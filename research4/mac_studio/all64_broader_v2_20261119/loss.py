"""Bounded full-suffix loss for the frozen broader-continuity corpus.

The 64-row head includes every supervised token, including EOS.  It never
truncates a suffix: overlong inputs fail before any tensor is constructed.
No MLX or model is imported at module import time.
"""

CONTEXT_LIMIT = 640
HEAD_WINDOW = 64


def geometry(ids, prefix, ceiling_padded=False):
    if not isinstance(ids, (list, tuple)) or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError('Nonnegative token IDs required')
    length = len(ids)
    if type(prefix) is not int or not 0 < prefix < length <= CONTEXT_LIMIT:
        raise ValueError('Untruncated, nonempty target within context required')
    if length - prefix > HEAD_WINDOW:
        raise ValueError('Supervised suffix exceeds selected head; truncation forbidden')
    if type(ceiling_padded) is not bool:
        raise ValueError('ceiling_padded must be boolean')
    width = CONTEXT_LIMIT + 1 if ceiling_padded else min(CONTEXT_LIMIT + 1, 1 + 32 * ((length + 31) // 32))
    if width < length:
        raise ValueError('Batch width would truncate the sequence')
    return {'semantic_tokens': length, 'prefix_tokens': prefix,
            'target_tokens': length - prefix, 'batch_width': width,
            'model_input_width': width - 1, 'ceiling_padded': ceiling_padded,
            'target_positions': list(range(prefix, length))}


def make_batch(runtime, ids, prefix, ceiling_padded=False):
    info = geometry(ids, prefix, ceiling_padded)
    padded = list(ids) + [0] * (info['batch_width'] - len(ids))
    return runtime.array([padded], dtype=runtime.int32), runtime.array([[prefix, len(ids)]], dtype=runtime.int32), info


def selected_suffix_components(runtime, neural, model, batch, lengths):
    """Return mean supervised NLL and token count using the full native body.

    Padded head positions are masked; the caller separately verifies the count
    equals the exact rendered suffix, including the assistant EOS token.
    """
    hidden = model.language_model.model(batch[:, :-1], cache=None)
    positions = lengths[:, 0:1] + runtime.arange(HEAD_WINDOW)[None, :]
    predictors = runtime.clip(positions - 1, 0, hidden.shape[1] - 1)
    selected = runtime.take_along_axis(hidden, predictors[:, :, None], axis=1)
    logits = model.language_model.lm_head(selected)
    targets = runtime.take_along_axis(batch, runtime.clip(positions, 0, batch.shape[1] - 1), axis=1)
    mask = positions < lengths[:, 1:]
    count = mask.sum()
    total = (neural.losses.cross_entropy(logits, targets) * mask).astype(runtime.float32).sum()
    return total / count, (count, logits, hidden)


def selected_suffix_loss(runtime, neural, model, batch, lengths):
    """Training objective omitting the large diagnostic auxiliaries."""
    hidden = model.language_model.model(batch[:, :-1], cache=None)
    positions = lengths[:, 0:1] + runtime.arange(HEAD_WINDOW)[None, :]
    predictors = runtime.clip(positions - 1, 0, hidden.shape[1] - 1)
    selected = runtime.take_along_axis(hidden, predictors[:, :, None], axis=1)
    logits = model.language_model.lm_head(selected)
    targets = runtime.take_along_axis(batch, runtime.clip(positions, 0, batch.shape[1] - 1), axis=1)
    valid = positions < lengths[:, 1:]
    count = valid.sum()
    total = (neural.losses.cross_entropy(logits, targets) * valid).astype(runtime.float32).sum()
    return total / count, count


def selected_suffix_vector(runtime, neural, model, batch, lengths):
    """Per-position likelihood for frozen DEV masks; caller must check bounds."""
    hidden = model.language_model.model(batch[:, :-1], cache=None)
    positions = lengths[:, 0:1] + runtime.arange(HEAD_WINDOW)[None, :]
    predictors = runtime.clip(positions - 1, 0, hidden.shape[1] - 1)
    selected = runtime.take_along_axis(hidden, predictors[:, :, None], axis=1)
    logits = model.language_model.lm_head(selected)
    targets = runtime.take_along_axis(batch, runtime.clip(positions, 0, batch.shape[1] - 1), axis=1)
    nll = neural.losses.cross_entropy(logits, targets).astype(runtime.float32)
    return nll, positions < lengths[:, 1:]
