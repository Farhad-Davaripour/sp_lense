"""Version-two exact supervised-suffix loss; no MLX import until called.

Shifted target position j is original token index j+1. A rendered prefix of
P IDs and full unpadded length L therefore supervises P <= position < L.
The native pinned trainer's inclusive upper bound is deliberately corrected
here; the completed/frozen last-layer experiment is never modified.
"""
LOSS_VERSION = 'shifted_suffix_P_le_position_lt_L.v2'


def target_positions(prefix, length, batch_width):
    if any(type(v) is not int for v in (prefix, length, batch_width)) or not 0 < prefix < length <= batch_width <= 1024:
        raise ValueError('Invalid untruncated prompt/target/batch lengths')
    return list(range(prefix, length))


def make_batch(runtime, token_ids, prefix):
    length = len(token_ids)
    width = min(1024, 1 + 32 * ((length + 31) // 32))
    target_positions(prefix, length, width)
    if any(type(token) is not int or token < 0 for token in token_ids): raise ValueError('Invalid token IDs')
    padded = list(token_ids) + [0] * (width-length)
    return (runtime.array([padded], dtype=runtime.int32),
            runtime.array([[prefix, length]], dtype=runtime.int32))


def sum_with_runtime(runtime, neural, model, batch, lengths):
    inputs, targets = batch[:, :-1], batch[:, 1:]
    logits = model(inputs)
    positions = runtime.arange(1, targets.shape[1]+1)
    mask = runtime.logical_and(positions >= lengths[:, 0:1], positions < lengths[:, 1:])
    nll = neural.losses.cross_entropy(logits, targets) * mask
    return nll.astype(runtime.float32).sum(), mask.sum()


def masked_target_sum(model, batch, lengths):
    import mlx.core as mx
    import mlx.nn as nn
    return sum_with_runtime(mx, nn, model, batch, lengths)


def masked_target_loss(model, batch, lengths):
    total, count = masked_target_sum(model, batch, lengths)
    return total/count, count
