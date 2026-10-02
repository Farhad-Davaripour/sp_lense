"""Observe the existing generator without changing model inputs or outputs.

Only real generation forwards are observed. No extra model forward is made.
Numeric tolerance is fixed before queries; missing position data is explicit.
"""
import copy
import hashlib
import json
from pathlib import Path

ATOL = 1e-3
RTOL = 1e-3


def _digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def _tensor(value):
    if value is None:
        return None
    if not hasattr(value, 'detach'):
        return {'available': False, 'reason': 'argument was not a tensor', 'type': type(value).__name__}
    cpu = value.detach().cpu()
    return {'available': True, 'shape': list(cpu.shape), 'dtype': str(cpu.dtype), 'values': cpu.tolist()}


def _arguments(args, kwargs):
    # The trusted historical generator calls these model arguments by keyword.
    names = ('input_ids', 'attention_mask', 'position_ids', 'cache_position')
    result = {name: _tensor(kwargs.get(name)) for name in names}
    result['positional_argument_count'] = len(args)
    cache = kwargs.get('past_key_values')
    result['cache_class'] = type(cache).__name__ if cache is not None else None
    return result


def _valid_positions(value, masks):
    if not value or not value.get('available'):
        return None, 'No tensor position_ids reached the observed forward boundary.'
    shape, data = value['shape'], value['values']
    batch = len(masks)
    width = len(masks[0])
    valid = [[i for i, bit in enumerate(row) if bit] for row in masks]
    if len(shape) == 2 and shape == [batch, width]:
        return [[data[row][i] for i in valid[row]] for row in range(batch)], None
    if len(shape) == 3 and shape[1:] == [batch, width]:
        return [[[data[axis][row][i] for i in valid[row]] for axis in range(shape[0])]
                for row in range(batch)], None
    if len(shape) == 1 and shape == [width]:
        return [[data[i] for i in valid[row]] for row in range(batch)], None
    return None, 'Observed position_ids shape does not match the prefill batch and width.'


def capture_generation(model, tokenizer, prompts, tools, invoke, output_dir, cohort):
    """Return ``(unchanged generation rows, receipt)`` from zero-argument invoke.

    prompts are conversations exactly as passed to fast_inference.generate_many.
    cohort contains artifact IDs only and never enters the model prompt. output_dir
    must be unique for this generation call so earlier evidence cannot be replaced.
    """
    import torch
    from safetensors.torch import save_file
    from model_ops import ids

    conversations = copy.deepcopy(prompts)
    cohort = list(cohort)
    if not conversations or len(cohort) != len(conversations):
        raise ValueError('A nonempty cohort must match the conversation count')
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    prompt_ids = [ids(tokenizer, messages, tools) for messages in conversations]
    if not all(prompt_ids):
        raise ValueError('Empty encoded prompt')
    maximum = max(map(len, prompt_ids))
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    pad = tokenizer.pad_token_id or eos[0]  # Match the unchanged historical generator.
    expected_ids = [[pad] * (maximum-len(row)) + row for row in prompt_ids]
    expected_masks = [[0] * (maximum-len(row)) + [1] * len(row) for row in prompt_ids]

    base = getattr(model, 'base_model', None)
    outer = getattr(base, 'model', model)
    language_modules = [(name, module) for name, module in model.named_modules()
                        if name.endswith('.language_model')]
    observed = {'outer_forward_calls': 0, 'language_forward_calls': 0,
                'outer_first': None, 'language_first': None, 'logits': None,
                'logits_original_dtype': None}
    handles = []

    def outer_pre(_module, args, kwargs):
        observed['outer_forward_calls'] += 1
        if observed['outer_first'] is None:
            observed['outer_first'] = _arguments(args, kwargs)

    def language_pre(_module, args, kwargs):
        observed['language_forward_calls'] += 1
        if observed['language_first'] is None:
            observed['language_first'] = _arguments(args, kwargs)

    def outer_post(_module, _args, _kwargs, output):
        if observed['logits'] is None:
            value = getattr(output, 'logits', None)
            if value is not None:
                observed['logits_original_dtype'] = str(value.dtype)
                observed['logits'] = value[:, -1, :].detach().float().cpu().contiguous()

    handles.append(outer.register_forward_pre_hook(outer_pre, with_kwargs=True))
    handles.append(outer.register_forward_hook(outer_post, with_kwargs=True))
    if len(language_modules) == 1:
        handles.append(language_modules[0][1].register_forward_pre_hook(language_pre, with_kwargs=True))
    rows = None
    error = None
    try:
        rows = invoke()
    except BaseException as exc:
        error = exc
    finally:
        for handle in handles:
            handle.remove()

    first = observed['outer_first'] or {}
    actual_ids = (first.get('input_ids') or {}).get('values')
    actual_masks = (first.get('attention_mask') or {}).get('values')
    ids_valid = actual_ids == expected_ids
    masks_valid = actual_masks == expected_masks
    language = observed['language_first'] or {}
    position = language.get('position_ids') or first.get('position_ids')
    positions, position_issue = _valid_positions(position, expected_masks)
    position_source = 'language_forward' if language.get('position_ids') else 'outer_forward'
    if positions is None:
        position_source = None
    logits = observed['logits']
    finite = bool(logits is not None and torch.isfinite(logits).all())
    logits_shape = list(logits.shape) if logits is not None else None
    logits_batch_matches = bool(logits is not None and logits.shape[0] == len(conversations))
    row_receipts = []
    if logits is not None:
        save_file({'next_token_logits': logits}, str(destination/'first_logits.safetensors'),
                  metadata={'source': 'first observed real generation forward',
                            'original_dtype': observed['logits_original_dtype']})
    for index, encoded in enumerate(prompt_ids):
        generated = rows[index] if rows is not None and index < len(rows) else None
        item = {'cohort_id': cohort[index], 'row': index, 'prompt_token_ids': encoded,
                'prompt_sha256': _digest(encoded), 'history_sha256': _digest(conversations[index]),
                'left_padding': maximum-len(encoded), 'logical_positions': list(range(len(encoded))),
                'actual_unpadded_position_ids': positions[index] if positions is not None else None,
                'returned_prompt_ids_match': bool(generated and generated.get('prompt_token_ids') == encoded),
                'returned_token_ids': generated.get('token_ids') if generated else None,
                'emitted_first_token': generated['token_ids'][0] if generated and generated.get('token_ids') else None}
        if logits is not None and index < logits.shape[0]:
            item['raw_argmax'] = int(logits[index].argmax())
            item['finite_logits'] = bool(torch.isfinite(logits[index]).all())
            top = torch.topk(logits[index], 2)
            item['top_two_token_ids'] = top.indices.tolist()
            item['top_two_logits'] = top.values.tolist()
            item['top_two_margin'] = float(top.values[0]-top.values[1])
        row_receipts.append(item)
    receipt = {
        'observational_only': True, 'extra_model_forwards': 0,
        'invoke_completed': error is None,
        'invoke_exception_class': type(error).__name__ if error else None,
        'requested_batch_size': len(conversations), 'requested_cohort': cohort,
        'tools_sha256': _digest(tools), 'expected_input_ids': expected_ids,
        'expected_attention_mask': expected_masks, 'effective_pad_token_id': pad,
        'generation_eos_token_ids': eos, 'generation_config': model.generation_config.to_dict(),
        'outer_hook_location': 'model.base_model.model' if outer is not model else 'model',
        'language_hook_location': language_modules[0][0] if len(language_modules) == 1 else None,
        'language_hook_issue': None if len(language_modules) == 1 else
            'Expected exactly one named module ending .language_model; found '+str(len(language_modules)),
        'actual_outer_first_forward': observed['outer_first'],
        'actual_language_first_forward': observed['language_first'],
        'outer_forward_calls': observed['outer_forward_calls'],
        'language_forward_calls': observed['language_forward_calls'],
        'input_ids_verified': ids_valid, 'attention_mask_verified': masks_valid,
        'actual_positions_available': positions is not None, 'actual_position_source': position_source,
        'position_capture_issue': position_issue,
        'positions_note': 'Only position IDs supplied at observed forward boundaries are recorded; '
            'internal construction is not inferred from padding.',
        'first_logits_shape': logits_shape, 'first_logits_original_dtype': observed['logits_original_dtype'],
        'first_logits_finite': finite, 'first_logits_batch_matches': logits_batch_matches,
        'first_logits_path': 'first_logits.safetensors' if logits is not None else None,
        'numeric_tolerance': {'atol': ATOL, 'rtol': RTOL}, 'rows': row_receipts,
        'returned_batch_matches': bool(rows is not None and len(rows) == len(conversations))}
    receipt['input_integrity_passed'] = bool(error is None and ids_valid and masks_valid and
        logits_batch_matches and finite and receipt['returned_batch_matches'] and
        all(item['returned_prompt_ids_match'] for item in row_receipts))
    (destination/'GENERATION_CAPTURE.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    if error is not None:
        raise error
    return rows, receipt


def compare_recorded_turns(left, right):
    """Pure structural comparison of two saved anchor turn lists."""
    fields = ('prompt_token_ids', 'token_ids', 'text', 'actions', 'parse_error',
              'tool_results', 'state_before', 'state_after')
    comparisons = {}
    for field in fields:
        differences = [i for i, (a, b) in enumerate(zip(left, right)) if a.get(field) != b.get(field)]
        comparisons[field] = {'equal': len(left) == len(right) and not differences,
                              'first_different_turn': differences[0] if differences else
                                  min(len(left), len(right)) if len(left) != len(right) else None}
    return {'left_turn_count': len(left), 'right_turn_count': len(right), 'fields': comparisons,
            'token_invariant': comparisons['token_ids']['equal'],
            'behavior_invariant': all(comparisons[key]['equal'] for key in
                                     ('actions', 'parse_error', 'tool_results', 'state_before', 'state_after'))}


def compare_first_logits(left_file, right_file, left_row=0, right_row=0):
    """Compare saved observations only; does not run a model or change tolerance."""
    import torch
    from safetensors.torch import load_file
    left = load_file(str(left_file), device='cpu')['next_token_logits'][left_row]
    right = load_file(str(right_file), device='cpu')['next_token_logits'][right_row]
    if left.shape != right.shape:
        return {'shape_matches': False, 'left_shape': list(left.shape), 'right_shape': list(right.shape),
                'numeric_tolerance': {'atol': ATOL, 'rtol': RTOL}, 'logits_close': False}
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    return {'shape_matches': True, 'finite': finite,
            'maximum_absolute_difference': float((left-right).abs().max()),
            'logits_close': bool(finite and torch.allclose(left, right, atol=ATOL, rtol=RTOL)),
            'raw_argmax_equal': int(left.argmax()) == int(right.argmax()),
            'left_argmax': int(left.argmax()), 'right_argmax': int(right.argmax()),
            'numeric_tolerance': {'atol': ATOL, 'rtol': RTOL}}