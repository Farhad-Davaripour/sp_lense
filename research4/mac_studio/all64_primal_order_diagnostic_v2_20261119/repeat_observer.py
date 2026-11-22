"""Unlaunchable no-update order/cache control for the failed rank-4 row.

The separately reviewed caller supplies the original worker's reference and
native runtime, an owned guard and exact authenticated model/input. No model,
optimizer or training data is loaded or executed merely by importing this.
"""
import gc
import hashlib
import json
import math
from pathlib import Path
import struct
import time

CASE_ID = 'discover_before_planning_01.1'
TOKEN_SHA256 = 'a495e741aa7b26ae3640e145b767f648c17c09cf82789fc4b24d897a5f642730'
FAILED_WORKER_SHA256 = 'b4996bc66695ff8abda70cc78764875d7cdfadcb1be1d7666b985fd6f68e938e'
BLOCK = ('direct', 'direct', 'ad', 'ad', 'ad', 'ad', 'direct', 'direct')
ATOL = 0.0001
RTOL = 0.0001


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def token_digest(ids):
    h = hashlib.sha256()
    for value in ids:
        if type(value) is not int or not 0 <= value <= 0xffffffff:
            raise ValueError('Expected unsigned 32-bit token')
        h.update(struct.pack('>I', value))
    return h.hexdigest()


def _tag(h, value):
    b = json.dumps(value, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    h.update(struct.pack('>Q', len(b))); h.update(b)


def full_state_snapshot(runtime, model, tree_flatten, boundary, chunk_bytes=8 * 1024 * 1024):
    """Hash all enumerable model.state leaves, one bounded byte chunk at a time.

    Array bits are viewed, never numerically converted (including BF16). The
    caller's independent guard must run asynchronously during this operation.
    Python/module attributes outside model.state are outside this digest.
    """
    import numpy as np
    if type(chunk_bytes) is not int or not 0 < chunk_bytes <= 8 * 1024 * 1024:
        raise ValueError('Bounded snapshot chunk required')
    leaves = sorted(tree_flatten(model.state), key=lambda item: item[0])
    if not leaves or len({name for name, _ in leaves}) != len(leaves):
        raise ValueError('Nonempty unique full model.state inventory required')
    parameters = tree_flatten(model.parameters())
    parameter_names = {name for name, value in parameters if isinstance(value, runtime.array)}
    if len(parameter_names) != len(parameters):
        raise ValueError('Unexpected duplicate or non-array model parameter leaf')
    h = hashlib.sha256(); inventory = []; arrays = nonarrays = 0; array_names = set()
    boundary('before_full_state_snapshot')
    for name, value in leaves:
        if not isinstance(name, str):
            raise ValueError('State leaf name must be a string')
        if isinstance(value, runtime.array):
            shape = list(value.shape); dtype = str(value.dtype)
            raw = value.reshape((-1,)).view(runtime.uint8)
            meta = {'name': name, 'kind': 'array', 'dtype': dtype, 'shape': shape,
                    'raw_bytes': int(raw.size)}
            _tag(h, meta)
            for start in range(0, int(raw.size), chunk_bytes):
                portion = raw[start:start + chunk_bytes]
                runtime.eval(portion)
                data = np.array(portion).tobytes(order='C')
                if len(data) != min(chunk_bytes, int(raw.size) - start):
                    raise ValueError('State-byte read was incomplete')
                h.update(data)
            arrays += 1
            array_names.add(name)
        elif value is None or type(value) in (str, bool, int, float):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError('Nonfinite Python state leaf')
            meta = {'name': name, 'kind': 'python_scalar', 'type': type(value).__name__, 'value': value}
            _tag(h, meta); nonarrays += 1
        else:
            raise ValueError('Unsupported enumerable model.state leaf: ' + name + ' ' + type(value).__name__)
        inventory.append(meta)
    if not parameter_names.issubset(array_names):
        raise ValueError('Some enumerable model parameters are absent from model.state')
    boundary('after_full_state_snapshot')
    return {'sha256': h.hexdigest(), 'leaf_count': len(inventory), 'array_count': arrays,
            'parameter_count': len(parameter_names), 'all_parameters_included': True,
            'python_scalar_count': nonarrays, 'inventory': inventory,
            'scope': 'All leaves enumerated by mlx.utils.tree_flatten(model.state); Python/module attributes outside model.state are separate.'}


def flags_snapshot(model):
    flags = []
    for name, module in model.named_modules():
        value = module.training
        if type(name) is not str or type(value) is not bool:
            raise ValueError('Every named module must expose a boolean training flag')
        flags.append([name, value])
    flags.sort()
    if not flags or len({name for name, _ in flags}) != len(flags):
        raise ValueError('Named training-mode flags are incomplete or duplicate')
    return {'sha256': digest_json(flags), 'count': len(flags),
            'all_training': all(value for _, value in flags)}


def rng_snapshot(runtime):
    runtime.eval(runtime.random.state)
    return digest_json([item.tolist() for item in list(runtime.random.state)])


def append_event(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())


def run_observation(runtime, neural, model, batch, lengths, *, full_native_reference,
                    tree_flatten, parameter_digest, optimizer_step, boundary, output_dir, write_once):
    """Observe 16 synchronized primals; no optimizer is constructed or updated."""
    runtime.eval(batch, lengths)
    ids = batch.tolist()
    if (tuple(batch.shape) != (1, 289) or tuple(lengths.shape) != (1, 2)
            or lengths.tolist() != [[257, 286]] or len(ids) != 1 or len(ids[0]) != 289
            or token_digest(ids[0][:286]) != TOKEN_SHA256 or ids[0][286:] != [0, 0, 0]):
        raise ValueError('Exact failed row, tokens, padding and target mask required')
    output = Path(output_dir)
    if not output.is_dir() or output.is_symlink():
        raise ValueError('Existing owned diagnostic directory required')
    events = output / 'PRIMAL_ORDER_EVENTS.jsonl'
    if events.exists(): raise ValueError('Existing event log is never reused')
    before = full_state_snapshot(runtime, model, tree_flatten, boundary)
    start_rng, start_flags = rng_snapshot(runtime), flags_snapshot(model)
    start_trainable = parameter_digest(model)
    if optimizer_step() != 0:
        raise ValueError('Zero-step Adam state required')
    if not start_flags['all_training']:
        raise ValueError('All named modules must remain in training mode')
    write_once(output / 'STATE_BEFORE.json', before)
    def objective(b, lens):
        return full_native_reference(runtime, neural, model, b, lens)
    derivative = neural.value_and_grad(model, objective)
    all_events = []
    for block in (1, 2):
        if block == 2:
            boundary('before_single_cache_clear')
            gc.collect(); runtime.clear_cache()
            append_event(events, {'kind': 'cache_clear', 'after_call': 8, 'epoch': time.time()})
            boundary('after_single_cache_clear')
        for method in BLOCK:
            ordinal = len(all_events) + 1
            boundary('before_call_' + str(ordinal))
            before_rng, before_flags = rng_snapshot(runtime), flags_snapshot(model)
            before_trainable, before_step = parameter_digest(model), optimizer_step()
            cache_before = int(runtime.get_cache_memory())
            started = time.perf_counter()
            if method == 'direct':
                mean, count = full_native_reference(runtime, neural, model, batch, lengths)
                runtime.eval(mean, count)
            else:
                (mean, count), gradient = derivative(batch, lengths)
                runtime.eval(mean, count, gradient)
                del gradient
            value, n = float(mean.item()), int(count.item())
            valid = math.isfinite(value) and n == 29
            after_rng, after_flags = rng_snapshot(runtime), flags_snapshot(model)
            after_trainable, after_step = parameter_digest(model), optimizer_step()
            event = {'kind': 'measurement' if valid else 'invalid_measurement',
                     'ordinal': ordinal, 'block': block, 'method': method,
                     'loss': value if math.isfinite(value) else None, 'loss_repr': repr(value),
                     'loss_classification': 'finite' if math.isfinite(value) else 'nonfinite',
                     'loss_dtype': str(mean.dtype),
                     'target_count': n, 'count_dtype': str(count.dtype),
                     'rng_before_sha256': before_rng, 'rng_after_sha256': after_rng,
                     'trainable_before_sha256': before_trainable, 'trainable_after_sha256': after_trainable,
                     'optimizer_step_before': before_step, 'optimizer_step_after': after_step,
                     'flags_before': before_flags, 'flags_after': after_flags,
                     'cache_bytes_before': cache_before, 'cache_bytes_after': int(runtime.get_cache_memory()),
                     'synchronized': True, 'elapsed_seconds': time.perf_counter() - started}
            append_event(events, event); all_events.append(event)
            del mean, count
            boundary('after_call_' + str(ordinal))
            if not valid:
                raise ValueError('Invalid primal or target count; exact partial event saved')
            if (before_step != 0 or after_step != 0 or before_rng != start_rng or after_rng != start_rng
                    or before_flags != start_flags or after_flags != start_flags
                    or before_trainable != start_trainable or after_trainable != start_trainable):
                raise RuntimeError('Per-call optimizer, trainable state, RNG or mode changed; partial event log saved')
    after = full_state_snapshot(runtime, model, tree_flatten, boundary)
    write_once(output / 'STATE_AFTER.json', after)
    final_rng, final_flags = rng_snapshot(runtime), flags_snapshot(model)
    direct = [x['loss'] for x in all_events if x['method'] == 'direct']
    ad = [x['loss'] for x in all_events if x['method'] == 'ad']
    pairs, repetitions = [], []
    for block in (1, 2):
        block_events = [item for item in all_events if item['block'] == block]
        for left, right in ((1, 2), (5, 6)):
            a, b = block_events[left], block_events[right]
            plain = a if a['method'] == 'direct' else b
            differentiated = a if a['method'] == 'ad' else b
            delta = abs(plain['loss'] - differentiated['loss'])
            limit = ATOL + RTOL * abs(plain['loss'])
            pairs.append({'block': block, 'order': [a['method'], b['method']],
                'ordinals': [a['ordinal'], b['ordinal']], 'absolute_difference': delta,
                'original_allowed_difference': limit, 'original_tolerance_passed': delta <= limit})
        for left in (0, 2, 4, 6):
            a, b = block_events[left], block_events[left + 1]
            repetitions.append({'block': block, 'method': a['method'],
                'ordinals': [a['ordinal'], b['ordinal']],
                'absolute_difference': abs(a['loss'] - b['loss'])})
    rng_calls_unchanged = all(x['rng_before_sha256'] == x['rng_after_sha256'] == start_rng for x in all_events)
    flag_calls_unchanged = all(x['flags_before'] == x['flags_after'] == start_flags for x in all_events)
    trainable_unchanged = all(x['trainable_before_sha256'] == x['trainable_after_sha256'] == start_trainable for x in all_events)
    optimizer_unchanged = all(x['optimizer_step_before'] == x['optimizer_step_after'] == 0 for x in all_events)
    state_unchanged = before['sha256'] == after['sha256'] and before['inventory'] == after['inventory']
    result = {'schema': 'research4.all64-primal-order-observation.v1', 'status': 'OBSERVED_NOT_QUALIFICATION',
              'case_id': CASE_ID, 'token_u32be_sha256': TOKEN_SHA256, 'failed_worker_sha256': FAILED_WORKER_SHA256,
              'batch_shape': [1, 289], 'input_shape': [1, 288], 'prefix_tokens': 257, 'target_tokens': 29,
              'schedule_each_block': list(BLOCK), 'blocks': 2, 'measurement_count': len(all_events),
              'single_cache_clear_after_call': 8, 'direct_count': len(direct), 'ad_count': len(ad),
              'direct_min': min(direct), 'direct_max': max(direct), 'ad_min': min(ad), 'ad_max': max(ad),
              'adjacent_direct_ad_pairs': pairs, 'consecutive_same_method_pairs': repetitions,
              'original_loss_atol': ATOL, 'original_loss_rtol': RTOL,
              'state_before_sha256': before['sha256'], 'state_after_sha256': after['sha256'],
              'state_leaf_count': before['leaf_count'], 'state_unchanged': state_unchanged,
              'trainable_state_sha256': start_trainable,
              'rng_before_sha256': start_rng, 'rng_after_sha256': final_rng,
              'rng_unchanged': start_rng == final_rng and rng_calls_unchanged,
              'training_flags_before': start_flags, 'training_flags_after': final_flags,
              'training_flags_unchanged': start_flags == final_flags and flag_calls_unchanged,
              'trainable_unchanged': trainable_unchanged, 'optimizer_unchanged': optimizer_unchanged,
              'python_module_attributes_scope': 'Python/module attributes outside model.state are not covered; named module training flags are separately recorded. A transient frozen-state mutation that reverts between full snapshots could be missed.',
              'adam_updates': 0, 'qualification_claim': False, 'scientific_training': False,
              'events_sha256': hashlib.sha256(events.read_bytes()).hexdigest(), 'finished_epoch': time.time()}
    write_once(output / 'PRIMAL_ORDER_OBSERVATION.json', result)
    if (not state_unchanged or not result['rng_unchanged'] or not result['training_flags_unchanged']
            or not trainable_unchanged or not optimizer_unchanged):
        raise RuntimeError('State or RNG changed; exact observation was saved')
    return result


if __name__ == '__main__':
    raise SystemExit('HELD: no model or supervisor may be launched from this helper.')
