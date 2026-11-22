"""Unlaunchable helper for a future read-only matched-theta diagnosis.

No model, optimizer, dataset, GPU library, or supervisor is loaded on import.
The separately reviewed owner must supply the exact failed row, model and
native reference. This helper executes a forward and an autodiff derivative,
but never constructs or updates an optimizer and never marks qualification.
"""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import time

CASE_ID = 'discover_before_planning_01.1'
TOKEN_SHA256 = 'a495e741aa7b26ae3640e145b767f648c17c09cf82789fc4b24d897a5f642730'
FAILED_WORKER_SHA256 = 'b4996bc66695ff8abda70cc78764875d7cdfadcb1be1d7666b985fd6f68e938e'
LOSS_ATOL = 0.0001
LOSS_RTOL = 0.0001
HEX = re.compile(r'^[0-9a-f]{64}$')


def digest_json(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def write_once(path, value):
    path = Path(path)
    if path.is_symlink() or not path.parent.is_dir():
        raise ValueError('Existing owned diagnostic directory required')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def finite_number(value):
    value = float(value)
    if math.isfinite(value):
        return {'value': value, 'repr': repr(value), 'classification': 'finite'}
    return {'value': None, 'repr': repr(value), 'classification': 'nonfinite'}


def snapshot(runtime, model, parameter_digest):
    theta = parameter_digest(model)
    if not isinstance(theta, str) or not HEX.fullmatch(theta):
        raise ValueError('Canonical F32 trainable-state SHA-256 required')
    runtime.eval(runtime.random.state)
    rng = [value.tolist() for value in list(runtime.random.state)]
    return {'trainable_state_sha256': theta, 'rng_state_sha256': digest_json(rng)}


def scalar_observation(value, count):
    dtype, count_dtype = str(value.dtype), str(count.dtype)
    return {'loss': finite_number(value.item()), 'target_count': int(count.item()),
            'loss_dtype': dtype, 'count_dtype': count_dtype}


def token_digest(ids):
    digest = hashlib.sha256()
    for token in ids:
        if type(token) is not int or not 0 <= token <= 0xffffffff:
            raise ValueError('Token IDs must be unsigned 32-bit integers')
        digest.update(struct.pack('>I', token))
    return digest.hexdigest()


def run_observation(runtime, neural, model, batch, lengths, *,
                    full_native_reference, parameter_digest, boundary,
                    case_id, token_sha256, output_dir):
    """Persist matching forward/AD values and state digests before any raise.

    `boundary` must enforce a separately owned deadline and resource guard;
    it is deliberately not represented as an authority check by this helper.
    The caller must authenticate actual token IDs, model/adapter configuration,
    source identity and process ownership before invoking this function.
    """
    if (case_id, token_sha256) != (CASE_ID, TOKEN_SHA256):
        raise ValueError('Exact failed TRAIN case/token binding required')
    if tuple(batch.shape) != (1, 289) or tuple(lengths.shape) != (1, 2):
        raise ValueError('Exact failed batch/lengths geometry required')
    if not callable(full_native_reference) or not callable(parameter_digest) or not callable(boundary):
        raise ValueError('Owned reference, state digest and guard boundary required')
    runtime.eval(batch, lengths)
    if lengths.tolist() != [[257, 286]]:
        raise ValueError('Exact original target mask required')
    actual_ids = batch.tolist()
    if (len(actual_ids) != 1 or len(actual_ids[0]) != 289
            or token_digest(actual_ids[0][:286]) != TOKEN_SHA256
            or actual_ids[0][286:] != [0, 0, 0]):
        raise ValueError('Padded input tokens do not match the failed case')
    out = Path(output_dir)
    if not out.is_dir() or out.is_symlink():
        raise ValueError('Owned diagnostic directory must exist')
    partial = {'schema': 'research4.all64-native-primal-observation.v1',
               'case_id': CASE_ID, 'token_u32be_sha256': TOKEN_SHA256,
               'failed_worker_sha256': FAILED_WORKER_SHA256,
               'batch_shape': [1, 289], 'input_shape': [1, 288],
               'prefix_tokens': 257, 'complete_tokens': 286, 'target_tokens': 29,
               'loss_atol': LOSS_ATOL, 'loss_rtol': LOSS_RTOL,
               'optimizer_constructed_by_observer': False, 'adam_updates': 0,
               'qualification_claim': False, 'scientific_training': False,
               'started_epoch': time.time()}
    try:
        boundary('before_plain_forward')
        before = snapshot(runtime, model, parameter_digest)
        plain_loss, plain_count = full_native_reference(runtime, neural, model, batch, lengths)
        runtime.eval(plain_loss, plain_count)
        plain = scalar_observation(plain_loss, plain_count)
        after_plain = snapshot(runtime, model, parameter_digest)
        partial.update(before=before, plain=plain, after_plain=after_plain)
        write_once(out / 'PLAIN_FORWARD.json', partial)
        # Match the failed worker: discard the forward arrays, collect, and
        # clear MLX cache before entering the differentiated reference.
        del plain_loss, plain_count
        gc.collect()
        runtime.clear_cache()
        boundary('after_plain_before_ad')
        def native_objective(b, lens):
            return full_native_reference(runtime, neural, model, b, lens)
        differentiated = neural.value_and_grad(model, native_objective)
        (ad_loss, ad_count), gradient = differentiated(batch, lengths)
        runtime.eval(ad_loss, ad_count, gradient)
        ad = scalar_observation(ad_loss, ad_count)
        after_ad = snapshot(runtime, model, parameter_digest)
        del gradient
        result = dict(partial, differentiated=ad, after_ad=after_ad,
                      theta_unchanged=(before['trainable_state_sha256'] == after_plain['trainable_state_sha256'] == after_ad['trainable_state_sha256']),
                      rng_unchanged=(before['rng_state_sha256'] == after_plain['rng_state_sha256'] == after_ad['rng_state_sha256']))
        a, b = plain['loss']['value'], ad['loss']['value']
        if a is None or b is None:
            result.update(status='NONFINITE', absolute_difference=None, original_allowed_difference=None,
                          original_tolerance_passed=False)
        else:
            difference = abs(a - b)
            limit = LOSS_ATOL + LOSS_RTOL * abs(a)
            result.update(status='MATCH' if difference <= limit and plain['target_count'] == ad['target_count'] == 29 else 'MISMATCH',
                          absolute_difference=difference, original_allowed_difference=limit,
                          original_tolerance_passed=(difference <= limit and plain['target_count'] == ad['target_count'] == 29))
        if not result['theta_unchanged'] or not result['rng_unchanged']:
            result['status'] = 'STATE_CHANGED'
        result['finished_epoch'] = time.time()
        write_once(out / 'NATIVE_PRIMAL_OBSERVATION.json', result)
        boundary('after_ad_observation_persisted')
        if result['status'] != 'MATCH':
            raise RuntimeError('Native primal control failed; exact evidence saved')
        return result
    except BaseException as exc:
        # A completed observation is never replaced with an abort record.
        if not (out / 'NATIVE_PRIMAL_OBSERVATION.json').exists():
            write_once(out / 'OBSERVATION_ABORTED.json', {'schema': partial['schema'],
                'status': 'ABORTED', 'error_type': type(exc).__name__, 'message': str(exc),
                'case_id': CASE_ID, 'adam_updates': 0, 'qualification_claim': False})
        raise


if __name__ == '__main__':
    raise SystemExit('HELD: helper cannot launch a model or bypass the separately reviewed owner.')
