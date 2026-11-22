"""One native plain/AD no-update observation for each pinned HPO width.

This helper has no model launcher or authority. A separately reviewed owner
authenticates the model, frozen fixture and native memory guard first.
"""
import gc
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import native_primal_observer as old

WIDTHS = (288, 544, 576, 608, 640)


def token_sha(ids):
    h = hashlib.sha256()
    for token in ids:
        if type(token) is not int or not 0 <= token <= 0xffffffff:
            raise ValueError('Unsigned token ID required')
        h.update(struct.pack('>I', token))
    return h.hexdigest()


def observe(runtime, neural, model, batch, lengths, expected, *,
            full_native_reference, parameter_digest, boundary, output_dir):
    """Record exact value/count/dtype/state before applying original gate."""
    keys = {'row_id', 'token_u32be_sha256', 'semantic_tokens', 'prefix_tokens',
            'target_tokens', 'model_input_width', 'batch_width', 'ceiling_padded'}
    if type(expected) is not dict or set(expected) != keys or expected['model_input_width'] not in WIDTHS:
        raise ValueError('Exact reviewed width fixture required')
    n, prefix, width = expected['semantic_tokens'], expected['prefix_tokens'], expected['batch_width']
    if (type(n) is not int or type(prefix) is not int or type(width) is not int
            or type(expected['ceiling_padded']) is not bool
            or not 0 < prefix < n <= 640 or width != expected['model_input_width'] + 1
            or expected['target_tokens'] != n - prefix
            or tuple(batch.shape) != (1, width) or tuple(lengths.shape) != (1, 2)):
        raise ValueError('Exact original width/target geometry required')
    runtime.eval(batch, lengths)
    items = batch.tolist()
    if (lengths.tolist() != [[prefix, n]] or len(items) != 1 or len(items[0]) != width
            or token_sha(items[0][:n]) != expected['token_u32be_sha256']
            or any(value != 0 for value in items[0][n:])):
        raise ValueError('Frozen row tokens or mask differ')
    out = Path(output_dir)
    if not out.is_dir() or out.is_symlink(): raise ValueError('Owned width directory required')
    partial = {'schema': 'research4.all64-force-fused-width-observation.v1',
        'case': expected, 'loss_atol': old.LOSS_ATOL, 'loss_rtol': old.LOSS_RTOL,
        'adam_updates': 0, 'qualification_claim': False, 'started_epoch': time.time()}
    boundary('before_plain_forward')
    before = old.snapshot(runtime, model, parameter_digest)
    mean, count = full_native_reference(runtime, neural, model, batch, lengths)
    runtime.eval(mean, count)
    plain = old.scalar_observation(mean, count)
    after_plain = old.snapshot(runtime, model, parameter_digest)
    partial.update(before=before, plain=plain, after_plain=after_plain)
    old.write_once(out / 'PLAIN_FORWARD.json', partial)
    del mean, count
    gc.collect(); runtime.clear_cache()
    boundary('after_plain_before_ad')
    def objective(b, lens): return full_native_reference(runtime, neural, model, b, lens)
    derivative = neural.value_and_grad(model, objective)
    (ad_mean, ad_count), gradient = derivative(batch, lengths)
    runtime.eval(ad_mean, ad_count, gradient)
    differentiated = old.scalar_observation(ad_mean, ad_count)
    after_ad = old.snapshot(runtime, model, parameter_digest)
    del gradient
    result = dict(partial, differentiated=differentiated, after_ad=after_ad,
        theta_unchanged=before['trainable_state_sha256'] == after_plain['trainable_state_sha256'] == after_ad['trainable_state_sha256'],
        rng_unchanged=before['rng_state_sha256'] == after_plain['rng_state_sha256'] == after_ad['rng_state_sha256'])
    a, b = plain['loss']['value'], differentiated['loss']['value']
    if a is None or b is None:
        result.update(status='NONFINITE', absolute_difference=None, original_allowed_difference=None,
                      original_tolerance_passed=False)
    else:
        diff, allowed = abs(a - b), old.LOSS_ATOL + old.LOSS_RTOL * abs(a)
        passed = (diff <= allowed and plain['target_count'] == differentiated['target_count'] == expected['target_tokens'])
        result.update(status='MATCH' if passed else 'MISMATCH', absolute_difference=diff,
                      original_allowed_difference=allowed, original_tolerance_passed=passed)
    if not result['theta_unchanged'] or not result['rng_unchanged']:
        result['status'] = 'STATE_CHANGED'
    result['finished_epoch'] = time.time()
    old.write_once(out / 'WIDTH_OBSERVATION.json', result)
    boundary('after_ad_observation_persisted')
    if result['status'] != 'MATCH':
        raise RuntimeError('Forced width primal did not match; exact evidence saved')
    return result
