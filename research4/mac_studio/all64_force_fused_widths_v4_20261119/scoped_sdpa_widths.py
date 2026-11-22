"""Owned diagnostic-only SDPA alias policy for one exact Qwen attention shape."""
from contextlib import contextmanager
import hashlib
import importlib.metadata
import inspect
import os
from pathlib import Path
import sys
from types import CodeType

SOURCE_PINS = {
    'mlx_lm/models/base.py': '63d8e58a4b6f8c0902719499c693647fbf486bbd5d22b09868a07d37e08c83d6',
    'mlx_lm/models/qwen3_next.py': '67ca08a4d08cb7be124965e95399735e556f1eaeb21a16bc23cb88d1ac035e11',
    'mlx/core/fast.pyi': 'e6ead453b46633174ef101388f4e6b7c02ed837fbee91db16aa3be39ea6f2d4b',
}
ENV_KEYS = ('MLX_SDPA_VJP_FALLBACK', 'MLX_ENABLE_TF32', 'MLX_METAL_GPU_ARCH')


def environment_snapshot():
    return {name: os.environ.get(name) for name in ENV_KEYS}


def verify_installed_policy_sources():
    for name, value in SOURCE_PINS.items():
        distribution = importlib.metadata.distribution('mlx-lm' if name.startswith('mlx_lm/') else 'mlx')
        path = Path(distribution.locate_file(name))
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != value:
            raise ValueError('Pinned native SDPA helper/API source differs')
    return dict(SOURCE_PINS)


def make_dispatch(original_helper, native_fast, receipt, force_fused, phase, native_bf16):
    """Pure factory also usable in model-free argument/restoration tests."""
    def dispatch(queries, keys, values, cache, scale, mask, sinks=None):
        _approved_width = queries.shape[2] if len(queries.shape) == 4 else None
        valid = (
            type(_approved_width) is int and _approved_width in (288, 544, 576, 608, 640)
            and list(queries.shape) == [1, 24, _approved_width, 256]
            and list(keys.shape) == [1, 4, _approved_width, 256]
            and list(values.shape) == [1, 4, _approved_width, 256]
            and queries.dtype == keys.dtype == values.dtype == native_bf16
            and cache is None and isinstance(mask, str) and mask == 'causal'
            and sinks is None and type(scale) in (int, float) and scale == .0625
        )
        if not valid:
            receipt['off_scope_calls'] += 1
            raise ValueError('Diagnostic SDPA alias received an unapproved shape/cache/mask/dtype')
        receipt['matched_calls'] += 1
        receipt['calls'].append({
            'phase': phase['name'], 'q_shape': list(queries.shape), 'k_shape': list(keys.shape),
            'v_shape': list(values.shape), 'dtype': str(queries.dtype), 'cache': None,
            'mask': mask, 'sinks': None, 'scale': scale, 'force_fused_requested': force_fused,
        })
        if force_fused:
            return native_fast(queries, keys, values, scale=scale, mask=mask, sinks=sinks, force_fused=True)
        return original_helper(queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks)
    return dispatch


@contextmanager
def scoped_sdpa_policy(*, owned_process_pid, policy, phase, expected_environment):
    if owned_process_pid != os.getpid() or policy not in ('default', 'force_fused'):
        raise ValueError('Explicit owned diagnostic policy required')
    if environment_snapshot() != expected_environment or expected_environment['MLX_SDPA_VJP_FALLBACK'] not in (None, '0'):
        raise ValueError('SDPA environment changed or the native fast VJP is disabled')
    pins = verify_installed_policy_sources()
    import mlx.core as mx
    from mlx_lm.models import base, qwen3_next
    original = qwen3_next.scaled_dot_product_attention
    native_fast = mx.fast.scaled_dot_product_attention
    if original is not base.scaled_dot_product_attention:
        raise ValueError('Native Qwen SDPA alias identity differs')
    source = Path(base.__file__)
    definitions = {code.co_name:code for code in compile(source.read_text(),original.__code__.co_filename,'exec',dont_inherit=True,optimize=sys.flags.optimize).co_consts if isinstance(code,CodeType)}
    if original.__globals__ is not base.__dict__ or original.__code__ != definitions['scaled_dot_product_attention']:
        raise ValueError('Original native SDPA helper code/namespace differs')
    if tuple(inspect.signature(original).parameters) != ('queries', 'keys', 'values', 'cache', 'scale', 'mask', 'sinks'):
        raise ValueError('Native SDPA helper signature changed')
    if qwen3_next.Qwen3NextAttention.__call__.__globals__ is not qwen3_next.__dict__:
        raise ValueError('Native attention class no longer resolves the reviewed module alias')
    receipt = {
        'schema': 'research4.scoped-block3-sdpa-policy.v1', 'policy': policy,
        'owned_process_pid': owned_process_pid, 'matched_calls': 0, 'off_scope_calls': 0,
        'calls': [], 'source_sha256': pins, 'environment': expected_environment,
        'original_helper_id': id(original), 'native_fast_id': id(native_fast),
        'restoration_verified': False, 'installed_files_modified': False,
        'kernel_regime_scope': 'Requested policy and tagged-source prediction, not kernel profiling',
    }
    target = make_dispatch(original, native_fast, receipt, policy == 'force_fused', phase, mx.bfloat16)
    def owned_dispatch(*args, **kwargs):
        if qwen3_next.scaled_dot_product_attention is not owned_dispatch or base.scaled_dot_product_attention is not original or mx.fast.scaled_dot_product_attention is not native_fast:
            raise RuntimeError('Owned SDPA alias/native function identity changed')
        return target(*args, **kwargs)
    try:
        qwen3_next.scaled_dot_product_attention = owned_dispatch
        yield receipt
    finally:
        unexpected = qwen3_next.scaled_dot_product_attention is not owned_dispatch
        qwen3_next.scaled_dot_product_attention = original
        receipt['restoration_verified'] = qwen3_next.scaled_dot_product_attention is original
        receipt['native_base_and_fast_untouched'] = base.scaled_dot_product_attention is original and mx.fast.scaled_dot_product_attention is native_fast
        receipt['environment_unchanged'] = environment_snapshot() == expected_environment
        verify_installed_policy_sources()
        if unexpected or not receipt['restoration_verified'] or not receipt['native_base_and_fast_untouched'] or not receipt['environment_unchanged']:
            raise RuntimeError('Owned SDPA policy restoration/integrity failed')
