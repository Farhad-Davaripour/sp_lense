"""Read-only numerical comparisons for the bounded SDPA VJP diagnostic.

The float64 CPU oracle is independent of MLX. Thresholds below are the
existing qualification's strict gradient threshold, not an assumption that
BF16 and float64 must agree. A miss is reported, never silently relaxed.
"""
import math
import numpy as np

from numpy_gqa_oracle import causal_gqa_vjp

ATOL = 1e-6
RTOL = 1e-3


def array_comparison(reference, observed):
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(observed, dtype=np.float64)
    if a.shape != b.shape or a.size == 0 or not np.isfinite(a).all():
        raise ValueError('Finite reference and equally shaped observed arrays required')
    if not np.isfinite(b).all():
        return {'shape': list(a.shape), 'observed_finite': False,
                'observed_nonfinite_count': int(np.count_nonzero(~np.isfinite(b))),
                'reference_l2': float(np.linalg.norm(a.ravel())), 'difference_l2': None,
                'global_relative_l2': None, 'maximum_absolute_difference': None,
                'elementwise_violations': None, 'atol': ATOL, 'rtol': RTOL,
                'passes_original_strict_gradient_gate': False}
    difference = b - a
    ref_norm = float(np.linalg.norm(a.ravel()))
    difference_norm = float(np.linalg.norm(difference.ravel()))
    relative = difference_norm / ref_norm if ref_norm else (0.0 if difference_norm == 0 else None)
    violations = int(np.count_nonzero(np.abs(difference) > ATOL + RTOL * np.abs(a)))
    if not all(math.isfinite(x) for x in (ref_norm, difference_norm)):
        raise ValueError('Comparison norm is not finite')
    return {'shape': list(a.shape), 'observed_finite': True, 'observed_nonfinite_count': 0, 'reference_l2': ref_norm,
            'difference_l2': difference_norm, 'global_relative_l2': relative,
            'maximum_absolute_difference': float(np.max(np.abs(difference))),
            'elementwise_violations': violations, 'atol': ATOL, 'rtol': RTOL,
            'passes_original_strict_gradient_gate': violations == 0 and relative is not None and relative <= RTOL}


def compare_fixture(q, k, v, cotangent, fused, fallback, *, ordinary=None, scale=0.0625):
    """Report BF16 ordinary-op reference first, with F64 as a diagnostic."""
    arrays = [np.asarray(value) for value in (q, k, v, cotangent)]
    if [tuple(a.shape) for a in arrays] != [(1, 24, 640, 256), (1, 4, 640, 256),
                                             (1, 4, 640, 256), (1, 24, 640, 256)]:
        raise ValueError('Exact frozen T640, 24Q/4KV fixture required')
    if set(fused) != {'dq', 'dk', 'dv'} or set(fallback) != {'dq', 'dk', 'dv'}:
        raise ValueError('Exact attention-gradient inventory required')
    if ordinary is not None and set(ordinary) != {'dq', 'dk', 'dv'}:
        raise ValueError('Exact ordinary BF16 gradient inventory required')
    _, reference = causal_gqa_vjp(*arrays, scale)
    result = {}
    for name, expected in zip(('dq', 'dk', 'dv'), reference):
        fused_array = np.asarray(fused[name])
        direct_comparison = (array_comparison(fused_array, fallback[name]) if np.isfinite(fused_array).all()
                             else {'available': False, 'reason': 'fused_reference_nonfinite'})
        result[name] = {'fused_vs_f64': array_comparison(expected, fused[name]),
                        'fallback_vs_f64': array_comparison(expected, fallback[name]),
                        'fallback_vs_fused': direct_comparison}
        if ordinary is not None:
            ordinary_array = np.asarray(ordinary[name])
            result[name]['ordinary_bf16_vs_f64'] = array_comparison(expected, ordinary_array)
            result[name]['fused_vs_ordinary_bf16'] = (
                array_comparison(ordinary_array, fused[name]) if np.isfinite(ordinary_array).all()
                else {'available': False, 'reason': 'ordinary_reference_nonfinite'})
            result[name]['fallback_vs_ordinary_bf16'] = (
                array_comparison(ordinary_array, fallback[name]) if np.isfinite(ordinary_array).all()
                else {'available': False, 'reason': 'ordinary_reference_nonfinite'})
    return result
