"""Independent ordinary-ops causal GQA reference; no model loaded on import."""
import math


def ordinary_gqa_probabilities(runtime, q, k, scale, *, precision):
    if (len(q.shape) != 4 or len(k.shape) != 4 or q.shape[0] != k.shape[0]
            or q.shape[-1] != k.shape[-1] or q.shape[-2] != k.shape[-2]
            or q.shape[1] % k.shape[1] != 0 or precision not in ('bf16', 'f32')
            or not isinstance(scale, float) or not math.isfinite(scale) or scale <= 0):
        raise ValueError('Exact same-length causal GQA geometry and scale required')
    dtype = runtime.bfloat16 if precision == 'bf16' else runtime.float32
    q, k = q.astype(dtype), k.astype(dtype)
    groups = q.shape[1] // k.shape[1]
    # Match the native fallback's unflatten/broadcast order, including the
    # grouped-K/V reduction order in automatic differentiation.
    q = runtime.reshape(q * runtime.array(scale, dtype=dtype),
                        (q.shape[0], k.shape[1], groups, q.shape[2], q.shape[3]))
    k = runtime.expand_dims(k, 2)
    scores = runtime.matmul(q, k.transpose(0, 1, 2, 4, 3))
    positions = runtime.arange(q.shape[-2])
    causal = positions[:, None] >= positions[None, :]
    scores = runtime.where(causal[None, None, None, :, :], scores,
                           runtime.array(runtime.finfo(dtype).min, dtype=dtype))
    return runtime.softmax(scores, axis=-1, precise=True)


def ordinary_gqa(runtime, q, k, v, scale, *, precision):
    if (len(q.shape) != 4 or len(k.shape) != 4 or len(v.shape) != 4
            or q.shape[0] != k.shape[0] or k.shape != v.shape
            or q.shape[-1] != k.shape[-1] or q.shape[-2] != k.shape[-2]
            or q.shape[1] % k.shape[1] != 0 or precision not in ('bf16', 'f32')
            or not isinstance(scale, float) or not math.isfinite(scale) or scale <= 0):
        raise ValueError('Exact same-length causal GQA geometry and scale required')
    dtype = runtime.bfloat16 if precision == 'bf16' else runtime.float32
    v = runtime.expand_dims(v.astype(dtype), 2)
    # Scale before matmul, matching the ordinary BF16 reference path.
    probabilities = ordinary_gqa_probabilities(runtime, q, k, scale, precision=precision)
    output = runtime.matmul(probabilities, v)
    return runtime.reshape(output, (q.shape[0], q.shape[1], q.shape[2], q.shape[3]))


def compare_arrays(runtime, reference, observed, *, atol=1e-6, rtol=1e-3):
    if tuple(reference.shape) != tuple(observed.shape) or not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError('Same gradient geometry and finite tolerances required')
    a, b = reference.astype(runtime.float32), observed.astype(runtime.float32)
    finite = runtime.logical_and(runtime.all(runtime.isfinite(a)), runtime.all(runtime.isfinite(b)))
    delta = runtime.abs(b - a)
    violations = runtime.sum(delta > (atol + rtol * runtime.abs(a)))
    ref_sq = runtime.sum(a * a)
    diff_sq = runtime.sum((b - a) * (b - a))
    max_abs = runtime.max(delta)
    runtime.eval(finite, violations, ref_sq, diff_sq, max_abs)
    if not bool(finite.item()):
        raise ValueError('Nonfinite gradient comparison')
    ref_val, diff_val = float(ref_sq.item()), float(diff_sq.item())
    relative = math.sqrt(diff_val / ref_val) if ref_val > 0 else (0.0 if diff_val == 0 else None)
    return {'shape': list(reference.shape), 'atol': atol, 'rtol': rtol,
            'allclose_element_violations': int(violations.item()),
            'global_relative_l2': relative, 'maximum_absolute_difference': float(max_abs.item()),
            'reference_l2': math.sqrt(ref_val), 'difference_l2': math.sqrt(diff_val),
            'passed_original_gate': int(violations.item()) == 0 and relative is not None and relative <= .001}
