"""Independent CPU float64 causal GQA forward and analytical VJP.

No MLX, model, or dataset is imported here.  Repeated query groups contribute
summed gradients to their shared key/value head. Inputs/cotangent must already
be authenticated; output is an independent high-precision reference, not an
assertion that BF16 must be bitwise identical.
"""
import math
import numpy as np


def causal_gqa_vjp(q, k, v, cotangent, scale):
    q, k, v, cotangent = (np.asarray(x, dtype=np.float64) for x in (q, k, v, cotangent))
    if (q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or cotangent.shape != q.shape
            or k.shape != v.shape or q.shape[0] != k.shape[0]
            or q.shape[2:] != k.shape[2:] or q.shape[1] % k.shape[1] != 0
            or not isinstance(scale, float) or not math.isfinite(scale) or scale <= 0
            or not all(np.isfinite(x).all() for x in (q, k, v, cotangent))):
        raise ValueError('Finite same-length causal GQA arrays and scale required')
    batch, heads, length, dim = q.shape
    kv_heads = k.shape[1]
    repeats = heads // kv_heads
    keys = np.repeat(k, repeats, axis=1)
    values = np.repeat(v, repeats, axis=1)
    scores = np.matmul(q, keys.swapaxes(-1, -2)) * scale
    mask = np.arange(length)[:, None] >= np.arange(length)[None, :]
    scores = np.where(mask[None, None, :, :], scores, -np.inf)
    scores -= np.max(scores, axis=-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    output = np.matmul(probabilities, values)
    dv_repeated = np.matmul(probabilities.swapaxes(-1, -2), cotangent)
    dp = np.matmul(cotangent, values.swapaxes(-1, -2))
    ds = probabilities * (dp - np.sum(dp * probabilities, axis=-1, keepdims=True))
    dq = np.matmul(ds, keys) * scale
    dk_repeated = np.matmul(ds.swapaxes(-1, -2), q) * scale
    dk = dk_repeated.reshape(batch, kv_heads, repeats, length, dim).sum(axis=2)
    dv = dv_repeated.reshape(batch, kv_heads, repeats, length, dim).sum(axis=2)
    if not all(np.isfinite(x).all() for x in (output, dq, dk, dv)):
        raise ValueError('Nonfinite CPU reference result')
    return output, (dq, dk, dv)
