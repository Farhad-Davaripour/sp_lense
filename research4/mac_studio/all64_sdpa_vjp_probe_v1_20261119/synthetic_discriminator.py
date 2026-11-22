"""Deterministic high-score toy fixture for detecting a VJP switch effect.

It never loads Qwen or user data.  NumPy randomness is local and does not
touch MLX model RNG.  The same saved BF16 tensors and cotangent are used by
the separately owned fused and fallback processes.
"""
import numpy as np


def build(runtime):
    rng = np.random.default_rng(20261119)
    q = (rng.standard_normal((1, 2, 32, 64)) * 30).astype(np.float32)
    k = (rng.standard_normal((1, 1, 32, 64)) * 30).astype(np.float32)
    v = rng.standard_normal((1, 1, 32, 64)).astype(np.float32)
    cotangent = rng.standard_normal((1, 2, 32, 64)).astype(np.float32)
    # The final query can attend to key 0; alignment ensures a high causal QK.
    k[0, 0, 0] = q[0, 0, -1]
    result = {name: runtime.array(value).astype(runtime.bfloat16) for name, value in
              [('q', q), ('k', k), ('v', v), ('cotangent', cotangent)]}
    runtime.eval(list(result.values()))
    return result


def causal_score_max(runtime, fixture):
    q = np.array(fixture['q'].astype(runtime.float32)).astype(np.float64)
    k = np.repeat(np.array(fixture['k'].astype(runtime.float32)).astype(np.float64), 2, axis=1)
    scores = np.abs(q @ k.transpose(0, 1, 3, 2))
    mask = np.tril(np.ones((32, 32), dtype=bool))
    return float(np.max(np.where(mask[None, None, :, :], scores, 0.0)))
