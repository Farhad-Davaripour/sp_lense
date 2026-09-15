"""Feature strategies for new experiments, independent of frozen score replay.

Inputs: PCA scores (n, k), J-lens scores (n, 18) in layer-major order
(three layers, six concepts), and activation norms (n,) or (n, 1).
Outputs: finite floating arrays (n, d), preserving row order. No fitting occurs.
"""


def build_features(strategy, pca, jlens, norms):
    import numpy as np

    pca, jlens, norms = (np.asarray(x, dtype=float) for x in (pca, jlens, norms))
    if pca.ndim != 2 or not pca.shape[0] or not pca.shape[1]:
        raise ValueError("pca must have shape (n, k) with n and k positive")
    n = pca.shape[0]
    if jlens.shape != (n, 18) or norms.shape not in {(n,), (n, 1)}:
        raise ValueError("expected jlens (n, 18) and norms (n,) or (n, 1)")
    if not all(np.isfinite(x).all() for x in (pca, jlens, norms)):
        raise ValueError("features must be finite")
    norms = norms.reshape(n, 1)
    if strategy == "pca_only":
        return pca.copy()
    if strategy == "raw_jlens":
        return np.concatenate((pca, jlens), axis=1)
    if strategy == "centered_jlens_norm":
        layers = jlens.reshape(n, 3, 6)
        centered = (layers - layers.mean(axis=2, keepdims=True)).reshape(n, 18)
        return np.concatenate((pca, centered, norms), axis=1)
    raise ValueError(f"Unknown feature strategy: {strategy}")
