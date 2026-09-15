"""Run a synthetic CPU-small feature extension without changing frozen study data."""

import json

from .feature_strategies import build_features


def main():
    import numpy as np

    pca = np.array([[1.0, 2.0], [3.0, 4.0]])
    jlens = np.arange(36.0).reshape(2, 18)
    features = build_features("centered_jlens_norm", pca, jlens, [10.0, 20.0])
    print(
        json.dumps(
            {
                "strategy": "centered_jlens_norm",
                "shape": list(features.shape),
                "finite": bool(np.isfinite(features).all()),
            }
        )
    )


if __name__ == "__main__":
    main()
