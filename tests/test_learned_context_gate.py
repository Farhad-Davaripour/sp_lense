from __future__ import annotations

import numpy as np
import pytest

from sp_lense.learned_context_gate import (
    binary_gate_metrics,
    conservative_separating_threshold,
    fit_balanced_ridge_gate,
    score_balanced_ridge_gate,
)


def test_balanced_ridge_gate_separates_simple_data() -> None:
    activations = np.asarray([[2.0, 0.0], [1.0, 0.1], [-2.0, 0.0], [-1.0, -0.1]])
    labels = [True, True, False, False]
    model = fit_balanced_ridge_gate(activations, labels, ridge=1e-3)
    scores = score_balanced_ridge_gate(model, activations)
    threshold = conservative_separating_threshold(scores, labels)
    metrics = binary_gate_metrics(scores, labels, threshold=threshold["threshold"])
    assert metrics["all_correct"] is True
    assert len(model["coefficient_sha256"]) == 64


def test_conservative_threshold_fails_closed_on_overlap() -> None:
    with pytest.raises(ValueError, match="not strictly separable"):
        conservative_separating_threshold([0.2, 0.1], [False, True])


def test_gate_rejects_degenerate_labels() -> None:
    with pytest.raises(ValueError, match="both label classes"):
        fit_balanced_ridge_gate(np.eye(2), [True, True], ridge=0.1)
