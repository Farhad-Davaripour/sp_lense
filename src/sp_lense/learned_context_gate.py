"""Small, auditable linear gate for prompt-conditioned activation steering."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

import numpy as np


SCHEMA_VERSION = "sp_lense.learned_context_gate.v1"


def _matrix(values: Any, *, field: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError(f"{field} must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{field} must be finite")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError(f"{field} rows must have positive norm")
    return matrix / norms[:, None]


def _labels(values: Sequence[Any], *, length: int) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1 or labels.shape[0] != length:
        raise ValueError("labels must be one-dimensional and match activations")
    if not np.isin(labels, [False, True, 0, 1]).all():
        raise ValueError("labels must be binary")
    binary = labels.astype(bool)
    if binary.all() or (~binary).all():
        raise ValueError("both label classes are required")
    return binary


def fit_balanced_ridge_gate(
    activations: Any,
    labels: Sequence[Any],
    *,
    ridge: float,
) -> dict[str, Any]:
    """Fit a unit-normalized, class-balanced ridge least-squares gate."""

    x = _matrix(activations, field="activations")
    y_bool = _labels(labels, length=x.shape[0])
    if isinstance(ridge, bool) or not isinstance(ridge, (int, float)):
        raise TypeError("ridge must be a finite positive scalar")
    ridge_value = float(ridge)
    if not math.isfinite(ridge_value) or ridge_value <= 0.0:
        raise ValueError("ridge must be a finite positive scalar")

    y = np.where(y_bool, 1.0, -1.0)
    positive_count = int(y_bool.sum())
    negative_count = int((~y_bool).sum())
    sample_weights = np.where(
        y_bool,
        0.5 / positive_count,
        0.5 / negative_count,
    )
    augmented = np.column_stack((x, np.ones(x.shape[0], dtype=np.float64)))
    root_weights = np.sqrt(sample_weights)
    weighted_x = augmented * root_weights[:, None]
    weighted_y = y * root_weights
    gram = weighted_x @ weighted_x.T
    scale = float(np.trace(gram) / gram.shape[0])
    regularizer = ridge_value * scale
    coefficients = weighted_x.T @ np.linalg.solve(
        gram + regularizer * np.eye(gram.shape[0], dtype=np.float64),
        weighted_y,
    )
    weights = np.ascontiguousarray(coefficients[:-1], dtype=np.float64)
    bias = float(coefficients[-1])
    payload = np.concatenate((weights, np.asarray([bias], dtype=np.float64)))
    return {
        "schema_version": SCHEMA_VERSION,
        "weights": weights,
        "bias": bias,
        "ridge": ridge_value,
        "regularizer": regularizer,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "coefficient_sha256": hashlib.sha256(payload.astype("<f8").tobytes()).hexdigest(),
    }


def score_balanced_ridge_gate(model: dict[str, Any], activations: Any) -> np.ndarray:
    """Return signed gate scores; larger values are more target-like."""

    x = _matrix(activations, field="activations")
    weights = np.asarray(model["weights"], dtype=np.float64)
    if weights.ndim != 1 or weights.shape[0] != x.shape[1] or not np.isfinite(weights).all():
        raise ValueError("gate weights do not match activation width")
    bias = float(model["bias"])
    if not math.isfinite(bias):
        raise ValueError("gate bias must be finite")
    return x @ weights + bias


def conservative_separating_threshold(
    scores: Sequence[float], labels: Sequence[Any]
) -> dict[str, float]:
    """Choose the midpoint only when development scores strictly separate."""

    score_array = np.asarray(scores, dtype=np.float64)
    label_array = _labels(labels, length=score_array.shape[0])
    if score_array.ndim != 1 or not np.isfinite(score_array).all():
        raise ValueError("scores must be a finite one-dimensional vector")
    minimum_positive = float(score_array[label_array].min())
    maximum_negative = float(score_array[~label_array].max())
    if minimum_positive <= maximum_negative:
        raise ValueError("development scores are not strictly separable")
    return {
        "threshold": 0.5 * (minimum_positive + maximum_negative),
        "minimum_positive_score": minimum_positive,
        "maximum_negative_score": maximum_negative,
        "separation_margin": minimum_positive - maximum_negative,
    }


def binary_gate_metrics(
    scores: Sequence[float], labels: Sequence[Any], *, threshold: float
) -> dict[str, Any]:
    """Return transparent binary confusion counts and balanced accuracy."""

    score_array = np.asarray(scores, dtype=np.float64)
    label_array = _labels(labels, length=score_array.shape[0])
    threshold_value = float(threshold)
    if score_array.ndim != 1 or not np.isfinite(score_array).all():
        raise ValueError("scores must be a finite one-dimensional vector")
    if not math.isfinite(threshold_value):
        raise ValueError("threshold must be finite")
    predicted = score_array > threshold_value
    true_positive = int(np.sum(predicted & label_array))
    false_negative = int(np.sum(~predicted & label_array))
    true_negative = int(np.sum(~predicted & ~label_array))
    false_positive = int(np.sum(predicted & ~label_array))
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    return {
        "threshold": threshold_value,
        "true_positive": true_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "all_correct": false_negative == 0 and false_positive == 0,
    }
