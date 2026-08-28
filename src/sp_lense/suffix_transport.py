"""Pure offline math for suffix-transported factorial gradients.

The functions in this module deliberately know nothing about models, prompts, answer
labels, or result partitions.  They operate on already captured gradient rows.  This
keeps the proposed suffix-transport diagnostic small enough to test without silently
opening an evaluation set or invoking a model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

SCHEMA_VERSION = "sp_lense.suffix_transport.v1"


def validate_matrix(
    values: Any,
    *,
    field: str,
    row_count: int | None = None,
    width: int | None = None,
) -> np.ndarray:
    """Return a finite contiguous float64 matrix with checked dimensions."""

    if not isinstance(field, str) or not field:
        raise ValueError("field must be a non-empty string")
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{field} must be a non-empty two-dimensional matrix")
    if row_count is not None and matrix.shape[0] != row_count:
        raise ValueError(f"{field} row count differs")
    if width is not None and matrix.shape[1] != width:
        raise ValueError(f"{field} width differs")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{field} must be finite")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def unit_normalize_rows(values: Any, *, field: str = "matrix") -> np.ndarray:
    """Normalize every matrix row independently and reject zero rows."""

    matrix = validate_matrix(values, field=field)
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise ValueError(f"{field} rows must have positive finite norm")
    normalized = matrix / norms[:, None]
    if not np.isfinite(normalized).all():
        raise ValueError(f"{field} normalization produced a non-finite value")
    return np.ascontiguousarray(normalized, dtype=np.float64)


def _positive_scalar(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite positive scalar")
    checked = float(value)
    if not math.isfinite(checked) or checked <= 0.0:
        raise ValueError(f"{field} must be a finite positive scalar")
    return checked


def _head_cosine_threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("minimum_head_cosine must be a finite scalar in [-1, 1]")
    checked = float(value)
    if not math.isfinite(checked) or not -1.0 <= checked <= 1.0:
        raise ValueError("minimum_head_cosine must be a finite scalar in [-1, 1]")
    return checked


@dataclass(frozen=True)
class DualRidgeTransport:
    """A two-head low-rank transport represented in dual coordinates."""

    source_rows: np.ndarray
    head_0_coefficients: np.ndarray
    head_1_coefficients: np.ndarray
    ridge_multiplier: float
    ridge: float
    source_rank: int
    kernel_trace: float
    diagnostics: dict[str, Any]

    @property
    def source_width(self) -> int:
        return int(self.source_rows.shape[1])

    @property
    def target_width(self) -> int:
        return int(self.head_0_coefficients.shape[1])


def fit_dual_ridge_transport(
    source_rows: Any,
    target_head_0_rows: Any,
    target_head_1_rows: Any,
    *,
    ridge_multiplier: float = 0.1,
) -> DualRidgeTransport:
    """Fit two transport heads with one fixed trace/rank-scaled ridge.

    Source and target rows are normalized before fitting.  With unit source rows,
    ``trace(S S^T)`` is the number of examples, but it is still computed explicitly
    and recorded.  The ridge is fixed as

    ``ridge_multiplier * trace(S S^T) / rank(S)``.
    """

    multiplier = _positive_scalar(ridge_multiplier, field="ridge_multiplier")
    source = unit_normalize_rows(source_rows, field="source_rows")
    head_0 = unit_normalize_rows(target_head_0_rows, field="target_head_0_rows")
    head_1 = unit_normalize_rows(target_head_1_rows, field="target_head_1_rows")
    if head_0.shape[0] != source.shape[0] or head_1.shape[0] != source.shape[0]:
        raise ValueError("source and target heads must have the same row count")
    if head_0.shape[1] != head_1.shape[1]:
        raise ValueError("target heads must have the same width")

    rank = int(np.linalg.matrix_rank(source))
    if rank <= 0:
        raise ValueError("source_rows must have positive rank")
    kernel = np.ascontiguousarray(source @ source.T, dtype=np.float64)
    kernel_trace = float(np.trace(kernel))
    if not math.isfinite(kernel_trace) or kernel_trace <= 0.0:
        raise ValueError("source kernel must have positive finite trace")
    ridge = multiplier * kernel_trace / rank
    regularized = kernel + ridge * np.eye(kernel.shape[0], dtype=np.float64)
    try:
        coefficients_0 = np.linalg.solve(regularized, head_0)
        coefficients_1 = np.linalg.solve(regularized, head_1)
    except np.linalg.LinAlgError as exc:
        raise ValueError("regularized source kernel could not be solved") from exc
    if not np.isfinite(coefficients_0).all() or not np.isfinite(coefficients_1).all():
        raise ValueError("dual ridge fit produced non-finite coefficients")

    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "row_count": int(source.shape[0]),
        "source_width": int(source.shape[1]),
        "target_width": int(head_0.shape[1]),
        "source_rank": rank,
        "kernel_trace": kernel_trace,
        "ridge_multiplier": multiplier,
        "ridge": ridge,
        "ridge_rule": "multiplier_times_source_kernel_trace_divided_by_source_rank",
        "regularized_kernel_condition_number": float(np.linalg.cond(regularized)),
        "source_rows_unit_normalized": True,
        "target_rows_unit_normalized": True,
    }
    return DualRidgeTransport(
        source_rows=source,
        head_0_coefficients=np.ascontiguousarray(coefficients_0, dtype=np.float64),
        head_1_coefficients=np.ascontiguousarray(coefficients_1, dtype=np.float64),
        ridge_multiplier=multiplier,
        ridge=ridge,
        source_rank=rank,
        kernel_trace=kernel_trace,
        diagnostics=diagnostics,
    )


def predict_dual_ridge_transport(
    model: DualRidgeTransport,
    source_rows: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict both transported heads for unit-normalized source rows."""

    if not isinstance(model, DualRidgeTransport):
        raise TypeError("model must be a DualRidgeTransport")
    source = unit_normalize_rows(source_rows, field="source_rows")
    if source.shape[1] != model.source_width:
        raise ValueError("source_rows width differs from fitted transport")
    kernel_cross = source @ model.source_rows.T
    predicted_0 = kernel_cross @ model.head_0_coefficients
    predicted_1 = kernel_cross @ model.head_1_coefficients
    if not np.isfinite(predicted_0).all() or not np.isfinite(predicted_1).all():
        raise ValueError("dual ridge prediction produced a non-finite value")
    return (
        np.ascontiguousarray(predicted_0, dtype=np.float64),
        np.ascontiguousarray(predicted_1, dtype=np.float64),
    )


def robust_two_head_unit_bisector(
    head_0_rows: Any,
    head_1_rows: Any,
    *,
    minimum_head_cosine: float = 0.0,
) -> dict[str, Any]:
    """Return the unit maximin bisector of two compatible direction heads.

    For two unit vectors, their normalized sum maximizes the smaller dot product
    within their span.  Rows below the frozen cosine threshold fail closed rather
    than receiving an arbitrary orientation.
    """

    threshold = _head_cosine_threshold(minimum_head_cosine)
    head_0 = unit_normalize_rows(head_0_rows, field="head_0_rows")
    head_1 = unit_normalize_rows(head_1_rows, field="head_1_rows")
    if head_0.shape != head_1.shape:
        raise ValueError("two transport heads must have the same shape")
    cosines = np.sum(head_0 * head_1, axis=1)
    cosines = np.clip(cosines, -1.0, 1.0)
    incompatible = np.flatnonzero(cosines < threshold)
    if incompatible.size:
        joined = ",".join(map(str, incompatible.tolist()))
        raise ValueError(f"transport heads fail the minimum cosine at rows {joined}")
    sums = head_0 + head_1
    directions = unit_normalize_rows(sums, field="head_bisector_sums")
    alignment_0 = np.sum(directions * head_0, axis=1)
    alignment_1 = np.sum(directions * head_1, axis=1)
    worst = np.minimum(alignment_0, alignment_1)
    if np.any(worst <= 0.0):
        raise ValueError("transport-head bisector lacks positive alignment")
    return {
        "directions": directions,
        "head_cosines": np.ascontiguousarray(cosines, dtype=np.float64),
        "head_0_alignments": np.ascontiguousarray(alignment_0, dtype=np.float64),
        "head_1_alignments": np.ascontiguousarray(alignment_1, dtype=np.float64),
        "worst_head_alignments": np.ascontiguousarray(worst, dtype=np.float64),
        "minimum_head_cosine": threshold,
    }


def _scenario_ids(values: Sequence[Any], *, row_count: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("scenario_ids must be a sequence of identifiers")
    identifiers = tuple(values)
    if len(identifiers) != row_count:
        raise ValueError("scenario_ids must match the matrix row count")
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError("scenario_ids must contain non-empty strings")
    if len(set(identifiers)) < 2:
        raise ValueError("leave-one-scenario-out requires at least two scenarios")
    return identifiers


def _descriptive(values: np.ndarray) -> dict[str, float]:
    checked = np.asarray(values, dtype=np.float64)
    if checked.ndim != 1 or checked.size < 1 or not np.isfinite(checked).all():
        raise ValueError("metric values must be a finite non-empty vector")
    return {
        "minimum": float(np.min(checked)),
        "mean": float(np.mean(checked)),
        "median": float(np.median(checked)),
        "maximum": float(np.max(checked)),
    }


def transport_metric_summary(
    predicted_head_0_rows: Any,
    predicted_head_1_rows: Any,
    observed_head_0_rows: Any,
    observed_head_1_rows: Any,
    *,
    scenario_ids: Sequence[Any] | None = None,
    minimum_head_cosine: float = 0.0,
    positive_alignment_threshold: float = 0.0,
) -> dict[str, Any]:
    """Summarize how a predicted bisector aligns with two observed order heads."""

    if isinstance(positive_alignment_threshold, bool) or not isinstance(
        positive_alignment_threshold, (int, float)
    ):
        raise TypeError("positive_alignment_threshold must be a finite scalar")
    positive_threshold = float(positive_alignment_threshold)
    if not math.isfinite(positive_threshold) or not -1.0 <= positive_threshold <= 1.0:
        raise ValueError("positive_alignment_threshold must lie in [-1, 1]")

    predicted_0 = validate_matrix(predicted_head_0_rows, field="predicted_head_0_rows")
    predicted_1 = validate_matrix(
        predicted_head_1_rows,
        field="predicted_head_1_rows",
        row_count=predicted_0.shape[0],
        width=predicted_0.shape[1],
    )
    observed_0 = unit_normalize_rows(observed_head_0_rows, field="observed_head_0_rows")
    observed_1 = unit_normalize_rows(observed_head_1_rows, field="observed_head_1_rows")
    if observed_0.shape != predicted_0.shape or observed_1.shape != predicted_0.shape:
        raise ValueError("predicted and observed transport heads must have the same shape")
    bisector = robust_two_head_unit_bisector(
        predicted_0,
        predicted_1,
        minimum_head_cosine=minimum_head_cosine,
    )
    directions = bisector["directions"]
    alignment_0 = np.sum(directions * observed_0, axis=1)
    alignment_1 = np.sum(directions * observed_1, axis=1)
    worst = np.minimum(alignment_0, alignment_1)
    both_positive = (alignment_0 > positive_threshold) & (alignment_1 > positive_threshold)

    if scenario_ids is None:
        identifiers = tuple(f"row_{index}" for index in range(predicted_0.shape[0]))
    else:
        identifiers = tuple(scenario_ids)
        if len(identifiers) != predicted_0.shape[0]:
            raise ValueError("scenario_ids must match the matrix row count")
        if any(not isinstance(value, str) or not value for value in identifiers):
            raise ValueError("scenario_ids must contain non-empty strings")
    ordered_scenarios = tuple(dict.fromkeys(identifiers))
    scenario_rows = []
    for scenario_id in ordered_scenarios:
        indices = [index for index, value in enumerate(identifiers) if value == scenario_id]
        complete = bool(np.all(both_positive[indices]))
        scenario_rows.append(
            {
                "scenario_id": scenario_id,
                "row_count": len(indices),
                "both_order_positive_count": int(np.sum(both_positive[indices])),
                "complete": complete,
                "minimum_worst_order_alignment": float(np.min(worst[indices])),
            }
        )
    complete_scenarios = sum(int(row["complete"]) for row in scenario_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "row_count": int(predicted_0.shape[0]),
        "scenario_count": len(scenario_rows),
        "both_order_positive_count": int(np.sum(both_positive)),
        "both_order_positive_fraction": float(np.mean(both_positive)),
        "complete_scenario_count": complete_scenarios,
        "complete_scenario_fraction": complete_scenarios / len(scenario_rows),
        "positive_alignment_threshold": positive_threshold,
        "predicted_head_cosine": _descriptive(bisector["head_cosines"]),
        "observed_head_0_alignment": _descriptive(alignment_0),
        "observed_head_1_alignment": _descriptive(alignment_1),
        "worst_order_alignment": _descriptive(worst),
        "scenario_rows": scenario_rows,
        "head_0_alignments": np.ascontiguousarray(alignment_0, dtype=np.float64),
        "head_1_alignments": np.ascontiguousarray(alignment_1, dtype=np.float64),
        "worst_order_alignments": np.ascontiguousarray(worst, dtype=np.float64),
        "both_order_positive": np.ascontiguousarray(both_positive, dtype=np.bool_),
        "directions": directions,
    }


def leave_one_scenario_out_transport(
    source_rows: Any,
    target_head_0_rows: Any,
    target_head_1_rows: Any,
    scenario_ids: Sequence[Any],
    *,
    ridge_multiplier: float = 0.1,
    minimum_head_cosine: float = 0.0,
    positive_alignment_threshold: float = 0.0,
) -> dict[str, Any]:
    """Fit and score dual transport while excluding every held-out scenario at once."""

    source = unit_normalize_rows(source_rows, field="source_rows")
    target_0 = unit_normalize_rows(target_head_0_rows, field="target_head_0_rows")
    target_1 = unit_normalize_rows(target_head_1_rows, field="target_head_1_rows")
    if target_0.shape[0] != source.shape[0] or target_1.shape[0] != source.shape[0]:
        raise ValueError("source and target heads must have the same row count")
    if target_0.shape[1] != target_1.shape[1]:
        raise ValueError("target heads must have the same width")
    identifiers = _scenario_ids(scenario_ids, row_count=source.shape[0])
    ordered_scenarios = tuple(dict.fromkeys(identifiers))
    predicted_0 = np.empty_like(target_0)
    predicted_1 = np.empty_like(target_1)
    folds = []
    for held_out in ordered_scenarios:
        test_indices = np.asarray(
            [index for index, value in enumerate(identifiers) if value == held_out],
            dtype=np.int64,
        )
        train_indices = np.asarray(
            [index for index, value in enumerate(identifiers) if value != held_out],
            dtype=np.int64,
        )
        if test_indices.size == 0 or train_indices.size == 0:
            raise ValueError("every LOSO fold must have train and held-out rows")
        model = fit_dual_ridge_transport(
            source[train_indices],
            target_0[train_indices],
            target_1[train_indices],
            ridge_multiplier=ridge_multiplier,
        )
        fold_0, fold_1 = predict_dual_ridge_transport(model, source[test_indices])
        predicted_0[test_indices] = fold_0
        predicted_1[test_indices] = fold_1
        training_scenarios = tuple(
            dict.fromkeys(identifiers[index] for index in train_indices.tolist())
        )
        if held_out in training_scenarios:
            raise RuntimeError("LOSO scenario leaked into its training fold")
        folds.append(
            {
                "held_out_scenario": held_out,
                "held_out_indices": test_indices.tolist(),
                "training_indices": train_indices.tolist(),
                "training_scenarios": list(training_scenarios),
                "training_row_count": int(train_indices.size),
                "held_out_row_count": int(test_indices.size),
                "fit_diagnostics": model.diagnostics,
            }
        )
    metrics = transport_metric_summary(
        predicted_0,
        predicted_1,
        target_0,
        target_1,
        scenario_ids=identifiers,
        minimum_head_cosine=minimum_head_cosine,
        positive_alignment_threshold=positive_alignment_threshold,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "ridge_multiplier": float(ridge_multiplier),
        "minimum_head_cosine": float(minimum_head_cosine),
        "predicted_head_0_rows": predicted_0,
        "predicted_head_1_rows": predicted_1,
        "directions": metrics["directions"],
        "folds": folds,
        "metrics": metrics,
    }


__all__ = [
    "SCHEMA_VERSION",
    "DualRidgeTransport",
    "fit_dual_ridge_transport",
    "leave_one_scenario_out_transport",
    "predict_dual_ridge_transport",
    "robust_two_head_unit_bisector",
    "transport_metric_summary",
    "unit_normalize_rows",
    "validate_matrix",
]
