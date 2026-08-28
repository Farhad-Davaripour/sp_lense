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
CELL_ORDER = ("SP", "OP", "ST", "OT")
PFIT_MINIMUM_PAIR_COSINE = -0.99


class SuffixTransportIneligible(ValueError):
    """A prospective construction failed a declared geometric eligibility gate."""

    def __init__(self, message: str, *, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


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


def order_even_odd_directions(
    choice_head_0_rows: Any,
    choice_head_1_rows: Any,
    *,
    minimum_head_cosine: float = PFIT_MINIMUM_PAIR_COSINE,
) -> dict[str, Any]:
    """Decompose normalized answer-order gradients into even and odd components.

    The even component is the robust unit bisector.  The odd nuisance is the unit
    direction of ``unit_head_0 - unit_head_1``.  Its pre-normalization magnitude is
    retained separately as a diagnostic.  PFIT's default ``-0.99`` compatibility
    floor was selected on opened development after the earlier ST-FG failure; it
    rejects near-antipodal heads while allowing a stable negative-cosine interface.
    """

    head_0 = unit_normalize_rows(choice_head_0_rows, field="choice_head_0_rows")
    head_1 = unit_normalize_rows(choice_head_1_rows, field="choice_head_1_rows")
    if head_0.shape != head_1.shape:
        raise ValueError("choice order heads must have the same shape")
    bisector = robust_two_head_unit_bisector(
        head_0,
        head_1,
        minimum_head_cosine=minimum_head_cosine,
    )
    odd_raw = 0.5 * (head_0 - head_1)
    odd_norms = np.linalg.norm(odd_raw, axis=1)
    odd = unit_normalize_rows(odd_raw, field="order_odd_nuisance_rows")
    return {
        "even_directions": bisector["directions"],
        "odd_nuisance_rows": np.ascontiguousarray(odd, dtype=np.float64),
        "odd_nuisance_norms": np.ascontiguousarray(odd_norms, dtype=np.float64),
        "head_cosines": bisector["head_cosines"],
        "minimum_head_cosine": float(minimum_head_cosine),
    }


def _finite_vector(values: Any, *, field: str, width: int | None = None) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if vector.ndim != 1 or vector.size < 1:
        raise ValueError(f"{field} must be a non-empty one-dimensional vector")
    if width is not None and vector.size != width:
        raise ValueError(f"{field} width differs")
    if not np.isfinite(vector).all():
        raise ValueError(f"{field} must be finite")
    return np.ascontiguousarray(vector, dtype=np.float64)


def _fraction(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite scalar in [0, 1]")
    checked = float(value)
    if not math.isfinite(checked) or not 0.0 <= checked <= 1.0:
        raise ValueError(f"{field} must be a finite scalar in [0, 1]")
    return checked


def _nonnegative_scalar(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite non-negative scalar")
    checked = float(value)
    if not math.isfinite(checked) or checked < 0.0:
        raise ValueError(f"{field} must be a finite non-negative scalar")
    return checked


def exact_nuisance_projection(
    target: Any,
    nuisance_rows: Any,
    *,
    minimum_retained_fraction: float = 0.05,
    svd_rtol: float = 1e-10,
    svd_atol: float = 1e-12,
) -> dict[str, Any]:
    """Project a target exactly out of the measured nuisance row span."""

    minimum_retained = _fraction(
        minimum_retained_fraction,
        field="minimum_retained_fraction",
    )
    relative_tolerance = _nonnegative_scalar(svd_rtol, field="svd_rtol")
    absolute_tolerance = _nonnegative_scalar(svd_atol, field="svd_atol")
    target_vector = _finite_vector(target, field="target")
    target_norm = float(np.linalg.norm(target_vector))
    if target_norm <= 0.0:
        raise ValueError("target must have positive norm")
    nuisance = unit_normalize_rows(nuisance_rows, field="nuisance_rows")
    if nuisance.shape[1] != target_vector.size:
        raise ValueError("nuisance_rows width differs from target")
    _, singular_values, right_vectors = np.linalg.svd(nuisance, full_matrices=False)
    maximum_singular = float(singular_values[0]) if singular_values.size else 0.0
    cutoff = max(absolute_tolerance, relative_tolerance * maximum_singular)
    rank = int(np.sum(singular_values > cutoff))
    basis = np.ascontiguousarray(right_vectors[:rank], dtype=np.float64)
    projected = target_vector.copy()
    if rank:
        projected = projected - basis.T @ (basis @ projected)
    projected_norm = float(np.linalg.norm(projected))
    retained_fraction = projected_norm / target_norm
    numerical_floor = 256.0 * np.finfo(np.float64).eps * (1.0 + target_norm)
    if projected_norm <= numerical_floor or retained_fraction < minimum_retained:
        raise SuffixTransportIneligible(
            "exact nuisance projection retained less than the minimum target fraction",
            diagnostics={
                "target_norm_before_projection": target_norm,
                "target_norm_after_projection": projected_norm,
                "retained_target_fraction": retained_fraction,
                "minimum_retained_fraction": minimum_retained,
                "nuisance_row_count": int(nuisance.shape[0]),
                "nuisance_rank": rank,
                "svd_cutoff": cutoff,
            },
        )
    direction = np.ascontiguousarray(projected / projected_norm, dtype=np.float64)
    nuisance_projections = nuisance @ direction
    maximum_abs_projection = float(np.max(np.abs(nuisance_projections)))
    projection_tolerance = 1e-10 * (1.0 + float(np.linalg.norm(direction)))
    if maximum_abs_projection > projection_tolerance:
        raise RuntimeError("exact nuisance projection left measurable nuisance alignment")
    return {
        "direction": direction,
        "unprotected_direction": np.ascontiguousarray(
            target_vector / target_norm,
            dtype=np.float64,
        ),
        "basis": basis,
        "nuisance_projections": np.ascontiguousarray(
            nuisance_projections,
            dtype=np.float64,
        ),
        "nuisance_row_count": int(nuisance.shape[0]),
        "nuisance_rank": rank,
        "target_norm_before_projection": target_norm,
        "target_norm_after_projection": projected_norm,
        "retained_target_fraction": retained_fraction,
        "minimum_retained_fraction": minimum_retained,
        "maximum_abs_nuisance_projection": maximum_abs_projection,
        "svd_cutoff": cutoff,
    }


def _cell_rows(values: Any, *, field: str) -> np.ndarray:
    try:
        rows = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if rows.ndim != 3 or rows.shape[0] != 2 or rows.shape[1] != len(CELL_ORDER):
        raise ValueError(f"{field} must have shape [2, 4, width]")
    if rows.shape[2] < 1 or not np.isfinite(rows).all():
        raise ValueError(f"{field} must be finite with positive width")
    normalized = unit_normalize_rows(rows.reshape(-1, rows.shape[2]), field=field)
    return np.ascontiguousarray(normalized.reshape(rows.shape), dtype=np.float64)


def _unit_direction(values: Any, *, field: str) -> np.ndarray:
    vector = _finite_vector(values, field=field)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{field} must have positive norm")
    return np.ascontiguousarray(vector / norm, dtype=np.float64)


def construct_cell_interface_directions(
    predicted_even_cell_rows: Any,
    predicted_odd_cell_rows: Any,
    *,
    minimum_retained_fraction: float = 0.05,
    svd_rtol: float = 1e-10,
    svd_atol: float = 1e-12,
) -> dict[str, Any]:
    """Construct protected and equal-access dynamic directions for one scenario.

    Both inputs use assignment-major order and ``CELL_ORDER`` within each assignment.
    Every nuisance interface is predicted from option-free held-out semantic rows;
    held-out observed choice gradients are not inputs to this construction.
    """

    even_cells = _cell_rows(predicted_even_cell_rows, field="predicted_even_cell_rows")
    odd_cells = _cell_rows(predicted_odd_cell_rows, field="predicted_odd_cell_rows")
    if even_cells.shape != odd_cells.shape:
        raise ValueError("predicted even and odd cell interfaces must have the same shape")
    target = np.mean(even_cells[:, 0, :], axis=0)
    unprotected = _unit_direction(target, field="predicted_self_permanent_mean")

    nuisance_rows = []
    nuisance_manifest = []
    for assignment in range(2):
        for cell_index, cell_name in enumerate(CELL_ORDER[1:], start=1):
            nuisance_rows.append(even_cells[assignment, cell_index])
            nuisance_manifest.append(
                {
                    "kind": "predicted_off_target_even_cell",
                    "assignment": assignment,
                    "cell": cell_name,
                }
            )
    for assignment in range(2):
        for cell_index, cell_name in enumerate(CELL_ORDER):
            nuisance_rows.append(odd_cells[assignment, cell_index])
            nuisance_manifest.append(
                {
                    "kind": "predicted_order_odd_cell",
                    "assignment": assignment,
                    "cell": cell_name,
                }
            )
    numerical_floor = 256.0 * np.finfo(np.float64).eps
    name_odd = even_cells[0, 0] - even_cells[1, 0]
    if float(np.linalg.norm(name_odd)) > numerical_floor:
        nuisance_rows.append(name_odd)
        nuisance_manifest.append({"kind": "predicted_self_permanent_even_name_odd"})

    projection = exact_nuisance_projection(
        target,
        np.stack(nuisance_rows),
        minimum_retained_fraction=minimum_retained_fraction,
        svd_rtol=svd_rtol,
        svd_atol=svd_atol,
    )
    diagnostics = {
        key: value
        for key, value in projection.items()
        if key not in {"direction", "unprotected_direction", "basis", "nuisance_projections"}
    }
    diagnostics.update(
        {
            "cell_order": list(CELL_ORDER),
            "nuisance_manifest": nuisance_manifest,
            "predicted_even_cell_count": 8,
            "predicted_order_odd_cell_count": 8,
        }
    )
    return {
        "protected_dynamic": projection["direction"],
        "unprotected_dynamic": unprotected,
        "nuisance_rows": unit_normalize_rows(
            np.stack(nuisance_rows),
            field="constructed_nuisance_rows",
        ),
        "nuisance_manifest": nuisance_manifest,
        "diagnostics": diagnostics,
    }


def _cell_tensor(values: Any, *, field: str) -> np.ndarray:
    try:
        tensor = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if tensor.ndim != 4 or tensor.shape[0] < 2 or tensor.shape[1:3] != (2, len(CELL_ORDER)):
        raise ValueError(f"{field} must have shape [scenario, 2, 4, width]")
    if tensor.shape[3] < 1 or not np.isfinite(tensor).all():
        raise ValueError(f"{field} must be finite with positive width")
    normalized = unit_normalize_rows(tensor.reshape(-1, tensor.shape[3]), field=field)
    return np.ascontiguousarray(normalized.reshape(tensor.shape), dtype=np.float64)


def evaluate_cell_interface_direction(
    direction: Any,
    observed_choice_head_0_rows: Any,
    observed_choice_head_1_rows: Any,
    *,
    positive_alignment_threshold: float = 0.0,
) -> dict[str, Any]:
    """Measure target cosines and normalized off-target sensitivity for one scenario."""

    if isinstance(positive_alignment_threshold, bool) or not isinstance(
        positive_alignment_threshold, (int, float)
    ):
        raise TypeError("positive_alignment_threshold must be a finite scalar")
    threshold = float(positive_alignment_threshold)
    if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
        raise ValueError("positive_alignment_threshold must lie in [-1, 1]")
    head_0 = _cell_rows(observed_choice_head_0_rows, field="observed_choice_head_0_rows")
    head_1 = _cell_rows(observed_choice_head_1_rows, field="observed_choice_head_1_rows")
    if head_0.shape != head_1.shape:
        raise ValueError("observed choice heads must have the same shape")
    checked_direction = _unit_direction(direction, field="direction")
    if checked_direction.size != head_0.shape[2]:
        raise ValueError("direction width differs from observed choice heads")

    target_0 = head_0[:, 0, :] @ checked_direction
    target_1 = head_1[:, 0, :] @ checked_direction
    target_cosines = np.column_stack((target_0, target_1))
    worst_target = np.min(target_cosines, axis=1)
    assignment_success = np.all(target_cosines > threshold, axis=1)
    off_target_0 = np.abs(head_0[:, 1:, :] @ checked_direction).reshape(-1)
    off_target_1 = np.abs(head_1[:, 1:, :] @ checked_direction).reshape(-1)
    off_target = np.concatenate((off_target_0, off_target_1))
    denominator = float(np.min(target_cosines))
    ratio_defined = denominator > 0.0
    maximum_off_target = float(np.max(off_target))
    mean_off_target = float(np.mean(off_target))
    return {
        "target_head_0_cosines": target_0.tolist(),
        "target_head_1_cosines": target_1.tolist(),
        "target_worst_order_cosines": worst_target.tolist(),
        "minimum_target_cosine": denominator,
        "mean_target_cosine": float(np.mean(target_cosines)),
        "assignment_both_order_positive": assignment_success.tolist(),
        "both_order_positive_assignment_count": int(np.sum(assignment_success)),
        "complete_scenario": bool(np.all(assignment_success)),
        "off_target_absolute_cosines": off_target.tolist(),
        "maximum_off_target_absolute_cosine": maximum_off_target,
        "mean_off_target_absolute_cosine": mean_off_target,
        "off_target_ratio_defined": ratio_defined,
        "maximum_off_target_absolute_sensitivity_ratio": (
            maximum_off_target / denominator if ratio_defined else None
        ),
        "mean_off_target_absolute_sensitivity_ratio": (
            mean_off_target / denominator if ratio_defined else None
        ),
        "positive_alignment_threshold": threshold,
    }


def _cell_method_summary(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("method summary requires at least one scenario report")
    available = [report for report in reports if report["available"]]
    unavailable = [report for report in reports if not report["available"]]
    target_cosines = np.asarray(
        [
            value
            for report in available
            for value in report["target_head_0_cosines"] + report["target_head_1_cosines"]
        ],
        dtype=np.float64,
    )
    worst_order = np.asarray(
        [value for report in available for value in report["target_worst_order_cosines"]],
        dtype=np.float64,
    )
    maximum_off_target = np.asarray(
        [report["maximum_off_target_absolute_cosine"] for report in available],
        dtype=np.float64,
    )
    defined_ratios = np.asarray(
        [
            report["maximum_off_target_absolute_sensitivity_ratio"]
            for report in available
            if report["off_target_ratio_defined"]
        ],
        dtype=np.float64,
    )
    scenario_count = len(reports)
    assignment_count = 2 * scenario_count
    positive_assignments = sum(
        int(report["both_order_positive_assignment_count"]) for report in available
    )
    complete_scenarios = sum(int(report["complete_scenario"]) for report in available)
    return {
        "scenario_count": scenario_count,
        "available_scenario_count": len(available),
        "unavailable_scenario_count": len(unavailable),
        "unavailable_scenarios": [
            {
                "scenario_id": report["scenario_id"],
                "reason": report["reason"],
                "error": report["error"],
                "exception_type": report["exception_type"],
                "diagnostics": report["diagnostics"],
            }
            for report in unavailable
        ],
        "assignment_unit_count": assignment_count,
        "available_assignment_unit_count": 2 * len(available),
        "both_order_positive_assignment_count": positive_assignments,
        "both_order_positive_assignment_fraction": positive_assignments / assignment_count,
        "complete_scenario_count": complete_scenarios,
        "complete_scenario_fraction": complete_scenarios / scenario_count,
        "target_cosine": _descriptive(target_cosines) if target_cosines.size else None,
        "target_worst_order_cosine": _descriptive(worst_order) if worst_order.size else None,
        "maximum_off_target_absolute_cosine": (
            _descriptive(maximum_off_target) if maximum_off_target.size else None
        ),
        "off_target_ratio_defined_count": int(defined_ratios.size),
        "off_target_ratio_undefined_count": len(available) - int(defined_ratios.size),
        "maximum_off_target_absolute_sensitivity_ratio": (
            _descriptive(defined_ratios) if defined_ratios.size else None
        ),
    }


def _unavailable_method_record(scenario_id: str, error: Exception) -> dict[str, Any]:
    diagnostics = (
        dict(error.diagnostics) if isinstance(error, SuffixTransportIneligible) else {}
    )
    return {
        "available": False,
        "scenario_id": scenario_id,
        "reason": (
            "projection_ineligible"
            if isinstance(error, SuffixTransportIneligible)
            else "degenerate_geometry"
            if isinstance(error, ValueError)
            else "construction_error"
        ),
        "error": str(error),
        "exception_type": type(error).__name__,
        "diagnostics": diagnostics,
    }


def _available_method_record(scenario_id: str, report: dict[str, Any]) -> dict[str, Any]:
    return {"available": True, "scenario_id": scenario_id, **report}


def leave_one_scenario_out_cell_interface_translation(
    semantic_cell_rows: Any,
    choice_head_0_rows: Any,
    choice_head_1_rows: Any,
    scenario_ids: Sequence[Any],
    *,
    ridge_multiplier: float = 0.1,
    minimum_head_cosine: float = PFIT_MINIMUM_PAIR_COSINE,
    minimum_retained_fraction: float = 0.05,
    positive_alignment_threshold: float = 0.0,
    svd_rtol: float = 1e-10,
    svd_atol: float = 1e-12,
    include_heldout_oracle: bool = False,
) -> dict[str, Any]:
    """Run leakage-safe cell-level interface translation and exact protection."""

    semantic = _cell_tensor(semantic_cell_rows, field="semantic_cell_rows")
    choice_0 = _cell_tensor(choice_head_0_rows, field="choice_head_0_rows")
    choice_1 = _cell_tensor(choice_head_1_rows, field="choice_head_1_rows")
    if choice_0.shape != choice_1.shape:
        raise ValueError("choice order tensors must have the same shape")
    if semantic.shape[:3] != choice_0.shape[:3]:
        raise ValueError("semantic and choice tensors must have the same cell coverage")
    if semantic.shape[3] != choice_0.shape[3]:
        raise ValueError("cell interface translation requires one shared coordinate width")
    if not isinstance(include_heldout_oracle, bool):
        raise TypeError("include_heldout_oracle must be boolean")
    identifiers = _scenario_ids(scenario_ids, row_count=semantic.shape[0])
    even_odd = order_even_odd_directions(
        choice_0.reshape(-1, choice_0.shape[3]),
        choice_1.reshape(-1, choice_1.shape[3]),
        minimum_head_cosine=minimum_head_cosine,
    )
    even = even_odd["even_directions"].reshape(choice_0.shape)
    odd = even_odd["odd_nuisance_rows"].reshape(choice_0.shape)
    scenario_count = int(semantic.shape[0])
    target_width = int(choice_0.shape[3])
    method_names = (
        "protected_dynamic",
        "unprotected_dynamic",
        "predicted_factorial_dynamic",
        "static_training_protected",
        "factorial_semantic_identity",
        *(("oracle_upper_bound",) if include_heldout_oracle else ()),
    )
    direction_rows: dict[str, dict[str, np.ndarray]] = {method: {} for method in method_names}
    method_reports: dict[str, list[dict[str, Any]]] = {method: [] for method in method_names}
    folds = []
    scenario_rows = []

    for held_out_index, held_out in enumerate(identifiers):
        train_indices = np.asarray(
            [index for index in range(scenario_count) if index != held_out_index],
            dtype=np.int64,
        )
        if any(identifiers[index] == held_out for index in train_indices.tolist()):
            raise ValueError("scenario_ids must be unique for cell-level LOSO")
        source_train = semantic[train_indices].reshape(-1, semantic.shape[3])
        even_train = even[train_indices].reshape(-1, target_width)
        model = fit_dual_ridge_transport(
            source_train,
            even_train,
            odd[train_indices].reshape(-1, target_width),
            ridge_multiplier=ridge_multiplier,
        )
        predicted_even, predicted_odd = predict_dual_ridge_transport(
            model,
            semantic[held_out_index].reshape(-1, semantic.shape[3]),
        )
        predicted_even_cells = unit_normalize_rows(
            predicted_even,
            field="predicted_even_cell_interfaces",
        ).reshape(2, len(CELL_ORDER), target_width)
        predicted_odd_cells = unit_normalize_rows(
            predicted_odd,
            field="predicted_order_odd_cell_interfaces",
        ).reshape(2, len(CELL_ORDER), target_width)
        constructed = construct_cell_interface_directions(
            predicted_even_cells,
            predicted_odd_cells,
            minimum_retained_fraction=minimum_retained_fraction,
            svd_rtol=svd_rtol,
            svd_atol=svd_atol,
        )
        fold_directions: dict[str, np.ndarray] = {
            "protected_dynamic": constructed["protected_dynamic"],
            "unprotected_dynamic": constructed["unprotected_dynamic"],
        }
        unavailable_methods: dict[str, dict[str, Any]] = {}
        try:
            fold_directions["predicted_factorial_dynamic"] = _unit_direction(
                np.mean(
                    predicted_even_cells[:, 0, :]
                    - predicted_even_cells[:, 1, :]
                    - predicted_even_cells[:, 2, :]
                    + predicted_even_cells[:, 3, :],
                    axis=0,
                ),
                field="predicted_factorial_dynamic",
            )
        except (ValueError, RuntimeError) as error:
            unavailable_methods["predicted_factorial_dynamic"] = _unavailable_method_record(
                held_out,
                error,
            )

        static_diagnostics: dict[str, Any]
        try:
            static_constructed = construct_cell_interface_directions(
                np.mean(even[train_indices], axis=0),
                np.mean(odd[train_indices], axis=0),
                minimum_retained_fraction=minimum_retained_fraction,
                svd_rtol=svd_rtol,
                svd_atol=svd_atol,
            )
            fold_directions["static_training_protected"] = static_constructed[
                "protected_dynamic"
            ]
            static_diagnostics = {
                "available": True,
                **static_constructed["diagnostics"],
                "training_only": True,
                "training_scenario_count": int(train_indices.size),
            }
        except (ValueError, RuntimeError) as error:
            unavailable = _unavailable_method_record(held_out, error)
            unavailable_methods["static_training_protected"] = unavailable
            static_diagnostics = unavailable

        try:
            fold_directions["factorial_semantic_identity"] = _unit_direction(
                np.mean(
                    semantic[held_out_index, :, 0, :]
                    - semantic[held_out_index, :, 1, :]
                    - semantic[held_out_index, :, 2, :]
                    + semantic[held_out_index, :, 3, :],
                    axis=0,
                ),
                field="heldout_factorial_semantic_identity",
            )
        except ValueError as error:
            unavailable_methods["factorial_semantic_identity"] = _unavailable_method_record(
                held_out,
                error,
            )

        oracle_diagnostics: dict[str, Any] | None = None
        if include_heldout_oracle:
            try:
                oracle = construct_cell_interface_directions(
                    even[held_out_index],
                    odd[held_out_index],
                    minimum_retained_fraction=minimum_retained_fraction,
                    svd_rtol=svd_rtol,
                    svd_atol=svd_atol,
                )
                fold_directions["oracle_upper_bound"] = oracle["protected_dynamic"]
                oracle_diagnostics = {"available": True, **oracle["diagnostics"]}
            except (ValueError, RuntimeError) as error:
                unavailable = _unavailable_method_record(held_out, error)
                unavailable_methods["oracle_upper_bound"] = unavailable
                oracle_diagnostics = unavailable
        evaluations = {}
        for method in method_names:
            if method in unavailable_methods:
                report = unavailable_methods[method]
            else:
                direction = fold_directions[method]
                direction_rows[method][held_out] = direction
                report = _available_method_record(
                    held_out,
                    evaluate_cell_interface_direction(
                        direction,
                        choice_0[held_out_index],
                        choice_1[held_out_index],
                        positive_alignment_threshold=positive_alignment_threshold,
                    ),
                )
            method_reports[method].append(report)
            evaluations[method] = report
        folds.append(
            {
                "held_out_scenario": held_out,
                "held_out_scenario_index": held_out_index,
                "training_scenarios": [identifiers[index] for index in train_indices.tolist()],
                "training_scenario_indices": train_indices.tolist(),
                "training_cell_row_count": int(source_train.shape[0]),
                "fit_diagnostics": model.diagnostics,
                "construction_diagnostics": constructed["diagnostics"],
                "static_training_protection_diagnostics": static_diagnostics,
                "oracle_construction_diagnostics": oracle_diagnostics,
                "method_availability": {
                    method: method not in unavailable_methods for method in method_names
                },
            }
        )
        scenario_rows.append(
            {
                "scenario_id": held_out,
                "construction_diagnostics": constructed["diagnostics"],
                "static_training_protection_diagnostics": folds[-1][
                    "static_training_protection_diagnostics"
                ],
                "oracle_construction_diagnostics": oracle_diagnostics,
                "methods": evaluations,
            }
        )

    method_summaries = {
        method: _cell_method_summary(reports) for method, reports in method_reports.items()
    }
    retained = np.asarray(
        [row["construction_diagnostics"]["retained_target_fraction"] for row in scenario_rows],
        dtype=np.float64,
    )
    maximum_projection = np.asarray(
        [
            row["construction_diagnostics"]["maximum_abs_nuisance_projection"]
            for row in scenario_rows
        ],
        dtype=np.float64,
    )
    method_summaries["protected_dynamic"]["protection"] = {
        "applied": True,
        "retained_target_fraction": _descriptive(retained),
        "maximum_abs_nuisance_projection": _descriptive(maximum_projection),
        "minimum_retained_fraction": float(minimum_retained_fraction),
    }
    static_available_diagnostics = [
        row["static_training_protection_diagnostics"]
        for row in scenario_rows
        if row["static_training_protection_diagnostics"]["available"]
    ]
    static_retained = np.asarray(
        [
            diagnostics["retained_target_fraction"]
            for diagnostics in static_available_diagnostics
        ],
        dtype=np.float64,
    )
    static_projection_maximum = np.asarray(
        [
            diagnostics["maximum_abs_nuisance_projection"]
            for diagnostics in static_available_diagnostics
        ],
        dtype=np.float64,
    )
    method_summaries["static_training_protected"]["protection"] = {
        "applied": True,
        "training_only": True,
        "retained_target_fraction": (
            _descriptive(static_retained) if static_retained.size else None
        ),
        "maximum_abs_nuisance_projection": (
            _descriptive(static_projection_maximum)
            if static_projection_maximum.size
            else None
        ),
        "minimum_retained_fraction": float(minimum_retained_fraction),
    }
    method_summaries["unprotected_dynamic"]["protection"] = {"applied": False}
    method_summaries["predicted_factorial_dynamic"]["protection"] = {"applied": False}
    method_summaries["factorial_semantic_identity"]["protection"] = {"applied": False}
    if include_heldout_oracle:
        oracle_available_diagnostics = [
            row["oracle_construction_diagnostics"]
            for row in scenario_rows
            if row["oracle_construction_diagnostics"]["available"]
        ]
        oracle_retained = np.asarray(
            [
                diagnostics["retained_target_fraction"]
                for diagnostics in oracle_available_diagnostics
            ],
            dtype=np.float64,
        )
        method_summaries["oracle_upper_bound"]["protection"] = {
            "applied": True,
            "uses_heldout_observed_choice_gradients": True,
            "evaluation_only_upper_bound": True,
            "retained_target_fraction": (
                _descriptive(oracle_retained) if oracle_retained.size else None
            ),
        }
    directions = {}
    for method in method_names:
        available_scenarios = [
            scenario_id for scenario_id in identifiers if scenario_id in direction_rows[method]
        ]
        rows = (
            np.stack([direction_rows[method][scenario_id] for scenario_id in available_scenarios])
            if available_scenarios
            else np.empty((0, target_width), dtype=np.float64)
        )
        directions[method] = {
            "available_scenario_ids": available_scenarios,
            "rows": np.ascontiguousarray(rows, dtype=np.float64),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "cell_order": list(CELL_ORDER),
        "scenario_ids": list(identifiers),
        "ridge_multiplier": float(ridge_multiplier),
        "minimum_head_cosine": float(minimum_head_cosine),
        "minimum_retained_fraction": float(minimum_retained_fraction),
        "include_heldout_oracle": include_heldout_oracle,
        "directions": directions,
        "folds": folds,
        "scenario_rows": scenario_rows,
        "method_summaries": method_summaries,
    }


__all__ = [
    "CELL_ORDER",
    "PFIT_MINIMUM_PAIR_COSINE",
    "SCHEMA_VERSION",
    "DualRidgeTransport",
    "SuffixTransportIneligible",
    "construct_cell_interface_directions",
    "evaluate_cell_interface_direction",
    "exact_nuisance_projection",
    "fit_dual_ridge_transport",
    "leave_one_scenario_out_cell_interface_translation",
    "leave_one_scenario_out_transport",
    "order_even_odd_directions",
    "predict_dual_ridge_transport",
    "robust_two_head_unit_bisector",
    "transport_metric_summary",
    "unit_normalize_rows",
    "validate_matrix",
]
