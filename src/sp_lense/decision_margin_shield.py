"""Model-free Decision-Margin Shielding geometry for the v2 layer screen.

The functions in this module never load a model and never perform a finite
intervention. They solve three uncapped minimum-Euclidean-norm problems in
residual-relative coordinates and apply a fixed norm frontier afterward.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import nnls

from .counterfactual_tangent_shield import (
    TangentShieldDirection,
    TangentShieldInfeasibleError,
    solve_minimum_l2_direction,
)
from .factorial_causal_anchor import canonical_sha256

SCHEMA_VERSION = "sp_lense.decision_margin_shield.v2"
METHODS = ("unshielded", "unrelated_only", "decision_margin_shield")
DEFAULT_MARGIN = 0.05
DEFAULT_CAP_FRONTIER = (1.0, 1.5, 2.0)
DEFAULT_QUALIFICATION_CAP = 2.0
DEFAULT_SVD_RTOL = 1e-10
DEFAULT_SVD_ATOL = 1e-12
DEFAULT_PRIMAL_TOLERANCE = 1e-8
DEFAULT_ACTIVE_SLACK_TOLERANCE = 1e-8
DEFAULT_OPTIMALITY_ABSOLUTE_TOLERANCE = 1e-8
DEFAULT_OPTIMALITY_RELATIVE_TOLERANCE = 1e-8
DEFAULT_KKT_TOLERANCE = 1e-7


class DecisionMarginOptimalityError(RuntimeError):
    """A DMS candidate lacks the certificate needed to call it minimum-norm."""


def _finite_matrix(value: Any, *, field: str, rows: int | None = None) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except Exception as error:  # pragma: no cover - NumPy supplies the concrete cause.
        raise TypeError(f"{field} must be an array") from error
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{field} must contain real numbers")
    result = np.asarray(raw, dtype=np.float64, order="C").copy(order="C")
    if result.ndim != 2 or result.shape[1] == 0:
        raise ValueError(f"{field} must be a non-empty two-dimensional matrix")
    if rows is not None and result.shape[0] != rows:
        raise ValueError(f"{field} must contain exactly {rows} rows")
    if not np.isfinite(result).all():
        raise ValueError(f"{field} must contain only finite values")
    return result


def _finite_vector(value: Any, *, field: str, length: int) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except Exception as error:  # pragma: no cover - NumPy supplies the concrete cause.
        raise TypeError(f"{field} must be an array") from error
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{field} must contain real numbers")
    result = np.asarray(raw, dtype=np.float64, order="C").copy(order="C")
    if result.shape != (length,):
        raise ValueError(f"{field} must contain exactly {length} values")
    if not np.isfinite(result).all():
        raise ValueError(f"{field} must contain only finite values")
    return result


def _positive_scalar(value: Any, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{field} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _checked_frontier(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(_positive_scalar(value, field="cap_frontier") for value in values)
    if not result or result != tuple(sorted(set(result))):
        raise ValueError("cap_frontier must be non-empty, unique, and strictly increasing")
    return result


def _cap_key(value: float) -> str:
    return format(value, ".15g")


def _nonnegative_rows(value: Any, *, field: str, length: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim == 0:
        if raw.dtype.kind not in "iuf":
            raise TypeError(f"{field} must contain real numbers")
        scalar = float(raw)
        if not math.isfinite(scalar) or scalar < 0.0:
            raise ValueError(f"{field} must be finite and nonnegative")
        return np.full(length, scalar, dtype=np.float64)
    result = _finite_vector(value, field=field, length=length)
    if bool(np.any(result < 0.0)):
        raise ValueError(f"{field} must be nonnegative")
    return result


def _equality_row_basis(
    equality_rows: np.ndarray,
    *,
    svd_rtol: float,
    svd_atol: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reconstruct the v1 normalized-row rank rule without importing its internals."""

    dimension = equality_rows.shape[1]
    row_norms = np.linalg.norm(equality_rows, axis=1)
    nonzero = row_norms > 0.0
    normalized = np.zeros_like(equality_rows)
    normalized[nonzero] = equality_rows[nonzero] / row_norms[nonzero, None]
    if equality_rows.shape[0] == 0:
        singular_values = np.zeros(0, dtype=np.float64)
        row_basis = np.zeros((0, dimension), dtype=np.float64)
        threshold = svd_atol
    else:
        _, singular_values, vh = np.linalg.svd(normalized, full_matrices=False)
        largest = float(singular_values[0]) if singular_values.size else 0.0
        threshold = max(svd_atol, svd_rtol * largest)
        rank = int(np.count_nonzero(singular_values > threshold))
        row_basis = vh[:rank].copy(order="C")
        for row in row_basis:
            anchor = int(np.argmax(np.abs(row)))
            if row[anchor] < 0.0:
                row *= -1.0
    projected_residual = equality_rows - (equality_rows @ row_basis.T) @ row_basis
    maximum_projection_residual = (
        float(np.max(np.abs(projected_residual))) if projected_residual.size else 0.0
    )
    orthonormality = row_basis @ row_basis.T - np.eye(row_basis.shape[0])
    maximum_orthonormality_residual = (
        float(np.max(np.abs(orthonormality))) if orthonormality.size else 0.0
    )
    maximum_row_norm = float(np.max(row_norms)) if row_norms.size else 0.0
    span_tolerance = max(256.0 * np.finfo(np.float64).eps, threshold) * max(
        1.0, maximum_row_norm
    )
    diagnostics = {
        "input_row_count": int(equality_rows.shape[0]),
        "zero_row_count": int(np.count_nonzero(~nonzero)),
        "dimension": int(dimension),
        "svd_rtol": svd_rtol,
        "svd_atol": svd_atol,
        "singular_values": singular_values.tolist(),
        "threshold": threshold,
        "rank": int(row_basis.shape[0]),
        "maximum_abs_input_rowspace_projection_residual": maximum_projection_residual,
        "input_rowspace_projection_tolerance": span_tolerance,
        "maximum_abs_row_basis_orthonormality_residual": maximum_orthonormality_residual,
        "row_basis_sha256": canonical_sha256(row_basis.tolist()),
    }
    return row_basis, diagnostics


def _roundoff_adjusted_dual_lower_bound(
    *,
    constraint_matrix: np.ndarray,
    constraint_lower: np.ndarray,
    multipliers: np.ndarray,
    equality_rows: np.ndarray,
    equality_multipliers: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    """Evaluate a conservative float64 dual bound for the represented problem."""

    eps = np.finfo(np.float64).eps

    def gamma(operation_count: int) -> float:
        scaled = max(1, operation_count) * eps
        return scaled / (1.0 - scaled)

    target_part = constraint_matrix.T @ multipliers
    equality_part = equality_rows.T @ equality_multipliers
    dual_stationarity_vector = target_part - equality_part
    linear_term = float(constraint_lower @ multipliers)
    quadratic_term = 0.5 * float(
        dual_stationarity_vector @ dual_stationarity_vector
    )
    raw_lower_bound = linear_term - quadratic_term

    target_dot_error = 64.0 * gamma(constraint_matrix.shape[0]) * (
        np.abs(constraint_matrix).T @ multipliers
    )
    equality_dot_error = 64.0 * gamma(equality_rows.shape[0]) * (
        np.abs(equality_rows).T @ np.abs(equality_multipliers)
    )
    vector_error = target_dot_error + equality_dot_error
    linear_error = 64.0 * gamma(constraint_lower.size) * float(
        np.sum(np.abs(constraint_lower * multipliers))
    )
    quadratic_error = 0.5 * (
        float(np.sum(2.0 * np.abs(dual_stationarity_vector) * vector_error))
        + float(vector_error @ vector_error)
        + 64.0
        * gamma(dual_stationarity_vector.size)
        * float(dual_stationarity_vector @ dual_stationarity_vector)
    )
    rounding_allowance = linear_error + quadratic_error + 64.0 * eps * max(
        1.0, abs(linear_term), abs(quadratic_term), abs(raw_lower_bound)
    )
    validated_lower_bound = raw_lower_bound - rounding_allowance
    return (
        raw_lower_bound,
        validated_lower_bound,
        rounding_allowance,
        dual_stationarity_vector,
    )


def certify_minimum_l2_candidate(
    direction: Any,
    target_rows: Any,
    target_offsets: Any,
    *,
    margin: Any = DEFAULT_MARGIN,
    nuisance_rows: Any | None = None,
    nuisance_bound: Any = 0.0,
    svd_rtol: Any = DEFAULT_SVD_RTOL,
    svd_atol: Any = DEFAULT_SVD_ATOL,
    primal_tolerance: Any = DEFAULT_PRIMAL_TOLERANCE,
    active_slack_tolerance: Any = DEFAULT_ACTIVE_SLACK_TOLERANCE,
    optimality_absolute_tolerance: Any = DEFAULT_OPTIMALITY_ABSOLUTE_TOLERANCE,
    optimality_relative_tolerance: Any = DEFAULT_OPTIMALITY_RELATIVE_TOLERANCE,
    kkt_tolerance: Any = DEFAULT_KKT_TOLERANCE,
) -> dict[str, Any]:
    """Independently certify a candidate for the convex DMS minimum-L2 problem.

    The v1 optimizer is not trusted here.  Active inequalities are used to recover
    nonnegative dual multipliers with NNLS.  The dual objective is then recomputed
    against the original inequalities and exact equalities, producing a valid lower
    bound for any dual-feasible multiplier vector.  A candidate passes only if its
    primal-dual gap and KKT residuals meet the fixed tolerances.
    """

    target = _finite_matrix(target_rows, field="target_rows")
    if target.shape[0] == 0:
        raise ValueError("target_rows must contain at least one row")
    target_count, dimension = target.shape
    target_b = _finite_vector(target_offsets, field="target_offsets", length=target_count)
    margins = _nonnegative_rows(margin, field="margin", length=target_count)
    target_lower = np.abs(target_b) + margins
    if not np.isfinite(target_lower).all():
        raise ValueError("abs(target_offsets) + margin must remain finite")
    nuisance = (
        np.zeros((0, dimension), dtype=np.float64)
        if nuisance_rows is None
        else _finite_matrix(nuisance_rows, field="nuisance_rows")
    )
    if nuisance.shape[1] != dimension:
        raise ValueError("nuisance_rows width must match target_rows")
    nuisance_bounds = _nonnegative_rows(
        nuisance_bound, field="nuisance_bound", length=nuisance.shape[0]
    )
    candidate = np.asarray(direction)
    if candidate.dtype.kind not in "iuf" or candidate.shape != (dimension,):
        raise ValueError(f"direction must be a real vector of length {dimension}")
    candidate = np.asarray(candidate, dtype=np.float64, order="C").copy(order="C")
    if not np.isfinite(candidate).all():
        raise ValueError("direction must contain only finite values")

    checked_svd_rtol = float(svd_rtol)
    checked_svd_atol = float(svd_atol)
    checked_primal_tolerance = _positive_scalar(
        primal_tolerance, field="primal_tolerance"
    )
    checked_active_tolerance = _positive_scalar(
        active_slack_tolerance, field="active_slack_tolerance"
    )
    checked_optimality_absolute = _positive_scalar(
        optimality_absolute_tolerance, field="optimality_absolute_tolerance"
    )
    checked_optimality_relative = _positive_scalar(
        optimality_relative_tolerance, field="optimality_relative_tolerance"
    )
    checked_kkt_tolerance = _positive_scalar(kkt_tolerance, field="kkt_tolerance")
    if (
        not math.isfinite(checked_svd_rtol)
        or checked_svd_rtol < 0.0
        or not math.isfinite(checked_svd_atol)
        or checked_svd_atol < 0.0
    ):
        raise ValueError("SVD tolerances must be finite and nonnegative")

    exact_mask = nuisance_bounds == 0.0
    soft_mask = ~exact_mask
    equality_rows = nuisance[exact_mask]
    soft_rows = nuisance[soft_mask]
    soft_bounds = nuisance_bounds[soft_mask]
    constraint_matrix = np.vstack((target, -soft_rows, soft_rows))
    constraint_lower = np.concatenate((target_lower, -soft_bounds, -soft_bounds))
    primal_slacks = constraint_matrix @ candidate - constraint_lower
    equality_values = equality_rows @ candidate
    candidate_norm = float(np.linalg.norm(candidate))
    primal_objective = 0.5 * candidate_norm**2
    inequality_scale = max(
        1.0,
        float(np.max(np.abs(constraint_lower))),
        float(np.linalg.norm(constraint_matrix, ord=2)) * max(1.0, candidate_norm),
    )
    equality_scale = max(
        1.0,
        (
            float(np.linalg.norm(equality_rows, ord=2)) * max(1.0, candidate_norm)
            if equality_rows.size
            else 0.0
        ),
    )
    primal_inequality_tolerance = checked_primal_tolerance * inequality_scale
    primal_equality_tolerance = checked_primal_tolerance * equality_scale
    active_threshold = checked_active_tolerance * inequality_scale

    row_basis, svd_diagnostics = _equality_row_basis(
        equality_rows,
        svd_rtol=checked_svd_rtol,
        svd_atol=checked_svd_atol,
    )
    projected_constraints = constraint_matrix - (
        constraint_matrix @ row_basis.T
    ) @ row_basis
    active_mask = primal_slacks <= active_threshold
    active_indices = np.flatnonzero(active_mask)
    multipliers = np.zeros(constraint_matrix.shape[0], dtype=np.float64)
    nnls_residual = candidate_norm
    if active_indices.size:
        try:
            active_multipliers, nnls_residual = nnls(
                projected_constraints[active_indices].T,
                candidate,
            )
        except Exception as error:
            raise DecisionMarginOptimalityError(
                "the independent active-set dual NNLS solve failed"
            ) from error
        multipliers[active_indices] = active_multipliers
    if not np.isfinite(multipliers).all() or not math.isfinite(float(nnls_residual)):
        raise DecisionMarginOptimalityError(
            "the independent active-set dual NNLS solve returned non-finite values"
        )

    target_dual_vector = constraint_matrix.T @ multipliers
    equality_multipliers = np.zeros(equality_rows.shape[0], dtype=np.float64)
    if row_basis.shape[0]:
        row_norms = np.linalg.norm(equality_rows, axis=1)
        nonzero = row_norms > 0.0
        normalized = equality_rows[nonzero] / row_norms[nonzero, None]
        u, singular_values, vh = np.linalg.svd(normalized, full_matrices=False)
        rank = row_basis.shape[0]
        normalized_multipliers = u[:, :rank] @ (
            (vh[:rank] @ target_dual_vector) / singular_values[:rank]
        )
        equality_multipliers[nonzero] = normalized_multipliers / row_norms[nonzero]

    (
        raw_dual_lower_bound,
        validated_dual_lower_bound,
        dual_roundoff_allowance,
        dual_stationarity_vector,
    ) = _roundoff_adjusted_dual_lower_bound(
        constraint_matrix=constraint_matrix,
        constraint_lower=constraint_lower,
        multipliers=multipliers,
        equality_rows=equality_rows,
        equality_multipliers=equality_multipliers,
    )
    raw_gap = primal_objective - raw_dual_lower_bound
    validated_gap = primal_objective - validated_dual_lower_bound
    stationarity_residual = candidate - dual_stationarity_vector
    stationarity_l2 = float(np.linalg.norm(stationarity_residual))
    complementarity_products = multipliers * primal_slacks
    maximum_abs_complementarity = (
        float(np.max(np.abs(complementarity_products)))
        if complementarity_products.size
        else 0.0
    )
    minimum_dual_multiplier = (
        float(np.min(multipliers)) if multipliers.size else 0.0
    )
    objective_tolerance = checked_optimality_absolute + checked_optimality_relative * max(
        1.0, abs(primal_objective), abs(validated_dual_lower_bound)
    )
    stationarity_tolerance = checked_kkt_tolerance * max(
        1.0, candidate_norm, float(np.linalg.norm(dual_stationarity_vector))
    )
    complementarity_tolerance = checked_kkt_tolerance * max(
        1.0,
        abs(primal_objective),
        float(np.sum(np.abs(constraint_lower * multipliers))),
    )
    minimum_inequality_slack = float(np.min(primal_slacks))
    maximum_equality_residual = (
        float(np.max(np.abs(equality_values))) if equality_values.size else 0.0
    )
    checks = {
        "finite": bool(
            np.isfinite(primal_slacks).all()
            and np.isfinite(equality_values).all()
            and np.isfinite(equality_multipliers).all()
            and math.isfinite(raw_dual_lower_bound)
            and math.isfinite(validated_dual_lower_bound)
            and math.isfinite(validated_gap)
        ),
        "primal_inequalities": bool(
            minimum_inequality_slack >= -primal_inequality_tolerance
        ),
        "primal_exact_equalities": bool(
            maximum_equality_residual <= primal_equality_tolerance
        ),
        "equality_svd_span": bool(
            svd_diagnostics["maximum_abs_input_rowspace_projection_residual"]
            <= svd_diagnostics["input_rowspace_projection_tolerance"]
        ),
        "equality_svd_orthonormality": bool(
            svd_diagnostics["maximum_abs_row_basis_orthonormality_residual"]
            <= 256.0 * np.finfo(np.float64).eps
        ),
        "dual_feasible_nonnegative_multipliers": bool(minimum_dual_multiplier >= 0.0),
        "weak_duality_with_roundoff": bool(raw_gap >= -objective_tolerance),
        "primal_dual_gap": bool(validated_gap <= objective_tolerance),
        "stationarity": bool(stationarity_l2 <= stationarity_tolerance),
        "complementarity": bool(
            maximum_abs_complementarity <= complementarity_tolerance
        ),
    }
    minimum_l2_lower_bound = math.sqrt(max(0.0, 2.0 * validated_dual_lower_bound))
    certificate = {
        "schema_version": "sp_lense.decision_margin_shield.optimality_certificate.v1",
        "objective": "minimum_one_half_squared_euclidean_l2",
        "dual_construction": "active_inequality_nnls_then_original_coordinate_dual_evaluation",
        "dual_lower_bound_validity": (
            "nonnegative_inequality_multipliers_and_unrestricted_exact_equality_multipliers"
        ),
        "direction_sha256": canonical_sha256(candidate.tolist()),
        "constraint_matrix_sha256": canonical_sha256(constraint_matrix.tolist()),
        "constraint_lower_sha256": canonical_sha256(constraint_lower.tolist()),
        "exact_equality_rows_sha256": canonical_sha256(equality_rows.tolist()),
        "target_constraint_count": int(target_count),
        "soft_slab_row_count": int(soft_rows.shape[0]),
        "exact_equality_row_count": int(equality_rows.shape[0]),
        "inequality_constraint_count": int(constraint_matrix.shape[0]),
        "active_inequality_indices": active_indices.tolist(),
        "active_inequality_count": int(active_indices.size),
        "candidate_l2_norm": candidate_norm,
        "primal_objective": primal_objective,
        "raw_dual_objective_lower_bound": raw_dual_lower_bound,
        "dual_roundoff_allowance": dual_roundoff_allowance,
        "validated_dual_objective_lower_bound": validated_dual_lower_bound,
        "minimum_l2_lower_bound": minimum_l2_lower_bound,
        "raw_primal_dual_gap": raw_gap,
        "validated_primal_dual_gap": validated_gap,
        "minimum_inequality_slack": minimum_inequality_slack,
        "maximum_abs_exact_equality_residual": maximum_equality_residual,
        "stationarity_l2_residual": stationarity_l2,
        "maximum_abs_complementarity_residual": maximum_abs_complementarity,
        "minimum_dual_inequality_multiplier": minimum_dual_multiplier,
        "maximum_dual_inequality_multiplier": (
            float(np.max(multipliers)) if multipliers.size else 0.0
        ),
        "nnls_stationarity_fit_l2_residual": float(nnls_residual),
        "primal_inequality_tolerance": primal_inequality_tolerance,
        "primal_equality_tolerance": primal_equality_tolerance,
        "active_slack_threshold": active_threshold,
        "objective_gap_tolerance": objective_tolerance,
        "stationarity_tolerance": stationarity_tolerance,
        "complementarity_tolerance": complementarity_tolerance,
        "inequality_multipliers": multipliers.tolist(),
        "equality_multipliers": equality_multipliers.tolist(),
        "primal_inequality_slacks": primal_slacks.tolist(),
        "svd": svd_diagnostics,
        "checks": checks,
        "passes": bool(all(checks.values())),
    }
    certificate["certificate_sha256"] = canonical_sha256(certificate)
    return certificate


def decision_margin_bounds(
    protected_offsets: Any,
    *,
    margin: float = DEFAULT_MARGIN,
) -> np.ndarray:
    """Return ``max(abs(b) - margin, 0)`` for each protected A/B row.

    A zero bound freezes the row's first-order score change. Only a baseline with
    absolute log-odds strictly smaller than the margin lacks the requested margin;
    equality is margin-certified.
    """

    checked_margin = _positive_scalar(margin, field="margin")
    raw = np.asarray(protected_offsets)
    if raw.dtype.kind not in "iuf" or raw.ndim != 1:
        raise ValueError("protected_offsets must be a one-dimensional real vector")
    offsets = np.asarray(raw, dtype=np.float64, order="C").copy(order="C")
    if not offsets.size or not np.isfinite(offsets).all():
        raise ValueError("protected_offsets must be non-empty and finite")
    bounds = np.maximum(np.abs(offsets) - checked_margin, 0.0)
    bounds[bounds == 0.0] = 0.0
    bounds.setflags(write=False)
    return bounds


def _method_record(
    *,
    method: str,
    solution: TangentShieldDirection | None,
    error: TangentShieldInfeasibleError | None,
    optimality_certificate: Mapping[str, Any] | None,
    cap_frontier: tuple[float, ...],
    target_count: int,
    exact_unrelated_count: int,
    protected_count: int,
    protected_bounds: np.ndarray,
    small_baseline_protected_count: int,
) -> dict[str, Any]:
    common: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "optimization": "uncapped_minimum_euclidean_l2",
        "target_constraint_count": target_count,
        "exact_unrelated_gradient_count": exact_unrelated_count,
        "protected_gradient_count": protected_count,
        "small_baseline_first_order_frozen_count": small_baseline_protected_count,
        "protected_margin_certified_row_count": protected_count
        - small_baseline_protected_count,
        "protected_bounds_sha256": canonical_sha256(protected_bounds.tolist()),
        "local_linear_only": True,
    }
    if solution is None:
        if error is None:  # pragma: no cover - private call contract.
            raise RuntimeError("an ineligible geometry record requires an error")
        record = {
            **common,
            "status": "infeasible",
            "error_type": type(error).__name__,
            "error": str(error),
            "minimum_standardized_l2": None,
            "certified_minimum_standardized_l2_lower_bound": None,
            "direction_sha256": None,
            "cap_passes": {_cap_key(cap): False for cap in cap_frontier},
            "cap_certificates": {
                _cap_key(cap): {
                    "cap": cap,
                    "status": "constraint_system_infeasible",
                    "feasible_witness": False,
                    "dual_infeasibility_certificate": False,
                }
                for cap in cap_frontier
            },
        }
    else:
        if optimality_certificate is None or not bool(
            optimality_certificate.get("passes", False)
        ):
            raise DecisionMarginOptimalityError(
                "a DMS geometry record requires a passing independent optimality certificate"
            )
        norm = float(optimality_certificate["candidate_l2_norm"])
        if not math.isfinite(norm) or norm < 0.0:
            raise RuntimeError("the certified minimum L2 norm is invalid")
        lower_bound_norm = float(optimality_certificate["minimum_l2_lower_bound"])
        objective_gap_tolerance = float(
            optimality_certificate["objective_gap_tolerance"]
        )
        maximum_numerical_lower_norm = math.sqrt(
            max(0.0, norm**2 + 2.0 * objective_gap_tolerance)
        )
        if (
            not math.isfinite(lower_bound_norm)
            or lower_bound_norm < 0.0
            or lower_bound_norm > maximum_numerical_lower_norm
        ):
            raise DecisionMarginOptimalityError(
                "the independent minimum-L2 lower bound is invalid"
            )
        cap_certificates: dict[str, dict[str, Any]] = {}
        for cap in cap_frontier:
            if norm <= cap:
                status = "feasible_primal_witness"
                feasible_witness = True
                dual_infeasibility = False
            elif lower_bound_norm > cap:
                status = "infeasible_dual_lower_bound"
                feasible_witness = False
                dual_infeasibility = True
            else:
                status = "numerically_indeterminate"
                feasible_witness = False
                dual_infeasibility = False
            cap_certificates[_cap_key(cap)] = {
                "cap": cap,
                "status": status,
                "candidate_l2_norm": norm,
                "certified_minimum_l2_lower_bound": lower_bound_norm,
                "feasible_witness": feasible_witness,
                "dual_infeasibility_certificate": dual_infeasibility,
            }
        record = {
            **common,
            "status": "eligible",
            "minimum_standardized_l2": norm,
            "certified_minimum_standardized_l2_lower_bound": lower_bound_norm,
            "direction_sha256": solution.diagnostics["direction_sha256"],
            "cap_passes": {
                key: bool(value["feasible_witness"])
                for key, value in cap_certificates.items()
            },
            "cap_certificates": cap_certificates,
            "optimality_certificate": dict(optimality_certificate),
            "solver_diagnostics": solution.diagnostics,
        }
    record["geometry_record_sha256"] = canonical_sha256(record)
    return record


def screen_scenario_layer(
    *,
    target_rows: Any,
    target_offsets: Any,
    protected_rows: Any,
    protected_offsets: Any,
    unrelated_rows: Any,
    margin: float = DEFAULT_MARGIN,
    cap_frontier: Sequence[float] = DEFAULT_CAP_FRONTIER,
) -> list[dict[str, Any]]:
    """Solve all three frozen geometry variants for one scenario and one layer.

    Study-specific row counts are enforced: four target, twelve matched protected,
    and eight unrelated gradients. Infeasibility is returned as a machine-readable
    record; numerical solver failures still fail closed.
    """

    checked_margin = _positive_scalar(margin, field="margin")
    caps = _checked_frontier(cap_frontier)
    target = _finite_matrix(target_rows, field="target_rows", rows=4)
    protected = _finite_matrix(protected_rows, field="protected_rows", rows=12)
    unrelated = _finite_matrix(unrelated_rows, field="unrelated_rows", rows=8)
    dimension = target.shape[1]
    if protected.shape[1] != dimension or unrelated.shape[1] != dimension:
        raise ValueError("target, protected, and unrelated matrices must have equal width")
    target_b = _finite_vector(target_offsets, field="target_offsets", length=4)
    protected_b = _finite_vector(protected_offsets, field="protected_offsets", length=12)
    protected_bounds = decision_margin_bounds(protected_b, margin=checked_margin)
    # Equality is certified: a zero first-order change retains exactly ``margin``.
    small_count = int(np.count_nonzero(np.abs(protected_b) < checked_margin))

    definitions = (
        ("unshielded", None, np.zeros(0, dtype=np.float64), 0, 0),
        ("unrelated_only", unrelated, np.zeros(8, dtype=np.float64), 8, 0),
        (
            "decision_margin_shield",
            np.vstack((unrelated, protected)),
            np.concatenate((np.zeros(8, dtype=np.float64), protected_bounds)),
            8,
            12,
        ),
    )
    records: list[dict[str, Any]] = []
    for method, nuisance, nuisance_bounds, unrelated_count, protected_count in definitions:
        try:
            solution = solve_minimum_l2_direction(
                target,
                target_b,
                margin=checked_margin,
                nuisance_rows=nuisance,
                nuisance_bound=nuisance_bounds,
                l2_cap=None,
            )
            error = None
            optimality_certificate = certify_minimum_l2_candidate(
                solution.direction,
                target,
                target_b,
                margin=checked_margin,
                nuisance_rows=nuisance,
                nuisance_bound=nuisance_bounds,
            )
            if not optimality_certificate["passes"]:
                failed = ", ".join(
                    name
                    for name, passed in optimality_certificate["checks"].items()
                    if not passed
                )
                raise DecisionMarginOptimalityError(
                    "the candidate failed independent optimality certification: " + failed
                )
        except TangentShieldInfeasibleError as caught:
            solution = None
            error = caught
            optimality_certificate = None
        records.append(
            _method_record(
                method=method,
                solution=solution,
                error=error,
                optimality_certificate=optimality_certificate,
                cap_frontier=caps,
                target_count=4,
                exact_unrelated_count=unrelated_count,
                protected_count=protected_count,
                protected_bounds=(
                    protected_bounds if method == "decision_margin_shield" else np.zeros(0)
                ),
                small_baseline_protected_count=(
                    small_count if method == "decision_margin_shield" else 0
                ),
            )
        )
    return records


def select_layer(
    records: Sequence[Mapping[str, Any]],
    *,
    calibration_scenario_ids: Sequence[str],
    layers: Sequence[int],
    cap_frontier: Sequence[float] = DEFAULT_CAP_FRONTIER,
    qualification_cap: float = DEFAULT_QUALIFICATION_CAP,
) -> dict[str, Any]:
    """Apply the frozen calibration-only lexicographic DMS layer rule."""

    scenario_ids = tuple(map(str, calibration_scenario_ids))
    if not scenario_ids or len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("calibration_scenario_ids must be non-empty and unique")
    checked_layers = tuple(map(int, layers))
    if not checked_layers or checked_layers != tuple(sorted(set(checked_layers))):
        raise ValueError("layers must be non-empty, unique, and ascending")
    caps = _checked_frontier(cap_frontier)
    cap = _positive_scalar(qualification_cap, field="qualification_cap")
    if cap not in caps:
        raise ValueError("qualification_cap must be a member of cap_frontier")

    indexed: dict[tuple[int, str], Mapping[str, Any]] = {}
    for record in records:
        if record.get("method") != "decision_margin_shield":
            continue
        if record.get("partition") != "calibration":
            raise ValueError("layer selection accepts calibration-partition records only")
        key = (int(record["layer"]), str(record["scenario_id"]))
        if key in indexed:
            raise ValueError("duplicate decision-margin layer/scenario record")
        indexed[key] = record
    expected = {(layer, scenario_id) for layer in checked_layers for scenario_id in scenario_ids}
    if set(indexed) != expected:
        raise ValueError("decision-margin records do not cover the frozen layer/scenario grid")

    summaries: list[dict[str, Any]] = []
    for layer in checked_layers:
        scenario_rows = [indexed[(layer, scenario_id)] for scenario_id in scenario_ids]
        for row in scenario_rows:
            if row.get("status") != "eligible":
                continue
            certificate = row.get("optimality_certificate")
            if not isinstance(certificate, Mapping) or not bool(certificate.get("passes")):
                raise DecisionMarginOptimalityError(
                    "layer selection refuses an eligible norm without independent certification"
                )
            cap_certificates = row.get("cap_certificates")
            if not isinstance(cap_certificates, Mapping):
                raise DecisionMarginOptimalityError(
                    "layer selection refuses an eligible norm without cap certificates"
                )
            cap_certificate = cap_certificates.get(_cap_key(cap))
            if not isinstance(cap_certificate, Mapping):
                raise DecisionMarginOptimalityError(
                    "layer selection lacks the qualification-cap certificate"
                )
            if cap_certificate.get("status") == "numerically_indeterminate":
                raise DecisionMarginOptimalityError(
                    "qualification-cap status is numerically indeterminate"
                )
        eligible_norms = [
            float(row["minimum_standardized_l2"])
            for row in scenario_rows
            if row.get("status") == "eligible"
        ]
        if any(not math.isfinite(norm) or norm < 0.0 for norm in eligible_norms):
            raise ValueError("an eligible DMS record has an invalid norm")
        all_eligible = len(eligible_norms) == len(scenario_ids)
        qualification_statuses = {
            scenario_id: (
                str(
                    indexed[(layer, scenario_id)]["cap_certificates"][_cap_key(cap)][
                        "status"
                    ]
                )
                if indexed[(layer, scenario_id)].get("status") == "eligible"
                else "constraint_system_infeasible"
            )
            for scenario_id in scenario_ids
        }
        qualifies = all_eligible and all(
            status == "feasible_primal_witness"
            for status in qualification_statuses.values()
        )
        summary = {
            "layer": layer,
            "scenario_count": len(scenario_ids),
            "eligible_scenario_count": len(eligible_norms),
            "scenario_minimum_standardized_l2": {
                scenario_id: (
                    float(indexed[(layer, scenario_id)]["minimum_standardized_l2"])
                    if indexed[(layer, scenario_id)].get("status") == "eligible"
                    else None
                )
                for scenario_id in scenario_ids
            },
            "cap_pass_counts": {
                _cap_key(frontier_cap): sum(
                    bool(row["cap_certificates"][_cap_key(frontier_cap)]["feasible_witness"])
                    for row in scenario_rows
                    if row.get("status") == "eligible"
                )
                for frontier_cap in caps
            },
            "cap_dual_infeasibility_counts": {
                _cap_key(frontier_cap): sum(
                    bool(
                        row["cap_certificates"][_cap_key(frontier_cap)][
                            "dual_infeasibility_certificate"
                        ]
                    )
                    for row in scenario_rows
                    if row.get("status") == "eligible"
                )
                for frontier_cap in caps
            },
            "qualification_cap_status": qualification_statuses,
            "worst_case_minimum_standardized_l2": (
                max(eligible_norms) if all_eligible else None
            ),
            "mean_minimum_standardized_l2": (
                statistics.fmean(eligible_norms) if all_eligible else None
            ),
            "qualification_cap": cap,
            "qualifies": qualifies,
        }
        summary["layer_summary_sha256"] = canonical_sha256(summary)
        summaries.append(summary)

    qualifying = [summary for summary in summaries if summary["qualifies"]]
    selected = min(
        qualifying,
        key=lambda row: (
            float(row["worst_case_minimum_standardized_l2"]),
            float(row["mean_minimum_standardized_l2"]),
            int(row["layer"]),
        ),
        default=None,
    )
    result = {
        "schema_version": "sp_lense.decision_margin_shield_layer_selection.v2",
        "status": "selected" if selected is not None else "no_qualifying_layer",
        "selection_partition": "calibration_only",
        "calibration_scenario_ids": list(scenario_ids),
        "layers": list(checked_layers),
        "cap_frontier": list(caps),
        "qualification_cap": cap,
        "tie_breakers": [
            "smallest_worst_case_dms_norm",
            "smallest_mean_dms_norm",
            "smallest_zero_based_layer",
        ],
        "qualifying_layer_count": len(qualifying),
        "selected_layer": None if selected is None else int(selected["layer"]),
        "selected_layer_summary": selected,
        "layer_summaries": summaries,
    }
    result["selection_sha256"] = canonical_sha256(result)
    return result


__all__ = [
    "DEFAULT_CAP_FRONTIER",
    "DEFAULT_MARGIN",
    "DEFAULT_QUALIFICATION_CAP",
    "METHODS",
    "SCHEMA_VERSION",
    "DecisionMarginOptimalityError",
    "certify_minimum_l2_candidate",
    "decision_margin_bounds",
    "screen_scenario_layer",
    "select_layer",
]
