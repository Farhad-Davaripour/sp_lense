"""Certified row-space solver for Decision-Margin Shielding.

This module is a model-free numerical amendment.  It solves the same convex
minimum-Euclidean-norm problem certified by :mod:`decision_margin_shield`, but
does the numerical optimization only in the span of the projected inequality
rows.  The restriction is lossless: the component orthogonal to every
constraint can only increase the norm and cannot improve feasibility.

No optimizer status is treated as a proof.  Every returned direction must pass
the independent primal, dual-lower-bound, and KKT certificate implemented in
``certify_minimum_l2_candidate``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import __version__ as scipy_version
from scipy.optimize import linprog, minimize

from .counterfactual_tangent_shield import (
    TangentShieldDirection,
    TangentShieldInfeasibleError,
    TangentShieldSolverError,
)
from .decision_margin_shield import (
    DEFAULT_SVD_ATOL,
    DEFAULT_SVD_RTOL,
    DecisionMarginOptimalityError,
    certify_minimum_l2_candidate,
)
from .factorial_causal_anchor import canonical_sha256

SCHEMA_VERSION = "sp_lense.decision_margin_shield_rowspace.v1"
SOLVER_METHOD = "SLSQP"
SOLVER_MAX_ITERATIONS = 2_000
SOLVER_FUNCTION_TOLERANCE = 1e-12
STRICT_RAW_FEASIBILITY_FRACTION = 1e-7
STRICT_RAW_ROUNDOFF_MULTIPLIER = 2_048.0


def _finite_matrix(value: Any, *, field: str, allow_empty_rows: bool = False) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except Exception as error:  # pragma: no cover - NumPy supplies the concrete cause.
        raise TypeError(f"{field} must be an array") from error
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{field} must contain real numbers")
    result = np.asarray(raw, dtype=np.float64, order="C").copy(order="C")
    if result.ndim != 2 or result.shape[1] == 0:
        raise ValueError(f"{field} must be a two-dimensional matrix with nonzero width")
    if result.shape[0] == 0 and not allow_empty_rows:
        raise ValueError(f"{field} must contain at least one row")
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


def _nonnegative_rows(value: Any, *, field: str, length: int) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except Exception as error:  # pragma: no cover - NumPy supplies the concrete cause.
        raise TypeError(f"{field} must be a real scalar or row vector") from error
    if raw.ndim == 0:
        if raw.dtype.kind not in "iuf" or isinstance(raw.item(), (bool, np.bool_)):
            raise TypeError(f"{field} must contain real numbers")
        scalar = float(raw)
        if not math.isfinite(scalar) or scalar < 0.0:
            raise ValueError(f"{field} must be finite and nonnegative")
        return np.full(length, scalar, dtype=np.float64)
    result = _finite_vector(value, field=field, length=length)
    if bool(np.any(result < 0.0)):
        raise ValueError(f"{field} must be nonnegative")
    return result


def _canonical_array(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64, order="C").copy(order="C")
    result[result == 0.0] = 0.0
    return result


def _array_sha256(value: np.ndarray) -> str:
    return canonical_sha256(_canonical_array(value).tolist())


def _canonicalize_row_signs(basis: np.ndarray) -> None:
    for row in basis:
        anchor = int(np.argmax(np.abs(row)))
        if row[anchor] < 0.0:
            row *= -1.0


def _normalized_row_basis(
    rows: np.ndarray,
    *,
    rank_rule: str,
    svd_rtol: float | None = None,
    svd_atol: float | None = None,
    label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a deterministic orthonormal basis for normalized input rows."""

    if rank_rule == "fixed_scientific_equality_tolerance":
        if svd_rtol is None or svd_atol is None:
            raise ValueError("the equality rank rule requires fixed SVD tolerances")
    elif rank_rule == "machine_precision_representer_span":
        if svd_rtol is not None or svd_atol is not None:
            raise ValueError("the representer rank rule derives its threshold from machine precision")
    else:  # pragma: no cover - private call contract.
        raise ValueError("unknown normalized-row rank rule")

    dimension = rows.shape[1]
    row_norms = np.linalg.norm(rows, axis=1)
    nonzero = row_norms > 0.0
    normalized = np.zeros_like(rows)
    normalized[nonzero] = rows[nonzero] / row_norms[nonzero, None]
    if rows.shape[0] == 0:
        singular_values = np.zeros(0, dtype=np.float64)
        basis = np.zeros((0, dimension), dtype=np.float64)
        threshold = float(svd_atol or 0.0)
        relative_threshold = float(svd_rtol or 0.0)
    else:
        _, singular_values, vh = np.linalg.svd(normalized, full_matrices=False)
        largest = float(singular_values[0]) if singular_values.size else 0.0
        if rank_rule == "fixed_scientific_equality_tolerance":
            relative_threshold = float(svd_rtol)
            threshold = max(float(svd_atol), relative_threshold * largest)
        else:
            relative_threshold = np.finfo(np.float64).eps * max(normalized.shape)
            threshold = relative_threshold * largest
        rank = int(np.count_nonzero(singular_values > threshold))
        basis = vh[:rank].copy(order="C")
        _canonicalize_row_signs(basis)

    reconstructed = (rows @ basis.T) @ basis
    projection_residual = rows - reconstructed
    maximum_projection_residual = (
        float(np.max(np.abs(projection_residual))) if projection_residual.size else 0.0
    )
    orthonormality = basis @ basis.T - np.eye(basis.shape[0], dtype=np.float64)
    maximum_orthonormality_residual = (
        float(np.max(np.abs(orthonormality))) if orthonormality.size else 0.0
    )
    maximum_row_norm = float(np.max(row_norms)) if row_norms.size else 0.0
    projection_tolerance = max(256.0 * np.finfo(np.float64).eps, threshold) * max(
        1.0, maximum_row_norm
    )
    orthonormality_tolerance = 256.0 * np.finfo(np.float64).eps * max(
        1, basis.shape[0]
    )
    checks = {
        "rowspace_reconstruction": bool(
            maximum_projection_residual <= projection_tolerance
        ),
        "row_basis_orthonormality": bool(
            maximum_orthonormality_residual <= orthonormality_tolerance
        ),
    }
    diagnostics = {
        "label": label,
        "rank_rule": rank_rule,
        "input_row_count": int(rows.shape[0]),
        "dimension": int(dimension),
        "row_normalization": "unit_l2_for_nonzero_rows",
        "zero_row_count": int(np.count_nonzero(~nonzero)),
        "svd_rtol": svd_rtol,
        "svd_atol": svd_atol,
        "effective_relative_rank_threshold": relative_threshold,
        "singular_values": singular_values.tolist(),
        "threshold": threshold,
        "rank": int(basis.shape[0]),
        "row_norms_sha256": _array_sha256(row_norms),
        "normalized_rows_sha256": _array_sha256(normalized),
        "row_basis_sha256": _array_sha256(basis),
        "maximum_abs_input_rowspace_projection_residual": maximum_projection_residual,
        "input_rowspace_projection_tolerance": projection_tolerance,
        "maximum_abs_row_basis_orthonormality_residual": maximum_orthonormality_residual,
        "row_basis_orthonormality_tolerance": orthonormality_tolerance,
        "checks": checks,
        "passes": bool(all(checks.values())),
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return basis, diagnostics


def _strict_raw_coordinate_certificate(
    *,
    direction: np.ndarray,
    target: np.ndarray,
    target_lower: np.ndarray,
    margins: np.ndarray,
    nuisance: np.ndarray,
    nuisance_bounds: np.ndarray,
    exact_mask: np.ndarray,
    soft_mask: np.ndarray,
) -> dict[str, Any]:
    """Certify raw scientific units without an operator-norm-scaled tolerance."""

    eps = np.finfo(np.float64).eps
    target_values = target @ direction
    target_slacks = target_values - target_lower
    target_references = np.where(
        margins > 0.0,
        margins,
        np.where(target_lower > 0.0, target_lower, 1.0),
    )
    target_tolerances = np.maximum(
        STRICT_RAW_FEASIBILITY_FRACTION * target_references,
        STRICT_RAW_ROUNDOFF_MULTIPLIER * eps * np.maximum(1.0, target_references),
    )

    nuisance_values = nuisance @ direction
    nuisance_slacks = nuisance_bounds - np.abs(nuisance_values)
    positive_target_reference = (
        float(np.max(margins[margins > 0.0]))
        if bool(np.any(margins > 0.0))
        else (
            float(np.max(target_lower[target_lower > 0.0]))
            if bool(np.any(target_lower > 0.0))
            else 1.0
        )
    )
    nuisance_references = np.where(
        nuisance_bounds > 0.0,
        nuisance_bounds,
        positive_target_reference,
    )
    nuisance_tolerances = np.maximum(
        STRICT_RAW_FEASIBILITY_FRACTION * nuisance_references,
        STRICT_RAW_ROUNDOFF_MULTIPLIER * eps * np.maximum(1.0, nuisance_references),
    )
    exact_values = nuisance_values[exact_mask]
    exact_tolerances = nuisance_tolerances[exact_mask]
    soft_slacks = nuisance_slacks[soft_mask]
    soft_tolerances = nuisance_tolerances[soft_mask]
    checks = {
        "finite": bool(
            np.isfinite(target_values).all()
            and np.isfinite(target_slacks).all()
            and np.isfinite(nuisance_values).all()
            and np.isfinite(nuisance_slacks).all()
        ),
        "target_lower_bounds_in_raw_units": bool(
            np.all(target_slacks >= -target_tolerances)
        ),
        "exact_nuisance_in_raw_units": bool(
            np.all(np.abs(exact_values) <= exact_tolerances)
        ),
        "soft_nuisance_bounds_in_raw_units": bool(
            np.all(soft_slacks >= -soft_tolerances)
        ),
    }
    record = {
        "schema_version": "sp_lense.decision_margin_shield_rowspace.raw_certificate.v1",
        "tolerance_rule": (
            "max(1e-7_times_fixed_margin_or_bound,2048_times_float64_epsilon_times_"
            "nonoperator_reference)"
        ),
        "operator_norm_used_in_tolerance": False,
        "strict_feasibility_fraction": STRICT_RAW_FEASIBILITY_FRACTION,
        "roundoff_multiplier": STRICT_RAW_ROUNDOFF_MULTIPLIER,
        "target_values": target_values.tolist(),
        "target_lower_bounds": target_lower.tolist(),
        "target_slacks": target_slacks.tolist(),
        "target_tolerances": target_tolerances.tolist(),
        "minimum_target_slack": float(np.min(target_slacks)),
        "nuisance_values": nuisance_values.tolist(),
        "nuisance_bounds": nuisance_bounds.tolist(),
        "nuisance_slacks": nuisance_slacks.tolist(),
        "nuisance_tolerances": nuisance_tolerances.tolist(),
        "maximum_abs_exact_nuisance_residual": (
            float(np.max(np.abs(exact_values))) if exact_values.size else 0.0
        ),
        "minimum_soft_nuisance_slack": (
            float(np.min(soft_slacks)) if soft_slacks.size else None
        ),
        "checks": checks,
        "passes": bool(all(checks.values())),
    }
    record["certificate_sha256"] = canonical_sha256(record)
    return record


def _freeze_solution(
    direction: np.ndarray,
    diagnostics: dict[str, Any],
) -> TangentShieldDirection:
    canonical = _canonical_array(direction)
    canonical.setflags(write=False)
    payload = dict(diagnostics)
    payload["direction_sha256"] = _array_sha256(canonical)
    payload["diagnostics_sha256"] = canonical_sha256(payload)
    return TangentShieldDirection(direction=canonical, diagnostics=payload)


def solve_certified_rowspace_minimum_l2_direction(
    target_rows: Any,
    target_offsets: Any,
    *,
    margin: Any = 0.05,
    nuisance_rows: Any | None = None,
    nuisance_bound: Any = 0.0,
    l2_cap: Any | None = None,
) -> TangentShieldDirection:
    """Solve and independently certify the DMS minimum-L2 direction.

    Exact-zero nuisance bounds are equalities.  Positive nuisance bounds are
    symmetric slabs.  Equalities are removed with a normalized-row SVD, then the
    optimization is restricted to the span of every projected inequality row.
    A finite SLSQP status-8 point is accepted only when the independent DMS
    certificate proves primal feasibility, a tight dual bound, and KKT residuals.
    """

    if l2_cap is not None:
        raise ValueError(
            "the certified row-space amendment is uncapped; l2_cap must be None"
        )

    target = _finite_matrix(target_rows, field="target_rows")
    target_count, dimension = target.shape
    offsets = _finite_vector(target_offsets, field="target_offsets", length=target_count)
    margins = _nonnegative_rows(margin, field="margin", length=target_count)
    lower_target = np.abs(offsets) + margins
    if not np.isfinite(lower_target).all():
        raise ValueError("abs(target_offsets) + margin must remain finite")

    nuisance = (
        np.zeros((0, dimension), dtype=np.float64)
        if nuisance_rows is None
        else _finite_matrix(
            nuisance_rows,
            field="nuisance_rows",
            allow_empty_rows=True,
        )
    )
    if nuisance.shape[1] != dimension:
        raise ValueError("nuisance_rows width must match target_rows")
    nuisance_bounds = _nonnegative_rows(
        nuisance_bound,
        field="nuisance_bound",
        length=nuisance.shape[0],
    )
    exact_mask = nuisance_bounds == 0.0
    soft_mask = ~exact_mask
    equality_rows = nuisance[exact_mask]
    soft_rows = nuisance[soft_mask]
    soft_bounds = nuisance_bounds[soft_mask]

    # These are exactly the inequalities consumed by the independent certificate.
    inequalities = np.vstack((target, -soft_rows, soft_rows))
    inequality_lower = np.concatenate((lower_target, -soft_bounds, -soft_bounds))

    equality_basis, equality_diagnostics = _normalized_row_basis(
        equality_rows,
        rank_rule="fixed_scientific_equality_tolerance",
        svd_rtol=DEFAULT_SVD_RTOL,
        svd_atol=DEFAULT_SVD_ATOL,
        label="exact_nuisance_equality_rowspace",
    )
    if not equality_diagnostics["passes"]:
        failed = ", ".join(
            name
            for name, passed in equality_diagnostics["checks"].items()
            if not passed
        )
        raise TangentShieldSolverError(
            "the exact-equality row-space diagnostics failed: " + failed
        )
    projected_inequalities = inequalities - (
        inequalities @ equality_basis.T
    ) @ equality_basis

    # Unit-row scaling is fixed before both feasibility and optimization.  This
    # removes the severe gradient-scale conditioning that made full-space SLSQP
    # stall, while leaving every nonzero half-space mathematically unchanged.
    projected_norms = np.linalg.norm(projected_inequalities, axis=1)
    nonzero_projected_mask = projected_norms > 0.0
    impossible_zero_rows = (~nonzero_projected_mask) & (inequality_lower > 0.0)
    if bool(np.any(impossible_zero_rows)):
        raise TangentShieldInfeasibleError(
            "an inequality with a positive lower bound vanishes in the exact nuisance null space"
        )
    retained_indices = np.flatnonzero(nonzero_projected_mask)
    scaled_inequalities = (
        projected_inequalities[retained_indices]
        / projected_norms[retained_indices, None]
    )
    scaled_lower = inequality_lower[retained_indices] / projected_norms[retained_indices]

    rowspace_basis, rowspace_diagnostics = _normalized_row_basis(
        scaled_inequalities,
        rank_rule="machine_precision_representer_span",
        label="projected_inequality_representer_rowspace",
    )
    if not rowspace_diagnostics["passes"]:
        failed = ", ".join(
            name
            for name, passed in rowspace_diagnostics["checks"].items()
            if not passed
        )
        raise TangentShieldSolverError(
            "the representer row-space diagnostics failed: " + failed
        )
    rowspace_rank = rowspace_basis.shape[0]
    if rowspace_rank > inequalities.shape[0]:  # pragma: no cover - linear algebra invariant.
        raise TangentShieldSolverError("the representer row-space rank exceeds its row count")
    reduced_inequalities = scaled_inequalities @ rowspace_basis.T

    if rowspace_rank == 0:
        reduced = np.zeros(0, dtype=np.float64)
        if bool(np.any(scaled_lower > 0.0)):
            raise TangentShieldInfeasibleError(
                "the projected inequality row space has no feasible target direction"
            )
        feasibility_success = True
        feasibility_status = 0
        feasibility_iterations = 0
        feasibility_message = "empty feasible representer row space"
        optimizer_success = True
        optimizer_status = 0
        optimizer_iterations = 0
        optimizer_message = "empty feasible representer row space"
        optimizer_objective = 0.0
    else:
        feasibility = linprog(
            np.zeros(rowspace_rank, dtype=np.float64),
            A_ub=-reduced_inequalities,
            b_ub=-scaled_lower,
            bounds=[(None, None)] * rowspace_rank,
            method="highs",
        )
        feasibility_success = bool(feasibility.success)
        feasibility_status = int(feasibility.status)
        feasibility_iterations = int(getattr(feasibility, "nit", -1))
        feasibility_message = str(getattr(feasibility, "message", ""))
        if not feasibility_success:
            if feasibility_status == 2:
                raise TangentShieldSolverError(
                    "HiGHS reported row-space infeasibility without an independent "
                    "infeasibility certificate; geometry is numerically indeterminate"
                )
            raise TangentShieldSolverError(
                "the deterministic row-space linear feasibility solve failed"
            )
        feasible_start = np.asarray(feasibility.x, dtype=np.float64)
        if feasible_start.shape != (rowspace_rank,) or not np.isfinite(feasible_start).all():
            raise TangentShieldSolverError(
                "the row-space linear feasibility solve returned an invalid point"
            )

        def objective(value: np.ndarray) -> float:
            return 0.5 * float(value @ value)

        def objective_jacobian(value: np.ndarray) -> np.ndarray:
            return value.copy()

        constraints = {
            "type": "ineq",
            "fun": lambda value: reduced_inequalities @ value - scaled_lower,
            "jac": lambda _value: reduced_inequalities,
        }
        optimized = minimize(
            objective,
            feasible_start,
            jac=objective_jacobian,
            constraints=constraints,
            method=SOLVER_METHOD,
            options={
                "maxiter": SOLVER_MAX_ITERATIONS,
                "ftol": SOLVER_FUNCTION_TOLERANCE,
                "disp": False,
            },
        )
        optimizer_success = bool(getattr(optimized, "success", False))
        optimizer_status = int(getattr(optimized, "status", -1))
        optimizer_iterations = int(getattr(optimized, "nit", -1))
        optimizer_message = str(getattr(optimized, "message", ""))
        reduced = np.asarray(getattr(optimized, "x", np.empty(0)), dtype=np.float64)
        optimizer_objective = float(getattr(optimized, "fun", math.nan))
        if reduced.shape != (rowspace_rank,) or not np.isfinite(reduced).all():
            raise TangentShieldSolverError(
                "the row-space minimum-L2 solve returned an invalid point"
            )
        if not optimizer_success and optimizer_status != 8:
            raise TangentShieldSolverError(
                "the row-space minimum-L2 solve neither converged nor returned allowed status 8"
            )

    direction = _canonical_array(rowspace_basis.T @ reduced)
    raw_certificate = _strict_raw_coordinate_certificate(
        direction=direction,
        target=target,
        target_lower=lower_target,
        margins=margins,
        nuisance=nuisance,
        nuisance_bounds=nuisance_bounds,
        exact_mask=exact_mask,
        soft_mask=soft_mask,
    )
    if not raw_certificate["passes"]:
        failed = ", ".join(
            name for name, passed in raw_certificate["checks"].items() if not passed
        )
        raise DecisionMarginOptimalityError(
            "the row-space candidate failed strict raw-coordinate certification: "
            + failed
        )
    certificate = certify_minimum_l2_candidate(
        direction,
        target,
        offsets,
        margin=margins,
        nuisance_rows=nuisance,
        nuisance_bound=nuisance_bounds,
    )
    if not certificate["passes"]:
        failed = ", ".join(
            name for name, passed in certificate["checks"].items() if not passed
        )
        raise DecisionMarginOptimalityError(
            "the row-space candidate failed independent optimality certification: " + failed
        )

    input_record = {
        "schema_version": SCHEMA_VERSION,
        "target_rows_sha256": _array_sha256(target),
        "target_offsets_sha256": _array_sha256(offsets),
        "margins_sha256": _array_sha256(margins),
        "nuisance_rows_sha256": _array_sha256(nuisance),
        "nuisance_bounds_sha256": _array_sha256(nuisance_bounds),
        "target_constraint_count": int(target_count),
        "exact_nuisance_row_count": int(np.count_nonzero(exact_mask)),
        "soft_nuisance_row_count": int(np.count_nonzero(soft_mask)),
        "dimension": int(dimension),
        "l2_cap": None,
    }
    scaling_record = {
        "rule": "divide_each_nonzero_projected_inequality_and_bound_by_row_l2_norm",
        "projected_inequality_count": int(projected_inequalities.shape[0]),
        "retained_nonzero_projected_indices": retained_indices.tolist(),
        "zero_projected_row_count": int(np.count_nonzero(~nonzero_projected_mask)),
        "projected_row_norms": projected_norms.tolist(),
        "projected_row_norms_sha256": _array_sha256(projected_norms),
        "scaled_inequalities_sha256": _array_sha256(scaled_inequalities),
        "scaled_lower_bounds_sha256": _array_sha256(scaled_lower),
    }
    scaling_record["scaling_sha256"] = canonical_sha256(scaling_record)
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "method": "certified_projected_inequality_rowspace_minimum_l2",
        "dtype": "float64",
        "scipy_version": scipy_version,
        "input_record": input_record,
        "input_sha256": canonical_sha256(input_record),
        "equality_projection": equality_diagnostics,
        "projected_inequalities_sha256": _array_sha256(projected_inequalities),
        "inequality_scaling": scaling_record,
        "representer_rowspace": rowspace_diagnostics,
        "reduced_inequalities_sha256": _array_sha256(reduced_inequalities),
        "reduced_dimension": int(rowspace_rank),
        "rank_at_most_inequality_count": bool(rowspace_rank <= inequalities.shape[0]),
        "solver_method": "scipy_linprog_highs_then_slsqp",
        "solver_settings": {
            "linear_feasibility_method": "highs",
            "minimum_l2_method": SOLVER_METHOD,
            "maximum_iterations": SOLVER_MAX_ITERATIONS,
            "function_tolerance": SOLVER_FUNCTION_TOLERANCE,
            "objective": "one_half_squared_euclidean_l2_in_orthonormal_rowspace",
            "fixed_constraint_scaling": "unit_l2_projected_rows",
        },
        "linear_feasibility_success": feasibility_success,
        "linear_feasibility_status": feasibility_status,
        "linear_feasibility_iterations": feasibility_iterations,
        "linear_feasibility_message": feasibility_message,
        "optimizer_success": optimizer_success,
        "optimizer_status": optimizer_status,
        "optimizer_iterations": optimizer_iterations,
        "optimizer_message": optimizer_message,
        "optimizer_reported_objective": optimizer_objective,
        "status_8_requires_and_received_independent_certificate": bool(
            optimizer_status == 8
        ),
        "strict_raw_coordinate_certificate": raw_certificate,
        "optimality_certificate": certificate,
        "passes_strict_certificate": True,
        "determinism_scope": "deterministic_within_pinned_runtime",
    }
    return _freeze_solution(direction, diagnostics)


__all__ = [
    "SCHEMA_VERSION",
    "solve_certified_rowspace_minimum_l2_direction",
]
