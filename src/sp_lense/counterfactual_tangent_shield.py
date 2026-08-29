"""Deterministic, model-free counterfactual tangent-shield constructions.

The primary construction solves the strictly convex problem

    minimize    0.5 * ||d||_2**2
    subject to  G d >= abs(b) + margin
                |H d| <= tau.

Rows whose nuisance bound is exactly zero are eliminated through an SVD null
basis.  Positive nuisance bounds remain symmetric linear inequalities.  An
optional L2 cap is certified after the minimum-norm solve: if the minimum-norm
point exceeds the cap, no feasible point can satisfy it.

The module performs no model calls.  All arithmetic, hashes, and certificates
use canonical CPU NumPy float64 arrays.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import __version__ as scipy_version
from scipy.optimize import linprog, minimize

SCHEMA_VERSION = "sp_lense.counterfactual_tangent_shield.v1"
SOLVER_METHOD = "SLSQP"
SOLVER_MAX_ITERATIONS = 2_000
SOLVER_FUNCTION_TOLERANCE = 1e-12
DEFAULT_SVD_RTOL = 1e-10
DEFAULT_SVD_ATOL = 1e-12
DEFAULT_CERTIFICATE_TOLERANCE = 2e-8


class TangentShieldError(RuntimeError):
    """Base class for fail-closed tangent-shield construction errors."""


class TangentShieldInfeasibleError(TangentShieldError):
    """The requested target, nuisance, and norm constraints are infeasible."""


class TangentShieldSolverError(TangentShieldError):
    """A numerical solver failed to return a strictly certified result."""


@dataclass(frozen=True, slots=True)
class TangentShieldDirection:
    """A canonical float64 direction and its machine-readable certificate."""

    direction: np.ndarray
    diagnostics: dict[str, Any]

    def as_record(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of the direction and diagnostics."""

        return {
            "direction": self.direction.tolist(),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class _PreparedProblem:
    target: np.ndarray
    offsets: np.ndarray
    margins: np.ndarray
    lower_bounds: np.ndarray
    nuisance: np.ndarray
    nuisance_bounds: np.ndarray
    exact_mask: np.ndarray
    soft_mask: np.ndarray
    null_basis: np.ndarray
    svd_diagnostics: dict[str, Any]
    l2_cap: float | None
    certificate_tolerance: float


def _finite_scalar(
    value: Any,
    *,
    field: str,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{field} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{field} must be nonnegative")
    return result


def _positive_scalar(value: Any, *, field: str) -> float:
    result = _finite_scalar(value, field=field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def _float64_array(value: Any, *, field: str, ndim: int) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except Exception as error:  # pragma: no cover - NumPy supplies the concrete cause.
        raise TypeError(f"{field} must be an array of real numbers") from error
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{field} must contain real numbers")
    if raw.ndim != ndim:
        raise ValueError(f"{field} must be {ndim}-dimensional")
    result = np.asarray(raw, dtype=np.float64, order="C").copy(order="C")
    if not np.isfinite(result).all():
        raise ValueError(f"{field} must contain only finite values")
    return result


def _matrix(
    value: Any,
    *,
    field: str,
    width: int | None = None,
    allow_empty_rows: bool = False,
) -> np.ndarray:
    result = _float64_array(value, field=field, ndim=2)
    if result.shape[1] == 0:
        raise ValueError(f"{field} must have nonzero width")
    if result.shape[0] == 0 and not allow_empty_rows:
        raise ValueError(f"{field} must contain at least one row")
    if width is not None and result.shape[1] != width:
        raise ValueError(f"{field} width must equal {width}")
    return result


def _vector(value: Any, *, field: str, length: int) -> np.ndarray:
    result = _float64_array(value, field=field, ndim=1)
    if result.shape != (length,):
        raise ValueError(f"{field} must have length {length}")
    return result


def _nonnegative_rows(value: Any, *, field: str, length: int) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except Exception as error:  # pragma: no cover - NumPy supplies the concrete cause.
        raise TypeError(f"{field} must be a real scalar or row vector") from error
    if raw.ndim == 0:
        scalar = _finite_scalar(raw.item(), field=field, nonnegative=True)
        return np.full(length, scalar, dtype=np.float64)
    result = _vector(value, field=field, length=length)
    if bool(np.any(result < 0.0)):
        raise ValueError(f"{field} must be nonnegative")
    return result


def _canonical_array(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64, order="C").copy(order="C")
    result[result == 0.0] = 0.0
    return result


def _array_sha256(value: np.ndarray) -> str:
    canonical = _canonical_array(value)
    header = json.dumps(
        {"dtype": "float64", "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    little_endian = canonical.astype("<f8", copy=False)
    return hashlib.sha256(header + b"\n" + little_endian.tobytes(order="C")).hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonicalize_basis_signs(basis: np.ndarray, *, vectors_are_columns: bool) -> None:
    count = basis.shape[1] if vectors_are_columns else basis.shape[0]
    for index in range(count):
        vector = basis[:, index] if vectors_are_columns else basis[index, :]
        anchor = int(np.argmax(np.abs(vector)))
        if vector[anchor] < 0.0:
            if vectors_are_columns:
                basis[:, index] *= -1.0
            else:
                basis[index, :] *= -1.0


def _svd_row_and_null_bases(
    rows: np.ndarray,
    *,
    svd_rtol: float,
    svd_atol: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    dimension = rows.shape[1]
    if rows.shape[0] == 0:
        row_basis = np.zeros((0, dimension), dtype=np.float64)
        null_basis = np.eye(dimension, dtype=np.float64)
        diagnostics = {
            "input_row_count": 0,
            "dimension": dimension,
            "row_normalization": "unit_l2_for_nonzero_rows",
            "zero_row_count": 0,
            "svd_rtol": svd_rtol,
            "svd_atol": svd_atol,
            "singular_values": [],
            "threshold": max(svd_atol, 0.0),
            "rank": 0,
            "null_dimension": dimension,
            "row_basis_sha256": _array_sha256(row_basis),
            "null_basis_sha256": _array_sha256(null_basis),
            "maximum_abs_input_null_residual": 0.0,
            "input_null_residual_tolerance": max(
                256.0 * np.finfo(np.float64).eps,
                svd_atol,
            ),
            "maximum_abs_null_orthonormality_residual": 0.0,
        }
        diagnostics["diagnostics_sha256"] = _json_sha256(diagnostics)
        return row_basis, null_basis, diagnostics

    row_norms = np.linalg.norm(rows, axis=1)
    nonzero = row_norms > 0.0
    normalized = np.zeros_like(rows)
    normalized[nonzero] = rows[nonzero] / row_norms[nonzero, None]
    _, singular_values, vh = np.linalg.svd(normalized, full_matrices=True)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    threshold = max(svd_atol, svd_rtol * largest)
    rank = int(np.count_nonzero(singular_values > threshold))
    row_basis = vh[:rank, :].copy(order="C")
    null_basis = vh[rank:, :].T.copy(order="C")
    _canonicalize_basis_signs(row_basis, vectors_are_columns=False)
    _canonicalize_basis_signs(null_basis, vectors_are_columns=True)

    input_null_residual = rows @ null_basis
    maximum_input_null_residual = (
        float(np.max(np.abs(input_null_residual))) if input_null_residual.size else 0.0
    )
    orthonormality = null_basis.T @ null_basis - np.eye(null_basis.shape[1])
    maximum_orthonormality_residual = (
        float(np.max(np.abs(orthonormality))) if orthonormality.size else 0.0
    )
    maximum_row_norm = float(np.max(row_norms)) if row_norms.size else 0.0
    span_tolerance = max(
        256.0 * np.finfo(np.float64).eps,
        threshold,
    ) * max(1.0, maximum_row_norm)
    if maximum_input_null_residual > span_tolerance:
        raise TangentShieldSolverError("the SVD null basis failed its original-row residual check")
    if maximum_orthonormality_residual > 256.0 * np.finfo(np.float64).eps:
        raise TangentShieldSolverError("the SVD null basis failed its orthonormality check")

    diagnostics = {
        "input_row_count": rows.shape[0],
        "dimension": dimension,
        "row_normalization": "unit_l2_for_nonzero_rows",
        "zero_row_count": int(np.count_nonzero(~nonzero)),
        "svd_rtol": svd_rtol,
        "svd_atol": svd_atol,
        "singular_values": singular_values.tolist(),
        "threshold": threshold,
        "rank": rank,
        "null_dimension": null_basis.shape[1],
        "normalized_rows_sha256": _array_sha256(normalized),
        "row_basis_sha256": _array_sha256(row_basis),
        "null_basis_sha256": _array_sha256(null_basis),
        "maximum_abs_input_null_residual": maximum_input_null_residual,
        "input_null_residual_tolerance": span_tolerance,
        "maximum_abs_null_orthonormality_residual": maximum_orthonormality_residual,
    }
    diagnostics["diagnostics_sha256"] = _json_sha256(diagnostics)
    return row_basis, null_basis, diagnostics


def _prepare_problem(
    target_rows: Any,
    target_offsets: Any,
    *,
    margin: Any,
    nuisance_rows: Any | None,
    nuisance_bound: Any,
    l2_cap: Any | None,
    svd_rtol: Any,
    svd_atol: Any,
    certificate_tolerance: Any,
) -> _PreparedProblem:
    target = _matrix(target_rows, field="target_rows")
    target_count, dimension = target.shape
    offsets = _vector(target_offsets, field="target_offsets", length=target_count)
    margins = _nonnegative_rows(margin, field="margin", length=target_count)
    lower_bounds = np.abs(offsets) + margins
    if not np.isfinite(lower_bounds).all():
        raise ValueError("abs(target_offsets) + margin must remain finite")

    nuisance = (
        np.zeros((0, dimension), dtype=np.float64)
        if nuisance_rows is None
        else _matrix(
            nuisance_rows,
            field="nuisance_rows",
            width=dimension,
            allow_empty_rows=True,
        )
    )
    nuisance_bounds = _nonnegative_rows(
        nuisance_bound,
        field="nuisance_bound",
        length=nuisance.shape[0],
    )
    exact_mask = nuisance_bounds == 0.0
    soft_mask = ~exact_mask

    relative_tolerance = _finite_scalar(svd_rtol, field="svd_rtol", nonnegative=True)
    absolute_tolerance = _finite_scalar(svd_atol, field="svd_atol", nonnegative=True)
    certificate = _positive_scalar(certificate_tolerance, field="certificate_tolerance")
    cap = (
        None
        if l2_cap is None
        else _finite_scalar(l2_cap, field="l2_cap", nonnegative=True)
    )
    _, null_basis, svd_diagnostics = _svd_row_and_null_bases(
        nuisance[exact_mask],
        svd_rtol=relative_tolerance,
        svd_atol=absolute_tolerance,
    )
    return _PreparedProblem(
        target=target,
        offsets=offsets,
        margins=margins,
        lower_bounds=lower_bounds,
        nuisance=nuisance,
        nuisance_bounds=nuisance_bounds,
        exact_mask=exact_mask,
        soft_mask=soft_mask,
        null_basis=null_basis,
        svd_diagnostics=svd_diagnostics,
        l2_cap=cap,
        certificate_tolerance=certificate,
    )


def _problem_input_record(problem: _PreparedProblem) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "target_rows_sha256": _array_sha256(problem.target),
        "target_offsets_sha256": _array_sha256(problem.offsets),
        "margin_sha256": _array_sha256(problem.margins),
        "target_lower_bounds_sha256": _array_sha256(problem.lower_bounds),
        "nuisance_rows_sha256": _array_sha256(problem.nuisance),
        "nuisance_bounds_sha256": _array_sha256(problem.nuisance_bounds),
        "target_constraint_count": problem.target.shape[0],
        "nuisance_constraint_count": problem.nuisance.shape[0],
        "exact_nuisance_row_count": int(np.count_nonzero(problem.exact_mask)),
        "soft_nuisance_row_count": int(np.count_nonzero(problem.soft_mask)),
        "dimension": problem.target.shape[1],
        "l2_cap": problem.l2_cap,
        "certificate_tolerance": problem.certificate_tolerance,
        "svd_diagnostics_sha256": problem.svd_diagnostics["diagnostics_sha256"],
    }


def _certificate(problem: _PreparedProblem, direction: np.ndarray) -> dict[str, Any]:
    target_values = problem.target @ direction
    target_residuals = target_values - problem.lower_bounds
    nuisance_values = problem.nuisance @ direction
    nuisance_slacks = problem.nuisance_bounds - np.abs(nuisance_values)
    norm = float(np.linalg.norm(direction))
    target_scale = max(1.0, float(np.max(np.abs(problem.lower_bounds))))
    nuisance_scale = max(
        1.0,
        float(np.max(problem.nuisance_bounds)) if problem.nuisance_bounds.size else 0.0,
        (
            float(np.linalg.norm(problem.nuisance, ord=2)) * max(1.0, norm)
            if problem.nuisance.size
            else 0.0
        ),
    )
    target_tolerance = problem.certificate_tolerance * target_scale
    nuisance_tolerance = problem.certificate_tolerance * nuisance_scale
    norm_tolerance = problem.certificate_tolerance * max(1.0, problem.l2_cap or 0.0)

    exact_values = nuisance_values[problem.exact_mask]
    soft_slacks = nuisance_slacks[problem.soft_mask]
    maximum_exact_residual = (
        float(np.max(np.abs(exact_values))) if exact_values.size else 0.0
    )
    minimum_soft_slack = float(np.min(soft_slacks)) if soft_slacks.size else None
    l2_cap_residual = problem.l2_cap - norm if problem.l2_cap is not None else None
    checks = {
        "finite": bool(
            np.isfinite(direction).all()
            and np.isfinite(target_values).all()
            and np.isfinite(nuisance_values).all()
            and math.isfinite(norm)
        ),
        "target": bool(float(np.min(target_residuals)) >= -target_tolerance),
        "exact_nuisance": bool(maximum_exact_residual <= nuisance_tolerance),
        "soft_nuisance": bool(
            minimum_soft_slack is None or minimum_soft_slack >= -nuisance_tolerance
        ),
        "l2_cap": bool(l2_cap_residual is None or l2_cap_residual >= -norm_tolerance),
    }
    return {
        "target_values": target_values.tolist(),
        "target_lower_bounds": problem.lower_bounds.tolist(),
        "target_constraint_residuals": target_residuals.tolist(),
        "minimum_target_constraint_residual": float(np.min(target_residuals)),
        "nuisance_values": nuisance_values.tolist(),
        "nuisance_bounds": problem.nuisance_bounds.tolist(),
        "nuisance_constraint_slacks": nuisance_slacks.tolist(),
        "maximum_abs_exact_nuisance_residual": maximum_exact_residual,
        "minimum_soft_nuisance_slack": minimum_soft_slack,
        "l2_norm": norm,
        "l2_cap": problem.l2_cap,
        "l2_cap_residual": l2_cap_residual,
        "target_tolerance": target_tolerance,
        "nuisance_tolerance": nuisance_tolerance,
        "norm_tolerance": norm_tolerance,
        "checks": checks,
        "passes": bool(all(checks.values())),
    }


def _frozen_solution(direction: np.ndarray, diagnostics: dict[str, Any]) -> TangentShieldDirection:
    canonical = _canonical_array(direction)
    canonical.setflags(write=False)
    payload = dict(diagnostics)
    payload["direction_sha256"] = _array_sha256(canonical)
    payload["diagnostics_sha256"] = _json_sha256(payload)
    return TangentShieldDirection(direction=canonical, diagnostics=payload)


def solve_minimum_l2_direction(
    target_rows: Any,
    target_offsets: Any,
    *,
    margin: Any = 0.0,
    nuisance_rows: Any | None = None,
    nuisance_bound: Any = 0.0,
    l2_cap: Any | None = None,
    svd_rtol: Any = DEFAULT_SVD_RTOL,
    svd_atol: Any = DEFAULT_SVD_ATOL,
    certificate_tolerance: Any = DEFAULT_CERTIFICATE_TOLERANCE,
) -> TangentShieldDirection:
    """Return the unique minimum-L2 direction satisfying all supplied constraints.

    The target lower bound is exactly ``abs(target_offsets) + margin``.  A scalar
    margin or nuisance bound is broadcast across rows; row vectors are also
    accepted.  Nuisance rows with zero bounds are exact equalities and positive
    bounds encode ``abs(H @ d) <= tau``.
    """

    problem = _prepare_problem(
        target_rows,
        target_offsets,
        margin=margin,
        nuisance_rows=nuisance_rows,
        nuisance_bound=nuisance_bound,
        l2_cap=l2_cap,
        svd_rtol=svd_rtol,
        svd_atol=svd_atol,
        certificate_tolerance=certificate_tolerance,
    )
    null_basis = problem.null_basis
    reduced_dimension = null_basis.shape[1]
    reduced_target = problem.target @ null_basis
    reduced_soft = problem.nuisance[problem.soft_mask] @ null_basis
    soft_bounds = problem.nuisance_bounds[problem.soft_mask]

    constraint_matrix = np.vstack((reduced_target, -reduced_soft, reduced_soft))
    constraint_lower = np.concatenate((problem.lower_bounds, -soft_bounds, -soft_bounds))
    # M z >= q is passed to linprog as -M z <= -q.
    if reduced_dimension == 0:
        reduced = np.zeros(0, dtype=np.float64)
        if bool(np.any(constraint_matrix @ reduced < constraint_lower)):
            raise TangentShieldInfeasibleError(
                "the exact nuisance span leaves no direction satisfying the linear constraints"
            )
        feasibility_status = 0
        feasibility_iterations = 0
        optimizer_status = 0
        optimizer_iterations = 0
    else:
        feasibility = linprog(
            np.zeros(reduced_dimension, dtype=np.float64),
            A_ub=-constraint_matrix,
            b_ub=-constraint_lower,
            bounds=[(None, None)] * reduced_dimension,
            method="highs",
        )
        feasibility_status = int(feasibility.status)
        feasibility_iterations = int(getattr(feasibility, "nit", -1))
        if not bool(feasibility.success):
            if feasibility_status == 2:
                raise TangentShieldInfeasibleError("the linear target/nuisance system is infeasible")
            raise TangentShieldSolverError("the deterministic linear feasibility solve failed")
        feasible_start = np.asarray(feasibility.x, dtype=np.float64)
        if feasible_start.shape != (reduced_dimension,) or not np.isfinite(feasible_start).all():
            raise TangentShieldSolverError("the feasibility solve returned an invalid point")

        def objective(value: np.ndarray) -> float:
            return 0.5 * float(value @ value)

        def objective_jac(value: np.ndarray) -> np.ndarray:
            return value.copy()

        constraints = {
            "type": "ineq",
            "fun": lambda value: constraint_matrix @ value - constraint_lower,
            "jac": lambda _value: constraint_matrix,
        }
        optimized = minimize(
            objective,
            feasible_start,
            jac=objective_jac,
            constraints=constraints,
            method=SOLVER_METHOD,
            options={
                "maxiter": SOLVER_MAX_ITERATIONS,
                "ftol": SOLVER_FUNCTION_TOLERANCE,
                "disp": False,
            },
        )
        optimizer_status = int(getattr(optimized, "status", -1))
        optimizer_iterations = int(getattr(optimized, "nit", -1))
        reduced = np.asarray(getattr(optimized, "x", np.empty(0)), dtype=np.float64)
        if (
            not bool(getattr(optimized, "success", False))
            or reduced.shape != (reduced_dimension,)
            or not np.isfinite(reduced).all()
        ):
            raise TangentShieldSolverError("the minimum-L2 solve did not converge")

    direction = _canonical_array(null_basis @ reduced)
    certificate = _certificate(problem, direction)
    if not certificate["passes"]:
        failed = ", ".join(
            name for name, passed in certificate["checks"].items() if not passed
        )
        if failed == "l2_cap":
            raise TangentShieldInfeasibleError("the minimum feasible L2 norm exceeds l2_cap")
        raise TangentShieldSolverError(f"the minimum-L2 result failed certification: {failed}")

    input_record = _problem_input_record(problem)
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "method": "minimum_l2_all_target_tangent_shield",
        "dtype": "float64",
        "scipy_version": scipy_version,
        "solver_method": SOLVER_METHOD,
        "solver_settings": {
            "linear_feasibility_method": "highs",
            "maximum_iterations": SOLVER_MAX_ITERATIONS,
            "function_tolerance": SOLVER_FUNCTION_TOLERANCE,
            "objective": "one_half_squared_l2_norm",
        },
        "input_record": input_record,
        "input_sha256": _json_sha256(input_record),
        "reduced_dimension": reduced_dimension,
        "svd": problem.svd_diagnostics,
        "linear_feasibility_status": feasibility_status,
        "linear_feasibility_iterations": feasibility_iterations,
        "optimizer_status": optimizer_status,
        "optimizer_iterations": optimizer_iterations,
        "certificate": certificate,
        "passes_strict_certificate": True,
        "deterministic_output": True,
    }
    return _frozen_solution(direction, diagnostics)


def build_projected_semantic_anchor_baseline(
    semantic_anchor: Any,
    target_rows: Any,
    target_offsets: Any,
    *,
    margin: Any = 0.0,
    nuisance_rows: Any | None = None,
    nuisance_bound: Any = 0.0,
    l2_cap: Any | None = None,
    svd_rtol: Any = DEFAULT_SVD_RTOL,
    svd_atol: Any = DEFAULT_SVD_ATOL,
    certificate_tolerance: Any = DEFAULT_CERTIFICATE_TOLERANCE,
) -> TangentShieldDirection:
    """Project an oriented semantic anchor and minimally scale it to all targets.

    The supplied sign is fixed; this function never flips the anchor after seeing
    the target rows.  Exact nuisance rows are projected out.  Positive nuisance
    bounds and the optional L2 cap impose upper limits on the scalar multiplier.
    """

    problem = _prepare_problem(
        target_rows,
        target_offsets,
        margin=margin,
        nuisance_rows=nuisance_rows,
        nuisance_bound=nuisance_bound,
        l2_cap=l2_cap,
        svd_rtol=svd_rtol,
        svd_atol=svd_atol,
        certificate_tolerance=certificate_tolerance,
    )
    anchor = _vector(
        semantic_anchor,
        field="semantic_anchor",
        length=problem.target.shape[1],
    )
    projected = _canonical_array(problem.null_basis @ (problem.null_basis.T @ anchor))
    projected_norm = float(np.linalg.norm(projected))
    slopes = problem.target @ projected
    soft_values = problem.nuisance[problem.soft_mask] @ projected
    soft_bounds = problem.nuisance_bounds[problem.soft_mask]

    lower_scale = 0.0
    upper_scale = math.inf
    for slope, bound in zip(slopes, problem.lower_bounds, strict=True):
        slope_value = float(slope)
        bound_value = float(bound)
        if slope_value > 0.0:
            lower_scale = max(lower_scale, bound_value / slope_value)
        elif bound_value > 0.0:
            raise TangentShieldInfeasibleError(
                "the projected semantic anchor has nonpositive slope for a positive target"
            )
        elif slope_value < 0.0:
            upper_scale = 0.0

    for nuisance_value, bound in zip(soft_values, soft_bounds, strict=True):
        magnitude = abs(float(nuisance_value))
        if magnitude > 0.0:
            upper_scale = min(upper_scale, float(bound) / magnitude)
    if problem.l2_cap is not None and projected_norm > 0.0:
        upper_scale = min(upper_scale, problem.l2_cap / projected_norm)

    scale_tolerance = problem.certificate_tolerance * max(
        1.0,
        lower_scale,
        upper_scale if math.isfinite(upper_scale) else 0.0,
    )
    if lower_scale > upper_scale + scale_tolerance:
        raise TangentShieldInfeasibleError(
            "the projected semantic anchor cannot satisfy targets within nuisance/norm limits"
        )
    if projected_norm == 0.0 and lower_scale > 0.0:
        raise TangentShieldInfeasibleError("the semantic anchor vanishes in the exact null space")

    direction = _canonical_array(lower_scale * projected)
    certificate = _certificate(problem, direction)
    if not certificate["passes"]:
        failed = ", ".join(
            name for name, passed in certificate["checks"].items() if not passed
        )
        raise TangentShieldSolverError(
            f"the projected semantic-anchor baseline failed certification: {failed}"
        )

    input_record = _problem_input_record(problem)
    input_record.update(
        {
            "semantic_anchor_sha256": _array_sha256(anchor),
            "projected_semantic_anchor_sha256": _array_sha256(projected),
        }
    )
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "method": "projected_semantic_anchor_baseline",
        "dtype": "float64",
        "input_record": input_record,
        "input_sha256": _json_sha256(input_record),
        "svd": problem.svd_diagnostics,
        "orientation_rule": "supplied_anchor_sign_no_posthoc_flip",
        "semantic_anchor_norm": float(np.linalg.norm(anchor)),
        "projected_semantic_anchor_norm": projected_norm,
        "target_slopes_before_scaling": slopes.tolist(),
        "soft_nuisance_values_before_scaling": soft_values.tolist(),
        "minimum_required_scale": lower_scale,
        "maximum_allowed_scale": upper_scale if math.isfinite(upper_scale) else None,
        "selected_scale": lower_scale,
        "certificate": certificate,
        "passes_strict_certificate": True,
        "deterministic_output": True,
    }
    return _frozen_solution(direction, diagnostics)


def build_seeded_random_null_control(
    dimension: Any,
    target_norm: Any,
    *,
    seed: Any,
    nuisance_rows: Any | None = None,
    svd_rtol: Any = DEFAULT_SVD_RTOL,
    svd_atol: Any = DEFAULT_SVD_ATOL,
    certificate_tolerance: Any = DEFAULT_CERTIFICATE_TOLERANCE,
) -> TangentShieldDirection:
    """Return a seeded isotropic control projected into the exact nuisance null."""

    if isinstance(dimension, (bool, np.bool_)) or not isinstance(
        dimension, (int, np.integer)
    ):
        raise TypeError("dimension must be an integer")
    width = int(dimension)
    if width <= 0:
        raise ValueError("dimension must be positive")
    norm = _finite_scalar(target_norm, field="target_norm", nonnegative=True)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed_value = int(seed)
    if seed_value < 0:
        raise ValueError("seed must be nonnegative")
    relative_tolerance = _finite_scalar(svd_rtol, field="svd_rtol", nonnegative=True)
    absolute_tolerance = _finite_scalar(svd_atol, field="svd_atol", nonnegative=True)
    certificate = _positive_scalar(certificate_tolerance, field="certificate_tolerance")
    nuisance = (
        np.zeros((0, width), dtype=np.float64)
        if nuisance_rows is None
        else _matrix(
            nuisance_rows,
            field="nuisance_rows",
            width=width,
            allow_empty_rows=True,
        )
    )
    _, null_basis, svd_diagnostics = _svd_row_and_null_bases(
        nuisance,
        svd_rtol=relative_tolerance,
        svd_atol=absolute_tolerance,
    )
    if norm > 0.0 and null_basis.shape[1] == 0:
        raise TangentShieldInfeasibleError(
            "a positive-norm random control does not exist in a zero-dimensional null space"
        )

    rng = np.random.default_rng(seed_value)
    attempt = 0
    if norm == 0.0:
        direction = np.zeros(width, dtype=np.float64)
    else:
        projected = np.zeros(width, dtype=np.float64)
        projected_norm = 0.0
        for attempt in range(1, 17):
            sample = rng.standard_normal(width, dtype=np.float64)
            projected = null_basis @ (null_basis.T @ sample)
            projected_norm = float(np.linalg.norm(projected))
            minimum_usable_norm = (
                256.0
                * np.finfo(np.float64).eps
                * max(1.0, float(np.linalg.norm(sample)))
            )
            if projected_norm > minimum_usable_norm:
                break
        else:  # pragma: no cover - requires adversarial RNG replacement.
            raise TangentShieldSolverError("seeded sampling did not produce a usable null vector")
        direction = _canonical_array((norm / projected_norm) * projected)

    nuisance_values = nuisance @ direction
    realized_norm = float(np.linalg.norm(direction))
    nuisance_scale = max(
        1.0,
        (
            float(np.linalg.norm(nuisance, ord=2)) * max(1.0, realized_norm)
            if nuisance.size
            else 0.0
        ),
    )
    nuisance_tolerance = certificate * nuisance_scale
    norm_tolerance = certificate * max(1.0, norm)
    maximum_nuisance_residual = (
        float(np.max(np.abs(nuisance_values))) if nuisance_values.size else 0.0
    )
    checks = {
        "finite": bool(np.isfinite(direction).all() and math.isfinite(realized_norm)),
        "exact_nuisance": bool(maximum_nuisance_residual <= nuisance_tolerance),
        "matched_norm": bool(abs(realized_norm - norm) <= norm_tolerance),
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise TangentShieldSolverError(f"the random-null control failed certification: {failed}")

    input_record = {
        "schema_version": SCHEMA_VERSION,
        "dimension": width,
        "target_norm": norm,
        "seed": seed_value,
        "nuisance_rows_sha256": _array_sha256(nuisance),
        "svd_diagnostics_sha256": svd_diagnostics["diagnostics_sha256"],
        "certificate_tolerance": certificate,
    }
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "method": "seeded_random_exact_null_control",
        "dtype": "float64",
        "rng": "numpy.default_rng.PCG64",
        "input_record": input_record,
        "input_sha256": _json_sha256(input_record),
        "svd": svd_diagnostics,
        "sample_attempt": attempt,
        "nuisance_values": nuisance_values.tolist(),
        "maximum_abs_exact_nuisance_residual": maximum_nuisance_residual,
        "requested_l2_norm": norm,
        "realized_l2_norm": realized_norm,
        "nuisance_tolerance": nuisance_tolerance,
        "norm_tolerance": norm_tolerance,
        "certificate_checks": checks,
        "passes_strict_certificate": True,
        "deterministic_output": True,
    }
    return _frozen_solution(direction, diagnostics)


__all__ = [
    "DEFAULT_CERTIFICATE_TOLERANCE",
    "DEFAULT_SVD_ATOL",
    "DEFAULT_SVD_RTOL",
    "SCHEMA_VERSION",
    "TangentShieldDirection",
    "TangentShieldError",
    "TangentShieldInfeasibleError",
    "TangentShieldSolverError",
    "build_projected_semantic_anchor_baseline",
    "build_seeded_random_null_control",
    "solve_minimum_l2_direction",
]
