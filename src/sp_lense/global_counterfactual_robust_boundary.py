"""Deterministic convex solver for global counterfactual robust boundaries.

This module is deliberately model-free.  Given affine target margins, protected
constraints, an unrelated-task equality span, and optional convex quadratic
budgets, it solves

    maximize    gamma
    subject to  A d - b >= gamma
                B d = 0
                C d >= q
                ||d||_2 <= rho
                0.5 d.T H d <= kappa
                0.5 ||R_j d||_2**2 <= kappa_j.

All numerical work is CPU NumPy/SciPy float64.  The target inequalities define
the positive orientation, so the returned direction is never post-hoc sign
flipped.  A second, fixed minimum-norm solve removes avoidable ambiguity among
near-optimal directions.  SciPy status is never trusted by itself: every primal
residual and the reported max-min objective are recomputed from the original
inputs, and uncertified output raises instead of being returned.

This is a numerical certificate, not a proof of global optimality from an exact
conic solver.  The problem is convex after negating the linear objective, but the
implementation still reports the observed multi-start spread and the fixed
near-optimality allowance used by the canonical tie-break.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import __version__ as scipy_version
from scipy.optimize import minimize as _MINIMIZE

_DUAL_MINIMIZE = _MINIMIZE

SCHEMA_VERSION = "sp_lense.global_counterfactual_robust_boundary.v1"
SOLVER_METHOD = "SLSQP"
SOLVER_MAX_ITERATIONS = 2_000
SOLVER_FUNCTION_TOLERANCE = 1e-12
PRIMAL_CERTIFICATE_TOLERANCE = 2e-8
CANONICAL_GAMMA_RELAXATION = 1e-9
MAX_DETERMINISTIC_STARTS = 12


class GCRBSError(RuntimeError):
    """Base class for fail-closed GCRBS solver errors."""


class GCRBSInfeasibleError(GCRBSError):
    """The protected constraints and budgets could not be certified feasible."""


class GCRBSSolverError(GCRBSError):
    """The optimizer did not return a strictly certified solution."""


@dataclass(frozen=True, slots=True)
class GCRBSSolution:
    """A certified direction, worst target margin, and machine-readable diagnostics."""

    direction: np.ndarray
    gamma: float
    diagnostics: dict[str, Any]

    def as_record(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of the solution."""

        return {
            "direction": self.direction.tolist(),
            "gamma": self.gamma,
            "diagnostics": dict(self.diagnostics),
        }


def _finite_scalar(value: Any, *, field: str, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{field} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{field} must be nonnegative")
    return result


def _float64_array(value: Any, *, field: str, ndim: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{field} must contain real numbers")
    if raw.ndim != ndim:
        raise ValueError(f"{field} must be {ndim}-dimensional")
    result = np.asarray(raw, dtype=np.float64, order="C")
    if not np.isfinite(result).all():
        raise ValueError(f"{field} must contain only finite values")
    return result.copy(order="C")


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
        raise ValueError(f"{field} width must equal the target dimension")
    return result


def _vector(value: Any, *, field: str, length: int) -> np.ndarray:
    result = _float64_array(value, field=field, ndim=1)
    if result.shape != (length,):
        raise ValueError(f"{field} length is inconsistent with its matrix")
    return result


def _canonical_array(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array, dtype=np.float64, order="C").copy(order="C")
    result[result == 0.0] = 0.0
    return result


def _array_sha256(array: np.ndarray) -> str:
    canonical = _canonical_array(array)
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


def _validate_symmetric_psd(matrix: np.ndarray, *, field: str) -> tuple[np.ndarray, float]:
    symmetry_scale = max(1.0, float(np.linalg.norm(matrix, ord=2)))
    symmetry_error = float(np.max(np.abs(matrix - matrix.T)))
    if symmetry_error > 1e-12 * symmetry_scale:
        raise ValueError(f"{field} must be symmetric")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    minimum = float(eigenvalues[0])
    psd_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if minimum < -1e-12 * psd_scale:
        raise ValueError(f"{field} must be positive semidefinite")
    return symmetric, minimum


def _null_space(equality: np.ndarray) -> tuple[np.ndarray, int, float]:
    dimension = equality.shape[1]
    if equality.shape[0] == 0:
        return np.eye(dimension, dtype=np.float64), 0, 0.0
    _, singular_values, vh = np.linalg.svd(equality, full_matrices=True)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    threshold = max(equality.shape) * np.finfo(np.float64).eps * largest
    rank = int(np.count_nonzero(singular_values > threshold))
    null = vh[rank:, :].T.copy(order="C")
    # SVD vector signs are arbitrary.  Fix each sign using its largest-magnitude
    # coordinate (with np.argmax's deterministic lowest-index tie break).
    for column_index in range(null.shape[1]):
        column = null[:, column_index]
        anchor = int(np.argmax(np.abs(column)))
        if column[anchor] < 0.0:
            null[:, column_index] *= -1.0
    residual = float(np.max(np.abs(equality @ null))) if null.shape[1] and equality.size else 0.0
    if residual > PRIMAL_CERTIFICATE_TOLERANCE:
        raise GCRBSSolverError("the computed unrelated-task null space failed certification")
    return null, rank, threshold


def _canonical_simplex_candidate(
    values: Any,
    *,
    target_count: int,
) -> np.ndarray | None:
    candidate = np.asarray(values, dtype=np.float64)
    if candidate.shape != (target_count,) or not np.isfinite(candidate).all():
        return None
    # SLSQP may return harmless negative roundoff at an active zero bound.  Clipping
    # and renormalizing constructs a new, actually nonnegative simplex point; the
    # bound is always recomputed from that point rather than from optimizer output.
    if float(np.min(candidate)) < -PRIMAL_CERTIFICATE_TOLERANCE:
        return None
    candidate = np.maximum(candidate, 0.0)
    total = float(np.sum(candidate))
    if not math.isfinite(total) or total <= 0.0:
        return None
    candidate /= total
    # Assign the final floating sum residual to the largest coordinate using a fixed
    # lowest-index tie break.  This keeps the stored simplex residual at roundoff scale.
    anchor = int(np.argmax(candidate))
    candidate[anchor] += 1.0 - float(np.sum(candidate))
    if float(np.min(candidate)) < 0.0:
        return None
    if abs(float(np.sum(candidate)) - 1.0) > 8.0 * np.finfo(np.float64).eps:
        return None
    return candidate.copy(order="C")


def _l2_relaxation_dual_upper_bound(
    *,
    reduced_target: np.ndarray,
    offsets: np.ndarray,
    l2_cap: float,
) -> dict[str, Any]:
    """Certify a cheap upper bound after relaxing all but null/L2 constraints.

    For every simplex vector ``lambda``,

    ``max_d min_i(a_i d - b_i) <= rho ||A.T lambda|| - b.T lambda``

    when ``d`` is restricted to the unrelated-task null and ``||d|| <= rho``.
    Removing protected and Fisher constraints can only increase the optimum.  The
    returned value is therefore valid even when SLSQP fails to improve the uniform
    simplex point.  No tightness claim is made.
    """

    target_count = reduced_target.shape[0]
    uniform = np.full(target_count, 1.0 / target_count, dtype=np.float64)
    uniform = _canonical_simplex_candidate(uniform, target_count=target_count)
    if uniform is None:  # pragma: no cover - construction is exact for positive m
        raise GCRBSSolverError("could not construct the uniform dual simplex point")

    def raw_bound(simplex: np.ndarray) -> float:
        aggregate = reduced_target.T @ simplex
        return l2_cap * float(np.linalg.norm(aggregate)) - float(offsets @ simplex)

    def certified_record(*, name: str, simplex: np.ndarray) -> dict[str, Any]:
        aggregate = reduced_target.T @ simplex
        support_term = l2_cap * float(np.linalg.norm(aggregate))
        offset_term = float(offsets @ simplex)
        raw = support_term - offset_term
        # A small outward roundoff allowance prevents a floating under-estimate from
        # being used as an exclusion certificate.  It is reported separately.
        arithmetic_scale = 1.0 + abs(support_term) + abs(offset_term)
        padding = (
            128.0
            * np.finfo(np.float64).eps
            * max(1, target_count, reduced_target.shape[1])
            * arithmetic_scale
        )
        return {
            "source": name,
            "lambda": simplex.tolist(),
            "lambda_sha256": _array_sha256(simplex),
            "minimum_lambda": float(np.min(simplex)),
            "simplex_sum": float(np.sum(simplex)),
            "simplex_sum_residual": float(np.sum(simplex)) - 1.0,
            "aggregate_target_sha256": _array_sha256(aggregate),
            "support_term": support_term,
            "offset_term": offset_term,
            "raw_upper_bound": raw,
            "roundoff_padding": padding,
            "certified_upper_bound": raw + padding,
            "passes_simplex_certificate": bool(
                float(np.min(simplex)) >= 0.0
                and abs(float(np.sum(simplex)) - 1.0) <= 8.0 * np.finfo(np.float64).eps
                and math.isfinite(raw)
            ),
        }

    candidates = [certified_record(name="uniform_fallback", simplex=uniform)]
    vertex_values = np.array(
        [
            raw_bound(np.eye(1, target_count, index, dtype=np.float64)[0])
            for index in range(target_count)
        ]
    )
    best_vertex_index = int(np.argmin(vertex_values))
    best_vertex = np.zeros(target_count, dtype=np.float64)
    best_vertex[best_vertex_index] = 1.0
    candidates.append(
        certified_record(name=f"best_vertex_{best_vertex_index}", simplex=best_vertex)
    )

    def objective(simplex: np.ndarray) -> float:
        return raw_bound(simplex)

    def objective_jac(simplex: np.ndarray) -> np.ndarray:
        aggregate = reduced_target.T @ simplex
        norm = float(np.linalg.norm(aggregate))
        support_gradient = (
            np.zeros(target_count, dtype=np.float64)
            if norm == 0.0
            else l2_cap * (reduced_target @ aggregate) / norm
        )
        return support_gradient - offsets

    constraint = {
        "type": "eq",
        "fun": lambda simplex: float(np.sum(simplex)) - 1.0,
        "jac": lambda _simplex: np.ones(target_count, dtype=np.float64),
    }
    optimizer_attempts: list[dict[str, Any]] = []
    starts = (("uniform", uniform), ("best_vertex", best_vertex))
    for start_name, start in starts:
        result = _DUAL_MINIMIZE(
            objective,
            start,
            jac=objective_jac,
            bounds=[(0.0, 1.0)] * target_count,
            constraints=[constraint],
            method=SOLVER_METHOD,
            options={
                "maxiter": SOLVER_MAX_ITERATIONS,
                "ftol": SOLVER_FUNCTION_TOLERANCE,
                "disp": False,
            },
        )
        success = bool(getattr(result, "success", False))
        canonical = _canonical_simplex_candidate(
            getattr(result, "x", np.empty(0)), target_count=target_count
        )
        accepted = bool(success and canonical is not None)
        optimizer_attempts.append(
            {
                "start_name": start_name,
                "optimizer_success": success,
                "optimizer_status": int(getattr(result, "status", -1)),
                "optimizer_iterations": int(getattr(result, "nit", -1)),
                "accepted_simplex": accepted,
            }
        )
        if accepted and canonical is not None:
            candidates.append(certified_record(name=f"slsqp_from_{start_name}", simplex=canonical))
    if not all(candidate["passes_simplex_certificate"] for candidate in candidates):
        raise GCRBSSolverError("an L2-relaxation dual candidate failed simplex certification")
    candidates.sort(
        key=lambda candidate: (
            float(candidate["certified_upper_bound"]),
            str(candidate["lambda_sha256"]),
            str(candidate["source"]),
        )
    )
    selected = dict(candidates[0])
    uniform_record = next(
        dict(candidate) for candidate in candidates if candidate["source"] == "uniform_fallback"
    )
    diagnostics = {
        "bound_type": "simplex_l2_support_after_relaxing_protected_and_quadratic_constraints",
        "retained_constraints": ["unrelated_equality_null", "l2_cap"],
        "relaxed_constraints": ["protected_affine", "global_metric", "group_metric"],
        "never_claims_tightness": True,
        "target_constraint_count": target_count,
        "uniform_fallback": uniform_record,
        "optimizer_attempts": optimizer_attempts,
        "optimizer_success_count": sum(
            bool(attempt["accepted_simplex"]) for attempt in optimizer_attempts
        ),
        "selected": selected,
        "certified_upper_bound": float(selected["certified_upper_bound"]),
        "passes_valid_upper_bound_construction": True,
    }
    diagnostics["certificate_sha256"] = _json_sha256(diagnostics)
    return diagnostics


def _quadratic_value(matrix: np.ndarray, vector: np.ndarray) -> float:
    return 0.5 * float(vector @ matrix @ vector)


def _factor_value(factor: np.ndarray, vector: np.ndarray) -> float:
    product = factor @ vector
    return 0.5 * float(product @ product)


def _within_tolerance(residual: float, *, scale: float = 1.0) -> bool:
    return residual >= -PRIMAL_CERTIFICATE_TOLERANCE * max(1.0, abs(scale))


def _budget_constraints(
    *,
    reduced_dimension: int,
    l2_cap: float,
    metric: np.ndarray | None,
    metric_budget: float | None,
    group_factors: tuple[np.ndarray, ...],
    group_budgets: tuple[float, ...],
    include_gamma: bool,
) -> list[dict[str, Any]]:
    tail = 1 if include_gamma else 0

    def direction(values: np.ndarray) -> np.ndarray:
        return values[:reduced_dimension]

    constraints: list[dict[str, Any]] = []

    def l2_fun(values: np.ndarray) -> float:
        reduced = direction(values)
        return l2_cap * l2_cap - float(reduced @ reduced)

    def l2_jac(values: np.ndarray) -> np.ndarray:
        reduced = direction(values)
        return np.concatenate((-2.0 * reduced, np.zeros(tail, dtype=np.float64)))

    constraints.append({"type": "ineq", "fun": l2_fun, "jac": l2_jac})

    if metric is not None and metric_budget is not None:

        def metric_fun(values: np.ndarray) -> float:
            reduced = direction(values)
            return metric_budget - _quadratic_value(metric, reduced)

        def metric_jac(values: np.ndarray) -> np.ndarray:
            reduced = direction(values)
            return np.concatenate((-(metric @ reduced), np.zeros(tail, dtype=np.float64)))

        constraints.append({"type": "ineq", "fun": metric_fun, "jac": metric_jac})

    for factor, budget in zip(group_factors, group_budgets, strict=True):

        def group_fun(
            values: np.ndarray,
            current_factor: np.ndarray = factor,
            current_budget: float = budget,
        ) -> float:
            reduced = direction(values)
            return current_budget - _factor_value(current_factor, reduced)

        def group_jac(values: np.ndarray, current_factor: np.ndarray = factor) -> np.ndarray:
            reduced = direction(values)
            gradient = -(current_factor.T @ (current_factor @ reduced))
            return np.concatenate((gradient, np.zeros(tail, dtype=np.float64)))

        constraints.append({"type": "ineq", "fun": group_fun, "jac": group_jac})
    return constraints


def _radial_limit(
    direction: np.ndarray,
    *,
    l2_cap: float,
    metric: np.ndarray | None,
    metric_budget: float | None,
    group_factors: tuple[np.ndarray, ...],
    group_budgets: tuple[float, ...],
) -> float:
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        return 0.0
    unit = direction / norm
    limits = [l2_cap]
    if metric is not None and metric_budget is not None:
        energy = _quadratic_value(metric, unit)
        if energy > 0.0:
            limits.append(math.sqrt(metric_budget / energy))
    for factor, budget in zip(group_factors, group_budgets, strict=True):
        energy = _factor_value(factor, unit)
        if energy > 0.0:
            limits.append(math.sqrt(budget / energy))
    return min(limits)


def _is_budget_feasible(
    reduced: np.ndarray,
    *,
    l2_cap: float,
    metric: np.ndarray | None,
    metric_budget: float | None,
    group_factors: tuple[np.ndarray, ...],
    group_budgets: tuple[float, ...],
) -> bool:
    if not _within_tolerance(l2_cap - float(np.linalg.norm(reduced)), scale=l2_cap):
        return False
    if (
        metric is not None
        and metric_budget is not None
        and not _within_tolerance(
            metric_budget - _quadratic_value(metric, reduced), scale=metric_budget
        )
    ):
        return False
    return all(
        _within_tolerance(budget - _factor_value(factor, reduced), scale=budget)
        for factor, budget in zip(group_factors, group_budgets, strict=True)
    )


def _restore_protected_feasibility(
    *,
    protected: np.ndarray,
    lower_bounds: np.ndarray,
    reduced_dimension: int,
    l2_cap: float,
    metric: np.ndarray | None,
    metric_budget: float | None,
    group_factors: tuple[np.ndarray, ...],
    group_budgets: tuple[float, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    if protected.shape[0] == 0:
        return np.zeros(reduced_dimension, dtype=np.float64), {
            "needed": False,
            "optimizer_success": True,
            "slack": 0.0,
        }
    initial_slack = max(0.0, float(np.max(lower_bounds))) + 1.0
    initial = np.concatenate((np.zeros(reduced_dimension), np.array([initial_slack])))

    def objective(values: np.ndarray) -> float:
        return float(values[-1]) + 1e-14 * float(values[:-1] @ values[:-1])

    def objective_jac(values: np.ndarray) -> np.ndarray:
        return np.concatenate((2e-14 * values[:-1], np.ones(1, dtype=np.float64)))

    def protected_fun(values: np.ndarray) -> np.ndarray:
        return protected @ values[:-1] - lower_bounds + values[-1]

    def protected_jac(_values: np.ndarray) -> np.ndarray:
        return np.column_stack((protected, np.ones(protected.shape[0], dtype=np.float64)))

    constraints: list[dict[str, Any]] = [
        {"type": "ineq", "fun": protected_fun, "jac": protected_jac},
        {
            "type": "ineq",
            "fun": lambda values: float(values[-1]),
            "jac": lambda _values: np.concatenate(
                (np.zeros(reduced_dimension, dtype=np.float64), np.ones(1, dtype=np.float64))
            ),
        },
    ]
    constraints.extend(
        _budget_constraints(
            reduced_dimension=reduced_dimension,
            l2_cap=l2_cap,
            metric=metric,
            metric_budget=metric_budget,
            group_factors=group_factors,
            group_budgets=group_budgets,
            include_gamma=True,
        )
    )
    result = _MINIMIZE(
        objective,
        initial,
        jac=objective_jac,
        constraints=constraints,
        method=SOLVER_METHOD,
        options={
            "maxiter": SOLVER_MAX_ITERATIONS,
            "ftol": SOLVER_FUNCTION_TOLERANCE,
            "disp": False,
        },
    )
    success = bool(getattr(result, "success", False))
    values = np.asarray(getattr(result, "x", np.empty(0)), dtype=np.float64)
    if not success or values.shape != (reduced_dimension + 1,) or not np.isfinite(values).all():
        raise GCRBSInfeasibleError("protected-constraint feasibility restoration did not converge")
    reduced = values[:-1]
    slack = float(values[-1])
    protected_residual = protected @ reduced - lower_bounds
    feasible = (
        slack <= PRIMAL_CERTIFICATE_TOLERANCE
        and float(np.min(protected_residual)) >= -PRIMAL_CERTIFICATE_TOLERANCE
        and _is_budget_feasible(
            reduced,
            l2_cap=l2_cap,
            metric=metric,
            metric_budget=metric_budget,
            group_factors=group_factors,
            group_budgets=group_budgets,
        )
    )
    if not feasible:
        raise GCRBSInfeasibleError(
            "protected constraints are infeasible under the equality and quadratic budgets"
        )
    return reduced.copy(), {
        "needed": True,
        "optimizer_success": True,
        "optimizer_status": int(getattr(result, "status", -1)),
        "optimizer_iterations": int(getattr(result, "nit", -1)),
        "slack": slack,
        "minimum_protected_residual": float(np.min(protected_residual)),
    }


def _deterministic_starts(
    *,
    target: np.ndarray,
    protected: np.ndarray,
    lower_bounds: np.ndarray,
    feasible: np.ndarray,
    l2_cap: float,
    metric: np.ndarray | None,
    metric_budget: float | None,
    group_factors: tuple[np.ndarray, ...],
    group_budgets: tuple[float, ...],
) -> tuple[tuple[str, np.ndarray], ...]:
    dimension = target.shape[1]
    proposals: list[tuple[str, np.ndarray]] = [
        ("zero", np.zeros(dimension, dtype=np.float64)),
        ("protected_feasible", feasible.copy()),
    ]
    vectors: list[tuple[str, np.ndarray]] = [("target_mean", np.mean(target, axis=0))]
    if protected.shape[0]:
        vectors.append(("protected_mean", np.mean(protected, axis=0)))
        least_squares, *_ = np.linalg.lstsq(protected, lower_bounds, rcond=None)
        proposals.append(("protected_least_squares", least_squares))
    if target.shape[0]:
        vectors.append(("target_first", target[0]))
        if target.shape[0] > 1:
            vectors.append(("target_last", target[-1]))
    for name, vector in vectors:
        limit = _radial_limit(
            vector,
            l2_cap=l2_cap,
            metric=metric,
            metric_budget=metric_budget,
            group_factors=group_factors,
            group_budgets=group_budgets,
        )
        norm = float(np.linalg.norm(vector))
        if norm > 0.0 and limit > 0.0:
            scaled = (0.75 * limit / norm) * vector
            proposals.append((f"{name}_positive", scaled))
            proposals.append((f"{name}_negative", -scaled))
    if dimension:
        for axis in range(min(2, dimension)):
            vector = np.zeros(dimension, dtype=np.float64)
            vector[axis] = 1.0
            limit = _radial_limit(
                vector,
                l2_cap=l2_cap,
                metric=metric,
                metric_budget=metric_budget,
                group_factors=group_factors,
                group_budgets=group_budgets,
            )
            proposals.append((f"axis_{axis}_positive", 0.5 * limit * vector))
    unique: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    for name, proposal in proposals:
        candidate = np.asarray(proposal, dtype=np.float64)
        if candidate.shape != (dimension,) or not np.isfinite(candidate).all():
            continue
        digest = _array_sha256(candidate)
        if digest in seen:
            continue
        seen.add(digest)
        unique.append((name, candidate.copy()))
        if len(unique) == MAX_DETERMINISTIC_STARTS:
            break
    return tuple(unique)


def _main_constraints(
    *,
    target: np.ndarray,
    offsets: np.ndarray,
    protected: np.ndarray,
    lower_bounds: np.ndarray,
    reduced_dimension: int,
    l2_cap: float,
    metric: np.ndarray | None,
    metric_budget: float | None,
    group_factors: tuple[np.ndarray, ...],
    group_budgets: tuple[float, ...],
) -> list[dict[str, Any]]:
    def target_fun(values: np.ndarray) -> np.ndarray:
        return target @ values[:-1] - offsets - values[-1]

    def target_jac(_values: np.ndarray) -> np.ndarray:
        return np.column_stack((target, -np.ones(target.shape[0], dtype=np.float64)))

    constraints: list[dict[str, Any]] = [{"type": "ineq", "fun": target_fun, "jac": target_jac}]
    if protected.shape[0]:

        def protected_fun(values: np.ndarray) -> np.ndarray:
            return protected @ values[:-1] - lower_bounds

        def protected_jac(_values: np.ndarray) -> np.ndarray:
            return np.column_stack((protected, np.zeros(protected.shape[0], dtype=np.float64)))

        constraints.append({"type": "ineq", "fun": protected_fun, "jac": protected_jac})
    constraints.extend(
        _budget_constraints(
            reduced_dimension=reduced_dimension,
            l2_cap=l2_cap,
            metric=metric,
            metric_budget=metric_budget,
            group_factors=group_factors,
            group_budgets=group_budgets,
            include_gamma=True,
        )
    )
    return constraints


def _certify_reduced_candidate(
    *,
    values: np.ndarray,
    target: np.ndarray,
    offsets: np.ndarray,
    protected: np.ndarray,
    lower_bounds: np.ndarray,
    l2_cap: float,
    metric: np.ndarray | None,
    metric_budget: float | None,
    group_factors: tuple[np.ndarray, ...],
    group_budgets: tuple[float, ...],
) -> tuple[bool, dict[str, Any]]:
    reduced = values[:-1]
    reported_gamma = float(values[-1])
    margins = target @ reduced - offsets
    target_residuals = margins - reported_gamma
    protected_residuals = protected @ reduced - lower_bounds
    norm = float(np.linalg.norm(reduced))
    metric_value = _quadratic_value(metric, reduced) if metric is not None else None
    group_values = tuple(_factor_value(factor, reduced) for factor in group_factors)
    target_ok = float(np.min(target_residuals)) >= -PRIMAL_CERTIFICATE_TOLERANCE * max(
        1.0, abs(reported_gamma), float(np.max(np.abs(margins)))
    )
    protected_ok = protected.shape[0] == 0 or float(
        np.min(protected_residuals)
    ) >= -PRIMAL_CERTIFICATE_TOLERANCE * max(1.0, float(np.max(np.abs(lower_bounds))))
    l2_ok = _within_tolerance(l2_cap - norm, scale=l2_cap)
    metric_ok = (
        metric_value is None
        or metric_budget is None
        or _within_tolerance(metric_budget - metric_value, scale=metric_budget)
    )
    groups_ok = all(
        _within_tolerance(budget - value, scale=budget)
        for value, budget in zip(group_values, group_budgets, strict=True)
    )
    finite = all(
        math.isfinite(value)
        for value in (
            reported_gamma,
            norm,
            *margins.tolist(),
            *target_residuals.tolist(),
            *protected_residuals.tolist(),
            *group_values,
        )
    ) and (metric_value is None or math.isfinite(metric_value))
    diagnostics = {
        "reported_gamma": reported_gamma,
        "recomputed_worst_margin": float(np.min(margins)),
        "minimum_target_residual": float(np.min(target_residuals)),
        "minimum_protected_residual": (
            float(np.min(protected_residuals)) if protected_residuals.size else None
        ),
        "l2_norm": norm,
        "l2_cap_residual": l2_cap - norm,
        "metric_value": metric_value,
        "metric_budget_residual": (
            metric_budget - metric_value
            if metric_value is not None and metric_budget is not None
            else None
        ),
        "group_values": list(group_values),
        "group_budget_residuals": [
            budget - value for value, budget in zip(group_values, group_budgets, strict=True)
        ],
        "passes": bool(finite and target_ok and protected_ok and l2_ok and metric_ok and groups_ok),
    }
    return diagnostics["passes"], diagnostics


def _canonical_minimum_norm(
    *,
    start: np.ndarray,
    gamma_floor: float,
    target: np.ndarray,
    offsets: np.ndarray,
    protected: np.ndarray,
    lower_bounds: np.ndarray,
    l2_cap: float,
    metric: np.ndarray | None,
    metric_budget: float | None,
    group_factors: tuple[np.ndarray, ...],
    group_budgets: tuple[float, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    dimension = target.shape[1]

    def objective(reduced: np.ndarray) -> float:
        return 0.5 * float(reduced @ reduced)

    def objective_jac(reduced: np.ndarray) -> np.ndarray:
        return reduced.copy()

    constraints: list[dict[str, Any]] = [
        {
            "type": "ineq",
            "fun": lambda reduced: target @ reduced - offsets - gamma_floor,
            "jac": lambda _reduced: target,
        }
    ]
    if protected.shape[0]:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda reduced: protected @ reduced - lower_bounds,
                "jac": lambda _reduced: protected,
            }
        )
    constraints.extend(
        _budget_constraints(
            reduced_dimension=dimension,
            l2_cap=l2_cap,
            metric=metric,
            metric_budget=metric_budget,
            group_factors=group_factors,
            group_budgets=group_budgets,
            include_gamma=False,
        )
    )
    result = _MINIMIZE(
        objective,
        start,
        jac=objective_jac,
        constraints=constraints,
        method=SOLVER_METHOD,
        options={
            "maxiter": SOLVER_MAX_ITERATIONS,
            "ftol": SOLVER_FUNCTION_TOLERANCE,
            "disp": False,
        },
    )
    success = bool(getattr(result, "success", False))
    reduced = np.asarray(getattr(result, "x", np.empty(0)), dtype=np.float64)
    if not success or reduced.shape != (dimension,) or not np.isfinite(reduced).all():
        raise GCRBSSolverError("canonical minimum-norm tie-break did not converge")
    margins = target @ reduced - offsets
    protected_residuals = protected @ reduced - lower_bounds
    feasible = (
        float(np.min(margins - gamma_floor))
        >= -PRIMAL_CERTIFICATE_TOLERANCE * max(1.0, abs(gamma_floor))
        and (
            protected.shape[0] == 0
            or float(np.min(protected_residuals))
            >= -PRIMAL_CERTIFICATE_TOLERANCE * max(1.0, float(np.max(np.abs(lower_bounds))))
        )
        and _is_budget_feasible(
            reduced,
            l2_cap=l2_cap,
            metric=metric,
            metric_budget=metric_budget,
            group_factors=group_factors,
            group_budgets=group_budgets,
        )
    )
    if not feasible:
        raise GCRBSSolverError("canonical minimum-norm output failed primal certification")
    return reduced.copy(), {
        "optimizer_success": True,
        "optimizer_status": int(getattr(result, "status", -1)),
        "optimizer_iterations": int(getattr(result, "nit", -1)),
        "gamma_floor": gamma_floor,
        "minimum_target_margin": float(np.min(margins)),
        "l2_norm": float(np.linalg.norm(reduced)),
    }


def solve_global_counterfactual_robust_boundary(
    *,
    target_matrix: Any,
    target_offsets: Any,
    l2_cap: Any,
    unrelated_equality_basis: Any | None = None,
    protected_matrix: Any | None = None,
    protected_lower_bounds: Any | None = None,
    metric_matrix: Any | None = None,
    metric_budget: Any | None = None,
    group_metric_factors: Any = (),
    group_metric_budgets: Any = (),
) -> GCRBSSolution:
    """Solve and strictly certify a generic GCRBS max-min problem.

    ``target_matrix`` is ``A`` and ``target_offsets`` is ``b`` in
    ``A @ d - b >= gamma``.  The caller must include every desired sign/order as
    a separate row.  Likewise, the caller must explicitly include both signs of
    protected constraints in ``C @ d >= q`` when both are required.

    The fixed solver procedure is: exact float64 validation, null-space reduction,
    protected-feasibility restoration, zero-first deterministic multistart SLSQP,
    then a minimum-L2 tie-break within ``CANONICAL_GAMMA_RELAXATION`` of the best
    certified margin.  No model, prompt, or evaluation outcome is accessed here.
    """

    target = _matrix(target_matrix, field="target_matrix")
    target_count, dimension = target.shape
    offsets = _vector(target_offsets, field="target_offsets", length=target_count)
    cap = _finite_scalar(l2_cap, field="l2_cap", nonnegative=True)

    equality = (
        np.zeros((0, dimension), dtype=np.float64)
        if unrelated_equality_basis is None
        else _matrix(
            unrelated_equality_basis,
            field="unrelated_equality_basis",
            width=dimension,
            allow_empty_rows=True,
        )
    )
    if (protected_matrix is None) != (protected_lower_bounds is None):
        raise ValueError("protected_matrix and protected_lower_bounds must be supplied together")
    if protected_matrix is None:
        protected = np.zeros((0, dimension), dtype=np.float64)
        lower_bounds = np.zeros(0, dtype=np.float64)
    else:
        protected = _matrix(
            protected_matrix,
            field="protected_matrix",
            width=dimension,
            allow_empty_rows=True,
        )
        lower_bounds = _vector(
            protected_lower_bounds,
            field="protected_lower_bounds",
            length=protected.shape[0],
        )

    if (metric_matrix is None) != (metric_budget is None):
        raise ValueError("metric_matrix and metric_budget must be supplied together")
    metric = None
    budget = None
    metric_minimum_eigenvalue = None
    if metric_matrix is not None:
        metric_raw = _matrix(metric_matrix, field="metric_matrix", width=dimension)
        if metric_raw.shape[0] != dimension:
            raise ValueError("metric_matrix must be square")
        metric, metric_minimum_eigenvalue = _validate_symmetric_psd(
            metric_raw, field="metric_matrix"
        )
        budget = _finite_scalar(metric_budget, field="metric_budget", nonnegative=True)

    try:
        raw_group_factors = tuple(group_metric_factors)
        raw_group_budgets = tuple(group_metric_budgets)
    except TypeError as error:
        raise TypeError("group metric factors and budgets must be finite sequences") from error
    if len(raw_group_factors) != len(raw_group_budgets):
        raise ValueError("group_metric_factors and group_metric_budgets must have equal length")
    factors = tuple(
        _matrix(value, field=f"group_metric_factors[{index}]", width=dimension)
        for index, value in enumerate(raw_group_factors)
    )
    budgets = tuple(
        _finite_scalar(value, field=f"group_metric_budgets[{index}]", nonnegative=True)
        for index, value in enumerate(raw_group_budgets)
    )

    null, equality_rank, equality_svd_threshold = _null_space(equality)
    reduced_dimension = null.shape[1]
    reduced_target = target @ null
    reduced_protected = protected @ null
    reduced_metric = null.T @ metric @ null if metric is not None else None
    reduced_factors = tuple(factor @ null for factor in factors)
    dual_upper_bound = _l2_relaxation_dual_upper_bound(
        reduced_target=reduced_target,
        offsets=offsets,
        l2_cap=cap,
    )

    input_record = {
        "schema_version": SCHEMA_VERSION,
        "target_matrix_sha256": _array_sha256(target),
        "target_offsets_sha256": _array_sha256(offsets),
        "unrelated_equality_basis_sha256": _array_sha256(equality),
        "protected_matrix_sha256": _array_sha256(protected),
        "protected_lower_bounds_sha256": _array_sha256(lower_bounds),
        "metric_matrix_sha256": _array_sha256(metric) if metric is not None else None,
        "metric_budget": budget,
        "group_metric_factor_sha256s": [_array_sha256(factor) for factor in factors],
        "group_metric_budgets": list(budgets),
        "l2_cap": cap,
        "dimension": dimension,
        "target_constraint_count": target_count,
        "unrelated_equality_count": equality.shape[0],
        "protected_constraint_count": protected.shape[0],
    }
    input_sha256 = _json_sha256(input_record)

    if reduced_dimension == 0:
        direction = np.zeros(dimension, dtype=np.float64)
        protected_residuals = protected @ direction - lower_bounds
        if protected_residuals.size and float(np.min(protected_residuals)) < (
            -PRIMAL_CERTIFICATE_TOLERANCE
        ):
            raise GCRBSInfeasibleError(
                "the unrelated equality span fixes d=0, which violates protected constraints"
            )
        best_gamma = float(np.min(target @ direction - offsets))
        feasibility_diagnostics = {
            "needed": False,
            "optimizer_success": True,
            "slack": 0.0,
        }
        main_attempts: list[dict[str, Any]] = []
        canonical_diagnostics = {
            "optimizer_success": True,
            "optimizer_status": 0,
            "optimizer_iterations": 0,
            "gamma_floor": best_gamma,
            "minimum_target_margin": best_gamma,
            "l2_norm": 0.0,
        }
        best_observed_gamma = best_gamma
    else:
        feasible, feasibility_diagnostics = _restore_protected_feasibility(
            protected=reduced_protected,
            lower_bounds=lower_bounds,
            reduced_dimension=reduced_dimension,
            l2_cap=cap,
            metric=reduced_metric,
            metric_budget=budget,
            group_factors=reduced_factors,
            group_budgets=budgets,
        )
        starts = _deterministic_starts(
            target=reduced_target,
            protected=reduced_protected,
            lower_bounds=lower_bounds,
            feasible=feasible,
            l2_cap=cap,
            metric=reduced_metric,
            metric_budget=budget,
            group_factors=reduced_factors,
            group_budgets=budgets,
        )
        constraints = _main_constraints(
            target=reduced_target,
            offsets=offsets,
            protected=reduced_protected,
            lower_bounds=lower_bounds,
            reduced_dimension=reduced_dimension,
            l2_cap=cap,
            metric=reduced_metric,
            metric_budget=budget,
            group_factors=reduced_factors,
            group_budgets=budgets,
        )

        def objective(values: np.ndarray) -> float:
            return -float(values[-1])

        def objective_jac(_values: np.ndarray) -> np.ndarray:
            return np.concatenate(
                (np.zeros(reduced_dimension, dtype=np.float64), -np.ones(1, dtype=np.float64))
            )

        certified: list[tuple[float, float, str, np.ndarray]] = []
        main_attempts = []
        for start_name, start_direction in starts:
            start_gamma = float(np.min(reduced_target @ start_direction - offsets)) - 1e-8
            initial = np.concatenate((start_direction, np.array([start_gamma])))
            result = _MINIMIZE(
                objective,
                initial,
                jac=objective_jac,
                constraints=constraints,
                method=SOLVER_METHOD,
                options={
                    "maxiter": SOLVER_MAX_ITERATIONS,
                    "ftol": SOLVER_FUNCTION_TOLERANCE,
                    "disp": False,
                },
            )
            success = bool(getattr(result, "success", False))
            values = np.asarray(getattr(result, "x", np.empty(0)), dtype=np.float64)
            certified_pass = False
            certificate: dict[str, Any] = {}
            if success and values.shape == (reduced_dimension + 1,) and np.isfinite(values).all():
                certified_pass, certificate = _certify_reduced_candidate(
                    values=values,
                    target=reduced_target,
                    offsets=offsets,
                    protected=reduced_protected,
                    lower_bounds=lower_bounds,
                    l2_cap=cap,
                    metric=reduced_metric,
                    metric_budget=budget,
                    group_factors=reduced_factors,
                    group_budgets=budgets,
                )
            attempt = {
                "start_name": start_name,
                "start_sha256": _array_sha256(start_direction),
                "optimizer_success": success,
                "optimizer_status": int(getattr(result, "status", -1)),
                "optimizer_iterations": int(getattr(result, "nit", -1)),
                "certificate_passes": certified_pass,
                "recomputed_worst_margin": certificate.get("recomputed_worst_margin"),
                "minimum_target_residual": certificate.get("minimum_target_residual"),
            }
            main_attempts.append(attempt)
            if certified_pass:
                recomputed_gamma = float(certificate["recomputed_worst_margin"])
                norm = float(np.linalg.norm(values[:-1]))
                certified.append((recomputed_gamma, norm, start_name, values[:-1].copy()))
        if not certified:
            raise GCRBSSolverError(
                "no deterministic max-min attempt returned a strictly certified solution"
            )
        certified.sort(key=lambda item: (-item[0], item[1], item[2], _array_sha256(item[3])))
        best_observed_gamma, _, best_start_name, best_reduced = certified[0]
        relaxation = CANONICAL_GAMMA_RELAXATION * max(1.0, abs(best_observed_gamma))
        gamma_floor = best_observed_gamma - relaxation
        canonical_reduced, canonical_diagnostics = _canonical_minimum_norm(
            start=best_reduced,
            gamma_floor=gamma_floor,
            target=reduced_target,
            offsets=offsets,
            protected=reduced_protected,
            lower_bounds=lower_bounds,
            l2_cap=cap,
            metric=reduced_metric,
            metric_budget=budget,
            group_factors=reduced_factors,
            group_budgets=budgets,
        )
        canonical_diagnostics["selected_main_start_name"] = best_start_name
        canonical_diagnostics["near_optimality_relaxation"] = relaxation
        direction = null @ canonical_reduced
        # Remove only exact-scale roundoff dust.  This is deterministic and is
        # followed by a full certificate against the original, unreduced inputs.
        cleanup_threshold = 64.0 * np.finfo(np.float64).eps * max(1.0, cap)
        direction[np.abs(direction) <= cleanup_threshold] = 0.0
        best_gamma = float(np.min(target @ direction - offsets))

    direction = _canonical_array(direction)
    target_values = target @ direction - offsets
    gamma = float(np.min(target_values))
    target_residuals = target_values - gamma
    equality_residuals = equality @ direction
    protected_residuals = protected @ direction - lower_bounds
    l2_norm = float(np.linalg.norm(direction))
    l2_residual = cap - l2_norm
    metric_value = _quadratic_value(metric, direction) if metric is not None else None
    metric_residual = (
        budget - metric_value if metric_value is not None and budget is not None else None
    )
    group_values = [_factor_value(factor, direction) for factor in factors]
    group_residuals = [
        group_budget - value for value, group_budget in zip(group_values, budgets, strict=True)
    ]
    equality_scale = max(1.0, float(np.linalg.norm(equality, ord=2)) * max(1.0, l2_norm))
    maximum_equality_residual = (
        float(np.max(np.abs(equality_residuals))) if equality_residuals.size else 0.0
    )
    checks = {
        "finite": bool(
            np.isfinite(direction).all()
            and np.isfinite(target_values).all()
            and np.isfinite(equality_residuals).all()
            and np.isfinite(protected_residuals).all()
            and math.isfinite(gamma)
        ),
        "target": bool(float(np.min(target_residuals)) >= -PRIMAL_CERTIFICATE_TOLERANCE),
        "unrelated_equality": bool(
            maximum_equality_residual <= PRIMAL_CERTIFICATE_TOLERANCE * equality_scale
        ),
        "protected": bool(
            not protected_residuals.size
            or float(np.min(protected_residuals))
            >= -PRIMAL_CERTIFICATE_TOLERANCE * max(1.0, float(np.max(np.abs(lower_bounds))))
        ),
        "l2_cap": bool(_within_tolerance(l2_residual, scale=cap)),
        "metric_budget": bool(
            metric_residual is None or _within_tolerance(metric_residual, scale=budget or 0.0)
        ),
        "group_budgets": bool(
            all(
                _within_tolerance(residual, scale=group_budget)
                for residual, group_budget in zip(group_residuals, budgets, strict=True)
            )
        ),
        "near_optimal_floor": bool(
            gamma
            >= float(canonical_diagnostics["gamma_floor"])
            - PRIMAL_CERTIFICATE_TOLERANCE
            * max(1.0, abs(float(canonical_diagnostics["gamma_floor"])))
        ),
        "primal_below_l2_relaxation_dual_upper_bound": bool(
            gamma
            <= float(dual_upper_bound["certified_upper_bound"])
            + PRIMAL_CERTIFICATE_TOLERANCE
            * max(1.0, abs(float(dual_upper_bound["certified_upper_bound"])))
        ),
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise GCRBSSolverError(f"final original-coordinate certificate failed: {failed}")

    worst_rows = np.flatnonzero(
        target_values <= gamma + PRIMAL_CERTIFICATE_TOLERANCE * max(1.0, abs(gamma))
    )
    orientation_anchor = int(worst_rows[0])
    direction.setflags(write=False)
    diagnostics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "solver_method": SOLVER_METHOD,
        "scipy_version": scipy_version,
        "solver_settings": {
            "max_iterations": SOLVER_MAX_ITERATIONS,
            "function_tolerance": SOLVER_FUNCTION_TOLERANCE,
            "primal_certificate_tolerance": PRIMAL_CERTIFICATE_TOLERANCE,
            "maximum_deterministic_starts": MAX_DETERMINISTIC_STARTS,
            "start_strategy": (
                "zero_first_then_protected_feasible_then_fixed_constraint_and_axis_seeds"
            ),
            "feasibility_objective": "minimum_nonnegative_shared_protected_slack",
            "max_min_objective": "maximize_gamma_equivalently_minimize_negative_gamma",
            "canonical_gamma_relative_relaxation": CANONICAL_GAMMA_RELAXATION,
            "canonical_tie_break": "minimum_l2_at_fixed_near_optimal_gamma_floor",
        },
        "dtype": "float64",
        "input_sha256": input_sha256,
        "input_record": input_record,
        "direction_sha256": _array_sha256(direction),
        "dimension": dimension,
        "reduced_dimension": reduced_dimension,
        "unrelated_equality_rank": equality_rank,
        "unrelated_equality_svd_threshold": equality_svd_threshold,
        "null_basis_sha256": _array_sha256(null),
        "metric_minimum_eigenvalue": metric_minimum_eigenvalue,
        "orientation_rule": "target_rows_define_positive_orientation_no_posthoc_sign_flip",
        "orientation_anchor_target_row": orientation_anchor,
        "orientation_anchor_target_value": float(target_values[orientation_anchor]),
        "canonical_tie_break": "minimum_l2_at_fixed_near_optimal_gamma_floor",
        "feasibility_restoration": feasibility_diagnostics,
        "main_attempts": main_attempts,
        "certified_main_attempt_count": sum(
            bool(attempt["certificate_passes"]) for attempt in main_attempts
        ),
        "best_observed_max_min_gamma": best_observed_gamma,
        "l2_relaxation_dual_upper_bound_certificate": dual_upper_bound,
        "certified_l2_relaxation_dual_upper_bound": float(
            dual_upper_bound["certified_upper_bound"]
        ),
        "dual_upper_bound_minus_primal_gamma": float(dual_upper_bound["certified_upper_bound"])
        - gamma,
        "canonical_tie_break_diagnostics": canonical_diagnostics,
        "recomputed_objective_gamma": gamma,
        "minimum_target_affine_value": gamma,
        "target_affine_values": target_values.tolist(),
        "target_constraint_residuals_at_reported_gamma": target_residuals.tolist(),
        "unrelated_equality_residuals": equality_residuals.tolist(),
        "maximum_abs_unrelated_equality_residual": maximum_equality_residual,
        "protected_constraint_residuals": protected_residuals.tolist(),
        "minimum_protected_constraint_residual": (
            float(np.min(protected_residuals)) if protected_residuals.size else None
        ),
        "l2_norm": l2_norm,
        "l2_cap_residual": l2_residual,
        "metric_quadratic_value": metric_value,
        "metric_budget_residual": metric_residual,
        "group_quadratic_values": group_values,
        "group_budget_residuals": group_residuals,
        "certificate_tolerance": PRIMAL_CERTIFICATE_TOLERANCE,
        "certificate_checks": checks,
        "passes_strict_primal_certificate": True,
        "positive_common_margin": bool(gamma > PRIMAL_CERTIFICATE_TOLERANCE),
    }
    hash_payload = dict(diagnostics)
    diagnostics["diagnostics_sha256"] = _json_sha256(hash_payload)
    return GCRBSSolution(direction=direction, gamma=gamma, diagnostics=diagnostics)


__all__ = [
    "CANONICAL_GAMMA_RELAXATION",
    "PRIMAL_CERTIFICATE_TOLERANCE",
    "SCHEMA_VERSION",
    "GCRBSError",
    "GCRBSInfeasibleError",
    "GCRBSSolution",
    "GCRBSSolverError",
    "solve_global_counterfactual_robust_boundary",
]
