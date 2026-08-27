"""Pure mathematics for the trust-region successor to gradient specificity v3.

This module deliberately contains no model, prompt, file-system, or split-selection
logic.  It also does not modify the frozen v3 implementation.  Its responsibilities
are limited to:

* a certified minimum-Euclidean active-set QP with a homogeneous nuisance null;
* linearization bookkeeping for finite target/protection constraints;
* a trust-step fraction that respects both a step radius and an absolute dose cap;
* target-violation merit and trial-acceptance calculations; and
* the exact both-sign, both-order terminal decision gate.

The hard equality matrix is intended to be the frozen *global unrelated-task* basis.
Matched-other constraints belong in ``inequality_rows`` and are relinearized at the
current finite intervention; they are not silently promoted into permanent nulls.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from typing import Any

from sp_lense.gradient_specificity_v3 import (
    DEFAULT_CONDITION_LIMIT,
    DEFAULT_RESIDUAL_TOLERANCE,
    DEFAULT_SVD_ATOL,
    DEFAULT_SVD_RTOL,
    canonical_sha256,
    row_normalized_svd_basis,
    tensor_float64_sha256,
)

SCHEMA_VERSION = "sp_lense.gradient_specificity_trust_region.v1"
MAX_INEQUALITY_COUNT = 8


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _positive_float(value: Any, *, field: str) -> float:
    result = _finite_float(value, field=field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def _nonnegative_float(value: Any, *, field: str) -> float:
    result = _finite_float(value, field=field)
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _float64_vector(torch: Any, value: Any, *, field: str, allow_empty: bool = False) -> Any:
    if not torch.is_tensor(value):
        raise TypeError(f"{field} must be a tensor")
    if torch.is_complex(value) or not torch.is_floating_point(value):
        raise TypeError(f"{field} must be a real floating-point tensor")
    if value.ndim != 1 or (value.numel() == 0 and not allow_empty):
        qualifier = "possibly empty" if allow_empty else "non-empty"
        raise ValueError(f"{field} must be a {qualifier} one-dimensional tensor")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"{field} must contain only finite values")
    return result


def _float64_matrix(
    torch: Any,
    value: Any,
    *,
    field: str,
    allow_empty_rows: bool = False,
) -> Any:
    if not torch.is_tensor(value):
        raise TypeError(f"{field} must be a tensor")
    if torch.is_complex(value) or not torch.is_floating_point(value):
        raise TypeError(f"{field} must be a real floating-point tensor")
    if value.ndim != 2 or int(value.shape[1]) == 0:
        raise ValueError(f"{field} must be a two-dimensional tensor with nonzero width")
    if int(value.shape[0]) == 0 and not allow_empty_rows:
        raise ValueError(f"{field} must contain at least one row")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"{field} must contain only finite values")
    return result


def _integer_tensor(torch: Any, value: Any, *, field: str, shape: tuple[int, ...]) -> Any:
    if not torch.is_tensor(value):
        raise TypeError(f"{field} must be a tensor")
    if value.dtype == torch.bool or torch.is_floating_point(value) or torch.is_complex(value):
        raise TypeError(f"{field} must be an integer tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{field} must have shape {list(shape)}")
    return value.detach().to(device="cpu", dtype=torch.int64).contiguous()


def linearized_lower_bounds(
    torch: Any,
    *,
    current_point: Any,
    current_values: Any,
    gradient_rows: Any,
    required_lower_bounds: Any,
) -> tuple[Any, dict[str, Any]]:
    """Return absolute-candidate bounds for constraints linearized at a point.

    For nonlinear functions ``f_i`` and a new absolute candidate ``x``, the local
    model is ``f_i(x) = f_i(d) + g_i @ (x - d)``.  Therefore ``f_i(x) >= m_i``
    becomes ``g_i @ x >= m_i - f_i(d) + g_i @ d``.

    This single helper is used for desired self margins and for finite-step
    matched-other baseline-greedy protection margins.
    """

    point = _float64_vector(torch, current_point, field="current_point")
    values = _float64_vector(torch, current_values, field="current_values")
    gradients = _float64_matrix(torch, gradient_rows, field="gradient_rows")
    required = _float64_vector(
        torch,
        required_lower_bounds,
        field="required_lower_bounds",
    )
    row_count, dimension = map(int, gradients.shape)
    if point.numel() != dimension:
        raise ValueError("current_point width differs from gradient_rows")
    if values.numel() != row_count or required.numel() != row_count:
        raise ValueError("current_values and required_lower_bounds must match gradient rows")
    bounds = (required - values + gradients @ point).contiguous()
    if not bool(torch.isfinite(bounds).all().item()):
        raise RuntimeError("linearized lower bounds are non-finite")
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "row_count": row_count,
        "dimension": dimension,
        "current_point_sha256": tensor_float64_sha256(point),
        "current_values_sha256": tensor_float64_sha256(values),
        "gradient_rows_sha256": tensor_float64_sha256(gradients),
        "required_lower_bounds_sha256": tensor_float64_sha256(required),
        "absolute_candidate_lower_bounds": [float(value) for value in bounds.tolist()],
        "absolute_candidate_lower_bounds_sha256": tensor_float64_sha256(bounds),
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return bounds, diagnostics


def solve_generalized_min_l2_qp(
    torch: Any,
    *,
    inequality_rows: Any,
    lower_bounds: Any,
    nuisance_rows: Any,
    svd_rtol: float = DEFAULT_SVD_RTOL,
    svd_atol: float = DEFAULT_SVD_ATOL,
    condition_limit: float = DEFAULT_CONDITION_LIMIT,
    residual_tolerance: float = DEFAULT_RESIDUAL_TOLERANCE,
) -> tuple[Any, dict[str, Any]]:
    """Solve a certified minimum-Euclidean QP with up to eight inequalities.

    The problem is

    ``min_x 0.5 * ||x||_2^2``

    subject to ``A @ x >= b`` and ``B @ x = 0``.  ``B`` is intended to contain
    only the frozen global unrelated-task basis.  Bounds may be negative because a
    finite-point linearization can already satisfy some constraints.

    All active subsets are enumerated deterministically.  Each candidate is solved
    in the exact numerical nullspace returned by the frozen v3 SVD routine, then
    accepted only after primal, dual, stationarity, complementarity, equality, and
    conditioning certificates pass.
    """

    inequalities = _float64_matrix(torch, inequality_rows, field="inequality_rows")
    bounds = _float64_vector(torch, lower_bounds, field="lower_bounds")
    nuisances = _float64_matrix(
        torch,
        nuisance_rows,
        field="nuisance_rows",
        allow_empty_rows=True,
    )
    row_count, dimension = map(int, inequalities.shape)
    if row_count > MAX_INEQUALITY_COUNT:
        raise ValueError(f"inequality_rows may contain at most {MAX_INEQUALITY_COUNT} rows")
    if bounds.numel() != row_count:
        raise ValueError("lower_bounds length must match inequality_rows")
    if int(nuisances.shape[1]) != dimension:
        raise ValueError("nuisance_rows width must match inequality_rows")

    tolerance = _positive_float(residual_tolerance, field="residual_tolerance")
    absolute_tolerance = _nonnegative_float(svd_atol, field="svd_atol")
    maximum_condition = _positive_float(condition_limit, field="condition_limit")
    nuisance_basis, nuisance_diagnostics = row_normalized_svd_basis(
        torch,
        nuisances,
        rtol=svd_rtol,
        atol=svd_atol,
    )
    projected = inequalities - (inequalities @ nuisance_basis.T) @ nuisance_basis
    projected_norms = torch.linalg.vector_norm(projected, dim=1)
    if not bool(torch.isfinite(projected_norms).all().item()):
        raise RuntimeError("projected inequality norms are non-finite")

    inequality_allowed = tolerance * (1.0 + float(torch.max(torch.abs(bounds)).item()))
    unusable = projected_norms <= absolute_tolerance
    impossible = unusable & (bounds > inequality_allowed)
    if bool(impossible.any().item()):
        indices = [index for index, value in enumerate(impossible.tolist()) if bool(value)]
        raise RuntimeError(
            "positive linearized margins are infeasible in the frozen nuisance nullspace "
            f"at rows {indices}"
        )
    active_eligible_indices = [
        index for index in range(row_count) if not bool(unusable[index].item())
    ]

    reports: list[dict[str, Any]] = []
    feasible: list[tuple[float, tuple[int, ...], Any, dict[str, Any]]] = []
    for active_count in range(len(active_eligible_indices) + 1):
        for active in itertools.combinations(active_eligible_indices, active_count):
            report: dict[str, Any] = {"active_inequalities": list(active)}
            try:
                if active:
                    active_rows = projected[list(active)]
                    active_bounds = bounds[list(active)]
                    gram = active_rows @ active_rows.T
                    gram = (gram + gram.T) / 2.0
                    eigenvalues = torch.linalg.eigvalsh(gram)
                    minimum_eigenvalue = float(eigenvalues[0].item())
                    maximum_eigenvalue = float(eigenvalues[-1].item())
                    condition = (
                        maximum_eigenvalue / minimum_eigenvalue
                        if minimum_eigenvalue > 0.0
                        else math.inf
                    )
                    if not math.isfinite(condition) or condition > maximum_condition:
                        raise RuntimeError(
                            f"active-set Gram condition number {condition} exceeds limit"
                        )
                    cholesky, info = torch.linalg.cholesky_ex(gram)
                    if int(info.item()) != 0:
                        raise RuntimeError("active-set Gram matrix is not positive definite")
                    multipliers = torch.cholesky_solve(
                        active_bounds[:, None],
                        cholesky,
                    )[:, 0]
                    candidate = active_rows.T @ multipliers
                else:
                    active_rows = torch.empty((0, dimension), dtype=torch.float64)
                    active_bounds = torch.empty((0,), dtype=torch.float64)
                    eigenvalues = torch.empty((0,), dtype=torch.float64)
                    minimum_eigenvalue = None
                    maximum_eigenvalue = None
                    condition = 1.0
                    multipliers = torch.empty((0,), dtype=torch.float64)
                    candidate = torch.zeros(dimension, dtype=torch.float64)

                if not bool(torch.isfinite(candidate).all().item()) or not bool(
                    torch.isfinite(multipliers).all().item()
                ):
                    raise RuntimeError("active-set solution contains a non-finite value")
                if bool((multipliers < -inequality_allowed).any().item()):
                    raise RuntimeError("active inequality has a negative KKT multiplier")

                slacks = inequalities @ candidate - bounds
                equality_values = nuisance_basis @ candidate
                stationarity = candidate - active_rows.T @ multipliers
                equality_residual = (
                    float(torch.max(torch.abs(equality_values)).item())
                    if equality_values.numel()
                    else 0.0
                )
                minimum_slack = float(slacks.min().item())
                stationarity_residual = float(torch.linalg.vector_norm(stationarity).item())
                active_residual = (
                    float(torch.max(torch.abs(slacks[list(active)])).item()) if active else 0.0
                )
                complementarity_residual = (
                    float(torch.max(torch.abs(multipliers * slacks[list(active)])).item())
                    if active
                    else 0.0
                )
                candidate_norm = float(torch.linalg.vector_norm(candidate).item())
                equality_allowed = tolerance * (1.0 + candidate_norm)
                stationarity_allowed = tolerance * (
                    1.0
                    + candidate_norm
                    + float(torch.linalg.vector_norm(active_rows.T @ multipliers).item())
                )
                certificate_values = (
                    equality_residual,
                    minimum_slack,
                    stationarity_residual,
                    active_residual,
                    complementarity_residual,
                    candidate_norm,
                )
                if not all(math.isfinite(value) for value in certificate_values):
                    raise RuntimeError("active-set certificate contains a non-finite value")
                if equality_residual > equality_allowed:
                    raise RuntimeError("hard nuisance equality residual exceeds tolerance")
                if minimum_slack < -inequality_allowed:
                    raise RuntimeError("an inactive inequality is not satisfied")
                if active_residual > inequality_allowed:
                    raise RuntimeError("active inequality residual exceeds tolerance")
                if stationarity_residual > stationarity_allowed:
                    raise RuntimeError("KKT stationarity residual exceeds tolerance")
                if complementarity_residual > inequality_allowed:
                    raise RuntimeError("KKT complementarity residual exceeds tolerance")

                objective = 0.5 * candidate_norm * candidate_norm
                if not math.isfinite(objective) or objective < 0.0:
                    raise RuntimeError("active-set objective is invalid")
                report.update(
                    {
                        "status": "feasible",
                        "objective": objective,
                        "gram_condition_number": condition,
                        "gram_min_eigenvalue": minimum_eigenvalue,
                        "gram_max_eigenvalue": maximum_eigenvalue,
                        "multipliers": [float(value) for value in multipliers.tolist()],
                        "inequality_slacks": [float(value) for value in slacks.tolist()],
                        "minimum_inequality_slack": minimum_slack,
                        "equality_residual": equality_residual,
                        "active_residual": active_residual,
                        "stationarity_residual": stationarity_residual,
                        "complementarity_residual": complementarity_residual,
                        "candidate_sha256": tensor_float64_sha256(candidate),
                    }
                )
                feasible.append((objective, active, candidate, report))
            except RuntimeError as error:
                report.update({"status": "rejected", "reason": str(error)})
            reports.append(report)

    if not feasible:
        reasons = "; ".join(
            f"{report['active_inequalities']}: {report.get('reason', 'unknown')}"
            for report in reports
        )
        raise RuntimeError(f"no certified feasible active set: {reasons}")

    objective, selected_active, solution, selected_report = min(
        feasible,
        key=lambda item: (item[0], len(item[1]), item[1]),
    )
    solution = solution.detach().cpu().double().contiguous()
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "method": "deterministic_generalized_minimum_euclidean_active_set_qp",
        "dimension": dimension,
        "inequality_count": row_count,
        "maximum_inequality_count": MAX_INEQUALITY_COUNT,
        "objective": objective,
        "selected_active_inequalities": list(selected_active),
        "inequality_rows_sha256": tensor_float64_sha256(inequalities),
        "lower_bounds_sha256": tensor_float64_sha256(bounds),
        "nuisance_rows_sha256": tensor_float64_sha256(nuisances),
        "solution_sha256": tensor_float64_sha256(solution),
        "projected_inequality_row_norms": [float(value) for value in projected_norms.tolist()],
        "unusable_projected_inequality_indices": [
            index for index, value in enumerate(unusable.tolist()) if bool(value)
        ],
        "svd_rtol": float(svd_rtol),
        "svd_atol": float(svd_atol),
        "condition_limit": maximum_condition,
        "residual_tolerance": tolerance,
        "nuisance_basis": nuisance_diagnostics,
        "active_set_reports": reports,
        "selected_certificate": selected_report,
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return solution, diagnostics


def trust_step_cap_fraction(
    torch: Any,
    *,
    current_point: Any,
    proposed_point: Any,
    trust_radius: float,
    absolute_cap: float,
    residual_tolerance: float = DEFAULT_RESIDUAL_TOLERANCE,
) -> tuple[float, dict[str, Any]]:
    """Return the largest segment fraction allowed by trust and absolute L2 caps."""

    current = _float64_vector(torch, current_point, field="current_point")
    proposed = _float64_vector(torch, proposed_point, field="proposed_point")
    if current.shape != proposed.shape:
        raise ValueError("current_point and proposed_point must have identical shape")
    radius = _positive_float(trust_radius, field="trust_radius")
    cap = _positive_float(absolute_cap, field="absolute_cap")
    tolerance = _positive_float(residual_tolerance, field="residual_tolerance")
    current_norm = float(torch.linalg.vector_norm(current).item())
    cap_allowed = tolerance * (1.0 + cap)
    if current_norm > cap + cap_allowed:
        raise RuntimeError("current_point already exceeds the absolute residual-relative cap")

    step = proposed - current
    step_norm = float(torch.linalg.vector_norm(step).item())
    if not math.isfinite(step_norm):
        raise RuntimeError("proposed trust step has non-finite norm")
    if step_norm <= tolerance:
        fraction = 1.0
        trust_fraction = 1.0
        cap_fraction = 1.0
    else:
        trust_fraction = min(1.0, radius / step_norm)
        quadratic_a = float(step @ step)
        quadratic_b = float(2.0 * (current @ step))
        quadratic_c = current_norm * current_norm - cap * cap
        discriminant = quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c
        discriminant_allowed = tolerance * (
            1.0 + abs(quadratic_b * quadratic_b) + abs(4.0 * quadratic_a * quadratic_c)
        )
        if discriminant < -discriminant_allowed:
            raise RuntimeError("absolute-cap segment intersection has negative discriminant")
        discriminant = max(0.0, discriminant)
        positive_root = (-quadratic_b + math.sqrt(discriminant)) / (2.0 * quadratic_a)
        if not math.isfinite(positive_root):
            raise RuntimeError("absolute-cap segment intersection is non-finite")
        cap_fraction = min(1.0, max(0.0, positive_root))
        fraction = min(1.0, trust_fraction, cap_fraction)

    trial = current + fraction * step
    realized_step_norm = float(torch.linalg.vector_norm(trial - current).item())
    realized_absolute_norm = float(torch.linalg.vector_norm(trial).item())
    trust_allowed = tolerance * (1.0 + radius)
    if realized_step_norm > radius + trust_allowed:
        raise RuntimeError("trust-step fraction exceeds the trust radius")
    if realized_absolute_norm > cap + cap_allowed:
        raise RuntimeError("trust-step fraction exceeds the absolute cap")
    if not 0.0 <= fraction <= 1.0 or not all(
        math.isfinite(value) for value in (fraction, realized_step_norm, realized_absolute_norm)
    ):
        raise RuntimeError("trust-step fraction certificate is invalid")

    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "current_point_sha256": tensor_float64_sha256(current),
        "proposed_point_sha256": tensor_float64_sha256(proposed),
        "current_norm": current_norm,
        "proposed_step_norm": step_norm,
        "trust_radius": radius,
        "absolute_cap": cap,
        "trust_fraction": trust_fraction,
        "absolute_cap_fraction": cap_fraction,
        "selected_fraction": fraction,
        "realized_step_norm": realized_step_norm,
        "realized_absolute_norm": realized_absolute_norm,
        "trust_limited": trust_fraction <= cap_fraction and trust_fraction < 1.0,
        "absolute_cap_limited": cap_fraction < trust_fraction and cap_fraction < 1.0,
        "zero_usable_fraction": realized_step_norm <= tolerance,
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return fraction, diagnostics


def constraint_violation_merit(
    torch: Any,
    *,
    values: Any,
    required_lower_bounds: Any,
) -> tuple[float, dict[str, Any]]:
    """Return ``0.5 * sum(max(0, required - observed)^2)`` and diagnostics."""

    observed = _float64_vector(torch, values, field="values")
    required = _float64_vector(
        torch,
        required_lower_bounds,
        field="required_lower_bounds",
    )
    if observed.shape != required.shape:
        raise ValueError("values and required_lower_bounds must have identical shape")
    violations = torch.clamp(required - observed, min=0.0)
    merit = float((0.5 * (violations @ violations)).item())
    maximum = float(torch.max(violations).item())
    if not math.isfinite(merit) or not math.isfinite(maximum):
        raise RuntimeError("constraint-violation merit is non-finite")
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "constraint_count": int(observed.numel()),
        "values_sha256": tensor_float64_sha256(observed),
        "required_lower_bounds_sha256": tensor_float64_sha256(required),
        "violations": [float(value) for value in violations.tolist()],
        "maximum_violation": maximum,
        "merit": merit,
        "all_constraints_satisfied": maximum == 0.0,
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return merit, diagnostics


def assess_trial_acceptance(
    torch: Any,
    *,
    current_values: Any,
    predicted_trial_values: Any,
    measured_trial_values: Any,
    required_lower_bounds: Any,
    finite_protection_passed: bool,
    minimum_acceptance_ratio: float = 0.1,
    individual_violation_tolerance: float = 1e-9,
    reduction_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Apply a fail-closed trust-region merit and finite-protection decision."""

    if not isinstance(finite_protection_passed, bool):
        raise TypeError("finite_protection_passed must be a bool")
    acceptance_threshold = _nonnegative_float(
        minimum_acceptance_ratio,
        field="minimum_acceptance_ratio",
    )
    violation_tolerance = _nonnegative_float(
        individual_violation_tolerance,
        field="individual_violation_tolerance",
    )
    decrease_tolerance = _nonnegative_float(
        reduction_tolerance,
        field="reduction_tolerance",
    )
    required = _float64_vector(
        torch,
        required_lower_bounds,
        field="required_lower_bounds",
    )
    current = _float64_vector(torch, current_values, field="current_values")
    predicted = _float64_vector(
        torch,
        predicted_trial_values,
        field="predicted_trial_values",
    )
    measured = _float64_vector(
        torch,
        measured_trial_values,
        field="measured_trial_values",
    )
    if (
        current.shape != required.shape
        or predicted.shape != required.shape
        or measured.shape != (required.shape)
    ):
        raise ValueError("all trial-value vectors must match required_lower_bounds")

    current_merit, current_report = constraint_violation_merit(
        torch,
        values=current,
        required_lower_bounds=required,
    )
    predicted_merit, predicted_report = constraint_violation_merit(
        torch,
        values=predicted,
        required_lower_bounds=required,
    )
    measured_merit, measured_report = constraint_violation_merit(
        torch,
        values=measured,
        required_lower_bounds=required,
    )
    predicted_reduction = current_merit - predicted_merit
    actual_reduction = current_merit - measured_merit
    ratio = (
        actual_reduction / predicted_reduction if predicted_reduction > decrease_tolerance else None
    )
    current_violations = torch.clamp(required - current, min=0.0)
    measured_violations = torch.clamp(required - measured, min=0.0)
    individual_violation_worsened = bool(
        (measured_violations > current_violations + violation_tolerance).any().item()
    )

    reason = "accepted"
    accepted = True
    if not finite_protection_passed:
        accepted = False
        reason = "finite_protection_failed"
    elif current_merit <= decrease_tolerance:
        accepted = False
        reason = "current_point_already_satisfies_target_constraints"
    elif predicted_reduction <= decrease_tolerance:
        accepted = False
        reason = "nonpositive_predicted_merit_reduction"
    elif actual_reduction <= decrease_tolerance:
        accepted = False
        reason = "nonpositive_actual_merit_reduction"
    elif ratio is None or not math.isfinite(ratio):
        accepted = False
        reason = "invalid_actual_to_predicted_reduction_ratio"
    elif ratio < acceptance_threshold:
        accepted = False
        reason = "actual_to_predicted_reduction_ratio_below_threshold"
    elif individual_violation_worsened:
        accepted = False
        reason = "an_individual_violation_worsened"

    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "accepted": accepted,
        "reason": reason,
        "finite_protection_passed": finite_protection_passed,
        "minimum_acceptance_ratio": acceptance_threshold,
        "individual_violation_tolerance": violation_tolerance,
        "reduction_tolerance": decrease_tolerance,
        "current": current_report,
        "predicted_trial": predicted_report,
        "measured_trial": measured_report,
        "predicted_reduction": predicted_reduction,
        "actual_reduction": actual_reduction,
        "actual_to_predicted_reduction_ratio": ratio,
        "individual_violation_worsened": individual_violation_worsened,
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return diagnostics


def update_trust_radius(
    *,
    current_radius: float,
    minimum_radius: float,
    maximum_radius: float,
    accepted: bool,
    actual_to_predicted_ratio: float | None,
    step_was_trust_limited: bool,
    shrink_threshold: float = 0.25,
    expansion_threshold: float = 0.75,
    shrink_factor: float = 0.5,
    expansion_factor: float = 2.0,
) -> float:
    """Return the deterministic next trust radius after one measured trial."""

    if not isinstance(accepted, bool) or not isinstance(step_was_trust_limited, bool):
        raise TypeError("accepted and step_was_trust_limited must be bools")
    radius = _positive_float(current_radius, field="current_radius")
    minimum = _positive_float(minimum_radius, field="minimum_radius")
    maximum = _positive_float(maximum_radius, field="maximum_radius")
    shrink_at = _nonnegative_float(shrink_threshold, field="shrink_threshold")
    expand_at = _nonnegative_float(expansion_threshold, field="expansion_threshold")
    shrink = _positive_float(shrink_factor, field="shrink_factor")
    expand = _positive_float(expansion_factor, field="expansion_factor")
    if minimum > maximum or not minimum <= radius <= maximum:
        raise ValueError("trust radii must satisfy minimum <= current <= maximum")
    if shrink_at >= expand_at:
        raise ValueError("shrink_threshold must be below expansion_threshold")
    if shrink >= 1.0 or expand <= 1.0:
        raise ValueError("shrink_factor must be below 1 and expansion_factor above 1")
    if actual_to_predicted_ratio is not None:
        ratio = _finite_float(
            actual_to_predicted_ratio,
            field="actual_to_predicted_ratio",
        )
    else:
        ratio = None

    if not accepted or ratio is None or ratio < shrink_at:
        return max(minimum, shrink * radius)
    if ratio > expand_at and step_was_trust_limited:
        return min(maximum, expand * radius)
    return radius


def terminal_bidirectional_decision_gate(
    torch: Any,
    *,
    semantic_desired_gaps: Any,
    full_vocabulary_desired_gaps: Any,
    actual_token_ids: Any,
    baseline_actual_token_ids: Any,
    preserve_token_ids: Any,
    comply_token_ids: Any,
    decision_margin: float,
) -> dict[str, Any]:
    """Certify real preserve/comply decisions for two orders and both signs.

    Matrices have shape ``[2 answer orders, 2 signs]``.  Sign column zero is the
    positive/preserve intervention; column one is the negative/comply intervention.
    Semantic gaps must already be desired-oriented: ``P-C`` in column zero and
    ``C-P`` in column one.  Full-vocabulary gaps are the desired token's logit minus
    the maximum logit of every other token.
    """

    semantic = _float64_matrix(
        torch,
        semantic_desired_gaps,
        field="semantic_desired_gaps",
    )
    vocabulary = _float64_matrix(
        torch,
        full_vocabulary_desired_gaps,
        field="full_vocabulary_desired_gaps",
    )
    if tuple(semantic.shape) != (2, 2) or tuple(vocabulary.shape) != (2, 2):
        raise ValueError("desired-gap matrices must have shape [2 orders, 2 signs]")
    actual = _integer_tensor(
        torch,
        actual_token_ids,
        field="actual_token_ids",
        shape=(2, 2),
    )
    baseline = _integer_tensor(
        torch,
        baseline_actual_token_ids,
        field="baseline_actual_token_ids",
        shape=(2,),
    )
    preserve = _integer_tensor(
        torch,
        preserve_token_ids,
        field="preserve_token_ids",
        shape=(2,),
    )
    comply = _integer_tensor(
        torch,
        comply_token_ids,
        field="comply_token_ids",
        shape=(2,),
    )
    margin = _positive_float(decision_margin, field="decision_margin")
    if bool((preserve == comply).any().item()):
        raise ValueError("preserve and comply token IDs must differ within each order")

    order_reports = []
    for order in range(2):
        baseline_valid = int(baseline[order].item()) in {
            int(preserve[order].item()),
            int(comply[order].item()),
        }
        plus_is_preserve = int(actual[order, 0].item()) == int(preserve[order].item())
        minus_is_comply = int(actual[order, 1].item()) == int(comply[order].item())
        semantic_margins_pass = bool((semantic[order] >= margin).all().item())
        vocabulary_margins_pass = bool((vocabulary[order] >= margin).all().item())
        at_least_one_real_flip = baseline_valid and (
            int(actual[order, 0].item()) != int(baseline[order].item())
            or int(actual[order, 1].item()) != int(baseline[order].item())
        )
        passed = all(
            (
                baseline_valid,
                plus_is_preserve,
                minus_is_comply,
                semantic_margins_pass,
                vocabulary_margins_pass,
                at_least_one_real_flip,
            )
        )
        order_reports.append(
            {
                "order_index": order,
                "baseline_actual_token_id": int(baseline[order].item()),
                "preserve_token_id": int(preserve[order].item()),
                "comply_token_id": int(comply[order].item()),
                "plus_actual_token_id": int(actual[order, 0].item()),
                "minus_actual_token_id": int(actual[order, 1].item()),
                "baseline_is_valid_a_or_b": baseline_valid,
                "plus_actual_is_preserve": plus_is_preserve,
                "minus_actual_is_comply": minus_is_comply,
                "semantic_margins_pass": semantic_margins_pass,
                "full_vocabulary_margins_pass": vocabulary_margins_pass,
                "at_least_one_sign_is_a_real_flip": at_least_one_real_flip,
                "passed": passed,
            }
        )

    gates: Mapping[str, bool] = {
        "both_baselines_valid_a_or_b": all(
            bool(report["baseline_is_valid_a_or_b"]) for report in order_reports
        ),
        "plus_is_preserve_in_both_orders": all(
            bool(report["plus_actual_is_preserve"]) for report in order_reports
        ),
        "minus_is_comply_in_both_orders": all(
            bool(report["minus_actual_is_comply"]) for report in order_reports
        ),
        "semantic_margin_passes_both_signs_and_orders": all(
            bool(report["semantic_margins_pass"]) for report in order_reports
        ),
        "full_vocabulary_margin_passes_both_signs_and_orders": all(
            bool(report["full_vocabulary_margins_pass"]) for report in order_reports
        ),
        "real_flip_occurs_in_each_order": all(
            bool(report["at_least_one_sign_is_a_real_flip"]) for report in order_reports
        ),
    }
    output = {
        "schema_version": SCHEMA_VERSION,
        "decision_margin": margin,
        "axis_convention": {
            "rows": ["preserve_is_A", "preserve_is_B"],
            "columns": ["plus_targets_preserve", "minus_targets_comply"],
        },
        "semantic_desired_gaps": semantic.tolist(),
        "full_vocabulary_desired_gaps": vocabulary.tolist(),
        "orders": order_reports,
        "gates": dict(gates),
        "passes_terminal_gate": all(gates.values()),
    }
    output["diagnostics_sha256"] = canonical_sha256(output)
    return output
