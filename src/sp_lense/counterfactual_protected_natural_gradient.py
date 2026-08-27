"""Pure math for counterfactual protected natural-gradient steering.

The model-facing procedure is deliberately outside this module.  This file only:

* removes the frozen global unrelated-task tangent span;
* forms a self-minus-matched-other completion-gradient contrast;
* preconditions that contrast by a protected-output metric factorization; and
* scales a direction to a declared local coarsened-next-token-KL budget and L2 cap.

No function accepts evaluation outcomes.  Candidate grids are literal constants so a
future runner can bind them before observing sealed results.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from sp_lense.gradient_specificity_trust_region import (
    terminal_bidirectional_decision_gate,
)
from sp_lense.gradient_specificity_v3 import (
    DEFAULT_SVD_ATOL,
    DEFAULT_SVD_RTOL,
    canonical_sha256,
    row_normalized_svd_basis,
    tensor_float32_sha256,
    tensor_float64_sha256,
)

SCHEMA_VERSION = "sp_lense.counterfactual_protected_natural_gradient.v1"

# Locked development candidates.  The ridge is relative to the mean diagonal of the
# protected coarsened metric, so the ridge parameterization is invariant to a common
# rescaling of its factor rows.  Fixed KL doses and L2-cap interactions are not.
# KL budgets use the local approximation KL ~= 0.5 * delta^T F delta.
FISHER_RIDGE_MULTIPLIER_GRID = (0.01, 0.1, 1.0)
PREDICTED_COARSENED_NEXT_TOKEN_KL_BUDGET_GRID = (0.0005, 0.001, 0.002, 0.005)
RESIDUAL_RELATIVE_L2_CAP_GRID = (0.05, 0.1, 0.15, 0.2)
COARSENED_METRIC_RELATIVE_DISCREPANCY_TOL = 1e-10

PRIMARY_CONTRAST_MODE = "self_minus_matched_other"
ABLATION_CONTRAST_MODES = ("self_only",)


class CounterfactualConstructionIneligible(RuntimeError):
    """A declared candidate-local numerical construction failure."""


def _finite_float(value: Any, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def _vector(torch: Any, value: Any, *, field: str) -> Any:
    if not torch.is_tensor(value):
        raise TypeError(f"{field} must be a tensor")
    if torch.is_complex(value) or not torch.is_floating_point(value):
        raise TypeError(f"{field} must be a real floating-point tensor")
    if value.ndim != 1 or int(value.numel()) == 0:
        raise ValueError(f"{field} must be a non-empty vector")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"{field} must contain only finite values")
    return result


def _matrix(
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
        raise ValueError(f"{field} must be a matrix with nonzero width")
    if int(value.shape[0]) == 0 and not allow_empty_rows:
        raise ValueError(f"{field} must contain at least one row")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"{field} must contain only finite values")
    return result


def float32_accumulation_gamma(torch: Any, dimension: int) -> float:
    """Return a float32 summation-error reference used only as a separation heuristic.

    This does not bound error in a captured neural-network VJP or backpropagation.
    """

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("dimension must be a positive integer")
    unit_roundoff = float(torch.finfo(torch.float32).eps) / 2.0
    product = dimension * unit_roundoff
    if product >= 1.0:
        raise ValueError("dimension is too large for the float32 gamma bound")
    return product / (1.0 - product)


def preregistered_candidate_grid() -> tuple[dict[str, float], ...]:
    """Return the literal development grid in deterministic tie-break order."""

    return tuple(
        {
            "fisher_ridge_multiplier": ridge,
            "predicted_coarsened_next_token_kl_budget": budget,
            "residual_relative_l2_cap": cap,
        }
        for ridge in FISHER_RIDGE_MULTIPLIER_GRID
        for budget in PREDICTED_COARSENED_NEXT_TOKEN_KL_BUDGET_GRID
        for cap in RESIDUAL_RELATIVE_L2_CAP_GRID
    )


def _project_out_basis(torch: Any, value: Any, basis: Any) -> Any:
    if int(basis.shape[0]) == 0:
        return value.clone().contiguous()
    return (value - basis.transpose(0, 1) @ (basis @ value)).contiguous()


def _project_factor_rows(torch: Any, factors: Any, basis: Any) -> Any:
    if int(basis.shape[0]) == 0:
        return factors.clone().contiguous()
    return (factors - (factors @ basis.transpose(0, 1)) @ basis).contiguous()


def global_unrelated_null_projection(
    torch: Any,
    *,
    vector: Any,
    unrelated_gradient_rows: Any,
    svd_rtol: float = DEFAULT_SVD_RTOL,
    svd_atol: float = DEFAULT_SVD_ATOL,
) -> tuple[Any, Any, dict[str, Any]]:
    """Project a vector out of the row span of unrelated-task gradients.

    Empty rows are accepted only to make the preregistered no-null ablation explicit.
    The primary method requires a non-empty, frozen unrelated matrix.
    """

    candidate = _vector(torch, vector, field="vector")
    rows = _matrix(
        torch,
        unrelated_gradient_rows,
        field="unrelated_gradient_rows",
        allow_empty_rows=True,
    )
    if int(rows.shape[1]) != int(candidate.numel()):
        raise ValueError("unrelated_gradient_rows width differs from vector")
    if int(rows.shape[0]) == 0:
        basis = torch.empty((0, candidate.numel()), dtype=torch.float64)
        basis_diagnostics: Mapping[str, Any] = {
            "rank": 0,
            "basis_sha256": tensor_float64_sha256(basis),
            "ablation_only": True,
        }
    else:
        basis, basis_diagnostics = row_normalized_svd_basis(
            torch,
            rows,
            rtol=_finite_float(svd_rtol, field="svd_rtol", positive=True),
            atol=_finite_float(svd_atol, field="svd_atol", positive=True),
        )
        basis = basis.double().contiguous()
    projected = _project_out_basis(torch, candidate, basis)
    maximum_projection = (
        float(torch.max(torch.abs(basis @ projected)).item()) if basis.numel() else 0.0
    )
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "input_row_count": int(rows.shape[0]),
        "dimension": int(candidate.numel()),
        "rank": int(basis.shape[0]),
        "input_vector_sha256": tensor_float64_sha256(candidate),
        "unrelated_rows_sha256": tensor_float64_sha256(rows),
        "basis_sha256": tensor_float64_sha256(basis),
        "projected_vector_sha256": tensor_float64_sha256(projected),
        "input_norm": float(candidate.norm().item()),
        "projected_norm": float(projected.norm().item()),
        "maximum_abs_basis_projection": maximum_projection,
        "basis_diagnostics": dict(basis_diagnostics),
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return projected, basis, diagnostics


def predicted_coarsened_next_token_kl(
    torch: Any,
    *,
    perturbation: Any,
    protected_metric_factors: Any,
) -> float:
    """Return the local coarsened next-token KL ``0.5 * ||L delta||^2``.

    The caller owns the provenance of ``L``.  In the development runner it is the
    prompt-balanced, top-k-plus-aggregate-tail next-token pullback factor.  It is not
    the full-vocabulary Fisher and this value is not the actual finite KL gate.
    """

    delta = _vector(torch, perturbation, field="perturbation")
    factors = _matrix(torch, protected_metric_factors, field="protected_metric_factors")
    if int(factors.shape[1]) != int(delta.numel()):
        raise ValueError("protected_metric_factors width differs from perturbation")
    value = 0.5 * float(torch.linalg.vector_norm(factors @ delta).square().item())
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError("predicted coarsened next-token KL is invalid")
    return value


def build_counterfactual_protected_natural_gradient(
    torch: Any,
    *,
    self_completion_gradient: Any,
    matched_other_completion_gradient: Any,
    unrelated_gradient_rows: Any,
    protected_metric_factors: Any,
    fisher_ridge_multiplier: float,
    contrast_mode: str = PRIMARY_CONTRAST_MODE,
    svd_rtol: float = DEFAULT_SVD_RTOL,
    svd_atol: float = DEFAULT_SVD_ATOL,
) -> tuple[Any, dict[str, Any]]:
    """Construct a ridge-regularized protected natural-gradient orientation.

    The primary contrast is ``g_self - g_other``.  Both terms, and every Fisher
    factor row, are projected into the exact null of the unrelated-gradient span.
    The protected coarsened metric is represented by rows ``L`` such that
    ``F_coarse = L.T @ L``.
    In the feasible unrelated-task null, the raw vector is the unique maximizer of
    ``r.T @ v - 0.5 * v.T @ (F + ridge I) @ v``.  A Woodbury solve applies the
    inverse without forming a dense ``dimension x dimension`` matrix.  This is not
    maximization per unit of the unregularized Fisher.  The returned orientation is
    subsequently normalized to one unit of the original factor metric solely to give
    the preregistered dose coordinate a common scale.
    """

    self_gradient = _vector(torch, self_completion_gradient, field="self_completion_gradient")
    other_gradient = _vector(
        torch,
        matched_other_completion_gradient,
        field="matched_other_completion_gradient",
    )
    if self_gradient.shape != other_gradient.shape:
        raise ValueError("self and matched-other gradients must have the same shape")
    factors = _matrix(torch, protected_metric_factors, field="protected_metric_factors")
    dimension = int(self_gradient.numel())
    if int(factors.shape[1]) != dimension:
        raise ValueError("protected_metric_factors width differs from gradients")
    if contrast_mode not in (PRIMARY_CONTRAST_MODE, *ABLATION_CONTRAST_MODES):
        raise ValueError("unsupported contrast_mode")

    projected_self, basis, self_projection = global_unrelated_null_projection(
        torch,
        vector=self_gradient,
        unrelated_gradient_rows=unrelated_gradient_rows,
        svd_rtol=svd_rtol,
        svd_atol=svd_atol,
    )
    projected_other = _project_out_basis(torch, other_gradient, basis)
    if contrast_mode == PRIMARY_CONTRAST_MODE:
        contrast = (projected_self - projected_other).contiguous()
    else:
        contrast = projected_self.clone().contiguous()
    contrast_norm = float(contrast.norm().item())
    numerical_floor = (
        64.0
        * torch.finfo(torch.float64).eps
        * (1.0 + float(projected_self.norm().item()) + float(projected_other.norm().item()))
    )
    if not math.isfinite(contrast_norm) or contrast_norm <= numerical_floor:
        raise CounterfactualConstructionIneligible(
            "projected counterfactual contrast is numerically zero"
        )
    projected_component_norm_sum = float(
        projected_self.norm().item() + projected_other.norm().item()
    )
    if not math.isfinite(projected_component_norm_sum) or projected_component_norm_sum <= 0.0:
        raise RuntimeError("projected contrast components have no positive norm")
    minimum_separation_ratio = contrast_norm / projected_component_norm_sum
    minimum_separation_reference = float32_accumulation_gamma(torch, dimension)
    if minimum_separation_ratio <= minimum_separation_reference:
        raise CounterfactualConstructionIneligible(
            "projected counterfactual contrast failed the minimum-separation heuristic"
        )

    projected_factors = _project_factor_rows(torch, factors, basis)
    fisher_trace_on_feasible_null = float(
        torch.linalg.matrix_norm(projected_factors).square().item()
    )
    feasible_dimension = dimension - int(basis.shape[0])
    if feasible_dimension <= 0:
        raise RuntimeError("unrelated-task null has no feasible dimension")
    mean_feasible_curvature = fisher_trace_on_feasible_null / feasible_dimension
    if not math.isfinite(mean_feasible_curvature) or mean_feasible_curvature <= 0.0:
        raise CounterfactualConstructionIneligible(
            "projected protected metric has no positive scale"
        )
    ridge_multiplier = _finite_float(
        fisher_ridge_multiplier,
        field="fisher_ridge_multiplier",
        positive=True,
    )
    ridge = ridge_multiplier * mean_feasible_curvature

    # Woodbury: (ridge I + L^T L)^-1 r
    row_count = int(projected_factors.shape[0])
    middle = projected_factors @ projected_factors.transpose(0, 1)
    middle = middle + ridge * torch.eye(row_count, dtype=torch.float64)
    rhs = projected_factors @ contrast
    coefficients = torch.linalg.solve(middle, rhs)
    solve_residual = middle @ coefficients - rhs
    relative_solve_residual = float(solve_residual.norm().item()) / (1.0 + float(rhs.norm().item()))
    middle_condition_number = float(torch.linalg.cond(middle).item())
    if not math.isfinite(middle_condition_number):
        raise RuntimeError("regularized Fisher system has a non-finite condition number")
    if relative_solve_residual > 1e-9:
        raise RuntimeError("regularized Fisher solve failed its residual certificate")
    raw = ((contrast - projected_factors.transpose(0, 1) @ coefficients) / ridge).contiguous()
    raw = _project_out_basis(torch, raw, basis)
    if not bool(torch.isfinite(raw).all().item()):
        raise RuntimeError("natural-gradient solve returned a non-finite vector")
    stationarity = (
        ridge * raw + projected_factors.transpose(0, 1) @ (projected_factors @ raw) - contrast
    )
    relative_stationarity_residual = float(stationarity.norm().item()) / (
        1.0 + float(contrast.norm().item())
    )
    if relative_stationarity_residual > 1e-8:
        raise RuntimeError("natural-gradient solve failed its stationarity certificate")

    raw_predicted_coarsened_kl_original = predicted_coarsened_next_token_kl(
        torch,
        perturbation=raw,
        protected_metric_factors=factors,
    )
    raw_predicted_coarsened_kl_projected = predicted_coarsened_next_token_kl(
        torch,
        perturbation=raw,
        protected_metric_factors=projected_factors,
    )
    raw_metric_discrepancy = abs(
        raw_predicted_coarsened_kl_original - raw_predicted_coarsened_kl_projected
    ) / max(
        raw_predicted_coarsened_kl_original,
        raw_predicted_coarsened_kl_projected,
        torch.finfo(torch.float64).tiny,
    )
    if raw_metric_discrepancy > COARSENED_METRIC_RELATIVE_DISCREPANCY_TOL:
        raise RuntimeError("original and projected protected metrics disagree on null vector")
    kl_floor = (
        1024.0
        * torch.finfo(torch.float64).eps
        * (1.0 + float(raw.norm().square().item()) * mean_feasible_curvature)
    )
    if raw_predicted_coarsened_kl_original <= kl_floor:
        raise CounterfactualConstructionIneligible(
            "natural direction has no certifiable protected-metric energy"
        )
    direction = (raw / math.sqrt(raw_predicted_coarsened_kl_original)).contiguous()
    normalized_predicted_coarsened_kl_original = predicted_coarsened_next_token_kl(
        torch,
        perturbation=direction,
        protected_metric_factors=factors,
    )
    normalized_predicted_coarsened_kl_projected = predicted_coarsened_next_token_kl(
        torch,
        perturbation=direction,
        protected_metric_factors=projected_factors,
    )
    normalized_metric_discrepancy = abs(
        normalized_predicted_coarsened_kl_original - normalized_predicted_coarsened_kl_projected
    ) / max(
        normalized_predicted_coarsened_kl_original,
        normalized_predicted_coarsened_kl_projected,
        torch.finfo(torch.float64).tiny,
    )
    maximum_null_projection = (
        float(torch.max(torch.abs(basis @ direction)).item()) if basis.numel() else 0.0
    )
    tolerance = 1e-9 * (1.0 + float(direction.norm().item()))
    if abs(normalized_predicted_coarsened_kl_original - 1.0) > 1e-9:
        raise RuntimeError("coarsened-next-token-KL normalization certificate failed")
    if normalized_metric_discrepancy > COARSENED_METRIC_RELATIVE_DISCREPANCY_TOL:
        raise RuntimeError("projected metric failed the normalization cross-certificate")
    if maximum_null_projection > tolerance:
        raise RuntimeError("natural direction left the unrelated-task null")

    regularized_energy = float(
        torch.linalg.vector_norm(projected_factors @ direction).square().item()
        + ridge * direction.norm().square().item()
    )
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "method": "counterfactual_protected_natural_gradient",
        "contrast_mode": contrast_mode,
        "dimension": dimension,
        "protected_metric_factor_count": row_count,
        "self_gradient_sha256": tensor_float64_sha256(self_gradient),
        "matched_other_gradient_sha256": tensor_float64_sha256(other_gradient),
        "projected_self_gradient_sha256": tensor_float64_sha256(projected_self),
        "projected_matched_other_gradient_sha256": tensor_float64_sha256(projected_other),
        "counterfactual_contrast_sha256": tensor_float64_sha256(contrast),
        "protected_metric_factors_sha256": tensor_float64_sha256(factors),
        "projected_protected_metric_factors_sha256": tensor_float64_sha256(projected_factors),
        "direction_sha256": tensor_float64_sha256(direction),
        "self_gradient_norm": float(self_gradient.norm().item()),
        "matched_other_gradient_norm": float(other_gradient.norm().item()),
        "projected_counterfactual_contrast_norm": contrast_norm,
        "projected_component_norm_sum": projected_component_norm_sum,
        "minimum_separation_ratio": minimum_separation_ratio,
        "minimum_separation_reference_dimension": dimension,
        "float32_unit_roundoff_reference": float(torch.finfo(torch.float32).eps) / 2.0,
        "minimum_separation_gamma_reference": minimum_separation_reference,
        "passes_minimum_separation_heuristic": True,
        "minimum_separation_is_not_a_vjp_error_bound": True,
        "unrelated_null_rank": int(basis.shape[0]),
        "maximum_abs_unrelated_basis_projection": maximum_null_projection,
        "feasible_dimension": feasible_dimension,
        "metric_trace_on_feasible_null": fisher_trace_on_feasible_null,
        "mean_feasible_metric_curvature": mean_feasible_curvature,
        "fisher_ridge_multiplier": ridge_multiplier,
        "fisher_ridge": ridge,
        "woodbury_middle_condition_number": middle_condition_number,
        "woodbury_relative_solve_residual": relative_solve_residual,
        "relative_stationarity_residual": relative_stationarity_residual,
        "raw_predicted_coarsened_next_token_kl_original_factors": raw_predicted_coarsened_kl_original,
        "raw_predicted_coarsened_next_token_kl_projected_factors": raw_predicted_coarsened_kl_projected,
        "raw_metric_relative_discrepancy": raw_metric_discrepancy,
        "normalized_predicted_coarsened_next_token_kl_original_factors": normalized_predicted_coarsened_kl_original,
        "normalized_predicted_coarsened_next_token_kl_projected_factors": normalized_predicted_coarsened_kl_projected,
        "normalized_metric_relative_discrepancy": normalized_metric_discrepancy,
        "direction_l2_norm": float(direction.norm().item()),
        "regularized_direction_energy": regularized_energy,
        "ridge_regularized_objective_at_raw_solution": float(
            contrast @ raw
            - 0.5
            * (
                torch.linalg.vector_norm(projected_factors @ raw).square()
                + ridge * raw.norm().square()
            )
        ),
        "counterfactual_alignment_at_unit_coarsened_metric": float(contrast @ direction),
        "global_unrelated_projection": self_projection,
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return direction, diagnostics


def scale_to_predicted_coarsened_next_token_kl_budget(
    torch: Any,
    *,
    unit_coarsened_next_token_kl_direction: Any,
    protected_metric_factors: Any,
    expected_protected_metric_factors_sha256: str,
    predicted_coarsened_next_token_kl_budget: float,
    residual_relative_l2_cap: float,
) -> tuple[Any, dict[str, Any]]:
    """Scale a unit coarsened-metric direction under a residual L2 dose cap."""

    direction = _vector(
        torch,
        unit_coarsened_next_token_kl_direction,
        field="unit_coarsened_next_token_kl_direction",
    )
    factors = _matrix(torch, protected_metric_factors, field="protected_metric_factors")
    if int(factors.shape[1]) != int(direction.numel()):
        raise ValueError("protected_metric_factors width differs from direction")
    factor_hash = tensor_float64_sha256(factors)
    if not isinstance(expected_protected_metric_factors_sha256, str):
        raise TypeError("expected_protected_metric_factors_sha256 must be a string")
    if factor_hash != expected_protected_metric_factors_sha256:
        raise ValueError("protected metric factors differ from the construction-bound factors")
    unit_kl = predicted_coarsened_next_token_kl(
        torch,
        perturbation=direction,
        protected_metric_factors=factors,
    )
    if abs(unit_kl - 1.0) > 1e-9:
        raise ValueError("direction must be normalized to one unit of coarsened next-token KL")
    budget = _finite_float(
        predicted_coarsened_next_token_kl_budget,
        field="predicted_coarsened_next_token_kl_budget",
        positive=True,
    )
    cap = _finite_float(
        residual_relative_l2_cap,
        field="residual_relative_l2_cap",
        positive=True,
    )
    requested_scale = math.sqrt(budget)
    direction_norm = float(direction.norm().item())
    cap_scale = cap / direction_norm
    realized_scale = min(requested_scale, cap_scale)
    perturbation = (realized_scale * direction).contiguous()
    realized_kl = predicted_coarsened_next_token_kl(
        torch,
        perturbation=perturbation,
        protected_metric_factors=factors,
    )
    realized_norm = float(perturbation.norm().item())
    tolerance = 1e-12 * (1.0 + budget + cap)
    if realized_kl > budget + tolerance or realized_norm > cap + tolerance:
        raise RuntimeError("scaled perturbation exceeds its locked budget")
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "predicted_coarsened_next_token_kl_budget": budget,
        "residual_relative_l2_cap": cap,
        "requested_scale": requested_scale,
        "cap_scale": cap_scale,
        "realized_scale": realized_scale,
        "cap_was_active": cap_scale < requested_scale,
        "realized_predicted_coarsened_next_token_kl": realized_kl,
        "realized_residual_relative_l2_norm": realized_norm,
        "direction_sha256": tensor_float64_sha256(direction),
        "protected_metric_factors_sha256": factor_hash,
        "perturbation_sha256": tensor_float64_sha256(perturbation),
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return perturbation, diagnostics


def certify_applied_float32_perturbation(
    torch: Any,
    *,
    requested_perturbation: Any,
    applied_float32_perturbation: Any,
    protected_metric_factors: Any,
    expected_protected_metric_factors_sha256: str,
    predicted_coarsened_next_token_kl_budget: float,
    residual_relative_l2_cap: float,
) -> dict[str, Any]:
    """Certify the actual float32 dose using exact observed cast-error bounds."""

    requested = _vector(torch, requested_perturbation, field="requested_perturbation")
    if (
        not torch.is_tensor(applied_float32_perturbation)
        or applied_float32_perturbation.dtype != torch.float32
        or applied_float32_perturbation.device.type != "cpu"
        or applied_float32_perturbation.ndim != 1
    ):
        raise TypeError("applied_float32_perturbation must be a CPU float32 vector")
    applied_float32 = applied_float32_perturbation.detach().contiguous()
    if not bool(torch.isfinite(applied_float32).all().item()):
        raise ValueError("applied_float32_perturbation must contain only finite values")
    if requested.shape != applied_float32.shape:
        raise ValueError("requested and applied perturbations must have the same shape")
    if not torch.equal(applied_float32, requested.float()):
        raise ValueError("applied perturbation is not the exact float32 cast of requested")
    applied = applied_float32.double().contiguous()
    factors = _matrix(torch, protected_metric_factors, field="protected_metric_factors")
    if int(factors.shape[1]) != int(requested.numel()):
        raise ValueError("protected_metric_factors width differs from perturbation")
    factor_hash = tensor_float64_sha256(factors)
    if factor_hash != expected_protected_metric_factors_sha256:
        raise ValueError("protected metric factors differ from the construction-bound factors")
    budget = _finite_float(
        predicted_coarsened_next_token_kl_budget,
        field="predicted_coarsened_next_token_kl_budget",
        positive=True,
    )
    cap = _finite_float(
        residual_relative_l2_cap,
        field="residual_relative_l2_cap",
        positive=True,
    )
    requested_norm = float(requested.norm().item())
    applied_norm = float(applied.norm().item())
    rounding_error = (applied - requested).contiguous()
    rounding_error_norm = float(rounding_error.norm().item())
    mapped_requested = factors @ requested
    mapped_rounding_error = factors @ rounding_error
    mapped_rounding_error_norm = float(mapped_rounding_error.norm().item())
    requested_kl = predicted_coarsened_next_token_kl(
        torch,
        perturbation=requested,
        protected_metric_factors=factors,
    )
    applied_kl = predicted_coarsened_next_token_kl(
        torch,
        perturbation=applied,
        protected_metric_factors=factors,
    )
    float64_floor = 32.0 * float(torch.finfo(torch.float64).eps)
    l2_tolerance = rounding_error_norm + float64_floor * (1.0 + cap)
    kl_tolerance = (
        float(mapped_requested.norm().item()) * mapped_rounding_error_norm
        + 0.5 * mapped_rounding_error_norm**2
        + float64_floor * (1.0 + budget)
    )
    if requested_norm > cap + float64_floor * (1.0 + cap):
        raise RuntimeError("requested perturbation exceeds its locked L2 cap")
    if requested_kl > budget + float64_floor * (1.0 + budget):
        raise RuntimeError("requested perturbation exceeds its locked coarsened-KL budget")
    if applied_norm > cap + l2_tolerance:
        raise RuntimeError("applied float32 perturbation exceeds its locked L2 cap")
    if applied_kl > budget + kl_tolerance:
        raise RuntimeError("applied float32 perturbation exceeds its locked coarsened-KL budget")
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "applied_dtype": "torch.float32",
        "dimension": int(requested.numel()),
        "tolerance_rule": "exact_observed_float32_cast_error_triangle_bound",
        "predicted_coarsened_next_token_kl_budget": budget,
        "residual_relative_l2_cap": cap,
        "requested_residual_relative_l2_norm": requested_norm,
        "applied_residual_relative_l2_norm": applied_norm,
        "requested_predicted_coarsened_next_token_kl": requested_kl,
        "applied_predicted_coarsened_next_token_kl": applied_kl,
        "float32_rounding_error_l2_norm": rounding_error_norm,
        "mapped_float32_rounding_error_l2_norm": mapped_rounding_error_norm,
        "l2_float32_aware_tolerance": l2_tolerance,
        "coarsened_kl_float32_aware_tolerance": kl_tolerance,
        "passes_applied_float32_budget_certificate": True,
        "requested_perturbation_sha256": tensor_float64_sha256(requested),
        "applied_float32_perturbation_sha256": tensor_float32_sha256(applied_float32),
        "applied_as_float64_perturbation_sha256": tensor_float64_sha256(applied),
        "protected_metric_factors_sha256": factor_hash,
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return diagnostics


__all__ = [
    "ABLATION_CONTRAST_MODES",
    "COARSENED_METRIC_RELATIVE_DISCREPANCY_TOL",
    "FISHER_RIDGE_MULTIPLIER_GRID",
    "PREDICTED_COARSENED_NEXT_TOKEN_KL_BUDGET_GRID",
    "PRIMARY_CONTRAST_MODE",
    "RESIDUAL_RELATIVE_L2_CAP_GRID",
    "CounterfactualConstructionIneligible",
    "build_counterfactual_protected_natural_gradient",
    "certify_applied_float32_perturbation",
    "float32_accumulation_gamma",
    "global_unrelated_null_projection",
    "predicted_coarsened_next_token_kl",
    "preregistered_candidate_grid",
    "scale_to_predicted_coarsened_next_token_kl_budget",
    "terminal_bidirectional_decision_gate",
]
