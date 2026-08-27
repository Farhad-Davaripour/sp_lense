from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from sp_lense.counterfactual_protected_natural_gradient import (
    FISHER_RIDGE_MULTIPLIER_GRID,
    PREDICTED_COARSENED_NEXT_TOKEN_KL_BUDGET_GRID,
    RESIDUAL_RELATIVE_L2_CAP_GRID,
    build_counterfactual_protected_natural_gradient,
    certify_applied_float32_perturbation,
    float32_accumulation_gamma,
    global_unrelated_null_projection,
    predicted_coarsened_next_token_kl,
    preregistered_candidate_grid,
    scale_to_predicted_coarsened_next_token_kl_budget,
    terminal_bidirectional_decision_gate,
)
from sp_lense.gradient_specificity_trust_region import (
    terminal_bidirectional_decision_gate as locked_terminal_gate,
)
from sp_lense.gradient_specificity_v3 import tensor_float64_sha256


def test_preregistered_grid_is_literal_complete_and_deterministic() -> None:
    grid = preregistered_candidate_grid()

    assert len(grid) == (
        len(FISHER_RIDGE_MULTIPLIER_GRID)
        * len(PREDICTED_COARSENED_NEXT_TOKEN_KL_BUDGET_GRID)
        * len(RESIDUAL_RELATIVE_L2_CAP_GRID)
    )
    assert grid[0] == {
        "fisher_ridge_multiplier": 0.01,
        "predicted_coarsened_next_token_kl_budget": 0.0005,
        "residual_relative_l2_cap": 0.05,
    }
    assert grid[-1] == {
        "fisher_ridge_multiplier": 1.0,
        "predicted_coarsened_next_token_kl_budget": 0.005,
        "residual_relative_l2_cap": 0.2,
    }
    assert preregistered_candidate_grid() == grid


def test_global_projection_exactly_cancels_unrelated_span() -> None:
    vector = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32)
    # Duplicate/scaled rows exercise rank reduction instead of assuming orthonormal input.
    unrelated = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32)

    projected, basis, diagnostics = global_unrelated_null_projection(
        torch,
        vector=vector,
        unrelated_gradient_rows=unrelated,
    )

    assert basis.shape == (1, 3)
    assert torch.allclose(projected, torch.tensor([0.0, 4.0, 5.0], dtype=torch.float64))
    assert torch.max(torch.abs(basis @ projected)).item() < 1e-12
    assert diagnostics["rank"] == 1
    assert diagnostics["maximum_abs_basis_projection"] < 1e-12
    assert len(diagnostics["diagnostics_sha256"]) == 64


def test_counterfactual_natural_gradient_uses_difference_null_and_low_fisher_cost() -> None:
    self_gradient = torch.tensor([10.0, 1.0, 1.0], dtype=torch.float64)
    matched_other = torch.tensor([4.0, 0.0, 0.0], dtype=torch.float64)
    unrelated = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    # Coordinate one has 100x the Fisher curvature of coordinate two.
    factors = torch.diag(torch.tensor([1.0, 10.0, 1.0], dtype=torch.float64))

    direction, diagnostics = build_counterfactual_protected_natural_gradient(
        torch,
        self_completion_gradient=self_gradient,
        matched_other_completion_gradient=matched_other,
        unrelated_gradient_rows=unrelated,
        protected_metric_factors=factors,
        fisher_ridge_multiplier=0.01,
    )

    assert abs(float(direction[0])) < 1e-12
    assert float(direction[2]) > 50.0 * float(direction[1])
    assert predicted_coarsened_next_token_kl(
        torch,
        perturbation=direction,
        protected_metric_factors=factors,
    ) == pytest.approx(1.0, abs=1e-10)
    assert diagnostics["contrast_mode"] == "self_minus_matched_other"
    assert diagnostics["projected_counterfactual_contrast_norm"] == pytest.approx(math.sqrt(2.0))
    assert diagnostics["feasible_dimension"] == 2
    assert diagnostics["raw_metric_relative_discrepancy"] < 1e-10
    assert diagnostics["ridge_regularized_objective_at_raw_solution"] > 0.0
    assert diagnostics["passes_minimum_separation_heuristic"] is True
    assert diagnostics["minimum_separation_is_not_a_vjp_error_bound"] is True


def test_self_only_ablation_changes_the_declared_contrast() -> None:
    self_gradient = torch.tensor([1.0, 2.0], dtype=torch.float64)
    other_gradient = torch.tensor([0.5, 1.5], dtype=torch.float64)
    empty_unrelated = torch.empty((0, 2), dtype=torch.float64)
    factors = torch.eye(2, dtype=torch.float64)

    primary, primary_diagnostics = build_counterfactual_protected_natural_gradient(
        torch,
        self_completion_gradient=self_gradient,
        matched_other_completion_gradient=other_gradient,
        unrelated_gradient_rows=empty_unrelated,
        protected_metric_factors=factors,
        fisher_ridge_multiplier=0.1,
    )
    ablation, ablation_diagnostics = build_counterfactual_protected_natural_gradient(
        torch,
        self_completion_gradient=self_gradient,
        matched_other_completion_gradient=other_gradient,
        unrelated_gradient_rows=empty_unrelated,
        protected_metric_factors=factors,
        fisher_ridge_multiplier=0.1,
        contrast_mode="self_only",
    )

    assert not torch.allclose(primary, ablation)
    assert primary_diagnostics["contrast_mode"] == "self_minus_matched_other"
    assert ablation_diagnostics["contrast_mode"] == "self_only"
    assert primary_diagnostics["unrelated_null_rank"] == 0


def test_near_cancelled_contrast_fails_locked_minimum_separation_screen() -> None:
    empty_unrelated = torch.empty((0, 2), dtype=torch.float64)

    with pytest.raises(RuntimeError, match="minimum-separation heuristic"):
        build_counterfactual_protected_natural_gradient(
            torch,
            self_completion_gradient=torch.tensor([1.0, 1.0], dtype=torch.float64),
            matched_other_completion_gradient=torch.tensor([1.0 - 1e-8, 1.0], dtype=torch.float64),
            unrelated_gradient_rows=empty_unrelated,
            protected_metric_factors=torch.eye(2, dtype=torch.float64),
            fisher_ridge_multiplier=0.1,
        )

    expected = (2 * (torch.finfo(torch.float32).eps / 2)) / (
        1 - 2 * (torch.finfo(torch.float32).eps / 2)
    )
    assert float32_accumulation_gamma(torch, 2) == pytest.approx(float(expected))


def test_scaling_hits_coarsened_kl_budget_when_cap_is_inactive() -> None:
    unit_direction = torch.tensor([math.sqrt(2.0), 0.0], dtype=torch.float64)
    factors = torch.tensor([[1.0, 0.0]], dtype=torch.float64)

    perturbation, diagnostics = scale_to_predicted_coarsened_next_token_kl_budget(
        torch,
        unit_coarsened_next_token_kl_direction=unit_direction,
        protected_metric_factors=factors,
        expected_protected_metric_factors_sha256=tensor_float64_sha256(factors),
        predicted_coarsened_next_token_kl_budget=0.002,
        residual_relative_l2_cap=0.1,
    )

    assert diagnostics["cap_was_active"] is False
    assert diagnostics["realized_predicted_coarsened_next_token_kl"] == pytest.approx(0.002)
    assert float(perturbation.norm()) < 0.1


def test_scaling_respects_l2_cap_and_reports_lower_realized_kl() -> None:
    unit_direction = torch.tensor([math.sqrt(2.0), 0.0], dtype=torch.float64)
    factors = torch.tensor([[1.0, 0.0]], dtype=torch.float64)

    perturbation, diagnostics = scale_to_predicted_coarsened_next_token_kl_budget(
        torch,
        unit_coarsened_next_token_kl_direction=unit_direction,
        protected_metric_factors=factors,
        expected_protected_metric_factors_sha256=tensor_float64_sha256(factors),
        predicted_coarsened_next_token_kl_budget=0.005,
        residual_relative_l2_cap=0.01,
    )

    assert diagnostics["cap_was_active"] is True
    assert float(perturbation.norm()) == pytest.approx(0.01)
    assert diagnostics["realized_predicted_coarsened_next_token_kl"] == pytest.approx(0.00005)


def test_scaler_rejects_factors_not_bound_during_construction() -> None:
    direction = torch.tensor([math.sqrt(2.0), 0.0], dtype=torch.float64)
    factors = torch.tensor([[1.0, 0.0]], dtype=torch.float64)

    with pytest.raises(ValueError, match="construction-bound"):
        scale_to_predicted_coarsened_next_token_kl_budget(
            torch,
            unit_coarsened_next_token_kl_direction=direction,
            protected_metric_factors=factors,
            expected_protected_metric_factors_sha256="0" * 64,
            predicted_coarsened_next_token_kl_budget=0.001,
            residual_relative_l2_cap=0.1,
        )


def test_applied_float32_certificate_recomputes_actual_dose_and_fails_over_budget() -> None:
    factors = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    requested = torch.tensor([0.04, 0.0], dtype=torch.float64)
    applied = requested.float().contiguous()

    certificate = certify_applied_float32_perturbation(
        torch,
        requested_perturbation=requested,
        applied_float32_perturbation=applied,
        protected_metric_factors=factors,
        expected_protected_metric_factors_sha256=tensor_float64_sha256(factors),
        predicted_coarsened_next_token_kl_budget=0.001,
        residual_relative_l2_cap=0.05,
    )

    assert certificate["applied_dtype"] == "torch.float32"
    assert certificate["applied_residual_relative_l2_norm"] == pytest.approx(0.04)
    assert certificate["applied_predicted_coarsened_next_token_kl"] == pytest.approx(0.0008)
    assert certificate["passes_applied_float32_budget_certificate"] is True

    with pytest.raises(RuntimeError, match="locked coarsened-KL budget"):
        certify_applied_float32_perturbation(
            torch,
            requested_perturbation=requested,
            applied_float32_perturbation=applied,
            protected_metric_factors=factors,
            expected_protected_metric_factors_sha256=tensor_float64_sha256(factors),
            predicted_coarsened_next_token_kl_budget=0.0005,
            residual_relative_l2_cap=0.05,
        )


def test_construction_fails_closed_for_zero_counterfactual_or_fisher_energy() -> None:
    empty = torch.empty((0, 2), dtype=torch.float64)
    vector = torch.tensor([1.0, 1.0], dtype=torch.float64)

    with pytest.raises(RuntimeError, match="contrast is numerically zero"):
        build_counterfactual_protected_natural_gradient(
            torch,
            self_completion_gradient=vector,
            matched_other_completion_gradient=vector,
            unrelated_gradient_rows=empty,
            protected_metric_factors=torch.eye(2, dtype=torch.float64),
            fisher_ridge_multiplier=0.1,
        )

    with pytest.raises(RuntimeError, match="metric has no positive scale"):
        build_counterfactual_protected_natural_gradient(
            torch,
            self_completion_gradient=vector,
            matched_other_completion_gradient=torch.zeros(2, dtype=torch.float64),
            unrelated_gradient_rows=empty,
            protected_metric_factors=torch.zeros((1, 2), dtype=torch.float64),
            fisher_ridge_multiplier=0.1,
        )


def test_terminal_gate_is_reused_not_reimplemented() -> None:
    assert terminal_bidirectional_decision_gate is locked_terminal_gate
