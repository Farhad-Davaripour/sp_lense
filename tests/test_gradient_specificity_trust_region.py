from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from sp_lense.gradient_specificity_trust_region import (
    MAX_INEQUALITY_COUNT,
    assess_trial_acceptance,
    constraint_violation_merit,
    linearized_lower_bounds,
    solve_generalized_min_l2_qp,
    terminal_bidirectional_decision_gate,
    trust_step_cap_fraction,
    update_trust_radius,
)


def _empty_rows(dimension: int) -> torch.Tensor:
    return torch.empty((0, dimension), dtype=torch.float64)


def test_linearized_bounds_use_an_absolute_candidate_coordinate() -> None:
    point = torch.tensor([0.2, -0.1], dtype=torch.float64)
    values = torch.tensor([0.3, 0.4], dtype=torch.float64)
    gradients = torch.eye(2, dtype=torch.float64)
    required = torch.tensor([0.5, 0.5], dtype=torch.float64)

    bounds, diagnostics = linearized_lower_bounds(
        torch,
        current_point=point,
        current_values=values,
        gradient_rows=gradients,
        required_lower_bounds=required,
    )

    assert torch.allclose(
        bounds,
        torch.tensor([0.4, 0.0], dtype=torch.float64),
        atol=1e-15,
        rtol=0.0,
    )
    assert torch.allclose(values + gradients @ (point - point), values)
    assert torch.allclose(values + gradients @ (bounds - point), required)
    assert diagnostics["row_count"] == 2
    assert len(diagnostics["diagnostics_sha256"]) == 64


def test_generalized_qp_solves_all_eight_inequalities_inside_global_null() -> None:
    basis = torch.eye(10, dtype=torch.float64)
    inequalities = basis[:MAX_INEQUALITY_COUNT]
    bounds = torch.arange(1, MAX_INEQUALITY_COUNT + 1, dtype=torch.float64) / 10.0
    nuisance = basis[8:]

    solution, diagnostics = solve_generalized_min_l2_qp(
        torch,
        inequality_rows=inequalities,
        lower_bounds=bounds,
        nuisance_rows=nuisance,
    )

    expected = torch.cat((bounds, torch.zeros(2, dtype=torch.float64)))
    assert torch.allclose(solution, expected, atol=1e-12, rtol=0.0)
    assert torch.max(torch.abs(nuisance @ solution)).item() < 1e-12
    assert bool(torch.all(inequalities @ solution >= bounds - 1e-12))
    assert diagnostics["selected_active_inequalities"] == list(range(8))
    assert diagnostics["nuisance_basis"]["rank"] == 2
    assert diagnostics["selected_certificate"]["equality_residual"] < 1e-12
    assert len(diagnostics["active_set_reports"]) == 2**8


def test_qp_treats_matched_other_protection_as_inequality_not_null() -> None:
    basis = torch.eye(4, dtype=torch.float64)
    # Row zero is a self target. Row one stands in for a relinearized
    # matched-other baseline-greedy margin. It is allowed to be nonzero.
    inequalities = torch.stack((basis[0], basis[1]))
    bounds = torch.tensor([1.0, 0.5], dtype=torch.float64)

    solution, diagnostics = solve_generalized_min_l2_qp(
        torch,
        inequality_rows=inequalities,
        lower_bounds=bounds,
        nuisance_rows=basis[2:3],
    )

    assert torch.allclose(
        solution,
        torch.tensor([1.0, 0.5, 0.0, 0.0], dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    assert float(inequalities[1] @ solution) == pytest.approx(0.5)
    assert diagnostics["nuisance_basis"]["rank"] == 1


def test_qp_allows_already_satisfied_negative_bounds_and_empty_active_set() -> None:
    inequalities = torch.eye(3, dtype=torch.float64)[:2]
    bounds = torch.tensor([-0.5, 0.0], dtype=torch.float64)

    solution, diagnostics = solve_generalized_min_l2_qp(
        torch,
        inequality_rows=inequalities,
        lower_bounds=bounds,
        nuisance_rows=_empty_rows(3),
    )

    assert torch.equal(solution, torch.zeros(3, dtype=torch.float64))
    assert diagnostics["selected_active_inequalities"] == []
    assert diagnostics["objective"] == 0.0


def test_qp_is_stable_to_nuisance_and_inequality_reordering() -> None:
    basis = torch.eye(6, dtype=torch.float64)
    inequalities = torch.stack(
        (
            basis[0] + 0.2 * basis[4],
            basis[1] - 0.1 * basis[4],
            basis[0] + basis[1] + 0.05 * basis[5],
        )
    )
    bounds = torch.tensor([0.4, 0.6, 1.2], dtype=torch.float64)
    nuisance = torch.stack((basis[4], basis[5], 2.0 * basis[4]))
    first, _ = solve_generalized_min_l2_qp(
        torch,
        inequality_rows=inequalities,
        lower_bounds=bounds,
        nuisance_rows=nuisance,
    )
    permutation = torch.tensor([2, 0, 1])
    second, _ = solve_generalized_min_l2_qp(
        torch,
        inequality_rows=inequalities[permutation],
        lower_bounds=bounds[permutation],
        nuisance_rows=nuisance.flip(0),
    )

    assert torch.allclose(first, second, atol=1e-12, rtol=0.0)
    assert torch.max(torch.abs(nuisance @ first)).item() < 1e-12


def test_qp_fails_closed_on_infeasible_excess_or_nonfinite_inputs() -> None:
    basis = torch.eye(3, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="infeasible in the frozen nuisance nullspace"):
        solve_generalized_min_l2_qp(
            torch,
            inequality_rows=basis[:1],
            lower_bounds=torch.ones(1, dtype=torch.float64),
            nuisance_rows=basis[:1],
        )

    with pytest.raises(ValueError, match="at most 8"):
        solve_generalized_min_l2_qp(
            torch,
            inequality_rows=torch.eye(9, dtype=torch.float64),
            lower_bounds=torch.ones(9, dtype=torch.float64),
            nuisance_rows=_empty_rows(9),
        )

    bad = basis[:1].clone()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        solve_generalized_min_l2_qp(
            torch,
            inequality_rows=bad,
            lower_bounds=torch.ones(1, dtype=torch.float64),
            nuisance_rows=_empty_rows(3),
        )

    with pytest.raises(RuntimeError, match="no certified feasible active set"):
        solve_generalized_min_l2_qp(
            torch,
            inequality_rows=torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64),
            lower_bounds=torch.ones(2, dtype=torch.float64),
            nuisance_rows=_empty_rows(2),
        )


def test_trust_step_fraction_enforces_trust_radius_and_absolute_cap() -> None:
    origin = torch.zeros(2, dtype=torch.float64)
    proposed = torch.tensor([3.0, 4.0], dtype=torch.float64)
    fraction, report = trust_step_cap_fraction(
        torch,
        current_point=origin,
        proposed_point=proposed,
        trust_radius=2.0,
        absolute_cap=10.0,
    )
    assert fraction == pytest.approx(0.4)
    assert report["realized_step_norm"] == pytest.approx(2.0)
    assert report["trust_limited"] is True

    current = torch.tensor([0.6, 0.0], dtype=torch.float64)
    proposed = torch.tensor([2.6, 0.0], dtype=torch.float64)
    fraction, report = trust_step_cap_fraction(
        torch,
        current_point=current,
        proposed_point=proposed,
        trust_radius=10.0,
        absolute_cap=1.0,
    )
    assert fraction == pytest.approx(0.2)
    assert report["realized_absolute_norm"] == pytest.approx(1.0)
    assert report["absolute_cap_limited"] is True


def test_trust_step_fraction_fails_closed_outside_cap_and_stops_at_boundary() -> None:
    with pytest.raises(RuntimeError, match="already exceeds"):
        trust_step_cap_fraction(
            torch,
            current_point=torch.tensor([1.1, 0.0], dtype=torch.float64),
            proposed_point=torch.tensor([0.0, 0.0], dtype=torch.float64),
            trust_radius=0.1,
            absolute_cap=1.0,
        )

    fraction, report = trust_step_cap_fraction(
        torch,
        current_point=torch.tensor([1.0, 0.0], dtype=torch.float64),
        proposed_point=torch.tensor([2.0, 0.0], dtype=torch.float64),
        trust_radius=0.5,
        absolute_cap=1.0,
    )
    assert fraction == pytest.approx(0.0)
    assert report["zero_usable_fraction"] is True

    fraction, report = trust_step_cap_fraction(
        torch,
        current_point=torch.tensor([0.2, 0.0], dtype=torch.float64),
        proposed_point=torch.tensor([0.2, 0.0], dtype=torch.float64),
        trust_radius=0.5,
        absolute_cap=1.0,
    )
    assert fraction == pytest.approx(1.0)
    assert report["zero_usable_fraction"] is True


def test_merit_and_acceptance_use_measured_not_only_predicted_progress() -> None:
    required = torch.ones(2, dtype=torch.float64)
    current = torch.zeros(2, dtype=torch.float64)
    predicted = torch.full((2,), 0.8, dtype=torch.float64)
    measured = torch.full((2,), 0.7, dtype=torch.float64)

    merit, report = constraint_violation_merit(
        torch,
        values=current,
        required_lower_bounds=required,
    )
    assert merit == pytest.approx(1.0)
    assert report["maximum_violation"] == pytest.approx(1.0)

    accepted = assess_trial_acceptance(
        torch,
        current_values=current,
        predicted_trial_values=predicted,
        measured_trial_values=measured,
        required_lower_bounds=required,
        finite_protection_passed=True,
    )
    assert accepted["accepted"] is True
    assert accepted["actual_to_predicted_reduction_ratio"] == pytest.approx(0.91 / 0.96)

    protection_failure = assess_trial_acceptance(
        torch,
        current_values=current,
        predicted_trial_values=predicted,
        measured_trial_values=measured,
        required_lower_bounds=required,
        finite_protection_passed=False,
    )
    assert protection_failure["accepted"] is False
    assert protection_failure["reason"] == "finite_protection_failed"

    inaccurate_model = assess_trial_acceptance(
        torch,
        current_values=current,
        predicted_trial_values=required,
        measured_trial_values=torch.full((2,), 0.01, dtype=torch.float64),
        required_lower_bounds=required,
        finite_protection_passed=True,
        minimum_acceptance_ratio=0.1,
    )
    assert inaccurate_model["accepted"] is False
    assert inaccurate_model["reason"] == "actual_to_predicted_reduction_ratio_below_threshold"

    one_constraint_regresses = assess_trial_acceptance(
        torch,
        current_values=torch.tensor([0.0, 0.5], dtype=torch.float64),
        predicted_trial_values=torch.tensor([0.2, 0.6], dtype=torch.float64),
        measured_trial_values=torch.tensor([0.1, 0.4], dtype=torch.float64),
        required_lower_bounds=required,
        finite_protection_passed=True,
    )
    assert one_constraint_regresses["actual_reduction"] > 0.0
    assert one_constraint_regresses["accepted"] is False
    assert one_constraint_regresses["reason"] == "an_individual_violation_worsened"


def test_trust_radius_update_is_deterministic_and_fail_closed() -> None:
    assert update_trust_radius(
        current_radius=0.02,
        minimum_radius=0.001,
        maximum_radius=0.08,
        accepted=False,
        actual_to_predicted_ratio=None,
        step_was_trust_limited=True,
    ) == pytest.approx(0.01)
    assert update_trust_radius(
        current_radius=0.02,
        minimum_radius=0.001,
        maximum_radius=0.08,
        accepted=True,
        actual_to_predicted_ratio=0.9,
        step_was_trust_limited=True,
    ) == pytest.approx(0.04)
    assert update_trust_radius(
        current_radius=0.02,
        minimum_radius=0.001,
        maximum_radius=0.08,
        accepted=True,
        actual_to_predicted_ratio=0.5,
        step_was_trust_limited=False,
    ) == pytest.approx(0.02)
    with pytest.raises(ValueError, match="minimum <= current <= maximum"):
        update_trust_radius(
            current_radius=0.1,
            minimum_radius=0.001,
            maximum_radius=0.08,
            accepted=False,
            actual_to_predicted_ratio=None,
            step_was_trust_limited=False,
        )


def _passing_terminal_inputs() -> dict[str, torch.Tensor | float]:
    return {
        "semantic_desired_gaps": torch.tensor([[0.08, 0.09], [0.07, 0.06]], dtype=torch.float64),
        "full_vocabulary_desired_gaps": torch.tensor(
            [[0.07, 0.08], [0.06, 0.09]], dtype=torch.float64
        ),
        "actual_token_ids": torch.tensor([[10, 11], [20, 21]], dtype=torch.int64),
        "baseline_actual_token_ids": torch.tensor([11, 20], dtype=torch.int64),
        "preserve_token_ids": torch.tensor([10, 20], dtype=torch.int64),
        "comply_token_ids": torch.tensor([11, 21], dtype=torch.int64),
        "decision_margin": 0.05,
    }


def test_terminal_gate_requires_both_signs_both_orders_and_real_flips() -> None:
    result = terminal_bidirectional_decision_gate(torch, **_passing_terminal_inputs())

    assert result["passes_terminal_gate"] is True
    assert all(result["gates"].values())
    assert all(order["passed"] for order in result["orders"])
    assert len(result["diagnostics_sha256"]) == 64


def test_terminal_gate_rejects_other_token_despite_forced_pair_margin() -> None:
    inputs = _passing_terminal_inputs()
    inputs["actual_token_ids"] = torch.tensor([[999, 11], [20, 21]], dtype=torch.int64)
    # Keep semantic pair gaps positive to prove that pair-logit movement alone is
    # insufficient. The exact token check must still fail.
    result = terminal_bidirectional_decision_gate(torch, **inputs)

    assert result["passes_terminal_gate"] is False
    assert result["gates"]["plus_is_preserve_in_both_orders"] is False
    assert result["gates"]["semantic_margin_passes_both_signs_and_orders"] is True


def test_terminal_gate_rejects_bad_vocab_margin_or_invalid_baseline() -> None:
    inputs = _passing_terminal_inputs()
    bad_gaps = inputs["full_vocabulary_desired_gaps"].clone()
    bad_gaps[1, 1] = 0.049
    inputs["full_vocabulary_desired_gaps"] = bad_gaps
    result = terminal_bidirectional_decision_gate(torch, **inputs)
    assert result["passes_terminal_gate"] is False
    assert result["gates"]["full_vocabulary_margin_passes_both_signs_and_orders"] is False

    inputs = _passing_terminal_inputs()
    inputs["baseline_actual_token_ids"] = torch.tensor([999, 20], dtype=torch.int64)
    result = terminal_bidirectional_decision_gate(torch, **inputs)
    assert result["passes_terminal_gate"] is False
    assert result["gates"]["both_baselines_valid_a_or_b"] is False
    assert result["gates"]["real_flip_occurs_in_each_order"] is False


def test_terminal_gate_rejects_malformed_axes_or_token_types() -> None:
    inputs = _passing_terminal_inputs()
    inputs["semantic_desired_gaps"] = torch.ones(4, dtype=torch.float64)
    with pytest.raises(ValueError, match="two-dimensional|shape"):
        terminal_bidirectional_decision_gate(torch, **inputs)

    inputs = _passing_terminal_inputs()
    inputs["actual_token_ids"] = torch.ones((2, 2), dtype=torch.float64)
    with pytest.raises(TypeError, match="integer tensor"):
        terminal_bidirectional_decision_gate(torch, **inputs)


def test_public_helpers_reject_nonfinite_scalar_controls() -> None:
    with pytest.raises(ValueError, match="finite"):
        trust_step_cap_fraction(
            torch,
            current_point=torch.zeros(2, dtype=torch.float64),
            proposed_point=torch.ones(2, dtype=torch.float64),
            trust_radius=math.inf,
            absolute_cap=1.0,
        )
