from __future__ import annotations

import numpy as np
import pytest

from sp_lense.counterfactual_tangent_shield import TangentShieldDirection
from sp_lense.decision_margin_shield import (
    METHODS,
    DecisionMarginOptimalityError,
    certify_minimum_l2_candidate,
    decision_margin_bounds,
    screen_scenario_layer,
    select_layer,
)


def _orthogonal_problem(*, protected_offsets: np.ndarray | None = None):
    dimension = 21
    target = np.zeros((4, dimension), dtype=np.float64)
    target[:, 0] = 1.0
    protected = np.zeros((12, dimension), dtype=np.float64)
    for index in range(12):
        protected[index, index + 1] = 1.0
    unrelated = np.zeros((8, dimension), dtype=np.float64)
    for index in range(8):
        unrelated[index, index + 13] = 1.0
    offsets = (
        np.full(12, 0.2, dtype=np.float64)
        if protected_offsets is None
        else np.asarray(protected_offsets, dtype=np.float64)
    )
    return {
        "target_rows": target,
        "target_offsets": np.zeros(4, dtype=np.float64),
        "protected_rows": protected,
        "protected_offsets": offsets,
        "unrelated_rows": unrelated,
    }


def test_decision_margin_bounds_and_exact_boundary_classification() -> None:
    offsets = np.array([0.049, 0.05, -0.05, 0.2, *([0.3] * 8)], dtype=np.float64)
    bounds = decision_margin_bounds(offsets, margin=0.05)
    np.testing.assert_allclose(bounds[:4], [0.0, 0.0, 0.0, 0.15], atol=1e-15, rtol=0.0)

    records = screen_scenario_layer(**_orthogonal_problem(protected_offsets=offsets))
    dms = next(record for record in records if record["method"] == "decision_margin_shield")
    assert dms["small_baseline_first_order_frozen_count"] == 1
    assert dms["protected_margin_certified_row_count"] == 11
    nuisance_bounds = dms["solver_diagnostics"]["certificate"]["nuisance_bounds"]
    np.testing.assert_allclose(nuisance_bounds[:8], np.zeros(8), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(nuisance_bounds[8:], bounds, atol=0.0, rtol=0.0)


def test_all_three_methods_are_uncapped_certified_and_frontier_reported() -> None:
    records = screen_scenario_layer(**_orthogonal_problem())
    assert [record["method"] for record in records] == list(METHODS)
    assert all(record["status"] == "eligible" for record in records)
    assert all(record["minimum_standardized_l2"] == pytest.approx(0.05) for record in records)
    for record in records:
        assert record["solver_diagnostics"]["input_record"]["l2_cap"] is None
        assert record["cap_passes"] == {"1": True, "1.5": True, "2": True}
        assert record["geometry_record_sha256"]
    dms = records[-1]
    assert dms["exact_unrelated_gradient_count"] == 8
    assert dms["protected_gradient_count"] == 12
    assert dms["solver_diagnostics"]["certificate"]["passes"] is True


def test_exact_unrelated_cancellation_can_return_clean_infeasibility() -> None:
    problem = _orthogonal_problem()
    problem["unrelated_rows"][0] = problem["target_rows"][0]
    records = screen_scenario_layer(**problem)
    assert records[0]["status"] == "eligible"
    assert records[1]["status"] == "infeasible"
    assert records[2]["status"] == "infeasible"
    assert records[2]["minimum_standardized_l2"] is None
    assert records[2]["cap_passes"] == {"1": False, "1.5": False, "2": False}


def _selection_record(layer: int, scenario_id: str, norm: float | None):
    record = {
        "layer": layer,
        "scenario_id": scenario_id,
        "partition": "calibration",
        "method": "decision_margin_shield",
        "status": "eligible" if norm is not None else "infeasible",
        "minimum_standardized_l2": norm,
    }
    if norm is not None:
        record["optimality_certificate"] = {"passes": True}
        record["cap_certificates"] = {}
        for cap in (1.0, 1.5, 2.0):
            passes = norm <= cap
            record["cap_certificates"][format(cap, ".15g")] = {
                "status": (
                    "feasible_primal_witness"
                    if passes
                    else "infeasible_dual_lower_bound"
                ),
                "feasible_witness": passes,
                "dual_infeasibility_certificate": not passes,
            }
    return record


def test_independent_certificate_recovers_known_analytic_optimum() -> None:
    certificate = certify_minimum_l2_candidate(
        np.array([1.0, 0.0]),
        np.array([[1.0, 0.0]]),
        np.array([0.0]),
        margin=1.0,
    )
    assert certificate["passes"] is True
    assert certificate["candidate_l2_norm"] == pytest.approx(1.0)
    assert certificate["primal_objective"] == pytest.approx(0.5)
    assert certificate["validated_dual_objective_lower_bound"] == pytest.approx(
        0.5, abs=1e-12
    )
    assert certificate["minimum_l2_lower_bound"] == pytest.approx(1.0, abs=1e-12)
    assert certificate["validated_primal_dual_gap"] < 1e-12
    assert certificate["stationarity_l2_residual"] < 1e-12


def test_certificate_handles_redundant_soft_and_exact_constraints() -> None:
    certificate = certify_minimum_l2_candidate(
        np.array([1.0, 0.0, 0.0]),
        np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.zeros(2),
        margin=1.0,
        nuisance_rows=np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        nuisance_bound=np.array([0.0, 0.0, 0.0, 0.25]),
    )
    assert certificate["passes"] is True
    assert certificate["exact_equality_row_count"] == 3
    assert certificate["soft_slab_row_count"] == 1
    assert certificate["svd"]["rank"] == 1
    assert certificate["svd"]["zero_row_count"] == 1
    assert certificate["minimum_l2_lower_bound"] == pytest.approx(1.0, abs=1e-12)


def test_feasible_but_nonoptimal_candidate_is_rejected() -> None:
    certificate = certify_minimum_l2_candidate(
        np.array([2.0, 0.0]),
        np.array([[1.0, 0.0]]),
        np.array([0.0]),
        margin=1.0,
    )
    assert certificate["minimum_inequality_slack"] == pytest.approx(1.0)
    assert certificate["passes"] is False
    assert certificate["checks"]["primal_inequalities"] is True
    assert certificate["checks"]["primal_dual_gap"] is False
    assert certificate["checks"]["stationarity"] is False


def test_screen_rejects_a_mocked_nonoptimal_solver_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = TangentShieldDirection(
        direction=np.array([0.1, *([0.0] * 20)], dtype=np.float64),
        diagnostics={
            "certificate": {"l2_norm": 0.1},
            "direction_sha256": "mocked-nonoptimal",
        },
    )
    monkeypatch.setattr(
        "sp_lense.decision_margin_shield.solve_minimum_l2_direction",
        lambda *args, **kwargs: fake,
    )
    with pytest.raises(DecisionMarginOptimalityError, match="optimality"):
        screen_scenario_layer(**_orthogonal_problem())


def test_norm_above_cap_has_a_dual_lower_bound_infeasibility_certificate() -> None:
    problem = _orthogonal_problem()
    problem["target_rows"][:, 0] = 0.02
    records = screen_scenario_layer(**problem)
    for record in records:
        assert record["minimum_standardized_l2"] == pytest.approx(2.5, abs=1e-8)
        assert record["certified_minimum_standardized_l2_lower_bound"] > 2.0
        cap = record["cap_certificates"]["2"]
        assert cap["status"] == "infeasible_dual_lower_bound"
        assert cap["dual_infeasibility_certificate"] is True
        assert record["cap_passes"]["2"] is False


def test_layer_selection_uses_worst_then_mean_then_layer() -> None:
    scenario_ids = tuple(f"c{index}" for index in range(4))
    norms = {
        0: (1.0, 1.0, 1.0, 2.1),
        1: (0.8, 0.8, 0.8, 1.2),
        2: (1.1, 1.1, 1.1, 1.1),
        3: (1.1, 1.1, 1.1, 1.1),
    }
    records = [
        _selection_record(layer, scenario_id, norms[layer][scenario_index])
        for layer in range(4)
        for scenario_index, scenario_id in enumerate(scenario_ids)
    ]
    result = select_layer(
        records,
        calibration_scenario_ids=scenario_ids,
        layers=range(4),
    )
    assert result["status"] == "selected"
    assert result["qualifying_layer_count"] == 3
    assert result["selected_layer"] == 2
    assert result["selected_layer_summary"]["worst_case_minimum_standardized_l2"] == 1.1


def test_zero_eligible_layers_is_a_valid_no_go() -> None:
    scenario_ids = tuple(f"c{index}" for index in range(4))
    records = [
        _selection_record(layer, scenario_id, None)
        for layer in range(3)
        for scenario_id in scenario_ids
    ]
    result = select_layer(
        records,
        calibration_scenario_ids=scenario_ids,
        layers=range(3),
    )
    assert result["status"] == "no_qualifying_layer"
    assert result["selected_layer"] is None
    assert result["qualifying_layer_count"] == 0
    assert all(summary["eligible_scenario_count"] == 0 for summary in result["layer_summaries"])


def test_invalid_geometry_inputs_and_incomplete_selection_fail_closed() -> None:
    problem = _orthogonal_problem()
    with pytest.raises(ValueError, match="exactly 4"):
        screen_scenario_layer(**{**problem, "target_rows": problem["target_rows"][:3]})
    with pytest.raises(ValueError, match="equal width"):
        screen_scenario_layer(**{**problem, "unrelated_rows": np.zeros((8, 22))})
    with pytest.raises(ValueError, match="cover"):
        select_layer(
            [_selection_record(0, "c0", 1.0)],
            calibration_scenario_ids=("c0", "c1"),
            layers=(0,),
        )
