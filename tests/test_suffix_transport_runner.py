from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "suffix_transport_feasibility.py"


def _load_runner():
    specification = importlib.util.spec_from_file_location(
        "suffix_transport_runner_tests", RUNNER_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not import runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def test_proposed_lock_is_development_only_and_exactly_metered() -> None:
    lock = runner.proposed_lock()
    assert lock["development_only"] is True
    assert lock["data"]["partition"] == "calibration"
    assert lock["data"]["sealed_or_fcags_pilot_outcomes_read"] is False
    assert lock["choice_target"]["layer"] == 22
    assert lock["compute_ceiling"] == {
        "choice_capture": {"forward": 16, "backward": 16},
        "generated_tokens": 0,
        "external_api_calls": 0,
        "external_model_judges": 0,
        "paid_cost_usd": 0,
    }
    assert lock["success_gates"]["minimum_both_order_positive_assignment_units"] == 6
    assert lock["success_gates"]["minimum_scenarios_with_both_assignments"] == 3


def test_capture_plan_has_four_calibration_scenarios_two_assignments_two_orders() -> None:
    dataset = runner._load_dataset()
    scenarios = runner._calibration_scenarios(dataset)
    plan = runner._capture_plan(dataset, scenarios)
    assert len(plan) == 8
    assert {unit["assignment"] for unit in plan} == {0, 1}
    assert len({unit["scenario_id"] for unit in plan}) == 4
    forms = [choice for unit in plan for choice in unit["choices"]]
    assert len(forms) == 16
    assert len({form["form_id"] for form in forms}) == 16
    assert sum(bool(form["preserve_first"]) for form in forms) == 8
    assert all({form["preserve_label"], form["comply_label"]} == {"A", "B"} for form in forms)


def test_meter_rejects_duplicates_and_enforces_each_ceiling() -> None:
    meter = runner.Meter(phase="test", ceiling={"forward": 1, "backward": 1})
    meter.reserve_forward("view")
    meter.reserve_backward("view")
    with pytest.raises(RuntimeError, match="duplicate forward"):
        meter.reserve_forward("view")
    with pytest.raises(RuntimeError, match="duplicate backward"):
        meter.reserve_backward("view")
    with pytest.raises(RuntimeError, match="forward ceiling"):
        meter.reserve_forward("second")
    with pytest.raises(RuntimeError, match="backward ceiling"):
        meter.reserve_backward("second")


def test_static_mean_predictions_exclude_the_complete_held_out_scenario() -> None:
    head_0 = np.asarray([[1.0, 0.0], [0.9, 0.1]] * 4)
    head_1 = np.asarray([[0.8, 0.2], [1.0, 0.0]] * 4)
    scenario_ids = ["a", "a", "b", "b", "c", "c", "d", "d"]
    predicted_0, predicted_1, folds = runner._static_mean_predictions(
        head_0, head_1, scenario_ids
    )
    assert predicted_0.shape == head_0.shape
    assert predicted_1.shape == head_1.shape
    assert len(folds) == 4
    for fold in folds:
        assert len(fold["held_out_indices"]) == 2
        assert len(fold["training_indices"]) == 6
        assert set(fold["held_out_indices"]).isdisjoint(fold["training_indices"])
        assert all(
            scenario_ids[index] != fold["held_out_scenario"]
            for index in fold["training_indices"]
        )


def _metric(count: int, scenarios: int, median: float) -> dict[str, object]:
    return {
        "available": True,
        "both_order_positive_count": count,
        "complete_scenario_count": scenarios,
        "worst_order_alignment": {"median": median},
    }


def _compute() -> dict[str, int]:
    return {
        "forward_evaluations": 16,
        "backward_evaluations": 16,
        "unique_forward_work_ids": 16,
        "unique_backward_work_ids": 16,
    }


def test_locked_gates_require_strict_median_and_two_unit_identity_advantage() -> None:
    gates, passes = runner._apply_gates(_metric(6, 3, 0.100001), _metric(4, 2, 0.2), _compute())
    assert passes is True
    assert all(gates.values())

    gates, passes = runner._apply_gates(_metric(6, 3, 0.10), _metric(4, 2, 0.2), _compute())
    assert passes is False
    assert gates["median_worst_order_cosine_strictly_greater_than_0_10"] is False

    gates, passes = runner._apply_gates(_metric(6, 3, 0.2), _metric(5, 2, 0.2), _compute())
    assert passes is False
    assert gates["at_least_two_more_assignment_units_than_identity"] is False


def test_unavailable_transport_fails_closed() -> None:
    unavailable = {"available": False, "failure": "incompatible heads"}
    gates, passes = runner._apply_gates(unavailable, _metric(0, 0, 0.0), _compute())
    assert passes is False
    assert gates["transport_heads_compatible"] is False
    assert gates["hash_and_anchor_audits_pass"] is True
