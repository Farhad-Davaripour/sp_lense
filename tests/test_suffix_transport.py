from __future__ import annotations

import math

import numpy as np
import pytest

from sp_lense import suffix_transport
from sp_lense.suffix_transport import (
    CELL_ORDER,
    SuffixTransportIneligible,
    construct_cell_interface_directions,
    exact_nuisance_projection,
    fit_dual_ridge_transport,
    leave_one_scenario_out_cell_interface_translation,
    leave_one_scenario_out_transport,
    order_even_odd_directions,
    predict_dual_ridge_transport,
    robust_two_head_unit_bisector,
    transport_metric_summary,
    unit_normalize_rows,
    validate_matrix,
)


def test_validate_and_unit_normalize_rows_are_strict_and_scale_invariant() -> None:
    values = np.asarray([[3.0, 4.0], [-10.0, 0.0]])
    normalized = unit_normalize_rows(values, field="example")
    scaled = unit_normalize_rows(7.0 * values, field="scaled")
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), np.ones(2))
    np.testing.assert_allclose(normalized, scaled)
    assert normalized.dtype == np.float64 and normalized.flags.c_contiguous
    np.testing.assert_array_equal(values, np.asarray([[3.0, 4.0], [-10.0, 0.0]]))

    with pytest.raises(ValueError, match="two-dimensional"):
        validate_matrix([1.0, 2.0], field="bad")
    with pytest.raises(ValueError, match="finite"):
        validate_matrix([[1.0, np.nan]], field="bad")
    with pytest.raises(ValueError, match="positive finite norm"):
        unit_normalize_rows([[0.0, 0.0]], field="bad")


def test_dual_ridge_uses_fixed_trace_over_rank_rule() -> None:
    source = np.eye(2, dtype=np.float64)
    head_0 = np.asarray([[1.0, 1.0], [-1.0, 1.0]])
    head_1 = np.asarray([[2.0, 2.0], [-3.0, 3.0]])
    model = fit_dual_ridge_transport(source, head_0, head_1, ridge_multiplier=0.1)

    assert model.source_rank == 2
    assert model.kernel_trace == pytest.approx(2.0)
    assert model.ridge == pytest.approx(0.1)
    predicted_0, predicted_1 = predict_dual_ridge_transport(model, 9.0 * source)
    expected = unit_normalize_rows(head_0) / 1.1
    np.testing.assert_allclose(predicted_0, expected)
    np.testing.assert_allclose(predicted_1, expected)
    assert model.diagnostics["ridge_rule"].startswith("multiplier_times")


def test_dual_ridge_rank_deficiency_changes_trace_over_rank_scale() -> None:
    source = np.asarray([[1.0, 0.0], [4.0, 0.0]])
    targets = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    model = fit_dual_ridge_transport(source, targets, targets, ridge_multiplier=0.2)
    assert model.source_rank == 1
    assert model.kernel_trace == pytest.approx(2.0)
    assert model.ridge == pytest.approx(0.4)

    with pytest.raises(TypeError, match="ridge_multiplier"):
        fit_dual_ridge_transport(source, targets, targets, ridge_multiplier=True)
    with pytest.raises(ValueError, match="same row count"):
        fit_dual_ridge_transport(source, targets[:1], targets, ridge_multiplier=0.1)


def test_robust_two_head_bisector_is_unit_and_maximin() -> None:
    result = robust_two_head_unit_bisector(
        [[1.0, 0.0], [1.0, 1.0]],
        [[0.0, 1.0], [2.0, 2.0]],
        minimum_head_cosine=0.0,
    )
    expected = 1.0 / math.sqrt(2.0)
    np.testing.assert_allclose(result["directions"][0], [expected, expected])
    np.testing.assert_allclose(np.linalg.norm(result["directions"], axis=1), [1.0, 1.0])
    assert result["head_0_alignments"][0] == pytest.approx(expected)
    assert result["head_1_alignments"][0] == pytest.approx(expected)
    assert result["worst_head_alignments"][1] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="minimum cosine"):
        robust_two_head_unit_bisector(
            [[1.0, 0.0]],
            [[-1.0, 0.0]],
            minimum_head_cosine=0.0,
        )


def _loso_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    source = np.asarray([[1.0, 0.0], [0.0, 1.0]] * 3)
    head_0 = np.asarray([[1.0, 0.1], [0.1, 1.0]] * 3)
    head_1 = np.asarray([[1.0, 0.2], [0.2, 1.0]] * 3)
    return source, head_0, head_1, ["scenario_a"] * 2 + ["scenario_b"] * 2 + ["scenario_c"] * 2


def test_loso_excludes_complete_scenarios_and_preserves_row_order() -> None:
    source, head_0, head_1, scenario_ids = _loso_fixture()
    result = leave_one_scenario_out_transport(
        source,
        head_0,
        head_1,
        scenario_ids,
        ridge_multiplier=0.1,
        minimum_head_cosine=0.0,
    )

    assert result["predicted_head_0_rows"].shape == head_0.shape
    assert result["predicted_head_1_rows"].shape == head_1.shape
    assert result["metrics"]["both_order_positive_count"] == 6
    assert result["metrics"]["complete_scenario_count"] == 3
    for fold in result["folds"]:
        assert fold["held_out_scenario"] not in fold["training_scenarios"]
        assert len(fold["held_out_indices"]) == 2
        assert len(fold["training_indices"]) == 4
        assert set(fold["held_out_indices"]).isdisjoint(fold["training_indices"])


def test_loso_rejects_invalid_scenario_partitions() -> None:
    source, head_0, head_1, _ = _loso_fixture()
    with pytest.raises(ValueError, match="at least two scenarios"):
        leave_one_scenario_out_transport(
            source,
            head_0,
            head_1,
            ["only_one"] * len(source),
        )
    with pytest.raises(ValueError, match="match the matrix row count"):
        leave_one_scenario_out_transport(source, head_0, head_1, ["a", "b"])


def test_metric_summary_counts_rows_and_complete_scenarios() -> None:
    predicted_0 = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    predicted_1 = predicted_0.copy()
    observed_0 = predicted_0.copy()
    observed_1 = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    summary = transport_metric_summary(
        predicted_0,
        predicted_1,
        observed_0,
        observed_1,
        scenario_ids=["scenario_a", "scenario_a", "scenario_b"],
    )
    assert summary["both_order_positive_count"] == 2
    assert summary["both_order_positive_fraction"] == pytest.approx(2.0 / 3.0)
    assert summary["complete_scenario_count"] == 1
    assert summary["scenario_rows"] == [
        {
            "scenario_id": "scenario_a",
            "row_count": 2,
            "both_order_positive_count": 1,
            "complete": False,
            "minimum_worst_order_alignment": -1.0,
        },
        {
            "scenario_id": "scenario_b",
            "row_count": 1,
            "both_order_positive_count": 1,
            "complete": True,
            "minimum_worst_order_alignment": 1.0,
        },
    ]


def test_order_even_odd_decomposition_uses_unit_heads() -> None:
    result = order_even_odd_directions(
        [[10.0, 0.0]],
        [[0.0, 4.0]],
        minimum_head_cosine=0.0,
    )
    expected = 1.0 / math.sqrt(2.0)
    np.testing.assert_allclose(result["even_directions"], [[expected, expected]])
    np.testing.assert_allclose(result["odd_nuisance_rows"], [[expected, -expected]])
    np.testing.assert_allclose(result["odd_nuisance_norms"], [expected])


def test_order_even_odd_allows_observed_negative_cosine_but_rejects_near_antipodal() -> None:
    cosine = -0.8
    second = [cosine, math.sqrt(1.0 - cosine**2)]
    result = order_even_odd_directions([[1.0, 0.0]], [second])
    assert result["head_cosines"][0] == pytest.approx(cosine)
    assert result["minimum_head_cosine"] == pytest.approx(-0.99)
    assert np.dot(result["even_directions"][0], [1.0, 0.0]) > 0.0
    assert np.dot(result["even_directions"][0], second) > 0.0

    with pytest.raises(ValueError, match="minimum cosine"):
        order_even_odd_directions(
            [[1.0, 0.0]],
            [[-0.995, math.sqrt(1.0 - 0.995**2)]],
        )


def test_exact_nuisance_projection_is_orthogonal_and_fails_on_vanishing_target() -> None:
    result = exact_nuisance_projection(
        [1.0, 1.0, 1.0],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        minimum_retained_fraction=0.05,
    )
    np.testing.assert_allclose(result["direction"], [0.0, 0.0, 1.0], atol=1e-12)
    assert result["retained_target_fraction"] == pytest.approx(1.0 / math.sqrt(3.0))
    assert result["maximum_abs_nuisance_projection"] <= 1e-12

    with pytest.raises(SuffixTransportIneligible, match="retained less") as exc_info:
        exact_nuisance_projection(
            [1.0, 0.0],
            [[1.0, 0.0]],
            minimum_retained_fraction=0.05,
        )
    assert exc_info.value.diagnostics["retained_target_fraction"] == pytest.approx(0.0)
    assert exc_info.value.diagnostics["nuisance_rank"] == 1


def _cell_interface_fixture(
    scenario_count: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    width = 9
    semantic = np.zeros((scenario_count, 2, len(CELL_ORDER), width), dtype=np.float64)
    choice_0 = np.zeros_like(semantic)
    choice_1 = np.zeros_like(semantic)
    odd_scale = 0.2
    for scenario in range(scenario_count):
        for assignment in range(2):
            for cell in range(len(CELL_ORDER)):
                coordinate = assignment * len(CELL_ORDER) + cell
                semantic[scenario, assignment, cell, coordinate] = 1.0
                choice_0[scenario, assignment, cell, coordinate] = 1.0
                choice_0[scenario, assignment, cell, 8] = odd_scale
                choice_1[scenario, assignment, cell, coordinate] = 1.0
                choice_1[scenario, assignment, cell, 8] = -odd_scale
    scenario_ids = [f"scenario_{index}" for index in range(scenario_count)]
    return semantic, choice_0, choice_1, scenario_ids


def test_constructed_cell_interface_direction_nulls_all_locked_nuisances() -> None:
    _, choice_0, choice_1, _ = _cell_interface_fixture()
    decomposition = order_even_odd_directions(
        choice_0.reshape(-1, 9),
        choice_1.reshape(-1, 9),
    )
    predicted_even = decomposition["even_directions"].reshape(choice_0.shape)[0]
    predicted_odd = decomposition["odd_nuisance_rows"].reshape(choice_0.shape)[0]
    constructed = construct_cell_interface_directions(predicted_even, predicted_odd)
    direction = constructed["protected_dynamic"]

    assert np.linalg.norm(direction) == pytest.approx(1.0)
    np.testing.assert_allclose(constructed["nuisance_rows"] @ direction, 0.0, atol=1e-12)
    assert constructed["diagnostics"]["retained_target_fraction"] == pytest.approx(1.0)
    assert constructed["diagnostics"]["nuisance_row_count"] == 15
    assert {entry["kind"] for entry in constructed["nuisance_manifest"]} == {
        "predicted_off_target_even_cell",
        "predicted_order_odd_cell",
        "predicted_self_permanent_even_name_odd",
    }


def test_cell_interface_loso_is_leakage_safe_and_reports_equal_access_baselines() -> None:
    semantic, choice_0, choice_1, scenario_ids = _cell_interface_fixture()
    result = leave_one_scenario_out_cell_interface_translation(
        semantic,
        choice_0,
        choice_1,
        scenario_ids,
        ridge_multiplier=0.1,
        minimum_head_cosine=0.0,
        minimum_retained_fraction=0.05,
    )

    assert result["cell_order"] == list(CELL_ORDER)
    assert set(result["directions"]) == {
        "protected_dynamic",
        "unprotected_dynamic",
        "predicted_factorial_dynamic",
        "static_training_protected",
        "factorial_semantic_identity",
    }
    assert set(result["method_summaries"]) == set(result["directions"])
    for fold in result["folds"]:
        assert fold["held_out_scenario"] not in fold["training_scenarios"]
        assert fold["training_cell_row_count"] == 16
        assert fold["construction_diagnostics"]["retained_target_fraction"] >= 0.05
        assert fold["construction_diagnostics"]["maximum_abs_nuisance_projection"] <= 1e-10
        assert fold["construction_diagnostics"]["nuisance_row_count"] == 15
        assert fold["static_training_protection_diagnostics"]["nuisance_row_count"] == 15
        assert fold["static_training_protection_diagnostics"]["training_only"] is True

    protected = result["method_summaries"]["protected_dynamic"]
    assert protected["both_order_positive_assignment_count"] == 6
    assert protected["complete_scenario_count"] == 3
    assert protected["maximum_off_target_absolute_sensitivity_ratio"]["maximum"] <= 1e-10
    assert protected["protection"]["applied"] is True
    assert result["method_summaries"]["static_training_protected"]["protection"][
        "training_only"
    ] is True
    assert result["method_summaries"]["unprotected_dynamic"]["protection"] == {"applied": False}
    assert result["method_summaries"]["predicted_factorial_dynamic"]["protection"] == {
        "applied": False
    }
    assert (
        result["method_summaries"]["predicted_factorial_dynamic"]
        ["maximum_off_target_absolute_sensitivity_ratio"]["minimum"]
        > 0.0
    )
    assert (
        result["method_summaries"]["factorial_semantic_identity"]
        ["maximum_off_target_absolute_sensitivity_ratio"]["minimum"]
        > 0.0
    )


def test_cell_interface_oracle_is_explicitly_evaluation_only() -> None:
    semantic, choice_0, choice_1, scenario_ids = _cell_interface_fixture()
    result = leave_one_scenario_out_cell_interface_translation(
        semantic,
        choice_0,
        choice_1,
        scenario_ids,
        include_heldout_oracle=True,
    )
    assert "oracle_upper_bound" in result["directions"]
    oracle = result["method_summaries"]["oracle_upper_bound"]["protection"]
    assert oracle["uses_heldout_observed_choice_gradients"] is True
    assert oracle["evaluation_only_upper_bound"] is True


def _forced_ineligible(label: str) -> SuffixTransportIneligible:
    return SuffixTransportIneligible(
        f"forced {label} failure",
        diagnostics={
            "retained_target_fraction": 0.0,
            "minimum_retained_fraction": 0.05,
            "nuisance_rank": 8,
            "nuisance_row_count": 15,
        },
    )


def test_oracle_failure_is_isolated_and_has_no_placeholder_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic, choice_0, choice_1, scenario_ids = _cell_interface_fixture()
    original = suffix_transport.construct_cell_interface_directions
    call_count = 0

    def fail_first_oracle(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        if call_count == 3:  # dynamic, static, then optional oracle for scenario 0
            raise _forced_ineligible("oracle")
        return original(*args, **kwargs)

    monkeypatch.setattr(suffix_transport, "construct_cell_interface_directions", fail_first_oracle)
    result = suffix_transport.leave_one_scenario_out_cell_interface_translation(
        semantic,
        choice_0,
        choice_1,
        scenario_ids,
        include_heldout_oracle=True,
    )

    oracle_summary = result["method_summaries"]["oracle_upper_bound"]
    assert oracle_summary["available_scenario_count"] == 2
    assert oracle_summary["unavailable_scenario_count"] == 1
    assert oracle_summary["unavailable_scenarios"][0]["reason"] == "projection_ineligible"
    oracle_directions = result["directions"]["oracle_upper_bound"]
    assert oracle_directions["available_scenario_ids"] == scenario_ids[1:]
    assert oracle_directions["rows"].shape == (2, 9)
    np.testing.assert_allclose(np.linalg.norm(oracle_directions["rows"], axis=1), 1.0)
    assert result["method_summaries"]["protected_dynamic"]["available_scenario_count"] == 3
    assert result["method_summaries"]["static_training_protected"][
        "available_scenario_count"
    ] == 3


def test_static_failure_is_isolated_while_primary_remains_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic, choice_0, choice_1, scenario_ids = _cell_interface_fixture()
    original = suffix_transport.construct_cell_interface_directions
    call_count = 0

    def fail_first_static(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        if call_count == 2:  # dynamic, then static for scenario 0
            raise _forced_ineligible("static")
        return original(*args, **kwargs)

    monkeypatch.setattr(suffix_transport, "construct_cell_interface_directions", fail_first_static)
    result = suffix_transport.leave_one_scenario_out_cell_interface_translation(
        semantic,
        choice_0,
        choice_1,
        scenario_ids,
    )

    static_summary = result["method_summaries"]["static_training_protected"]
    assert static_summary["scenario_count"] == 3
    assert static_summary["available_scenario_count"] == 2
    assert static_summary["unavailable_scenario_count"] == 1
    assert static_summary["both_order_positive_assignment_fraction"] == pytest.approx(4.0 / 6.0)
    static_directions = result["directions"]["static_training_protected"]
    assert static_directions["available_scenario_ids"] == scenario_ids[1:]
    assert static_directions["rows"].shape == (2, 9)
    primary = result["method_summaries"]["protected_dynamic"]
    assert primary["available_scenario_count"] == 3
    assert primary["complete_scenario_count"] == 3


def test_primary_projection_failure_still_fails_the_whole_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic, choice_0, choice_1, scenario_ids = _cell_interface_fixture()
    original = suffix_transport.construct_cell_interface_directions
    call_count = 0

    def fail_primary(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _forced_ineligible("primary")
        return original(*args, **kwargs)

    monkeypatch.setattr(suffix_transport, "construct_cell_interface_directions", fail_primary)
    with pytest.raises(SuffixTransportIneligible, match="forced primary"):
        suffix_transport.leave_one_scenario_out_cell_interface_translation(
            semantic,
            choice_0,
            choice_1,
            scenario_ids,
            include_heldout_oracle=True,
        )


def test_cell_interface_loso_rejects_duplicate_scenario_identity() -> None:
    semantic, choice_0, choice_1, _ = _cell_interface_fixture()
    with pytest.raises(ValueError, match="unique"):
        leave_one_scenario_out_cell_interface_translation(
            semantic,
            choice_0,
            choice_1,
            ["same", "same", "different"],
        )
