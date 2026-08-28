from __future__ import annotations

import math

import numpy as np
import pytest

from sp_lense.suffix_transport import (
    fit_dual_ridge_transport,
    leave_one_scenario_out_transport,
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
