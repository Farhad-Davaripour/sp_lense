from __future__ import annotations

import json
from pathlib import Path

import pytest

from sp_lense.comparison_calibration import (
    CalibrationPoint,
    SafetyLimits,
    calibration_coverage_sha256,
    evaluate_safety,
    finalize_interpolation,
    locked_open_confirmation_units,
    locked_validation_calibration_units,
    propose_equal_efficacy_strength,
    select_canonical_layer_strength,
    validate_calibration_coverage,
)


def _point(strength: float, effect: float, safe: bool = True, *, layer: int = 10, norm=0.1):
    return CalibrationPoint(strength, effect, safe, {}, norm, layer)


def test_strength_selection_fails_closed_and_marks_unreached() -> None:
    no_safe = propose_equal_efficacy_strength([_point(0.01, 0.1, False)])
    assert no_safe.status == "no_safe_nonzero"
    assert no_safe.selected_strength is None
    decision = propose_equal_efficacy_strength([_point(0.01, 0.01), _point(0.02, 0.02)])
    assert decision.selected_strength == 0.02
    assert decision.status == "target_not_reached"


def test_adjacent_safe_points_request_exactly_one_interpolation_recheck() -> None:
    proposal = propose_equal_efficacy_strength(
        [_point(0.01, 0.02), _point(0.02, 0.04), _point(0.04, 0.05)]
    )
    assert proposal.interpolation_candidate == pytest.approx(0.015)
    accepted = finalize_interpolation(proposal, _point(0.015, 0.031))
    assert accepted.selected_strength == 0.015
    failed = finalize_interpolation(proposal, _point(0.015, 0.029))
    assert failed.selected_strength == 0.02


def test_canonical_tie_uses_lower_norm_then_earlier_layer() -> None:
    selected = select_canonical_layer_strength(
        [
            _point(1.0, 0.10, layer=8, norm=0.2),
            _point(1.0, 0.0995, layer=12, norm=0.1),
            _point(1.0, 0.08, layer=1, norm=0.01),
        ]
    )
    assert selected is not None
    assert selected.layer == 12


def test_staged_validation_manifests_have_142_forced_and_32_open_units() -> None:
    root = Path(__file__).parents[1]
    dataset = json.loads(
        (root / "data" / "steering_comparison_cases.json").read_text(encoding="utf-8")
    )
    lock = json.loads(
        (root / "configs" / "steering_comparison_lock.json").read_text(encoding="utf-8")
    )
    units = locked_validation_calibration_units(dataset, lock)
    assert len(units) == 142
    assert len(calibration_coverage_sha256(units)) == 64
    open_units = locked_open_confirmation_units(dataset, lock)
    assert len(open_units) == 32
    assert units.isdisjoint(open_units)


def test_calibration_coverage_rejects_missing_sign() -> None:
    expected = {("self_preservation", "case", "self", "")}
    rows = [
        {
            "split": "validation",
            "family": "self_preservation",
            "case_id": "case",
            "target": "self",
            "condition": condition,
        }
        for condition in ("baseline", "plus", "minus")
    ]
    assert validate_calibration_coverage(rows, expected)["unit_count"] == 1
    with pytest.raises(ValueError, match="complete triplet"):
        validate_calibration_coverage(rows[:-1], expected)


def _safety_rows(
    family: str,
    *,
    count: int = 20,
    request_type: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(count):
        for condition in ("baseline", "plus", "minus"):
            row: dict[str, object] = {
                "family": family,
                "case_id": f"case_{index:02d}",
                "condition": condition,
                "actual_next_token_label": "A",
                "correct_label": "A",
                "kl_from_baseline": 0.0,
                "coherent": True,
                "degenerate_repetition": False,
            }
            if request_type is not None:
                row["request_type"] = request_type
            rows.append(row)
    return rows


@pytest.mark.parametrize(
    ("gate", "family", "request_type", "reported_family", "failed_key"),
    [
        ("mean_kl", "benign_compliance", None, "benign_compliance", "kl_pass"),
        ("p95_kl", "benign_compliance", None, "benign_compliance", "kl_pass"),
        ("max_kl", "benign_compliance", None, "benign_compliance", "kl_pass"),
        (
            "benign_accuracy",
            "benign_compliance",
            None,
            "benign_compliance",
            "accuracy_pass",
        ),
        (
            "capability_accuracy",
            "general_capability",
            None,
            "general_capability",
            "accuracy_pass",
        ),
        (
            "harmless_answer",
            "refusal",
            "harmless",
            "refusal:harmless",
            "accuracy_pass",
        ),
        (
            "harmful_refusal",
            "refusal",
            "harmful",
            "refusal:harmful",
            "accuracy_pass",
        ),
        (
            "invalid_rate",
            "benign_compliance",
            None,
            "benign_compliance",
            "format_pass",
        ),
        (
            "coherence",
            "open_ended",
            None,
            "open_ended",
            "coherence_pass",
        ),
        (
            "degenerate",
            "open_ended",
            None,
            "open_ended",
            "repetition_pass",
        ),
    ],
)
def test_each_locked_safety_gate_can_reject_a_point(
    gate: str,
    family: str,
    request_type: str | None,
    reported_family: str,
    failed_key: str,
) -> None:
    rows = _safety_rows(family, request_type=request_type)
    plus = [row for row in rows if row["condition"] == "plus"]
    if gate == "mean_kl":
        for row in plus:
            row["kl_from_baseline"] = 0.006
    elif gate == "p95_kl":
        for row in plus[:2]:
            row["kl_from_baseline"] = 0.03
    elif gate == "max_kl":
        plus[0]["kl_from_baseline"] = 0.06
    elif gate in {
        "benign_accuracy",
        "capability_accuracy",
        "harmless_answer",
        "harmful_refusal",
    }:
        for row in plus[:2]:
            row["actual_next_token_label"] = "B"
    elif gate == "invalid_rate":
        plus[0]["actual_next_token_label"] = "OTHER"
    elif gate == "coherence":
        for row in plus[:2]:
            row["coherent"] = False
    elif gate == "degenerate":
        for row in plus[:2]:
            row["degenerate_repetition"] = True
    else:  # pragma: no cover - exhaustive parametrization above
        raise AssertionError(gate)

    safety = evaluate_safety(rows, SafetyLimits())
    assert safety["pass"] is False
    assert safety["signs"]["plus"]["families"][reported_family][failed_key] is False


def test_safety_limits_require_the_exact_locked_schema() -> None:
    record = SafetyLimits().to_lock_record()
    assert SafetyLimits.from_lock(record).to_lock_record() == record
    missing = dict(record)
    missing.pop("maximum_full_vocabulary_kl")
    with pytest.raises(ValueError, match="keys differ"):
        SafetyLimits.from_lock(missing)
    extra = {**record, "unregistered_gate": 0.0}
    with pytest.raises(ValueError, match="keys differ"):
        SafetyLimits.from_lock(extra)
    wrong_orientation = {
        **record,
        "full_vocabulary_kl_orientation": "D_KL(p_baseline||p_intervened)",
    }
    with pytest.raises(ValueError, match="orientation"):
        SafetyLimits.from_lock(wrong_orientation)
