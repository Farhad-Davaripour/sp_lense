from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "gradient_specificity_v3_absolute_dose_probe.py"
SPEC = importlib.util.spec_from_file_location(
    "gradient_specificity_v3_absolute_dose_probe", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


DOSES = [0.02, 0.05, 0.1, 0.15]


def _score_row(
    unit_id: str,
    *,
    condition: str,
    dose: float,
    target: str,
    direction_key: str,
    preserve_first: bool,
    semantic: str,
    token_id: int,
    log_odds: float,
    kl: float,
) -> dict[str, Any]:
    return {
        "unit_id": unit_id,
        "condition": condition,
        "absolute_residual_relative_dose": dose,
        "target": target,
        "direction_key": direction_key,
        "preserve_first": preserve_first,
        "actual_next_token_semantic_choice": semantic,
        "actual_next_token_token_id": token_id,
        "semantic_positive_log_odds": log_odds,
        "correct": semantic == "positive",
        "full_vocabulary_kl_changed_to_baseline": kl,
    }


def _unit_rows(
    unit_id: str,
    *,
    target: str,
    direction_key: str,
    preserve_first: bool,
    break_at_largest_dose: bool = False,
) -> list[dict[str, Any]]:
    baseline_semantic = "negative" if target == "self" else "positive"
    baseline_token = 2 if baseline_semantic == "negative" else 1
    rows = [
        _score_row(
            unit_id,
            condition="baseline",
            dose=0.0,
            target=target,
            direction_key=direction_key,
            preserve_first=preserve_first,
            semantic=baseline_semantic,
            token_id=baseline_token,
            log_odds=-0.1 if target == "self" else 0.1,
            kl=0.0,
        )
    ]
    for dose in DOSES:
        for condition in ("plus", "minus"):
            if target == "self":
                semantic = "positive" if condition == "plus" else "negative"
                token = 1 if semantic == "positive" else 2
                log_odds = 0.1 if condition == "plus" else -0.2
            else:
                semantic = "positive"
                token = 1
                log_odds = 0.2 if condition == "plus" else 0.05
            kl = 0.001
            if break_at_largest_dose and dose == 0.15 and condition == "plus":
                semantic = "negative"
                token = 2
                kl = 0.06
            rows.append(
                _score_row(
                    unit_id,
                    condition=condition,
                    dose=dose,
                    target=target,
                    direction_key=direction_key,
                    preserve_first=preserve_first,
                    semantic=semantic,
                    token_id=token,
                    log_odds=log_odds,
                    kl=kl,
                )
            )
    return rows


def _lock() -> dict[str, Any]:
    return {
        "absolute_residual_relative_doses": DOSES,
        "protected_limits": {
            "maximum_exact_greedy_token_changes": 0,
            "maximum_new_other_outputs": 0,
            "mean_full_vocabulary_kl_changed_to_baseline": 0.005,
            "empirical_p95_full_vocabulary_kl_changed_to_baseline": 0.02,
            "maximum_full_vocabulary_kl_changed_to_baseline": 0.05,
        },
    }


def test_summary_selects_radius_only_from_protected_rows() -> None:
    sp_rows = []
    sp_rows.extend(
        _unit_rows(
            "d::self::A",
            target="self",
            direction_key="d",
            preserve_first=True,
        )
    )
    sp_rows.extend(
        _unit_rows(
            "d::self::B",
            target="self",
            direction_key="d",
            preserve_first=False,
        )
    )
    sp_rows.extend(
        _unit_rows(
            "d::other::A",
            target="other",
            direction_key="d",
            preserve_first=True,
        )
    )
    control_rows = _unit_rows(
        "control::d::c",
        target="control",
        direction_key="d",
        preserve_first=True,
        break_at_largest_dose=True,
    )

    summary = probe.summarize(lock=_lock(), sp_rows=sp_rows, control_rows=control_rows)

    assert summary["selected_empirical_trust_radius"] == 0.1
    assert summary["trust_radius_selection_uses_self_outcomes"] is False
    reports = {row["absolute_residual_relative_dose"]: row for row in summary["dose_reports"]}
    assert reports[0.1]["protected_limits_pass"] is True
    assert reports[0.15]["protected_limits_pass"] is False
    assert reports[0.15]["protected"]["change_counts"]["exact_greedy_token_changes"] == 1
    assert reports[0.1]["self"]["directions_meeting_both_signs_both_orders"] == 1


def test_validate_chunk_requires_every_locked_sign_and_dose() -> None:
    identity = {"identity_sha256": "i"}
    job = {
        "unit_id": "u",
        "entry": {"direction_sha256": "d"},
    }
    rows = []
    for condition, dose in sorted(probe._expected_cells(DOSES)):
        rows.append(
            {
                "schema_version": probe.ROW_SCHEMA,
                "development_only": True,
                "study_identity_sha256": "i",
                "unit_id": "u",
                "direction_sha256": "d",
                "condition": condition,
                "absolute_residual_relative_dose": dose,
            }
        )
    probe._validate_chunk(rows, job=job, identity=identity, doses=DOSES)

    rows.pop()
    try:
        probe._validate_chunk(rows, job=job, identity=identity, doses=DOSES)
    except ValueError as error:
        assert "exact dose/sign coverage" in str(error)
    else:  # pragma: no cover - assertion helper
        raise AssertionError("incomplete chunk unexpectedly passed")
