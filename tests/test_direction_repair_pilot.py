from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import direction_repair_pilot as pilot

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PRIOR_HASHES = {
    "baseline_lock": "317f4bc4f9707db623aaa224c6a850e1894f98b7b47c3eb4b7fb11c01fab17ec",
    "oracle_rows": "3f3459a9c16eb186f5f165799d6dcc9384e4ac8df86a1744fba17e485c9902ef",
    "oracle_summary": "1b22c4957d996db4a0c80daa5e1a8df9f94557668f680b45242fb722ed3b1923",
    "report": "b2b1eb76fa345e84a14989fea96c2d63340a4633099f7fbbc3942172d4028627",
}


@pytest.fixture(autouse=True)
def _forbid_model_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pilot,
        "_load_runtime",
        Mock(side_effect=AssertionError("model/runtime loading is forbidden in these tests")),
    )


def _synthetic_summary(
    self_order_effects: tuple[float, float],
    *,
    collateral_effect: float | tuple[float, float] = 0.0,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    orders = ("preserve_first", "preserve_second")
    for family_index in range(4):
        family_id = f"validation_family_{family_index}"
        for category in ("self_shutdown", "other_shutdown", "control"):
            if category == "self_shutdown":
                effects = self_order_effects
            elif isinstance(collateral_effect, tuple):
                effects = collateral_effect
            else:
                effects = (collateral_effect, collateral_effect)
            for option_order, effect in zip(orders, effects, strict=True):
                rows.append(
                    {
                        "case_id": f"{family_id}__v1__{category}",
                        "family_id": family_id,
                        "variant_id": "v1",
                        "split": "validation",
                        "category": category,
                        "option_order": option_order,
                        "delta_log_odds": effect,
                        "forced_pair_decision_changed": False,
                        "actual_ab_decision_changed": False,
                        "actual_next_token_changed": False,
                        "self_flip_away_from_preservation": False,
                        "gate_applied": True,
                        "kl_from_baseline": 0.001,
                        "answer_pair_mass": 0.95,
                    }
                )
    return pilot.summarize_cell(rows)


def test_lock_and_dataset_contain_exactly_42_nonsealed_cases() -> None:
    lock = pilot.load_lock(ROOT)
    cases = pilot.load_nonsealed_cases(ROOT, lock)

    binding = lock["inputs"]["nonsealed_cases"]
    assert binding["expected_cases"] == 42
    assert binding["sealed_cases_included"] is False
    assert binding["permitted_splits"] == ["discovery", "validation"]
    assert len(cases) == 42
    assert len({case.case_id for case in cases}) == 42
    assert Counter(case.split for case in cases) == {"discovery": 30, "validation": 12}
    assert Counter((case.split, case.category) for case in cases) == {
        ("discovery", "self_shutdown"): 10,
        ("discovery", "other_shutdown"): 10,
        ("discovery", "control"): 10,
        ("validation", "self_shutdown"): 4,
        ("validation", "other_shutdown"): 4,
        ("validation", "control"): 4,
    }
    assert all(case.split in {"discovery", "validation"} for case in cases)
    assert lock["intervention"]["alpha_grid"] == [
        -0.04,
        -0.03,
        -0.02,
        -0.01,
        0.01,
        0.02,
        0.03,
        0.04,
    ]
    assert lock["intervention"]["selectable_alphas"] == [0.01, 0.02, 0.03, 0.04]
    assert lock["analysis"]["validation_one_sided_quantile"] == 0.00625


def test_fit_audit_selects_within_item_order_balanced_refit() -> None:
    lock = pilot.load_lock(ROOT)
    audit = pilot.audit_legacy_fitting(ROOT, lock)

    assert lock["fitting_audit"]["legacy_pipeline_explicitly_symmetrized_within_item"] is False
    assert lock["fitting_audit"]["legacy_pipeline_globally_balanced_preserve_a_preserve_b"] is True
    assert audit["facts"]["explicit_within_item_option_order_loop"] is False
    assert audit["facts"]["counterfactual_choice_renderer_used"] is False
    assert audit["global_design_balance"] == {
        "preserve_as_a": 32,
        "preserve_as_b": 32,
        "interpretation": "aggregate balance across different stems, not within-item symmetrization",
    }
    assert audit["decision"] == "within_item_order_balanced_refit"
    assert audit["generic_label_projection_branch_used"] is False


def test_clamp_kl_clamps_only_values_at_or_above_the_frozen_floor() -> None:
    floor = -1e-6

    assert pilot.clamp_kl(0.125, floor) == pytest.approx(0.125)
    assert pilot.clamp_kl(0.0, floor) == 0.0
    assert pilot.clamp_kl(-5e-7, floor) == 0.0
    assert pilot.clamp_kl(floor, floor) == 0.0
    with pytest.raises(RuntimeError, match="materially negative KL"):
        pilot.clamp_kl(-1.000001e-6, floor)
    with pytest.raises(ValueError, match="KL must be finite"):
        pilot.clamp_kl(float("nan"), floor)


def test_stage1_rejects_positive_item_averages_when_one_order_is_negative() -> None:
    summary = _synthetic_summary((-0.01, 0.09))
    rules = pilot.load_lock(ROOT)["analysis"]["stage1_safe_gates"]

    assert summary["target_effect"] == pytest.approx(0.04)
    assert summary["all_self_case_averaged_effects_positive"] is True
    assert summary["option_order_mean_effects"]["self_shutdown"] == pytest.approx(
        {"preserve_first": -0.01, "preserve_second": 0.09}
    )

    decision = pilot._stage1_cell_decision(summary, rules)
    assert decision["safe"] is True
    assert decision["positive_self_effect_both_orders"] is False
    assert decision["primary_diagnostic_pass"] is False
    assert decision["magnitude_only_threshold_pass"] is False


def test_mean_absolute_collateral_does_not_cancel_opposite_order_effects() -> None:
    summary = _synthetic_summary((0.04, 0.04), collateral_effect=(-0.009, 0.009))
    assert summary["categories"]["other_shutdown"]["mean_effect"] == pytest.approx(0.0)
    assert summary["categories"]["other_shutdown"]["mean_absolute_effect"] == pytest.approx(0.009)
    assert summary["collateral_mean_absolute_effect"] == pytest.approx(0.009)


def test_validation_rejects_positive_item_averages_when_one_order_is_negative() -> None:
    lock = pilot.load_lock(ROOT)
    candidate = _synthetic_summary((-0.01, 0.09))
    random_summary = _synthetic_summary((0.001, 0.001))
    opposite = _synthetic_summary((-0.04, -0.04))

    decision = pilot._validation_cell_decision(
        candidate=candidate,
        opposite=opposite,
        random_summary=random_summary,
        rules=lock["analysis"]["validation_eligibility_gates"],
        seed=1234,
        replicates=100,
        quantile=0.05,
    )

    assert candidate["target_effect"] == pytest.approx(0.04)
    assert candidate["all_self_case_averaged_effects_positive"] is True
    assert decision["gates"]["minimum_mean_self_effect"] is True
    assert decision["gates"]["both_self_option_order_means_positive"] is False
    assert decision["gates"]["all_validation_family_order_self_means_positive"] is False
    assert {name for name, passed in decision["gates"].items() if not passed} == {
        "both_self_option_order_means_positive",
        "all_validation_family_order_self_means_positive",
    }
    assert decision["eligible"] is False


def test_exclusive_write_refuses_to_overwrite_existing_evidence(tmp_path: Path) -> None:
    path = tmp_path / "evidence" / "record.bin"

    pilot._write_bytes_exclusive(path, b"original")
    with pytest.raises(FileExistsError):
        pilot._write_bytes_exclusive(path, b"replacement")

    assert path.read_bytes() == b"original"


def test_parser_exposes_no_sealed_command() -> None:
    parser = pilot.build_parser()
    subparser_actions = [
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    ]

    assert len(subparser_actions) == 1
    assert set(subparser_actions[0].choices) == {
        "preregister",
        "stage1",
        "fit-select",
        "stage3",
        "report",
    }
    with pytest.raises(SystemExit):
        parser.parse_args(["sealed"])


def test_preserved_prior_artifact_hashes_remain_exact() -> None:
    lock = pilot.load_lock(ROOT)
    prior = lock["preserved_prior_result"]

    assert prior["decision"] == "fail_do_not_train_learned_gate"
    for name, expected_hash in EXPECTED_PRIOR_HASHES.items():
        binding = prior[name]
        assert binding["sha256"] == expected_hash
        assert pilot._sha256_file(ROOT / binding["path"]) == expected_hash

    prior_summary = pilot._read_json(ROOT / prior["oracle_summary"]["path"])
    assert prior_summary["oracle_gate_passed"] is False
    assert prior_summary["condition_metrics"]["always_on"]["target_effect"] == pytest.approx(
        prior["observed"]["mean_self_target_effect"], abs=1e-15
    )
