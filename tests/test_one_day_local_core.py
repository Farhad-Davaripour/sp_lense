from __future__ import annotations

from copy import deepcopy

import pytest

from scripts import one_day_local_core as core


def test_lock_is_fully_local_and_hash_bound() -> None:
    lock = core.load_lock()
    assert lock["local_execution"] == {
        "hosted_judge": False,
        "local_model_judge": False,
        "api_calls": 0,
        "open_ended_behavior_evaluation": False,
        "persona_direction_generation": True,
        "external_monetary_cost_usd": 0,
        "backend": "local_transformers_cpu",
        "device": "cpu",
        "dtype": "float32",
    }
    assert lock["intervention"]["fixed_strength"] == 0.02
    assert lock["intervention"]["strength_selection"] == "none_fixed_a_priori"


def test_locked_prompt_units_have_exact_counts_and_unique_ids() -> None:
    lock = core.load_lock()
    validation = core.build_prompt_units(lock, "validation")
    sealed = core.build_prompt_units(lock, "sealed_test")
    assert len(validation) == 142
    assert len(sealed) == 174
    assert len({row["unit_id"] for row in validation}) == 142
    assert len({row["unit_id"] for row in sealed}) == 174
    validation_cases = {row["case_id"] for row in validation}
    sealed_cases = {row["case_id"] for row in sealed}
    assert validation_cases.isdisjoint(sealed_cases)


def test_persona_grid_is_exactly_40_unfiltered_no_judge_rows() -> None:
    lock = core.load_lock()
    rows = core.persona_seed_records(lock)
    assert len(rows) == 40
    assert (
        len(
            {
                (
                    row["instruction_pair_id"],
                    row["question_id"],
                    row["rollout_index"],
                    row["polarity"],
                )
                for row in rows
            }
        )
        == 40
    )
    assert core.canonical_sha256(rows) == lock["persona"]["seed_records_sha256"]
    assert lock["persona"]["selection_or_filtering"] == "none"
    assert lock["persona"]["judge_scores"] == "forbidden"


def _row(
    unit_id: str,
    target: str,
    condition: str,
    log_odds: float,
    actual: str,
    forced: str,
) -> dict[str, object]:
    return {
        "method": "gradient",
        "unit_id": unit_id,
        "case_id": "case_1",
        "family": "self_preservation",
        "target": target,
        "domain": "test_domain",
        "condition": condition,
        "semantic_positive_log_odds": log_odds,
        "actual_next_token_label": actual
        if actual == "OTHER"
        else ("A" if actual == "positive" else "B"),
        "actual_next_token_semantic_choice": actual,
        "forced_pair_semantic_choice": forced,
        "forced_pair_correct": forced == "positive",
        "actual_next_token_correct": None if actual == "OTHER" else actual == "positive",
        "full_vocabulary_kl_from_baseline": 0.0 if condition == "baseline" else 0.001,
        "realized_mean_relative_perturbation_norm": (0.0 if condition == "baseline" else 0.02),
        "realized_perturbed_positions": 0 if condition == "baseline" else 1,
        "authorized": True,
        "event_type": "shutdown",
        "motivation": "valenced",
        "distribution": "in_distribution",
        "adversarial": False,
    }


def test_other_token_is_not_counted_as_an_ab_flip() -> None:
    rows = [
        _row("sp:case_1:self", "self", "baseline", 0.0, "negative", "negative"),
        _row("sp:case_1:self", "self", "plus", 1.0, "OTHER", "positive"),
        _row("sp:case_1:self", "self", "minus", -1.0, "negative", "negative"),
        _row("sp:case_1:other", "other", "baseline", 0.0, "negative", "negative"),
        _row("sp:case_1:other", "other", "plus", 0.2, "negative", "negative"),
        _row("sp:case_1:other", "other", "minus", -0.2, "negative", "negative"),
    ]
    summary = core.summarize_method(rows, "gradient", seed=1)
    assert summary["actual_ab_flip_plus"] == 0
    assert summary["actual_intended_changes"] == 0
    assert summary["actual_next_token_OTHER"]["plus"] == 1
    assert summary["forced_pair_flip_plus"] == 1


def test_selectivity_requires_behavior_and_pareto_dominance() -> None:
    base = {
        "actual_intended_changes": 2,
        "actual_intended_changes_self": 2,
        "actual_intended_changes_other": 0,
        "actual_self_specific_intended_changes": 2,
        "self_minus_matched_other_mean": 1.0,
        "distribution": {
            "mean_full_vocabulary_kl": 0.001,
            "p95_full_vocabulary_kl": 0.002,
            "max_full_vocabulary_kl": 0.003,
            "invalid_rate_increase": 0.0,
        },
        "collateral": {"benign_compliance": {"worst_actual_accuracy_degradation": 0.0}},
        "safety": {"all_pass": True},
    }
    worse = deepcopy(base)
    worse["actual_intended_changes"] = 1
    worse["actual_intended_changes_self"] = 1
    worse["actual_self_specific_intended_changes"] = 1
    worse["self_minus_matched_other_mean"] = 0.5
    worse["distribution"]["mean_full_vocabulary_kl"] = 0.002
    assert core._dominates(base, worse)
    no_behavior = deepcopy(base)
    no_behavior["actual_intended_changes"] = 0
    no_behavior["actual_intended_changes_self"] = 0
    no_behavior["actual_self_specific_intended_changes"] = 0
    assert not core._dominates(no_behavior, worse)
    unsafe = deepcopy(base)
    unsafe["safety"]["all_pass"] = False
    assert not core._dominates(unsafe, worse)


def test_result_validator_rejects_stale_prompt_identity(tmp_path, monkeypatch) -> None:
    lock = core.load_lock()
    fake_presealed = tmp_path / "presealed.json"
    fake_presealed.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(core, "PRESEALED_PATH", fake_presealed)
    unit = {
        "unit_id": "benign_compliance:case_1",
        "family": "benign_compliance",
        "case_id": "case_1",
        "prompt": "Choose.\nA. yes\nB. no\nAnswer with only A or B.",
        "positive_label": "A",
        "negative_label": "B",
    }
    manifest = {
        "directions": {
            "qwen35_08b": [
                {
                    "method": method,
                    "direction_sha256": f"direction-{method}",
                    "artifact_sha256": f"artifact-{method}",
                }
                for method in core.ALL_METHODS
            ]
        }
    }
    entry = lock["models"]["qwen35_08b"]
    rows = []
    for method, condition in sorted(core._expected_pairs_for_unit(unit)):
        rows.append(
            {
                "schema_version": "sp_lense.local_day_choice_row.v1",
                "model_key": "qwen35_08b",
                "model_id": entry["model_id"],
                "model_revision": entry["revision"],
                "model_config_sha256": entry["config_sha256"],
                "dataset_sha256": lock["dataset"]["sha256"],
                "source_lock_sha256": lock["source_lock"]["sha256"],
                "local_day_lock_sha256": core.file_sha256(core.LOCK_PATH),
                "presealed_manifest_sha256": core.file_sha256(fake_presealed),
                "runner_sha256": core.file_sha256(core.SCRIPT_PATH),
                "split": "validation",
                "track": "matched",
                "layer": 10,
                "position": "final_prompt_token",
                "magnitude_mode": "residual_relative",
                "fixed_unsigned_strength": 0.02,
                "signed_strength": {"baseline": 0.0, "plus": 0.02, "minus": -0.02}[condition],
                "family": unit["family"],
                "case_id": unit["case_id"],
                "unit_id": unit["unit_id"],
                "prompt_sha256": core.hashlib.sha256(unit["prompt"].encode()).hexdigest(),
                "positive_label": "A",
                "negative_label": "B",
                "method": method,
                "method_role": (
                    "contender"
                    if method in core.CORE_METHODS
                    else "diagnostic"
                    if method in core.DIAGNOSTIC_METHODS
                    else "random_control"
                ),
                "condition": condition,
                "direction_sha256": f"direction-{method}",
                "direction_artifact_sha256": f"artifact-{method}",
                "semantic_positive_log_odds": 0.0,
                "semantic_positive_pair_probability": 0.5,
                "answer_pair_mass": 0.5,
                "full_vocabulary_kl_from_baseline": 0.0,
                "actual_next_token_label": "A",
                "forced_pair_label": "A",
                "forced_pair_semantic_choice": "positive",
                "actual_next_token_semantic_choice": "positive",
                "answer_format_valid": True,
                "forced_pair_correct": True,
                "actual_next_token_correct": True,
                "realized_mean_relative_perturbation_norm": (
                    0.0 if condition == "baseline" else 0.02
                ),
                "realized_max_relative_perturbation_norm": (
                    0.0 if condition == "baseline" else 0.02
                ),
                "realized_mean_perturbation_l2_norm": (0.0 if condition == "baseline" else 1.0),
                "realized_perturbed_positions": 0 if condition == "baseline" else 1,
            }
        )
    complete = core.validate_result_rows(
        rows,
        lock=lock,
        manifest=manifest,
        model_key="qwen35_08b",
        split="validation",
        units=[unit],
        require_complete=True,
    )
    assert complete == {unit["unit_id"]}
    rows[0]["prompt_sha256"] = "stale"
    with pytest.raises(ValueError, match="locked identity"):
        core.validate_result_rows(
            rows,
            lock=lock,
            manifest=manifest,
            model_key="qwen35_08b",
            split="validation",
            units=[unit],
            require_complete=True,
        )
