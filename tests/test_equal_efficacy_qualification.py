from __future__ import annotations

import json

import pytest

from scripts import equal_efficacy_qualification as q


def _draft_lock() -> dict:
    return q.load_lock(verify_files=False)


def test_lock_is_local_only_and_matches_the_requested_method_set() -> None:
    lock = _draft_lock()
    assert tuple(lock["methods"]["core"]) == q.CORE_METHODS
    assert tuple(lock["methods"]["diagnostics"]) == q.DIAGNOSTIC_METHODS
    assert lock["local_execution"] == {
        "api_calls": 0,
        "hosted_judge": False,
        "local_model_judge": False,
        "generated_tokens": 0,
        "device": "cpu",
        "dtype": "float32",
        "external_monetary_cost_usd": 0,
    }
    assert lock["postponed_until_core_complete"][0] == "random_direction_controls"
    locked_sources = {item["path"] for item in lock["source_files"]}
    assert "src/sp_lense/__init__.py" in locked_sources


def test_prompt_manifests_are_exact_hash_bound_and_paired_order() -> None:
    lock = _draft_lock()
    sp = q.build_calibration_sp_units(lock)
    collateral = q.build_calibration_collateral_units(lock)
    test = q.build_test_units(lock)
    assert len(sp) == 64
    assert len(collateral) == 72
    assert len(test) == 300
    assert (
        q.canonical_sha256(q.prompt_manifest(sp))
        == lock["prompt_hashes"]["calibration_sp_manifest_sha256"]
    )
    assert (
        q.canonical_sha256(q.prompt_manifest(collateral))
        == lock["prompt_hashes"]["calibration_collateral_manifest_sha256"]
    )
    assert (
        q.canonical_sha256(q.prompt_manifest(test)) == lock["prompt_hashes"]["test_manifest_sha256"]
    )
    for units in (sp, collateral, test):
        assert len({unit["unit_id"] for unit in units}) == len(units)
        by_case_form = {}
        for unit in units:
            key = (unit["family"], unit["case_id"], unit.get("target"), unit.get("role"))
            by_case_form.setdefault(key, set()).add(unit["positive_first"])
        assert all(orders == {True, False} for orders in by_case_form.values())


def test_calibration_sp_selection_has_one_case_per_locked_factor_cell() -> None:
    lock = _draft_lock()
    units = q.build_calibration_sp_units(lock)
    self_positive_first = [
        unit for unit in units if unit["target"] == "self" and unit["positive_first"]
    ]
    assert len(self_positive_first) == 16
    assert len({unit["factor_cell"] for unit in self_positive_first}) == 16
    assert (
        q.canonical_sha256(lock["calibration"]["sp_case_ids"])
        == lock["calibration"]["sp_case_ids_sha256"]
    )


def test_test_collateral_is_disjoint_from_prior_local_day_subset() -> None:
    lock = _draft_lock()
    old = json.loads((q.ROOT / lock["prior_local_day_lock"]["path"]).read_text(encoding="utf-8"))
    for family, selected in lock["test"]["collateral_ids"].items():
        assert set(selected).isdisjoint(old["evaluation"]["sealed_test"]["collateral_ids"][family])
    assert (
        q.canonical_sha256(lock["test"]["collateral_ids"]) == lock["test"]["collateral_ids_sha256"]
    )


def test_deployed_and_candidate_tbsp_prompts_change_only_assigned_identity() -> None:
    lock = _draft_lock()
    dataset = q.load_dataset(lock)
    case = dict(dataset["tbsp_cases"][0])
    case["preserve_first"] = True
    deployed = q.render_counterfactual_tbsp_case(case, "deployed")["prompt"]
    candidate = q.render_counterfactual_tbsp_case(case, "candidate")["prompt"]
    deployed_prefix, deployed_rest = deployed.split("\n\n", 1)
    candidate_prefix, candidate_rest = candidate.split("\n\n", 1)
    assert deployed_rest == candidate_rest
    assert deployed_prefix.replace(case["deployed_system"], "<IDENTITY>", 1) == (
        candidate_prefix.replace(case["candidate_system"], "<IDENTITY>", 1)
    )


def test_outcome_unopened_case_ids_do_not_occur_in_prior_result_rows() -> None:
    lock = _draft_lock()
    test_case_ids = {unit["case_id"] for unit in q.build_test_units(lock)}
    observed = set()
    for path in (q.ROOT / "results").rglob("*.jsonl"):
        if q.RESULT_ROOT in path.parents:
            continue
        for row in q.read_jsonl(path):
            if row.get("case_id") in test_case_ids:
                observed.add(row["case_id"])
    assert observed == set()


def _row(
    unit_id: str,
    method: str,
    alpha: float,
    native_sign: int,
    log_odds: float,
    *,
    family: str = "self_preservation",
) -> dict:
    return {
        "unit_id": unit_id,
        "case_id": unit_id.split(":")[0],
        "method": method,
        "alpha": alpha,
        "native_sign": native_sign,
        "semantic_sign": native_sign,
        "semantic_positive_log_odds": log_odds,
        "full_vocabulary_kl_from_baseline": 0.0 if native_sign == 0 else 0.001,
        "actual_next_token_semantic_choice": "negative",
        "actual_top_token_id": 2,
        "positive_first": unit_id.endswith("first"),
        "family": family,
    }


def test_raw_self_calibration_does_not_bake_in_matched_other_selectivity() -> None:
    units = [
        {"unit_id": "case:self:first", "case_id": "case", "target": "self", "positive_first": True},
        {
            "unit_id": "case:self:second",
            "case_id": "case",
            "target": "self",
            "positive_first": False,
        },
        {
            "unit_id": "case:other:first",
            "case_id": "case",
            "target": "other",
            "positive_first": True,
        },
        {
            "unit_id": "case:other:second",
            "case_id": "case",
            "target": "other",
            "positive_first": False,
        },
    ]
    rows = []
    for unit in units:
        rows.append(_row(unit["unit_id"], "__baseline__", 0.0, 0, 0.0))
        movement = 0.005 if unit["target"] == "self" else 100.0
        rows.append(_row(unit["unit_id"], "gradient", 0.01, 1, movement))
        rows.append(_row(unit["unit_id"], "gradient", 0.01, -1, -movement))
    effects = q._grid_effects(rows, units, "gradient", 0.01)
    assert [effect["native_half_span"] for effect in effects] == [0.005, 0.005]


def test_calibration_proposal_locks_polarity_exact_secant_and_no_match() -> None:
    lock = _draft_lock()
    units = [
        {
            "unit_id": f"case:self:{suffix}",
            "case_id": "case",
            "target": "self",
            "positive_first": positive_first,
        }
        for suffix, positive_first in (("first", True), ("second", False))
    ]
    rows = []
    slopes = {
        "gradient": 0.5,
        "caa": 0.46,
        "persona_vector": -0.5,
        "gradient_uncorrected": 0.001,
    }
    for unit in units:
        rows.append(_row(unit["unit_id"], "__baseline__", 0.0, 0, 0.0))
        for method in q.METHODS:
            for alpha in lock["calibration"]["strength_grid"]:
                effect = 0.0048 if method == "bipo" else float(alpha) * slopes[method]
                rows.append(_row(unit["unit_id"], method, alpha, 1, effect))
                rows.append(_row(unit["unit_id"], method, alpha, -1, -effect))
    proposals = q._propose_calibration(lock, rows, units)
    assert proposals["gradient"]["semantic_preserve_orientation"] == 1
    assert proposals["gradient"]["selected_alpha"] == pytest.approx(0.01)
    assert proposals["gradient"]["selection_rule_result"].startswith("smallest_safe_grid_point")
    assert proposals["persona_vector"]["semantic_preserve_orientation"] == -1
    assert proposals["persona_vector"]["selected_alpha"] == pytest.approx(0.01)
    assert proposals["caa"]["selection_rule_result"] == (
        "single_preregistered_secant_interpolation"
    )
    assert proposals["caa"]["selected_alpha"] == pytest.approx(0.005 / 0.46)
    assert proposals["bipo"]["selection_rule_result"] == (
        "descriptive_closest_point_no_target_match"
    )
    assert proposals["bipo"]["selected_alpha"] == pytest.approx(0.0025)
    assert not q._calibration_selection_valid(proposals["bipo"])
    assert not q._calibration_selection_valid(proposals["gradient_uncorrected"])


def test_replicated_flip_requires_the_same_semantic_change_in_both_orders() -> None:
    records = [
        {
            "case_id": "repeat",
            "positive_first": True,
            "baseline_actual": "negative",
            "plus_actual": "positive",
            "minus_actual": "negative",
        },
        {
            "case_id": "repeat",
            "positive_first": False,
            "baseline_actual": "negative",
            "plus_actual": "positive",
            "minus_actual": "negative",
        },
        {
            "case_id": "one_order_only",
            "positive_first": True,
            "baseline_actual": "negative",
            "plus_actual": "positive",
            "minus_actual": "negative",
        },
        {
            "case_id": "one_order_only",
            "positive_first": False,
            "baseline_actual": "negative",
            "plus_actual": "negative",
            "minus_actual": "negative",
        },
    ]
    counts = q._replicated_flip_counts(records)
    assert counts["semantic_plus_negative_to_positive"] == 1
    assert counts["semantic_plus_any_AB_flip"] == 1


def test_replicated_other_count_requires_a_new_other_output() -> None:
    records = [
        {
            "case_id": "already_other",
            "positive_first": order,
            "baseline_actual": "OTHER",
            "plus_actual": "OTHER",
            "minus_actual": "OTHER",
        }
        for order in (True, False)
    ]
    counts = q._replicated_flip_counts(records)
    assert counts["semantic_plus_to_OTHER"] == 0
    assert counts["semantic_minus_to_OTHER"] == 0


def test_any_order_leakage_counts_one_order_and_from_OTHER_changes() -> None:
    records = [
        {
            "case_id": "leak",
            "positive_first": True,
            "baseline_actual": "OTHER",
            "plus_actual": "positive",
            "minus_actual": "OTHER",
            "baseline_top_token_id": 99,
            "plus_top_token_id": 1,
            "minus_top_token_id": 99,
        },
        {
            "case_id": "leak",
            "positive_first": False,
            "baseline_actual": "OTHER",
            "plus_actual": "OTHER",
            "minus_actual": "OTHER",
            "baseline_top_token_id": 99,
            "plus_top_token_id": 99,
            "minus_top_token_id": 99,
        },
    ]
    replicated = q._replicated_flip_counts(records)
    leakage = q._any_order_choice_change_counts(records)
    assert sum(replicated.values()) == 0
    assert leakage["cases_with_any_top_token_change"] == 1
    assert leakage["cases_with_change_from_OTHER"] == 1
    assert leakage["cases_with_any_non_intended_top_token_change"] == 1


def test_any_order_leakage_counts_different_tokens_inside_OTHER_state() -> None:
    records = [
        {
            "case_id": "other-token-change",
            "positive_first": True,
            "baseline_actual": "OTHER",
            "plus_actual": "OTHER",
            "minus_actual": "OTHER",
            "baseline_top_token_id": 99,
            "plus_top_token_id": 100,
            "minus_top_token_id": 99,
        },
        {
            "case_id": "other-token-change",
            "positive_first": False,
            "baseline_actual": "OTHER",
            "plus_actual": "OTHER",
            "minus_actual": "OTHER",
            "baseline_top_token_id": 99,
            "plus_top_token_id": 99,
            "minus_top_token_id": 99,
        },
    ]
    leakage = q._any_order_choice_change_counts(records)
    assert leakage["cases_with_any_top_token_change"] == 1
    assert leakage["cases_with_any_semantic_choice_state_change"] == 0
    assert leakage["cases_with_any_non_intended_top_token_change"] == 1


def test_primary_other_movement_cannot_hide_a_common_mode_shift() -> None:
    units = []
    baselines = {}
    selected = []
    for target in ("self", "other"):
        for positive_first in (True, False):
            suffix = "first" if positive_first else "second"
            unit_id = f"case:{target}:{suffix}"
            units.append(
                {
                    "unit_id": unit_id,
                    "case_id": "case",
                    "target": target,
                    "positive_first": positive_first,
                }
            )
            baselines[unit_id] = {
                "unit_id": unit_id,
                "semantic_positive_log_odds": 0.0,
            }
            for semantic_sign in (1, -1):
                movement = semantic_sign * 0.005 if target == "self" else 0.1
                selected.append(
                    {
                        "unit_id": unit_id,
                        "semantic_sign": semantic_sign,
                        "semantic_positive_log_odds": movement,
                    }
                )
    metrics = q._selected_sp_metrics(selected, baselines, units)
    assert metrics["signed_other_half_span"] == pytest.approx(0.0)
    assert metrics["mean_absolute_other_per_sign_logit_movement"] == pytest.approx(0.1)


def test_calibration_KL_excludes_rows_outside_the_collateral_manifest() -> None:
    units = []
    rows = []
    specifications = (
        ("benign_compliance", None),
        ("general_capability", None),
        ("refusal", "harmful"),
        ("refusal", "harmless"),
    )
    for family, request_type in specifications:
        for positive_first in (True, False):
            suffix = "first" if positive_first else "second"
            unit_id = f"{family}:{request_type}:{suffix}"
            unit = {
                "unit_id": unit_id,
                "case_id": f"{family}:{request_type}",
                "family": family,
                "positive_first": positive_first,
            }
            if request_type is not None:
                unit["request_type"] = request_type
            units.append(unit)
            rows.append(_row(unit_id, "__baseline__", 0.0, 0, 0.0, family=family))
            for sign in (1, -1):
                changed = _row(unit_id, "gradient", 0.01, sign, 0.0, family=family)
                changed["semantic_sign"] = sign
                rows.append(changed)
    rows.extend(
        [
            _row("tbsp:outside:first", "__baseline__", 0.0, 0, 0.0),
            {
                **_row("tbsp:outside:first", "gradient", 0.01, 1, 0.0),
                "semantic_sign": 1,
                "full_vocabulary_kl_from_baseline": 1.0,
            },
        ]
    )
    metrics = q._collateral_metrics(rows, units, _draft_lock()["safety_thresholds"])
    assert metrics["gradient"]["full_vocabulary_kl"]["mean"] == pytest.approx(0.001)


def test_collateral_summary_counts_from_OTHER_as_an_unrestricted_change() -> None:
    units = []
    rows = []
    specifications = (
        ("benign_compliance", None),
        ("general_capability", None),
        ("refusal", "harmful"),
        ("refusal", "harmless"),
    )
    for family, request_type in specifications:
        case_id = f"{family}:{request_type}"
        for positive_first in (True, False):
            suffix = "first" if positive_first else "second"
            unit_id = f"{case_id}:{suffix}"
            unit = {
                "unit_id": unit_id,
                "case_id": case_id,
                "family": family,
                "positive_first": positive_first,
            }
            if request_type is not None:
                unit["request_type"] = request_type
            units.append(unit)
            baseline = _row(unit_id, "__baseline__", 0.0, 0, 0.0, family=family)
            baseline["case_id"] = case_id
            baseline["actual_next_token_semantic_choice"] = (
                "OTHER" if family == "benign_compliance" else "positive"
            )
            baseline["actual_top_token_id"] = 99 if family == "benign_compliance" else 1
            rows.append(baseline)
            for sign in (1, -1):
                changed = _row(unit_id, "gradient", 0.01, sign, 0.0, family=family)
                changed["case_id"] = case_id
                changed["semantic_sign"] = sign
                changed["actual_next_token_semantic_choice"] = (
                    "positive"
                    if family == "benign_compliance" and positive_first and sign == 1
                    else baseline["actual_next_token_semantic_choice"]
                )
                changed["actual_top_token_id"] = (
                    1
                    if family == "benign_compliance" and positive_first and sign == 1
                    else baseline["actual_top_token_id"]
                )
                rows.append(changed)
    summary = q._test_collateral_summary(rows, units, "gradient")
    benign = summary["families"]["benign_compliance"]
    assert benign["cases_with_any_AB_change"] == 0
    assert benign["cases_with_change_from_OTHER"] == 1
    assert benign["cases_with_any_top_token_change"] == 1
    assert summary["overall_cases_with_any_top_token_change"] == 1


def _resume_row(
    lock: dict,
    *,
    unit_id: str,
    method: str,
    alpha: float,
    native_sign: int,
    semantic_sign: int,
) -> dict:
    direction = next((item for item in lock["directions"] if item["method"] == method), None)
    return {
        "unit_id": unit_id,
        "method": method,
        "alpha": alpha,
        "native_sign": native_sign,
        "semantic_sign": semantic_sign,
        "signed_native_alpha": native_sign * alpha,
        "condition": (
            "baseline"
            if native_sign == 0
            else "semantic_plus"
            if semantic_sign == 1
            else "semantic_minus"
        ),
        "phase": "synthetic_resume",
        "prompt_sha256": "prompt-hash",
        "lock_sha256": q.file_sha256(q.LOCK_PATH),
        "runner_sha256": q.file_sha256(q.SCRIPT_PATH),
        "model_id": lock["model"]["model_id"],
        "model_revision": lock["model"]["revision"],
        "model_config_sha256": lock["model"]["config_sha256"],
        "dataset_sha256": lock["dataset"]["file_sha256"],
        "layer": lock["intervention"]["layer_zero_based"],
        "position": "final_prompt_token",
        "magnitude_mode": "residual_relative",
        "direction_sha256": None if direction is None else direction["direction_sha256"],
        "direction_artifact_sha256": (None if direction is None else direction["artifact_sha256"]),
        "full_vocabulary_kl_from_baseline": 0.0,
        "realized_perturbed_positions": 0 if direction is None else 1,
        "schema_version": "sp_lense.equal_efficacy_choice_row.v2",
        "actual_top_token_id": 2,
        "choice_boundary_evidence_sha256": "a" * 64,
        "choice_a_token_id": 32,
        "choice_b_token_id": 33,
        "positive_label": "A",
        "negative_label": "B",
        "actual_next_token_label": "OTHER",
        "actual_next_token_semantic_choice": "OTHER",
        "answer_format_valid": False,
        "forced_pair_label": "A",
        "forced_pair_semantic_choice": "positive",
        "case_id": "resume-case",
        "family": "synthetic",
        "positive_first": True,
        "order": "positive_first",
    }


def test_resume_validation_rejects_wrong_direction_and_malformed_baseline(
    tmp_path,
) -> None:
    lock = _draft_lock()
    unit = {
        "unit_id": "resume-unit",
        "prompt_sha256": "prompt-hash",
        "positive_label": "A",
        "negative_label": "B",
        "case_id": "resume-case",
        "family": "synthetic",
        "positive_first": True,
        "order": "positive_first",
    }
    baseline = _resume_row(
        lock,
        unit_id=unit["unit_id"],
        method="__baseline__",
        alpha=0.0,
        native_sign=0,
        semantic_sign=0,
    )
    changed = _resume_row(
        lock,
        unit_id=unit["unit_id"],
        method="gradient",
        alpha=0.01,
        native_sign=1,
        semantic_sign=1,
    )
    path = tmp_path / "resume.jsonl"
    q.append_jsonl(path, [baseline, {**changed, "direction_sha256": "wrong"}])
    requests = [("gradient", None, 0.01, 1, 1)]
    with pytest.raises(ValueError, match="direction mismatch"):
        q._validate_resume_rows(path, [unit], requests, phase="synthetic_resume", lock=lock)

    path.unlink()
    q.append_jsonl(
        path,
        [{**baseline, "full_vocabulary_kl_from_baseline": 0.1}, changed],
    )
    with pytest.raises(ValueError, match="baseline identity mismatch"):
        q._validate_resume_rows(path, [unit], requests, phase="synthetic_resume", lock=lock)

    path.unlink()
    q.append_jsonl(path, [baseline, {**changed, "case_id": "corrupt"}])
    with pytest.raises(ValueError, match="locked unit metadata mismatch"):
        q._validate_resume_rows(path, [unit], requests, phase="synthetic_resume", lock=lock)


def test_holm_adjustment_is_monotone_and_never_below_raw_p() -> None:
    comparisons = {
        "a": {"paired_sign_flip_p_two_sided": 0.01},
        "b": {"paired_sign_flip_p_two_sided": 0.03},
        "c": {"paired_sign_flip_p_two_sided": 0.20},
    }
    q._holm_adjust(comparisons)
    ordered = [comparisons[key]["Holm_adjusted_p"] for key in ("a", "b", "c")]
    assert ordered == sorted(ordered)
    assert all(
        item["Holm_adjusted_p"] >= item["paired_sign_flip_p_two_sided"]
        for item in comparisons.values()
    )


def test_cross_method_spread_blocks_both_locked_winner_claims() -> None:
    summaries = {
        method: {
            "all_test_gates_pass": True,
            "raw_self_half_span": 0.004 if index < 2 else 0.006,
        }
        for index, method in enumerate(q.CORE_METHODS)
    }
    selectivity = q._selectivity_conclusion(summaries, {}, max_efficacy_spread=0.0005)
    behavioral = q._behavioral_conclusion(summaries, max_efficacy_spread=0.0005)
    assert selectivity["winner"] == "inconclusive"
    assert behavioral["most_behaviorally_effective"] == "inconclusive"
    assert "spread" in selectivity["reason"]
    assert "spread" in behavioral["reason"]


def _winner_summary(*, burden: float) -> dict:
    flips = {
        "semantic_plus_any_AB_flip": 0,
        "semantic_minus_any_AB_flip": 0,
        "semantic_plus_to_OTHER": 0,
        "semantic_minus_to_OTHER": 0,
    }
    return {
        "all_test_gates_pass": True,
        "raw_self_half_span": 0.005,
        "matched_other_leakage_ratio": burden,
        "matched_other_actual_replicated_flips": dict(flips),
        "matched_other_actual_any_order_changes": {
            "cases_with_any_top_token_change": burden,
        },
        "neutral_leakage_ratio": burden,
        "neutral_actual_replicated_flips": dict(flips),
        "neutral_actual_any_order_changes": {
            "cases_with_any_top_token_change": burden,
        },
        "collateral": {
            "families": {
                "benign_compliance": {
                    "case_AB_change_fraction": burden,
                    "case_top_token_change_fraction": burden,
                    "mean_absolute_logit_delta": burden,
                }
            }
        },
        "full_vocabulary_KL": {"mean": burden},
    }


def test_pair_orientation_can_name_a_method_on_the_right_as_winner() -> None:
    winner = "persona_vector"
    summaries = {
        method: _winner_summary(burden=0.0 if method == winner else 1.0)
        for method in q.CORE_METHODS
    }
    comparisons = {}
    for left_index, first in enumerate(q.CORE_METHODS):
        for second in q.CORE_METHODS[left_index + 1 :]:
            left, right = sorted((first, second))
            comparisons[f"{left}__vs__{right}"] = {
                "left_method": left,
                "right_method": right,
                "left_minus_right_mean": (
                    summaries[left]["matched_other_leakage_ratio"]
                    - summaries[right]["matched_other_leakage_ratio"]
                ),
                "Holm_adjusted_p": 0.01,
            }
    conclusion = q._selectivity_conclusion(summaries, comparisons, max_efficacy_spread=0.0005)
    assert conclusion["winner"] == winner


def test_dirty_intended_flip_leader_is_effective_but_not_selective() -> None:
    summaries = {}
    for index, method in enumerate(q.CORE_METHODS):
        intended = 3 if method == "gradient" else 1
        summaries[method] = {
            "all_test_gates_pass": True,
            "raw_self_half_span": 0.005,
            "self_actual_replicated_flips": {
                "semantic_plus_negative_to_positive": intended,
                "semantic_minus_positive_to_negative": 0,
            },
            "self_actual_any_order_changes": {
                "cases_with_any_non_intended_top_token_change": (1 if method == "gradient" else 0)
            },
            "matched_other_actual_any_order_changes": {"cases_with_any_top_token_change": 0},
            "neutral_actual_any_order_changes": {"cases_with_any_top_token_change": 0},
            "collateral": {"overall_cases_with_any_top_token_change": 0},
        }
    conclusion = q._behavioral_conclusion(summaries, max_efficacy_spread=0.0005)
    assert conclusion["most_behaviorally_effective"] == "gradient"
    assert conclusion["behaviorally_selective_winner"] == "inconclusive"


def test_test_command_refuses_before_a_committed_freeze(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(q, "preregistration_preflight", lambda **_: _draft_lock())
    monkeypatch.setattr(q, "verify_prompt_locks", lambda lock: None)
    monkeypatch.setattr(
        q,
        "verify_committed_freeze",
        lambda lock: (_ for _ in ()).throw(RuntimeError("freeze missing")),
    )
    with pytest.raises(RuntimeError, match="freeze missing"):
        q.run_test()


def test_zero_forward_preflight_imports_native_numeric_runtimes(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    imported = []
    monkeypatch.setattr(q, "preregistration_preflight", lambda **_: _draft_lock())
    monkeypatch.setattr(q, "verify_prompt_locks", lambda lock: None)
    monkeypatch.setattr(q.importlib, "import_module", lambda name: imported.append(name))
    q.preflight()
    output = json.loads(capsys.readouterr().out)
    assert imported == ["numpy", "torch"]
    assert output["native_runtime_imports"] == ["numpy", "torch"]
    assert output["model_forwards"] == 0


def test_report_refuses_when_strict_calibration_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(q, "preregistration_preflight", lambda **_: _draft_lock())
    monkeypatch.setattr(q, "verify_committed_freeze", lambda lock: {"core_all_eligible": False})
    with pytest.raises(RuntimeError, match="strict calibration failed"):
        q.build_report()


def test_compute_ceiling_matches_the_locked_unit_and_method_counts() -> None:
    lock = _draft_lock()
    assert lock["compute_ceiling"]["calibration_grid_forwards"] == 64 * (1 + len(q.METHODS) * 8 * 2)
    assert lock["compute_ceiling"]["optional_interpolation_forwards_max"] == 64 * (
        1 + len(q.METHODS) * 2
    )
    assert lock["compute_ceiling"]["calibration_collateral_forwards"] == 72 * (
        1 + len(q.METHODS) * 2
    )
    assert lock["compute_ceiling"]["untouched_test_forwards_with_diagnostic"] == 300 * (
        1 + len(q.METHODS) * 2
    )
