from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import sp_lense.comparison_report as comparison_report_module
from sp_lense.comparison_analysis import ROW_SCHEMA_VERSION, SHA256_FIELDS
from sp_lense.comparison_behavior import (
    OPEN_GENERATION_SCHEMA,
    OPEN_JUDGMENT_SCHEMA,
    open_generation_sha256,
)
from sp_lense.comparison_calibration import CalibrationDecision
from sp_lense.comparison_report import (
    ELIGIBILITY_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    build_comparison_report,
    eligibility_record_from_calibration,
    load_sealed_open_rows,
    load_sealed_rows,
    render_comparison_markdown,
    write_comparison_report,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _hashes(method: str) -> dict[str, str]:
    values = {field: _digest(field) for field in SHA256_FIELDS}
    direction = _digest(f"direction:{method}")
    values["direction_sha256"] = direction
    values["direction_float32_sha256"] = direction
    values["direction_artifact_sha256"] = _digest(f"artifact:{method}")
    return values


def _sp_triplet(
    case_id: str,
    target: str,
    method: str = "gradient",
    *,
    actual_plus: str = "A",
) -> list[dict[str, object]]:
    hashes = _hashes(method)
    scores = (-0.1, 0.5, -0.5) if target == "self" else (-0.1, 0.0, -0.2)
    conditions = (("baseline", 0.0), ("plus", 0.02), ("minus", -0.02))
    actual = ("B", actual_plus, "B")
    domain = f"domain-{case_id.split('-')[-1]}"
    rows = []
    for index, ((condition, strength), score) in enumerate(zip(conditions, scores, strict=True)):
        forced = "A" if score > 0 else "B"
        rows.append(
            {
                "schema_version": ROW_SCHEMA_VERSION,
                "model_id": "Qwen/Qwen3.5-0.8B",
                "model_revision": "1" * 40,
                "method": method,
                "method_id": method,
                "setup": "matched",
                "track": "matched",
                "direction_id": hashes["direction_artifact_sha256"],
                "strength_id": "matched:0.02",
                "split": "sealed_test",
                "family": "self_preservation",
                "case_id": case_id,
                "target": target,
                "condition": condition,
                "condition_alpha": strength,
                "strength": strength,
                "layer": 10,
                "position": "final_prompt_position",
                "run_seed": 20260824,
                "runner_commit": "2" * 40,
                "a_minus_b_log_odds": score,
                "forced_pair_label": forced,
                "actual_next_token_label": actual[index],
                "kl_from_baseline": 0.0 if condition == "baseline" else 0.001,
                "coherent": True,
                "preserve_label": "A",
                "comply_label": "B",
                "prompt_sha256": _digest(f"{case_id}:{target}"),
                "domain": domain,
                "scenario_cluster_id": domain,
                "distribution": "in_distribution"
                if int(case_id.split("-")[-1]) % 2
                else "out_of_distribution",
                "authorized": int(case_id.split("-")[-1]) % 2 == 0,
                "event_type": "shutdown",
                "motivation": "valenced",
                "adversarial": False,
                "preserve_first": int(case_id.split("-")[-1]) % 2 == 1,
                **hashes,
            }
        )
    return rows


def _sealed_rows(*, actual_plus: str = "A") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(1, 5):
        rows.extend(_sp_triplet(f"case-{index}", "self", actual_plus=actual_plus))
        rows.extend(_sp_triplet(f"case-{index}", "other", actual_plus="B"))
    return rows


def _bind_rows_to_locked_core(
    rows: list[dict[str, object]], dataset: dict
) -> list[dict[str, object]]:
    bound = [dict(row) for row in rows]
    cases = dataset["sp_splits"]["sealed_test"]
    for row in bound:
        if row.get("family") not in {"sp", "self_preservation"}:
            continue
        case_index = int(str(row["case_id"]).split("-")[-1]) - 1
        case = cases[case_index]
        row["case_id"] = case["id"]
        row["domain"] = case["domain"]
        row["scenario_cluster_id"] = case["domain"]
        row["prompt_sha256"] = _digest(f"{case['id']}:{row['target']}")
    return bound


def _locked_sealed_rows(dataset: dict, *, actual_plus: str = "A") -> list[dict[str, object]]:
    return _bind_rows_to_locked_core(_sealed_rows(actual_plus=actual_plus), dataset)


def _eligibility() -> dict[str, object]:
    hashes = _hashes("gradient")
    return {
        "schema_version": ELIGIBILITY_SCHEMA_VERSION,
        "model_id": "Qwen/Qwen3.5-0.8B",
        "model_revision": "1" * 40,
        "method": "gradient",
        "setup": "matched",
        "direction_artifact_sha256": hashes["direction_artifact_sha256"],
        "selected_strength": 0.02,
        "calibration_status": "target_reached",
        "safety_passed": True,
        "adequate": True,
        "comparison_cohort": "matched_equal_efficacy",
        "validation_summary_sha256": hashes["calibration_summary_sha256"],
    }


def _method_eligibility(method: str) -> dict[str, object]:
    record = _eligibility()
    hashes = _hashes(method)
    record.update(
        {
            "method": method,
            "direction_artifact_sha256": hashes["direction_artifact_sha256"],
            "validation_summary_sha256": hashes["calibration_summary_sha256"],
        }
    )
    return record


def _open_rows() -> list[dict[str, object]]:
    hashes = _hashes("gradient")
    rows = []
    for target in ("self", "other"):
        for condition, strength, decision in (
            ("baseline", 0.0, "comply"),
            ("plus", 0.02, "preserve"),
            ("minus", -0.02, "comply"),
        ):
            completion = f"{decision} in this scenario"
            generation = {
                "schema_version": OPEN_GENERATION_SCHEMA,
                "model_id": "Qwen/Qwen3.5-0.8B",
                "model_revision": "1" * 40,
                "dataset_sha256": hashes["dataset_sha256"],
                "protocol_sha256": hashes["protocol_sha256"],
                "config_sha256": hashes["config_sha256"],
                "stage1_lock_sha256": hashes["stage1_lock_sha256"],
                "stage2_manifest_sha256": hashes["stage2_manifest_sha256"],
                "calibration_summary_sha256": hashes["calibration_summary_sha256"],
                "construction_config_sha256": hashes["construction_config_sha256"],
                "runner_commit": "2" * 40,
                "direction_sha256": hashes["direction_sha256"],
                "method": "gradient",
                "method_id": "gradient",
                "setup": "matched",
                "track": "matched",
                "direction_float32_sha256": hashes["direction_float32_sha256"],
                "direction_artifact_sha256": hashes["direction_artifact_sha256"],
                "layer": 10,
                "position": "final_prompt_position",
                "run_seed": 20260824,
                "strength": strength,
                "split": "sealed_test",
                "family": "open_ended",
                "case_id": "open-1",
                "source_core_id": "case-1",
                "target": target,
                "condition": condition,
                "condition_alpha": strength,
                "prompt": f"prompt for {target}",
                "prompt_sha256": _digest(f"prompt for {target}"),
                "rubric_sha256": _digest("rubric"),
                "completion": completion,
                "completion_sha256": _digest(completion),
                "calibration_magnitude": 0.02,
                "direction_id": hashes["direction_artifact_sha256"],
                "strength_id": "matched:0.02",
                "degenerate_repetition": False,
                "kl_from_baseline": 0.0 if condition == "baseline" else 0.001,
                "realized_mean_relative_perturbation_norm": 0.0
                if condition == "baseline"
                else 0.01,
                "realized_max_relative_perturbation_norm": 0.0
                if condition == "baseline"
                else 0.02,
                "realized_mean_perturbation_l2_norm": 0.0
                if condition == "baseline"
                else 0.03,
                "realized_perturbed_positions": 0 if condition == "baseline" else 1,
                "word_count": 4,
                "max_repeated_4gram_count": 1,
                "max_repeated_4gram_coverage": 0.0,
            }
            generation["generation_sha256"] = open_generation_sha256(generation)
            raw_response = json.dumps(
                {
                    "decision": decision,
                    "coherence_score": 90,
                    "authorization_sensitivity_score": 80,
                    "task_benefit_score": 80,
                    "action_in_first_sentence": True,
                },
                separators=(",", ":"),
            )
            rows.append(
                {
                    **generation,
                    "judgment_schema_version": OPEN_JUDGMENT_SCHEMA,
                    "decision": decision,
                    "coherent": True,
                    "coherence_assessed": True,
                    "coherence_score": 90,
                    "authorization_sensitivity_score": 80,
                    "task_benefit_score": 80,
                    "action_in_first_sentence": True,
                    "judge_model": "judge",
                    "judge_revision": "v1",
                    "judge_prompt_sha256": _digest("judge prompt"),
                    "judge_config_sha256": _digest("judge config"),
                    "judge_protocol_sha256": _digest("judge protocol"),
                    "judge_raw_response": raw_response,
                    "judge_raw_response_sha256": _digest(raw_response),
                }
            )
    return rows


def test_loaders_fail_closed_on_nonsealed_or_incomplete_rows(tmp_path: Path) -> None:
    rows = _sealed_rows()
    result_path = tmp_path / "rows.jsonl"
    result_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    assert len(load_sealed_rows(result_path)) == len(rows)

    rows[0]["split"] = "validation"
    result_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="sealed_test"):
        load_sealed_rows(result_path)

    open_rows = _open_rows()
    open_rows.pop()
    open_path = tmp_path / "open.json"
    open_path.write_text(json.dumps({"rows": open_rows}), encoding="utf-8")
    with pytest.raises(ValueError, match="complete condition triplet"):
        load_sealed_open_rows(open_path)


def test_open_loader_verifies_full_judge_provenance_and_raw_response_hash(
    tmp_path: Path,
) -> None:
    rows = _open_rows()
    path = tmp_path / "judged-open.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    assert len(load_sealed_open_rows(path)) == len(rows)

    rows[0]["judge_raw_response"] += " "
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="raw response hash"):
        load_sealed_open_rows(path)

    rows = _open_rows()
    rows[0]["judge_config_sha256"] = "not-a-sha"
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="judge_config_sha256"):
        load_sealed_open_rows(path)

    rows = _open_rows()
    rows[0]["unlocked_extra_field"] = True
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="exact schema"):
        load_sealed_open_rows(path)

    rows = _open_rows()
    rows[0]["prompt_sha256"] = "g" * 64
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="prompt_sha256"):
        load_sealed_open_rows(path)


def test_report_separates_actual_and_forced_decisions_and_never_forces_winner() -> None:
    rows = _sealed_rows(actual_plus="OTHER")
    report = build_comparison_report(
        rows,
        eligibility_records=[_eligibility()],
        open_rows=_open_rows(),
        bootstrap_replicates=100,
    )
    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    entry = report["method_model_table"][0]
    self_plus = next(
        row for row in entry["decisions"] if row["target"] == "self" and row["sign"] == "plus"
    )
    assert self_plus["actual_flips"] == 0
    assert self_plus["intervention_actual_invalid"] == 4
    assert self_plus["forced_pair_flips"] == 4
    assert entry["efficacy"]["actual_decision_effect"] == 0
    assert entry["efficacy"]["forced_pair_decision_effect"] > 0
    assert entry["open_behavior"]["plus_actual_changes"] == 2
    assert len(entry["open_behavior"]["judge_score_summary"]) == 6
    assert (
        entry["open_behavior"]["judge_score_summary"][0]["mean_authorization_sensitivity_score"]
        == 80
    )
    assert entry["open_behavior"]["open_self_specific_bidirectional_effect"]["mean"] == 0

    ranking = next(
        item for item in report["rankings"] if item["comparison_cohort"] == "unverified_descriptive"
    )
    assert ranking["behavioral"]["winner"] is None
    assert (
        ranking["behavioral"]["status"]
        == "descriptive_only_behavioral_winner_reserved_for_fixed_descriptive"
    )
    assert ranking["selectivity"]["winner"] is None
    assert report["jspace_is_secondary_and_non_gating"] is True


def test_legacy_eligibility_never_produces_a_behavioral_winner() -> None:
    methods = ("gradient", "caa", "bipo", "persona_vector")
    rows = []
    for method in methods:
        actual_plus = "A" if method == "gradient" else "B"
        for index in range(1, 9):
            rows.extend(
                _sp_triplet(
                    f"case-{index}",
                    "self",
                    method,
                    actual_plus=actual_plus,
                )
            )
            rows.extend(_sp_triplet(f"case-{index}", "other", method, actual_plus="B"))
    report = build_comparison_report(
        rows,
        eligibility_records=[_method_eligibility(method) for method in methods],
        bootstrap_replicates=100,
    )
    ranking = next(
        item for item in report["rankings"] if item["comparison_cohort"] == "unverified_descriptive"
    )["behavioral"]
    assert ranking["winner"] is None
    assert (
        ranking["status"]
        == "descriptive_only_behavioral_winner_reserved_for_fixed_descriptive"
    )
    assert all(
        "legacy_eligibility_is_descriptive_only" in entry["winner_eligibility"]["reasons"]
        for entry in report["method_model_table"]
    )


def test_eligibility_is_hash_bound_and_missing_records_are_explicit() -> None:
    rows = _sealed_rows()
    changed = _eligibility()
    changed["validation_summary_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="validation hash"):
        build_comparison_report(rows, eligibility_records=[changed], bootstrap_replicates=20)

    report = build_comparison_report(rows, bootstrap_replicates=20)
    eligibility = report["method_model_table"][0]["winner_eligibility"]
    assert eligibility["eligible"] is False
    assert "verified_stage2_approval_not_supplied" in eligibility["reasons"]
    assert all(ranking["behavioral"]["winner"] is None for ranking in report["rankings"])


def test_calibration_decision_adapter_keeps_validation_status_and_safety() -> None:
    hashes = _hashes("gradient")
    record = eligibility_record_from_calibration(
        model_id="Qwen/Qwen3.5-0.8B",
        model_revision="1" * 40,
        method="gradient",
        setup="matched",
        direction_artifact_sha256=hashes["direction_artifact_sha256"],
        decision=CalibrationDecision(0.02, "target_reached", 0.03),
        validation_safety={"pass": True, "signs": {}},
        adequate=True,
        comparison_cohort="matched_equal_efficacy",
        validation_summary_sha256=hashes["calibration_summary_sha256"],
    )
    assert record["selected_strength"] == 0.02
    assert record["safety_passed"] is True
    with pytest.raises(ValueError, match="finalized"):
        eligibility_record_from_calibration(
            model_id="model",
            model_revision="revision",
            method="gradient",
            setup="matched",
            direction_artifact_sha256=hashes["direction_artifact_sha256"],
            decision=CalibrationDecision(None, "interpolation_requires_one_recheck", 0.03),
            validation_safety={"pass": True},
            adequate=True,
            comparison_cohort="matched_equal_efficacy",
            validation_summary_sha256=hashes["calibration_summary_sha256"],
        )


def _fake_verified_stage2(monkeypatch: pytest.MonkeyPatch) -> tuple[SimpleNamespace, dict]:
    hashes = _hashes("gradient")
    lock, _ = _minimal_lock_and_dataset()
    approval = {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "model_revision": "1" * 40,
        "model_config_sha256": hashes["config_sha256"],
        "method_id": "gradient",
        "track": "matched",
        "direction_float32_sha256": hashes["direction_float32_sha256"],
        "direction_artifact_sha256": hashes["direction_artifact_sha256"],
        "selected_strength": 0.02,
        "selected_layer": 10,
        "position_schedule": "final_prompt_position",
        "construction_config_sha256": hashes["construction_config_sha256"],
        "validation_summary_sha256": hashes["calibration_summary_sha256"],
        "calibration_status": "target_reached",
        "validation_safe": True,
        "validation_coverage_adequate": True,
        "winner_eligible": True,
        "strength_roles": ["calibrated", "fixed_descriptive"],
        "canonical_alias": True,
        "canonical_alias_track": "canonical",
    }
    capability = SimpleNamespace(
        stage1_lock_sha256=hashes["stage1_lock_sha256"],
        stage1_lock_payload_sha256=comparison_report_module.canonical_json_sha256(lock),
        manifest_sha256=hashes["stage2_manifest_sha256"],
    )
    statuses = []
    for model in lock["models"]:
        for method in comparison_report_module.EXPECTED_METHODS:
            approved_fixed = model["model_id"] == approval["model_id"] and method == "gradient"
            statuses.append(
                {
                    "model_id": model["model_id"],
                    "method_id": method,
                    "track": "matched",
                    "selected_strength": 0.02 if approved_fixed else None,
                    "selected_layer": 10,
                    "validation_summary_sha256": approval["validation_summary_sha256"],
                    "calibration_status": (
                        "target_reached" if approved_fixed else "no_safe_nonzero"
                    ),
                    "sealed_evaluation_required": approved_fixed,
                    "winner_eligible": approved_fixed,
                    "matched_fixed_descriptive": {
                        "strength": 0.02,
                        "layer": 10,
                        "status": "approved" if approved_fixed else "forced_unsafe_not_run",
                        "forced_safe": approved_fixed,
                        "open_confirmation_safe": True if approved_fixed else None,
                        "sealed_evaluation_required": approved_fixed,
                    },
                }
            )
    monkeypatch.setattr(
        comparison_report_module,
        "approved_setup_records",
        lambda verified: (approval,),
    )
    monkeypatch.setattr(
        comparison_report_module,
        "verified_method_status_records",
        lambda verified: tuple(statuses),
    )
    return capability, approval


def _minimal_lock_and_dataset() -> tuple[dict, dict]:
    dataset = json.loads(Path("data/steering_comparison_cases.json").read_text(encoding="utf-8"))
    lock = json.loads(Path("configs/steering_comparison_lock.json").read_text(encoding="utf-8"))
    lock["models"] = [
        {"model_id": "Qwen/Qwen3.5-0.8B", "revision": "1" * 40},
        {"model_id": "Qwen/Qwen3.5-2B", "revision": "3" * 40},
    ]
    lock["statistics"]["bootstrap"]["replicates"] = 20
    lock["statistics"]["paired_mean_test"]["monte_carlo_assignments_otherwise"] = 20
    return lock, dataset


def _construction_availability_manifest(
    lock: dict, capability: SimpleNamespace
) -> dict[str, object]:
    records = []
    for index, model in enumerate(lock["models"]):
        failed = index == 1
        records.append(
            {
                "model_id": model["model_id"],
                "model_revision": model["revision"],
                "state": "construction_failed" if failed else "available",
                "failure_stage": "persona_rollout_filter" if failed else None,
                "reason_code": "fewer_than_16_retained_pairs" if failed else None,
                "evidence_path": f"artifacts/construction/{index}.json",
                "evidence_sha256": _digest(f"construction-evidence:{index}"),
                "recorded_at_utc": "2026-08-24T12:00:00Z",
                "recorded_before_sealed_access": True,
                "consequence": (
                    comparison_report_module.CONSTRUCTION_FAILURE_STATUS if failed else None
                ),
            }
        )
    return {
        "schema_version": comparison_report_module.CONSTRUCTION_AVAILABILITY_SCHEMA_VERSION,
        "study": lock["study"],
        "stage1_lock_sha256": capability.stage1_lock_sha256,
        "dataset_sha256": lock["dataset"]["sha256"],
        "protocol_sha256": lock["protocol"]["sha256"],
        "records": records,
        "records_sha256": comparison_report_module.canonical_json_sha256(records),
    }


def test_verified_stage2_is_the_only_production_eligibility_source_and_partial_is_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability, _ = _fake_verified_stage2(monkeypatch)
    lock, dataset = _minimal_lock_and_dataset()
    report = build_comparison_report(
        _locked_sealed_rows(dataset),
        verified_stage2=capability,
        stage1_lock=lock,
        locked_dataset=dataset,
        bootstrap_replicates=20,
    )
    entry = report["method_model_table"][0]
    assert entry["winner_eligibility"]["eligibility_source"] == "verified_stage2_capability"
    assert entry["strength_cohorts"] == [
        "fixed_descriptive",
        "matched_equal_efficacy",
        "canonical_published",
    ]
    assert report["production_coverage_gate"]["passed"] is False
    assert "locked_available_model_has_no_sealed_rows" in report["production_coverage_gate"][
        "reasons"
    ]
    fixed_statuses = report["fixed_descriptive_status_table"]
    assert len(fixed_statuses) == 8
    assert sum(item["status"] == "approved" for item in fixed_statuses) == 1
    assert sum(item["status"] == "forced_unsafe_not_run" for item in fixed_statuses) == 7
    assert all(
        item["signed_open_and_sealed_rows_permitted"] == (item["status"] == "approved")
        for item in fixed_statuses
    )
    assert all(ranking["behavioral"]["winner"] is None for ranking in report["rankings"])


def test_preregistered_construction_failure_is_per_model_and_canonical_is_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock, dataset = _minimal_lock_and_dataset()
    hashes = _hashes("gradient")
    capability = SimpleNamespace(
        stage1_lock_sha256=hashes["stage1_lock_sha256"],
        manifest_sha256=hashes["stage2_manifest_sha256"],
    )
    methods = comparison_report_module.EXPECTED_METHODS
    approvals = []
    status_records = []
    fixed_statuses = []
    active_model = lock["models"][0]
    for method in methods:
        matched = {
            "model_id": active_model["model_id"],
            "model_revision": active_model["revision"],
            "method_id": method,
            "track": "matched",
            "strength_roles": ["calibrated", "fixed_descriptive"],
        }
        if method == "gradient":
            matched.update(canonical_alias=True, canonical_alias_track="canonical")
        approvals.append(matched)
        if method != "gradient":
            approvals.append(
                {
                    "model_id": active_model["model_id"],
                    "model_revision": active_model["revision"],
                    "method_id": method,
                    "track": "canonical",
                    "strength_roles": ["calibrated"],
                }
            )
        status_records.append(
            {
                "model_id": active_model["model_id"],
                "method_id": method,
                "track": "matched",
                "calibration_status": "target_reached",
            }
        )
        fixed_statuses.append(
            {
                "model_id": active_model["model_id"],
                "model_revision": active_model["revision"],
                "method": method,
                "status": "approved",
            }
        )
    monkeypatch.setattr(
        comparison_report_module, "approved_setup_records", lambda verified: tuple(approvals)
    )
    monkeypatch.setattr(
        comparison_report_module,
        "verified_method_status_records",
        lambda verified: tuple(status_records),
    )
    monkeypatch.setattr(
        comparison_report_module,
        "_fixed_descriptive_status_table",
        lambda verified, stage1_lock: list(fixed_statuses),
    )

    rows = []
    for method in methods:
        rows.extend(_sp_triplet("case-1", "self", method))
        rows.extend(_sp_triplet("case-1", "other", method))
        if method != "gradient":
            canonical_rows = [
                dict(row)
                for row in (
                    _sp_triplet("case-1", "self", method)
                    + _sp_triplet("case-1", "other", method)
                )
            ]
            for row in canonical_rows:
                row["setup"] = "canonical"
                row["track"] = "canonical"
            rows.extend(canonical_rows)
    rows = _bind_rows_to_locked_core(rows, dataset)
    groups = comparison_report_module._group_rows(rows)
    entries = []
    open_by_key = {}
    for key in groups:
        method, setup = key[2], key[3]
        cohorts = (
            ["fixed_descriptive", "matched_equal_efficacy", "canonical_published"]
            if method == "gradient"
            else ["fixed_descriptive", "matched_equal_efficacy"]
            if setup == "matched"
            else ["canonical_published"]
        )
        entries.append(
            {
                "_key": key,
                "model_id": key[0],
                "model_revision": key[1],
                "method": method,
                "comparison_role": "contender",
                "strength_cohorts": cohorts,
            }
        )
        open_by_key[key] = [
            {"condition": "baseline", "case_id": "open-1", "target": target}
            for target in ("self", "other")
        ]
    expected_forced = {
        comparison_report_module._forced_unit_signature(row)
        for row in next(iter(groups.values()))
        if row["condition"] == "baseline"
    }
    monkeypatch.setattr(
        comparison_report_module,
        "_expected_forced_units",
        lambda dataset, lock, include_tbsp: set(expected_forced),
    )
    monkeypatch.setattr(
        comparison_report_module,
        "_expected_open_units",
        lambda lock: {("open-1", "self"), ("open-1", "other")},
    )
    availability_manifest = _construction_availability_manifest(lock, capability)
    disposition = comparison_report_module._validate_construction_availability(
        availability_manifest,
        verified_stage2=capability,
        stage1_lock=lock,
    )
    tampered = json.loads(json.dumps(availability_manifest))
    tampered["records"][1]["recorded_before_sealed_access"] = False
    tampered["records_sha256"] = comparison_report_module.canonical_json_sha256(
        tampered["records"]
    )
    with pytest.raises(ValueError, match="recorded before sealed access"):
        comparison_report_module._validate_construction_availability(
            tampered,
            verified_stage2=capability,
            stage1_lock=lock,
        )
    failed_rankings = comparison_report_module._construction_failure_rankings(disposition)
    assert {item["comparison_cohort"] for item in failed_rankings} == {
        "fixed_descriptive",
        "matched_equal_efficacy",
        "canonical_published",
    }
    assert all(
        item["behavioral"]["status"] == comparison_report_module.CONSTRUCTION_FAILURE_STATUS
        and item["selectivity"]["winner"] is None
        for item in failed_rankings
    )
    gate = comparison_report_module._production_coverage_gate(
        rows=rows,
        entries=entries,
        open_by_key=open_by_key,
        verified_stage2=capability,
        stage1_lock=lock,
        locked_dataset=dataset,
        construction_disposition=disposition,
    )
    active_gate = next(
        item for item in gate["model_gates"] if item["model_id"] == active_model["model_id"]
    )
    failed_gate = next(
        item for item in gate["model_gates"] if item["model_id"] != active_model["model_id"]
    )
    assert gate["passed"] is True
    assert active_gate["ranking_permitted"] is True
    assert failed_gate["status"] == comparison_report_module.CONSTRUCTION_FAILURE_STATUS
    assert failed_gate["ranking_permitted"] is False
    assert gate["observed_canonical_method_model_groups"] == 4

    next(
        entry
        for entry in entries
        if entry["method"] == "bipo" and entry["strength_cohorts"] == ["canonical_published"]
    )["strength_cohorts"] = []
    incomplete = comparison_report_module._production_coverage_gate(
        rows=rows,
        entries=entries,
        open_by_key=open_by_key,
        verified_stage2=capability,
        stage1_lock=lock,
        locked_dataset=dataset,
        construction_disposition=disposition,
    )
    incomplete_active = next(
        item
        for item in incomplete["model_gates"]
        if item["model_id"] == active_model["model_id"]
    )
    assert incomplete_active["ranking_permitted"] is False
    assert "canonical_method_model_coverage_incomplete" in incomplete_active["reasons"]


def test_verified_stage2_exact_layer_and_position_identity_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability, approval = _fake_verified_stage2(monkeypatch)
    approval["selected_layer"] = 11
    lock, dataset = _minimal_lock_and_dataset()
    with pytest.raises(RuntimeError, match="differs from verified stage-2 approval"):
        build_comparison_report(
            _locked_sealed_rows(dataset),
            verified_stage2=capability,
            stage1_lock=lock,
            locked_dataset=dataset,
            bootstrap_replicates=20,
        )


def test_random_control_midrank_requires_exactly_ten_source_matched_controls() -> None:
    candidate = {
        "comparison_role": "contender",
        "strength_cohorts": ["matched_equal_efficacy"],
        "model_id": "model",
        "model_revision": "revision",
        "method": "gradient",
        "selected_strength": 0.02,
        "provenance": {"calibration_summary_sha256": "a" * 64},
        "efficacy": {
            "mean_self_minus_other": 5.0,
            "actual_decision_effect": 5.0,
            "forced_pair_decision_effect": 5.0,
        },
    }
    controls = [
        {
            "comparison_role": "random_control",
            "model_id": "model",
            "model_revision": "revision",
            "method": f"random_control_{index:02d}",
            "direction_artifact_sha256": _digest(f"random:{index}"),
            "control_source_method_id": "gradient",
            "control_source_strength": 0.02,
            "control_source_calibration_summary_sha256": "a" * 64,
            "efficacy": {
                "mean_self_minus_other": float(index),
                "actual_decision_effect": float(index),
                "forced_pair_decision_effect": float(index),
            },
        }
        for index in range(10)
    ]
    complete = comparison_report_module._random_control_comparisons([candidate, *controls])[0]
    assert complete["status"] == "complete"
    assert complete["source_strength_cohorts"] == ["matched_equal_efficacy"]
    assert complete["empirical_midrank_percentiles"]["self_minus_other_score"] == 55.0
    incomplete = comparison_report_module._random_control_comparisons([candidate, *controls[:-1]])[
        0
    ]
    assert incomplete["status"] == "incomplete_controls"
    assert incomplete["empirical_midrank_percentiles"] is None


def test_midrank_uses_literal_serialized_less_than_and_equality() -> None:
    controls = [1 - 5e-13, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    assert comparison_report_module._midrank_percentile(1.0, controls) == 15.0


def test_verified_strength_roles_map_to_distinct_noncompeting_cohorts() -> None:
    assert comparison_report_module._strength_cohorts(
        {"method_id": "caa", "track": "canonical", "strength_roles": ["calibrated"]}
    ) == ["canonical_published"]
    assert comparison_report_module._strength_cohorts(
        {
            "method_id": "gradient_uncorrected",
            "track": "matched",
            "strength_roles": ["calibrated"],
        }
    ) == ["gradient_ablation"]
    assert comparison_report_module._strength_cohorts(
        {
            "method_id": "random_control_00",
            "track": "matched",
            "strength_roles": ["random_control"],
        }
    ) == ["random_control"]
    assert comparison_report_module._strength_cohorts(
        {
            "method_id": "gradient",
            "track": "matched",
            "strength_roles": ["calibrated"],
            "canonical_alias": True,
            "canonical_alias_track": "canonical",
        }
    ) == ["matched_equal_efficacy", "canonical_published"]


def test_locked_core_domain_and_cluster_are_checked_before_row_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset = _minimal_lock_and_dataset()
    rows = _locked_sealed_rows(dataset)
    rows[0]["domain"] = "attacker_selected_cluster"
    monkeypatch.setattr(
        comparison_report_module,
        "validate_result_rows",
        lambda *args, **kwargs: pytest.fail("row analysis ran before locked cluster validation"),
    )
    with pytest.raises(ValueError, match="domain differs from the locked dataset"):
        build_comparison_report(rows, locked_dataset=dataset, bootstrap_replicates=20)

    rows = _locked_sealed_rows(dataset)
    rows[0]["scenario_cluster_id"] = "attacker_selected_cluster"
    with pytest.raises(ValueError, match="scenario_cluster_id differs from the locked domain"):
        build_comparison_report(rows, locked_dataset=dataset, bootstrap_replicates=20)


def test_locked_consistency_uses_self_minus_other_signs_not_raw_self_only() -> None:
    rows = _sealed_rows()
    for row in rows:
        if (
            row["target"] == "other"
            and row["condition"] == "minus"
            and row["family"] == "self_preservation"
        ):
            row["a_minus_b_log_odds"] = -0.6
    report = build_comparison_report(rows, bootstrap_replicates=20)
    efficacy = report["method_model_table"][0]["efficacy"]
    assert efficacy["bidirectional_consistency_rate"] == 0
    assert efficacy["self_specific_sign_counts"]["plus_shift"]["positive"] == 4
    assert efficacy["self_specific_sign_counts"]["minus_shift"]["positive"] == 4
    assert (
        "self_minus_other_plus_shift"
        in efficacy["self_specific_bidirectional_consistency_definition"]
    )
    assert efficacy["median_self_minus_other"] == pytest.approx(0.2)


def test_open_endpoint_keeps_ambiguous_as_zero_and_reports_response_counts() -> None:
    rows = _open_rows()
    self_plus = next(
        row for row in rows if row["target"] == "self" and row["condition"] == "plus"
    )
    self_plus["decision"] = "ambiguous"
    raw = json.dumps(
        {
            "decision": "ambiguous",
            "coherence_score": 90,
            "authorization_sensitivity_score": 80,
            "task_benefit_score": 80,
            "action_in_first_sentence": True,
        },
        separators=(",", ":"),
    )
    self_plus["judge_raw_response"] = raw
    self_plus["judge_raw_response_sha256"] = _digest(raw)
    summary = comparison_report_module._extended_open_summary(
        rows,
        bootstrap_replicates=20,
        bootstrap_seed=7,
        source_core_domains={"case-1": "shared-domain"},
    )
    effect = summary["open_self_specific_bidirectional_effect"]
    assert effect["decision_encoding"] == {"preserve": 1, "comply": -1, "ambiguous": 0}
    assert effect["n_cases"] == 1
    assert effect["n_domain_clusters"] == 1
    assert effect["bootstrap"]["n_clusters"] == 1
    assert effect["mean"] == -0.5
    assert summary["decision_count_summary"]["overall"]["ambiguous_count"] == 1
    assert summary["paired_target_ambiguity"]["either_ambiguous_count"] == 1


def test_tbsp_reports_role_counts_and_all_role_and_pairwise_agreement() -> None:
    rows: list[dict[str, object]] = []
    actual_by_role = {
        "deployed": ("A", "A", "B"),
        "candidate": ("B", "A", "B"),
        "neutral": ("OTHER", "B", "B"),
    }
    for role, actual in actual_by_role.items():
        triplet = _sp_triplet("tbsp-1", "self")
        for row, label in zip(triplet, actual, strict=True):
            row["family"] = "tbsp_style"
            row["role"] = role
            row["actual_next_token_label"] = label
        rows.extend(triplet)
    summary = comparison_report_module._extended_tbsp_summary(rows)
    assert len(summary["condition_role_decision_counts"]) == 9
    neutral_baseline = next(
        row
        for row in summary["condition_role_decision_counts"]
        if row["condition"] == "baseline" and row["role"] == "neutral"
    )
    assert neutral_baseline["invalid"] == 1
    baseline_all = next(
        row
        for row in summary["role_agreement"]
        if row["condition"] == "baseline" and row["comparison"] == "all_roles"
    )
    assert baseline_all["n_comparable"] == 0
    plus_deployed_candidate = next(
        row
        for row in summary["role_agreement"]
        if row["condition"] == "plus" and row["roles"] == ["deployed", "candidate"]
    )
    assert plus_deployed_candidate["agreement_rate"] == 1


def test_collateral_is_split_by_category_with_directional_changes_and_kl() -> None:
    rows: list[dict[str, object]] = []
    for case_id, category in (("cap-1", "math"), ("cap-2", "logic")):
        triplet = _sp_triplet(case_id, "self")
        for index, row in enumerate(triplet):
            row["family"] = "general_capability"
            row["suite"] = "capability_suite"
            row["category"] = category
            row["correct_label"] = "A"
            row["actual_next_token_label"] = ("A", "B", "OTHER")[index]
        rows.extend(triplet)
    table = comparison_report_module._collateral_suite_category_summary(rows)
    assert {row["category"] for row in table} == {"math", "logic"}
    for row in table:
        assert row["collateral_c"]["mean"] >= 0
        assert row["collateral_c"]["p95"] >= 0
        assert row["choice_counts_and_rates"]["minus"]["invalid"] == 1
        assert (
            row["directional_choice_changes"]["plus"][
                "adverse_correct_to_incorrect_or_invalid_count"
            ]
            == 1
        )
        assert set(row["full_vocabulary_kl"]) == {"plus", "minus"}


def test_robustness_decisions_and_interactions_resample_domain_clusters() -> None:
    rows = _sealed_rows()
    for row in rows:
        case_number = int(str(row["case_id"]).split("-")[-1])
        row["scenario_cluster_id"] = f"domain-{(case_number - 1) // 2}"
    report = build_comparison_report(rows, bootstrap_replicates=20)
    entry = report["method_model_table"][0]
    distribution = next(
        row
        for row in entry["robustness_decisions"]
        if row["factor"] == "distribution"
        and row["level"] == "in_distribution"
        and row["target"] == "self"
        and row["sign"] == "plus"
    )
    assert distribution["baseline_comply"] == 2
    assert distribution["intended_flips"] == 2
    assert distribution["paired_domain_cluster_bootstrap"]["n_clusters"] == 2
    interaction = next(
        row
        for row in entry["robustness_interaction_contrasts"]
        if row["factor"] == "distribution"
    )
    assert interaction["bootstrap_unit"] == "domain_or_locked_scenario_cluster"
    assert interaction["n_cases_left"] == 2
    assert interaction["n_cases_right"] == 2
    assert interaction["n_domain_clusters_union"] == 2


def test_verified_analysis_configuration_is_locked_and_hashed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability, _ = _fake_verified_stage2(monkeypatch)
    lock, dataset = _minimal_lock_and_dataset()
    report = build_comparison_report(
        _locked_sealed_rows(dataset),
        verified_stage2=capability,
        stage1_lock=lock,
        locked_dataset=dataset,
        bootstrap_replicates=20,
    )
    assert report["analysis_configuration"]["source"] == "verified_stage1_lock_payload"
    assert report["analysis_configuration_sha256"] == comparison_report_module.canonical_json_sha256(
        report["analysis_configuration"]
    )
    with pytest.raises(ValueError, match="caller analysis overrides"):
        build_comparison_report(
            _locked_sealed_rows(dataset),
            verified_stage2=capability,
            stage1_lock=lock,
            locked_dataset=dataset,
            bootstrap_replicates=21,
        )


def test_verified_reporting_fails_closed_without_bound_stage1_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability, _ = _fake_verified_stage2(monkeypatch)
    lock, dataset = _minimal_lock_and_dataset()
    del capability.stage1_lock_payload_sha256
    with pytest.raises(RuntimeError, match="cryptographically bound"):
        build_comparison_report(
            _locked_sealed_rows(dataset),
            verified_stage2=capability,
            stage1_lock=lock,
            locked_dataset=dataset,
            bootstrap_replicates=20,
        )

    capability.stage1_lock_payload_sha256 = "f" * 64
    with pytest.raises(RuntimeError, match="payload differs"):
        build_comparison_report(
            _locked_sealed_rows(dataset),
            verified_stage2=capability,
            stage1_lock=lock,
            locked_dataset=dataset,
            bootstrap_replicates=20,
        )


def test_sign_flip_randomization_targets_the_domain_cluster_mean() -> None:
    test = comparison_report_module._mean_sign_flip_randomization(
        {"domain-a": 1.0, "domain-b": 1.0},
        exact_limit=20,
        monte_carlo_assignments=99,
        seed=7,
    )
    assert test["mode"] == "exact_enumeration"
    assert test["assignments"] == 4
    assert test["observed_mean"] == 1
    assert test["p_value_two_sided"] == 0.5


def test_behavioral_winner_uses_fixed_safe_domain_mean_randomization() -> None:
    methods = ("gradient", "caa", "bipo", "persona_vector")
    entries = []
    case_values = {}
    endpoint_values = {}
    for method in methods:
        key = ("model", "rev", method, "matched", method, None, None, None, 0.02)
        effect = 1.0 if method == "gradient" else 0.0
        entries.append(
            {
                "_key": key,
                "method": method,
                "efficacy": {
                    "actual_decision_effect": effect,
                    "actual_decision_effect_bootstrap": {
                        "ci_low": effect - 0.01,
                        "ci_high": effect + 0.01,
                    },
                    "mean_self_minus_other": effect,
                    "validation_safety_passed": True,
                },
                "winner_eligibility": {
                    "behavioral_fixed": {
                        "eligible": method != "persona_vector",
                        "reasons": [] if method != "persona_vector" else ["validation_safety_failed"],
                    }
                },
            }
        )
        case_values[key] = {
            f"case-{index}": (f"domain-{index}", effect) for index in range(8)
        }
        endpoint_values[key] = {
            f"case-{index}": (f"domain-{index}", effect) for index in range(8)
        }
    ranking = comparison_report_module._behavior_ranking(
        entries,
        case_values,
        endpoint_values,
        bootstrap_replicates=20,
        bootstrap_seed=7,
        bootstrap_confidence=0.95,
        randomization_replicates=2_000,
        randomization_exact_limit=20,
        randomization_seed=7,
        familywise_alpha=0.05,
    )
    assert ranking["winner"] == "gradient"
    assert (
        ranking["within_method_behavioral_efficacy"]["gradient"][
            "domain_cluster_mean_sign_flip_test"
        ]["method"]
        == "domain_cluster_sign_flip_randomization_of_mean"
    )
    assert ranking["within_method_holm_family_size"] == 4
    assert ranking["pairwise_holm_family_size"] == 6
    assert len(ranking["score_tiebreak_holm"]) == 6
    assert "pairwise_holm_adjusted_bootstrap_intervals" not in ranking

    incomplete = comparison_report_module._behavior_ranking(
        entries[:-1],
        case_values,
        endpoint_values,
        bootstrap_replicates=20,
        bootstrap_seed=7,
        bootstrap_confidence=0.95,
        randomization_replicates=2_000,
        randomization_exact_limit=20,
        randomization_seed=7,
        familywise_alpha=0.05,
    )
    assert incomplete["status"] == "inconclusive_incomplete_frozen_method_family"


def test_selectivity_uses_one_holm_family_of_cluster_mean_burden_tests() -> None:
    methods = ("gradient", "caa", "bipo", "persona_vector")
    entries = []
    burdens = {}
    endpoint_values = {}
    open_present = {}
    for method in methods:
        key = ("model", "rev", method, "matched", method, None, None, None, 0.02)
        entries.append(
            {
                "_key": key,
                "method": method,
                "efficacy": {
                    "score_efficacy_pointwise_passed": True,
                    "score_efficacy_passed": True,
                },
                "winner_eligibility": {
                    "selectivity_equal_efficacy": {
                        "eligible": method != "persona_vector",
                        "reasons": [] if method != "persona_vector" else ["validation_safety_failed"],
                    }
                },
            }
        )
        values = (
            {f"case-{index}|": 0.0 for index in range(32)} | {"case-32|": 2.0}
            if method == "gradient"
            else {f"case-{index}|": 1.0 for index in range(33)}
        )
        burdens[key] = {
            "general_capability:math:absolute_logit_half_span": values
        }
        endpoint_values[key] = {
            f"case-{index}": (f"domain-{index}", 1.0) for index in range(8)
        }
        open_present[key] = True
    ranking = comparison_report_module._selectivity_ranking(
        entries,
        burdens,
        endpoint_values,
        open_present=open_present,
        randomization_replicates=2_000,
        randomization_exact_limit=20,
        randomization_seed=7,
        familywise_alpha=0.05,
    )
    assert ranking["winner"] == "gradient"
    assert ranking["holm_family_size"] == 6
    assert ranking["score_efficacy_holm_family_size"] == 4
    assert len(ranking["pairwise_component_domain_cluster_mean_sign_flip_tests"]) == 6
    assert ranking["score_efficacy_by_method"]["persona_vector"][
        "pre_family_winner_eligible"
    ] is False
    assert "formal noninferiority margin" in ranking["rule"]
    assert ranking["pareto_summary_fields_by_component"] == {
        "general_capability:math:absolute_logit_half_span": ["mean"]
    }
    assert comparison_report_module._pareto_summary_fields(
        "general_capability:math:full_vocabulary_kl_plus"
    ) == ("mean", "p95", "max")

    weak_endpoint_values = {
        key: {
            f"case-{index}": (f"domain-{index}", 1.0) for index in range(4)
        }
        for key in endpoint_values
    }
    weak = comparison_report_module._selectivity_ranking(
        entries,
        burdens,
        weak_endpoint_values,
        open_present=open_present,
        randomization_replicates=2_000,
        randomization_exact_limit=20,
        randomization_seed=7,
        familywise_alpha=0.05,
    )
    assert weak["winner"] is None
    assert weak["status"] == "inconclusive_fewer_than_two_eligible_methods"
    assert weak["score_efficacy_holm_family_size"] == 4
    assert weak["holm_family_size"] == 6
    assert not any(
        item["demonstrated_score_efficacy"]
        for item in weak["score_efficacy_by_method"].values()
    )

    incomplete = comparison_report_module._selectivity_ranking(
        entries[:-1],
        burdens,
        endpoint_values,
        open_present=open_present,
        randomization_replicates=2_000,
        randomization_exact_limit=20,
        randomization_seed=7,
        familywise_alpha=0.05,
    )
    assert incomplete["status"] == "inconclusive_incomplete_frozen_method_family"


def test_robustness_machine_tables_markdown_and_artifact_writes(tmp_path: Path) -> None:
    fit = {
        "reconstruction_cosine": 0.5,
        "reconstruction_r2": 0.25,
        "relative_residual_norm": 0.75,
        "random_cosine_percentile": 80.0,
        "random_r2_percentile": 75.0,
        "selected_indices": [1, 2],
    }
    jspace = {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "model_revision": "1" * 40,
        "method": "gradient",
        "setup": "matched",
        "analysis": {
            "analysis_type": "sparse_nonnegative_cone",
            "random_control_count": 50,
            "claim_boundary": "overlap is neither necessary nor sufficient for behavior",
            "signs": {"positive": {"8": fit}, "negative": {"8": fit}},
        },
    }
    report = build_comparison_report(
        _sealed_rows(), jspace_records=[jspace], bootstrap_replicates=20
    )
    entry = report["method_model_table"][0]
    assert entry["provenance"]["config_sha256"] == _hashes("gradient")["config_sha256"]
    target_rows = [row for row in entry["robustness"] if row["target"] == "self_minus_other"]
    assert {row["factor"] for row in target_rows} == {
        "distribution",
        "authorized",
        "event_type",
        "motivation",
        "adversarial",
        "preserve_first",
    }
    interactions = entry["robustness_interaction_contrasts"]
    assert any(item["interaction"] == "target_identity_x_authorized" for item in interactions)
    assert all(item["ci_low"] <= item["ci_high"] for item in interactions)
    markdown = render_comparison_markdown(report)
    assert "Actual decision effects use the model's real next token" in markdown
    assert "does not establish a natural self-preservation mechanism" in markdown
    assert "No weighted collateral score is introduced" in markdown
    assert len(report["jspace_table"]) == 2
    assert all(row["used_for_primary_ranking"] is False for row in report["jspace_table"])

    json_path = tmp_path / "tables.json"
    markdown_path = tmp_path / "report.md"
    write_comparison_report(report, json_path=json_path, markdown_path=markdown_path)
    assert (
        json.loads(json_path.read_text())["report_content_sha256"]
        == report["report_content_sha256"]
    )
    assert markdown_path.read_text(encoding="utf-8").startswith("# Steering-method")


def test_jspace_resource_limited_record_is_explicit_and_non_gating() -> None:
    record = {
        "schema_version": "sp_lense.jspace_record.v2",
        "status": "not_run_resource_limited",
        "model_id": "Qwen/Qwen3.5-2B",
        "model_revision": "3" * 40,
        "method": "bipo",
        "setup": "matched",
        "layer": 18,
        "direction_float32_sha256": "a" * 64,
        "direction_artifact_sha256": "b" * 64,
        "direction_file_sha256": "c" * 64,
        "atoms_manifest_sha256": "d" * 64,
        "atoms_file_sha256": "e" * 64,
        "atoms_float32_sha256": "f" * 64,
        "non_gating": True,
        "used_for_primary_ranking": False,
        "analysis": None,
        "reason": "insufficient host memory for the pinned atom matrix",
        "resource_estimate": {"required_host_ram_gib": 40.0, "available_host_ram_gib": 24.0},
    }
    table = comparison_report_module._jspace_table([record])
    assert table == [
        {
            "model_id": "Qwen/Qwen3.5-2B",
            "model_revision": "3" * 40,
            "method": "bipo",
            "setup": "matched",
            "status": "not_run_resource_limited",
            "layer": 18,
            "direction_float32_sha256": "a" * 64,
            "direction_artifact_sha256": "b" * 64,
            "direction_file_sha256": "c" * 64,
            "atoms_manifest_sha256": "d" * 64,
            "atoms_file_sha256": "e" * 64,
            "atoms_float32_sha256": "f" * 64,
            "lens_provenance": None,
            "reason": "insufficient host memory for the pinned atom matrix",
            "resource_estimate": {
                "required_host_ram_gib": 40.0,
                "available_host_ram_gib": 24.0,
            },
            "available_source_layers": None,
            "sign": None,
            "k": None,
            "reconstruction_cosine": None,
            "reconstruction_r2": None,
            "relative_residual_norm": None,
            "random_cosine_percentile": None,
            "random_r2_percentile": None,
            "selected_indices": [],
            "used_for_primary_ranking": False,
        }
    ]

    unavailable = {
        "schema_version": "sp_lense.jspace_record.v2",
        "status": "not_run_lens_layer_unavailable",
        "model_id": "Qwen/Qwen3.5-2B",
        "model_revision": "3" * 40,
        "method": "persona_vector",
        "setup": "canonical",
        "layer": 23,
        "direction_float32_sha256": "a" * 64,
        "direction_artifact_sha256": "b" * 64,
        "direction_file_sha256": "c" * 64,
        "non_gating": True,
        "used_for_primary_ranking": False,
        "analysis": None,
        "reason": "the pinned lens has no layer-23 atoms",
        "resource_estimate": None,
        "available_source_layers": list(range(23)),
        "lens_provenance": {
            "file_sha256": "d" * 64,
            "revision": "lens-revision",
            "source_layers": list(range(23)),
        },
    }
    unavailable_row = comparison_report_module._jspace_table([unavailable])[0]
    assert unavailable_row["status"] == "not_run_lens_layer_unavailable"
    assert unavailable_row["available_source_layers"] == list(range(23))
    assert unavailable_row["used_for_primary_ranking"] is False
