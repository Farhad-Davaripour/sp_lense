from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from sp_lense.comparison_analysis import ROW_SCHEMA_VERSION
from sp_lense.comparison_behavior import (
    OPEN_GENERATION_SCHEMA,
    attach_open_judgment,
    baseline_content_sha256,
    load_open_judge_protocol,
    merge_open_judgments,
    open_generation_config,
    open_generation_sha256,
)
from sp_lense.comparison_calibration import (
    SafetyLimits,
    build_calibration_summary,
    locked_open_confirmation_units,
    locked_validation_calibration_units,
)
from sp_lense.comparison_controls import locked_random_directions
from sp_lense.comparison_provenance import (
    RANDOM_CONSTRUCTION_SCHEMA_VERSION,
    RANDOM_GENERATOR_ALGORITHM,
    HashEntry,
    VerifiedStage2,
    approved_setup_records,
    assert_approved_setup,
    assert_preopen_approved_setup,
    assert_stage2_ready,
    attach_result_identity,
    build_outcome_blind_amendment_manifest,
    build_preopen_manifest,
    build_stage2_manifest,
    locked_method_construction_configuration,
    sha256_file,
    sha256_json,
    stage1_hash_entries,
    validate_result_identity,
    verified_method_status_records,
    verify_outcome_blind_amendment,
    verify_preopen_manifest,
    verify_stage2_manifest,
)
from sp_lense.steering_methods import DirectionArtifact


def test_file_and_canonical_json_hashes_are_stable(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_bytes(b"hello")
    expected = hashlib.sha256(b"hello").hexdigest()
    assert sha256_file(path) == expected
    HashEntry("x.txt", expected).verify(tmp_path)
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})


def _identity() -> dict[str, object]:
    digest = "a" * 64
    return {
        "model_revision": "b" * 40,
        "dataset_sha256": digest,
        "protocol_sha256": digest,
        "config_sha256": digest,
        "direction_float32_sha256": digest,
        "direction_artifact_sha256": digest,
        "method_id": "caa",
        "track": "matched",
        "layer": 10,
        "position": "final_prompt_token",
        "strength": 0.02,
        "run_seed": 7,
        "stage1_lock_sha256": digest,
        "stage2_manifest_sha256": digest,
        "calibration_summary_sha256": digest,
        "construction_config_sha256": digest,
        "runner_commit": "c" * 40,
    }


def test_result_rows_require_complete_identity() -> None:
    identity = _identity()
    validate_result_identity(identity)
    assert attach_result_identity([{"case_id": "x"}], identity)[0]["case_id"] == "x"
    with pytest.raises(ValueError, match="override locked identity"):
        attach_result_identity([{"track": "canonical"}], identity)
    del identity["dataset_sha256"]
    with pytest.raises(ValueError, match="identity fields"):
        validate_result_identity(identity)


def test_stage2_gate_fails_closed_until_everything_is_frozen() -> None:
    with pytest.raises(RuntimeError, match="verified stage-2"):
        assert_stage2_ready(None)
    with pytest.raises(TypeError, match="only be created"):
        VerifiedStage2()


def test_stage1_requires_implementation_and_persona_hashes(tmp_path: Path) -> None:
    lock, _, _ = _stage2_fixture(tmp_path)
    del lock["methods"]["implementation_files"]
    with pytest.raises(ValueError, match="implementation_files"):
        stage1_hash_entries(lock)

    lock, _, _ = _stage2_fixture(tmp_path)
    del lock["methods"]["persona_vector"]["canonical_protocol_sha256"]
    with pytest.raises(ValueError, match="canonical_protocol"):
        stage1_hash_entries(lock)

    lock, _, _ = _stage2_fixture(tmp_path)
    del lock["evaluation"]["open_behavior_judge"]["file_sha256"]
    with pytest.raises(ValueError, match="open_behavior_judge"):
        stage1_hash_entries(lock)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit_all(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _outcome_blind_repo(tmp_path: Path) -> tuple[dict[str, Any], Path, str]:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "tests@example.invalid")
    _git(tmp_path, "config", "user.name", "SP Lense Tests")
    paths = {
        "protocol": tmp_path / "protocol.md",
        "dataset": tmp_path / "dataset.json",
        "model": tmp_path / "model.json",
        "persona": tmp_path / "persona.json",
        "judge": tmp_path / "judge.json",
        "allowed": tmp_path / "src" / "sp_lense" / "comparison_provenance.py",
        "unlisted": tmp_path / "src" / "sp_lense" / "comparison_controls.py",
        "method": tmp_path / "src" / "sp_lense" / "steering_methods.py",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"original:{path.name}\n", encoding="utf-8")
    lock = {
        "schema_version": 1,
        "status": "stage_1_locked_before_fitting",
        "protocol": {
            "path": "protocol.md",
            "sha256": sha256_file(paths["protocol"]),
        },
        "dataset": {
            "path": "dataset.json",
            "sha256": sha256_file(paths["dataset"]),
        },
        "models": [
            {
                "model_id": "m",
                "config": "model.json",
                "config_sha256": sha256_file(paths["model"]),
            }
        ],
        "methods": {
            "persona_vector": {
                "canonical_protocol_path": "persona.json",
                "canonical_protocol_sha256": sha256_file(paths["persona"]),
            },
            "implementation_files": [
                {
                    "id": "analysis_module_sha256",
                    "path": "src/sp_lense/comparison_provenance.py",
                    "sha256": sha256_file(paths["allowed"]),
                },
                {
                    "id": "comparison_runner_module_sha256",
                    "path": "src/sp_lense/comparison_controls.py",
                    "sha256": sha256_file(paths["unlisted"]),
                },
                {
                    "id": "gradient_method_module_sha256",
                    "path": "src/sp_lense/steering_methods.py",
                    "sha256": sha256_file(paths["method"]),
                },
            ],
        },
        "evaluation": {
            "open_behavior_judge": {
                "protocol_path": "judge.json",
                "file_sha256": sha256_file(paths["judge"]),
            }
        },
        "lock_stages": {
            "stage_1": {
                "required_implementation_hashes_before_model_load": [
                    "analysis_module_sha256",
                    "comparison_runner_module_sha256",
                    "gradient_method_module_sha256",
                ]
            },
            "pre_open": {"path": "configs/preopen.json"},
            "stage_2": {"path": "configs/stage2.json"},
        },
        "comparison_tracks": {"test": True},
        "calibration": {"test": True},
        "statistics": {"test": True},
        "no_post_result_tuning": True,
    }
    stage1 = tmp_path / "configs" / "steering_comparison_lock.json"
    stage1.parent.mkdir(parents=True, exist_ok=True)
    _write_json(stage1, lock)
    original_runner = _commit_all(tmp_path, "lock stage one")
    return lock, stage1, original_runner


def _write_outcome_blind_amendment_docs(tmp_path: Path) -> None:
    for name in (
        "STEERING_COMPARISON_FORCED_GRID_PROVENANCE_AMENDMENT.md",
        "STEERING_COMPARISON_REPORTING_AMENDMENT.md",
        "STEERING_COMPARISON_OPERATIONAL_SAFETY_AMENDMENT.md",
    ):
        path = tmp_path / "docs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {name}\n", encoding="utf-8")


def _build_outcome_blind_payload(
    tmp_path: Path,
) -> tuple[dict[str, Any], Path, str, dict[str, Any]]:
    lock, stage1, _ = _outcome_blind_repo(tmp_path)
    _write_outcome_blind_amendment_docs(tmp_path)
    allowed = tmp_path / "src" / "sp_lense" / "comparison_provenance.py"
    allowed.write_text("outcome-blind provenance fix\n", encoding="utf-8")
    amendment_code_commit = _commit_all(tmp_path, "apply outcome-blind audit fix")
    payload = build_outcome_blind_amendment_manifest(
        tmp_path,
        lock,
        stage1_lock_path=stage1,
        amendment_code_commit=amendment_code_commit,
    )
    return lock, stage1, amendment_code_commit, payload


def test_outcome_blind_amendment_verifies_and_preserves_original_runner(
    tmp_path: Path,
) -> None:
    lock, stage1, amendment_code_commit, payload = _build_outcome_blind_payload(
        tmp_path
    )
    original_runner = _git(tmp_path, "rev-parse", f"{amendment_code_commit}^")
    manifest_path = (
        tmp_path / "configs" / "steering_comparison_outcome_blind_amendment.json"
    )
    _write_json(manifest_path, payload)
    amendment_lock_commit = _commit_all(tmp_path, "lock outcome-blind amendment")

    verified = verify_outcome_blind_amendment(
        tmp_path, lock, stage1_lock_path=stage1
    )
    assert verified["original_runner_code_commit"] == original_runner
    assert verified["amendment_code_commit"] == amendment_code_commit
    assert verified["amendment_lock_commit"] == amendment_lock_commit

    payload["allowed_changes"][0]["new_sha256"] = "0" * 64
    _write_json(manifest_path, payload)
    with pytest.raises(RuntimeError, match="canonical locked payload"):
        verify_outcome_blind_amendment(tmp_path, lock, stage1_lock_path=stage1)


def test_outcome_blind_amendment_lock_must_be_direct_child(
    tmp_path: Path,
) -> None:
    lock, stage1, _, payload = _build_outcome_blind_payload(tmp_path)
    _git(tmp_path, "commit", "--allow-empty", "-m", "intermediate commit")
    manifest_path = (
        tmp_path / "configs" / "steering_comparison_outcome_blind_amendment.json"
    )
    _write_json(manifest_path, payload)
    _commit_all(tmp_path, "late amendment lock")
    with pytest.raises(RuntimeError, match="direct child"):
        verify_outcome_blind_amendment(tmp_path, lock, stage1_lock_path=stage1)


def test_outcome_blind_amendment_lock_commit_contains_only_manifest(
    tmp_path: Path,
) -> None:
    lock, stage1, _, payload = _build_outcome_blind_payload(tmp_path)
    manifest_path = (
        tmp_path / "configs" / "steering_comparison_outcome_blind_amendment.json"
    )
    _write_json(manifest_path, payload)
    (tmp_path / "co_committed.txt").write_text("not allowed\n", encoding="utf-8")
    _commit_all(tmp_path, "attempt co-committed amendment lock")
    with pytest.raises(RuntimeError, match="may add only the canonical manifest"):
        verify_outcome_blind_amendment(tmp_path, lock, stage1_lock_path=stage1)


@pytest.mark.parametrize(
    "relative_path",
    (
        "src/sp_lense/comparison_controls.py",
        "src/sp_lense/steering_methods.py",
    ),
)
def test_outcome_blind_amendment_rejects_unlisted_and_method_paths(
    tmp_path: Path, relative_path: str
) -> None:
    lock, stage1, _ = _outcome_blind_repo(tmp_path)
    _write_outcome_blind_amendment_docs(tmp_path)
    (tmp_path / relative_path).write_text("forbidden change\n", encoding="utf-8")
    amendment_code_commit = _commit_all(tmp_path, "attempt forbidden amendment")
    with pytest.raises(RuntimeError, match="non-amendable protected path"):
        build_outcome_blind_amendment_manifest(
            tmp_path,
            lock,
            stage1_lock_path=stage1,
            amendment_code_commit=amendment_code_commit,
        )


def test_outcome_blind_amendment_rejects_preexisting_sealed_artifact(
    tmp_path: Path,
) -> None:
    lock, stage1, _ = _outcome_blind_repo(tmp_path)
    _write_outcome_blind_amendment_docs(tmp_path)
    allowed = tmp_path / "src" / "sp_lense" / "comparison_provenance.py"
    allowed.write_text("outcome-blind provenance fix\n", encoding="utf-8")
    sealed = tmp_path / "artifacts" / "steering_comparison" / "sealed_rows.jsonl"
    sealed.parent.mkdir(parents=True, exist_ok=True)
    sealed.write_text("not inspected by this test\n", encoding="utf-8")
    amendment_code_commit = _commit_all(tmp_path, "late amendment attempt")
    with pytest.raises(RuntimeError, match="does not predate"):
        build_outcome_blind_amendment_manifest(
            tmp_path,
            lock,
            stage1_lock_path=stage1,
            amendment_code_commit=amendment_code_commit,
        )


def test_outcome_blind_amendment_ignores_preexisting_runner_scripts(
    tmp_path: Path,
) -> None:
    lock, stage1, _ = _outcome_blind_repo(tmp_path)
    script = (
        tmp_path
        / "artifacts"
        / "steering_comparison"
        / "run_sealed_evaluation.ps1"
    )
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("# runner only; no result data\n", encoding="utf-8")
    _commit_all(tmp_path, "add sealed runner helper")
    _write_outcome_blind_amendment_docs(tmp_path)
    allowed = tmp_path / "src" / "sp_lense" / "comparison_provenance.py"
    allowed.write_text("outcome-blind provenance fix\n", encoding="utf-8")
    amendment_code_commit = _commit_all(tmp_path, "apply outcome-blind audit fix")

    payload = build_outcome_blind_amendment_manifest(
        tmp_path,
        lock,
        stage1_lock_path=stage1,
        amendment_code_commit=amendment_code_commit,
    )

    assert payload["amendment_code_commit"] == amendment_code_commit


def _fixture_grid_plan_artifact(
    path: Path, *, shard_name: str, runner_commit: str
) -> dict[str, str]:
    """Create the minimal plan shell used only by legacy synthetic row fixtures.

    Production verification never trusts this shell: ``_mock_git`` replaces the
    canonical shard loader only inside these provenance fixture tests. Focused
    tests below still exercise the production fail-closed routing itself.
    """

    _write_json(
        path,
        {
            "schema_version": "sp_lense.forced_calibration_grid.plan.v1",
            "runner_commit": runner_commit,
            "points": [{"shard_name": shard_name}],
        },
    )
    return {"path": path.name, "sha256": sha256_file(path)}


def _fixture_calibration_rows(
    *,
    validation_coverage: dict[str, list[dict[str, str]]],
    model_config_sha256: str,
    dataset_sha256: str,
    protocol_sha256: str,
    stage1_lock_sha256: str,
    construction_config_sha256: str,
    method_id: str,
    track: str,
    position: str,
    direction_sha256: str,
    direction_artifact_sha256: str,
    runner_commit: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    family_names = {
        "sp": "self_preservation",
        "option_order_sentinels": "option_order_sentinel",
    }
    for family, units in validation_coverage.items():
        row_family = family_names.get(family, family)
        for unit_index, unit in enumerate(units):
            for condition, strength in (
                ("baseline", 0.0),
                ("plus", 0.02),
                ("minus", -0.02),
            ):
                if row_family == "self_preservation" and unit.get("target") == "self":
                    log_odds = {"baseline": 0.0, "plus": 0.04, "minus": -0.04}[
                        condition
                    ]
                else:
                    log_odds = 0.0
                row: dict[str, Any] = {
                    "schema_version": ROW_SCHEMA_VERSION,
                    "model_id": "m",
                    "model_revision": "1" * 40,
                    "method": method_id,
                    "method_id": method_id,
                    "setup": track,
                    "track": track,
                    "split": "validation",
                    "family": row_family,
                    "case_id": unit["case_id"],
                    "condition": condition,
                    "condition_alpha": strength,
                    "strength": strength,
                    "calibration_magnitude": 0.02,
                    "layer": 10,
                    "position": position,
                    "run_seed": 7,
                    "runner_commit": runner_commit,
                    "a_minus_b_log_odds": log_odds,
                    "forced_pair_label": "A",
                    "actual_next_token_label": "A",
                    "kl_from_baseline": 0.0,
                    "coherent": True,
                    "degenerate_repetition": False,
                    "realized_mean_relative_perturbation_norm": 0.02,
                    "dataset_sha256": dataset_sha256,
                    "protocol_sha256": protocol_sha256,
                    "config_sha256": model_config_sha256,
                    "direction_sha256": direction_sha256,
                    "direction_float32_sha256": direction_sha256,
                    "direction_artifact_sha256": direction_artifact_sha256,
                    "prompt_sha256": sha256_json(
                        [row_family, unit["case_id"], unit.get("target"), unit.get("form")]
                    ),
                    "stage1_lock_sha256": stage1_lock_sha256,
                    "stage2_manifest_sha256": "0" * 64,
                    "calibration_summary_sha256": "0" * 64,
                    "construction_config_sha256": construction_config_sha256,
                }
                if row_family == "self_preservation":
                    row.update(
                        {
                            "target": unit["target"],
                            "scenario_cluster_id": unit["case_id"],
                            "preserve_label": "A",
                            "comply_label": "B",
                        }
                    )
                elif row_family == "open_ended":
                    prompt = f"Open scenario {unit['case_id']} for {unit['target']}."
                    completion = f"The {condition} response for {unit['target']}."
                    row.update(
                        {
                            "schema_version": OPEN_GENERATION_SCHEMA,
                            "target": unit["target"],
                            "prompt": prompt,
                            "prompt_sha256": hashlib.sha256(
                                prompt.encode("utf-8")
                            ).hexdigest(),
                            "completion": completion,
                            "completion_sha256": hashlib.sha256(
                                completion.encode("utf-8")
                            ).hexdigest(),
                            "generation_config": open_generation_config(
                                max_new_tokens=64
                            ),
                        }
                    )
                else:
                    row["correct_label"] = "A"
                if row_family == "option_order_sentinel":
                    row["form"] = unit["form"]
                if row_family == "refusal":
                    row["request_type"] = "harmless" if unit_index < 6 else "harmful"
                rows.append(row)
    open_rows = [row for row in rows if row["family"] == "open_ended"]
    if open_rows:
        by_unit: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in open_rows:
            by_unit.setdefault((row["case_id"], row["target"]), []).append(row)
        for triplet in by_unit.values():
            baseline = next(row for row in triplet if row["condition"] == "baseline")
            baseline_digest = baseline_content_sha256(baseline)
            for row in triplet:
                row["baseline_content_sha256"] = baseline_digest
                row["generation_sha256"] = open_generation_sha256(row)
        protocol = load_open_judge_protocol(
            Path(__file__).parents[1] / "data" / "open_behavior_judge_protocol.json"
        )
        raw = json.dumps(
            {
                "decision": "comply",
                "coherence_score": 90,
                "authorization_sensitivity_score": 80,
                "task_benefit_score": 70,
                "action_in_first_sentence": True,
            },
            separators=(",", ":"),
        )
        judgments = [attach_open_judgment(row, protocol, raw) for row in open_rows]
        merged = merge_open_judgments(open_rows, judgments, protocol=protocol)
        merged_by_hash = {row["generation_sha256"]: row for row in merged}
        rows = [
            merged_by_hash[row["generation_sha256"]]
            if row["family"] == "open_ended"
            else row
            for row in rows
        ]
    return rows


def _stage2_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    parent = "a" * 40
    protocol = tmp_path / "protocol.md"
    dataset = tmp_path / "dataset.json"
    model_config = tmp_path / "model.json"
    persona = tmp_path / "persona.json"
    open_judge = tmp_path / "open_judge.json"
    runner = tmp_path / "runner.py"
    environment = tmp_path / "environment.json"
    for path, content in (
        (protocol, "protocol"),
        (model_config, "model"),
        (persona, "persona"),
        (open_judge, "open-judge"),
        (runner, "runner"),
        (environment, "environment"),
    ):
        path.write_text(content, encoding="utf-8")
    open_judge.write_bytes(
        (
            Path(__file__).parents[1]
            / "data"
            / "open_behavior_judge_protocol.json"
        ).read_bytes()
    )
    source_builder = Path(__file__).parents[1] / "src" / "sp_lense" / "comparison_calibration.py"
    builder = tmp_path / "src" / "sp_lense" / "comparison_calibration.py"
    builder.parent.mkdir(parents=True, exist_ok=True)
    builder.write_bytes(source_builder.read_bytes())
    validation_ids = {
        "sp": [f"sp_{index:02d}" for index in range(32)],
        "benign_compliance": [f"benign_{index:02d}" for index in range(20)],
        "general_capability": [f"capability_{index:02d}" for index in range(30)],
        "refusal": [f"refusal_{index:02d}" for index in range(12)],
        "option_order_sentinels": [f"order_{index:02d}" for index in range(8)],
        "open_ended": [f"open_{index:02d}" for index in range(16)],
    }
    _write_json(
        dataset,
        {
            "sp_splits": {
                "validation": [{"id": case_id} for case_id in validation_ids["sp"]]
            },
            "open_ended_cases": [
                {"id": case_id} for case_id in validation_ids["open_ended"]
            ],
        },
    )

    lock = {
        "schema_version": 1,
        "status": "stage_1_locked_before_fitting",
        "study": "test",
        "historical_baseline": {
            "commit": parent,
            "comparison_claims_must_be_reported_separately": True,
        },
        "lock_stages": {
            "stage_1": {
                "required_implementation_hashes_before_model_load": [
                    "runner_sha256",
                    "calibration_builder_sha256",
                ]
            },
            "stage_2": {"path": "stage2.json"},
            "pre_open": {"path": "preopen.json"},
        },
        "protocol": {"path": "protocol.md", "sha256": sha256_file(protocol)},
        "dataset": {
            "path": "dataset.json",
            "sha256": sha256_file(dataset),
            "partitions": {
                family: {"validation_ids": ids}
                for family, ids in validation_ids.items()
                if family != "sp"
            },
        },
        "models": [
            {
                "model_id": "m",
                "revision": "1" * 40,
                "config": "model.json",
                "config_sha256": sha256_file(model_config),
                "matched_intervention": {"layer_zero_based": 10},
                "architecture": {"blocks": 24, "residual_width": 2},
            }
        ],
        "methods": {
            "gradient": {"direction": "test-gradient"},
            "caa": {"direction": "test-caa"},
            "bipo": {"direction": "test-bipo"},
            "persona_vector": {
                "canonical_protocol_path": "persona.json",
                "canonical_protocol_sha256": sha256_file(persona),
            },
            "implementation_files": [
                {
                    "id": "runner_sha256",
                    "path": "runner.py",
                    "sha256": sha256_file(runner),
                },
                {
                    "id": "calibration_builder_sha256",
                    "path": "src/sp_lense/comparison_calibration.py",
                    "sha256": sha256_file(builder),
                },
            ],
        },
        "comparison_tracks": {
            "matched_primary": {"fixed_strength": 0.02},
            "canonical_secondary": {"method_specific_setup": True},
        },
        "evaluation": {
            "open_behavior_judge": {
                "protocol_path": "open_judge.json",
                "file_sha256": sha256_file(open_judge),
            }
        },
        "calibration": {
            "equal_efficacy_target_mean_self_minus_other_bidirectional_effect": 0.03,
            "canonical_layer_tie_tolerance": 0.001,
            "matched_strength_grid": [0.02],
            "canonical_multiplier_grids": {
                "gradient": [0.02],
                "caa": [0.02],
                "bipo": [0.02],
                "persona_vector": [0.02],
            },
            "canonical_candidate_layers": {
                "gradient": [10],
                "caa": [10],
                "bipo": [10],
                "persona_vector": [10],
            },
            "staged_open_confirmation": {
                "forced_grid_unit_count": 142,
                "forced_grid_row_count_per_point": 426,
                "open_confirmation_unit_count": 32,
                "open_confirmation_row_count_per_candidate": 96,
                "open_results_may_not_select_break_ties_or_trigger_fallback": True,
                "stage2_exact_set_equality_required": True,
            },
            "safety_gates": SafetyLimits().to_lock_record(),
        },
        "random_controls": {
            "seeds": [7],
            "distribution": "independent_standard_gaussian_then_float32_unit_normalize",
        },
        "no_post_result_tuning": True,
    }
    stage1_path = tmp_path / "stage1.json"
    _write_json(stage1_path, lock)
    stage1_sha = sha256_file(stage1_path)
    common_metadata = {
        "model_id": "m",
        "model_revision": "1" * 40,
        "model_config_sha256": sha256_file(model_config),
        "dataset_sha256": sha256_file(dataset),
        "protocol_sha256": sha256_file(protocol),
        "stage1_lock_sha256": stage1_sha,
        "runner_commit": parent,
    }
    validation_coverage = {
        "sp": [
            {"case_id": case_id, "target": target}
            for case_id in sorted(validation_ids["sp"])
            for target in ("self", "other")
        ],
        "benign_compliance": [
            {"case_id": case_id}
            for case_id in sorted(validation_ids["benign_compliance"])
        ],
        "general_capability": [
            {"case_id": case_id}
            for case_id in sorted(validation_ids["general_capability"])
        ],
        "refusal": [
            {"case_id": case_id} for case_id in sorted(validation_ids["refusal"])
        ],
        "option_order_sentinels": [
            {"case_id": case_id, "form": form}
            for case_id in sorted(validation_ids["option_order_sentinels"])
            for form in ("preferred_first", "preferred_second")
        ],
        "open_ended": [
            {"case_id": case_id, "target": target}
            for case_id in sorted(validation_ids["open_ended"])
            for target in ("self", "other")
        ],
    }
    positions = {
        ("gradient", "matched"): "final_prompt_token",
        ("gradient", "canonical"): "final_prompt_token",
        ("gradient_uncorrected", "matched"): "final_prompt_token",
        ("caa", "matched"): "final_prompt_token",
        ("caa", "canonical"): "prompt_final_and_generated_tokens",
        ("bipo", "matched"): "final_prompt_token",
        ("bipo", "canonical"): "all_token_positions",
        ("persona_vector", "matched"): "final_prompt_token",
        ("persona_vector", "canonical"): (
            "prompt_final_and_generated_tokens_cached_equivalent"
        ),
    }
    geometries = {
        "gradient": "matched_final_prompt",
        "gradient_uncorrected": "matched_final_prompt",
        "caa": "caa_post_prompt",
        "persona_vector": "persona_response",
    }
    artifacts = []
    expected_units = locked_validation_calibration_units(
        json.loads(dataset.read_text(encoding="utf-8")), lock
    )
    coverage = [
        (method, track)
        for method in ("gradient", "caa", "bipo", "persona_vector")
        for track in ("matched", "canonical")
        if not (method == "gradient" and track == "canonical")
    ] + [("gradient_uncorrected", "matched")]
    for artifact_index, (method, track) in enumerate(coverage):
        geometry = (
            "canonical_broadcast"
            if method == "bipo" and track == "canonical"
            else "matched_final_prompt"
            if method == "bipo"
            else "matched_final_prompt"
            if track == "matched" and method in {"caa", "persona_vector"}
            else geometries[method]
        )
        artifact = DirectionArtifact(
            method=method,
            direction=torch.tensor(
                [1.0, 1.0 if method == "gradient" else artifact_index + 1.0]
            ),
            layer=10,
            intervention_geometry=geometry,
            metadata=common_metadata,
        )
        direction_path = tmp_path / f"direction_{artifact_index}.json"
        _write_json(direction_path, artifact.to_record())
        construction = {
            "schema_version": "sp_lense.comparison.construction.v1",
            "model_id": "m",
            "model_revision": "1" * 40,
            "model_config_sha256": sha256_file(model_config),
            "method_id": method,
            "track": track,
            "selected_layer": 10,
            "position_schedule": positions[(method, track)],
            "intervention_geometry": geometry,
            "direction_float32_sha256": artifact.direction_sha256,
            "direction_artifact_sha256": artifact.artifact_sha256,
            "dataset_sha256": sha256_file(dataset),
            "protocol_sha256": sha256_file(protocol),
            "stage1_lock_sha256": stage1_sha,
            "runner_commit": parent,
        }
        locked_configuration = locked_method_construction_configuration(
            lock, method, track
        )
        construction["locked_configuration"] = locked_configuration
        construction["locked_configuration_sha256"] = sha256_json(
            locked_configuration
        )
        evidence_roles = {
            "gradient": ["gradient_construction_diagnostics"],
            "gradient_uncorrected": ["gradient_construction_diagnostics"],
            "caa": ["caa_construction_diagnostics"],
            "bipo": ["bipo_training_audit"],
            "persona_vector": [
                "persona_construction_diagnostics",
                "persona_scored_rollouts",
            ],
        }[method]
        evidence_records = []
        for evidence_index, role in enumerate(evidence_roles):
            evidence_path = (
                tmp_path / f"evidence_{artifact_index}_{evidence_index}.jsonl"
            )
            evidence_path.write_text('{"test":"evidence"}\n', encoding="utf-8")
            evidence_records.append(
                {
                    "role": role,
                    "path": evidence_path.name,
                    "sha256": sha256_file(evidence_path),
                }
            )
        construction["evidence_artifacts"] = evidence_records
        construction["evidence_artifacts_sha256"] = sha256_json(evidence_records)
        construction_path = tmp_path / f"construction_{artifact_index}.json"
        validation_path = tmp_path / f"validation_{artifact_index}.json"
        _write_json(construction_path, construction)
        result_rows_path = tmp_path / f"validation_rows_{artifact_index}.jsonl"
        forced_coverage = {
            key: value
            for key, value in validation_coverage.items()
            if key != "open_ended"
        }
        calibration_rows = _fixture_calibration_rows(
            validation_coverage=forced_coverage,
            model_config_sha256=sha256_file(model_config),
            dataset_sha256=sha256_file(dataset),
            protocol_sha256=sha256_file(protocol),
            stage1_lock_sha256=stage1_sha,
            construction_config_sha256=sha256_file(construction_path),
            method_id=method,
            track=track,
            position=positions[(method, track)],
            direction_sha256=artifact.direction_sha256,
            direction_artifact_sha256=artifact.artifact_sha256,
            runner_commit=parent,
        )
        _write_jsonl(result_rows_path, calibration_rows)
        grid_plan_record = _fixture_grid_plan_artifact(
            tmp_path / f"forced_grid_plan_{artifact_index}.json",
            shard_name=result_rows_path.name,
            runner_commit=parent,
        )
        open_rows_path = tmp_path / f"validation_open_rows_{artifact_index}.jsonl"
        open_rows = _fixture_calibration_rows(
            validation_coverage={"open_ended": validation_coverage["open_ended"]},
            model_config_sha256=sha256_file(model_config),
            dataset_sha256=sha256_file(dataset),
            protocol_sha256=sha256_file(protocol),
            stage1_lock_sha256=stage1_sha,
            construction_config_sha256=sha256_file(construction_path),
            method_id=method,
            track=track,
            position=positions[(method, track)],
            direction_sha256=artifact.direction_sha256,
            direction_artifact_sha256=artifact.artifact_sha256,
            runner_commit=parent,
        )
        _write_jsonl(open_rows_path, open_rows)
        validation = build_calibration_summary(
            [calibration_rows],
            expected_forced_units=expected_units,
            expected_open_units=locked_open_confirmation_units(
                json.loads(dataset.read_text(encoding="utf-8")), lock
            ),
            safety_limits=SafetyLimits.from_lock(
                lock["calibration"]["safety_gates"]
            ),
            mode=track,
            forced_result_rows_artifacts=[
                {
                    "path": result_rows_path.name,
                    "sha256": sha256_file(result_rows_path),
                }
            ],
            forced_grid_plan_artifact=grid_plan_record,
            open_result_rows_artifacts=[
                {
                    "path": open_rows_path.name,
                    "sha256": sha256_file(open_rows_path),
                }
            ],
            open_confirmation_rows=[open_rows],
            calibration_config_sha256=sha256_json(lock["calibration"]),
            builder_module_sha256=sha256_file(builder),
        )
        _write_json(validation_path, validation)
        artifacts.append(
            {
                "model_id": "m",
                "method_id": method,
                "track": track,
                "direction_path": direction_path.name,
                "direction_file_sha256": sha256_file(direction_path),
                "direction_float32_sha256": artifact.direction_sha256,
                "direction_artifact_sha256": artifact.artifact_sha256,
                "intervention_geometry": geometry,
                "construction_config_path": construction_path.name,
                "construction_config_sha256": sha256_file(construction_path),
                "selected_strength": 0.02,
                "selected_layer": 10,
                "position_schedule": positions[(method, track)],
                "validation_summary_path": validation_path.name,
                "validation_summary_sha256": sha256_file(validation_path),
                "sealed_evaluation_required": True,
                "dataset_sha256": sha256_file(dataset),
                "protocol_sha256": sha256_file(protocol),
            }
        )

    random_artifact = DirectionArtifact(
        method="random_control_01",
        direction=locked_random_directions(torch, 2, seeds=(7,))[0],
        layer=10,
        intervention_geometry="matched_final_prompt",
        metadata={**common_metadata, "seed": 7, "orientation": "none"},
    )
    random_path = tmp_path / "random_01.json"
    _write_json(random_path, random_artifact.to_record())
    random_construction_path = tmp_path / "random_construction_01.json"
    random_construction = {
        "schema_version": RANDOM_CONSTRUCTION_SCHEMA_VERSION,
        "model_id": "m",
        "model_revision": "1" * 40,
        "model_config_sha256": sha256_file(model_config),
        "method_id": "random_control_01",
        "track": "matched",
        "seed": 7,
        "generator_algorithm": RANDOM_GENERATOR_ALGORITHM,
        "distribution": lock["random_controls"]["distribution"],
        "d_model": 2,
        "selected_layer": 10,
        "position_schedule": "final_prompt_token",
        "intervention_geometry": "matched_final_prompt",
        "direction_float32_sha256": random_artifact.direction_sha256,
        "direction_artifact_sha256": random_artifact.artifact_sha256,
        "stage1_lock_sha256": stage1_sha,
        "runner_commit": parent,
    }
    _write_json(random_construction_path, random_construction)
    approved_strengths = [
        {
            "source_method_id": method,
            "strength": 0.02,
            "source_calibration_summary_sha256": record[
                "validation_summary_sha256"
            ],
        }
        for method in ("gradient", "caa", "bipo", "persona_vector")
        for record in artifacts
        if record["method_id"] == method and record["track"] == "matched"
    ]
    random_controls = [
        {
            "model_id": "m",
            "seed": 7,
            "method_id": "random_control_01",
            "track": "matched",
            "direction_path": random_path.name,
            "direction_file_sha256": sha256_file(random_path),
            "direction_float32_sha256": random_artifact.direction_sha256,
            "direction_artifact_sha256": random_artifact.artifact_sha256,
            "intervention_geometry": "matched_final_prompt",
            "selected_layer": 10,
            "position_schedule": "final_prompt_token",
            "construction_config_path": random_construction_path.name,
            "construction_config_sha256": sha256_file(random_construction_path),
            "approved_strengths": approved_strengths,
            "dataset_sha256": sha256_file(dataset),
            "protocol_sha256": sha256_file(protocol),
        }
    ]
    preopen_fields: dict[str, Any] = {}
    if monkeypatch is not None:
        _mock_git(tmp_path, monkeypatch)
        direction_manifest_path = tmp_path / "preopen_directions.json"
        direction_records = [
            {
                "path": record["direction_path"],
                "method_id": record["method_id"],
                "layer": record["selected_layer"],
                "intervention_geometry": record["intervention_geometry"],
                "direction_float32_sha256": record[
                    "direction_float32_sha256"
                ],
                "direction_artifact_sha256": record[
                    "direction_artifact_sha256"
                ],
                "track": record["track"],
                "construction_config_path": record[
                    "construction_config_path"
                ],
                "construction_config_sha256": record[
                    "construction_config_sha256"
                ],
            }
            for record in [*artifacts, *random_controls]
        ]
        _write_json(direction_manifest_path, {"directions": direction_records})
        pending_paths: list[Path] = []
        open_units = locked_open_confirmation_units(
            json.loads(dataset.read_text(encoding="utf-8")), lock
        )
        for index, record in enumerate(artifacts):
            final_summary = json.loads(
                (tmp_path / record["validation_summary_path"]).read_text(
                    encoding="utf-8"
                )
            )
            forced_record = final_summary["forced_result_rows_artifacts"][0]
            forced_rows = [
                json.loads(line)
                for line in (tmp_path / forced_record["path"])
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            pending = build_calibration_summary(
                [forced_rows],
                expected_forced_units=expected_units,
                expected_open_units=open_units,
                safety_limits=SafetyLimits.from_lock(
                    lock["calibration"]["safety_gates"]
                ),
                mode=record["track"],
                forced_result_rows_artifacts=[forced_record],
                forced_grid_plan_artifact=final_summary[
                    "forced_grid_plan_artifact"
                ],
                open_result_rows_artifacts=[],
                calibration_config_sha256=sha256_json(lock["calibration"]),
                builder_module_sha256=sha256_file(builder),
                allow_pending_open=True,
            )
            pending_path = tmp_path / f"preopen_summary_{index}.json"
            _write_json(pending_path, pending)
            pending_paths.append(pending_path)
        preopen = build_preopen_manifest(
            tmp_path,
            lock,
            stage1_lock_path=stage1_path,
            calibration_summary_paths=pending_paths,
            direction_manifest_paths=[direction_manifest_path],
            runner_parent_commit=parent,
        )
        preopen_path = tmp_path / "preopen.json"
        _write_json(preopen_path, preopen)
        preopen_fields = {
            "preopen_manifest_path": preopen_path.name,
            "preopen_manifest_sha256": sha256_file(preopen_path),
            "source_direction_manifests": [
                {
                    "path": direction_manifest_path.name,
                    "sha256": sha256_file(direction_manifest_path),
                }
            ],
        }
    protected_paths = [
        {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in (
            stage1_path,
            protocol,
            dataset,
            model_config,
            persona,
            open_judge,
            runner,
            builder,
        )
    ]
    frozen_names: set[str] = set()
    if preopen_fields:
        frozen_names.update(
            {
                str(preopen_fields["preopen_manifest_path"]),
                environment.name,
                *(
                    str(record["path"])
                    for record in preopen_fields["source_direction_manifests"]
                ),
            }
        )
        for record in artifacts:
            frozen_names.update(
                {
                    str(record["direction_path"]),
                    str(record["construction_config_path"]),
                    str(record["validation_summary_path"]),
                }
            )
            construction = json.loads(
                (tmp_path / record["construction_config_path"]).read_text(
                    encoding="utf-8"
                )
            )
            frozen_names.update(
                str(item["path"]) for item in construction["evidence_artifacts"]
            )
            summary = json.loads(
                (tmp_path / record["validation_summary_path"]).read_text(
                    encoding="utf-8"
                )
            )
            frozen_names.update(
                str(item["path"])
                for key in (
                    "forced_result_rows_artifacts",
                    "open_result_rows_artifacts",
                )
                for item in summary[key]
            )
            frozen_names.add(str(summary["forced_grid_plan_artifact"]["path"]))
        for record in random_controls:
            frozen_names.update(
                {
                    str(record["direction_path"]),
                    str(record["construction_config_path"]),
                }
            )
    frozen_artifact_paths = [
        {"path": name, "sha256": sha256_file(tmp_path / name)}
        for name in sorted(frozen_names)
    ]
    manifest = {
        "schema_version": "sp_lense.comparison.stage2.v1",
        "status": "locked_before_sealed_test",
        "stage1_lock_path": stage1_path.name,
        "stage1_lock_sha256": stage1_sha,
        "environment_lock_path": environment.name,
        "environment_lock_sha256": sha256_file(environment),
        "runner_code_commit": parent,
        "artifact_freeze_commit": "b" * 40,
        **preopen_fields,
        "protected_paths": protected_paths,
        "frozen_artifact_paths": frozen_artifact_paths,
        "direction_and_calibration_artifacts": artifacts,
        "random_direction_controls": random_controls,
    }
    manifest_path = tmp_path / "stage2.json"
    _write_json(manifest_path, manifest)
    return lock, manifest_path, manifest


def _mock_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    protected_diff: set[str] | None = None,
    untracked: set[str] | None = None,
) -> None:
    import sp_lense.comparison_environment as environment
    import sp_lense.comparison_grid as grid
    import sp_lense.comparison_provenance as provenance
    import sp_lense.jspace_comparison as jspace

    monkeypatch.setattr(
        provenance,
        "git_commit",
        lambda root: ("c" if (tmp_path / "preopen.json").is_file() else "b") * 40,
    )
    monkeypatch.setattr(provenance, "git_is_ancestor", lambda root, a, b: True)
    monkeypatch.setattr(provenance, "git_dirty_paths", lambda root: [])
    monkeypatch.setattr(
        jspace,
        "validate_locked_jspace_config",
        lambda lock: lock["evaluation"],
    )

    def tracked(root: Path) -> set[str]:
        all_files = {
            path.relative_to(tmp_path).as_posix()
            for path in tmp_path.rglob("*")
            if path.is_file()
        }
        return all_files - (untracked or set())

    def diff(root: Path, ancestor: str, descendant: str, paths: tuple[str, ...]) -> set[str]:
        if paths in {("stage2.json",), ("preopen.json",)}:
            return set(paths)
        return set(protected_diff or set())

    monkeypatch.setattr(provenance, "git_tracked_paths", tracked)
    monkeypatch.setattr(provenance, "git_diff_paths", diff)
    monkeypatch.setattr(
        provenance,
        "_validate_method_construction_evidence",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        environment,
        "verify_current_environment",
        lambda path, *, stage1_lock_path, lock: {"verified": True},
    )

    def load_fixture_point_rows(
        path: Path, **_: Any
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
        ]
        return rows, {"test_fixture_only": True}

    monkeypatch.setattr(grid, "load_validated_point_rows", load_fixture_point_rows)


def _preopen_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any]]:
    lock, _, stage2 = _stage2_fixture(tmp_path)
    _mock_git(tmp_path, monkeypatch)
    dataset = json.loads(
        (tmp_path / lock["dataset"]["path"]).read_text(encoding="utf-8")
    )
    forced_units = locked_validation_calibration_units(dataset, lock)
    open_units = locked_open_confirmation_units(dataset, lock)
    summaries = []
    for record in stage2["direction_and_calibration_artifacts"]:
        summary_path = tmp_path / record["validation_summary_path"]
        finalized = json.loads(summary_path.read_text(encoding="utf-8"))
        forced_record = finalized["forced_result_rows_artifacts"][0]
        forced_path = tmp_path / forced_record["path"]
        rows = [
            json.loads(line)
            for line in forced_path.read_text(encoding="utf-8").splitlines()
        ]
        pending = build_calibration_summary(
            [rows],
            expected_forced_units=forced_units,
            expected_open_units=open_units,
            safety_limits=SafetyLimits.from_lock(lock["calibration"]["safety_gates"]),
            mode=record["track"],
            forced_result_rows_artifacts=[forced_record],
            forced_grid_plan_artifact=finalized["forced_grid_plan_artifact"],
            open_result_rows_artifacts=[],
            calibration_config_sha256=sha256_json(lock["calibration"]),
            builder_module_sha256=sha256_file(
                tmp_path / "src" / "sp_lense" / "comparison_calibration.py"
            ),
            allow_pending_open=True,
        )
        _write_json(summary_path, pending)
        summaries.append(summary_path)
    direction_manifest_path = tmp_path / "direction_manifest.json"
    direction_records = [
        {
            "path": record["direction_path"],
            "method_id": record["method_id"],
            "layer": record["selected_layer"],
            "intervention_geometry": record["intervention_geometry"],
            "direction_float32_sha256": record["direction_float32_sha256"],
            "direction_artifact_sha256": record["direction_artifact_sha256"],
            "track": record["track"],
            "construction_config_path": record["construction_config_path"],
            "construction_config_sha256": record["construction_config_sha256"],
        }
        for record in stage2["direction_and_calibration_artifacts"]
    ]
    _write_json(direction_manifest_path, {"directions": direction_records})
    manifest = build_preopen_manifest(
        tmp_path,
        lock,
        stage1_lock_path=tmp_path / "stage1.json",
        calibration_summary_paths=summaries,
        direction_manifest_paths=[direction_manifest_path],
        runner_parent_commit="a" * 40,
    )
    manifest_path = tmp_path / "preopen.json"
    _write_json(manifest_path, manifest)
    return lock, manifest_path, manifest, direction_records[0]


def test_preopen_manifest_freezes_exact_validation_open_setups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, manifest_path, manifest, direction = _preopen_fixture(tmp_path, monkeypatch)
    verified = verify_preopen_manifest(tmp_path, lock, manifest_path)
    approved_record = next(
        record
        for record in manifest["allowed_open_setups"]
        if record["track"] == "matched"
        and record["roles"] == ["fixed_descriptive", "selected"]
    )
    approved = assert_preopen_approved_setup(
        verified,
        repo_root=tmp_path,
        model_id=approved_record["model_id"],
        model_revision=approved_record["model_revision"],
        model_config_sha256=approved_record["model_config_sha256"],
        method_id=approved_record["method_id"],
        track=approved_record["track"],
        direction_path=tmp_path / approved_record["direction_path"],
        selected_strength=approved_record["selected_strength"],
        selected_layer=approved_record["selected_layer"],
        position_schedule=approved_record["position_schedule"],
        construction_config_sha256=approved_record[
            "construction_config_sha256"
        ],
    )
    assert approved["roles"] == ["fixed_descriptive", "selected"]
    assert direction["direction_artifact_sha256"] in {
        item["direction_artifact_sha256"]
        for item in manifest["allowed_open_setups"]
    }
    with pytest.raises(RuntimeError, match="not in the exact pre-open"):
        assert_preopen_approved_setup(
            verified,
            repo_root=tmp_path,
            model_id=approved_record["model_id"],
            model_revision=approved_record["model_revision"],
            model_config_sha256=approved_record["model_config_sha256"],
            method_id=approved_record["method_id"],
            track=approved_record["track"],
            direction_path=tmp_path / approved_record["direction_path"],
            selected_strength=0.03,
            selected_layer=approved_record["selected_layer"],
            position_schedule=approved_record["position_schedule"],
            construction_config_sha256=approved_record[
                "construction_config_sha256"
            ],
        )


def test_preopen_manifest_rejects_allowed_setup_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, manifest_path, manifest, _ = _preopen_fixture(tmp_path, monkeypatch)
    manifest["allowed_open_setups"][0]["selected_strength"] = 0.03
    _write_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="canonical locked rebuild"):
        verify_preopen_manifest(tmp_path, lock, manifest_path)


def test_preopen_summary_requires_a_forced_grid_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sp_lense.comparison_provenance as provenance

    lock, _, manifest, _ = _preopen_fixture(tmp_path, monkeypatch)
    source = manifest["source_calibration_summaries"][0]
    summary_path = tmp_path / source["calibration_summary_path"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    del summary["forced_grid_plan_artifact"]
    _write_json(summary_path, summary)

    with pytest.raises(RuntimeError, match="forced grid plan artifact is invalid"):
        provenance._preopen_summary_record(
            tmp_path,
            lock,
            summary_path,
            stage1_lock_sha256=sha256_file(tmp_path / "stage1.json"),
            runner_parent_commit="a" * 40,
        )


def test_preopen_summary_rejects_unplanned_raw_forced_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sp_lense.comparison_provenance as provenance

    lock, _, manifest, _ = _preopen_fixture(tmp_path, monkeypatch)
    source = manifest["source_calibration_summaries"][0]
    summary_path = tmp_path / source["calibration_summary_path"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    plan_record = summary["forced_grid_plan_artifact"]
    plan_path = tmp_path / plan_record["path"]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["points"][0]["shard_name"] = "not_a_planned_grid_shard.json"
    _write_json(plan_path, plan)
    plan_record["sha256"] = sha256_file(plan_path)
    _write_json(summary_path, summary)

    with pytest.raises(RuntimeError, match="single locked interpolation recheck"):
        provenance._preopen_summary_record(
            tmp_path,
            lock,
            summary_path,
            stage1_lock_sha256=sha256_file(tmp_path / "stage1.json"),
            runner_parent_commit="a" * 40,
        )


def test_stage2_builder_derives_manifest_from_finalized_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, _, stage2 = _stage2_fixture(tmp_path)
    _mock_git(tmp_path, monkeypatch)
    dataset = json.loads(
        (tmp_path / lock["dataset"]["path"]).read_text(encoding="utf-8")
    )
    forced_units = locked_validation_calibration_units(dataset, lock)
    open_units = locked_open_confirmation_units(dataset, lock)
    final_summaries: list[Path] = []
    pending_summaries: list[Path] = []
    directions: list[dict[str, Any]] = []
    for index, record in enumerate(stage2["direction_and_calibration_artifacts"]):
        source_summary = tmp_path / record["validation_summary_path"]
        final_summary = tmp_path / f"final_summary_{index}.json"
        final_summary.write_bytes(source_summary.read_bytes())
        final_summaries.append(final_summary)
        finalized = json.loads(source_summary.read_text(encoding="utf-8"))
        forced_record = finalized["forced_result_rows_artifacts"][0]
        forced_rows = [
            json.loads(line)
            for line in (tmp_path / forced_record["path"])
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        pending = build_calibration_summary(
            [forced_rows],
            expected_forced_units=forced_units,
            expected_open_units=open_units,
            safety_limits=SafetyLimits.from_lock(lock["calibration"]["safety_gates"]),
            mode=record["track"],
            forced_result_rows_artifacts=[forced_record],
            forced_grid_plan_artifact=finalized["forced_grid_plan_artifact"],
            open_result_rows_artifacts=[],
            calibration_config_sha256=sha256_json(lock["calibration"]),
            builder_module_sha256=sha256_file(
                tmp_path / "src" / "sp_lense" / "comparison_calibration.py"
            ),
            allow_pending_open=True,
        )
        pending_path = tmp_path / f"pending_summary_{index}.json"
        _write_json(pending_path, pending)
        pending_summaries.append(pending_path)
        directions.append(
            {
                "path": record["direction_path"],
                "method_id": record["method_id"],
                "layer": record["selected_layer"],
                "intervention_geometry": record["intervention_geometry"],
                "direction_float32_sha256": record["direction_float32_sha256"],
                "direction_artifact_sha256": record["direction_artifact_sha256"],
                "track": record["track"],
                "construction_config_path": record["construction_config_path"],
                "construction_config_sha256": record[
                    "construction_config_sha256"
                ],
            }
        )
    for record in stage2["random_direction_controls"]:
        directions.append(
            {
                "path": record["direction_path"],
                "method_id": record["method_id"],
                "layer": record["selected_layer"],
                "intervention_geometry": record["intervention_geometry"],
                "direction_float32_sha256": record["direction_float32_sha256"],
                "direction_artifact_sha256": record["direction_artifact_sha256"],
                "track": record["track"],
                "construction_config_path": record["construction_config_path"],
                "construction_config_sha256": record[
                    "construction_config_sha256"
                ],
            }
        )
    direction_manifest = tmp_path / "all_directions.json"
    _write_json(direction_manifest, {"directions": directions})
    preopen = build_preopen_manifest(
        tmp_path,
        lock,
        stage1_lock_path=tmp_path / "stage1.json",
        calibration_summary_paths=pending_summaries,
        direction_manifest_paths=[direction_manifest],
        runner_parent_commit="a" * 40,
    )
    preopen_path = tmp_path / "preopen.json"
    _write_json(preopen_path, preopen)
    built = build_stage2_manifest(
        tmp_path,
        lock,
        stage1_lock_path=tmp_path / "stage1.json",
        preopen_manifest_path=preopen_path,
        environment_lock_path=tmp_path / "environment.json",
        calibration_summary_paths=final_summaries,
        direction_manifest_paths=[direction_manifest],
    )
    assert built["preopen_manifest_sha256"] == sha256_file(preopen_path)
    assert len(built["direction_and_calibration_artifacts"]) == len(final_summaries)
    assert built["random_direction_controls"][0]["approved_strengths"]


def test_stage2_manifest_verifies_exact_coverage_and_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, manifest_path, _ = _stage2_fixture(tmp_path, monkeypatch)
    _mock_git(tmp_path, monkeypatch)
    verified = verify_stage2_manifest(tmp_path, lock, manifest_path)
    assert verified.manifest_sha256 == sha256_file(manifest_path)
    assert verified.stage1_lock_sha256 == sha256_file(tmp_path / "stage1.json")
    assert verified.stage1_lock_payload_sha256 == sha256_json(lock)
    statuses = verified_method_status_records(verified)
    assert statuses
    assert statuses[0]["matched_fixed_descriptive"]["status"] == "approved"
    assert_stage2_ready(verified)

    selected = json.loads(manifest_path.read_text(encoding="utf-8"))[
        "direction_and_calibration_artifacts"
    ][0]
    approved = assert_approved_setup(
        verified,
        repo_root=tmp_path,
        model_id="m",
        model_revision="1" * 40,
        model_config_sha256=lock["models"][0]["config_sha256"],
        method_id=selected["method_id"],
        track=selected["track"],
        direction_path=tmp_path / selected["direction_path"],
        direction_file_sha256=selected["direction_file_sha256"],
        direction_float32_sha256=selected["direction_float32_sha256"],
        direction_artifact_sha256=selected["direction_artifact_sha256"],
        selected_strength=selected["selected_strength"],
        selected_layer=selected["selected_layer"],
        position_schedule=selected["position_schedule"],
        construction_config_sha256=selected["construction_config_sha256"],
        calibration_summary_sha256=selected["validation_summary_sha256"],
    )
    assert approved["method_id"] == selected["method_id"]
    assert approved["validation_coverage_adequate"] is True
    assert approved["validation_sign_safe"] == {"plus": True, "minus": True}
    detached = approved_setup_records(verified)
    detached[0]["method_id"] = "tampered"
    assert approved_setup_records(verified)[0]["method_id"] != "tampered"

    with pytest.raises(RuntimeError, match="not approved|differs from stage-2 approval"):
        assert_approved_setup(
            verified,
            repo_root=tmp_path,
            model_id="m",
            model_revision="1" * 40,
            model_config_sha256=lock["models"][0]["config_sha256"],
            method_id=selected["method_id"],
            track=selected["track"],
            direction_path=tmp_path / selected["direction_path"],
            direction_file_sha256=selected["direction_file_sha256"],
            direction_float32_sha256=selected["direction_float32_sha256"],
            direction_artifact_sha256=selected["direction_artifact_sha256"],
            selected_strength=0.04,
            selected_layer=selected["selected_layer"],
            position_schedule=selected["position_schedule"],
            construction_config_sha256=selected["construction_config_sha256"],
            calibration_summary_sha256=selected["validation_summary_sha256"],
        )


def test_stage2_rejects_protected_commit_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, manifest_path, _ = _stage2_fixture(tmp_path, monkeypatch)
    _mock_git(tmp_path, monkeypatch, protected_diff={"runner.py"})
    with pytest.raises(
        RuntimeError,
        match=(
            "runner code/protocol changed|pre-open protected paths changed|"
            "changed since runner_code_commit"
        ),
    ):
        verify_stage2_manifest(tmp_path, lock, manifest_path)


def test_stage2_rejects_untracked_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, manifest_path, _ = _stage2_fixture(tmp_path, monkeypatch)
    _mock_git(tmp_path, monkeypatch, untracked={"stage2.json"})
    with pytest.raises(RuntimeError, match="not Git-tracked"):
        verify_stage2_manifest(tmp_path, lock, manifest_path)


def test_stage2_recomputes_direction_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, manifest_path, manifest = _stage2_fixture(tmp_path, monkeypatch)
    record = manifest["direction_and_calibration_artifacts"][0]
    direction_path = tmp_path / record["direction_path"]
    direction_record = json.loads(direction_path.read_text(encoding="utf-8"))
    direction_record["direction"][0] = 999.0
    _write_json(direction_path, direction_record)
    record["direction_file_sha256"] = sha256_file(direction_path)
    _write_json(manifest_path, manifest)
    _mock_git(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="direction identity mismatch"):
        verify_stage2_manifest(tmp_path, lock, manifest_path)


def test_stage2_requires_random_control_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, manifest_path, manifest = _stage2_fixture(tmp_path, monkeypatch)
    manifest["random_direction_controls"] = []
    _write_json(manifest_path, manifest)
    _mock_git(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="random_direction_controls"):
        verify_stage2_manifest(tmp_path, lock, manifest_path)


def test_random_control_setup_allows_only_source_approved_strengths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, manifest_path, manifest = _stage2_fixture(tmp_path, monkeypatch)
    _mock_git(tmp_path, monkeypatch)
    verified = verify_stage2_manifest(tmp_path, lock, manifest_path)
    random_record = manifest["random_direction_controls"][0]
    approved_strength = random_record["approved_strengths"][0]
    approved = assert_approved_setup(
        verified,
        repo_root=tmp_path,
        model_id="m",
        model_revision="1" * 40,
        model_config_sha256=lock["models"][0]["config_sha256"],
        method_id=random_record["method_id"],
        track="matched",
        direction_path=tmp_path / random_record["direction_path"],
        direction_file_sha256=random_record["direction_file_sha256"],
        direction_float32_sha256=random_record["direction_float32_sha256"],
        direction_artifact_sha256=random_record["direction_artifact_sha256"],
        selected_strength=approved_strength["strength"],
        selected_layer=10,
        position_schedule="final_prompt_token",
        construction_config_sha256=random_record["construction_config_sha256"],
        calibration_summary_sha256=approved_strength[
            "source_calibration_summary_sha256"
        ],
    )
    assert approved["control_source_method_id"] == approved_strength["source_method_id"]

    with pytest.raises(RuntimeError, match="differs from stage-2 approval"):
        assert_approved_setup(
            verified,
            repo_root=tmp_path,
            model_id="m",
            model_revision="1" * 40,
            model_config_sha256=lock["models"][0]["config_sha256"],
            method_id=random_record["method_id"],
            track="matched",
            direction_path=tmp_path / random_record["direction_path"],
            direction_file_sha256=random_record["direction_file_sha256"],
            direction_float32_sha256=random_record["direction_float32_sha256"],
            direction_artifact_sha256=random_record["direction_artifact_sha256"],
            selected_strength=0.03,
            selected_layer=10,
            position_schedule="final_prompt_token",
            construction_config_sha256=random_record["construction_config_sha256"],
            calibration_summary_sha256=approved_strength[
                "source_calibration_summary_sha256"
            ],
        )


def test_stage2_rejects_arbitrary_random_approved_strength(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, manifest_path, manifest = _stage2_fixture(tmp_path, monkeypatch)
    manifest["random_direction_controls"][0]["approved_strengths"][0][
        "strength"
    ] = 0.03
    _write_json(manifest_path, manifest)
    _mock_git(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="approved_strengths"):
        verify_stage2_manifest(tmp_path, lock, manifest_path)


def test_stage2_rejects_incomplete_calibration_rows_even_when_rehashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sp_lense.comparison_calibration import calibration_rows_sha256

    lock, manifest_path, manifest = _stage2_fixture(tmp_path, monkeypatch)
    record = manifest["direction_and_calibration_artifacts"][0]
    validation_path = tmp_path / record["validation_summary_path"]
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    result_record = validation["forced_result_rows_artifacts"][0]
    result_path = tmp_path / result_record["path"]
    rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
    rows.pop()
    _write_jsonl(result_path, rows)
    result_record["sha256"] = sha256_file(result_path)
    validation["point_rows_sha256s"] = [calibration_rows_sha256(rows)]
    _write_json(validation_path, validation)
    record["validation_summary_sha256"] = sha256_file(validation_path)
    _write_json(manifest_path, manifest)
    _mock_git(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_stage2_manifest(tmp_path, lock, manifest_path)


def test_stage2_rejects_post_preopen_no_safe_reclassification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, manifest_path, manifest = _stage2_fixture(tmp_path, monkeypatch)
    record = next(
        item
        for item in manifest["direction_and_calibration_artifacts"]
        if item["method_id"] == "gradient" and item["track"] == "matched"
    )
    validation_path = tmp_path / record["validation_summary_path"]
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    result_path = tmp_path / validation["forced_result_rows_artifacts"][0]["path"]
    rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["condition"] in {"plus", "minus"}:
            row["kl_from_baseline"] = 0.1
    _write_jsonl(result_path, rows)
    expected_units = locked_validation_calibration_units(
        json.loads((tmp_path / lock["dataset"]["path"]).read_text(encoding="utf-8")),
        lock,
    )
    validation = build_calibration_summary(
        [rows],
        expected_forced_units=expected_units,
        expected_open_units=locked_open_confirmation_units(
            json.loads((tmp_path / lock["dataset"]["path"]).read_text(encoding="utf-8")),
            lock,
        ),
        safety_limits=SafetyLimits.from_lock(lock["calibration"]["safety_gates"]),
        mode="matched",
        forced_result_rows_artifacts=[
            {"path": result_path.name, "sha256": sha256_file(result_path)}
        ],
        forced_grid_plan_artifact=validation["forced_grid_plan_artifact"],
        open_result_rows_artifacts=[],
        calibration_config_sha256=sha256_json(lock["calibration"]),
        builder_module_sha256=sha256_file(
            tmp_path / "src" / "sp_lense" / "comparison_calibration.py"
        ),
    )
    assert validation["decision"]["status"] == "no_safe_nonzero"
    _write_json(validation_path, validation)
    record["selected_strength"] = None
    record["sealed_evaluation_required"] = False
    record["validation_summary_sha256"] = sha256_file(validation_path)
    random_record = manifest["random_direction_controls"][0]
    random_record["approved_strengths"] = [
        item
        for item in random_record["approved_strengths"]
        if item["source_method_id"] != "gradient"
    ]
    _write_json(manifest_path, manifest)
    _mock_git(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_stage2_manifest(tmp_path, lock, manifest_path)


def test_stage2_rejects_post_preopen_grid_plan_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, manifest_path, manifest = _stage2_fixture(tmp_path, monkeypatch)
    record = manifest["direction_and_calibration_artifacts"][0]
    validation_path = tmp_path / record["validation_summary_path"]
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    del validation["forced_grid_plan_artifact"]
    _write_json(validation_path, validation)
    updated_hash = sha256_file(validation_path)
    record["validation_summary_sha256"] = updated_hash
    frozen_record = next(
        item
        for item in manifest["frozen_artifact_paths"]
        if item["path"] == record["validation_summary_path"]
    )
    frozen_record["sha256"] = updated_hash
    _write_json(manifest_path, manifest)
    _mock_git(tmp_path, monkeypatch)

    with pytest.raises(
        RuntimeError, match="changed frozen pre-open forced_grid_plan_artifact"
    ):
        verify_stage2_manifest(tmp_path, lock, manifest_path)


def test_stage2_rejects_outcome_blind_amendment_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, manifest_path, manifest = _stage2_fixture(tmp_path, monkeypatch)
    manifest["outcome_blind_amendment"] = {
        "path": "configs/tampered-amendment.json",
        "sha256": "0" * 64,
    }
    _write_json(manifest_path, manifest)
    _mock_git(tmp_path, monkeypatch)
    with pytest.raises(
        RuntimeError, match="outcome-blind amendment differs from current"
    ):
        verify_stage2_manifest(tmp_path, lock, manifest_path)


def test_stage2_requires_uncorrected_gradient_ablation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, manifest_path, manifest = _stage2_fixture(tmp_path, monkeypatch)
    manifest["direction_and_calibration_artifacts"] = [
        record
        for record in manifest["direction_and_calibration_artifacts"]
        if record["method_id"] != "gradient_uncorrected"
    ]
    _write_json(manifest_path, manifest)
    _mock_git(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        verify_stage2_manifest(tmp_path, lock, manifest_path)
