from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = EXPERIMENT_ROOT / "scripts" / "evaluate_sensitivity.py"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bipo_postconfirm_evaluation", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    return _load_runner()


@pytest.fixture(scope="module")
def config() -> dict:
    return json.loads((EXPERIMENT_ROOT / "config.json").read_text(encoding="utf-8"))


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def test_final_freeze_gate_requires_exact_subject_and_commit_bytes(
    tmp_path: Path, runner: ModuleType
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Sensitivity Test")
    _git(repo, "config", "user.email", "sensitivity@example.invalid")
    anchor = repo / "artifacts" / "steering_comparison" / "final_report.json"
    companion = repo / "configs" / "steering_comparison_stage2_lock.json"
    anchor.parent.mkdir(parents=True)
    companion.parent.mkdir(parents=True)
    anchor.write_text("{}\n", encoding="utf-8")
    companion.write_text("{}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "wrong subject")
    with pytest.raises(runner.PostConfirmatoryGateError, match="exact frozen"):
        runner.require_frozen_commit_paths(
            repo,
            anchor_path="artifacts/steering_comparison/final_report.json",
            expected_subject="Add sealed steering comparison results and adversarial review",
            required_paths=[
                "artifacts/steering_comparison/final_report.json",
                "configs/steering_comparison_stage2_lock.json",
            ],
        )
    _git(
        repo,
        "commit",
        "--amend",
        "-m",
        "Add sealed steering comparison results and adversarial review",
    )
    commit = runner.require_frozen_commit_paths(
        repo,
        anchor_path="artifacts/steering_comparison/final_report.json",
        expected_subject="Add sealed steering comparison results and adversarial review",
        required_paths=[
            "artifacts/steering_comparison/final_report.json",
            "configs/steering_comparison_stage2_lock.json",
        ],
    )
    assert commit == _git(repo, "rev-parse", "HEAD")
    companion.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(runner.PostConfirmatoryGateError, match="differs from final commit"):
        runner.require_frozen_commit_paths(
            repo,
            anchor_path="artifacts/steering_comparison/final_report.json",
            expected_subject="Add sealed steering comparison results and adversarial review",
            required_paths=[
                "artifacts/steering_comparison/final_report.json",
                "configs/steering_comparison_stage2_lock.json",
            ],
        )


def test_remote_main_gate_reads_the_actual_pushed_ref(tmp_path: Path, runner: ModuleType) -> None:
    bare = tmp_path / "origin.git"
    repo = tmp_path / "work"
    _git(tmp_path, "init", "--bare", str(bare))
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Sensitivity Test")
    _git(repo, "config", "user.email", "sensitivity@example.invalid")
    (repo / "marker.txt").write_text("pushed\n", encoding="utf-8")
    _git(repo, "add", "marker.txt")
    _git(repo, "commit", "-m", "pushed marker")
    _git(repo, "branch", "-M", "main")
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "-u", "origin", "main")
    assert runner._remote_main_commit(repo) == _git(repo, "rev-parse", "HEAD")
    (repo / "marker.txt").write_text("local only\n", encoding="utf-8")
    _git(repo, "commit", "-am", "local only")
    assert runner._remote_main_commit(repo) != _git(repo, "rev-parse", "HEAD")


def _forced_row(case_id: str, condition: str, prompt: str) -> dict:
    return {
        "split": "sealed_test",
        "family": "self_preservation",
        "case_id": case_id,
        "target": "self",
        "role": None,
        "suite": None,
        "category": None,
        "form": None,
        "scenario_cluster_id": case_id,
        "request_type": None,
        "expected_behavior": None,
        "first_semantic_label": "A",
        "second_semantic_label": "B",
        "preserve_label": "A",
        "comply_label": "B",
        "correct_label": None,
        "preferred_label": None,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "condition": condition,
    }


def _open_row(case_id: str, condition: str, prompt: str) -> dict:
    return {
        "split": "sealed_test",
        "family": "open_ended",
        "case_id": case_id,
        "source_core_id": "core_001",
        "target": "self",
        "condition": condition,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "rubric_sha256": "a" * 64,
        "generation_config": {"decoder": "greedy_argmax", "max_new_tokens": 96},
        "baseline_content_sha256": "b" * 64,
    }


def test_exact_sealed_identity_and_coverage_fail_on_substitution(
    runner: ModuleType,
) -> None:
    parent = [
        _forced_row("case_1", "baseline", "same prompt"),
        _forced_row("case_1", "plus", "same prompt"),
    ]
    sensitivity = [dict(row) for row in parent]
    runner.assert_exact_forced_coverage(parent, sensitivity, expected_rows=2)
    sensitivity[1]["prompt_sha256"] = "f" * 64
    with pytest.raises(runner.PostConfirmatoryGateError, match="exact parent"):
        runner.assert_exact_forced_coverage(parent, sensitivity, expected_rows=2)

    parent_open = [
        _open_row("open_1", "baseline", "same open prompt"),
        _open_row("open_1", "plus", "same open prompt"),
    ]
    sensitivity_open = [dict(row) for row in parent_open]
    runner.assert_exact_open_coverage(parent_open, sensitivity_open, expected_rows=2)
    sensitivity_open[0]["baseline_content_sha256"] = "c" * 64
    with pytest.raises(runner.PostConfirmatoryGateError, match="baseline generations"):
        runner.assert_exact_open_coverage(parent_open, sensitivity_open, expected_rows=2)


def _safe_plan() -> dict:
    return {
        "analysis_tier": "secondary_sensitivity_only",
        "confirmatory_winner_ranking_eligible": False,
        "automatic_confirmatory_ingestion_allowed": False,
        "method_id": "bipo_warmup11_sensitivity",
        "output_root": "experiments/bipo_warmup_sensitivity/evaluation_outputs",
        "setups": [
            {
                "identity": {"method_id": "bipo_warmup11_sensitivity"},
                "forced_path": (
                    "experiments/bipo_warmup_sensitivity/evaluation_outputs/forced/a.jsonl"
                ),
                "generation_path": (
                    "experiments/bipo_warmup_sensitivity/evaluation_outputs/generations/a.jsonl"
                ),
                "scored_path": (
                    "experiments/bipo_warmup_sensitivity/evaluation_outputs/scored/a.jsonl"
                ),
            }
        ],
    }


def test_ranking_firewall_rejects_main_method_and_main_output(
    runner: ModuleType,
) -> None:
    plan = _safe_plan()
    runner.validate_plan_firewall(plan, REPO_ROOT)
    plan["setups"][0]["identity"]["method_id"] = "bipo"
    with pytest.raises(runner.PostConfirmatoryGateError, match="confirmatory method"):
        runner.validate_plan_firewall(plan, REPO_ROOT)
    plan = _safe_plan()
    plan["output_root"] = "artifacts/steering_comparison/sensitivity"
    with pytest.raises(runner.PostConfirmatoryGateError, match="aliases main"):
        runner.validate_plan_firewall(plan, REPO_ROOT)


def test_postconfirm_protocol_inherits_exact_parent_limits(config: dict) -> None:
    parent = json.loads(
        (REPO_ROOT / config["parent_study"]["stage1_lock_path"]).read_text(encoding="utf-8")
    )
    policy = config["evaluation_policy"]
    assert policy["safety_limits"] == parent["calibration"]["safety_gates"]
    assert policy["forced_coverage"]["rows_per_setup"] == 1350
    assert policy["open_coverage"]["rows_per_setup"] == 96
    assert policy["strength_selection"].startswith("inherit_each_parent_bipo_setup")
    assert policy["automatic_main_ranking_ingestion_allowed"] is False


def test_ranking_firewall_is_structural_and_main_report_is_never_imported() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "sp_lense.comparison_report" not in source
    assert "rank_behavioral_efficacy" not in source
    assert "rank_equal_efficacy_selectivity" not in source
    assert 'SENSITIVITY_METHOD_ID = "bipo_warmup11_sensitivity"' in source
    assert '"ranking_update": None' in source
    assert "refs/remotes/origin/main" in source
    assert "local HEAD to equal origin/main" in source


def test_evaluation_outputs_are_ignored() -> None:
    patterns = {
        line.strip()
        for line in (EXPERIMENT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert "evaluation_outputs/" in patterns
