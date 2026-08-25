from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

import pytest

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = EXPERIMENT_ROOT / "scripts" / "run_sensitivity.py"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bipo_warmup_sensitivity_runner", RUNNER_PATH)
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


def test_lock_verifies_without_model_loading(runner: ModuleType) -> None:
    observed = runner.verify_experiment(
        repo_root=REPO_ROOT,
        experiment_root=EXPERIMENT_ROOT,
    )
    assert observed["status"] == "verified_outcome_blind_secondary_sensitivity"
    assert observed["planned_artifact_count"] == 4
    assert observed["total_optimizer_steps"] == 320
    assert observed["warmup_steps"] == 11
    assert observed["confirmatory_ranking_eligible"] is False


def test_warmup_is_nearest_integer_to_published_fraction(config: dict) -> None:
    derivation = config["warmup_derivation"]
    assert math.ceil(608 / 4) * 20 == 3040
    assert math.ceil(64 / 4) * 20 == 320
    exact_target = 320 * (100 / 3040)
    assert exact_target == pytest.approx(10.526315789473685)
    assert round(exact_target) == 11
    assert abs((11 / 320) - (100 / 3040)) < abs((10 / 320) - (100 / 3040))
    assert derivation["selected_warmup_steps"] == 11
    assert derivation["selected_warmup_fraction"] == pytest.approx(0.034375)
    assert derivation["confirmatory_warmup_fraction"] == pytest.approx(0.3125)


def test_only_training_change_is_warmup(config: dict, runner: ModuleType) -> None:
    differences = runner._training_differences(
        config["confirmatory_training"],
        config["sensitivity_training"],
    )
    assert differences == {"warmup_steps": (100, 11)}


def test_four_new_artifact_ids_and_hashes_are_unique(config: dict, runner: ModuleType) -> None:
    plans = config["planned_artifacts"]
    assert {(plan["model_tag"], plan["track"]) for plan in plans} == {
        (model, track)
        for model in ("qwen35_08b", "qwen35_2b")
        for track in ("matched", "canonical")
    }
    assert len({plan["artifact_id"] for plan in plans}) == 4
    assert len({plan["artifact_identity_sha256"] for plan in plans}) == 4
    for plan in plans:
        assert plan["artifact_id"].startswith("bipo_warmup11_")
        assert (
            runner.canonical_sha256(plan["artifact_identity"]) == plan["artifact_identity_sha256"]
        )


def test_ranking_and_path_firewalls_are_explicit(config: dict) -> None:
    role = config["analysis_role"]
    assert role["analysis_tier"] == "secondary_sensitivity_only"
    assert role["confirmatory_winner_ranking_eligible"] is False
    assert role["automatic_confirmatory_ingestion_allowed"] is False
    assert role["sensitivity_method_id"] == "bipo_warmup11_sensitivity"
    assert role["sensitivity_method_id"] != role["parent_method_id"]
    for plan in config["planned_artifacts"]:
        assert plan["output_directory"].startswith("experiments/bipo_warmup_sensitivity/outputs/")
        assert not plan["output_directory"].startswith("artifacts/steering_comparison/")
        assert plan["manifest_filename"] == "sensitivity_manifest.json"
        assert plan["manifest_filename"] != "direction_manifest.json"


def test_existing_confirmatory_vectors_are_hash_bound_and_future_ones_are_not_invented(
    config: dict,
) -> None:
    for plan in config["planned_artifacts"]:
        reference = plan["confirmatory_reference_at_registration"]
        if plan["model_tag"] == "qwen35_08b":
            assert reference["status"] == "present_hash_bound_read_only"
            assert all(
                isinstance(reference[field], str) and len(reference[field]) == 64
                for field in ("file_sha256", "artifact_sha256", "direction_float32_sha256")
            )
        else:
            assert reference == {
                "status": "pending_confirmatory_construction",
                "file_sha256": None,
                "artifact_sha256": None,
                "direction_float32_sha256": None,
            }


def test_runner_does_not_import_confirmatory_reporting() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "comparison_report" not in source
    assert "build_final_report" not in source
    assert 'root.glob("*/directions/bipo_*/*")' in source
    assert "confirmatory BiPO files changed" in source


def test_outputs_are_git_ignored() -> None:
    patterns = {
        line.strip()
        for line in (EXPERIMENT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert "outputs/" in patterns
