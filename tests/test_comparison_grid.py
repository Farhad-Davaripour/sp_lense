from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sp_lense.comparison_analysis import ROW_SCHEMA_VERSION, canonical_json_sha256
from sp_lense.comparison_dataset import load_comparison_dataset
from sp_lense.comparison_fit import write_direction_artifact
from sp_lense.comparison_grid import (
    EXPECTED_POINTS_PER_MODEL,
    BaselineLogitsCache,
    baseline_logits_float32_sha256,
    build_forced_prompt_units,
    derive_forced_grid_specs,
    load_validated_point_rows,
    resolve_forced_grid_plan,
    run_forced_grid,
)
from sp_lense.comparison_provenance import (
    MAIN_CONSTRUCTION_SCHEMA_VERSION,
    locked_method_construction_configuration,
    locked_position_schedule,
    sha256_file,
    sha256_json,
)
from sp_lense.steering_methods import DirectionArtifact

LOCK_PATH = Path("configs/steering_comparison_lock.json")
DATASET_PATH = Path("data/steering_comparison_cases.json")
STAGE1_SHA256 = "d" * 64
RUNNER_COMMIT = "2" * 40


def _lock_and_dataset() -> tuple[dict, dict]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    dataset = load_comparison_dataset(DATASET_PATH, expected_sha256=lock["dataset"]["sha256"])
    return lock, dataset


def _geometry(method: str, track: str) -> str:
    if track == "matched" or method.startswith("gradient"):
        return "matched_final_prompt"
    return {
        "caa": "caa_post_prompt",
        "bipo": "canonical_broadcast",
        "persona_vector": "persona_response",
    }[method]


def _evidence_roles(method: str) -> tuple[str, ...]:
    return {
        "gradient": ("gradient_construction_diagnostics",),
        "gradient_uncorrected": ("gradient_construction_diagnostics",),
        "caa": ("caa_construction_diagnostics",),
        "bipo": ("bipo_training_audit",),
        "persona_vector": (
            "persona_construction_diagnostics",
            "persona_scored_rollouts",
        ),
    }[method]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _artifact_manifest(tmp_path: Path, lock: dict, model: dict) -> Path:
    specs = derive_forced_grid_specs(lock, model["model_id"])
    keys = sorted({(item.method_id, item.track, item.layer) for item in specs})
    records = []
    evidence_by_method: dict[str, list[dict[str, str]]] = {}
    for method in sorted({key[0] for key in keys}):
        evidence = []
        for role in _evidence_roles(method):
            evidence_path = tmp_path / "evidence" / f"{method}_{role}.json"
            _write_json(evidence_path, {"method": method, "role": role})
            evidence.append(
                {"role": role, "path": str(evidence_path), "sha256": sha256_file(evidence_path)}
            )
        evidence_by_method[method] = evidence
    for method, track, layer in keys:
        direction = torch.zeros(model["architecture"]["residual_width"], dtype=torch.float32)
        direction[layer % direction.numel()] = 1.0
        artifact = DirectionArtifact(
            method=method,
            direction=direction,
            layer=layer,
            intervention_geometry=_geometry(method, track),
            metadata={
                "model_id": model["model_id"],
                "model_revision": model["revision"],
                "model_config_sha256": model["config_sha256"],
                "dataset_sha256": lock["dataset"]["sha256"],
                "protocol_sha256": lock["protocol"]["sha256"],
                "stage1_lock_sha256": STAGE1_SHA256,
                "runner_commit": RUNNER_COMMIT,
                "track": track,
            },
        )
        stem = f"{method}_{track}_l{layer:02d}"
        direction_path = tmp_path / "directions" / f"{stem}.json"
        direction_record = write_direction_artifact(direction_path, artifact)
        locked_configuration = locked_method_construction_configuration(lock, method, track)
        evidence = evidence_by_method[method]
        construction = {
            "schema_version": MAIN_CONSTRUCTION_SCHEMA_VERSION,
            "model_id": model["model_id"],
            "model_revision": model["revision"],
            "model_config_sha256": model["config_sha256"],
            "method_id": method,
            "track": track,
            "selected_layer": layer,
            "position_schedule": locked_position_schedule(method, track),
            "intervention_geometry": _geometry(method, track),
            "direction_float32_sha256": artifact.direction_sha256,
            "direction_artifact_sha256": artifact.artifact_sha256,
            "dataset_sha256": lock["dataset"]["sha256"],
            "protocol_sha256": lock["protocol"]["sha256"],
            "stage1_lock_sha256": STAGE1_SHA256,
            "runner_commit": RUNNER_COMMIT,
            "locked_configuration": locked_configuration,
            "locked_configuration_sha256": sha256_json(locked_configuration),
            "evidence_artifacts": evidence,
            "evidence_artifacts_sha256": sha256_json(evidence),
        }
        construction_path = tmp_path / "directions" / f"{stem}.construction.json"
        _write_json(construction_path, construction)
        records.append(
            {
                **direction_record,
                "track": track,
                "construction_config_path": str(construction_path),
                "construction_config_sha256": sha256_file(construction_path),
            }
        )
    manifest_path = tmp_path / "direction_manifest.json"
    _write_json(manifest_path, {"directions": records})
    return manifest_path


@pytest.fixture(scope="module")
def resolved_plan(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, dict, dict, Path]:
    lock, dataset = _lock_and_dataset()
    model = lock["models"][0]
    root = tmp_path_factory.mktemp("grid-plan")
    manifest = _artifact_manifest(root, lock, model)
    plan = resolve_forced_grid_plan(
        repo_root=root,
        lock=lock,
        stage1_lock_sha256=STAGE1_SHA256,
        model_id=model["model_id"],
        direction_manifest_paths=[manifest],
        runner_commit=RUNNER_COMMIT,
    )
    return plan, lock, dataset, root


def test_exact_grid_has_the_locked_250_point_breakdown() -> None:
    lock, _ = _lock_and_dataset()
    for model in lock["models"]:
        specs = derive_forced_grid_specs(lock, model["model_id"])
        assert len(specs) == EXPECTED_POINTS_PER_MODEL
        assert sum(item.track == "matched" for item in specs) == 30
        assert sum(item.track == "canonical" and item.method_id == "caa" for item in specs) == 96
        assert sum(item.track == "canonical" and item.method_id == "bipo" for item in specs) == 4
        assert (
            sum(item.track == "canonical" and item.method_id == "persona_vector" for item in specs)
            == 120
        )


def test_plan_resolves_every_direction_and_construction_fail_closed(
    resolved_plan: tuple[dict, dict, dict, Path],
) -> None:
    plan, lock, _, root = resolved_plan
    assert plan["expected_point_count"] == 250
    assert len(plan["points"]) == 250
    assert (
        len({(point["method_id"], point["track"], point["layer"]) for point in plan["points"]})
        == 54
    )
    assert all(not Path(item["path"]).is_absolute() for item in plan["direction_manifests"])
    assert all(
        not Path(point["direction_path"]).is_absolute()
        and not Path(point["construction_config_path"]).is_absolute()
        for point in plan["points"]
    )
    construction_path = root / plan["points"][0]["construction_config_path"]
    original = construction_path.read_text(encoding="utf-8")
    construction_path.write_text(original + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="construction config hash"):
        resolve_forced_grid_plan(
            repo_root=root,
            lock=lock,
            stage1_lock_sha256=STAGE1_SHA256,
            model_id=plan["model_id"],
            direction_manifest_paths=[root / "direction_manifest.json"],
            runner_commit=RUNNER_COMMIT,
        )
    construction_path.write_text(original, encoding="utf-8")


def test_prompt_projection_is_exactly_validation_only() -> None:
    lock, dataset = _lock_and_dataset()
    units = build_forced_prompt_units(dataset, lock)
    assert len(units) == 142
    assert len({unit.key for unit in units}) == 142
    validation_core = {case["id"] for case in dataset["sp_splits"]["validation"]}
    sealed_core = {case["id"] for case in dataset["sp_splits"]["sealed_test"]}
    observed_core = {unit.case_id for unit in units if unit.family == "self_preservation"}
    assert observed_core == validation_core
    assert not observed_core & sealed_core
    assert all(unit.family not in {"open_ended", "tbsp_style"} for unit in units)


def test_baseline_cache_separates_identity_key_from_logit_content() -> None:
    plan = {
        "model_id": "model",
        "model_revision": "1" * 40,
        "model_config_sha256": "2" * 64,
    }
    first_cache = BaselineLogitsCache(plan)
    second_cache = BaselineLogitsCache(plan)
    _, first_key, first_content = first_cache.get_or_compute(
        "same prompt", lambda: torch.tensor([1.0, 2.0])
    )
    _, second_key, second_content = second_cache.get_or_compute(
        "same prompt", lambda: torch.tensor([1.0, 3.0])
    )
    assert first_key == second_key
    assert first_content != second_content
    assert first_content == baseline_logits_float32_sha256([1.0, 2.0])
    assert first_content != baseline_logits_float32_sha256(torch.tensor([[1.0, 2.0]]))


def _fake_rows(
    backend: object,
    point: dict,
    prompt_units: tuple,
    baseline_cache: BaselineLogitsCache,
    plan: dict,
) -> list[dict]:
    del backend
    rows = []
    for unit in prompt_units:
        _, baseline_key, baseline_logits_hash = baseline_cache.get_or_compute(
            unit.prompt, lambda: [0.0]
        )
        boundary_hash = hashlib.sha256(f"boundary:{unit.prompt}".encode()).hexdigest()
        for condition, sign in (("baseline", 0), ("plus", 1), ("minus", -1)):
            signed = sign * float(point["strength"])
            rows.append(
                {
                    "schema_version": ROW_SCHEMA_VERSION,
                    "model_id": plan["model_id"],
                    "model_revision": plan["model_revision"],
                    "dataset_sha256": plan["dataset_sha256"],
                    "protocol_sha256": plan["protocol_sha256"],
                    "config_sha256": plan["model_config_sha256"],
                    "stage1_lock_sha256": plan["stage1_lock_sha256"],
                    "stage2_manifest_sha256": "0" * 64,
                    "calibration_summary_sha256": "0" * 64,
                    "construction_config_sha256": point["construction_config_sha256"],
                    "runner_commit": plan["runner_commit"],
                    "direction_sha256": point["direction_float32_sha256"],
                    "direction_float32_sha256": point["direction_float32_sha256"],
                    "direction_artifact_sha256": point["direction_artifact_sha256"],
                    "prompt_sha256": hashlib.sha256(unit.prompt.encode("utf-8")).hexdigest(),
                    "baseline_cache_key_sha256": baseline_key,
                    "baseline_logits_float32_sha256": baseline_logits_hash,
                    "choice_boundary_evidence_sha256": boundary_hash,
                    "choice_a_token_id": 32,
                    "choice_b_token_id": 33,
                    "method": point["method_id"],
                    "method_id": point["method_id"],
                    "setup": point["track"],
                    "track": point["track"],
                    "direction_id": point["direction_artifact_sha256"],
                    "strength_id": f"{point['track']}:{point['strength']:.12g}",
                    "calibration_magnitude": point["strength"],
                    "split": "validation",
                    "family": unit.family,
                    "case_id": unit.case_id,
                    "condition": condition,
                    "condition_alpha": signed,
                    "strength": signed,
                    "layer": point["layer"],
                    "position": point["position_schedule"],
                    "run_seed": plan["run_seed"],
                    "a_minus_b_log_odds": float(sign),
                    "forced_pair_label": "A",
                    "actual_next_token_label": "A",
                    "kl_from_baseline": 0.0 if sign == 0 else 0.001,
                    "coherent": True,
                    "realized_mean_relative_perturbation_norm": abs(signed),
                    **dict(unit.extra_fields),
                }
            )
    return rows


def test_runner_loads_once_reuses_baselines_and_validates_resume(
    resolved_plan: tuple[dict, dict, dict, Path], tmp_path: Path, monkeypatch
) -> None:
    plan, lock, dataset, root = resolved_plan
    selected = [point["point_sha256"] for point in plan["points"][:2]]
    loads = []
    runtime_checks = []

    monkeypatch.setattr(
        "sp_lense.comparison_grid.validate_locked_choice_runtime",
        lambda backend, runtime: runtime_checks.append((backend.name, runtime)) or {},
    )

    def factory(value: dict) -> object:
        loads.append(value["plan_sha256"])
        model = next(item for item in lock["models"] if item["model_id"] == value["model_id"])
        return SimpleNamespace(
            name="single-resident-model",
            config=SimpleNamespace(
                config_path=Path(model["config"]).resolve(),
                model=SimpleNamespace(
                    id=model["model_id"],
                    revision=model["revision"],
                    prompt_format="chat",
                ),
            ),
            model=SimpleNamespace(
                cfg=SimpleNamespace(
                    n_layers=model["architecture"]["blocks"],
                    d_model=model["architecture"]["residual_width"],
                )
            ),
        )

    first = run_forced_grid(
        plan=plan,
        lock=lock,
        dataset=dataset,
        repo_root=root,
        output_dir=tmp_path,
        backend_factory=factory,
        point_evaluator=_fake_rows,
        only_point_sha256s=selected,
    )
    assert loads == [plan["plan_sha256"]]
    assert runtime_checks == [("single-resident-model", lock["models"][0]["runtime"])]
    assert first["written_count"] == 2
    assert first["baseline_forward_computation_count"] == 142
    assert first["remaining_count"] == 248

    loaded_rows, loaded_validation = load_validated_point_rows(
        first["shards"][0]["path"],
        plan=plan,
        lock=lock,
        dataset=dataset,
        repo_root=root,
    )
    assert len(loaded_rows) == 426
    assert loaded_validation["row_count"] == 426
    assert {row["baseline_logits_float32_sha256"] for row in loaded_rows} == {
        baseline_logits_float32_sha256([0.0])
    }

    second = run_forced_grid(
        plan=plan,
        lock=lock,
        dataset=dataset,
        repo_root=root,
        output_dir=tmp_path,
        backend_factory=lambda value: pytest.fail("resume loaded a model for completed points"),
        point_evaluator=_fake_rows,
        only_point_sha256s=selected,
    )
    assert second["validated_existing_count"] == 2
    assert second["written_count"] == 0
    assert second["model_loaded_by_factory"] is False

    shard_path = Path(first["shards"][0]["path"])
    original_shard = shard_path.read_text(encoding="utf-8")
    corrupted = json.loads(shard_path.read_text(encoding="utf-8"))
    corrupted["rows"][0]["case_id"] = "tampered"
    _write_json(shard_path, corrupted)
    with pytest.raises(ValueError, match="content hash"):
        run_forced_grid(
            plan=plan,
            lock=lock,
            dataset=dataset,
            repo_root=root,
            output_dir=tmp_path,
            backend_factory=lambda value: pytest.fail("corrupt resume loaded the model"),
            point_evaluator=_fake_rows,
            only_point_sha256s=selected,
        )

    shard_path.write_text(original_shard, encoding="utf-8")
    inconsistent = json.loads(original_shard)
    inconsistent["rows"][0]["choice_a_token_id"] = 99
    payload = {key: value for key, value in inconsistent.items() if key != "shard_content_sha256"}
    inconsistent["shard_content_sha256"] = canonical_json_sha256(payload)
    _write_json(shard_path, inconsistent)
    with pytest.raises(ValueError, match="mismatched baseline or choice-boundary"):
        run_forced_grid(
            plan=plan,
            lock=lock,
            dataset=dataset,
            repo_root=root,
            output_dir=tmp_path,
            backend_factory=lambda value: pytest.fail("inconsistent resume loaded the model"),
            point_evaluator=_fake_rows,
            only_point_sha256s=selected,
        )


def test_runner_rejects_the_wrong_resident_model_before_evaluation(
    resolved_plan: tuple[dict, dict, dict, Path], tmp_path: Path
) -> None:
    plan, lock, dataset, root = resolved_plan
    evaluated = []

    wrong_backend = SimpleNamespace(
        config=SimpleNamespace(
            config_path=Path(lock["models"][0]["config"]).resolve(),
            model=SimpleNamespace(
                id="wrong/model",
                revision=plan["model_revision"],
                prompt_format="chat",
            ),
        ),
        model=SimpleNamespace(
            cfg=SimpleNamespace(
                n_layers=24,
                d_model=lock["models"][0]["architecture"]["residual_width"],
            )
        ),
    )
    with pytest.raises(RuntimeError, match="resident backend differs"):
        run_forced_grid(
            plan=plan,
            lock=lock,
            dataset=dataset,
            repo_root=root,
            output_dir=tmp_path,
            backend=wrong_backend,
            point_evaluator=lambda *args: evaluated.append(args) or [],
            only_point_sha256s=[plan["points"][0]["point_sha256"]],
        )
    assert evaluated == []
