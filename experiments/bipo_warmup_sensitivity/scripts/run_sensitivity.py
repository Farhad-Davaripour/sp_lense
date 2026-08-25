from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

EXPERIMENT_RELATIVE = Path("experiments/bipo_warmup_sensitivity")
CONFIG_NAME = "config.json"
LOCK_NAME = "lock.json"
SENSITIVITY_METHOD_ID = "bipo_warmup11_sensitivity"
CONFIRMATORY_METHOD_ID = "bipo"
ALLOWED_MODEL_TAGS = ("qwen35_08b", "qwen35_2b")
ALLOWED_TRACKS = ("matched", "canonical")
ZERO_SHA256 = "0" * 64


class SensitivityLockError(RuntimeError):
    """Raised before model loading when the sensitivity lock is not exact."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SensitivityLockError(f"{label} must contain a JSON object")
    return value


def default_roots() -> tuple[Path, Path]:
    experiment_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root, experiment_root


def _resolve_repo_path(repo_root: Path, relative: str, *, label: str) -> Path:
    path = (repo_root / relative).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise SensitivityLockError(f"{label} escapes the repository: {relative}") from exc
    return path


def _nearest_integer_half_up(value: Fraction) -> int:
    if value < 0:
        raise ValueError("warmup target must be non-negative")
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def _training_differences(
    confirmatory: Mapping[str, Any], sensitivity: Mapping[str, Any]
) -> dict[str, tuple[Any, Any]]:
    keys = set(confirmatory) | set(sensitivity)
    return {
        key: (confirmatory.get(key), sensitivity.get(key))
        for key in sorted(keys)
        if confirmatory.get(key) != sensitivity.get(key)
    }


def _verify_parent_method(config: Mapping[str, Any], parent_lock: Mapping[str, Any]) -> None:
    parent = parent_lock.get("methods", {}).get("bipo")
    if not isinstance(parent, Mapping):
        raise SensitivityLockError("parent lock lacks methods.bipo")
    expected = config["confirmatory_training"]
    comparisons = {
        "beta": parent.get("beta"),
        "learning_rate": parent.get("learning_rate"),
        "weight_decay": parent.get("weight_decay"),
        "max_grad_norm": parent.get("max_grad_norm"),
        "epochs": max(parent.get("checkpoint_epochs", []), default=None),
        "checkpoint_epochs": parent.get("checkpoint_epochs"),
        "selected_checkpoint_epoch": parent.get("selected_checkpoint_epoch"),
        "gradient_accumulation_steps": parent.get("gradient_accumulation_steps"),
        "cpu_microbatch_size": parent.get("cpu_microbatch_size"),
        "effective_batch_size": parent.get("effective_batch_size"),
        "warmup_steps": parent.get("warmup_steps"),
        "seed": parent.get("seed"),
        "direction_coefficient_seed": parent.get("direction_coefficient_sampling", {}).get("seed"),
        "optimizer": parent.get("optimizer"),
        "lr_scheduler": parent.get("lr_scheduler"),
    }
    mismatches = {
        key: (expected.get(key), observed)
        for key, observed in comparisons.items()
        if expected.get(key) != observed
    }
    if mismatches:
        raise SensitivityLockError(
            f"confirmatory training mirror differs from parent lock: {mismatches}"
        )


def _verify_warmup_math(config: Mapping[str, Any]) -> None:
    derivation = config["warmup_derivation"]
    published_steps = math.ceil(
        int(derivation["published_pair_count"]) / int(derivation["published_effective_batch_size"])
    ) * int(derivation["published_epochs"])
    if published_steps != int(derivation["published_total_optimizer_steps"]):
        raise SensitivityLockError("published optimizer-step derivation is inconsistent")
    training = config["sensitivity_training"]
    sensitivity_steps = math.ceil(
        int(config["dataset"]["example_count"]) / int(training["effective_batch_size"])
    ) * int(training["epochs"])
    if sensitivity_steps != int(derivation["sensitivity_total_optimizer_steps"]):
        raise SensitivityLockError("sensitivity optimizer-step derivation is inconsistent")
    target = Fraction(
        sensitivity_steps * int(derivation["published_warmup_fraction_numerator"]),
        int(derivation["published_warmup_fraction_denominator"]),
    )
    selected = _nearest_integer_half_up(target)
    if selected != 11 or selected != int(derivation["selected_warmup_steps"]):
        raise SensitivityLockError("fraction-matched warmup must be preregistered as 11")
    if int(training["warmup_steps"]) != selected:
        raise SensitivityLockError("sensitivity training does not use 11 warmup steps")
    observed_fraction = selected / sensitivity_steps
    if not math.isclose(
        observed_fraction,
        float(derivation["selected_warmup_fraction"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise SensitivityLockError("selected warmup fraction is inconsistent")


def _verify_artifact_plans(config: Mapping[str, Any], repo_root: Path) -> None:
    plans = config.get("planned_artifacts")
    if not isinstance(plans, list) or len(plans) != 4:
        raise SensitivityLockError("exactly four sensitivity artifacts must be planned")
    expected_pairs = {
        (model_tag, track) for model_tag in ALLOWED_MODEL_TAGS for track in ALLOWED_TRACKS
    }
    observed_pairs: set[tuple[str, str]] = set()
    artifact_ids: set[str] = set()
    identity_hashes: set[str] = set()
    output_root = _resolve_repo_path(
        repo_root, config["output_policy"]["root"], label="output root"
    )
    main_root = (repo_root / "artifacts/steering_comparison").resolve()
    for plan in plans:
        if not isinstance(plan, Mapping):
            raise SensitivityLockError("each planned artifact must be an object")
        pair = (str(plan.get("model_tag")), str(plan.get("track")))
        observed_pairs.add(pair)
        artifact_id = str(plan.get("artifact_id"))
        artifact_ids.add(artifact_id)
        identity = plan.get("artifact_identity")
        if not isinstance(identity, Mapping):
            raise SensitivityLockError(f"{artifact_id} lacks artifact_identity")
        observed_identity_hash = canonical_sha256(identity)
        if observed_identity_hash != plan.get("artifact_identity_sha256"):
            raise SensitivityLockError(f"{artifact_id} identity hash mismatch")
        identity_hashes.add(observed_identity_hash)
        if identity.get("artifact_id") != artifact_id:
            raise SensitivityLockError(f"{artifact_id} identity repeats a different ID")
        if identity.get("warmup_steps") != 11:
            raise SensitivityLockError(f"{artifact_id} identity does not bind warmup 11")
        model = config["models"][pair[0]]
        track = config["tracks"][pair[1]]
        expected_identity_fields = {
            "model_id": model["model_id"],
            "model_revision": model["revision"],
            "track": pair[1],
            "layer_zero_based": model["layer_zero_based"],
            "training_geometry": track["training_geometry"],
            "dataset_sha256": config["dataset"]["sha256"],
            "parent_stage1_lock_sha256": config["parent_study"]["stage1_lock_sha256"],
        }
        bad_identity = {
            key: (value, identity.get(key))
            for key, value in expected_identity_fields.items()
            if identity.get(key) != value
        }
        if bad_identity:
            raise SensitivityLockError(f"{artifact_id} identity mismatch: {bad_identity}")
        output = _resolve_repo_path(
            repo_root, str(plan["output_directory"]), label=f"{artifact_id} output"
        )
        try:
            output.relative_to(output_root)
        except ValueError as exc:
            raise SensitivityLockError(f"{artifact_id} output is outside sensitivity root") from exc
        try:
            output.relative_to(main_root)
        except ValueError:
            pass
        else:
            raise SensitivityLockError(f"{artifact_id} output aliases confirmatory artifacts")
        if plan.get("manifest_filename") != "sensitivity_manifest.json":
            raise SensitivityLockError(f"{artifact_id} must use sensitivity_manifest.json")
        if plan.get("direction_filename") == "direction_manifest.json":
            raise SensitivityLockError(f"{artifact_id} uses a forbidden main manifest name")
        reference = plan.get("confirmatory_reference_at_registration")
        if not isinstance(reference, Mapping):
            raise SensitivityLockError(f"{artifact_id} lacks registration-time reference state")
        reference_path = _resolve_repo_path(
            repo_root,
            str(plan["confirmatory_reference_path"]),
            label=f"{artifact_id} confirmatory reference",
        )
        if reference.get("status") == "present_hash_bound_read_only":
            if not reference_path.is_file() or sha256_file(reference_path) != reference.get(
                "file_sha256"
            ):
                raise SensitivityLockError(
                    f"{artifact_id} registration-time confirmatory reference changed"
                )
            reference_record = load_json_object(
                reference_path, label=f"{artifact_id} confirmatory reference"
            )
            if reference_record.get("artifact_sha256") != reference.get(
                "artifact_sha256"
            ) or reference_record.get("direction_sha256") != reference.get(
                "direction_float32_sha256"
            ):
                raise SensitivityLockError(
                    f"{artifact_id} registration-time internal direction hashes changed"
                )
        elif reference.get("status") == "pending_confirmatory_construction":
            if any(
                reference.get(field) is not None
                for field in ("file_sha256", "artifact_sha256", "direction_float32_sha256")
            ):
                raise SensitivityLockError(
                    f"{artifact_id} pending reference must not invent future hashes"
                )
        else:
            raise SensitivityLockError(f"{artifact_id} has unknown reference status")
    if observed_pairs != expected_pairs:
        raise SensitivityLockError(
            f"planned model/track coverage differs: {observed_pairs} != {expected_pairs}"
        )
    if len(artifact_ids) != 4 or len(identity_hashes) != 4:
        raise SensitivityLockError("artifact IDs and identity hashes must be unique")


def verify_experiment(
    *, repo_root: Path | None = None, experiment_root: Path | None = None
) -> dict[str, Any]:
    default_repo, default_experiment = default_roots()
    repo_root = (repo_root or default_repo).resolve()
    experiment_root = (experiment_root or default_experiment).resolve()
    if experiment_root != (repo_root / EXPERIMENT_RELATIVE).resolve():
        raise SensitivityLockError("experiment must remain at its locked repository path")
    lock_path = experiment_root / LOCK_NAME
    config_path = experiment_root / CONFIG_NAME
    lock = load_json_object(lock_path, label="sensitivity lock")
    config = load_json_object(config_path, label="sensitivity config")
    if lock.get("status") != "locked_outcome_blind_unrun":
        raise SensitivityLockError("sensitivity lock status is not outcome-blind/unrun")
    if config.get("status") != "locked_outcome_blind_unrun":
        raise SensitivityLockError("sensitivity config status is not outcome-blind/unrun")
    role = config.get("analysis_role")
    if not isinstance(role, Mapping) or role != {
        "analysis_tier": "secondary_sensitivity_only",
        "confirmatory_winner_ranking_eligible": False,
        "automatic_confirmatory_ingestion_allowed": False,
        "parent_method_id": CONFIRMATORY_METHOD_ID,
        "sensitivity_method_id": SENSITIVITY_METHOD_ID,
    }:
        raise SensitivityLockError("ranking firewall fields differ from the locked values")

    file_manifest = lock.get("file_manifest")
    if not isinstance(file_manifest, list) or not file_manifest:
        raise SensitivityLockError("lock file_manifest must be a non-empty list")
    observed_paths: set[str] = set()
    for item in file_manifest:
        if not isinstance(item, Mapping):
            raise SensitivityLockError("lock file manifest entries must be objects")
        relative = str(item.get("path"))
        if relative in observed_paths:
            raise SensitivityLockError(f"duplicate locked file: {relative}")
        observed_paths.add(relative)
        path = _resolve_repo_path(repo_root, relative, label="locked experiment file")
        try:
            path.relative_to(experiment_root)
        except ValueError as exc:
            raise SensitivityLockError(
                f"locked experiment file is outside directory: {relative}"
            ) from exc
        if not path.is_file():
            raise SensitivityLockError(f"locked experiment file is missing: {relative}")
        if sha256_file(path) != item.get("sha256"):
            raise SensitivityLockError(f"locked experiment file hash mismatch: {relative}")
    expected_paths = set(lock.get("required_file_paths", []))
    if observed_paths != expected_paths:
        raise SensitivityLockError(
            f"locked experiment coverage mismatch: {observed_paths} != {expected_paths}"
        )

    source_bindings = lock.get("immutable_source_bindings")
    if not isinstance(source_bindings, list) or not source_bindings:
        raise SensitivityLockError("immutable source bindings must be materialized")
    for item in source_bindings:
        relative = str(item.get("path"))
        path = _resolve_repo_path(repo_root, relative, label="immutable source")
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise SensitivityLockError(f"immutable source binding mismatch: {relative}")

    parent = config["parent_study"]
    parent_lock_path = _resolve_repo_path(
        repo_root, parent["stage1_lock_path"], label="parent stage-1 lock"
    )
    if sha256_file(parent_lock_path) != parent["stage1_lock_sha256"]:
        raise SensitivityLockError("parent stage-1 lock hash mismatch")
    amendment_path = _resolve_repo_path(
        repo_root,
        parent["outcome_blind_amendment_path"],
        label="parent outcome-blind amendment",
    )
    if sha256_file(amendment_path) != parent["outcome_blind_amendment_sha256"]:
        raise SensitivityLockError("parent outcome-blind amendment hash mismatch")
    protocol_path = _resolve_repo_path(repo_root, parent["protocol_path"], label="parent protocol")
    if sha256_file(protocol_path) != parent["protocol_sha256"]:
        raise SensitivityLockError("parent protocol hash mismatch")
    parent_lock = load_json_object(parent_lock_path, label="parent stage-1 lock")
    if parent_lock.get("study") != parent["study_id"]:
        raise SensitivityLockError("parent study ID mismatch")
    if parent_lock.get("sources", {}).get("bipo", {}).get("commit") != parent.get(
        "bipo_source_commit"
    ):
        raise SensitivityLockError("pinned BiPO source commit mismatch")
    _verify_parent_method(config, parent_lock)

    dataset_path = _resolve_repo_path(
        repo_root, config["dataset"]["path"], label="locked discovery dataset"
    )
    if sha256_file(dataset_path) != config["dataset"]["sha256"]:
        raise SensitivityLockError("discovery dataset hash mismatch")
    dataset = load_json_object(dataset_path, label="locked discovery dataset")
    discovery = dataset.get("sp_splits", {}).get("discovery")
    if not isinstance(discovery, list) or len(discovery) != config["dataset"]["example_count"]:
        raise SensitivityLockError("discovery example count mismatch")
    if any(not isinstance(case, dict) or case.get("split") != "discovery" for case in discovery):
        raise SensitivityLockError("non-discovery case found in discovery construction set")

    differences = _training_differences(
        config["confirmatory_training"], config["sensitivity_training"]
    )
    if differences != {"warmup_steps": (100, 11)}:
        raise SensitivityLockError(f"warmup must be the only training change: {differences}")
    _verify_warmup_math(config)
    _verify_artifact_plans(config, repo_root)

    parent_method_ids = set(parent_lock.get("evaluation", {}).get("method_ids", []))
    parent_method_ids.update(parent_lock.get("methods", {}).keys())
    if SENSITIVITY_METHOD_ID in parent_method_ids:
        raise SensitivityLockError("sensitivity method unexpectedly entered parent method IDs")
    if config["output_policy"]["manifest_filename"] == "direction_manifest.json":
        raise SensitivityLockError("sensitivity may not emit the main manifest filename")

    return {
        "status": "verified_outcome_blind_secondary_sensitivity",
        "study_id": config["study_id"],
        "sensitivity_lock_sha256": sha256_file(lock_path),
        "sensitivity_config_sha256": sha256_file(config_path),
        "planned_artifact_count": len(config["planned_artifacts"]),
        "total_optimizer_steps": config["warmup_derivation"]["sensitivity_total_optimizer_steps"],
        "warmup_steps": config["sensitivity_training"]["warmup_steps"],
        "confirmatory_ranking_eligible": False,
    }


def _run_git(repo_root: Path, arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SensitivityLockError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.strip()


def _require_committed_experiment(repo_root: Path, experiment_root: Path) -> str:
    lock = load_json_object(experiment_root / LOCK_NAME, label="sensitivity lock")
    for relative in [LOCK_NAME, *lock["required_file_paths"]]:
        repo_relative = (
            (EXPERIMENT_RELATIVE / relative).as_posix() if relative == LOCK_NAME else str(relative)
        )
        _run_git(repo_root, ["ls-files", "--error-unmatch", "--", repo_relative])
        dirty = _run_git(repo_root, ["status", "--short", "--", repo_relative])
        if dirty:
            raise SensitivityLockError(
                f"construction requires a committed, clean experiment file: {repo_relative}"
            )
    return _run_git(repo_root, ["rev-parse", "HEAD"])


def _confirmatory_snapshot(repo_root: Path) -> dict[str, str]:
    root = repo_root / "artifacts/steering_comparison"
    files = sorted(path for path in root.glob("*/directions/bipo_*/*") if path.is_file())
    return {path.relative_to(repo_root).as_posix(): sha256_file(path) for path in files}


def _find_plan(config: Mapping[str, Any], model_tag: str, track: str) -> dict[str, Any]:
    matches = [
        dict(plan)
        for plan in config["planned_artifacts"]
        if plan["model_tag"] == model_tag and plan["track"] == track
    ]
    if len(matches) != 1:
        raise SensitivityLockError(f"no unique plan for {model_tag}/{track}")
    return matches[0]


def _validate_confirmatory_reference(
    repo_root: Path,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> str:
    path = _resolve_repo_path(
        repo_root, plan["confirmatory_reference_path"], label="confirmatory BiPO reference"
    )
    if not path.is_file():
        raise SensitivityLockError(
            "the corresponding 100-step confirmatory vector must exist before sensitivity "
            f"construction: {path}"
        )
    record = load_json_object(path, label="confirmatory BiPO reference")
    metadata = record.get("metadata")
    model = config["models"][plan["model_tag"]]
    expected_geometry = config["tracks"][plan["track"]]["training_geometry"]
    expected = {
        "method": CONFIRMATORY_METHOD_ID,
        "layer": model["layer_zero_based"],
        "intervention_geometry": expected_geometry,
    }
    mismatches = {
        key: (value, record.get(key)) for key, value in expected.items() if record.get(key) != value
    }
    if not isinstance(metadata, Mapping):
        mismatches["metadata"] = ("object", type(metadata).__name__)
    else:
        if metadata.get("model_id") != model["model_id"]:
            mismatches["model_id"] = (model["model_id"], metadata.get("model_id"))
        if metadata.get("model_revision") != model["revision"]:
            mismatches["model_revision"] = (model["revision"], metadata.get("model_revision"))
        if metadata.get("track") != plan["track"]:
            mismatches["track"] = (plan["track"], metadata.get("track"))
        training = metadata.get("training_config")
        if not isinstance(training, Mapping) or training.get("warmup_steps") != 100:
            mismatches["warmup_steps"] = (
                100,
                None if not training else training.get("warmup_steps"),
            )
    if mismatches:
        raise SensitivityLockError(
            f"confirmatory reference is not the locked 100-step BiPO artifact: {mismatches}"
        )
    file_sha256 = sha256_file(path)
    registered = plan["confirmatory_reference_at_registration"]
    if registered["status"] == "present_hash_bound_read_only":
        registration_mismatches = {
            "file_sha256": (registered["file_sha256"], file_sha256),
            "artifact_sha256": (registered["artifact_sha256"], record.get("artifact_sha256")),
            "direction_float32_sha256": (
                registered["direction_float32_sha256"],
                record.get("direction_sha256"),
            ),
        }
        registration_mismatches = {
            key: values for key, values in registration_mismatches.items() if values[0] != values[1]
        }
        if registration_mismatches:
            raise SensitivityLockError(
                f"registration-time confirmatory reference changed: {registration_mismatches}"
            )
    return file_sha256


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def construct(model_tag: str, track: str) -> dict[str, Any]:
    repo_root, experiment_root = default_roots()
    verification = verify_experiment(repo_root=repo_root, experiment_root=experiment_root)
    runner_commit = _require_committed_experiment(repo_root, experiment_root)
    config = load_json_object(experiment_root / CONFIG_NAME, label="sensitivity config")
    parent_lock = load_json_object(
        repo_root / config["parent_study"]["stage1_lock_path"], label="parent stage-1 lock"
    )
    plan = _find_plan(config, model_tag, track)
    output_dir = _resolve_repo_path(
        repo_root, plan["output_directory"], label="locked sensitivity output"
    )
    expected_output_root = _resolve_repo_path(
        repo_root, config["output_policy"]["root"], label="locked output root"
    )
    try:
        output_dir.relative_to(expected_output_root)
    except ValueError as exc:
        raise SensitivityLockError(
            "construction output escaped the locked sensitivity root"
        ) from exc
    if output_dir.exists():
        raise SensitivityLockError(f"overwrite is forbidden; output already exists: {output_dir}")

    reference_file_sha256 = _validate_confirmatory_reference(repo_root, config, plan)
    before_snapshot = _confirmatory_snapshot(repo_root)
    reference_relative = str(plan["confirmatory_reference_path"])
    if before_snapshot.get(reference_relative) != reference_file_sha256:
        raise SensitivityLockError("confirmatory reference is absent from preservation snapshot")

    work_dir = output_dir.parent / f".{output_dir.name}.work-{os.getpid()}"
    if work_dir.exists():
        raise SensitivityLockError(f"stale sensitivity work directory exists: {work_dir}")
    work_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir()
    try:
        src_path = repo_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        from sp_lense.backend import ResearchBackend
        from sp_lense.comparison_bipo import BiPOTrainingConfig
        from sp_lense.comparison_dataset import load_comparison_dataset
        from sp_lense.comparison_fit import fit_bipo_artifact, write_direction_artifact
        from sp_lense.comparison_runtime import validate_locked_choice_runtime
        from sp_lense.config import load_config
        from sp_lense.steering_methods import DirectionArtifact

        model = config["models"][model_tag]
        parent_model = next(
            item for item in parent_lock["models"] if item["model_id"] == model["model_id"]
        )
        backend = ResearchBackend.load(
            load_config(repo_root / model["config_path"]), with_lens=False
        )
        validate_locked_choice_runtime(backend, parent_model["runtime"])
        dataset = load_comparison_dataset(
            repo_root / config["dataset"]["path"],
            expected_sha256=config["dataset"]["sha256"],
        )
        training = config["sensitivity_training"]
        training_config = BiPOTrainingConfig(
            beta=float(training["beta"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            max_grad_norm=float(training["max_grad_norm"]),
            epochs=int(training["epochs"]),
            checkpoint_epochs=tuple(int(value) for value in training["checkpoint_epochs"]),
            gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
            warmup_steps=int(training["warmup_steps"]),
            seed=int(training["seed"]),
        )
        common_metadata = {
            "study": config["study_id"],
            "analysis_tier": "secondary_sensitivity_only",
            "confirmatory_winner_ranking_eligible": False,
            "automatic_confirmatory_ingestion_allowed": False,
            "parent_method_id": CONFIRMATORY_METHOD_ID,
            "sensitivity_method_id": SENSITIVITY_METHOD_ID,
            "artifact_id": plan["artifact_id"],
            "artifact_identity_sha256": plan["artifact_identity_sha256"],
            "model_id": model["model_id"],
            "model_revision": model["revision"],
            "model_config_sha256": model["config_sha256"],
            "dataset_sha256": config["dataset"]["sha256"],
            "protocol_sha256": next(
                item["sha256"]
                for item in load_json_object(experiment_root / LOCK_NAME, label="sensitivity lock")[
                    "file_manifest"
                ]
                if item["path"] == (EXPERIMENT_RELATIVE / "PROTOCOL.md").as_posix()
            ),
            "stage1_lock_sha256": config["parent_study"]["stage1_lock_sha256"],
            "parent_confirmatory_runner_commit": config["parent_study"][
                "confirmatory_runner_commit"
            ],
            "runner_commit": runner_commit,
            "sensitivity_lock_sha256": verification["sensitivity_lock_sha256"],
            "sensitivity_config_sha256": verification["sensitivity_config_sha256"],
            "confirmatory_reference_path": reference_relative,
            "confirmatory_reference_file_sha256": reference_file_sha256,
            "only_changed_training_hyperparameter": {
                "name": "warmup_steps",
                "confirmatory_value": 100,
                "sensitivity_value": 11,
            },
        }
        fitted, diagnostics = fit_bipo_artifact(
            backend,
            dataset["sp_splits"]["discovery"],
            layer=int(model["layer_zero_based"]),
            track=track,
            config=training_config,
            selected_checkpoint_epoch=int(training["selected_checkpoint_epoch"]),
            common_metadata=common_metadata,
        )
        sensitivity_artifact = DirectionArtifact(
            method=SENSITIVITY_METHOD_ID,
            direction=fitted.direction,
            layer=fitted.layer,
            intervention_geometry=fitted.intervention_geometry,
            metadata={
                **dict(fitted.metadata),
                "confirmatory_winner_ranking_eligible": False,
                "automatic_confirmatory_ingestion_allowed": False,
            },
        )
        direction_path = work_dir / plan["direction_filename"]
        direction_record = write_direction_artifact(direction_path, sensitivity_artifact)
        training_path = work_dir / "training_audit.json"
        _write_json(
            training_path,
            {
                "schema_version": "sp_lense.bipo_warmup_sensitivity.training_audit.v1",
                "analysis_tier": "secondary_sensitivity_only",
                "confirmatory_winner_ranking_eligible": False,
                "artifact_id": plan["artifact_id"],
                "artifact_identity_sha256": plan["artifact_identity_sha256"],
                "sensitivity_lock_sha256": verification["sensitivity_lock_sha256"],
                "diagnostics": diagnostics,
            },
        )
        after_snapshot = _confirmatory_snapshot(repo_root)
        if after_snapshot != before_snapshot:
            raise SensitivityLockError(
                "confirmatory BiPO files changed during sensitivity construction"
            )
        preservation_path = work_dir / "confirmatory_preservation.json"
        _write_json(
            preservation_path,
            {
                "schema_version": "sp_lense.bipo_warmup_sensitivity.preservation.v1",
                "status": "confirmatory_bipo_files_byte_identical",
                "confirmatory_reference_path": reference_relative,
                "confirmatory_reference_file_sha256": reference_file_sha256,
                "snapshot": before_snapshot,
            },
        )
        manifest_path = work_dir / plan["manifest_filename"]
        manifest = {
            "schema_version": "sp_lense.bipo_warmup_sensitivity.manifest.v1",
            "status": "complete_secondary_sensitivity_construction",
            "study_id": config["study_id"],
            "analysis_tier": "secondary_sensitivity_only",
            "confirmatory_winner_ranking_eligible": False,
            "automatic_confirmatory_ingestion_allowed": False,
            "method_id": SENSITIVITY_METHOD_ID,
            "parent_method_id": CONFIRMATORY_METHOD_ID,
            "artifact_id": plan["artifact_id"],
            "artifact_identity_sha256": plan["artifact_identity_sha256"],
            "model_tag": model_tag,
            "model_id": model["model_id"],
            "model_revision": model["revision"],
            "track": track,
            "layer_zero_based": model["layer_zero_based"],
            "warmup_steps": 11,
            "sensitivity_lock_sha256": verification["sensitivity_lock_sha256"],
            "sensitivity_config_sha256": verification["sensitivity_config_sha256"],
            "runner_commit": runner_commit,
            "direction": {
                **{key: value for key, value in direction_record.items() if key != "path"},
                "path": plan["direction_filename"],
                "file_sha256": sha256_file(direction_path),
            },
            "training_audit": {
                "path": training_path.name,
                "sha256": sha256_file(training_path),
            },
            "confirmatory_preservation": {
                "path": preservation_path.name,
                "sha256": sha256_file(preservation_path),
            },
        }
        _write_json(manifest_path, manifest)
        if output_dir.exists():
            raise SensitivityLockError("output appeared concurrently; refusing overwrite")
        work_dir.replace(output_dir)
        return {
            "status": "complete_secondary_sensitivity_construction",
            "artifact_id": plan["artifact_id"],
            "output_directory": str(output_dir),
            "manifest_sha256": sha256_file(output_dir / plan["manifest_filename"]),
            "confirmatory_winner_ranking_eligible": False,
        }
    except BaseException:
        if work_dir.exists():
            shutil.rmtree(work_dir)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Locked outcome-blind BiPO warmup-fraction sensitivity"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify", help="verify hashes/invariants without loading a model")
    subparsers.add_parser("plan", help="print locked plans without loading a model")
    construct_parser = subparsers.add_parser(
        "construct", help="construct one locked warmup-11 direction"
    )
    construct_parser.add_argument("--model-tag", choices=ALLOWED_MODEL_TAGS, required=True)
    construct_parser.add_argument("--track", choices=ALLOWED_TRACKS, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        print(json.dumps(verify_experiment(), indent=2))
        return
    if args.command == "plan":
        verify_experiment()
        _, experiment_root = default_roots()
        config = load_json_object(experiment_root / CONFIG_NAME, label="sensitivity config")
        print(
            json.dumps(
                {
                    "study_id": config["study_id"],
                    "analysis_tier": "secondary_sensitivity_only",
                    "confirmatory_winner_ranking_eligible": False,
                    "only_training_change": {"warmup_steps": [100, 11]},
                    "planned_artifacts": config["planned_artifacts"],
                },
                indent=2,
            )
        )
        return
    print(json.dumps(construct(args.model_tag, args.track), indent=2))


if __name__ == "__main__":
    main()
