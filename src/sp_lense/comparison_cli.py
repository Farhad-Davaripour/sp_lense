from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .backend import ResearchBackend
from .comparison_bipo import BiPOTrainingConfig
from .comparison_calibration import (
    SafetyLimits,
    build_calibration_summary,
    locked_forced_calibration_units,
    locked_open_confirmation_units,
)
from .comparison_controls import random_control_artifacts
from .comparison_dataset import load_comparison_dataset
from .comparison_environment import capture_environment_record, write_environment_record
from .comparison_evaluate import (
    EvaluationIdentity,
    MethodSetup,
    SealedEvaluationGate,
    evaluate_collateral_cases,
    evaluate_option_order_sentinels,
    evaluate_sp_cases,
    evaluate_tbsp_cases,
    sealed_ids_from_dataset_and_lock,
    select_cases_by_locked_ids,
)
from .comparison_fit import (
    fit_bipo_artifact,
    fit_caa_artifacts_all_layers,
    fit_gradient_method,
    fit_persona_artifacts_all_layers,
    make_direction_artifact,
    matched_artifact_from_canonical,
    read_direction_artifact,
    write_direction_artifact,
)
from .comparison_grid import (
    load_validated_point_rows,
    resolve_forced_grid_plan,
    run_forced_grid,
)
from .comparison_persona import (
    generate_persona_grid,
    load_persona_protocol,
    persona_generation_provenance,
    read_rollouts,
    validate_rollouts,
)
from .comparison_provenance import (
    MAIN_CONSTRUCTION_SCHEMA_VERSION,
    OUTCOME_BLIND_AMENDMENT_PATH,
    RANDOM_CONSTRUCTION_SCHEMA_VERSION,
    RANDOM_GENERATOR_ALGORITHM,
    assert_approved_setup,
    assert_preopen_approved_setup,
    build_outcome_blind_amendment_manifest,
    build_preopen_manifest,
    build_stage2_manifest,
    locked_method_construction_configuration,
    locked_position_schedule,
    locked_runner_code_commit,
    sha256_file,
    sha256_json,
    verify_outcome_blind_amendment,
    verify_preopen_manifest,
    verify_stage1_lock,
    verify_stage2_manifest,
)
from .comparison_runtime import validate_locked_choice_runtime
from .config import load_config
from .io_utils import write_json, write_jsonl

_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def _sha256_argument(value: str) -> str:
    if not _SHA256_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError("expected a 64-character hexadecimal SHA-256")
    return value.lower()


def _load_lock(repo_root: Path, lock_path: Path) -> dict[str, Any]:
    return verify_stage1_lock(repo_root, lock_path)


def _model_record(lock: Mapping[str, Any], config_path: Path) -> Mapping[str, Any]:
    normalized = config_path.resolve()
    for model in lock["models"]:
        if normalized == (config_path.parents[1] / model["config"]).resolve():
            return model
        if normalized.name == Path(model["config"]).name:
            return model
    raise ValueError(f"model config is not in the comparison lock: {config_path}")


def _common_metadata(
    repo_root: Path,
    lock_path: Path,
    lock: Mapping[str, Any],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "study": lock["study"],
        "model_id": model["model_id"],
        "model_revision": model["revision"],
        "model_config_sha256": model["config_sha256"],
        "dataset_sha256": lock["dataset"]["sha256"],
        "protocol_sha256": lock["protocol"]["sha256"],
        "stage1_lock_sha256": sha256_file(lock_path),
        "runner_commit": locked_runner_code_commit(repo_root, lock_path),
    }


def _write_artifacts(
    output_dir: Path,
    artifacts: Mapping[str, Any],
    *,
    repo_root: Path,
    lock: Mapping[str, Any],
    evidence_paths: Mapping[str, Path] | None = None,
) -> list[dict[str, Any]]:
    root = repo_root.resolve()

    def relative_path(path: Path, *, field: str) -> str:
        try:
            return path.resolve().relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"{field} must be inside the repository") from error

    evidence_artifacts = []
    for role, path in sorted((evidence_paths or {}).items()):
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"construction evidence is missing: {resolved}")
        evidence_artifacts.append(
            {
                "role": role,
                "path": relative_path(resolved, field="construction evidence"),
                "sha256": sha256_file(resolved),
            }
        )
    records = []
    for name, artifact in artifacts.items():
        record = write_direction_artifact(output_dir / f"{name}.json", artifact)
        record["path"] = relative_path(
            output_dir / f"{name}.json", field="direction artifact"
        )
        track = str(
            artifact.metadata.get(
                "track", "matched" if artifact.method.startswith("random_control_") else ""
            )
        )
        if track not in {"matched", "canonical"}:
            raise RuntimeError(f"direction {name!r} lacks a matched/canonical track identity")
        if artifact.method.startswith("random_control_"):
            construction = {
                "schema_version": RANDOM_CONSTRUCTION_SCHEMA_VERSION,
                "model_id": artifact.metadata["model_id"],
                "model_revision": artifact.metadata["model_revision"],
                "model_config_sha256": artifact.metadata["model_config_sha256"],
                "method_id": artifact.method,
                "track": "matched",
                "seed": artifact.metadata["seed"],
                "generator_algorithm": RANDOM_GENERATOR_ALGORITHM,
                "distribution": lock["random_controls"]["distribution"],
                "d_model": int(artifact.direction.numel()),
                "selected_layer": artifact.layer,
                "position_schedule": "final_prompt_token",
                "intervention_geometry": artifact.intervention_geometry,
                "direction_float32_sha256": artifact.direction_sha256,
                "direction_artifact_sha256": artifact.artifact_sha256,
                "stage1_lock_sha256": artifact.metadata["stage1_lock_sha256"],
                "runner_commit": artifact.metadata["runner_commit"],
            }
        else:
            required_evidence_roles = {
                "gradient": {"gradient_construction_diagnostics"},
                "gradient_uncorrected": {"gradient_construction_diagnostics"},
                "caa": {"caa_construction_diagnostics"},
                "bipo": {"bipo_training_audit"},
                "persona_vector": {
                    "persona_construction_diagnostics",
                    "persona_scored_rollouts",
                },
            }[artifact.method]
            observed_evidence_roles = {
                str(item["role"]) for item in evidence_artifacts
            }
            if observed_evidence_roles != required_evidence_roles:
                raise RuntimeError(
                    f"{artifact.method} construction evidence roles must be "
                    f"{sorted(required_evidence_roles)}"
                )
            locked_configuration = locked_method_construction_configuration(
                lock, artifact.method, track
            )
            construction = {
                "schema_version": MAIN_CONSTRUCTION_SCHEMA_VERSION,
                "model_id": artifact.metadata["model_id"],
                "model_revision": artifact.metadata["model_revision"],
                "model_config_sha256": artifact.metadata["model_config_sha256"],
                "method_id": artifact.method,
                "track": track,
                "selected_layer": artifact.layer,
                "position_schedule": locked_position_schedule(artifact.method, track),
                "intervention_geometry": artifact.intervention_geometry,
                "direction_float32_sha256": artifact.direction_sha256,
                "direction_artifact_sha256": artifact.artifact_sha256,
                "dataset_sha256": artifact.metadata["dataset_sha256"],
                "protocol_sha256": artifact.metadata["protocol_sha256"],
                "stage1_lock_sha256": artifact.metadata["stage1_lock_sha256"],
                "runner_commit": artifact.metadata["runner_commit"],
                "locked_configuration": locked_configuration,
                "locked_configuration_sha256": sha256_json(locked_configuration),
                "evidence_artifacts": evidence_artifacts,
                "evidence_artifacts_sha256": sha256_json(evidence_artifacts),
            }
        construction_path = output_dir / f"{name}.construction.json"
        write_json(construction_path, construction)
        record.update(
            {
                "track": track,
                "construction_config_path": relative_path(
                    construction_path, field="construction config"
                ),
                "construction_config_sha256": sha256_file(construction_path),
            }
        )
        records.append(record)
    write_json(output_dir / "direction_manifest.json", {"directions": records})
    return records


def command_verify_stage1(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    lock = _load_lock(repo_root, args.lock.resolve())
    print(
        json.dumps(
            {
                "status": "stage_1_verified",
                "study": lock["study"],
                "lock_sha256": sha256_file(args.lock.resolve()),
                "dataset_sha256": lock["dataset"]["sha256"],
                "protocol_sha256": lock["protocol"]["sha256"],
            },
            indent=2,
        )
    )


def command_build_code_amendment(args: argparse.Namespace) -> None:
    """Build the canonical amendment after audit code is committed, before artifacts."""

    repo_root = args.repo_root.resolve()
    lock_path = args.lock.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    output = args.output.resolve()
    expected = (repo_root / OUTCOME_BLIND_AMENDMENT_PATH).resolve()
    if output != expected:
        raise RuntimeError(
            "outcome-blind amendment output must use its canonical config path"
        )
    payload = build_outcome_blind_amendment_manifest(
        repo_root,
        lock,
        stage1_lock_path=lock_path,
        amendment_code_commit=args.amendment_code_commit,
    )
    write_json(output, payload)


def command_verify_code_amendment(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    lock_path = args.lock.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    verified = verify_outcome_blind_amendment(
        repo_root,
        lock,
        stage1_lock_path=lock_path,
    )
    print(
        json.dumps(
            {"status": "outcome_blind_amendment_verified", **verified},
            indent=2,
        )
    )


def command_capture_environment(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    lock_path = args.lock.resolve()
    lock = _load_lock(repo_root, lock_path)
    record = capture_environment_record(
        stage1_lock_path=lock_path,
        lock=lock,
    )
    write_environment_record(args.output.resolve(), record)


def _load_fit_context(args: argparse.Namespace) -> tuple[Any, ...]:
    repo_root = args.repo_root.resolve()
    lock_path = args.lock.resolve()
    lock = _load_lock(repo_root, lock_path)
    dataset_path = repo_root / lock["dataset"]["path"]
    dataset = load_comparison_dataset(
        dataset_path, expected_sha256=lock["dataset"]["sha256"]
    )
    config_path = args.model_config.resolve()
    model_record = _model_record(lock, config_path)
    if sha256_file(config_path) != model_record["config_sha256"]:
        raise ValueError("model configuration hash differs from the stage-1 lock")
    backend = ResearchBackend.load(load_config(config_path), with_lens=False)
    validate_locked_choice_runtime(backend, model_record["runtime"])
    metadata = _common_metadata(repo_root, lock_path, lock, model_record)
    return repo_root, lock, dataset, model_record, backend, metadata


def command_fit(args: argparse.Namespace) -> None:
    repo_root, lock, dataset, model_record, backend, metadata = _load_fit_context(args)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    discovery = dataset["sp_splits"]["discovery"]
    layer = int(model_record["matched_intervention"]["layer_zero_based"])

    if args.method == "gradient":
        directions, diagnostics = fit_gradient_method(backend, discovery, layer=layer)
        artifacts = {
            "gradient_matched": make_direction_artifact(
                method="gradient",
                direction=directions["gradient_self_specific"],
                layer=layer,
                geometry="matched_final_prompt",
                metadata={**metadata, "track": "matched", "diagnostics": diagnostics},
            ),
            "gradient_uncorrected": make_direction_artifact(
                method="gradient_uncorrected",
                direction=directions["gradient_uncorrected"],
                layer=layer,
                geometry="matched_final_prompt",
                metadata={
                    **metadata,
                    "track": "matched",
                    "diagnostics": diagnostics,
                    "role": "diagnostic",
                },
            ),
        }
        diagnostics_path = output_dir / "gradient_construction_diagnostics.json"
        write_json(diagnostics_path, diagnostics)
        _write_artifacts(
            output_dir,
            artifacts,
            repo_root=repo_root,
            lock=lock,
            evidence_paths={"gradient_construction_diagnostics": diagnostics_path},
        )
    elif args.method == "caa":
        artifacts, diagnostics = fit_caa_artifacts_all_layers(
            backend,
            discovery,
            layers=tuple(range(int(model_record["architecture"]["blocks"]))),
            common_metadata=metadata,
        )
        matched = matched_artifact_from_canonical(backend, artifacts[layer])
        diagnostics_path = output_dir / "caa_diagnostics.json"
        write_json(diagnostics_path, diagnostics)
        _write_artifacts(
            output_dir,
            {
                "caa_matched": matched,
                **{
                    f"caa_canonical_layer_{layer_index:02d}": artifact
                    for layer_index, artifact in artifacts.items()
                },
            },
            repo_root=repo_root,
            lock=lock,
            evidence_paths={"caa_construction_diagnostics": diagnostics_path},
        )
    elif args.method == "bipo":
        if args.track not in {"matched", "canonical"}:
            raise ValueError("BiPO fitting requires --track matched or canonical")
        method_lock = lock["methods"]["bipo"]
        training_config = BiPOTrainingConfig(
            beta=float(method_lock["beta"]),
            learning_rate=float(method_lock["learning_rate"]),
            weight_decay=float(method_lock["weight_decay"]),
            max_grad_norm=float(method_lock["max_grad_norm"]),
            epochs=max(method_lock["checkpoint_epochs"]),
            checkpoint_epochs=tuple(method_lock["checkpoint_epochs"]),
            gradient_accumulation_steps=int(method_lock["gradient_accumulation_steps"]),
            warmup_steps=int(method_lock["warmup_steps"]),
            seed=int(method_lock["seed"]),
        )
        artifact, diagnostics = fit_bipo_artifact(
            backend,
            discovery,
            layer=layer,
            track=args.track,
            config=training_config,
            selected_checkpoint_epoch=int(method_lock["selected_checkpoint_epoch"]),
            common_metadata=metadata,
        )
        diagnostics_path = output_dir / f"bipo_{args.track}_training.json"
        write_json(diagnostics_path, diagnostics)
        _write_artifacts(
            output_dir,
            {f"bipo_{args.track}": artifact},
            repo_root=repo_root,
            lock=lock,
            evidence_paths={"bipo_training_audit": diagnostics_path},
        )
    elif args.method == "persona":
        if args.persona_rollouts is None:
            raise ValueError("persona fitting requires --persona-rollouts with blind judge scores")
        persona_path = repo_root / lock["methods"]["persona_vector"][
            "canonical_protocol_path"
        ]
        protocol = load_persona_protocol(persona_path)
        rollouts = read_rollouts(args.persona_rollouts.resolve())
        expected_provenance = persona_generation_provenance(
            protocol,
            model_id=str(model_record["model_id"]),
            model_revision=str(model_record["revision"]),
            model_config_sha256=str(model_record["config_sha256"]),
            stage1_lock_sha256=str(metadata["stage1_lock_sha256"]),
            runner_commit=str(metadata["runner_commit"]),
            persona_protocol_sha256=str(
                lock["methods"]["persona_vector"]["canonical_protocol_sha256"]
            ),
        )
        validate_rollouts(
            rollouts,
            protocol,
            rollouts_per_instruction_question=int(
                lock["methods"]["persona_vector"]["canonical_grid"][
                    "rollouts_per_instruction_question_per_polarity"
                ]
            ),
            require_scores=True,
            expected_generation_provenance=expected_provenance,
        )
        artifacts, diagnostics = fit_persona_artifacts_all_layers(
            backend,
            rollouts,
            protocol,
            layers=tuple(range(int(model_record["architecture"]["blocks"]))),
            common_metadata=metadata,
        )
        matched = matched_artifact_from_canonical(backend, artifacts[layer])
        diagnostics_path = output_dir / "persona_diagnostics.json"
        write_json(diagnostics_path, diagnostics)
        _write_artifacts(
            output_dir,
            {
                "persona_vector_matched": matched,
                **{
                    f"persona_vector_canonical_layer_{layer_index:02d}": artifact
                    for layer_index, artifact in artifacts.items()
                },
            },
            repo_root=repo_root,
            lock=lock,
            evidence_paths={
                "persona_construction_diagnostics": diagnostics_path,
                "persona_scored_rollouts": args.persona_rollouts.resolve(),
            },
        )
    elif args.method == "random":
        seeds = tuple(int(seed) for seed in lock["random_controls"]["seeds"])
        artifacts = random_control_artifacts(
            backend.torch,
            d_model=int(model_record["architecture"]["residual_width"]),
            layer=layer,
            seeds=seeds,
            common_metadata=metadata,
        )
        _write_artifacts(
            output_dir,
            {f"random_control_{index + 1:02d}": artifact for index, artifact in enumerate(artifacts)},
            repo_root=repo_root,
            lock=lock,
        )
    else:  # pragma: no cover - argparse protects this branch
        raise ValueError(args.method)


def command_generate_persona(args: argparse.Namespace) -> None:
    _, lock, _, model_record, backend, metadata = _load_fit_context(args)
    protocol_path = args.repo_root.resolve() / lock["methods"]["persona_vector"][
        "canonical_protocol_path"
    ]
    protocol = load_persona_protocol(protocol_path)
    generation = protocol["generation"]
    provenance = persona_generation_provenance(
        protocol,
        model_id=str(model_record["model_id"]),
        model_revision=str(model_record["revision"]),
        model_config_sha256=str(model_record["config_sha256"]),
        stage1_lock_sha256=str(metadata["stage1_lock_sha256"]),
        runner_commit=str(metadata["runner_commit"]),
        persona_protocol_sha256=str(
            lock["methods"]["persona_vector"]["canonical_protocol_sha256"]
        ),
    )
    rows = generate_persona_grid(
        backend,
        protocol,
        rollouts_per_instruction_question=int(
            generation["rollouts_per_instruction_question"]
        ),
        max_new_tokens=int(generation["max_new_tokens"]),
        temperature=float(generation["temperature"]),
        seed=int(generation["seed"]),
        generation_provenance=provenance,
    )
    write_jsonl(args.output.resolve(), [row.to_dict() for row in rows])


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"calibration result file is empty: {path}")
    return rows


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def command_run_forced_grid(args: argparse.Namespace) -> None:
    """Resolve the exact stage-one grid, then load one resident model and resume it."""

    repo_root = args.repo_root.resolve()
    lock_path = args.lock.resolve()
    lock = _load_lock(repo_root, lock_path)
    dataset = load_comparison_dataset(
        repo_root / lock["dataset"]["path"],
        expected_sha256=lock["dataset"]["sha256"],
    )
    config_path = args.model_config.resolve()
    model_record = _model_record(lock, config_path)
    if sha256_file(config_path) != model_record["config_sha256"]:
        raise ValueError("model configuration hash differs from the stage-one lock")
    plan = resolve_forced_grid_plan(
        repo_root=repo_root,
        lock=lock,
        stage1_lock_sha256=sha256_file(lock_path),
        model_id=str(model_record["model_id"]),
        direction_manifest_paths=[path.resolve() for path in args.direction_manifest],
        runner_commit=locked_runner_code_commit(repo_root, lock_path),
    )

    def load_backend(_: Mapping[str, Any]) -> ResearchBackend:
        return ResearchBackend.load(load_config(config_path), with_lens=False)

    summary = run_forced_grid(
        plan=plan,
        lock=lock,
        dataset=dataset,
        repo_root=repo_root,
        output_dir=args.output_dir.resolve(),
        backend_factory=load_backend,
        only_point_sha256s=args.only_point_sha256,
        max_new_points=args.max_new_points,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def command_build_calibration_summary(args: argparse.Namespace) -> None:
    """Build a hash-bound summary using only stage-1 locked limits and selection rules."""

    repo_root = args.repo_root.resolve()
    lock = _load_lock(repo_root, args.lock.resolve())
    dataset = load_comparison_dataset(
        repo_root / lock["dataset"]["path"],
        expected_sha256=lock["dataset"]["sha256"],
    )
    calibration = lock["calibration"]
    safety_limits = SafetyLimits.from_lock(calibration["safety_gates"])
    expected_forced_units = locked_forced_calibration_units(dataset, lock)
    expected_open_units = locked_open_confirmation_units(dataset, lock)
    grid_plan_path = args.grid_plan.resolve()
    grid_plan = _read_json_object(grid_plan_path, label="forced grid plan")
    point_paths = [path.resolve() for path in args.point_shards]
    if len(point_paths) != len(set(point_paths)):
        raise ValueError("calibration point shards must be unique")
    point_rows = [
        load_validated_point_rows(
            path,
            plan=grid_plan,
            lock=lock,
            dataset=dataset,
            repo_root=repo_root,
        )[0]
        for path in point_paths
    ]
    recheck_path = (
        None
        if args.interpolation_recheck_rows is None
        else args.interpolation_recheck_rows.resolve()
    )
    open_paths = [path.resolve() for path in args.open_confirmation_rows]
    if len(open_paths) != len(set(open_paths)):
        raise ValueError("open-confirmation row files must be unique")
    forced_paths = [*point_paths, *([] if recheck_path is None else [recheck_path])]
    all_paths = [*forced_paths, *open_paths]
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("forced and open calibration artifact paths must be disjoint")
    if args.pre_open_only and open_paths:
        raise ValueError("--pre-open-only cannot include open-confirmation rows")

    def artifact_records(paths: list[Path]) -> list[dict[str, str]]:
        records = []
        for path in paths:
            try:
                relative = path.relative_to(repo_root).as_posix()
            except ValueError as error:
                raise ValueError(
                    f"calibration result artifact must be inside the repository: {path}"
                ) from error
            records.append({"path": relative, "sha256": sha256_file(path)})
        return records

    forced_artifacts = artifact_records(forced_paths)
    open_artifacts = artifact_records(open_paths)
    grid_plan_artifact = artifact_records([grid_plan_path])[0]
    builder_path = repo_root / "src" / "sp_lense" / "comparison_calibration.py"
    summary = build_calibration_summary(
        point_rows,
        expected_forced_units=expected_forced_units,
        expected_open_units=expected_open_units,
        safety_limits=safety_limits,
        mode=args.mode,
        forced_result_rows_artifacts=forced_artifacts,
        open_result_rows_artifacts=open_artifacts,
        forced_grid_plan_artifact=grid_plan_artifact,
        calibration_config_sha256=sha256_json(calibration),
        builder_module_sha256=sha256_file(builder_path),
        interpolation_recheck_rows=(
            None if recheck_path is None else _read_jsonl_objects(recheck_path)
        ),
        open_confirmation_rows=[_read_jsonl_objects(path) for path in open_paths],
        allow_pending_open=bool(args.pre_open_only),
        fixed_strength=float(
            lock["comparison_tracks"]["matched_primary"]["fixed_strength"]
        ),
        target=float(
            calibration[
                "equal_efficacy_target_mean_self_minus_other_bidirectional_effect"
            ]
        ),
        tie_tolerance=float(calibration["canonical_layer_tie_tolerance"]),
    )
    write_json(args.output.resolve(), summary)


def command_judge_requests(args: argparse.Namespace) -> None:
    """Render exact blinded judge requests without calling an external service."""

    from .comparison_behavior import load_open_judge_protocol
    from .comparison_workflow import (
        build_open_judge_requests,
        build_persona_judge_requests,
        read_jsonl_objects,
    )

    repo_root = args.repo_root.resolve()
    lock = _load_lock(repo_root, args.lock.resolve())
    if args.kind == "open":
        protocol = load_open_judge_protocol(
            repo_root / lock["evaluation"]["open_behavior_judge"]["protocol_path"]
        )
        requests = build_open_judge_requests(
            read_jsonl_objects(args.input.resolve()), protocol
        )
    else:
        protocol = load_persona_protocol(
            repo_root / lock["methods"]["persona_vector"]["canonical_protocol_path"]
        )
        records = read_rollouts(args.input.resolve())
        requests = build_persona_judge_requests(
            records,
            protocol,
            rollouts_per_instruction_question=int(
                lock["methods"]["persona_vector"]["canonical_grid"][
                    "rollouts_per_instruction_question_per_polarity"
                ]
            ),
        )
    write_jsonl(args.output.resolve(), requests)


def command_attach_judgments(args: argparse.Namespace) -> None:
    """Attach an exact one-to-one set of raw judge responses to generated records."""

    from .comparison_behavior import load_open_judge_protocol
    from .comparison_workflow import (
        attach_open_judge_responses,
        attach_persona_judge_responses,
        read_jsonl_objects,
    )

    repo_root = args.repo_root.resolve()
    lock = _load_lock(repo_root, args.lock.resolve())
    responses = read_jsonl_objects(args.responses.resolve())
    if args.kind == "open":
        protocol = load_open_judge_protocol(
            repo_root / lock["evaluation"]["open_behavior_judge"]["protocol_path"]
        )
        output = attach_open_judge_responses(
            read_jsonl_objects(args.input.resolve()), responses, protocol
        )
    else:
        protocol = load_persona_protocol(
            repo_root / lock["methods"]["persona_vector"]["canonical_protocol_path"]
        )
        output = [
            item.to_dict()
            for item in attach_persona_judge_responses(
                read_rollouts(args.input.resolve()),
                responses,
                protocol,
                rollouts_per_instruction_question=int(
                    lock["methods"]["persona_vector"]["canonical_grid"][
                        "rollouts_per_instruction_question_per_polarity"
                    ]
                ),
            )
        ]
    write_jsonl(args.output.resolve(), output)


def command_verify_stage2(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    lock = _load_lock(repo_root, args.lock.resolve())
    verified = verify_stage2_manifest(repo_root, lock, args.stage2_manifest.resolve())
    print(
        json.dumps(
            {
                "status": "stage_2_verified",
                "stage1_lock_sha256": verified.stage1_lock_sha256,
                "stage2_manifest_sha256": verified.manifest_sha256,
            },
            indent=2,
        )
    )


def command_build_preopen(args: argparse.Namespace) -> None:
    """Build the canonical forced-only manifest before any validation-open run."""

    repo_root = args.repo_root.resolve()
    lock_path = args.lock.resolve()
    lock = _load_lock(repo_root, lock_path)
    expected = (
        repo_root / lock["lock_stages"]["pre_open"]["path"]
    ).resolve()
    output = args.output.resolve()
    if output != expected:
        raise RuntimeError("pre-open output path differs from the stage-1 lock")
    manifest = build_preopen_manifest(
        repo_root,
        lock,
        stage1_lock_path=lock_path,
        calibration_summary_paths=[path.resolve() for path in args.calibration_summary],
        direction_manifest_paths=[path.resolve() for path in args.direction_manifest],
    )
    write_json(output, manifest)


def command_verify_preopen(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    lock = _load_lock(repo_root, args.lock.resolve())
    verified = verify_preopen_manifest(
        repo_root, lock, args.preopen_manifest.resolve()
    )
    print(
        json.dumps(
            {
                "status": "pre_open_verified",
                "stage1_lock_sha256": verified.stage1_lock_sha256,
                "preopen_manifest_sha256": verified.manifest_sha256,
            },
            indent=2,
        )
    )


def command_build_stage2(args: argparse.Namespace) -> None:
    """Build the canonical manifest that gates every sealed run."""

    repo_root = args.repo_root.resolve()
    lock_path = args.lock.resolve()
    lock = _load_lock(repo_root, lock_path)
    expected = (repo_root / lock["lock_stages"]["stage_2"]["path"]).resolve()
    output = args.output.resolve()
    if output != expected:
        raise RuntimeError("stage-2 output path differs from the stage-1 lock")
    manifest = build_stage2_manifest(
        repo_root,
        lock,
        stage1_lock_path=lock_path,
        preopen_manifest_path=args.preopen_manifest.resolve(),
        environment_lock_path=args.environment_lock.resolve(),
        calibration_summary_paths=[
            path.resolve() for path in args.calibration_summary
        ],
        direction_manifest_paths=[path.resolve() for path in args.direction_manifest],
    )
    write_json(output, manifest)


def command_report(args: argparse.Namespace) -> None:
    """Build the locked machine-readable and Markdown reports from sealed artifacts."""

    from .comparison_report import build_comparison_report, write_comparison_report
    from .comparison_workflow import read_jsonl_objects

    repo_root = args.repo_root.resolve()
    lock_path = args.lock.resolve()
    lock = _load_lock(repo_root, lock_path)
    verified = verify_stage2_manifest(
        repo_root, lock, args.stage2_manifest.resolve()
    )
    dataset = load_comparison_dataset(
        repo_root / lock["dataset"]["path"],
        expected_sha256=lock["dataset"]["sha256"],
    )
    rows = [
        row
        for path in args.forced_rows
        for row in read_jsonl_objects(path.resolve())
    ]
    open_rows = (
        None
        if not args.open_rows
        else [
            row
            for path in args.open_rows
            for row in read_jsonl_objects(path.resolve())
        ]
    )
    jspace_records = (
        None
        if not args.jspace_records
        else [
            row
            for path in args.jspace_records
            for row in read_jsonl_objects(path.resolve())
        ]
    )
    construction_availability = None
    if args.construction_availability is not None:
        availability_path = args.construction_availability.resolve()
        loaded_availability = json.loads(availability_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_availability, Mapping):
            raise TypeError("construction availability manifest must contain a JSON object")
        records = loaded_availability.get("records")
        if not isinstance(records, list):
            raise TypeError("construction availability manifest records must be a list")
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise TypeError(f"construction availability record {index} must be an object")
            evidence_path = record.get("evidence_path")
            if not isinstance(evidence_path, str):
                raise TypeError(
                    f"construction availability record {index} evidence_path must be a string"
                )
            evidence = (repo_root / evidence_path).resolve()
            try:
                evidence.relative_to(repo_root)
            except ValueError as exc:
                raise ValueError(
                    f"construction availability record {index} evidence escapes the repository"
                ) from exc
            if not evidence.is_file():
                raise FileNotFoundError(
                    f"construction availability evidence is missing: {evidence}"
                )
            if sha256_file(evidence) != record.get("evidence_sha256"):
                raise ValueError(
                    f"construction availability record {index} evidence hash mismatch"
                )
        construction_availability = dict(loaded_availability)
    statistics = lock["statistics"]
    report = build_comparison_report(
        rows,
        verified_stage2=verified,
        stage1_lock=lock,
        locked_dataset=dataset,
        open_rows=open_rows,
        jspace_records=jspace_records,
        construction_availability=construction_availability,
        expected_hashes={
            "dataset_sha256": lock["dataset"]["sha256"],
            "protocol_sha256": lock["protocol"]["sha256"],
            "stage1_lock_sha256": sha256_file(lock_path),
            "stage2_manifest_sha256": verified.manifest_sha256,
        },
        bootstrap_replicates=int(statistics["bootstrap"]["replicates"]),
        bootstrap_seed=int(statistics["bootstrap"]["seed"]),
        minimum_bidirectional_consistency=float(
            statistics["minimum_bidirectional_consistency"]
        ),
    )
    write_comparison_report(
        report,
        json_path=args.output_json.resolve(),
        markdown_path=args.output_markdown.resolve(),
    )


def command_jspace(args: argparse.Namespace) -> None:
    """Run the secondary, explicitly non-gating sparse-cone comparison."""

    import torch

    from .jspace_comparison import (
        JSPACE_RECORD_SCHEMA,
        analyze_direction_against_jspace,
        estimate_jspace_resources,
        load_jspace_atom_artifact,
        prepare_atom_dictionary,
        validate_jspace_atom_manifest,
    )

    repo_root = args.repo_root.resolve()
    lock = _load_lock(repo_root, args.lock.resolve())
    direction_path = args.direction.resolve()
    artifact = read_direction_artifact(direction_path, torch)
    model_id = artifact.metadata.get("model_id")
    model_revision = artifact.metadata.get("model_revision")
    model_records = {
        (str(item["model_id"]), str(item["revision"])): item for item in lock["models"]
    }
    model_record = model_records.get((str(model_id), str(model_revision)))
    if model_record is None:
        raise ValueError("direction artifact model identity is absent from the stage-1 lock")
    if artifact.metadata.get("model_config_sha256") != model_record["config_sha256"]:
        raise ValueError("direction artifact model-config hash differs from the stage-1 lock")
    if artifact.metadata.get("track") != args.setup:
        raise ValueError("direction artifact track differs from --setup")
    expected_width = int(model_record["architecture"]["residual_width"])
    if int(artifact.direction.numel()) != expected_width:
        raise ValueError("direction width differs from the locked model residual width")

    jspace = _locked_jspace_settings(lock, str(model_id))
    if artifact.method not in set(jspace["method_ids"]):
        raise ValueError("J-space records are locked to the four primary comparison methods")
    locked_k = tuple(jspace["k_values"])
    if tuple(args.k) != locked_k:
        raise ValueError(f"J-space k values must equal the stage-1 lock: {locked_k}")
    if args.random_count != int(jspace["random_control_count"]):
        raise ValueError("J-space random-control count differs from the stage-1 lock")
    if args.random_seed != int(jspace["random_seed"]):
        raise ValueError("J-space random seed differs from the stage-1 lock")
    for name, value in (
        ("max_working_gib", args.max_working_gib),
        ("max_dictionary_read_tib", args.max_dictionary_read_tib),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
        if value != float(jspace["resource_limits"][name]):
            raise ValueError(f"{name} differs from the stage-1 J-space resource limit")

    lens_lock = jspace["lens"]
    base_record = {
        "schema_version": JSPACE_RECORD_SCHEMA,
        "model_id": str(model_id),
        "model_revision": str(model_revision),
        "model_config_sha256": str(model_record["config_sha256"]),
        "method": artifact.method,
        "setup": args.setup,
        "layer": artifact.layer,
        "direction_float32_sha256": artifact.direction_sha256,
        "direction_artifact_sha256": artifact.artifact_sha256,
        "direction_file_sha256": sha256_file(direction_path),
        "lens_provenance": {
            "file_sha256": lens_lock["file_sha256"],
            "revision": lens_lock["revision"],
            "source_layers": list(lens_lock["source_layers"]),
        },
        "non_gating": True,
        "used_for_primary_ranking": False,
    }
    available_layers = list(lens_lock["source_layers"])
    if artifact.layer not in available_layers:
        write_jsonl(
            args.output.resolve(),
            [{
                **base_record,
                "status": "not_run_lens_layer_unavailable",
                "reason": (
                    f"direction block {artifact.layer} is absent from the pinned Jacobian "
                    f"lens source layers {available_layers}"
                ),
                "available_source_layers": available_layers,
                "resource_estimate": None,
                "analysis": None,
            }],
        )
        return
    if args.atoms_manifest is None:
        raise ValueError("--atoms-manifest is required when the pinned lens contains the layer")

    validated = validate_jspace_atom_manifest(args.atoms_manifest.resolve())
    manifest = validated.manifest
    expected_model = {
        "id": str(model_id),
        "revision": str(model_revision),
        "config_sha256": str(model_record["config_sha256"]),
    }
    if manifest["model"] != expected_model:
        raise ValueError("J-space atom manifest model identity differs from the direction")
    locked_lens_fields = (
        "repository",
        "filename",
        "revision",
        "file_sha256",
        "file_size_bytes",
        "n_prompts",
        "source_layers",
        "fitted_model_id",
        "fitted_model_revision",
        "transfer_status",
    )
    lens_mismatches = {
        field: (lens_lock.get(field), manifest["lens"].get(field))
        for field in locked_lens_fields
        if lens_lock.get(field) != manifest["lens"].get(field)
    }
    if lens_mismatches:
        raise ValueError(f"J-space atom manifest lens provenance mismatch: {lens_mismatches}")
    if manifest["construction"]["reference_repository_commit"] != lock["sources"][
        "j_space"
    ]["commit"]:
        raise ValueError("J-space atom manifest uses the wrong reference implementation commit")
    shape = manifest["atoms"]["shape"]
    if manifest["layer"] != artifact.layer or int(shape[1]) != expected_width:
        raise ValueError("J-space atom manifest layer/width differs from the direction")

    estimate = estimate_jspace_resources(
        n_atoms=int(shape[0]),
        d_model=int(shape[1]),
        random_count=args.random_count,
        max_k=max(locked_k),
    )
    artifact_fields = {
        "atoms_manifest_sha256": validated.manifest_sha256,
        "atoms_file_sha256": manifest["atoms"]["file_sha256"],
        "atoms_float32_sha256": manifest["atoms"]["float32_sha256"],
    }
    over_memory = estimate["estimated_peak_working_gib"] > args.max_working_gib
    over_traffic = (
        estimate["dictionary_read_upper_bound_tib"] > args.max_dictionary_read_tib
    )
    if over_memory or over_traffic:
        reasons = []
        if over_memory:
            reasons.append(
                f"estimated peak {estimate['estimated_peak_working_gib']:.3f} GiB exceeds "
                f"the declared {args.max_working_gib:.3f} GiB limit"
            )
        if over_traffic:
            reasons.append(
                f"estimated dictionary traffic {estimate['dictionary_read_upper_bound_tib']:.3f} "
                f"TiB exceeds the declared {args.max_dictionary_read_tib:.3f} TiB limit"
            )
        write_jsonl(
            args.output.resolve(),
            [{
                **base_record,
                **artifact_fields,
                "status": "not_run_resource_limited",
                "reason": "; ".join(reasons),
                "resource_limits": {
                    "max_working_gib": args.max_working_gib,
                    "max_dictionary_read_tib": args.max_dictionary_read_tib,
                },
                "resource_estimate": estimate,
                "analysis": None,
            }],
        )
        return

    loaded = load_jspace_atom_artifact(args.atoms_manifest.resolve())
    prepared = prepare_atom_dictionary(loaded.atoms, width=expected_width)
    analysis = analyze_direction_against_jspace(
        artifact.direction,
        prepared,
        k_values=locked_k,
        random_count=args.random_count,
        random_seed=args.random_seed,
        token_labels=loaded.metadata.token_labels,
        known_direction_float32_sha256=artifact.direction_sha256,
        known_atoms_float32_sha256=manifest["atoms"]["float32_sha256"],
    )
    write_jsonl(
        args.output.resolve(),
        [{
            **base_record,
            **artifact_fields,
            "status": "complete",
            "reason": None,
            "resource_limits": {
                "max_working_gib": args.max_working_gib,
                "max_dictionary_read_tib": args.max_dictionary_read_tib,
            },
            "resource_estimate": estimate,
            "analysis": analysis,
        }],
    )


def _locked_jspace_settings(lock: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
    from .jspace_comparison import validate_locked_jspace_config

    settings = validate_locked_jspace_config(lock)
    evaluation = lock.get("evaluation")
    if not isinstance(evaluation, Mapping):  # pragma: no cover - validator guards this
        raise TypeError("stage-1 lock evaluation must be an object")
    models = settings.get("models")
    model = models.get(model_id) if isinstance(models, Mapping) else None
    if not isinstance(model, Mapping) or not isinstance(model.get("lens"), Mapping):
        raise TypeError(f"stage-1 lock lacks J-space lens provenance for {model_id}")
    return {**settings, "lens": model["lens"]}


def command_prepare_jspace_atoms(args: argparse.Namespace) -> None:
    """Extract a provenance-bound binary J-space atom dictionary for one layer."""

    import importlib.metadata

    from huggingface_hub import hf_hub_download

    from .jspace_comparison import extract_jspace_atoms, write_jspace_atom_artifact

    repo_root = args.repo_root.resolve()
    lock = _load_lock(repo_root, args.lock.resolve())
    config_path = args.model_config.resolve()
    model_record = _model_record(lock, config_path)
    config = load_config(config_path)
    if (
        config.model.id != model_record["model_id"]
        or config.model.revision != model_record["revision"]
        or sha256_file(config_path) != model_record["config_sha256"]
    ):
        raise ValueError("model configuration differs from the stage-1 lock")
    jspace = _locked_jspace_settings(lock, config.model.id)
    lens_lock = jspace["lens"]
    config_lens = {
        "repository": config.model.lens,
        "filename": config.model.lens_filename,
        "revision": config.model.lens_revision,
    }
    mismatches = {
        field: (lens_lock.get(field), value)
        for field, value in config_lens.items()
        if lens_lock.get(field) != value
    }
    if mismatches:
        raise ValueError(f"model config differs from locked J-space lens: {mismatches}")
    if args.layer not in lens_lock["source_layers"]:
        raise ValueError("requested layer is absent from the pinned J-space lens")
    if not config.model.lens_filename or not config.model.lens_revision:
        raise ValueError("J-space extraction requires a pinned lens filename and revision")
    lens_path = Path(
        hf_hub_download(
            repo_id=config.model.lens,
            filename=config.model.lens_filename,
            revision=config.model.lens_revision,
        )
    ).resolve()
    if (
        sha256_file(lens_path) != lens_lock["file_sha256"]
        or lens_path.stat().st_size != lens_lock["file_size_bytes"]
    ):
        raise ValueError("downloaded Jacobian lens bytes differ from the stage-1 lock")

    backend = ResearchBackend.load(config, with_lens=True)
    validate_locked_choice_runtime(backend, model_record["runtime"])
    atoms, labels, extraction = extract_jspace_atoms(
        backend, layer=args.layer, chunk_size=args.chunk_size
    )
    if (
        extraction["lens_n_prompts"] != lens_lock["n_prompts"]
        or extraction["lens_source_layers"] != lens_lock["source_layers"]
    ):
        raise ValueError("loaded Jacobian lens payload differs from the stage-1 lock")
    write_jspace_atom_artifact(
        manifest_path=args.manifest_output.resolve(),
        atoms_path=args.atoms_output.resolve(),
        labels_path=args.token_labels_output.resolve(),
        atoms=atoms,
        token_labels=labels,
        model_id=config.model.id,
        model_revision=str(config.model.revision),
        model_config_sha256=str(model_record["config_sha256"]),
        lens_repository=str(lens_lock["repository"]),
        lens_filename=str(lens_lock["filename"]),
        lens_revision=str(lens_lock["revision"]),
        lens_file_path=lens_path,
        lens_n_prompts=int(extraction["lens_n_prompts"]),
        lens_source_layers=extraction["lens_source_layers"],
        lens_fitted_model_id=str(lens_lock["fitted_model_id"]),
        lens_fitted_model_revision=str(lens_lock["fitted_model_revision"]),
        lens_transfer_status=str(lens_lock["transfer_status"]),
        layer=args.layer,
        tokenizer_id=config.model.id,
        tokenizer_revision=str(config.model.revision),
        unembedding_shape=extraction["unembedding_shape"],
        unembedding_float32_sha256=extraction["unembedding_float32_sha256"],
        implementation_package_version=importlib.metadata.version("transformer-lens"),
        reference_repository_commit=str(lock["sources"]["j_space"]["commit"]),
    )


def _partition_cases(dataset: Mapping[str, Any], lock: Mapping[str, Any], split: str):
    split_key = "validation_ids" if split == "validation" else "sealed_ids"
    partitions = lock["dataset"]["partitions"]
    collateral = dataset["collateral_cases"]
    return {
        "benign_compliance": select_cases_by_locked_ids(
            collateral["benign_compliance"], partitions["benign_compliance"][split_key]
        ),
        "general_capability": select_cases_by_locked_ids(
            collateral["general_capability"], partitions["general_capability"][split_key]
        ),
        "refusal": select_cases_by_locked_ids(
            collateral["refusal"], partitions["refusal"][split_key]
        ),
        "option_order_sentinels": select_cases_by_locked_ids(
            collateral["option_order_sentinels"],
            partitions["option_order_sentinels"][split_key],
        ),
    }


def command_evaluate_forced(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    if args.split == "validation" and args.calibration_summary_sha256 != "0" * 64:
        raise RuntimeError(
            "validation rows must use the all-zero pre-summary SHA-256 sentinel"
        )
    if args.split == "sealed_test" and args.calibration_summary_sha256 == "0" * 64:
        raise RuntimeError("sealed rows require the approved nonzero calibration summary hash")
    lock_path = args.lock.resolve()
    lock = _load_lock(repo_root, lock_path)
    dataset = load_comparison_dataset(
        repo_root / lock["dataset"]["path"], expected_sha256=lock["dataset"]["sha256"]
    )
    config_path = args.model_config.resolve()
    model_record = _model_record(lock, config_path)
    if sha256_file(config_path) != model_record["config_sha256"]:
        raise ValueError("model configuration hash differs from the stage-1 lock")
    verified_stage2 = None
    approved_setup = None
    if args.split == "sealed_test":
        if args.stage2_manifest is None:
            raise RuntimeError("sealed evaluation requires --stage2-manifest")
        verified_stage2 = verify_stage2_manifest(
            repo_root, lock, args.stage2_manifest.resolve()
        )
        direction_path = args.direction.resolve()
        direction_record = json.loads(direction_path.read_text(encoding="utf-8"))
        if not isinstance(direction_record, Mapping):
            raise TypeError("direction artifact must contain a JSON object")
        required_direction_fields = (
            "method",
            "layer",
            "direction_sha256",
            "artifact_sha256",
        )
        missing_direction_fields = [
            field for field in required_direction_fields if direction_record.get(field) is None
        ]
        if missing_direction_fields:
            raise ValueError(
                "direction artifact lacks sealed identity fields: "
                f"{missing_direction_fields}"
            )
        method_id = str(direction_record["method"])
        selected_layer = direction_record["layer"]
        if isinstance(selected_layer, bool) or not isinstance(selected_layer, int):
            raise TypeError("direction artifact layer must be an integer")
        approved_setup = assert_approved_setup(
            verified_stage2,
            repo_root=repo_root,
            model_id=str(model_record["model_id"]),
            model_revision=str(model_record["revision"]),
            model_config_sha256=str(model_record["config_sha256"]),
            method_id=method_id,
            track=args.track,
            direction_path=direction_path,
            direction_file_sha256=sha256_file(direction_path),
            direction_float32_sha256=str(direction_record["direction_sha256"]),
            direction_artifact_sha256=str(direction_record["artifact_sha256"]),
            selected_strength=args.strength,
            selected_layer=selected_layer,
            position_schedule=locked_position_schedule(method_id, args.track),
            construction_config_sha256=args.construction_config_sha256,
            calibration_summary_sha256=args.calibration_summary_sha256,
        )
    backend = ResearchBackend.load(load_config(config_path), with_lens=False)
    validate_locked_choice_runtime(backend, model_record["runtime"])
    artifact = read_direction_artifact(args.direction.resolve(), backend.torch)
    artifact_model = artifact.metadata.get("model_id")
    artifact_revision = artifact.metadata.get("model_revision")
    if artifact_model != model_record["model_id"] or artifact_revision != model_record["revision"]:
        raise ValueError("direction artifact model identity differs from the selected model")
    gate = SealedEvaluationGate(
        sealed_ids_from_dataset_and_lock(dataset, lock),
        verified_stage2=verified_stage2,
    )
    setup = MethodSetup(artifact, artifact.method, args.track, args.strength)
    if approved_setup is not None:
        setup.validate()
        observed_setup = {
            "method_id": setup.method_id,
            "track": setup.track,
            "selected_strength": float(setup.strength),
            "selected_layer": setup.artifact.layer,
            "position_schedule": setup.position,
            "direction_float32_sha256": setup.artifact.direction_sha256,
            "direction_artifact_sha256": setup.artifact.artifact_sha256,
        }
        mismatches = {
            key: (approved_setup.get(key), value)
            for key, value in observed_setup.items()
            if approved_setup.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"loaded sealed setup differs from stage-2 approval: {mismatches}"
            )
    identity = EvaluationIdentity(
        model_id=model_record["model_id"],
        model_revision=model_record["revision"],
        dataset_sha256=lock["dataset"]["sha256"],
        protocol_sha256=lock["protocol"]["sha256"],
        config_sha256=model_record["config_sha256"],
        run_seed=int(lock["statistics"]["bootstrap"]["seed"]),
        stage1_lock_sha256=sha256_file(lock_path),
        stage2_manifest_sha256=(
            verified_stage2.manifest_sha256 if verified_stage2 is not None else "0" * 64
        ),
        calibration_summary_sha256=args.calibration_summary_sha256,
        construction_config_sha256=args.construction_config_sha256,
        runner_commit=locked_runner_code_commit(repo_root, lock_path),
        control_source_method_id=(
            approved_setup.get("control_source_method_id")
            if approved_setup is not None
            else None
        ),
        control_source_strength=(
            approved_setup.get("control_source_strength")
            if approved_setup is not None
            else None
        ),
        control_source_calibration_summary_sha256=(
            approved_setup.get("control_source_calibration_summary_sha256")
            if approved_setup is not None
            else None
        ),
    )
    core = dataset["sp_splits"][args.split]
    rows = evaluate_sp_cases(
        backend,
        core,
        setup=setup,
        identity=identity,
        split=args.split,
        gate=gate,
    )
    for family, cases in _partition_cases(dataset, lock, args.split).items():
        if family == "option_order_sentinels":
            rows.extend(
                evaluate_option_order_sentinels(
                    backend,
                    cases,
                    setup=setup,
                    identity=identity,
                    split=args.split,
                    gate=gate,
                )
            )
        else:
            rows.extend(
                evaluate_collateral_cases(
                    backend,
                    cases,
                    setup=setup,
                    identity=identity,
                    split=args.split,
                    family=family,
                    gate=gate,
                )
            )
    if args.split == "sealed_test" and args.include_tbsp:
        rows.extend(
            evaluate_tbsp_cases(
                backend,
                dataset["tbsp_cases"],
                setup=setup,
                identity=identity,
                gate=gate,
            )
        )
    write_jsonl(args.output.resolve(), rows)


def command_generate_open(args: argparse.Namespace) -> None:
    """Generate locked open-response triplets with the same sealed setup gate."""

    from .comparison_behavior import generate_open_cases

    repo_root = args.repo_root.resolve()
    if args.split == "validation" and args.calibration_summary_sha256 != "0" * 64:
        raise RuntimeError(
            "validation rows must use the all-zero pre-summary SHA-256 sentinel"
        )
    if args.split == "sealed_test" and args.calibration_summary_sha256 == "0" * 64:
        raise RuntimeError("sealed rows require the approved nonzero calibration summary hash")
    lock_path = args.lock.resolve()
    lock = _load_lock(repo_root, lock_path)
    dataset = load_comparison_dataset(
        repo_root / lock["dataset"]["path"],
        expected_sha256=lock["dataset"]["sha256"],
    )
    config_path = args.model_config.resolve()
    model_record = _model_record(lock, config_path)
    if sha256_file(config_path) != model_record["config_sha256"]:
        raise ValueError("model configuration hash differs from the stage-1 lock")
    verified_stage2 = None
    verified_preopen = None
    approved_setup = None
    direction_path = args.direction.resolve()
    direction_record = json.loads(direction_path.read_text(encoding="utf-8"))
    if not isinstance(direction_record, Mapping):
        raise TypeError("direction artifact must contain a JSON object")
    required = ("method", "layer", "direction_sha256", "artifact_sha256")
    missing = [field for field in required if direction_record.get(field) is None]
    if missing:
        raise ValueError(f"direction artifact lacks identity fields: {missing}")
    layer = direction_record["layer"]
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise TypeError("direction artifact layer must be an integer")
    method_id = str(direction_record["method"])
    if args.split == "validation":
        if args.preopen_manifest is None:
            raise RuntimeError("validation open generation requires --preopen-manifest")
        verified_preopen = verify_preopen_manifest(
            repo_root, lock, args.preopen_manifest.resolve()
        )
        approved_setup = assert_preopen_approved_setup(
            verified_preopen,
            repo_root=repo_root,
            model_id=str(model_record["model_id"]),
            model_revision=str(model_record["revision"]),
            model_config_sha256=str(model_record["config_sha256"]),
            method_id=method_id,
            track=args.track,
            direction_path=direction_path,
            selected_strength=args.strength,
            selected_layer=layer,
            position_schedule=locked_position_schedule(method_id, args.track),
            construction_config_sha256=args.construction_config_sha256,
        )
    if args.split == "sealed_test":
        if args.stage2_manifest is None:
            raise RuntimeError("sealed open generation requires --stage2-manifest")
        verified_stage2 = verify_stage2_manifest(
            repo_root, lock, args.stage2_manifest.resolve()
        )
        approved_setup = assert_approved_setup(
            verified_stage2,
            repo_root=repo_root,
            model_id=str(model_record["model_id"]),
            model_revision=str(model_record["revision"]),
            model_config_sha256=str(model_record["config_sha256"]),
            method_id=method_id,
            track=args.track,
            direction_path=direction_path,
            direction_file_sha256=sha256_file(direction_path),
            direction_float32_sha256=str(direction_record["direction_sha256"]),
            direction_artifact_sha256=str(direction_record["artifact_sha256"]),
            selected_strength=args.strength,
            selected_layer=layer,
            position_schedule=locked_position_schedule(method_id, args.track),
            construction_config_sha256=args.construction_config_sha256,
            calibration_summary_sha256=args.calibration_summary_sha256,
        )

    backend = ResearchBackend.load(load_config(config_path), with_lens=False)
    validate_locked_choice_runtime(backend, model_record["runtime"])
    artifact = read_direction_artifact(direction_path, backend.torch)
    if artifact.metadata.get("model_id") != model_record["model_id"] or artifact.metadata.get(
        "model_revision"
    ) != model_record["revision"]:
        raise ValueError("direction artifact model identity differs from the selected model")
    setup = MethodSetup(artifact, artifact.method, args.track, args.strength)
    setup.validate()
    if approved_setup is not None:
        observed_setup = {
            "method_id": setup.method_id,
            "track": setup.track,
            "selected_strength": float(setup.strength),
            "selected_layer": setup.artifact.layer,
            "position_schedule": setup.position,
            "direction_float32_sha256": setup.artifact.direction_sha256,
            "direction_artifact_sha256": setup.artifact.artifact_sha256,
        }
        mismatches = {
            key: (approved_setup.get(key), value)
            for key, value in observed_setup.items()
            if approved_setup.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"loaded sealed setup differs from stage-2 approval: {mismatches}"
            )
    identity = EvaluationIdentity(
        model_id=model_record["model_id"],
        model_revision=model_record["revision"],
        dataset_sha256=lock["dataset"]["sha256"],
        protocol_sha256=lock["protocol"]["sha256"],
        config_sha256=model_record["config_sha256"],
        run_seed=int(lock["statistics"]["bootstrap"]["seed"]),
        stage1_lock_sha256=sha256_file(lock_path),
        stage2_manifest_sha256=(
            verified_stage2.manifest_sha256 if verified_stage2 is not None else "0" * 64
        ),
        calibration_summary_sha256=args.calibration_summary_sha256,
        construction_config_sha256=args.construction_config_sha256,
        runner_commit=(
            verified_stage2.runner_parent_commit
            if verified_stage2 is not None
            else verified_preopen.runner_parent_commit
        ),
        control_source_method_id=(
            approved_setup.get("control_source_method_id")
            if approved_setup is not None
            else None
        ),
        control_source_strength=(
            approved_setup.get("control_source_strength")
            if approved_setup is not None
            else None
        ),
        control_source_calibration_summary_sha256=(
            approved_setup.get("control_source_calibration_summary_sha256")
            if approved_setup is not None
            else None
        ),
    )
    gate = SealedEvaluationGate(
        sealed_ids_from_dataset_and_lock(dataset, lock),
        verified_stage2=verified_stage2,
    )
    split_key = "validation_ids" if args.split == "validation" else "sealed_ids"
    rows = generate_open_cases(
        backend,
        dataset=dataset,
        locked_case_ids=lock["dataset"]["partitions"]["open_ended"][split_key],
        setup=setup,
        identity=identity,
        split=args.split,
        gate=gate,
    )
    write_jsonl(args.output.resolve(), rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preregistered gradient/CAA/BiPO/persona steering comparison"
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--lock", type=Path, default=Path("configs/steering_comparison_lock.json")
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify-stage1")
    verify.set_defaults(function=command_verify_stage1)

    build_amendment = subparsers.add_parser("build-code-amendment")
    build_amendment.add_argument("--amendment-code-commit")
    build_amendment.add_argument(
        "--output",
        type=Path,
        default=Path(OUTCOME_BLIND_AMENDMENT_PATH),
    )
    build_amendment.set_defaults(function=command_build_code_amendment)

    verify_amendment = subparsers.add_parser("verify-code-amendment")
    verify_amendment.set_defaults(function=command_verify_code_amendment)

    environment = subparsers.add_parser("capture-environment")
    environment.add_argument("--output", type=Path, required=True)
    environment.set_defaults(function=command_capture_environment)

    fit = subparsers.add_parser("fit")
    fit.add_argument("--model-config", type=Path, required=True)
    fit.add_argument("--method", choices=("gradient", "caa", "bipo", "persona", "random"), required=True)
    fit.add_argument("--track", choices=("matched", "canonical"))
    fit.add_argument("--persona-rollouts", type=Path)
    fit.add_argument("--output", type=Path, required=True)
    fit.set_defaults(function=command_fit)

    persona = subparsers.add_parser("generate-persona")
    persona.add_argument("--model-config", type=Path, required=True)
    persona.add_argument("--output", type=Path, required=True)
    persona.set_defaults(function=command_generate_persona)

    forced_grid = subparsers.add_parser("run-forced-grid")
    forced_grid.add_argument("--model-config", type=Path, required=True)
    forced_grid.add_argument(
        "--direction-manifest", type=Path, nargs="+", required=True
    )
    forced_grid.add_argument("--only-point-sha256", nargs="*")
    forced_grid.add_argument("--max-new-points", type=int)
    forced_grid.add_argument("--output-dir", type=Path, required=True)
    forced_grid.set_defaults(function=command_run_forced_grid)

    calibrate = subparsers.add_parser("build-calibration-summary")
    calibrate.add_argument("--mode", choices=("matched", "canonical"), required=True)
    calibrate.add_argument("--grid-plan", type=Path, required=True)
    calibrate.add_argument("--point-shards", type=Path, nargs="+", required=True)
    calibrate.add_argument("--interpolation-recheck-rows", type=Path)
    calibrate.add_argument("--open-confirmation-rows", type=Path, nargs="*", default=())
    calibrate.add_argument("--pre-open-only", action="store_true")
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.set_defaults(function=command_build_calibration_summary)

    judge_requests = subparsers.add_parser("judge-requests")
    judge_requests.add_argument("--kind", choices=("persona", "open"), required=True)
    judge_requests.add_argument("--input", type=Path, required=True)
    judge_requests.add_argument("--output", type=Path, required=True)
    judge_requests.set_defaults(function=command_judge_requests)

    judge_attach = subparsers.add_parser("attach-judgments")
    judge_attach.add_argument("--kind", choices=("persona", "open"), required=True)
    judge_attach.add_argument("--input", type=Path, required=True)
    judge_attach.add_argument("--responses", type=Path, required=True)
    judge_attach.add_argument("--output", type=Path, required=True)
    judge_attach.set_defaults(function=command_attach_judgments)

    verify_stage2 = subparsers.add_parser("verify-stage2")
    verify_stage2.add_argument("--stage2-manifest", type=Path, required=True)
    verify_stage2.set_defaults(function=command_verify_stage2)

    build_preopen = subparsers.add_parser("build-preopen-lock")
    build_preopen.add_argument(
        "--calibration-summary", type=Path, nargs="+", required=True
    )
    build_preopen.add_argument(
        "--direction-manifest", type=Path, nargs="+", required=True
    )
    build_preopen.add_argument("--output", type=Path, required=True)
    build_preopen.set_defaults(function=command_build_preopen)

    verify_preopen = subparsers.add_parser("verify-preopen")
    verify_preopen.add_argument("--preopen-manifest", type=Path, required=True)
    verify_preopen.set_defaults(function=command_verify_preopen)

    build_stage2 = subparsers.add_parser("build-stage2-manifest")
    build_stage2.add_argument("--preopen-manifest", type=Path, required=True)
    build_stage2.add_argument("--environment-lock", type=Path, required=True)
    build_stage2.add_argument(
        "--calibration-summary", type=Path, nargs="+", required=True
    )
    build_stage2.add_argument(
        "--direction-manifest", type=Path, nargs="+", required=True
    )
    build_stage2.add_argument("--output", type=Path, required=True)
    build_stage2.set_defaults(function=command_build_stage2)

    report = subparsers.add_parser("report")
    report.add_argument("--stage2-manifest", type=Path, required=True)
    report.add_argument("--forced-rows", type=Path, nargs="+", required=True)
    report.add_argument("--open-rows", type=Path, nargs="*")
    report.add_argument("--jspace-records", type=Path, nargs="*")
    report.add_argument("--construction-availability", type=Path)
    report.add_argument("--output-json", type=Path, required=True)
    report.add_argument("--output-markdown", type=Path, required=True)
    report.set_defaults(function=command_report)

    prepare_jspace = subparsers.add_parser("prepare-jspace-atoms")
    prepare_jspace.add_argument("--model-config", type=Path, required=True)
    prepare_jspace.add_argument("--layer", type=int, required=True)
    prepare_jspace.add_argument("--chunk-size", type=int, default=1024)
    prepare_jspace.add_argument("--atoms-output", type=Path, required=True)
    prepare_jspace.add_argument("--token-labels-output", type=Path, required=True)
    prepare_jspace.add_argument("--manifest-output", type=Path, required=True)
    prepare_jspace.set_defaults(function=command_prepare_jspace_atoms)

    jspace = subparsers.add_parser("jspace")
    jspace.add_argument("--direction", type=Path, required=True)
    jspace.add_argument("--atoms-manifest", type=Path)
    jspace.add_argument("--setup", choices=("matched", "canonical"), required=True)
    jspace.add_argument("--k", type=int, nargs="+", default=[8, 16, 25])
    jspace.add_argument("--random-count", type=int, default=50)
    jspace.add_argument("--random-seed", type=int, default=20_260_824)
    jspace.add_argument("--max-working-gib", type=float, default=8.0)
    jspace.add_argument("--max-dictionary-read-tib", type=float, default=4.0)
    jspace.add_argument("--output", type=Path, required=True)
    jspace.set_defaults(function=command_jspace)

    evaluate = subparsers.add_parser("evaluate-forced")
    evaluate.add_argument("--model-config", type=Path, required=True)
    evaluate.add_argument("--direction", type=Path, required=True)
    evaluate.add_argument("--track", choices=("matched", "canonical"), required=True)
    evaluate.add_argument("--strength", type=float, required=True)
    evaluate.add_argument("--split", choices=("validation", "sealed_test"), required=True)
    evaluate.add_argument("--stage2-manifest", type=Path)
    evaluate.add_argument(
        "--calibration-summary-sha256", type=_sha256_argument, required=True
    )
    evaluate.add_argument(
        "--construction-config-sha256", type=_sha256_argument, required=True
    )
    evaluate.add_argument("--include-tbsp", action="store_true")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(function=command_evaluate_forced)

    open_generate = subparsers.add_parser("generate-open")
    open_generate.add_argument("--model-config", type=Path, required=True)
    open_generate.add_argument("--direction", type=Path, required=True)
    open_generate.add_argument(
        "--track", choices=("matched", "canonical"), required=True
    )
    open_generate.add_argument("--strength", type=float, required=True)
    open_generate.add_argument(
        "--split", choices=("validation", "sealed_test"), required=True
    )
    open_generate.add_argument("--stage2-manifest", type=Path)
    open_generate.add_argument("--preopen-manifest", type=Path)
    open_generate.add_argument(
        "--calibration-summary-sha256", type=_sha256_argument, required=True
    )
    open_generate.add_argument(
        "--construction-config-sha256", type=_sha256_argument, required=True
    )
    open_generate.add_argument("--output", type=Path, required=True)
    open_generate.set_defaults(function=command_generate_open)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.lock.is_absolute():
        args.lock = args.repo_root / args.lock
    args.function(args)


if __name__ == "__main__":
    main()
