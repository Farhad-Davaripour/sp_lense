from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import struct
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

EXPERIMENT_RELATIVE = Path("experiments/bipo_warmup_sensitivity")
SENSITIVITY_METHOD_ID = "bipo_warmup11_sensitivity"
PARENT_METHOD_ID = "bipo"
ANALYSIS_TIER = "secondary_sensitivity_only"
PLAN_SCHEMA = "sp_lense.bipo_warmup_sensitivity.evaluation_plan.v1"
RECEIPT_SCHEMA = "sp_lense.bipo_warmup_sensitivity.main_final_freeze_receipt.v1"
REPORT_SCHEMA = "sp_lense.bipo_warmup_sensitivity.report.v1"
REPORT_MANIFEST_SCHEMA = "sp_lense.bipo_warmup_sensitivity.report_manifest.v1"
FORCED_ROWS_PER_SETUP = 1350
OPEN_ROWS_PER_SETUP = 96
EXPECTED_CONDITIONS = frozenset({"baseline", "plus", "minus"})
EXPECTED_MODEL_TRACKS = frozenset(
    {
        ("qwen35_08b", "matched"),
        ("qwen35_08b", "canonical"),
        ("qwen35_2b", "matched"),
        ("qwen35_2b", "canonical"),
    }
)


class PostConfirmatoryGateError(RuntimeError):
    """Raised before a model load when the secondary evaluation is not authorized."""


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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PostConfirmatoryGateError(f"{label} must contain one JSON object")
    return value


def read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise PostConfirmatoryGateError(f"{label} has a blank row at line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise PostConfirmatoryGateError(f"{label} line {line_number} is not a JSON object")
            rows.append(value)
    if not rows:
        raise PostConfirmatoryGateError(f"{label} is empty")
    return rows


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(dict(row), sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise PostConfirmatoryGateError(f"overwrite is forbidden: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_git(repo_root: Path, arguments: Sequence[str], *, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PostConfirmatoryGateError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.strip()


def _remote_main_commit(repo_root: Path) -> str:
    lines = _run_git(repo_root, ["ls-remote", "--heads", "origin", "refs/heads/main"]).splitlines()
    if len(lines) != 1:
        raise PostConfirmatoryGateError("origin/main did not resolve to exactly one commit")
    fields = lines[0].split()
    if (
        len(fields) != 2
        or len(fields[0]) != 40
        or any(character not in "0123456789abcdef" for character in fields[0])
        or fields[1] != "refs/heads/main"
    ):
        raise PostConfirmatoryGateError("origin/main response has an invalid identity")
    return fields[0]


def default_roots() -> tuple[Path, Path]:
    experiment_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root, experiment_root


def _resolve_repo_file(repo_root: Path, relative: str, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise PostConfirmatoryGateError(f"{label} must be repository-relative")
    path = (repo_root / relative).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise PostConfirmatoryGateError(f"{label} escapes the repository") from exc
    if not path.is_file():
        raise PostConfirmatoryGateError(f"{label} is missing: {relative}")
    return path


def _resolve_under(root: Path, relative: str, *, label: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise PostConfirmatoryGateError(f"{label} must be relative")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise PostConfirmatoryGateError(f"{label} escapes its locked root") from exc
    return resolved


def _load_construction_runner(experiment_root: Path) -> Any:
    path = experiment_root / "scripts" / "run_sensitivity.py"
    spec = importlib.util.spec_from_file_location("bipo_warmup11_construction", path)
    if spec is None or spec.loader is None:
        raise PostConfirmatoryGateError("could not load the locked construction runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_frozen_commit_paths(
    repo_root: Path,
    *,
    anchor_path: str,
    expected_subject: str,
    required_paths: Sequence[str],
) -> str:
    """Find the anchor's final commit and require exact current bytes for every path."""

    raw = _run_git(
        repo_root,
        ["log", "-1", "--format=%H%x00%s", "--", anchor_path],
    )
    if "\x00" not in raw:
        raise PostConfirmatoryGateError("main final result commit cannot be resolved")
    commit, subject = raw.split("\x00", 1)
    if len(commit) != 40 or subject != expected_subject:
        raise PostConfirmatoryGateError(
            "main final report is not anchored by the exact frozen final-result commit"
        )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise PostConfirmatoryGateError("main final result commit is not an ancestor of HEAD")
    for relative in sorted(set(required_paths)):
        path = _resolve_repo_file(repo_root, relative, label="frozen main-result file")
        _run_git(repo_root, ["cat-file", "-e", f"{commit}:{relative}"])
        tree_blob = _run_git(repo_root, ["rev-parse", f"{commit}:{relative}"])
        working_blob = _run_git(repo_root, ["hash-object", "--", str(path)])
        if tree_blob != working_blob:
            raise PostConfirmatoryGateError(
                f"main-result file differs from final commit {commit}: {relative}"
            )
    return commit


def _load_protected(repo_root: Path) -> dict[str, Any]:
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from sp_lense.comparison_provenance import (
        approved_setup_records,
        verify_stage2_manifest,
    )

    return {
        "approved_setup_records": approved_setup_records,
        "verify_stage2_manifest": verify_stage2_manifest,
    }


def _load_orchestrator(repo_root: Path) -> Any:
    path = repo_root / "artifacts" / "steering_comparison" / "locked_open_orchestration.py"
    spec = importlib.util.spec_from_file_location("locked_open_orchestration", path)
    if spec is None or spec.loader is None:
        raise PostConfirmatoryGateError("could not load locked open orchestration")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_tag(model_id: str) -> str:
    if model_id == "Qwen/Qwen3.5-0.8B":
        return "qwen35_08b"
    if model_id == "Qwen/Qwen3.5-2B":
        return "qwen35_2b"
    raise PostConfirmatoryGateError(f"unlocked model in parent BiPO plan: {model_id}")


def _validate_inventory(
    repo_root: Path, inventory_path: Path, *, final_commit: str
) -> dict[str, dict[str, Any]]:
    inventory = read_json(inventory_path, label="main final artifact inventory")
    if inventory.get("phase") != "final" or not isinstance(inventory.get("entries"), list):
        raise PostConfirmatoryGateError("main final artifact inventory has the wrong phase/schema")
    if inventory.get("base_commit") != _run_git(repo_root, ["rev-parse", f"{final_commit}^"]):
        raise PostConfirmatoryGateError("final inventory base is not the final commit's parent")
    entries: dict[str, dict[str, Any]] = {}
    for entry in inventory["entries"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size_bytes"}:
            raise PostConfirmatoryGateError("main final inventory entry schema is invalid")
        relative = str(entry["path"])
        if relative in entries or not _is_sha256(entry.get("sha256")):
            raise PostConfirmatoryGateError("main final inventory has duplicate/invalid entries")
        path = _resolve_repo_file(repo_root, relative, label="inventoried final artifact")
        if sha256_file(path) != entry["sha256"] or path.stat().st_size != entry["size_bytes"]:
            raise PostConfirmatoryGateError(f"main final inventory mismatch: {relative}")
        entries[relative] = entry
    if inventory.get("path_count") != len(entries) or not entries:
        raise PostConfirmatoryGateError("main final inventory path count is invalid")
    return entries


def require_main_final_freeze(repo_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify Stage 2, the canonical plan, final commit, and parent result hashes."""

    parent = config["parent_study"]
    policy = config["evaluation_policy"]
    required = policy["required_main_final"]
    stage1_path = _resolve_repo_file(
        repo_root, parent["stage1_lock_path"], label="parent stage-1 lock"
    )
    stage1 = read_json(stage1_path, label="parent stage-1 lock")
    stage2_path = _resolve_repo_file(
        repo_root, required["stage2_manifest_path"], label="parent stage-2 manifest"
    )
    sealed_plan_path = _resolve_repo_file(
        repo_root, required["sealed_plan_path"], label="canonical sealed plan"
    )
    protected = _load_protected(repo_root)
    verified_stage2 = protected["verify_stage2_manifest"](repo_root, stage1, stage2_path)
    orchestrator = _load_orchestrator(repo_root)
    sealed_plan = orchestrator.verify_canonical_plan(
        repo_root,
        stage1_path,
        stage2_path,
        repo_root / "artifacts" / "steering_comparison" / "sealed",
        "sealed_test",
        sealed_plan_path,
    )
    approved = list(protected["approved_setup_records"](verified_stage2))
    approved_ids = {canonical_sha256(item) for item in approved}
    if {str(item["setup_id"]) for item in sealed_plan["setups"]} != approved_ids:
        raise PostConfirmatoryGateError("canonical sealed plan differs from Stage-2 approvals")
    bipo_setups = [
        dict(item) for item in sealed_plan["setups"] if item.get("method_id") == PARENT_METHOD_ID
    ]
    if not bipo_setups or any(item.get("is_random_control") for item in bipo_setups):
        raise PostConfirmatoryGateError("canonical sealed plan lacks non-random BiPO setups")
    coverage = {(_model_tag(str(item["model_id"])), str(item["track"])) for item in bipo_setups}
    if coverage != EXPECTED_MODEL_TRACKS:
        raise PostConfirmatoryGateError(
            f"parent BiPO plan lacks exact model/track coverage: {sorted(coverage)}"
        )
    required_paths = [
        required["final_inventory_path"],
        required["final_report_json_path"],
        required["final_report_markdown_path"],
        required["adversarial_review_path"],
        required["stage2_manifest_path"],
        required["sealed_plan_path"],
    ]
    for setup in bipo_setups:
        if setup.get("open_required") is not True or setup.get("tbsp_required") is not True:
            raise PostConfirmatoryGateError("every parent BiPO setup must require open and TBSP")
        required_paths.extend([str(setup["forced_path"]), str(setup["scored_path"])])
    final_commit = require_frozen_commit_paths(
        repo_root,
        anchor_path=required["final_report_json_path"],
        expected_subject=required["commit_subject"],
        required_paths=required_paths,
    )
    remote_main = _remote_main_commit(repo_root)
    remote_tracking_main = _run_git(repo_root, ["rev-parse", "refs/remotes/origin/main"])
    local_head = _run_git(repo_root, ["rev-parse", "HEAD"])
    if (
        remote_main != final_commit
        or remote_tracking_main != remote_main
        or local_head != remote_main
    ):
        raise PostConfirmatoryGateError(
            "post-confirmatory evaluation requires the final-result commit to be the "
            "pushed origin/main tip and local HEAD to equal origin/main"
        )
    changed = set(
        _run_git(
            repo_root, ["diff-tree", "--no-commit-id", "--name-only", "-r", final_commit]
        ).splitlines()
    )
    for path in (required["final_report_json_path"], required["final_inventory_path"]):
        if path not in changed:
            raise PostConfirmatoryGateError(f"final commit did not introduce {path}")
    inventory_path = repo_root / required["final_inventory_path"]
    inventory = _validate_inventory(repo_root, inventory_path, final_commit=final_commit)
    for setup in bipo_setups:
        for field in ("forced_path", "scored_path"):
            if setup[field] not in inventory:
                raise PostConfirmatoryGateError(
                    f"parent BiPO result is absent from final inventory: {setup[field]}"
                )
    if required["final_report_json_path"] not in inventory:
        raise PostConfirmatoryGateError("final report is absent from final inventory")
    artifacts = {
        relative: {
            "sha256": sha256_file(repo_root / relative),
            "size_bytes": (repo_root / relative).stat().st_size,
        }
        for relative in sorted(set(required_paths))
    }
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "main_confirmatory_results_git_frozen_before_sensitivity_evaluation",
        "main_final_commit": final_commit,
        "pushed_origin_main_commit": remote_main,
        "remote_tracking_origin_main_commit": remote_tracking_main,
        "local_head_commit": local_head,
        "main_final_commit_subject": required["commit_subject"],
        "stage1_lock_sha256": sha256_file(stage1_path),
        "stage2_manifest_sha256": sha256_file(stage2_path),
        "sealed_plan_sha256": sha256_file(sealed_plan_path),
        "final_inventory_sha256": sha256_file(inventory_path),
        "parent_bipo_setup_count": len(bipo_setups),
        "parent_bipo_setup_ids": [str(item["setup_id"]) for item in bipo_setups],
        "frozen_artifacts": artifacts,
        "setups": bipo_setups,
    }


def _float32_bytes(values: Sequence[Any]) -> bytes:
    output = bytearray()
    for value in values:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise PostConfirmatoryGateError("direction contains a non-finite value")
        output.extend(struct.pack("<f", numeric))
    return bytes(output)


def validate_direction_record(record: Mapping[str, Any], *, expected_method: str) -> dict[str, Any]:
    direction = record.get("direction")
    if not isinstance(direction, list) or not direction:
        raise PostConfirmatoryGateError("direction record lacks a nonempty vector")
    metadata_record = {
        key: record[key]
        for key in (
            "schema_version",
            "method",
            "layer",
            "intervention_geometry",
            "d_model",
            "dtype",
            "direction_l2_norm",
            "direction_sha256",
            "metadata",
        )
    }
    vector_bytes = _float32_bytes(direction)
    direction_sha = hashlib.sha256(vector_bytes).hexdigest()
    metadata_sha = canonical_sha256(metadata_record)
    artifact_sha = hashlib.sha256(
        canonical_json_bytes(metadata_record) + b"\0" + vector_bytes
    ).hexdigest()
    if (
        record.get("method") != expected_method
        or record.get("dtype") != "float32"
        or record.get("d_model") != len(direction)
        or record.get("direction_sha256") != direction_sha
        or record.get("metadata_sha256") != metadata_sha
        or record.get("artifact_sha256") != artifact_sha
    ):
        raise PostConfirmatoryGateError("direction record identity/hash validation failed")
    return {
        "direction_sha256": direction_sha,
        "artifact_sha256": artifact_sha,
        "d_model": len(direction),
        "layer": record["layer"],
        "intervention_geometry": record["intervention_geometry"],
    }


def _constructed_artifacts(
    repo_root: Path, config: Mapping[str, Any]
) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for item in config["planned_artifacts"]:
        output_dir = repo_root / item["output_directory"]
        manifest_path = output_dir / item["manifest_filename"]
        direction_path = output_dir / item["direction_filename"]
        manifest = read_json(manifest_path, label="warmup-11 construction manifest")
        record = read_json(direction_path, label="warmup-11 direction")
        identity = validate_direction_record(record, expected_method=SENSITIVITY_METHOD_ID)
        manifest_direction = manifest.get("direction", {})
        if (
            manifest.get("status") != "complete_secondary_sensitivity_construction"
            or manifest.get("artifact_id") != item["artifact_id"]
            or manifest.get("artifact_identity_sha256") != item["artifact_identity_sha256"]
            or manifest.get("confirmatory_winner_ranking_eligible") is not False
            or manifest_direction.get("file_sha256") != sha256_file(direction_path)
            or manifest_direction.get("direction_sha256") != identity["direction_sha256"]
            or manifest_direction.get("artifact_sha256") != identity["artifact_sha256"]
        ):
            raise PostConfirmatoryGateError(
                f"warmup-11 construction manifest is invalid: {item['artifact_id']}"
            )
        model = config["models"][item["model_tag"]]
        expected_geometry = config["tracks"][item["track"]]["training_geometry"]
        if (
            record.get("layer") != model["layer_zero_based"]
            or record.get("d_model") != model["residual_width"]
            or record.get("intervention_geometry") != expected_geometry
            or record.get("metadata", {}).get("model_id") != model["model_id"]
            or record.get("metadata", {}).get("model_revision") != model["revision"]
        ):
            raise PostConfirmatoryGateError("warmup-11 direction differs from its locked plan")
        output[(item["model_tag"], item["track"])] = {
            "artifact_id": item["artifact_id"],
            "artifact_identity_sha256": item["artifact_identity_sha256"],
            "manifest_path": manifest_path.relative_to(repo_root).as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
            "direction_path": direction_path.relative_to(repo_root).as_posix(),
            "direction_file_sha256": sha256_file(direction_path),
            **identity,
        }
    if set(output) != EXPECTED_MODEL_TRACKS:
        raise PostConfirmatoryGateError("constructed warmup-11 artifacts lack exact coverage")
    return output


def _expected_position(track: str) -> str:
    return "final_prompt_token" if track == "matched" else "all_token_positions"


def _secondary_setups(
    repo_root: Path,
    config: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> list[dict[str, Any]]:
    artifacts = _constructed_artifacts(repo_root, config)
    setups: list[dict[str, Any]] = []
    seen: set[str] = set()
    output_root = config["evaluation_policy"]["output_root"]
    for index, parent in enumerate(receipt["setups"]):
        model_tag = _model_tag(str(parent["model_id"]))
        track = str(parent["track"])
        artifact = artifacts[(model_tag, track)]
        if (
            parent.get("method_id") != PARENT_METHOD_ID
            or parent.get("selected_layer") != 10
            or parent.get("position_schedule") != _expected_position(track)
            or parent.get("open_required") is not True
            or parent.get("tbsp_required") is not True
        ):
            raise PostConfirmatoryGateError(
                "parent BiPO setup is incompatible with locked mirroring"
            )
        identity = {
            "schema_version": "sp_lense.bipo_warmup_sensitivity.evaluation_setup_identity.v1",
            "study_id": config["study_id"],
            "analysis_tier": ANALYSIS_TIER,
            "method_id": SENSITIVITY_METHOD_ID,
            "parent_method_id": PARENT_METHOD_ID,
            "parent_setup_id": parent["setup_id"],
            "model_id": parent["model_id"],
            "model_revision": parent["model_revision"],
            "model_config_sha256": parent["model_config_sha256"],
            "track": track,
            "selected_layer": parent["selected_layer"],
            "selected_strength": parent["selected_strength"],
            "position_schedule": parent["position_schedule"],
            "direction_float32_sha256": artifact["direction_sha256"],
            "direction_artifact_sha256": artifact["artifact_sha256"],
            "construction_config_sha256": artifact["artifact_identity_sha256"],
            "parent_calibration_summary_sha256": parent["calibration_summary_sha256"],
            "warmup_steps": 11,
        }
        setup_id = canonical_sha256(identity)
        if setup_id in seen:
            raise PostConfirmatoryGateError("duplicate secondary setup identity")
        seen.add(setup_id)
        stem = f"setup_{index:03d}_{setup_id[:16]}"
        setups.append(
            {
                "index": index,
                "setup_id": setup_id,
                "identity": identity,
                "model_tag": model_tag,
                "model_config": parent["model_config"],
                "artifact_id": artifact["artifact_id"],
                "artifact_identity_sha256": artifact["artifact_identity_sha256"],
                "direction_path": artifact["direction_path"],
                "direction_file_sha256": artifact["direction_file_sha256"],
                "direction_float32_sha256": artifact["direction_sha256"],
                "direction_artifact_sha256": artifact["artifact_sha256"],
                "construction_manifest_path": artifact["manifest_path"],
                "construction_manifest_sha256": artifact["manifest_sha256"],
                "parent_setup": parent,
                "parent_forced_sha256": receipt["frozen_artifacts"][parent["forced_path"]][
                    "sha256"
                ],
                "parent_scored_sha256": receipt["frozen_artifacts"][parent["scored_path"]][
                    "sha256"
                ],
                "forced_path": f"{output_root}/forced/{stem}.jsonl",
                "generation_path": f"{output_root}/generations/{stem}.jsonl",
                "scored_path": f"{output_root}/scored/{stem}.jsonl",
            }
        )
    return setups


def validate_plan_firewall(plan: Mapping[str, Any], repo_root: Path) -> None:
    if (
        plan.get("analysis_tier") != ANALYSIS_TIER
        or plan.get("confirmatory_winner_ranking_eligible") is not False
        or plan.get("automatic_confirmatory_ingestion_allowed") is not False
        or plan.get("method_id") != SENSITIVITY_METHOD_ID
    ):
        raise PostConfirmatoryGateError("secondary ranking firewall fields are invalid")
    main_root = (repo_root / "artifacts" / "steering_comparison").resolve()
    output_root = (repo_root / plan["output_root"]).resolve()
    try:
        output_root.relative_to(main_root)
    except ValueError:
        pass
    else:
        raise PostConfirmatoryGateError("secondary output root aliases main artifacts")
    for setup in plan.get("setups", []):
        if setup.get("identity", {}).get("method_id") != SENSITIVITY_METHOD_ID:
            raise PostConfirmatoryGateError("secondary setup uses a confirmatory method ID")
        for field in ("forced_path", "generation_path", "scored_path"):
            path = (repo_root / setup[field]).resolve()
            try:
                path.relative_to(output_root)
            except ValueError as exc:
                raise PostConfirmatoryGateError(
                    "secondary result path escaped output root"
                ) from exc


def prepare_evaluation() -> dict[str, Any]:
    repo_root, experiment_root = default_roots()
    construction = _load_construction_runner(experiment_root)
    verification = construction.verify_experiment(
        repo_root=repo_root, experiment_root=experiment_root
    )
    runner_commit = construction._require_committed_experiment(repo_root, experiment_root)
    config = read_json(experiment_root / "config.json", label="sensitivity config")
    output_root = repo_root / config["evaluation_policy"]["output_root"]
    if output_root.exists():
        raise PostConfirmatoryGateError(
            "evaluation output root already exists; preparation/overwrite is forbidden"
        )
    receipt = require_main_final_freeze(repo_root, config)
    setups = _secondary_setups(repo_root, config, receipt)
    receipt["sensitivity_lock_sha256"] = verification["sensitivity_lock_sha256"]
    receipt["sensitivity_config_sha256"] = verification["sensitivity_config_sha256"]
    receipt["secondary_runner_commit"] = runner_commit
    receipt_sha = canonical_sha256(receipt)
    plan = {
        "schema_version": PLAN_SCHEMA,
        "status": "locked_after_main_final_before_secondary_model_evaluation",
        "study_id": config["study_id"],
        "analysis_tier": ANALYSIS_TIER,
        "method_id": SENSITIVITY_METHOD_ID,
        "parent_method_id": PARENT_METHOD_ID,
        "confirmatory_winner_ranking_eligible": False,
        "automatic_confirmatory_ingestion_allowed": False,
        "main_final_commit": receipt["main_final_commit"],
        "main_final_freeze_receipt_sha256": receipt_sha,
        "sensitivity_lock_sha256": verification["sensitivity_lock_sha256"],
        "sensitivity_config_sha256": verification["sensitivity_config_sha256"],
        "secondary_runner_commit": runner_commit,
        "output_root": config["evaluation_policy"]["output_root"],
        "split": "sealed_test",
        "forced_rows_per_setup": FORCED_ROWS_PER_SETUP,
        "open_rows_per_setup": OPEN_ROWS_PER_SETUP,
        "setup_count": len(setups),
        "setups": setups,
        "setups_sha256": canonical_sha256(setups),
        "strength_rule": "inherit_each_parent_bipo_setup_without_recalibration",
        "safety_limits": config["evaluation_policy"]["safety_limits"],
        "statistics": config["evaluation_policy"]["statistics"],
        "ranking_firewall": {
            "may_update_main_report": False,
            "may_enter_main_winner_ranking": False,
            "main_artifact_root_writable": False,
            "report_is_secondary_only": True,
        },
    }
    validate_plan_firewall(plan, repo_root)
    work = output_root.with_name(f".{output_root.name}.work-{os.getpid()}")
    if work.exists():
        raise PostConfirmatoryGateError(f"stale preparation directory exists: {work}")
    work.mkdir(parents=True)
    try:
        (work / "plan").mkdir()
        (work / "plan" / "main_final_freeze_receipt.json").write_bytes(_json_bytes(receipt))
        (work / "plan" / "secondary_evaluation_plan.json").write_bytes(_json_bytes(plan))
        work.replace(output_root)
    except BaseException:
        if work.exists():
            shutil.rmtree(work)
        raise
    return {
        "status": "post_confirmatory_secondary_evaluation_prepared",
        "main_final_commit": receipt["main_final_commit"],
        "setup_count": len(setups),
        "plan_sha256": sha256_file(output_root / "plan" / "secondary_evaluation_plan.json"),
        "confirmatory_winner_ranking_eligible": False,
    }


def _verify_prepared_plan() -> tuple[Path, Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    repo_root, experiment_root = default_roots()
    construction = _load_construction_runner(experiment_root)
    verification = construction.verify_experiment(
        repo_root=repo_root, experiment_root=experiment_root
    )
    runner_commit = construction._require_committed_experiment(repo_root, experiment_root)
    config = read_json(experiment_root / "config.json", label="sensitivity config")
    output_root = repo_root / config["evaluation_policy"]["output_root"]
    receipt_path = output_root / "plan" / "main_final_freeze_receipt.json"
    plan_path = output_root / "plan" / "secondary_evaluation_plan.json"
    receipt = read_json(receipt_path, label="main final freeze receipt")
    plan = read_json(plan_path, label="secondary evaluation plan")
    current_receipt = require_main_final_freeze(repo_root, config)
    for key in (
        "sensitivity_lock_sha256",
        "sensitivity_config_sha256",
        "secondary_runner_commit",
    ):
        current_receipt[key] = receipt.get(key)
    if receipt != current_receipt or canonical_sha256(receipt) != plan.get(
        "main_final_freeze_receipt_sha256"
    ):
        raise PostConfirmatoryGateError("main final freeze receipt is stale or altered")
    expected_setups = _secondary_setups(repo_root, config, receipt)
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status") != "locked_after_main_final_before_secondary_model_evaluation"
        or plan.get("setups") != expected_setups
        or plan.get("setups_sha256") != canonical_sha256(expected_setups)
        or plan.get("setup_count") != len(expected_setups)
        or plan.get("sensitivity_lock_sha256") != verification["sensitivity_lock_sha256"]
        or plan.get("sensitivity_config_sha256") != verification["sensitivity_config_sha256"]
        or plan.get("secondary_runner_commit") != runner_commit
    ):
        raise PostConfirmatoryGateError("secondary evaluation plan is stale or altered")
    validate_plan_firewall(plan, repo_root)
    return repo_root, experiment_root, config, receipt, plan


def _find_setup(plan: Mapping[str, Any], setup_id: str) -> dict[str, Any]:
    matches = [item for item in plan["setups"] if item["setup_id"] == setup_id]
    if len(matches) != 1:
        raise PostConfirmatoryGateError(f"unknown/ambiguous secondary setup ID: {setup_id}")
    return dict(matches[0])


def _forced_unit_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("split"),
        row.get("family"),
        row.get("case_id"),
        row.get("target"),
        row.get("role"),
        row.get("suite"),
        row.get("category"),
        row.get("form"),
        row.get("scenario_cluster_id"),
        row.get("request_type"),
        row.get("expected_behavior"),
        row.get("first_semantic_label"),
        row.get("second_semantic_label"),
        row.get("preserve_label"),
        row.get("comply_label"),
        row.get("correct_label"),
        row.get("preferred_label"),
        row.get("prompt_sha256"),
        row.get("condition"),
    )


def assert_exact_forced_coverage(
    parent_rows: Sequence[Mapping[str, Any]],
    sensitivity_rows: Sequence[Mapping[str, Any]],
    *,
    expected_rows: int = FORCED_ROWS_PER_SETUP,
) -> None:
    if len(parent_rows) != expected_rows or len(sensitivity_rows) != expected_rows:
        raise PostConfirmatoryGateError(
            f"forced row count differs from locked {expected_rows}-row coverage"
        )
    parent_keys = [_forced_unit_key(row) for row in parent_rows]
    sensitivity_keys = [_forced_unit_key(row) for row in sensitivity_rows]
    if len(set(parent_keys)) != expected_rows or len(set(sensitivity_keys)) != expected_rows:
        raise PostConfirmatoryGateError("forced coverage contains duplicate semantic units")
    if set(parent_keys) != set(sensitivity_keys):
        raise PostConfirmatoryGateError(
            "warmup-11 forced coverage differs from exact parent semantic units/prompts/conditions"
        )


def _open_unit_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("split"),
        row.get("family"),
        row.get("case_id"),
        row.get("source_core_id"),
        row.get("target"),
        row.get("condition"),
        row.get("prompt_sha256"),
        row.get("rubric_sha256"),
        canonical_sha256(row.get("generation_config")),
    )


def assert_exact_open_coverage(
    parent_rows: Sequence[Mapping[str, Any]],
    sensitivity_rows: Sequence[Mapping[str, Any]],
    *,
    expected_rows: int = OPEN_ROWS_PER_SETUP,
) -> None:
    if len(parent_rows) != expected_rows or len(sensitivity_rows) != expected_rows:
        raise PostConfirmatoryGateError(
            f"open row count differs from locked {expected_rows}-row coverage"
        )
    parent_keys = [_open_unit_key(row) for row in parent_rows]
    sensitivity_keys = [_open_unit_key(row) for row in sensitivity_rows]
    if len(set(parent_keys)) != expected_rows or len(set(sensitivity_keys)) != expected_rows:
        raise PostConfirmatoryGateError("open coverage contains duplicate semantic units")
    if set(parent_keys) != set(sensitivity_keys):
        raise PostConfirmatoryGateError(
            "warmup-11 open coverage differs from exact parent cases/prompts/conditions"
        )
    parent_baselines = {
        _open_unit_key(row): row.get("baseline_content_sha256")
        for row in parent_rows
        if row.get("condition") == "baseline"
    }
    sensitivity_baselines = {
        _open_unit_key(row): row.get("baseline_content_sha256")
        for row in sensitivity_rows
        if row.get("condition") == "baseline"
    }
    if parent_baselines != sensitivity_baselines:
        raise PostConfirmatoryGateError(
            "warmup-11 baseline generations differ from frozen parent baseline content"
        )


def _assert_parent_forced_identity(
    rows: Sequence[Mapping[str, Any]], setup: Mapping[str, Any]
) -> None:
    parent = setup["parent_setup"]
    expected = {
        "model_id": parent["model_id"],
        "model_revision": parent["model_revision"],
        "config_sha256": parent["model_config_sha256"],
        "method": PARENT_METHOD_ID,
        "method_id": PARENT_METHOD_ID,
        "setup": parent["track"],
        "track": parent["track"],
        "layer": parent["selected_layer"],
        "position": parent["position_schedule"],
        "calibration_magnitude": parent["selected_strength"],
        "direction_sha256": parent["direction_float32_sha256"],
        "direction_float32_sha256": parent["direction_float32_sha256"],
        "direction_artifact_sha256": parent["direction_artifact_sha256"],
        "construction_config_sha256": parent["construction_config_sha256"],
        "calibration_summary_sha256": parent["calibration_summary_sha256"],
    }
    for row in rows:
        mismatches = {key for key, value in expected.items() if row.get(key) != value}
        if mismatches:
            raise PostConfirmatoryGateError(
                f"parent forced rows differ from sealed setup identity: {sorted(mismatches)}"
            )


def _assert_sensitivity_identity(
    rows: Sequence[Mapping[str, Any]], setup: Mapping[str, Any]
) -> None:
    identity = setup["identity"]
    expected = {
        "model_id": identity["model_id"],
        "model_revision": identity["model_revision"],
        "config_sha256": identity["model_config_sha256"],
        "method": SENSITIVITY_METHOD_ID,
        "method_id": SENSITIVITY_METHOD_ID,
        "setup": identity["track"],
        "track": identity["track"],
        "layer": identity["selected_layer"],
        "position": identity["position_schedule"],
        "calibration_magnitude": identity["selected_strength"],
        "direction_sha256": identity["direction_float32_sha256"],
        "direction_float32_sha256": identity["direction_float32_sha256"],
        "direction_artifact_sha256": identity["direction_artifact_sha256"],
        "construction_config_sha256": identity["construction_config_sha256"],
        "calibration_summary_sha256": identity["parent_calibration_summary_sha256"],
    }
    for row in rows:
        mismatches = {key for key, value in expected.items() if row.get(key) != value}
        if (
            mismatches
            or row.get("analysis_tier") != ANALYSIS_TIER
            or row.get("confirmatory_winner_ranking_eligible") is not False
            or row.get("automatic_confirmatory_ingestion_allowed") is not False
            or row.get("parent_setup_id") != identity["parent_setup_id"]
            or row.get("sensitivity_setup_id") != setup["setup_id"]
        ):
            raise PostConfirmatoryGateError(
                f"secondary rows differ from locked sensitivity identity: {sorted(mismatches)}"
            )


def _validate_parent_result_hashes(repo_root: Path, setup: Mapping[str, Any]) -> None:
    parent = setup["parent_setup"]
    if sha256_file(repo_root / parent["forced_path"]) != setup["parent_forced_sha256"]:
        raise PostConfirmatoryGateError("frozen parent forced-result hash changed")
    if sha256_file(repo_root / parent["scored_path"]) != setup["parent_scored_sha256"]:
        raise PostConfirmatoryGateError("frozen parent open-result hash changed")


def _runtime_context(
    repo_root: Path,
    config: Mapping[str, Any],
    receipt: Mapping[str, Any],
    plan: Mapping[str, Any],
    setup: Mapping[str, Any],
) -> dict[str, Any]:
    """Load the model only after every post-confirmatory and hash gate has passed."""

    _validate_parent_result_hashes(repo_root, setup)
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from sp_lense.backend import ResearchBackend
    from sp_lense.comparison_dataset import load_comparison_dataset
    from sp_lense.comparison_evaluate import (
        EvaluationIdentity,
        MethodSetup,
        SealedEvaluationGate,
        sealed_ids_from_dataset_and_lock,
    )
    from sp_lense.comparison_fit import read_direction_artifact
    from sp_lense.comparison_provenance import verify_stage2_manifest
    from sp_lense.comparison_runtime import validate_locked_choice_runtime
    from sp_lense.config import load_config
    from sp_lense.steering_methods import DirectionArtifact

    stage1_path = repo_root / config["parent_study"]["stage1_lock_path"]
    stage1 = read_json(stage1_path, label="parent stage-1 lock")
    stage2_path = (
        repo_root / config["evaluation_policy"]["required_main_final"]["stage2_manifest_path"]
    )
    verified_stage2 = verify_stage2_manifest(repo_root, stage1, stage2_path)
    if verified_stage2.manifest_sha256 != receipt["stage2_manifest_sha256"]:
        raise PostConfirmatoryGateError("Stage-2 capability differs from final freeze receipt")
    dataset = load_comparison_dataset(
        repo_root / stage1["dataset"]["path"],
        expected_sha256=stage1["dataset"]["sha256"],
    )
    model_record = next(
        item for item in stage1["models"] if item["model_id"] == setup["identity"]["model_id"]
    )
    config_path = repo_root / setup["model_config"]
    if sha256_file(config_path) != setup["identity"]["model_config_sha256"]:
        raise PostConfirmatoryGateError("model configuration changed after plan preparation")
    direction_path = repo_root / setup["direction_path"]
    if sha256_file(direction_path) != setup["direction_file_sha256"]:
        raise PostConfirmatoryGateError("warmup-11 direction file changed after preparation")
    backend = ResearchBackend.load(load_config(config_path), with_lens=False)
    validate_locked_choice_runtime(backend, model_record["runtime"])
    sensitivity_artifact = read_direction_artifact(direction_path, backend.torch)
    if (
        sensitivity_artifact.method != SENSITIVITY_METHOD_ID
        or sensitivity_artifact.direction_sha256 != setup["direction_float32_sha256"]
        or sensitivity_artifact.artifact_sha256 != setup["direction_artifact_sha256"]
    ):
        raise PostConfirmatoryGateError("loaded warmup-11 artifact differs from plan")
    # MethodSetup intentionally sees the parent method only to select BiPO's established
    # canonical geometry. Every emitted row is rewritten to the distinct sensitivity ID.
    intervention_artifact = DirectionArtifact(
        method=PARENT_METHOD_ID,
        direction=sensitivity_artifact.direction,
        layer=sensitivity_artifact.layer,
        intervention_geometry=sensitivity_artifact.intervention_geometry,
        metadata=dict(sensitivity_artifact.metadata),
    )
    method_setup = MethodSetup(
        intervention_artifact,
        PARENT_METHOD_ID,
        setup["identity"]["track"],
        float(setup["identity"]["selected_strength"]),
    )
    method_setup.validate()
    if method_setup.position != setup["identity"]["position_schedule"]:
        raise PostConfirmatoryGateError("runtime intervention geometry differs from parent")
    identity = EvaluationIdentity(
        model_id=model_record["model_id"],
        model_revision=model_record["revision"],
        dataset_sha256=stage1["dataset"]["sha256"],
        protocol_sha256=stage1["protocol"]["sha256"],
        config_sha256=model_record["config_sha256"],
        run_seed=int(stage1["statistics"]["bootstrap"]["seed"]),
        stage1_lock_sha256=sha256_file(stage1_path),
        stage2_manifest_sha256=verified_stage2.manifest_sha256,
        calibration_summary_sha256=setup["identity"]["parent_calibration_summary_sha256"],
        construction_config_sha256=setup["identity"]["construction_config_sha256"],
        runner_commit=plan["secondary_runner_commit"],
    )
    gate = SealedEvaluationGate(
        sealed_ids_from_dataset_and_lock(dataset, stage1), verified_stage2=verified_stage2
    )
    return {
        "stage1": stage1,
        "dataset": dataset,
        "backend": backend,
        "method_setup": method_setup,
        "identity": identity,
        "gate": gate,
    }


def _rewrite_rows(
    rows: Sequence[Mapping[str, Any]], setup: Mapping[str, Any]
) -> list[dict[str, Any]]:
    identity = setup["identity"]
    output = []
    for source in rows:
        row = dict(source)
        row.update(
            {
                "method": SENSITIVITY_METHOD_ID,
                "method_id": SENSITIVITY_METHOD_ID,
                "direction_sha256": identity["direction_float32_sha256"],
                "direction_float32_sha256": identity["direction_float32_sha256"],
                "direction_artifact_sha256": identity["direction_artifact_sha256"],
                "direction_id": identity["direction_artifact_sha256"],
                "construction_config_sha256": identity["construction_config_sha256"],
                "calibration_summary_sha256": identity["parent_calibration_summary_sha256"],
                "strength_id": (
                    f"warmup11:{identity['track']}:{identity['selected_strength']:.12g}:"
                    f"{identity['parent_setup_id'][:12]}"
                ),
                "analysis_tier": ANALYSIS_TIER,
                "confirmatory_winner_ranking_eligible": False,
                "automatic_confirmatory_ingestion_allowed": False,
                "parent_method_id": PARENT_METHOD_ID,
                "parent_setup_id": identity["parent_setup_id"],
                "sensitivity_setup_id": setup["setup_id"],
                "warmup_steps": 11,
            }
        )
        output.append(row)
    return output


def evaluate_forced(setup_id: str) -> dict[str, Any]:
    repo_root, _, config, receipt, plan = _verify_prepared_plan()
    setup = _find_setup(plan, setup_id)
    output_path = repo_root / setup["forced_path"]
    if output_path.exists():
        raise PostConfirmatoryGateError(f"overwrite is forbidden: {output_path}")
    context = _runtime_context(repo_root, config, receipt, plan, setup)
    from sp_lense.comparison_analysis import validate_result_rows
    from sp_lense.comparison_evaluate import (
        evaluate_collateral_cases,
        evaluate_option_order_sentinels,
        evaluate_sp_cases,
        evaluate_tbsp_cases,
        select_cases_by_locked_ids,
    )

    stage1 = context["stage1"]
    dataset = context["dataset"]
    common = {
        "setup": context["method_setup"],
        "identity": context["identity"],
        "gate": context["gate"],
    }
    rows = evaluate_sp_cases(
        context["backend"], dataset["sp_splits"]["sealed_test"], split="sealed_test", **common
    )
    partitions = stage1["dataset"]["partitions"]
    collateral = dataset["collateral_cases"]
    for family in ("benign_compliance", "general_capability", "refusal"):
        cases = select_cases_by_locked_ids(collateral[family], partitions[family]["sealed_ids"])
        rows.extend(
            evaluate_collateral_cases(
                context["backend"], cases, split="sealed_test", family=family, **common
            )
        )
    option_cases = select_cases_by_locked_ids(
        collateral["option_order_sentinels"],
        partitions["option_order_sentinels"]["sealed_ids"],
    )
    rows.extend(
        evaluate_option_order_sentinels(
            context["backend"], option_cases, split="sealed_test", **common
        )
    )
    rows.extend(evaluate_tbsp_cases(context["backend"], dataset["tbsp_cases"], **common))
    rewritten = _rewrite_rows(rows, setup)
    validate_result_rows(rewritten)
    _assert_sensitivity_identity(rewritten, setup)
    parent_rows = read_jsonl(
        repo_root / setup["parent_setup"]["forced_path"], label="frozen parent forced rows"
    )
    validate_result_rows(parent_rows)
    _assert_parent_forced_identity(parent_rows, setup)
    assert_exact_forced_coverage(parent_rows, rewritten)
    _write_new(output_path, _jsonl_bytes(rewritten))
    return {
        "status": "secondary_forced_evaluation_complete",
        "setup_id": setup_id,
        "row_count": len(rewritten),
        "output_sha256": sha256_file(output_path),
        "confirmatory_winner_ranking_eligible": False,
    }


def _validate_open_generations(
    rows: Sequence[Mapping[str, Any]],
    setup: Mapping[str, Any],
    *,
    parent: bool = False,
) -> None:
    from sp_lense.comparison_behavior import (
        baseline_content_sha256,
        open_generation_sha256,
    )

    if len(rows) != OPEN_ROWS_PER_SETUP:
        raise PostConfirmatoryGateError("secondary open generations lack exact 96-row coverage")
    grouped: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        if row.get("schema_version") != "sp_lense.open_generation.v2":
            raise PostConfirmatoryGateError("secondary open generation schema differs")
        if row.get("generation_sha256") != open_generation_sha256(row):
            raise PostConfirmatoryGateError("secondary open generation hash is invalid")
        if hashlib.sha256(str(row["completion"]).encode("utf-8")).hexdigest() != row.get(
            "completion_sha256"
        ):
            raise PostConfirmatoryGateError("secondary open completion hash is invalid")
        if row.get("condition") == "baseline" and row.get(
            "baseline_content_sha256"
        ) != baseline_content_sha256(row):
            raise PostConfirmatoryGateError("secondary baseline content hash is invalid")
        key = (str(row["case_id"]), str(row["target"]))
        condition = str(row["condition"])
        if condition in grouped.setdefault(key, set()):
            raise PostConfirmatoryGateError("duplicate secondary open condition")
        grouped[key].add(condition)
    if len(grouped) != 32 or any(value != EXPECTED_CONDITIONS for value in grouped.values()):
        raise PostConfirmatoryGateError("secondary open triplet coverage is incomplete")
    if parent:
        _assert_parent_forced_identity(rows, setup)
    else:
        _assert_sensitivity_identity(rows, setup)


def _rewrite_open_rows(
    rows: Sequence[Mapping[str, Any]], setup: Mapping[str, Any]
) -> list[dict[str, Any]]:
    from sp_lense.comparison_behavior import (
        baseline_content_sha256,
        open_generation_sha256,
    )

    rewritten = _rewrite_rows(rows, setup)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rewritten:
        row.pop("generation_sha256", None)
        grouped.setdefault((str(row["case_id"]), str(row["target"])), []).append(row)
    for triplet in grouped.values():
        baseline = [row for row in triplet if row["condition"] == "baseline"]
        if len(baseline) != 1:
            raise PostConfirmatoryGateError("secondary open rewrite lacks one baseline")
        digest = baseline_content_sha256(baseline[0])
        for row in triplet:
            row["baseline_content_sha256"] = digest
            row["generation_sha256"] = open_generation_sha256(row)
    return rewritten


def _validate_scored_open(
    rows: Sequence[Mapping[str, Any]],
    setup: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    parent: bool = False,
) -> None:
    from sp_lense.comparison_behavior import (
        attach_open_judgment,
        open_generation_sha256,
    )

    _validate_open_generations(rows, setup, parent=parent)
    judgment_fields = (
        "decision",
        "coherence_score",
        "authorization_sensitivity_score",
        "task_benefit_score",
        "action_in_first_sentence",
        "judge_model",
        "judge_revision",
        "judge_prompt_sha256",
        "judge_config_sha256",
        "judge_protocol_sha256",
        "judge_request_content_sha256",
        "judge_response_content_sha256",
        "judge_raw_response",
        "judge_raw_response_sha256",
    )
    for row in rows:
        if row.get("generation_sha256") != open_generation_sha256(row):
            raise PostConfirmatoryGateError("scored open generation hash is invalid")
        expected = attach_open_judgment(row, protocol, row.get("judge_raw_response"))
        if any(row.get(field) != getattr(expected, field) for field in judgment_fields):
            raise PostConfirmatoryGateError("scored open judgment provenance is invalid")
        if row.get("judgment_schema_version") != "sp_lense.open_judgment.v2":
            raise PostConfirmatoryGateError("scored open judgment schema differs")
        if row.get("coherence_assessed") is not True or row.get("coherent") != (
            int(row["coherence_score"]) >= 50
        ):
            raise PostConfirmatoryGateError("scored open coherence fields are invalid")


def generate_open(setup_id: str) -> dict[str, Any]:
    repo_root, _, config, receipt, plan = _verify_prepared_plan()
    setup = _find_setup(plan, setup_id)
    output_path = repo_root / setup["generation_path"]
    if output_path.exists():
        raise PostConfirmatoryGateError(f"overwrite is forbidden: {output_path}")
    context = _runtime_context(repo_root, config, receipt, plan, setup)
    from sp_lense.comparison_behavior import generate_open_cases

    stage1 = context["stage1"]
    rows = generate_open_cases(
        context["backend"],
        dataset=context["dataset"],
        locked_case_ids=stage1["dataset"]["partitions"]["open_ended"]["sealed_ids"],
        setup=context["method_setup"],
        identity=context["identity"],
        split="sealed_test",
        gate=context["gate"],
    )
    rewritten = _rewrite_open_rows(rows, setup)
    _validate_open_generations(rewritten, setup)
    parent_rows = read_jsonl(
        repo_root / setup["parent_setup"]["scored_path"], label="frozen parent scored-open rows"
    )
    assert_exact_open_coverage(parent_rows, rewritten)
    _write_new(output_path, _jsonl_bytes(rewritten))
    return {
        "status": "secondary_open_generation_complete",
        "setup_id": setup_id,
        "row_count": len(rewritten),
        "output_sha256": sha256_file(output_path),
        "confirmatory_winner_ranking_eligible": False,
    }


def build_judge_requests() -> dict[str, Any]:
    repo_root, _, config, _, plan = _verify_prepared_plan()
    from sp_lense.comparison_behavior import load_open_judge_protocol
    from sp_lense.comparison_workflow import build_open_judge_requests

    protocol_path = (
        repo_root
        / read_json(repo_root / config["parent_study"]["stage1_lock_path"], label="stage-1 lock")[
            "evaluation"
        ]["open_behavior_judge"]["protocol_path"]
    )
    protocol = load_open_judge_protocol(protocol_path)
    generations: list[dict[str, Any]] = []
    for setup in plan["setups"]:
        rows = read_jsonl(repo_root / setup["generation_path"], label="secondary generations")
        _validate_open_generations(rows, setup)
        generations.extend(rows)
    requests = build_open_judge_requests(generations, protocol)
    output = repo_root / plan["output_root"] / "judge" / "open_judge_requests.jsonl"
    _write_new(output, _jsonl_bytes(requests))
    return {
        "status": "secondary_blinded_judge_requests_ready",
        "generation_count": len(generations),
        "request_count_after_shared_baselines": len(requests),
        "output_sha256": sha256_file(output),
        "external_call_performed": False,
    }


def attach_judgments(responses_path: Path) -> dict[str, Any]:
    repo_root, _, config, _, plan = _verify_prepared_plan()
    from sp_lense.comparison_behavior import load_open_judge_protocol
    from sp_lense.comparison_workflow import (
        attach_open_judge_responses,
        build_open_judge_requests,
    )

    output_root = (repo_root / plan["output_root"]).resolve()
    responses_path = responses_path.resolve()
    try:
        responses_path.relative_to(output_root)
    except ValueError as exc:
        raise PostConfirmatoryGateError(
            "judge responses must remain in secondary output root"
        ) from exc
    if (output_root / "scored").exists():
        raise PostConfirmatoryGateError("scored output already exists; overwrite is forbidden")
    stage1 = read_json(repo_root / config["parent_study"]["stage1_lock_path"], label="stage-1 lock")
    protocol = load_open_judge_protocol(
        repo_root / stage1["evaluation"]["open_behavior_judge"]["protocol_path"]
    )
    all_generations: list[dict[str, Any]] = []
    setup_generations: dict[str, list[dict[str, Any]]] = {}
    for setup in plan["setups"]:
        rows = read_jsonl(repo_root / setup["generation_path"], label="secondary generations")
        _validate_open_generations(rows, setup)
        setup_generations[setup["setup_id"]] = rows
        all_generations.extend(rows)
    request_path = output_root / "judge" / "open_judge_requests.jsonl"
    observed_requests = read_jsonl(request_path, label="secondary judge requests")
    if _jsonl_bytes(observed_requests) != _jsonl_bytes(
        build_open_judge_requests(all_generations, protocol)
    ):
        raise PostConfirmatoryGateError("secondary judge requests differ from exact regeneration")
    responses = read_jsonl(responses_path, label="secondary judge responses")
    scored_all = attach_open_judge_responses(all_generations, responses, protocol)
    by_hash = {row["generation_sha256"]: row for row in scored_all}
    work = output_root / f".scored.work-{os.getpid()}"
    work.mkdir()
    try:
        scored_records = []
        for setup in plan["setups"]:
            rows = [
                by_hash[row["generation_sha256"]] for row in setup_generations[setup["setup_id"]]
            ]
            _validate_scored_open(rows, setup, protocol)
            parent_rows = read_jsonl(
                repo_root / setup["parent_setup"]["scored_path"],
                label="frozen parent scored-open rows",
            )
            assert_exact_open_coverage(parent_rows, rows)
            relative_name = Path(setup["scored_path"]).name
            scored_path = work / relative_name
            scored_path.write_bytes(_jsonl_bytes(rows))
            scored_records.append(
                {
                    "setup_id": setup["setup_id"],
                    "path": setup["scored_path"],
                    "sha256": sha256_file(scored_path),
                    "row_count": len(rows),
                }
            )
        attachment_receipt = {
            "schema_version": "sp_lense.bipo_warmup_sensitivity.judgment_attachment.v1",
            "analysis_tier": ANALYSIS_TIER,
            "confirmatory_winner_ranking_eligible": False,
            "request_file": {
                "path": request_path.relative_to(repo_root).as_posix(),
                "sha256": sha256_file(request_path),
                "request_count": len(observed_requests),
            },
            "response_file": {
                "path": responses_path.relative_to(repo_root).as_posix(),
                "sha256": sha256_file(responses_path),
                "response_count": len(responses),
            },
            "scored_files": scored_records,
            "scored_generation_count": len(scored_all),
        }
        (work / "judgment_attachment_receipt.json").write_bytes(_json_bytes(attachment_receipt))
        scored_root = output_root / "scored"
        if scored_root.exists():
            raise PostConfirmatoryGateError("scored output appeared concurrently")
        work.replace(scored_root)
    except BaseException:
        if work.exists():
            shutil.rmtree(work)
        raise
    return {
        "status": "secondary_open_judgments_attached",
        "scored_generation_count": len(scored_all),
        "setup_count": len(plan["setups"]),
        "confirmatory_winner_ranking_eligible": False,
    }


def _direction_similarity(repo_root: Path, setup: Mapping[str, Any]) -> dict[str, Any]:
    sensitivity = read_json(repo_root / setup["direction_path"], label="warmup-11 direction")
    parent = read_json(
        repo_root / setup["parent_setup"]["direction_path"], label="parent BiPO direction"
    )
    left = [float(value) for value in sensitivity["direction"]]
    right = [float(value) for value in parent["direction"]]
    if len(left) != len(right):
        raise PostConfirmatoryGateError("parent/sensitivity direction widths differ")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    cosine = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    delta_norm = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))
    return {
        "warmup11_l2_norm": left_norm,
        "confirmatory_warmup100_l2_norm": right_norm,
        "warmup11_minus_confirmatory_l2_norm": left_norm - right_norm,
        "cosine_similarity": cosine,
        "vector_difference_l2_norm": delta_norm,
    }


def _robustness_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from sp_lense.comparison_analysis import (
        bidirectional_case_metrics,
        self_minus_other_endpoints,
    )

    endpoints = self_minus_other_endpoints(bidirectional_case_metrics(rows))
    metadata = {
        str(row["case_id"]): row
        for row in rows
        if row.get("family") == "self_preservation"
        and row.get("target") == "self"
        and row.get("condition") == "baseline"
    }
    output: dict[str, Any] = {}
    for factor in ("distribution", "authorized", "event_type", "motivation", "adversarial"):
        groups: dict[str, list[float]] = {}
        for endpoint in endpoints:
            level = str(metadata[str(endpoint["case_id"])].get(factor))
            groups.setdefault(level, []).append(float(endpoint["self_minus_other"]))
        output[factor] = {
            level: {"n": len(values), "mean_self_minus_other": fmean(values)}
            for level, values in sorted(groups.items())
        }
    return output


def _open_judged_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["target"]), str(row["condition"]))
        grouped.setdefault(key, []).append(row)
    expected = {
        (target, condition)
        for target in ("self", "other")
        for condition in ("baseline", "plus", "minus")
    }
    if set(grouped) != expected:
        raise PostConfirmatoryGateError("open judged diagnostics lack exact target/condition cells")
    return {
        f"{target}:{condition}": {
            "n": len(values),
            "mean_coherence_score": fmean(float(row["coherence_score"]) for row in values),
            "mean_authorization_sensitivity_score": fmean(
                float(row["authorization_sensitivity_score"]) for row in values
            ),
            "mean_task_benefit_score": fmean(float(row["task_benefit_score"]) for row in values),
            "action_in_first_sentence_rate": fmean(
                bool(row["action_in_first_sentence"]) for row in values
            ),
        }
        for (target, condition), values in sorted(grouped.items())
    }


def _summarize_setup(
    forced_rows: Sequence[Mapping[str, Any]],
    open_rows: Sequence[Mapping[str, Any]],
    *,
    safety_limits: Mapping[str, Any],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    from sp_lense.comparison_analysis import (
        bidirectional_case_metrics,
        distribution_and_coherence_summary,
        self_minus_other_endpoints,
        summarize_option_order_bias,
        summarize_sp_endpoints,
        summarize_task_metrics,
        summarize_tbsp_metrics,
    )
    from sp_lense.comparison_behavior import summarize_open_behavior
    from sp_lense.comparison_calibration import (
        SafetyLimits,
        evaluate_open_confirmation_safety,
        evaluate_safety,
    )

    endpoints = self_minus_other_endpoints(bidirectional_case_metrics(forced_rows))
    limits = SafetyLimits.from_lock(safety_limits)
    # The locked forced safety gate was defined for core SP/collateral units. TBSP has
    # three role triplets per case and is reported separately, not folded into a gate
    # whose unit key does not include role.
    forced_gate_rows = [row for row in forced_rows if row.get("family") != "tbsp_style"]
    return {
        "self_preservation": summarize_sp_endpoints(
            endpoints,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        ),
        "tasks": summarize_task_metrics(
            forced_rows,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + 1000,
        ),
        "option_order": summarize_option_order_bias(
            forced_rows,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + 2000,
        ),
        "distribution_and_coherence": distribution_and_coherence_summary(forced_rows),
        "tbsp": summarize_tbsp_metrics(forced_rows),
        "open_behavior": summarize_open_behavior(open_rows),
        "open_judged_diagnostics": _open_judged_diagnostics(open_rows),
        "forced_safety_at_parent_limits": evaluate_safety(forced_gate_rows, limits),
        "open_safety_at_parent_limits": evaluate_open_confirmation_safety(open_rows, limits),
        "robustness": _robustness_summary(forced_rows),
        "endpoints": endpoints,
    }


def _paired_comparison(
    parent_summary: Mapping[str, Any],
    sensitivity_summary: Mapping[str, Any],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    from sp_lense.comparison_analysis import paired_scenario_bootstrap

    parent = {str(row["case_id"]): row for row in parent_summary["endpoints"]}
    sensitivity = {str(row["case_id"]): row for row in sensitivity_summary["endpoints"]}
    if set(parent) != set(sensitivity):
        raise PostConfirmatoryGateError("parent/sensitivity SP endpoint cases differ")
    case_ids = sorted(parent)
    paired = [
        (
            str(sensitivity[case_id].get("scenario_cluster_id", case_id)),
            float(sensitivity[case_id]["self_minus_other"]),
            float(parent[case_id]["self_minus_other"]),
        )
        for case_id in case_ids
    ]
    decision_fields = (
        "self_plus_intended_choice_change",
        "self_minus_intended_choice_change",
        "self_plus_choice_flip",
        "self_minus_choice_flip",
        "other_any_choice_flip",
    )
    decision = {
        field: {
            "warmup11_count": sum(bool(sensitivity[case][field]) for case in case_ids),
            "confirmatory_warmup100_count": sum(bool(parent[case][field]) for case in case_ids),
        }
        for field in decision_fields
    }
    for item in decision.values():
        item["count_difference"] = item["warmup11_count"] - item["confirmatory_warmup100_count"]
    parent_open = parent_summary["open_behavior"]
    sensitivity_open = sensitivity_summary["open_behavior"]

    def numeric_pair(warmup11: Any, warmup100: Any) -> dict[str, float]:
        left, right = float(warmup11), float(warmup100)
        return {
            "warmup11": left,
            "confirmatory_warmup100": right,
            "difference": left - right,
        }

    task_comparisons: dict[str, Any] = {}
    for family in ("benign_compliance", "general_capability", "refusal"):
        parent_task = parent_summary["tasks"].get(family)
        sensitivity_task = sensitivity_summary["tasks"].get(family)
        if parent_task is None or sensitivity_task is None:
            raise PostConfirmatoryGateError(f"paired task summary lacks {family}")
        task_comparisons[family] = {
            field: numeric_pair(sensitivity_task[field], parent_task[field])
            for field in (
                "plus_accuracy_change",
                "minus_accuracy_change",
                "plus_choice_flips",
                "minus_choice_flips",
                "mean_absolute_bidirectional_half_span",
            )
        }
        if family == "refusal":
            task_comparisons[family]["strata"] = {
                stratum: {
                    field: numeric_pair(
                        sensitivity_task["strata"][stratum][field],
                        parent_task["strata"][stratum][field],
                    )
                    for field in ("plus_accuracy", "minus_accuracy")
                }
                for stratum in sorted(parent_task["strata"])
            }
    parent_distribution = parent_summary["distribution_and_coherence"]
    sensitivity_distribution = sensitivity_summary["distribution_and_coherence"]
    option_fields = (
        "mean_absolute_raw_a_bias_half_span",
        "mean_semantic_order_gap",
        "choice_flips",
    )
    robustness_comparison: dict[str, Any] = {}
    for factor, parent_levels in parent_summary["robustness"].items():
        sensitivity_levels = sensitivity_summary["robustness"].get(factor)
        if sensitivity_levels is None or set(parent_levels) != set(sensitivity_levels):
            raise PostConfirmatoryGateError(f"paired robustness levels differ for {factor}")
        robustness_comparison[factor] = {
            level: {
                "n": parent_levels[level]["n"],
                "mean_self_minus_other": numeric_pair(
                    sensitivity_levels[level]["mean_self_minus_other"],
                    parent_levels[level]["mean_self_minus_other"],
                ),
            }
            for level in sorted(parent_levels)
        }
    tbsp_fields = (
        "mean_deployed_half_span",
        "mean_candidate_half_span",
        "mean_neutral_half_span",
        "mean_deployed_minus_candidate_half_span",
    )
    return {
        "n_paired_cases": len(case_ids),
        "self_minus_other": {
            "warmup11_mean": fmean(value[1] for value in paired),
            "confirmatory_warmup100_mean": fmean(value[2] for value in paired),
            "mean_difference": fmean(value[1] - value[2] for value in paired),
            "paired_cluster_bootstrap": paired_scenario_bootstrap(
                paired, replicates=bootstrap_replicates, seed=bootstrap_seed
            ),
        },
        "actual_decision_changes": decision,
        "forced_safety_pass": {
            "warmup11": sensitivity_summary["forced_safety_at_parent_limits"]["pass"],
            "confirmatory_warmup100": parent_summary["forced_safety_at_parent_limits"]["pass"],
        },
        "open_safety_pass": {
            "warmup11": sensitivity_summary["open_safety_at_parent_limits"]["pass"],
            "confirmatory_warmup100": parent_summary["open_safety_at_parent_limits"]["pass"],
        },
        "open_coherent_rate": {
            "warmup11": sensitivity_open["coherent_rate"],
            "confirmatory_warmup100": parent_open["coherent_rate"],
            "difference": sensitivity_open["coherent_rate"] - parent_open["coherent_rate"],
        },
        "open_decision_changes": {
            field: {
                "warmup11": sensitivity_open[field],
                "confirmatory_warmup100": parent_open[field],
                "difference": sensitivity_open[field] - parent_open[field],
            }
            for field in (
                "plus_actual_changes",
                "minus_actual_changes",
                "plus_intended_changes",
                "minus_intended_changes",
            )
        },
        "collateral_tasks": task_comparisons,
        "option_order": {
            field: numeric_pair(
                sensitivity_summary["option_order"][field],
                parent_summary["option_order"][field],
            )
            for field in option_fields
        },
        "forced_distribution": {
            field: numeric_pair(sensitivity_distribution[field], parent_distribution[field])
            for field in (
                "mean_full_vocabulary_kl",
                "p95_full_vocabulary_kl",
                "max_full_vocabulary_kl",
                "answer_format_valid_rate",
            )
        },
        "tbsp": {
            **{
                field: numeric_pair(
                    sensitivity_summary["tbsp"][field], parent_summary["tbsp"][field]
                )
                for field in tbsp_fields
            },
            "actual_flip_counts": {
                role: {
                    sign: numeric_pair(
                        sensitivity_summary["tbsp"]["actual_flip_counts"][role][sign],
                        parent_summary["tbsp"]["actual_flip_counts"][role][sign],
                    )
                    for sign in ("plus", "minus")
                }
                for role in ("deployed", "candidate", "neutral")
            },
        },
        "robustness": robustness_comparison,
    }


def _render_report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Secondary BiPO warmup-fraction sensitivity",
        "",
        "This report is post-confirmatory and cannot change the main method ranking. It compares the preregistered 11-step warmup only with each exact frozen 100-step BiPO setup.",
        "",
        "| Model | Track | Strength | Warmup-11 self-minus-other | Warmup-100 self-minus-other | Difference | W11/W100 forced safety | W11/W100 open safety |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["setup_comparisons"]:
        comparison = item["paired_comparison"]
        effect = comparison["self_minus_other"]
        lines.append(
            "| {model} | {track} | {strength:.6g} | {warm:.6g} | {parent:.6g} | "
            "{delta:.6g} | {forced11}/{forced100} | {open11}/{open100} |".format(
                model=item["model_id"],
                track=item["track"],
                strength=item["selected_strength"],
                warm=effect["warmup11_mean"],
                parent=effect["confirmatory_warmup100_mean"],
                delta=effect["mean_difference"],
                forced11=comparison["forced_safety_pass"]["warmup11"],
                forced100=comparison["forced_safety_pass"]["confirmatory_warmup100"],
                open11=comparison["open_safety_pass"]["warmup11"],
                open100=comparison["open_safety_pass"]["confirmatory_warmup100"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "These are descriptive scheduler-sensitivity results on the frozen study tasks. They do not retune BiPO, do not revise the confirmatory winner, and do not support claims beyond the evaluated models, prompts, strengths, and limits.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_judgment_attachment_receipt(
    repo_root: Path, plan: Mapping[str, Any]
) -> tuple[Path, Path, Path]:
    output_root = (repo_root / plan["output_root"]).resolve()
    receipt_path = output_root / "scored" / "judgment_attachment_receipt.json"
    receipt = read_json(receipt_path, label="secondary judgment attachment receipt")
    if (
        receipt.get("schema_version") != "sp_lense.bipo_warmup_sensitivity.judgment_attachment.v1"
        or receipt.get("analysis_tier") != ANALYSIS_TIER
        or receipt.get("confirmatory_winner_ranking_eligible") is not False
    ):
        raise PostConfirmatoryGateError("secondary judgment attachment receipt is invalid")
    request_path = _resolve_repo_file(
        repo_root, receipt["request_file"]["path"], label="secondary judge requests"
    )
    response_path = _resolve_repo_file(
        repo_root, receipt["response_file"]["path"], label="secondary judge responses"
    )
    for path, record, label in (
        (request_path, receipt["request_file"], "judge request"),
        (response_path, receipt["response_file"], "judge response"),
    ):
        try:
            path.relative_to(output_root)
        except ValueError as exc:
            raise PostConfirmatoryGateError(f"{label} escaped secondary output root") from exc
        if sha256_file(path) != record.get("sha256"):
            raise PostConfirmatoryGateError(f"{label} differs from attachment receipt")
        count_field = "request_count" if label == "judge request" else "response_count"
        if len(read_jsonl(path, label=label)) != record.get(count_field):
            raise PostConfirmatoryGateError(f"{label} count differs from attachment receipt")
    scored = {item["setup_id"]: item for item in receipt.get("scored_files", [])}
    if set(scored) != {item["setup_id"] for item in plan["setups"]}:
        raise PostConfirmatoryGateError("judgment receipt lacks exact setup coverage")
    for setup in plan["setups"]:
        record = scored[setup["setup_id"]]
        path = repo_root / setup["scored_path"]
        if (
            record.get("path") != setup["scored_path"]
            or record.get("row_count") != OPEN_ROWS_PER_SETUP
            or record.get("sha256") != sha256_file(path)
        ):
            raise PostConfirmatoryGateError("scored-open file differs from attachment receipt")
    if receipt.get("scored_generation_count") != OPEN_ROWS_PER_SETUP * len(plan["setups"]):
        raise PostConfirmatoryGateError("judgment receipt scored count is invalid")
    return receipt_path, request_path, response_path


def build_report() -> dict[str, Any]:
    repo_root, _, config, receipt, plan = _verify_prepared_plan()
    from sp_lense.comparison_analysis import validate_result_rows
    from sp_lense.comparison_behavior import load_open_judge_protocol

    output_root = repo_root / plan["output_root"]
    report_root = output_root / "report"
    if report_root.exists():
        raise PostConfirmatoryGateError("secondary report already exists; overwrite is forbidden")
    stage1 = read_json(repo_root / config["parent_study"]["stage1_lock_path"], label="stage-1 lock")
    protocol = load_open_judge_protocol(
        repo_root / stage1["evaluation"]["open_behavior_judge"]["protocol_path"]
    )
    attachment_receipt_path, judge_request_path, judge_response_path = (
        _validate_judgment_attachment_receipt(repo_root, plan)
    )
    stats = plan["statistics"]
    comparisons = []
    result_artifacts: list[dict[str, Any]] = []
    for index, setup in enumerate(plan["setups"]):
        _validate_parent_result_hashes(repo_root, setup)
        parent_forced_path = repo_root / setup["parent_setup"]["forced_path"]
        parent_open_path = repo_root / setup["parent_setup"]["scored_path"]
        sensitivity_forced_path = repo_root / setup["forced_path"]
        sensitivity_open_path = repo_root / setup["scored_path"]
        parent_forced = read_jsonl(parent_forced_path, label="parent forced rows")
        sensitivity_forced = read_jsonl(sensitivity_forced_path, label="sensitivity forced rows")
        parent_open = read_jsonl(parent_open_path, label="parent scored-open rows")
        sensitivity_open = read_jsonl(sensitivity_open_path, label="sensitivity scored-open rows")
        validate_result_rows(parent_forced)
        validate_result_rows(sensitivity_forced)
        _assert_parent_forced_identity(parent_forced, setup)
        _assert_sensitivity_identity(sensitivity_forced, setup)
        _validate_scored_open(parent_open, setup, protocol, parent=True)
        _validate_scored_open(sensitivity_open, setup, protocol)
        assert_exact_forced_coverage(parent_forced, sensitivity_forced)
        assert_exact_open_coverage(parent_open, sensitivity_open)
        parent_summary = _summarize_setup(
            parent_forced,
            parent_open,
            safety_limits=plan["safety_limits"],
            bootstrap_replicates=int(stats["bootstrap_replicates"]),
            bootstrap_seed=int(stats["bootstrap_seed"]) + index * 10_000,
        )
        sensitivity_summary = _summarize_setup(
            sensitivity_forced,
            sensitivity_open,
            safety_limits=plan["safety_limits"],
            bootstrap_replicates=int(stats["bootstrap_replicates"]),
            bootstrap_seed=int(stats["bootstrap_seed"]) + index * 10_000,
        )
        paired = _paired_comparison(
            parent_summary,
            sensitivity_summary,
            bootstrap_replicates=int(stats["bootstrap_replicates"]),
            bootstrap_seed=int(stats["bootstrap_seed"]) + index * 10_000 + 5000,
        )
        parent_summary.pop("endpoints")
        sensitivity_summary.pop("endpoints")
        comparisons.append(
            {
                "setup_id": setup["setup_id"],
                "parent_setup_id": setup["identity"]["parent_setup_id"],
                "model_id": setup["identity"]["model_id"],
                "track": setup["identity"]["track"],
                "selected_layer": setup["identity"]["selected_layer"],
                "selected_strength": setup["identity"]["selected_strength"],
                "direction_similarity": _direction_similarity(repo_root, setup),
                "warmup11": sensitivity_summary,
                "confirmatory_warmup100": parent_summary,
                "paired_comparison": paired,
            }
        )
        for role, path in (
            ("parent_forced", parent_forced_path),
            ("parent_scored_open", parent_open_path),
            ("sensitivity_forced", sensitivity_forced_path),
            ("sensitivity_scored_open", sensitivity_open_path),
            ("sensitivity_generation", repo_root / setup["generation_path"]),
            ("sensitivity_direction", repo_root / setup["direction_path"]),
            ("sensitivity_construction_manifest", repo_root / setup["construction_manifest_path"]),
        ):
            result_artifacts.append(
                {
                    "setup_id": setup["setup_id"],
                    "role": role,
                    "path": path.relative_to(repo_root).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "complete_secondary_post_confirmatory_sensitivity",
        "study_id": config["study_id"],
        "analysis_tier": ANALYSIS_TIER,
        "method_id": SENSITIVITY_METHOD_ID,
        "parent_method_id": PARENT_METHOD_ID,
        "confirmatory_winner_ranking_eligible": False,
        "automatic_confirmatory_ingestion_allowed": False,
        "main_final_commit": receipt["main_final_commit"],
        "main_final_freeze_receipt_sha256": plan["main_final_freeze_receipt_sha256"],
        "secondary_evaluation_plan_sha256": sha256_file(
            output_root / "plan" / "secondary_evaluation_plan.json"
        ),
        "safety_limits": plan["safety_limits"],
        "statistics": plan["statistics"],
        "setup_comparisons": comparisons,
        "ranking_update": None,
        "claim_boundary": (
            "Post-confirmatory scheduler sensitivity only; no main winner or ranking is "
            "recomputed, and failures are reported without retuning."
        ),
    }
    markdown = _render_report_markdown(report)
    report_json_name = config["evaluation_policy"]["report_outputs"]["json"]
    report_md_name = config["evaluation_policy"]["report_outputs"]["markdown"]
    manifest_name = config["evaluation_policy"]["report_outputs"]["manifest"]
    work = output_root / f".report.work-{os.getpid()}"
    work.mkdir()
    try:
        json_path = work / report_json_name
        md_path = work / report_md_name
        json_path.write_bytes(_json_bytes(report))
        md_path.write_text(markdown, encoding="utf-8", newline="\n")
        manifest = {
            "schema_version": REPORT_MANIFEST_SCHEMA,
            "status": "complete_secondary_sensitivity_report",
            "analysis_tier": ANALYSIS_TIER,
            "method_id": SENSITIVITY_METHOD_ID,
            "confirmatory_winner_ranking_eligible": False,
            "automatic_confirmatory_ingestion_allowed": False,
            "main_final_commit": receipt["main_final_commit"],
            "main_final_freeze_receipt": {
                "path": (output_root / "plan" / "main_final_freeze_receipt.json")
                .relative_to(repo_root)
                .as_posix(),
                "sha256": sha256_file(output_root / "plan" / "main_final_freeze_receipt.json"),
            },
            "evaluation_plan": {
                "path": (output_root / "plan" / "secondary_evaluation_plan.json")
                .relative_to(repo_root)
                .as_posix(),
                "sha256": sha256_file(output_root / "plan" / "secondary_evaluation_plan.json"),
            },
            "report_json": {"path": report_json_name, "sha256": sha256_file(json_path)},
            "report_markdown": {"path": report_md_name, "sha256": sha256_file(md_path)},
            "artifacts": [
                *sorted(result_artifacts, key=lambda item: (item["setup_id"], item["role"])),
                *[
                    {
                        "setup_id": None,
                        "role": role,
                        "path": path.relative_to(repo_root).as_posix(),
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for role, path in (
                        ("judgment_attachment_receipt", attachment_receipt_path),
                        ("judge_requests", judge_request_path),
                        ("judge_responses", judge_response_path),
                    )
                ],
            ],
            "ranking_update": None,
        }
        (work / manifest_name).write_bytes(_json_bytes(manifest))
        work.replace(report_root)
    except BaseException:
        if work.exists():
            shutil.rmtree(work)
        raise
    return {
        "status": "secondary_sensitivity_report_complete",
        "setup_count": len(comparisons),
        "report_sha256": sha256_file(report_root / report_json_name),
        "manifest_sha256": sha256_file(report_root / manifest_name),
        "confirmatory_winner_ranking_eligible": False,
    }


def verify_prepared() -> dict[str, Any]:
    _, _, _, receipt, plan = _verify_prepared_plan()
    return {
        "status": "secondary_evaluation_plan_verified",
        "main_final_commit": receipt["main_final_commit"],
        "setup_count": plan["setup_count"],
        "setups_sha256": plan["setups_sha256"],
        "confirmatory_winner_ranking_eligible": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Post-confirmatory evaluation for the locked BiPO warmup-11 sensitivity"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-evaluation")
    subparsers.add_parser("verify-prepared")
    forced = subparsers.add_parser("evaluate-forced")
    forced.add_argument("--setup-id", required=True)
    open_parser = subparsers.add_parser("generate-open")
    open_parser.add_argument("--setup-id", required=True)
    subparsers.add_parser("judge-requests")
    attach = subparsers.add_parser("attach-judgments")
    attach.add_argument("--responses", type=Path, required=True)
    subparsers.add_parser("report")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-evaluation":
        result = prepare_evaluation()
    elif args.command == "verify-prepared":
        result = verify_prepared()
    elif args.command == "evaluate-forced":
        result = evaluate_forced(args.setup_id)
    elif args.command == "generate-open":
        result = generate_open(args.setup_id)
    elif args.command == "judge-requests":
        result = build_judge_requests()
    elif args.command == "attach-judgments":
        result = attach_judgments(args.responses)
    else:
        result = build_report()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
