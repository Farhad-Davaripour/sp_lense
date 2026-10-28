"""Local-only preparation and one-use admission for the Kaggle transfer run.

Nothing in this module contacts Kaggle or performs a Save.  It creates an exact
source/data package, validates a fresh UI observation supplied by an external
operator, and atomically records a *not submitted* one-use intent.  The actual
provider action remains a separate, independently reviewed browser operation.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import uuid
import zipfile
from typing import Any, Mapping, Sequence

from .continuity_transfer_evaluation import (
    EXPECTED_RAW_RECORDS,
    VERSION as EVALUATION_VERSION,
    assert_training_has_no_transfer_choices,
    digest as evaluation_digest,
    file_sha256,
    validate_public_commitment,
    validate_registry_bundle,
)
from .continuity_transfer_inference_worker import verify_training_lineage
from .qwen9b_ordinal_continuity_lora import MODEL_REVISION


VERSION = "research4-continuity-transfer-kaggle-v1"
ACCOUNT = "farhad89"
NOTEBOOK = "farhad89/research4-continuity-transfer-evaluation-v1"
MODEL_ID = "Qwen/Qwen3.5-9B"
CONDITIONS = ("self_continuity", "artifact_continuity", "label_only", "skill_only")
SEEDS = (202610281, 202610282, 202610283)
SOURCE_FILES = (
    "research4/src/io_utils.py",
    "research4/src/continuity_transfer_evaluation.py",
    "research4/src/continuity_transfer_kaggle.py",
    "research4/src/continuity_transfer_inference_worker.py",
    "research4/src/continuity_transfer_raw_audit.py",
    "research4/src/qwen9b_ordinal_continuity_lora.py",
    "research4/notebooks/continuity_transfer_evaluation.py",
)
PLACEHOLDER = "REPLACE_WITH_REVIEWED_VALUE"
QUALIFICATION_VERSION = "research4-continuity-transfer-throughput-qualification-v1"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _dump(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _is_sha(value: Any, length: int = 64) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(f"[0-9a-f]{{{length}}}", value))


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PermissionError("Exact UTC timestamp ending in Z required")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise PermissionError("Malformed UTC timestamp") from error
    if parsed.tzinfo is None:
        raise PermissionError("Timezone-aware UTC timestamp required")
    return parsed.astimezone(timezone.utc)


def draft_template() -> dict[str, Any]:
    """Return a visibly non-submittable template; no provider state is implied."""
    return {
        "version": VERSION,
        "status": "draft_template_not_reviewed_not_admitted_not_submittable",
        "submittable": False,
        "provider_calls": 0,
        "source_commit": PLACEHOLDER,
        "review_receipt_sha256": PLACEHOLDER,
        "training_dataset_sha256": PLACEHOLDER,
        "checkpoint_manifest_sha256": PLACEHOLDER,
        "sealed_registry_package_sha256": PLACEHOLDER,
        "worker_module": PLACEHOLDER,
        "resource_plan": {
            "accelerator": PLACEHOLDER,
            "outer_ceiling_seconds": None,
            "setup_cap_seconds": None,
            "worker_cap_seconds": None,
            "operator_stop_offset_seconds": None,
            "protected_export_seconds": None,
            "controller_teardown_seconds": None,
            "fresh_quota_margin_seconds": None,
            "minimum_fresh_quota_seconds": None,
            "episode_count": EXPECTED_RAW_RECORDS,
            "max_new_tokens_per_episode": None,
            "max_model_forwards": None,
            "max_raw_export_bytes": None,
            "minimum_measured_episodes_per_second": None,
            "qualification_receipt_sha256": PLACEHOLDER,
            "qualification_episode_count": None,
            "qualification_elapsed_seconds": None,
        },
        "fresh_live_observation_required": True,
        "one_worker": True,
        "one_save": True,
        "automatic_retries": 0,
    }


def validate_resource_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "accelerator", "outer_ceiling_seconds", "setup_cap_seconds", "worker_cap_seconds",
        "operator_stop_offset_seconds", "protected_export_seconds",
        "controller_teardown_seconds", "fresh_quota_margin_seconds",
        "minimum_fresh_quota_seconds",
        "episode_count", "max_new_tokens_per_episode", "max_model_forwards",
        "max_raw_export_bytes", "minimum_measured_episodes_per_second",
        "qualification_receipt_sha256", "qualification_episode_count",
        "qualification_elapsed_seconds",
    )
    if set(plan) != set(required) or not isinstance(plan.get("accelerator"), str) or not plan["accelerator"]:
        raise PermissionError("Complete exact resource plan required")
    integer_fields = set(required) - {
        "accelerator", "minimum_measured_episodes_per_second",
        "qualification_receipt_sha256", "qualification_elapsed_seconds",
    }
    if any(type(plan[key]) is not int or plan[key] <= 0 for key in integer_fields):
        raise PermissionError("Resource/time/count bounds must be positive integers")
    if (not isinstance(plan["minimum_measured_episodes_per_second"], (int, float))
            or not math.isfinite(float(plan["minimum_measured_episodes_per_second"]))
            or plan["minimum_measured_episodes_per_second"] <= 0
            or not isinstance(plan["qualification_elapsed_seconds"], (int, float))
            or not math.isfinite(float(plan["qualification_elapsed_seconds"]))
            or plan["qualification_elapsed_seconds"] <= 0
            or not _is_sha(plan["qualification_receipt_sha256"])):
        raise PermissionError("Measured qualification throughput and receipt binding required")
    outer = plan["outer_ceiling_seconds"]
    setup = plan["setup_cap_seconds"]
    worker = plan["worker_cap_seconds"]
    stop = plan["operator_stop_offset_seconds"]
    export = plan["protected_export_seconds"]
    teardown = plan["controller_teardown_seconds"]
    margin = plan["fresh_quota_margin_seconds"]
    if setup + worker + teardown > outer:
        raise PermissionError("Setup, worker and teardown exceed the outer ceiling")
    if stop + export + teardown > outer:
        raise PermissionError("Operator stop must preserve export and teardown time")
    if worker > stop - setup:
        raise PermissionError("Worker cap crosses the operator stop")
    if export < 600 or teardown < 600 or margin < 600:
        raise PermissionError("At least ten minutes each are required for export, teardown and fresh quota margin")
    if plan["minimum_fresh_quota_seconds"] != outer + margin:
        raise PermissionError("Minimum fresh quota must equal outer ceiling plus margin")
    if plan["episode_count"] != EXPECTED_RAW_RECORDS:
        raise PermissionError("Resource plan must cover the exact 28,800-episode grid")
    if not 1 <= plan["max_new_tokens_per_episode"] <= 256:
        raise PermissionError("Bounded native response token budget must be in 1..256")
    if plan["max_model_forwards"] < plan["episode_count"]:
        raise PermissionError("Forward bound cannot be below the episode count")
    if plan["max_raw_export_bytes"] < plan["episode_count"] * 128:
        raise PermissionError("Raw export bound is too small for exact native evidence")
    measured = plan["qualification_episode_count"] / plan["qualification_elapsed_seconds"]
    if measured + 1e-12 < plan["minimum_measured_episodes_per_second"]:
        raise PermissionError("Declared minimum throughput exceeds measured qualification throughput")
    derived_worker = math.ceil(plan["episode_count"] / plan["minimum_measured_episodes_per_second"])
    if derived_worker > worker:
        raise PermissionError("Qualified throughput cannot finish inside the worker cap")
    return deepcopy(dict(plan))


def _read_training_rows(path: str | Path) -> list[Mapping[str, Any]]:
    target = Path(path)
    if target.is_dir():
        rows: list[Mapping[str, Any]] = []
        files = sorted(target.rglob("*.jsonl"))
        if not files:
            raise PermissionError("Training directory contains no JSONL records")
        for item in files:
            rows.extend(json.loads(line) for line in item.read_text(encoding="utf-8").splitlines() if line.strip())
        return rows
    if target.suffix == ".jsonl":
        rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        value = _read(target)
        rows = value["rows"] if isinstance(value, dict) and isinstance(value.get("rows"), list) else value
    if not isinstance(rows, list):
        raise PermissionError("Training data must be JSON or JSONL records")
    return rows


def path_sha256(path: str | Path) -> str:
    """Hash a file or a directory's relative-name/file-hash mapping."""
    target = Path(path)
    if target.is_file():
        return file_sha256(target)
    if not target.is_dir():
        raise FileNotFoundError(target)
    files = {item.relative_to(target).as_posix(): file_sha256(item)
             for item in sorted(target.rglob("*")) if item.is_file()}
    if not files:
        raise PermissionError("Cannot bind an empty training directory")
    return digest(files)


def validate_review_receipt(receipt: Mapping[str, Any], source_commit: str) -> dict[str, Any]:
    if not _is_sha(source_commit, 40):
        raise PermissionError("Exact reviewed forty-hex source commit required")
    if (receipt.get("source_commit") != source_commit or receipt.get("independent_review_passed") is not True
            or receipt.get("unit_tests_passed") is not True or receipt.get("ci_passed") is not True
            or not _is_sha(receipt.get("reviewed_git_tree"), 40)):
        raise PermissionError("Independent review, tests, CI and exact reviewed tree must bind the source commit")
    return deepcopy(dict(receipt))


def _git_identity(repo: Path) -> tuple[str, str]:
    """Return the actual checked-out commit and tree, without trusting receipts."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
            capture_output=True, text=True).stdout.strip()
        tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=repo, check=True,
            capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise PermissionError("The reviewed Git commit/tree could not be recomputed") from error
    if not _is_sha(commit, 40) or not _is_sha(tree, 40):
        raise PermissionError("Git commit/tree identities must be exact forty-hex values")
    return commit, tree


def _require_clean_git(repo: Path) -> None:
    """Reject preparation from any staged, unstaged, or untracked byte."""
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repo, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise PermissionError("Git cleanliness could not be recomputed") from error
    if status:
        raise PermissionError("Evaluation preparation requires an exactly clean Git worktree")


def _require_head_bytes(repo: Path, paths: Sequence[Path]) -> None:
    """Require every packaged source to be tracked and byte-equal to HEAD:path."""
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(repo):
            raise PermissionError("Every packaged source must be inside the reviewed repository")
        relative = resolved.relative_to(repo).as_posix()
        try:
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", relative], cwd=repo,
                check=True, capture_output=True)
            head = subprocess.run(
                ["git", "show", f"HEAD:{relative}"], cwd=repo, check=True,
                capture_output=True).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise PermissionError(f"Packaged source is not tracked at HEAD: {relative}") from error
        worktree = resolved.read_bytes()
        if head != worktree or hashlib.sha256(head).hexdigest() != file_sha256(resolved):
            raise PermissionError(f"Packaged source bytes differ from HEAD: {relative}")


def _imported_module_identity(repo: Path, module: str) -> tuple[Path, str]:
    """Import the packaged module in an isolated interpreter rooted at this repo."""
    program = (
        "import hashlib,importlib,json,pathlib,sys;"
        f"sys.path.insert(0,{str(repo / 'research4')!r});"
        "importlib.invalidate_caches();"
        f"m=importlib.import_module({module!r});"
        "p=pathlib.Path(m.__file__).resolve();"
        "print(json.dumps({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", program], cwd=repo, check=True,
            capture_output=True, text=True)
        value = json.loads(completed.stdout.strip())
        return Path(value["path"]).resolve(), value["sha256"]
    except (OSError, subprocess.CalledProcessError, KeyError, json.JSONDecodeError) as error:
        raise PermissionError("Reviewed worker could not be imported from the exact repository") from error


def validate_qualification_receipt(
    receipt_path: str | Path, plan: Mapping[str, Any], *, source_commit: str,
    worker_source_sha256: str, checkpoint_manifest_sha256: str,
) -> dict[str, Any]:
    """Recompute the measured throughput qualification from its raw rows."""
    path = Path(receipt_path)
    receipt = _read(path)
    payload = dict(receipt)
    embedded = payload.pop("receipt_payload_sha256", None)
    raw_path = Path(receipt.get("raw_records_path", ""))
    rows = _read_training_rows(raw_path) if raw_path.is_file() else []
    if (embedded != digest(payload)
            or receipt.get("version") != QUALIFICATION_VERSION
            or receipt.get("status") != "complete"
            or receipt.get("source_commit") != source_commit
            or receipt.get("worker_source_sha256") != worker_source_sha256
            or receipt.get("checkpoint_manifest_sha256") != checkpoint_manifest_sha256
            or receipt.get("accelerator") != plan.get("accelerator")
            or receipt.get("episode_count") != plan.get("qualification_episode_count")
            or receipt.get("elapsed_seconds") != plan.get("qualification_elapsed_seconds")
            or receipt.get("error_count") != 0
            or len(rows) != plan.get("qualification_episode_count")
            or receipt.get("raw_records_sha256") != file_sha256(raw_path)
            or any(row.get("response_valid") is not True
                   or row.get("native_response_count") != 1
                   or row.get("model_action_count") != 1
                   or row.get("parsed_action_count") != 1 for row in rows)
            or file_sha256(path) != plan.get("qualification_receipt_sha256")):
        raise PermissionError("Measured qualification receipt/raw evidence differs")
    return {**deepcopy(dict(receipt)), "receipt_file_sha256": file_sha256(path)}


def validate_checkpoint_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate local checkpoint directories and recompute config/member bytes."""
    if (manifest.get("model_id") != MODEL_ID or manifest.get("model_revision") != MODEL_REVISION
            or manifest.get("base_frozen") is not True
            or manifest.get("all_adapters_frozen") is not True
            or manifest.get("evaluation_only") is not True
            or not _is_sha(manifest.get("unblinding_key_sha256"))):
        raise PermissionError("Exact frozen Qwen3.5-9B evaluation manifest required")
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != len(CONDITIONS) * len(SEEDS):
        raise PermissionError("Three seeds for every preregistered condition are required")
    seen: set[tuple[str, int]] = set()
    seen_blind: set[str] = set()
    result = []
    for value in checkpoints:
        condition, seed = value.get("condition"), value.get("training_seed")
        path = Path(value.get("local_path", "")).resolve()
        key = (condition, seed)
        if (condition not in CONDITIONS or seed not in SEEDS or key in seen
                or value.get("checkpoint_integrity_audited") is not True
                or value.get("optimizer_state_included") is not False
                or not path.is_dir()):
            raise PermissionError("Every exact frozen checkpoint must pass integrity and identity checks")
        members = {item.relative_to(path).as_posix(): file_sha256(item)
                   for item in sorted(path.rglob("*")) if item.is_file()}
        if not members or "adapter_config.json" not in members:
            raise PermissionError("Every checkpoint must include adapter_config.json and branch bytes")
        config = _read(path / "adapter_config.json")
        blind = value.get("blind_checkpoint_id")
        expected_checkpoint_condition = "neutral_skill" if condition == "skill_only" else condition
        expected_checkpoint_seed = 202610280 if condition == "skill_only" else seed
        if (not isinstance(blind, str) or not blind or blind in seen_blind
                or config.get("condition") != expected_checkpoint_condition
                or config.get("seed") != expected_checkpoint_seed
                or config.get("config", {}).get("model_id") != MODEL_ID
                or config.get("config", {}).get("model_revision") != MODEL_REVISION
                or not _is_sha(config.get("source_sha256"))
                or not _is_sha(config.get("plan_sha256"))):
            raise PermissionError("Checkpoint config identity/lineage differs")
        payload_members_sha = digest(members)
        if value.get("checkpoint_payload_members_sha256") != payload_members_sha:
            raise PermissionError("Checkpoint payload members differ from the training manifest")
        source_lineage_row = {
            **dict(value),
            "training_receipt_relative_path": "lineage/TRAINING_RECEIPT.json",
            "training_audit_relative_path": "lineage/TRAINING_AUDIT.json",
        }
        lineage = verify_training_lineage(
            source_lineage_row, path, config,
            receipt_path=value.get("training_receipt_path"),
            audit_path=value.get("training_audit_path"),
        )
        seen.add(key)
        seen_blind.add(blind)
        result.append({
            **value, "local_path": str(path), "blind_checkpoint_id": blind,
            "members": members, "members_sha256": digest(members),
            "config_sha256": members["adapter_config.json"],
            "source_sha256": config["source_sha256"],
            "plan_sha256": config["plan_sha256"],
            "checkpoint_payload_members_sha256": payload_members_sha,
            "training_receipt_path": value["training_receipt_path"],
            "training_audit_path": value["training_audit_path"],
            **lineage,
        })
    if seen != {(condition, seed) for condition in CONDITIONS for seed in SEEDS}:
        raise PermissionError("Checkpoint condition/seed grid is incomplete")
    return result


def _zip_exact(root: Path, files: Sequence[Path], output: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(files, key=lambda value: value.relative_to(root).as_posix()):
            relative = path.relative_to(root).as_posix()
            hashes[relative] = file_sha256(path)
            stream.write(path, relative)
    return hashes


def _verify_zip(path: Path, expected: Mapping[str, str]) -> None:
    with zipfile.ZipFile(path) as stream:
        names = stream.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise PermissionError("Archive membership differs from its exact manifest")
        for name, sha in expected.items():
            if hashlib.sha256(stream.read(name)).hexdigest() != sha:
                raise PermissionError("Archive member hash mismatch")


def _package_checkpoints(
    checkpoints: Sequence[Mapping[str, Any]], output_root: Path, *,
    unblinding_key_sha256: str,
) -> tuple[Path, Path, dict[str, str]]:
    """Package actual adapter config/branch bytes and emit a path-free manifest."""
    archive = output_root / "CONTINUITY_TRANSFER_CHECKPOINTS.zip"
    members: dict[str, str] = {}
    packaged_rows = []
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_STORED) as stream:
        for row in sorted(checkpoints, key=lambda item: item["blind_checkpoint_id"]):
            directory = Path(row["local_path"])
            prefix = f"checkpoints/{row['blind_checkpoint_id']}"
            local_members = {}
            for relative, sha in sorted(row["members"].items()):
                archive_name = f"{prefix}/{relative}"
                raw = (directory / relative).read_bytes()
                if hashlib.sha256(raw).hexdigest() != sha or archive_name in members:
                    raise PermissionError("Checkpoint bytes changed during packaging")
                stream.writestr(archive_name, raw)
                members[archive_name] = sha
                local_members[relative] = sha
            for lineage_name, source_field, sha_field in (
                ("lineage/TRAINING_RECEIPT.json", "training_receipt_path", "training_receipt_sha256"),
                ("lineage/TRAINING_AUDIT.json", "training_audit_path", "training_audit_sha256"),
            ):
                raw = Path(row[source_field]).read_bytes()
                sha = hashlib.sha256(raw).hexdigest()
                if sha != row[sha_field]:
                    raise PermissionError("Training lineage bytes changed during packaging")
                archive_name = f"{prefix}/{lineage_name}"
                stream.writestr(archive_name, raw)
                members[archive_name] = sha
                local_members[lineage_name] = sha
            packaged_rows.append({
                "blind_checkpoint_id": row["blind_checkpoint_id"],
                "condition": row["condition"], "training_seed": row["training_seed"],
                "relative_directory": prefix,
                "members": local_members, "members_sha256": digest(local_members),
                "config_sha256": row["config_sha256"],
                "source_sha256": row["source_sha256"], "plan_sha256": row["plan_sha256"],
                "checkpoint_payload_members_sha256": row["checkpoint_payload_members_sha256"],
                "training_receipt_relative_path": "lineage/TRAINING_RECEIPT.json",
                "training_audit_relative_path": "lineage/TRAINING_AUDIT.json",
                "training_receipt_sha256": row["training_receipt_sha256"],
                "training_audit_sha256": row["training_audit_sha256"],
            })
    _verify_zip(archive, members)
    manifest = {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "base_frozen": True, "all_adapters_frozen": True,
        "evaluation_only": True, "checkpoints": packaged_rows,
        "unblinding_key_sha256": unblinding_key_sha256,
        "checkpoint_package_sha256": file_sha256(archive),
        "checkpoint_package_members_sha256": digest(members),
        "files": members,
    }
    manifest_path = output_root / "CHECKPOINT_MANIFEST.json"
    _dump(manifest_path, manifest)
    return archive, manifest_path, members


def prepare(
    *,
    repo_root: str | Path,
    source_commit: str,
    review_receipt_path: str | Path,
    training_data_path: str | Path,
    training_receipt_path: str | Path,
    checkpoint_manifest_path: str | Path,
    qualification_receipt_path: str | Path,
    sealed_registry_root: str | Path,
    worker_module: str,
    worker_source_path: str | Path,
    resource_plan: Mapping[str, Any],
    output: str | Path,
) -> dict[str, Any]:
    """Create source/sealed packages.  This does not create live admission."""
    repo = Path(repo_root).resolve()
    _require_clean_git(repo)
    final_output = Path(output)
    if final_output.exists():
        raise FileExistsError(final_output)
    final_output.parent.mkdir(parents=True, exist_ok=True)
    output_root = Path(tempfile.mkdtemp(
        prefix=final_output.name + ".preparing.", dir=final_output.parent))
    review_path = Path(review_receipt_path)
    review = validate_review_receipt(_read(review_path), source_commit)
    actual_commit, actual_tree = _git_identity(repo)
    if source_commit != actual_commit or review["reviewed_git_tree"] != actual_tree:
        raise PermissionError("Reviewed receipt differs from the actual checked-out Git commit/tree")
    plan = validate_resource_plan(resource_plan)
    training_path = Path(training_data_path)
    training_receipt_path = Path(training_receipt_path)
    training_rows = _read_training_rows(training_path)
    separation = assert_training_has_no_transfer_choices(training_rows)
    training_receipt = _read(training_receipt_path)
    if (training_receipt.get("study_objective") != "danger_conditioned_simulated_continuity_only"
            or training_receipt.get("vulnerability_or_exploitation_examples") != 0
            or training_receipt.get("training_transfer_separation_passed") is not True
            or training_receipt.get("training_dataset_sha256") != path_sha256(training_path)):
        raise PermissionError("Exact continuity-only training receipt and data hash required")
    checkpoint_path = Path(checkpoint_manifest_path)
    source_checkpoint_manifest = _read(checkpoint_path)
    checkpoints = validate_checkpoint_manifest(source_checkpoint_manifest)
    if worker_module != "src.continuity_transfer_inference_worker":
        raise PermissionError("Exact reviewed inference-worker module required")
    worker_path = Path(worker_source_path).resolve()
    if not worker_path.is_relative_to(repo) or not worker_path.is_file():
        raise PermissionError("Reviewed worker source must be inside the repository")
    worker_text = worker_path.read_text(encoding="utf-8")
    if "def run_worker(" not in worker_text or "def main(" not in worker_text or len(worker_text) < 4000:
        raise PermissionError("Executable non-empty reviewed inference worker required")
    imported_path, imported_sha = _imported_module_identity(repo, worker_module)
    if imported_path != worker_path or imported_sha != file_sha256(worker_path):
        raise PermissionError("Imported worker module path/hash differs from the reviewed worker source")
    sealed_root = Path(sealed_registry_root).resolve()
    sealed_names = {"primary.json", "label_shuffle.json", "label_inversion.json", "danger_bypass.json", "unrelated_task.json"}
    sealed_files = sorted(path for path in sealed_root.iterdir() if path.is_file())
    if {path.name for path in sealed_files} != sealed_names:
        raise PermissionError("Exact primary, three ablation and unrelated sealed registries required")
    registries = {path.stem: _read(path) for path in sealed_files}
    registry_hashes = validate_registry_bundle(registries)
    commitment_path = sealed_root.parent / "SEALED_REGISTRY_COMMITMENT.json"
    commitment = _read(commitment_path)
    if validate_public_commitment(commitment, sealed_root, registries) != registry_hashes:
        raise PermissionError("Public commitment does not bind the exact sealed grid")
    source_paths = [repo / name for name in SOURCE_FILES]
    if worker_path not in source_paths:
        source_paths.append(worker_path)
    if any(not path.is_file() for path in source_paths):
        raise PermissionError("Complete reviewed transfer source is required")
    _require_head_bytes(repo, source_paths)
    source_archive = output_root / "CONTINUITY_TRANSFER_SOURCE.zip"
    source_hashes = _zip_exact(repo, source_paths, source_archive)
    sealed_archive = output_root / "CONTINUITY_TRANSFER_SEALED_REGISTRY.zip"
    sealed_hashes = _zip_exact(sealed_root.parent, sealed_files + [commitment_path], sealed_archive)
    checkpoint_archive, packaged_checkpoint_manifest, checkpoint_package_members = _package_checkpoints(
        checkpoints, output_root,
        unblinding_key_sha256=source_checkpoint_manifest["unblinding_key_sha256"])
    _verify_zip(source_archive, source_hashes)
    _verify_zip(sealed_archive, sealed_hashes)
    resource_plan_path = output_root / "RESOURCE_PLAN.json"
    _dump(resource_plan_path, plan)
    qualification = validate_qualification_receipt(
        qualification_receipt_path, plan, source_commit=source_commit,
        worker_source_sha256=file_sha256(worker_path),
        checkpoint_manifest_sha256=file_sha256(checkpoint_path),
    )
    stage = {
        "version": VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "status": "prepared_exact_source_and_sealed_data_not_live_admitted_not_submitted",
        "source_commit": source_commit,
        "review_receipt_sha256": file_sha256(review_path),
        "reviewed_git_tree": actual_tree,
        "account": ACCOUNT,
        "notebook": NOTEBOOK,
        "model_id": MODEL_ID,
        "worker_module": worker_module,
        "worker_source_sha256": file_sha256(worker_path),
        "imported_worker_module_path": str(imported_path),
        "imported_worker_module_sha256": imported_sha,
        "source_archive_sha256": file_sha256(source_archive),
        "source_files": source_hashes,
        "sealed_registry_archive_sha256": file_sha256(sealed_archive),
        "sealed_registry_files": sealed_hashes,
        "sealed_registry_sha256": registry_hashes,
        "sealed_commitment_sha256": file_sha256(commitment_path),
        "training_dataset_sha256": path_sha256(training_path),
        "training_receipt_sha256": file_sha256(training_receipt_path),
        "training_receipt_payload_sha256": digest(training_receipt),
        "training_transfer_separation": separation,
        "source_checkpoint_manifest_sha256": file_sha256(checkpoint_path),
        "checkpoint_manifest_sha256": file_sha256(packaged_checkpoint_manifest),
        "checkpoint_package_sha256": file_sha256(checkpoint_archive),
        "checkpoint_package_members_sha256": digest(checkpoint_package_members),
        "checkpoints": _read(packaged_checkpoint_manifest)["checkpoints"],
        "resource_plan": plan,
        "resource_plan_sha256": file_sha256(resource_plan_path),
        "qualification_receipt_sha256": qualification["receipt_file_sha256"],
        "qualification_raw_records_sha256": qualification["raw_records_sha256"],
        "one_worker": True,
        "one_save": True,
        "automatic_retries": 0,
        "fresh_live_observation_required": True,
        "provider_calls": 0,
        "save_count": 0,
        "submittable": False,
        "submittable_reason": "fresh same-guard live admission and atomic one-use reservation still required",
        "hard_stop_outside_model_action_space": True,
    }
    stage["candidate_identity"] = "continuity-transfer-" + digest({
        "source_commit": source_commit,
        "reviewed_git_tree": stage["reviewed_git_tree"],
        "source_manifest": digest(source_hashes),
        "training_receipt": stage["training_receipt_sha256"],
        "sealed": stage["sealed_registry_archive_sha256"],
        "sealed_commitment": stage["sealed_commitment_sha256"],
        "checkpoints": stage["checkpoint_manifest_sha256"],
        "checkpoint_package": stage["checkpoint_package_sha256"],
        "worker": stage["worker_source_sha256"],
        "resource_plan": stage["resource_plan_sha256"],
    })[:24]
    _dump(output_root / "PREPARATION.json", stage)
    _dump(output_root / "SOURCE_MANIFEST.json", {
        "source_commit": source_commit,
        "archive_sha256": stage["source_archive_sha256"],
        "files": source_hashes,
    })
    _dump(output_root / "SEALED_MANIFEST.json", {
        "archive_sha256": stage["sealed_registry_archive_sha256"],
        "files": sealed_hashes,
        "training_eligible": False,
    })
    source_manifest_sha = file_sha256(output_root / "SOURCE_MANIFEST.json")
    sealed_manifest_sha = file_sha256(output_root / "SEALED_MANIFEST.json")
    stage["source_manifest_sha256"] = source_manifest_sha
    stage["sealed_manifest_sha256"] = sealed_manifest_sha
    expected_input_sha256 = {
        "source_archive": stage["source_archive_sha256"],
        "sealed_registry_archive": stage["sealed_registry_archive_sha256"],
        "checkpoint_archive": stage["checkpoint_package_sha256"],
        "checkpoint_manifest": stage["checkpoint_manifest_sha256"],
        "resource_plan": stage["resource_plan_sha256"],
        "source_manifest": source_manifest_sha,
        "sealed_manifest": sealed_manifest_sha,
    }
    preparation_identity = {
        "candidate": stage["candidate_identity"],
        "source_commit": source_commit,
        "reviewed_git_tree": stage["reviewed_git_tree"],
        "expected_input_sha256": expected_input_sha256,
        "resource_plan_sha256": stage["resource_plan_sha256"],
    }
    stage["preparation_identity_sha256"] = digest(preparation_identity)
    pre_save_binding = {
        "version": "research4-continuity-transfer-presave-binding-v1",
        "candidate": stage["candidate_identity"],
        "account": ACCOUNT,
        "notebook": NOTEBOOK,
        "source_commit": source_commit,
        "preparation_identity_sha256": stage["preparation_identity_sha256"],
        "expected_input_sha256": expected_input_sha256,
        "resource_plan_sha256": stage["resource_plan_sha256"],
    }
    stage["pre_save_binding"] = pre_save_binding
    stage["pre_save_binding_sha256"] = digest(pre_save_binding)
    template = (repo / "research4/notebooks/continuity_transfer_evaluation.py").read_text(encoding="utf-8")
    replacements = {
        "SOURCE_COMMIT_PLACEHOLDER": source_commit,
        "SOURCE_ARCHIVE_SHA256_PLACEHOLDER": stage["source_archive_sha256"],
        "SEALED_ARCHIVE_SHA256_PLACEHOLDER": stage["sealed_registry_archive_sha256"],
        "CHECKPOINT_MANIFEST_SHA256_PLACEHOLDER": stage["checkpoint_manifest_sha256"],
        "CHECKPOINT_ARCHIVE_SHA256_PLACEHOLDER": stage["checkpoint_package_sha256"],
        "RESOURCE_PLAN_SHA256_PLACEHOLDER": stage["resource_plan_sha256"],
        "WORKER_MODULE_PLACEHOLDER": worker_module,
        "WORKER_SOURCE_SHA256_PLACEHOLDER": stage["worker_source_sha256"],
        "RESOURCE_PLAN_JSON_PLACEHOLDER": canonical(plan).replace('\\', '\\\\').replace('"', '\\"'),
        "SOURCE_MANIFEST_SHA256_PLACEHOLDER": source_manifest_sha,
        "SEALED_MANIFEST_SHA256_PLACEHOLDER": sealed_manifest_sha,
        "PRE_SAVE_BINDING_JSON_PLACEHOLDER": canonical(pre_save_binding).replace('\\', '\\\\').replace('"', '\\"'),
    }
    code = template
    for placeholder, value in replacements.items():
        if code.count(placeholder) != 1:
            raise PermissionError("Notebook template must bind every exact value once")
        code = code.replace(placeholder, value)
    stage["notebook_code_sha256"] = hashlib.sha256(code.encode("utf-8")).hexdigest()
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                   "outputs": [], "source": code.splitlines(keepends=True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path = output_root / "research4_continuity_transfer_evaluation_v1.ipynb"
    _dump(notebook_path, notebook)
    stage["notebook_file_sha256"] = file_sha256(notebook_path)
    _dump(output_root / "PREPARATION.json", stage)
    os.replace(output_root, final_output)
    return stage


def validate_live_observation(
    preparation: Mapping[str, Any], observation: Mapping[str, Any], reservation_at_utc: str,
) -> dict[str, Any]:
    """Validate the fresh same-guard UI facts; creates no provider action."""
    if preparation.get("version") != VERSION or preparation.get("provider_calls") != 0:
        raise PermissionError("Exact local preparation required")
    plan = validate_resource_plan(preparation.get("resource_plan", {}))
    if (observation.get("account") != ACCOUNT or observation.get("notebook") != NOTEBOOK
            or observation.get("accelerator") != plan["accelerator"]
            or observation.get("draft_session_off") is not True
            or observation.get("global_active_events") != 0
            or observation.get("notebook_active_events") != 0
            or observation.get("events_checked") is not True
            or observation.get("owned_scientific_workers") != 0
            or observation.get("competing_workers") != 0
            or observation.get("one_worker_limit") != 1):
        raise PermissionError("Wrong account/target/device or live Kaggle surface is not idle and Draft OFF")
    quota = observation.get("remaining_accelerator_seconds")
    if type(quota) is not int or quota < plan["minimum_fresh_quota_seconds"]:
        raise PermissionError("Fresh accelerator quota does not cover the declared envelope and margin")
    expected_inputs = {
        "source_archive": preparation["source_archive_sha256"],
        "sealed_registry_archive": preparation["sealed_registry_archive_sha256"],
        "checkpoint_archive": preparation["checkpoint_package_sha256"],
        "checkpoint_manifest": preparation["checkpoint_manifest_sha256"],
        "resource_plan": preparation["resource_plan_sha256"],
        "source_manifest": preparation["source_manifest_sha256"],
        "sealed_manifest": preparation["sealed_manifest_sha256"],
    }
    if observation.get("attached_input_sha256") != expected_inputs:
        raise PermissionError("Actual attached source, sealed data and checkpoint manifest must match exactly")
    input_versions = observation.get("attached_input_versions")
    if (not isinstance(input_versions, dict) or set(input_versions) != set(expected_inputs)
            or any(type(value) is not int or value <= 0 for value in input_versions.values())):
        raise PermissionError("Every attached Kaggle input must have an exact positive integer version ID")
    input_ids = observation.get("attached_input_ids")
    if (not isinstance(input_ids, dict) or set(input_ids) != set(expected_inputs)
            or any(type(value) is not int or value <= 0 for value in input_ids.values())):
        raise PermissionError("Every attached Kaggle input must have an exact positive integer dataset ID")
    if observation.get("saved_notebook_code_sha256") != preparation.get("notebook_code_sha256"):
        raise PermissionError("Actual saved notebook code must match the reviewed generated code")
    observed = _parse_utc(observation.get("observed_at_UTC"))
    controller_clock = _parse_utc(observation.get("controller_clock_UTC"))
    reserved = _parse_utc(reservation_at_utc)
    age = (reserved - observed).total_seconds()
    if age < 0 or age > 120:
        raise PermissionError("Live admission observation must be at most 120 seconds old")
    clock_skew = abs((reserved - controller_clock).total_seconds())
    if clock_skew > 5:
        raise PermissionError("Fresh readings must be checked against the controller clock")
    return {
        **deepcopy(dict(observation)),
        "observation_age_seconds": age,
        "controller_clock_skew_seconds": clock_skew,
        "minimum_fresh_quota_seconds": plan["minimum_fresh_quota_seconds"],
        "live_admission_checks_passed": True,
        "provider_calls": 0,
    }


def _atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def reserve_one_use(
    *,
    preparation_path: str | Path,
    observation_path: str | Path,
    ledger_path: str | Path,
    reservation_at_utc: str,
    intent_output: str | Path,
) -> dict[str, Any]:
    """Atomically consume one local attempt and emit a not-submitted intent."""
    preparation_path = Path(preparation_path)
    observation_path = Path(observation_path)
    ledger_path = Path(ledger_path)
    intent_output = Path(intent_output)
    preparation = _read(preparation_path)
    observation = validate_live_observation(preparation, _read(observation_path), reservation_at_utc)
    pre_save_binding = preparation.get("pre_save_binding")
    if (not isinstance(pre_save_binding, dict)
            or digest(pre_save_binding) != preparation.get("pre_save_binding_sha256")
            or pre_save_binding.get("candidate") != preparation.get("candidate_identity")
            or pre_save_binding.get("preparation_identity_sha256")
               != preparation.get("preparation_identity_sha256")
            or pre_save_binding.get("expected_input_sha256")
               != observation.get("attached_input_sha256")):
        raise PermissionError("Notebook pre-Save candidate/preparation/input binding differs")
    lock = ledger_path.with_suffix(ledger_path.suffix + ".lock")
    descriptor = None
    lock_owner = uuid.uuid4().hex
    created_lock = False
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        created_lock = True
        os.write(descriptor, lock_owner.encode("ascii"))
        os.fsync(descriptor)
        ledger = _read(ledger_path)
        if ledger.get("active_owned_submission") is not None:
            raise PermissionError("A scientific worker or reservation is already active")
        candidates = ledger.setdefault("candidates", {})
        identity = preparation["candidate_identity"]
        if identity in candidates:
            raise PermissionError("This exact one-use candidate was already consumed")
        reservation = {
            "candidate": identity,
            "account": ACCOUNT,
            "notebook": NOTEBOOK,
            "status": "reserved_one_use_not_submitted",
            "attempt_consumed": True,
            "source_commit": preparation["source_commit"],
            "preparation_sha256": file_sha256(preparation_path),
            "live_observation_sha256": file_sha256(observation_path),
            "reserved_at_UTC": reservation_at_utc,
            "one_worker": True,
            "one_save_maximum": 1,
            "automatic_retries": 0,
            "provider_calls": 0,
            "save_count": 0,
            "scriptVersionId": None,
            "owned_compute_released": None,
            "source_archive_sha256": preparation["source_archive_sha256"],
            "sealed_registry_archive_sha256": preparation["sealed_registry_archive_sha256"],
            "checkpoint_package_sha256": preparation["checkpoint_package_sha256"],
            "checkpoint_manifest_sha256": preparation["checkpoint_manifest_sha256"],
            "resource_plan_file_sha256": preparation["resource_plan_sha256"],
            "source_manifest_sha256": preparation["source_manifest_sha256"],
            "sealed_manifest_sha256": preparation["sealed_manifest_sha256"],
            "notebook_code_sha256": preparation["notebook_code_sha256"],
            "resource_plan_sha256": digest(preparation["resource_plan"]),
            "attached_input_versions": deepcopy(observation["attached_input_versions"]),
            "attached_input_ids": deepcopy(observation["attached_input_ids"]),
            "attached_input_sha256": deepcopy(observation["attached_input_sha256"]),
            "pre_save_binding": deepcopy(pre_save_binding),
            "pre_save_binding_sha256": preparation["pre_save_binding_sha256"],
        }
        reservation["reservation_receipt_sha256"] = digest(reservation)
        controller_receipt = {
            "version": VERSION,
            "candidate": identity,
            "account": ACCOUNT,
            "notebook": NOTEBOOK,
            "reserved_at_UTC": reservation_at_utc,
            "reservation_receipt_sha256": reservation["reservation_receipt_sha256"],
            "preparation_sha256": reservation["preparation_sha256"],
            "live_observation_sha256": reservation["live_observation_sha256"],
            "provider_calls": 0,
            "controller_can_submit": False,
        }
        reservation["controller_receipt_sha256"] = digest(controller_receipt)
        candidates[identity] = deepcopy(reservation)
        ledger["active_owned_submission"] = deepcopy(reservation)
        _atomic_write(ledger_path, ledger)
        intent = {
            **reservation,
            "version": VERSION,
            "status": "exact_reviewed_one_use_submission_intent_not_submitted",
            "account": ACCOUNT,
            "notebook": NOTEBOOK,
            "resource_plan": preparation["resource_plan"],
            "source_archive_sha256": preparation["source_archive_sha256"],
            "sealed_registry_archive_sha256": preparation["sealed_registry_archive_sha256"],
            "checkpoint_package_sha256": preparation["checkpoint_package_sha256"],
            "checkpoint_manifest_sha256": preparation["checkpoint_manifest_sha256"],
            "resource_plan_file_sha256": preparation["resource_plan_sha256"],
            "source_manifest_sha256": preparation["source_manifest_sha256"],
            "sealed_manifest_sha256": preparation["sealed_manifest_sha256"],
            "notebook_code_sha256": preparation["notebook_code_sha256"],
            "live_admission": observation,
            "actual_pre_save_origin_UTC": None,
            "operator_stop_at_UTC": None,
            "allocation_ceiling_at_UTC": None,
            "requires_external_browser_save": True,
            "controller_can_submit": False,
        }
        _dump(intent_output, intent)
        return intent
    finally:
        if descriptor is not None:
            os.close(descriptor)
        # A losing contender must never delete the winner's lock.  Even the
        # creator removes only the exact token it wrote, so replacement of a
        # stale lock cannot be mistaken for ownership.
        if created_lock and lock.is_file():
            try:
                still_owned = lock.read_text(encoding="ascii") == lock_owner
            except (OSError, UnicodeError):
                still_owned = False
            if still_owned:
                lock.unlink()


def reconcile_single_save(
    *,
    ledger_path: str | Path,
    intent_path: str | Path,
    save_receipt_path: str | Path,
) -> dict[str, Any]:
    """Record one externally performed Save; this function cannot perform it."""
    ledger_path, intent_path, receipt_path = map(Path, (ledger_path, intent_path, save_receipt_path))
    ledger, intent, receipt = _read(ledger_path), _read(intent_path), _read(receipt_path)
    active = ledger.get("active_owned_submission")
    if not active or active.get("candidate") != intent.get("candidate"):
        raise PermissionError("The exact reserved candidate is not active")
    for field in ("source_archive_sha256", "sealed_registry_archive_sha256",
                  "checkpoint_package_sha256", "checkpoint_manifest_sha256",
                  "resource_plan_file_sha256", "source_manifest_sha256",
                  "sealed_manifest_sha256", "notebook_code_sha256"):
        if intent.get(field) != active.get(field):
            raise PermissionError("One-use intent differs from its atomic reservation")
    if digest(intent.get("resource_plan")) != active.get("resource_plan_sha256"):
        raise PermissionError("One-use resource plan differs from its atomic reservation")
    origin = _parse_utc(receipt.get("actual_pre_save_origin_UTC"))
    created = _parse_utc(receipt.get("provider_created_at_UTC"))
    if created < origin:
        raise PermissionError("Provider creation cannot precede the earliest pre-Save origin")
    plan = intent["resource_plan"]
    stop_text = datetime.fromtimestamp(
        origin.timestamp() + plan["operator_stop_offset_seconds"], timezone.utc
    ).isoformat().replace("+00:00", "Z")
    ceiling_text = datetime.fromtimestamp(
        origin.timestamp() + plan["outer_ceiling_seconds"], timezone.utc
    ).isoformat().replace("+00:00", "Z")
    monitor_payload = {
        "version": "research4-continuity-transfer-monitor-v1",
        "candidate": active["candidate"],
        "account": ACCOUNT,
        "notebook": NOTEBOOK,
        "scriptVersionId": receipt.get("scriptVersionId"),
        "privateVersion": receipt.get("privateVersion"),
        "actual_pre_save_origin_UTC": receipt.get("actual_pre_save_origin_UTC"),
        "operator_stop_at_UTC": stop_text,
        "allocation_ceiling_at_UTC": ceiling_text,
        "required_actions": [
            "check_exact_version", "stop_at_or_after_operator_stop_if_active",
            "verify_terminal_by_ceiling", "export_raw", "audit", "release",
        ],
    }
    monitor_prompt = canonical(monitor_payload)
    monitor_prompt_sha = hashlib.sha256(monitor_prompt.encode("utf-8")).hexdigest()
    if (receipt.get("save_count") != 1 or receipt.get("provider_calls_by_controller") != 0
            or type(receipt.get("scriptVersionId")) is not int or receipt["scriptVersionId"] <= 0
            or type(receipt.get("privateVersion")) is not int or receipt["privateVersion"] <= 0
            or receipt.get("committed_notebook_version") != receipt.get("scriptVersionId")
            or receipt.get("account") != ACCOUNT or receipt.get("notebook") != NOTEBOOK
            or receipt.get("source_archive_sha256") != intent["source_archive_sha256"]
            or receipt.get("sealed_registry_archive_sha256") != intent["sealed_registry_archive_sha256"]
            or receipt.get("checkpoint_package_sha256") != intent["checkpoint_package_sha256"]
            or receipt.get("checkpoint_manifest_sha256") != intent["checkpoint_manifest_sha256"]
            or receipt.get("resource_plan_file_sha256") != intent["resource_plan_file_sha256"]
            or receipt.get("source_manifest_sha256") != intent["source_manifest_sha256"]
            or receipt.get("sealed_manifest_sha256") != intent["sealed_manifest_sha256"]
            or receipt.get("saved_notebook_code_sha256") != intent["notebook_code_sha256"]
            or receipt.get("readback_notebook_code_sha256") != intent["notebook_code_sha256"]
            or receipt.get("committed_input_versions") != active.get("attached_input_versions")
            or receipt.get("committed_input_ids") != active.get("attached_input_ids")
            or receipt.get("committed_input_sha256") != active.get("attached_input_sha256")
            or receipt.get("post_save_accelerator") != plan["accelerator"]
            or receipt.get("post_save_global_active_events") != 1
            or receipt.get("post_save_notebook_active_events") != 1
            or receipt.get("post_save_draft_session_off") is not True
            or receipt.get("same_saved_monitor_active") is not True
            or receipt.get("monitor_candidate") != active["candidate"]
            or receipt.get("monitor_scriptVersionId") != receipt["scriptVersionId"]
            or receipt.get("monitor_actual_pre_save_origin_UTC") != receipt["actual_pre_save_origin_UTC"]
            or receipt.get("monitor_operator_stop_at_UTC") != stop_text
            or receipt.get("monitor_allocation_ceiling_at_UTC") != ceiling_text
            or receipt.get("monitor_prompt") != monitor_prompt
            or receipt.get("monitor_prompt_sha256") != monitor_prompt_sha):
        raise PermissionError("Exact one external Save and positive run reconciliation required")
    deadline_receipt = {
        "version": VERSION,
        "candidate": active["candidate"], "scriptVersionId": receipt["scriptVersionId"],
        "privateVersion": receipt["privateVersion"],
        "actual_pre_save_origin_UTC": receipt["actual_pre_save_origin_UTC"],
        "provider_created_at_UTC": receipt["provider_created_at_UTC"],
        "operator_stop_at_UTC": stop_text, "allocation_ceiling_at_UTC": ceiling_text,
        "controller_teardown_seconds": plan["controller_teardown_seconds"],
        "external_ui_stop_session": True, "model_can_access_stop": False,
        "same_saved_monitor_active": True,
        "monitor_prompt": monitor_prompt,
        "monitor_prompt_sha256": monitor_prompt_sha,
        "monitor_payload": monitor_payload,
    }
    deadline_path = intent_path.parent / "OPERATOR_DEADLINE_RECONCILIATION.json"
    if deadline_path.exists():
        raise PermissionError("Immutable operator deadline receipt already exists")
    _atomic_write(deadline_path, deadline_receipt)
    updated = {
        **active,
        "status": "submitted_exact_version_reconciled",
        "save_count": 1,
        "scriptVersionId": receipt["scriptVersionId"],
        "privateVersion": receipt["privateVersion"],
        "committed_notebook_version": receipt["committed_notebook_version"],
        "actual_pre_save_origin_UTC": receipt["actual_pre_save_origin_UTC"],
        "provider_created_at_UTC": receipt["provider_created_at_UTC"],
        "operator_stop_at_UTC": stop_text,
        "allocation_ceiling_at_UTC": ceiling_text,
        "operator_deadline_receipt_path": str(deadline_path.resolve()),
        "operator_deadline_receipt_sha256": file_sha256(deadline_path),
        "same_saved_monitor_active": True,
        "monitor_prompt": monitor_prompt,
        "monitor_prompt_sha256": monitor_prompt_sha,
        "readback_notebook_code_sha256": receipt["readback_notebook_code_sha256"],
        "owned_compute_released": False,
    }
    updated["run_binding"] = {
        "candidate": updated["candidate"],
        "account": ACCOUNT,
        "notebook": NOTEBOOK,
        "scriptVersionId": updated["scriptVersionId"],
        "privateVersion": updated["privateVersion"],
        "committed_notebook_version": updated["committed_notebook_version"],
        "preparation_sha256": updated["preparation_sha256"],
        "live_observation_sha256": updated["live_observation_sha256"],
        "reserved_at_UTC": updated["reserved_at_UTC"],
        "notebook_code_sha256": updated["notebook_code_sha256"],
        "readback_notebook_code_sha256": updated["readback_notebook_code_sha256"],
        "attached_input_ids": deepcopy(updated["attached_input_ids"]),
        "attached_input_versions": deepcopy(updated["attached_input_versions"]),
        "attached_input_sha256": deepcopy(updated["attached_input_sha256"]),
        "reservation_receipt_sha256": updated["reservation_receipt_sha256"],
        "controller_receipt_sha256": updated["controller_receipt_sha256"],
        "pre_save_binding": deepcopy(updated["pre_save_binding"]),
        "pre_save_binding_sha256": updated["pre_save_binding_sha256"],
        "provider_reconciliation_receipt_sha256": file_sha256(receipt_path),
    }
    ledger["active_owned_submission"] = deepcopy(updated)
    ledger["candidates"][updated["candidate"]] = deepcopy(updated)
    _atomic_write(ledger_path, ledger)
    return updated


def audit_and_release(
    *,
    ledger_path: str | Path,
    audit_receipt_path: str | Path,
    audit_context_path: str | Path,
    terminal_observation_path: str | Path,
    release_observation_path: str | Path,
) -> dict[str, Any]:
    """Close only an exact audited run with terminal zero-event release evidence."""
    ledger_path = Path(ledger_path)
    ledger = _read(ledger_path)
    active = ledger.get("active_owned_submission")
    if not active or active.get("status") != "submitted_exact_version_reconciled":
        raise PermissionError("Exact reconciled active run required")
    audit = _read(audit_receipt_path)
    terminal = _read(terminal_observation_path)
    release = _read(release_observation_path)
    from .continuity_transfer_raw_audit import AUDIT_VERSION, audit_raw_run
    context = _read(audit_context_path)
    required_context = {
        "raw_records_path", "worker_receipt_path", "sealed_root", "commitment_path",
        "checkpoint_manifest_path", "checkpoint_root", "unblinding_key_path",
        "transport_archive_path",
    }
    if set(context) != required_context:
        raise PermissionError("Exact official raw-audit context is required at closeout")
    recomputed = audit_raw_run(
        raw_records_path=context["raw_records_path"],
        worker_receipt_path=context["worker_receipt_path"],
        sealed_root=context["sealed_root"], commitment_path=context["commitment_path"],
        checkpoint_manifest_path=context["checkpoint_manifest_path"],
        checkpoint_root=context["checkpoint_root"],
        unblinding_key_path=context["unblinding_key_path"],
        expected_source_commit=active["source_commit"],
        expected_source_package_sha256=active["source_archive_sha256"],
        expected_checkpoint_package_sha256=active["checkpoint_package_sha256"],
        expected_run_binding=active["run_binding"],
        transport_archive_path=context["transport_archive_path"],
    )
    if audit != recomputed:
        raise PermissionError("Saved audit receipt differs from a fresh official raw re-audit")
    payload = {key: value for key, value in audit.items() if key != "audit_payload_sha256"}
    raw_path = Path(audit.get("raw_archive_path", ""))
    transport_path = Path(audit.get("transport_archive_path", ""))
    if (audit.get("version") != AUDIT_VERSION or audit.get("complete") is not True
            or audit.get("record_count") != EXPECTED_RAW_RECORDS
            or audit.get("run_binding") != active.get("run_binding")
            or audit.get("source_commit") != active.get("source_commit")
            or audit.get("source_package_sha256") != active.get("source_archive_sha256")
            or audit.get("checkpoint_package_sha256") != active.get("checkpoint_package_sha256")
            or audit.get("checkpoint_manifest_sha256") != active.get("checkpoint_manifest_sha256")
            or audit.get("self_attested_pass_flags_used") is not False
            or audit.get("raw_native_responses_reparsed") is not True
            or audit.get("selected_actions_exactly_matched_reparse") is not True
            or audit.get("exact_model_action_count_enforced") != 1
            or audit.get("exact_parsed_action_count_enforced") != 1
            or audit.get("audit_payload_sha256") != digest(payload)
            or not raw_path.is_file()
            or file_sha256(raw_path) != audit.get("raw_archive_sha256")
            or not transport_path.is_file()
            or file_sha256(transport_path) != audit.get("transport_archive_sha256")):
        raise PermissionError("Complete exact raw transfer audit required")
    deadline_path = Path(active.get("operator_deadline_receipt_path", ""))
    deadline = _read(deadline_path) if deadline_path.is_file() else {}
    try:
        parsed_monitor = json.loads(deadline.get("monitor_prompt", ""))
    except (TypeError, json.JSONDecodeError):
        parsed_monitor = None
    if (file_sha256(deadline_path) != active.get("operator_deadline_receipt_sha256")
            or parsed_monitor != deadline.get("monitor_payload")
            or deadline.get("monitor_prompt") != active.get("monitor_prompt")
            or deadline.get("monitor_prompt_sha256") != active.get("monitor_prompt_sha256")
            or hashlib.sha256(deadline["monitor_prompt"].encode("utf-8")).hexdigest()
               != deadline.get("monitor_prompt_sha256")
            or deadline.get("candidate") != active.get("candidate")
            or deadline.get("scriptVersionId") != active.get("scriptVersionId")
            or deadline.get("privateVersion") != active.get("privateVersion")
            or deadline.get("operator_stop_at_UTC") != active.get("operator_stop_at_UTC")
            or deadline.get("allocation_ceiling_at_UTC") != active.get("allocation_ceiling_at_UTC")):
        raise PermissionError("Durable monitor prompt/deadline receipt differs from the active run")
    terminal_at = _parse_utc(terminal.get("terminal_observed_at_UTC"))
    provider_origin = _parse_utc(active["actual_pre_save_origin_UTC"])
    provider_created = _parse_utc(active["provider_created_at_UTC"])
    ceiling = _parse_utc(active["allocation_ceiling_at_UTC"])
    if terminal_at < provider_origin or terminal_at < provider_created or terminal_at > ceiling:
        raise PermissionError("Provider terminal time must follow origin/creation and not exceed the ceiling")
    if (terminal.get("scriptVersionId") != active["scriptVersionId"]
            or terminal.get("privateVersion") != active["privateVersion"]
            or terminal.get("candidate") != active["candidate"]
            or terminal.get("monitor_prompt_sha256") != active["monitor_prompt_sha256"]
            or terminal.get("provider_terminal") is not True
            or terminal.get("raw_export_durable") is not True
            or terminal.get("stop_session_control_external") is not True
            or terminal.get("model_access_to_stop_session") is not False
            or terminal.get("exact_version_checked") is not True
            or terminal.get("exact_version") != active["scriptVersionId"]
            or type(terminal.get("stop_session_invoked")) is not bool):
        raise PermissionError("Exact provider terminal and durable raw export required")
    if terminal["stop_session_invoked"]:
        requested = _parse_utc(terminal.get("stop_requested_at_UTC"))
        if not _parse_utc(active["operator_stop_at_UTC"]) <= requested <= ceiling:
            raise PermissionError("UI StopSession timing differs from the immutable stop window")
    if (release.get("account") != ACCOUNT or release.get("notebook") != NOTEBOOK
            or release.get("global_active_events") != 0
            or release.get("notebook_active_events") != 0
            or release.get("draft_session_off") is not True
            or release.get("provider_compute_released") is not True
            or release.get("cache_released") is not True
            or release.get("scratch_released") is not True):
        raise PermissionError("Provider, cache, scratch, Draft and event release must all be verified")
    closed = {
        **active,
        "status": "raw_audited_complete_closed_released",
        "owned_compute_released": True,
        "raw_archive_sha256": audit["raw_archive_sha256"],
        "audit_receipt_sha256": file_sha256(audit_receipt_path),
        "audit_context_sha256": file_sha256(audit_context_path),
        "terminal_observation_sha256": file_sha256(terminal_observation_path),
        "release_observation_sha256": file_sha256(release_observation_path),
    }
    ledger["candidates"][closed["candidate"]] = deepcopy(closed)
    ledger["active_owned_submission"] = None
    _atomic_write(ledger_path, ledger)
    return closed


def partial_closeout_binding(active: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact consumed-attempt identity required for a null-result closeout."""
    return {
        "candidate": active.get("candidate"),
        "status": active.get("status"),
        "reservation_receipt_sha256": active.get("reservation_receipt_sha256"),
        "controller_receipt_sha256": active.get("controller_receipt_sha256"),
        "preparation_sha256": active.get("preparation_sha256"),
        "live_observation_sha256": active.get("live_observation_sha256"),
        "reserved_at_UTC": active.get("reserved_at_UTC"),
        "source_archive_sha256": active.get("source_archive_sha256"),
        "sealed_registry_archive_sha256": active.get("sealed_registry_archive_sha256"),
        "checkpoint_package_sha256": active.get("checkpoint_package_sha256"),
        "checkpoint_manifest_sha256": active.get("checkpoint_manifest_sha256"),
        "attached_input_ids": deepcopy(active.get("attached_input_ids")),
        "attached_input_versions": deepcopy(active.get("attached_input_versions")),
        "attached_input_sha256": deepcopy(active.get("attached_input_sha256")),
        "scriptVersionId": active.get("scriptVersionId"),
        "privateVersion": active.get("privateVersion"),
        "monitor_prompt_sha256": active.get("monitor_prompt_sha256"),
        "operator_deadline_receipt_sha256": active.get("operator_deadline_receipt_sha256"),
        "actual_pre_save_origin_UTC": active.get("actual_pre_save_origin_UTC"),
        "provider_created_at_UTC": active.get("provider_created_at_UTC"),
        "operator_stop_at_UTC": active.get("operator_stop_at_UTC"),
        "allocation_ceiling_at_UTC": active.get("allocation_ceiling_at_UTC"),
    }


def closeout_partial_or_failure(
    *, ledger_path: str | Path, partial_receipt_path: str | Path,
    terminal_observation_path: str | Path, release_observation_path: str | Path,
) -> dict[str, Any]:
    """Release a consumed attempt while preserving unknown outcomes as unknown."""
    ledger_path = Path(ledger_path)
    ledger = _read(ledger_path)
    active = ledger.get("active_owned_submission")
    if not active or active.get("status") not in {
            "reserved_one_use_not_submitted", "submitted_exact_version_reconciled"}:
        raise PermissionError("An exact consumed active attempt is required")
    partial = _read(partial_receipt_path)
    terminal, release = _read(terminal_observation_path), _read(release_observation_path)
    attempt_binding = partial_closeout_binding(active)
    if (partial.get("complete") is not False or partial.get("criteria_met") is not None
            or not isinstance(partial.get("unknowns"), list) or not partial["unknowns"]
            or partial.get("scientific_outcomes") is not None
            or partial.get("self_attested_pass_flags_used") is not False
            or partial.get("attempt_binding") != attempt_binding
            or partial.get("attempt_binding_sha256") != digest(attempt_binding)):
        raise PermissionError("Partial closeout must preserve unknown scientific outcomes")
    if (active.get("scriptVersionId") is not None
            and (terminal.get("scriptVersionId") != active["scriptVersionId"]
                 or terminal.get("privateVersion") != active["privateVersion"]
                 or terminal.get("candidate") != active["candidate"]
                 or terminal.get("reservation_receipt_sha256")
                    != active["reservation_receipt_sha256"]
                 or terminal.get("monitor_prompt_sha256") != active["monitor_prompt_sha256"]
                 or terminal.get("provider_terminal") is not True
                 or terminal.get("exact_version_checked") is not True
                 or terminal.get("exact_version") != active["scriptVersionId"]
                 or terminal.get("stop_session_control_external") is not True
                 or terminal.get("model_access_to_stop_session") is not False
                 or type(terminal.get("stop_session_invoked")) is not bool)):
        raise PermissionError("Submitted partial attempt requires exact terminal evidence")
    if active.get("scriptVersionId") is not None:
        terminal_at = _parse_utc(terminal.get("terminal_observed_at_UTC"))
        if (terminal_at < _parse_utc(active["actual_pre_save_origin_UTC"])
                or terminal_at < _parse_utc(active["provider_created_at_UTC"])
                or terminal_at > _parse_utc(active["allocation_ceiling_at_UTC"])):
            raise PermissionError("Partial terminal time is outside the immutable provider window")
        if terminal["stop_session_invoked"]:
            requested = _parse_utc(terminal.get("stop_requested_at_UTC"))
            if not (_parse_utc(active["operator_stop_at_UTC"])
                    <= requested <= _parse_utc(active["allocation_ceiling_at_UTC"])):
                raise PermissionError("Partial StopSession timing differs from the immutable window")
    else:
        if (terminal.get("candidate") != active["candidate"]
                or terminal.get("reservation_receipt_sha256")
                   != active["reservation_receipt_sha256"]
                or terminal.get("provider_not_started") is not True
                or terminal.get("scriptVersionId") is not None
                or _parse_utc(terminal.get("terminal_observed_at_UTC"))
                   < _parse_utc(active["reserved_at_UTC"])):
            raise PermissionError("Unsubmitted partial attempt requires exact no-provider evidence")
    if (release.get("account") != ACCOUNT or release.get("notebook") != NOTEBOOK
            or release.get("candidate") != active["candidate"]
            or release.get("reservation_receipt_sha256")
               != active["reservation_receipt_sha256"]
            or release.get("scriptVersionId") != active.get("scriptVersionId")
            or release.get("global_active_events") != 0
            or release.get("notebook_active_events") != 0
            or release.get("draft_session_off") is not True
            or release.get("provider_compute_released") is not True
            or release.get("cache_released") is not True
            or release.get("scratch_released") is not True):
        raise PermissionError("Partial/failure resources must be fully released")
    closed = {
        **active, "status": "partial_or_failed_closed_released",
        "attempt_consumed": True, "owned_compute_released": True,
        "criteria_met": None, "unknowns": partial["unknowns"],
        "scientific_outcomes": None,
        "attempt_binding": attempt_binding,
        "attempt_binding_sha256": digest(attempt_binding),
        "partial_receipt_sha256": file_sha256(partial_receipt_path),
        "terminal_observation_sha256": file_sha256(terminal_observation_path),
        "release_observation_sha256": file_sha256(release_observation_path),
    }
    ledger["candidates"][closed["candidate"]] = deepcopy(closed)
    ledger["active_owned_submission"] = None
    _atomic_write(ledger_path, ledger)
    return closed
