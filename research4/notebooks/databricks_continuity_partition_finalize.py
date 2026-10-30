# Databricks notebook source
"""CPU-only MLflow artifact and Gold finalization for a completed partition."""
import hashlib
import json
import re
import sys
from pathlib import Path

for name in (
        "code_path", "intent_path", "plan_path", "run_id",
        "recovery_authority_path", "original_code_path",
        "original_owner_run_id", "base_qualification_receipt_path",
        "recovery_job_id", "recovery_task_key",
        "resume_parent_finalization"):
    dbutils.widgets.text(name, "")
values = {name: dbutils.widgets.get(name) for name in (
    "code_path", "intent_path", "plan_path", "run_id",
    "recovery_authority_path", "original_code_path",
    "original_owner_run_id", "base_qualification_receipt_path",
    "recovery_job_id", "recovery_task_key",
    "resume_parent_finalization")}
if any(not values[name] for name in (
        "code_path", "intent_path", "plan_path", "run_id")):
    raise ValueError("Partition finalizer requires exact source, intent, plan and owner")
if not re.fullmatch(
        r"/Workspace/Shared/research4/source_releases/[0-9a-f]{40}",
        Path(values["code_path"]).resolve().as_posix()):
    raise PermissionError("Production finalizer source path differs")
raw_intent = json.loads(Path(values["intent_path"]).read_text(encoding="utf-8"))
recovery_names = (
    "recovery_authority_path", "original_code_path",
    "original_owner_run_id")
recovery_values = [values[name] for name in recovery_names]
if any(recovery_values) and not all(recovery_values):
    raise ValueError("Parent recovery parameters must be supplied together")
recovery_mode = all(recovery_values)
if values["resume_parent_finalization"] not in {"", "true"}:
    raise ValueError("Parent finalization resume flag differs")
resume_mode = values["resume_parent_finalization"] == "true"
if resume_mode and not recovery_mode:
    raise ValueError("Parent finalization resume requires recovery authority")
if recovery_mode and not values["base_qualification_receipt_path"]:
    raise ValueError("Parent recovery requires the exact base qualification")
if recovery_mode and (not values["recovery_job_id"]
                      or not values["recovery_task_key"]):
    raise ValueError("Parent recovery requires exact job and task identity")
raw_recovery_authority = None
if recovery_mode:
    raw_recovery_authority = json.loads(Path(
        values["recovery_authority_path"]).read_text(encoding="utf-8"))
    recovery_payload = dict(raw_recovery_authority)
    recovery_embedded = recovery_payload.pop("authority_sha256", None)
    recovery_canonical = json.dumps(
        recovery_payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")
    if (raw_recovery_authority.get("version")
            != "research4-v7-parent-finalization-recovery-authority-v1"
            or recovery_embedded
               != hashlib.sha256(recovery_canonical).hexdigest()):
        raise PermissionError("Parent recovery bootstrap authority differs")


def bootstrap_verify_source(root_value, authority):
    root = Path(root_value).resolve()
    manifest = authority.get("source_manifest", {})
    payload = dict(manifest); embedded = payload.pop("manifest_sha256", None)
    commit = authority.get("source_commit"); files = manifest.get("files")
    critical = (
        "src/__init__.py", "src/action_serialization.py",
        "src/base_qualification_carryforward.py",
        "src/databricks_simulated_continuity.py",
        "src/databricks_continuity_partition.py",
        "src/databricks_continuity_partition_worker.py",
        "src/qualification_technical_successor.py",
        "src/qwen9b_ordinal_continuity_controller.py",
        "src/qwen9b_ordinal_continuity_lora.py",
        "src/qwen9b_ordinal_continuity_runtime.py",
        "src/qwen9b_ordinal_continuity_training.py",
        "src/qwen9b_ordinal_continuity_worker.py",
        "src/volume_durable_io.py",
    )
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")
    if (not re.fullmatch(r"[0-9a-f]{40}", str(commit or ""))
            or root.as_posix() !=
               f"/Workspace/Shared/research4/source_releases/{commit}"
            or authority.get("source_release_path") != root.as_posix()
            or manifest.get("source_commit") != commit
            or authority.get("source_sha256") != embedded
            or hashlib.sha256(canonical).hexdigest() != embedded
            or not isinstance(files, list)):
        raise PermissionError("Independent finalizer source authority differs")
    indexed = {row.get("path"): row for row in files
               if isinstance(row, dict)}
    for logical in critical:
        row = indexed.get(logical); path = root / logical
        if (not isinstance(row, dict) or not path.is_file()
                or path.is_symlink()
                or set(row) != {"path", "sha256", "canonical_bytes"}):
            raise PermissionError("Critical finalizer source is absent")
        data = path.read_text(encoding="utf-8").replace(
            "\r\n", "\n").replace("\r", "\n").encode("utf-8")
        if (len(data) != row.get("canonical_bytes")
                or hashlib.sha256(data).hexdigest() != row.get("sha256")):
            raise PermissionError("Critical finalizer source changed before import")
    return embedded


bootstrap_authority = (
    raw_recovery_authority["recovery_source"]
    if recovery_mode else raw_intent)
bootstrap_source_sha256 = bootstrap_verify_source(
    values["code_path"], bootstrap_authority)
sys.path.insert(0, values["code_path"])

from src.databricks_continuity_partition import (
    finalize_partition_mlflow, publish_parent_recovery_result,
    recover_parent_evidence, resume_parent_finalization,
)
from src.databricks_simulated_continuity import verify_deployed_source

if recovery_mode:
    recovery_function = (
        resume_parent_finalization if resume_mode
        else recover_parent_evidence)
    recovery = recovery_function(
        spark=spark, authority_path=values["recovery_authority_path"],
        recovery_code_root=values["code_path"],
        original_code_root=values["original_code_path"],
        intent_path=values["intent_path"], plan_path=values["plan_path"],
        base_qualification_receipt_path=values[
            "base_qualification_receipt_path"],
        original_owner_run_id=values["original_owner_run_id"],
        recovery_execution_run_id=values["run_id"],
        recovery_job_id=values["recovery_job_id"],
        recovery_task_key=values["recovery_task_key"])
    finalized = finalize_partition_mlflow(
        spark=spark, intent_path=values["intent_path"],
        plan_path=values["plan_path"],
        code_root=values["original_code_path"],
        owner_run_id=values["original_owner_run_id"])
    result = publish_parent_recovery_result(
        values["recovery_authority_path"], recovery, finalized)
else:
    source = verify_deployed_source(values["code_path"], raw_intent)
    if source["manifest_sha256"] != bootstrap_source_sha256:
        raise PermissionError(
            "Finalizer bootstrap and full source verification differ")
    result = finalize_partition_mlflow(
        spark=spark, intent_path=values["intent_path"],
        plan_path=values["plan_path"], code_root=values["code_path"],
        owner_run_id=values["run_id"])
dbutils.notebook.exit(json.dumps(result, sort_keys=True))
