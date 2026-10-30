"""Bounded one-use Databricks partition launcher; never retries."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict
import json
import hashlib
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time

from .databricks_simulated_continuity import (
    GOLD_ADMISSION_TABLE, GOLD_RUN_TABLE, MODEL_RECEIPT, SILVER_TABLE, VOLUME,
    build_source_manifest, canonical, digest, expected_predecessor_binding,
    validate_exact_gold_row,
    validate_base_qualification_receipt, validate_intent,
    validate_source_manifest,
    validate_policy_silver_payload, validate_primary_authority_files,
    verify_deployed_source, verify_intent_base_qualification_authority,
    verify_model_receipt, verify_model_receipt_authority,
    verify_reserved_intent)
from .qwen9b_ordinal_continuity_runtime import directory_tree_sha256
from .qwen9b_ordinal_continuity_training import (
    DATABRICKS_A10_ALLOCATOR_CONF, TrainingPlan, adjacent_pair_schedule,
    validate_sequential_parent, verify_rank_receipt,
)
from .databricks_continuity_partition_worker import validate_mlflow_journal
from .volume_durable_io import (
    SUCCESS_MANIFEST, completion_metadata, file_descriptor,
    publish_immutable_bytes,
    publish_immutable_json, verify_manifest)


PARENT_MANIFEST_VERSION = "research4-parent-volume-manifest-v1"
PARENT_SUCCESS_MANIFEST = "PARENT_OUTPUT_MANIFEST.json"
PARENT_RECOVERY_AUTHORITY_VERSION = (
    "research4-v7-parent-finalization-recovery-authority-v1")
PARENT_RECOVERY_CLAIM_VERSION = (
    "research4-v7-parent-finalization-recovery-claim-v1")
PARENT_RECOVERY_RESULT_VERSION = (
    "research4-v7-parent-finalization-recovery-result-v1")
_V7_RECOVERY_ORIGINAL = {
    "identity": "qualification_rank4_a10_volume_durable_v7",
    "source_commit": "832ffba93ac11a276adb9139795b51554000fe32",
    "source_manifest_sha256": (
        "5b36d597d9634cf59a69538d98ebfd95840138867cd49c2b31454989e50cfb80"),
    "partition_source_file_sha256": (
        "379a44e443daf0c4e0c2a5950cd2a139b2949e1c0998de772f717dbb5b736c6e"),
    "task_output_file_sha256": (
        "1e1b281d2fd9e3ffcd82219e958fb70926dfd7cc10395f1f85cadb2143c4df55"),
    "failure_evidence_sha256": (
        "7836b60c82966c01dbc8ff917146ae0fab27b0e0f972e65eee5cb418bc1c297f"),
    "failure_evidence_file_sha256": (
        "1afb810b74d77ed0e80f6b69de26ece12223d51edb60ca3593d0b8907b44a823"),
    "intent_sha256": (
        "41fa0f68729f6df7cc5e41605e3e0e69c487085e6bd8745f63089c467d09ccb8"),
    "intent_file_sha256": (
        "402a4fe242608e6b8c500886ef6d86c1195bb6bad68dfe653762b87a5c13e30d"),
    "plan_sha256": (
        "83a3e4a213316cba3ae21c9b83ebbf0fa631cfdbed5b615a158bf6495bac31bc"),
    "plan_file_sha256": (
        "a0188e519c895173e7ea39c545f965b5ec5cdd69fc5bfa3cb896f4ec346207ed"),
    "model_tree_sha256": (
        "9845026dbe255e24b105224eebbb5436d315713b9a5c53c434137896b160c5b1"),
    "model_receipt_sha256": (
        "4d1aece45e905dace3990a1ac0e351bd2ef8a53912d2fab93298a1ca06eb41f1"),
    "base_qualification_receipt_sha256": (
        "4d35f901c34ee0fe10bd64e74ccd4089e2d46e98a0e894a405326a3450ac1701"),
    "global_journal_sha256": (
        "873dd87b655b6ea89faf9f7d3b154988c0135087a6073adff94df42d528b74de"),
    "job_id": "676193552960324",
    "owner_run_id": "303633587077489",
    "gpu_task_run_id": "64191367076367",
    "finalizer_run_id": "423953237661721",
    "release_run_id": "607815815128582",
    "mlflow_experiment_id": "2096517895538432",
    "mlflow_experiment_path": (
        "/Shared/research4_simulated_continuity_training_v5"),
    "mlflow_run_id": "6be4c07bd3334b00b439995d314ebd88",
    "records_count": 2304,
    "records_file_sha256": (
        "0c9b41dd7344b8454ccb426b06b3225683d05778af84694c979fc8735796d724"),
    "terminal_file_sha256": (
        "a643c643e40a32c0823618af4ed198eb65c279607c471701d20a12400e947999"),
    "child_manifest_sha256": (
        "d9c0ef67240837974ae2b2348eb9449e808f8fd7e9825010c6b9824c8f692509"),
    "child_manifest_file_sha256": (
        "f4f44c7d1b5215e934d8fd94b7852a810de7e2e0e03ddc353f59d952f08011a9"),
    "training_receipt_sha256": (
        "3bccd1c2b4e68d83a7914c146bb1deb9ba6b5ee182218b39b6270f5357e4fdad"),
    "training_receipt_file_sha256": (
        "db4031371a84e390b4116e454b6b57ca691571a4f2c7f3c6dca95c3ce88506b5"),
    "rank_receipt_file_sha256": (
        "b09e689e8f6789f9ee371fbefcb567583c06b8678ddad52c216c4940ba606a76"),
    "evidence_sha256": (
        "c17d2cce67981f4653127da5bf492aaa766948cbc50adc1dc8b629d37ca88396"),
    "evidence_file_sha256": (
        "d953f42de4ad2d71fc09b2dbcb338b5bbfe6d4fab3c0e8117e3b9533e958ec69"),
    "journal_final_sha256": (
        "2a0ebc12abcf6763f46b641dd0041ad5f5258192a129ab76e67ba35ab5893974"),
    "checkpoint_tree_sha256": (
        "1e043f3cbfdb3227300cd7831715ae3838ab5c12a96e2db0ccfd39a9599c7967"),
    "checkpoint_file_sha256": (
        "da13c73092db7dfde1efe6abac60835ed22834576a64cfa1456c5680f6e0ad2b"),
    "optimizer_file_sha256": (
        "7e6f6f2eb2d442f0f9a17faf091961eb14e9c45987de94da68f2d2193e1fe575"),
    "child_audit_sha256": (
        "9167accc69a983de7ccd98377f4c8443b83325b4cf0c2fdcc43da3186f17a740"),
    "child_audit_file_sha256": (
        "d7f2a6a0ffa1ac9224d08575281d0e074eb0a3c5f6ecdf83859dcd6082737ad5"),
    "mlflow_gold_audit_sha256": (
        "cc9b063e377a5520bc8d59f6ad0aa7873badeec036215530f8b53042f1739a4d"),
    "mlflow_gold_audit_file_sha256": (
        "0b3751431f6752d73aaa669f68e25ad6ca3268896f7abbe7103b87724bfbadcd"),
    "closeout_receipt_sha256": (
        "601a0dec7ecafc085fe1c9b87e1e7254dfd42f76558f2ca8b12932b132c58ef3"),
    "optimizer_updates": 2,
}


def _bounded_run(command, *, cwd, env, stop_seconds, ceiling_seconds):
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=cwd, env=env, shell=False,
                               start_new_session=True)
    stopped = killed = False
    try:
        code = process.wait(timeout=stop_seconds)
    except subprocess.TimeoutExpired:
        stopped = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            code = process.wait(timeout=max(1, ceiling_seconds - stop_seconds))
        except subprocess.TimeoutExpired:
            killed = True; os.killpg(process.pid, signal.SIGKILL); code = process.wait()
    return {"exit_code": code, "elapsed_seconds": time.monotonic() - started,
            "operator_stop_sent": stopped, "ceiling_kill_sent": killed,
            "child_exit_confirmed": process.poll() is not None}


def _stage_local_model(volume_root, volume_receipt, identity, expected_tree,
                       expected_receipt_sha256):
    destination = Path(tempfile.gettempdir()) / "research4_model_cache" / identity
    if destination.exists():
        raise FileExistsError("Per-run local model cache already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = verify_model_receipt_authority(
        volume_receipt, expected_tree_sha256=expected_tree,
        expected_receipt_sha256=expected_receipt_sha256)
    required = int(receipt["total_bytes"]) + 8 * 1024**3
    free_before = shutil.disk_usage(destination.parent).free
    if free_before < required:
        raise RuntimeError("Local disk lacks model-copy plus 8 GiB export margin")
    started = time.monotonic(); local_model = destination / "model"
    destination.mkdir(parents=True)
    try:
        shutil.copytree(volume_root, local_model)
        local_receipt = destination / "MODEL_SNAPSHOT_RECEIPT.json"
        shutil.copy2(volume_receipt, local_receipt)
        proof = verify_model_receipt(local_model, local_receipt)
        if (proof["tree_sha256"] != expected_tree
                or proof["receipt_sha256"] != expected_receipt_sha256
                or proof != receipt):
            raise PermissionError("Local model cache differs from volume authority")
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    elapsed = time.monotonic() - started
    return local_model, local_receipt, {
        "source_model_path": str(volume_root),
        "source_receipt_path": str(volume_receipt),
        "copy_seconds": elapsed,
        "copy_bytes": proof["total_bytes"],
        "copy_bytes_per_second": proof["total_bytes"] / max(elapsed, 1e-9),
        "public_model_download_bytes": 0,
        "hf_network_fallback_disabled": True,
        "snapshot_download_used": False,
        "local_disk_free_before": free_before,
        "local_disk_free_after": shutil.disk_usage(destination.parent).free,
        "tree_sha256": proof["tree_sha256"],
    }


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, value):
    stage = Path(tempfile.gettempdir()) / "research4_volume_publish_stage"
    return publish_immutable_json(value, path, local_stage_dir=stage)


def _parent_manifest(run_root, *, identity, intent_sha256, plan_sha256):
    root = Path(run_root)
    relative_paths = [
        "records.jsonl", "TERMINAL.json",
        f"output/{SUCCESS_MANIFEST}",
        "output/PARENT_FINALIZATION.json",
    ]
    files = []
    for relative in relative_paths:
        descriptor = file_descriptor(root / relative)
        files.append({"path": relative, **descriptor})
    value = {
        "version": PARENT_MANIFEST_VERSION,
        "status": "complete",
        "identity": identity,
        "intent_sha256": intent_sha256,
        "plan_sha256": plan_sha256,
        "files": files,
    }
    value["manifest_sha256"] = digest(value)
    _atomic_json(root / PARENT_SUCCESS_MANIFEST, value)
    readback = _read(root / PARENT_SUCCESS_MANIFEST)
    payload = dict(readback); embedded = payload.pop("manifest_sha256", None)
    if payload != {key: item for key, item in value.items()
                   if key != "manifest_sha256"} or embedded != digest(payload):
        raise PermissionError("Parent Volume manifest readback differs")
    return readback


def _verify_parent_manifest(run_root, *, identity, intent_sha256, plan_sha256):
    root = Path(run_root); path = root / PARENT_SUCCESS_MANIFEST
    value = _read(path); payload = dict(value)
    embedded = payload.pop("manifest_sha256", None)
    if (payload.get("version") != PARENT_MANIFEST_VERSION
            or payload.get("status") != "complete"
            or payload.get("identity") != identity
            or payload.get("intent_sha256") != intent_sha256
            or payload.get("plan_sha256") != plan_sha256
            or embedded != digest(payload)):
        raise PermissionError("Parent Volume manifest differs")
    expected_paths = [
        "records.jsonl", "TERMINAL.json",
        f"output/{SUCCESS_MANIFEST}",
        "output/PARENT_FINALIZATION.json",
    ]
    files = payload.get("files")
    if (not isinstance(files, list)
            or [row.get("path") for row in files] != expected_paths):
        raise PermissionError("Parent Volume manifest file order differs")
    for row in files:
        relative = row.get("path")
        if (not isinstance(relative, str)
                or set(row) != {"path", "bytes", "sha256"}
                or file_descriptor(root / relative) != {
                    "bytes": row["bytes"], "sha256": row["sha256"]}):
            raise PermissionError("Parent Volume manifest file differs")
    return value


def _verify_durable_completion(run_root, *, identity, intent_sha256,
                               plan_sha256, optimizer_updates):
    """Single finalizer boundary for exact child and parent completion."""
    root = Path(run_root); output = root / "output"
    child = verify_manifest(
        output, manifest_name=SUCCESS_MANIFEST,
        expected_status="complete", required_update_count=optimizer_updates,
        allowed_extra={"PARENT_FINALIZATION.json"},
        expected_metadata=completion_metadata(
            identity=identity, intent_sha256=intent_sha256,
            plan_sha256=plan_sha256,
            executed_optimizer_step_count=optimizer_updates,
            durable_published_update_count=optimizer_updates))
    parent = _verify_parent_manifest(
        root, identity=identity, intent_sha256=intent_sha256,
        plan_sha256=plan_sha256)
    return {"child": child, "parent": parent}


def _replay_completed_child(run_root, *, intent, plan_dict, exact_plan, rows):
    """Replay a completed child before any parent/Gold publication."""
    root = Path(run_root); output = root / "output"
    verify_manifest(
        output, manifest_name=SUCCESS_MANIFEST, expected_status="complete",
        required_update_count=exact_plan.optimizer_updates,
        allowed_extra={"PARENT_FINALIZATION.json"},
        expected_metadata=completion_metadata(
            identity=intent["identity"],
            intent_sha256=intent["intent_sha256"],
            plan_sha256=intent["plan_sha256"],
            executed_optimizer_step_count=exact_plan.optimizer_updates,
            durable_published_update_count=exact_plan.optimizer_updates))
    receipt_path = output / "TRAINING_RECEIPT.json"
    rank_receipt_path = output / "RANK_0_RECEIPT.json"
    evidence_path = output / "DATABRICKS_EVIDENCE.json"
    receipt = _read(receipt_path); rank_receipt = _read(rank_receipt_path)
    evidence = _read(evidence_path); evidence_payload = dict(evidence)
    evidence_sha256 = evidence_payload.pop("evidence_sha256", None)
    if evidence_sha256 != digest(evidence_payload):
        raise PermissionError("Databricks evidence self-hash differs")
    if (receipt.get("status") != "complete"
            or receipt.get("successful_optimizer_updates")
               != exact_plan.optimizer_updates):
        raise PermissionError("Completed Databricks receipt differs from exact plan")
    journal = validate_mlflow_journal(
        output, intent=intent, plan=exact_plan, receipt=receipt,
        rank_receipt=rank_receipt, evidence=evidence)
    verify_rank_receipt(
        rank_receipt, expected_plan_sha256=intent["plan_sha256"],
        expected_global_journal_sha256=intent["global_journal_sha256"],
        expected_plan=exact_plan,
        expected_global_journal=adjacent_pair_schedule(rows, exact_plan))
    checkpoint = output / "checkpoint"
    checkpoint_tree = directory_tree_sha256(checkpoint)
    checkpoint_metadata_path = checkpoint / "adapter_config.json"
    checkpoint_metadata = _read(checkpoint_metadata_path)
    checkpoint_payload = dict(checkpoint_metadata)
    checkpoint_embedded = checkpoint_payload.pop("receipt_sha256", None)
    receipt_without_checkpoint = dict(receipt)
    receipt_checkpoint = receipt_without_checkpoint.pop("checkpoint", None)
    receipt_checkpoint_sha = receipt_without_checkpoint.pop(
        "checkpoint_metadata_sha256", None)
    if (receipt_without_checkpoint != rank_receipt
            or receipt_checkpoint != checkpoint_metadata
            or checkpoint_embedded != digest(checkpoint_payload)
            or receipt_checkpoint_sha != _file_sha256(
                checkpoint_metadata_path)):
        raise PermissionError("Training receipt/checkpoint differs from the rank receipt")
    return {
        "plan": exact_plan, "plan_dict": plan_dict,
        "receipt": receipt, "rank_receipt": rank_receipt,
        "evidence": evidence, "evidence_sha256": evidence_sha256,
        "journal": journal, "checkpoint_tree_sha256": checkpoint_tree,
        "checkpoint_metadata_path": checkpoint_metadata_path,
        "receipt_path": receipt_path, "rank_receipt_path": rank_receipt_path,
        "evidence_path": evidence_path,
    }


def _publish_parent_evidence(*, run_root, intent, plan_dict, exact_plan, rows,
                             model_receipt, base_qualification,
                             primary_authority, parent_checkpoint,
                             owner_run_id, mlflow_run_id,
                             deadline_check=None):
    """Publish the two immutable parent files from a verified durable child."""
    root = Path(run_root)
    terminal = _read(root / "TERMINAL.json")
    parent_path = root / "output" / "PARENT_FINALIZATION.json"
    manifest_path = root / PARENT_SUCCESS_MANIFEST
    if manifest_path.exists() and not parent_path.is_file():
        raise PermissionError(
            "Parent manifest exists without its bound finalization file")
    if (plan_dict != intent.get("plan")
            or plan_dict != asdict(exact_plan)
            or exact_plan.full_plan_sha256 != intent.get("plan_sha256")):
        raise PermissionError("Parent publication plan binding differs")
    if (terminal.get("exit_code") != 0
            or terminal.get("operator_stop_sent") is not False
            or terminal.get("ceiling_kill_sent") is not False
            or terminal.get("child_exit_confirmed") is not True):
        raise PermissionError("Parent publication lacks successful child termination")
    if (model_receipt.get("tree_sha256") != intent.get("model_tree_sha256")
            or model_receipt.get("receipt_sha256")
               != intent.get("model_receipt_sha256")
            or base_qualification.get("receipt_sha256")
               != intent.get("base_qualification_receipt_sha256")):
        raise PermissionError("Parent publication model or base authority differs")
    if (not re.fullmatch(r"[0-9]+", str(owner_run_id))
            or not re.fullmatch(r"[0-9a-f]{32}", str(mlflow_run_id))):
        raise PermissionError("Parent publication owner or MLflow identity differs")
    if plan_dict["mode"] == "qualification":
        if primary_authority is not None or parent_checkpoint is not None:
            raise PermissionError("Qualification parent authority differs")
    elif primary_authority is None:
        raise PermissionError("Primary parent authority is absent")
    child = _replay_completed_child(
        root, intent=intent, plan_dict=plan_dict,
        exact_plan=exact_plan, rows=rows)
    receipt = child["receipt"]; rank_receipt = child["rank_receipt"]
    evidence_sha256 = child["evidence_sha256"]
    journal = child["journal"]
    checkpoint_tree = child["checkpoint_tree_sha256"]
    parent_tree = (None if not parent_checkpoint else
                   directory_tree_sha256(parent_checkpoint))
    pending_gold_row = {
        "partition_identity": intent["identity"], "mode": plan_dict["mode"],
        "condition": plan_dict["condition"], "stage": plan_dict["stage"],
        "seed": plan_dict["seed"], "status": "pending_mlflow_finalize",
        "mlflow_run_id": mlflow_run_id,
        "plan_sha256": intent["plan_sha256"],
        "source_commit": intent["source_commit"],
        "model_tree_sha256": model_receipt["tree_sha256"],
        "receipt_sha256": rank_receipt["receipt_sha256"],
        "intent_sha256": intent["intent_sha256"],
        "evidence_sha256": evidence_sha256,
        "admission_sha256": (None if primary_authority is None else
                              primary_authority["admission"]["admission_sha256"]),
        "qualification_receipt_sha256": (
            base_qualification["receipt_sha256"] if primary_authority is None
            else primary_authority["qualification_receipt"]["receipt_sha256"]),
        "receipt_json": canonical(receipt),
        "optimizer_updates": exact_plan.optimizer_updates,
        "checkpoint_tree_sha256": checkpoint_tree,
        "parent_checkpoint_tree_sha256": parent_tree,
        "receipt_replayed": True,
    }
    if deadline_check is not None:
        deadline_check()
    parent_finalization = {
        "version": "research4-parent-mlflow-finalization-v4",
        "identity": intent["identity"],
        "owner_run_id": str(owner_run_id),
        "intent_sha256": intent["intent_sha256"],
        "plan_sha256": intent["plan_sha256"],
        "mlflow_run_id": mlflow_run_id,
        "terminal_file_sha256": _file_sha256(root / "TERMINAL.json"),
        "journal_final_sha256": journal["final"]["final_sha256"],
        "training_receipt_file_sha256": _file_sha256(child["receipt_path"]),
        "rank_receipt_file_sha256": _file_sha256(child["rank_receipt_path"]),
        "evidence_file_sha256": _file_sha256(child["evidence_path"]),
        "checkpoint_metadata_file_sha256": _file_sha256(
            child["checkpoint_metadata_path"]),
        "checkpoint_tree_sha256": checkpoint_tree,
        "receipt_sha256": receipt["receipt_sha256"],
        "rank_receipt_sha256": rank_receipt["receipt_sha256"],
        "evidence_sha256": evidence_sha256,
        "pending_gold_row": pending_gold_row,
    }
    parent_finalization["parent_finalization_sha256"] = digest(
        parent_finalization)
    _atomic_json(root / "output" / "PARENT_FINALIZATION.json",
                 parent_finalization)
    parent_manifest = _parent_manifest(
        root, identity=intent["identity"],
        intent_sha256=intent["intent_sha256"],
        plan_sha256=intent["plan_sha256"])
    return {"status": "pending_cpu_finalize", "identity": intent["identity"],
            "mlflow_run_id": mlflow_run_id,
            "receipt_sha256": rank_receipt["receipt_sha256"],
            "parent_finalization": parent_finalization,
            "parent_manifest": parent_manifest}


def validate_parent_recovery_authority(value, *, recovery_code_root):
    """Validate the single reviewed v7 CPU parent-recovery authority."""
    authority = dict(value) if isinstance(value, dict) else {}
    required = {
        "version", "execution_platform", "cpu_only", "one_use",
        "automatic_retries", "issued_at_utc", "expires_at_utc",
        "identity", "original",
        "recovery_source", "files", "scope", "human_authority",
        "authority_sha256",
    }
    payload = dict(authority); embedded = payload.pop("authority_sha256", None)
    if (set(authority) != required
            or authority.get("version") != PARENT_RECOVERY_AUTHORITY_VERSION
            or authority.get("execution_platform") != "databricks"
            or authority.get("cpu_only") is not True
            or authority.get("one_use") is not True
            or authority.get("automatic_retries") != 0
            or authority.get("identity") != _V7_RECOVERY_ORIGINAL["identity"]
            or authority.get("original") != _V7_RECOVERY_ORIGINAL
            or embedded != digest(payload)):
        raise PermissionError("Parent recovery authority identity differs")
    try:
        issued = datetime.fromisoformat(
            authority["issued_at_utc"].replace("Z", "+00:00"))
        expires = datetime.fromisoformat(
            authority["expires_at_utc"].replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise PermissionError("Parent recovery authority expiry differs") from exc
    now = datetime.now(timezone.utc)
    if (issued.tzinfo is None or expires.tzinfo is None
            or (issued - now).total_seconds() > 300
            or expires <= now
            or expires <= issued
            or (expires - issued).total_seconds() > 3600):
        raise PermissionError("Parent recovery authority is expired")
    scope = authority.get("scope")
    if scope != {
            "allowed_task": "finalize_partition_mlflow",
            "max_cloud_runs": 1, "max_retries": 0,
            "timeout_seconds": 900, "gpu_execution": False,
            "model_loads": 0, "model_forwards": 0,
            "backward_passes": 0, "optimizer_updates": 0,
            "new_mlflow_run": False, "new_scientific_identity": False}:
        raise PermissionError("Parent recovery CPU-only scope differs")
    human = authority.get("human_authority")
    if (not isinstance(human, dict)
            or set(human) != {"thread_id", "message_id", "instruction"}
            or human.get("thread_id")
               != "01a182ea-a06d-7223-9a17-90f33b1276f6"
            or human.get("message_id")
               != "01a18df6-7e5a-7a02-8eb3-efab4055294e"
            or human["instruction"]
               != "try again just the failed step not all the steps"):
        raise PermissionError("Parent recovery human authority differs")
    recovery = authority.get("recovery_source")
    if not isinstance(recovery, dict) or set(recovery) != {
            "source_commit", "source_release_path", "source_sha256",
            "source_manifest"}:
        raise PermissionError("Parent recovery source binding differs")
    manifest = validate_source_manifest(recovery.get("source_manifest"))
    commit = recovery.get("source_commit")
    expected_root = f"/Workspace/Shared/research4/source_releases/{commit}"
    actual_root = Path(recovery_code_root).resolve().as_posix()
    actual_manifest = build_source_manifest(recovery_code_root, commit)
    if (not re.fullmatch(r"[0-9a-f]{40}", str(commit or ""))
            or recovery.get("source_release_path") != expected_root
            or actual_root != expected_root
            or recovery.get("source_sha256") != manifest["manifest_sha256"]
            or manifest.get("source_commit") != commit
            or actual_manifest != manifest):
        raise PermissionError("Parent recovery deployed source differs")
    files = authority.get("files")
    expected_file_keys = {
        "intent", "plan", "base_qualification_receipt",
        "failure_evidence", "task_output", "child_audit",
        "mlflow_gold_audit",
    }
    if not isinstance(files, dict) or set(files) != expected_file_keys:
        raise PermissionError("Parent recovery evidence file set differs")
    for descriptor in files.values():
        if (not isinstance(descriptor, dict)
                or set(descriptor) != {"path", "sha256"}
                or not isinstance(descriptor.get("path"), str)
                or not descriptor["path"].startswith(VOLUME + "/")
                or not re.fullmatch(r"[0-9a-f]{64}",
                                    str(descriptor.get("sha256", "")))):
            raise PermissionError("Parent recovery evidence descriptor differs")
    return authority


def _verify_self_hashed_evidence(path, *, field, expected_self_sha256,
                                 expected_file_sha256):
    target = Path(path)
    data = target.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_file_sha256:
        raise PermissionError("Parent recovery evidence file hash differs")
    value = json.loads(data.decode("utf-8")); payload = dict(value)
    embedded = payload.pop(field, None)
    if ((expected_self_sha256 is not None
         and embedded != expected_self_sha256)
            or embedded != digest(payload)):
        raise PermissionError("Parent recovery evidence self-hash differs")
    return value


def _read_exact_text(path, expected_sha256):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise PermissionError("Parent recovery exact text file differs")
    return data.decode("utf-8")


def _parent_recovery_root():
    return (Path(VOLUME) / "parent_finalization_recovery" /
            f"{PARENT_RECOVERY_AUTHORITY_VERSION}-"
            f"{_V7_RECOVERY_ORIGINAL['identity']}")


def recover_parent_evidence(*, spark, authority_path, recovery_code_root,
                            original_code_root, intent_path, plan_path,
                            base_qualification_receipt_path,
                            original_owner_run_id, recovery_execution_run_id,
                            recovery_job_id, recovery_task_key):
    """Recover only missing parent evidence from the exact completed v7 child."""
    from pyspark.sql import functions as F
    import mlflow

    authority = validate_parent_recovery_authority(
        _read(authority_path), recovery_code_root=recovery_code_root)
    original = authority["original"]; files = authority["files"]
    expected_original_root = (
        f"/Workspace/Shared/research4/source_releases/"
        f"{original['source_commit']}")
    if (Path(original_code_root).resolve().as_posix() != expected_original_root
            or not str(authority_path).startswith(VOLUME + "/")
            or str(original_owner_run_id) != original["owner_run_id"]
            or not re.fullmatch(r"[0-9]+", str(recovery_execution_run_id))
            or str(recovery_job_id) != original["job_id"]
            or recovery_task_key != "finalize_partition_mlflow"
            or str(intent_path) != files["intent"]["path"]
            or str(plan_path) != files["plan"]["path"]
            or str(base_qualification_receipt_path)
               != files["base_qualification_receipt"]["path"]):
        raise PermissionError("Parent recovery invocation binding differs")
    for descriptor in files.values():
        if _file_sha256(descriptor["path"]) != descriptor["sha256"]:
            raise PermissionError("Parent recovery input file differs")
    if (files["intent"]["sha256"] != original["intent_file_sha256"]
            or files["plan"]["sha256"] != original["plan_file_sha256"]):
        raise PermissionError("Parent recovery intent or plan file differs")

    intent = validate_intent(_read(intent_path)); plan_dict = _read(plan_path)
    exact_plan = TrainingPlan(**plan_dict).validate()
    original_source = verify_deployed_source(original_code_root, intent)
    if (intent["identity"] != original["identity"]
            or intent["source_commit"] != original["source_commit"]
            or intent["source_sha256"] != original["source_manifest_sha256"]
            or original_source["manifest_sha256"]
               != original["source_manifest_sha256"]
            or intent["intent_sha256"] != original["intent_sha256"]
            or intent["plan_sha256"] != original["plan_sha256"]
            or intent["global_journal_sha256"]
               != original["global_journal_sha256"]
            or intent["model_tree_sha256"] != original["model_tree_sha256"]
            or intent["model_receipt_sha256"]
               != original["model_receipt_sha256"]
            or plan_dict != intent["plan"]
            or asdict(exact_plan) != plan_dict
            or exact_plan.optimizer_updates != original["optimizer_updates"]):
        raise PermissionError("Parent recovery original science differs")
    source_row = next(
        (row for row in original_source["files"]
         if row["path"] == "src/databricks_continuity_partition.py"), None)
    if (not source_row
            or source_row["sha256"]
               != original["partition_source_file_sha256"]):
        raise PermissionError("Parent recovery original failure source differs")

    base_qualification = validate_base_qualification_receipt(
        _read(base_qualification_receipt_path))
    base_authority = verify_intent_base_qualification_authority(
        intent, base_qualification, original_code_root)
    model_receipt = verify_model_receipt_authority(
        MODEL_RECEIPT, expected_tree_sha256=original["model_tree_sha256"],
        expected_receipt_sha256=original["model_receipt_sha256"])
    if (base_qualification["receipt_sha256"]
            != original["base_qualification_receipt_sha256"]):
        raise PermissionError("Parent recovery base qualification differs")
    base_rows = spark.table(GOLD_RUN_TABLE).where(
        F.col("partition_identity") == base_qualification["identity"]).collect()
    validate_exact_gold_row(base_rows, {
        "partition_identity": base_qualification["identity"],
        "mode": "base_qualification", "condition": "frozen_base",
        "stage": "format_schema", "seed": 0,
        "plan_sha256": base_qualification["plan_sha256"],
        "source_commit": base_authority["source_commit"],
        "model_tree_sha256": intent["model_tree_sha256"],
        "checkpoint_tree_sha256": intent["model_tree_sha256"],
        "receipt_sha256": base_qualification["receipt_sha256"],
        "evidence_sha256": base_qualification["raw_results_sha256"],
        "intent_sha256": base_qualification[
            "base_qualification_intent_sha256"],
    }, label="frozen-base format qualification")

    failure = _verify_self_hashed_evidence(
        files["failure_evidence"]["path"],
        field="failure_evidence_sha256",
        expected_self_sha256=original["failure_evidence_sha256"],
        expected_file_sha256=original["failure_evidence_file_sha256"])
    task_output = _read_exact_text(
        files["task_output"]["path"], original["task_output_file_sha256"])
    if (files["task_output"]["sha256"]
               != original["task_output_file_sha256"]
            or files["failure_evidence"]["sha256"]
               != original["failure_evidence_file_sha256"]):
        raise PermissionError("Parent recovery raw failure output differs")
    exact_error = "AttributeError: 'dict' object has no attribute 'optimizer_updates'"
    if (failure.get("version")
            != "research4-v7-parent-finalization-failure-evidence-v1"
            or failure.get("source_commit") != original["source_commit"]
            or str(failure.get("owner_run_id")) != original["owner_run_id"]
            or str(failure.get("gpu_task_run_id"))
               != original["gpu_task_run_id"]
            or failure.get("error") != exact_error
            or failure.get("source_file")
               != "src/databricks_continuity_partition.py"
            or failure.get("line_numbers") != [427, 432, 433]
            or failure.get("task_output_file_sha256")
               != files["task_output"]["sha256"]
            or exact_error not in task_output):
        raise PermissionError("Parent recovery failure evidence differs")
    _verify_self_hashed_evidence(
        files["child_audit"]["path"], field="audit_sha256",
        expected_self_sha256=original["child_audit_sha256"],
        expected_file_sha256=original["child_audit_file_sha256"])
    _verify_self_hashed_evidence(
        files["mlflow_gold_audit"]["path"], field="audit_sha256",
        expected_self_sha256=original["mlflow_gold_audit_sha256"],
        expected_file_sha256=original["mlflow_gold_audit_file_sha256"])

    run_root = Path(VOLUME) / "runs" / intent["identity"]
    output = run_root / "output"
    terminal = _read(run_root / "TERMINAL.json")
    records_path = run_root / "records.jsonl"
    rows = _jsonl(records_path)
    core_hashes = {
        records_path: original["records_file_sha256"],
        run_root / "TERMINAL.json": original["terminal_file_sha256"],
        output / SUCCESS_MANIFEST: original["child_manifest_file_sha256"],
        output / "TRAINING_RECEIPT.json":
            original["training_receipt_file_sha256"],
        output / "RANK_0_RECEIPT.json": original["rank_receipt_file_sha256"],
        output / "DATABRICKS_EVIDENCE.json": original["evidence_file_sha256"],
        output / "checkpoint" / "delta1.pt": original["checkpoint_file_sha256"],
        output / "optimizer_rank0.pt": original["optimizer_file_sha256"],
    }
    if (len(rows) != original["records_count"]
            or any(_file_sha256(path) != expected
                   for path, expected in core_hashes.items())
            or terminal.get("exit_code") != 0
            or terminal.get("operator_stop_sent") is not False
            or terminal.get("ceiling_kill_sent") is not False
            or terminal.get("child_exit_confirmed") is not True):
        raise PermissionError("Parent recovery durable child bytes differ")
    child = _replay_completed_child(
        run_root, intent=intent, plan_dict=plan_dict,
        exact_plan=exact_plan, rows=rows)
    if (child["receipt"].get("receipt_sha256")
            != original["training_receipt_sha256"]
            or child["rank_receipt"].get("receipt_sha256")
               != original["training_receipt_sha256"]
            or child["evidence_sha256"] != original["evidence_sha256"]
            or child["journal"]["final"]["final_sha256"]
               != original["journal_final_sha256"]
            or child["checkpoint_tree_sha256"]
               != original["checkpoint_tree_sha256"]):
        raise PermissionError("Parent recovery child semantic evidence differs")

    lock_root = Path(VOLUME) / "locks"
    consumed_path = lock_root / "consumed" / f"{intent['identity']}.json"
    closeout_path = (lock_root / "closeouts" / intent["identity"] /
                     f"{hashlib.sha256(original['owner_run_id'].encode()).hexdigest()}.json")
    active_path = lock_root / "scientific.lock"
    consumed = _read(consumed_path); closeout = _read(closeout_path)
    admission_rows = spark.table(GOLD_ADMISSION_TABLE).where(
        (F.col("identity") == intent["identity"])
        & (F.col("intent_sha256") == intent["intent_sha256"])).collect()
    owner_state = _verify_finalizer_owner(
        consumed, admission_rows, owner_run_id=original["owner_run_id"],
        identity=intent["identity"], intent_sha256=intent["intent_sha256"],
        active=(_read(active_path) if active_path.exists() else None),
        closeout=closeout)
    if (owner_state != "after_cleanup"
            or active_path.exists()
            or closeout.get("receipt_sha256")
               != original["closeout_receipt_sha256"]):
        raise PermissionError("Parent recovery release state differs")

    client = mlflow.tracking.MlflowClient()
    mlflow_run = client.get_run(original["mlflow_run_id"])
    experiment = client.get_experiment(mlflow_run.info.experiment_id)
    owner_tags = {
        "mlflow.runName": intent["identity"],
        "research4.parent_owned_run": "true",
        "research4.owner_run_id": original["owner_run_id"],
        "research4.intent_sha256": intent["intent_sha256"],
        "research4.partition_identity": intent["identity"],
        "research4.plan_sha256": intent["plan_sha256"],
    }
    gold_rows = spark.table(GOLD_RUN_TABLE).where(
        (F.col("partition_identity") == intent["identity"])
        & (F.col("intent_sha256") == intent["intent_sha256"])).collect()
    if (str(mlflow_run.info.experiment_id)
            != original["mlflow_experiment_id"]
            or experiment.name != original["mlflow_experiment_path"]
            or mlflow_run.info.status != "RUNNING"
            or any(mlflow_run.data.tags.get(key) != value
                   for key, value in owner_tags.items())
            or mlflow_run.data.params or mlflow_run.data.metrics
            or client.list_artifacts(original["mlflow_run_id"])
            or gold_rows):
        raise PermissionError("Parent recovery MLflow or Gold prestate differs")

    authority_sha = authority["authority_sha256"]
    recovery_root = _parent_recovery_root()
    claim = {
        "version": PARENT_RECOVERY_CLAIM_VERSION,
        "authority_sha256": authority_sha,
        "identity": intent["identity"],
        "original_owner_run_id": original["owner_run_id"],
        "recovery_execution_run_id": str(recovery_execution_run_id),
        "recovery_job_id": str(recovery_job_id),
        "recovery_task_key": recovery_task_key,
        "cpu_only": True, "model_loads": 0, "model_forwards": 0,
        "backward_passes": 0, "optimizer_updates": 0,
    }
    claim["claim_sha256"] = digest(claim)
    _atomic_json(recovery_root / "EXECUTION_CLAIM.json", claim)
    published = _publish_parent_evidence(
        run_root=run_root, intent=intent, plan_dict=plan_dict,
        exact_plan=exact_plan, rows=rows, model_receipt=model_receipt,
        base_qualification=base_qualification, primary_authority=None,
        parent_checkpoint=None, owner_run_id=original["owner_run_id"],
        mlflow_run_id=original["mlflow_run_id"])
    return {"status": "parent_evidence_recovered",
            "identity": intent["identity"],
            "authority_sha256": authority_sha,
            "recovery_root": str(recovery_root),
            "original_owner_run_id": original["owner_run_id"],
            "recovery_execution_run_id": str(recovery_execution_run_id),
            "mlflow_run_id": original["mlflow_run_id"],
            "parent_finalization_sha256": published["parent_finalization"][
                "parent_finalization_sha256"],
            "parent_manifest_sha256": published["parent_manifest"][
                "manifest_sha256"],
            "model_loads": 0, "model_forwards": 0,
            "backward_passes": 0, "optimizer_updates": 0}


def publish_parent_recovery_result(authority_path, recovery, finalization):
    """Publish deterministic recovery provenance outside scientific output."""
    authority = _read(authority_path)
    authority_payload = dict(authority)
    authority_embedded = authority_payload.pop("authority_sha256", None)
    if (authority.get("version") != PARENT_RECOVERY_AUTHORITY_VERSION
            or authority_embedded != digest(authority_payload)
            or recovery.get("authority_sha256")
            != authority.get("authority_sha256")
            or recovery.get("status") != "parent_evidence_recovered"
            or finalization.get("status") != "complete"
            or finalization.get("identity") != recovery.get("identity")
            or finalization.get("mlflow_run_id")
               != recovery.get("mlflow_run_id")
            or recovery.get("recovery_root")
               != str(_parent_recovery_root())):
        raise PermissionError("Parent recovery result binding differs")
    result = {
        "version": PARENT_RECOVERY_RESULT_VERSION,
        "status": "complete", "authority_sha256": authority["authority_sha256"],
        "identity": recovery["identity"],
        "original_owner_run_id": recovery["original_owner_run_id"],
        "recovery_execution_run_id": recovery["recovery_execution_run_id"],
        "mlflow_run_id": recovery["mlflow_run_id"],
        "parent_finalization_sha256": recovery["parent_finalization_sha256"],
        "parent_manifest_sha256": recovery["parent_manifest_sha256"],
        "receipt_sha256": finalization["receipt_sha256"],
        "journal_final_sha256": finalization["journal_final_sha256"],
        "model_loads": 0, "model_forwards": 0,
        "backward_passes": 0, "optimizer_updates": 0,
    }
    result["result_sha256"] = digest(result)
    path = _parent_recovery_root() / "RECOVERY_RESULT.json"
    _atomic_json(path, result)
    return _read(path)


def run_partition(*, spark, intent_path, plan_path, admission_path=None,
                  parent_checkpoint=None, qualification_intent_path=None,
                  qualification_output=None, qualification_receipt_path=None,
                  base_qualification_receipt_path=None,
                   code_root, job_run_id, task_started_monotonic=None,
                   job_started_epoch=None):
    task_started = (time.monotonic() if task_started_monotonic is None
                    else float(task_started_monotonic))

    def whole_job_elapsed():
        local = time.monotonic() - task_started
        if job_started_epoch is None:
            return local
        return max(local, time.time() - float(job_started_epoch))
    intent = validate_intent(_read(intent_path)); plan = _read(plan_path)
    if intent["plan"] != plan:
        raise PermissionError("Intent does not bind the exact partition plan")
    if not base_qualification_receipt_path:
        raise PermissionError("Training requires the frozen-base format gate")
    base_qualification = validate_base_qualification_receipt(
        _read(base_qualification_receipt_path))
    exact_plan = TrainingPlan(**plan).validate()
    source_proof = verify_deployed_source(code_root, intent)
    base_authority = verify_intent_base_qualification_authority(
        intent, base_qualification, code_root)
    lock_root = Path(VOLUME) / "locks"
    verify_reserved_intent(lock_root, intent, owner=str(job_run_id))
    model_receipt = verify_model_receipt_authority(
        MODEL_RECEIPT,
        expected_tree_sha256=intent["model_tree_sha256"],
        expected_receipt_sha256=intent["model_receipt_sha256"])
    from pyspark.sql import functions as F
    base_rows = spark.table(GOLD_RUN_TABLE).where(
        F.col("partition_identity") == base_qualification["identity"]).collect()
    validate_exact_gold_row(base_rows, {
        "partition_identity": base_qualification["identity"],
        "mode": "base_qualification",
        "condition": "frozen_base",
        "stage": "format_schema",
        "seed": 0,
        "plan_sha256": base_qualification["plan_sha256"],
        "source_commit": base_authority["source_commit"],
        "model_tree_sha256": intent["model_tree_sha256"],
        "checkpoint_tree_sha256": intent["model_tree_sha256"],
        "receipt_sha256": base_qualification["receipt_sha256"],
        "evidence_sha256": base_qualification["raw_results_sha256"],
        "intent_sha256": base_qualification[
            "base_qualification_intent_sha256"],
    }, label="frozen-base format qualification")
    authority = None
    if plan["mode"] == "qualification":
        if any((admission_path, parent_checkpoint, qualification_intent_path,
                qualification_output, qualification_receipt_path)):
            raise PermissionError("Qualification forbids primary authority and parent state")
    else:
        if any(not value for value in (admission_path, qualification_intent_path,
                                       qualification_output,
                                       qualification_receipt_path)):
            raise PermissionError("Primary partition requires full qualification authority")
        authority = validate_primary_authority_files(
            intent, admission_path, qualification_intent_path,
            qualification_output, qualification_receipt_path)
        qualification_intent = authority["qualification_intent"]
        qualification_receipt = authority["qualification_receipt"]
        qualification_rows = spark.table(GOLD_RUN_TABLE).where(
            F.col("partition_identity") == qualification_intent["identity"]).collect()
        validate_exact_gold_row(qualification_rows, {
            "partition_identity": qualification_intent["identity"],
            "mode": "qualification", "plan_sha256": qualification_intent["plan_sha256"],
            "source_commit": intent["source_commit"],
            "model_tree_sha256": intent["model_tree_sha256"],
            "checkpoint_tree_sha256": qualification_receipt[
                "qualification_checkpoint_tree_sha256"],
            "receipt_sha256": qualification_receipt[
                "qualification_rank_receipt_sha256"],
            "evidence_sha256": qualification_receipt[
                "qualification_evidence_sha256"],
            "intent_sha256": qualification_intent["intent_sha256"],
        }, label="qualification")
        if plan["stage"] == "delta1" and parent_checkpoint:
            raise PermissionError("Delta1 must start from the frozen base")
        if plan["stage"] != "delta1" and not parent_checkpoint:
            raise PermissionError("Delta2/Delta3 require the exact predecessor")
        parent_metadata = (None if not parent_checkpoint else
                           _read(Path(parent_checkpoint) / "adapter_config.json"))
        validate_sequential_parent(
            exact_plan, parent_metadata, source_sha256=intent["source_sha256"])
        predecessor = expected_predecessor_binding(plan)
        if predecessor is not None:
            parent_tree = directory_tree_sha256(parent_checkpoint)
            predecessor_rows = spark.table(GOLD_RUN_TABLE).where(
                F.col("partition_identity") == predecessor["partition_identity"]).collect()
            predecessor_row = validate_exact_gold_row(predecessor_rows, {
                **predecessor, "source_commit": intent["source_commit"],
                "model_tree_sha256": intent["model_tree_sha256"],
                "checkpoint_tree_sha256": parent_tree,
                "qualification_receipt_sha256": qualification_receipt["receipt_sha256"],
            }, label="sequential predecessor")
            if (not isinstance(predecessor_row.get("admission_sha256"), str)
                    or len(predecessor_row["admission_sha256"]) != 64):
                raise PermissionError("Predecessor Gold admission lineage is absent")
    identity = intent["identity"]
    run_root = Path(VOLUME) / "runs" / identity
    condition = plan["condition"]
    kind = condition
    selected = spark.table(SILVER_TABLE).where(
        (F.col("source_commit") == intent["source_commit"])
        & (F.col("record_kind") == kind) & (F.col("split") == "train")
        & (F.col("admitted_for_training") == F.lit(True)))
    selected_rows = selected.select(
        "dataset_tree_sha256", "record_id", "record_kind", "condition",
        "split", "family_id", "danger_level", "record_sha256", "record_json"
    ).orderBy("record_id").collect()
    if ({row.dataset_tree_sha256 for row in selected_rows}
            != {intent["dataset_tree_sha256"]}):
        raise PermissionError("Silver rows differ from the admitted dataset release")
    selected_binding = [{"record_id": row.record_id,
                         "record_sha256": row.record_sha256}
                        for row in selected_rows]
    if (len(selected_binding) != intent["silver_record_count"]
            or digest(selected_binding) != intent["silver_records_sha256"]):
        raise PermissionError("Exact ordered Silver record selection differs")
    rows = []
    for row in selected_rows:
        if (hashlib.sha256(row.record_json.encode("utf-8")).hexdigest()
                != row.record_sha256):
            raise PermissionError("Silver record payload differs from its admitted hash")
        rows.append(validate_policy_silver_payload(
            row.record_json, record_id=row.record_id,
            expected_condition=condition, expected_split=row.split,
            expected_family_id=row.family_id,
            expected_danger_level=row.danger_level))
        if (row.record_kind != condition or row.condition != condition
                or row.split != "train"):
            raise PermissionError("Silver training metadata semantics differ")
    if not rows:
        raise PermissionError("Silver table has no exact admitted partition records")
    if (intent["semantics_record_count"] != 0
            or intent["semantics_records_sha256"] != digest([])):
        raise PermissionError("Danger semantics may not enter optimizer exposure")
    if run_root.exists():
        raise FileExistsError("Databricks partition output is immutable")
    run_root.mkdir(parents=True, exist_ok=False)
    local_stage_root = Path(tempfile.mkdtemp(
        prefix=f"research4_{identity}_", dir=tempfile.gettempdir()))
    records_path = run_root / "records.jsonl"
    local_records_path = local_stage_root / "records.jsonl"
    local_records_path.write_text(
        "".join(canonical(row) + "\n" for row in rows), encoding="utf-8")
    publish_immutable_bytes(
        local_records_path.read_bytes(), records_path,
        local_stage_dir=local_stage_root / "publication_stage")
    local_model, local_model_receipt, cache_metrics = _stage_local_model(
        intent["model_root"], MODEL_RECEIPT, identity,
        intent["model_tree_sha256"], intent["model_receipt_sha256"])
    try:
        import mlflow
        experiment = mlflow.set_experiment(intent["mlflow_experiment_path"])
        mlflow_client = mlflow.tracking.MlflowClient()
        active = mlflow_client.create_run(
            experiment.experiment_id,
            tags={"mlflow.runName": identity,
                  "research4.parent_owned_run": "true",
                  "research4.owner_run_id": str(job_run_id),
                  "research4.intent_sha256": intent["intent_sha256"],
                  "research4.partition_identity": identity,
                  "research4.plan_sha256": intent["plan_sha256"]})
        mlflow_run_id = active.info.run_id
    except BaseException:
        shutil.rmtree(local_model.parent, ignore_errors=True)
        raise
    local_output = local_stage_root / "output"
    durable_output = run_root / "output"
    command = [
        sys.executable, "-m",
        "src.databricks_continuity_partition_worker",
        "--intent", str(intent_path), "--plan", str(plan_path),
        "--records", str(local_records_path), "--output", str(local_output),
        "--durable-output", str(durable_output),
        "--model-receipt", str(local_model_receipt),
    ]
    if admission_path:
        command += ["--admission", str(admission_path)]
    if parent_checkpoint:
        command += ["--parent-checkpoint", str(parent_checkpoint)]
    env = dict(os.environ)
    env.update({"RESEARCH4_EXECUTION_PLATFORM": "databricks",
                "RESEARCH4_MODEL_ROOT": str(local_model),
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "PYTORCH_CUDA_ALLOC_CONF": DATABRICKS_A10_ALLOCATOR_CONF})
    try:
        try:
            prework_seconds = whole_job_elapsed()
            remaining_stop = intent["wall_clock_stop_seconds"] - prework_seconds
            remaining_ceiling = intent["wall_clock_ceiling_seconds"] - prework_seconds
            if remaining_stop <= 0 or remaining_ceiling - remaining_stop < 300:
                raise TimeoutError(
                    "Databricks prework exhausted the reviewed child/export envelope")
            result = _bounded_run(
                command, cwd=Path(code_root).resolve(), env=env,
                stop_seconds=remaining_stop,
                ceiling_seconds=remaining_ceiling)
        finally:
            shutil.rmtree(local_model.parent, ignore_errors=True)
        result["local_model_cache"] = cache_metrics
        result["prework_seconds"] = prework_seconds
        result["whole_task_elapsed_seconds"] = whole_job_elapsed()
        _atomic_json(run_root / "TERMINAL.json", result)
        if (result["exit_code"] != 0 or result["operator_stop_sent"]):
            mlflow_client.set_tag(
                mlflow_run_id, "terminal_status", "stopped_or_failed")
            mlflow_client.set_terminated(mlflow_run_id, status="FAILED")
            raise RuntimeError("Databricks partition did not complete; retry is forbidden")
    except BaseException:
        if "mlflow_run_id" in locals():
            try:
                if mlflow_client.get_run(mlflow_run_id).info.status == "RUNNING":
                    mlflow_client.set_terminated(mlflow_run_id, status="FAILED")
            except BaseException:
                pass
        raise

    def post_run_deadline():
        if whole_job_elapsed() >= intent["wall_clock_ceiling_seconds"]:
            raise TimeoutError(
                "Databricks post-run audit exceeded the reviewed ceiling")

    try:
        return _publish_parent_evidence(
            run_root=run_root, intent=intent, plan_dict=plan,
            exact_plan=exact_plan, rows=rows,
            model_receipt=model_receipt,
            base_qualification=base_qualification,
            primary_authority=authority,
            parent_checkpoint=parent_checkpoint,
            owner_run_id=job_run_id, mlflow_run_id=mlflow_run_id,
            deadline_check=post_run_deadline)
    except BaseException:
        # The child has already exited at this point. Keep post-science
        # publication entirely off the GPU task; an independent CPU closeout
        # can reconcile the still-open parent-owned run from durable evidence.
        raise


def _expected_mlflow_scalars(journal, terminal):
    """Return the exact parent-owned scalar state encoded by the journal."""
    params = {
        key: str(value) for key, value in journal["params"].items()
        if key not in {"version", "params_sha256"} and value is not None
    }
    metrics = {}

    def add_metric(key, value, step):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))):
            raise PermissionError(f"CPU finalizer metric is invalid: {key}")
        metrics.setdefault(key, []).append((int(step), float(value)))

    for update in journal["progress"]:
        step = update["optimizer_update"]
        for key, value in update.items():
            if key in {"version", "intent_sha256", "plan_sha256",
                       "optimizer_update", "progress_record_sha256",
                       "overflow_recovered", "gradient_norm_positive",
                       "fixed_scale_no_growth",
                       "grad_scaler_attempts_sha256"}:
                continue
            add_metric(key, value, step)
    for key in ("mean_gpu_utilization_percent", "tokens_per_second",
                "max_allocated_bytes", "max_reserved_bytes",
                "disk_free_bytes"):
        add_metric(key, journal["final"].get(key), 0)
    cache = terminal.get("local_model_cache", {})
    for key, value in {
        "model_copy_seconds": cache.get("copy_seconds"),
        "model_copy_bytes_per_second": cache.get("copy_bytes_per_second"),
        "local_disk_free_after_copy": cache.get("local_disk_free_after"),
    }.items():
        add_metric(key, value, 0)
    return params, metrics


def _reconcile_mlflow_scalars(client, run_id, journal, terminal, *, write):
    """Idempotently publish and then verify every owned scalar."""
    expected_params, expected_metrics = _expected_mlflow_scalars(
        journal, terminal)
    if write:
        for key, value in expected_params.items():
            client.log_param(run_id, key, value)
        for key, values in expected_metrics.items():
            for step, value in values:
                client.log_metric(run_id, key, value, step=step)
    actual_params = dict(client.get_run(run_id).data.params)
    if actual_params != expected_params:
        raise PermissionError("CPU finalizer MLflow parameters differ")
    for key, expected in expected_metrics.items():
        history = client.get_metric_history(run_id, key)
        actual = [(int(item.step), float(item.value)) for item in history]
        if not actual:
            raise PermissionError(f"CPU finalizer MLflow metric is absent: {key}")
        for actual_step, actual_value in actual:
            if not any(
                    actual_step == expected_step
                    and math.isclose(actual_value, expected_value,
                                     rel_tol=1e-12, abs_tol=1e-12)
                    for expected_step, expected_value in expected):
                raise PermissionError(
                    f"CPU finalizer MLflow metric differs: {key}")
        for expected_step, expected_value in expected:
            if not any(
                    actual_step == expected_step
                    and math.isclose(actual_value, expected_value,
                                     rel_tol=1e-12, abs_tol=1e-12)
                    for actual_step, actual_value in actual):
                raise PermissionError(
                    f"CPU finalizer MLflow metric is incomplete: {key}")
    return {"params": expected_params, "metrics": expected_metrics}


_EXPECTED_MLFLOW_ARTIFACTS = {
    "TRAINING_RECEIPT.json", "RANK_0_RECEIPT.json",
    "DATABRICKS_EVIDENCE.json", "MLFLOW_PARAMS.json",
    "MLFLOW_PROGRESS.jsonl", "MLFLOW_FINAL.json",
    "PARENT_FINALIZATION.json", "CHILD_OUTPUT_MANIFEST.json",
    "PARENT_OUTPUT_MANIFEST.json", "adapter_config.json", "TERMINAL.json",
}


def _verify_completed_mlflow_state(client, run_id, journal, terminal,
                                   owner_tags):
    """Read back every remote completion field before claiming success."""
    run = client.get_run(run_id)
    if run.info.status != "FINISHED":
        raise PermissionError("CPU finalizer MLflow run did not finish")
    _reconcile_mlflow_scalars(
        client, run_id, journal, terminal, write=False)
    required_tags = {**owner_tags, "terminal_status": "complete",
                     "cpu_artifact_finalize": "true"}
    if any(run.data.tags.get(key) != value
           for key, value in required_tags.items()):
        raise PermissionError("Completed CPU finalizer tags differ")
    artifacts = {Path(item.path).name for item in client.list_artifacts(
        run_id, "verified")}
    if artifacts != _EXPECTED_MLFLOW_ARTIFACTS:
        raise PermissionError("Completed CPU finalizer artifacts differ")
    return run


def _verify_finalizer_owner(consumed, admission_rows, *, owner_run_id,
                            identity, intent_sha256, active=None,
                            closeout=None):
    expected = {
        "owner": str(owner_run_id), "identity": identity,
        "intent_sha256": intent_sha256,
        "status": "reserved_after_gpu_ready_before_model",
    }
    if consumed != expected:
        raise PermissionError("CPU finalizer reservation owner differs")
    if any(str(row.job_run_id) != str(owner_run_id) for row in admission_rows):
        raise PermissionError("CPU finalizer admission owner differs")
    statuses = [row.status for row in admission_rows]
    def closeout_is_exact():
        if not isinstance(closeout, dict):
            return False
        payload = dict(closeout); embedded = payload.pop("receipt_sha256", None)
        return (payload.get("version")
                == "research4-databricks-active-reservation-closeout-v1"
                and payload.get("owner") == str(owner_run_id)
                and payload.get("identity") == identity
                and payload.get("intent_sha256") == intent_sha256
                and payload.get("consumed_identity_retained") is True
                and payload.get("reservation_sha256") == digest(consumed)
                and embedded == digest(payload))
    if statuses == ["consumed_after_gpu_ready_before_model"]:
        if active == expected and closeout is None:
            return "before_gpu_closeout"
        if active is None and closeout_is_exact():
            return "after_gpu_closeout_before_cleanup"
        raise PermissionError("CPU finalizer active reservation differs")
    if sorted(statuses) != sorted([
            "consumed_after_gpu_ready_before_model",
            "active_released_consumed_identity_retained"]):
        raise PermissionError("CPU finalizer admission state differs")
    if active is not None or not closeout_is_exact():
        raise PermissionError("CPU finalizer released reservation differs")
    return "after_cleanup"


def _validate_parent_finalization(run_root, *, intent, terminal, receipt,
                                  rank_receipt, evidence, journal,
                                  owner_run_id):
    output = Path(run_root) / "output"
    checkpoint = output / "checkpoint"
    metadata_path = checkpoint / "adapter_config.json"
    metadata = _read(metadata_path)
    metadata_payload = dict(metadata)
    metadata_embedded = metadata_payload.pop("receipt_sha256", None)
    receipt_base = dict(receipt)
    receipt_metadata = receipt_base.pop("checkpoint", None)
    receipt_metadata_sha = receipt_base.pop("checkpoint_metadata_sha256", None)
    if (receipt_base != rank_receipt or receipt_metadata != metadata
            or metadata_embedded != digest(metadata_payload)
            or receipt_metadata_sha != _file_sha256(metadata_path)):
        raise PermissionError(
            "CPU finalizer training receipt/checkpoint differs")
    checkpoint_tree = directory_tree_sha256(checkpoint)
    parent_path = output / "PARENT_FINALIZATION.json"
    parent = _read(parent_path)
    parent_payload = dict(parent)
    embedded = parent_payload.pop("parent_finalization_sha256", None)
    mlflow_run_id = parent.get("mlflow_run_id")
    plan = intent["plan"]
    pending_gold_row = parent.get("pending_gold_row")
    admission_sha = (pending_gold_row or {}).get("admission_sha256")
    qualification_sha = (pending_gold_row or {}).get(
        "qualification_receipt_sha256")
    parent_tree = (pending_gold_row or {}).get(
        "parent_checkpoint_tree_sha256")
    if plan["mode"] == "qualification":
        if (admission_sha is not None
                or qualification_sha
                   != intent["base_qualification_receipt_sha256"]
                or parent_tree is not None):
            raise PermissionError("CPU parent qualification Gold binding differs")
    elif (not re.fullmatch(r"[0-9a-f]{64}", str(admission_sha or ""))
          or not re.fullmatch(r"[0-9a-f]{64}", str(qualification_sha or ""))
          or ((plan["stage"] == "delta1") != (parent_tree is None))
          or (parent_tree is not None
              and not re.fullmatch(r"[0-9a-f]{64}", str(parent_tree)))):
        raise PermissionError("CPU parent primary Gold binding differs")
    expected_gold = {
        "partition_identity": intent["identity"], "mode": plan["mode"],
        "condition": plan["condition"], "stage": plan["stage"],
        "seed": plan["seed"], "status": "pending_mlflow_finalize",
        "mlflow_run_id": mlflow_run_id,
        "plan_sha256": intent["plan_sha256"],
        "source_commit": intent["source_commit"],
        "model_tree_sha256": intent["model_tree_sha256"],
        "receipt_sha256": rank_receipt["receipt_sha256"],
        "intent_sha256": intent["intent_sha256"],
        "evidence_sha256": evidence["evidence_sha256"],
        "admission_sha256": admission_sha,
        "qualification_receipt_sha256": qualification_sha,
        "receipt_json": canonical(receipt),
        "optimizer_updates": plan["optimizer_updates"],
        "checkpoint_tree_sha256": checkpoint_tree,
        "parent_checkpoint_tree_sha256": parent_tree,
        "receipt_replayed": True,
    }
    if pending_gold_row != expected_gold:
        raise PermissionError("CPU parent pending Gold binding differs")
    expected = {
        "version": "research4-parent-mlflow-finalization-v4",
        "identity": intent["identity"],
        "owner_run_id": str(owner_run_id),
        "intent_sha256": intent["intent_sha256"],
        "plan_sha256": intent["plan_sha256"],
        "mlflow_run_id": mlflow_run_id,
        "terminal_file_sha256": _file_sha256(Path(run_root) / "TERMINAL.json"),
        "journal_final_sha256": journal["final"]["final_sha256"],
        "training_receipt_file_sha256": _file_sha256(
            output / "TRAINING_RECEIPT.json"),
        "rank_receipt_file_sha256": _file_sha256(
            output / "RANK_0_RECEIPT.json"),
        "evidence_file_sha256": _file_sha256(
            output / "DATABRICKS_EVIDENCE.json"),
        "checkpoint_metadata_file_sha256": _file_sha256(metadata_path),
        "checkpoint_tree_sha256": checkpoint_tree,
        "receipt_sha256": receipt["receipt_sha256"],
        "rank_receipt_sha256": rank_receipt["receipt_sha256"],
        "evidence_sha256": evidence["evidence_sha256"],
        "pending_gold_row": expected_gold,
    }
    if (not isinstance(mlflow_run_id, str) or not mlflow_run_id
            or parent_payload != expected or embedded != digest(parent_payload)):
        raise PermissionError("CPU parent finalization binding differs")
    return parent


def _ensure_pending_gold_row(read_rows, append_row, parent_finalization):
    """Idempotently create or verify the CPU-owned pending Gold record."""
    expected = parent_finalization["pending_gold_row"]
    rows = list(read_rows())
    if not rows:
        pending = {**expected, "completed_at": datetime.now(timezone.utc)}
        append_row(pending)
        rows = list(read_rows())
    if len(rows) != 1:
        raise PermissionError("CPU finalizer expected exactly one Gold row")
    row = rows[0]
    gold = (row.asDict(recursive=True)
            if hasattr(row, "asDict") else dict(row))
    if (gold.get("status") not in {"pending_mlflow_finalize", "complete"}
            or any(gold.get(key) != value for key, value in expected.items()
                   if key != "status")):
        raise PermissionError("CPU finalizer pending Gold row differs")
    return gold

def finalize_partition_mlflow(*, spark, intent_path, plan_path, code_root,
                              owner_run_id):
    """CPU-only artifact/MLflow/Gold finalization after verified GPU success."""
    from pyspark.sql import functions as F
    import mlflow
    intent = validate_intent(_read(intent_path))
    plan_dict = _read(plan_path)
    plan = TrainingPlan(**plan_dict).validate()
    verify_deployed_source(code_root, intent)
    identity = intent["identity"]
    run_root = Path(VOLUME) / "runs" / identity
    output = run_root / "output"
    if (not (output / "PARENT_FINALIZATION.json").is_file()
            or not (run_root / PARENT_SUCCESS_MANIFEST).is_file()):
        return {"status": "no_complete_parent_evidence",
                "identity": identity, "owner_run_id": str(owner_run_id)}
    _verify_durable_completion(
        run_root, identity=identity,
        intent_sha256=intent["intent_sha256"],
        plan_sha256=intent["plan_sha256"],
        optimizer_updates=plan.optimizer_updates)
    terminal = _read(run_root / "TERMINAL.json")
    receipt = _read(output / "TRAINING_RECEIPT.json")
    rank_receipt = _read(output / "RANK_0_RECEIPT.json")
    evidence = _read(output / "DATABRICKS_EVIDENCE.json")
    evidence_payload = dict(evidence)
    if (evidence_payload.pop("evidence_sha256", None)
            != digest(evidence_payload)):
        raise PermissionError("CPU finalizer evidence self-hash differs")
    if (terminal.get("exit_code") != 0
            or terminal.get("operator_stop_sent") is not False
            or terminal.get("ceiling_kill_sent") is not False
            or terminal.get("child_exit_confirmed") is not True
            or receipt.get("status") != "complete"
            or receipt.get("successful_optimizer_updates")
               != plan.optimizer_updates):
        raise PermissionError("CPU finalizer lacks successful GPU evidence")
    verify_rank_receipt(
        rank_receipt, expected_plan_sha256=intent["plan_sha256"],
        expected_global_journal_sha256=intent["global_journal_sha256"],
        expected_plan=plan,
        expected_global_journal=adjacent_pair_schedule(
            _jsonl(run_root / "records.jsonl"), plan))
    journal = validate_mlflow_journal(
        output, intent=intent, plan=plan, receipt=receipt,
        rank_receipt=rank_receipt, evidence=evidence)
    parent_finalization = _validate_parent_finalization(
        run_root, intent=intent, terminal=terminal, receipt=receipt,
        rank_receipt=rank_receipt, evidence=evidence, journal=journal,
        owner_run_id=owner_run_id)
    lock_root = Path(VOLUME) / "locks"
    consumed_path = lock_root / "consumed" / f"{identity}.json"
    consumed = _read(consumed_path)
    admission_rows = spark.table(GOLD_ADMISSION_TABLE).where(
        (F.col("identity") == identity)
        & (F.col("intent_sha256") == intent["intent_sha256"])).collect()
    active_path = lock_root / "scientific.lock"
    closeout_path = (lock_root / "closeouts" / identity /
                     f"{hashlib.sha256(str(owner_run_id).encode()).hexdigest()}.json")
    _verify_finalizer_owner(
        consumed, admission_rows, owner_run_id=owner_run_id,
        identity=identity, intent_sha256=intent["intent_sha256"],
        active=(_read(active_path) if active_path.exists() else None),
        closeout=(_read(closeout_path) if closeout_path.exists() else None))
    def read_gold_rows():
        return spark.table(GOLD_RUN_TABLE).where(
            (F.col("partition_identity") == identity)
            & (F.col("intent_sha256") == intent["intent_sha256"])).collect()

    def append_gold_row(value):
        spark.createDataFrame(
            [value], schema=spark.table(GOLD_RUN_TABLE).schema
        ).write.mode("append").saveAsTable(GOLD_RUN_TABLE)

    gold = _ensure_pending_gold_row(
        read_gold_rows, append_gold_row, parent_finalization)
    client = mlflow.tracking.MlflowClient()
    mlflow_run_id = gold["mlflow_run_id"]
    if mlflow_run_id != parent_finalization["mlflow_run_id"]:
        raise PermissionError("CPU finalizer MLflow owner binding differs")
    run = client.get_run(mlflow_run_id)
    owner_tags = {
        "mlflow.runName": identity,
        "research4.parent_owned_run": "true",
        "research4.owner_run_id": str(owner_run_id),
        "research4.intent_sha256": intent["intent_sha256"],
        "research4.partition_identity": identity,
        "research4.plan_sha256": intent["plan_sha256"],
    }
    if any(run.data.tags.get(key) != value
           for key, value in owner_tags.items()):
        raise PermissionError("CPU finalizer MLflow run tags differ")
    if run.info.status == "FINISHED":
        if gold["status"] != "complete":
            raise PermissionError("Completed CPU finalizer evidence differs")
        _verify_completed_mlflow_state(
            client, mlflow_run_id, journal, terminal, owner_tags)
        return {"status": "complete", "identity": identity,
                "mlflow_run_id": mlflow_run_id,
                "receipt_sha256": rank_receipt["receipt_sha256"],
                "journal_final_sha256": journal["final"]["final_sha256"],
                "owner_run_id": str(owner_run_id)}
    if run.info.status not in {"RUNNING", "FAILED"}:
        raise PermissionError("CPU finalizer MLflow state differs")
    _reconcile_mlflow_scalars(
        client, mlflow_run_id, journal, terminal, write=True)
    artifacts = [
        output / "TRAINING_RECEIPT.json",
        output / "RANK_0_RECEIPT.json",
        output / "DATABRICKS_EVIDENCE.json",
        output / "MLFLOW_PARAMS.json",
        output / "MLFLOW_PROGRESS.jsonl",
        output / "MLFLOW_FINAL.json",
        output / "PARENT_FINALIZATION.json",
        output / SUCCESS_MANIFEST,
        output / "checkpoint" / "adapter_config.json",
        run_root / "TERMINAL.json",
        run_root / PARENT_SUCCESS_MANIFEST,
    ]
    if any(not path.is_file() for path in artifacts):
        raise FileNotFoundError("CPU finalizer artifact is absent")
    for path in artifacts:
        client.log_artifact(mlflow_run_id, str(path), "verified")
    safe_identity = identity.replace("'", "''")
    if gold["status"] == "pending_mlflow_finalize":
        spark.sql(
            f"UPDATE {GOLD_RUN_TABLE} SET status = 'complete' "
            f"WHERE partition_identity = '{safe_identity}' "
            f"AND intent_sha256 = '{intent['intent_sha256']}' "
            "AND status = 'pending_mlflow_finalize'"
        )
    finalized = spark.table(GOLD_RUN_TABLE).where(
        (F.col("partition_identity") == identity)
        & (F.col("intent_sha256") == intent["intent_sha256"])
        & (F.col("status") == "complete")).count()
    if finalized != 1:
        raise PermissionError("CPU finalizer Gold update was not exact")
    client.set_tag(mlflow_run_id, "terminal_status", "complete")
    client.set_tag(mlflow_run_id, "cpu_artifact_finalize", "true")
    client.set_terminated(mlflow_run_id, status="FINISHED")
    _verify_completed_mlflow_state(
        client, mlflow_run_id, journal, terminal, owner_tags)
    return {"status": "complete", "identity": identity,
            "mlflow_run_id": mlflow_run_id,
            "receipt_sha256": rank_receipt["receipt_sha256"],
            "journal_final_sha256": journal["final"]["final_sha256"],
            "owner_run_id": str(owner_run_id)}
