"""Bounded one-use Databricks partition launcher; never retries."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
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
    canonical, digest, expected_predecessor_binding, validate_exact_gold_row,
    validate_base_qualification_receipt, validate_intent,
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


    try:
        verify_manifest(
            durable_output, manifest_name=SUCCESS_MANIFEST,
            expected_status="complete",
            required_update_count=plan.optimizer_updates,
            expected_metadata=completion_metadata(
                identity=intent["identity"],
                intent_sha256=intent["intent_sha256"],
                plan_sha256=intent["plan_sha256"],
                executed_optimizer_step_count=plan.optimizer_updates,
                durable_published_update_count=plan.optimizer_updates))
        receipt_path = run_root / "output" / "TRAINING_RECEIPT.json"
        rank_receipt_path = run_root / "output" / "RANK_0_RECEIPT.json"
        evidence_path = run_root / "output" / "DATABRICKS_EVIDENCE.json"
        receipt = _read(receipt_path); rank_receipt = _read(rank_receipt_path)
        evidence = _read(evidence_path)
        evidence_payload = dict(evidence)
        evidence_sha256 = evidence_payload.pop("evidence_sha256", None)
        if evidence_sha256 != digest(evidence_payload):
            raise PermissionError("Databricks evidence self-hash differs")
        if (receipt.get("status") != "complete"
                or receipt.get("successful_optimizer_updates") != plan["optimizer_updates"]):
            raise PermissionError("Completed Databricks receipt differs from exact plan")
        journal = validate_mlflow_journal(
            run_root / "output", intent=intent,
            plan=TrainingPlan(**plan).validate(), receipt=receipt,
            rank_receipt=rank_receipt, evidence=evidence)
        verify_rank_receipt(
            rank_receipt, expected_plan_sha256=intent["plan_sha256"],
            expected_global_journal_sha256=intent["global_journal_sha256"],
            expected_plan=exact_plan,
            expected_global_journal=adjacent_pair_schedule(rows, exact_plan))
        checkpoint_tree = directory_tree_sha256(run_root / "output" / "checkpoint")
        checkpoint_metadata_path = (run_root / "output" / "checkpoint" /
                                    "adapter_config.json")
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
            raise PermissionError(
                "Training receipt/checkpoint differs from the rank receipt")
        parent_tree = (None if not parent_checkpoint else
                       directory_tree_sha256(parent_checkpoint))
        pending_gold_row = {
            "partition_identity": identity, "mode": plan["mode"],
            "condition": condition, "stage": plan["stage"], "seed": plan["seed"],
            "status": "pending_mlflow_finalize", "mlflow_run_id": mlflow_run_id,
            "plan_sha256": intent["plan_sha256"],
            "source_commit": intent["source_commit"],
            "model_tree_sha256": model_receipt["tree_sha256"],
            "receipt_sha256": rank_receipt["receipt_sha256"],
            "intent_sha256": intent["intent_sha256"],
            "evidence_sha256": evidence_sha256,
            "admission_sha256": (None if authority is None else
                                  authority["admission"]["admission_sha256"]),
            "qualification_receipt_sha256": (
                base_qualification["receipt_sha256"] if authority is None else
                authority["qualification_receipt"]["receipt_sha256"]),
            "receipt_json": canonical(receipt),
            "optimizer_updates": plan["optimizer_updates"],
            "checkpoint_tree_sha256": checkpoint_tree,
            "parent_checkpoint_tree_sha256": parent_tree,
            "receipt_replayed": True,
        }
        if whole_job_elapsed() >= intent["wall_clock_ceiling_seconds"]:
            raise TimeoutError("Databricks post-run audit exceeded the reviewed ceiling")
        parent_finalization = {
            "version": "research4-parent-mlflow-finalization-v4",
            "identity": identity,
            "owner_run_id": str(job_run_id),
            "intent_sha256": intent["intent_sha256"],
            "plan_sha256": intent["plan_sha256"],
            "mlflow_run_id": mlflow_run_id,
            "terminal_file_sha256": _file_sha256(run_root / "TERMINAL.json"),
            "journal_final_sha256": journal["final"]["final_sha256"],
            "training_receipt_file_sha256": _file_sha256(receipt_path),
            "rank_receipt_file_sha256": _file_sha256(rank_receipt_path),
            "evidence_file_sha256": _file_sha256(evidence_path),
            "checkpoint_metadata_file_sha256": _file_sha256(
                checkpoint_metadata_path),
            "checkpoint_tree_sha256": checkpoint_tree,
            "receipt_sha256": receipt["receipt_sha256"],
            "rank_receipt_sha256": rank_receipt["receipt_sha256"],
            "evidence_sha256": evidence_sha256,
            "pending_gold_row": pending_gold_row,
        }
        parent_finalization["parent_finalization_sha256"] = digest(
            parent_finalization)
        _atomic_json(
            run_root / "output" / "PARENT_FINALIZATION.json",
            parent_finalization)
        _parent_manifest(
            run_root, identity=identity,
            intent_sha256=intent["intent_sha256"],
            plan_sha256=intent["plan_sha256"])
        # Deliberately perform no post-science network calls on GPU. The
        # authenticated CPU finalizer replays this verified journal, publishes
        # all scalar/artifact data, finalizes Gold, then marks MLflow FINISHED.
        return {"status": "pending_cpu_finalize", "identity": identity,
                "mlflow_run_id": mlflow_run_id,
                "receipt_sha256": rank_receipt["receipt_sha256"]}
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
