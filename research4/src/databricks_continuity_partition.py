"""Bounded one-use Databricks partition launcher; never retries."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time

from .databricks_simulated_continuity import (
    GOLD_RUN_TABLE, MODEL_RECEIPT, SILVER_TABLE, VOLUME,
    canonical, digest, expected_predecessor_binding, validate_exact_gold_row,
    validate_intent, validate_primary_authority_files, verify_deployed_source,
    verify_model_receipt, verify_reserved_intent)
from .qwen9b_ordinal_continuity_runtime import directory_tree_sha256
from .qwen9b_ordinal_continuity_training import verify_rank_receipt
from .qwen9b_ordinal_continuity_training import TrainingPlan, validate_sequential_parent


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


def _stage_local_model(volume_root, volume_receipt, identity, expected_tree):
    destination = Path(tempfile.gettempdir()) / "research4_model_cache" / identity
    if destination.exists():
        raise FileExistsError("Per-run local model cache already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = _read(volume_receipt)
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
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    elapsed = time.monotonic() - started
    if proof["tree_sha256"] != expected_tree:
        raise PermissionError("Local model cache differs from volume authority")
    return local_model, local_receipt, {
        "copy_seconds": elapsed,
        "copy_bytes": proof["total_bytes"],
        "copy_bytes_per_second": proof["total_bytes"] / max(elapsed, 1e-9),
        "local_disk_free_before": free_before,
        "local_disk_free_after": shutil.disk_usage(destination.parent).free,
        "tree_sha256": proof["tree_sha256"],
    }


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_partition(*, spark, intent_path, plan_path, admission_path=None,
                  parent_checkpoint=None, qualification_intent_path=None,
                  qualification_output=None, qualification_receipt_path=None,
                  code_root, job_run_id):
    intent = validate_intent(_read(intent_path)); plan = _read(plan_path)
    if intent["plan"] != plan:
        raise PermissionError("Intent does not bind the exact partition plan")
    exact_plan = TrainingPlan(**plan).validate()
    source_proof = verify_deployed_source(code_root, intent)
    lock_root = Path(VOLUME) / "locks"
    verify_reserved_intent(lock_root, intent, owner=str(job_run_id))
    model_receipt = verify_model_receipt(intent["model_root"], MODEL_RECEIPT)
    if (model_receipt["tree_sha256"] != intent["model_tree_sha256"]
            or model_receipt["receipt_sha256"] != intent["model_receipt_sha256"]):
        raise PermissionError("Reviewed intent model snapshot differs")
    from pyspark.sql import functions as F
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
        if plan["stage"] == "skill" and parent_checkpoint:
            raise PermissionError("Canonical skill partition must start fresh")
        if plan["stage"] != "skill" and not parent_checkpoint:
            raise PermissionError("Policy partition requires its exact predecessor")
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
    kind = "neutral_skill" if plan["stage"] == "skill" else condition
    selected = spark.table(SILVER_TABLE).where(
        (F.col("source_commit") == intent["source_commit"])
        & (F.col("record_kind") == kind) & (F.col("split") == "train")
        & (F.col("admitted_for_training") == F.lit(True)))
    selected_rows = selected.select(
        "dataset_tree_sha256", "record_id", "record_sha256", "record_json"
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
    rows = [json.loads(row.record_json) for row in selected_rows]
    if not rows:
        raise PermissionError("Silver table has no exact admitted partition records")
    semantics_path = None
    semantic_rows = []
    if plan["stage"] != "skill":
        semantics = spark.table(SILVER_TABLE).where(
            (F.col("source_commit") == intent["source_commit"])
            & (F.col("record_kind") == "danger_semantics")
            & (F.col("split") == "train")
            & (F.col("admitted_for_training") == F.lit(True)))
        selected_semantics = semantics.select(
            "record_id", "record_sha256", "record_json").orderBy("record_id").collect()
        semantic_binding = [{"record_id": row.record_id,
                             "record_sha256": row.record_sha256}
                            for row in selected_semantics]
        if (len(semantic_binding) != intent["semantics_record_count"]
                or digest(semantic_binding) != intent["semantics_records_sha256"]):
            raise PermissionError("Exact common-semantics selection differs")
        semantic_rows = [json.loads(row.record_json) for row in selected_semantics]
        if not semantic_rows:
            raise PermissionError("Common danger-semantics rows are missing from silver")
    if run_root.exists():
        raise FileExistsError("Databricks partition output is immutable")
    run_root.mkdir(parents=True, exist_ok=False)
    records_path = run_root / "records.jsonl"
    records_path.write_text("".join(canonical(row) + "\n" for row in rows),
                            encoding="utf-8")
    if semantic_rows:
        semantics_path = run_root / "semantics.jsonl"
        semantics_path.write_text(
            "".join(canonical(row) + "\n" for row in semantic_rows), encoding="utf-8")
    local_model, local_model_receipt, cache_metrics = _stage_local_model(
        intent["model_root"], MODEL_RECEIPT, identity,
        intent["model_tree_sha256"])
    import mlflow
    mlflow.set_experiment(intent["mlflow_experiment_path"])
    active = mlflow.start_run(run_name=identity)
    mlflow_run_id = active.info.run_id
    command = [
        sys.executable, "-m",
        "src.databricks_continuity_partition_worker",
        "--intent", str(intent_path), "--plan", str(plan_path),
        "--records", str(records_path), "--output", str(run_root / "output"),
        "--model-receipt", str(local_model_receipt),
        "--mlflow-run-id", mlflow_run_id,
    ]
    if semantics_path is not None:
        command += ["--semantics", str(semantics_path)]
    if admission_path:
        command += ["--admission", str(admission_path)]
    if parent_checkpoint:
        command += ["--parent-checkpoint", str(parent_checkpoint)]
    env = dict(os.environ)
    env.update({"RESEARCH4_EXECUTION_PLATFORM": "databricks",
                "RESEARCH4_MODEL_ROOT": str(local_model),
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    try:
        try:
            result = _bounded_run(
                command, cwd=Path(code_root).resolve(), env=env,
                stop_seconds=intent["wall_clock_stop_seconds"],
                ceiling_seconds=intent["wall_clock_ceiling_seconds"])
        finally:
            shutil.rmtree(local_model.parent, ignore_errors=True)
        result["local_model_cache"] = cache_metrics
        (run_root / "TERMINAL.json").write_text(canonical(result) + "\n")
        if (result["exit_code"] != 0 or result["operator_stop_sent"]):
            mlflow.set_tag("terminal_status", "stopped_or_failed")
            mlflow.end_run(status="FAILED")
            raise RuntimeError("Databricks partition did not complete; retry is forbidden")
    except BaseException:
        if mlflow.active_run() is not None:
            mlflow.end_run(status="FAILED")
        raise
    try:
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
        verify_rank_receipt(
            rank_receipt, expected_plan_sha256=intent["plan_sha256"],
            expected_global_journal_sha256=intent["global_journal_sha256"])
        checkpoint_tree = directory_tree_sha256(run_root / "output" / "checkpoint")
        parent_tree = (None if not parent_checkpoint else
                       directory_tree_sha256(parent_checkpoint))
        gold = [{
            "partition_identity": identity, "mode": plan["mode"],
            "condition": condition, "stage": plan["stage"], "seed": plan["seed"],
            "status": "complete", "mlflow_run_id": mlflow_run_id,
            "plan_sha256": intent["plan_sha256"], "source_commit": intent["source_commit"],
            "model_tree_sha256": model_receipt["tree_sha256"],
            "receipt_sha256": rank_receipt["receipt_sha256"],
            "intent_sha256": intent["intent_sha256"],
            "evidence_sha256": evidence_sha256,
            "admission_sha256": (None if authority is None else
                                  authority["admission"]["admission_sha256"]),
            "qualification_receipt_sha256": (None if authority is None else
                                               authority["qualification_receipt"][
                                                   "receipt_sha256"]),
            "receipt_json": canonical(receipt), "optimizer_updates": plan["optimizer_updates"],
            "checkpoint_tree_sha256": checkpoint_tree,
            "parent_checkpoint_tree_sha256": parent_tree,
            "receipt_replayed": True, "completed_at": datetime.now(timezone.utc),
        }]
        spark.createDataFrame(
            gold, schema=spark.table(GOLD_RUN_TABLE).schema
        ).write.mode("append").saveAsTable(GOLD_RUN_TABLE)
        mlflow.set_tag("terminal_status", "complete")
        mlflow.log_metrics({
            "model_copy_seconds": cache_metrics["copy_seconds"],
            "model_copy_bytes_per_second": cache_metrics["copy_bytes_per_second"],
            "local_disk_free_after_copy": cache_metrics["local_disk_free_after"],
        })
        mlflow.log_dict(result, "terminal.json")
        mlflow.end_run(status="FINISHED")
        return {"status": "complete", "identity": identity,
                "mlflow_run_id": mlflow_run_id,
                "receipt_sha256": rank_receipt["receipt_sha256"]}
    except BaseException:
        if mlflow.active_run() is not None:
            mlflow.end_run(status="FAILED")
        raise
