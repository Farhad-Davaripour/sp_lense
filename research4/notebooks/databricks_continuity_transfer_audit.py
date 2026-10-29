# Databricks notebook source
"""Independent CPU replay/audit of the exact Databricks transfer grid."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import sys
import zipfile

for name in ("code_path", "intent_path", "run_id"):
    dbutils.widgets.text(name, "")
values = {name: dbutils.widgets.get(name) for name in (
    "code_path", "intent_path", "run_id")}
if any(not value for value in values.values()):
    raise ValueError("Transfer audit requires exact code, intent and run owner")
raw_intent = json.loads(Path(values["intent_path"]).read_text(encoding="utf-8"))


def _bootstrap_verify_source(root_value, authority):
    root = Path(root_value).resolve(); manifest = authority.get("source_manifest", {})
    payload = dict(manifest); embedded = payload.pop("manifest_sha256", None)
    commit = authority.get("source_commit"); files = manifest.get("files")
    canonical_bytes = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    critical = (
        "src/__init__.py", "src/databricks_simulated_continuity.py",
        "src/databricks_continuity_partition.py",
        "src/continuity_transfer_databricks.py",
        "src/continuity_transfer_evaluation.py",
        "src/continuity_transfer_inference_worker.py",
        "src/continuity_transfer_raw_audit.py",
        "src/qwen9b_ordinal_continuity_lora.py",
    )
    if (not re.fullmatch(r"[0-9a-f]{40}", str(commit or ""))
            or not root.as_posix().endswith(f"/research4/source_releases/{commit}")
            or authority.get("source_sha256") != embedded
            or hashlib.sha256(canonical_bytes).hexdigest() != embedded
            or not isinstance(files, list)):
        raise PermissionError("Independent transfer source authority differs")
    indexed = {row.get("path"): row for row in files if isinstance(row, dict)}
    for logical in critical:
        row = indexed.get(logical); path = root / logical
        if not isinstance(row, dict) or not path.is_file() or path.is_symlink():
            raise PermissionError("Critical transfer source is absent")
        data = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace(
            "\r", "\n").encode("utf-8")
        if (len(data) != row.get("canonical_bytes")
                or hashlib.sha256(data).hexdigest() != row.get("sha256")):
            raise PermissionError("Critical transfer source changed before import")
    return embedded


bootstrap_source_sha256 = _bootstrap_verify_source(values["code_path"], raw_intent)
sys.path.insert(0, values["code_path"])

from src.continuity_transfer_databricks import (
    TRANSFER_GOLD_TABLE, build_databricks_run_binding,
    validate_transfer_intent, verify_transfer_deployed_source,
    verify_transfer_inputs,
)
from src.continuity_transfer_evaluation import canonical, digest, file_sha256
from src.continuity_transfer_raw_audit import audit_raw_run
from src.databricks_simulated_continuity import VOLUME
from pyspark.sql import functions as F

intent = validate_transfer_intent(raw_intent)
source = verify_transfer_deployed_source(values["code_path"], intent)
if source["manifest_sha256"] != bootstrap_source_sha256:
    raise PermissionError("Transfer bootstrap and full source verification differ")
proof = verify_transfer_inputs(intent)
consumed_path = (Path(VOLUME) / "locks" / "consumed"
                 / (intent["identity"] + ".json"))
reservation = json.loads(consumed_path.read_text(encoding="utf-8"))
expected_binding = build_databricks_run_binding(
    intent, job_run_id=values["run_id"], reservation=reservation)
output = Path(intent["output_root"])
actual_binding = json.loads((output / "RUN_BINDING.json").read_text(
    encoding="utf-8"))
if actual_binding != expected_binding:
    raise PermissionError("Worker run binding differs from the consumed reservation")
raw = output / "raw" / "RAW_NATIVE_RECORDS.jsonl"
worker_receipt = output / "raw" / "WORKER_RECEIPT.json"
if (not raw.is_file() or not worker_receipt.is_file()
        or raw.stat().st_size > intent["resource_plan"]["max_raw_export_bytes"]):
    raise PermissionError("Bounded raw transfer evidence is absent or oversized")
transport = output / "TRANSPORT.zip"
if transport.exists():
    raise FileExistsError("Transfer transport is immutable")
with zipfile.ZipFile(transport, "x", compression=zipfile.ZIP_STORED) as archive:
    archive.write(raw, "continuity_transfer/RAW_NATIVE_RECORDS.jsonl")
    archive.write(worker_receipt, "continuity_transfer/WORKER_RECEIPT.json")
audit_path = output / "INDEPENDENT_AUDIT.json"
audit = audit_raw_run(
    raw_records_path=raw, worker_receipt_path=worker_receipt,
    sealed_root=intent["sealed_root"], commitment_path=intent["commitment_path"],
    checkpoint_manifest_path=intent["checkpoint_manifest_path"],
    checkpoint_root=intent["checkpoint_root"],
    unblinding_key_path=intent["unblinding_key_path"],
    expected_source_commit=intent["source_commit"],
    expected_source_package_sha256=intent["source_sha256"],
    expected_checkpoint_package_sha256=intent["checkpoint_package_sha256"],
    expected_run_binding=expected_binding, transport_archive_path=transport,
    expected_resource_plan=intent["resource_plan"],
    output_path=audit_path,
)
mlflow_binding = json.loads((output / "MLFLOW.json").read_text(encoding="utf-8"))
if (mlflow_binding.get("status") != "raw_complete_pending_independent_audit"
        or not mlflow_binding.get("mlflow_run_id")):
    raise PermissionError("Transfer MLflow ownership binding differs")
row = {
    "identity": intent["identity"], "job_run_id": values["run_id"],
    "status": "pending_mlflow_finalize", "source_commit": intent["source_commit"],
    "intent_sha256": intent["intent_sha256"],
    "run_binding_sha256": digest(expected_binding),
    "model_tree_sha256": intent["model_tree_sha256"],
    "base_qualification_receipt_sha256": intent[
        "base_qualification_receipt_sha256"],
    "checkpoint_manifest_sha256": intent["checkpoint_manifest_sha256"],
    "sealed_commitment_sha256": intent["sealed_commitment_sha256"],
    "raw_records_sha256": file_sha256(raw),
    "worker_receipt_sha256": file_sha256(worker_receipt),
    "transport_archive_sha256": file_sha256(transport),
    "audit_sha256": file_sha256(audit_path),
    "criteria_met": audit["criteria_met"], "record_count": audit["record_count"],
    "mlflow_run_id": mlflow_binding["mlflow_run_id"],
    "claim_scope_json": canonical(audit["claim_scope"]),
    "audit_json": canonical(audit), "completed_at": datetime.now(timezone.utc),
}
existing = spark.table(TRANSFER_GOLD_TABLE).where(
    (F.col("identity") == intent["identity"])
    & (F.col("intent_sha256") == intent["intent_sha256"])
    & (F.col("job_run_id") == values["run_id"])
).limit(1).count()
if existing:
    raise PermissionError("Transfer audit Gold identity was already written")
spark.createDataFrame([row], schema=spark.table(TRANSFER_GOLD_TABLE).schema).write.mode(
    "append").saveAsTable(TRANSFER_GOLD_TABLE)

import mlflow
from mlflow.tracking import MlflowClient
mlflow_client = MlflowClient()
mlflow_client.log_dict(
    mlflow_binding["mlflow_run_id"], audit, "independent_audit.json")
mlflow_client.log_metric(
    mlflow_binding["mlflow_run_id"], "audited_record_count",
    audit["record_count"])
mlflow_client.log_metric(
    mlflow_binding["mlflow_run_id"], "criteria_met",
    int(audit["criteria_met"] is True))
mlflow_client.set_tag(
    mlflow_binding["mlflow_run_id"], "terminal_status",
    "audit_complete_pending_gold_finalize")

spark.sql(
    f"UPDATE {TRANSFER_GOLD_TABLE} SET status = 'complete_audited' "
    f"WHERE identity = '{intent['identity']}' "
    f"AND intent_sha256 = '{intent['intent_sha256']}' "
    f"AND job_run_id = '{values['run_id']}' "
    "AND status = 'pending_mlflow_finalize'"
)
finalized = spark.table(TRANSFER_GOLD_TABLE).where(
    (F.col("identity") == intent["identity"])
    & (F.col("intent_sha256") == intent["intent_sha256"])
    & (F.col("job_run_id") == values["run_id"])
    & (F.col("status") == "complete_audited")).count()
if finalized != 1:
    raise PermissionError("Transfer Gold finalization was not exact")
mlflow_client.set_tag(
    mlflow_binding["mlflow_run_id"], "terminal_status", "complete_audited")
mlflow_client.set_terminated(
    mlflow_binding["mlflow_run_id"], status="FINISHED")

dbutils.notebook.exit(json.dumps({
    "status": "complete_audited", "identity": intent["identity"],
    "record_count": audit["record_count"],
    "criteria_met": audit["criteria_met"],
    "audit_sha256": row["audit_sha256"],
}))
