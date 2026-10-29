# Databricks notebook source
"""Publish the CPU-only content-addressed transfer-evaluation package."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import sys

for name in ("code_path", "plan_path"):
    dbutils.widgets.text(name, "")
code_path = dbutils.widgets.get("code_path")
plan_path = dbutils.widgets.get("plan_path")
if not code_path or not plan_path:
    raise ValueError("Transfer packaging requires exact code and plan paths")
raw_plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))


def _bootstrap_verify_source(root_value, authority):
    root = Path(root_value).resolve(); manifest = authority.get("source_manifest", {})
    payload = dict(manifest); embedded = payload.pop("manifest_sha256", None)
    commit = authority.get("source_commit"); files = manifest.get("files")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    critical = (
        "src/__init__.py", "src/databricks_simulated_continuity.py",
        "src/continuity_transfer_packaging.py",
        "src/continuity_transfer_evaluation.py",
        "src/qwen9b_ordinal_continuity_controller.py",
        "src/qwen9b_ordinal_continuity_lora.py",
        "src/qwen9b_ordinal_continuity_runtime.py",
        "src/qwen9b_ordinal_continuity_training.py",
    )
    if (not re.fullmatch(r"[0-9a-f]{40}", str(commit or ""))
            or not root.as_posix().endswith(f"/research4/source_releases/{commit}")
            or authority.get("source_sha256") != embedded
            or hashlib.sha256(encoded).hexdigest() != embedded
            or not isinstance(files, list)):
        raise PermissionError("Independent packaging source authority differs")
    indexed = {row.get("path"): row for row in files if isinstance(row, dict)}
    for logical in critical:
        row = indexed.get(logical); path = root / logical
        if (not isinstance(row, dict) or not path.is_file() or path.is_symlink()
                or set(row) != {"path", "sha256", "canonical_bytes"}):
            raise PermissionError("Critical packaging source is absent")
        data = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace(
            "\r", "\n").encode("utf-8")
        if (len(data) != row.get("canonical_bytes")
                or hashlib.sha256(data).hexdigest() != row.get("sha256")):
            raise PermissionError("Critical packaging source changed before import")
    return embedded


bootstrap_source_sha256 = _bootstrap_verify_source(code_path, raw_plan)
sys.path.insert(0, code_path)

from pyspark.sql import functions as F
from src.continuity_transfer_packaging import (
    TRANSFER_PACKAGE_GOLD_TABLE, publish_transfer_package,
    validate_packaging_plan,
)
from src.databricks_simulated_continuity import (
    GOLD_RUN_TABLE, GOLD_SUMMARY_TABLE, SILVER_TABLE, canonical,
    verify_deployed_source,
)
from src.qwen9b_ordinal_continuity_runtime import file_sha256

plan = validate_packaging_plan(raw_plan)
source = verify_deployed_source(code_path, plan)
if source["manifest_sha256"] != bootstrap_source_sha256:
    raise PermissionError("Packaging bootstrap and full source verification differ")
base_receipt = json.loads(Path(
    plan["base_qualification_receipt_path"]).read_text(encoding="utf-8"))
capacity_receipt = json.loads(Path(
    plan["capacity_qualification_receipt_path"]).read_text(encoding="utf-8"))
gold_rows = [row.asDict(recursive=True) for row in spark.table(
    GOLD_RUN_TABLE).where(
        (F.col("source_commit") == plan["source_commit"])
        & (F.col("model_tree_sha256") == plan["model_tree_sha256"])
    ).collect()]
summary_rows = [row.asDict(recursive=True) for row in spark.table(
    GOLD_SUMMARY_TABLE).collect()]
silver = spark.table(SILVER_TABLE).where(
    (F.col("source_commit") == plan["source_commit"])
    & (F.col("admitted_for_training") == F.lit(True))
).select("record_id", "record_sha256", "record_json").orderBy("record_id").collect()
training_rows = []
for row in silver:
    if hashlib.sha256(row.record_json.encode("utf-8")).hexdigest() != row.record_sha256:
        raise PermissionError("Silver training row bytes differ from their admitted hash")
    value = json.loads(row.record_json)
    if value.get("record_id") != row.record_id:
        raise PermissionError("Silver training row identity differs")
    training_rows.append(value)

receipt = publish_transfer_package(
    plan=plan, base_receipt=base_receipt,
    capacity_receipt=capacity_receipt, gold_rows=gold_rows,
    summary_rows=summary_rows, training_rows=training_rows)
row = {
    "package_identity": receipt["package_content_sha256"],
    "status": receipt["status"], "source_commit": receipt["source_commit"],
    "model_tree_sha256": receipt["model_tree_sha256"],
    "gold_matrix_summary_sha256": receipt["gold_matrix_summary_sha256"],
    "package_content_sha256": receipt["package_content_sha256"],
    "package_root": receipt["package_root"],
    "checkpoint_package_sha256": receipt["checkpoint_package_sha256"],
    "checkpoint_manifest_sha256": receipt["checkpoint_manifest_sha256"],
    "unblinding_key_path": receipt["unblinding_key_path"],
    "unblinding_key_sha256": receipt["unblinding_key_sha256"],
    "sealed_commitment_sha256": receipt["sealed_commitment_sha256"],
    "training_separation_receipt_sha256": receipt[
        "training_separation_receipt_sha256"],
    "packaging_receipt_sha256": receipt["receipt_sha256"],
    "package_files_sha256": receipt["package_files_sha256"],
    "receipt_json": canonical(receipt), "completed_at": datetime.now(timezone.utc),
}
selected = spark.table(TRANSFER_PACKAGE_GOLD_TABLE).where(
    F.col("package_identity") == row["package_identity"])
existing = [item.asDict(recursive=True) for item in selected.collect()]
stable_fields = tuple(key for key in row if key != "completed_at")
if existing:
    if len(existing) != 1 or any(existing[0].get(key) != row[key]
                                 for key in stable_fields):
        raise PermissionError("Existing Gold transfer package row differs")
else:
    spark.createDataFrame(
        [row], schema=spark.table(TRANSFER_PACKAGE_GOLD_TABLE).schema
    ).write.mode("append").saveAsTable(TRANSFER_PACKAGE_GOLD_TABLE)

dbutils.notebook.exit(json.dumps({
    "status": "complete_cpu_packaged", "gpu_runs_started": 0,
    "package_content_sha256": receipt["package_content_sha256"],
    "package_root": receipt["package_root"],
    "packaging_receipt_sha256": receipt["receipt_sha256"],
}))
