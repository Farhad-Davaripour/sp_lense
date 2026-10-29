# Databricks notebook source
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, re, sys

dbutils.widgets.text("code_path", "")
dbutils.widgets.text("source_commit", "")
dbutils.widgets.text("model_tree_sha256", "")
dbutils.widgets.text("base_qualification_intent_path", "")
dbutils.widgets.text("base_qualification_receipt_path", "")
dbutils.widgets.text("qualification_receipt_path", "")
code_path = dbutils.widgets.get("code_path")
source_commit = dbutils.widgets.get("source_commit")
model_tree = dbutils.widgets.get("model_tree_sha256")
base_intent_path = dbutils.widgets.get("base_qualification_intent_path")
base_path = dbutils.widgets.get("base_qualification_receipt_path")
qualification_path = dbutils.widgets.get("qualification_receipt_path")
if (not code_path or len(source_commit) != 40 or len(model_tree) != 64
        or not base_intent_path or not base_path or not qualification_path):
    raise ValueError("Exact code/source/model identity is required")
raw_base_intent = json.loads(Path(base_intent_path).read_text(encoding="utf-8"))


def _bootstrap_verify_source(root_value, authority):
    root = Path(root_value).resolve(); manifest = authority.get("source_manifest", {})
    payload = dict(manifest); embedded = payload.pop("manifest_sha256", None)
    commit = authority.get("source_commit"); files = manifest.get("files")
    canonical_bytes = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    critical = (
        "src/__init__.py", "src/databricks_simulated_continuity.py",
        "src/qwen9b_ordinal_continuity_runtime.py",
        "src/qwen9b_ordinal_continuity_training.py",
        "src/qwen9b_ordinal_continuity_lora.py",
    )
    if (not re.fullmatch(r"[0-9a-f]{40}", str(commit or ""))
            or commit != source_commit
            or not root.as_posix().endswith(f"/research4/source_releases/{commit}")
            or authority.get("source_sha256") != embedded
            or hashlib.sha256(canonical_bytes).hexdigest() != embedded
            or not isinstance(files, list)):
        raise PermissionError("Independent Gold-audit source authority differs")
    indexed = {row.get("path"): row for row in files if isinstance(row, dict)}
    for logical in critical:
        row = indexed.get(logical); path = root / logical
        if not isinstance(row, dict) or not path.is_file() or path.is_symlink():
            raise PermissionError("Critical Gold-audit source is absent")
        data = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace(
            "\r", "\n").encode("utf-8")
        if (len(data) != row.get("canonical_bytes")
                or hashlib.sha256(data).hexdigest() != row.get("sha256")):
            raise PermissionError("Critical Gold-audit source changed before import")
    return embedded


bootstrap_source_sha256 = _bootstrap_verify_source(code_path, raw_base_intent)
sys.path.insert(0, code_path)

from src.databricks_simulated_continuity import (
    GOLD_RUN_TABLE, GOLD_SUMMARY_TABLE, build_source_manifest, canonical, gold_summary,
    validate_base_qualification_intent, validate_base_qualification_receipt,
    validate_exact_gold_row, verify_deployed_source)

from pyspark.sql import functions as F
base_receipt = validate_base_qualification_receipt(json.loads(
    Path(base_path).read_text(encoding="utf-8")))
base_intent = validate_base_qualification_intent(raw_base_intent)
source_proof = verify_deployed_source(code_path, base_intent)
if (source_proof["manifest_sha256"] != bootstrap_source_sha256
        or base_receipt["base_qualification_intent_sha256"]
           != base_intent["intent_sha256"]):
    raise PermissionError("Gold audit source/base intent lineage differs")
qualification_receipt = json.loads(Path(
    qualification_path).read_text(encoding="utf-8"))
source_manifest = build_source_manifest(code_path, source_commit)
if (source_manifest["manifest_sha256"] != base_receipt["source_sha256"]
        or qualification_receipt.get("source_sha256")
           != base_receipt["source_sha256"]):
    raise PermissionError("Gold audit source differs from frozen qualification evidence")
selected = spark.table(GOLD_RUN_TABLE).where(
    (F.col("source_commit") == source_commit)
    & (F.col("model_tree_sha256") == model_tree))
rows = [row.asDict(recursive=True) for row in selected.collect()]
base_rows = [row for row in rows if row["mode"] == "base_qualification"]
base_row = validate_exact_gold_row(base_rows, {
    "partition_identity": base_receipt["identity"],
    "mode": "base_qualification", "condition": "frozen_base",
    "stage": "format_schema", "seed": 0,
    "plan_sha256": base_receipt["plan_sha256"],
    "source_commit": source_commit, "model_tree_sha256": model_tree,
    "checkpoint_tree_sha256": model_tree,
    "receipt_sha256": base_receipt["receipt_sha256"],
    "evidence_sha256": base_receipt["raw_results_sha256"],
    "intent_sha256": base_receipt["base_qualification_intent_sha256"],
}, label="frozen-base qualification")
capacity = [row for row in rows if row["mode"] == "qualification"]
if len(capacity) != 1:
    raise PermissionError("Gold audit requires exactly one capacity qualifier")
policies = [row for row in rows if row["mode"] == "primary"]
summary = gold_summary(
    base_receipt, qualification_receipt, capacity[0], policies)
matrix_identity = summary["summary_sha256"]
if spark.table(GOLD_SUMMARY_TABLE).where(
        f"matrix_identity = '{matrix_identity}'").limit(1).count():
    raise PermissionError("This audited matrix summary already exists")
record = [{
    "matrix_identity": matrix_identity, "status": "complete",
    "baseline": "frozen_base",
    "frozen_base_model_sha256": summary["frozen_base_model_sha256"],
    "base_qualification_receipt_sha256": summary[
        "base_qualification_receipt_sha256"],
    "capacity_qualification_receipt_sha256": summary[
        "capacity_qualification_receipt_sha256"],
    "policy_partition_count": 27, "summary_sha256": summary["summary_sha256"],
    "summary_json": canonical(summary), "audited_at": datetime.now(timezone.utc),
}]
spark.createDataFrame(record).write.mode("append").saveAsTable(GOLD_SUMMARY_TABLE)
dbutils.notebook.exit(json.dumps(summary))
