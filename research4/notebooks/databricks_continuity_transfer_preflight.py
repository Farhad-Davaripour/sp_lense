# Databricks notebook source
"""Read-only preflight for one exact continuity transfer evaluation."""
import hashlib
import json
import re
import sys
from pathlib import Path

for name in ("code_path", "intent_path"):
    dbutils.widgets.text(name, "")
code_path = dbutils.widgets.get("code_path")
intent_path = dbutils.widgets.get("intent_path")
if not code_path or not intent_path:
    raise ValueError("Transfer preflight requires exact code and intent paths")
raw_intent = json.loads(Path(intent_path).read_text(encoding="utf-8"))


def _bootstrap_verify_source(root_value, authority):
    root = Path(root_value).resolve(); manifest = authority.get("source_manifest", {})
    payload = dict(manifest); embedded = payload.pop("manifest_sha256", None)
    commit = authority.get("source_commit"); files = manifest.get("files")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False).encode("utf-8")
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
            or hashlib.sha256(canonical).hexdigest() != embedded
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


bootstrap_source_sha256 = _bootstrap_verify_source(code_path, raw_intent)
sys.path.insert(0, code_path)

from pyspark.sql import functions as F
from src.databricks_continuity_partition import validate_exact_gold_row
from src.databricks_simulated_continuity import GOLD_RUN_TABLE
from src.continuity_transfer_databricks import (
    verify_transfer_deployed_source, verify_transfer_inputs,
    validate_transfer_intent,
)

intent = validate_transfer_intent(raw_intent)
source = verify_transfer_deployed_source(code_path, intent)
if source["manifest_sha256"] != bootstrap_source_sha256:
    raise PermissionError("Transfer bootstrap and full source verification differ")
proof = verify_transfer_inputs(intent)
base = proof["base_qualification"]
rows = spark.table(GOLD_RUN_TABLE).where(
    F.col("partition_identity") == base["identity"]).collect()
validate_exact_gold_row(rows, {
    "partition_identity": base["identity"], "mode": "base_qualification",
    "condition": "frozen_base", "stage": "format_schema", "seed": 0,
    "plan_sha256": base["plan_sha256"],
    "source_commit": intent["source_commit"],
    "model_tree_sha256": intent["model_tree_sha256"],
    "checkpoint_tree_sha256": intent["model_tree_sha256"],
    "receipt_sha256": base["receipt_sha256"],
    "evidence_sha256": base["raw_results_sha256"],
    "intent_sha256": base["base_qualification_intent_sha256"],
}, label="frozen-base qualification before transfer evaluation")

dbutils.notebook.exit(json.dumps({
    "status": "validated_not_reserved", "identity": intent["identity"],
    "source_manifest_sha256": source["manifest_sha256"],
    "checkpoint_count": proof["checkpoint_count"],
    "record_count": intent["resource_plan"]["episode_count"],
    "gpu_runs_started": 0,
}))
