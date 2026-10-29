# Databricks notebook source
"""Read-only preflight for the zero-update frozen-base format qualification."""
import json
import hashlib
import re
import sys
from pathlib import Path

for name in ("code_path", "intent_path"):
    dbutils.widgets.text(name, "")
code_path = dbutils.widgets.get("code_path")
intent_path = dbutils.widgets.get("intent_path")
if not code_path or not intent_path:
    raise ValueError("Frozen-base preflight requires exact code and intent paths")
raw_intent = json.loads(Path(intent_path).read_text(encoding="utf-8"))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _bootstrap_verify_source(root_value, authority):
    """Verify the authority/lock implementation before importing it."""
    root = Path(root_value).resolve()
    manifest = authority.get("source_manifest")
    if not isinstance(manifest, dict):
        raise PermissionError("Source bootstrap manifest is absent")
    payload = dict(manifest)
    embedded = payload.pop("manifest_sha256", None)
    commit = authority.get("source_commit")
    files = manifest.get("files")
    critical = (
        "src/__init__.py", "src/action_serialization.py",
        "src/databricks_simulated_continuity.py",
        "src/databricks_continuity_partition.py",
        "src/qwen9b_ordinal_continuity_runtime.py",
        "src/qwen9b_ordinal_continuity_training.py",
        "src/qwen9b_ordinal_continuity_lora.py",
    )
    if (not re.fullmatch(r"[0-9a-f]{40}", str(commit or ""))
            or not root.as_posix().endswith(
                f"/research4/source_releases/{commit}")
            or authority.get("source_sha256") != embedded
            or embedded != hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
            or not isinstance(files, list)):
        raise PermissionError("Independent source bootstrap authority differs")
    by_path = {row.get("path"): row for row in files if isinstance(row, dict)}
    for logical in critical:
        row = by_path.get(logical)
        path = root / logical
        if (not isinstance(row, dict) or not path.is_file() or path.is_symlink()
                or set(row) != {"path", "sha256", "canonical_bytes"}):
            raise PermissionError("Critical source bootstrap file is absent")
        content = path.read_bytes().decode("utf-8")
        data = content.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
        if (len(data) != row.get("canonical_bytes")
                or hashlib.sha256(data).hexdigest() != row.get("sha256")):
            raise PermissionError("Critical source changed before import")
    return embedded


bootstrap_source_sha256 = _bootstrap_verify_source(code_path, raw_intent)
sys.path.insert(0, code_path)

from pyspark.sql import functions as F
from src.databricks_simulated_continuity import (
    MODEL_RECEIPT,
    SILVER_TABLE,
    validate_base_qualification_silver_rows,
    validate_base_qualification_intent,
    verify_deployed_source,
    verify_model_receipt,
)

intent = validate_base_qualification_intent(raw_intent)
source = verify_deployed_source(code_path, intent)
if source["manifest_sha256"] != bootstrap_source_sha256:
    raise PermissionError("Bootstrap and full source verification differ")
model = verify_model_receipt(intent["model_root"], MODEL_RECEIPT)
if (model["tree_sha256"] != intent["model_tree_sha256"]
        or model["receipt_sha256"] != intent["model_receipt_sha256"]):
    raise PermissionError("Base qualification model snapshot differs")
selected = spark.table(SILVER_TABLE).where(
    (F.col("source_commit") == intent["source_commit"])
    & (F.col("record_kind") == "base_qualification")
    & (F.col("split") == "qualification")
    & (F.col("admitted_for_training") == F.lit(False))).select(
        "source_commit", "dataset_tree_sha256", "record_id", "record_kind",
        "condition", "split", "family_id", "danger_level", "record_sha256",
        "record_json", "admitted_for_training", "dataset_release_path",
        "dataset_release_receipt_sha256").orderBy("record_id").collect()
validated = validate_base_qualification_silver_rows(selected, intent)
dbutils.notebook.exit(json.dumps({
    "status": "validated_not_reserved",
    "identity": intent["identity"],
    "source_manifest_sha256": source["manifest_sha256"],
    "decision_count": len(validated),
    "optimizer_updates": 0,
}))
