# Databricks notebook source
from datetime import datetime, timezone
from pathlib import Path
import json, os, re, sys, uuid

for name in ("code_path", "source_commit", "dataset_release_path"):
    dbutils.widgets.text(name, "")
code_path = dbutils.widgets.get("code_path")
source_commit = dbutils.widgets.get("source_commit")
dataset_release_path = dbutils.widgets.get("dataset_release_path")
if (not code_path or not re.fullmatch(r"[0-9a-f]{40}", source_commit)
        or Path(code_path).resolve().as_posix() != (
            f"/Workspace/Shared/research4/source_releases/{source_commit}")
        or not dataset_release_path):
    raise ValueError("Exact code, commit and dataset release paths are required")
sys.path.insert(0, code_path)

from src.databricks_simulated_continuity import (
    BRONZE_TABLE, MODEL_ROOT, SILVER_TABLE, VOLUME, bronze_rows, digest,
    silver_rows, silver_partition_bindings, stable_rows_sha256,
    verify_dataset_release)
from transformers import AutoTokenizer
from pyspark.sql import functions as F


def atomic_json(path, value):
    path = Path(path)
    temporary = path.parent / (".pending-" + uuid.uuid4().hex + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def commit_or_verify(table, expected, fields):
    selected = spark.table(table).where(F.col("source_commit") == source_commit)
    actual = [row.asDict(recursive=True) for row in selected.select(*fields).collect()]
    expected_sha = stable_rows_sha256(expected, fields)
    if actual:
        if stable_rows_sha256(actual, fields) != expected_sha:
            raise PermissionError(f"Existing {table} rows differ from this exact release")
        return {"status": "verified_existing", "rows": len(actual),
                "stable_rows_sha256": expected_sha}
    spark.createDataFrame(
        expected, schema=spark.table(table).schema
    ).write.mode("append").saveAsTable(table)
    readback = [row.asDict(recursive=True) for row in spark.table(table).where(
        F.col("source_commit") == source_commit).select(*fields).collect()]
    if (len(readback) != len(expected)
            or stable_rows_sha256(readback, fields) != expected_sha):
        raise PermissionError(f"{table} readback differs after append")
    return {"status": "appended_and_verified", "rows": len(readback),
            "stable_rows_sha256": expected_sha}


release = verify_dataset_release(dataset_release_path)
generated = dataset_release_path
bronze = bronze_rows(generated, source_commit)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ROOT, local_files_only=True, trust_remote_code=False, use_fast=True)
silver = silver_rows(generated, source_commit, tokenizer=tokenizer)
if (sum(row["admitted_for_training"] is True for row in silver) != 6912
        or sum(row["admitted_for_training"] is False for row in silver) != 288):
    raise PermissionError("Silver is not exact 6912 policy + 288 base checks")
bindings = silver_partition_bindings(silver, generated)
now = datetime.now(timezone.utc)
for row in bronze:
    row.update({
        "dataset_release_path": release["dataset_release_path"],
        "dataset_release_receipt_sha256": release["receipt_sha256"],
        "dataset_tree_sha256": release["dataset_tree_sha256"],
        "ingested_at": now,
    })
for row in silver:
    row.update({
        "dataset_release_path": release["dataset_release_path"],
        "dataset_release_receipt_sha256": release["receipt_sha256"],
        "transformed_at": now,
    })

bronze_fields = (
    "source_commit", "source_path", "source_sha256", "row_number",
    "record_kind", "condition", "split", "record_id", "record_json",
    "dataset_release_path", "dataset_release_receipt_sha256",
    "dataset_tree_sha256",
)
silver_fields = (
    "source_commit", "dataset_tree_sha256", "record_id", "record_kind",
    "condition", "split", "family_id", "danger_level", "record_sha256",
    "record_json", "admitted_for_training", "dataset_release_path",
    "dataset_release_receipt_sha256",
)
bronze_commit = commit_or_verify(BRONZE_TABLE, bronze, bronze_fields)
silver_commit = commit_or_verify(SILVER_TABLE, silver, silver_fields)

binding_root = Path(VOLUME) / "silver_bindings" / source_commit
binding_root.mkdir(parents=True, exist_ok=True)
for stale in binding_root.glob(".pending-*.tmp"):
    stale.unlink()
expected_names = {f"{identity}.json" for identity in bindings}
unexpected = {path.name for path in binding_root.glob("*.json")
              if path.name != "COMPLETION.json"} - expected_names
if unexpected:
    raise PermissionError("Silver binding directory contains unexpected files")
for identity, value in bindings.items():
    path = binding_root / f"{identity}.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise PermissionError("Existing Silver binding differs")
    else:
        atomic_json(path, value)

completion = {
    "version": "research4-databricks-data-stage-completion-v2",
    "source_commit": source_commit,
    "dataset_release_path": release["dataset_release_path"],
    "dataset_release_receipt_sha256": release["receipt_sha256"],
    "dataset_tree_sha256": release["dataset_tree_sha256"],
    "bronze": bronze_commit,
    "silver": silver_commit,
    "partition_binding_count": len(bindings),
    "partition_bindings_sha256": digest(bindings),
    "completed_at": now.isoformat(),
}
completion["receipt_sha256"] = digest(completion)
completion_path = binding_root / "COMPLETION.json"
if completion_path.exists():
    prior = json.loads(completion_path.read_text(encoding="utf-8"))
    prior_payload = dict(prior)
    prior_sha256 = prior_payload.pop("receipt_sha256", None)
    if (prior_payload.get("version") != completion["version"]
            or prior_sha256 != digest(prior_payload)):
        raise PermissionError("Existing data-stage completion self-hash differs")
    for key in ("source_commit", "dataset_release_path",
                "dataset_release_receipt_sha256", "dataset_tree_sha256",
                "partition_binding_count", "partition_bindings_sha256"):
        if prior.get(key) != completion[key]:
            raise PermissionError("Existing data-stage completion differs")
    for key, current in (("bronze", bronze_commit), ("silver", silver_commit)):
        previous = prior.get(key, {})
        if (previous.get("rows") != current["rows"]
                or previous.get("stable_rows_sha256")
                   != current["stable_rows_sha256"]
                or previous.get("status") not in {
                    "appended_and_verified", "verified_existing"}):
            raise PermissionError("Existing data-stage table proof differs")
    completion = prior
else:
    atomic_json(completion_path, completion)

dbutils.notebook.exit(json.dumps({
    "status": "data_ready", "bronze_rows": len(bronze),
    "silver_rows": len(silver), "partition_bindings": len(bindings),
    "silver_policy_training_rows": 6912,
    "silver_base_qualification_nontraining_rows": 288,
    "dataset_tree_sha256": release["dataset_tree_sha256"],
    "completion_receipt_sha256": completion["receipt_sha256"],
}))
