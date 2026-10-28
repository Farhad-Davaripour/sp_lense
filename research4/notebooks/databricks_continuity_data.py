# Databricks notebook source
from datetime import datetime, timezone
from pathlib import Path
import json, sys

for name in ("code_path", "source_commit"):
    dbutils.widgets.text(name, "")
code_path, source_commit = dbutils.widgets.get("code_path"), dbutils.widgets.get("source_commit")
if not code_path or len(source_commit) != 40:
    raise ValueError("Exact synchronized code path and source commit are required")
sys.path.insert(0, code_path)

from src.databricks_simulated_continuity import (
    BRONZE_TABLE, MODEL_ROOT, SILVER_TABLE, VOLUME, bronze_rows, silver_rows,
    silver_partition_bindings)
from transformers import AutoTokenizer

generated = code_path + "/simulated_continuity_v1/generated"
bronze = bronze_rows(generated, source_commit)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ROOT, local_files_only=True, trust_remote_code=False, use_fast=True)
silver = silver_rows(generated, source_commit, tokenizer=tokenizer)
bindings = silver_partition_bindings(silver)
now = datetime.now(timezone.utc)
for row in bronze:
    row["ingested_at"] = now
for row in silver:
    row["transformed_at"] = now

if spark.table(BRONZE_TABLE).where(f"source_commit = '{source_commit}'").limit(1).count():
    raise PermissionError("Bronze source commit was already ingested")
if spark.table(SILVER_TABLE).where(f"source_commit = '{source_commit}'").limit(1).count():
    raise PermissionError("Silver source commit was already materialized")
spark.createDataFrame(bronze).write.mode("append").saveAsTable(BRONZE_TABLE)
spark.createDataFrame(silver).write.mode("append").saveAsTable(SILVER_TABLE)
binding_root = Path(VOLUME) / "silver_bindings" / source_commit
binding_root.mkdir(parents=True, exist_ok=False)
for identity, value in bindings.items():
    (binding_root / f"{identity}.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n")
dbutils.notebook.exit(json.dumps({"status": "data_ready", "bronze_rows": len(bronze),
                                  "silver_rows": len(silver),
                                  "partition_bindings": len(bindings)}))
