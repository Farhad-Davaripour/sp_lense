# Databricks notebook source
from datetime import datetime, timezone
import json, sys

dbutils.widgets.text("code_path", "")
dbutils.widgets.text("source_commit", "")
dbutils.widgets.text("model_tree_sha256", "")
code_path = dbutils.widgets.get("code_path")
source_commit = dbutils.widgets.get("source_commit")
model_tree = dbutils.widgets.get("model_tree_sha256")
if not code_path or len(source_commit) != 40 or len(model_tree) != 64:
    raise ValueError("Exact code/source/model identity is required")
sys.path.insert(0, code_path)

from src.databricks_simulated_continuity import (
    GOLD_RUN_TABLE, GOLD_SUMMARY_TABLE, canonical, gold_summary)

from pyspark.sql import functions as F
selected = spark.table(GOLD_RUN_TABLE).where(
    (F.col("mode") == "primary") & (F.col("source_commit") == source_commit)
    & (F.col("model_tree_sha256") == model_tree))
rows = [row.asDict(recursive=True) for row in selected.collect()]
skill = [row for row in rows if row["stage"] == "skill"]
policies = [row for row in rows if row["stage"] != "skill"]
if len(skill) != 1:
    raise PermissionError("Gold audit requires exactly one canonical skill partition")
summary = gold_summary(skill[0], policies)
matrix_identity = summary["summary_sha256"]
if spark.table(GOLD_SUMMARY_TABLE).where(
        f"matrix_identity = '{matrix_identity}'").limit(1).count():
    raise PermissionError("This audited matrix summary already exists")
record = [{
    "matrix_identity": matrix_identity, "status": "complete",
    "canonical_skill_checkpoint_sha256": summary["canonical_skill_checkpoint_sha256"],
    "policy_partition_count": 27, "summary_sha256": summary["summary_sha256"],
    "summary_json": canonical(summary), "audited_at": datetime.now(timezone.utc),
}]
spark.createDataFrame(record).write.mode("append").saveAsTable(GOLD_SUMMARY_TABLE)
dbutils.notebook.exit(json.dumps(summary))
