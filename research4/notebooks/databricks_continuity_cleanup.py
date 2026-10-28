# Databricks notebook source
"""ALL_DONE owner-bound release; the one-use identity stays consumed forever."""
from datetime import datetime, timezone
import json, sys
from pathlib import Path

for name in ("code_path", "intent_path", "run_id"):
    dbutils.widgets.text(name, "")
values = {name: dbutils.widgets.get(name) for name in (
    "code_path", "intent_path", "run_id")}
if any(not value for value in values.values()):
    raise ValueError("Cleanup requires exact code, intent and job-run owner")
sys.path.insert(0, values["code_path"])

from pyspark.sql import functions as F
from src.databricks_simulated_continuity import (
    GOLD_ADMISSION_TABLE, VOLUME, close_active_reservation, validate_intent)

intent = validate_intent(json.loads(Path(values["intent_path"]).read_text(
    encoding="utf-8")))
lock_root = Path(VOLUME) / "locks"
consumed = lock_root / "consumed" / (intent["identity"] + ".json")
if not consumed.exists():
    dbutils.notebook.exit(json.dumps({
        "status": "no_reservation_to_release", "identity": intent["identity"]}))
receipt = close_active_reservation(
    lock_root, intent, values["run_id"], "job_all_done_cleanup")
status = "active_released_consumed_identity_retained"
existing = spark.table(GOLD_ADMISSION_TABLE).where(
    (F.col("identity") == intent["identity"])
    & (F.col("intent_sha256") == intent["intent_sha256"])
    & (F.col("job_run_id") == values["run_id"])
    & (F.col("status") == status)).limit(1).count()
if existing == 0:
    spark.createDataFrame([{
        "identity": intent["identity"], "intent_sha256": intent["intent_sha256"],
        "job_run_id": values["run_id"], "status": status,
        "consumed_at": datetime.now(timezone.utc),
    }]).write.mode("append").saveAsTable(GOLD_ADMISSION_TABLE)
dbutils.notebook.exit(json.dumps({
    "status": status, "identity": intent["identity"],
    "closeout_receipt_sha256": receipt["receipt_sha256"]}))
