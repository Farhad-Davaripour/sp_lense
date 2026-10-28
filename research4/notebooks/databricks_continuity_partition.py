# Databricks notebook source
from pathlib import Path
import json, sys

for name in ("code_path", "intent_path", "plan_path", "admission_path",
             "parent_checkpoint", "qualification_intent_path",
             "qualification_output", "qualification_receipt_path",
             "expected_mode"):
    dbutils.widgets.text(name, "")
values = {name: dbutils.widgets.get(name) for name in (
    "code_path", "intent_path", "plan_path", "admission_path", "parent_checkpoint",
    "qualification_intent_path", "qualification_output",
    "qualification_receipt_path")}
if not values["code_path"] or not values["intent_path"] or not values["plan_path"]:
    raise ValueError("Exact code, intent and plan paths are required")
sys.path.insert(0, values["code_path"])
intent_preview = json.loads(Path(values["intent_path"]).read_text())
if intent_preview.get("plan", {}).get("mode") != dbutils.widgets.get("expected_mode"):
    raise PermissionError("Job resource does not admit this partition mode")

from src.databricks_continuity_partition import run_partition
from src.databricks_simulated_continuity import VOLUME, close_active_reservation, validate_intent

intent = validate_intent(intent_preview)
try:
    result = run_partition(
        spark=spark, intent_path=values["intent_path"], plan_path=values["plan_path"],
        admission_path=values["admission_path"] or None,
        parent_checkpoint=values["parent_checkpoint"] or None,
        qualification_intent_path=values["qualification_intent_path"] or None,
        qualification_output=values["qualification_output"] or None,
        qualification_receipt_path=values["qualification_receipt_path"] or None,
        code_root=values["code_path"], job_run_id=dbutils.widgets.get("run_id"))
except BaseException:
    close_active_reservation(
        Path(VOLUME) / "locks", intent, dbutils.widgets.get("run_id"),
        "partition_notebook_failed")
    raise
else:
    close_active_reservation(
        Path(VOLUME) / "locks", intent, dbutils.widgets.get("run_id"),
        "partition_notebook_complete")
dbutils.notebook.exit(json.dumps(result))
