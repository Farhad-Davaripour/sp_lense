# Databricks notebook source
"""CPU-only MLflow artifact and Gold finalization for a completed partition."""
import json
import re
import sys
from pathlib import Path

for name in ("code_path", "intent_path", "plan_path", "run_id"):
    dbutils.widgets.text(name, "")
values = {name: dbutils.widgets.get(name) for name in (
    "code_path", "intent_path", "plan_path", "run_id")}
if any(not value for value in values.values()):
    raise ValueError("Partition finalizer requires exact source, intent, plan and owner")
if not re.fullmatch(
        r"/Workspace/Shared/research4/source_releases/[0-9a-f]{40}",
        Path(values["code_path"]).resolve().as_posix()):
    raise PermissionError("Production finalizer source path differs")
sys.path.insert(0, values["code_path"])

from src.databricks_continuity_partition import finalize_partition_mlflow

result = finalize_partition_mlflow(
    spark=spark, intent_path=values["intent_path"],
    plan_path=values["plan_path"], code_root=values["code_path"],
    owner_run_id=values["run_id"])
dbutils.notebook.exit(json.dumps(result, sort_keys=True))
