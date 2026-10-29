# Databricks notebook source
import json, re, sys
from pathlib import Path

dbutils.widgets.text("code_path", "")
code_path = dbutils.widgets.get("code_path")
if (not code_path or not re.fullmatch(
        r"/Workspace/Shared/research4/source_releases/[0-9a-f]{40}",
        Path(code_path).resolve().as_posix())):
    raise ValueError("Exact synchronized code_path is required")
sys.path.insert(0, code_path)

from src.databricks_simulated_continuity import catalog_ddl

for statement in catalog_ddl():
    spark.sql(statement)

dbutils.notebook.exit(json.dumps({"status": "catalog_ready", "ddl_count": len(catalog_ddl())}))
