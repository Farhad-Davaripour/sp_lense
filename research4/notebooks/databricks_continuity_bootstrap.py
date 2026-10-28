# Databricks notebook source
import json, sys

dbutils.widgets.text("code_path", "")
code_path = dbutils.widgets.get("code_path")
if not code_path:
    raise ValueError("Exact synchronized code_path is required")
sys.path.insert(0, code_path)

from src.databricks_simulated_continuity import catalog_ddl

for statement in catalog_ddl():
    spark.sql(statement)

dbutils.notebook.exit(json.dumps({"status": "catalog_ready", "ddl_count": len(catalog_ddl())}))
