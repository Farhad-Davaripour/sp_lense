# Databricks notebook source
# MAGIC %pip install PyYAML==6.0.3 databricks-sdk==0.153.0

# COMMAND ----------
from pathlib import Path
import os
import sys

_here = Path.cwd()
_root = next((p for p in (_here, *_here.parents) if (p / "src/config.py").is_file()), None)
if _root is None:
    raise RuntimeError("Run this notebook from the cloned Research 3 Git folder")
sys.path.insert(0, str(_root))

def parameter(name, default=""):
    try:
        return dbutils.widgets.get(name)
    except Exception:
        dbutils.widgets.text(name, default)
        return default

for name, key in (("catalog", "DATABRICKS_CATALOG"), ("schema", "DATABRICKS_SCHEMA"),
                  ("volume", "DATABRICKS_VOLUME"), ("warehouse_id", "DATABRICKS_WAREHOUSE_ID")):
    value = parameter(name)
    if value:
        os.environ[key] = value

# COMMAND ----------
from src.persist_data import main
args = ["--execute"] if parameter("execute", "false").lower() == "true" else []
result = main(args)

# COMMAND ----------
import json
dbutils.notebook.exit(json.dumps(result))
