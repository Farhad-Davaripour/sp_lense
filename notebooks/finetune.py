# Databricks notebook source
# MAGIC %pip install PyYAML==6.0.3 databricks-sdk==0.153.0 psutil==7.2.2 transformers==5.19.0 peft==0.21.2 safetensors==0.7.0

# COMMAND ----------
# MAGIC %pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

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
                  ("volume", "DATABRICKS_VOLUME")):
    value = parameter(name)
    if value:
        os.environ[key] = value

# COMMAND ----------
from src.finetune import main
args = ["--engine-args", parameter("engine_args", "[]"), "--max-seconds", parameter("max_seconds", "7200")]
if parameter("resume_2b_admission"):
    args += ["--resume-2b-admission", parameter("resume_2b_admission")]
if parameter("execute", "false").lower() == "true":
    args.append("--execute")
result = main(args)

# COMMAND ----------
import json
dbutils.notebook.exit(json.dumps(result))
