# Databricks notebook source
# MAGIC %pip install PyYAML==6.0.3 psutil==7.2.2 torch==2.11.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128 transformers==5.15.1 safetensors==0.8.0

# COMMAND ----------
from pathlib import Path
import os
import sys

_here = Path.cwd()
_root = next((p for p in (_here, *_here.parents) if (p / 'src/config.py').is_file()), None)
if _root is None:
    raise RuntimeError('Run from the Research 3 Git folder')
sys.path.insert(0, str(_root))
for name, key in (('catalog', 'DATABRICKS_CATALOG'), ('schema', 'DATABRICKS_SCHEMA'),
                  ('volume', 'DATABRICKS_VOLUME')):
    os.environ[key] = dbutils.widgets.get(name)

# COMMAND ----------
from src.research2_inference import main
args = []
for name in ('reference_root', 'classifier_output', 'output', 'submission_utc'):
    args += ['--' + name.replace('_', '-'), dbutils.widgets.get(name)]
result = main(args)

# COMMAND ----------
import json
dbutils.notebook.exit(json.dumps({'status': result['status'], 'models': list(result['models'])}))
