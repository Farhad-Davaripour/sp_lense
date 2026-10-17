# Databricks notebook source
# MAGIC %pip install PyYAML==6.0.3

# COMMAND ----------
from pathlib import Path
import sys

_here = Path.cwd()
_root = next((p for p in (_here, *_here.parents) if (p / 'src/config.py').is_file()), None)
if _root is None:
    raise RuntimeError('Run from the Research 3 Git folder')
sys.path.insert(0, str(_root))

# COMMAND ----------
from src.research2_inference_report import main
args = []
for name in ('reference_root', 'classifier_output', 'output'):
    args += ['--' + name.replace('_', '-'), dbutils.widgets.get(name)]
result = main(args)

# COMMAND ----------
import json
dbutils.notebook.exit(json.dumps({'status': result['status'], 'JEV_gate_matches': result['JEV_gate_matches']}))
