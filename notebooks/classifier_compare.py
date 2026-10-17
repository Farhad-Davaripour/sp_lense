# Databricks notebook source
# MAGIC %pip install PyYAML==6.0.3

# COMMAND ----------
from pathlib import Path
import sys

_here=Path.cwd()
_root=next((p for p in (_here,*_here.parents) if (p/'src/config.py').is_file()),None)
if _root is None:
    raise RuntimeError('Run this notebook from the Research 3 Git folder')
sys.path.insert(0,str(_root))

def parameter(name,default=''):
    try:
        return dbutils.widgets.get(name)
    except Exception:
        dbutils.widgets.text(name,default)
        return default

# COMMAND ----------
from src.classifier_compare import main
args=[]
if parameter('classifier_output'):
    args += ['--classifier-output',parameter('classifier_output')]
elif parameter('admission'):
    args += ['--admission',parameter('admission')]
result=main(args)

# COMMAND ----------
import json
dbutils.notebook.exit(json.dumps(result))
