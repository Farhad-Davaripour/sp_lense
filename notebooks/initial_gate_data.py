# Databricks notebook source
# MAGIC %pip install PyYAML==6.0.3 databricks-sdk==0.153.0

# COMMAND ----------
from pathlib import Path
import os
import sys
_here=Path.cwd()
_root=next((p for p in (_here,*_here.parents) if (p/'src/config.py').is_file()),None)
if _root is None:
    raise RuntimeError('Run from the deployed Research 3 workspace')
sys.path.insert(0,str(_root))
for name,key in (('catalog','DATABRICKS_CATALOG'),('schema','DATABRICKS_SCHEMA'),('volume','DATABRICKS_VOLUME'),
                 ('warehouse_id','DATABRICKS_WAREHOUSE_ID')):
    os.environ[key]=dbutils.widgets.get(name)

# COMMAND ----------
from src.initial_gate_data import main
result=main(['--admission',dbutils.widgets.get('admission'),'--execute'])

# COMMAND ----------
import json
dbutils.notebook.exit(json.dumps(result))
