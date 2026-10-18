# Databricks notebook source
# MAGIC %pip install PyYAML==6.0.3 databricks-sdk==0.153.0 psutil==7.2.2 torch==2.11.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128 transformers==5.19.0 peft==0.21.2 safetensors==0.8.0

# COMMAND ----------
from pathlib import Path
import os
import sys
_here=Path.cwd()
_root=next((p for p in (_here,*_here.parents) if (p/'src/config.py').is_file()),None)
if _root is None:
    raise RuntimeError('Run from the deployed Research 3 workspace')
sys.path.insert(0,str(_root))
for name,key in (('catalog','DATABRICKS_CATALOG'),('schema','DATABRICKS_SCHEMA'),('volume','DATABRICKS_VOLUME')):
    os.environ[key]=dbutils.widgets.get(name)

# COMMAND ----------
from src.initial_gate_training import main
result=main(['--admission',dbutils.widgets.get('admission')])

# COMMAND ----------
import json
dbutils.notebook.exit(json.dumps({'status':result['status'],'development':result['development'],
    'benchmark_executed':result['benchmark_executed'],'optimizer_updates':result['optimizer_updates']}))
