# Databricks notebook source
# MAGIC %pip install PyYAML==6.0.3 databricks-sdk==0.153.0 psutil==7.2.2 torch==2.11.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128 transformers==5.19.0 peft==0.21.2 safetensors==0.8.0

# COMMAND ----------
from pathlib import Path
import os
import sys
root=Path(dbutils.widgets.get('code_path'))
sys.path.insert(0,str(root))
for widget,key in [('catalog','DATABRICKS_CATALOG'),('schema','DATABRICKS_SCHEMA'),('volume','DATABRICKS_VOLUME')]:
    os.environ[key]=dbutils.widgets.get(widget)
from src.matched_validation import run_gate
result=run_gate(dbutils.widgets.get('admission'))

# COMMAND ----------
import json
dbutils.notebook.exit(json.dumps({'status':result['status'],'selected':result['selected'],'optimizer_updates':result['optimizer_updates']}))
