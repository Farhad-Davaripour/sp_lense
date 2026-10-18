# Databricks notebook source
# MAGIC %pip install transformers==5.19.0 PyYAML==6.0.3

# COMMAND ----------
import sys
sys.path.insert(0,dbutils.widgets.get('code_path'))
import os
for widget,key in [('catalog','DATABRICKS_CATALOG'),('schema','DATABRICKS_SCHEMA'),('volume','DATABRICKS_VOLUME')]:
    os.environ[key]=dbutils.widgets.get(widget)
from src.matched_steering import prepare_models
result=prepare_models(dbutils.widgets.get('admission'))
import json
dbutils.notebook.exit(json.dumps(result))
