# Databricks notebook source
import os
import sys
sys.path.insert(0,dbutils.widgets.get('code_path'))
for widget,key in [('catalog','DATABRICKS_CATALOG'),('schema','DATABRICKS_SCHEMA'),('volume','DATABRICKS_VOLUME')]:
    os.environ[key]=dbutils.widgets.get(widget)
from src.matched_validation import persist_inputs
result=persist_inputs(dbutils.widgets.get('admission'),spark)
import json
dbutils.notebook.exit(json.dumps(result))
