# Databricks notebook source
import sys
import json
from pathlib import Path
sys.path.insert(0,dbutils.widgets.get('code_path'))
from src.matched_steering import summarize
admission=json.loads(Path(dbutils.widgets.get('admission')).read_bytes())
result=summarize(admission['output'])
dbutils.notebook.exit(json.dumps(result))
