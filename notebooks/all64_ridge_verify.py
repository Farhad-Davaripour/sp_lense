# Databricks notebook source
import json
import sys
from pathlib import Path

sys.path.insert(0, dbutils.widgets.get('code_path'))
from src.all64_ridge import summarize

admission = json.loads(Path(dbutils.widgets.get('admission')).read_bytes())
dbutils.notebook.exit(json.dumps(summarize(admission['output'])))
