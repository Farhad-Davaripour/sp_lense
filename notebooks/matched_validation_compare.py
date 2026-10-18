# Databricks notebook source
from pathlib import Path
import sys
sys.path.insert(0,str(Path(dbutils.widgets.get('code_path'))))
from src.matched_validation import compare
result=compare(dbutils.widgets.get('admission'))
import json
dbutils.notebook.exit(json.dumps(result))
