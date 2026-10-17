# Databricks notebook source
from pathlib import Path
import sys,json
root=Path(dbutils.widgets.get('code_path'))
if not str(root).startswith('/Workspace/Shared/research4-semantic-gate/deployed/'):raise RuntimeError('Not a Research 4 deployment')
sys.path.insert(0,str(root))
from src.audit_delta import run
result=run(root,dbutils.widgets.get('run_id'),dbutils.widgets.get('source_commit'),dbutils.widgets.get('pilot_source_commit'),dbutils.widgets.get('pilot_run_id'),spark)
if not result['audit_pass']:raise RuntimeError(json.dumps(result))
dbutils.notebook.exit(json.dumps(result))
