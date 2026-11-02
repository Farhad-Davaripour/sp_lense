# Databricks notebook source
from pathlib import Path
import json,sys
root=Path(dbutils.widgets.get('code_path'))
if not str(root).startswith('/Workspace/Shared/research4-semantic-gate/deployed/'):raise RuntimeError('Not a Research 4 deployment')
if 'src' in sys.modules and Path(sys.modules['src'].__file__).resolve().parent!=root.resolve()/'src':raise RuntimeError('An unrelated source package is already loaded')
sys.path.insert(0,str(root))
from src.recovery_runtime import run
result=run(root,dbutils.widgets.get('run_id'),dbutils.widgets.get('source_commit'),dbutils.widgets.get('mode'),dbutils.widgets.get('probe_run_id'),spark)
if result['status']!='complete':raise RuntimeError(json.dumps(result))
dbutils.notebook.exit(json.dumps(result))
