# Databricks notebook source
from pathlib import Path
import json,sys

root=Path(dbutils.widgets.get('code_path'))
commit=dbutils.widgets.get('source_commit')
run_id=dbutils.widgets.get('run_id')
mode=dbutils.widgets.get('mode')
input_harness=dbutils.widgets.get('input_harness')
if root.name!=commit or not root.as_posix().startswith('/Workspace/Shared/research4-semantic-gate/deployed/'):
    raise ValueError('Unexpected discovery release')
sys.path.insert(0,str(root))
from src.discovery_runtime import run
from src.io_utils import dump
from src.runtime import archive

output=Path('/Volumes/workspace/research4/artifacts/discovery_redteam_v1')/mode/commit/run_id
result=run(root,output,mode,commit,input_harness or None)
result.update(Databricks_run_id=run_id)
dump(output/'DATABRICKS_RECEIPT.json',result)
result['export']=archive(output)
dump(output.parent/(run_id+'_EXPORT_RECEIPT.json'),result)
if result['status']!='complete':raise RuntimeError(json.dumps(result))
dbutils.notebook.exit(json.dumps(result))
