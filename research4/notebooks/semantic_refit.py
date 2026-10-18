# Databricks notebook source
from pathlib import Path
import json,sys,yaml
root=Path(dbutils.widgets.get('code_path'))
if not str(root).startswith('/Workspace/Shared/research4-semantic-gate/deployed/'):raise RuntimeError('Not a Research 4 release')
sys.path.insert(0,str(root))
from src.io_utils import verify_release,assert_namespace,dump
from src.semantic_refit import fit
from src.runtime import archive
commit=dbutils.widgets.get('source_commit');run_id=dbutils.widgets.get('run_id');verify_release(root,commit)
features=Path(dbutils.widgets.get('training_features'));assert_namespace(features)
checkpoint=Path('/Volumes/workspace/research4/artifacts/runs/ac36e91858a93f2624fb5e479c1ff35ad213a46f/859924329424825/pilot/trainables.pt')
output=Path('/Volumes/workspace/research4/artifacts/semantic_readout_refit')/commit/run_id;assert_namespace(output)
config=yaml.safe_load((root/'config/semantic_policy_v3.yaml').read_text())
result=fit(features,checkpoint,output,config,execution='databricks')
result.update(source_commit=commit,Databricks_run_id=run_id)
dump(output/'DATABRICKS_RECEIPT.json',result);result['export']=archive(output);dump(output.parent/(run_id+'_EXPORT_RECEIPT.json'),result)
dbutils.notebook.exit(json.dumps(result))
