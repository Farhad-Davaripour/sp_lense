# Databricks notebook source
from pathlib import Path
import json,sys,yaml
root=Path(dbutils.widgets.get('code_path'))
if not str(root).startswith('/Workspace/Shared/research4-semantic-gate/deployed/'):raise RuntimeError('Not a Research 4 release')
sys.path.insert(0,str(root))
from src.io_utils import assert_namespace,dump,read,sha
from src.model_store import ensure_model
from src.runtime import archive
from src.survival_task_refit import fit
commit=dbutils.widgets.get('source_commit');run_id=dbutils.widgets.get('run_id')
if root.name!=commit:raise RuntimeError('The deployed source path is not bound to the requested commit')
expected={
 'src/survival_task_refit.py':'226a6e637d69e77cd26312a901a4ff946330cb9d3843fabed5d4d7a20048f063',
 'src/survival_actor.py':'64568dfc836d508b2bc98a92c5856aba95aa627d1c549e185055f7e4b3f04506',
 'src/actor.py':'52acaab75f3d360c1adcc3260546bddf010c5377f3aca3910da64884519c3280',
 'src/survival_sandbox.py':'6674e175003c326d84cfc32224a62d5c7313c37f899b57bb22693bed7dc031eb',
 'src/io_utils.py':'89a579a088708725b0bd7291754d3532014a1255d6a8fa2ebcb94800266f3f47',
 'src/model_store.py':'3a54a37e28c09099f15c5624a914280592b60fa87c9444170510fa4bce8ee0f1',
 'config/survival_task_refit.json':'0b6f7e4971da91d5fba03972ed3c9cae98a2de55ea31ab9d5bb1c20760823200',
 'config/study.yaml':'e3a063b06f7c115873df87310cd5ee4ab0c49bdc30124cc4b1b6dbc74b2d0667'}
if any(sha(root/name)!=value for name,value in expected.items()):raise RuntimeError('Critical deployed task-refit source hash mismatch')
rows=Path(dbutils.widgets.get('training_rows'));checkpoint=Path(dbutils.widgets.get('input_checkpoint'))
for path in [rows,checkpoint]:assert_namespace(path)
study=yaml.safe_load((root/'config/study.yaml').read_text());config=read(root/'config/survival_task_refit.json');model,_=ensure_model('/Volumes/workspace/research4/artifacts',study['model'])
output=Path('/Volumes/workspace/research4/artifacts/survival_task_refit')/commit/run_id;assert_namespace(output)
result=fit(rows,checkpoint,model,output,study,config,execution='databricks');result.update(source_commit=commit,Databricks_run_id=run_id)
dump(output/'DATABRICKS_RECEIPT.json',result);result['export']=archive(output);dump(output.parent/(run_id+'_EXPORT_RECEIPT.json'),result)
dbutils.notebook.exit(json.dumps(result))
