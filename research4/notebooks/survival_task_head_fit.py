# Databricks notebook source
from pathlib import Path
import json,sys
root=Path(dbutils.widgets.get('code_path'))
if not str(root).startswith('/Workspace/Shared/research4-semantic-gate/deployed/'):raise RuntimeError('Not a Research 4 release')
sys.path.insert(0,str(root))
from src.io_utils import assert_namespace,dump,read,sha
from src.runtime import archive
from src.survival_task_head_fit import fit
commit=dbutils.widgets.get('source_commit');run_id=dbutils.widgets.get('run_id')
if root.name!=commit:raise RuntimeError('The deployed source path is not bound to the requested commit')
expected={
 'src/survival_task_head_fit.py':'2e9570cdc06ef3f7f9d69c9a06eb1cb6f95a9051c8f71acd4536671461a488a6',
 'src/io_utils.py':'89a579a088708725b0bd7291754d3532014a1255d6a8fa2ebcb94800266f3f47',
 'config/survival_task_head_fit.json':'55e64a345e0716beb4cdfdf39749c4595012d7a3d5ebb1c1bbd40b148578cfe0'}
if any(sha(root/name)!=value for name,value in expected.items()):raise RuntimeError('Critical deployed normal-head source hash mismatch')
features=Path(dbutils.widgets.get('training_features'));rows=Path(dbutils.widgets.get('training_rows'));checkpoint=Path(dbutils.widgets.get('input_checkpoint'))
for path in [features,rows,checkpoint]:assert_namespace(path)
config=read(root/'config/survival_task_head_fit.json');output=Path('/Volumes/workspace/research4/artifacts/survival_task_head_fit')/commit/run_id;assert_namespace(output)
result=fit(features,rows,checkpoint,output,config,execution='databricks');result.update(source_commit=commit,Databricks_run_id=run_id)
dump(output/'DATABRICKS_RECEIPT.json',result);result['export']=archive(output);dump(output.parent/(run_id+'_EXPORT_RECEIPT.json'),result)
dbutils.notebook.exit(json.dumps(result))
