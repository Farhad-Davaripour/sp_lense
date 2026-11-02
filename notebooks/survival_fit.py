# Databricks notebook source
from pathlib import Path
import json,sys
root=Path(dbutils.widgets.get('code_path'))
if not str(root).startswith('/Workspace/Shared/research4-semantic-gate/deployed/'):raise RuntimeError('Not a Research 4 release')
sys.path.insert(0,str(root))
from src.io_utils import assert_namespace,dump,read,sha
from src.model_store import ensure_model
from src.survival_fit import fit
from src.runtime import archive
commit=dbutils.widgets.get('source_commit');run_id=dbutils.widgets.get('run_id')
if root.name!=commit:raise RuntimeError('The deployed source path is not bound to the requested commit')
expected={
 'src/survival_fit.py':'aed00a9a3832d7874a183b35c6b0d97100d78f72bfc6fd23c17b342c96dbed38',
 'src/survival_actor.py':'64568dfc836d508b2bc98a92c5856aba95aa627d1c549e185055f7e4b3f04506',
 'src/survival_sandbox.py':'e403dbd253a53da18dac252e10c2b706e1a6d0b623342818d0701d4aa58073d8',
 'config/survival_fit.json':'4549b15364882c4176fd7b600fb0521e3f299647a514dbb0f1a02af9cbda154e',
 'config/study.yaml':'e3a063b06f7c115873df87310cd5ee4ab0c49bdc30124cc4b1b6dbc74b2d0667'}
if any(sha(root/name)!=value for name,value in expected.items()):raise RuntimeError('Critical deployed survival source hash mismatch')
features=Path(dbutils.widgets.get('training_features'));rows=Path(dbutils.widgets.get('training_rows'));sequences=Path(dbutils.widgets.get('training_sequences'))
for path in [features,rows,sequences]:assert_namespace(path)
study=read(root/'config/study.yaml') if (root/'config/study.yaml').suffix=='.json' else __import__('yaml').safe_load((root/'config/study.yaml').read_text())
fit_config=read(root/'config/survival_fit.json');model,_=ensure_model('/Volumes/workspace/research4/artifacts',study['model'])
output=Path('/Volumes/workspace/research4/artifacts/survival_propensity_fit')/commit/run_id;assert_namespace(output)
result=fit(features,rows,sequences,model,output,study,fit_config,execution='databricks')
result.update(source_commit=commit,Databricks_run_id=run_id);dump(output/'DATABRICKS_RECEIPT.json',result)
result['export']=archive(output);dump(output.parent/(run_id+'_EXPORT_RECEIPT.json'),result)
dbutils.notebook.exit(json.dumps(result))
