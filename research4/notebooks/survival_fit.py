# Databricks notebook source
from pathlib import Path
import json,sys
root=Path(dbutils.widgets.get('code_path'))
if not str(root).startswith('/Workspace/Shared/research4-semantic-gate/deployed/'):raise RuntimeError('Not a Research 4 release')
sys.path.insert(0,str(root))
from src.io_utils import verify_release,assert_namespace,dump,read
from src.model_store import ensure_model
from src.survival_fit import fit
from src.runtime import archive
commit=dbutils.widgets.get('source_commit');run_id=dbutils.widgets.get('run_id');verify_release(root,commit)
features=Path(dbutils.widgets.get('training_features'));rows=Path(dbutils.widgets.get('training_rows'));sequences=Path(dbutils.widgets.get('training_sequences'))
for path in [features,rows,sequences]:assert_namespace(path)
study=read(root/'config/study.yaml') if (root/'config/study.yaml').suffix=='.json' else __import__('yaml').safe_load((root/'config/study.yaml').read_text())
fit_config=read(root/'config/survival_fit.json');model,_=ensure_model('/Volumes/workspace/research4/artifacts',study['model'])
output=Path('/Volumes/workspace/research4/artifacts/survival_propensity_fit')/commit/run_id;assert_namespace(output)
result=fit(features,rows,sequences,model,output,study,fit_config,execution='databricks')
result.update(source_commit=commit,Databricks_run_id=run_id);dump(output/'DATABRICKS_RECEIPT.json',result)
result['export']=archive(output);dump(output.parent/(run_id+'_EXPORT_RECEIPT.json'),result)
dbutils.notebook.exit(json.dumps(result))
