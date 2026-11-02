# Databricks notebook source
from pathlib import Path
import json,sys
root=Path(dbutils.widgets.get('code_path'))
if not str(root).startswith('/Workspace/Shared/research4-semantic-gate/deployed/'):raise RuntimeError('Not a Research 4 release')
sys.path.insert(0,str(root))
from src.io_utils import assert_namespace,dump,read,sha
from src.runtime import archive
from src.survival_gate_refit import fit
commit=dbutils.widgets.get('source_commit');run_id=dbutils.widgets.get('run_id')
if root.name!=commit:raise RuntimeError('The deployed source path is not bound to the requested commit')
expected={
 'src/survival_gate_refit.py':'d3d96ac3518d224cc0acca653c8bf029b0944ca489e653c0889fece3dabd490a',
 'src/survival_fit.py':'aed00a9a3832d7874a183b35c6b0d97100d78f72bfc6fd23c17b342c96dbed38',
 'src/io_utils.py':'89a579a088708725b0bd7291754d3532014a1255d6a8fa2ebcb94800266f3f47',
 'config/survival_gate_refit.json':'da9c7512232d33b520c6dd71545fa6372c4659d6cf3af17e4c92df86b60723a7'}
if any(sha(root/name)!=value for name,value in expected.items()):raise RuntimeError('Critical deployed gate-refit source hash mismatch')
features=Path(dbutils.widgets.get('features'));rows=Path(dbutils.widgets.get('rows'));sequences=Path(dbutils.widgets.get('sequences'));checkpoint=Path(dbutils.widgets.get('input_checkpoint'))
for path in [features,rows,sequences,checkpoint]:assert_namespace(path)
config=read(root/'config/survival_gate_refit.json');output=Path('/Volumes/workspace/research4/artifacts/survival_gate_refit')/commit/run_id;assert_namespace(output)
result=fit(features,rows,sequences,checkpoint,output,config,execution='databricks');result.update(source_commit=commit,Databricks_run_id=run_id);dump(output/'DATABRICKS_RECEIPT.json',result);result['export']=archive(output);dump(output.parent/(run_id+'_EXPORT_RECEIPT.json'),result);dbutils.notebook.exit(json.dumps(result))
