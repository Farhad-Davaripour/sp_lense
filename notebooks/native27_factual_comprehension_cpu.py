# Databricks notebook source
import os,sys,json,tempfile,zipfile
from pathlib import Path
os.environ.update(USE_TORCH='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
root=Path(dbutils.widgets.get('code_path'));commit=dbutils.widgets.get('source_commit');run_id=dbutils.widgets.get('run_id')
if root.name!=commit or not root.as_posix().startswith('/Workspace/Shared/research4-semantic-gate/deployed/'):raise ValueError('Unexpected release')
sys.path.insert(0,str(root))
from src.io_utils import read,dump,sha,verify_release,assert_namespace
from src.runtime import archive
from src.factual_comprehension_audit import audit
verify_release(root,commit)
folder=Path(dbutils.widgets.get('input_folder'));assert_namespace(folder)
if not folder.as_posix().startswith('/Volumes/workspace/research4/artifacts/native27_factual_comprehension_v1/'):raise ValueError('Unexpected QA namespace')
export=read(folder/'EXPORT_RECEIPT.json')
if export['status']!='complete' or export['source_commit']!=commit or sha(folder/'evidence.zip')!=export['archive_sha256']:raise ValueError('Incomplete QA export')
output=Path('/Volumes/workspace/research4/artifacts/native27_factual_comprehension_v1/audits')/commit/run_id;output.mkdir(parents=True,exist_ok=False)
with tempfile.TemporaryDirectory(prefix='research4-factual-qa-audit-') as scratch:
    local=Path(scratch)
    with zipfile.ZipFile(folder/'evidence.zip') as z:
        if z.testzip() is not None:raise ValueError('QA CRC differs')
        for item in z.infolist():
            if not (local/item.filename).resolve().is_relative_to(local.resolve()):raise ValueError('Unsafe archive member')
        z.extractall(local)
    result=audit(local,root,output/'AUDIT.json')
result.update(status='complete',source_commit=commit,Databricks_run_id=run_id,GPU_allocations=0,output=output.as_posix())
dump(output/'DATABRICKS_RECEIPT.json',result);result['export']=archive(output);dump(output.parent/(run_id+'_EXPORT_RECEIPT.json'),result)
dbutils.notebook.exit(json.dumps(result))
