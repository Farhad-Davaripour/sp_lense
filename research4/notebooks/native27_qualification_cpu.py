# Databricks notebook source
import json,os,sys,tempfile,zipfile
from pathlib import Path

os.environ.update(USE_TORCH='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
root=Path(dbutils.widgets.get('code_path'));commit=dbutils.widgets.get('source_commit');run_id=dbutils.widgets.get('run_id')
if root.name!=commit or not root.as_posix().startswith('/Workspace/Shared/research4-semantic-gate/deployed/'):raise ValueError('Unexpected native auditor release')
sys.path.insert(0,str(root))
from src.io_utils import dump,read,sha,verify_release,assert_namespace
from src.runtime import archive
from src.native27_audit import audit

verify_release(root,commit)
folder=Path(dbutils.widgets.get('input_folder'));assert_namespace(folder)
export=read(folder/'EXPORT_RECEIPT.json')
if export['status']!='complete' or sha(folder/'evidence.zip')!=export['archive_sha256']:raise ValueError('Incomplete or changed native export')
namespace=export.get('artifact_namespace','native27_qualification_v1')
if namespace not in ['native27_qualification_v1','native27_qualification_v2']:raise ValueError('Unexpected native artifact namespace')
output=Path('/Volumes/workspace/research4/artifacts')/namespace/'audits'/commit/run_id
output.mkdir(parents=True,exist_ok=False)
with tempfile.TemporaryDirectory(prefix='research4-native-audit-') as temporary:
    local=Path(temporary)
    with zipfile.ZipFile(folder/'evidence.zip') as z:
        if z.testzip() is not None:raise ValueError('Native archive CRC failed')
        for item in z.infolist():
            if not (local/item.filename).resolve().is_relative_to(local.resolve()):raise ValueError('Unsafe native archive member')
        z.extractall(local)
    result=audit(local,root,output/'AUDIT.json')
result.update(status='complete',source_commit=commit,Databricks_run_id=run_id,GPU_allocations=0)
dump(output/'DATABRICKS_RECEIPT.json',result);result['export']=archive(output)
dump(output.parent/(run_id+'_EXPORT_RECEIPT.json'),result)
dbutils.notebook.exit(json.dumps(result))
