# Databricks notebook source
import os,sys,json,tempfile,zipfile
from pathlib import Path
os.environ.update(USE_TORCH='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
root=Path(dbutils.widgets.get('code_path'));commit=dbutils.widgets.get('source_commit');run_id=dbutils.widgets.get('run_id');mode=dbutils.widgets.get('mode')
if root.name!=commit or not root.as_posix().startswith('/Workspace/Shared/research4-semantic-gate/deployed/'):raise ValueError('Unexpected release')
sys.path.insert(0,str(root))
from src.io_utils import read,dump,sha,verify_release,assert_namespace
from src.runtime import archive
from src.risk_readout_analysis import fit,audit
from databricks.sdk import WorkspaceClient
verify_release(root,commit)
live=WorkspaceClient().jobs.get_run(run_id=int(run_id))
if not live.start_time or mode not in ['fit','audit']:raise ValueError('Authenticated job required')
folder=Path(dbutils.widgets.get('input_folder'));assert_namespace(folder)
if not str(folder).startswith('/Volumes/workspace/research4/artifacts/native27_risk_readout_v1/'):raise ValueError('Unexpected snapshot namespace')
export=read(folder/'EXPORT_RECEIPT.json')
if export['status']!='complete' or export['source_commit']!=commit or sha(folder/'evidence.zip')!=export['archive_sha256']:raise ValueError('Incomplete snapshot export')
output=Path('/Volumes/workspace/research4/artifacts/native27_risk_readout_v1')/mode/commit/run_id;output.mkdir(parents=True,exist_ok=False)
with tempfile.TemporaryDirectory(prefix='research4-risk-cpu-') as scratch:
    local=Path(scratch)
    with zipfile.ZipFile(folder/'evidence.zip') as z:
        if z.testzip() is not None:raise ValueError('CRC failure')
        for item in z.infolist():
            if not (local/item.filename).resolve().is_relative_to(local.resolve()):raise ValueError('Unsafe archive path')
        z.extractall(local)
    if mode=='fit':result=fit(local,root,output)
    else:
        fitted=Path(dbutils.widgets.get('fit_folder'));assert_namespace(fitted)
        if not fitted.as_posix().startswith('/Volumes/workspace/research4/artifacts/native27_risk_readout_v1/fit/'+commit+'/'):raise ValueError('Unexpected fit artifact')
        for name,value in read(fitted/'MANIFEST.json').items():
            if not (fitted/name).resolve().is_relative_to(fitted.resolve()) or sha(fitted/name)!=value:raise ValueError('Fit manifest differs')
        result=audit(local,fitted,root,output/'AUDIT.json')
result.update(Databricks_run_id=run_id,source_commit=commit,mode=mode,GPU_allocations=0,output=output.as_posix())
dump(output/'DATABRICKS_RECEIPT.json',result);result['export']=archive(output)
dump(output.parent/(run_id+'_EXPORT_RECEIPT.json'),result)
dbutils.notebook.exit(json.dumps(result))
