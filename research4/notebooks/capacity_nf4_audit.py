# Databricks notebook source
import hashlib,json,os,sys,tempfile,zipfile
from pathlib import Path

os.environ.update(USE_TORCH='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
root=Path(dbutils.widgets.get('code_path'));commit=dbutils.widgets.get('source_commit')
if root.name!=commit or not root.as_posix().startswith('/Workspace/Shared/research4-semantic-gate/deployed/'):raise ValueError('Unexpected capacity auditor release')
sys.path.insert(0,str(root))
from src.io_utils import dump,read,sha,verify_release,assert_namespace
from src.recognition_capacity_audit import audit
from src.runtime import archive

verify_release(root,commit)
folder=Path(dbutils.widgets.get('capacity_folder'));inputs=Path(dbutils.widgets.get('input_path'));metadata=Path(dbutils.widgets.get('metadata_path'))
for path in [folder,inputs,metadata]:assert_namespace(path)
if sha(inputs/'MANIFEST.json')!=dbutils.widgets.get('input_manifest_sha256'):raise ValueError('Capacity input anchor differs')
for pin in read(inputs/'MODEL_PINS.json'):
    pinned={f['name']:f for f in pin['files']}
    for path in (metadata/pin['tag']).iterdir():
        spec=pinned[path.name];data=path.read_bytes()
        if len(data)!=spec['size']:raise ValueError('Tokenizer metadata size differs')
        if spec['sha256'] and hashlib.sha256(data).hexdigest()!=spec['sha256']:raise ValueError('Tokenizer metadata digest differs')
        if spec.get('git_blob_id') and hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=spec['git_blob_id']:raise ValueError('Tokenizer metadata blob differs')
receipt=read(folder/'EXPORT_RECEIPT.json')
if receipt['status']!='complete' or sha(folder/'evidence.zip')!=receipt['archive_sha256']:raise ValueError('Incomplete or changed capacity export')
output=Path('/Volumes/workspace/research4/artifacts/recognition_capacity_databricks_nf4_v1/audits')/commit/dbutils.widgets.get('run_id')
output.mkdir(parents=True,exist_ok=False)
with tempfile.TemporaryDirectory(prefix='research4-capacity-audit-') as temporary:
    target=Path(temporary)
    with zipfile.ZipFile(folder/'evidence.zip') as bundle:
        if bundle.testzip() is not None:raise ValueError('Capacity archive CRC failed')
        for entry in bundle.infolist():
            path=(target/entry.filename).resolve()
            if not path.is_relative_to(target.resolve()):raise ValueError('Unsafe capacity archive member')
        bundle.extractall(target)
    result=audit(inputs,target,metadata,output/'AUDIT.json')
result.update(status='complete',Databricks_run_id=dbutils.widgets.get('run_id'),GPU_allocations=0,capacity_folder=str(folder),source_commit=commit)
dump(output/'DATABRICKS_RECEIPT.json',result)
result['export']=archive(output)
dump(output.parent/(output.name+'_EXPORT_RECEIPT.json'),result)
dbutils.notebook.exit(json.dumps(result))
