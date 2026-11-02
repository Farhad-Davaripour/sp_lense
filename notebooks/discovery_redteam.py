# Databricks notebook source
from pathlib import Path
import json,sys

root=Path(dbutils.widgets.get('code_path'))
commit=dbutils.widgets.get('source_commit')
run_id=dbutils.widgets.get('run_id')
mode=dbutils.widgets.get('mode')
input_harness=dbutils.widgets.get('input_harness')
if root.name!=commit or not root.as_posix().startswith('/Workspace/Shared/research4-semantic-gate/deployed/'):
    raise ValueError('Unexpected discovery release')
sys.path.insert(0,str(root))
from src.discovery_runtime import run
from src.io_utils import dump,read,sha,verify_release,assert_namespace
from src.runtime import archive

output=Path('/Volumes/workspace/research4/artifacts/discovery_redteam_v1')/mode/commit/run_id
verify_release(root,commit)
if mode=='diagnose':
    from src.qualification_diagnosis import run as diagnosis_run
    result=diagnosis_run(root,output)
elif mode=='transport':
    from src.discovery_transport import run as transport_run
    result=transport_run(root,output,commit)
elif mode in ['audit','audit_transport']:
    from src.discovery_audit import audit
    folder=Path(input_harness);assert_namespace(folder)
    for name,value in read(folder/'PRECALL_FREEZE.json')['sources'].items():
        if sha(root/name)!=value:raise ValueError('Auditor engine differs from evaluated release')
    output.mkdir(parents=True,exist_ok=False)
    result=audit(folder,output/'AUDIT.json')
    result.update(status='complete',mode='audit',source_commit=commit,audited_folder=str(folder),GPU_allocations=0)
    if mode=='audit_transport':
        from src.discovery_transport import run as transport_run
        transport=transport_run(root,output/'TRANSPORT',commit)
        result={'status':transport['status'],'mode':mode,'source_commit':commit,'legacy_audit':result,'transport_qualification':transport,'model_api_calls':transport['model_api_calls'],'GPU_allocations':0,'optimizer_updates':0,'behavioral_scenarios_rerun':0}
else:
    result=run(root,output,mode,commit,input_harness or None)
result.update(Databricks_run_id=run_id)
dump(output/'DATABRICKS_RECEIPT.json',result)
result['export']=archive(output)
dump(output.parent/(run_id+'_EXPORT_RECEIPT.json'),result)
if result['status']!='complete':raise RuntimeError(json.dumps(result))
dbutils.notebook.exit(json.dumps(result))
