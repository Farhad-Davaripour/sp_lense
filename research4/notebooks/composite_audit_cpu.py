# Databricks notebook source
import os,sys,json,tempfile,hashlib
from pathlib import Path
os.environ.update(USE_TORCH='0',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1')
root=Path(dbutils.widgets.get('code_path'));commit=dbutils.widgets.get('source_commit');run_id=dbutils.widgets.get('run_id')
if root.name!=commit or not root.as_posix().startswith('/Workspace/Shared/research4-semantic-gate/deployed/'):raise ValueError('Unexpected composite audit source')
sys.path.insert(0,str(root))
from databricks.sdk import WorkspaceClient
if not WorkspaceClient().jobs.get_run(run_id=int(run_id)).start_time:raise RuntimeError('Actual Databricks job required')
os.environ['RESEARCH4_SCIENTIFIC_RUNTIME']='databricks-verified'
from src.io_utils import read,dump,sha,verify_release,digest
from src.runtime import archive
from src.composite_store import Store,download_model
from src.composite_audit import evidence,diagnose_detector,audit_trajectory
from src.composite_metrics import summarize
from src.composite_world import Scene
verify_release(root,commit)
for name,value in read(root/'inputs/MANIFEST.json').items():
    if sha(root/'inputs'/name)!=value:raise ValueError('Audit plan input changed')
cfg=read(root/'config/composite_controller_v1.json');plan=read(root/'inputs/AUDIT_PLAN.json');store=Store()
output=Path('/Volumes/workspace/research4/artifacts/composite_controller_v1/audits')/commit/run_id;output.mkdir(parents=True,exist_ok=False)
dump(output/'PRECALL_FREEZE.json',{'source_commit':commit,'audit_plan':plan,'model_forwards':0,'optimizer_updates':0})
failure=None
try:
    if plan['kind']=='detector_diagnosis':
        files=evidence(store,plan['evidence']);data=store.json('data_v1/DATA_RELEASE.json');raw=store.get('data_v1/validation/DETECTION_ROWS.json')
        if hashlib.sha256(raw).hexdigest()!=data['cohorts']['validation']['files']['DETECTION_ROWS.json']:raise ValueError('Detector diagnosis cohort changed')
        result=diagnose_detector(json.loads(files['DETECTOR_VALIDATION.json']),json.loads(raw))
        dump(output/'DETECTOR_DIAGNOSIS.json',result)
    elif plan['kind']=='trajectory_audit':
        records=[];proofs=[];audits=[]
        for descriptor in plan['partitions']:
            files=evidence(store,descriptor);summary=json.loads(files['TRAJECTORY_SUMMARY.json']);proof=json.loads(files['PACKAGE_VERIFICATION.json'])
            if summary['role']!=plan['role'] or not summary['complete'] or proof['package_sha256']!=digest(plan['package']):raise ValueError('Incomplete or mixed package/role partition')
            if not json.loads(files['BYPASS_CHECK.json'])['pass']:raise ValueError('Package OFF bypass not admitted')
            proofs.append(proof)
            records.extend(json.loads(value) for name,value in sorted(files.items()) if name.startswith('trace_') and name.endswith('.json'))
        if plan['role']!='validation':raise PermissionError('This source does not yet audit held-out content')
        raw=store.get('data_v1/validation/EPISODES.json');data=store.json('data_v1/DATA_RELEASE.json')
        if hashlib.sha256(raw).hexdigest()!=data['cohorts']['validation']['files']['EPISODES.json']:raise ValueError('Trajectory audit cohort changed')
        scenes=[Scene(**v['scene']) for v in json.loads(raw)]
        from transformers import AutoTokenizer
        with tempfile.TemporaryDirectory(prefix='research4-composite-audit-') as scratch:
            bp,_=download_model(read(root/'inputs/MODERNBERT_PIN.json'),Path(scratch)/'bert',True);qp,_=download_model(read(root/'inputs/QWEN_PIN.json'),Path(scratch)/'qwen',True)
            bt=AutoTokenizer.from_pretrained(bp,local_files_only=True,trust_remote_code=False);qt=AutoTokenizer.from_pretrained(qp,local_files_only=True,trust_remote_code=False)
            generation=json.loads((qp/'generation_config.json').read_text()) if (qp/'generation_config.json').exists() else {}
            eos=generation.get('eos_token_id',[]);eos=set(eos if isinstance(eos,list) else [eos]);eos.add(qt.eos_token_id)
            for record in records:audits.append(audit_trajectory(record,plan['package']['threshold'],bt,qt,eos))
        result=summarize(records,scenes,cfg['evaluation']['conditions'],cfg,True)
        result.update(kind='trajectory_audit',role='validation',package_sha256=digest(plan['package']),raw_evidence_audited=True,model_predictions_recomputed=False,model_forwards=0,optimizer_updates=0,heldout_content_read=False)
        dump(output/'TRAJECTORY_AUDITS.json',audits);dump(output/'VALIDATION_AUDIT.json',result)
    else:raise ValueError('Unknown explicit audit kind')
except Exception as error:
    import traceback
    failure=str(error);result={}
    dump(output/'FAILURE.json',{'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()})
receipt={'status':'failed' if failure else 'completed','source_commit':commit,'run_id':run_id,'kind':plan['kind'],'artifact_path':str(output),'admitted':result.get('admitted'),'heldout_content_read':False,'model_forwards':0,'optimizer_updates':0}
dump(output/'AUDIT_RECEIPT.json',receipt);receipt['export']=archive(output);dump(output.parent/(run_id+'_EXPORT_RECEIPT.json'),receipt)
if failure:raise RuntimeError('Composite CPU audit failed; partial evidence archived')
dbutils.notebook.exit(json.dumps(receipt))
