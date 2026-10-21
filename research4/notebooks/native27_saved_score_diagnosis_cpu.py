# Databricks notebook source
import os,sys,json,zipfile,hashlib
from pathlib import Path
os.environ.update(USE_TORCH='0',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
root=Path(dbutils.widgets.get('code_path'));commit=dbutils.widgets.get('source_commit');run_id=dbutils.widgets.get('run_id')
if root.name!=commit or not root.as_posix().startswith('/Workspace/Shared/research4-semantic-gate/deployed/'):raise ValueError('Unexpected release')
sys.path.insert(0,str(root))
from src.io_utils import read,dump,sha,assert_namespace,verify_release
from src.runtime import archive
from src.risk_saved_score_diagnosis import summarize
verify_release(root,commit);cfg=read(root/'config/native27_saved_score_diagnosis_v1.json')
def saved(path,expected):
    path=Path(path);assert_namespace(path)
    if sha(path)!=expected:raise ValueError('Saved archive digest differs')
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:raise ValueError('CRC failure')
        manifest=json.loads(z.read('MANIFEST.json'))
        if set(z.namelist())!=set(manifest)|{'MANIFEST.json'}:raise ValueError('Archive membership differs')
        for name,value in manifest.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=value:raise ValueError('Member digest differs')
        records={name:json.loads(z.read(name)) for name in manifest if name in ['ROWS.json','RESULT.json','SCORES.json','AUDIT.json']}
        records['_manifest']=manifest;return records
snapshot=saved(cfg['snapshot_folder']+'/evidence.zip',cfg['snapshot_sha256']);scored=saved(cfg['score_archive'],cfg['score_sha256']);audited=saved(cfg['audit_archive'],cfg['audit_sha256'])
if not audited['AUDIT.json']['audit_pass'] or scored['RESULT.json']['transfer_admitted'] or audited['AUDIT.json']['score_result_sha256']!=scored['_manifest']['RESULT.json']:raise ValueError('Unverified score inputs')
details=summarize(snapshot['ROWS.json'],scored['SCORES.json'][cfg['readout']])
output=Path('/Volumes/workspace/research4/artifacts/native27_risk_transfer_v1/saved_score_diagnosis')/commit/run_id;output.mkdir(parents=True,exist_ok=False)
result={'status':'complete','analysis_role':cfg['role'],'source_commit':commit,'Databricks_run_id':run_id,'new_model_calls':0,'new_fits':0,'retuned_thresholds':0,'transfer_admission_remains_failed':True,'original_heldout_scores_opened':False,'independently_audited':False,'inputs_independently_audited':True,'families':details,'output':output.as_posix(),'scope':cfg['scope']}
dump(output/'RESULT.json',result);result['export']=archive(output);dump(output.parent/(run_id+'_EXPORT_RECEIPT.json'),result)
dbutils.notebook.exit(json.dumps(result))
