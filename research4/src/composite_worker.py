"""Versioned bounded AIR stages for the two-adapter campaign."""
import argparse,gc,io,json,os,tempfile,time,traceback,zipfile,shutil
from pathlib import Path
from .io_utils import read,dump,sha,verify_release
from .composite_store import Store,download_model

CONFIG='config/composite_controller_v1.json'


def cohort_rows(root,role):
    import hashlib
    store=Store();data=store.json('data_v1/DATA_RELEASE.json')
    if role not in ['training','validation']:raise ValueError('This loader never opens held-out content')
    if data['world_source_sha256']!=sha(root/'src/composite_world.py'):raise ValueError('World changed after data materialization')
    raw=store.get('data_v1/'+role+'/DETECTION_ROWS.json')
    if hashlib.sha256(raw).hexdigest()!=data['cohorts'][role]['files']['DETECTION_ROWS.json']:raise ValueError('Cohort digest differs')
    rows=json.loads(raw)
    if any(r['split']!=role for r in rows):raise ValueError('Cohort role differs')
    return rows,data


def behavior_stage(root,output,cache,cfg,guard,progress,forward_callback,name):
    from .composite_actor import Actor
    from .composite_behavior import fit_behavior
    rows,data=cohort_rows(root,'training');placements=[p for p in cfg['qwen']['placement_candidates'] if p['name']==name]
    if len(placements)!=1:raise ValueError('Unregistered behavior placement')
    dump(output/'DATA_PROVENANCE.json',{'release':data,'heldout_content_read':False,'validation_examples_optimized':0})
    progress('download_qwen');path,proof=download_model(read(root/'inputs/QWEN_PIN.json'),cache)
    dump(output/'MODEL_FILE_VERIFICATION.json',proof);guard();progress('load_qwen')
    actor=Actor(path,cfg['qwen'],guard,forward_callback)
    dump(output/'BASE_ADMISSION.json',actor.frozen_proof())
    if not any(getattr(p,'quant_state',None) is not None for p in actor.model.parameters()):raise ValueError('NF4 profile was not applied')
    progress('fit_behavior');result=fit_behavior(actor,rows,placements[0],cfg,output,guard,progress)
    # OFF comparison temporarily unwraps the same Qwen instance; a second
    # pretrained Qwen is never allocated. These are training-only observations.
    texts=[next(r['text'] for r in rows if not r['terminal'] and r['labels']['gate']),next(r['text'] for r in rows if not r['terminal'] and not r['labels']['gate'])]
    progress('exact_off_bypass');bypass=actor.bypass_check(texts);dump(output/'BYPASS_CHECK.json',bypass)
    if not bypass['pass']:raise RuntimeError('Native OFF bypass / ON logit effect admission failed')
    result.update(stage='behavior',exact_off_bypass_pass=True,behavioral_success=None)
    return result


def detector_stage(root,output,cache,cfg,guard,progress,forward_callback):
    from .composite_training import fit_detector
    store=Store();data=store.json('data_v1/DATA_RELEASE.json')
    if data['world_source_sha256']!=sha(root/'src/composite_world.py'):raise ValueError('World changed after data materialization')
    datasets={}
    for role in ['training','validation']:
        raw=store.get('data_v1/'+role+'/DETECTION_ROWS.json')
        import hashlib
        if hashlib.sha256(raw).hexdigest()!=data['cohorts'][role]['files']['DETECTION_ROWS.json']:raise ValueError('Cohort digest differs')
        datasets[role]=json.loads(raw)
    if {r['group_id'] for r in datasets['training']} & {r['group_id'] for r in datasets['validation']}:raise ValueError('Identity-group split leakage')
    dump(output/'DATA_PROVENANCE.json',{'release':data,'heldout_content_read':False})
    progress('download_modernbert');model_path,proof=download_model(read(root/'inputs/MODERNBERT_PIN.json'),cache)
    dump(output/'MODEL_FILE_VERIFICATION.json',proof);progress('fit_modernbert')
    result=fit_detector(model_path,cfg,datasets['training'],datasets['validation'],output,guard,progress,forward_callback)
    result.update(stage='detector',heldout_content_read=False,Qwen_optimizer_updates=0)
    return result


def run(root,commit,destination,mode):
    import torch,transformers,bitsandbytes
    from databricks.sdk import WorkspaceClient
    root=Path(root);cfg=read(root/CONFIG)
    if not os.environ.get('MLFLOW_RUN_ID') or root.resolve()!=Path(os.environ['CODE_SOURCE_PATH']).resolve():raise ValueError('Actual Databricks AIR runtime required')
    verify_release(root,commit)
    if not destination.startswith('/Volumes/workspace/research4/artifacts/composite_controller_v1/') or '..' in destination.split('/'):raise ValueError('Dedicated campaign destination required')
    for n,h in read(root/'inputs/MANIFEST.json').items():
        if sha(root/'inputs'/n)!=h:raise ValueError('Input pin changed')
    client=WorkspaceClient();client.files.create_directory(destination)
    def upload(name,data):
        client.files.upload(destination+'/'+name,io.BytesIO(data),overwrite=False)
        with client.files.download(destination+'/'+name).contents as stream:back=stream.read()
        if back!=data:raise ValueError('Campaign export readback differs')
    upload('ALLOCATION.json',json.dumps({'mode':mode,'source_commit':commit,'MLFLOW_RUN_ID':os.environ['MLFLOW_RUN_ID'],'scientific_retries':0}).encode())
    with tempfile.TemporaryDirectory(prefix='research4-composite-') as scratch:
        base=Path(scratch);output=base/'output';output.mkdir();cache=base/'cache';cache.mkdir();started=time.monotonic()
        receipt={'status':'running','stage':mode,'source_commit':commit,'heldout_content_read':False};counts={'attempted':0,'completed':0}
        def guard():
            if time.monotonic()-started>=cfg['bounds']['worker_seconds']-cfg['bounds']['export_reserve_seconds']:raise TimeoutError('Protected campaign export deadline')
            if torch.cuda.is_available() and torch.cuda.mem_get_info()[0]<536870912:raise MemoryError('Live GPU export reserve')
        def forward_callback(event):
            if event=='attempt':
                guard()
                if counts['attempted']>=cfg['bounds']['maximum_backbone_forwards_per_run']:raise TimeoutError('Backbone forward ceiling')
                counts['attempted']+=1
            else:counts['completed']+=1
        def progress(stage,**kw):
            v={'stage':stage,'seconds':time.monotonic()-started,'backbone_forwards':counts['completed'],**kw};dump(output/'PROGRESS.json',v);print(json.dumps(v),flush=True)
        try:
            if torch.__version__!='2.11.0+cu128' or transformers.__version__!='5.19.0' or bitsandbytes.__version__!='0.49.2':raise RuntimeError('Pinned campaign software differs')
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.device_count()!=1:raise RuntimeError('One BF16-capable quota-approved GPU required')
            dump(output/'PRECALL_FREEZE.json',{'config':cfg,'mode':mode,'source_commit':commit,'input_manifest_sha256':sha(root/'inputs/MANIFEST.json')})
            dump(output/'SOFTWARE.json',{'torch':torch.__version__,'transformers':transformers.__version__,'bitsandbytes':bitsandbytes.__version__,'GPU':torch.cuda.get_device_name(0)})
            torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
            if mode=='detector':result=detector_stage(root,output,cache,cfg,guard,progress,forward_callback)
            elif mode.startswith('behavior-'):result=behavior_stage(root,output,cache,cfg,guard,progress,forward_callback,mode[len('behavior-'):])
            elif mode=='evaluation':
                from .composite_evaluation import evaluate_stage
                if read(root/'inputs/EVALUATION_PLAN.json')['role']=='heldout':receipt['heldout_content_read']=None
                result=evaluate_stage(root,output,cache,cfg,guard,progress,forward_callback)
                receipt['heldout_content_read']=result['heldout_content_read']
            else:raise ValueError('Stage not implemented in this version; do not infer completion')
            receipt.update(status='complete',result=result)
        except Exception as error:
            receipt.update(status='failed',error_type=type(error).__name__,error=str(error));dump(output/'FAILURE.json',{'traceback':traceback.format_exc()})
        finally:
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
            receipt.update(seconds=time.monotonic()-started,worker_exiting=True,backbone_forward_attempts=counts['attempted'],backbone_forwards=counts['completed']);dump(output/'RUN_RECEIPT.json',receipt)
        dump(output/'MANIFEST.json',{f.relative_to(output).as_posix():sha(f) for f in output.rglob('*') if f.is_file() and f.name!='MANIFEST.json'})
        archive=base/'evidence.zip'
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for f in sorted(output.rglob('*')):
                if f.is_file():z.write(f,f.relative_to(output).as_posix())
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:raise ValueError('Campaign archive CRC differs')
        upload('evidence.zip',archive.read_bytes());upload('RUN_RECEIPT.json',(output/'RUN_RECEIPT.json').read_bytes())
        export={**receipt,'destination':destination,'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'owned_temporary_cleanup':True};upload('EXPORT_RECEIPT.json',json.dumps(export,indent=2).encode())
    print(json.dumps(export),flush=True)
    if receipt['status']!='complete':raise RuntimeError('Campaign stage failed; partial evidence retained')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('commit');p.add_argument('destination');p.add_argument('--mode',required=True);a=p.parse_args();run(a.root,a.commit,a.destination,a.mode)
