"""Source-bound single child V4 frozen actor partition bootstrap; no resubmission."""
import hashlib,json,os,subprocess,sys,time,traceback,venv
from pathlib import Path

START=time.monotonic()
SOURCE_COMMIT='SOURCE_COMMIT_PLACEHOLDER'
MANIFEST_SHA256='PACKAGE_MANIFEST_SHA256_PLACEHOLDER'
assert Path('/kaggle/input').is_dir() and Path('/kaggle/working').is_dir()
Path('/kaggle/temp').mkdir(exist_ok=True)
matches=list(Path('/kaggle/input').rglob('BOUNDED_ACTOR_EVALUATION_PACKAGE_FILES.json'))
assert len(matches)==1,'Attach exactly one frozen V4 actor evaluation source package'
manifest_path=matches[0]
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==MANIFEST_SHA256
manifest=json.loads(manifest_path.read_text())
assert manifest['source_commit']==SOURCE_COMMIT
assert manifest['version']=='bounded-behavior-frozen-actor-evaluation-v4'
assert manifest_path.stat().st_size+sum((manifest_path.parent/name).stat().st_size for name in manifest['files'])<=256*1024**2
root=Path('/kaggle/temp/research4_bounded_eval_inputs')/SOURCE_COMMIT
root.mkdir(parents=True,exist_ok=False)
for name,expected in manifest['files'].items():
    assert name.startswith('source/') and '\\' not in name and ':' not in name
    assert not any(p in ['', '.', '..'] for p in name.split('/'))
    assert not (name.startswith('source/inputs/') and any(s in name.lower() for s in ['heldout','sealed','.env']))
    source_file=(manifest_path.parent/name).resolve()
    assert source_file.is_relative_to(manifest_path.parent.resolve())
    raw=source_file.read_bytes();assert hashlib.sha256(raw).hexdigest()==expected
    target=(root/name).resolve();assert target.is_relative_to(root.resolve())
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as stream:stream.write(raw)
with (root/'BOUNDED_ACTOR_EVALUATION_PACKAGE_FILES.json').open('xb') as stream:stream.write(manifest_path.read_bytes())
source=root/'source';plan_path=source/'inputs/BOUNDED_ACTOR_EVALUATION_STAGE.json'
stage=json.loads(plan_path.read_text())
assert stage['version']==manifest['version'] and stage['endpoint'] in ['behavior','competence']
assert stage['optimizer_updates']==0
assert stage['scientific_GPU_count']==2 and stage['visible_devices']=='0,1'
assert stage['native_interface_profile']=='bounded-fictional-record-list-native-v4'
output=Path(stage['destination']);output.parent.mkdir(parents=True,exist_ok=True)
child={'child_started':False,'child_exit_confirmed':None,'exit_code':None}
receipt={'source_commit':SOURCE_COMMIT,'notebook':stage['notebook'],'mode':stage['version'],
  'owner_thread_id':stage['owner_thread_id'],'phase':stage['phase'],'endpoint':stage['endpoint'],'status':'setting_up',
  'model_forwards':None,'optimizer_updates':None,'heldout_content_read':False,
  'automatic_scientific_retries':0,'full_fit_completed':False,
  'actual_training_capacity_qualified':False}
try:
    environment=Path('/kaggle/temp/research4_bounded_eval_environment')
    venv.EnvBuilder(system_site_packages=True,with_pip=False).create(environment)
    python=environment/'bin/python'
    probe=subprocess.run([str(python),'-c','import torch; print(torch.__version__)'],
      text=True,capture_output=True,check=True,timeout=90)
    if probe.stdout.strip()!='2.11.0+cu128':
        subprocess.run([sys.executable,'-m','pip','--python',str(python),'install',
          '--disable-pip-version-check','--only-binary=:all:','--index-url',
          'https://download.pytorch.org/whl/cu128','torch==2.11.0+cu128'],check=True,timeout=900)
    subprocess.run([sys.executable,'-m','pip','--python',str(python),'install',
      '--disable-pip-version-check','--only-binary=:all:','--index-url','https://pypi.org/simple',
      'transformers==5.19.0','numpy==2.2.2','psutil==7.2.2','safetensors==0.8.0',
      'accelerate==1.15.0','bitsandbytes==0.49.2'],check=True,timeout=600)
    env={k:v for k,v in os.environ.items() if not k.startswith(('DATABRICKS_','RESEARCH4_'))}
    env.pop('PYTORCH_ALLOC_CONF',None);env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    env.update(PYTORCH_ALLOC_CONF='expandable_segments:True',
      RESEARCH4_EXECUTION_PLATFORM='kaggle',CUDA_VISIBLE_DEVICES='0,1',
      RESEARCH4_BOUNDED_EVAL_SOURCE_ROOT=str(source),
      RESEARCH4_BOUNDED_EVAL_STAGE_SHA256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
      TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',
      HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_SHARED_BLOBS='1')
    def find_parent(key,suffix):
        parents=[p for p in Path('/kaggle/input').rglob('*.zip') if p.as_posix().endswith(suffix)]
        assert len(parents)==1,'Attach each exact complete parent raw archive once: '+key
        env[key]=str(parents[0])
    for key,commit,phase in [
      ('RESEARCH4_BOUNDED_ALIAS_FIT_ARCHIVE','82cf3ecf41e799facd93b8c11e05c6d1f569599c','full_fit'),
      ('RESEARCH4_BOUNDED_ALIAS_QUALIFICATION_ARCHIVE','10eae62507d38f1507fb7443cf579ca5b6acc29f','qualification')]:
        find_parent(key,'research4_campaign/runs/'+commit+'/task_skill_training_v1/'+phase+'_01/evidence.zip')
    parent=json.loads((source/'inputs/BOUNDED_FIT_PARENT_EXPECTED.json').read_text())
    direct=parent.get('fit_mode')=='bounded-fictional-behavior-actor-two-t4-sampling-a1-direct-v1'
    amended=direct or parent.get('fit_mode')=='bounded-fictional-behavior-actor-two-t4-sampling-a1'
    training='bounded_behavior_sampling_training_a1' if amended else 'bounded_behavior_training_v4'
    data='bounded_sampling_amendment_v1' if amended else 'bounded_behavior_v4_data'
    fit_phase='direct_full_fit_01' if direct else 'full_fit_01'
    find_parent('RESEARCH4_BOUNDED_FULL_FIT_ARCHIVE','research4_campaign/runs/'+parent['source_commit']+'/'+training+'/'+fit_phase+'/evidence.zip')
    find_parent('RESEARCH4_BOUNDED_DATA_ARCHIVE','research4_campaign/runs/'+parent['data_source_commit']+'/'+data+'/export_01_EVIDENCE.zip')
    if not direct:
        find_parent('RESEARCH4_BOUNDED_QUALIFICATION_ARCHIVE','research4_campaign/runs/'+parent['qualification_source_commit']+'/'+training+'/qualification_01/evidence.zip')
    sys.path.insert(0,str(source))
    from src.composite_runtime import run_logged
    remaining=stage['worker_seconds']
    assert stage['hard_timeout_seconds']-int(time.monotonic()-START)-180>=remaining,'Full registered worker/export envelope must remain'
    result=run_logged([str(python),'-m','src.composite_bounded_behavior_evaluation_worker','--root',str(source),'--commit',SOURCE_COMMIT],
      source,env,output.parent/'BOUNDED_ACTOR_WORKER_LOG.txt',remaining,
      exit_observer=lambda value:child.update(value))
    receipt.update(status='external_hard_timeout' if result['timed_out'] else 'worker_exited',
      exit_code=result['exit_code'],timed_out=result['timed_out'])
except Exception as error:
    receipt.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
finally:
    try:
        sys.path.insert(0,str(source))
        from src.composite_task_skill_rescue import rescue_owned
        receipt['raw_prefix_rescue']=rescue_owned(output,stage,child)
    except Exception as error:receipt['raw_prefix_rescue']={'status':'unconfirmed','error':str(error),'scientific_result':None}
    receipt.update(child=child,seconds=time.monotonic()-START,provider_compute_release_verified=False)
    with (output.parent/'BOUNDED_ACTOR_CONTROLLER_RECEIPT.json').open('x') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt))
assert receipt['status']=='worker_exited' and receipt.get('exit_code')==0 and child.get('child_exit_confirmed') is True,'Preserve this exact consumed attempt; never retry automatically'
