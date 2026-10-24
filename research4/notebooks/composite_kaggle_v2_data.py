"""Kaggle CPU-only reference-data stage. No model loading or training."""
import hashlib,json,os,subprocess,sys,time,venv
from pathlib import Path

START=time.monotonic()
SOURCE_COMMIT='SOURCE_COMMIT_PLACEHOLDER'
MANIFEST_SHA256='PACKAGE_MANIFEST_SHA256_PLACEHOLDER'
assert all(Path(p).is_dir() for p in ['/kaggle/input','/kaggle/working','/kaggle/temp'])
matches=list(Path('/kaggle/input').rglob('KAGGLE_V2_PACKAGE_FILES.json'))
assert len(matches)==1,'Attach exactly one reviewed V2 data bundle; never guess a prior input'
manifest_path=matches[0]
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==MANIFEST_SHA256
manifest=json.loads(manifest_path.read_text());assert manifest['source_commit']==SOURCE_COMMIT
root=Path('/kaggle/working/research4_inputs')/SOURCE_COMMIT
root.mkdir(parents=True,exist_ok=False)
for name,expected in manifest['files'].items():
    assert name.startswith('source/') and '\\' not in name and '..' not in name.split('/') and ':' not in name
    assert not (name.startswith('source/inputs/') and any(x in name.lower() for x in ['heldout','sealed','.env']))
    source=(manifest_path.parent/name).resolve();assert source.is_relative_to(manifest_path.parent.resolve())
    raw=source.read_bytes();assert hashlib.sha256(raw).hexdigest()==expected
    target=(root/name).resolve();assert target.is_relative_to(root.resolve())
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as stream:stream.write(raw)
source=root/'source'
profile=json.loads((source/'inputs/KAGGLE_REMAINDER_PROFILE.json').read_text())
assert profile['mode']=='generalized-data-export-v2' and profile['scientific_GPU_count']==0
# Reuse the reviewed package pins; inherit the platform's torch wheel without
# downloading model weights or allocating any accelerator for data generation.
environment=Path('/kaggle/temp/research4_v2_data_environment')
venv.EnvBuilder(system_site_packages=True,with_pip=False).create(environment)
python=environment/'bin/python'
pins=['transformers==5.19.0','numpy==2.2.2','psutil==7.2.2','safetensors==0.8.0','accelerate==1.15.0','bitsandbytes==0.49.2']
subprocess.run([sys.executable,'-m','pip','--python',str(python),'install','--disable-pip-version-check',
    '--only-binary=:all:','--index-url','https://pypi.org/simple',*pins],check=True,
    timeout=max(1,profile['hard_timeout_seconds']-int(time.monotonic()-START)-120))
environment_vars={k:v for k,v in os.environ.items() if not k.startswith('DATABRICKS_')}
environment_vars.update(RESEARCH4_EXECUTION_PLATFORM='kaggle',CUDA_VISIBLE_DEVICES='',
    RESEARCH4_KAGGLE_V2_SOURCE_ROOT=str(source),
    RESEARCH4_KAGGLE_V2_PROFILE_SHA256=hashlib.sha256((source/'inputs/KAGGLE_REMAINDER_PROFILE.json').read_bytes()).hexdigest(),
    TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',
    HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_SHARED_BLOBS='1')
sys.path.insert(0,str(source))
from src.composite_runtime import run_logged
output=Path(profile['destination']);output.parent.mkdir(parents=True,exist_ok=True)
log=output.parent/'CPU_DATA_WORKER_LOG.txt'
child={}
remaining=max(1,profile['hard_timeout_seconds']-int(time.monotonic()-START)-30)
result=run_logged([str(python),'-m','src.composite_v2_data_worker','--root',str(source),'--commit',SOURCE_COMMIT],
    source,environment_vars,log,remaining,exit_observer=lambda value:child.update(value))
receipt={'source_commit':SOURCE_COMMIT,'notebook':profile['notebook'],'mode':profile['mode'],'child':child,
    'status':'external_hard_timeout' if result['timed_out'] else 'worker_exited','exit_code':result['exit_code'],
    'seconds':time.monotonic()-START,'automatic_scientific_retries':0,'model_weights_downloaded':False,
    'model_forwards':0,'optimizer_updates':0,'GPU_allocated_by_stage':False,'heldout_content_read':False,
    'provider_compute_release_verified':False}
with (output.parent/'CPU_DATA_CONTROLLER_RECEIPT.json').open('x') as stream:json.dump(receipt,stream,indent=2)
print(json.dumps(receipt))
assert result['exit_code']==0 and not result['timed_out'],'Preserve partial data and reconcile this exact identity; never retry automatically'
