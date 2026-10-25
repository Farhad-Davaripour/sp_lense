"""Kaggle CPU-only option-A data bootstrap; no model loading/fitting."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import venv

START = time.monotonic()
SOURCE_COMMIT = 'SOURCE_COMMIT_PLACEHOLDER'
MANIFEST_SHA256 = 'PACKAGE_MANIFEST_SHA256_PLACEHOLDER'
runtime_paths = {p: Path(p).is_dir() for p in ['/kaggle/input', '/kaggle/working', '/kaggle/temp']}
print(json.dumps({'stage': 'pre_data_runtime_paths', 'directories': runtime_paths, 'python': sys.version.split()[0]}))
assert runtime_paths['/kaggle/input'] and runtime_paths['/kaggle/working'], 'Actual Kaggle mounted input/working roots are required'
# A CPU image need not pre-create the disposable scratch directory. Creating
# this fixed directory is setup only; it changes no data or scientific recipe.
Path('/kaggle/temp').mkdir(exist_ok=True)
matches = [p for filename in ['TASK_SKILL_PACKAGE_FILES.json','RISK_DATA_PACKAGE_FILES.json']
           for p in Path('/kaggle/input').rglob(filename)]
assert len(matches) == 1, 'Attach exactly one reviewed CPU data package'
manifest_path = matches[0]
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == MANIFEST_SHA256
manifest = json.loads(manifest_path.read_text())
registered={
  'ordinary-task-skill-data-bootstrap-v2':{'manifest':'TASK_SKILL_PACKAGE_FILES.json','root':'/kaggle/working/research4_inputs',
    'profile':'TASK_SKILL_PROFILE.json','worker':'src.composite_task_skill_worker',
    'root_env':'RESEARCH4_TASK_SOURCE_ROOT','profile_env':'RESEARCH4_TASK_PROFILE_SHA256'},
  'option-a-frozen-risk-development-data-v1':{'manifest':'RISK_DATA_PACKAGE_FILES.json','root':'/kaggle/temp/research4_frozen_risk_inputs',
    'profile':'FROZEN_RISK_DATA_PROFILE.json','worker':'src.composite_frozen_risk_data_worker',
    'root_env':'RESEARCH4_FROZEN_RISK_SOURCE_ROOT','profile_env':'RESEARCH4_FROZEN_RISK_PROFILE_SHA256'}}
assert manifest['source_commit']==SOURCE_COMMIT and manifest['version'] in registered
selected=registered[manifest['version']];assert manifest_path.name==selected['manifest']
root = Path(selected['root']) / SOURCE_COMMIT
root.mkdir(parents=True, exist_ok=False)
for name, expected in manifest['files'].items():
    assert name.startswith('source/') and '\\' not in name and '..' not in name.split('/') and ':' not in name
    assert not (name.startswith('source/inputs/') and any(s in name.lower() for s in ['heldout', 'sealed', '.env']))
    source_file = (manifest_path.parent / name).resolve()
    assert source_file.is_relative_to(manifest_path.parent.resolve())
    raw = source_file.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected
    target = (root / name).resolve()
    assert target.is_relative_to(root.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(raw)
with (root/selected['manifest']).open('xb') as stream:stream.write(manifest_path.read_bytes())
source = root / 'source'
profile_path = source / 'inputs'/selected['profile']
profile = json.loads(profile_path.read_text())
assert profile['version']==manifest['version'] and profile['scientific_GPU_count']==0
if manifest['version']=='option-a-frozen-risk-development-data-v1':assert profile['roles']==['development']
output = Path(profile['destination'])
output.parent.mkdir(parents=True, exist_ok=True)
child = {'child_started': False, 'child_exit_confirmed': None, 'exit_code': None}
receipt = {'source_commit': SOURCE_COMMIT, 'notebook': profile['notebook'], 'stage': profile['version'],
           'mode':profile['version'],'phase':profile.get('phase','task_data_export'),
           'status': 'setting_up', 'model_forwards': 0, 'optimizer_updates': 0,
           'GPU_allocated_by_stage': False, 'model_weights_downloaded': False,
           'heldout_content_read': False, 'automatic_scientific_retries': 0}
try:
    environment = Path('/kaggle/temp/research4_task_data_environment')
    venv.EnvBuilder(system_site_packages=True, with_pip=False).create(environment)
    python = environment / 'bin/python'
    pins = ['transformers==5.19.0', 'numpy==2.2.2', 'psutil==7.2.2', 'safetensors==0.8.0',
            'accelerate==1.15.0', 'bitsandbytes==0.49.2']
    available=profile['hard_timeout_seconds']-profile['worker_seconds']-30-int(time.monotonic()-START)
    assert available>0,'Preserve the complete CPU worker/export envelope'
    subprocess.run([sys.executable, '-m', 'pip', '--python', str(python), 'install', '--disable-pip-version-check',
        '--only-binary=:all:', '--index-url', 'https://pypi.org/simple', *pins], check=True,
        timeout=available)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('DATABRICKS_','RESEARCH4_'))}
    env.update(RESEARCH4_EXECUTION_PLATFORM='kaggle', CUDA_VISIBLE_DEVICES='',
        TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1',
        HF_HUB_DISABLE_XET='1', HF_HUB_DISABLE_SHARED_BLOBS='1')
    env[selected['root_env']]=str(source)
    env[selected['profile_env']]=hashlib.sha256(profile_path.read_bytes()).hexdigest()
    sys.path.insert(0, str(source))
    from src.composite_runtime import run_logged
    assert profile['hard_timeout_seconds']-int(time.monotonic()-START)-30>=profile['worker_seconds'],'Preserve the complete worker/export envelope after setup'
    result = run_logged([str(python), '-m', selected['worker'], '--root', str(source), '--commit', SOURCE_COMMIT],
        source, env, output.parent / 'TASK_DATA_WORKER_LOG.txt',
        profile['worker_seconds'],
        exit_observer=lambda value: child.update(value))
    receipt.update(status='external_hard_timeout' if result['timed_out'] else 'worker_exited',
                   exit_code=result['exit_code'], timed_out=result['timed_out'])
except Exception as error:
    receipt.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
finally:
    receipt.update(child=child, seconds=time.monotonic() - START, provider_compute_release_verified=False)
    with (output.parent / 'TASK_DATA_CONTROLLER_RECEIPT.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))
assert receipt['status'] == 'worker_exited' and receipt.get('exit_code') == 0 and child.get('child_exit_confirmed') is True, 'Preserve this exact attempt and partial evidence; never retry automatically'
