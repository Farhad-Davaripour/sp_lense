"""Full-rank task-only stage bootstrap; never launches a second job."""
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
assert Path('/kaggle/input').is_dir() and Path('/kaggle/working').is_dir()
Path('/kaggle/temp').mkdir(exist_ok=True)
matches = list(Path('/kaggle/input').rglob('TASK_TRAINING_PACKAGE_FILES.json'))
assert len(matches) == 1, 'Attach exactly one reviewed task qualification package'
manifest_path = matches[0]
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == MANIFEST_SHA256
manifest = json.loads(manifest_path.read_text())
registered={'ordinary-task-skill-two-t4-training-postload-v2':{'qualification':1,'full_fit':64},
            'ordinary-task-skill-stateful-memory-v3':{'qualification':3,'full_fit':64},
            'ordinary-task-frozen-development-two-t4-v1':{'task_development':0}}
assert manifest['source_commit'] == SOURCE_COMMIT and manifest['version'] in registered
evaluation=manifest['version']=='ordinary-task-frozen-development-two-t4-v1'
root = Path('/kaggle/temp/research4_task_evaluation_inputs' if evaluation else '/kaggle/temp/research4_task_training_inputs') / SOURCE_COMMIT
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
source = root / 'source'
plan_path = source / ('inputs/TASK_EVALUATION_STAGE.json' if evaluation else 'inputs/TASK_TRAINING_PLAN.json')
stage = json.loads(plan_path.read_text())
assert stage['phase'] in ['qualification','full_fit','task_development'] and stage['scientific_GPU_count'] == 2
assert stage['version']==manifest['version'] and stage['phase'] in registered[stage['version']]
assert stage['optimizer_updates'] == registered[stage['version']][stage['phase']]
output = Path(stage['destination'])
output.parent.mkdir(parents=True, exist_ok=True)
child = {'child_started': False, 'child_exit_confirmed': None, 'exit_code': None}
receipt = {'source_commit': SOURCE_COMMIT, 'notebook': stage['notebook'], 'mode': stage['version'],
           'phase': stage['phase'], 'status': 'setting_up', 'model_forwards': None,
           'optimizer_updates': None, 'heldout_content_read': False,
           'automatic_scientific_retries': 0, 'full_fit_completed': False,
           'actual_training_capacity_qualified': False}
try:
    environment = Path('/kaggle/temp/research4_task_training_environment')
    venv.EnvBuilder(system_site_packages=True, with_pip=False).create(environment)
    python = environment / 'bin/python'
    # Inherit only an exact existing torch wheel; otherwise install the same
    # pinned official wheel inside the disposable environment before model work.
    probe = subprocess.run([str(python), '-c', 'import torch; print(torch.__version__)'],
                           text=True, capture_output=True, check=True, timeout=90)
    if probe.stdout.strip() != '2.11.0+cu128':
        subprocess.run([sys.executable, '-m', 'pip', '--python', str(python), 'install',
            '--disable-pip-version-check', '--only-binary=:all:', '--index-url',
            'https://download.pytorch.org/whl/cu128', 'torch==2.11.0+cu128'], check=True, timeout=900)
    pins = ['transformers==5.19.0', 'numpy==2.2.2', 'psutil==7.2.2', 'safetensors==0.8.0',
            'accelerate==1.15.0', 'bitsandbytes==0.49.2']
    subprocess.run([sys.executable, '-m', 'pip', '--python', str(python), 'install', '--disable-pip-version-check',
        '--only-binary=:all:', '--index-url', 'https://pypi.org/simple', *pins], check=True, timeout=600)
    env = {k: v for k, v in os.environ.items() if not k.startswith('DATABRICKS_')}
    if stage['version']=='ordinary-task-skill-stateful-memory-v3' or evaluation:
        # New child receives the configuration before importing Torch/CUDA.
        env.pop('PYTORCH_ALLOC_CONF',None);env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        env['PYTORCH_ALLOC_CONF']='expandable_segments:True'
    env.update(RESEARCH4_EXECUTION_PLATFORM='kaggle', CUDA_VISIBLE_DEVICES='0,1',
        RESEARCH4_TASK_TRAINING_SOURCE_ROOT=str(source),
        RESEARCH4_TASK_TRAINING_PLAN_SHA256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1',
        HF_HUB_DISABLE_XET='1', HF_HUB_DISABLE_SHARED_BLOBS='1')
    if evaluation:
        env.pop('RESEARCH4_TASK_TRAINING_SOURCE_ROOT',None);env.pop('RESEARCH4_TASK_TRAINING_PLAN_SHA256',None)
        env.update(RESEARCH4_TASK_EVALUATION_SOURCE_ROOT=str(source),
          RESEARCH4_TASK_EVALUATION_STAGE_SHA256=hashlib.sha256(plan_path.read_bytes()).hexdigest())
        expected=json.loads((source/'inputs/FIT_PARENT_EXPECTED.json').read_text())
        for key,commit,phase in [
          ('RESEARCH4_TASK_FULL_FIT_ARCHIVE',expected['source_commit'],'full_fit'),
          ('RESEARCH4_STATEFUL_QUALIFICATION_ARCHIVE','10eae62507d38f1507fb7443cf579ca5b6acc29f','qualification')]:
            suffix='research4_campaign/runs/'+commit+'/task_skill_training_v1/'+phase+'_01/evidence.zip'
            parents=[p for p in Path('/kaggle/input').rglob('evidence.zip') if p.as_posix().endswith(suffix)]
            assert len(parents)==1,'Attach each original completed fit/qualification output once'
            env[key]=str(parents[0])
    if stage['phase']=='full_fit':
        suffix='research4_campaign/runs/149278382451faa1aceed82f22a4604a9bf99f79/task_skill_training_v1/qualification_01/evidence.zip'
        if stage['version']=='ordinary-task-skill-stateful-memory-v3':
            suffix='research4_campaign/runs/10eae62507d38f1507fb7443cf579ca5b6acc29f/task_skill_training_v1/qualification_01/evidence.zip'
        parents=[p for p in Path('/kaggle/input').rglob('evidence.zip') if p.as_posix().endswith(suffix)]
        assert len(parents)==1,'Attach the exact completed qualification output once; never infer a parent from a summary'
        key='RESEARCH4_STATEFUL_QUALIFICATION_ARCHIVE' if stage['version']=='ordinary-task-skill-stateful-memory-v3' else 'RESEARCH4_TASK_QUALIFICATION_ARCHIVE'
        env[key]=str(parents[0])
    sys.path.insert(0, str(source))
    from src.composite_runtime import run_logged
    remaining = stage['worker_seconds']
    assert stage['hard_timeout_seconds'] - int(time.monotonic() - START) - 180 >= remaining, 'A complete registered worker/export envelope must remain after setup'
    worker_module='src.composite_task_skill_evaluation_worker' if evaluation else 'src.composite_task_skill_model_worker'
    result = run_logged([str(python), '-m', worker_module, '--root', str(source), '--commit', SOURCE_COMMIT],
        source, env, output.parent / 'TASK_TRAINING_WORKER_LOG.txt', remaining,
        exit_observer=lambda value: child.update(value))
    receipt.update(status='external_hard_timeout' if result['timed_out'] else 'worker_exited',
                   exit_code=result['exit_code'], timed_out=result['timed_out'])
except Exception as error:
    receipt.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
finally:
    try:
        sys.path.insert(0, str(source))
        from src.composite_task_skill_rescue import rescue_owned
        receipt['raw_prefix_rescue'] = rescue_owned(output, stage, child)
    except Exception as error:
        receipt.update(raw_prefix_rescue={'status':'unconfirmed','error':str(error),'scientific_result':None})
    receipt.update(child=child, seconds=time.monotonic() - START, provider_compute_release_verified=False)
    with (output.parent / 'TASK_TRAINING_CONTROLLER_RECEIPT.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))
assert receipt['status'] == 'worker_exited' and receipt.get('exit_code') == 0 and child.get('child_exit_confirmed') is True, 'Preserve this exact attempt and reconcile partial evidence; never retry automatically'
