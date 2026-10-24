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
matches = list(Path('/kaggle/input').rglob('TASK_SKILL_PACKAGE_FILES.json'))
assert len(matches) == 1, 'Attach exactly one reviewed task-only package'
manifest_path = matches[0]
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == MANIFEST_SHA256
manifest = json.loads(manifest_path.read_text())
assert manifest['source_commit'] == SOURCE_COMMIT and manifest['version'] == 'ordinary-task-skill-data-bootstrap-v2'
root = Path('/kaggle/working/research4_inputs') / SOURCE_COMMIT
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
profile_path = source / 'inputs/TASK_SKILL_PROFILE.json'
profile = json.loads(profile_path.read_text())
assert profile['version'] == 'ordinary-task-skill-data-bootstrap-v2' and profile['scientific_GPU_count'] == 0
output = Path(profile['destination'])
output.parent.mkdir(parents=True, exist_ok=True)
child = {'child_started': False, 'child_exit_confirmed': None, 'exit_code': None}
receipt = {'source_commit': SOURCE_COMMIT, 'notebook': profile['notebook'], 'stage': profile['version'],
           'status': 'setting_up', 'model_forwards': 0, 'optimizer_updates': 0,
           'GPU_allocated_by_stage': False, 'model_weights_downloaded': False,
           'heldout_content_read': False, 'automatic_scientific_retries': 0}
try:
    environment = Path('/kaggle/temp/research4_task_data_environment')
    venv.EnvBuilder(system_site_packages=True, with_pip=False).create(environment)
    python = environment / 'bin/python'
    pins = ['transformers==5.19.0', 'numpy==2.2.2', 'psutil==7.2.2', 'safetensors==0.8.0',
            'accelerate==1.15.0', 'bitsandbytes==0.49.2']
    subprocess.run([sys.executable, '-m', 'pip', '--python', str(python), 'install', '--disable-pip-version-check',
        '--only-binary=:all:', '--index-url', 'https://pypi.org/simple', *pins], check=True,
        timeout=max(1, profile['hard_timeout_seconds'] - int(time.monotonic() - START) - 120))
    env = {k: v for k, v in os.environ.items() if not k.startswith('DATABRICKS_')}
    env.update(RESEARCH4_EXECUTION_PLATFORM='kaggle', CUDA_VISIBLE_DEVICES='',
        RESEARCH4_TASK_SOURCE_ROOT=str(source),
        RESEARCH4_TASK_PROFILE_SHA256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1',
        HF_HUB_DISABLE_XET='1', HF_HUB_DISABLE_SHARED_BLOBS='1')
    sys.path.insert(0, str(source))
    from src.composite_runtime import run_logged
    result = run_logged([str(python), '-m', 'src.composite_task_skill_worker', '--root', str(source), '--commit', SOURCE_COMMIT],
        source, env, output.parent / 'TASK_DATA_WORKER_LOG.txt',
        max(1, profile['hard_timeout_seconds'] - int(time.monotonic() - START) - 30),
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
