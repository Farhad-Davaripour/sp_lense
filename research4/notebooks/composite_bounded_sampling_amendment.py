"""One exact V4 Kaggle CPU data bootstrap; no model loading or fitting."""
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
matches = list(Path('/kaggle/input').rglob('BOUNDED_SAMPLING_AMENDMENT_PACKAGE_FILES.json'))
assert len(matches) == 1, 'Attach exactly one reviewed V4 CPU package'
manifest_path = matches[0]
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == MANIFEST_SHA256
manifest = json.loads(manifest_path.read_text())
assert manifest['source_commit'] == SOURCE_COMMIT and manifest['version'] == 'bounded-behavior-approved-sampling-amendment-v1-data'
assert manifest['notebook'] == 'farhad89/research4-bounded-sampling-amendment-v1'
assert manifest_path.stat().st_size + sum((manifest_path.parent/name).stat().st_size for name in manifest['files']) <=128*1024**2
root = Path('/kaggle/temp/research4_sampling_amendment_inputs') / SOURCE_COMMIT
root.mkdir(parents=True, exist_ok=False)
for name, expected in manifest['files'].items():
    assert name.startswith('source/') and '\\' not in name and ':' not in name
    assert not any(p in ['', '.', '..'] for p in name.split('/'))
    assert not (name.startswith('source/inputs/') and any(s in name.lower() for s in ['heldout', 'sealed', '.env']))
    origin = (manifest_path.parent / name).resolve()
    assert origin.is_relative_to(manifest_path.parent.resolve())
    raw = origin.read_bytes(); assert hashlib.sha256(raw).hexdigest() == expected
    target = (root / name).resolve(); assert target.is_relative_to(root.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream: stream.write(raw)
with (root / 'BOUNDED_SAMPLING_AMENDMENT_PACKAGE_FILES.json').open('xb') as stream:
    stream.write(manifest_path.read_bytes())
source = root / 'source'
profile_path = source / 'inputs/BOUNDED_SAMPLING_AMENDMENT_PROFILE.json'
profile = json.loads(profile_path.read_text())
assert profile['source_commit'] == SOURCE_COMMIT and profile['version'] == manifest['version']
assert profile['phase'] == 'bounded_sampling_amendment_v1' and profile['scientific_GPU_count'] == 0
assert profile['visible_devices'] == '' and profile['owner_thread_id'] == '01a175f0-fc99-76b1-8bb2-d64f22afbcd7'
output = Path(profile['destination']); output.parent.mkdir(parents=True, exist_ok=True)
setup_failure = None
try:
    environment = Path('/kaggle/temp/research4_sampling_amendment_environment')
    venv.EnvBuilder(system_site_packages=True, with_pip=False).create(environment)
    python = environment / 'bin/python'
    # Torch is used solely by the original raw tensor auditor, with no models.
    available = lambda: profile['hard_timeout_seconds'] - profile['worker_seconds'] - 30 - int(time.monotonic() - START)
    assert available() > 0, 'Preserve the full unchanged worker/export window'
    probe = subprocess.run([str(python), '-c', 'import torch; print(torch.__version__)'],
                           text=True, capture_output=True, timeout=min(90, available()))
    if probe.returncode != 0 or probe.stdout.strip() != '2.11.0+cpu':
        subprocess.run([sys.executable, '-m', 'pip', '--python', str(python), 'install',
            '--disable-pip-version-check', '--only-binary=:all:', '--index-url',
            'https://download.pytorch.org/whl/cpu', 'torch==2.11.0+cpu'], check=True, timeout=available())
    pins = ['transformers==5.19.0', 'numpy==2.2.2', 'safetensors==0.8.0']
    assert available() > 0
    subprocess.run([sys.executable, '-m', 'pip', '--python', str(python), 'install',
        '--disable-pip-version-check', '--only-binary=:all:', '--index-url', 'https://pypi.org/simple',
        *pins], check=True, timeout=available())
    assert available() > 0
    probe_code = ("import json,sys,torch; from importlib.metadata import version; "
                  "print(json.dumps({'python':sys.version.split()[0],'torch':str(torch.__version__),"
                  "'torch_cuda_build':torch.version.cuda,'packages':{name:version(name) for name in "
                  "['transformers','numpy','safetensors']}}))")
    probed = subprocess.run([str(python), '-c', probe_code], text=True, capture_output=True,
                            check=True, timeout=min(90, available()))
    sys.path.insert(0, str(source))
    from src.composite_bounded_sampling_worker import verify_cpu_environment
    cpu_runtime = verify_cpu_environment(json.loads(probed.stdout))
    setup_receipt = {'source_commit': SOURCE_COMMIT, 'mode': profile['version'],
        'phase': profile['phase'], 'owner_thread_id': profile['owner_thread_id'],
        'status': 'CPU_environment_ready', 'cpu_runtime': cpu_runtime,
        'seconds': time.monotonic() - START, 'model_forwards': 0, 'optimizer_updates': 0}
    with (output.parent / 'BOUNDED_SAMPLING_SETUP_RECEIPT.json').open('x') as stream:
        json.dump(setup_receipt, stream, indent=2)
    print(json.dumps(setup_receipt))
    env = {k: v for k, v in os.environ.items() if not k.startswith(('DATABRICKS_', 'RESEARCH4_'))}
    env.update(RESEARCH4_EXECUTION_PLATFORM='kaggle', CUDA_VISIBLE_DEVICES='',
        RESEARCH4_SAMPLING_SOURCE_ROOT=str(source),
        RESEARCH4_SAMPLING_PROFILE_SHA256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
        HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_XET='1', HF_HUB_DISABLE_SHARED_BLOBS='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    for key, commit, phase in [
        ('RESEARCH4_BOUNDED_ALIAS_FIT_ARCHIVE', '82cf3ecf41e799facd93b8c11e05c6d1f569599c', 'full_fit'),
        ('RESEARCH4_BOUNDED_ALIAS_QUALIFICATION_ARCHIVE', '10eae62507d38f1507fb7443cf579ca5b6acc29f', 'qualification')]:
        suffix = 'research4_campaign/runs/' + commit + '/task_skill_training_v1/' + phase + '_01/evidence.zip'
        parents = [p for p in Path('/kaggle/input').rglob('evidence.zip') if p.as_posix().endswith(suffix)]
        assert len(parents) == 1, 'Attach each exact original alias raw dependency once'
        env[key] = str(parents[0])
    sys.path.insert(0, str(source))
    from src.composite_bounded_sampling_controller import run
    receipt = run(source, SOURCE_COMMIT, python, env, elapsed_setup_seconds=time.monotonic() - START)
except Exception as error:
    setup_failure = {'source_commit': SOURCE_COMMIT, 'mode': profile['version'],
        'phase': profile['phase'], 'notebook': profile['notebook'],
        'owner_thread_id': profile['owner_thread_id'], 'status': 'failed_setup',
        'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc(),
        'child': {'child_started': False, 'child_exit_confirmed': True, 'exit_code': None},
        'model_forwards': 0, 'optimizer_updates': 0, 'automatic_scientific_retries': 0,
        'heldout_content_read': False, 'provider_compute_release_verified': False}
    path = output.parent / 'BOUNDED_SAMPLING_CONTROLLER_RECEIPT.json'
    if not path.exists():
        with path.open('x') as stream: json.dump(setup_failure, stream, indent=2)
    receipt = setup_failure
print(json.dumps(receipt))
assert (receipt['status'] == 'worker_exited' and receipt.get('exit_code') == 0
        and receipt['child']['child_exit_confirmed'] is True), 'Preserve this consumed attempt; no automatic retries'
