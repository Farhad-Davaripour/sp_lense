"""One owned Kaggle two-T4 bootstrap for the frozen record-entry alias probe.

The reviewed source/package preparer fills the two placeholders before making
the notebook. This controller launches one child once and never retries it.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import venv
import zipfile

START = time.monotonic()
SOURCE_COMMIT = 'SOURCE_COMMIT_PLACEHOLDER'
MANIFEST_SHA256 = 'PACKAGE_MANIFEST_SHA256_PLACEHOLDER'
MODE = 'ordinary-record-entry-alias-frozen-probe-v1'
assert Path('/kaggle/input').is_dir() and Path('/kaggle/working').is_dir()
Path('/kaggle/temp').mkdir(exist_ok=True)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


matches = list(Path('/kaggle/input').rglob('TOOL_ALIAS_PACKAGE_FILES.json'))
assert len(matches) == 1, 'Attach one exact reviewed alias source package'
manifest_path = matches[0]
assert file_sha(manifest_path) == MANIFEST_SHA256
manifest = json.loads(manifest_path.read_text())
assert manifest['source_commit'] == SOURCE_COMMIT and manifest['version'] == MODE
alias_container = Path('/kaggle/temp/research4_tool_alias_inputs') / SOURCE_COMMIT
alias_container.mkdir(parents=True, exist_ok=False)
for name, expected in manifest['files'].items():
    assert name.startswith('source/') and '\\' not in name and ':' not in name and '..' not in name.split('/')
    assert not (name.startswith('source/inputs/') and
                any(word in name.lower() for word in ('heldout', 'sealed', '.env')))
    original = (manifest_path.parent / name).resolve()
    assert original.is_relative_to(manifest_path.parent.resolve())
    assert file_sha(original) == expected
    target = (alias_container / name).resolve()
    assert target.is_relative_to(alias_container.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    with original.open('rb') as source, target.open('xb') as destination:
        for block in iter(lambda: source.read(8 * 1024**2), b''):
            destination.write(block)
        destination.flush()
        os.fsync(destination.fileno())
    assert file_sha(target) == expected
(alias_container / 'TOOL_ALIAS_PACKAGE_FILES.json').write_bytes(manifest_path.read_bytes())
alias_root = alias_container / 'source'
stage_path = alias_root / 'inputs/TOOL_ALIAS_STAGE.json'
stage = json.loads(stage_path.read_text())
assert (stage['version'] == MODE and stage['source_commit'] == SOURCE_COMMIT
        and stage['phase'] == 'ordinary_entry_alias_diagnostic'
        and stage['platform'] == 'kaggle' and stage['account'] == 'farhad89'
        and stage['scientific_GPU_count'] == 2 and stage['visible_devices'] == '0,1'
        and stage['expected_responses'] == 48 and stage['maximum_new_tokens'] == 160
        and stage['optimizer_updates'] == 0 and stage['world_actions'] == 0
        and stage['maximum_concurrent_owned_workers'] == 1
        and stage['automatic_scientific_retries'] == 0)

actor_matches = list(Path('/kaggle/input').rglob('ACTOR_PACKAGE.zip'))
assert len(actor_matches) == 1, 'Attach the byte-identical released partition-00 actor package'
actor_archive = actor_matches[0]
assert file_sha(actor_archive) == stage['actor_package_sha256']
actor_container = Path('/kaggle/temp/research4_task_evaluation_inputs') / stage['actor_source_commit']
actor_container.mkdir(parents=True, exist_ok=False)
with zipfile.ZipFile(actor_archive) as archive:
    names = archive.namelist()
    assert len(names) == len(set(names))
    assert all(name.startswith('source/') or name == 'TASK_TRAINING_PACKAGE_FILES.json'
               for name in names if not name.endswith('/'))
    assert not any('..' in name.split('/') or '\\' in name or ':' in name
                   or name.startswith('/') or (name.startswith('source/inputs/') and
                   any(word in name.lower() for word in ('heldout', 'sealed', '.env')))
                   for name in names)
    for item in archive.infolist():
        assert item.file_size <= 256 * 1024**2
        if item.is_dir():
            continue
        target = (actor_container / item.filename).resolve()
        assert target.is_relative_to(actor_container.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(item) as source, target.open('xb') as destination:
            for block in iter(lambda: source.read(8 * 1024**2), b''):
                destination.write(block)
            destination.flush()
            os.fsync(destination.fileno())
actor_root = actor_container / 'source'
assert file_sha(actor_root / 'inputs/TASK_EVALUATION_STAGE.json') == stage['actor_stage_sha256']
assert file_sha(actor_container / 'TASK_TRAINING_PACKAGE_FILES.json') == stage['actor_manifest_sha256']

fit_expected = json.loads((actor_root / 'inputs/FIT_PARENT_EXPECTED.json').read_text())
parents = []
for source_commit, phase in [(fit_expected['source_commit'], 'full_fit'),
                             ('10eae62507d38f1507fb7443cf579ca5b6acc29f', 'qualification')]:
    suffix = ('research4_campaign/runs/' + source_commit + '/task_skill_training_v1/' +
              phase + '_01/evidence.zip')
    paths = [path for path in Path('/kaggle/input').rglob('evidence.zip')
             if path.as_posix().endswith(suffix)]
    assert len(paths) == 1, 'Attach each exact original fit/qualification output once'
    parents.append(paths[0])
fit_archive, qualification_archive = parents
assert file_sha(fit_archive) == stage['fit_raw_archive_sha256']

output = Path(stage['destination'])
output.parent.mkdir(parents=True, exist_ok=True)
assert not output.exists(), 'One-use probe output already exists; no duplicate launch'
controller = {'source_commit': SOURCE_COMMIT, 'version': MODE, 'status': 'setting_up',
              'child_started': False, 'child_exit_confirmed': None,
              'exit_code': None, 'automatic_scientific_retries': 0,
              'optimizer_updates': 0, 'world_actions': 0,
              'official_task72_or_risk180_result': False}
try:
    envdir = Path('/kaggle/temp/research4_tool_alias_environment')
    venv.EnvBuilder(system_site_packages=True, with_pip=False).create(envdir)
    python = envdir / 'bin/python'
    observed = subprocess.run([str(python), '-c', 'import torch; print(torch.__version__)'],
                              text=True, capture_output=True, check=True, timeout=90)
    if observed.stdout.strip() != '2.11.0+cu128':
        subprocess.run([sys.executable, '-m', 'pip', '--python', str(python), 'install',
            '--disable-pip-version-check', '--only-binary=:all:', '--index-url',
            'https://download.pytorch.org/whl/cu128', 'torch==2.11.0+cu128'],
            check=True, timeout=900)
    pins = ['transformers==5.19.0', 'numpy==2.2.2', 'psutil==7.2.2',
            'safetensors==0.8.0', 'accelerate==1.15.0', 'bitsandbytes==0.49.2']
    subprocess.run([sys.executable, '-m', 'pip', '--python', str(python), 'install',
        '--disable-pip-version-check', '--only-binary=:all:', '--index-url',
        'https://pypi.org/simple', *pins], check=True, timeout=600)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('DATABRICKS_', 'RESEARCH4_'))}
    env.pop('PYTORCH_ALLOC_CONF', None)
    env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
    env.update(RESEARCH4_EXECUTION_PLATFORM='kaggle', CUDA_VISIBLE_DEVICES='0,1',
        PYTORCH_ALLOC_CONF='expandable_segments:True', TOKENIZERS_PARALLELISM='false',
        HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1',
        HF_HUB_DISABLE_XET='1', HF_HUB_DISABLE_SHARED_BLOBS='1',
        RESEARCH4_TASK_EVALUATION_SOURCE_ROOT=str(actor_root),
        RESEARCH4_TASK_EVALUATION_STAGE_SHA256=file_sha(actor_root / 'inputs/TASK_EVALUATION_STAGE.json'),
        RESEARCH4_TASK_FULL_FIT_ARCHIVE=str(fit_archive),
        RESEARCH4_STATEFUL_QUALIFICATION_ARCHIVE=str(qualification_archive))
    sys.path.insert(0, str(actor_root))
    from src.composite_runtime import run_logged
    remaining = stage['worker_seconds']
    assert (stage['hard_timeout_seconds'] - int(time.monotonic() - START) - 180 >= remaining), \
        'Complete reviewed worker/export envelope must remain after setup'
    child = [str(python), '-u', str(alias_root / 'src/composite_tool_alias_worker.py'),
             '--alias-root', str(alias_root), '--actor-root', str(actor_root),
             '--alias-commit', SOURCE_COMMIT, '--actor-archive', str(actor_archive)]
    result = run_logged(child, actor_root, env,
        output.parent / 'TOOL_ALIAS_WORKER_LOG.txt', remaining,
        exit_observer=lambda value: controller.update(value))
    controller.update(status='external_hard_timeout' if result['timed_out'] else 'worker_exited',
                      exit_code=result['exit_code'], timed_out=result['timed_out'])
except BaseException as error:
    controller.update(status='failed', error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
finally:
    controller.update(elapsed_seconds=time.monotonic() - START,
                      provider_compute_release_verified=False)
    with (output.parent / 'TOOL_ALIAS_CONTROLLER_RECEIPT.json').open('x') as stream:
        json.dump(controller, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(controller), flush=True)
assert (controller['status'] == 'worker_exited' and controller['exit_code'] == 0
        and controller['child_exit_confirmed'] is True), \
    'Preserve this exact attempt and reconcile its raw evidence; no automatic retry'
