# Kaggle notebook cell. The operator supplies the exact reviewed bundle SHA.
import hashlib
import base64
import json
import os
from pathlib import Path
import subprocess
import zipfile

expected_sha = RESEARCH4_PACKAGE_SHA256
source_commit = RESEARCH4_SOURCE_COMMIT
python = '/kaggle/temp/research4_environment_v1/bin/python'
assert Path(python).is_file(), 'Run the qualified pinned environment cells first'
packages = list(Path('/kaggle/input').rglob('FROZEN_PACKAGE.zip'))
manifests = list(Path('/kaggle/input').rglob('PACKAGE_FILES.json'))
assert (len(packages), len(manifests)) in [(1, 0), (0, 1)], 'Attach exactly one frozen Research 4 bundle'
root = Path('/kaggle/working/research4_inputs') / source_commit
root.mkdir(parents=True, exist_ok=False)
if packages:
    package = packages[0]
    assert hashlib.sha256(package.read_bytes()).hexdigest() == expected_sha
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
        assert sum(item.file_size for item in archive.infolist()) < 256 * 1024**2
        for item in archive.infolist():
            target = (root / item.filename).resolve()
            assert target.is_relative_to(root.resolve()) and '\\' not in item.filename
            if item.filename.startswith('artifacts/'):
                assert 'heldout' not in item.filename.lower()
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(item))
else:
    manifest = manifests[0]
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == RESEARCH4_PACKAGE_FILES_SHA256
    contents = json.loads(manifest.read_text())
    assert contents['source_commit'] == source_commit
    total = 0
    for relative, expected in contents['files'].items():
        origin = (manifest.parent / relative).resolve()
        target = (root / relative).resolve()
        assert origin.is_relative_to(manifest.parent.resolve())
        assert target.is_relative_to(root.resolve()) and '\\' not in relative
        if relative.startswith('artifacts/'):
            assert 'heldout' not in relative.lower()
        data = origin.read_bytes()
        total += len(data)
        assert total < 256 * 1024**2
        assert hashlib.sha256(data).hexdigest() == expected
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
contents_file = root / 'PACKAGE_FILES.json'
assert hashlib.sha256(contents_file.read_bytes()).hexdigest() == RESEARCH4_PACKAGE_FILES_SHA256
contents = json.loads(contents_file.read_text())
for relative, descriptor in contents.get('encoded_files', {}).items():
    stored = (root / descriptor['stored_path']).resolve()
    target = (root / relative).resolve()
    assert stored.is_relative_to(root.resolve()) and target.is_relative_to(root.resolve())
    assert '\\' not in relative and '\\' not in descriptor['stored_path']
    if relative.startswith('artifacts/'):
        assert 'heldout' not in relative.lower()
    decoded = base64.b64decode(stored.read_bytes(), validate=True)
    assert hashlib.sha256(decoded).hexdigest() == descriptor['decoded_sha256']
    with target.open('xb') as stream:
        stream.write(decoded)
source = root / 'source'
pin = json.loads((source / 'inputs/EXECUTION_PLATFORM.json').read_text())
assert pin['source_commit'] == source_commit and pin['scientific_retries'] == 0
assert pin['mode'] == 'detector-rank32-v5' and pin['hard_timeout_seconds'] == 2700
scratch = Path('/kaggle/working/research4_campaign/scratch') / source_commit
scratch.mkdir(parents=True, exist_ok=False)
environment = {key: value for key, value in os.environ.items()
               if not key.startswith('DATABRICKS_')}
environment.update(RESEARCH4_EXECUTION_PLATFORM='kaggle',
                   RESEARCH4_KAGGLE_INPUT_ROOT=pin['input_root'],
                   RESEARCH4_KAGGLE_MANIFEST_SHA256=pin['transport_manifest_sha256'],
                   CUDA_VISIBLE_DEVICES='0', TOKENIZERS_PARALLELISM='false',
                   HF_HUB_DISABLE_TELEMETRY='1', TMPDIR=str(scratch))
log = scratch.parent / (source_commit + '_WORKER_LOG.txt')
controller = {'source_commit': source_commit, 'package_sha256': expected_sha,
              'mode': pin['mode'], 'notebook': pin['notebook'],
              'heldout_content_read': False, 'scientific_retries': 0}
print('Starting the complete frozen recognition recipe; no automatic retry.', flush=True)
try:
    with log.open('x') as stream:
        result = subprocess.run([python, '-m', 'src.composite_worker', str(source),
                                 source_commit, pin['destination'], '--mode', pin['mode']],
                                cwd=source, env=environment, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=2700)
    controller.update(status='worker_exited', exit_code=result.returncode)
except subprocess.TimeoutExpired:
    controller.update(status='external_hard_timeout', exit_code=None,
                      scientific_execution_extent='inspect retained scratch evidence')
finally:
    (scratch.parent / (source_commit + '_CONTROLLER_RECEIPT.json')).write_text(
        json.dumps(controller, indent=2) + '\n')
print(json.dumps(controller), flush=True)
print(log.read_text()[-30000:], flush=True)
