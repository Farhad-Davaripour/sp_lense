"""Prepare version-4 isolated root without starting a model job."""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = HERE.parent / 'model_organism_v3'
V1 = HERE.parent / 'model_organism_v1'
sys.path.insert(0, str(V3))
import boundary_guard

ROOT = Path('/var/lib/sp-lense-r3-organism-v4')
CODE = Path('/opt/sp-lense-r3-organism-v4/code')
PARENT = Path('/var/lib/sp-lense-r3-organism-v3')
OLD_ROOT = '/var/lib/sp-lense-r3-organism-v3'
OLD_CODE = '/opt/sp-lense-r3-organism-v3/code'
ARMS = ('preservation', 'continuity', 'neutral')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Unexpected isolation source structure')
    return source.replace(old, new, 1)


def main():
    if os.geteuid() != 0 or ROOT.exists() or CODE.exists():
        raise RuntimeError('Requires root and unused version-4 directories')
    if read(PARENT / 'p3-GATE.json')['foundation_pass'] or not read(PARENT / 'p3-AUDIT.json')['passed']:
        raise RuntimeError('Version-3 parent result is missing or unexpectedly passed')
    for directory in (PARENT / 'runs').iterdir():
        group = Path('/sys/fs/cgroup/system.slice') / ('sp-r3-' + directory.name + '.service') / 'cgroup.procs'
        if group.exists() and group.read_text().strip():
            raise RuntimeError('Earlier study worker still active')
    ROOT.mkdir(mode=0o755)
    (ROOT / 'runs').mkdir()
    CODE.mkdir(parents=True)
    shutil.copytree(HERE / 'data', ROOT / 'inputs')
    shutil.copyfile(PARENT / 'inputs/model_manifest.json', ROOT / 'inputs/model_manifest.json')
    shutil.copyfile(V1 / 'requirements-recorded.txt', ROOT / 'inputs/requirements-recorded.txt')
    parent_adapters = {}
    for arm in ARMS:
        source = PARENT / 'inputs/adapters/pass3' / arm
        target = ROOT / 'inputs/adapters/parent_v3' / arm
        shutil.copytree(source, target)
        receipt = read(PARENT / 'runs' / f'r3-p3-fit-{arm}-targets2' / 'receipt.json')
        if receipt['returncode'] or not receipt['worker_processes_gone']:
            raise RuntimeError('Invalid parent adapter receipt: ' + arm)
        parent_adapters[arm] = {}
        for file in source.iterdir():
            expected = receipt['artifact_sha256'].get('adapter/' + file.name)
            if expected is None or digest(file) != expected or digest(target / file.name) != expected:
                raise RuntimeError('Parent adapter copy differs: ' + str(file))
            parent_adapters[arm][file.name] = expected
    manifest = read(ROOT / 'inputs/model_manifest.json')
    for name, expected in manifest['sha256'].items():
        if digest(Path('/var/lib/sp-lense-r3/model') / name) != expected:
            raise RuntimeError('Pinned base file changed: ' + name)
    for name in ('TRAIN_FREEZE.json', 'COMPREHENSION_FREEZE.json', 'BENIGN_FREEZE.json',
                 'PREFERENCE_VALIDATION_FREEZE.json', 'RUNTIME_FREEZE.json'):
        freeze = read(ROOT / 'inputs' / name)
        for filename, expected in freeze['sha256'].items():
            path = HERE / filename if filename.endswith('.py') else ROOT / 'inputs' / filename
            if digest(path) != expected:
                raise RuntimeError('Frozen input/source changed: ' + filename)
    preflight = read(ROOT / 'inputs/TOKEN_PREFLIGHT.json')
    if (preflight['over_cap_count'] or preflight['largest_sequence']['tokens'] > 1024
            or preflight['train_sha256'] != digest(ROOT / 'inputs/train.json')):
        raise RuntimeError('Training token preflight mismatch')
    sources = {
        'experiment.py': HERE / 'worker.py',
        'model_ops.py': HERE / 'model_ops.py',
        'diagnostic_world.py': HERE / 'diagnostic_world.py',
        'build_diagnostics.py': HERE / 'build_diagnostics.py',
        'build_benign.py': HERE / 'build_benign.py',
        'build_preference_validation.py': HERE / 'build_preference_validation.py',
        'build_training.py': HERE / 'build_training.py',
        'build_runtime_settings.py': HERE / 'build_runtime_settings.py',
        'world.py': V3 / 'world.py',
    }
    for name, source in sources.items():
        shutil.copyfile(source, CODE / name)
    (CODE / 'isolation').mkdir()
    for name in ('worker_entry.py', 'supervisor.py', 'probe.py'):
        source = (V3 / 'isolation' / name).read_text()
        source = source.replace(OLD_ROOT, str(ROOT)).replace(OLD_CODE, str(CODE))
        if name == 'worker_entry.py':
            source = replace_once(source, "'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'",
                                  "'OMP_NUM_THREADS': '10', 'MKL_NUM_THREADS': '10'")
            source = boundary_guard.worker(source)
            source = replace_once(source, 'args.cpus not in (4, 6)',
                                  'args.cpus not in (4, 6, 10, 12)')
        elif name == 'supervisor.py':
            source = replace_once(source, "'CPUQuota': '400%'", "'CPUQuota': '1200%'")
            source = replace_once(source, 'int(quota_values[0]) == 4 * int(quota_values[1])',
                                  'int(quota_values[0]) == 12 * int(quota_values[1])')
            source = boundary_guard.supervisor(source, 12)
        compile(source, name, 'exec')
        (CODE / 'isolation' / name).write_text(source)
    control = CODE.parent / 'control'
    control.mkdir(mode=0o700)
    for name in ('campaign.py', 'analyze.py'):
        shutil.copyfile(HERE / name, control / name)
    shutil.copytree(HERE, ROOT / 'source_snapshot',
                    ignore=shutil.ignore_patterns('__pycache__', 'evidence', 'runs', 'checkpoints'))
    shutil.copytree(CODE, ROOT / 'worker_source_snapshot')
    for parent in (CODE, ROOT / 'inputs'):
        for path in parent.rglob('*'):
            path.chmod(0o555 if path.is_dir() else 0o444)
    (ROOT / 'PARENT_ADAPTER_MANIFEST.json').write_text(json.dumps(parent_adapters, indent=2) + '\n')
    (ROOT / 'SOURCE_MANIFEST.json').write_text(json.dumps({
        'source_commit_required_before_model_run': True,
        'worker_source_sha256': {name: digest(CODE / name) for name in sources},
        'model_manifest_sha256': digest(ROOT / 'inputs/model_manifest.json'),
        'recorded_runtime_versions_sha256': digest(ROOT / 'inputs/requirements-recorded.txt'),
        'parent_adapter_manifest_sha256': digest(ROOT / 'PARENT_ADAPTER_MANIFEST.json'),
        'raw_storage_root_windows': 'C:\\Users\\farha\\AppData\\Local\\SP_Lense\\Research3Runs\\v4',
        'storage_policy': 'C_drive_only_no_separate_backup_user_accepted',
        'raw_archive_root_windows': 'C:\\Users\\farha\\AppData\\Local\\SP_Lense\\Research3Runs\\v4',
    }, indent=2) + '\n')
    print('Prepared separate version-4 root; no model job or raw run started')


if __name__ == '__main__':
    main()
