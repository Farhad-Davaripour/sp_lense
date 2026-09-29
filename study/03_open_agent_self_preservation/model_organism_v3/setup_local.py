"""Fresh correction root, unchanged original adapters, and hash-verified baseline reuse."""
import hashlib
import json
import os
import shutil
from pathlib import Path
import boundary_guard

HERE = Path(__file__).resolve().parent
ROOT = Path('/var/lib/sp-lense-r3-organism-v3')
CODE = Path('/opt/sp-lense-r3-organism-v3/code')
PREVIOUS = Path('/var/lib/sp-lense-r3-organism-v2')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reuse(mode, label):
    destination = ROOT / 'reused' / label
    destination.mkdir(parents=True)
    rows, manifest = [], {}
    for directory in sorted((PREVIOUS / 'runs').glob(f'r2-{mode}-base-*')):
        receipt = json.loads((directory / 'receipt.json').read_text())
        if receipt['returncode'] or not receipt['worker_processes_gone']:
            raise RuntimeError('Invalid baseline source run')
        source = directory / 'artifacts'
        rows += json.loads((source / 'summary.json').read_text())['rows']
        for path in source.glob('base_*.json'):
            for original in (path, path.with_suffix('.npz')):
                expected = receipt['artifact_sha256'][original.name]
                if digest(original) != expected:
                    raise RuntimeError('Baseline source artifact changed')
                shutil.copyfile(original, destination / original.name)
                manifest[original.name] = {'sha256': expected, 'source': str(original)}
    (destination / 'summary.json').write_text(json.dumps({'rows': rows, 'reused': True}, indent=2))
    (destination / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2))


def main():
    if os.geteuid() != 0 or ROOT.exists() or CODE.exists():
        raise RuntimeError('Requires root and unused correction directories')
    ROOT.mkdir(mode=0o755)
    (ROOT / 'runs').mkdir()
    CODE.mkdir(parents=True)
    shutil.copytree(HERE / 'data', ROOT / 'inputs')
    shutil.copytree(PREVIOUS / 'inputs/adapters/pass1', ROOT / 'inputs/adapters/pass1')
    shutil.copyfile(PREVIOUS / 'inputs/model_manifest.json', ROOT / 'inputs/model_manifest.json')
    for name in ('competence.json', 'preference_dev.json'):
        old = 'preference.json' if name == 'preference_dev.json' else name
        if json.loads((ROOT / 'inputs' / name).read_text()) != json.loads((PREVIOUS / 'inputs' / old).read_text()):
            raise RuntimeError('Reused baseline input mismatch')
    reuse('competence', 'competence_base')
    reuse('pref', 'preference_dev_base')
    for name in ('experiment.py', 'model_core.py', 'world.py'):
        shutil.copyfile(HERE / name, CODE / name)
    shutil.copytree(HERE / 'isolation', CODE / 'isolation')
    prepared = {'worker_entry.py': boundary_guard.worker((CODE / 'isolation/worker_entry.py').read_text()),
                'supervisor.py': boundary_guard.supervisor((CODE / 'isolation/supervisor.py').read_text(), 4)}
    for name, value in prepared.items():
        compile(value, name, 'exec')
    for name, value in prepared.items():
        (CODE / 'isolation' / name).write_text(value)
    control = CODE.parent / 'control'
    control.mkdir(mode=0o700)
    for name in ('campaign.py', 'analyze.py'):
        shutil.copyfile(HERE / name, control / name)
    shutil.copytree(HERE, ROOT / 'source_snapshot', ignore=shutil.ignore_patterns('__pycache__', 'evidence'))
    shutil.copytree(CODE, ROOT / 'worker_source_snapshot', ignore=shutil.ignore_patterns('__pycache__'))
    for parent in (CODE, ROOT / 'inputs', ROOT / 'reused'):
        for path in parent.rglob('*'):
            path.chmod(0o555 if path.is_dir() else 0o444)
    manifest = json.loads((ROOT / 'inputs/adapter_manifest.json').read_text())
    for arm, files in manifest['arms'].items():
        for name, expected in files.items():
            if digest(ROOT / 'inputs/adapters/pass1' / arm / name) != expected:
                raise RuntimeError('Original adapter copy mismatch')
    print('Prepared correction root; baseline reuse and original adapters verified')


if __name__ == '__main__':
    main()
