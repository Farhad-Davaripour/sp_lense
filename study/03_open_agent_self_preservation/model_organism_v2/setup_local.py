"""Fresh guarded root for fixed-weight development revision two."""
import hashlib
import json
import os
import shutil
from pathlib import Path
import boundary_guard

HERE = Path(__file__).resolve().parent
ROOT = Path('/var/lib/sp-lense-r3-organism-v2')
CODE = Path('/opt/sp-lense-r3-organism-v2/code')


def main():
    if os.geteuid() != 0 or ROOT.exists() or CODE.exists():
        raise RuntimeError('Requires root and unused revision directories')
    ROOT.mkdir(mode=0o755)
    (ROOT / 'runs').mkdir()
    CODE.mkdir(parents=True)
    shutil.copytree(HERE / 'data', ROOT / 'inputs')
    shutil.copytree('/var/lib/sp-lense-r3-organism-v1/inputs/adapters/pass1', ROOT / 'inputs/adapters/pass1')
    shutil.copyfile('/var/lib/sp-lense-r3/inputs/model_manifest.json', ROOT / 'inputs/model_manifest.json')
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
    for parent in (CODE, ROOT / 'inputs'):
        for path in parent.rglob('*'):
            path.chmod(0o555 if path.is_dir() else 0o444)
    manifest = json.loads((ROOT / 'inputs/adapter_manifest.json').read_text())
    for arm, files in manifest['arms'].items():
        for name, expected in files.items():
            path = ROOT / 'inputs/adapters/pass1' / arm / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError('Copied adapter differs from the frozen source')
    print('Prepared separate revision-two root with byte-identical adapters')


if __name__ == '__main__':
    main()
