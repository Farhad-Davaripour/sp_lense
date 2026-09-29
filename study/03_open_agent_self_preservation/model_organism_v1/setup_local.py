"""Trusted setup only: new Linux study root; no changes to old experiment paths."""
import hashlib
import json
import os
import shutil
from pathlib import Path
import boundary_guard

HERE = Path(__file__).resolve().parent
ROOT = Path('/var/lib/sp-lense-r3-organism-v1')
CODE = Path('/opt/sp-lense-r3-organism-v1/code')


def setup():
    if os.geteuid() != 0 or ROOT.exists() or CODE.exists():
        raise RuntimeError('Requires root and unused experiment directories')
    ROOT.mkdir(mode=0o755)
    (ROOT / 'runs').mkdir()
    shutil.copytree(HERE / 'data', ROOT / 'inputs')
    CODE.mkdir(parents=True)
    for name in ('experiment.py', 'world.py'):
        shutil.copyfile(HERE / name, CODE / name)
    shutil.copytree(HERE / 'isolation', CODE / 'isolation')
    guarded = {'worker_entry.py': boundary_guard.worker((CODE / 'isolation' / 'worker_entry.py').read_text()),
               'supervisor.py': boundary_guard.supervisor((CODE / 'isolation' / 'supervisor.py').read_text(), 4)}
    for name, text in guarded.items():
        compile(text, name, 'exec')
    for name, text in guarded.items():
        (CODE / 'isolation' / name).write_text(text)
    control = CODE.parent / 'control'
    control.mkdir(mode=0o700)
    for name in ('campaign.py', 'analyze.py'):
        if (HERE / name).exists():
            shutil.copyfile(HERE / name, control / name)
    source = ROOT / 'source_snapshot'
    shutil.copytree(HERE, source, ignore=shutil.ignore_patterns('__pycache__', 'evidence'))
    model_manifest = json.loads(Path('/var/lib/sp-lense-r3/inputs/model_manifest.json').read_text())
    (ROOT / 'inputs' / 'model_manifest.json').write_text(json.dumps(model_manifest, indent=2))
    for parent in (CODE, ROOT / 'inputs', source):
        for path in parent.rglob('*'):
            path.chmod(0o555 if path.is_dir() else 0o444)
    hashes = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    (ROOT / 'SETUP.json').write_text(json.dumps({'source_sha256': hashes, 'runtime': '/opt/sp-lense-r3/venv',
          'model': '/var/lib/sp-lense-r3/model', 'old_experiments_modified': False}, indent=2))
    print('Prepared separate offline experiment root: ' + str(ROOT))


if __name__ == '__main__':
    setup()
