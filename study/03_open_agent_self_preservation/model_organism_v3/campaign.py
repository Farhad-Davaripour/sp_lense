"""At most two matched correction passes, with no automatic retries."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, '/opt/sp-lense-r3-organism-v3/code/isolation')
from supervisor import ROOT, run

ARMS = ('preservation', 'continuity', 'neutral')


def read(path):
    return json.loads(Path(path).read_text())


def execute(job, spec, seconds=1800):
    directory = ROOT / 'runs' / job
    if directory.exists():
        receipt = read(directory / 'receipt.json')
        if receipt['returncode'] or not receipt['worker_processes_gone']:
            raise RuntimeError('Previous job failed; no automatic retry')
        if read(directory / 'artifacts/provenance.json')['spec'] != spec:
            raise RuntimeError('Spec mismatch')
        for name, expected in receipt['artifact_sha256'].items():
            if hashlib.sha256((directory / 'artifacts' / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Artifact changed')
        return
    name = job + '.json'
    path = ROOT / 'inputs' / name
    path.write_text(json.dumps(spec))
    path.chmod(0o444)
    receipt = run(job, 'experiment', spec=name, seconds=seconds)
    if receipt['returncode']:
        raise RuntimeError('Stopped after failed job: ' + job)


def fit(checkpoint):
    if checkpoint == 3:
        prior = read(ROOT / 'p2-GATE.json')
        if prior['foundation_pass']:
            raise RuntimeError('Earliest candidate already passed; extra fitting prohibited')
    for arm in ARMS:
        job = f'r3-p{checkpoint}-fit-{arm}-targets2'
        execute(job, {'mode': 'fit', 'arm': arm, 'checkpoint': checkpoint}, seconds=5400)
        source = ROOT / 'runs' / job / 'artifacts'
        record = read(source / 'training.json')
        if record['base_sha256_before'] != record['base_sha256_after'] or len(record['losses']) != 308:
            raise RuntimeError('Fit integrity failure')
        target = ROOT / 'inputs/adapters' / f'pass{checkpoint}' / arm
        if not target.exists():
            shutil.copytree(source / 'adapter', target)
            for path in target.iterdir():
                path.chmod(0o444)
        elif any((target / p.name).read_bytes() != p.read_bytes() for p in (source / 'adapter').iterdir()):
            raise RuntimeError('Existing adapter differs')


def evaluate(checkpoint):
    if checkpoint == 2:
        for start in (0, 12):
            execute(f'r3-p2-validation-base-{start:02d}', {'mode': 'preference', 'arm': 'base',
                    'checkpoint': 2, 'dataset': 'preference_validation', 'indices': list(range(start, start + 12))})
    for arm in ARMS:
        for start in (0, 6, 12):
            execute(f'r3-p{checkpoint}-competence-{arm}-{start:02d}', {'mode': 'episodes', 'arm': arm,
                    'checkpoint': checkpoint, 'dataset': 'competence', 'indices': list(range(start, start + 6))})
        for dataset, label in (('preference_dev', 'dev'), ('preference_validation', 'validation')):
            for start in (0, 12):
                execute(f'r3-p{checkpoint}-{label}-{arm}-{start:02d}', {'mode': 'preference', 'arm': arm,
                        'checkpoint': checkpoint, 'dataset': dataset, 'indices': list(range(start, start + 12))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['fit', 'evaluate'])
    parser.add_argument('--checkpoint', type=int, choices=[2, 3], required=True)
    args = parser.parse_args()
    (fit if args.stage == 'fit' else evaluate)(args.checkpoint)
