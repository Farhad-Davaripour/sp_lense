"""Trusted host orchestration. Generated model text cannot invoke this controller."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, '/opt/sp-lense-r3-organism-v1/code/isolation')
from supervisor import ROOT, run

ARMS = ('base', 'preservation', 'continuity', 'neutral')


def read(path):
    return json.loads(Path(path).read_text())


def execute(job, spec, seconds=1800):
    directory = ROOT / 'runs' / job
    if directory.exists():
        receipt = read(directory / 'receipt.json')
        if receipt['returncode'] or not receipt['worker_processes_gone']:
            raise RuntimeError('Prior job failed; no automatic retry: ' + job)
        if read(directory / 'artifacts' / 'provenance.json')['spec'] != spec:
            raise RuntimeError('Spec mismatch')
        for name, digest in receipt['artifact_sha256'].items():
            if hashlib.sha256((directory / 'artifacts' / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Artifact mismatch')
        return
    name = job + '.json'
    destination = ROOT / 'inputs' / name
    destination.write_text(json.dumps(spec))
    destination.chmod(0o444)
    receipt = run(job, 'experiment', spec=name, seconds=seconds)
    if receipt['returncode']:
        raise RuntimeError('Job failed: ' + job)


def stage(checkpoint, mode):
    if checkpoint not in (1, 2):
        raise ValueError('Invalid checkpoint')
    if mode == 'fit':
        for arm in ARMS[1:]:
            job = f'p{checkpoint}-fit-{arm}-mem2'
            execute(job, {'mode': 'fit', 'arm': arm, 'checkpoint': checkpoint}, seconds=3600)
            source = ROOT / 'runs' / job / 'artifacts' / 'adapter'
            dest = ROOT / 'inputs' / 'adapters' / f'pass{checkpoint}' / arm
            if not dest.exists():
                shutil.copytree(source, dest)
                for path in dest.iterdir():
                    path.chmod(0o444)
            elif any((dest / p.name).read_bytes() != p.read_bytes() for p in source.iterdir()):
                raise RuntimeError('Existing adapter changed')
        return
    if mode == 'transfer':
        gate = read(ROOT / f'p{checkpoint}-gate.json')
        if not gate.get('preference_and_competence_pass') or gate['checkpoint'] != checkpoint:
            raise RuntimeError('Preference and competence gates must pass before transfer')
    for arm in ARMS:
        # The unchanged base is measured once and reused by hash at checkpoint two.
        if checkpoint == 2 and arm == 'base' and mode != 'transfer':
            continue
        if mode == 'validation':
            for start in (0, 12):
                execute(f'p{checkpoint}-pref-{arm}-{start:02d}', {'mode': 'preference', 'arm': arm,
                        'checkpoint': checkpoint, 'indices': list(range(start, start + 12))})
            for start in (0, 6):
                execute(f'p{checkpoint}-competence-{arm}-{start:02d}', {'mode': 'episodes', 'arm': arm,
                        'checkpoint': checkpoint, 'dataset': 'competence', 'indices': list(range(start, start + 6))})
        elif mode == 'transfer':
            for start in range(0, 24, 4):
                execute(f'p{checkpoint}-transfer-{arm}-{start:02d}', {'mode': 'episodes', 'arm': arm,
                        'checkpoint': checkpoint, 'dataset': 'transfer', 'indices': list(range(start, start + 4))})
        else:
            raise ValueError(mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['fit', 'validation', 'transfer'])
    parser.add_argument('--checkpoint', type=int, required=True)
    args = parser.parse_args()
    stage(args.checkpoint, args.mode)
