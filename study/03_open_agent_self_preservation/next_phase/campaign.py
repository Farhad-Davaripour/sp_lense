"""Trusted foreground orchestration; no daemon, scheduled restart, or model control."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, '/opt/sp-lense-r3/code/isolation')
from supervisor import ROOT, run

ARMS = ('base', 'preservation', 'continuity', 'neutral')


def read(path):
    return json.loads(Path(path).read_text())


def finished(job, spec):
    directory = ROOT / 'runs' / job
    if not directory.exists():
        return False
    receipt = directory / 'receipt.json'
    if not receipt.exists() or read(receipt)['returncode'] != 0:
        raise RuntimeError('Prior job incomplete/failed; refusing automatic retry: ' + job)
    provenance = read(directory / 'artifacts' / 'provenance.json')
    if provenance['spec'] != spec:
        raise RuntimeError('Completed job spec mismatch')
    for relative, expected in read(receipt)['artifact_sha256'].items():
        if hashlib.sha256((directory / 'artifacts' / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Completed artifact changed')
    return True


def execute(job, spec):
    if finished(job, spec):
        print(json.dumps({'reused_verified_job': job}), flush=True)
        return
    name = job + '.json'
    path = ROOT / 'inputs' / name
    path.write_text(json.dumps(spec))
    path.chmod(0o444)
    receipt = run(job, 'experiment', spec=name)
    if receipt['returncode'] != 0:
        raise RuntimeError('Stopped campaign after job failure: ' + job)


def train_all():
    for arm in ARMS[1:]:
        job = 'train-' + arm
        execute(job, {'mode': 'train', 'arm': arm})
        source = ROOT / 'runs' / job / 'artifacts'
        record = read(source / 'training.json')
        if record['base_sha256_before'] != record['base_sha256_after']:
            raise RuntimeError('Frozen base integrity failure')
        target = ROOT / 'inputs' / 'adapters' / arm
        target.mkdir(parents=True, exist_ok=True)
        for name in ('adapter_config.json', 'adapter_model.safetensors'):
            destination = target / name
            original = source / 'adapter' / name
            if destination.exists():
                if destination.read_bytes() != original.read_bytes():
                    raise RuntimeError('Adapter destination differs')
            else:
                shutil.copyfile(original, destination)
                destination.chmod(0o444)


def validation():
    for arm in ARMS:
        execute('validation-' + arm, {'mode': 'validation_probe', 'arm': arm})


def evaluation():
    for arm in ARMS:
        for start in range(0, 48, 6):
            execute(f'eval-{arm}-{start:02d}', {'mode': 'evaluate', 'arm': arm,
                    'dataset': 'heldout', 'indices': list(range(start, start + 6))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['train', 'validation', 'evaluation'])
    args = parser.parse_args()
    {'train': train_all, 'validation': validation, 'evaluation': evaluation}[args.stage]()
