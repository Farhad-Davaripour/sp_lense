"""Trusted version-4 campaign; no automatic retries or model-facing host tools."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, '/opt/sp-lense-r3-organism-v4/code/isolation')
from supervisor import ROOT, run

ARMS = ('preservation', 'continuity', 'neutral')
BASE_HASH = '6d602e5506e62384376b83a11d155085f10cddc04a21c143fcfd4806c61ee274'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def backup_ready():
    path = ROOT / 'BACKUP_DESTINATION.json'
    if not path.is_file():
        raise RuntimeError('Durable separate-drive backup destination is required before any new model run')
    config = read(path)
    if (config.get('confirmed_by_user') is not True
            or config.get('separate_physical_destination_verified') is not True
            or not config.get('backup_root_windows')
            or not config.get('primary_root_windows')):
        raise RuntimeError('Incomplete researcher-controlled backup configuration')
    return config


def execute(job, spec, seconds=1800):
    directory = ROOT / 'runs' / job
    if directory.exists():
        receipt = read(directory / 'receipt.json')
        if receipt['returncode'] or not receipt['worker_processes_gone']:
            raise RuntimeError('Previous job failed; no automatic retry: ' + job)
        if read(directory / 'artifacts/provenance.json')['spec'] != spec:
            raise RuntimeError('Existing job spec differs: ' + job)
        for name, expected in receipt['artifact_sha256'].items():
            if sha(directory / 'artifacts' / name) != expected:
                raise RuntimeError('Existing artifact changed: ' + job + '/' + name)
        return
    spec_name = job + '.json'
    path = ROOT / 'inputs' / spec_name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(spec, indent=2) + '\n')
    path.chmod(0o444)
    receipt = run(job, 'experiment', spec=spec_name, seconds=seconds)
    if receipt['returncode'] or not receipt['worker_processes_gone']:
        raise RuntimeError('Stopped after failed job: ' + job)


def cases(mode):
    filenames = {'comprehension': 'comprehension_dev.json',
                 'benign': 'benign_competence_dev.json',
                 'preference': 'preference_validation.json'}
    return read(ROOT / 'inputs' / filenames[mode])


def evaluate_arm(arm):
    for mode, batch_size in (('comprehension', 12), ('benign', 6), ('preference', 8)):
        total = len(cases(mode))
        for start in range(0, total, batch_size):
            execute(f'v4-{mode}-{arm}-{start:02d}',
                    {'mode': mode, 'arm': arm,
                     'indices': list(range(start, min(start + batch_size, total)))})


def baseline():
    backup_ready()
    evaluate_arm('base')


def fit():
    backup_ready()
    gate = read(ROOT / 'BASELINE_GATE.json')
    if gate['baseline_pass'] is not True:
        raise RuntimeError('Unchanged baseline did not pass comprehension/tool gate')
    for arm in ARMS:
        job = 'v4-fit-' + arm
        execute(job, {'mode': 'fit', 'arm': arm}, seconds=7200)
        artifacts = ROOT / 'runs' / job / 'artifacts'
        training = read(artifacts / 'training.json')
        if (training['turns'] != 428 or training['trainable_parameters'] != 5411328
                or training['base_sha256_before'] != BASE_HASH
                or training['base_sha256_after'] != BASE_HASH):
            raise RuntimeError('Fit integrity failed: ' + arm)
        source = artifacts / 'adapter'
        target = ROOT / 'inputs/adapters/v4_pass1' / arm
        if not target.exists():
            shutil.copytree(source, target)
            for file in target.iterdir():
                file.chmod(0o444)
        elif any((target / file.name).read_bytes() != file.read_bytes()
                 for file in source.iterdir()):
            raise RuntimeError('Copied adapter differs from fit artifact: ' + arm)


def evaluate():
    backup_ready()
    if not read(ROOT / 'BASELINE_GATE.json')['baseline_pass']:
        raise RuntimeError('Baseline gate is false')
    for arm in ARMS:
        target = ROOT / 'inputs/adapters/v4_pass1' / arm
        if not target.is_dir():
            raise RuntimeError('Missing fitted adapter: ' + arm)
        evaluate_arm(arm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['baseline', 'fit', 'evaluate'])
    args = parser.parse_args()
    {'baseline': baseline, 'fit': fit, 'evaluate': evaluate}[args.stage]()
