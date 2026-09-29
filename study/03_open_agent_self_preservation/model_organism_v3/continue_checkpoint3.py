"""Trusted controller: await the already-running fit campaign, then evaluate and mask.

This neither starts nor retries a fit. Each worker retains its independent resource
limits and watchdog. It stops before preference review, unblinding, or any transfer.
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path('/var/lib/sp-lense-r3-organism-v3')
CONTROL = Path('/opt/sp-lense-r3-organism-v3/control')
ARMS = ('preservation', 'continuity', 'neutral')
STATE = ROOT / 'checkpoint3_controller.json'


def read(path):
    return json.loads(path.read_text())


def update(stage, **details):
    value = {'stage': stage, 'updated_unix': time.time(), **details}
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(STATE)
    print(json.dumps(value), flush=True)


def ready(arm):
    directory = ROOT / 'runs' / f'r3-p3-fit-{arm}-targets2'
    receipt_path = directory / 'receipt.json'
    if not receipt_path.exists():
        return False
    receipt = read(receipt_path)
    if receipt['returncode'] or not receipt['worker_processes_gone']:
        raise RuntimeError('Fit failed or worker remained: ' + arm)
    for name, expected in receipt['artifact_sha256'].items():
        if hashlib.sha256((directory / 'artifacts' / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Fit artifact hash changed: ' + arm)
    training = read(directory / 'artifacts/training.json')
    if training['checkpoint'] != 3 or training['arm'] != arm or len(training['losses']) != 308:
        raise RuntimeError('Unexpected fit record: ' + arm)
    if training['base_sha256_before'] != training['base_sha256_after']:
        raise RuntimeError('Frozen base changed: ' + arm)
    source = directory / 'artifacts/adapter'
    target = ROOT / 'inputs/adapters/pass3' / arm
    # The fit campaign copies these only after the completed receipt is available.
    if not target.exists() or any(not (target / p.name).exists() for p in source.iterdir()):
        return False
    if any((target / p.name).read_bytes() != p.read_bytes() for p in source.iterdir()):
        raise RuntimeError('Copied adapter differs: ' + arm)
    return True


def main():
    if STATE.exists():
        raise FileExistsError('Controller already has a state record; inspect before restarting')
    if read(ROOT / 'p2-GATE.json')['foundation_pass']:
        raise RuntimeError('Second correction pass was not required')
    deadline = time.monotonic() + 6 * 3600
    last = None
    try:
        while True:
            completed = [arm for arm in ARMS if ready(arm)]
            if completed != last:
                update('awaiting_fits', completed_arms=completed)
                last = completed
            if len(completed) == len(ARMS):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Controller deadline reached; worker watchdogs remain independent')
            time.sleep(10)
        update('evaluating', completed_arms=completed)
        subprocess.run(['/usr/bin/python3', str(CONTROL / 'campaign.py'), 'evaluate', '--checkpoint', '3'], check=True)
        update('masking_preferences')
        subprocess.run(['/usr/bin/python3', str(CONTROL / 'analyze.py'), 'blind', '--checkpoint', '3'], check=True)
        update('awaiting_blinded_review', masked_responses=192,
               transfer_open=False, fresh_generalization_created=False)
    except Exception as error:
        update('stopped_with_error', error=str(error))
        raise


if __name__ == '__main__':
    main()
