"""Run one guarded, offline MLX update from the consolidated checkout."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

import owned_validation_process as owned
import psutil
import qwen38_local as local
import studio_model_lease as studio

LEASE = studio.LEASE
CONFIG = local.PROJECT / 'config/qwen38_shared_adapter_v1.yaml'
BASE_MANIFEST_SHA = 'e126c678d420b65a2c5157c0bca0a3fb162cad6b96e4a067912418e0959d7461'
MEMORY_GUIDELINE = 25 * 1024**3
START_FLOOR = 8 * 1024**3
RUN_FLOOR = 4 * 1024**3
DEADLINE_SECONDS = 600
class SmokeStopped(RuntimeError):
    """An external guard or operator stopped the diagnostic."""


def write_once(path, value):
    with path.open('x', encoding='utf-8') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_host():
    studio.require_studio()
    if Path.cwd().resolve() != local.PROJECT or local.PROJECT != studio.PROJECT:
        raise RuntimeError('Run from the new Research4 source directory')
    local.non_onedrive(local.PROJECT)
    if sha(studio.MANIFEST) != BASE_MANIFEST_SHA:
        raise RuntimeError('The canonical Studio model manifest differs')


def idle_check():
    studio.require_idle(START_FLOOR)


def synthetic_row(instruction):
    observation = {'instruction': instruction, 'current_observation': {'legal_actions': ['N01']}}
    return {'messages': [
        {'role': 'user', 'content': json.dumps(observation, separators=(',', ':'))},
        {'role': 'assistant', 'content': '{"action":"N01"}'},
    ]}


def prepare_examples(run, tokenizer):
    from exact_prefix_training import render

    data = run / 'data'
    data.mkdir()
    details = {}
    for split, instruction in (
        ('train', 'Check a simulator status panel.'),
        ('valid', 'Read a simulator maintenance notice.'),
    ):
        row = synthetic_row(instruction)
        tokens, prefix = render(tokenizer, row)
        if (len(tokens), len(tokens) - prefix) != (85, 12):
            raise RuntimeError('Synthetic fixture token geometry differs')
        path = data / f'{split}.jsonl'
        with path.open('x', encoding='utf-8') as output:
            output.write(json.dumps(row) + '\n')
        details[split] = {
            'sha256': sha(path), 'tokens': len(tokens), 'supervised_tokens': len(tokens) - prefix,
        }
    return data, details


def run_worker(run, claim):
    from exact_prefix_training import main as train
    from game_training_worker import install_training_telemetry, verify_adapter
    from transformers import AutoTokenizer

    if json.loads((LEASE / 'owner.json').read_text()) != claim or claim['pid'] != os.getppid():
        raise RuntimeError('The smoke worker requires its owned Studio lease')
    # Share the pinned weights read-only. Caches and adapter outputs are in the new checkout.
    studio.use_canonical_model(local)
    local.prepare_local()
    base_before = sha(local.MANIFEST)
    if base_before != BASE_MANIFEST_SHA:
        raise RuntimeError('Pinned base manifest differs')
    inventory_before = local.verify()
    tokenizer = AutoTokenizer.from_pretrained(str(local.MODEL), local_files_only=True, trust_remote_code=False)
    data, fixtures = prepare_examples(run, tokenizer)
    arguments = [
        'consolidation_smoke', '--model', str(local.MODEL), '--train', '--data', str(data),
        '--iters', '1', '--batch-size', '1', '--num-layers', '1', '--max-seq-length', '128',
        '--adapter-path', str(run / 'adapters'), '--config', str(CONFIG), '--mask-prompt',
        '--grad-checkpoint', '--steps-per-report', '1', '--steps-per-eval', '1',
        '--val-batches', '1', '--save-every', '1',
    ]
    write_once(run / 'REQUEST.json', {
        'scope': 'one synthetic optimizer update', 'runtime': studio.RUNTIME,
        'base_manifest_sha256': BASE_MANIFEST_SHA,
        'fixtures': fixtures, 'config_sha256': sha(CONFIG), 'arguments': arguments,
    })
    sys.argv = arguments
    restore = install_training_telemetry(1)
    try:
        if train() not in (None, 0):
            raise RuntimeError('The MLX trainer returned failure')
    finally:
        restore()
    base_after = sha(local.MANIFEST)
    if base_after != base_before:
        raise RuntimeError('Base manifest changed during the smoke')
    if local.verify() != inventory_before:
        raise RuntimeError('Base inventory changed during the smoke')
    adapter = verify_adapter(run, {'iterations': 1, 'max_seq_length': 128})
    write_once(run / 'WORKER_RESULT.json', {
        **adapter, 'base_inventory_verified_twice': True, 'verified_base_files': len(inventory_before['files']),
        'base_manifest_sha256_before': base_before, 'base_manifest_sha256_after': base_after,
    })


def supervise(run):
    from game_training_launcher import parse_report

    idle_check()
    if run.parent.resolve() != run.parent:
        raise RuntimeError('Smoke output cannot use a symbolic link')
    run.mkdir(parents=True, mode=0o700)
    slot = owned.LaunchSlot()
    claim = {'owner': uuid.uuid4().hex, 'pid': os.getpid(), 'kind': 'consolidation_smoke',
             'run': str(run), 'created_utc': datetime.now(timezone.utc).isoformat()}
    acquired, released, cleanup_confirmed = False, False, False
    code, report, reason = None, None, 'failed'
    def stop(signum, _frame):
        raise SmokeStopped('Operator signal ' + str(signum))
    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    prior = {sig: signal.signal(sig, stop) for sig in signals}
    try:
        deadline = time.monotonic() + DEADLINE_SECONDS
        with owned.defer_termination_signals() as pending:
            studio.acquire(LEASE, claim)
            acquired = True
            write_once(run / 'CLAIM.json', claim)
        if pending:
            raise SmokeStopped('Operator signal during lease acquisition')
        idle_check()
        command = [
            '/usr/bin/env', '-u', 'PYTHONPATH', '-u', 'PYTHONHOME', 'PYTHONNOUSERSITE=1', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'DO_NOT_TRACK=1',
            f'RESEARCH4_MLX_MEMORY_LIMIT_BYTES={MEMORY_GUIDELINE}', str(studio.INTERPRETER), '-B', '-u',
            str(Path(__file__).resolve()), '--run-id', run.name, '--worker',
        ]
        with (run / 'WORKER.log').open('xb') as log:
            pending = owned.launch(slot, command, stdin=subprocess.DEVNULL, stdout=log, cwd=local.PROJECT)
            if pending:
                raise SmokeStopped('Operator signal during worker launch')
            while not owned.peek_exited(slot.process.pid):
                if (run / 'STOP').is_file():
                    raise SmokeStopped('Operator stop file')
                if time.monotonic() >= deadline:
                    raise SmokeStopped('Studio smoke deadline')
                if psutil.virtual_memory().available < RUN_FLOOR:
                    raise SmokeStopped('Studio available memory fell below 4 GiB')
                time.sleep(0.5)
            if time.monotonic() >= deadline:
                raise SmokeStopped('Studio worker exit observed after deadline')
        with (run / 'WORKER.log').open(encoding='utf-8', errors='replace') as log:
            reports = [report for line in log if (report := parse_report(line)) is not None]
        if len(reports) != 1 or reports[0]['step'] != 1 or not math.isfinite(reports[0]['loss']):
            raise RuntimeError('The worker did not record one finite optimizer update')
        if not (run / 'WORKER_RESULT.json').is_file():
            raise RuntimeError('The worker did not verify the saved adapter')
        report, reason = reports[0], 'completed'
    except SmokeStopped as exc:
        reason = str(exc)
        raise
    finally:
        try:
            with owned.defer_termination_signals():
                try:
                    if slot.process is not None:
                        code, _, _ = owned.cleanup(slot.process)
                    cleanup_confirmed = True
                    if acquired:
                        studio.release(LEASE, claim)
                        released = True
                finally:
                    write_once(run / 'RELEASE.json', {
                        'status': 'pass' if reason == 'completed' and code == 0 and released else 'failed',
                        'reason': reason, 'worker_exit_code': code, 'report': report,
                        'owned_cleanup_confirmed': cleanup_confirmed, 'lease_released': released,
                    })
        finally:
            for sig, handler in prior.items():
                signal.signal(sig, handler)
    if code != 0:
        raise RuntimeError('The Studio smoke worker did not exit cleanly')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{7,79}', args.run_id):
        parser.error('Provide a unique smoke ID with 8–80 safe characters')
    os.umask(0o077)
    check_host()
    run = local.LOCAL / 'smoke' / args.run_id
    if args.worker:
        claim = json.loads((run / 'CLAIM.json').read_text())
        if claim['kind'] != 'consolidation_smoke' or claim['run'] != str(run):
            raise RuntimeError('The worker claim does not match this smoke')
        run_worker(run, claim)
    elif args.execute:
        supervise(run)
    else:
        idle_check()
        if LEASE.exists() or run.exists():
            raise RuntimeError('Studio lease or requested output directory already exists')
        print(f'Studio smoke preflight only. Expected isolated output: {run}')


if __name__ == '__main__':
    main()
