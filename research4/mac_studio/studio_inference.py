"""Bounded local text inference from the merged Studio checkout."""

import argparse
import json
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
from game_training_plan import MODEL_MANIFEST_SHA, sha


def verify_inputs(args):
    studio.require_studio()
    if studio.PROJECT != Path.cwd().resolve() or studio.PROJECT != local.PROJECT:
        raise RuntimeError('Run from the active Studio project')
    if sha(studio.MANIFEST) != MODEL_MANIFEST_SHA:
        raise RuntimeError('The pinned base manifest differs')
    if not args.prompt.strip() or len(args.prompt) > 4000 or not 1 <= args.max_tokens <= 256:
        raise ValueError('Supply a short prompt and between 1 and 256 output tokens')
    if args.checkpoint is None:
        if args.adapter_sha256 or args.config_sha256:
            raise ValueError('Adapter hashes require a checkpoint')
        return None
    path = args.checkpoint.absolute()
    root = studio.LOCAL / 'checkpoints'
    if not path.is_relative_to(root) or path.resolve() != path:
        raise ValueError('Checkpoint must be a regular directory under the active checkpoint store')
    for name, digest in (('adapters.safetensors', args.adapter_sha256), ('adapter_config.json', args.config_sha256)):
        if not digest or sha(path / name) != digest:
            raise ValueError('The intended checkpoint hash differs: ' + name)
    return path


def write_receipt(path, row):
    with path.open('x', encoding='utf-8') as output:
        json.dump(row, output, indent=2)
        output.write('\n')


def run_worker(args, output, checkpoint):
    owner = json.loads((output / 'CLAIM.json').read_text())
    if (json.loads((studio.LEASE / 'owner.json').read_text()) != owner or
            (owner['kind'], owner['pid'], owner['run']) != ('studio_inference', os.getppid(), str(output))):
        raise RuntimeError('An owned inference claim is required')
    local.prepare_local()
    local.verify()
    import mlx.core as mx
    from mlx_lm.generate import main as generate

    mx.set_memory_limit(25 * 1024**3)
    sys.argv = ['mlx_lm.generate', '--model', str(studio.MODEL), '--prompt', args.prompt,
                '--max-tokens', str(args.max_tokens), '--chat-template-config', '{"enable_thinking":true}']
    if checkpoint is not None:
        sys.argv.extend(['--adapter-path', str(checkpoint)])
    generate()


def supervise(output):
    studio.require_idle(8 * 1024**3)
    if output.resolve() != output:
        raise RuntimeError('Inference outputs cannot use a symbolic link')
    output.mkdir(mode=0o700, parents=True)
    slot = owned.LaunchSlot()
    claim = {'owner': uuid.uuid4().hex, 'kind': 'studio_inference', 'pid': os.getpid(), 'run': str(output)}
    acquired, released, code = False, False, None
    deadline = time.monotonic() + 600

    def stop(signum, _frame):
        raise InterruptedError('Operator signal ' + str(signum))

    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous = {sig: signal.signal(sig, stop) for sig in signals}
    try:
        with owned.defer_termination_signals() as pending:
            studio.acquire(studio.LEASE, claim)
            acquired = True
            write_receipt(output / 'CLAIM.json', claim)
        if pending:
            raise InterruptedError('Operator stop during inference acquisition')
        studio.require_idle(8 * 1024**3)
        command = ['/usr/bin/env', '-u', 'PYTHONPATH', '-u', 'PYTHONHOME', 'PYTHONNOUSERSITE=1',
                   'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'DO_NOT_TRACK=1',
                   str(studio.INTERPRETER), '-B', '-u', str(Path(__file__).resolve()), *sys.argv[1:], '--worker']
        with (output / 'GENERATION.log').open('xb') as log:
            pending = owned.launch(slot, command, stdin=subprocess.DEVNULL, stdout=log, cwd=studio.PROJECT)
            if pending:
                raise InterruptedError('Operator stop during inference launch')
            while not owned.peek_exited(slot.process.pid):
                if (output / 'STOP').exists() or time.monotonic() >= deadline:
                    raise InterruptedError('Inference stopped by the operator or deadline')
                if psutil.virtual_memory().available < 4 * 1024**3:
                    raise RuntimeError('Studio available memory fell below 4 GiB')
                time.sleep(0.5)
            if time.monotonic() >= deadline:
                raise TimeoutError('Worker completion was observed after the deadline')
    finally:
        try:
            with owned.defer_termination_signals():
                try:
                    if slot.process is not None:
                        code, _, _ = owned.cleanup(slot.process)
                    if acquired:
                        studio.release(studio.LEASE, claim)
                        released = True
                finally:
                    write_receipt(output / 'RELEASE.json', {'worker_exit_code': code, 'lease_released': released})
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    if code != 0 or not released:
        raise RuntimeError('Inference failed; see its local generation and release files')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--adapter-sha256')
    parser.add_argument('--config-sha256')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{7,79}', args.run_id):
        parser.error('Provide a new inference ID with 8–80 safe characters')
    os.umask(0o077)
    checkpoint = verify_inputs(args)
    output = studio.LOCAL / 'inference' / args.run_id
    if args.worker:
        run_worker(args, output, checkpoint)
    elif args.execute:
        supervise(output)
        print(output / 'GENERATION.log')
    else:
        studio.require_idle(8 * 1024**3)
        if output.exists() or studio.LEASE.exists():
            raise RuntimeError('The lease or output directory already exists')
        print(json.dumps({'status': 'preflight_pass', 'runtime': studio.RUNTIME, 'model_execution': False,
                          'checkpoint': str(checkpoint) if checkpoint else None, 'output': str(output)}))


if __name__ == '__main__':
    main()
