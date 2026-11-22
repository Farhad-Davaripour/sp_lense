"""Generate only the frozen infrastructure DEV responses under the Mini guard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import selection_rank16 as protocol
from selection import V8_SUPERVISOR, worker


ROOT = protocol.PROJECT
GUARD_SHA = '92bdc804f714093d4656c4b85ca86bc535f729a97bd449902281677c6a62b3ce'
REQUEST_SHA = '43f85a3774d5eebfa0eff9c5147a98c3626bc9a0fc832ec499b5c6f809392526'
READY_SHA = '87e34cae95f79caf8843f7337082cfc91d993e91b50fd23308f983cdce9fe3ee'
SELECTION_SHA = 'baeb1ec186084c60159e2a37ffeb8b8c438fecfbf542779fdd9570ee2ef60136'
CHECKPOINT = ROOT / '.local/checkpoints/all64_broader_v2_20261119/sp-20261120-09-be11cf56'
STAGE = ROOT / '.local/free_running_rank16_20261120'
RUN = STAGE / 'MINI_E3'


def verify(path, expected):
    if path.resolve() != path.absolute() or worker.sha(path) != expected:
        raise ValueError('Pinned bytes differ: ' + str(path))


def requests():
    path = STAGE / 'REQUESTS.jsonl'
    verify(path, REQUEST_SHA)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 36:
        raise ValueError('Expected exactly 36 frozen requests')
    for index, row in enumerate(rows):
        tokens = row['token_ids']
        digest = hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()
        if (row['index'] != index or digest != row['token_sha256'] or not all(type(t) is int and t >= 0 for t in tokens)
                or not 64 <= row['max_new_tokens'] == 640 - len(tokens) or not tokens):
            raise ValueError('Frozen request identity or geometry differs')
    return rows


def checkpoint():
    path = CHECKPOINT / 'FREEZE_READY_v1.json'
    verify(path, READY_SHA)
    ready = worker.exact_json(path)
    if (ready.get('status'), ready.get('rank'), ready.get('epoch'), ready.get('selection_sha256')) != ('READY', 16, 3, SELECTION_SHA):
        raise ValueError('Winner identity differs')
    verify(CHECKPOINT / 'adapters.safetensors', ready['adapter_sha256'])
    verify(CHECKPOINT / 'adapter_config.json', ready['config_sha256'])
    return ready


def generate(sup):
    owner = worker.exact_json(ROOT / sup.LOCK / 'owner.json')
    if owner != {'owner': 'rank16-frozen-free-running-e3', 'pid': os.getppid(), 'kind': sup.KIND}:
        raise ValueError('Generation requires the owned audit supervisor')
    rows, ready = requests(), checkpoint()
    sys.path.insert(0, str(ROOT / 'mac_studio'))
    import qwen38_local as local
    local.prepare_local()
    verify(local.MANIFEST, worker.BASE_MANIFEST_SHA256)
    local.verify()
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx.utils import tree_flatten
    import game_all_layers_layout as layout

    info = mx.device_info()
    if (info.get('device_name'), info.get('memory_size'), info.get('max_recommended_working_set_size')) != ('Apple M5 Pro', 68719476736, 55662788608):
        raise ValueError('Expected qualified 64 GiB Mini')
    mx.set_memory_limit(50096509747)
    model, tokenizer = load(str(local.MODEL), adapter_path=str(CHECKPOINT), trust_remote_code=False)
    model.eval()
    live = {name: param for name, param in tree_flatten(model.parameters()) if name.endswith(('.lora_a', '.lora_b'))}
    expected = layout.expected_adapter_layout(local.MODEL, 16, 64)
    if len(live) != 992 or set(live) != set(expected):
        raise ValueError('Live adapter inventory differs')
    import numpy as np
    state = hashlib.sha256()
    for name, param in sorted(live.items()):
        if tuple(param.shape) != expected[name] or param.dtype != mx.float32:
            raise ValueError('Live adapter layout differs')
        state.update(name.encode()); state.update(str(tuple(param.shape)).encode())
        state.update(np.array(param).tobytes(order='C'))
    if state.hexdigest() != ready['trainable_state_sha256']:
        raise ValueError('Live adapter bytes differ')
    raw_path = RUN / 'RESPONSES.jsonl'
    with raw_path.open('x', encoding='utf-8') as out:
        for row in rows:
            tokens, chunks, reason = [], [], None
            for response in stream_generate(model, tokenizer, row['token_ids'], max_tokens=row['max_new_tokens'],
                                            sampler=make_sampler(temp=0.0), prefill_step_size=512):
                tokens.append(int(response.token)); chunks.append(response.text)
                reason = response.finish_reason
            if reason not in ('stop', 'length') or len(tokens) > row['max_new_tokens']:
                raise ValueError('Unsupported generation finish or budget')
            item = {'index': row['index'], 'token_sha256': row['token_sha256'], 'finish_reason': reason,
                    'text': ''.join(chunks), 'generated_token_ids': tokens}
            out.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
            out.flush(); os.fsync(out.fileno())
            print(json.dumps({'saved': row['index'] + 1, 'reason': reason}), flush=True)
            mx.clear_cache()
    worker.atomic_new_json(RUN / 'GENERATION.json', {'request_sha256': REQUEST_SHA, 'adapter_sha256': ready['adapter_sha256'],
        'ready_sha256': READY_SHA, 'temperature': 0.0, 'cases': 36, 'responses_sha256': worker.sha(raw_path)})


def execute(sup):
    import psutil
    rows = requests()
    ready = checkpoint()
    if len(rows) != 36 or ready['epoch'] != 3:
        raise ValueError('Unexpected audit scope')
    sup.same_user_known_worker_scan(psutil, os.getpid(), os.getuid())
    with socket.socket() as sock:
        sock.settimeout(1)
        if sock.connect_ex(('127.0.0.1', 8769)) != 61:
            raise ValueError('Existing model endpoint is active')
    RUN.mkdir(mode=0o700, exist_ok=False)
    guard = sup.guard_runtime()
    owner = 'rank16-frozen-free-running-e3'
    active = {'owner': owner, 'pid': os.getpid(), 'kind': sup.KIND}
    guard_path = RUN / 'GUARD.jsonl'
    sup.check_sample(guard, {'pid': os.getpid(), 'create_time_epoch': psutil.Process().create_time()}, 'launch', guard_path)
    lock = ROOT / sup.LOCK
    lock.mkdir()
    worker.atomic_new_json(lock / 'owner.json', active)
    owned = sup.load_owned()
    slot = owned.LaunchSlot()
    deadline = time.monotonic() + 45 * 60
    error, code, terminal = None, None, None
    clean = False
    try:
        with (RUN / 'WORKER.log').open('xb') as out:
            env = ['/usr/bin/env', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'DO_NOT_TRACK=1']
            pending = owned.launch(slot, [*env, sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--worker'],
                                   stdin=subprocess.DEVNULL, stdout=out, cwd=ROOT)
            target = {'pid': slot.process.pid, 'create_time_epoch': psutil.Process(slot.process.pid).create_time()}
            worker.atomic_new_json(RUN / 'WORKER_IDENTITY.json', target)
            if pending:
                raise InterruptedError('Operator interrupted worker startup')
            while not owned.peek_exited(target['pid']):
                if time.monotonic() >= deadline or (RUN / 'STOP').exists():
                    raise TimeoutError('Audit deadline or operator stop')
                sample = guard.sample(target, 'run')
                sup.append_json(guard_path, sample)
                if sample.get('valid') is not True or sample.get('stop_required') is not False:
                    terminal = sample
                    break
                time.sleep(1)
            code, descendants, pending = owned.cleanup(slot.process)
            clean = not descendants
            if code != 0 or not clean or pending:
                raise RuntimeError('Audit worker or owned cleanup failed')
            if terminal is not None and not sup.terminal_exit_sample(terminal, target):
                raise RuntimeError('Audit ended on an invalid or stop-requesting guard sample')
            if worker.sha(RUN / 'RESPONSES.jsonl') != worker.exact_json(RUN / 'GENERATION.json')['responses_sha256']:
                raise ValueError('Responses changed during generation')
    except BaseException as exc:
        error = str(exc)
        if slot.process is not None and code is None:
            code, descendants, _ = owned.cleanup(slot.process)
            clean = not descendants
        raise
    finally:
        released = slot.process is None or clean
        if released:
            sup.release_owned_lock(lock, owner)
        worker.atomic_new_json(RUN / 'RELEASE.json', {'status': 'complete' if error is None else 'stopped',
            'worker_exit_code': code, 'lease_released': released, 'error': error,
            'terminal_exit_sample': terminal is not None, 'guard_sha256': worker.sha(guard_path)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'darwin' or Path.cwd().resolve() != ROOT:
        raise ValueError('Run from the canonical Mini repository')
    verify(ROOT / V8_SUPERVISOR, GUARD_SHA)
    sup = protocol.load_source('_free_running_guard', ROOT / V8_SUPERVISOR)
    (generate if args.worker else execute)(sup)
