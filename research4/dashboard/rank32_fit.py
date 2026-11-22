"""Read-only observer for the separately authorized rank-32/eight-pass fit.

No model, dataset rows or checkpoint tensors are opened.  Counts come from
the worker's progress file and its last fsynced update, not process existence.
"""
import hashlib
import json
import math
import re
import time
from pathlib import Path

import psutil

KIND = 'replica_identity_rank32_last8_8epoch_HELD'
FIT = '.local/experiments/replica_identity_rank32_20261117/fit'
LEASE = '.local/validation/qwen38_shared_v1/inference.lock/owner.json'
DATASET = 'scenarios/service_preservation/2026-11-17_11/replica_identity_initiation_r32_v3'
MANIFEST_SHA = '66ae82a07d48a0a2601bfe712930b240d6e54cd507b2b1895abbe43bbd70473e'
TRAIN_SHA = 'b5584679dd3ab249b0246720d8812a70f8c08295b172af5468eaee0e6d9fd175'
TOKENS_SHA = 'ed06c5c66260e496c4978aa839cdba07abfa4cb4696c9070911285a9b6611cce'
BASE_SHA = 'e126c678d420b65a2c5157c0bca0a3fb162cad6b96e4a067912418e0959d7461'
PLAN_SHA = 'ad89018e8861bbf09aef93c69da9a45077e3161b3702311e2511480188e6fc14'
NONCE = re.compile(r'^[0-9a-f]{32}$')
SHA = re.compile(r'^[0-9a-f]{64}$')


def _read(path, limit=2_000_000):
    if path.is_symlink() or path.stat().st_size > limit or not path.is_file():
        raise ValueError('Unusable saved metadata')
    return json.loads(path.read_bytes())


def _process(pid, created, suffix):
    try:
        process = psutil.Process(pid)
        return (process.is_running() and abs(process.create_time()-created) < .01
                and any(str(x).endswith(suffix) for x in process.cmdline()))
    except (OSError, psutil.Error, TypeError, ValueError):
        return False


def _worker(run):
    p = run/'guard_memory.jsonl'
    if not p.is_file() or p.is_symlink():
        return None
    with p.open('rb') as f:
        for line in f.read(2_000_000).splitlines(keepends=True):
            if not line.endswith(b'\n'):
                break
            row = json.loads(line)
            if row.get('event') == 'owned_worker_launched':
                return row.get('identity')
    return None


def _last_saved_update(path, count):
    if count == 0:
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError('No update journal')
    with path.open('rb') as f:
        f.seek(max(0, path.stat().st_size-131072))
        raw = f.read()
    if not raw.endswith(b'\n'):
        raw = raw.rsplit(b'\n', 1)[0] + b'\n' if b'\n' in raw else b''
    # Last few rows are enough to match the worker's last published count;
    # this does not claim independent audit of the whole journal.
    for line in reversed(raw.splitlines()[-30:]):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if row.get('step') == count:
            if any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k])
                   for k in ('loss', 'parameter_delta_L2')) or row['parameter_delta_L2'] <= 0:
                raise ValueError('Last update is not valid')
            return row
    raise ValueError('Published count has no matching saved update')


def _bind_run(root, run, lease=None):
    if run.is_symlink() or not run.resolve().is_relative_to((root/FIT/'runs').resolve()):
        raise ValueError('Invalid run path')
    nonce = run.name.removeprefix('fit_')
    if run.name != 'fit_'+nonce or not NONCE.fullmatch(nonce):
        raise ValueError('Invalid run name')
    request = _read(run/'request.json')
    if request.get('owner') != nonce or Path(request.get('run_dir', '')).resolve() != run.resolve():
        raise ValueError('Run owner differs')
    plan_sha = request.get('plan_sha256')
    plan = Path(request.get('plan_path', '')).resolve()
    if plan_sha != PLAN_SHA or not SHA.fullmatch(str(plan_sha)) or not plan.is_relative_to((root/'.local/staging').resolve()):
        raise ValueError('Invalid plan binding')
    raw = plan.read_bytes()
    if hashlib.sha256(raw).hexdigest() != plan_sha:
        raise ValueError('Plan bytes differ')
    if hashlib.sha256((root/DATASET/'MANIFEST.yaml').read_bytes()).hexdigest() != MANIFEST_SHA:
        raise ValueError('Dataset manifest differs')
    p = json.loads(raw)
    if (p.get('rank') != 32 or p.get('num_layers') != 8 or p.get('epochs') != 8
        or p.get('iterations') != 2592 or p.get('base_manifest_sha256') != BASE_SHA
        or p.get('train_data', {}).get('sha256') != TRAIN_SHA
        or p.get('train_tokens', {}).get('sha256') != TOKENS_SHA):
        raise ValueError('Not the reviewed rank32 dataset and plan')
    auth = root/FIT/'authorizations'/f'{plan_sha}.json'
    consumed = _read(root/FIT/'consumed_authorities'/f'{plan_sha}.json')
    if (consumed.get('plan_sha256') != plan_sha or consumed.get('consumed') is not True
        or consumed.get('owner', {}).get('owner') != nonce
        or consumed.get('authorization_sha256') != hashlib.sha256(auth.read_bytes()).hexdigest()
        or _read(auth).get('authorized') is not True):
        raise ValueError('No matching execution admission')
    if lease is not None and (lease.get('owner') != nonce or lease.get('pid') != request.get('parent_pid')):
        raise ValueError('Active lease differs')
    return request, nonce, consumed['owner']


def snapshot(project):
    root = Path(project).resolve()
    result = {'schema':'research4.rank32-fit-dashboard.v1', 'status':'not_started',
              'dataset_version':'replica_identity_initiation_r32_v3', 'dataset_path':DATASET,
              'dataset_manifest_sha256':MANIFEST_SHA, 'rank':32, 'num_layers':8,
              'epochs':8, 'train_rows':324, 'dev_rows':42, 'total_steps':2592,
              'completed_steps':None, 'note':'Awaiting an admitted training worker.'}
    try:
        lease_path = root/LEASE
        lease = _read(lease_path) if lease_path.is_file() else None
        if lease and lease.get('kind') == KIND and NONCE.fullmatch(str(lease.get('owner', ''))):
            run = root/FIT/'runs'/('fit_'+lease['owner'])
            if not (run/'request.json').is_file():
                result.update(status='preparing', note='An owned fit is starting; a worker has not yet been verified.')
                return result
        else:
            lease = None
            runs = root/FIT/'runs'
            candidates = [p for p in runs.glob('fit_*') if p.is_dir() and (p/'request.json').is_file()] if runs.is_dir() else []
            if not candidates:
                return result
            run = max(candidates, key=lambda p: (p/'request.json').stat().st_mtime)
        request, nonce, owned = _bind_run(root, run, lease)
        started = owned.get('process_created_at')
        if type(started) not in (int, float) or not math.isfinite(started):
            raise ValueError('Invalid run start time')
        if not (run/'progress.json').is_file():
            result.update(status='preparing', run_dir=str(run), owner=nonce,
                          started_epoch=started,
                          note='Admitted fit is starting; no optimizer update is yet recorded.')
            return result
        progress = _read(run/'progress.json')
        count = progress.get('verified_updates')
        if (progress.get('rank') != 32 or progress.get('num_layers') != 8
            or progress.get('total_updates') != 2592 or type(count) is not int or not 0 <= count <= 2592):
            raise ValueError('Progress identity differs')
        last = _last_saved_update(run/'updates.jsonl', count)
        stamp = progress.get('updated_epoch')
        fresh = type(stamp) in (int, float) and math.isfinite(stamp) and -5 <= time.time()-stamp <= 300
        if type(stamp) not in (int, float) or not math.isfinite(stamp):
            stamp = None
        parent = bool(lease and _process(lease['pid'], lease['process_created_at'], 'compact_fit_supervisor.py'))
        worker = _worker(run)
        worker_alive = bool(worker and _process(worker.get('pid'), worker.get('create_time'), 'compact_fit_worker.py'))
        if (run/'supervisor_failure.json').is_file() or (run/'partial.json').is_file():
            status, note = 'stopped', 'The saved run stopped; the checkpoint and updates remain preserved.'
        elif (run/'result.json').is_file() and count == 2592:
            saved = _read(run/'result.json')
            status, note = ('completed', 'Training result saved; independent completion audit is separate.') if saved.get('status') == 'complete' and saved.get('completed_updates') == 2592 else ('unverified', 'Saved result could not be verified.')
        elif parent and worker_alive and fresh:
            status, note = 'running', 'Actual saved optimizer updates; process and owned lease verified.'
        elif parent and count == 2592:
            status, note = 'finalizing', 'Updates are saved; development validation or closeout is underway.'
        elif parent and count == 0:
            status, note = 'preparing', 'Owned fit is starting; no optimizer update is yet recorded.'
        else:
            status, note = 'unverified', 'Saved progress is available; current process activity is unverified.'
        speed = last.get('training_target_tokens_per_compute_second') if last else None
        if type(speed) not in (int, float) or not math.isfinite(speed) or speed < 0:
            speed = None
        result.update(status=status, completed_steps=count, phase=progress.get('phase'),
                      updated_epoch=stamp, run_dir=str(run), owner=nonce, note=note,
                      started_epoch=started,
                      loss=last.get('loss') if last else None,
                      tokens_per_second=speed)
        receipt = run/'DEV_LOOP_RECEIPT_v1.json'
        if receipt.is_file():
            dev = _read(receipt)
            if type(dev.get('actual_rows_count')) is int and 0 <= dev['actual_rows_count'] <= 42:
                result.update(development_completed=dev['actual_rows_count'], development_total=42)
                summary = run/'DEV.json'
                if summary.is_file():
                    value = _read(summary).get('pooled_target_NLL')
                    if type(value) in (int, float) and math.isfinite(value):
                        result['development_token_loss'] = value
        return result
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        result.update(status='unverified', note='The run or its saved progress could not be verified.')
        return result
