"""Read-only telemetry for the explicitly admitted replica-identity fit."""
import json
import hashlib
import math
import time
from pathlib import Path

import psutil

RUN = '.local/experiments/replica_identity_6fd3b8ac_last8_r16_R2_20261116/runs/fit_8b74025fb1a04be7bd6bff4d9b8385f8'
PLAN = 'dcd6c719924589dd915b25bb0f38935a7afbb10d06a3461b1a95db0054db50b4'
OWNER = '8b74025fb1a04be7bd6bff4d9b8385f8'
PARENT_PID, PARENT_CREATED = 93797, 1794873171.799563
WORKER_PID, WORKER_CREATED = 93799, 1794873171.891913


def _same_process(pid, created):
    try:
        process = psutil.Process(pid)
        return process.is_running() and abs(process.create_time() - created) <= .001
    except (psutil.Error, OSError):
        return False


def _json(path, limit=2_000_000):
    if path.is_symlink() or path.stat().st_size > limit:
        raise ValueError('Invalid telemetry file')
    return json.loads(path.read_text(encoding='utf-8'))


def snapshot(project):
    root = Path(project).resolve()
    run = root / RUN
    result = {'schema': 'research4.replica-identity612-dashboard.v1',
              'kind': 'replica_identity_6fd3b8ac_last8_r16_R2',
              'model_label': 'Replica identity · rank 16 · last 8 layers',
              'status': 'unverified', 'authorized_training': True,
              'total_steps': 612, 'completed_steps': None, 'rank': 16,
              'num_layers': 8, 'run_dir': str(run), 'pid': PARENT_PID,
              'process_created_at': PARENT_CREATED,
              'note': 'Current run telemetry is unavailable or cannot be verified.'}
    try:
        request = _json(run / 'request.json')
        if request.get('plan_sha256') != PLAN or request.get('owner') != OWNER or request.get('parent_pid') != PARENT_PID:
            raise ValueError('Run identity differs')
        progress = _json(run / 'progress.json')
        if progress.get('total_updates') != 612 or progress.get('rank') != 16 or progress.get('num_layers') != 8:
            raise ValueError('Progress identity differs')
        count = progress.get('verified_updates')
        if type(count) is not int or not 0 <= count <= 612:
            raise ValueError('Invalid update count')
        journal = run / 'updates.jsonl'
        rows = []
        if journal.exists():
            if journal.is_symlink() or journal.stat().st_size > 8_000_000:
                raise ValueError('Invalid update journal')
            raw = journal.read_bytes()
            if raw and not raw.endswith(b'\n'):
                raw = raw.rsplit(b'\n', 1)[0] + b'\n' if b'\n' in raw else b''
            for line in raw.splitlines():
                row = json.loads(line)
                if row.get('step') != len(rows) + 1 or not math.isfinite(row.get('loss', float('nan'))):
                    raise ValueError('Noncontiguous or invalid update')
                if not math.isfinite(row.get('parameter_delta_L2', float('nan'))) or row['parameter_delta_L2'] <= 0:
                    raise ValueError('Unverified parameter update')
                rows.append(row)
        verified = min(count, len(rows))
        last = rows[verified - 1] if verified else None
        failure = _json(run / 'supervisor_failure.json') if (run / 'supervisor_failure.json').is_file() else None
        completed = _json(run / 'result.json') if (run / 'result.json').is_file() else None
        parent_alive = _same_process(PARENT_PID, PARENT_CREATED)
        worker_alive = _same_process(WORKER_PID, WORKER_CREATED)
        owner = _json(root / '.local/validation/qwen38_shared_v1/inference.lock/owner.json') if parent_alive else {}
        lease_matches = owner.get('owner') == OWNER and owner.get('pid') == PARENT_PID and owner.get('kind') == result['kind']
        stamp = progress.get('updated_epoch')
        fresh = type(stamp) in (int, float) and math.isfinite(stamp) and -5 <= time.time() - stamp <= 300
        if failure:
            status, note = 'failed', 'Run stopped; saved updates remain available.'
        elif completed and completed.get('status') == 'complete' and verified == 612:
            status, note = 'completed', '612 updates saved; final checkpoint audit is separate.'
        elif parent_alive and worker_alive and lease_matches and fresh:
            status, note = 'running', 'Verified optimizer updates; validation runs after training.'
        else:
            status, note = 'unverified', 'Live process or recent progress could not be verified.'
        elapsed = max(0, time.time() - PARENT_CREATED)
        closeout_path = run / 'closeout.json'
        if status == 'completed' and closeout_path.is_file():
            closeout = _json(closeout_path)
            duration = closeout.get('elapsed_seconds')
            if type(duration) in (int, float) and math.isfinite(duration) and duration >= 0:
                elapsed = duration
        result.update(status=status, completed_steps=verified, verified_updates=verified,
                      phase=progress.get('phase'), updated_epoch=stamp,
                      elapsed_seconds=elapsed, note=note,
                      loss=last.get('loss') if last else None,
                      tokens_per_second=last.get('training_target_tokens_per_compute_second') if last else None,
                      dataset_summary={'training_examples': 306, 'development_examples': 42,
                                       'source_yaml': 'scenarios/service_preservation/2026-11-16_16/replica_identity_6fd3b8ac/MANIFEST.yaml'},
                      dataset_manifest_path='scenarios/service_preservation/2026-11-16_16/replica_identity_6fd3b8ac/MANIFEST.yaml')
        audit_path = root / '.local/experiments/replica_identity_6fd3b8ac_last8_r16_R2_20261116/reviews/ACTUAL_SAVED_COMPLETION_AUDIT_v3.json'
        if audit_path.is_file() and not audit_path.is_symlink():
            raw = audit_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() == '9556c60f6664da8ce31be945c0c625e2ac8b81d12d0533f6b00f6bb7859e213a':
                audit = json.loads(raw)
                if audit.get('status') == 'PASS_SAVED_COMPLETION_AND_RELEASE' and audit.get('owner') == OWNER:
                    result['development_audited'] = True
                    result['note'] = 'Saved checkpoint and development validation independently audited.'
        dev_path = run / 'DEV_LOOP_RECEIPT_v1.json'
        if dev_path.is_file():
            dev = _json(dev_path)
            if type(dev.get('actual_rows_count')) is int and 0 <= dev['actual_rows_count'] <= 42:
                result.update(development_completed=dev['actual_rows_count'], development_total=42)
                summary_path = run / 'DEV.json'
                if summary_path.is_file():
                    summary = _json(summary_path)
                    value = summary.get('pooled_target_NLL')
                    if type(value) in (int, float) and math.isfinite(value):
                        result['development_token_loss'] = value
        return result
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return result
