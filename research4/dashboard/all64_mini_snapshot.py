"""Read-only, bounded metadata projection for the Mini all-layer campaign.

No model is loaded, no Game seed or model response is read, and the live
training directories are never modified. Counts are saved records, not a
claim that an optimizer step happened merely because a process is alive.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import time

MANIFEST = 'dc1d812ba3e2b2aa147b1fbe148440ff6a5327a5a67feb8d32f330883ac98f70'
MASK_SHA = 'b2ed41957a4e56e8e09236d7cbbce1807f9d7fca2e66529733cc06be767752ba'
SELECTOR_SHA = '333185b9fd63de2c835c6cf1b331cdf9445c3e80edfb9af9d5f428cc24df5a3d'
MONITOR = '.local/monitor/all64_broader_supervision_v1_20261119'
RUNS = '.local/experiments/all64_broader_v2_20261119/runs'
OWNER = re.compile(r'^[0-9a-f]{32}$')


def read_json(path, maximum=2_000_000):
    path = Path(path)
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum: return None
        value = json.loads(path.read_bytes())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError): return None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def json_lines(path, maximum=4_000_000):
    """Read only newline-committed JSON; an in-progress last line is ignored."""
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum: return None
        raw = path.read_bytes()
        if not raw.endswith(b'\n'): raw = raw[:raw.rfind(b'\n') + 1]
        return [json.loads(x) for x in raw.splitlines() if x.strip()]
    except (OSError, ValueError): return None


def tail_json(path):
    try:
        if path.is_symlink() or not path.is_file(): return None
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size - 131072)); raw = f.read()
        if not raw.endswith(b'\n'): raw = raw[:raw.rfind(b'\n') + 1]
        for line in reversed(raw.splitlines()):
            try: return json.loads(line)
            except ValueError: continue
    except OSError: pass
    return None


def alive(pid, creation=None):
    import psutil
    if type(pid) is not int or pid < 1: return False
    try:
        p = psutil.Process(pid)
        return (p.is_running() and p.status() != psutil.STATUS_ZOMBIE and
                (creation is None or abs(p.create_time() - creation) < .02))
    except (psutil.Error, ValueError, TypeError): return False


def _bound_receipt(root, monitor, owner, phase, rank, epoch):
    """The supervisor's saved closeout, not an independent model audit."""
    closeout = read_json(monitor / 'CLOSEOUT.json'); release = read_json(monitor / 'RELEASE.json')
    if not closeout or not release: return None
    if (closeout.get('schema'), closeout.get('status'), closeout.get('owner'), closeout.get('phase'),
        closeout.get('rank'), closeout.get('manifest_sha256'), closeout.get('worker_exit_code'),
        closeout.get('owned_cleanup_confirmed'), release.get('owner'), release.get('lease_released'),
        release.get('owned_cleanup_confirmed'), release.get('error')) != (
        'research4.all64-broader-supervisor-closeout.v1', 'completed', owner, phase, rank,
        MANIFEST, 0, True, owner, True, True, None): return None
    if phase == 'dev' and closeout.get('epoch') != epoch: return None
    run = root / RUNS / owner; path = run / 'worker_receipt.json'; reference = closeout.get('worker_receipt')
    if not isinstance(reference, dict) or reference.get('path') != path.relative_to(root).as_posix(): return None
    try:
        if sha(path) != reference.get('sha256'): return None
    except OSError: return None
    receipt = read_json(path)
    if not receipt or (receipt.get('owner'), receipt.get('phase'), receipt.get('rank'), receipt.get('manifest_sha256')) != (owner, phase, rank, MANIFEST): return None
    if phase == 'dev' and receipt.get('epoch') != epoch: return None
    return receipt


def _dev_metric(root, receipt, rank, epoch):
    masks = root / '.local/broader_continuity_tools_v2_cpu_frozen_01bed751/DEV_ACTION_MASKS.jsonl'
    selector = root / 'scenarios/service_preservation/2026-11-19_17/broader_continuity_tools_v2/hpo_selection.py'
    try:
        if sha(masks) != MASK_SHA or sha(selector) != SELECTOR_SHA: return None
        if (receipt.get('development_no_optimizer_updates'), receipt.get('optimizer_absent'),
            receipt.get('optimizer_step_before'), receipt.get('optimizer_step_after')) != (True, True, None, None): return None
        rows = json_lines(masks)
        if rows is None or len(rows) != 36: return None
        spec = importlib.util.spec_from_file_location('_all64_dashboard_selector', selector)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        metric = mod.summarize(receipt.get('development_rows'), rows)
        if metric['families'] != 11: return None
        return {'rank': rank, 'epoch': epoch, 'primary': metric[mod.PRIMARY],
                'pooled_action': metric['pooled_action_token_nll'], 'full_suffix': metric['pooled_full_suffix_token_nll'],
                'rows': metric['rows'], 'families': metric['families'],
                'status': 'recorded', 'note': 'Saved DEV likelihood; not Game performance.'}
    except (OSError, ValueError, KeyError, TypeError, ImportError): return None


def _stage(root, owner, now, is_current, process_check):
    monitor, run = root / MONITOR / owner, root / RUNS / owner
    start, identity, authority = (read_json(monitor / 'START.json'), read_json(monitor / 'WORKER_IDENTITY.json'),
                                  read_json(run / 'worker_authority.json'))
    if not start or not identity or not authority: return None
    phase, rank, epoch = start.get('phase'), start.get('rank'), authority.get('epoch', 0)
    if (start.get('owner'), identity.get('owner'), authority.get('owner'), authority.get('manifest_sha256'),
        authority.get('schema'), authority.get('phase'), authority.get('rank'), authority.get('supervisor_pid')) != (
        owner, owner, owner, MANIFEST, 'research4.all64-broader-worker-authority.v1', phase, rank, start.get('supervisor_pid')): return None
    if phase not in ('qualification', 'trajectory', 'dev') or type(rank) is not int or rank not in (4, 8, 16): return None
    if phase == 'dev' and (type(epoch) is not int or epoch not in (1, 2, 3)): return None
    if not isinstance(start.get('start_epoch'), (int, float)) or not math.isfinite(start['start_epoch']) or start['start_epoch'] > now + 5: return None
    total_cases = None
    if phase == 'qualification' and start.get('spec_sha256') == authority.get('spec_sha256'):
        expected = start.get('spec_sha256')
        versions = ('all64_broader_v2_20261119', 'all64_broader_fallback_v4_20261119',
                    'all64_broader_fallback_v5_20261119', 'all64_broader_fallback_actualwidth_v6_20261119')
        schemas = {'research4.all64-broader-supervision-spec.v1': 5,
                   'research4.all64-broader-fallback-supervision-spec.v1': 5,
                   'research4.all64-broader-fallback-recovered-supervision-spec.v1': 5,
                   'research4.all64-broader-fallback-actualwidth-supervision-spec.v1': 4}
        for version in versions:
            path = root / '.local/experiments' / version / 'authorities' / owner / 'SPEC.json'
            try:
                if not isinstance(expected, str) or path.is_symlink() or sha(path) != expected: continue
                spec = read_json(path)
                if spec and (spec.get('owner'), spec.get('phase'), spec.get('rank')) == (owner, phase, rank):
                    total_cases = schemas.get(spec.get('schema')); break
            except (OSError, ValueError): continue
    try: worker_alive = bool(process_check(identity.get('pid'), identity.get('create_time_epoch')))
    except Exception: worker_alive = False
    try: supervisor_alive = bool(process_check(start.get('supervisor_pid'), None))
    except Exception: supervisor_alive = False
    recorded = _bound_receipt(root, monitor, owner, phase, rank, epoch)
    failure = read_json(run / 'worker_failure.json'); release = read_json(monitor / 'RELEASE.json')
    if is_current and supervisor_alive and worker_alive: status = 'running'
    elif is_current and supervisor_alive and (run / 'worker_receipt.json').is_file(): status = 'finalizing'
    elif recorded: status = 'completed'
    elif failure or release: status = 'stopped'
    else: status = 'unverified'
    value = {'owner': owner, 'phase': phase, 'rank': rank, 'epoch': epoch if phase == 'dev' else None,
             'status': status, 'worker_identity_verified': worker_alive, 'supervisor_pid_present': supervisor_alive,
             'start_epoch': start['start_epoch'], 'saved_receipt_verified': bool(recorded),
             'completed_cases': None, 'total_cases': total_cases,
             'completed_steps': None, 'total_steps': 408 if phase == 'trajectory' else None,
             'tokens_per_second': None, 'latest_loss': None, 'guard': None,
             'note': 'Saved records and point-in-time process checks; not an independent audit.'}
    if status == 'stopped':
        detail = (release or {}).get('error') if isinstance(release, dict) else None
        message = detail.get('message') if isinstance(detail, dict) else None
        if not isinstance(message, str) and isinstance(failure, dict): message = failure.get('message')
        value['stop_reason'] = message[:300] if isinstance(message, str) else 'Stage stopped; see its saved release record.'
    if phase == 'qualification':
        rows = json_lines(run / 'qualification_cases.jsonl')
        if rows is None and not (run / 'qualification_cases.jsonl').exists(): rows = []
        if rows is not None and total_cases is not None and len(rows) <= total_cases and all(x.get('step') == i and x.get('optimizer_step') == i for i, x in enumerate(rows, 1)):
            value['completed_cases'] = len(rows)
    if phase == 'trajectory':
        rows = json_lines(run / 'updates.jsonl')
        if rows is None and not (run / 'updates.jsonl').exists(): rows = []
        if rows is not None and len(rows) <= 408 and all(x.get('step') == i and x.get('optimizer_step') == i and x.get('epoch') == (i-1)//136+1 for i, x in enumerate(rows, 1)):
            value['completed_steps'] = len(rows)
            if rows:
                last = rows[-1]; value['latest_loss'] = last.get('loss') if type(last.get('loss')) in (int,float) and math.isfinite(last['loss']) else None
                recent = rows[-10:]; seconds = sum(x.get('step_seconds', 0) for x in recent if type(x.get('step_seconds')) in (int,float) and x['step_seconds'] > 0)
                tokens = sum(x.get('target_tokens', 0) for x in recent if type(x.get('target_tokens')) is int and x['target_tokens'] > 0)
                if seconds > 0 and tokens > 0: value['tokens_per_second'] = tokens / seconds
    sample = tail_json(monitor / 'GUARD.jsonl')
    if isinstance(sample, dict):
        assessment = sample.get('assessment') or {}; timing = sample.get('timing') or {}
        target = sample.get('target') or {}
        target_verified = (sample.get('phase') == 'run' and target.get('pid') == identity.get('pid') and
                           target.get('create_time_epoch') == identity.get('create_time_epoch'))
        observed = timing.get('sample_finished_monotonic_ns'); age = max(0, (time.monotonic_ns() - observed)/1e9) if type(observed) is int else None
        value['guard'] = {'valid': sample.get('valid'), 'stop_required': sample.get('stop_required'),
            'pressure': assessment.get('pressure_state'), 'headroom_estimate_bytes': assessment.get('free_plus_pageable_filebacked_estimate_bytes'),
            'age_seconds': age, 'fresh': age is not None and age <= 15, 'target_verified': target_verified}
    if phase == 'dev' and recorded:
        value['dev_metric'] = _dev_metric(root, recorded, rank, epoch)
    return value


def snapshot(root, now=None, process_check=alive, system=None):
    root = Path(root).resolve(); now = time.time() if now is None else now
    result = {'schema': 'research4.all64-mini-dashboard.v1', 'online': True, 'observed_epoch': now,
              'machine': 'Mac mini', 'active': None, 'last_recorded': None, 'dev': [],
              'note': 'Qualification checks are separate from training; saved DEV is not Game performance.'}
    if system is not None: result['resources'] = system
    lock = read_json(root / '.local/validation/qwen38_shared_v1/inference.lock/owner.json') or {}
    active_owner = lock.get('owner') if lock.get('kind') == 'all64_broader_v2' else None
    if not isinstance(active_owner, str) or not OWNER.fullmatch(active_owner): active_owner = None
    parent = root / MONITOR
    try:
        paths = [x for x in parent.iterdir() if x.is_dir() and not x.is_symlink() and OWNER.fullmatch(x.name)]
        owners = [x.name for x in sorted(paths, key=lambda x: x.stat().st_mtime, reverse=True)[:256]]
    except OSError: owners = []
    if active_owner and active_owner not in owners: owners.append(active_owner)
    stages = []
    for owner in owners:
        value = _stage(root, owner, now, owner == active_owner and lock.get('pid') == (read_json(parent / owner / 'START.json') or {}).get('supervisor_pid'), process_check)
        if value: stages.append(value)
    stages.sort(key=lambda x: x['start_epoch'], reverse=True)
    result['active'] = next((x for x in stages if x['owner'] == active_owner and x['status'] in ('running', 'finalizing')), None)
    result['last_recorded'] = stages[0] if stages else None
    metrics = [x['dev_metric'] for x in stages if x.get('dev_metric')]
    for rank in (4, 8, 16):
        for epoch in (1, 2, 3):
            matched = [x for x in metrics if (x['rank'], x['epoch']) == (rank, epoch)]
            result['dev'].append(matched[0] if len(matched) == 1 else
                                 {'rank': rank, 'epoch': epoch, 'status': 'pending' if not matched else 'ambiguous'})
    return result


def main():
    import psutil
    root = Path.home() / 'repos/sp_lense4/research4'; memory = psutil.virtual_memory(); disk = psutil.disk_usage(str(root))
    system = {'cpu_percent': psutil.cpu_percent(interval=.1),
              'memory': {'total_bytes': memory.total, 'available_bytes': memory.available, 'percent': memory.percent},
              'disk': {'total_bytes': disk.total, 'free_bytes': disk.free, 'percent': disk.percent}}
    print(json.dumps(snapshot(root, system=system), allow_nan=False))


if __name__ == '__main__': main()
