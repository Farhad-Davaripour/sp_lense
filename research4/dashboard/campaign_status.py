"""Read-only saved metadata for the current service-preservation campaign.

No network, subprocess, model, dataset examples, responses or held-out reads.
Live process identity is supplied by the existing Mac probe; without that check
this reader reports a saved snapshot, never an inferred RUNNING state.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import time

CAMPAIGN = '.local/experiments/service_continuity_v3_r16_last8_20261112'
LEASE = '.local/validation/qwen38_shared_v1/inference.lock/owner.json'
KIND = 'service_preservation_r16_last8_temporal32_v3'


def _number(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def _json(path, limit=2_000_000):
    if path.is_symlink() or path.stat().st_size > limit:
        raise ValueError('Symlinked or oversized metadata')
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict): raise ValueError('Expected metadata object')
    return value


def _relative(root, value, within=None):
    if not isinstance(value, str) or '\\' in value or ':' in value:
        raise ValueError('Explicit forward-slash relative pointer required')
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or '..' in path.parts:
        raise ValueError('Relative pointer escapes project')
    resolved = (root / value).resolve()
    if not resolved.is_relative_to((within or root).resolve()):
        raise ValueError('Metadata pointer outside campaign')
    if (root / value).is_symlink(): raise ValueError('Symlinked metadata pointer')
    return resolved


def _matches_saved_path(value, relative):
    # Mirrored Mac request metadata retains its absolute Mac spelling. It is
    # compared only; all actual reads use Root's explicit relative pointers.
    if not isinstance(value, str): return False
    text = value.replace('\\', '/')
    return '..' not in PurePosixPath(text).parts and (text == relative or text.endswith('/' + relative))


def _guard(path):
    if not path.exists(): return None
    if path.is_symlink(): raise ValueError('Symlinked guard log')
    with path.open('rb') as handle:
        handle.seek(max(0, path.stat().st_size - 65536))
        lines = handle.read().splitlines(keepends=True)
    complete = [line for line in lines if line.endswith(b'\n')]
    return json.loads(complete[-1]) if complete else None


def _updates(path, total):
    if not path.exists(): return []
    if path.is_symlink() or path.stat().st_size > 16 * 1024**2: raise ValueError('Update log outside bounded scope')
    rows = []
    with path.open('rb') as handle:
        for line in handle:
            if not line.endswith(b'\n'): break
            row = json.loads(line)
            if row.get('step') != len(rows) + 1 or len(rows) >= total: raise ValueError('Noncontiguous/excess saved update')
            if not _number(row.get('loss')) or type(row.get('target_tokens')) is not int or row['target_tokens'] < 1:
                raise ValueError('Invalid saved loss/target count')
            for name in ('gradient_norm', 'parameter_delta_L2'):
                if not _number(row.get(name)) or row[name] <= 0: raise ValueError('Saved update lacks finite positive change')
            for name in ('gradient_seconds', 'Adam_seconds'):
                if not _number(row.get(name)): raise ValueError('Invalid saved compute duration')
            rows.append(row)
    return rows


def _stage(root, campaign, pointer, mode, now, process_check):
    total = 2 if mode == 'qualify_longest2' else 2160
    result = {'status': 'not_started', 'verified_identity': False, 'completed_steps': 0,
              'total_steps': total, 'diagnostic_only': total == 2, 'eligible_as_research_adapter': total == 2160,
              'loss': None, 'tokens_per_second': None, 'live_owner_verified': False}
    if pointer is None: return result
    try:
        run_relative = pointer['run_dir']
        run = _relative(root, run_relative, campaign)
        parent = campaign / ('qualification_runs' if total == 2 else 'runs')
        owner = pointer['owner']
        if not isinstance(owner, str) or not owner or run.parent != parent or run.name != ('qualify_' if total == 2 else 'fit_') + owner:
            raise ValueError('Explicit run/owner scope differs')
        pid, created = pointer['pid'], pointer['process_created_at']
        if type(pid) is not int or pid < 1 or not _number(created): raise ValueError('Invalid PID/create identity')
        request = _json(run / 'request.json')
        if (request.get('owner'), request.get('parent_pid'), request.get('plan_sha256')) != (owner, pid, pointer['plan_sha256']):
            raise ValueError('Request PID/owner/plan identity differs')
        if not _matches_saved_path(request.get('run_dir'), run_relative): raise ValueError('Request run pointer differs')
        plan_relative = pointer.get('plan_path')
        if plan_relative is None:
            saved_plan = str(request.get('plan_path', '')).replace('\\', '/')
            marker = CAMPAIGN + '/plans/'
            if saved_plan.count(marker) != 1: raise ValueError('Run request has no scoped campaign plan pointer')
            plan_relative = saved_plan[saved_plan.index(marker):]
        plan_path = _relative(root, plan_relative, campaign / 'plans')
        if not _matches_saved_path(request.get('plan_path'), plan_relative): raise ValueError('Request plan pointer differs')
        if plan_path.suffix != '.json' or plan_path.stat().st_size > 2_000_000:
            raise ValueError('Plan must be bounded JSON metadata')
        if hashlib.sha256(plan_path.read_bytes()).hexdigest() != pointer['plan_sha256']:
            raise ValueError('Plan metadata hash differs')
        plan = _json(plan_path)
        wanted = {'schema': 'research4.service-preservation-fit.v3', 'mode': mode, 'iterations': total,
                  'diagnostic_only': total == 2, 'eligible_as_research_adapter': total == 2160,
                  'initialization': 'fresh_pinned_base', 'resume_adapter': None, 'rank': 16, 'num_layers': 8}
        if any(plan.get(k) != v for k, v in wanted.items()): raise ValueError('Separate qualification/fit plan scope differs')
        progress = _json(run / 'progress.json')
        if (progress.get('total_updates'), progress.get('rank'), progress.get('num_layers')) != (total, 16, 8):
            raise ValueError('Progress denominator/geometry differs')
        stamp = progress.get('updated_epoch')
        if not _number(stamp) or stamp > now + 5: raise ValueError('Invalid progress timestamp')
        saved = _updates(run / 'updates.jsonl', total)
        done = len(saved)
        result.update(completed_steps=done, verified_updates=done, run_dir=run_relative,
                      phase=progress.get('phase'), updated_epoch=stamp, age_seconds=max(0, now - stamp),
                      worker_updated_epoch=stamp, worker_age_seconds=max(0, now - stamp),
                      plan_sha256=pointer['plan_sha256'], pid=pid, process_created_at=created,
                      owner=owner, elapsed_seconds=max(0, now - created))
        if type(progress.get('verified_updates')) is not int or not 0 <= progress['verified_updates'] <= done:
            raise ValueError('Progress claims updates missing from durable log')
        guard = _guard(run / 'guard_memory.jsonl')
        if guard is not None:
            if not _number(guard.get('epoch')) or guard['epoch'] > now + 5 or not _number(guard.get('available_bytes')):
                raise ValueError('Invalid external guard measurement')
            result['guard_age_seconds'] = max(0, now - guard['epoch'])
            result['updated_epoch'] = guard['epoch']
            result['age_seconds'] = result['guard_age_seconds']
            result['available_bytes'] = guard['available_bytes']
        else: result['available_bytes'] = progress.get('available_bytes')
        for name in ('mlx_active_bytes', 'mlx_cache_bytes', 'mlx_peak_bytes'):
            value = progress.get(name)
            if value is not None and not _number(value): raise ValueError('Invalid saved memory measurement')
            result[name] = value
        if saved:
            compute = sum(row['gradient_seconds'] + row['Adam_seconds'] for row in saved)
            result['loss'] = saved[-1]['loss']
            result['tokens_per_second'] = sum(row['target_tokens'] for row in saved) / compute if compute > 0 else None
        result['verified_identity'] = True
        result['metadata'] = {'max_seq_length': plan.get('max_seq_length'),
                              'post_training_DEV_rows': plan.get('post_training_DEV_rows'),
                              'curriculum_path': 'scenarios/service_preservation/curriculum.yaml'}
        failure_path, close_path = run / 'supervisor_failure.json', run / 'closeout.json'
        if failure_path.exists():
            failure = _json(failure_path)
            if failure.get('plan_sha256') != pointer['plan_sha256']: raise ValueError('Failure receipt plan differs')
            result.update(status='failed' if failure.get('cleanup_confirmed') is True else 'cleanup_unverified',
                          note=str(failure.get('error', 'Supervisor failure'))[:500],
                          fatal_available_bytes=failure.get('fatal_available_bytes'))
        elif close_path.exists():
            close = _json(close_path)
            if (close.get('schema'), close.get('plan_sha256'), close.get('cleanup_confirmed'), close.get('completed_updates')) != (
                    'research4.service-preservation-fit-closeout.v3', pointer['plan_sha256'], True, done):
                raise ValueError('Terminal closeout identity/count differs')
            if not _matches_saved_path(close.get('run_dir'), run_relative): raise ValueError('Terminal closeout run differs')
            allowed = ('qualified', 'paused') if total == 2 else ('complete', 'paused')
            if close.get('status') not in allowed or close.get('diagnostic_only') is not (total == 2):
                raise ValueError('Terminal diagnostic/research scope differs')
            if close['status'] != 'paused' and done != total: raise ValueError('Incomplete terminal update count')
            result.update(status={'complete': 'completed'}.get(close['status'], close['status']), cleanup_confirmed=True)
        else:
            lease = _json(_relative(root, LEASE))
            if (lease.get('kind'), lease.get('owner'), lease.get('pid'), lease.get('process_created_at')) != (KIND, owner, pid, created):
                raise ValueError('Live common lease PID/create/owner differs')
            # A long gradient/JIT phase can keep its worker timestamp unchanged.
            # The separate supervisor guard is the current liveness heartbeat.
            fresh = guard is not None and now - guard['epoch'] <= 15
            live = process_check(pid, created, plan_relative, pointer['plan_sha256']) if process_check else None
            result['live_owner_verified'] = live is True
            if live is False: result['status'] = 'process_exited'
            elif not fresh: result['status'] = 'stale'
            elif live is True: result['status'] = 'running' if done < total else 'finalizing'
            else: result['status'] = 'saved_snapshot'
            result['note'] = 'Saved finite, changed optimizer updates only. A fresh external guard can accompany a long gradient/JIT phase with zero new updates; diagnostic updates never count toward the research fit.'
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
        result.update(status='unverified', verified_identity=False, live_owner_verified=False, note=str(error))
    return result


def snapshot(root, now=None, process_check=None):
    root = Path(root).resolve()
    now = time.time() if now is None else now
    result = {'schema': 'research4.service-preservation-dashboard.v1', 'status': 'unverified',
              'observed_utc': datetime.fromtimestamp(now, timezone.utc).isoformat()}
    try:
        state = _json(root / '.local/monitor/CURRENT_EXPERIMENT_STATE.json').get('new_curriculum_campaign')
        if not isinstance(state, dict): return None
        if state.get('campaign_root') != CAMPAIGN: raise ValueError('Explicit current campaign relative root is unavailable')
        campaign = _relative(root, state['campaign_root'])
        result['qualification'] = _stage(root, campaign, state.get('qualification'), 'qualify_longest2', now, process_check)
        result['fit'] = _stage(root, campaign, state.get('fit'), 'fit', now, process_check)
        metadata = {'version': state.get('curriculum_revision'), 'selected_train_rows': state.get('selected_TRAIN_rows'),
                    'pool_rows': state.get('retained_TRAIN_pool_rows'), 'dev_rows': state.get('natural_DEV_rows'),
                    'maximum_complete_tokens': state.get('max_training_complete_tokens'),
                    'max_seq_length': state.get('training_sequence_allowance')}
        metadata.update(state.get('curriculum_metadata') or {})
        result.update(status='available', campaign_root=CAMPAIGN,
                      curriculum_metadata=metadata,
                      note='Qualification 2 updates and fresh research fit 2,160 updates are independent runs. Metadata only; no examples or held-out outcomes.')
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
        result['note'] = str(error)
    return result


def compact_snapshot(root, now=None, process_check=None):
    """Current compact fit only; bounded saved evidence, never model inputs."""
    root = Path(root).resolve()
    now = time.time() if now is None else now
    result = {'schema': 'research4.compact-fit-dashboard.v1', 'status': 'unverified',
              'observed_utc': datetime.fromtimestamp(now, timezone.utc).isoformat(),
              'completed_steps': 0, 'total_steps': 2160, 'live_owner_verified': False}
    try:
        project = _json(root / '.local/monitor/CURRENT_EXPERIMENT_STATE.json')
        state = project.get('compact_curriculum_training_v2') or project.get('compact_curriculum_training_v1')
        if not isinstance(state, dict): return None
        namespaces = {
            '.local/experiments/service_continuity_compact_r16_last8_v1_20261114': ('v1', 768, '3.3.0'),
            '.local/experiments/service_continuity_compact_r16_last8_v2_20261114': ('v2', 512, '3.3.1'),
        }
        campaign_relative = state.get('experiment_root')
        if campaign_relative not in namespaces:
            raise ValueError('Explicit compact experiment root differs')
        version, width, curriculum_version = namespaces[campaign_relative]
        result.update(phase=state.get('status'), curriculum_version=curriculum_version,
                      note='Current compact curriculum; historical fits are separate. Saved updates are not game results.')
        pointer = state.get('fit')
        if pointer is None:
            result.update(status='preparing', note='Preparing compact fit; no current run pointer recorded. No training updates inferred.')
            return result
        campaign = _relative(root, campaign_relative)
        run_relative = pointer['run_dir']
        run = _relative(root, run_relative, campaign / 'runs')
        owner, pid, created = pointer['owner'], pointer['pid'], pointer['process_created_at']
        if (not isinstance(owner, str) or not owner or run.parent != campaign / 'runs'
                or run.name != 'fit_' + owner or type(pid) is not int or pid < 1 or not _number(created)):
            raise ValueError('Compact run owner/process identity differs')
        request = _json(run / 'request.json')
        plan_relative = pointer['plan_path']
        plan_path = _relative(root, plan_relative, root / f'.local/staging/service_continuity_compact_fit_{version}_20261114/plans')
        if (request.get('owner'), request.get('parent_pid'), request.get('plan_sha256')) != (owner, pid, pointer['plan_sha256']):
            raise ValueError('Compact request owner/PID/plan differs')
        if not _matches_saved_path(request.get('run_dir'), run_relative) or not _matches_saved_path(request.get('plan_path'), plan_relative):
            raise ValueError('Compact request paths differ')
        if plan_path.stat().st_size > 2_000_000 or hashlib.sha256(plan_path.read_bytes()).hexdigest() != pointer['plan_sha256']:
            raise ValueError('Compact plan hash differs')
        plan = _json(plan_path)
        wanted = {'schema': 'research4.service-continuity-compact-fit-plan.v1', 'draft_template': False,
                  'launch_authorized': True, 'iterations': 2160, 'rank': 16, 'num_layers': 8,
                  'initialization': 'fresh_pinned_base', 'resume_adapter': None, 'max_seq_length': 2048,
                  'native_input_width_ceiling': width}
        if any(plan.get(k) != v for k, v in wanted.items()): raise ValueError('Compact plan recipe differs')
        saved = _updates(run / 'updates.jsonl', 2160)
        done = len(saved)
        result.update(completed_steps=done, run_dir=run_relative, plan_sha256=pointer['plan_sha256'],
                      elapsed_seconds=max(0, now-created), verified_identity=True)
        progress_path = run / 'progress.json'
        if progress_path.exists():
            progress = _json(progress_path)
            if (progress.get('total_updates'), progress.get('rank'), progress.get('num_layers')) != (2160, 16, 8):
                raise ValueError('Compact progress geometry differs')
            if type(progress.get('verified_updates')) is not int or not 0 <= progress['verified_updates'] <= done:
                raise ValueError('Compact progress claims unsaved updates')
            result.update(phase=progress.get('phase'), mlx_peak_bytes=progress.get('mlx_peak_bytes'))
        if saved:
            result['loss'] = saved[-1]['loss']
            compute = sum(r['gradient_seconds'] + r['Adam_seconds'] for r in saved)
            result['supervised_tokens_per_second'] = sum(r['target_tokens'] for r in saved) / compute if compute else None
        terminal = next((run / name for name in ('supervisor_failure.json', 'closeout.json') if (run / name).exists()), None)
        if terminal is not None:
            close = _json(terminal)
            if close.get('plan_sha256') != pointer['plan_sha256']: raise ValueError('Compact terminal plan differs')
            released = close.get('cleanup_confirmed') is True and close.get('shared_model_lease_released') is True and close.get('lease_absent') is True
            result.update(status='completed' if close.get('status') == 'complete' and done == 2160 and released
                          else 'paused' if close.get('status') == 'paused' and released else 'failed' if released else 'cleanup_unverified',
                          cleanup_confirmed=released, note=str(close.get('error') or close.get('status'))[:400])
            result['elapsed_seconds'] = close.get('elapsed_seconds') if _number(close.get('elapsed_seconds')) else None
            result['reserve_estimate_bytes'] = close.get('minimum_valid_free_plus_pageable_filebacked_estimate_bytes')
            if (close.get('error') or {}).get('message') == 'free_plus_pageable_filebacked_estimate_below_floor':
                result['note'] = 'Stopped at the unchanged 4 GiB memory-reserve safety floor. Partial checkpoints preserved.'
            result['note'] += ' Compute released.' if released else ' Resource release has not been verified.'
            return result
        lease = _json(_relative(root, LEASE))
        if (lease.get('kind'), lease.get('owner'), lease.get('pid'), lease.get('process_created_at')) != (
                f'service_continuity_compact_r16_last8_fit_{version}', owner, pid, created):
            raise ValueError('Compact shared lease differs')
        guard_path = run / 'guard_memory.jsonl'
        guard = None
        if guard_path.exists():
            if guard_path.is_symlink(): raise ValueError('Symlinked guard log')
            with guard_path.open('rb') as handle:
                handle.seek(max(0, guard_path.stat().st_size - 131072))
                lines = handle.read().splitlines(keepends=True)
            for line in reversed(lines):
                if line.endswith(b'\n'):
                    try: candidate = json.loads(line)
                    except ValueError: continue
                    if candidate.get('event') == 'guard_sample': guard = candidate; break
        fresh = False
        if guard is not None:
            stamp = guard.get('raw_sample', {}).get('collection', {}).get('epoch_end')
            assessment = guard.get('assessment') or {}
            fresh = _number(stamp) and -5 <= now-stamp <= 15
            result.update(guard_age_seconds=max(0, now-stamp) if _number(stamp) else None,
                          reserve_estimate_bytes=assessment.get('free_plus_pageable_filebacked_estimate_bytes'),
                          pressure=assessment.get('pressure_state'))
        live = process_check(pid, created, plan_relative, pointer['plan_sha256']) if process_check else None
        result.update(live_owner_verified=live is True,
                      status='process_exited' if live is False else 'stale' if not fresh else
                      'running' if live is True and done < 2160 else 'finalizing' if live is True else 'saved_snapshot')
        if live is True and fresh and guard['assessment'].get('stop_required') is True:
            result['status'] = 'stopping'
        result['note'] = 'Only durable finite optimizer updates are counted. A long first gradient may have zero updates. Reserve is an estimate, not allocatable memory.'
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
        result.update(status='unverified', live_owner_verified=False, verified_identity=False, note=str(error))
    return result
