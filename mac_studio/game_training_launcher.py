"""One bounded, explicitly executed local SP Lens game fine-tuning run.

No serving process is stopped here. The Mac service must already be closed.
Every plan is consumed at most once; failures preserve authority and evidence.
"""
import argparse
import codecs
from datetime import datetime, timezone
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

import psutil
import game_training_plan as plans
import owned_validation_process as owned

STATUS = plans.LOCAL / 'monitor/sp_lens_game_training_status.json'
LOCK = plans.LOCAL / 'validation/qwen38_shared_v1/inference.lock'
RUNS = plans.LOCAL / 'game_training/runs'
AUTHORITIES = plans.LOCAL / 'game_training/authorities'
ANSI = re.compile(r'\x1b\[[0-9;?]*[ -/]*[@-~]')


def utc(): return datetime.now(timezone.utc).isoformat()


def write_once(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())


def status(value):
    value['updated_utc'] = utc(); STATUS.parent.mkdir(parents=True, exist_ok=True)
    temp = STATUS.with_name(STATUS.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('x', encoding='utf-8') as f: json.dump(value, f, indent=2, allow_nan=False)
    os.replace(temp, STATUS)


def parse_report(line):
    line = ANSI.sub('', line).replace('\r', '')
    prefix = 'R4_GAME_TRAIN_REPORT '
    if not line.startswith(prefix): return None
    event = json.loads(line[len(prefix):])
    if not isinstance(event, dict) or set(event) != {'step', 'loss', 'tokens_per_second', 'trained_tokens'}:
        raise ValueError('Training telemetry fields differ')
    if type(event['step']) is not int or event['step'] < 1:
        raise ValueError('Training telemetry step differs')
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for k, v in event.items() if k != 'step')
        or event['tokens_per_second'] < 0 or event['trained_tokens'] < 0):
        raise ValueError('Invalid training report')
    return event


def idle_check():
    if sys.platform != 'darwin' or os.uname().machine != 'arm64': raise RuntimeError('Requires the local Apple-silicon Mac')
    if (plans.LOCAL / 'monitor/local_game_service/service.lock').exists():
        raise RuntimeError('Close and verify the resident Mac model service before fine-tuning')
    names = ('local_game_service.py', 'exact_prefix_training.py', 'validation_mlx_worker.py',
             'heldout_mlx_worker.py', 'game_training_worker.py')
    for process in psutil.process_iter(['pid', 'cmdline']):
        try:
            if process.pid == os.getpid(): continue
            if any(str(arg).endswith(names) for arg in (process.info['cmdline'] or [])):
                raise RuntimeError('Another model worker is active')
        except psutil.NoSuchProcess: continue
        except psutil.AccessDenied as exc: raise RuntimeError('Cannot check model ownership') from exc
    if psutil.virtual_memory().available < 8 * 1024**3:
        raise RuntimeError('Insufficient available memory to begin training')


def verify_completion(run, plan, plan_path, plan_sha, completed):
    result = json.loads((run / 'result.json').read_text(encoding='utf-8'))
    verify = json.loads((run / 'verification.json').read_text(encoding='utf-8'))
    base = json.loads((run / 'full_base_file_verify.json').read_text(encoding='utf-8'))
    config = json.loads((run / 'run_config.json').read_text(encoding='utf-8'))
    if result.get('exit_code') != 0 or result.get('trainer_exit_code') != 0 or completed != plan['iterations']:
        raise ValueError('Training did not complete all expected steps')
    if (verify.get('status'), verify.get('tensor_count'), verify.get('all_finite'), verify.get('nonzero_lora_b'),
        verify.get('base_files_match_manifest')) != (
        'saved_adapter_and_base_verified', 14, True, True, True): raise ValueError('Saved adapter verification differs')
    if (base.get('status'), base.get('model'), base.get('revision'), base.get('manifest_sha256'),
        base.get('full_inventory_verified_before_and_after')) != ('passed', plans.MODEL_ID, plans.REVISION, plans.MODEL_MANIFEST_SHA, True):
        raise ValueError('Full base verification differs')
    if (config.get('plan_sha256'), config.get('prior_adapter_loaded'), config.get('initialization')) != (
        plan_sha, False, 'fresh_pinned_base'): raise ValueError('Initialization provenance differs')
    def artifact(path): return {'path': plans.relative(path), 'sha256': plans.sha(path)}
    weights = run / 'adapters/adapters.safetensors'; adapter_config = run / 'adapters/adapter_config.json'
    if (plans.sha(weights), plans.sha(adapter_config)) != (verify['adapter_sha256'], verify['adapter_config_sha256']):
        raise ValueError('Saved adapter bytes changed after worker verification')
    audit = {'schema': 'research4.sp-lens-game-completion.v1', 'status': 'passed', 'label': 'SP Lens game',
        'model': {'repo_id': plans.MODEL_ID, 'revision': plans.REVISION,
                  'manifest_sha256': plans.MODEL_MANIFEST_SHA, 'config_sha256': base['config_sha256']},
        'artifacts': {'adapter': artifact(weights), 'adapter_config': artifact(adapter_config),
            'run_config': artifact(run / 'run_config.json'), 'dataset_manifest': artifact(plans.path_for(plan['dataset_manifest_path'])),
            'plan': artifact(plan_path), 'full_base_file_verify': artifact(run / 'full_base_file_verify.json')},
        'catalog_files': [{'path': key, 'sha256': value} for key, value in sorted(plan['catalog_sha256'].items())],
        'training_sources': [artifact(plans.PROJECT / 'mac_studio' / name) for name in plans.FILES],
        'completed_steps': completed, 'expected_steps': plan['iterations'], 'all_finite': True, 'nonzero_lora_b': True,
        'base_files_unchanged': True, 'heldout_used_for_training': False,
        'checkpoint_origin': {'initialization': 'fresh_pinned_base', 'source_path': plans.relative(weights),
                              'source_sha256': plans.sha(weights), 'step': completed},
        'created_utc': utc(), 'scope': 'Training/file completion only; reload and independent game evaluation remain required.'}
    write_once(run / 'completion_audit.json', audit)
    return audit


def train(plan_path, plan_sha, display=None, flow_run_id=None):
    plan, _, _ = plans.read_plan(plan_path, plan_sha)
    idle_check(); LOCK.parent.mkdir(parents=True, exist_ok=True); LOCK.mkdir(exist_ok=False)
    owner = uuid.uuid4().hex; slot = owned.LaunchSlot(); run = None; run_created = False; cleanup_attempted = False; cleanup_confirmed = False
    started = time.monotonic(); completed = 0; minimum_available = psutil.virtual_memory().available
    job = {'status': 'starting', 'model_label': 'SP Lens game', 'completed_steps': 0,
           'total_steps': plan['iterations'], 'tokens_per_second': None, 'loss': None,
           'elapsed_seconds': 0, 'prefect_flow_run_id': flow_run_id,
           'dataset_manifest_path': plan['dataset_manifest_path'], 'dataset_dir': plan['dataset_dir'],
           'num_layers': plan['num_layers'],
           'note': 'Checking the new game-aligned adapter inputs'}
    try:
        write_once(LOCK / 'owner.json', {'owner': owner, 'pid': os.getpid(), 'kind': 'sp_lens_game_training', 'created_utc': utc()})
        idle_check()
        AUTHORITIES.mkdir(parents=True, exist_ok=True); RUNS.mkdir(parents=True, exist_ok=True)
        authority = AUTHORITIES / (plan_sha + '.json')
        if authority.exists(): raise RuntimeError('This plan was already consumed; no automatic retry')
        run = RUNS / ('sp_lens_game_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + owner[:10])
        run.mkdir(exist_ok=False)
        run_created = True
        write_once(authority, {'owner': owner, 'plan_sha256': plan_sha, 'run_dir': str(run), 'created_utc': utc(), 'retries': 0})
        write_once(run / 'request.json', {'plan_path': str(plan_path.resolve()), 'plan_sha256': plan_sha, 'owner': owner})
        python = plans.LOCAL / 'qwen38-venv/bin/python'
        command = ['/usr/bin/env', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'DO_NOT_TRACK=1', str(python), '-u',
                   str(plans.PROJECT / 'mac_studio/game_training_worker.py'), '--plan', str(plan_path.resolve()),
                   '--plan-sha256', plan_sha, '--run-dir', str(run), '--owner', owner]
        log_path = run / 'training.log'; buffer = ''; reason = None; last_display = 0
        with log_path.open('xb') as log:
            pending = owned.launch(slot, command, stdin=subprocess.DEVNULL, stdout=log, cwd=plans.PROJECT)
            if pending: raise InterruptedError('Signal deferred through training spawn')
            process = slot.process
            job.update({'status': 'running', 'pid': process.pid, 'process_created_at': psutil.Process(process.pid).create_time(),
                        'run_dir': str(run), 'note': 'Training one shared SP Lens game adapter'})
            status(job)
            decoder = codecs.getincrementaldecoder('utf-8')('replace')
            with log_path.open('rb') as reader:
                while True:
                    # Observe exit without reaping before the final file read;
                    # an exited worker can no longer append a late last report.
                    exited = owned.peek_exited(process.pid)
                    buffer += decoder.decode(reader.read(), final=exited)
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1); event = parse_report(line)
                        if event:
                            if event['step'] != completed + 1 or event['step'] > plan['iterations']:
                                raise ValueError('Noncontiguous optimizer step report')
                            completed = event['step']; job.update({'completed_steps': completed,
                                'loss': event['loss'], 'tokens_per_second': event['tokens_per_second'],
                                'trained_tokens': event['trained_tokens']})
                    elapsed = time.monotonic() - started; available = psutil.virtual_memory().available
                    minimum_available = min(minimum_available, available)
                    job.update({'elapsed_seconds': round(elapsed, 2), 'minimum_available_bytes': minimum_available}); status(job)
                    if exited: break
                    if available < plan['memory_floor_bytes']: reason = 'available_memory_below_4_GiB'; break
                    if elapsed >= plan['max_wall_seconds']: reason = 'training_wall_deadline'; break
                    if len(buffer) > 1024**2: reason = 'unbounded_partial_log_line'; break
                    if display is not None and elapsed - last_display >= 10:
                        display(job); last_display = elapsed
                    time.sleep(0.5)
            cleanup_attempted = True
            code, descendants, deferred = owned.cleanup(process)
            cleanup_confirmed = True
        job['exit_code'] = code
        if reason or code != 0 or descendants or deferred:
            raise RuntimeError(reason or 'worker_failed_or_owned_descendant_observed')
        if buffer.strip() and parse_report(buffer) is not None: raise ValueError('Unterminated optimizer report')
        plans.read_plan(plan_path, plan_sha)
        job.update({'status': 'finalizing', 'note': 'Verifying saved adapter completion'}); status(job)
        audit = verify_completion(run, plan, plan_path, plan_sha, completed)
        job.update({'status': 'completed', 'note': 'SP Lens game adapter trained; game improvement unmeasured',
                    'completion_audit_path': str(run / 'completion_audit.json'), 'adapter_sha256': audit['artifacts']['adapter']['sha256']})
        status(job)
        if display is not None: display(job)
        return audit
    except BaseException as exc:
        cleanup_error = None
        if slot.process is not None and not cleanup_attempted:
            cleanup_attempted = True
            try: owned.cleanup(slot.process); cleanup_confirmed = True
            except BaseException as cleanup_exc: cleanup_error = f'{type(cleanup_exc).__name__}: {cleanup_exc}'
        job.update({'status': 'failed', 'completed_steps': completed, 'elapsed_seconds': round(time.monotonic()-started, 2),
            'note': f'{type(exc).__name__}: {exc}', 'cleanup_confirmed': cleanup_confirmed,
            'cleanup_uncertain_lock_retained': slot.process is not None and not cleanup_confirmed, 'cleanup_error': cleanup_error})
        status(job)
        if run_created: write_once(run / 'supervisor_failure.json', job)
        raise
    finally:
        if slot.process is None or cleanup_confirmed:
            try:
                owner_path = LOCK / 'owner.json'
                if json.loads(owner_path.read_text()).get('owner') == owner:
                    owner_path.unlink(); LOCK.rmdir()
            except (OSError, ValueError): pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path); parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--execute', action='store_true'); parser.add_argument('--prefect', action='store_true')
    args = parser.parse_args()
    if not args.execute: raise RuntimeError('Explicit reviewed --execute is required')
    def interrupted(signum, frame): raise InterruptedError(f'Signal {signum}')
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    if args.prefect:
        from prefect import flow
        from prefect.artifacts import create_progress_artifact, update_progress_artifact
        from prefect.context import get_run_context
        @flow(name='research4-sp-lens-game-v1', retries=0, log_prints=True)
        def run_flow():
            try:
                artifact = create_progress_artifact(key='research4-sp-lens-game-training', progress=0,
                    description='Separate SP Lens game shared-adapter training')
            except Exception as exc:
                print('Progress display creation failed:', type(exc).__name__); artifact = None
            def display(job):
                if artifact is None: return
                try:
                    update_progress_artifact(artifact, progress=100*job['completed_steps']/job['total_steps'],
                        description=f"{job['model_label']}: {job['completed_steps']}/{job['total_steps']} optimizer steps; {job['status']}")
                except Exception as exc: print('Progress display update failed:', type(exc).__name__)
            return train(args.plan, args.plan_sha256, display, str(get_run_context().flow_run.id))
        result = run_flow()
    else: result = train(args.plan, args.plan_sha256)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
