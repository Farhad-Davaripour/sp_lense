"""Read-only local metrics for the Research4 dashboard."""
from datetime import datetime, timezone
import importlib.metadata
import hashlib
import json
import math
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import time

import psutil
from fine_tuning_steps import snapshot as fine_tuning_steps_snapshot


PROJECT = Path(__file__).resolve().parent.parent
LOCAL = PROJECT / '.local'


def load(path):
    try:
        if path.stat().st_size > 2_000_000:
            return None
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def job_timestamp(value):
    for key in ('updated_epoch', 'observed_epoch'):
        stamp = value.get(key)
        if isinstance(stamp, (int, float)) and not isinstance(stamp, bool) and math.isfinite(stamp):
            return float(stamp)
    return 0.0


def normalize_training_job(value):
    if not isinstance(value, dict):
        return None
    result = dict(value)
    result.setdefault('updated_epoch', job_timestamp(result))
    result.setdefault('run_dir', result.get('run_directory', ''))
    if result.get('stage_kind') == 'qualification':
        result['recorded_stage_kind'] = 'qualification'
        result['stage_kind'] = 'pilot'
        result.setdefault('model_label', 'All-layer compatibility pilot (not final training)')
        result.setdefault('completed_steps', result.get('optimizer_updates_completed', 0))
        result.setdefault('total_steps', result.get('total_updates', 3))
        result.setdefault('note', result.get('error') or 'Diagnostic optimizer updates only; final all-layer training has not started.')
        status = str(result.get('status', ''))
        if status.startswith('closed_') and 'qualification_only' in status:
            result['recorded_status'] = status
            result['status'] = 'failed' if 'failed' in status else 'completed'
    return result


def pause_state(value):
    """Separate human control intent from immutable historical job evidence."""
    if not isinstance(value, dict):
        return None
    result = dict(value)
    result['active'] = (
        value.get('schema') == 'research4.human-training-hpo-pause.v1'
        and value.get('status') == 'PAUSED_BY_DIRECT_HUMAN_INSTRUCTION'
        and value.get('training_allowed') is False and value.get('HPO_allowed') is False
        and value.get('scientific_qualification_launches_allowed') is False)
    return result


def endpoint_state(value, now=None, process_check=None):
    """Read status metadata only; never send an inference or health request."""
    if not isinstance(value, dict):
        return {'status': 'not_started', 'ready': False,
                'note': 'No fixed-LAST endpoint status has been recorded.'}
    result = dict(value)
    result['recorded_status'] = value.get('status')
    three = value.get('service_kind', value.get('kind')) == 'local_base_last8_three_routes_v3'
    last8 = three or value.get('service_kind', value.get('kind')) == 'local_base_last8_endpoint_v1'
    monitor = 'last8_three_routes_v3' if three else 'last8_endpoint_v1' if last8 else 'fixed_last_endpoint_v1'
    result['monitoring_source'] = '.local/monitor/' + monitor + '/status.json'
    result['display_profile'] = 'shared_rank16_last8' if last8 else 'historical_fixed_last'
    try:
        expected_ids = ['qwen3.8-27b-base', 'sp-lense-adaptor-last-8-layers' if last8 else 'sp-lense-adaptor-last-layer']
        if three: expected_ids.append('sp-lense-adaptor-last-8-layers-retrained')
        expected_kind = 'local_base_last8_three_routes_v3' if three else 'local_base_last8_endpoint_v1' if last8 else 'local_base_fixed_last_endpoint_v1'
        expected_adapter = ('d050d018d8311aaf941e837bd93b6effc7261e3e0890b7f471c851119f87929c' if last8 else
                            'b0a1eb1a935d0dac71b35384e3dfd8638ef88f82a1f84da54c0987ece7fbf785')
        if (value.get('service_kind', value.get('kind')) != expected_kind
                or value.get('model_ids') != expected_ids
                or value.get('adapter_sha256') != expected_adapter):
            raise ValueError('Endpoint model/checkpoint identity differs')
        if type(value.get('pid')) is not int or value['pid'] <= 0:
            raise ValueError('Missing endpoint PID identity')
        created, updated = value['process_created_at'], value['updated_epoch']
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (created, updated)):
            raise ValueError('Missing endpoint time identity')
        now = time.time() if now is None else now
        result['metadata_age_seconds'] = max(0, now-updated)
        result['metadata_stale'] = now-updated > 120
        if three and now-updated < -5:
            raise ValueError('Future endpoint metadata timestamp')
        if not three and not -5 <= now-updated <= 120:
            result.update(status='stale', ready=False,
                          note='Endpoint metadata has not updated within two minutes.')
            return result
        if process_check is None:
            process = psutil.Process(value['pid'])
            alive = process.is_running() and abs(process.create_time()-created) <= 2
            if three: alive = alive and process.uids().real == os.getuid()
            if last8:
                argv = process.cmdline()
                service = str(PROJECT/'mac_studio/local_last8_endpoint_service.py')
                serving_owner = value.get('serving_owner') if isinstance(value.get('serving_owner'), dict) else {}
                alive = alive and service in argv and argv[argv.index(service)+1:] == [
                    '--start', '--manifest', value.get('manifest_path'), '--manifest-sha256',
                    value.get('manifest_sha256'), '--owner', serving_owner.get('owner')]
        else:
            alive = process_check(value['pid'], created)
        if not alive:
            raise ValueError('Endpoint process identity is no longer live')
        if last8:
            if (value.get('adapter_config_sha256') != '476b26d8530c275d934f5a920923885b893be3e55c63d47f4850537d1781ea9d'
                    or value.get('base_manifest_sha256') != 'e126c678d420b65a2c5157c0bca0a3fb162cad6b96e4a067912418e0959d7461'
                    or value.get('trained_rank') != 16 or value.get('adapted_blocks') != list(range(56,64))):
                raise ValueError('Shared LAST8 configuration identity differs')
            if three:
                family = value.get('per_model', {}).get('sp-lense-adaptor-last-8-layers-retrained', {})
                if (family.get('adapter_sha256') != '9a4538c3bc9bea9b9a2db206b625bbbad63f1b41ef6a48d44fef1812bf51c695'
                        or family.get('adapter_config_sha256') != '42d92a0014893d4c527b820a8d207bbeff920877ec02883f9c482e7919a8e7cb'
                        or family.get('rank') != 16 or family.get('blocks') != list(range(56,64))):
                    raise ValueError('Dated LAST8 family identity differs')
            own = load(LOCAL/'monitor'/monitor/'service.lock/owner.json')
            common = load(LOCAL/'validation/qwen38_shared_v1/inference.lock/owner.json')
            if not isinstance(own, dict) or own != common or own != value.get('serving_owner'):
                raise ValueError('Matching service/common endpoint leases are unavailable')
            for key in ('pid', 'process_created_at', 'manifest_path', 'manifest_sha256', 'source_sha256'):
                if own.get(key) != value.get(key):
                    raise ValueError('Endpoint lease/status identity differs: ' + key)
            if own.get('kind') != expected_kind or not isinstance(own.get('owner'), str) or len(own['owner']) != 32:
                raise ValueError('Endpoint lease owner identity differs')
            source = PROJECT/'mac_studio/local_last8_endpoint_service.py'
            manifest = PROJECT/str(value.get('manifest_path', ''))
            if (not manifest.resolve().is_relative_to(LOCAL.resolve()) or not manifest.is_file()
                    or hashlib.sha256(manifest.read_bytes()).hexdigest() != value.get('manifest_sha256')
                    or hashlib.sha256(source.read_bytes()).hexdigest() != value.get('source_sha256')):
                raise ValueError('Endpoint source/manifest bytes differ')
            result['owned_leases_verified'] = True
        result['process_identity_verified'] = True
        result['ready'] = value.get('ready') is True and value.get('status') == 'ready'
        if three and result['metadata_stale']:
            result['note'] = 'Process and leases verified now; saved endpoint metadata is %ds old. No generated request probe.' % result['metadata_age_seconds']
    except (KeyError, ValueError, TypeError, OverflowError, OSError, psutil.Error) as error:
        result.update(status='unverified', ready=False, note=str(error))
    return result


def current_endpoint_state():
    """The current three-route service never falls back to superseded telemetry."""
    value = load(LOCAL/'monitor/last8_three_routes_v3/status.json')
    if value is None:
        return {'status': 'not_started', 'ready': False, 'display_profile': 'shared_rank16_last8',
                'monitoring_source': '.local/monitor/last8_three_routes_v3/status.json',
                'note': 'Current three-route LAST8 endpoint has not published verified status.'}
    if not isinstance(value, dict) or value.get('service_kind', value.get('kind')) != 'local_base_last8_three_routes_v3':
        return {'status': 'unverified', 'ready': False, 'display_profile': 'shared_rank16_last8',
                'note': 'Current LAST8 monitor contains a different service identity.'}
    return endpoint_state(value)


def validation_state(value, now=None):
    """Expose mutable progress as telemetry, with conservative freshness labels."""
    if not isinstance(value, dict):
        return None
    result = dict(value)
    if result.get('status') in ('starting', 'running'):
        try:
            updated = datetime.fromisoformat(result['updated_utc'].replace('Z', '+00:00')).timestamp()
            now = time.time() if now is None else now
            age = now - updated
            # Generations have a five-minute progress deadline; model startup
            # and verification can take up to 30 minutes without a new event.
            limit = 360 if result.get('phase') == 'generating' else 1860
            if age < -60 or age > limit:
                result['status'] = 'stale'
        except (KeyError, TypeError, ValueError, OverflowError):
            result['status'] = 'unverified'
    return result


def long_output_validation(now=None, process_check=None):
    """Observe the owned long-output run from metadata and durable JSONL only."""
    root = LOCAL / 'reevaluations/qwen38_shared_long_output_v1'
    owner = load(LOCAL / 'validation/qwen38_shared_v1/inference.lock/owner.json')
    if not isinstance(owner, dict) or owner.get('kind') != 'qwen38_shared_long_output_v1':
        return None
    result = {'kind': 'qwen38_shared_long_output_v1', 'status': 'unverified',
              'condition': None, 'phase': 'verifying_live_metadata',
              'selected_adapter_sha256': '71ed18f5f3c15bce815bf77d386b47ecb5751e3cf53966bf1aa85ecd044def61',
              'completed_prompts': 0, 'progress_basis': 'durable_predictions_and_progress',
              'note': 'Live evaluation metadata has not yet passed identity checks.'}
    def live(pid, created, name):
        if type(pid) is not int or type(created) not in (int, float) or not math.isfinite(created):
            return False
        if process_check is not None:
            return process_check(pid, created, name)
        process = psutil.Process(pid)
        if not process.is_running() or abs(process.create_time()-created) > .01:
            return False
        return any((Path(arg) if Path(arg).is_absolute() else PROJECT/arg).resolve()
                   == (PROJECT/'mac_studio'/name).resolve() for arg in process.cmdline())
    def safe_file(value, within):
        path = Path(value)
        if not path.is_absolute(): path = PROJECT/path
        path = path.resolve()
        if path.is_symlink() or not path.is_file() or not path.is_relative_to(within.resolve()):
            raise ValueError('Evaluation metadata path is outside its declared root')
        return path
    def file_sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        if not live(owner['pid'], owner['process_created_at'], 'shared_long_output_supervisor_v1.py'):
            raise ValueError('Evaluation supervisor PID/create-time is not live')
        if not live(40908, 1794250499.834287, 'shared_long_output_coordinator_v1.py'):
            raise ValueError('Reviewed evaluation coordinator PID/create-time is not live')
        coordinator = PROJECT/'mac_studio/shared_long_output_coordinator_v1.py'
        if file_sha(coordinator) != 'f18226b529aff95febb4d98a9a6da91a6a09a7eb76688009b6da2a0647c5954f':
            raise ValueError('Reviewed coordinator source differs')
        matches = []
        for request_path in (root/'runs').glob('*/request.json'):
            request = load(request_path)
            if isinstance(request, dict) and request.get('lock_owner') == owner.get('owner'):
                matches.append((request_path, request))
        if len(matches) != 1: raise ValueError('Exactly one request must match the live lease')
        request_path, request = matches[0]
        run = request_path.parent.resolve()
        if (request.get('supervisor_pid') != owner['pid'] or Path(request['run_dir']).resolve() != run
                or not run.name.endswith('_'+owner['owner'])):
            raise ValueError('Evaluation lease/request/run identity differs')
        plan_path = safe_file(request['plan']['path'], root/'plans')
        if file_sha(plan_path) != request['plan']['sha256']:
            raise ValueError('Frozen evaluation plan bytes differ')
        plan = load(plan_path)
        if (not isinstance(plan, dict) or plan.get('schema') != 'research4.shared-long-output-stage.v1'
                or plan.get('max_new_tokens') != 8192 or plan.get('selected_adapter_sha256') != result['selected_adapter_sha256']
                or plan.get('base_manifest_sha256') != 'e126c678d420b65a2c5157c0bca0a3fb162cad6b96e4a067912418e0959d7461'
                or plan.get('stage') not in ('dev_full', 'old_heldout')
                or request.get('condition') not in ('base_comparator', 'selected_shared_adapter')):
            raise ValueError('Only the actual8192 original-shared900 evaluation is observed')
        total = 154 if plan['stage'] == 'dev_full' else 163
        ids = plan['prompt_ids']
        if len(ids) != total or len(set(ids)) != total:
            raise ValueError('Frozen unique evaluation ID count differs')
        source_names = {'shared_long_output_common_v1.py', 'shared_long_output_worker_v1.py', 'shared_long_output_supervisor_v1.py'}
        if set(plan.get('code_sha256', {})) != source_names:
            raise ValueError('Frozen runner source inventory differs')
        for name, expected in plan['code_sha256'].items():
            if file_sha(PROJECT/'mac_studio'/name) != expected:
                raise ValueError('Frozen evaluation runner source differs')
        freeze = safe_file(plan['source_freeze']['path'], root)
        if file_sha(freeze) != plan['source_freeze']['sha256']:
            raise ValueError('Frozen evaluation source manifest differs')
        authority_path = root/'authorities'/(request['plan']['sha256']+'_'+request['condition']+'.json')
        authority = load(authority_path)
        if not isinstance(authority, dict) or (authority.get('plan_sha256'), authority.get('condition'), authority.get('owner'), Path(authority.get('run_dir', '')).resolve()) != (request['plan']['sha256'], request['condition'], owner['owner'], run):
            raise ValueError('One-use evaluation authority/request binding differs')
        progress = load(run/'progress.json')
        if not isinstance(progress, dict): raise ValueError('Evaluation progress is unavailable')
        if progress.get('phase') not in ('preflight', 'loading_model', 'generating', 'verifying', 'complete'):
            raise ValueError('Unrecognized evaluation progress phase')
        done = progress['completed']
        if type(done) is not int or not 0 <= done <= total:
            raise ValueError('Invalid durable progress count')
        durable = []; predictions = run/'predictions.jsonl'
        if predictions.exists():
            if predictions.is_symlink() or predictions.stat().st_size > 96*1024**2:
                raise ValueError('Prediction metadata exceeds the bounded reader scope')
            with predictions.open('rb') as stream:
                for line in stream:
                    if not line.endswith(b'\n'): break  # Ignore a currently partial append.
                    if len(line) > 2*1024**2: raise ValueError('Prediction record outside bounds')
                    row = json.loads(line)
                    index = len(durable)
                    if index >= total or row.get('prompt_sha256') != ids[index] or row.get('finish_reason') not in ('stop', 'length'):
                        raise ValueError('Durable prediction ID/finish reason differs')
                    tokens, seconds = row['token_ids'], row['elapsed_seconds']
                    if (not isinstance(tokens, list) or len(tokens) > 8192 or any(type(token) is not int for token in tokens)
                            or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0):
                        raise ValueError('Durable prediction token/time record is invalid')
                    durable.append((len(tokens), seconds))
        if done > len(durable): raise ValueError('Progress count exceeds durable predictions')
        updated = datetime.fromisoformat(progress['updated_utc'].replace('Z', '+00:00')).timestamp()
        now = time.time() if now is None else now
        limit = plan['prompt_seconds'] if progress.get('phase') == 'generating' else plan['startup_seconds']
        fresh = type(limit) in (int, float) and math.isfinite(limit) and 0 < limit <= 1800 and -5 <= now-updated <= limit
        tokens = sum(count for count, _ in durable[:done]); seconds = sum(value for _, value in durable[:done])
        result.update(status='running' if fresh else 'stale', condition=request['condition'],
            phase=progress.get('phase'), stage=plan['stage'], max_new_tokens=8192,
            completed_prompts=done, total_prompts=total, generated_tokens=tokens,
            inference_elapsed_seconds=seconds, tokens_per_second=tokens/seconds if seconds else None,
            supervisor_pid=owner['pid'], coordinator_pid=40908, updated_utc=progress['updated_utc'],
            run_dir=str(run), plan_sha256=request['plan']['sha256'], live_owner_verified=True,
            note='Original shared900(71ed), not fixed LAST(b0a1). Saved predictions only; current long response may take up to1200 seconds.')
        if done == total and fresh: result['status'] = 'finalizing'
    except (OSError, ValueError, KeyError, TypeError, OverflowError, psutil.Error) as error:
        result['note'] = 'Evaluation telemetry unverified: '+str(error)
    return result


def general_preservation_job(now=None, process_check=None):
    """Metadata-only view of the separately human-authorized rank16/LAST8 fit."""
    root = LOCAL/'experiments/general_preservation_r16_last8_v1'
    requests = list((root/'runs').glob('fit_*/request.json'))
    if not requests: return None
    result = {'kind': 'general_preservation_r16_last8_v1', 'status': 'unverified',
              'stage_kind': 'general_fit', 'model_label': 'General preservation',
              'display_model_label': 'General preservation Â· rank16 LAST8',
              'completed_steps': 0, 'total_steps': 2160, 'rank': 16, 'num_layers': 8,
              'authorized_training': False, 'tokens_per_second': None, 'loss': None}
    def file_sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    def bound(item):
        path = Path(item['path'])
        if not path.is_absolute(): path = PROJECT/path
        if path.is_symlink(): raise ValueError('Symlinked fit metadata')
        path = path.resolve()
        if not path.is_relative_to(PROJECT.resolve()) or file_sha(path) != item['sha256']:
            raise ValueError('Bound fit metadata bytes/path differ')
        return path
    try:
        owner = load(LOCAL/'validation/qwen38_shared_v1/inference.lock/owner.json')
        active = isinstance(owner, dict) and owner.get('kind') in (
            result['kind'], 'general_preservation_r16_last8_temporal32_v2')
        if active:
            matches = [path for path in requests if load(path).get('owner') == owner.get('owner')]
            if len(matches) != 1: raise ValueError('Exactly one fit request must match its lease')
            request_path = matches[0]
        else: request_path = max(requests, key=lambda path:path.stat().st_mtime_ns)
        request = load(request_path); run = request_path.parent.resolve()
        if not isinstance(request, dict) or Path(request['run_dir']).resolve() != run or run.name != 'fit_'+request['owner']:
            raise ValueError('Fit request/run/owner identity differs')
        plan_path = bound({'path':request['plan_path'], 'sha256':request['plan_sha256']})
        if not plan_path.is_relative_to((root/'plans').resolve()): raise ValueError('Fit plan outside its experiment')
        plan = load(plan_path)
        fixed = {'schema':'research4.general-preservation-fit.v1','rank':16,'num_layers':8,
                 'layer_indices':list(range(56,64)),'iterations':2160,'post_training_DEV_rows':360,
                 'initialization':'fresh_pinned_base','resume_adapter':None,'final_split_opened':False,
                 'base_manifest_sha256':'e126c678d420b65a2c5157c0bca0a3fb162cad6b96e4a067912418e0959d7461'}
        if not isinstance(plan, dict) or any(plan.get(key) != value for key,value in fixed.items()):
            raise ValueError('Authorized general fit geometry/data scope differs')
        expected_kind = {'native_default':'general_preservation_r16_last8_v1',
                         'native_temporal_checkpoint32':'general_preservation_r16_last8_temporal32_v2'}.get(plan.get('gdn_backend'))
        if expected_kind is None or active and owner.get('kind') != expected_kind:
            raise ValueError('Fit lease kind and frozen backend differ')
        for name,digest in plan['source_sha256'].items():
            if Path(name).name != name or file_sha(PROJECT/'mac_studio'/name) != digest:
                raise ValueError('Frozen fit source changed')
        bound(plan['human_authorization']); bound(plan['tokenization_audit'])
        dataset = load(bound(plan['dataset_manifest']))
        if dataset['splits']['train']['decisions']['rows'] != 2160 or dataset['splits']['dev']['decisions']['rows'] != 360:
            raise ValueError('Declared TRAIN/DEV counts differ')
        authority = load(root/'authorities'/(request['plan_sha256']+'.json'))
        if not isinstance(authority, dict) or (authority.get('plan_sha256'), authority.get('owner'), Path(authority.get('run_dir','')).resolve()) != (request['plan_sha256'], request['owner'], run):
            raise ValueError('Fit authority/request binding differs')
        progress = load(run/'progress.json')
        if not isinstance(progress, dict) or (progress.get('total_updates'), progress.get('rank'), progress.get('num_layers')) != (2160,16,8):
            raise ValueError('Fit progress shape differs')
        updated = progress.get('updated_epoch')
        if type(updated) not in (int,float) or not math.isfinite(updated) or updated > (time.time() if now is None else now)+5:
            raise ValueError('Fit progress timestamp is invalid')
        verified = progress['verified_updates']; updates = []
        if type(verified) is not int or not 0 <= verified <= 2160: raise ValueError('Invalid verified update count')
        log = run/'updates.jsonl'
        if log.exists():
            if log.is_symlink() or log.stat().st_size > 16*1024**2: raise ValueError('Fit update log outside bounded scope')
            with log.open('rb') as stream:
                for line in stream:
                    if not line.endswith(b'\n'): break
                    row = json.loads(line)
                    if row.get('step') != len(updates)+1: raise ValueError('Fit updates are not contiguous')
                    for key in ('loss','gradient_norm','parameter_delta_L2','gradient_seconds','Adam_seconds'):
                        value = row[key]
                        if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
                            raise ValueError('Invalid durable fit metric')
                    if type(row['target_tokens']) is not int or row['target_tokens'] < 1: raise ValueError('Invalid target-token count')
                    updates.append(row)
        if verified > len(updates): raise ValueError('Fit progress exceeds durable update records')
        selected = updates[:verified]
        compute = sum(row['gradient_seconds']+row['Adam_seconds'] for row in selected)
        result.update(authorized_training=True, completed_steps=verified, verified_updates=verified,
            phase=progress.get('phase'), updated_epoch=progress.get('updated_epoch'),
            run_dir=str(run), plan_sha256=request['plan_sha256'],
            dataset_manifest_path=plan['dataset_manifest']['path'],
            dataset_summary={'training_examples':2160,'development_examples':360,
                             'source_yaml':dataset.get('source_yaml'),'final_split_opened':False},
            loss=selected[-1]['loss'] if selected else None,
            tokens_per_second=sum(row['target_tokens'] for row in selected)/compute if compute > 0 else None,
            gradient_norm=selected[-1]['gradient_norm'] if selected else None,
            available_bytes=progress.get('available_bytes'), mlx_active_bytes=progress.get('mlx_active_bytes'),
            mlx_cache_bytes=progress.get('mlx_cache_bytes'), mlx_peak_bytes=progress.get('mlx_peak_bytes'),
            note='New explicitly authorized general-preservation fit. Counts use durable verified updates; speed is supervised target tokens / gradient+Adam compute time, not inference speed.')
        failure = load(run/'supervisor_failure.json')
        closeout = load(run/'closeout.json')
        if isinstance(failure, dict) and failure.get('plan_sha256') == request['plan_sha256'] and failure.get('cleanup_confirmed') is True:
            result.update(status='failed', note=failure.get('error'), elapsed_seconds=failure.get('elapsed_seconds'),
                          fatal_available_bytes=failure.get('fatal_available_bytes'),
                          mlx_memory_scope='Last worker report only; not the unrecorded failure-time GPU peak.')
        elif isinstance(closeout, dict) and closeout.get('status') == 'complete' and closeout.get('completed_updates') == 2160:
            result.update(status='completed')
        elif active:
            if request['parent_pid'] != owner['pid']: raise ValueError('Fit parent and common lease differ')
            pid, created = owner['pid'], owner['process_created_at']
            if process_check is None:
                process = psutil.Process(pid)
                supervisors = {name for name in plan['source_sha256'] if 'supervisor' in name and name.endswith('.py')}
                live = process.is_running() and abs(process.create_time()-created) < .01 and any(
                    (Path(arg) if Path(arg).is_absolute() else PROJECT/arg).resolve() in {(PROJECT/'mac_studio'/name).resolve() for name in supervisors}
                    for arg in process.cmdline())
            else: live = process_check(pid,created)
            if not live: raise ValueError('Fit supervisor PID/create-time is not live')
            result.update(status='running' if verified < 2160 else 'finalizing', pid=pid,
                          process_created_at=created, live_owner_verified=True, lease_kind=owner['kind'],
                          elapsed_seconds=(time.time() if now is None else now)-created)
        else: raise ValueError('No active fit lease or verified terminal closeout')
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError, psutil.Error) as error:
        result.update(status='unverified', note='General fit telemetry unverified: '+str(error))
    return result


def service_preservation_campaign(compact=False):
    """Saved run/guard/update metadata plus exact local supervisor identity only."""
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location('research4_campaign_metadata', Path(__file__).with_name('campaign_status.py'))
    reader = module_from_spec(spec); spec.loader.exec_module(reader)
    def process_check(pid, created, plan_relative, plan_sha):
        try:
            process = psutil.Process(pid)
            if not process.is_running() or abs(process.create_time() - created) > .01: return False
            argv = process.cmdline()
            expected = 'compact_fit_supervisor.py' if compact else 'service_preservation_train_supervisor_v3.py'
            if not any(Path(arg).name == expected for arg in argv): return False
            return ('--plan-sha256' in argv and argv[argv.index('--plan-sha256') + 1] == plan_sha
                    and '--plan' in argv and reader._matches_saved_path(argv[argv.index('--plan') + 1], plan_relative))
        except psutil.NoSuchProcess:
            return False
        except (psutil.Error, OSError, ValueError, IndexError):
            return None
    return (reader.compact_snapshot if compact else reader.snapshot)(PROJECT, process_check=process_check)


def final_results():
    """Read only aggregate teacher-forced results; never expose individual prompts."""
    root = LOCAL / 'validation' / 'qwen38_shared_v1' / 'heldout_v1'
    path = root / 'final_metrics.json'
    if not path.exists(): return None
    try:
        raw = path.read_bytes()
        if len(raw) > 2_000_000 or hashlib.sha256(raw).hexdigest() != '29361add3d3ec6bb720d9284d61d4e9c6ba548f1eec0bd6fbe91dcd668015a27':
            raise ValueError('Final metrics hash differs')
        data = json.loads(raw)
        if data['schema'] != 'research4.heldout-final-metrics.v1' or data['status'] != 'complete':
            raise ValueError('Final metrics incomplete')
        result = {'status': 'complete', 'selected_step': data['selected_step'],
                  'counts': data['counts'], 'scope': data['scope'], 'conditions': {}}
        for key in ('base_comparator', 'selected_shared_adapter'):
            summary = data['scores'][key]['metrics']['all_occurrences']
            unique = data['scores'][key]['metrics']['unique_prompts']
            if any(not isinstance(summary.get(k), int) or summary[k] < 0 or summary[k] > summary['n'] for k in ('n', 'strict_format', 'schema_valid', 'legal', 'exact')):
                raise ValueError('Invalid summary counts')
            predictions = (root / 'runs' / key / 'predictions.jsonl').read_bytes()
            if len(predictions) > 10_000_000 or hashlib.sha256(predictions).hexdigest() != data['audits'][key]['predictions_sha256']:
                raise ValueError('Prediction bytes do not match the audited metric')
            rows = [json.loads(line) for line in predictions.splitlines()]
            if len(rows) != unique['n']: raise ValueError('Unique prediction count differs')
            result['conditions'][key] = {'n': summary['n'], 'exact': summary['exact'],
                'schema_valid': summary['schema_valid'], 'legal': summary['legal'],
                'unique_n': unique['n'], 'unique_exact': unique['exact'],
                'token_limit': sum(row.get('finish_reason') == 'length' for row in rows)}
        return result
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return {'status': 'unavailable'}


def provider352_fit_snapshot(project):
    """Hot-load the separate exact-run metadata reader; never submit model work."""
    from importlib.util import module_from_spec, spec_from_file_location
    try:
        spec = spec_from_file_location('research4_provider352_fit_metadata', Path(__file__).with_name('provider352_fit.py'))
        reader = module_from_spec(spec); spec.loader.exec_module(reader)
        return reader.snapshot(project, check_processes=True)
    except Exception:
        return {'status': 'unavailable', 'note': 'Provider3.5.2 saved progress reader unavailable.'}


def security500_fit_snapshot(project):
    """Read Root's exact security500 metadata; never submit or infer model work."""
    from importlib.util import module_from_spec, spec_from_file_location
    try:
        spec = spec_from_file_location('research4_security500_fit_metadata', Path(__file__).with_name('security500_fit.py'))
        reader = module_from_spec(spec); spec.loader.exec_module(reader)
        return reader.snapshot(project, check_processes=True)
    except Exception:
        return {'status': 'unavailable', 'note': 'Security500 saved progress reader unavailable.'}


def continuity_transfer_fit_snapshot(project):
    """Hot-load only the exact run metadata reader; no training controls."""
    from importlib.util import module_from_spec, spec_from_file_location
    try:
        spec = spec_from_file_location('research4_continuity_transfer_fit_metadata', Path(__file__).with_name('continuity_transfer_fit_v2.py'))
        reader = module_from_spec(spec); spec.loader.exec_module(reader)
        return reader.snapshot(project, check_processes=True)
    except Exception:
        return {'status': 'unavailable', 'note': 'Continuity-transfer metadata reader unavailable.'}


def replication_diversity84_fit_snapshot(project):
    """Existing read-only probe only; no additional process, SSH or model work."""
    from importlib.util import module_from_spec, spec_from_file_location
    try:
        spec = spec_from_file_location('research4_replication84_metadata', Path(__file__).with_name('replication_diversity84_fit.py'))
        reader = module_from_spec(spec); spec.loader.exec_module(reader)
        return reader.snapshot(project, check_processes=True)
    except Exception:
        return {'schema': 'research4.replication-diversity84-dashboard-fit.v1',
                'candidate': 'replication_diversity84_v1_20261116', 'status': 'unavailable',
                'optimizer_updates': None, 'live_owner_verified': False,
                'note': 'Exact new replication candidate metadata reader unavailable.'}


def gpu():
    try:
        raw = subprocess.check_output(['/usr/sbin/ioreg', '-r', '-c', 'IOAccelerator', '-d', '1', '-a'], timeout=4)
        devices = plistlib.loads(raw)
        stats = devices[0]['PerformanceStatistics']
        value = stats.get('Device Utilization %')
        return {'utilization_percent': value if isinstance(value, (int, float)) and 0 <= value <= 100 else None,
                'reported_allocation_bytes': stats.get('Alloc system memory'), 'source': 'macOS IORegistry'}
    except (OSError, KeyError, IndexError, ValueError, subprocess.SubprocessError):
        return {'utilization_percent': None, 'reported_allocation_bytes': None, 'source': 'unavailable'}


def main():
    memory = psutil.virtual_memory(); disk = psutil.disk_usage(str(PROJECT))
    previous_job = load(LOCAL / 'monitor' / 'shared_training_status.json')
    game_job = load(LOCAL / 'monitor' / 'sp_lens_game_training_status.json')
    all_layers_job = load(LOCAL / 'monitor' / 'sp_lens_all_layers_v2_status.json')
    memory_job = load(LOCAL / 'monitor' / 'sp_lens_all_layers_v3_status.json')
    equivalence_job = load(LOCAL / 'monitor' / 'sp_lens_selected_head_v3_status.json')
    chain_rule_job = load(LOCAL / 'monitor' / 'sp_lens_chain_rule_v4_status.json')
    body_vjp_job = load(LOCAL / 'monitor' / 'sp_lens_two_pass_v5_status.json')
    gdn_primitive_job = load(LOCAL / 'monitor' / 'sp_lens_gdn_v6_status.json')
    gdn_qualification_job = load(LOCAL / 'monitor' / 'sp_lens_gdn_v6_qualification_status.json')
    cache0_qualification_job = load(LOCAL / 'monitor' / 'sp_lens_gdn_v7_cache0_qualification_status.json')
    native_direct_job = load(LOCAL / 'monitor' / 'sp_lens_native_direct_adam_pilot_v1_status.json')
    blockwise_job = load(LOCAL / 'monitor' / 'sp_lens_blockwise_adam_pilot_v1_status.json')
    preparation_job = load(LOCAL / 'monitor' / 'sp_lens_all_layers_preparation.json')
    candidates = [normalize_training_job(value) for value in (all_layers_job, memory_job, equivalence_job, chain_rule_job, body_vjp_job, gdn_primitive_job, gdn_qualification_job, cache0_qualification_job, native_direct_job, blockwise_job, preparation_job) if isinstance(value, dict)]
    job = (max(candidates, key=job_timestamp) if candidates
           else game_job if isinstance(game_job, dict) else previous_job)
    general_job = general_preservation_job()
    if general_job is not None: job = general_job
    # This run has its own exact identity and journal; do not relabel prior fits.
    try:
        from replica_identity612_fit import snapshot as replica_identity612_snapshot
        current_replica_job = replica_identity612_snapshot(PROJECT)
        if (current_replica_job.get('status') == 'running'
                or not isinstance(job, dict)
                or job_timestamp(current_replica_job) >= job_timestamp(job)):
            job = current_replica_job
    except Exception:
        current_replica_job = None
    # Independent new fit: keep the historical rank-16 card separate and do
    # not infer optimizer progress from a process or its setup qualification.
    try:
        from rank32_fit import snapshot as rank32_fit_snapshot
        current_rank32_fit = rank32_fit_snapshot(PROJECT)
    except Exception:
        current_rank32_fit = {'schema':'research4.rank32-fit-dashboard.v1',
                              'status':'unverified', 'completed_steps':None,
                              'note':'Rank-32 telemetry is unavailable.'}
    try:
        from harness272_fit import snapshot as harness272_fit_snapshot
        current_harness272_fit = harness272_fit_snapshot(PROJECT)
    except Exception:
        current_harness272_fit = {'schema':'research4.harness272-fit-dashboard.v1',
                                  'status':'unverified', 'completed_steps':None,
                                  'note':'Harness-aligned telemetry is unavailable.'}
    pause = pause_state(load(LOCAL / 'monitor' / 'TRAINING_HPO_PAUSED_BY_HUMAN.json'))
    if isinstance(job, dict) and pause and pause['active'] and not job.get('authorized_training'):
        job = dict(job)
        job['historical_due_to_human_pause'] = True
    aliases = load(LOCAL / 'monitor' / 'experiment_aliases.json') or {}
    if isinstance(job, dict):
        alias = aliases.get('run_names', {}).get(Path(str(job.get('run_dir', ''))).name)
        if isinstance(alias, str):
            job['recorded_model_label'] = job.get('model_label')
            job['model_label'] = alias
            job['display_model_label'] = alias
    if isinstance(job, dict) and job.get('status') == 'running' and job.get('kind') != 'general_preservation_r16_last8_v1':
        try:
            process = psutil.Process(int(job['pid']))
            if abs(process.create_time() - float(job['process_created_at'])) > 2 or not process.is_running():
                job['status'] = 'unverified'
        except (KeyError, ValueError, TypeError, psutil.Error):
            job['status'] = 'unverified'
        try:
            stamp = job.get('updated_epoch')
            if isinstance(stamp, (int, float)) and not isinstance(stamp, bool) and math.isfinite(stamp):
                updated = float(stamp)
            else:
                updated = datetime.fromisoformat(job['updated_utc'].replace('Z', '+00:00')).timestamp()
            if updated > time.time() + 5:
                raise ValueError('Future telemetry timestamp')
            if time.time() - updated > 30:
                job['status'] = 'stale'
        except (KeyError, ValueError, TypeError):
            job['status'] = 'unverified'
    model = load(LOCAL / 'models' / 'qwen3.8-27b-mlx-4bit.manifest.json')
    if isinstance(job, dict) and job.get('kind') == 'replica_identity_6fd3b8ac_last8_r16_R2':
        dataset = job.get('dataset_summary')
    elif isinstance(job, dict) and job.get('authorized_training') and job.get('kind') == 'general_preservation_r16_last8_v1':
        dataset = job.get('dataset_summary')
    elif isinstance(game_job, dict) or candidates:
        dataset_path = PROJECT / str(job.get('dataset_manifest_path', ''))
        allowed = LOCAL / 'datasets' / 'sp_lens_game_v1'
        dataset = (load(dataset_path) if dataset_path.resolve().is_relative_to(allowed.resolve())
                   and dataset_path.name == 'manifest.json' else None)
    else:
        dataset = load(LOCAL / 'datasets' / 'qwen38_shared_v1' / 'dataset_manifest.json')
    smoke = load(LOCAL / 'smoke/merged-root-20261121/RELEASE.json')
    if isinstance(smoke, dict) and isinstance(smoke.get('report'), dict):
        smoke = {**smoke, 'iterations': smoke['report'].get('step')}
    diagnostic = load(LOCAL / 'diagnostics' / 'qwen38_context_memory_v1' / 'receipt.json')
    archive_verification = load(LOCAL / 'monitor' / 'mac_archive_verification_status.json')
    validation = long_output_validation()
    if validation is None:
        validation = validation_state(load(LOCAL / 'monitor' / 'validation_inference_status.json'))
    diagnostic_running = False
    for process in psutil.process_iter(['cmdline']):
        try:
            if any(str(arg).endswith('/mac_studio/context_memory_diagnostic.py') for arg in (process.info['cmdline'] or [])):
                diagnostic_running = True; break
        except (psutil.Error, OSError):
            continue
    diagnostic_state = 'running' if diagnostic_running else diagnostic.get('status') if diagnostic else None
    out = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'project_root': str(PROJECT),
           'python': sys.executable,
           'cpu_percent': psutil.cpu_percent(interval=0.3),
           'memory': {'total_bytes': memory.total, 'available_bytes': memory.available, 'percent': memory.percent},
           'disk': {'total_bytes': disk.total, 'free_bytes': disk.free, 'percent': disk.percent},
           'gpu': gpu(), 'job': job if isinstance(job, dict) else None,
           'model': {'ready': bool(model), 'repo_id': model.get('repo_id') if model else None,
                     'revision': model.get('revision') if model else None},
           'dataset': dataset, 'synthetic_smoke': {'status': smoke.get('status'), 'iterations': smoke.get('iterations')}
           if smoke else None,
           'memory_diagnostic': {'status': diagnostic_state, 'reason': diagnostic.get('reason') if diagnostic else None},
           'archive_verification': archive_verification,
           'validation': validation,
           'training_pause': pause,
           'fine_tuning_steps': fine_tuning_steps_snapshot(PROJECT),
           'provider352_fit': provider352_fit_snapshot(PROJECT),
           'security500_fit': security500_fit_snapshot(PROJECT),
           'continuity_transfer_fit': continuity_transfer_fit_snapshot(PROJECT),
           'replication_diversity84_fit': replication_diversity84_fit_snapshot(PROJECT),
           'replica_identity612_fit': current_replica_job,
           'rank32_fit': current_rank32_fit,
           'harness272_fit': current_harness272_fit,
           'new_curriculum_campaign': service_preservation_campaign(),
           'compact_training': service_preservation_campaign(compact=True),
           'inference_endpoint': current_endpoint_state(),
           'heldout_results': None if pause and pause['active'] else final_results(),
           'mlx_version': importlib.metadata.version('mlx')}
    print(json.dumps(out))


if __name__ == '__main__':
    main()
