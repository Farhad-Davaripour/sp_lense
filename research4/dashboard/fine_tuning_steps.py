"""Read-only NEXT-candidate dashboard; never reads data bodies or contacts a worker.

Optional saved-progress contract (all keys below are closed):
  candidate.dashboard_steps = {"schema": "research4.dashboard-steps.v1",
                               "stages": {stage_id: {"path": ..., "sha256": ...}}}
Receipts live ONLY under CONTRACT_ROOT, as <stage_id>.json or
<stage_id>_<version>.json. Each receipt contains schema, candidate, source_root,
dataset, stage, recorded_utc, status, and optional detail (<=500 characters).
schema is research4.dashboard-stage.v1; candidate is CANDIDATE; source_root and
dataset must match the current candidate. Status is completed/active/pending/
blocked/unavailable; train alone also permits stopped. dataset_build additionally requires generated_train_rows
and generated_dev_rows. train/validate require completed and total (2160/500).
train may supply supervised_target_tokens and compute_seconds together, and
eta_seconds. These are saved measurements/estimates, never live telemetry.
Other optional keys are rejected. Completed/active stages require completed
prerequisites; model stages additionally require model_or_fit_admitted=true.

No recursive discovery, process probe, raw journal, TRAIN/DEV examples, model
output, private evaluation, prediction or checkpoint body is opened here.

Optional candidate.runtime_progress = {path, sha256} binds an immutable receipt
under CONTRACT_ROOT/runtime[_version].json. Its exact keys are schema
(research4.dashboard-runtime.v1), candidate, source_root, dataset, recorded_utc,
run_root, plan ({path, sha256}), request_sha256, run_config_sha256, and
model_manifest_sha256. Only the exact width768 or warning-once plan/root pairs
and one fit_<32hex> run under the matching root/runs are accepted. The hashed request and run_config must agree
with that run, plan, source map and frozen model. After those checks, only that
run's bounded progress.json is read dynamically. It contains worker-verified
update counters, not a process-liveness assertion or release/closeout evidence.
The only journal exception is a <=64 KiB tail of that exact run's updates.jsonl,
for a finite numeric last-verified-update compute throughput measurement. The
worker supplies no cumulative throughput, elapsed or per-row DEV counters.
Warning-once runtime also requires a saved active TRAIN receipt. A stopped
receipt must be replaced before any new runtime counters can be displayed.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import time

CANDIDATE = 'ongoing_public_view_cpu_full_candidate_v3'
STATE = '.local/monitor/CURRENT_EXPERIMENT_STATE.json'
SOURCE = 'proposals/service_preservation_ongoing_20261114/candidate_public_view_cpu_v3_full_20261114'
DATASET = '.local/datasets/ongoing_service_public_view_cpu_v3_full_20261114'
CONTRACT_ROOT = '.local/monitor/fine_tuning_steps/' + CANDIDATE
DECISION = '.local/reviews/game_ongoing_public_view_review_20261114/CRITICAL_REVIEW_AND_FULL_CPU_BUILD_DECISION_v1.json'
SOURCE_REVIEW = '.local/reviews/ongoing_public_view_cpu_v2_20261114/INDEPENDENT_CRITICAL_REVIEW_FINAL_v1.json'
RUNTIME_REVIEW = '.local/reviews/prior_native_length_evidence_20261114/PRIOR_NATIVE_LENGTH_EVIDENCE_v2.json'
RUNTIME_SHA = 'fe7115f6734d32ac3d0c011dbe82ccf20c46bf7466c889ddb322c1aaf06a81bc'
MAX_METADATA_BYTES = 1_000_000
STALE_SECONDS = 3600
RUNTIME_STALE_SECONDS = 300
FIT_ROOT = '.local/experiments/ongoing_service_r16_last8_v2_width768_20261114'
FIT_PLAN = '.local/staging/ongoing_service_fit_v2_width768_20261114/plans/FIT_WIDTH768_ADMITTED_v1.json'
FIT_PLAN_SHA = 'f8807b177ad8835ce72f0d84db31004e02e4881bfe0cc1f6be01554540e764a4'
WARNING_FIT_ROOT = '.local/experiments/ongoing_service_r16_last8_warning_once_v1_20261114'
WARNING_FIT_PLAN = '.local/staging/ongoing_service_fit_v3_warning_once_20261114/plans/FIT_WARNING_ONCE_ADMITTED_v1.json'
WARNING_FIT_PLAN_SHA = '77ca214b0b90d0bf08e38b4e5fb5eca13ee427d330ccd7488122035b143ddfb2'
MODEL_SHA = 'e126c678d420b65a2c5157c0bca0a3fb162cad6b96e4a067912418e0959d7461'
MAX_UPDATE_TAIL_BYTES = 65536
STAGES = (
    ('source_checks', 'Reviewed public-view source checks (v2)'),
    ('saved_runtime', 'Review saved historical runtime'),
    ('dataset_build', 'Build and select full TRAIN / DEV candidate'),
    ('full_audit', 'Audit full data, causal coverage and token widths'),
    ('input_resource_plan', 'Review input-length scope and resource plan'),
    ('train', 'Train shared adapter: 2,160 updates'),
    ('validate', 'Validate on 500 DEV rows'),
    ('checkpoint_audit', 'Audit checkpoint and saved training evidence'),
    ('game_comparison', 'Run independent game comparison'),
)
STATUSES = {'completed', 'active', 'pending', 'blocked', 'unavailable'}
READ_ERRORS = (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError)


def _metadata_path(root, relative):
    if not isinstance(relative, str) or '\\' in relative or ':' in relative:
        raise ValueError('Metadata requires an explicit relative path')
    parts = PurePosixPath(relative).parts
    if not parts or relative.startswith('/') or '..' in parts:
        raise ValueError('Metadata path escapes repository')
    path = root
    for part in parts:
        path = path / part
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Linked/reparse metadata is outside the read scope')
    if not path.resolve().is_relative_to(root) or not stat.S_ISREG(info.st_mode):
        raise ValueError('Metadata must be a regular repository file')
    return path


def _read(root, relative, digest=None):
    path = _metadata_path(root, relative)
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError('Metadata exceeds size bound')
    with path.open('rb') as handle:
        raw = handle.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError('Metadata exceeds size bound')
    if digest is not None and (not isinstance(digest, str) or
                              not re.fullmatch('[0-9a-f]{64}', digest) or
                              hashlib.sha256(raw).hexdigest() != digest):
        raise ValueError('Saved metadata hash does not match')
    value = json.loads(raw.decode('utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Metadata must be an object')
    return value


def _ref(root, reference, expected_path=None):
    if not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}:
        raise ValueError('Expected exact path / SHA256 metadata reference')
    if expected_path is not None and reference['path'] != expected_path:
        raise ValueError('Saved metadata reference has a different scope')
    return _read(root, reference['path'], reference['sha256'])


def _epoch(value, now):
    if not isinstance(value, str):
        raise ValueError('Missing saved metadata time')
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is None or stamp.timestamp() > now + 300:
        raise ValueError('Invalid saved metadata time')
    return stamp.timestamp()


def _count(value, total):
    if type(value) is not int or not 0 <= value <= total:
        raise ValueError('Invalid saved progress counter')
    return value


def _positive(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        return False


def _progress(root, candidate, stage, reference, now):
    path = reference.get('path', '') if isinstance(reference, dict) else ''
    if not re.fullmatch(re.escape(CONTRACT_ROOT + '/' + stage) + r'(?:_[A-Za-z0-9_-]+)?\.json', path):
        raise ValueError('Progress receipt is outside its closed metadata namespace')
    receipt = _ref(root, reference)
    required = {'schema', 'candidate', 'source_root', 'dataset', 'stage', 'recorded_utc', 'status'}
    extras = {'detail'}
    if stage == 'dataset_build':
        required |= {'generated_train_rows', 'generated_dev_rows'}
    if stage in ('train', 'validate'):
        required |= {'completed', 'total'}
    if stage == 'train':
        extras |= {'supervised_target_tokens', 'compute_seconds', 'eta_seconds'}
    if not required <= set(receipt) or set(receipt) - required - extras:
        raise ValueError('Progress receipt differs from the closed metadata contract')
    if (receipt['schema'], receipt['candidate'], receipt['source_root'], receipt['dataset'], receipt['stage']) != (
            'research4.dashboard-stage.v1', CANDIDATE, SOURCE, DATASET, stage):
        raise ValueError('Progress receipt candidate identity differs')
    if receipt['status'] not in STATUSES | ({'stopped'} if stage == 'train' else set()):
        raise ValueError('Unknown saved stage status')
    if not isinstance(receipt.get('detail', ''), str) or len(receipt.get('detail', '')) > 500:
        raise ValueError('Invalid progress description')
    age = max(0, now - _epoch(receipt['recorded_utc'], now))
    out = {'status': receipt['status'], 'saved_status': receipt['status'], 'recorded_utc': receipt['recorded_utc'],
           'detail': receipt.get('detail', 'Saved stage receipt; no live worker check.'),
           'evidence_path': path, 'stale': age > STALE_SECONDS}
    if stage == 'dataset_build':
        train = _count(receipt['generated_train_rows'], 2160)
        dev = _count(receipt['generated_dev_rows'], 500)
        out.update(generated_train_rows=train, generated_dev_rows=dev)
        out['detail'] = f'Generated {train:,} / 2,160 TRAIN and {dev:,} / 500 DEV rows. ' + out['detail']
        if receipt['status'] == 'completed' and (train, dev) != (2160, 500):
            raise ValueError('Completed dataset receipt lacks full generated counts')
    if stage in ('train', 'validate'):
        total = 2160 if stage == 'train' else 500
        completed = _count(receipt['completed'], total)
        if type(receipt['total']) is not int or receipt['total'] != total:
            raise ValueError('Saved progress denominator differs')
        if receipt['status'] == 'completed' and completed != total:
            raise ValueError('Completed stage lacks its full counter')
        out.update(completed=completed, total=total)
    if stage == 'train':
        tokens, seconds = receipt.get('supervised_target_tokens'), receipt.get('compute_seconds')
        if tokens is not None or seconds is not None:
            if not _positive(tokens) or type(tokens) is not int or not _positive(seconds) or not out['completed']:
                raise ValueError('Invalid saved training throughput measurements')
            out['training_tokens_per_second'] = tokens / seconds
        if 'eta_seconds' in receipt:
            eta = receipt['eta_seconds']
            if type(eta) not in (int, float) or not math.isfinite(eta) or eta < 0 or not out['completed']:
                raise ValueError('Invalid saved training ETA')
            out['training_eta_seconds'] = eta
    if out['stale'] and out['status'] == 'active':
        out.update(status='unavailable', detail='Saved active stage receipt is stale; current activity is unverified.')
    return out


def _runtime_progress(root, reference, now):
    """Read one explicitly bound worker's small saved counters, never its journal."""
    path = reference.get('path', '') if isinstance(reference, dict) else ''
    if not re.fullmatch(re.escape(CONTRACT_ROOT + '/runtime') + r'(?:_[A-Za-z0-9_-]+)?\.json', path):
        raise ValueError('Runtime identity is outside its closed metadata namespace')
    identity = _ref(root, reference)
    keys = {'schema', 'candidate', 'source_root', 'dataset', 'recorded_utc',
            'run_root', 'plan', 'request_sha256', 'run_config_sha256', 'model_manifest_sha256'}
    if set(identity) != keys or (identity['schema'], identity['candidate'], identity['source_root'], identity['dataset']) != (
            'research4.dashboard-runtime.v1', CANDIDATE, SOURCE, DATASET):
        raise ValueError('Runtime identity differs from the closed candidate contract')
    _epoch(identity['recorded_utc'], now)
    run_root = identity['run_root']
    profiles = ((FIT_ROOT, FIT_PLAN, FIT_PLAN_SHA),
                (WARNING_FIT_ROOT, WARNING_FIT_PLAN, WARNING_FIT_PLAN_SHA))
    matches = [(plan_path, plan_sha) for fit_root, plan_path, plan_sha in profiles
               if isinstance(run_root, str) and re.fullmatch(re.escape(fit_root + '/runs/fit_') + '[0-9a-f]{32}', run_root)]
    if len(matches) != 1:
        raise ValueError('Runtime run is outside the exact admitted fit roots')
    plan_path, plan_sha = matches[0]
    if identity['plan'] != {'path': plan_path, 'sha256': plan_sha} or identity['model_manifest_sha256'] != MODEL_SHA:
        raise ValueError('Runtime plan or frozen model identity differs')
    if any(not isinstance(identity[key], str) or not re.fullmatch('[0-9a-f]{64}', identity[key])
           for key in ('request_sha256', 'run_config_sha256')):
        raise ValueError('Immutable actual request/config SHA256 bindings are required')
    plan = _ref(root, identity['plan'], plan_path)
    if (plan.get('schema'), plan.get('base_manifest_sha256'), plan.get('iterations'), plan.get('post_training_DEV_rows')) != (
            'research4.service-continuity-compact-fit-plan.v1', MODEL_SHA, 2160, 500):
        raise ValueError('Pinned runtime plan has a different training scope')
    request = _read(root, run_root + '/request.json', identity['request_sha256'])
    config = _read(root, run_root + '/run_config.json', identity['run_config_sha256'])
    if set(request) != {'plan_path', 'plan_sha256', 'run_dir', 'owner', 'parent_pid',
                        'original_origin_monotonic', 'original_deadline_monotonic'}:
        raise ValueError('Actual run request differs from the worker contract')
    absolute_run = request['run_dir']
    suffix = '/' + run_root
    if not isinstance(absolute_run, str) or not absolute_run.startswith('/') or not absolute_run.endswith(suffix) or '..' in PurePosixPath(absolute_run).parts:
        raise ValueError('Actual run request path does not bind this run')
    project_prefix = absolute_run[:-len(suffix)]
    if (request['plan_path'] != project_prefix + '/' + plan_path or request['plan_sha256'] != plan_sha or
            request['owner'] != run_root.rsplit('fit_', 1)[1] or
            type(request['parent_pid']) is not int or request['parent_pid'] <= 0 or
            not _positive(request['original_origin_monotonic']) or
            request['original_deadline_monotonic'] != request['original_origin_monotonic'] + 172800 or
            config.get('plan_sha256') != plan_sha or any(config.get(key) != value for key, value in plan.items())):
        raise ValueError('Actual request/config plan, source or owner binding differs')
    progress_path = run_root + '/progress.json'
    progress = _read(root, progress_path)
    required = {'phase', 'verified_updates', 'total_updates', 'rank', 'num_layers', 'learning_rate', 'updated_epoch',
                'available_bytes', 'mlx_active_bytes', 'mlx_cache_bytes', 'mlx_peak_bytes'}
    extras = {'next_update', 'gradient_seconds', 'optimizer_step_observed', 'last_checkpoint'}
    if not required <= set(progress) or set(progress) - required - extras:
        raise ValueError('Worker progress differs from the closed counter contract')
    count = _count(progress['verified_updates'], 2160)
    if (type(progress['total_updates']) is not int or progress['total_updates'] != 2160 or
            type(progress['rank']) is not int or progress['rank'] != 16 or
            type(progress['num_layers']) is not int or progress['num_layers'] != 8 or progress['learning_rate'] != 1e-5 or
            any(type(progress[key]) is not int or progress[key] < 0 for key in
                ('available_bytes', 'mlx_active_bytes', 'mlx_cache_bytes', 'mlx_peak_bytes'))):
        raise ValueError('Worker progress recipe or counters differ')
    phase = progress['phase']
    if phase not in {'loading_model', 'before_gradient', 'gradient_evaluated', 'Adam_evaluated',
                     'update_verified', 'paused', 'DEV_evaluating', 'complete'}:
        raise ValueError('Unknown worker progress phase')
    if ((phase == 'loading_model' and count != 0) or
            (phase in {'DEV_evaluating', 'complete'} and count != 2160) or
            (phase == 'update_verified' and count == 0)):
        raise ValueError('Worker phase contradicts its verified counter')
    if phase in {'before_gradient', 'gradient_evaluated'} and (
            type(progress.get('next_update')) is not int or progress['next_update'] != count + 1 or count == 2160):
        raise ValueError('Next gradient is not the next unverified update')
    if phase == 'gradient_evaluated' and not _positive(progress.get('gradient_seconds')):
        raise ValueError('Missing measured gradient duration')
    if phase == 'Adam_evaluated' and (
            type(progress.get('optimizer_step_observed')) is not int or progress['optimizer_step_observed'] != count + 1 or count == 2160):
        raise ValueError('Adam phase is not the next unverified update')
    checkpoint = progress.get('last_checkpoint')
    if checkpoint is not None and (not isinstance(checkpoint, str) or
            not re.fullmatch(r'adapters/step_[0-9]{6}', checkpoint) or int(checkpoint[-6:]) > count):
        raise ValueError('Saved checkpoint pointer exceeds verified updates')
    epoch = progress['updated_epoch']
    if not _positive(epoch) or epoch > now + 300:
        raise ValueError('Invalid saved worker progress time')
    age = max(0, now - epoch)
    return {'run_root': run_root, 'phase': phase, 'verified_updates': count, 'total_updates': 2160,
            'recorded_utc': datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace('+00:00', 'Z'),
            'age_seconds': age, 'stale': age > RUNTIME_STALE_SECONDS,
            'live_process_verified': False, 'last_checkpoint': checkpoint, 'evidence_path': progress_path}


def _last_update_measurement(root, runtime):
    """Bounded numeric journal exception; no whole-journal or example reads."""
    if not runtime['verified_updates']:
        return None
    path = _metadata_path(root, runtime['run_root'] + '/updates.jsonl')
    with path.open('rb') as handle:
        size = handle.seek(0, 2)
        offset = max(0, size - MAX_UPDATE_TAIL_BYTES)
        handle.seek(offset)
        tail = handle.read(MAX_UPDATE_TAIL_BYTES)
    lines = tail.split(b'\n')
    if offset:
        lines = lines[1:]
    # Only newline-terminated records are durable complete journal entries.
    for line in reversed(lines[:-1]):
        if len(line) > 4096:
            raise ValueError('Numeric update record exceeds its size bound')
        row = json.loads(line.decode('utf-8'))
        if not isinstance(row, dict):
            raise ValueError('Numeric update record must be an object')
        step = _count(row.get('step'), 2160)
        if not step or step > runtime['verified_updates']:
            continue
        keys = {'step', 'loss', 'target_tokens', 'gradient_norm', 'parameter_delta_L2',
                'gradient_seconds', 'Adam_seconds', 'training_target_tokens_per_compute_second',
                'mlx_peak_bytes', 'available_bytes', 'source_scope'}
        if (set(row) != keys or type(row['target_tokens']) is not int or
                any(not _positive(row[key]) for key in ('target_tokens', 'gradient_norm', 'parameter_delta_L2',
                    'gradient_seconds', 'Adam_seconds', 'training_target_tokens_per_compute_second')) or
                type(row['loss']) not in (int, float) or not math.isfinite(row['loss']) or row['loss'] < 0 or
                any(type(row[key]) is not int or row[key] < 0 for key in ('mlx_peak_bytes', 'available_bytes')) or
                row['source_scope'] != 'Actual adapter training; not inference token speed.'):
            raise ValueError('Last verified update lacks finite actual training measurements')
        rate = row['target_tokens'] / (row['gradient_seconds'] + row['Adam_seconds'])
        if not math.isclose(rate, row['training_target_tokens_per_compute_second'], rel_tol=1e-12):
            raise ValueError('Last verified update throughput ratio differs')
        return {'step': step, 'target_tokens': row['target_tokens'],
                'compute_seconds': row['gradient_seconds'] + row['Adam_seconds'],
                'training_tokens_per_second': rate, 'scope': 'last verified update'}
    return None


def snapshot(project, now=None):
    """Return saved NEXT-candidate stages. No return value asserts live activity."""
    now = time.time() if now is None else now
    result = {'status': 'unavailable', 'title': 'Next shared-adapter fine-tuning steps',
              'candidate': CANDIDATE, 'recorded_utc': None, 'age_seconds': None,
              'stale': False, 'scope': 'Saved Root metadata only; no live process or model check.',
              'planned_train_rows': None, 'planned_dev_rows': None,
              'generated_train_rows': None, 'generated_dev_rows': None,
              'optimizer_updates': None, 'training_tokens_per_second': None,
              'last_verified_update_training_tokens_per_second': None,
              'training_throughput_scope': None, 'training_eta_seconds': None,
              'terminal_training_status': None, 'runtime_progress': None, 'warnings': [],
              'stages': [{'id': key, 'title': title, 'status': 'unavailable',
                          'detail': 'Current candidate metadata unavailable.'} for key, title in STAGES]}
    stages = {item['id']: item for item in result['stages']}
    root = Path(project).resolve()
    try:
        state = _read(root, STATE)
        candidate = state[CANDIDATE]
        if not isinstance(candidate, dict) or (candidate.get('source_root'), candidate.get('dataset')) != (SOURCE, DATASET):
            raise ValueError('Current candidate namespace differs')
        counts = candidate.get('counts', {})
        if type(counts.get('TRAIN')) is not int or type(counts.get('DEV')) is not int or (counts['TRAIN'], counts['DEV']) != (2160, 500):
            raise ValueError('Planned current candidate counts differ')
        age = max(0, now - _epoch(state.get('updated_utc'), now))
        result.update(status='available', recorded_utc=state['updated_utc'], age_seconds=age,
                      stale=age > STALE_SECONDS, planned_train_rows=2160, planned_dev_rows=500)
        for stage in stages.values():
            stage.update(status='pending', detail='Waiting for the preceding reviewed stage.')
        stages['dataset_build'].update(status='unavailable', detail='CPU build authority is not yet verified.')
        stages['input_resource_plan'].update(status='blocked', detail='Native 512 input scope and current resource-plan admission remain unresolved.')
        stages['train'].update(status='blocked', completed=0, total=2160,
                               detail='No model or fit is admitted for this candidate; 0 / 2,160 optimizer updates.')
        stages['validate'].update(completed=0, total=500, detail='500 planned DEV rows; validation has not started.')
        if candidate.get('model_or_fit_admitted') is not False:
            stages['train'].update(status='unavailable', completed=None,
                                   detail='No verified saved training progress receipt for this candidate.')
            stages['validate'].update(completed=None)
        cpu_build_authorized = False
        try:
            decision = _ref(root, candidate.get('decision'), DECISION)
            if decision.get('schema') != 'research4.game-public-view-critical-review-and-next-cpu-step.v1' or decision.get('full_CPU_dataset_build_authorized') is not True:
                raise ValueError('CPU dataset build authority is absent')
            cpu_build_authorized = True
            if candidate.get('status') == 'AUTHORIZED_CPU_BUILD_ONLY_NATIVE512_SCOPE_UNRESOLVED':
                stages['dataset_build'].update(status='active' if not result['stale'] else 'unavailable',
                    detail='CPU build and action-anchored selection authorized: 2,160 TRAIN / 500 DEV planned. Generated counts are not recorded; this is a saved work-stage declaration.')
        except READ_ERRORS as error:
            result['warnings'].append('CPU build decision: ' + str(error))
        try:
            prior = state.get('ongoing_public_view_cpu_candidate_v2', {})
            review = _ref(root, prior.get('independent_final_critical_review'), SOURCE_REVIEW)
            if review.get('schema') != 'research4.public-view-v2-independent-final-critical-review.v1':
                raise ValueError('Source review schema differs')
            checked = review['confirmed_source_and_critical_checks']
            tests, causal = checked['existing_source_tests'], checked['causal_records']
            if (tests.get('passed'), tests.get('independently_run'), causal.get('decisions')) != (15, 15, 816) or tests.get('source_hashes_match_frozen_source') is not True or causal.get('all_backend_before_action_receipt_after_replayed_exactly') is not True:
                raise ValueError('Source review aggregate does not match the reported checks')
            stages['source_checks'].update(status='completed', evidence_path=SOURCE_REVIEW,
                detail='Prior frozen v2 evidence: 15 / 15 CPU source tests and 816 independently replayed causal decisions. Full v3 dataset audit remains separate.')
        except READ_ERRORS as error:
            stages['source_checks'].update(status='unavailable', detail='Exact prior source-review receipt could not be verified.')
            result['warnings'].append('Source checks: ' + str(error))
        try:
            reference = candidate.get('historical_length_review')
            if not isinstance(reference, dict) or reference.get('sha256') != RUNTIME_SHA:
                raise ValueError('Expected the saved-runtime v2 reconciliation receipt')
            review = _ref(root, reference, RUNTIME_REVIEW)
            if review.get('schema') != 'research4.prior-native-length-evidence.v2' or review.get('status') != 'PASS_SAVED_RUNTIME_RECONCILIATION_ONLY_HELD_NO_ADMISSION':
                raise ValueError('Saved-runtime reconciliation is not complete')
            stages['saved_runtime'].update(status='completed', evidence_path=RUNTIME_REVIEW,
                detail='Historical saved-runtime reconciliation complete. Historical input widths do not admit this candidate or change its native 512 guard.')
        except READ_ERRORS as error:
            stages['saved_runtime'].update(status='unavailable', detail='Exact saved-runtime v2 receipt is unavailable or unverified.')
            result['warnings'].append('Saved runtime: ' + str(error))

        contract = candidate.get('dashboard_steps')
        if contract is not None:
            if not isinstance(contract, dict) or set(contract) != {'schema', 'stages'} or contract['schema'] != 'research4.dashboard-steps.v1' or not isinstance(contract['stages'], dict) or set(contract['stages']) - {key for key, _ in STAGES[2:]}:
                raise ValueError('Invalid closed saved-progress contract')
            prerequisites = {'dataset_build': [], 'full_audit': ['dataset_build'],
                'input_resource_plan': ['source_checks', 'saved_runtime', 'dataset_build', 'full_audit'],
                'train': ['input_resource_plan'], 'validate': ['train'],
                'checkpoint_audit': ['validate'], 'game_comparison': ['checkpoint_audit']}
            for key, _ in STAGES[2:]:
                if key in contract['stages']:
                    try:
                        progress = _progress(root, candidate, key, contract['stages'][key], now)
                        if progress['status'] in ('active', 'completed', 'stopped'):
                            if any(stages[item]['status'] != 'completed' for item in prerequisites[key]):
                                raise ValueError('Preceding required stages are not completed')
                            if key == 'dataset_build' and not cpu_build_authorized:
                                raise ValueError('CPU dataset build authority is not verified')
                            if key in ('train', 'validate', 'checkpoint_audit', 'game_comparison') and candidate.get('model_or_fit_admitted') is not True:
                                raise ValueError('Current candidate has no model/fit admission')
                        stages[key].update(progress)
                    except READ_ERRORS as error:
                        stages[key].update(status='unavailable', detail='Saved progress receipt could not be verified.')
                        if key in ('train', 'validate'):
                            stages[key]['completed'] = None
                        result['warnings'].append(key + ': ' + str(error))
        if candidate.get('runtime_progress') is not None and stages['train']['status'] == 'stopped':
            result['warnings'].append('Runtime progress cannot replace a stopped training receipt; a new active stage receipt is required.')
        elif candidate.get('runtime_progress') is not None:
            try:
                if candidate.get('model_or_fit_admitted') is not True or any(
                        stages[key]['status'] != 'completed' for key in
                        ('source_checks', 'saved_runtime', 'dataset_build', 'full_audit', 'input_resource_plan')):
                    raise ValueError('Runtime counters require fit admission and completed reviewed prerequisites')
                runtime = _runtime_progress(root, candidate['runtime_progress'], now)
                if runtime['run_root'].startswith(WARNING_FIT_ROOT + '/') and stages['train'].get('saved_status') != 'active':
                    raise ValueError('Warning-once runtime requires a verified active training receipt')
                count, phase = runtime['verified_updates'], runtime['phase']
                detail = f'Saved worker phase: {phase}; {count:,} / 2,160 verified optimizer updates. Process activity has not been probed.'
                status = 'completed' if count == 2160 else 'active'
                if phase == 'paused':
                    status = 'blocked'
                    detail += ' Saved paused boundary; no resume is authorized.'
                if runtime['stale']:
                    status = 'unavailable'
                    detail += ' Saved progress is stale; current activity is unverified.'
                if runtime['last_checkpoint'] is not None:
                    detail += f" Saved worker checkpoint pointer: {runtime['last_checkpoint']}; checkpoint audit remains separate."
                stages['train'].update(status=status, completed=count, total=2160, detail=detail,
                    recorded_utc=runtime['recorded_utc'], stale=runtime['stale'], evidence_path=runtime['evidence_path'])
                # The worker has no cumulative compute/token measurements. Never
                # reuse throughput or ETA from a separate saved-stage receipt.
                stages['train'].pop('training_tokens_per_second', None)
                stages['train'].pop('training_eta_seconds', None)
                try:
                    measurement = _last_update_measurement(root, runtime)
                    if measurement is not None:
                        runtime['last_verified_update_measurement'] = measurement
                        stages['train']['training_tokens_per_second'] = measurement['training_tokens_per_second']
                        result['last_verified_update_training_tokens_per_second'] = measurement['training_tokens_per_second']
                        result['training_throughput_scope'] = measurement['scope']
                except READ_ERRORS as error:
                    result['warnings'].append('Last verified update throughput: ' + str(error))
                if phase in {'DEV_evaluating', 'complete'} and stages['validate']['status'] != 'completed':
                    stages['validate'].update(status='unavailable' if runtime['stale'] else 'active',
                        completed=None, total=500, recorded_utc=runtime['recorded_utc'], stale=runtime['stale'],
                        detail='Worker reached DEV evaluation after 2,160 verified updates. No per-row DEV counter is emitted; 500 rows await a separately verified completion receipt. Process and closeout are unverified.')
                result.update(runtime_progress=runtime, recorded_utc=runtime['recorded_utc'],
                    age_seconds=runtime['age_seconds'], stale=runtime['stale'])
            except READ_ERRORS as error:
                stages['train'].update(status='unavailable', completed=None,
                    detail='Bound runtime progress could not be verified.')
                stages['train'].pop('training_tokens_per_second', None)
                stages['train'].pop('training_eta_seconds', None)
                result['warnings'].append('Runtime progress: ' + str(error))
        result['generated_train_rows'] = stages['dataset_build'].get('generated_train_rows')
        result['generated_dev_rows'] = stages['dataset_build'].get('generated_dev_rows')
        result['optimizer_updates'] = stages['train'].get('completed', 0) if stages['train']['status'] != 'unavailable' else None
        if result['runtime_progress'] is not None:
            result['optimizer_updates'] = result['runtime_progress']['verified_updates']
        result['training_tokens_per_second'] = stages['train'].get('training_tokens_per_second')
        result['training_eta_seconds'] = stages['train'].get('training_eta_seconds')
        if stages['train']['status'] == 'stopped':
            result['terminal_training_status'] = 'stopped'
    except READ_ERRORS as error:
        result['warnings'].append(str(error))
    return result
