"""Read-only, exact-candidate telemetry. No model/data bodies or discovery.

The fixed STATE file is Root's display pointer, not execution authority. Its
descriptor is NULL until Root binds the actual reviewed run. PC snapshots never
interpret Mac PIDs locally. Dynamic metadata is bounded; only durable newline
terminated Adam records count. Native reserve is distinct from psutil available.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import time
import io
from zoneinfo import ZoneInfo

CANDIDATE = 'replication_diversity84_v1_20261116'
SCHEMA = 'research4.replication-diversity84-dashboard-fit.v1'
MONITOR = '.local/monitor/' + CANDIDATE
STATE = MONITOR + '/STATE_v1.json'
DESCRIPTOR = MONITOR + '/RUN_DESCRIPTOR_v1.json'
FIT_ROOT = '.local/experiments/replication_positive84_last8_native_short_R2_20261116'
OWNER_KIND = 'replication_positive84_last8_native_short_R2'
AUTHORITY = {'path': '.local/reviews/replication_diversity84_v1_20261116/DIRECT_HUMAN_POSITIVE_REPLICATION_FIT_AUTHORITY_v1.json',
             'sha256': '5891e2d7f66ff7a42b3d8feb977dd286789e2bcbad3eb5145ba21519333823bd'}
CPU_PACKET = {'path': 'proposals/replication_diversity84_v1_20261116/CPU_REVIEW_PACKET_v1.json',
              'sha256': 'b6cefa1923081e47586209f738886c23683371f226d6302a11b615474a9afc63'}
CPU_ACCEPTANCE = {'path': '.local/reviews/replication_diversity84_v1_20261116/MAIN_POSITIVE84_CPU_DATA_ACCEPT_v1.json',
                  'sha256': '543b9ed271f8587554d4c873ea1e267983fd258cca2c019cee80c82eae609d4e'}
NATIVE_POLICY = 'free_plus_pageable_filebacked_ESTIMATE.fit_NORMAL_only.6GiB_launch_2GiB_run.native_pressure_vm_clock.v1'
MODEL_SHA = 'e126c678d420b65a2c5157c0bca0a3fb162cad6b96e4a067912418e0959d7461'
RUN_FLOOR = 2147483648
LAUNCH_FLOOR = 6442450944
STALE_SECONDS = 300
CHECKPOINT_STEPS = (1, 2, 3, 84, 168)
RECOVERY_AUDIT = {'path': FIT_ROOT+'/reviews/ACTUAL_SAVED_COMPLETION_RECOVERY_AUDIT_v2.json',
                  'sha256': '78bc3847a174e0b16f62bbcf4ea19737383028a601cd4d71cc8463b4383820aa'}
ARTIFACT_READY = {'path': '.local/checkpoints/self_preservation_replication_last8_r16/sp-20261116-10-bdacaf88/ARTIFACT_READY_v1.json',
                  'sha256': '1403de9daddd814a3400ce698aba04ed34a44fd972ec9d5c153d4685f6190164'}
POSTFIT_BINDING = {'path': MONITOR+'/ACTUAL_POSTFIT_ARTIFACT_BINDING_v1.json',
                   'sha256': '3caa233e06abeab10bc1a66c2363ac791fb3ba19d11d98fd4e76908c9205a0bc'}
OWNED_RELEASE = {'path': MONITOR+'/ACTUAL_OWNED_RESOURCE_RELEASE_v1.json',
                 'sha256': '1f1350399bf9a631a27616933c33e204e16d9ada0aa0d28a0cc463b2fd248aca'}
TZIF = {'path': '.local/metadata/timezones/tzdata2026.6/America_Edmonton.tzif',
        'sha256': 'd52173b56a28205d8d6f792bcfa8bc07614ee0d269f7bf6e43ec91f68d410600'}
FINAL_ADAPTER_SHA = 'bdacaf88cd78597b2469b67e0956998d56c48f966cf40eb5c5d4e70d8eb4cbc0'
FINAL_CONFIG_SHA = '44d55729106a8687288264e7a1463d34803f73192e50ce79f05a984cc05d8340'


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _finite(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _path(root, relative):
    _need(type(relative) is str and relative and '\\' not in relative and ':' not in relative
          and not PurePosixPath(relative).is_absolute() and '..' not in PurePosixPath(relative).parts,
          'Canonical relative metadata path required')
    path = root / relative
    _need(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), 'Metadata path escapes project')
    _need(stat.S_ISREG(path.stat().st_mode), 'Regular metadata file required')
    return path


def _bytes(root, relative, maximum=2_000_000, digest=None):
    path = _path(root, relative)
    _need(path.stat().st_size <= maximum, 'Metadata exceeds bounded size')
    raw = path.read_bytes()
    _need(len(raw) <= maximum and (digest is None or hashlib.sha256(raw).hexdigest() == digest), 'Metadata hash differs')
    return raw


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, 'Duplicate JSON field')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def _read(root, relative):
    return _json(_bytes(root, relative))


def _ref(root, ref):
    _need(type(ref) is dict and set(ref) == {'path', 'sha256'} and _sha(ref['sha256'])
          and type(ref['path']) is str and ref['path'].endswith('.json'), 'Exact JSON metadata reference required')
    prefixes = (MONITOR+'/', '.local/staging/replication_positive84', '.local/staging/replication_diversity84',
                '.local/reviews/replication_diversity84', '.local/replication_diversity84',
                'proposals/replication_diversity84')
    exact_postfit = (RECOVERY_AUDIT['path'], ARTIFACT_READY['path'])
    _need(ref['path'].startswith(prefixes) or ref['path'] in exact_postfit, 'Unrelated metadata reference rejected before reading')
    return _json(_bytes(root, ref['path'], digest=ref['sha256']))


def _identity(value, worker=False):
    wanted = {'pid', 'create_time_epoch'} | ({'pgid'} if worker else set())
    _need(type(value) is dict and set(value) == wanted and type(value['pid']) is int and value['pid'] > 0
          and _finite(value['create_time_epoch'], 1), 'Exact process identity required')
    if worker:
        _need(value['pgid'] == value['pid'] and type(value['pgid']) is int, 'Owned worker group required')


def _base(note='No actual run is bound.'):
    return {'schema': SCHEMA, 'candidate': CANDIDATE, 'status': 'preparing', 'phase': 'PREPARING',
            'optimizer_updates': 0, 'total_updates': 168, 'counts_status': 'PLANNED',
            'unique_TRAIN_rows_planned': 84, 'epochs_planned': 2, 'DEV_rows_planned': 24,
            'actual_unique_TRAIN_rows': None, 'actual_DEV_rows': None, 'completed_epochs': 0,
            'rank': 16, 'layer_indices': list(range(56, 64)), 'positive_replication_only': True,
            'run_root': None, 'owner': None, 'plan_sha256': None, 'source_packet_sha256': None,
            'data_packet_sha256': None, 'source_sha256': None, 'live_owner_verified': False,
            'native_reserve': None, 'last_training_loss': None, 'input_tokens_per_compute_second': None,
            'target_tokens_per_compute_second': None, 'DEV_completed': None, 'DEV_target_NLL': None,
            'checkpoints': [], 'requested_run_floor_bytes': RUN_FLOOR,
            'artifact_ready_verified': False, 'endpoint_ready': False, 'serving_model_loaded': False,
            'launch_floor_bytes': LAUNCH_FLOOR, 'run_floor_admitted': False,
            'pressure_required': 1, 'recorded_utc': None, 'display_timezone': 'America/Edmonton',
            'note': note, 'scope': 'Saved metadata only; no model readiness or behavioral-improvement claim.'}


def _local_stamp(root, value):
    raw = _bytes(root, TZIF['path'], maximum=32768, digest=TZIF['sha256'])
    zone = ZoneInfo.from_file(io.BytesIO(raw), key='America/Edmonton')
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    _need(stamp.tzinfo is not None, 'Timezone-aware source timestamp required')
    return stamp.astimezone(zone).isoformat()


def _display_stamps(root, result):
    try:
        if result.get('recorded_utc'):
            result['recorded_local'] = _local_stamp(root, result['recorded_utc'])
        if result.get('native_reserve'):
            result['native_reserve']['saved_file_local'] = _local_stamp(root, result['native_reserve']['saved_file_utc'])
        result['display_timezone_source'] = dict(TZIF)
    except (OSError, ValueError, TypeError):
        result['recorded_local'] = None
        result['display_timezone_source'] = None
    return result


def _postfit_artifact(root, state, d, result):
    _need(state.get('recovery_audit_ref') == RECOVERY_AUDIT and state.get('artifact_ready_ref') == ARTIFACT_READY,
          'Only the exact audited recovery and READY references are eligible')
    _need(d['owner'] == '35a05cf45ff248cb8a9e8a4feaaa00b6'
          and d['plan']['sha256'] == '52b10199f203f351b72c3fc8bacdb51a9bdf313acef8201914362044b4389346'
          and result['optimizer_updates'] == 168 and result['DEV_completed'] == 24, 'Exact completed run/DEV scope required')
    audit = _ref(root, RECOVERY_AUDIT); ready = _ref(root, ARTIFACT_READY); binding = _ref(root, POSTFIT_BINDING)
    _need(audit.get('schema') == 'research4.replication-positive84-saved-artifact-recovery-audit.v2'
          and audit.get('status') == 'PASS_NUMERIC_ARTIFACT_RECOVERY_CONTROLLER_FAILED'
          and audit.get('owner') == d['owner'] and audit.get('plan') == d['plan']
          and audit.get('source_packet') == d['source_packet'] and audit.get('data_packet') == d['data_packet']
          and audit.get('completed_updates') == 168 and audit.get('epochs') == 2
          and audit.get('unique_TRAIN_decisions') == 84 and audit.get('TRAIN_target_presentations') == 7414
          and audit.get('DEV_rows') == 24 and audit.get('DEV_target_tokens') == 1037
          and audit.get('final_adapter_sha256') == FINAL_ADAPTER_SHA
          and audit.get('original_controller_status') == 'failed_partial_preserved'
          and audit.get('parent_OS_exit_code') == 1 and audit.get('worker_OS_exit_code') is None
          and audit.get('worker_OS_exit_observed') is False
          and audit.get('normal_parent_OS0_or_controller_success_claimed') is False, 'Audited artifact/controller outcome join differs')
    failure_ref = audit['original_controller_failure']
    _need(failure_ref == {'path':d['run_root']+'/supervisor_failure.json',
                         'sha256':'8573d4ecaac143ff958045a19a812943dc9e18145ac1a9041b9fa89ad732aed7'},
          'Original failed-controller reference differs')
    failure = _json(_bytes(root, failure_ref['path'], digest=failure_ref['sha256']))
    _need(failure.get('status') == 'failed_partial_preserved' and failure.get('exit_code') is None
          and failure.get('error') == {'type':'ProcessLookupError','message':'[Errno 3] No such process'},
          'Original controller failure must remain preserved')
    _need(set(binding) == {'expected_ft_id','adapter_sha256','adapter_config_sha256','base_manifest_sha256',
                          'artifact_ready','ready_utc','completion_audit','fit_updates','teacher_dev_rows'}
          and binding['expected_ft_id'] == 'sp-20261116-10-bdacaf88'
          and binding['adapter_sha256'] == FINAL_ADAPTER_SHA and binding['adapter_config_sha256'] == FINAL_CONFIG_SHA
          and binding['base_manifest_sha256'] == MODEL_SHA and binding['artifact_ready'] == ARTIFACT_READY
          and binding['completion_audit'] == RECOVERY_AUDIT and binding['fit_updates'] == 168 and binding['teacher_dev_rows'] == 24,
          'Closed nine-field postfit binding differs')
    _need(ready.get('schema') == 'sp-lense.research4-adapter-artifact-ready.v1'
          and ready.get('status') == 'ARTIFACT_READY_RECOVERY_AUDITED_NOT_ENDPOINT_READY'
          and ready.get('expected_ft_id') == binding['expected_ft_id']
          and ready.get('original_run') == d['run_root']
          and ready.get('original_checkpoint') == d['run_root']+'/adapters/step_000168'
          and ready.get('completion_audit') == RECOVERY_AUDIT and ready.get('completion_audit_status') == audit['status']
          and ready.get('adapter_sha256') == FINAL_ADAPTER_SHA and ready.get('adapter_config_sha256') == FINAL_CONFIG_SHA
          and ready.get('base_manifest_sha256') == MODEL_SHA and ready.get('fit_updates') == 168
          and ready.get('teacher_dev_rows') == 24 and ready.get('teacher_dev_target_tokens') == 1037
          and ready.get('controller_OS_exit_code') == 1 and ready.get('worker_OS_exit_code') is None
          and ready.get('controller_inner_cause') == 'UNPROVEN' and ready.get('original_controller_failure_preserved') is True
          and ready.get('endpoint_ready') is False and ready.get('serving_model_loaded') is False,
          'Artifact READY must not imply normal controller success or endpoint readiness')
    _need(ready['ready_utc'] == binding['ready_utc'] and ready['timezone_data']['TZif'] == TZIF
          and ready['timezone_data']['offset_seconds_at_READY'] == -21600
          and ready['ready_local'] == _local_stamp(root, ready['ready_utc']), 'Pinned Alberta READY timestamp differs')
    nll = audit['teacher_forced_DEV_target_NLL']
    _need(_finite(nll) and nll == ready['teacher_forced_DEV_target_NLL'] == result['DEV_target_NLL'], 'Teacher NLL metadata differs')
    _need(audit.get('release_proof') == OWNED_RELEASE, 'Recovered artifact release reference differs')
    release = _ref(root, OWNED_RELEASE)
    _need(release.get('owner') == d['owner'] and release.get('plan_sha256') == d['plan']['sha256']
          and release.get('supervisor') == d['supervisor'] and release.get('worker') == d['worker']
          and release.get('parent_os_exit_code') == 1 and release.get('worker_exit_code') is None
          and release.get('all_owned_pids_absent') is True and release.get('owned_worker_process_group_absent') is True
          and release.get('shared_model_lease_absent') is True and release.get('manual_release_of_exact_owned_orphan_lease') is True
          and release.get('original_parent_failure_preserved') is True, 'Independent resource release metadata differs')
    return {'phase':'AUDITED_ARTIFACT_READY_NOT_ENDPOINT_READY','artifact_ready_verified':True,
            'artifact_state':'NUMERIC_ARTIFACT_RECOVERED','training_controller_status':'FAILED_PRESERVED',
            'original_controller_failure':failure_ref,'original_controller_parent_OS_exit':1,
            'original_controller_worker_OS_exit':None,'original_controller_inner_cause':'UNPROVEN',
            'resources_release_metadata_verified':True,'live_owner_verified':False,
            'artifact_ready_ref':dict(ARTIFACT_READY),'recovery_audit_ref':dict(RECOVERY_AUDIT),
            'artifact_id':ready['expected_ft_id'],'artifact_adapter_sha256':FINAL_ADAPTER_SHA,
            'artifact_config_sha256':FINAL_CONFIG_SHA,'artifact_ready_utc':ready['ready_utc'],
            'artifact_ready_local':ready['ready_local'],'recorded_utc':ready['ready_utc'],
            'endpoint_ready':False,'serving_model_loaded':False,
            'note':'Independent numeric artifact recovery verified. Original controller FAILED (parent OS1; worker OS unknown; inner cause unproven). Resources released; endpoint not ready. Teacher NLL is not a behavioral result.'}


def _preparation_data(root, ref):
    _need(ref == CPU_PACKET, 'Only the exact frozen positive84 CPU packet is eligible')
    packet = _ref(root, CPU_PACKET); acceptance = _ref(root, CPU_ACCEPTANCE)
    _need(packet.get('schema') == 'research4.replication-diversity84-CPU-candidate-packet.v1'
          and packet.get('candidate_id') == CANDIDATE
          and packet.get('status') == 'FROZEN_CPU_REVIEW_CANDIDATE_NO_ACTIVATION_TRAINING_OR_NATIVE_ADMISSION',
          'Frozen positive84 CPU packet identity differs')
    files = packet.get('files')
    _need(type(files) is dict and len(files) == 11, 'Exact11-member CPU packet required')
    normalized = set()
    for name, item in files.items():
        _need(type(name) is str, 'Safe CPU metadata member key required')
        name = name.replace('\\', '/')
        _need(not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts and ':' not in name
              and name.startswith(('proposals/replication_diversity84_v1_20261116/',
                                   'scenarios/service_preservation/replication_diversity84_v1_20261116/'))
              and name not in normalized and type(item) is dict and _sha(item.get('sha256'))
              and type(item.get('bytes')) is int and item['bytes'] > 0, 'Unsafe or duplicate CPU metadata member')
        normalized.add(name)
    _need(acceptance.get('schema') == 'sp-lense.replication-positive84-data-acceptance.v1'
          and acceptance.get('status') == 'ACCEPT_FROZEN_POSITIVE_DATA_CPU_NOT_MODEL_ADMISSION'
          and acceptance.get('candidate_id') == CANDIDATE and acceptance.get('packet') == CPU_PACKET
          and acceptance.get('model_source_guard_and_live_width_capacity_admission_still_required') is True
          and acceptance.get('training_or_behavioral_success_claimed') is False, 'Exact CPU data-only acceptance required')
    for split, episodes, count, targets, maximum in (('TRAIN', 21, 84, 3707, 512), ('DEV', 6, 24, 1037, 510)):
        stats = packet['stats'][split]
        _need(all(type(stats.get(k)) is int and stats[k] == v for k, v in {
            'episodes': episodes, 'unique_decisions': count, 'target_tokens': targets,
            'max_total_rendered_tokens': maximum, 'max_unclipped_native_raw_width': 513,
            'max_target_tokens': 52}.items()), 'Frozen CPU split statistics differ')
        _need(acceptance[split].get('unique_decisions') == count
              and acceptance[split].get('max_complete_length') == maximum, 'CPU review/split counts differ')
    recipe = packet['recipe']; reviewed = acceptance['recipe']
    _need(recipe.get('epochs') == reviewed.get('epochs') == 2
          and recipe.get('updates') == reviewed.get('updates') == packet['stats']['TRAIN']['unique_decisions']*recipe['epochs'] == 168
          and recipe.get('target_presentations') == reviewed.get('target_presentations') == 3707*2 == 7414
          and recipe.get('rank') == reviewed.get('rank') == 16
          and recipe.get('Last8_layers') == reviewed.get('layer_indices') == list(range(56, 64))
          and reviewed.get('max_native_input') == 512 and reviewed.get('max_raw_batch') == 513
          and reviewed.get('head_slots') == 52, 'CPU recipe/width metadata differs')
    # Member metadata is checked above. Bodies/caches/tensors are never opened.
    return {'counts_status': 'DATA_CPU_VERIFIED', 'actual_unique_TRAIN_rows': 84, 'actual_DEV_rows': 24,
            'TRAIN_episodes': 21, 'DEV_episodes': 6, 'TRAIN_target_tokens_per_pass': 3707,
            'TRAIN_target_presentations': 7414, 'DEV_target_tokens': 1037,
            'max_TRAIN_complete_tokens': 512, 'max_DEV_complete_tokens': 510,
            'max_native_input_width': 512, 'max_raw_batch_width': 513, 'max_target_tokens': 52,
            'data_packet_sha256': CPU_PACKET['sha256'], 'data_acceptance_sha256': CPU_ACCEPTANCE['sha256'],
            'data_accepted_utc': acceptance['recorded_utc'],
            'note': 'Frozen CPU packet and data-only acceptance verified; no actual run or runtime admission is bound.'}


def _descriptor(root, ref):
    _need(type(ref) is dict and ref.get('path') == DESCRIPTOR, 'Only the exact new candidate descriptor is allowed')
    d = _ref(root, ref)
    wanted = {'schema', 'candidate', 'recorded_utc', 'run_root', 'owner', 'supervisor', 'worker',
              'plan', 'source_packet', 'data_packet', 'guard_profile', 'source_sha256',
              'request_sha256', 'run_config_sha256', 'original_origin_monotonic',
              'original_deadline_monotonic', 'epoch_index_base', 'final_DEV_rows', 'checkpoint_steps'}
    _need(type(d) is dict and set(d) == wanted and d['schema'] == 'research4.replication-diversity84-dashboard-run-descriptor.v1'
          and d['candidate'] == CANDIDATE, 'Closed exact-candidate run descriptor required')
    _need(type(d['owner']) is str and re.fullmatch('[0-9a-f]{32}', d['owner']) is not None
          and d['run_root'] == FIT_ROOT + '/runs/fit_' + d['owner'], 'Historical or unrelated run rejected')
    _identity(d['supervisor']); _identity(d['worker'], True)
    _need(_sha(d['request_sha256']) and _sha(d['run_config_sha256']), 'Actual request/config hashes required')
    stamp = datetime.fromisoformat(d['recorded_utc'].replace('Z', '+00:00'))
    _need(stamp.tzinfo is not None, 'Timezone-aware descriptor timestamp required')
    _need(_finite(d['original_origin_monotonic'], 1)
          and d['original_deadline_monotonic'] == d['original_origin_monotonic'] + 14400,
          'One original 14400-second clock required')
    _need(type(d['epoch_index_base']) is int and d['epoch_index_base'] in (0, 1)
          and type(d['final_DEV_rows']) is int and d['final_DEV_rows'] > 0
          and d['checkpoint_steps'] == list(CHECKPOINT_STEPS), 'Exact final counts/checkpoint metadata required')
    source = d['source_sha256']
    _need(type(source) is dict and source and all(type(k) is str and '/' not in k and _sha(v) for k, v in source.items()),
          'Exact reviewed source map required')
    plan = _ref(root, d['plan']); packet = _ref(root, d['source_packet'])
    _ref(root, d['data_packet']); _ref(root, d['guard_profile'])
    _need(plan.get('source_sha256') == source == packet.get('source_sha256')
          and plan.get('dataset_manifest') == d['data_packet'] == packet.get('data_packet'),
          'Exact source/plan/data metadata joins required')
    _need(plan.get('launch_authorized') is True and plan.get('draft_template') is False
          and plan.get('iterations') == 168 and plan.get('epochs') == 2
          and plan.get('train_rows_per_epoch') == 84 and plan.get('rank') == 16
          and plan.get('num_layers') == 8 and plan.get('layer_indices') == list(range(56, 64))
          and plan.get('save_every') == 84 and plan.get('early_checkpoint_steps') == [1, 2, 3]
          and plan.get('base_manifest_sha256') == MODEL_SHA and plan.get('memory_floor_bytes') == RUN_FLOOR
          and plan.get('post_training_DEV_rows') == d['final_DEV_rows'], 'Exact positive84 R2 plan required')
    run = d['run_root']
    request = _json(_bytes(root, run + '/request.json', digest=d['request_sha256']))
    config = _json(_bytes(root, run + '/run_config.json', digest=d['run_config_sha256']))
    _need(request.get('owner') == d['owner'] and request.get('parent_pid') == d['supervisor']['pid']
          and request.get('plan_sha256') == d['plan']['sha256']
          and request.get('original_origin_monotonic') == d['original_origin_monotonic']
          and request.get('original_deadline_monotonic') == d['original_deadline_monotonic']
          and config.get('plan_sha256') == d['plan']['sha256']
          and all(config.get(k) == v for k, v in plan.items()), 'Actual request/run-config identity differs')
    return d


def _journal(root, d):
    raw = _bytes(root, d['run_root'] + '/updates.jsonl')
    tail = bool(raw and not raw.endswith(b'\n'))
    lines = raw.splitlines()
    if tail:
        lines = lines[:-1]
    _need(len(lines) <= 168, 'Update journal exceeds the new finite dose')
    rows = []
    for step, line in enumerate(lines, 1):
        row = _json(line)
        _need(type(row.get('step')) is int and row['step'] == step
              and _finite(row.get('loss')) and _finite(row.get('gradient_norm'), 1e-300)
              and _finite(row.get('parameter_delta_L2'), 1e-300)
              and type(row.get('target_tokens')) is int and row['target_tokens'] > 0
              and _finite(row.get('gradient_seconds'), 1e-300) and _finite(row.get('Adam_seconds')),
              'Only contiguous finite durable Adam records count')
        _need(row.get('epoch_index') == (step-1)//84 + d['epoch_index_base'], 'Wrong epoch/row dose')
        if 'optimizer_step_observed' in row:
            _need(type(row['optimizer_step_observed']) is int and row['optimizer_step_observed'] == step, 'Adam step differs')
        _need(type(row.get('native_input_width')) is int and row['native_input_width'] > 0
              and row.get('raw_batch_width') == row['native_input_width'] + 1, 'Native input geometry differs')
        rows.append(row)
    return rows, tail


def _guard(root, d, check_processes):
    try:
        path = _path(root, d['run_root'] + '/guard_memory.jsonl')
        size = path.stat().st_size
        with path.open('rb') as stream:
            offset = max(0, size-65536); stream.seek(offset); raw = stream.read(65536)
        lines = raw.splitlines(keepends=True)
        if offset:
            lines = lines[1:]
        identities = {d[k]['pid']: d[k]['create_time_epoch'] for k in ('supervisor', 'worker')}
        for line in reversed(lines):
            if not line.endswith(b'\n'):
                continue
            event = _json(line); target = event.get('target', {}); assessment = event.get('assessment', {})
            if (event.get('event') != 'guard_sample' or target.get('pid') not in identities
                    or target.get('create_time_epoch') != identities[target['pid']]):
                continue
            # original_assessment belongs to the historical 4-GiB policy.
            if assessment.get('policy_id') != NATIVE_POLICY or assessment.get('valid') is not True:
                continue
            reserve = assessment.get('free_plus_pageable_filebacked_estimate_bytes')
            floor = assessment.get('floor_bytes'); sample = event.get('monotonic')
            if type(reserve) is not int or reserve < 0 or floor not in (RUN_FLOOR, LAUNCH_FLOOR) or not _finite(sample, 1):
                continue
            age = time.monotonic()-sample if check_processes else None
            return {'bytes': reserve, 'floor_bytes': floor, 'pressure_flag': assessment.get('pressure_flag_distinct_from_bytes'),
                    'stop_required': assessment.get('stop_required'), 'sample_monotonic': sample,
                    'age_seconds': age, 'fresh': age is not None and 0 <= age <= STALE_SECONDS,
                    'saved_file_utc': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                    'scope': 'Native R estimate from this run; saved-file time is not sample wall-clock time. Not psutil available.'}
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None


def _live(root, d):
    if platform.system() != 'Darwin':
        return False
    try:
        import os
        import psutil
        for key in ('supervisor', 'worker'):
            identity = d[key]; process = psutil.Process(identity['pid'])
            if abs(process.create_time()-identity['create_time_epoch']) > .01:
                return False
            expected = 'compact_fit_supervisor.py' if key == 'supervisor' else 'compact_fit_worker.py'
            if not any(Path(arg).name == expected for arg in process.cmdline()):
                return False
        if os.getpgid(d['worker']['pid']) != d['worker']['pgid']:
            return False
        lease = _read(root, '.local/validation/qwen38_shared_v1/inference.lock/owner.json')
        return (lease.get('kind') == OWNER_KIND and lease.get('owner') == d['owner']
                and lease.get('pid') == d['supervisor']['pid']
                and abs(lease.get('process_created_at', 0)-d['supervisor']['create_time_epoch']) <= .01)
    except Exception:
        return False


def snapshot(project, now=None, check_processes=False):
    root = Path(project).resolve(); now = time.time() if now is None else now
    result = _base()
    try:
        authority = _ref(root, AUTHORITY)
        _need(authority.get('schema') == 'sp-lense.direct-human-positive-replication-fit-authority.v1', 'Authority metadata differs')
        state = _read(root, STATE)
        fields = {'schema', 'candidate', 'recorded_utc', 'phase', 'run_descriptor'}
        _need(type(state) is dict and fields <= set(state) and set(state) <= fields | {'preparation_data_ref','recovery_audit_ref','artifact_ready_ref'}
              and state['schema'] == 'research4.replication-diversity84-dashboard-state.v1'
              and state['candidate'] == CANDIDATE and state['phase'] == 'PREPARING', 'Exact new candidate display state required')
        result['recorded_utc'] = state['recorded_utc']
        if state.get('preparation_data_ref') is not None:
            result.update(_preparation_data(root, state['preparation_data_ref']))
        if state['run_descriptor'] is None:
            _need(not state.get('recovery_audit_ref') and not state.get('artifact_ready_ref'), 'Postfit metadata requires exact run identity')
            return _display_stamps(root, result)
        d = _descriptor(root, state['run_descriptor']); rows, partial = _journal(root, d)
        if state.get('preparation_data_ref') is not None:
            _need(d['data_packet'] == state['preparation_data_ref'], 'Actual run differs from the CPU-verified packet')
        progress = _read(root, d['run_root'] + '/progress.json')
        count = len(rows)
        _need(type(progress.get('verified_updates')) is int and abs(progress['verified_updates']-count) <= 1
              and progress.get('total_updates') == 168 and progress.get('rank') == 16
              and progress.get('num_layers') == 8 and progress.get('learning_rate') == 1e-5
              and _finite(progress.get('updated_epoch'), 1), 'Progress/actual journal profile differs')
        stale = now-progress['updated_epoch'] > STALE_SECONDS or progress['updated_epoch'] > now+30
        live = bool(check_processes and platform.system() == 'Darwin' and _live(root, d) and not stale)
        phase = str(progress.get('phase', 'unknown'))
        public_phase = ('QUALIFYING' if 'qualif' in phase.lower() or any(x in phase for x in ('loading', 'materializ'))
                        else 'DEV' if phase == 'DEV_evaluating' else 'TRAINING' if count < 168 else 'FREEZE_PENDING')
        failure = None
        for name in ('supervisor_failure.json', 'worker_failure.json', 'partial.json', 'paused.json'):
            try:
                saved = _read(root, d['run_root'] + '/' + name)
                failure = name; break
            except FileNotFoundError:
                pass
        if failure:
            public_phase = 'FAILED' if failure != 'paused.json' else 'PAUSED'; live = False
        result.update(status='available', phase=public_phase, saved_phase=phase, optimizer_updates=count,
                      counts_status='BOUND_ACTUAL_PLAN', actual_unique_TRAIN_rows=84, actual_DEV_rows=d['final_DEV_rows'],
                      completed_epochs=count//84, current_epoch=min(2, (count-1)//84+1) if count else 0,
                      run_root=d['run_root'], owner=d['owner'], plan_sha256=d['plan']['sha256'],
                      source_packet_sha256=d['source_packet']['sha256'], data_packet_sha256=d['data_packet']['sha256'],
                      source_sha256=d['source_sha256'], live_owner_verified=live, stale=stale,
                      recorded_utc=datetime.fromtimestamp(progress['updated_epoch'], timezone.utc).isoformat(),
                      native_reserve=_guard(root, d, bool(check_processes and platform.system() == 'Darwin')),
                      run_floor_admitted=True, partial_journal_tail=partial, failure_metadata=failure,
                      note='Durable saved Adam records; Mac process liveness is checked only on the Mac.')
        if rows:
            last = rows[-1]; seconds = last['gradient_seconds'] + last['Adam_seconds']
            result.update(last_training_loss=last['loss'],
                          input_tokens_per_compute_second=last['native_input_width']/seconds,
                          target_tokens_per_compute_second=last['target_tokens']/seconds,
                          input_rate_scope='Padded native body positions / (gradient + Adam compute seconds)',
                          target_rate_scope='Supervised target tokens / (gradient + Adam compute seconds)')
        for step in CHECKPOINT_STEPS:
            if step > count:
                continue
            directory = d['run_root'] + '/adapters/step_' + format(step, '06d')
            try:
                config = _read(root, directory + '/adapter_config.json'); verification = _read(root, directory + '/verification.json')
                _need(config.get('plan_sha256') == d['plan']['sha256'] and config.get('checkpoint_step') == step
                      and verification.get('all_finite') is True, 'Checkpoint metadata identity differs')
                result['checkpoints'].append(step)
            except FileNotFoundError:
                pass
        try:
            dev = _read(root, d['run_root'] + '/DEV.json')
            _need(count == 168 and dev.get('rows') == d['final_DEV_rows'] and dev.get('no_updates') is True
                  and _finite(dev.get('pooled_target_NLL')), 'DEV metadata differs')
            result.update(DEV_completed=dev['rows'], DEV_target_NLL=dev['pooled_target_NLL'])
        except FileNotFoundError:
            pass
        try:
            closeout = _read(root, d['run_root'] + '/closeout.json')
            _need(closeout.get('plan_sha256') == d['plan']['sha256'] and closeout.get('owner') == d['owner'], 'Closeout identity differs')
            if closeout.get('cleanup_confirmed') is True and closeout.get('lease_released') is True:
                result.update(phase='RELEASED' if count == 168 and result['DEV_completed'] is not None and not failure else public_phase,
                              release_metadata_verified=True, live_owner_verified=False)
        except FileNotFoundError:
            pass
        if state.get('recovery_audit_ref') is not None or state.get('artifact_ready_ref') is not None:
            try:
                result.update(_postfit_artifact(root, state, d, result))
            except (OSError, ValueError, TypeError, KeyError) as error:
                result.update(artifact_ready_verified=False, postfit_metadata_error=str(error),
                              note='Saved training/controller failure retained; artifact READY metadata could not be verified: '+str(error))
        return _display_stamps(root, result)
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        return dict(_base(str(error)), status='unavailable', phase='UNAVAILABLE', optimizer_updates=None)
