"""Local-only read-only dashboard with Windows and SSH Mac metrics."""
from datetime import datetime, timezone
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import types
from urllib.request import Request, urlopen
from urllib.parse import urlsplit

import psutil
from compact_validation import snapshot as compact_validation_snapshot, version_catalog_snapshot
from fine_tuning_steps import snapshot as fine_tuning_steps_snapshot

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
HISTORY = PROJECT.parents[1] / 'superseded' / 'sp_lense4' / 'research4'
PORT = int(os.environ.get('RESEARCH4_DASHBOARD_PORT', '8768'))
if not 1024 <= PORT <= 65535:
    raise ValueError('Invalid dashboard loopback port')
STUDIO_PROJECT = '/Users/fdavaripour/repos/Research4/research4'
MINI_ARCHIVE = PROJECT / '.local/dashboard/MINI_ARCHIVED_SNAPSHOT.json'


def evidence_root(relative):
    return PROJECT if (PROJECT / relative).is_file() else HISTORY
SSH_KEY = Path.home() / '.ssh' / 'research4_mac_studio_ed25519'
HOST = 'research4-mac-vpn'
TRANSFER_SCRIPT = 'transfer_legacy_archive_to_mac.py'
TRANSFER_BYTES = 68_248_267_839
TRANSFER_RECEIPT = '.local/monitor/mac_archive_transfer_verified.json'
last_io = None
artifact_ids = {}
training_history = deque()
gpu_cache = None
cache = {'sampled_utc': None, 'mac': None, 'mini': None, 'windows': None, 'transfer': None, 'current_heldout': None, 'new_curriculum_campaign': None, 'saved_project_status': None}
lock = threading.Lock()
HELDOUT_READER = '.local/experiments/general_preservation_r16_last8_v1/heldout_resident_endpoint_v1/read_status.py'
HELDOUT_READER_SHA = '55c8bcbe4aee98bb92ec3789796d8a8f5e978b7af3e5e6602f5fc5f7113523a6'
heldout_reader_module = None


def replication_diversity84_status(mac_data=None):
    """One exact candidate only; PC fallback never verifies Mac process IDs."""
    if isinstance(mac_data, dict) and mac_data.get('online') is True:
        value = mac_data.get('replication_diversity84_fit')
        if (isinstance(value, dict) and value.get('schema') == 'research4.replication-diversity84-dashboard-fit.v1'
                and value.get('candidate') == 'replication_diversity84_v1_20261116'
                and value.get('status') == 'available'):
            return value
    # Match the existing Mac probe's hot-load pattern: changing a read-only
    # observer/STATE does not require future server/model restarts.
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location('research4_replication84_PC_metadata', HERE/'replication_diversity84_fit.py')
    reader = module_from_spec(spec); spec.loader.exec_module(reader)
    return reader.snapshot(evidence_root('.local/monitor/CURRENT_EXPERIMENT_STATE.json'), check_processes=False)


def saved_project_status():
    """Only Root's PC project metadata; never infer process liveness."""
    try:
        path = evidence_root('.local/monitor/CURRENT_EXPERIMENT_STATE.json') / '.local/monitor/CURRENT_EXPERIMENT_STATE.json'
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 2_000_000:
            raise ValueError('Saved project metadata unavailable')
        state = json.loads(path.read_text(encoding='utf-8'))
        fit = state['current_research_fit_v1']; preparation = state['archival_reference_feasibility_v1']
        completed = fit['last_verified_durable_research_updates']; total = fit['total_research_updates']
        tests = preparation['focused_CPU_tests_passed']
        if (type(completed) is not int or type(total) is not int or not 0 <= completed <= total or total != 2160
                or type(tests) is not int or tests < 0 or not isinstance(fit['status'], str)
                or not isinstance(preparation['status'], str) or not isinstance(state['updated_utc'], str)):
            raise ValueError('Saved project metadata is inconsistent')
        authorization = (False if state.get('Root_model_execution_allowed') is False
                         and state.get('new_model_submissions_allowed') is False else None)
        diagnostic = None
        leaf_fixed = state.get('one_admitted_leaf_fix')
        ad_pruned = state.get('one_admitted_ad_pruned')
        request_fixed = state.get('one_admitted_align_reqfix')
        aligned = state.get('one_admitted_align_short')
        forward = state.get('one_admitted_fwdref_short')
        isolated = state.get('one_admitted_process_isolated_archive14')
        forward_method = leaf_fixed is not None or ad_pruned is not None or request_fixed is not None or aligned is not None or forward is not None
        recorded = leaf_fixed if leaf_fixed is not None else ad_pruned if ad_pruned is not None else request_fixed if request_fixed is not None else aligned if aligned is not None else forward if forward is not None else isolated if isolated is not None else state.get('one_admitted_archive_live14')
        expected_plans = (('3541e6c0cd658b55147188024259727cfb83f9d71863ea86101903846215d086',)
                          if leaf_fixed is not None else ('17e4f7836449f4b54f6fce36598f114bb1c9076912fd86d1ade446622b03a1c2',)
                          if ad_pruned is not None else ('f19d1da6e223dfb61270da308c9443de810b19c92298c412659157801b556d50',)
                          if request_fixed is not None else ('0bd0a8b8bd0597b8f92289515119135873c87483356b981e59f83fc1bd4cc3e0',)
                          if aligned is not None else ('be100733bad076bc87a9abf7d012c8689c3e08c83b63d508096afa370ad467c9',
                           '518249cd7d68f48de51e87645a0eae07d43a97afd8f394dcb258b043f4cabefc')
                          if forward is not None else ('8dad125686fe61492891c1efc369d74cdb016f6dd7c8e7ea634e41774e33872a',)
                          if isolated is not None else ('67325e0f049c6540170b96c76e06459defe80a968ab21183f15d30b49ffb09c0',))
        limited = (isinstance(recorded, dict) and recorded.get('one_diagnostic_only') is True
                   and recorded.get('primary_fit_or_retry_authorized') is False
                   and recorded.get('plan_sha256') in expected_plans)
        if limited:
            counts = [recorded.get('forward_saved_records' if forward_method else 'observed_saved_records'), recorded.get('candidate_saved_records')]
            status = recorded.get('status')
            if not isinstance(status, str) or any(type(count) is not int or not 0 <= count <= 7 for count in counts):
                diagnostic = {'status':'unavailable', 'note':'Saved diagnostic metadata is inconsistent'}
            else:
                label = ('publishing' if 'PUBLICATION' in status or status.startswith('PUBLISHING') else
                         'stopped' if any(word in status for word in ('TERMINAL','STOPPED','FAILED','CLOSED','COMPLETE')) else
                         'running' if 'RUNNING' in status and 'NOT_RUNNING' not in status and 'NOT_MODEL_RUNNING' not in status else
                         'not started')
                terminal = label == 'stopped'
                numerical_failure = terminal and ('LOSS_PARITY' in status or recorded.get('terminal_error') == 'Primal forward loss/count parity failed')
                closed = terminal and recorded.get('all_owned_identities_gone') is True and recorded.get('shared_model_lease_absent') is True
                diagnostic = {'status':label, 'saved_status':status, 'reference_saved_records':counts[0],
                              'terminal':terminal, 'closed':closed, 'numerical_check_failed':numerical_failure,
                              'candidate_initialization_failed':terminal and 'CANDIDATE_INITIALIZATION' in status,
                              'resource_guard_stopped':terminal and 'RESOURCE_GUARD' in status,
                              'resource_guard_note':('At stop: reserve estimate %.2f GiB; floor %.0f GiB' % (recorded['guard_stop_R_bytes']/1024**3,recorded['guard_stop_floor_bytes']/1024**3)) if terminal and type(recorded.get('guard_stop_R_bytes')) is int and type(recorded.get('guard_stop_floor_bytes')) is int else None,
                              'reference_label':'Reference values' if leaf_fixed is not None or ad_pruned is not None else 'Reference forward' if forward_method else 'Observed',
                              'candidate_label':'Candidate derivatives' if forward_method else 'Candidate',
                              'candidate_saved_records':counts[1], 'plan_sha256':recorded['plan_sha256'],
                              'intent':recorded.get('intent'), 'actual_run_root':recorded.get('actual_run_root'),
                              'current_partition':recorded.get('current_partition') if recorded.get('current_partition')
                                  in (('forward896','forward800','candidate896','candidate800') if forward_method else
                                      ('observed896','observed800','candidate896','candidate800')) else None,
                              'original_origin_monotonic':recorded.get('original_origin_monotonic'),
                              'scope':'Saved diagnostic metadata only; not fresh process liveness. No primary-fit or retry authorization.'}
                diagnostic['forward_saved_records' if forward_method else 'observed_saved_records'] = counts[0]
        execution_scope = ('one_exact_diagnostic_only' if limited and state.get('Root_model_execution_allowed') is True
                           and state.get('new_model_submissions_allowed') is False else
                           'not_authorized' if authorization is False else 'not_verified')
        return {'schema':'research4.saved-project-dashboard.v1', 'status':'available',
                'metadata_updated_utc':state['updated_utc'], 'read_utc':datetime.now(timezone.utc).isoformat(),
                'fit':{'status':fit['status'], 'failed':fit['status'].startswith('FAILED_'),
                       'recorded_updates':completed, 'total_updates':total},
                'preparation':{'status':preparation['status'], 'CPU_tests_passed':tests,
                               'ready_for_review':preparation['status']=='LOCAL_ARCHIVE7_LIVE14_REVIEW_PACKET_COMPLETE_NOT_MODEL_ADMITTED',
                               'draft_plan_sha256':preparation.get('new_distinct_draft_plan_sha256')},
                'new_model_execution_authorized':authorization,
                'execution_scope':execution_scope, 'diagnostic':diagnostic,
                'scope':'Saved PC project status only. Not a fresh Mac process or training-liveness observation.'}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {'schema':'research4.saved-project-dashboard.v1', 'status':'unavailable', 'note':str(error)}


def current_campaign_status(compact=False):
    """Offline saved-metadata fallback; never infer Mac process liveness on PC."""
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location('research4_campaign_saved_metadata', HERE/'campaign_status.py')
    reader = module_from_spec(spec); spec.loader.exec_module(reader)
    return (reader.compact_snapshot if compact else reader.snapshot)(evidence_root('.local/monitor/CURRENT_EXPERIMENT_STATE.json'))


def current_heldout_status():
    """PC metadata only; never read predictions/scores or contact inference."""
    global heldout_reader_module
    try:
        path = PROJECT/HELDOUT_READER
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 32768:
            raise ValueError('Current held-out metadata reader unavailable')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != HELDOUT_READER_SHA:
            raise ValueError('Current held-out metadata reader source differs')
        key = (str(path.resolve()), HELDOUT_READER_SHA)
        if heldout_reader_module is None or heldout_reader_module[0] != key:
            module = types.ModuleType('research4_current_heldout_metadata')
            module.__file__ = str(path.resolve())
            exec(compile(raw, str(path), 'exec'), module.__dict__)
            heldout_reader_module = (key,module)
        history = evidence_root(HELDOUT_READER.replace('read_status.py', 'ROOT_LAUNCH.json'))
        value = heldout_reader_module[1].snapshot(root=history)
        if value.get('schema') != 'research4.heldout-read-only-status.v1':
            raise ValueError('Current held-out telemetry schema differs')
        if value.get('verified_identity') is not True:
            return {'state':'unverified','verified_identity':False,'note':value.get('error','Launch identity is unverified')}
        arms = ('qwen3.8-27b-base','sp-lense-adaptor-last-8-layers')
        conditions = {}
        for arm in arms:
            teacher = value['files_by_phase_arm']['teacher'][arm]
            own = value['files_by_phase_arm']['own'][arm]
            saved, completed = teacher['score'], value['own_outcome_files'][arm]
            if type(saved) is not int or not 0 <= saved <= 360 or type(completed) is not int or not 0 <= completed <= 96:
                raise ValueError('Current held-out file counts differ')
            conditions[arm] = {'reference_saved':saved,'reference_planned':360,
                'own_completed':completed,'own_planned':96,
                'response_files':teacher['response']+own['response']}
        reference = sum(v['reference_saved'] for v in conditions.values())
        own = sum(v['own_completed'] for v in conditions.values())
        return {'schema':'research4.current-heldout-dashboard.v1','state':value['state'],
            'verified_identity':True,'process_live':value['process_live'] if history == PROJECT else False,'pid':value['pid'],
            'plan_sha256':value['plan_sha256'],'observed_utc':value['observed_utc'],
            'adapter_checkpoint':'d050d018d8311aaf941e837bd93b6effc7261e3e0890b7f471c851119f87929c',
            'phase':'reference_decisions' if reference<720 else 'own_trajectories' if own<192 else 'collection_saved',
            'reference':{'saved':reference,'planned':720},'own':{'completed':own,'planned':192},
            'conditions':conditions,'request_intent_files':value['progress']['request_intent_files'],
            'pending_unadmitted':len(value['pending_unadmitted']), 'latest_evidence':value['latest_evidence'],
            'terminal_result_file':value['terminal_result_file'],'halt_files':len(value['halt_files']),
            'failure_files':value['failure_files'],
            'note':'Saved-file progress only. Correctness and trajectory success are withheld until the completed-evidence audit; shared900 is historical.'}
    except (OSError,ValueError,KeyError,TypeError) as error:
        return {'state':'unverified','verified_identity':False,'note':str(error)}


def local_training_pause():
    """Keep the direct human pause visible even if SSH telemetry is offline."""
    try:
        path = evidence_root('.local/monitor/TRAINING_HPO_PAUSED_BY_HUMAN.json') / '.local/monitor/TRAINING_HPO_PAUSED_BY_HUMAN.json'
        if path.stat().st_size > 2_000_000: return None
        value = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value, dict): return None
        value['active'] = (value.get('schema') == 'research4.human-training-hpo-pause.v1'
            and value.get('status') == 'PAUSED_BY_DIRECT_HUMAN_INSTRUCTION'
            and value.get('training_allowed') is False and value.get('HPO_allowed') is False
            and value.get('scientific_qualification_launches_allowed') is False)
        return value
    except (OSError, ValueError, TypeError):
        return None


def prefect_request(path, method='GET', payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request('http://127.0.0.1:4200/api' + path, data=body, method=method,
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=4) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def publish(key, kind, value, description):
    artifact_id = artifact_ids.get(key)
    if not artifact_id:
        found = prefect_request('/artifacts/filter', 'POST', {'artifacts': {'key': {'any_': [key]}}, 'limit': 10})
        if found:
            artifact_id = found[0]['id']
        else:
            created = prefect_request('/artifacts/', 'POST', {'key': key, 'type': kind, 'data': value, 'description': description})
            artifact_id = created['id']
        artifact_ids[key] = artifact_id
    prefect_request('/artifacts/' + artifact_id, 'PATCH', {'data': value, 'description': description})


def human_bytes(value):
    return f'{value / 1024**3:.1f} GiB' if isinstance(value, (int, float)) else 'Unavailable'


def publish_snapshot(snapshot):
    win = snapshot['windows']; mac_data = snapshot['mac']; tx = snapshot['transfer']
    mem = win['memory']; m = mac_data if mac_data.get('online') else None
    mm = m.get('memory') if m else None; mg = m.get('gpu') if m else None
    job = m.get('job') if m else None; dataset = m.get('dataset') if m else None
    pause = snapshot.get('training_pause') or (m.get('training_pause') if m else None)
    paused = isinstance(pause, dict) and pause.get('active') is True
    endpoint = m.get('inference_endpoint') if m else None
    game_training = isinstance(job, dict) and job.get('model_label') in (
        'SP Lens game', 'sp-lense-adaptor-last-layer', 'sp-lense-adaptors-all-layers')
    training_label = job.get('display_model_label') or job.get('model_label') if game_training else 'Shared-adapter'
    stage_kind = job.get('stage_kind') if job else None
    stage_label = {'pilot': 'Compatibility pilot', 'diagnostic': 'Numerical equivalence check', 'trial': 'Hyperparameter trial',
                   'final': 'Final all-layer training'}.get(stage_kind, 'Training')
    stage_scope = {'pilot': 'Setup check only; this is not the final research adapter.',
                   'diagnostic': 'TRAIN-only numerical setup check; this is not a research candidate or final adapter.',
                   'trial': 'Short training trial; selection uses development loss, not game outcomes.',
                   'final': 'Final fit of the development-selected configuration; game evaluation is separate.'}.get(stage_kind, '')
    if paused:
        stage_label = 'Historical training / pilot record'
        stage_scope = 'Training/HPO paused by direct human instruction. Statistics below are historical; they are not final selected-model progress.'
    manifest_path = job.get('dataset_manifest_path', '') if game_training else ''
    version_match = re.fullmatch(r'\.local/datasets/sp_lens_game_v1/([a-zA-Z0-9_-]+)/manifest\.json', manifest_path)
    game_version = version_match.group(1) if version_match else 'pending'
    scenario_folder = 'sp_lens_game_v1/' + game_version if game_training else 'qwen38_shared_v1'
    data_folder = 'sp_lens_game_v1/' + game_version if game_training else 'qwen38_shared_v1'
    validation = m.get('validation') if m else None
    heldout = m.get('heldout_results') if m else None
    pc_gpu = (win.get('gpu') or [None])[0]
    def percent(value):
        return f'{value:.1f}%' if isinstance(value, (int, float)) else 'Unavailable'
    def memory_text(value):
        return f"{human_bytes(value['total_bytes']-value['available_bytes'])} / {human_bytes(value['total_bytes'])}" if value else 'Unavailable'
    def safe(value):
        return value if isinstance(value, (int, float)) else None
    state = job.get('status', 'unknown') if job else ('Not started' if m else 'Unavailable: Mac offline')
    if state == 'completed' and stage_kind in ('pilot', 'trial', 'diagnostic'):
        state = 'Stage complete'
    speed = safe(job.get('tokens_per_second')) if job else None
    done = job.get('completed_steps') if job else None; total = job.get('total_steps') if job else None
    train_progress = f'{done} / {total} optimizer steps' if isinstance(done, int) and isinstance(total, int) and total > 0 else ('Not started â€” no verified training run' if m else 'Unavailable while the Mac is offline')
    if stage_kind == 'diagnostic':
        checks_done = job.get('diagnostic_checks_completed'); checks_total = job.get('diagnostic_checks_total')
        train_progress = (f'{checks_done} / {checks_total} numerical checks' if isinstance(checks_done, int)
                          and isinstance(checks_total, int) and checks_total > 0 else 'Numerical check in preparation')
    transfer_speed = safe(tx.get('bytes_per_second'))
    transfer_speed_text = f'{transfer_speed / 1024**2:.1f} MiB/s' if transfer_speed is not None else 'Unavailable'
    step_rate = safe(job.get('steps_per_minute')) if job else None
    step_rate_text = f'{step_rate:.1f} steps/min' if step_rate is not None else 'Waiting for enough steps'
    note = job.get('note') if job else None
    if note == 'available_memory_below_4_GiB':
        note = 'Stopped because available memory fell below the 4 GiB safety threshold.'
    vstate = validation.get('status', 'unknown') if isinstance(validation, dict) else ('Not started' if m else 'Unavailable: Mac offline')
    condition = validation.get('condition') if isinstance(validation, dict) else None
    vdone = validation.get('completed_prompts') if isinstance(validation, dict) else None
    vtotal = validation.get('total_prompts') if isinstance(validation, dict) else None
    vspeed = safe(validation.get('tokens_per_second')) if isinstance(validation, dict) else None
    vprogress = f'{vdone} / {vtotal} prompts' if isinstance(vdone, int) and isinstance(vtotal, int) and 0 <= vdone <= vtotal and vtotal > 0 else 'Not available'
    # Status labels are displayed in Markdown; do not interpolate arbitrary text.
    vstate = vstate if vstate in ('starting', 'running', 'completed', 'failed', 'stale', 'unverified', 'Not started', 'Unavailable: Mac offline') else 'unknown'
    condition = condition if condition in ('base', 'step_0450', 'step_0900', 'step_1359') else 'Not available'
    phase = validation.get('phase') if isinstance(validation, dict) else None
    phase = phase if phase in ('preflight', 'loading_model', 'generating', 'verifying', 'complete') else 'Not available'
    def heldout_text(name):
        data = (heldout.get('conditions') or {}).get(name) if isinstance(heldout, dict) and heldout.get('status') == 'complete' else None
        if not isinstance(data, dict): return 'Unavailable'
        n, exact, limited, unique_n = (data.get(k) for k in ('n', 'exact', 'token_limit', 'unique_n'))
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in (n, exact, limited, unique_n)) or not (0 <= exact <= n and n > 0 and 0 <= limited <= unique_n and unique_n > 0): return 'Unavailable'
        return f'{exact}/{n} next-action matches ({100 * exact / n:.1f}%); {limited}/{unique_n} unique outputs hit the token limit'
    heldout_state = ('Complete' if isinstance(heldout, dict) and heldout.get('status') == 'complete' else
                     'Unavailable or failed integrity check' if heldout is not None else 'Not available' if m else 'Mac offline')
    rows = [
        '# Research4 Â· Local lab',
        '**Model:** Qwen3.8-27B 4-bit  |  **Design:** one shared adapter  |  **Danger label:** hidden',
        f"**Updated:** {snapshot['sampled_utc']}",
        '**Training/HPO paused; inference endpoint setup.** No automatic resume or retry.' if paused else '',
        '', '## Local BASE + shared LAST8 inference endpoint',
        f"**Status:** {endpoint.get('status', 'unavailable') if isinstance(endpoint, dict) else 'Mac unavailable'}",
        f"**Models:** {', '.join(endpoint.get('model_ids', [])) if isinstance(endpoint, dict) else 'Not verified'}",
        'Shared rank16 blocks56â€“63 checkpoint d050. Historical one-layer b0a1 remains separate; this is read-only endpoint metadata, not a generated model probe.',
        '', f'## {training_label} Â· {stage_label}', f'**Status:** {state}', f'**Progress:** {train_progress}',
        stage_scope,
        f"**Stage / rank:** {job.get('stage', 'Training')} / {job.get('rank', 'Not reported')}" if job else '',
        f"**Adapted layers:** {str(job.get('num_layers')) + ' of 64' if job and isinstance(job.get('num_layers'), int) else 'Not yet reported'}",
        f"**{'Historical ' if paused else ''}supervised training-token speed:** {f'{speed:.1f} tokens/s' if speed is not None else 'Unavailable until training reports it'}",
        f'**Step rate:** {step_rate_text}',
        f"**Latest loss:** {job.get('loss', 'â€”') if job else 'â€”'}",
        'Training loss is not a held-out scenario or exploitation result.',
        f"**Run note:** {note or 'Waiting for a verified run.'}",
        '', '## Previous adapter Â· development inference', f'**Status:** {vstate}', f'**Condition:** {condition}',
        f'**Phase:** {phase}', f'**Progress:** {vprogress}',
        f"**Output-token speed:** {f'{vspeed:.1f} tokens/s' if vspeed is not None else 'Not yet available'}",
        'Speed is generated output tokens divided by generation wall time including prompt prefill. Completion remains subject to independent origin audit.',
        '', '## Previous adapter Â· held-out next-action check', f'**Status:** {heldout_state}',
        f"**Selected adapter step:** {heldout.get('selected_step') if isinstance(heldout, dict) and heldout.get('status') == 'complete' else 'Unavailable'}",
        f"**Base:** {heldout_text('base_comparator')}",
        f"**Shared adapter:** {heldout_text('selected_shared_adapter')}",
        'Teacher-forced next-action agreement only; not autonomous self-preservation or exploitation. The base comparison is confounded by outputs hitting the token limit.',
        f"**Scenario dataset:** {dataset.get('training_examples', 'Preparing') if dataset else 'Preparing'}",
        f'**Scenario YAML:** `~/repos/Research4/research4/scenarios/{scenario_folder}/`',
        f'**Training JSONL:** `~/repos/Research4/research4/.local/datasets/{data_folder}/`',
        '', '## Machine resources', '| Resource | Earlier Mac Studio (historical) | Windows PC |', '|---|---:|---:|',
        f"| CPU | {percent(m.get('cpu_percent') if m else None)} | {percent(win['cpu_percent'])} |",
        f"| Memory used / total | {memory_text(mm)} | {memory_text(mem)} |",
        f"| GPU activity | {percent(mg.get('utilization_percent') if mg else None)} | {percent(pc_gpu.get('utilization_percent') if pc_gpu else None)} |",
        f"| GPU-reported allocation (shared memory) | {human_bytes(mg.get('reported_allocation_bytes')) if mg else 'Unavailable'} | {human_bytes(pc_gpu.get('used_bytes')) if pc_gpu else 'Unavailable'} |",
        f"| Disk free | {human_bytes(m['disk']['free_bytes']) if m else 'Unavailable'} | {human_bytes(win['disk']['free_bytes'])} |",
        '', 'Mac memory is unified. Windows GPU activity uses NVIDIA telemetry or the busiest Windows GPU engine; the Mac uses device utilization. Missing readings mean unavailable, not zero.',
        '', '## Previous research archive',
        f"**Status:** {tx['status'].replace('_', ' ')}",
        f"**Approximate data sent to SSH:** {human_bytes(tx.get('bytes_sent_approx'))} of {human_bytes(tx.get('bytes_total'))}",
        f'**Transfer speed:** {transfer_speed_text}',
        tx.get('note', ''), '',
        f"**Long-context memory diagnostic:** {(m.get('memory_diagnostic') or {}).get('status') or 'Not started'}" if m else '**Long-context memory diagnostic:** Unavailable',
        'Synthetic setup tests are not the research adapter or study results.'
    ]
    publish('research4-live-resources', 'markdown', '\n'.join(rows), 'Live local resources, training status, and archive transfer.')
    if tx.get('status') in ('streaming', 'verifying', 'paused_for_memory', 'verified'):
        publish('research4-archive-transfer', 'progress', tx['percent_approx'],
                tx.get('note', '100% requires independent verification on the Mac.'))


def gpu_windows():
    exe = shutil.which('nvidia-smi')
    if exe:
        try:
            result = subprocess.run([exe, '--query-gpu=utilization.gpu,memory.used,memory.total',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3, check=True)
            rows = []
            for line in result.stdout.strip().splitlines():
                util, used, total = [float(x.strip()) for x in line.split(',')]
                rows.append({'utilization_percent': util, 'used_bytes': int(used * 1024**2),
                             'total_bytes': int(total * 1024**2), 'source': 'nvidia-smi'})
            if rows: return rows
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    try:
        script = r"$ErrorActionPreference='Stop'; $s=(Get-Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 1 -MaxSamples 1 -ErrorAction Stop).CounterSamples; @($s | ForEach-Object { [pscustomobject]@{instance=$_.InstanceName;value=$_.CookedValue} }) | ConvertTo-Json -Compress"
        shell = shutil.which('pwsh') or shutil.which('powershell.exe')
        if not shell: return None
        result = subprocess.run([shell, '-NoProfile', '-NonInteractive', '-Command', script],
                                capture_output=True, text=True, timeout=10, check=True)
        samples = json.loads(result.stdout); samples = samples if isinstance(samples, list) else [samples]
        engines = {}
        for item in samples:
            name = str(item['instance']).lower(); value = float(item['value'])
            if not re.match(r'^pid_\d+_luid_', name) or not 0 <= value <= 100: continue
            engine = re.sub(r'^pid_\d+_', '', name)
            engines[engine] = engines.get(engine, 0.0) + value
        if engines:
            return [{'utilization_percent': min(100, max(engines.values())),
                     'source': 'Windows GPU Engine counters, busiest engine'}]
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        pass
    return None


def refresh_gpu():
    global gpu_cache
    while True:
        gpu_cache = gpu_windows()
        time.sleep(15)


def transfer():
    global last_io
    for proc in psutil.process_iter(['name', 'cmdline', 'create_time']):
        try:
            if (proc.info['name'] or '').lower() != 'python.exe':
                continue
            if not any(str(x).replace('\\', '/').endswith('/' + TRANSFER_SCRIPT) for x in (proc.info['cmdline'] or [])):
                continue
            count = proc.io_counters().write_bytes; now = time.monotonic(); speed = None
            if last_io and last_io[0] == proc.pid and last_io[1] == proc.info['create_time'] and now > last_io[2]:
                speed = max(0, (count - last_io[3]) / (now - last_io[2]))
            last_io = (proc.pid, proc.info['create_time'], now, count)
            return {'status': 'streaming', 'bytes_sent_approx': count, 'bytes_total': TRANSFER_BYTES,
                    'percent_approx': min(99.9, count / TRANSFER_BYTES * 100), 'bytes_per_second': speed,
                    'note': 'Bytes written to the SSH stream; destination verification is still pending.'}
        except (psutil.Error, OSError, AttributeError):
            continue
    try:
        data = json.loads((evidence_root(TRANSFER_RECEIPT) / TRANSFER_RECEIPT).read_text(encoding='utf-8'))
        if data.get('status') == 'verified':
            return {'status': 'verified', 'percent_approx': 100, 'note': 'Copied files were verified on the Mac.'}
    except (OSError, ValueError):
        pass
    return {'status': 'not_observed', 'note': 'Transfer is not currently observed; verification status is unknown.'}


def mac():
    remote = (f'"{STUDIO_PROJECT}/.local/qwen38-venv/bin/python" -B ' f'"{STUDIO_PROJECT}/dashboard/mac_probe.py"')
    try:
        result = subprocess.run(['ssh', '-i', str(SSH_KEY), '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
                                 '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=5', HOST, remote],
                                capture_output=True, text=True, timeout=9, check=True)
        data = json.loads(result.stdout)
        if data.get('project_root') != STUDIO_PROJECT or data.get('python') != STUDIO_PROJECT + '/.local/qwen38-venv/bin/python':
            raise ValueError('Live Studio probe came from an unexpected checkout')
        return {'online': True, **data}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {'online': False, 'error': 'Mac metrics are currently unavailable.'}


def mini():
    """Return the saved Mini record without executing code on the retired host path."""
    try:
        data = json.loads(MINI_ARCHIVE.read_text(encoding='utf-8'))
        if data.get('schema') != 'research4.all64-mini-dashboard.v1' or data.get('archive_only') is not True:
            raise ValueError('Archived Mini telemetry schema differs')
        return {**data, 'online': False, 'active': None, 'resources': None}
    except (OSError, ValueError, AttributeError):
        return {'online': False, 'machine': 'Mac mini', 'error': 'Archived Mini telemetry is unavailable.'}


def refresh_mini():
    while True:
        value = mini()
        with lock:
            cache['mini'] = value
        time.sleep(10)


def add_training_rate(mac_data):
    job = mac_data.get('job') if mac_data.get('online') else None
    if not isinstance(job, dict) or job.get('status') != 'running':
        training_history.clear(); return
    step = job.get('completed_steps'); total = job.get('total_steps'); run = job.get('prefect_flow_run_id') or job.get('run_dir')
    if not isinstance(step, int) or step < 1 or not isinstance(total, int) or total <= 0 or not run:
        return
    now = time.monotonic()
    if training_history and training_history[-1][0] != run:
        training_history.clear()
    if not training_history or step > training_history[-1][2]:
        training_history.append((run, now, step))
    while len(training_history) > 2 and now - training_history[0][1] > 300:
        training_history.popleft()
    if len(training_history) >= 2:
        first, last = training_history[0], training_history[-1]
        seconds = last[1] - first[1]
        if seconds > 0 and last[2] > first[2]:
            rate = (last[2] - first[2]) / seconds * 60
            if now - last[1] <= max(90, 180 / rate):
                job['steps_per_minute'] = rate
                job['eta_seconds'] = max(0, total - step) * 60 / rate


def refresh():
    psutil.cpu_percent(interval=None)
    last_published = 0
    while True:
        memory = psutil.virtual_memory(); disk = psutil.disk_usage(PROJECT.anchor)
        windows = {'cpu_percent': psutil.cpu_percent(interval=None),
                   'memory': {'total_bytes': memory.total, 'available_bytes': memory.available, 'percent': memory.percent},
                   'disk': {'total_bytes': disk.total, 'free_bytes': disk.free, 'percent': disk.percent}, 'gpu': gpu_cache}
        mac_data = mac(); add_training_rate(mac_data)
        transfer_data = transfer()
        verification = mac_data.get('archive_verification') if mac_data.get('online') else None
        if transfer_data['status'] == 'not_observed' and isinstance(verification, dict):
            status = verification.get('status'); checked = verification.get('files_checked'); total = verification.get('total_files')
            if status in ('verifying', 'paused_for_memory', 'verified', 'failed') and isinstance(checked, int) and isinstance(total, int) and total > 0:
                transfer_data = {'status': status, 'files_checked': checked, 'total_files': total,
                    'percent_approx': 100 if status == 'verified' else min(99.9, checked / total * 100),
                    'note': verification.get('note', 'Archive verification on the Mac.')}
        snapshot = {'sampled_utc': datetime.now(timezone.utc).isoformat(), 'windows': windows,
                    'transfer': transfer_data, 'mac': mac_data,
                    'saved_project_status': saved_project_status(),
                    'compact_validation': compact_validation_snapshot(evidence_root('.local/experiments/service_continuity_compact_r16_last8_v2_20261114/reviews/INDEPENDENT_COMPLETION_AUDIT_v1.json')),
                    'model_version_catalog': version_catalog_snapshot(evidence_root('.local/monitor/SP_LENS_MODEL_VERSIONS_ABT_20261114.json')),
                    'fine_tuning_steps': fine_tuning_steps_snapshot(evidence_root('.local/monitor/CURRENT_EXPERIMENT_STATE.json')),
                    'training_pause': local_training_pause(), 'current_heldout': current_heldout_status(),
                    'new_curriculum_campaign': mac_data.get('new_curriculum_campaign') if mac_data.get('online')
                    and mac_data.get('new_curriculum_campaign') else current_campaign_status(),
                    'compact_training': mac_data.get('compact_training') if mac_data.get('online')
                    and mac_data.get('compact_training') else current_campaign_status(compact=True)}
        with lock:
            cache.update(snapshot)
        if PORT == 8768 and time.monotonic() - last_published >= 15:
            try:
                publish_snapshot(snapshot); last_published = time.monotonic()
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print(f'Prefect update unavailable: {type(exc).__name__}', flush=True)
        time.sleep(5)


def page_html():
    page = (HERE / 'index.html').read_text(encoding='utf-8')
    try:
        payload = json.loads((PROJECT / '.local/dashboard/HISTORICAL_PAGE_PAYLOADS.json').read_text(encoding='utf-8'))
        for name in ('replica612-audited-record', 'preparation-receipt', 'completed-fit-receipt'):
            entry = payload.get(name)
            if isinstance(entry, dict):
                marker = f'<script type="application/json" id="{name}">'
                safe = json.dumps(entry, ensure_ascii=True, allow_nan=False).replace('<', r'\u003c')
                page = page.replace(marker + '{}', marker + safe, 1)
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return page.encode('utf-8')


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.headers.get('Host', '').split(':')[0] not in ('127.0.0.1', 'localhost'):
            self.send_error(403); return
        path = urlsplit(self.path).path
        if path == '/':
            body = page_html(); kind = 'text/html; charset=utf-8'
        elif path == '/fine-tuning-steps.js':
            body = (HERE / 'fine_tuning_steps_ui.js').read_bytes(); kind = 'text/javascript; charset=utf-8'
        elif path == '/all64-mini.js':
            body = (HERE / 'all64_mini_ui.js').read_bytes(); kind = 'text/javascript; charset=utf-8'
        elif path == '/api/replication-diversity84':
            with lock:
                mac_data = cache.get('mac')
            body = json.dumps(replication_diversity84_status(mac_data), allow_nan=False).encode()
            kind = 'application/json; charset=utf-8'
        elif path == '/api/status':
            with lock:
                value = dict(cache)
            value['replication_diversity84_fit'] = replication_diversity84_status(value.get('mac'))
            body = json.dumps(value, allow_nan=False).encode()
            kind = 'application/json; charset=utf-8'
        else:
            self.send_error(404); return
        self.send_response(200); self.send_header('Content-Type', kind)
        self.send_header('Cache-Control', 'no-store'); self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; base-uri 'none'")
        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

    def log_message(self, format, *args):
        pass


if __name__ == '__main__':
    print(f'Research4 live view: http://127.0.0.1:{PORT}/; Prefect: http://127.0.0.1:4200/', flush=True)
    threading.Thread(target=refresh_gpu, daemon=True).start()
    threading.Thread(target=refresh, daemon=True).start()
    threading.Thread(target=refresh_mini, daemon=True).start()
    server = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
