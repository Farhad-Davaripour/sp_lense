"""Prospective human-authorized evaluation resource envelope; no allocation."""
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime,timedelta,timezone
import math,re
from .io_utils import digest,read,sha

RESOURCE_VERSION='bounded-actor-partial-validation-resource-v1'
AUTHORITY_FILE='HUMAN_PARTIAL_VALIDATION_AUTHORITY.json'
QUOTA_FILE='RESOURCE_QUOTA_OBSERVATION.json'
RESOURCE_FILE='RESOURCE_AMENDMENT_PLAN.json'
OPERATOR_FILE='OPERATOR_BACKSTOP_PLAN.json'
RESOURCE_INPUTS={AUTHORITY_FILE,QUOTA_FILE,RESOURCE_FILE,OPERATOR_FILE}
AUTHORITY_SHA='29627183b1aec418166413fd8cee71722d2425e5191fcc4f94745376cfa8852c'
CHECKPOINT_SHA='9998bb60e16186edcf03137ca0b4168a72a258cd6acdfb075faa9bcfce1a24e2'


def _number(value,name):
    if type(value) not in [int,float] or not math.isfinite(value):
        raise PermissionError('Finite numeric '+name+' required')
    return value


def _timestamp(value):
    if not isinstance(value,str):raise PermissionError('An actual timezone-aware observation timestamp is required')
    try:result=datetime.fromisoformat(value.replace('Z','+00:00'))
    except ValueError as error:raise PermissionError('Malformed observation timestamp') from error
    if result.tzinfo is None:raise PermissionError('An actual timezone-aware observation timestamp is required')
    return result.astimezone(timezone.utc)


def validate_authority(value):
    if not isinstance(value,dict):raise PermissionError('The verified human resource authority object is required')
    expected={'human_user_message_id':'01a17d4e-0c04-7be0-80f9-170a5eececc5',
        'execution_owner_thread_id':'01a175f0-fc99-76b1-8bb2-d64f22afbcd7',
        'prospective_resource_amendment_required':True,
        'resource_envelope_must_fit_fresh_quota_with_startup_export_teardown_margin':True,
        'receipt_is_not_scientific_admission_or_audit_PASS':True}
    if any(digest(value.get(k))!=digest(v) for k,v in expected.items()):
        raise PermissionError('The exact human partial-validation resource authority is required')
    return deepcopy(value)


def resource_plan(observation):
    if not isinstance(observation,dict):raise PermissionError('An actual quota observation object is required')
    if observation.get('account')!='farhad89':raise PermissionError('Actual existing-account quota observation required')
    quota=_number(observation.get('remaining_accelerator_seconds'),'available quota')
    _timestamp(observation.get('observed_at_UTC'))
    if not re.fullmatch('[0-9a-f]{64}',observation.get('evidence_sha256','')):
        raise PermissionError('The saved actual quota observation must be content bound')
    hard=min(30000,math.floor((quota-1800)/300)*300)
    if hard<10800:raise PermissionError('Insufficient quota for the minimum prospective resource envelope')
    return {'version':RESOURCE_VERSION,'authority_sha256':AUTHORITY_SHA,'checkpoint_sha256':CHECKPOINT_SHA,
        'quota_observation_sha256':digest(observation),'observed_available_accelerator_seconds':quota,
        'hard_timeout_seconds':hard,'setup_cap_seconds':1800,'worker_seconds':hard-3600,
        'export_reserve_seconds':1800,'work_before_export_seconds':hard-5400,
        'controller_tail_seconds':1800,'minimum_fresh_quota_margin_seconds':1800,
        'minimum_fresh_quota_seconds':hard+1800,'operator_stop_offset_seconds':hard-1200,
        'operator_ceiling_offset_seconds':hard,'scientific_cases_arms_and_criteria_unchanged':True,
        'partial_is_complete_partition_PASS':False,'automatic_scientific_retries':0}


def validate_plan(value,observation):
    expected=resource_plan(observation)
    if digest(value)!=digest(expected):raise PermissionError('Exact deterministic prospective resource plan required')
    return expected


def operator_plan(plan):
    return {'version':'bounded-actor-root-UI-backstop-v1','resource_policy_sha256':digest(plan),
        'authority_sha256':AUTHORITY_SHA,'checkpoint_sha256':CHECKPOINT_SHA,
        'origin':'exact_consumed_provider_submission_creation_UTC',
        'stop_offset_seconds':plan['operator_stop_offset_seconds'],
        'ceiling_offset_seconds':plan['operator_ceiling_offset_seconds'],
        'execution_owner_thread_id':'01a175f0-fc99-76b1-8bb2-d64f22afbcd7',
        'mechanism':'root_UI_stop_using_existing_owner_heartbeat',
        'scheduler_is_hard_watchdog':False,'new_watcher_or_worker':False,
        'actual_provider_origin_and_absolute_deadlines_required_after_submission':True}


def amend_stage(stage,authority,observation,plan):
    validate_authority(authority);plan=validate_plan(plan,observation)
    if stage.get('checkpoint_sha256')!=CHECKPOINT_SHA:
        raise PermissionError('The resource amendment preserves the exact independently audited frozen checkpoint')
    result=deepcopy(stage)
    for key in ['hard_timeout_seconds','worker_seconds','export_reserve_seconds']:
        result[key]=plan[key]
    result.update(resource_amendment_version=RESOURCE_VERSION,resource_authority_sha256=AUTHORITY_SHA,
        resource_policy_sha256=digest(plan),quota_observation_sha256=digest(observation),
        resource_amendment_plan=plan,operator_backstop_plan=operator_plan(plan),
        whole_notebook_deadline_enforcement='independent_root_UI_provider_age_backstop',
        declared_setup_cap_seconds=1800,declared_controller_tail_seconds=1800,
        actual_external_child_timer_required=True,actual_whole_notebook_timer_installed=False)
    return result


def read_resource_inputs(inputs):
    from pathlib import Path
    inputs=Path(inputs)
    if sha(inputs/AUTHORITY_FILE)!=AUTHORITY_SHA:raise PermissionError('Pinned human resource authority receipt changed')
    authority=validate_authority(read(inputs/AUTHORITY_FILE));observation=read(inputs/QUOTA_FILE)
    plan=validate_plan(read(inputs/RESOURCE_FILE),observation)
    if read(inputs/OPERATOR_FILE)!=operator_plan(plan):raise PermissionError('Exact source-bound operator backstop plan required')
    return authority,observation,plan


def setup_budget(stage,elapsed,maximum=None):
    elapsed=_number(elapsed,'setup elapsed seconds')
    if elapsed<0:raise PermissionError('Setup elapsed time cannot be negative')
    plan=_timings(stage)
    remaining=plan['setup_cap_seconds']-elapsed
    if remaining<=0:raise TimeoutError('Prospective setup cap reached; preserve this consumed attempt')
    if maximum is None:return remaining
    maximum=_number(maximum,'setup subprocess timeout')
    if maximum<=0:raise PermissionError('Positive setup subprocess timeout required')
    return min(maximum,remaining)


def child_launch_budget(stage,elapsed):
    setup_budget(stage,elapsed)
    plan=stage['resource_amendment_plan']
    if stage['hard_timeout_seconds']-elapsed<stage['worker_seconds']+plan['controller_tail_seconds']:
        raise TimeoutError('The complete fixed child and controller-tail windows must remain')
    return stage['worker_seconds']


def _timings(stage):
    plan=stage.get('resource_amendment_plan',{});hard=plan.get('hard_timeout_seconds')
    if (stage.get('resource_amendment_version')!=RESOURCE_VERSION or type(hard) is not int
            or not 10800<=hard<=30000 or hard%300 or stage.get('checkpoint_sha256')!=CHECKPOINT_SHA
            or stage.get('resource_policy_sha256')!=digest(plan)
            or any(stage.get(k)!=plan.get(k) for k in ['hard_timeout_seconds','worker_seconds','export_reserve_seconds'])
            or plan.get('setup_cap_seconds')!=1800 or plan.get('worker_seconds')!=hard-3600
            or plan.get('export_reserve_seconds')!=1800 or plan.get('controller_tail_seconds')!=1800):
        raise PermissionError('Exact internally consistent source-bound resource timings required')
    return plan


def tail_budget(stage,elapsed):
    elapsed=_number(elapsed,'whole notebook elapsed seconds');plan=_timings(stage)
    if elapsed<0:raise PermissionError('Notebook elapsed time cannot be negative')
    remaining=min(plan['controller_tail_seconds'],stage['hard_timeout_seconds']-elapsed)
    if remaining<=0:raise TimeoutError('Registered notebook resource ceiling reached')
    return remaining


@contextmanager
def deadline_alarm(seconds):
    """A Linux deadline for blocking setup/finalization, not a provider stop."""
    import signal
    seconds=_number(seconds,'local deadline seconds')
    if seconds<=0:raise TimeoutError('No declared local deadline remains')
    if not hasattr(signal,'SIGALRM'):raise RuntimeError('The registered Linux setup/finalization timer is required')
    def expired(*args):raise TimeoutError('Prospective setup/finalization deadline reached')
    previous=signal.signal(signal.SIGALRM,expired)
    old_timer=signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        if old_timer[0]>0:signal.setitimer(signal.ITIMER_REAL,*old_timer)


def validate_controller_resource(stage,controller,allow_unknown=False):
    """Bind captured controller facts without turning incomplete facts into PASS."""
    if type(allow_unknown) is not bool:raise PermissionError('An explicit complete or unknown-evidence mode is required')
    if 'resource_amendment_version' not in stage:
        return {'resource_amendment':False,'legacy_validation_unchanged':True}
    plan=_timings(stage);unknown=[]
    if controller is None:
        if not allow_unknown:raise PermissionError('Actual resource controller evidence is required')
        return {'resource_amendment':True,'resource_policy_sha256':stage['resource_policy_sha256'],
            'controller_evidence_present':False,'complete_resource_observation':False,
            'unknown_fields':['controller'],'scientific_admission_created':False}
    if not isinstance(controller,dict):raise PermissionError('Captured controller evidence must be an object')
    expected={'resource_amendment_version':RESOURCE_VERSION,'resource_policy_sha256':stage['resource_policy_sha256'],
        'operator_backstop_plan':stage['operator_backstop_plan'],'declared_hard_timeout_seconds':stage['hard_timeout_seconds'],
        'declared_child_timeout_seconds':stage['worker_seconds'],'declared_setup_cap_seconds':1800,
        'declared_controller_tail_seconds':1800,'actual_whole_notebook_timer_installed':False,
        'whole_notebook_deadline_enforcement':'independent_root_UI_provider_age_backstop',
        'notebook_clock_origin':'first_cell_execution_monotonic_not_provider_creation'}
    def require(name,value):
        if name not in controller or controller[name] is None:
            if not allow_unknown:raise PermissionError('Missing actual controller resource fact: '+name)
            unknown.append(name);return None
        if digest(controller[name])!=digest(value):
            raise PermissionError('Actual controller contradicts its admitted resource plan: '+name)
        return controller[name]
    for name,value in expected.items():require(name,value)
    child_record=controller.get('child')
    if child_record is None:child=None
    elif isinstance(child_record,dict):child=child_record.get('child_started')
    else:raise PermissionError('Actual child evidence must be an object or unknown')
    if child not in [True,False,None] or type(child) not in [bool,type(None)]:
        raise PermissionError('Actual child-start status must be true, false or unknown')
    if child is not True and not allow_unknown:raise PermissionError('A complete resource observation requires an actual child')
    installed=controller.get('actual_external_child_timer_installed')
    tail_installed=controller.get('actual_controller_tail_timer_installed')
    for name,value in [('actual_external_child_timer_installed',installed),('actual_controller_tail_timer_installed',tail_installed)]:
        if value is None:
            if not allow_unknown:raise PermissionError('Missing actual controller timer fact: '+name)
            unknown.append(name)
        elif type(value) is not bool:raise PermissionError('Actual timer observations must be booleans')
        elif not value and not allow_unknown:raise PermissionError('Complete actual timer installation required')
    if child is False and installed is True:raise PermissionError('An unstarted child cannot have an installed child timer')
    if child is None:unknown.append('child.child_started')
    if child is True or 'external_child_timeout_argument_seconds' in controller:
        require('external_child_timeout_argument_seconds',stage['worker_seconds'])
    if installed is True:
        require('actual_child_timeout_seconds',stage['worker_seconds'])
    elif controller.get('actual_child_timeout_seconds') is not None:
        raise PermissionError('An uninstalled/unknown child timer cannot claim an actual timeout value')
    measured={}
    for name,ceiling in [('setup_completed_elapsed_seconds',plan['setup_cap_seconds']),('seconds',stage['hard_timeout_seconds'])]:
        value=controller.get(name)
        if value is None:
            if not allow_unknown:raise PermissionError('Missing actual controller elapsed time: '+name)
            unknown.append(name);measured[name]=None
        else:
            value=_number(value,name)
            if value<0:raise PermissionError('Actual controller elapsed time cannot be negative')
            measured[name]=value<=ceiling
            if measured[name] is False and not allow_unknown:raise PermissionError('Complete controller exceeded its admitted resource envelope')
    complete=(child is True and installed is True and tail_installed is True
        and not unknown and all(value is True for value in measured.values()))
    return {'resource_amendment':True,'resource_policy_sha256':stage['resource_policy_sha256'],
        'controller_evidence_present':True,'child_started':child,'external_child_timer_installed':installed,
        'controller_tail_timer_installed':tail_installed,'elapsed_within_declared_bounds':measured,
        'whole_notebook_timer_installed':False if controller.get('actual_whole_notebook_timer_installed') is False else None,
        'complete_resource_observation':complete,'unknown_fields':sorted(set(unknown)),
        'actual_operator_stop_required_for_earlier_completion':False,'scientific_admission_created':False}


def operator_receipt(binding,provider_created_at_UTC):
    """Make a source-bound stop plan after exact submission, never claim it ran."""
    created=_timestamp(provider_created_at_UTC);plan=binding.get('operator_backstop_plan',{})
    if (binding.get('resource_amendment_version')!=RESOURCE_VERSION
            or type(binding.get('scriptVersionId')) is not int or binding['scriptVersionId']<=0
            or not re.fullmatch('[0-9a-f]{40}',binding.get('source_commit',''))
            or not binding.get('identity') or binding.get('checkpoint_sha256')!=CHECKPOINT_SHA
            or plan.get('resource_policy_sha256')!=binding.get('resource_policy_sha256')
            or plan.get('stop_offset_seconds')!=binding.get('hard_timeout_seconds',0)-1200
            or plan.get('ceiling_offset_seconds')!=binding.get('hard_timeout_seconds')
            or plan.get('checkpoint_sha256')!=CHECKPOINT_SHA or plan.get('authority_sha256')!=AUTHORITY_SHA):
        raise PermissionError('An exact consumed source-bound resource submission is required')
    return {'version':'bounded-actor-operator-deadlines-v1','source_commit':binding['source_commit'],
        'identity':binding['identity'],'scriptVersionId':binding['scriptVersionId'],
        'resource_policy_sha256':binding['resource_policy_sha256'],'checkpoint_sha256':CHECKPOINT_SHA,
        'provider_created_at_UTC':created.isoformat(),
        'operator_stop_at_UTC':(created+timedelta(seconds=plan['stop_offset_seconds'])).isoformat(),
        'allocation_ceiling_at_UTC':(created+timedelta(seconds=plan['ceiling_offset_seconds'])).isoformat(),
        'actual_stop_performed':False,'scheduler_is_hard_watchdog':False}
