"""Read one Root-bound provider352 run; metadata only, no discovery or actions.

The immutable descriptor is RUN_DESCRIPTOR_v1.json under CONTRACT_ROOT. Its
closed fields are schema, recorded_utc, run_root, plan{path,sha256},
request_sha256, run_config_sha256, supervisor{pid,create_time_epoch},
worker{pid,create_time_epoch}. Only exact PLAN/ROOT below are eligible.
TRAIN/DEV bodies, token arrays, tensors, private games and authority files are
never read. Journal is bounded, newline-complete, contiguous from update1.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import time

CONTRACT_ROOT='.local/monitor/provider352_fit'
DESCRIPTOR=CONTRACT_ROOT+'/RUN_DESCRIPTOR_v1.json'
DESCRIPTOR_SHA='bdabe34ba9c10519128aec65f9c0437a132ee82c5528014f19e6a668149cb9ef'
FIT_ROOT='.local/experiments/provider352_service_preservation_r16_last8_strict512_20261114'
STAGE='.local/staging/provider352_strict512_draft_20261114'
PLAN=STAGE+'/plans/PROVIDER352_FIT_ADMITTED_v1.json'
PLAN_SHA='ca582107fa244d5cdbecb3aa2ed910c9558385198564b9d189fdcba56673e52b'
MODEL_SHA='e126c678d420b65a2c5157c0bca0a3fb162cad6b96e4a067912418e0959d7461'
DATA_MANIFEST='proposals/service_preservation_small_world_20261114/provider352_full_cpu_20261114/FULL_FREEZE_MANIFEST.json'
DATA_SHA='72f480e68a11e13f8356af1d46a61d9d2a8e30fc9fcd12ff9f950f4ab716c12c'
AUDIT='.local/staging/provider352_full_cpu_20261114/TOKEN_AUDIT.json'
AUDIT_SHA='6d6d44143cef16e46253be5bb7e36ffe07f21a510a1fb89100235b83c38a7a71'
SOURCE={'compact_fit_common.py':'b878f01cad0c274f18031de5ab1bcb65ec732f426011090265257d0978e744cd',
        'compact_fit_worker.py':'87b26b5ad0758c2088200cd0ba0d141c8d49f6bb61e974eeb882a83ad24de45c',
        'compact_fit_supervisor.py':'1e9435dd72c20fa9ca452ef0f70eed55a2505d31729519ff9d736863c4f8d0c2',
        'compact_fit_native_guard.py':'cb75b2817d3e07981ee5b3b88788df0aa45906a83fb30925cfc611d3efc66448',
        'FIT_NATIVE_GUARD_SCOPE_v1.json':'4fdca0bd273cc3afdfb4907f2a9b6d791ae562f27ad56928ae07eb144ee98d62'}
MAX_METADATA_BYTES=2_000_000
MAX_JOURNAL_BYTES=4_000_000
STALE_SECONDS=300
ACTUAL_AUDIT=FIT_ROOT+'/reviews/ACTUAL_INDEPENDENT_COMPLETION_AUDIT_V3_v1.json'
ACTUAL_AUDIT_SHA='d9ac268723614db8a1334fd9dffe90e1a6dde4a28d3f74c4e88f0f9a60d2e532'
ACTUAL_AUDIT_PEER='.local/reviews/provider352_actual_completion_independent_20261115/ACTUAL_REPORT_PROOF_REVIEW_v1.json'
ACTUAL_AUDIT_PEER_SHA='b27cb10e4fc2525fa3907bd0a302ddd828001fc3646198eb9e72af65470470ce'


def _actual_audit_verified(root,dev_loss):
    try:
        audit=_read(root,ACTUAL_AUDIT,ACTUAL_AUDIT_SHA)
        peer=_read(root,ACTUAL_AUDIT_PEER,ACTUAL_AUDIT_PEER_SHA)
        return (audit.get('status')=='PASS_PROVIDER352_SAVED_FINAL_COMPLETION' and audit.get('errors')==[]
                and audit.get('plan_sha256')==PLAN_SHA and audit.get('completed_verified_updates')==2160
                and audit.get('DEV_rows')==500 and audit.get('facts',{}).get('DEV_pooled_target_NLL')==dev_loss
                and audit.get('completion_auditor_sha256')=='79710b96753e119d33ccfbbbead0bede9a797e18995832ffe8b55ea98c804610'
                and peer.get('actual_report_sha256')==ACTUAL_AUDIT_SHA and peer.get('plan_sha256')==PLAN_SHA
                and peer.get('status')=='ACCEPT_ACTUAL_SAVED_COMPLETION_REPORT_AND_PROOF_JOINS_NO_DUPLICATE_PAYLOAD_SCAN'
                and peer.get('parent_OS_exit_observed') is False and peer.get('parent_controller_exit_code') is None)
    except (OSError,ValueError,TypeError,KeyError,AttributeError,OverflowError):
        return False


def _positive(x):return type(x)in(int,float)and math.isfinite(x)and x>0
def _count(x,total=2160):
    if type(x)is not int or not 0<=x<=total:raise ValueError('Invalid actual update count')
    return x


def _file(root,relative):
    if not isinstance(relative,str)or ':'in relative or '\\'in relative or relative.startswith('/')or '..'in PurePosixPath(relative).parts:
        raise ValueError('Metadata path is outside the repository')
    path=root
    for part in PurePosixPath(relative).parts:
        path=path/part;info=path.lstat()
        if stat.S_ISLNK(info.st_mode)or getattr(info,'st_file_attributes',0)&0x400:raise ValueError('Linked metadata rejected')
    if not path.resolve().is_relative_to(root)or not path.is_file():raise ValueError('Regular metadata file required')
    return path


def _bytes(root,relative,limit=MAX_METADATA_BYTES,digest=None):
    path=_file(root,relative)
    if path.stat().st_size>limit:raise ValueError('Metadata exceeds bound')
    with path.open('rb')as stream:raw=stream.read(limit+1)
    if len(raw)>limit:raise ValueError('Metadata exceeds bound')
    if digest is not None and (not isinstance(digest,str)or not re.fullmatch('[0-9a-f]{64}',digest)or hashlib.sha256(raw).hexdigest()!=digest):raise ValueError('Immutable metadata hash differs')
    return raw


def _read(root,relative,digest=None):
    value=json.loads(_bytes(root,relative,digest=digest))
    if type(value)is not dict:raise ValueError('Metadata object required')
    return value


def _identity(value):
    if type(value)is not dict or set(value)!={'pid','create_time_epoch'}or type(value['pid'])is not int or value['pid']<=0 or not _positive(value['create_time_epoch']):raise ValueError('Exact process identity required')
    return value


def _journal(root,run):
    path=run+'/updates.jsonl'
    try:raw=_bytes(root,path,MAX_JOURNAL_BYTES)
    except FileNotFoundError:return []
    rows=[]
    keys={'step','loss','target_tokens','gradient_norm','parameter_delta_L2','gradient_seconds','Adam_seconds',
          'training_target_tokens_per_compute_second','mlx_peak_bytes','available_bytes','source_scope'}
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b'\n'):break # Worker newline+fsync contract; partial tail is not an update.
        if len(line)>16384:raise ValueError('Journal line exceeds bound')
        row=json.loads(line)
        if type(row)is not dict or set(row)!=keys or row['step']!=len(rows)+1 or type(row['step'])is not int or len(rows)>=2160:raise ValueError('Actual journal must be contiguous from update1')
        if type(row['loss'])not in(int,float)or not math.isfinite(row['loss'])or row['loss']<0:raise ValueError('Invalid finite loss')
        if type(row['target_tokens'])is not int or not all(_positive(row[k])for k in('target_tokens','gradient_norm','parameter_delta_L2','gradient_seconds','Adam_seconds','training_target_tokens_per_compute_second')):raise ValueError('Journal lacks finite positive actual Adam change')
        speed=row['target_tokens']/(row['gradient_seconds']+row['Adam_seconds'])
        if not math.isclose(speed,row['training_target_tokens_per_compute_second'],rel_tol=1e-6,abs_tol=1e-9):raise ValueError('Journal throughput differs from measured compute')
        if any(type(row[k])is not int or row[k]<0 for k in('mlx_peak_bytes','available_bytes'))or row['source_scope']!='Actual adapter training; not inference token speed.':raise ValueError('Journal measurement scope differs')
        rows.append(row)
    return rows


def _processes(descriptor,absolute_plan):
    try:
        import psutil
        for role,filename in(('supervisor','compact_fit_supervisor.py'),('worker','compact_fit_worker.py')):
            item=descriptor[role];process=psutil.Process(item['pid'])
            if not process.is_running()or abs(process.create_time()-item['create_time_epoch'])>.01:return False
            argv=process.cmdline()
            if not any(Path(a).name==filename for a in argv):return False
            if '--plan-sha256'not in argv or argv[argv.index('--plan-sha256')+1]!=PLAN_SHA:return False
            if '--plan'not in argv or argv[argv.index('--plan')+1]!=absolute_plan:return False
        return True
    except Exception:return False


def snapshot(project,now=None,check_processes=False):
    root=Path(project).resolve();now=time.time()if now is None else now
    unavailable=dict(schema='research4.provider352-dashboard-fit.v1',status='unavailable',optimizer_updates=None,
                     total_updates=2160,dev_completed=None,dev_total=500,live_owner_verified=False,
                     note='No verified provider3.5.2 run metadata; preparation is separate from training.')
    try:
        raw_descriptor=_bytes(root,DESCRIPTOR,digest=DESCRIPTOR_SHA);d=json.loads(raw_descriptor)
        wanted={'schema','recorded_utc','run_root','plan','request_sha256','run_config_sha256','supervisor','worker'}
        if type(d)is not dict or set(d)!=wanted or d['schema']!='research4.provider352-dashboard-run-descriptor.v1':raise ValueError('Closed run descriptor required')
        stamp=datetime.fromisoformat(d['recorded_utc'].replace('Z','+00:00'))
        if stamp.tzinfo is None or stamp.timestamp()>now+300:raise ValueError('Invalid descriptor time')
        _identity(d['supervisor']);_identity(d['worker'])
        run=d['run_root']
        if not isinstance(run,str)or not re.fullmatch(re.escape(FIT_ROOT+'/runs/fit_')+'[0-9a-f]{32}',run):raise ValueError('Old alias or unrelated run rejected')
        if d['plan']!={'path':PLAN,'sha256':PLAN_SHA}:raise ValueError('Only exact provider352 admitted plan is eligible')
        plan=_read(root,PLAN,PLAN_SHA)
        if plan.get('source_sha256')!=SOURCE or plan.get('base_manifest_sha256')!=MODEL_SHA or plan.get('draft_template')is not False or plan.get('launch_authorized')is not True or plan.get('iterations')!=2160 or plan.get('post_training_DEV_rows')!=500:raise ValueError('Exact plan identity differs')
        if plan.get('dataset_manifest')!={'path':DATA_MANIFEST,'sha256':DATA_SHA}or plan.get('tokenization_audit')!={'path':AUDIT,'sha256':AUDIT_SHA}:raise ValueError('Dataset/audit binding differs')
        _read(root,DATA_MANIFEST,DATA_SHA);audit=_read(root,AUDIT,AUDIT_SHA)
        if audit.get('status')!='PASS'or audit.get('over_budget_rows')!=0:raise ValueError('Full-data audit does not pass')
        for name,digest in SOURCE.items():_bytes(root,STAGE+'/'+name,digest=digest)
        request=_read(root,run+'/request.json',d['request_sha256']);config=_read(root,run+'/run_config.json',d['run_config_sha256'])
        if set(request)!={'plan_path','plan_sha256','run_dir','owner','parent_pid','original_origin_monotonic','original_deadline_monotonic'}:raise ValueError('Actual request schema differs')
        absolute=request['run_dir'];suffix='/'+run
        if not isinstance(absolute,str)or not absolute.startswith('/')or not absolute.endswith(suffix)or '..'in PurePosixPath(absolute).parts:raise ValueError('Actual run path differs')
        absolute_plan=absolute[:-len(suffix)]+'/'+PLAN
        if request['plan_path']!=absolute_plan or request['plan_sha256']!=PLAN_SHA or request['owner']!=run.rsplit('fit_',1)[1]or request['parent_pid']!=d['supervisor']['pid']or not _positive(request['original_origin_monotonic'])or request['original_deadline_monotonic']!=request['original_origin_monotonic']+172800:raise ValueError('Actual request owner/clock/plan differs')
        if config.get('plan_sha256')!=PLAN_SHA or any(config.get(k)!=v for k,v in plan.items()):raise ValueError('Run config is not the exact plan')
        p=_read(root,run+'/progress.json')
        required={'phase','verified_updates','total_updates','rank','num_layers','learning_rate','updated_epoch','available_bytes','mlx_active_bytes','mlx_cache_bytes','mlx_peak_bytes'}
        if not required<=set(p)or set(p)-required-{'next_update','gradient_seconds','optimizer_step_observed','last_checkpoint'}:raise ValueError('Progress schema differs')
        verified=_count(p['verified_updates'])
        if type(p['total_updates'])is not int or p['total_updates']!=2160 or type(p['rank'])is not int or p['rank']!=16 or type(p['num_layers'])is not int or p['num_layers']!=8 or p['learning_rate']!=1e-5:raise ValueError('Progress recipe differs')
        if any(type(p[k])is not int or p[k]<0 for k in('available_bytes','mlx_active_bytes','mlx_cache_bytes','mlx_peak_bytes')):raise ValueError('Invalid progress memory fields')
        phase=p['phase']
        if phase not in{'loading_model','before_gradient','gradient_evaluated','Adam_evaluated','update_verified','paused','DEV_evaluating','complete'}:raise ValueError('Unknown progress phase')
        if phase=='loading_model'and verified!=0 or phase in{'DEV_evaluating','complete'}and verified!=2160 or phase=='update_verified'and verified==0:raise ValueError('Phase/counter mismatch')
        if phase in{'before_gradient','gradient_evaluated'}and(type(p.get('next_update'))is not int or p['next_update']!=verified+1):raise ValueError('Invalid next-gradient boundary')
        if phase=='Adam_evaluated'and(type(p.get('optimizer_step_observed'))is not int or p['optimizer_step_observed']!=verified+1):raise ValueError('Unverified Adam boundary differs')
        if not _positive(p['updated_epoch'])or p['updated_epoch']>now+300:raise ValueError('Invalid progress time')
        journal=_journal(root,run);count=min(verified,len(journal))
        terminal=None
        for filename in('supervisor_failure.json','closeout.json'):
            try:t=_read(root,run+'/'+filename)
            except FileNotFoundError:continue
            if terminal is not None:raise ValueError('Ambiguous terminal closeout')
            owner=t.get('owner',{});identity=t.get('identity',{})
            if (t.get('schema')!='research4.service-continuity-compact-fit-owned-closeout.v1'or t.get('plan_sha256')!=PLAN_SHA
                or t.get('run_dir')!=absolute or owner.get('owner')!=request['owner']or owner.get('pid')!=d['supervisor']['pid']
                or owner.get('process_created_at')!=d['supervisor']['create_time_epoch']or identity.get('pid')!=d['worker']['pid']
                or identity.get('create_time')!=d['worker']['create_time_epoch']or t.get('status')not in{'failed_partial_preserved','paused','complete'}):raise ValueError('Terminal plan/owner/process binding differs')
            terminal=t
        dev=None;dev_loss=None;status='saved_progress'
        if terminal:
            status={'failed_partial_preserved':'stopped','paused':'paused','complete':'complete'}[terminal['status']]
            if terminal.get('cleanup_confirmed')is not True or terminal.get('cleanup_uncertain_lock_retained')is not False:status='terminal_cleanup_unverified'
            saved=_read(root,run+'/TERMINAL_SAVED_MANIFEST.json',terminal['terminal_manifest_sha256'])
            if saved.get('schema')!='research4.service-continuity-compact-fit-terminal-saved.v1':raise ValueError('Terminal saved manifest differs')
            for name in('progress.json','updates.jsonl'):
                entry=saved.get('files',{}).get(name)
                if entry is not None:_bytes(root,run+'/'+name,MAX_JOURNAL_BYTES if name=='updates.jsonl'else MAX_METADATA_BYTES,entry['sha256'])
                elif name=='progress.json'or count:raise ValueError('Terminal manifest does not bind saved progress')
            if terminal['status']=='complete':
                if count!=2160 or terminal.get('completed_updates')!=2160 or status!='complete':raise ValueError('Completed closeout lacks complete verified counters/cleanup')
                result=_read(root,run+'/result.json',saved['files']['result.json']['sha256'])
                validation=_read(root,run+'/DEV.json',saved['files']['DEV.json']['sha256'])
                if result.get('status')!='complete'or result.get('plan_sha256')!=PLAN_SHA or result.get('completed_updates')!=2160 or result.get('base_files_verified_before_and_after')is not True or result.get('final_opened')is not False:raise ValueError('Result metadata does not establish completed training')
                if type(validation.get('rows'))is not int or validation['rows']!=500 or validation.get('no_updates')is not True or validation.get('final_opened')is not False or type(validation.get('pooled_target_NLL'))not in(int,float)or not math.isfinite(validation['pooled_target_NLL'])or validation['pooled_target_NLL']<0:raise ValueError('DEV completion is not established')
                dev=500;dev_loss=validation['pooled_target_NLL']
        live=_processes(d,absolute_plan)if check_processes and terminal is None else False
        if status=='saved_progress'and check_processes and not live:status='saved_owner_unconfirmed'
        age=max(0,now-p['updated_epoch'])
        last=journal[count-1]if count else None
        note='Saved worker counter capped by contiguous finite actual-Adam journal. Process liveness is separate from optimizer progress.'
        if len(journal)<verified:note+=' Journal trails the saved counter; only the smaller count is displayed.'
        return dict(schema='research4.provider352-dashboard-fit.v1',status='available',plan_sha256=PLAN_SHA,
                    run_root=run,descriptor_sha256=hashlib.sha256(raw_descriptor).hexdigest(),training_status=status,
                    phase=phase,optimizer_updates=count,worker_reported_updates=verified,journal_updates=len(journal),total_updates=2160,
                    dev_completed=dev,dev_total=500,dev_pooled_target_NLL=dev_loss,
                    preparation_verified=True,live_owner_verified=live,liveness_checked=check_processes,
                    stale=age>STALE_SECONDS,age_seconds=age,recorded_utc=datetime.fromtimestamp(p['updated_epoch'],timezone.utc).isoformat(),
                    last_training_loss=last['loss']if last else None,
                    last_training_target_tokens_per_compute_second=last['training_target_tokens_per_compute_second']if last else None,
                    measurement_scope='Last verified training update only; not inference speed or cumulative throughput.',
                    recorded_mlx_peak_bytes=p['mlx_peak_bytes'],
                    final_saved_audit_verified=_actual_audit_verified(root,dev_loss) if status=='complete' else False,
                    training_eta_seconds=None,note=note)
    except (OSError,ValueError,TypeError,KeyError,AttributeError,OverflowError):return unavailable
