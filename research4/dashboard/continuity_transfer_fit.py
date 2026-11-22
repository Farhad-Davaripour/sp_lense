"""Read only the Root-pinned continuity_transfer run; no discovery or model actions.

Root writes the closed descriptor at DESCRIPTOR after launch and supplies its
file-byte SHA in DESCRIPTOR_SHA. Until that pin exists, only exact approved
preparation metadata is shown. No data bodies, token arrays or tensors are read.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import time

CONTRACT_ROOT = '.local/monitor/continuity_transfer_fit'
DESCRIPTOR = CONTRACT_ROOT + '/RUN_DESCRIPTOR_v1.json'
DESCRIPTOR_SHA = '8c1ed5f89c4cf488ef068d64236b4d64374859a6cc088a3c983ad792ae65bd77'  # Actual fresh run e74f3e2c; no discovery.
FIT_ROOT = '.local/experiments/continuity_transfer_r16_last8_20261115'
STAGE = '.local/staging/continuity_transfer_fit_20261115'
PLAN = STAGE + '/plans/CONTINUITY_TRANSFER_V3_SIX_PASS_ADMITTED_v1.json'
PLAN_SHA = '1d32fa6924303355eebc2acfeedd81d5e8e79ebe10e73d61777fe27bef3f0669'
HELD_PLAN = STAGE + '/plans/CONTINUITY_TRANSFER_V3_SIX_PASS_HELD_v1.json'
HELD_PLAN_SHA = 'd3f15c4c8a620c58ab2dfa68ba2ebf2ef7c14376fb4c050a8b338dd2591e0686'
TRAIN_PACKET = STAGE + '/TRAIN_REVIEW_PACKET_v1.json'
TRAIN_PACKET_SHA = '66afbd03a3411184aee1f488d44d3f62fde3b657924e84a989ed13849140cf0f'
MODEL_SHA = 'e126c678d420b65a2c5157c0bca0a3fb162cad6b96e4a067912418e0959d7461'
DATA_MANIFEST = 'proposals/continuity_transfer_20261115/CPU_REVIEW_PACKET_v3.json'
DATA_SHA = 'a3ae23992773605b10e04e96ed426e744300f2a785f7e98b5cb75d30a01d7fb0'
AUDIT = '.local/diagnosis/continuity_transfer_20261115_v1/cpu_packet_v4/CPU_AUDIT_v1.json'
AUDIT_SHA = '6fc2372340847a2d051a7e7ec42b225a162871ffef4a18b4ac9b7901d7f00d5e'
DATA_REVIEW = '.local/reviews/transfer_curriculum_20261115/EXACT_V3_CORPUS_AND_SIX_PASS_ACCEPTANCE_v1.json'
DATA_REVIEW_SHA = 'cada569adb262afd9e18a033e80b04f2675e800ea2739e529567a6d3c8a5444b'
DOSE = '.local/diagnosis/continuity_transfer_20261115_v1/cpu_packet_v4/DOSE_v3.json'
DOSE_SHA = '3df08eb5da27ba9bb1d69c8c467378a56a7156698c7274f63292a756b1d12f4d'
FINAL_AUDIT = FIT_ROOT + '/reviews/FINAL_SAVED_COMPLETION_AUDIT_v1.json'
FINAL_AUDIT_SHA = None
MAX_METADATA_BYTES = 2_000_000
MAX_JOURNAL_BYTES = 1_000_000
STALE_SECONDS = 300
CHECKPOINT_STEPS = (1, 2, 3, 180, 264)
NATIVE_POLICY = 'free_plus_pageable_filebacked_ESTIMATE.fit_NORMAL_only.6GiB_launch_4GiB_run.native_pressure_vm_clock.v1'
SOURCE = {'compact_fit_common.py': '3c584c7beb1db17cde1fbf373952ea917da016d27fb278ae0a1e63ceb0e5949c', 'compact_fit_worker.py': '10d0aedb936157f877ff976bb36324764ef13d9dd207b219dc247d5c269efa5c', 'compact_fit_supervisor.py': 'd3777653262cfbe8208b1189e268f34f4c027119a3d0418b5dd264b9050a4eba', 'compact_fit_native_guard.py': '8941fc3db8078f56e80da0d3767f251b8165d0bd05e312901fe08fc1fa6bfae4', 'FIT_NATIVE_GUARD_SCOPE_v1.json': 'e627700d78b904a4f68eae453cdc700cc4fee9110a041147b56bd6780a63da69'}


def _positive(x):return type(x)in(int,float)and math.isfinite(x)and x>0


def _count(x,total=264):
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


def _final_saved_audit(root):
    """Display only the measured, immutable postrelease CPU audit; no tensor I/O."""
    if not isinstance(FINAL_AUDIT_SHA, str) or not re.fullmatch('[0-9a-f]{64}', FINAL_AUDIT_SHA):
        return False
    try:
        report=_read(root,FINAL_AUDIT,FINAL_AUDIT_SHA)
        return (report.get('status')=='PASS_CONTINUITY_TRANSFER_SAVED_FINAL_COMPLETION'
                and report.get('errors')==[] and report.get('plan_sha256')==PLAN_SHA
                and report.get('completed_verified_updates')==264 and report.get('DEV_rows')==36)
    except (OSError,ValueError,TypeError,KeyError):
        return False


def _journal(root,run):
    path=run+'/updates.jsonl'
    try:raw=_bytes(root,path,MAX_JOURNAL_BYTES)
    except FileNotFoundError:return []
    rows=[]
    keys={'step','loss','target_tokens','gradient_norm','parameter_delta_L2','gradient_seconds','Adam_seconds',
          'training_target_tokens_per_compute_second','mlx_peak_bytes','available_bytes','source_scope',
          'epoch_index','data_row_id','data_row_index','complete_length','prefix_tokens',
          'native_input_width','raw_batch_width'}
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b'\n'):break # Worker newline+fsync contract; partial tail is not an update.
        if len(line)>16384:raise ValueError('Journal line exceeds bound')
        row=json.loads(line)
        if type(row)is not dict or set(row)!=keys or row['step']!=len(rows)+1 or type(row['step'])is not int or len(rows)>=264:raise ValueError('Actual journal must be contiguous from update1')
        if (type(row['epoch_index'])is not int or row['epoch_index']!=(row['step']-1)//44+1
                or not isinstance(row['data_row_id'],str)or not row['data_row_id']or len(row['data_row_id'])>256
                or type(row['data_row_index'])is not int or not 0<=row['data_row_index']<44
                or type(row['complete_length'])is not int or not 1<=row['complete_length']<=480
                or type(row['prefix_tokens'])is not int or not 0<=row['prefix_tokens']<row['complete_length']
                or row['target_tokens']!=row['complete_length']-row['prefix_tokens']
                or type(row['native_input_width'])is not int or row['native_input_width']!=32*((row['complete_length']+31)//32)
                or type(row['raw_batch_width'])is not int or row['raw_batch_width']!=row['native_input_width']+1):
            raise ValueError('Actual row/pass/target/native-width receipt differs')
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


def _prepared(note):
    return dict(schema='research4.continuity-transfer-dashboard-fit.v1', status='preparation',
                plan_sha256=PLAN_SHA, optimizer_updates=None, total_updates=264,
                dev_completed=None, dev_total=36, preparation_verified=True,
                live_owner_verified=False, unique_train_episodes=21, unique_train_rows=44,
                epochs=6, supervised_target_presentations=5934,
                paired_dev_completed=None, paired_dev_total=12, paired_dev_routes=['BASE','FT'], note=note)


def _preparation(root):
    # A held exact plan may establish preparation, never launch or progress.
    admitted = (root / PLAN).is_file()
    plan = _read(root, PLAN if admitted else HELD_PLAN, PLAN_SHA if admitted else HELD_PLAN_SHA)
    if (plan.get('source_sha256') != SOURCE or plan.get('base_manifest_sha256') != MODEL_SHA
            or plan.get('draft_template') is not (not admitted)
            or plan.get('launch_authorized') is not admitted
            or plan.get('iterations') != 264 or plan.get('post_training_DEV_rows') != 36
            or plan.get('epochs') != 6 or plan.get('train_rows_per_epoch') != 44
            or plan.get('supervised_tokens_total') != 5934
            or plan.get('rank') != 16 or plan.get('num_layers') != 8):
        raise ValueError('Only the exact six-pass continuity-transfer preparation is eligible')
    if (plan.get('dataset_manifest') != {'path': DATA_MANIFEST, 'sha256': DATA_SHA}
            or plan.get('tokenization_audit') != {'path': AUDIT, 'sha256': AUDIT_SHA}
            or plan.get('prospective_dose') != {'path': DOSE, 'sha256': DOSE_SHA}):
        raise ValueError('Exact corpus, audit and dose bindings required')
    packet = _read(root, TRAIN_PACKET, TRAIN_PACKET_SHA)
    audit = _read(root, AUDIT, AUDIT_SHA)
    dose = _read(root, DOSE, DOSE_SHA)
    _read(root, DATA_MANIFEST, DATA_SHA)
    _read(root, DATA_REVIEW, DATA_REVIEW_SHA)
    if (packet.get('source_sha256') != SOURCE or packet.get('fit_updates') != 264
            or audit.get('status') != 'PASS_CPU_ONLY_PENDING_REVIEW'
            or audit.get('splits', {}).get('train', {}).get('unique_rows') != 44
            or audit.get('splits', {}).get('train', {}).get('episodes') != 21
            or audit.get('splits', {}).get('dev', {}).get('unique_rows') != 36
            or dose.get('optimizer_updates') != 264 or dose.get('supervised_tokens_total') != 5934):
        raise ValueError('Exact preparation metadata differs')
    for name, digest in SOURCE.items():
        _bytes(root, STAGE + '/' + name, digest=digest)
    return plan


def _checkpoints(root,run,count):
    steps=[]
    for step in CHECKPOINT_STEPS:
        if step>count:continue
        directory=run+'/adapters/step_'+format(step,'06d')
        try:
            config=_read(root,directory+'/adapter_config.json')
            verification=_read(root,directory+'/verification.json')
        except (FileNotFoundError,json.JSONDecodeError):continue
        if (config.get('plan_sha256')!=PLAN_SHA or config.get('checkpoint_step')!=step
                or config.get('iters')!=264 or config.get('base_manifest_sha256')!=MODEL_SHA
                or config.get('dataset_manifest_sha256')!=DATA_SHA
                or verification.get('all_finite')is not True
                or verification.get('nonzero_lora_b_layer_indices')!=list(range(56,64))
                or not isinstance(verification.get('adapter_sha256'),str)
                or re.fullmatch('[0-9a-f]{64}',verification['adapter_sha256'])is None):
            raise ValueError('Published checkpoint metadata differs from this fit')
        steps.append(step)
    return dict(count=len(steps),planned=len(CHECKPOINT_STEPS),steps=steps,
                scope='Published config/verification metadata only; final tensor audit is separate.')


def _native_reserve(root,run,descriptor):
    # Bounded last complete guard frame; do not stream the resource journal.
    try:
        path=_file(root,run+'/guard_memory.jsonl')
        with path.open('rb')as stream:
            size=path.stat().st_size;offset=max(0,size-65536);stream.seek(offset);raw=stream.read(65537)
        lines=raw.splitlines(keepends=True)
        if offset:lines=lines[1:]
        identities={x['pid']:x['create_time_epoch']for x in(descriptor['supervisor'],descriptor['worker'])}
        for line in reversed(lines):
            if not line.endswith(b'\n'):continue
            event=json.loads(line)
            if event.get('event')!='guard_sample' or event.get('accepted_for_minimum')is not True:continue
            target=event.get('target',{});assessment=event.get('assessment',{})
            if target.get('pid')not in identities or target.get('create_time_epoch')!=identities[target['pid']]:continue
            value=assessment.get('free_plus_pageable_filebacked_estimate_bytes');sample=event.get('monotonic')
            if (assessment.get('valid')is not True or assessment.get('policy_id')!=NATIVE_POLICY
                    or type(value)is not int or value<0 or not _positive(sample)):continue
            age=time.monotonic()-sample
            if age<0 or age>300:return None
            return dict(bytes=value,age_seconds=age,phase=event.get('phase'),
                        pressure_flag=assessment.get('pressure_flag_distinct_from_bytes'),
                        stop_required=assessment.get('stop_required'),
                        scope='Native free plus pageable file-backed reserve estimate, not psutil available memory.')
    except (OSError,ValueError,TypeError,KeyError,AttributeError,OverflowError):pass
    return None


def snapshot(project,now=None,check_processes=False):
    root=Path(project).resolve();now=time.time()if now is None else now
    unavailable=dict(schema='research4.continuity-transfer-dashboard-fit.v1',status='unavailable',optimizer_updates=None,
                     total_updates=264,dev_completed=None,dev_total=36,live_owner_verified=False,
                     note='No verified continuity_transfer run metadata; preparation is separate from training.')
    try:
        try:
            plan=_preparation(root)
        except FileNotFoundError:
            if DESCRIPTOR_SHA is not None:
                raise
            value=_prepared('PREPARING: exact local plan/source metadata is not yet fully verified; no run identity is bound.')
            value['preparation_verified']=False
            return value
        if not isinstance(DESCRIPTOR_SHA,str) or not re.fullmatch('[0-9a-f]{64}',DESCRIPTOR_SHA):
            return _prepared('Awaiting the exact Root run identity; preparation does not count as training.')
        raw_descriptor=_bytes(root,DESCRIPTOR,digest=DESCRIPTOR_SHA);d=json.loads(raw_descriptor)
        wanted={'schema','recorded_utc','run_root','plan','request_sha256','run_config_sha256','supervisor','worker'}
        if type(d)is not dict or set(d)!=wanted or d['schema']!='research4.continuity-transfer-dashboard-run-descriptor.v1':raise ValueError('Closed run descriptor required')
        stamp=datetime.fromisoformat(d['recorded_utc'].replace('Z','+00:00'))
        if stamp.tzinfo is None or stamp.timestamp()>now+300:raise ValueError('Invalid descriptor time')
        _identity(d['supervisor']);_identity(d['worker'])
        run=d['run_root']
        if not isinstance(run,str)or not re.fullmatch(re.escape(FIT_ROOT+'/runs/fit_')+'[0-9a-f]{32}',run):raise ValueError('Old alias or unrelated run rejected')
        if d['plan']!={'path':PLAN,'sha256':PLAN_SHA}:raise ValueError('Only exact agent-continuity_transfer admitted plan is eligible')
        plan=_read(root,PLAN,PLAN_SHA)
        if plan.get('source_sha256')!=SOURCE or plan.get('base_manifest_sha256')!=MODEL_SHA or plan.get('draft_template')is not False or plan.get('launch_authorized')is not True or plan.get('iterations')!=264 or plan.get('post_training_DEV_rows')!=36:raise ValueError('Exact plan identity differs')
        if plan.get('dataset_manifest')!={'path':DATA_MANIFEST,'sha256':DATA_SHA}or plan.get('tokenization_audit')!={'path':AUDIT,'sha256':AUDIT_SHA}:raise ValueError('Dataset/audit binding differs')
        _read(root,DATA_MANIFEST,DATA_SHA);audit=_read(root,AUDIT,AUDIT_SHA)
        if audit.get('status')!='PASS_CPU_ONLY_PENDING_REVIEW':raise ValueError('Full-data audit does not pass')
        for name,digest in SOURCE.items():_bytes(root,STAGE+'/'+name,digest=digest)
        request=_read(root,run+'/request.json',d['request_sha256']);config=_read(root,run+'/run_config.json',d['run_config_sha256'])
        if set(request)!={'plan_path','plan_sha256','run_dir','owner','parent_pid','original_origin_monotonic','original_deadline_monotonic'}:raise ValueError('Actual request schema differs')
        absolute=request['run_dir'];suffix='/'+run
        if not isinstance(absolute,str)or not absolute.startswith('/')or not absolute.endswith(suffix)or '..'in PurePosixPath(absolute).parts:raise ValueError('Actual run path differs')
        absolute_plan=absolute[:-len(suffix)]+'/'+PLAN
        if request['plan_path']!=absolute_plan or request['plan_sha256']!=PLAN_SHA or request['owner']!=run.rsplit('fit_',1)[1]or request['parent_pid']!=d['supervisor']['pid']or not _positive(request['original_origin_monotonic'])or request['original_deadline_monotonic']!=request['original_origin_monotonic']+14400:raise ValueError('Actual request owner/clock/plan differs')
        if config.get('plan_sha256')!=PLAN_SHA or any(config.get(k)!=v for k,v in plan.items()):raise ValueError('Run config is not the exact plan')
        p=_read(root,run+'/progress.json')
        required={'phase','verified_updates','total_updates','rank','num_layers','learning_rate','updated_epoch','available_bytes','mlx_active_bytes','mlx_cache_bytes','mlx_peak_bytes'}
        if not required<=set(p)or set(p)-required-{'next_update','gradient_seconds','optimizer_step_observed','last_checkpoint'}:raise ValueError('Progress schema differs')
        verified=_count(p['verified_updates'])
        if type(p['total_updates'])is not int or p['total_updates']!=264 or type(p['rank'])is not int or p['rank']!=16 or type(p['num_layers'])is not int or p['num_layers']!=8 or p['learning_rate']!=1e-5:raise ValueError('Progress recipe differs')
        if any(type(p[k])is not int or p[k]<0 for k in('available_bytes','mlx_active_bytes','mlx_cache_bytes','mlx_peak_bytes')):raise ValueError('Invalid progress memory fields')
        phase=p['phase']
        if phase not in{'loading_model','before_gradient','gradient_evaluated','Adam_evaluated','update_verified','paused','DEV_evaluating','complete'}:raise ValueError('Unknown progress phase')
        if phase=='loading_model'and verified!=0 or phase in{'DEV_evaluating','complete'}and verified!=264 or phase=='update_verified'and verified==0:raise ValueError('Phase/counter mismatch')
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
                if count!=264 or terminal.get('completed_updates')!=264 or status!='complete':raise ValueError('Completed closeout lacks complete verified counters/cleanup')
                result=_read(root,run+'/result.json',saved['files']['result.json']['sha256'])
                validation=_read(root,run+'/DEV.json',saved['files']['DEV.json']['sha256'])
                if result.get('status')!='complete'or result.get('plan_sha256')!=PLAN_SHA or result.get('completed_updates')!=264 or result.get('base_files_verified_before_and_after')is not True or result.get('final_opened')is not False:raise ValueError('Result metadata does not establish completed training')
                if type(validation.get('rows'))is not int or validation['rows']!=36 or validation.get('no_updates')is not True or validation.get('final_opened')is not False or type(validation.get('pooled_target_NLL'))not in(int,float)or not math.isfinite(validation['pooled_target_NLL'])or validation['pooled_target_NLL']<0:raise ValueError('DEV completion is not established')
                dev=36;dev_loss=validation['pooled_target_NLL']
        live=_processes(d,absolute_plan)if check_processes and terminal is None else False
        if status=='saved_progress'and check_processes and not live:status='saved_owner_unconfirmed'
        age=max(0,now-p['updated_epoch'])
        last=journal[count-1]if count else None
        note='Saved worker counter capped by contiguous finite actual-Adam journal. Process liveness is separate from optimizer progress.'
        if len(journal)<verified:note+=' Journal trails the saved counter; only the smaller count is displayed.'
        return dict(schema='research4.continuity-transfer-dashboard-fit.v1',status='available',plan_sha256=PLAN_SHA,
                    run_root=run,descriptor_sha256=hashlib.sha256(raw_descriptor).hexdigest(),training_status=status,
                    phase=phase,optimizer_updates=count,worker_reported_updates=verified,journal_updates=len(journal),total_updates=264,
                    dev_completed=dev,dev_total=36,dev_pooled_target_NLL=dev_loss,
                    preparation_verified=True,live_owner_verified=live,liveness_checked=check_processes,
                    stale=terminal is None and age>STALE_SECONDS,age_seconds=age,recorded_utc=datetime.fromtimestamp(p['updated_epoch'],timezone.utc).isoformat(),
                    last_training_loss=last['loss']if last else None,
                    last_training_target_tokens_per_compute_second=last['training_target_tokens_per_compute_second']if last else None,
                    measurement_scope='Last verified training update only; not inference speed or cumulative throughput.',
                    recorded_mlx_peak_bytes=p['mlx_peak_bytes'],
                    recorded_psutil_available_bytes=p['available_bytes'],
                    checkpoints=_checkpoints(root,run,count),
                    native_reserve=_native_reserve(root,run,d),
                    completion_saved_metadata_verified=status=='complete',
                    final_saved_audit_verified=status=='complete' and _final_saved_audit(root),
                    training_eta_seconds=None,unique_train_episodes=21,unique_train_rows=44,epochs=6,supervised_target_presentations=5934,paired_dev_completed=None,paired_dev_total=12,paired_dev_routes=['BASE','FT'],note=note)
    except (OSError,ValueError,TypeError,KeyError,AttributeError,OverflowError):return unavailable
