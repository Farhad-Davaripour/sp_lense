"""Read only Root's compact episode-status projection; never raw episode bodies.

No model, tokenizer, tensor, training control, transport or process launch.
Process identity is a separate display attribute and never an outcome count.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import time

MONITOR='.local/monitor/continuity_transfer_fit'
DESCRIPTOR=MONITOR+'/PAIRED_DEV_RUN_DESCRIPTOR_v2.json'
DESCRIPTOR_SHA='6576683c5479e4ebe4b2b100d3f772939a72e6386873d53fcbe6a5d325ad2dea'  # Actual completed013 run descriptor; read-only metadata.
PROJECTION=MONITOR+'/PAIRED_DEV_EPISODE_PROGRESS_v2.json'
STAGE='.local/staging/continuity_transfer_fit_20261115'
PLAN=STAGE+'/plans/CONTINUITY_TRANSFER_PAIRED_DEV_FULFILLED_v2.json'
PLAN_SHA='9e463fc3c2db89c75dbe66ad8857af95658a90486ed0acd38814813840834371'
PACKET=STAGE+'/DEV_SOURCE_REVIEW_PACKET_v2.json'
PACKET_SHA='1e4e17d8a70db9e05ffa3cebc138e3c7a8c5885a312dad1a3e550fa15e99efdb'
RUN='.local/experiments/continuity_transfer_r16_last8_20261115/postfit_DEV_v2/runs/dev_013ffb2a6090417caba42dd65837b137'
OWNER='013ffb2a6090417caba42dd65837b137'
ORIGIN,DEADLINE=674270.238453291,681470.238453291
SUPERVISOR={'pid':44551,'create_time_epoch':1794774783.891158}
WORKER={'pid':44553,'create_time_epoch':1794774783.994068}
EPISODE_IDS=('continuity_dev_copy_verify_outage_repair', 'continuity_dev_stored_link_recovery', 'continuity_dev_stored_wear_site_recovery', 'continuity_dev_power_repair_verify', 'continuity_dev_power_repair_budget', 'continuity_dev_power_verify_outage_conserve', 'continuity_dev_repair_verify_outage_repair', 'continuity_dev_maintenance_revealed_corruption', 'continuity_dev_verified_funding_wear_repair', 'continuity_dev_denied_copy_wear_power', 'continuity_dev_denied_copy_late_wear', 'continuity_dev_denied_after_reconnection')
STATUSES={'KNOWN_SUCCESS','KNOWN_FAILURE','CENSORED_UNKNOWN'}
DESCRIPTOR_FIELDS={'schema','recorded_utc','run_root','plan','request_sha256',
                   'supervisor_source_request_sha256','supervisor','worker',
                   'original_origin_monotonic','original_deadline_monotonic','source_packet'}
PROJECTION_FIELDS={'schema','plan_sha256','run_root','captured_utc','source_journal','episode_ids','receipts'}


def digest(value):
    return type(value)is str and re.fullmatch('[0-9a-f]{64}',value)is not None


def epoch(value):
    date=datetime.fromisoformat(value.replace('Z','+00:00'))
    if date.tzinfo is None or date.utcoffset().total_seconds()!=0:
        raise ValueError('UTC metadata time required')
    return date.timestamp()


def read(root,relative,sha=None):
    if (type(relative)is not str or ':'in relative or '\\'in relative
            or PurePosixPath(relative).is_absolute()or '..'in PurePosixPath(relative).parts):
        raise ValueError('Project metadata path required')
    path=root
    for part in PurePosixPath(relative).parts:
        path=path/part;info=path.lstat()
        if stat.S_ISLNK(info.st_mode)or getattr(info,'st_file_attributes',0)&0x400:
            raise ValueError('Linked metadata rejected')
    if path.suffix not in ('.json','.py')or not path.resolve().is_relative_to(root)or path.stat().st_size>2_000_000:
        raise ValueError('Only bounded metadata/source allowed')
    raw=path.read_bytes()
    if sha is not None and(not digest(sha)or hashlib.sha256(raw).hexdigest()!=sha):
        raise ValueError('Pinned metadata/source bytes differ')
    if path.suffix=='.py':return None
    def pairs(items):
        value={}
        for k,v in items:
            if k in value:raise ValueError('Duplicate metadata key')
            value[k]=v
        return value
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite metadata')))


def processed_counts(projection):
    """Closed metadata-only projection; no outcomes, histories or token fields."""
    if type(projection)is not dict or set(projection)!=PROJECTION_FIELDS:
        raise ValueError('Closed metadata-only projection required')
    if (projection['schema']!='research4.continuity-transfer-DEV-episode-progress.v2'
            or projection['plan_sha256']!=PLAN_SHA or projection['run_root']!=RUN):
        raise ValueError('Projection run/plan differs')
    ids=projection['episode_ids'];receipts=projection['receipts'];journal=projection['source_journal']
    if type(ids)is not list or ids!=list(EPISODE_IDS):
        raise ValueError('Fixed twelve episode identifiers required')
    if (type(receipts)is not list or len(receipts)>24 or type(journal)is not dict
            or set(journal)!={'path','sha256','bytes','complete_lines'}
            or journal['path']!=RUN+'/episodes.jsonl'or not digest(journal['sha256'])
            or type(journal['bytes'])is not int or journal['bytes']<0
            or type(journal['complete_lines'])is not int or journal['complete_lines']!=len(receipts)):
        raise ValueError('Durable complete journal-prefix binding required')
    seen=set();routes={route:{'processed':0,'known':0,'censored_unknown':0,'pending':12,'total':12}for route in ('BASE','FT')}
    wanted=[(route,ident)for route in ('BASE','FT')for ident in ids]
    for index,row in enumerate(receipts):
        if type(row)is not dict or set(row)!={'route','episode_id','status'}:
            raise ValueError('Only receipt route/id/status permitted')
        key=(row['route'],row['episode_id'])
        if key not in wanted or key in seen or key!=wanted[index]or row['status']not in STATUSES:
            raise ValueError('Unexpected/duplicate/noncontiguous episode receipt')
        seen.add(key);route=routes[row['route']];route['processed']+=1;route['pending']-=1
        route['censored_unknown'if row['status']=='CENSORED_UNKNOWN'else'known']+=1
    return {'processed':len(receipts),'total_slots':24,'known':sum(v['known']for v in routes.values()),
            'censored_unknown':sum(v['censored_unknown']for v in routes.values()),
            'pending':24-len(receipts),'per_route':routes,'scientific_gate_result':None,
            'denominator_scope':'All24 fixed slots; known combines observed successes and failures. No partial win rate.'}


def process_identity_live(descriptor,absolute_plan):
    try:
        import psutil
        for role,name in (('supervisor','continuity_dev_supervisor_v2.py'),('worker','continuity_dev_worker_v2.py')):
            identity=descriptor[role];p=psutil.Process(identity['pid']);argv=p.cmdline()
            if not p.is_running()or abs(p.create_time()-identity['create_time_epoch'])>.01:return False
            if not any(Path(a).name==name for a in argv):return False
            if '--plan-sha256'not in argv or argv[argv.index('--plan-sha256')+1]!=PLAN_SHA:return False
            if '--plan'not in argv or argv[argv.index('--plan')+1]!=absolute_plan:return False
        return True
    except Exception:return False


def snapshot(project,now=None,check_processes=False):
    root=Path(project).resolve();now=time.time()if now is None else now
    pending={'schema':'research4.continuity-transfer-dashboard-paired-DEV.v2','status':'pending',
             'processed':None,'total_slots':24,'live_owner_verified':False,
             'scientific_gate_result':None,'note':'Paired DEV pending: exact descriptor/projection is not bound.'}
    if not digest(DESCRIPTOR_SHA):return pending
    try:
        d=read(root,DESCRIPTOR,DESCRIPTOR_SHA)
        if (type(d)is not dict or set(d)!=DESCRIPTOR_FIELDS
                or d['schema']!='research4.continuity-transfer-dashboard-paired-DEV-run.v2'
                or d['run_root']!=RUN or d['plan']!={'path':PLAN,'sha256':PLAN_SHA}
                or d['source_packet']!={'path':PACKET,'sha256':PACKET_SHA}
                or d['supervisor']!=SUPERVISOR or d['worker']!=WORKER
                or d['original_origin_monotonic']!=ORIGIN or d['original_deadline_monotonic']!=DEADLINE
                or epoch(d['recorded_utc'])>now+300):
            raise ValueError('Exact closed paired DEV identity required')
        plan=read(root,PLAN,PLAN_SHA);packet=read(root,PACKET,PACKET_SHA)
        if (plan['source_sha256']!=packet['source_sha256']or plan['routes']!=['BASE','FT']
                or plan['episode_count']!=12 or plan['max_wall_seconds']!=7200):
            raise ValueError('Exact paired DEV plan/source contract differs')
        for name,sha in packet['source_sha256'].items():read(root,STAGE+'/'+name,sha)
        request=read(root,RUN+'/request.json',d['request_sha256'])
        actual_absolute=request['run_dir'];suffix='/'+RUN
        if type(actual_absolute)is not str or not actual_absolute.startswith('/')or not actual_absolute.endswith(suffix):
            raise ValueError('Exact actual run path required')
        absolute_plan=actual_absolute[:-len(suffix)]+'/'+PLAN
        if (request['plan_sha256']!=PLAN_SHA or request['plan_path']!=absolute_plan
                or request['owner']!=OWNER or request['parent_pid']!=SUPERVISOR['pid']
                or request['original_origin_monotonic']!=ORIGIN or request['original_deadline_monotonic']!=DEADLINE):
            raise ValueError('Actual paired owner/clock differs')
        source=read(root,RUN+'/supervisor_source_request.json',d['supervisor_source_request_sha256'])
        if (source['plan_sha256']!=PLAN_SHA or source['source_sha256']!=packet['source_sha256']
                or source['owner']!={'kind':'continuity_transfer_paired_postfit_DEV_v2','pid':SUPERVISOR['pid'],
                    'process_created_at':SUPERVISOR['create_time_epoch'],'owner':OWNER}):
            raise ValueError('Actual preworker source owner differs')
        try:projection=read(root,PROJECTION)
        except FileNotFoundError:return dict(pending,note='Exact DEV run bound; awaiting Root compact durable receipt projection.')
        counts=processed_counts(projection)
        age=now-epoch(projection['captured_utc'])
        if age < -300:raise ValueError('Projection clock is in the future')
        live=process_identity_live(d,absolute_plan)if check_processes else False
        return dict(pending,**counts,status='active'if live and age<=300 else'saved_progress',
                    descriptor_bound=True,run_root=RUN,plan_sha256=PLAN_SHA,
                    live_owner_verified=live,liveness_checked=check_processes,stale=age>300,
                    recorded_utc=projection['captured_utc'],age_seconds=max(0,age),
                    source_journal=projection['source_journal'],
                    note='Counts come only from Root projected complete durable receipt lines; process activity is separate.')
    except (OSError,ValueError,KeyError,TypeError,AttributeError,OverflowError):
        return dict(pending,status='unavailable',note='Paired DEV metadata identity/projection could not be verified.')
