"""Dedicated bounded frozen qualification child; no compute submission API."""
import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import signal
import re
import sys
import time
import traceback
import uuid

from .io_utils import read, sha, digest
from .composite_exports import exclusive_output, make_evidence_archive
from .composite_probe_ownership import verify_owned_cache
from .composite_frozen_probe import (authorized_load_plan, verify_runtime, load_probe_actor,
    collect_frozen_probes, load_training_rows, select_training_probes, BOUNDS, MODE)

OUTPUT_BASE = '/kaggle/working/research4_campaign/runs'
DOWNLOAD_BYTES = 55_575_985_859
GiB = 1024**3
DISK_POLICY = {'download_file_bytes': DOWNLOAD_BYTES, 'transport_overhead_bytes': 2*GiB,
               'export_reserve_bytes': 5*GiB, 'maximum_raw_evidence_bytes': GiB,
               'terminal_metadata_headroom_bytes':4*1024**2,
               'transport': 'hub_http_no_xet', 'persistent_weight_export': False}


def atomic_json(path, value):
    path = Path(path); temporary = path.with_name(path.name + '.'+uuid.uuid4().hex+'.pending')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def disk_admission(cache, output, model_pin, usage=None):
    import shutil
    usage = shutil.disk_usage if usage is None else usage
    total = sum(value['size'] for value in model_pin['files'])
    if total != DOWNLOAD_BYTES:
        raise ValueError('The original complete checkpoint download budget changed')
    cache_free = usage(cache).free; output_free = usage(output).free
    required = (total + DISK_POLICY['transport_overhead_bytes'] + DISK_POLICY['export_reserve_bytes']
                + DISK_POLICY['maximum_raw_evidence_bytes'])
    if cache_free < required or output_free < DISK_POLICY['export_reserve_bytes'] + DISK_POLICY['maximum_raw_evidence_bytes']:
        raise OSError('Insufficient disk for the complete checkpoint, transport overhead and protected exports')
    return {'download_file_bytes':total,'required_cache_free_bytes':required,
            'observed_cache_free_bytes':cache_free,'observed_output_free_bytes':output_free,
            'disk_policy':DISK_POLICY,'runtime_model_fit_admitted':False}


def transport_environment(cache):
    cache = Path(cache)
    return {'HF_HOME':str(cache/'hf_home'), 'HF_XET_CACHE':str(cache/'xet'),
            'HF_HUB_DISABLE_XET':'1', 'HF_HUB_DISABLE_SHARED_BLOBS':'1',
            'HF_HUB_DISABLE_SYMLINKS':'0', 'HF_HUB_DISABLE_IMPLICIT_TOKEN':'1',
            'HF_HUB_DISABLE_TELEMETRY':'1', 'TMPDIR':str(cache/'scratch')}


def verify_transport(cache):
    expected = transport_environment(cache)
    if any(os.environ.get(key) != value for key,value in expected.items()):
        raise ValueError('Owned HTTP cache environment differs; set it before importing Hub libraries')
    return expected


def output_target(destination, commit, token, base=OUTPUT_BASE):
    if not re.fullmatch('[0-9a-f]{40}',commit) or not re.fullmatch('[0-9a-f]{32}',token):
        raise ValueError('Invalid source/output ownership identity')
    target = Path(destination).resolve(); root = Path(base).resolve()
    if target == root or not target.is_relative_to(root) or commit not in target.parts:
        raise ValueError('Frozen output escaped its source-bound Research4 directory')
    return target


def reserve_export_space(target, commit, token, size=5*GiB, allocator=None):
    target = Path(target); marker = read(target/'OUTPUT_OWNER.json')
    if marker != {'source_commit':commit,'token':token,'path':str(target.resolve())}:
        raise PermissionError('Export reservation belongs to another output owner')
    reserve = target/'EXPORT_RESERVE.bin'
    if allocator is None:
        if not hasattr(os,'posix_fallocate'):
            raise RuntimeError('The qualified Linux filesystem must support physical export reservation')
        allocator = os.posix_fallocate
    with reserve.open('xb') as stream:
        allocator(stream.fileno(),0,size)
        stream.flush(); os.fsync(stream.fileno())
    if reserve.stat().st_size != size:
        raise OSError('Physical export reservation did not reach its declared size')
    return {'path':str(reserve),'reserved_bytes':size}


def release_export_space(target, commit, token):
    target = Path(target); marker = read(target/'OUTPUT_OWNER.json')
    if marker != {'source_commit':commit,'token':token,'path':str(target.resolve())}:
        raise PermissionError('Cannot release another output owner\'s reserve')
    reserve = target/'EXPORT_RESERVE.bin'
    if reserve.is_symlink(): raise PermissionError('Export reserve became a symlink')
    if reserve.exists(): reserve.unlink()
    return {'export_reserve_released':not reserve.exists()}


@contextmanager
def protected_alarm(remaining):
    if remaining<=0: raise TimeoutError('Frozen protected export deadline')
    if not hasattr(signal,'SIGALRM'):
        raise RuntimeError('The bounded child requires the qualified Linux timer')
    def expired(*args): raise TimeoutError('Frozen protected export deadline')
    previous = signal.signal(signal.SIGALRM,expired)
    old_timer = signal.setitimer(signal.ITIMER_REAL,max(0.001,remaining))
    try: yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous)
        if old_timer[0] > 0: signal.setitimer(signal.ITIMER_REAL,*old_timer)


class ForwardBudget:
    """Exact controlled-exit hook counts; durable lower bounds on abrupt kill."""
    def __init__(self, start, clock, journal, resource_check=lambda:None, bounds=None):
        self.start=start; self.clock=clock; self.journal=journal; self.resource_check=resource_check
        self.bounds=BOUNDS if bounds is None else bounds
        self.attempted=0; self.completed=0

    def snapshot(self):
        return {'attempted':self.attempted,'completed':self.completed,
                'source':'observed_root_forward_hooks','at_monotonic':self.clock()}

    def check(self):
        if self.clock()-self.start >= self.bounds['worker_seconds']-self.bounds['export_reserve_seconds']:
            raise TimeoutError('Frozen protected export deadline')
        self.resource_check()

    def event(self,event):
        if event=='attempt':
            self.check()
            if self.attempted >= self.bounds['maximum_backbone_forwards']:
                raise TimeoutError('Frozen forward ceiling')
            self.attempted+=1
        elif event=='complete':
            if self.completed >= self.attempted: raise ValueError('Completion without an attempted root forward')
            self.completed+=1
        else: raise ValueError('Unknown forward event')
        self.journal(self.snapshot())


class Services:
    clock=staticmethod(time.monotonic)
    authorize=staticmethod(authorized_load_plan)
    runtime=staticmethod(verify_runtime)
    ownership=staticmethod(verify_owned_cache)
    disk=staticmethod(disk_admission)
    reserve=staticmethod(reserve_export_space)
    release_reserve=staticmethod(release_export_space)
    alarm=staticmethod(protected_alarm)
    load=staticmethod(load_probe_actor)
    collect=staticmethod(collect_frozen_probes)
    verify_transport=staticmethod(verify_transport)
    output_base=OUTPUT_BASE

    @staticmethod
    def preflight_data(context):
        rows,provenance=load_training_rows(context); selected=select_training_probes(rows)
        return {'provenance':provenance,'conditions':[row['condition'] for row in selected],
                'state_ids':[row['state_id'] for row in selected],'heldout_content_read':False}

    @staticmethod
    def resources():
        import torch
        from .composite_frozen_probe import guard_frozen_devices
        guard_frozen_devices(torch.cuda)

    @staticmethod
    def release_model():
        gc.collect()
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def run(root, commit, destination, cache, owner_token, _services=None):
    services=Services() if _services is None else _services
    started=services.clock()  # Includes authorization, disk checks and every download.
    root=Path(root); actor=None
    config=read(root/'config/composite_controller_rank32_v2.json')['qwen']
    context=services.authorize(root,commit,config)  # Proposals stop before outputs/cache/model work.
    pin=context['pin']
    if pin['destination'] != destination or pin['cache_root'] != cache or pin['disk_policy'] != DISK_POLICY:
        raise ValueError('Frozen worker paths or disk policy differ from its declared freeze')
    services.ownership(cache,commit,owner_token)
    services.verify_transport(cache)
    target=output_target(destination,commit,owner_token,services.output_base)
    upload=exclusive_output(target,services.output_base)
    upload('OUTPUT_OWNER.json',json.dumps({'source_commit':commit,'token':owner_token,'path':str(target)}).encode())
    raw=target/'raw'; raw.mkdir()
    budget=ForwardBudget(started,services.clock,lambda value:atomic_json(raw/'FORWARD_COUNTS.json',value),services.resources)
    atomic_json(raw/'FORWARD_COUNTS.json',budget.snapshot())
    receipt={'status':'running','stage':MODE,'source_commit':commit,'heldout_content_read':False,
             'optimizer_updates':0,'behavior_adapter_fitting':False,'scientific_retries':0,
             'authorization_sha256':context['authorization_sha256'],'model_load_started':False,
             'model_instance_count':0,'inference_started':False}
    def persist(name,value):
        from .composite_runtime import contained
        path=contained(raw,name)
        encoded=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
        payload_limit=DISK_POLICY['maximum_raw_evidence_bytes']-DISK_POLICY['terminal_metadata_headroom_bytes']
        if len(encoded)+sum(file.stat().st_size for file in raw.rglob('*') if file.is_file()) > payload_limit:
            raise ValueError('Raw evidence export bound exceeded; no truncation')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream: stream.write(encoded)
        if path.read_bytes()!=encoded: raise ValueError('Frozen partial artifact readback differs')
        if name=='FROZEN_LOADED_LAYOUT.json': receipt['model_instance_count']=1
    try:
        persist('PRECALL_FREEZE.json',{'source_commit':commit,'mode':MODE,'pin':pin,
                'input_manifest_sha256':sha(root/'inputs/MANIFEST.json'),'bounds':BOUNDS,'disk_policy':DISK_POLICY})
        remaining=BOUNDS['worker_seconds']-BOUNDS['export_reserve_seconds']-(services.clock()-started)
        with services.alarm(remaining):
            persist('HARDWARE_PREFLIGHT.json',services.runtime(context))
            persist('DATA_PREFLIGHT.json',services.preflight_data(context))
            persist('DISK_ADMISSION.json',services.disk(cache,target,context['model_pin']))
            services.reserve(target,commit,owner_token)
            budget.check()
            receipt['model_load_started']=True; receipt['model_instance_count']=None
            actor, loaded_context, proof=services.load(root,commit,config,Path(cache)/'models',budget.check,budget.event,persist)
            receipt['model_instance_count']=1
            if loaded_context['source_commit']!=commit or loaded_context['authorization_sha256']!=context['authorization_sha256']:
                raise ValueError('Loader returned another source or authority freeze')
            budget.check(); receipt['inference_started']=True
            result=services.collect(actor,context,persist)
            if len(result.get('probes',[]))!=2 or result.get('forward_counts')!=actor.frozen_probe_counts:
                raise ValueError('A complete collection must retain both probes and its exact hook counts')
            if result.get('optimizer_updates')!=0 or result.get('behavior_adapter_fitting') is not False or result.get('heldout_content_read') is not False:
                raise ValueError('Frozen collector scope differs')
            persist('FROZEN_QUALIFICATION_RECORD.json',result)
            receipt.update(status='complete',probe_count=2,full_validation_admitted=False,
                           runtime_memory_admitted=False,cross_device_numerical_parity='not_established')
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error))
        try: persist('FAILURE.json',{'traceback':traceback.format_exc()})
        except Exception as export_error: receipt['failure_detail_export_error']=str(export_error)
    finally:
        # No further model call occurs after these final hook counts are captured.
        actor_counts=dict(actor.frozen_probe_counts) if actor is not None else {'attempted':0,'completed':0}
        owner_counts={'attempted':budget.attempted,'completed':budget.completed}
        if actor is not None and (any(actor_counts[key] < owner_counts[key] for key in owner_counts) or (receipt['status']=='complete' and actor_counts!=owner_counts)):
            receipt.update(status='failed',error_type='CounterMismatch',error='Actor/owner forward accounting differs')
        final_counts=actor_counts if actor is not None else owner_counts
        receipt.update(forward_counts=final_counts,owner_forward_counts=owner_counts,
                       counter_sources_consistent=actor_counts==owner_counts if actor is not None else True,
                       counts_exact_at_controlled_exit=True,worker_exiting=True,
                       seconds=services.clock()-started,full_validation_admitted=False,
                       platform_compute_release='requires_live_control_plane_verification')
        actor=None
        try: services.release_model(); receipt['owned_model_references_released']=True
        except Exception as error: receipt['model_release_error']=str(error)
        try: receipt.update(services.release_reserve(target,commit,owner_token))
        except Exception as error: receipt['export_reserve_release_error']=str(error)
        terminal_size=len((json.dumps(receipt,indent=2)+'\n').encode())+len(json.dumps(final_counts).encode())
        if terminal_size>DISK_POLICY['terminal_metadata_headroom_bytes']//2:
            raise ValueError('Terminal metadata exceeded its reserved share; raw failed evidence remains intact')
        atomic_json(raw/'FORWARD_COUNTS.json',{**final_counts,'sealed_for_exit':True,'source':'observed_root_forward_hooks'})
        atomic_json(raw/'RUN_RECEIPT.json',receipt)
    archive=target/'evidence.zip'
    proof=make_evidence_archive(raw,archive,DISK_POLICY['maximum_raw_evidence_bytes'])
    # Archive already exists locally; use distinct final names through the exclusive writer.
    upload('FROZEN_EVIDENCE.zip',archive.read_bytes())
    upload('RUN_RECEIPT.json',(raw/'RUN_RECEIPT.json').read_bytes())
    export={**receipt,**proof,'destination':destination,'cache_cleanup':'parent_only_after_child_exit'}
    upload('EXPORT_RECEIPT.json',json.dumps(export,indent=2).encode())
    print(json.dumps(export),flush=True)
    return export


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ['root','commit','destination','cache','owner-token']: parser.add_argument('--'+key,required=True)
    args=parser.parse_args(); value=run(args.root,args.commit,args.destination,args.cache,args.owner_token)
    if value['status']!='complete': raise SystemExit(1)
