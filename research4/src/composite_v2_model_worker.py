"""Owned V2 qualification/full-fit orchestration; no compute submission API."""
import argparse,base64,gc,json,os,time,traceback
from pathlib import Path
from .io_utils import read,digest
from .composite_v2_model_contract import validate_model_metadata,load_training_export,MODE
from .composite_owned_artifacts import OwnedArtifacts
from .composite_generalized_interface import PROFILE
from .composite_v2_training import run_fit
from .composite_v2_raw_exports import disk_admission,provider_binding


class Services:
    clock=staticmethod(time.monotonic)
    fit=staticmethod(run_fit)
    @staticmethod
    def metadata(root,commit):
        from .composite_generalized_data import require_runtime
        from .composite_store import Store
        from .workspace_accounts import verify_runtime_workspace
        require_runtime();context=validate_model_metadata(root,commit);store=Store()
        actual=verify_runtime_workspace(Path(root),store.client.config.host)
        if actual['host']!=context['workspace']['host']:raise PermissionError('Actual model workspace differs')
        return context,store
    @staticmethod
    def inputs(context,store,scratch):
        import psutil
        if psutil.virtual_memory().available<12*1024**3:raise MemoryError('Full raw-state audit/serialization host-memory reserve unavailable')
        stage=context['stage']
        context['provider_run_id']=provider_binding(stage,context['workspace'],store.json(stage['provider_binding_path']))
        archive=store.get(stage['data_export']['path']);qualification=None
        if stage['phase']=='full_fit':
            from .composite_v2_parent_audit import qualification_from_descriptor
            qualification=qualification_from_descriptor(stage,context['workspace'],store,scratch,archive)
        rows,data=load_training_export(archive,context,qualification_evidence=qualification)
        return rows,data
    @staticmethod
    def runtime():
        import torch,transformers,bitsandbytes
        from importlib.metadata import version
        from .composite_frozen_probe import verify_software
        verify_software(str(torch.__version__),{name:version(name) for name in
            ['transformers','bitsandbytes','accelerate','safetensors','numpy','psutil']})
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():raise PermissionError('One BF16-capable visible GPU required')
        return {'visible_GPUs':1,'name':torch.cuda.get_device_name(0),'total_bytes':torch.cuda.mem_get_info(0)[1]}
    @staticmethod
    def resources():
        import torch
        if torch.cuda.mem_get_info(0)[0]<512*1024**2:raise MemoryError('Live GPU export reserve')
    @staticmethod
    def alarm(seconds):
        from .composite_frozen_worker import protected_alarm
        return protected_alarm(seconds)
    @staticmethod
    def load(context,owner,guard,forward):
        from .composite_frozen_worker import transport_environment
        from .composite_frozen_probe import verify_metadata_bytes
        from .composite_store import download_model
        cache=owner.scratch/'cache';cache.mkdir();os.environ.update(transport_environment(cache))
        pin=read(Path(context['root'])/'inputs/QWEN_PIN.json')
        owner.json('DISK_ADMISSION.json',disk_admission(cache,owner.raw,pin,context['stage']))
        reserve=owner.scratch/'EXPORT_RESERVE.bin'
        if not hasattr(os,'posix_fallocate'):raise RuntimeError('Qualified Linux physical export reservation required')
        with reserve.open('xb') as stream:os.posix_fallocate(stream.fileno(),0,context['stage']['physical_export_reserve_bytes'])
        small,proof=download_model(pin,cache,metadata_only=True)
        encoded={item['name']:base64.b64encode((small/item['name']).read_bytes()).decode()
                 for item in pin['files'] if not item['name'].endswith('.safetensors')}
        verify_metadata_bytes(pin,proof,encoded)
        owner.json('MODEL_METADATA_BYTES.json',encoded);owner.json('METADATA_FILE_VERIFICATION.json',proof)
        guard();path,proof=download_model(pin,cache);owner.json('MODEL_FILE_VERIFICATION.json',proof)
        from .composite_actor import Actor
        actor=Actor(path,context['configuration']['qwen'],guard,forward,native_interface_profile=PROFILE)
        actor.v2_training_plan=context['stage']
        owner.json('MODEL_LOAD_ADMISSION.json',{'loading':actor.frozen_loading_proof,'kernel':actor.kernel_proof,
            'frozen_original_parameters':actor.frozen_proof(),'training_memory_fit':None})
        return actor
    @staticmethod
    def release(actor):
        # Caller drops its last scientific model reference before cache cleanup.
        gc.collect()
        import torch
        torch.cuda.synchronize(0)
        torch.cuda.empty_cache()


def run(root,commit,_services=None,scratch_base=None):
    services=Services() if _services is None else _services;started=services.clock()
    if _services is None:
        context,store=services.metadata(root,commit);rows=data=None
    else:context,store,rows,data=services.authorize(root,commit)  # Explicit public fake-service path only.
    stage=context['stage'];cfg=context['configuration'];bounds=cfg['bounds']
    prefix=stage['destination'][len(context['workspace']['artifact_volume']+'/composite_controller_v1/'):]
    owner=OwnedArtifacts(store,prefix,commit,base=scratch_base,maximum_bytes=stage.get('maximum_raw_evidence_bytes',1024**3))
    counts={'attempted':0,'completed':0};actor=None
    receipt={'mode':MODE,'source_commit':commit,'phase':stage['phase'],'status':'running','heldout_content_read':False,
        'model_instances':None,'optimizer_updates':None,'native_interface_profile':PROFILE,'scientific_retries':0,
        'public_fixture_test_mode':_services is not None,'full_fit_completed':False}
    receipt['provider_run_id']=context.get('provider_run_id')
    def guard():
        if services.clock()-started>=bounds['worker_seconds']-bounds['export_reserve_seconds']:raise TimeoutError('Protected model export deadline')
        services.resources()
    def forward(event):
        guard()
        if event=='attempt':
            if counts['attempted']>=bounds['maximum_backbone_forwards_per_run']:raise TimeoutError('Forward ceiling')
            counts['attempted']+=1
        elif event=='complete':
            if counts['completed']>=counts['attempted']:raise ValueError('Completion without attempted forward')
            counts['completed']+=1
            if counts['completed']%10==0:
                owner.json('forward_progress/'+str(counts['completed']).zfill(6)+'.json',
                    {'counts_lower_bound':dict(counts),'counts_exact_at_exit':False})
        else:raise ValueError('Unknown forward event')
    def persist(name,value):return owner.json(name,value,terminal=name=='TRAINING_TERMINAL_JOURNAL.json',require_durable=name!='TRAINING_TERMINAL_JOURNAL.json')
    try:
        with services.alarm(bounds['worker_seconds']-bounds['export_reserve_seconds']-(services.clock()-started)):
            owner.json('MODEL_METADATA_PREFLIGHT.json',{'source_commit':commit,'phase':stage['phase'],'model_work_started':False})
            owner.json('HARDWARE_PREFLIGHT.json',services.runtime());guard()
            if _services is None:
                rows,data=services.inputs(context,store,owner.scratch)
                receipt['provider_run_id']=context['provider_run_id']
            owner.json('PRECALL_FREEZE.json',{'stage':stage,'workspace':context['workspace'],'data_release_sha256':digest(data)})
            actor=services.load(context,owner,guard,forward);receipt['model_instances']=1
            result=services.fit(actor,rows,cfg,stage,owner.raw,guard,lambda *a,**k:None,persist)
            owner.json('TRAINING_RESULT.json',result)
            receipt.update(status='complete',optimizer_updates=result['optimizer_updates'],full_fit_completed=result['full_fit_completed'])
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error))
        owner.json('FAILURE.json',{'traceback':traceback.format_exc()},terminal=True,require_durable=False)
    finally:
        actor=None
        try:services.release(actor)
        except Exception as error:receipt.update(status='failed',model_release_error=str(error))
        reserve=owner.scratch/'EXPORT_RESERVE.bin'
        if reserve.exists():reserve.unlink()  # Exact owned file, after scientific work has exited.
        receipt.update(forward_counts=dict(counts),seconds=services.clock()-started,worker_exiting=True,
            counts_exact_at_controlled_exit=True,provider_compute_release_verified=False,full_scientific_validation_complete=False)
        journal=owner.raw/'TRAINING_TERMINAL_JOURNAL.json'
        if journal.exists():
            value=read(journal);receipt.update(last_training_journal=value,
                optimizer_updates_completed_lower_bound=value.get('optimizer_steps_completed'),
                optimizer_mutation_outcome_exact=value.get('optimizer_steps_completed')==value.get('optimizer_steps_attempted'))
        if _services is not None:receipt.update(full_fit_completed=False,scientific_qualification_admitted=False)
        owner.json('RUN_RECEIPT.json',receipt,terminal=True,require_durable=False)
    export=owner.finalize()
    if not export['durable_complete']:
        return {**receipt,'status':'export_unconfirmed','export':export,'owned_scratch_removed':False,
                'scientific_qualification_admitted':False,'model_result_preserved_durably':False}
    cleanup=owner.cleanup(True)
    return {**receipt,'export':export,'cleanup':cleanup,'scientific_qualification_admitted':False,
            'raw_evidence_audit_required':True,'provider_compute_release_verified':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    value=run(a.root,a.commit);print(json.dumps(value))
    if value['status']!='complete':raise SystemExit(1)
