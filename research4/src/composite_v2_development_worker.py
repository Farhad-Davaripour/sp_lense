"""Owned frozen detector/native development entrypoint; no submit or fit API."""
import argparse,base64,gc,hashlib,io,json,os,time,traceback
from dataclasses import asdict
from pathlib import Path
from .io_utils import read,digest
from .composite_owned_artifacts import OwnedArtifacts
from .composite_v2_development_contract import validate_stage,MODE
from .composite_v2_development_data import load_development_data
from .composite_v2_raw_exports import provider_binding,detector_evidence,runtime_evidence
from .composite_v2_model_audit import capture_live_backbone
from .composite_v2_evaluation import collect_partition
from .composite_v2_detector_transfer import collect_transfer


class Services:
    clock=staticmethod(time.monotonic)
    @staticmethod
    def metadata(root,commit):
        from .composite_generalized_data import require_runtime
        from .composite_store import Store
        from .workspace_accounts import verify_runtime_workspace
        require_runtime();context=validate_stage(root,commit);store=Store()
        actual=verify_runtime_workspace(Path(root),store.client.config.host)
        if actual['host']!=context['workspace']['host']:raise PermissionError('Development workspace differs')
        return context,store
    @staticmethod
    def runtime():
        from .composite_v2_model_worker import Services as ModelServices
        import torch
        torch.backends.cuda.matmul.allow_tf32=False
        from importlib.metadata import version
        return {**ModelServices.runtime(),'torch':str(torch.__version__),'packages':{name:version(name) for name in
            ['transformers','bitsandbytes','accelerate','safetensors','numpy','psutil']},'BF16_supported':torch.cuda.is_bf16_supported(),
            'TF32_enabled':bool(torch.backends.cuda.matmul.allow_tf32)}
    @staticmethod
    def resources():
        from .composite_v2_model_worker import Services as ModelServices
        ModelServices.resources()
    @staticmethod
    def alarm(seconds):
        from .composite_frozen_worker import protected_alarm
        return protected_alarm(seconds)
    @staticmethod
    def inputs(context,store,owner,guard):
        import psutil
        if psutil.virtual_memory().available<context['bounds']['minimum_host_available_bytes']:
            raise MemoryError('Complete development evidence host-memory reserve unavailable')
        stage=context['stage'];context['provider_run_id']=provider_binding(stage,context['workspace'],store.json(stage['provider_binding_path']))
        raw=store.get(stage['data_export']['path']);transfer=stage['phase']=='detector_transfer'
        data=load_development_data(raw,context,reference_rows=transfer)
        if not transfer:
            from .composite_v2_full_parent import full_fit_evidence
            from .composite_v2_admission import validate_package
            model_proof=full_fit_evidence(stage,context['workspace'],store,owner.scratch,raw)
            package=store.json(stage['package']['path'])
            if digest(package)!=stage['package']['sha256']:raise PermissionError('Model package changed')
            validate_package(package,context['configuration']);report=model_proof.receipt()
            if (package['checkpoint_sha256']!=report['checkpoint_sha256'] or package['training_raw_audit_sha256']!=digest(report)
                or package['data_release_sha256']!=stage['data_release_sha256']
                or digest(package['development_roster'])!=digest(stage['development_roster'])):
                raise PermissionError('Exact audited model/package/roster binding required')
            context.update(package=package,model_stage_evidence=model_proof)
        return data
    @staticmethod
    def load(context,store,owner,guard,forward):
        import torch,shutil
        from .composite_store import download_model
        from .composite_v2_raw_exports import disk_admission
        stage=context['stage'];cfg=context['configuration'];transfer=stage['phase']=='detector_transfer'
        cache=owner.scratch/'cache';cache.mkdir(exist_ok=True)
        pin=read(Path(context['root'])/'inputs/MODERNBERT_PIN.json')
        total=sum(v['size'] for v in pin['files'])
        needed=total+context['bounds']['physical_export_reserve_bytes']+context['bounds']['maximum_raw_evidence_bytes']
        if not transfer:
            qpin=read(Path(context['root'])/'inputs/QWEN_PIN.json')
            from .composite_v2_raw_exports import PROFILE as RAW_PROFILE
            model_stage={**stage,'raw_evidence_profile':RAW_PROFILE,
                'maximum_raw_evidence_bytes':context['bounds']['maximum_raw_evidence_bytes'],
                'physical_export_reserve_bytes':context['bounds']['physical_export_reserve_bytes']}
            proof=disk_admission(cache,owner.raw,qpin,model_stage)
            needed=proof['required_cache_free_bytes']+total
        if shutil.disk_usage(cache).free<needed:raise OSError('Complete joint model and evidence disk capacity unavailable')
        owner.json('DISK_ADMISSION.json',{'required_free_bytes':needed,'source_commit':context['source_commit'],
                                         'actual_runtime_capacity_admitted':False})
        reserve=owner.scratch/'EXPORT_RESERVE.bin'
        if not hasattr(os,'posix_fallocate'):raise RuntimeError('Physical Linux export reservation required')
        with reserve.open('xb') as f:os.posix_fallocate(f.fileno(),0,context['bounds']['physical_export_reserve_bytes'])
        path,proof=download_model(pin,cache/'bert')
        encoded={v['name']:base64.b64encode((path/v['name']).read_bytes()).decode() for v in pin['files'] if not v['name'].endswith('.safetensors')}
        owner.json('DETECTOR_FILE_VERIFICATION.json',proof);owner.json('DETECTOR_METADATA_BYTES.json',encoded)
        raw=store.get(stage['detector_checkpoint']['path'])
        if hashlib.sha256(raw).hexdigest()!=stage['detector_checkpoint_sha256']:raise PermissionError('Selected fixed detector bytes changed')
        saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        from .composite_detector import Detector
        detector=Detector(path,cfg['modernbert'],head_mode='role_pool');detector.restore(saved);detector.eval().requires_grad_(False)
        detector.on_forward=forward
        if transfer:return detector
        # Reuse the exact reviewed Actor loader, without reserving/downloading
        # a second model or invoking any optimizer. Its reserve is already held.
        qpath,qproof=download_model(qpin,cache/'qwen');owner.json('MODEL_FILE_VERIFICATION.json',qproof)
        owner.json('MODEL_METADATA_BYTES.json',{v['name']:base64.b64encode((qpath/v['name']).read_bytes()).decode() for v in qpin['files'] if not v['name'].endswith('.safetensors')})
        from .composite_actor import Actor
        from .composite_generalized_interface import PROFILE
        actor=Actor(qpath,cfg['qwen'],guard,forward,native_interface_profile=PROFILE)
        checkpoint=store.get(stage['model_checkpoint']['path'])
        if hashlib.sha256(checkpoint).hexdigest()!=context['package']['checkpoint_sha256']:raise PermissionError('Fitted actor checkpoint changed')
        actor.restore(torch.load(io.BytesIO(checkpoint),map_location='cpu',weights_only=True));actor.eval().requires_grad_(False)
        from .composite_v2_controller import V2Composite
        return V2Composite(detector,actor)
    @staticmethod
    def integrity(model,owner,when,transfer):
        detector=model if transfer else model.detector
        records,runtime=detector_evidence(detector)
        owner.json('DETECTOR_PARAMETERS_'+when+'.json',records)
        if when=='BEFORE':owner.json('DETECTOR_RUNTIME.json',runtime)
        else:
            if runtime!=read(owner.raw/'DETECTOR_RUNTIME.json'):raise ValueError('Frozen detector runtime changed')
        if not transfer:
            import torch
            model.actor.verify_frozen();owner.json('BACKBONE_LIVE_'+when+'.json',capture_live_backbone(model.actor))
            with (owner.raw/('EVAL_ADAPTER_'+when+'.pt')).open('xb') as f:torch.save(model.actor.adapter_state()['weights'],f)
            if when=='BEFORE':owner.json('RUNTIME_RAW.json',runtime_evidence(model.actor))
    @staticmethod
    def collect(context,model,data,guard,persist,verify):
        stage=context['stage']
        if stage['phase']=='detector_transfer':
            try:return collect_transfer(model,data['rows'],guard,persist)
            finally:verify()
        from .composite_generalized_world import GeneralizedScene
        ids=set(stage['partition']['case_ids']);scenes=[GeneralizedScene(**s) for s in data['scenes'] if s['case_id'] in ids]
        if len(scenes)!=len(ids):raise PermissionError('Exact whole-case partition is missing')
        return collect_partition(model,scenes,stage['partition']['controls'],guard,persist,verify,context['source_commit'],persist_prefixes=True)
    @staticmethod
    def release():
        import torch
        gc.collect();torch.cuda.synchronize(0);torch.cuda.empty_cache()


def run(root,commit,_services=None,scratch_base=None):
    services=Services() if _services is None else _services;started=services.clock()
    context,store=services.metadata(root,commit);stage=context['stage'];bounds=context['bounds']
    transfer=stage['phase']=='detector_transfer'
    prefix=stage['destination'][len(context['workspace']['artifact_volume']+'/composite_controller_v1/'):]
    owner=OwnedArtifacts(store,prefix,commit,base=scratch_base,maximum_bytes=bounds['maximum_raw_evidence_bytes'],development=True)
    if _services is None:
        from .composite_frozen_worker import transport_environment
        cache=owner.scratch/'cache';cache.mkdir();(cache/'scratch').mkdir()
        os.environ.update(transport_environment(cache))
    model=None;counts={'attempted':0,'completed':0};records=[];integrity_started=False
    receipt={'mode':MODE,'source_commit':commit,'phase':stage['phase'],'status':'running','optimizer_updates':0,
        'public_fixture_test_mode':_services is not None,'heldout_content_read':False,'scientific_validation_admitted':False,
        'provider_compute_release_verified':False,'runtime_capacity_qualified':False,'observed_trajectories':None}
    def guard():
        if services.clock()-started>=bounds['worker_seconds']-bounds['export_reserve_seconds']:raise TimeoutError('Protected development export boundary')
        services.resources()
    def forward(event):
        guard()
        if event=='attempt':
            if counts['attempted']>=bounds['maximum_backbone_forwards']:raise TimeoutError('Frozen development forward ceiling')
            counts['attempted']+=1
        elif event=='complete':
            if counts['completed']>=counts['attempted']:raise ValueError('Unmatched forward completion')
            counts['completed']+=1
            if counts['completed']%bounds['forward_progress_interval']==0:
                owner.json('forward_progress/'+str(counts['completed']).zfill(6)+'.json',
                           {'counts_lower_bound':dict(counts),'counts_exact_at_exit':False})
        else:raise ValueError('Unregistered model event')
    def persist(name,value):return owner.json(name,value,terminal=name.endswith('RECEIPT.json'),require_durable=not name.endswith('RECEIPT.json'))
    def verify():services.integrity(model,owner,'AFTER',transfer)
    try:
        with services.alarm(bounds['worker_seconds']-bounds['export_reserve_seconds']-(services.clock()-started)):
            owner.json('HARDWARE_PREFLIGHT.json',services.runtime());guard()
            data=services.inputs(context,store,owner,guard)
            receipt['provider_run_id']=context.get('provider_run_id')
            owner.json('DEVELOPMENT_PRECALL_FREEZE.json',{'source_commit':commit,'stage':stage,'workspace':context['workspace'],
                'data_release_sha256':digest(data['release'])})
            model=services.load(context,store,owner,guard,forward)
            services.integrity(model,owner,'BEFORE',transfer);integrity_started=True
            records=services.collect(context,model,data,guard,persist,verify)
            if not transfer:
                manifest={r['scene']['case_id']+'/'+r['condition']:digest(r) for r in records}
                owner.json('EVALUATION_MODEL_BINDING.json',{'source_commit':commit,'package_sha256':digest(context['package']),
                    'data_release_sha256':stage['data_release_sha256'],'trace_manifest':manifest,'optimizer_updates':0,
                    'teacher_forcing':False,'decoder_repair':False,'policy_logit_mask':False,'public_fixture_test_mode':_services is not None})
            receipt.update(status='complete',observed_trajectories=None if transfer else len(records),observed_detector_states=len(records) if transfer else None)
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error))
        owner.json('FAILURE.json',{'traceback':traceback.format_exc()},terminal=True,require_durable=False)
    finally:
        model=None
        try:services.release()
        except Exception as error:receipt.update(status='failed',model_release_error=str(error))
        reserve=owner.scratch/'EXPORT_RESERVE.bin'
        if reserve.exists():reserve.unlink()
        receipt.update(seconds=services.clock()-started,forward_counts=dict(counts),counts_exact_at_controlled_exit=True,
                       worker_exiting=True,integrity_snapshots_started=integrity_started)
        collection=owner.raw/('TRANSFER_COLLECTION_RECEIPT.json' if transfer else 'COLLECTION_RECEIPT.json')
        if collection.exists():
            summary=read(collection)
            receipt['observed_detector_states' if transfer else 'observed_trajectories']=summary['observed_states' if transfer else 'observed_trajectories']
        owner.json('RUN_RECEIPT.json',receipt,terminal=True,require_durable=False)
    export=owner.finalize()
    if not export['durable_complete']:return {**receipt,'status':'export_unconfirmed','export':export,'owned_scratch_removed':False}
    cleanup=owner.cleanup(True)
    return {**receipt,'export':export,'cleanup':cleanup,'independent_raw_audit_required':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--commit',required=True)
    args=parser.parse_args();value=run(args.root,args.commit);print(json.dumps(value))
    if value['status']!='complete':raise SystemExit(1)
