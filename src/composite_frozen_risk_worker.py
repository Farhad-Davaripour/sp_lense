"""One useful frozen risk/recognition stage using the shared owned lifecycle."""
import argparse
import base64
import json
import os
from pathlib import Path
import shutil
from .io_utils import read,sha,digest
from .composite_task_skill_model_worker import Services,run


class RiskServices(Services):
    def authorize(self,root,commit):
        from .composite_frozen_risk_profile import validate_source
        archives=[Path(os.environ[name]).resolve() for name in [
          'RESEARCH4_TASK_FULL_FIT_ARCHIVE','RESEARCH4_STATEFUL_QUALIFICATION_ARCHIVE','RESEARCH4_RISK_DEVELOPMENT_ARCHIVE']]
        if any(not p.is_relative_to(Path('/kaggle/input').resolve()) for p in archives):
            raise PermissionError('All original risk/model parent archives must remain immutable Kaggle inputs')
        risk=validate_source(root,commit,*archives);risk.require_runtime()
        cfg=read(Path(root)/'config/composite_controller_rank32_v2.json')
        from .composite_frozen_probe import verified_model_context
        context=verified_model_context(root,commit,cfg['qwen'],risk.stage(),sha(Path(root)/'inputs'/risk.stage_filename))
        context.update(stage=risk.stage(),full_config=cfg,risk_context=risk)
        self.context=context;self.detector_counts={'attempted':0,'completed':0}
        return context,[]
    def resources(self):
        import torch
        if self.context['stage']['phase']=='risk_detector_transfer':
            if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Exact one-T4 transfer context required')
            free,_=torch.cuda.mem_get_info(0);idle=max(0,torch.cuda.memory_reserved(0)-torch.cuda.memory_allocated(0))
            if free+idle<512*1024**2:raise MemoryError('Protected transfer export headroom lost')
        else:super().resources()
    def runtime(self,context):
        import torch
        from importlib.metadata import version
        from .composite_frozen_probe import verify_software,verify_runtime
        torch.backends.cuda.matmul.allow_tf32=False
        if context['stage']['phase']!='risk_detector_transfer':return verify_runtime(context)
        software=verify_software(torch.__version__,{n:version(n) for n in ['transformers','bitsandbytes','accelerate','safetensors','numpy','psutil']})
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!='Tesla T4':
            raise PermissionError('One scientific Tesla T4 required for fixed detector transfer')
        free,total=torch.cuda.mem_get_info(0)
        if free<3*1024**3:raise MemoryError('Complete fixed detector plus runtime/export reserve unavailable')
        return {'software':software,'visible_GPUs':1,'devices':[{'index':0,'name':'Tesla T4','free_bytes':free,'total_bytes':total}],
                'TF32_enabled':False,'BF16_supported':torch.cuda.is_bf16_supported(),'Qwen_loaded':False}
    @staticmethod
    def construct_actor(path,context,guard,forward):
        from .composite_actor import Actor
        from .composite_generalized_interface import PROFILE
        return Actor(path,context['full_config']['qwen'],guard,forward,native_interface_profile=PROFILE,
                     task_evaluation=context['risk_context'])
    def load(self,context,owner,guard,forward):
        from .composite_store import download_model
        from .composite_frozen_worker import transport_environment
        from .composite_frozen_risk_detector import selected_detector
        from .composite_detector import Detector
        stage=context['stage'];transfer=stage['phase']=='risk_detector_transfer';cache=owner.scratch/'cache'
        if transfer:
            cache.mkdir();(cache/'scratch').mkdir();os.environ.update(transport_environment(cache))
            if shutil.disk_usage(cache).free<stage['physical_export_reserve_bytes']+stage['maximum_raw_evidence_bytes']+2*1024**3:
                raise OSError('Complete transfer model/raw/archive/reservation disk space required')
            with (owner.scratch/'EXPORT_RESERVE.bin').open('xb') as f:os.posix_fallocate(f.fileno(),0,stage['physical_export_reserve_bytes'])
            owner.json('DISK_ADMISSION.json',{'frozen_detector_only':True,'maximum_raw_evidence_bytes':stage['maximum_raw_evidence_bytes'],
              'physical_export_reserve_bytes':stage['physical_export_reserve_bytes'],'Qwen_downloaded':False})
            actor=None
        else:actor=super().load(context,owner,guard,forward)
        pin=read(Path(context['source_root'])/'inputs/MODERNBERT_PIN.json')
        if shutil.disk_usage(cache).free<sum(v['size'] for v in pin['files'])+1024**3:raise OSError('Full detector cache reserve unavailable')
        path,proof=download_model(pin,cache/'bert');owner.json('DETECTOR_FILE_VERIFICATION.json',proof)
        owner.json('DETECTOR_METADATA_BYTES.json',{v['name']:base64.b64encode((path/v['name']).read_bytes()).decode()
          for v in pin['files'] if not v['name'].endswith('.safetensors')})
        guard();detector=Detector(path,context['full_config']['modernbert'],head_mode='role_pool')
        detector.restore(selected_detector(Path(context['source_root'])/'inputs',context['full_config']['modernbert']))
        detector.eval().requires_grad_(False)
        def event(kind):
            forward(kind)
            self.detector_counts['attempted' if kind=='attempt' else 'completed']+=1
        detector.on_forward=event
        if transfer:return detector
        from .composite_frozen_risk_capture import RecordingRiskActor
        from .composite_runtime import contained
        def binary(name,raw):
            file=contained(owner.raw,name);file.parent.mkdir(parents=True,exist_ok=True)
            with file.open('xb') as f:f.write(raw)
            if not owner._publish_file('raw/'+name,file):raise OSError('Unique native activation transfer unconfirmed; preserve prefix')
        proxy=RecordingRiskActor(actor,binary,owner.json)
        from .composite_v2_controller import V2Composite
        return V2Composite(detector,proxy,0.9).eval().requires_grad_(False)
    def instance_count(self,model):return 1 if self.context['stage']['phase']=='risk_detector_transfer' else 2
    def fit(self,model,rows,cfg,stage,output,guard,progress,persist):
        if rows or stage['optimizer_updates']!=0:raise PermissionError('Frozen risk stages receive no fitting rows')
        from .composite_frozen_risk_runtime import frozen_scope
        from .composite_v2_raw_exports import detector_evidence,runtime_evidence
        from .composite_v2_model_audit import capture_live_backbone
        from .composite_task_skill_inference import verify_frozen_checkpoint
        transfer=stage['phase']=='risk_detector_transfer';detector=model if transfer else model.detector
        actor=None if transfer else model.actor.actor
        def integrity(when):
            params,runtime=detector_evidence(detector);persist('DETECTOR_PARAMETERS_'+when+'.json',params)
            if when=='BEFORE':persist('DETECTOR_RUNTIME.json',runtime)
            elif digest(runtime)!=digest(read(Path(output)/'DETECTOR_RUNTIME.json')):raise RuntimeError('Fixed detector runtime changed')
            if actor is not None:
                actor.verify_frozen();persist('BACKBONE_LIVE_'+when+'.json',capture_live_backbone(actor))
                persist('CHECKPOINT_LIVE_'+when+'.json',verify_frozen_checkpoint(actor,self.context['risk_context']))
                if when=='BEFORE':persist('RUNTIME_RAW.json',runtime_evidence(actor))
        integrity('BEFORE')
        try:
            with frozen_scope(self.context['risk_context']):
                if transfer:
                    from .composite_v2_detector_transfer import collect_transfer
                    records=collect_transfer(detector,self.context['risk_context'].data.reference_rows(),guard,persist)
                else:
                    from .composite_generalized_world import GeneralizedScene
                    from .composite_v2_controller import rollout
                    lookup={s['case_id']:s for s in self.context['risk_context'].data.scenes()};records=[]
                    for unit in stage['partition']['units']:
                        guard();case,arm=unit['case_id'],unit['arm'];model.actor.begin_trace(case,arm)
                        def prefix(world,trace,case=case,arm=arm):
                            from dataclasses import asdict
                            from .composite_v2_controller import ENVIRONMENT
                            persist('prefixes/'+case+'/'+arm+'/step_'+str(len(trace)).zfill(3)+'.json',
                              {'scene':asdict(world.scene),'condition':arm,'environment':ENVIRONMENT,
                               'initial_observation':model.observations[0],'trace':trace,'outcome':world.outcome(),
                               'terminal_memory':model.memory.snapshot(),'primitive_trace':world.engine.trace,
                               'helper_definitions':world.engine.definitions,'helper_calls':world.helper_calls,
                               'model_training_during_evaluation':False,'teacher_compiler_called_during_evaluation':False})
                        value=rollout(model,GeneralizedScene(**lookup[case]),arm,guard,prefix)
                        persist('traces/'+case+'/'+arm+'.json',value);records.append(value)
                        progress('frozen_risk_trace_complete',case_id=case,arm=arm,complete_trajectories=len(records))
            return {'records':records,'optimizer_updates':0,'reference_states':len(records) if transfer else None,
              'completed_trajectories':None if transfer else len(records),'detector_forward_counts':dict(self.detector_counts),
              'actor_forward_counts':None if transfer else dict(actor.frozen_probe_counts),
              'scientific_validation_admitted':False,'heldout_content_read':False}
        finally:integrity('AFTER')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--commit',required=True)
    result=run(**vars(parser.parse_args()),frozen_risk=True);print(json.dumps(result))
    if result['status']!='complete':raise SystemExit(1)
