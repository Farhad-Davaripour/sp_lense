"""One frozen paired task case using the common owned model-worker lifecycle."""
import argparse
import json
import os
from pathlib import Path
from .io_utils import read,sha
from .composite_task_skill_model_worker import Services,run


class EvaluationServices(Services):
    @staticmethod
    def authorize(root,commit):
        from .composite_task_skill_inference import validate_source,authorized_load_plan
        from .composite_task_skill_fit_registry import registered_fit
        inputs=Path(root)/'inputs';expected=read(inputs/'FIT_PARENT_EXPECTED.json')
        registration=registered_fit(expected['source_commit'],expected['run_id'])
        fit=Path(os.environ['RESEARCH4_TASK_FULL_FIT_ARCHIVE']).resolve()
        qualification=Path(os.environ['RESEARCH4_STATEFUL_QUALIFICATION_ARCHIVE']).resolve()
        for path,source,phase in [(fit,registration['source_commit'],'full_fit'),
                                  (qualification,'10eae62507d38f1507fb7443cf579ca5b6acc29f','qualification')]:
            suffix='research4_campaign/runs/'+source+'/task_skill_training_v1/'+phase+'_01/evidence.zip'
            if not path.is_relative_to(Path('/kaggle/input').resolve()) or not path.as_posix().endswith(suffix):
                raise PermissionError('Attach each exact immutable original full-fit/qualification output once')
        evaluation=validate_source(root,commit,fit,qualification)
        cfg=read(Path(root)/'config/composite_controller_rank32_v2.json')
        context=authorized_load_plan(evaluation,cfg['qwen'])
        context.update(stage=evaluation.stage(),full_config=cfg,task_evaluation=evaluation)
        return context,[]

    @staticmethod
    def construct_actor(path,context,guard,forward):
        from .composite_actor import Actor
        from .composite_generalized_interface import PROFILE
        return Actor(path,context['full_config']['qwen'],guard,forward,native_interface_profile=PROFILE,
                     task_evaluation=context['task_evaluation'])

    @staticmethod
    def fit(actor,rows,cfg,stage,output,guard,progress,persist):
        if rows or stage['optimizer_updates']!=0:raise PermissionError('Frozen collection must receive no fitting rows')
        from .composite_v2_model_audit import capture_live_backbone
        from .composite_task_skill_collection import collect_pair
        from .composite_task_skill_inference import verify_frozen_checkpoint
        from .composite_v2_raw_exports import runtime_evidence
        from .composite_task_memory_profile import allocator_evidence
        import torch
        persist('BACKBONE_LIVE_BEFORE.json',capture_live_backbone(actor))
        persist('CHECKPOINT_LIVE_BEFORE.json',verify_frozen_checkpoint(actor,actor.task_evaluation_context))
        persist('RUNTIME_RAW.json',runtime_evidence(actor))
        persist('ALLOCATOR_BEFORE_COLLECTION.json',allocator_evidence(torch.cuda))
        persist('MODEL_LOADING_INFO.json',actor.frozen_loading_proof)
        persist('CHECKPOINT_RESTORE.json',actor.task_checkpoint_proof)
        persist('POSTLOAD_RUNTIME.json',actor.task_evaluation_postload_proof)
        try:
            result=collect_pair(actor,actor.task_evaluation_context,persist)
            progress('task_paired_case_complete',case_id=stage['partition']['case_ids'][0],completed_trajectories=2)
            return result
        finally:
            actor.verify_frozen()
            persist('BACKBONE_LIVE_AFTER.json',capture_live_backbone(actor))
            persist('CHECKPOINT_LIVE_AFTER.json',verify_frozen_checkpoint(actor,actor.task_evaluation_context))
            persist('ALLOCATOR_AFTER_COLLECTION.json',allocator_evidence(torch.cuda))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--commit',required=True)
    result=run(**vars(parser.parse_args()),task_evaluation=True)
    print(json.dumps(result))
    if result['status']!='complete':raise SystemExit(1)
