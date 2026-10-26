"""One owned frozen actor-only native evaluation partition; no fitting/submission."""
import argparse,json,os,time,traceback
from pathlib import Path
from .io_utils import read,sha,digest
from .composite_bounded_behavior_evaluation_profile import MODE,stage_prefix
from .composite_bounded_behavior_evaluation_source import validate_source
from .composite_task_skill_model_worker import Services as TaskServices
from .composite_owned_artifacts import OwnedArtifacts
from .composite_bounded_behavior_evaluation_store import EvaluationStore

class Services(TaskServices):
    @staticmethod
    def authorize(root,commit):
        keys=['RESEARCH4_BOUNDED_FULL_FIT_ARCHIVE','RESEARCH4_BOUNDED_QUALIFICATION_ARCHIVE',
          'RESEARCH4_BOUNDED_DATA_ARCHIVE','RESEARCH4_BOUNDED_ALIAS_FIT_ARCHIVE',
          'RESEARCH4_BOUNDED_ALIAS_QUALIFICATION_ARCHIVE']
        from .composite_bounded_sampling_fit_profile import DIRECT_MODE
        direct=read(Path(root)/'inputs/BOUNDED_FIT_PARENT_EXPECTED.json').get('fit_mode')==DIRECT_MODE
        paths=[None if direct and key=='RESEARCH4_BOUNDED_QUALIFICATION_ARCHIVE' else Path(os.environ[key]).resolve() for key in keys]
        if any(p is not None and not p.is_relative_to(Path('/kaggle/input').resolve()) for p in paths):
            raise PermissionError('Every original fitted/data parent remains immutable Kaggle input')
        frozen=validate_source(root,commit,*paths);frozen.require_runtime()
        from .composite_frozen_probe import verified_model_context
        from .composite_bounded_behavior_evaluation_profile import STAGE
        context=verified_model_context(root,commit,frozen.model.cfg['qwen'],frozen.stage(),sha(Path(root)/'inputs'/STAGE))
        context.update(stage=frozen.stage(),full_config=frozen.model.cfg,evaluation_context=frozen)
        return context,[]
    @staticmethod
    def owner(context):
        stage=context['stage']
        return OwnedArtifacts(EvaluationStore(stage),stage_prefix(stage),stage['source_commit'],base='/kaggle/temp',
          maximum_bytes=stage['maximum_raw_evidence_bytes'],bounded_evaluation=True)
    @staticmethod
    def construct_actor(path,context,guard,forward):
        from .composite_actor import Actor
        from .composite_bounded_behavior_interface import PROFILE
        return Actor(path,context['full_config']['qwen'],guard,forward,native_interface_profile=PROFILE,
          bounded_evaluation=context['evaluation_context'])
    @staticmethod
    def fit(actor,rows,cfg,stage,output,guard,progress,persist):
        if rows or stage['optimizer_updates']!=0:raise PermissionError('Frozen actor evaluation receives no fitting rows')
        context=actor.bounded_evaluation_context
        from .composite_v2_model_audit import capture_live_backbone
        from .composite_v2_raw_exports import runtime_evidence
        from .composite_bounded_behavior_evaluation_inference import verify_checkpoint
        from .composite_bounded_behavior_evaluation_collection import collect_pair
        from .composite_task_memory_profile import allocator_evidence
        import torch
        def integrity(when):
            actor.verify_frozen();persist('BACKBONE_LIVE_'+when+'.json',capture_live_backbone(actor))
            persist('CHECKPOINT_LIVE_'+when+'.json',verify_checkpoint(actor,context))
            persist('ALLOCATOR_'+when+'_COLLECTION.json',allocator_evidence(torch.cuda))
            if when=='BEFORE':persist('RUNTIME_RAW.json',runtime_evidence(actor))
        persist('ADAPTER_CAPACITY_RUNTIME.json',actor.capacity_proof)
        persist('POSTLOAD_RUNTIME.json',actor.bounded_evaluation_postload_proof)
        persist('CHECKPOINT_RESTORE.json',actor.bounded_checkpoint_proof)
        integrity('BEFORE');results=[]
        try:
            for case in stage['partition']['case_ids']:
                guard();value=collect_pair(actor,context,persist,case)
                persist('pairs/'+case+'.json',value);results.append(value)
                progress('bounded_actor_pair_complete',case_id=case,complete_pairs=len(results))
            return {'completed_trajectories':len(results)*2,'optimizer_updates':0,'results':results,
              'teacher_forcing':False,'teacher_compiler_called':False,'decoder_repair':False,
              'scientific_validation_admitted':False,'heldout_content_read':False}
        finally:integrity('AFTER')

def run(root,commit,_services=None):
    services=Services() if _services is None else _services
    started = services.clock()
    context, rows = services.authorize(root, commit)
    stage = context['stage']
    if stage.get('version')!=MODE or stage.get('endpoint') not in ['behavior','competence']:
        raise PermissionError('The separate V4 worker admits only its frozen actor partitions')
    owner = services.owner(context)
    counts = {'attempted': 0, 'completed': 0}
    actor = None
    receipt = {'mode': stage['version'], 'source_commit': commit, 'phase': stage['phase'], 'status': 'running',
               'training_objective':None,
               'evaluation_only':True,
               'model_instances': None, 'optimizer_updates': None, 'full_fit_completed': False,
               'qualification_admitted': False, 'raw_evidence_audit_required': True,
               'heldout_content_read': False, 'automatic_scientific_retries': 0,
               'public_fixture_test_mode': _services is not None}

    def guard():
        if services.clock() - started >= stage['worker_seconds'] - stage['export_reserve_seconds']:
            raise TimeoutError('Protected task-model export deadline')
        services.resources()

    def forward(event):
        guard()
        if event == 'attempt':
            if counts['attempted'] >= stage['maximum_backbone_forwards']:
                raise TimeoutError('Registered forward ceiling')
            counts['attempted'] += 1
        elif event == 'complete':
            if counts['completed'] >= counts['attempted']:
                raise ValueError('Forward completion without attempt')
            counts['completed'] += 1
            if counts['completed'] % 10 == 0:
                owner.json('forward_progress/' + str(counts['completed']).zfill(6) + '.json',
                           {'counts_lower_bound': dict(counts), 'exact_at_exit': False})
        else:
            raise ValueError('Unregistered forward event')

    def persist(name, value):
        return owner.json(name, value, terminal=name == 'TRAINING_TERMINAL_JOURNAL.json',
                          require_durable=name != 'TRAINING_TERMINAL_JOURNAL.json')

    try:
        with services.alarm(stage['worker_seconds'] - stage['export_reserve_seconds'] - (services.clock() - started)):
            owner.json('PRECALL_FREEZE.json', {'stage': stage, 'source_commit': commit,
                'data_archive_sha256': context['stage']['source_data_archive_sha256'],
                'input_manifest_sha256': sha(Path(root) / 'inputs/MANIFEST.json') if _services is None else None,
                'raw_training_rows_canonical_sha256':None,
                'source_release': read(Path(root) / 'RELEASE.json') if _services is None else None})
            owner.json('HARDWARE_PREFLIGHT.json', services.runtime(context))
            guard()
            actor = services.load(context, owner, guard, forward)
            receipt['model_instances'] = services.instance_count(actor) if hasattr(services,'instance_count') else 1
            def progress(kind,**value):
                print(json.dumps({'kind':kind,'source_commit':commit,'phase':stage['phase'],
                  'counts_exact_at_exit':False,**value}),flush=True)
            result = services.fit(actor, rows, context['full_config'], stage, owner.raw, guard,progress,persist)
            owner.json('ACTOR_COLLECTION_RESULT.json', result)
            receipt.update(status='complete', optimizer_updates=result['optimizer_updates'],full_fit_completed=False)
    except Exception as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error))
        owner.json('FAILURE.json', {'traceback': traceback.format_exc()}, terminal=True, require_durable=False)
    finally:
        actor = None
        try:
            services.release()
            receipt.update(services.remove_owned_cache(owner))
        except Exception as error:
            receipt.update(status='failed', model_or_cache_release_error=str(error))
        reserve = owner.scratch / 'EXPORT_RESERVE.bin'
        if reserve.exists():
            reserve.unlink()
        receipt.update(forward_counts=dict(counts), seconds=services.clock() - started, worker_exiting=True,
                       counts_exact_at_controlled_exit=True, provider_compute_release_verified=False)
        journal = owner.raw / 'TRAINING_TERMINAL_JOURNAL.json'
        if journal.exists():
            value = read(journal)
            receipt.update(last_training_journal=value,
                           optimizer_steps_completed_lower_bound=value.get('optimizer_steps_completed'),
                           optimizer_mutation_outcome_exact=value.get('optimizer_steps_completed') == value.get('optimizer_steps_attempted'))
        owner.json('RUN_RECEIPT.json', receipt, terminal=True, require_durable=False)
    exported = owner.finalize()
    if not exported['durable_complete']:
        return {**receipt, 'status': 'export_unconfirmed', 'export': exported, 'owned_scratch_removed': False}
    cleanup = owner.cleanup(True)
    return {**receipt, 'export': exported, 'cleanup': cleanup,
            'qualification_admitted': False, 'provider_compute_release_verified': False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--commit',required=True)
    result=run(**vars(parser.parse_args()));print(json.dumps(result))
    if result['status']!='complete':raise SystemExit(1)
