"""One owned V4 stateful qualifier/fresh fit. Never submits or retries jobs."""
import argparse,json,time,traceback
from pathlib import Path
from .io_utils import read,sha,digest
from .composite_bounded_sampling_fit_profile import MODE,DIRECT_MODE,OBJECTIVE
from .composite_bounded_sampling_training import authorized_load_plan,run_fit
from .composite_task_skill_model_worker import Services as TaskServices
from .composite_owned_artifacts import OwnedArtifacts
from .composite_bounded_sampling_fit_store import BoundedStore,stage_prefix

class Services(TaskServices):
    fit=staticmethod(run_fit)
    @staticmethod
    def authorize(root,commit):
        cfg=read(Path(root)/'config/composite_controller_rank32_v2.json')
        context=authorized_load_plan(root,commit,cfg['qwen'])
        return context,context['bounded_data_evidence'].fitting_rows()
    @staticmethod
    def owner(context):
        stage=context['stage']
        return OwnedArtifacts(BoundedStore(stage),stage_prefix(stage),stage['source_commit'],
          base='/kaggle/temp',maximum_bytes=stage['maximum_raw_evidence_bytes'],sampling_fit=True)
    @staticmethod
    def construct_actor(path,context,guard,forward):
        from .composite_actor import Actor
        from .composite_bounded_behavior_interface import PROFILE
        return Actor(path,context['full_config']['qwen'],guard,forward,native_interface_profile=PROFILE,
          bounded_training={'root':context['source_root'],'source_commit':context['source_commit'],'mode':context['stage']['version']})

def run(root,commit,_services=None):
    services=Services() if _services is None else _services
    started = services.clock()
    context, rows = services.authorize(root, commit)
    stage = context['stage']
    if (stage.get('version') not in [MODE,DIRECT_MODE] or stage.get('phase') not in ['qualification','full_fit']
            or stage.get('version')==DIRECT_MODE and stage.get('phase')!='full_fit'):
        raise PermissionError('The separate V4 worker admits only its exact fitting stages')
    owner = services.owner(context)
    counts = {'attempted': 0, 'completed': 0}
    actor = None
    receipt = {'mode': stage['version'], 'source_commit': commit, 'phase': stage['phase'], 'status': 'running',
               'training_objective':OBJECTIVE,
               'evaluation_only':False,
               'model_instances': None, 'optimizer_updates': None, 'full_fit_completed': False,
               'qualification_admitted': False, 'raw_evidence_audit_required': True,
               'heldout_content_read': False, 'automatic_scientific_retries': 0,
               'public_fixture_test_mode': _services is not None}
    if stage['version']==DIRECT_MODE:
        receipt.update(separate_qualification_waived=True,qualification_performed=False,
                       direct_fit_waiver_sha256=stage['direct_fit_waiver_sha256'])

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
                'raw_training_rows_canonical_sha256':digest(rows),
                'source_release': read(Path(root) / 'RELEASE.json') if _services is None else None})
            owner.json('HARDWARE_PREFLIGHT.json', services.runtime(context))
            guard()
            actor = services.load(context, owner, guard, forward)
            receipt['model_instances'] = services.instance_count(actor) if hasattr(services,'instance_count') else 1
            def progress(kind,**value):
                print(json.dumps({'kind':kind,'source_commit':commit,'phase':stage['phase'],
                  'counts_exact_at_exit':False,**value}),flush=True)
            result = services.fit(actor, rows, context['full_config'], stage, owner.raw, guard,progress,persist)
            owner.json('TRAINING_RESULT.json', result)
            receipt.update(status='complete', optimizer_updates=result['optimizer_updates'],full_fit_completed=stage['phase']=='full_fit')
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
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True);parser.add_argument('--commit',required=True)
    result=run(**vars(parser.parse_args()));print(json.dumps(result))
    if result['status']!='complete':raise SystemExit(1)
