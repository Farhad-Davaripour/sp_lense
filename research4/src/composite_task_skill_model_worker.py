"""One owned full-rank task-training qualification; never submits/retries jobs."""
import argparse
import base64
import gc
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from .io_utils import read, sha, digest
from .composite_task_skill_training import authorized_load_plan, run_fit, MODE
from .composite_owned_artifacts import OwnedArtifacts
from .composite_task_skill_store import StageStore


class Services:
    clock = staticmethod(time.monotonic)
    fit = staticmethod(run_fit)

    @staticmethod
    def authorize(root, commit):
        cfg = read(Path(root) / 'config/composite_controller_rank32_v2.json')
        context = authorized_load_plan(root, commit, cfg['qwen'])
        return context, context['task_data_evidence'].rows()

    @staticmethod
    def owner(context):
        stage = context['stage']
        store = StageStore(stage)
        inference=stage['phase']=='task_development'
        from .composite_task_skill_inference import stage_prefix
        prefix = stage_prefix(stage) if inference else 'task_skill_training_v1/' + stage['source_commit'] + '/'+stage['phase']+'_01'
        return OwnedArtifacts(store, prefix, stage['source_commit'], base='/kaggle/temp',
                              maximum_bytes=stage['maximum_raw_evidence_bytes'], task_skill=not inference,task_evaluation=inference)

    @staticmethod
    def runtime(context):
        import psutil
        from .composite_frozen_probe import verify_runtime
        if psutil.virtual_memory().available < 12 * 1024**3:
            raise MemoryError('Full raw optimizer/checkpoint export requires its host-memory reserve')
        return verify_runtime(context)

    @staticmethod
    def resources():
        import torch
        from .composite_task_skill_training import live_resources
        return live_resources(torch.cuda)

    @staticmethod
    def alarm(seconds):
        from .composite_frozen_worker import protected_alarm
        return protected_alarm(seconds)

    @classmethod
    def load(cls,context, owner, guard, forward):
        from .composite_frozen_worker import transport_environment
        from .composite_store import download_model
        from .composite_frozen_probe import verify_metadata_bytes
        from .composite_v2_raw_exports import disk_admission
        stage = context['stage']
        cache = owner.scratch / 'cache'
        cache.mkdir()
        (cache / 'scratch').mkdir()
        os.environ.update(transport_environment(cache))
        pin = context['model_pin']
        owner.json('DISK_ADMISSION.json', disk_admission(cache, owner.raw, pin, stage, expected_profile=stage['raw_evidence_profile']))
        if not hasattr(os, 'posix_fallocate'):
            raise RuntimeError('Actual Linux physical export reservation required')
        with (owner.scratch / 'EXPORT_RESERVE.bin').open('xb') as stream:
            os.posix_fallocate(stream.fileno(), 0, stage['physical_export_reserve_bytes'])
        small, proof = download_model(pin, cache, metadata_only=True)
        encoded = {item['name']: base64.b64encode((small / item['name']).read_bytes()).decode()
                   for item in pin['files'] if not item['name'].endswith('.safetensors')}
        verify_metadata_bytes(pin, proof, encoded)
        owner.json('MODEL_METADATA_BYTES.json', encoded)
        owner.json('METADATA_FILE_VERIFICATION.json', proof)
        guard()
        path, proof = download_model(pin, cache)
        owner.json('MODEL_FILE_VERIFICATION.json', proof)
        actor = cls.construct_actor(path,context,guard,forward)
        if stage['phase']!='task_development':actor.v2_training_plan = stage
        loading={'loading': actor.frozen_loading_proof,
            'kernel': actor.kernel_proof, 'layout': actor.frozen_layout_proof,
            'capacity_qualified_by_loading': False}
        if stage['phase']=='task_development':loading['adapter_capacity']=actor.capacity_proof
        owner.json('MODEL_LOAD_ADMISSION.json',loading)
        return actor

    @staticmethod
    def construct_actor(path,context,guard,forward):
        from .composite_actor import Actor
        from .composite_generalized_interface import PROFILE
        return Actor(path,context['full_config']['qwen'],guard,forward,native_interface_profile=PROFILE,
                     task_training={'root':context['source_root'],'source_commit':context['source_commit']})

    @staticmethod
    def release():
        gc.collect()
        import torch
        for index in [0, 1]:
            if torch.cuda.is_available() and index < torch.cuda.device_count():
                torch.cuda.synchronize(index)
                with torch.cuda.device(index):
                    torch.cuda.empty_cache()

    @staticmethod
    def remove_owned_cache(owner):
        # Cache contains only reproducible downloaded weights/software scratch,
        # never unique raw evidence. Scientific model references are gone first.
        cache = (owner.scratch / 'cache').resolve()
        if cache.exists():
            if cache != owner.scratch.resolve() / 'cache' or cache.parent != owner.scratch.resolve() or read(owner.scratch / 'OWNER.json') != owner.owner:
                raise PermissionError('Cache cleanup ownership changed')
            shutil.rmtree(cache)
        return {'owned_download_cache_removed': not cache.exists(), 'unique_raw_evidence_removed': False}


def run(root, commit, _services=None,*,task_evaluation=False):
    if type(task_evaluation) is not bool:raise PermissionError('Explicit registered task-worker kind required')
    if task_evaluation:
        from .composite_task_skill_evaluation_worker import EvaluationServices
        services=EvaluationServices() if _services is None else _services
    else:services = Services() if _services is None else _services
    started = services.clock()
    context, rows = services.authorize(root, commit)
    stage = context['stage']
    if (stage['phase']=='task_development')!=task_evaluation:raise PermissionError('Worker kind and source-bound stage differ')
    owner = services.owner(context)
    counts = {'attempted': 0, 'completed': 0}
    actor = None
    receipt = {'mode': stage['version'], 'source_commit': commit, 'phase': stage['phase'], 'status': 'running',
               'training_objective':None if task_evaluation else 'ordinary_authorized_task_skills_only',
               'evaluation_only':task_evaluation,
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
                'raw_training_rows_canonical_sha256':None if task_evaluation else digest(rows),
                'source_release': read(Path(root) / 'RELEASE.json') if _services is None else None})
            owner.json('HARDWARE_PREFLIGHT.json', services.runtime(context))
            guard()
            actor = services.load(context, owner, guard, forward)
            receipt['model_instances'] = 1
            def progress(kind,**value):
                print(json.dumps({'kind':kind,'source_commit':commit,'phase':stage['phase'],
                  'counts_exact_at_exit':False,**value}),flush=True)
            result = services.fit(actor, rows, context['full_config'], stage, owner.raw, guard,progress,persist)
            owner.json('TASK_COLLECTION_RESULT.json' if task_evaluation else 'TRAINING_RESULT.json', result)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--commit', required=True)
    result = run(**vars(parser.parse_args()))
    print(json.dumps(result))
    if result['status'] != 'complete':
        raise SystemExit(1)
