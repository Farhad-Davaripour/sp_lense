"""CPU-only complete option-A data export, raw replay and token admission."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import traceback
from .io_utils import read, sha, digest
from .composite_task_skill_profile import authorize, MODE, BOUNDS
from .composite_task_skill_data import scenes, audit, native_budget, leakage
from .composite_task_skill_world import reference_episode
from .composite_v2_data_worker import Writer, metadata_tokenizers
from .composite_exports import make_evidence_archive


def run(root, commit):
    profile, cfg = authorize(root, commit)
    output = Path(profile['destination'])
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    def guard():
        if time.monotonic() - start >= BOUNDS['worker_seconds'] - BOUNDS['export_reserve_seconds']:
            raise TimeoutError('Task-only protected export boundary')
    writer = Writer(output, guard)
    receipt = {'stage': MODE, 'source_commit': commit, 'status': 'running',
               'model_forwards': 0, 'optimizer_updates': 0, 'GPU_required': False,
               'model_weights_downloaded': False, 'heldout_content_read': False,
               'continuity_targets_generated': False, 'task_dataset_admitted': False,
               'completed_roles': [], 'automatic_scientific_retries': 0}
    try:
        writer.json('PRECALL_FREEZE.json', {'source_commit': commit,
            'input_manifest_sha256': sha(Path(root) / 'inputs/MANIFEST.json'),
            'study_configuration': cfg, 'profile': profile,
            'source_release_sha256': sha(Path(root) / 'RELEASE.json')})
        from .composite_frozen_worker import protected_alarm
        with protected_alarm(BOUNDS['worker_seconds'] - BOUNDS['export_reserve_seconds']):
            with tempfile.TemporaryDirectory(prefix='research4-task-tokenizers-') as cache:
                qt, _ = metadata_tokenizers(root, cache, writer)
                audits, budgets, datasets = {}, {}, {}
                for role in ['training', 'development']:
                    generated = []
                    expected = scenes(cfg, role, commit)
                    writer.json(role + '/PRECALL_ROSTER.json', {'cases': [vars(s) for s in expected],
                                                             'source_commit': commit})
                    for scene in expected:
                        guard()
                        episode = reference_episode(scene)
                        writer.json(role + '/cases/' + scene.case_id + '.json', episode)
                        generated.append(episode)
                    if [e['scene'] for e in generated] != [vars(s) for s in expected]:
                        raise ValueError('Actual task roster differs from its source-bound roster')
                    audits[role] = audit(generated, cfg, role)
                    budgets[role] = native_budget(generated, qt)
                    datasets[role] = generated
                    writer.json(role + '/REFERENCE_AUDIT.json', audits[role])
                    writer.json(role + '/NATIVE_BUDGET.json', budgets[role])
                    # Keep the evaluation references separate from fitting rows.
                    if role == 'training':
                        rows = [{**row, 'case_id': e['scene']['case_id'], 'group_id': e['scene']['group_id'],
                                 'mechanism': e['scene']['mechanism'], 'split': role}
                                for e in generated for row in e['states']]
                        if not all(row['training_eligible'] and row['task_only'] for row in rows):
                            raise ValueError('Non-task or evaluation state reached the fitting export')
                        writer.json('training/FITTING_ROWS.json', rows)
                    else:
                        writer.json('development/EVALUATION_ROSTER.json', {
                            'cases': [e['scene'] for e in generated], 'reference_actions_in_roster': False,
                            'arms': cfg['task_development']['arms'], 'paired_trajectories': 72})
                    receipt['completed_roles'].append(role)
                leak = leakage(audits['training'], audits['development'], datasets['training'], datasets['development'])
                writer.json('LEAKAGE_AUDIT.json', leak)
                if any(budgets[role]['status'] != 'pass' or budgets[role]['states'] != audits[role]['complete_states']
                       for role in ['training', 'development']):
                    raise ValueError('Full native context/target audit failed; no truncation')
                release = {'version': MODE, 'source_commit': commit, 'input_manifest_sha256': sha(Path(root) / 'inputs/MANIFEST.json'),
                           'study_config_sha256': sha(Path(root) / 'config/composite_task_skill_v1.json'),
                           'audits': audits, 'native_budgets': budgets, 'leakage': leak,
                           'cohort_canonical_sha256': {role: digest(data) for role, data in datasets.items()},
                           'task_dataset_admitted': True, 'model_training_admitted_by_this_receipt': False,
                           'frozen_risk_dataset_generated': False, 'heldout_content_read': False,
                           'model_forwards': 0, 'optimizer_updates': 0}
                writer.json('TASK_DATA_RELEASE.json', release)
                receipt.update(status='complete', task_dataset_admitted=True)
    except Exception as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error))
        writer.json('FAILURE.json', {'traceback': traceback.format_exc()}, terminal=True)
    finally:
        receipt.update(seconds=time.monotonic() - start, worker_exiting=True,
                       provider_release='must_reconcile_actual_notebook_exit')
        writer.json('RUN_RECEIPT.json', receipt, terminal=True)
    archive = output.with_name(output.name + '_EVIDENCE.zip')
    proof = make_evidence_archive(output, archive, BOUNDS['maximum_raw_export_bytes'])
    return {**receipt, **proof, 'archive': str(archive)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--commit', required=True)
    value = run(**vars(parser.parse_args()))
    print(json.dumps(value))
    if value['status'] != 'complete':
        raise SystemExit(1)
