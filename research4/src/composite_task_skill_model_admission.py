"""One-use full-rank task qualification admission; no provider submission API."""
from pathlib import Path
from .io_utils import read, sha
from .composite_submission_admission import reserve, reconcile, guard
from .composite_task_skill_training import validate_model_source, PLAN, DATA_CLOSEOUT_SHA
from .composite_kaggle_v2_profile import HUMAN_SHA
from .composite_task_skill_profile import SELECTION_SHA
from .composite_kaggle_v2_admission import candidate_rule

CANDIDATE = 'task_skill_gpu_qualification_v1'
ATTEMPTS = 'qualification_attempts'


def validate_preparation(path):
    path = Path(path)
    value = read(path)
    root = path.parent / 'bundle/source'
    stage, _, evidence = validate_model_source(root, value['source_commit'])
    if value.get('status') != 'prepared_task_qualification_not_submitted' or value.get('phase') != 'qualification':
        raise PermissionError('Exact fresh task qualification preparation required')
    for name, key in [('TASK_QUALIFICATION_PACKAGE.zip','bundle_sha256'),
                      ('bundle/TASK_TRAINING_PACKAGE_FILES.json','bundle_manifest_sha256'),
                      ('research4_task_skill_qualification.ipynb','notebook_file_sha256'),
                      ('bundle/source/inputs/' + PLAN,'plan_sha256'),
                      ('bundle/source/inputs/MANIFEST.json','input_manifest_sha256')]:
        if sha(path.parent / name) != value[key]:
            raise PermissionError('Frozen qualification package/code changed')
    if value['notebook'] != stage['notebook'] or value['training_states'] != len(evidence.rows()):
        raise PermissionError('Qualification identity or complete task dataset differs')
    return {'source_commit':value['source_commit'], 'notebook':stage['notebook'], 'mode':stage['version'],
            'phase':'qualification', 'package_sha256':value['bundle_sha256'],
            'authorization_sha256':value['plan_sha256'], 'source_data_archive_sha256':value['data_archive_sha256'],
            'kaggle_remainder_scope_sha256':HUMAN_SHA, 'scope_A_selection_sha256':SELECTION_SHA,
            'provider_scope':'human-selected-Kaggle-remainder-v1', 'fresh_stage':True,
            'scientific_GPU_count':2, 'optimizer_updates':1, 'full_fit_completed':False,
            'hard_timeout_seconds':stage['hard_timeout_seconds'], 'automatic_scientific_retries':0}


def reserve_qualification(campaign_root, preparation_file, live_snapshot, guard_token=None):
    root = Path(campaign_root)
    lock, token, _ = guard(root, guard_token)
    try:
        prior = read(root / 'SCIENTIFIC_EXECUTION_LEDGER.json')['candidates'].get('ordinary_task_skill_data_bootstrap_v2')
        if (not prior or prior.get('status') != 'complete_task_only_data_independently_audited_CPU_released'
                or prior.get('owned_compute_released') is not True
                or sha(prior['closeout_receipt']) != DATA_CLOSEOUT_SHA):
            raise PermissionError('Complete independently audited/released task-data parent required')
    except Exception:
        if lock.exists() and read(lock).get('token') == token:
            lock.unlink()
        raise
    return reserve(root, preparation_file, live_snapshot, CANDIDATE, ATTEMPTS,
                   'r4-task-qualification-', validate_preparation, candidate_rule, token)


def reconcile_qualification(campaign_root, identity, script_version_id=None, guard_token=None):
    return reconcile(campaign_root, identity, CANDIDATE, ATTEMPTS, script_version_id, guard_token)
