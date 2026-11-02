"""One-use admission for the exact task-only CPU stage; no submission API."""
from pathlib import Path
from .io_utils import read, sha
from .composite_task_skill_profile import validate_source, PROFILE, SELECTION_SHA
from .composite_kaggle_v2_profile import HUMAN_SHA
from .composite_submission_admission import reserve, reconcile
from .composite_kaggle_v2_admission import candidate_rule
from .composite_submission_admission import guard

CANDIDATE = 'ordinary_task_skill_data_bootstrap_v2'
ATTEMPTS = 'data_attempts'


def validate_preparation(path):
    path = Path(path)
    value = read(path)
    root = path.parent / 'bundle/source'
    profile, _ = validate_source(root, value['source_commit'])
    if value.get('status') != 'prepared_task_only_CPU_data_not_submitted':
        raise PermissionError('Fresh task-only preparation required')
    for name, key in [('TASK_SKILL_DATA_PACKAGE.zip', 'bundle_sha256'),
                      ('bundle/TASK_SKILL_PACKAGE_FILES.json', 'bundle_manifest_sha256'),
                      ('research4_task_skill_data.ipynb', 'notebook_file_sha256'),
                      ('bundle/source/inputs/' + PROFILE, 'profile_sha256'),
                      ('bundle/source/inputs/MANIFEST.json', 'input_manifest_sha256')]:
        if sha(path.parent / name) != value[key]:
            raise PermissionError('Task-only package/code bytes changed')
    if value['notebook'] != profile['notebook'] or value['human_selection_sha256'] != SELECTION_SHA:
        raise PermissionError('Notebook or option-A selection changed')
    return {'source_commit': value['source_commit'], 'notebook': profile['notebook'],
            'mode': profile['version'], 'package_sha256': value['bundle_sha256'],
            'authorization_sha256': sha(root / 'inputs' / PROFILE),
            'kaggle_remainder_scope_sha256': HUMAN_SHA,
            'scope_A_selection_sha256': SELECTION_SHA,
            'provider_scope': 'human-selected-Kaggle-remainder-v1', 'fresh_stage': True,
            'scientific_GPU_count': 0, 'hard_timeout_seconds': 900, 'automatic_scientific_retries': 0}


def reserve_data(campaign_root, preparation_file, live_snapshot, guard_token=None):
    campaign_root = Path(campaign_root)
    lock, token, _ = guard(campaign_root, guard_token)
    try:
        parent = read(campaign_root / 'SCIENTIFIC_EXECUTION_LEDGER.json')['candidates'].get('ordinary_task_skill_data_v1')
        if (not parent or parent.get('status') != 'failed_pre_data_bootstrap_audited_released'
                or parent.get('scientific_execution_started') is not False
                or parent.get('owned_compute_released') is not True
                or len(parent.get('data_attempts', [])) != 1
                or parent['data_attempts'][0].get('scriptVersionId') != 361861821):
            raise PermissionError('Scientific execution/unknown parents never qualify for bootstrap repair')
    except Exception:
        if lock.exists() and read(lock).get('token') == token:
            lock.unlink()
        raise
    return reserve(campaign_root, preparation_file, live_snapshot, CANDIDATE, ATTEMPTS,
                   'r4-task-skill-data-v2-', validate_preparation, candidate_rule, token)


def reconcile_data(campaign_root, identity, script_version_id=None, guard_token=None):
    return reconcile(campaign_root, identity, CANDIDATE, ATTEMPTS, script_version_id, guard_token)
