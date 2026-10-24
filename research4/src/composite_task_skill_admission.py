"""One-use admission for the exact task-only CPU stage; no submission API."""
from pathlib import Path
from .io_utils import read, sha
from .composite_task_skill_profile import validate_source, PROFILE, SELECTION_SHA
from .composite_kaggle_v2_profile import HUMAN_SHA
from .composite_submission_admission import reserve, reconcile
from .composite_kaggle_v2_admission import candidate_rule

CANDIDATE = 'ordinary_task_skill_data_v1'
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
    return reserve(campaign_root, preparation_file, live_snapshot, CANDIDATE, ATTEMPTS,
                   'r4-task-skill-data-', validate_preparation, candidate_rule, guard_token)


def reconcile_data(campaign_root, identity, script_version_id=None, guard_token=None):
    return reconcile(campaign_root, identity, CANDIDATE, ATTEMPTS, script_version_id, guard_token)
