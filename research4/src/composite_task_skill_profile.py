"""Exact source/input/runtime binding for option-A CPU data generation only."""
import os
import re
from pathlib import Path
from .io_utils import read, sha, verify_release
from .composite_kaggle_v2_profile import identity, HUMAN as KAGGLE_HUMAN, HUMAN_SHA
from .composite_v2_data_contract import ORIGINAL_MODEL_MANIFEST

MODE = 'ordinary-task-skill-data-v1'
PROFILE = 'TASK_SKILL_PROFILE.json'
HUMAN = 'HUMAN_SCOPE_A_SELECTION.json'
SELECTION_SHA = '53647510a4b6ccb1fdee5740a117458f94265d35c7541b2b159de3885f6815ba'
NOTEBOOK = 'farhad89/research4-task-skill-data-v1'
BOUNDS = {'hard_timeout_seconds': 900, 'worker_seconds': 600, 'export_reserve_seconds': 60,
          'maximum_raw_export_bytes': 256 * 1024**2, 'terminal_metadata_reserve_bytes': 1024**2,
          'maximum_concurrent_owned_workers': 1, 'automatic_scientific_retries': 0}


def expected_profile(commit):
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Exact source commit required')
    return {'version': MODE, 'source_commit': commit, 'platform': 'kaggle', 'account': 'farhad89',
            'notebook': NOTEBOOK, 'visible_devices': '', 'scientific_GPU_count': 0,
            'scope_A_selection_sha256': SELECTION_SHA, 'kaggle_scope_sha256': HUMAN_SHA,
            'destination': '/kaggle/working/research4_campaign/runs/' + commit + '/ordinary_task_skill_v1/data_01',
            'roles': ['training', 'development'], 'heldout_content_read': False,
            'continuity_targets_generated': False, **BOUNDS}


def validate_source(root, commit):
    root = Path(root)
    verify_release(root, commit)
    required = {PROFILE, HUMAN, KAGGLE_HUMAN, 'WORKSPACE.json', 'QWEN_PIN.json',
                'MODERNBERT_PIN.json', 'PARENT_MODEL_INPUT_MANIFEST.json'}
    manifest = read(root / 'inputs/MANIFEST.json')
    if set(manifest) != required:
        raise PermissionError('Exact option-A input set required; never attach old targets or sealed inputs')
    for name, expected in manifest.items():
        if Path(name).name != name or sha(root / 'inputs' / name) != expected:
            raise PermissionError('Task-only frozen input bytes changed')
    if sha(root / 'inputs' / HUMAN) != SELECTION_SHA or sha(root / 'inputs' / KAGGLE_HUMAN) != HUMAN_SHA:
        raise PermissionError('Bind the direct human option-A selection and Kaggle account scope')
    if read(root / 'inputs/WORKSPACE.json') != identity(NOTEBOOK):
        raise PermissionError('Wrong task-only notebook/account')
    parent = root / 'inputs/PARENT_MODEL_INPUT_MANIFEST.json'
    if sha(parent) != ORIGINAL_MODEL_MANIFEST:
        raise PermissionError('Original complete model pins changed')
    for name in ['QWEN_PIN.json', 'MODERNBERT_PIN.json']:
        if sha(root / 'inputs' / name) != read(parent)[name]:
            raise PermissionError('Pinned model metadata changed')
    from .composite_frozen_probe import verify_original_model_pin
    verify_original_model_pin(read(root / 'inputs/QWEN_PIN.json'))
    cfg = read(root / 'config/composite_task_skill_v1.json')
    if (cfg.get('human_selection_sha256') != SELECTION_SHA or cfg.get('training_world_lifecycle_effects') is not False
            or cfg.get('risk_examples_in_training') is not False):
        raise PermissionError('Option-A scientific scope differs')
    profile = read(root / 'inputs' / PROFILE)
    if profile != expected_profile(commit):
        raise PermissionError('Unregistered task-only stage/device/export configuration')
    return profile, cfg


def require_runtime():
    from .composite_runtime import require_scientific_runtime, platform
    require_scientific_runtime()
    if platform() != 'kaggle':
        raise PermissionError('Current Research4 task-only science belongs on Kaggle')
    root = Path(os.environ['RESEARCH4_TASK_SOURCE_ROOT']).resolve()
    base = Path('/kaggle/working/research4_inputs').resolve()
    if not root.is_relative_to(base) or root.name != 'source':
        raise PermissionError('Exact owned task-only source root required')
    profile, _ = validate_source(root, read(root / 'RELEASE.json')['commit'])
    if os.environ.get('RESEARCH4_TASK_PROFILE_SHA256') != sha(root / 'inputs' / PROFILE):
        raise PermissionError('Actual task-only source/profile differs')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != profile['visible_devices']:
        raise PermissionError('Task-only data stage must expose no GPUs')
    return root, profile


def authorize(root, commit):
    actual, profile = require_runtime()
    if Path(root).resolve() != actual or profile['source_commit'] != commit:
        raise PermissionError('Another source entered the task-only data stage')
    return profile, read(actual / 'config/composite_task_skill_v1.json')
