"""Inert recovery profile preparation and an explicit human-exception gate.

This module grants no authority. Operators must resolve the recorded human
message and the shared execution ledger before making the one approved submit.
"""
import copy
import os
from pathlib import Path
from .io_utils import read, sha

PROFILE = 'kaggle-recognition-complete-75min-v1'
AUTHORIZATION = 'RECOVERY_AUTHORIZATION.json'
WORKER_SECONDS = 4200
HARD_SECONDS = 4500


def execution_config(root, commit, mode, config):
    root = Path(root)
    path = root / 'inputs/EXECUTION_PLATFORM.json'
    if not path.exists():
        return config
    pin = read(path)
    recovery = pin.get('recovery')
    if recovery is None:
        if pin.get('hard_timeout_seconds', 2700) != 2700:
            raise PermissionError('A changed time bound requires an explicit recovery exception')
        return config
    if os.environ.get('RESEARCH4_EXECUTION_PLATFORM') != 'kaggle' or mode != 'detector-rank32-v5':
        raise PermissionError('Recovery profile is recognition-only on the frozen Kaggle runtime')
    if pin['source_commit'] != commit or pin['hard_timeout_seconds'] != HARD_SECONDS:
        raise ValueError('Recovery source or hard bound differs')
    if recovery['profile'] != PROFILE or recovery['authorization_file'] != AUTHORIZATION:
        raise ValueError('Unregistered recovery profile')
    authorization = read(root / 'inputs' / AUTHORIZATION)
    if authorization.get('status') != 'explicit_human_exception' or authorization.get('authorized_by') != 'human_user':
        raise PermissionError('A proposal does not authorize another scientific run or longer bounds')
    expected = {'profile': PROFILE, 'parent_run_id': recovery['parent_run_id'],
                'source_commit': commit, 'notebook': pin['notebook'],
                'original_scientific_input_manifest_sha256': pin['original_scientific_input_manifest_sha256'],
                'transport_manifest_sha256': pin['transport_manifest_sha256'],
                'maximum_fresh_recovery_submissions': 1,
                'fresh_full_run_not_exact_continuation': True,
                'use_selected_checkpoint_weights': False, 'initialization_seed': 202610213,
                'maximum_optimizer_updates': 1200,
                'validation_updates': [200, 400, 600, 800, 1000, 1200],
                'hard_timeout_seconds': HARD_SECONDS, 'worker_seconds': WORKER_SECONDS,
                'scientific_visible_GPU_count': 1, 'maximum_backbone_forwards': 30000,
                'automatic_scientific_retries': 0, 'maximum_concurrent_research_runs': 1}
    if any(authorization.get(key) != value for key, value in expected.items()):
        raise PermissionError('Human exception does not bind this exact recovery proposal')
    if not authorization.get('human_message_evidence_locator'):
        raise PermissionError('Operator must retain the actual human authorization evidence')
    recipe = root / 'config/composite_detector_rank32_v5.json'
    if authorization.get('recipe_sha256') != sha(recipe):
        raise ValueError('Recovery scientific recipe changed')
    if authorization.get('original_protocol_sha256') != sha(root / 'config/composite_controller_rank32_v2.json'):
        raise ValueError('Recovery original protocol changed')
    if read(recipe)['steps'] != 1200 or read(recipe)['validation_updates'] != [200, 400, 600, 800, 1000, 1200]:
        raise ValueError('Recovery may not trim updates or validation')
    bounds = config['bounds']
    if (bounds['worker_seconds'], bounds['hard_timeout_minutes'], bounds['export_reserve_seconds'],
        bounds['maximum_backbone_forwards_per_run'], bounds['automatic_retries'],
        bounds['maximum_concurrent_research_runs'], bounds['GPU_count']) != (2400, 45, 180, 30000, 0, 1, 1):
        raise ValueError('Recovery must start from the unchanged original bounded profile')
    result = copy.deepcopy(config)
    result['bounds'].update(worker_seconds=WORKER_SECONDS, hard_timeout_minutes=HARD_SECONDS // 60)
    result['execution_recovery'] = {'profile': PROFILE, 'parent_run_id': recovery['parent_run_id'],
                                   'authorization_sha256': sha(root / 'inputs' / AUTHORIZATION),
                                   'kind': 'one_explicitly_authorized_fresh_complete_run',
                                   'exact_continuation': False}
    return result
