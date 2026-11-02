"""Exact V4 actor-data source/parent/role profile. No model or submit API."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import re

from .io_utils import read, sha, digest
from .composite_bounded_behavior_profile import CONFIG_SHA, validate_preparation
from .composite_bounded_behavior_parent import require_parent
from .composite_bounded_behavior_world import Scene, WORLD_VERSION
from .composite_generalized_data import cohort_contract
from .composite_kaggle_v2_profile import HUMAN as KAGGLE_HUMAN, HUMAN_SHA, identity
from .composite_source_archive import verify_transport

MODE = 'bounded-fictional-behavior-v4-data-export'
PHASE = 'bounded_behavior_data_export'
OWNER_THREAD = '01a175f0-fc99-76b1-8bb2-d64f22afbcd7'
PROFILE = 'BOUNDED_BEHAVIOR_DATA_PROFILE.json'
MANIFEST = 'BOUNDED_BEHAVIOR_DATA_PACKAGE_FILES.json'
SCOPE = 'HUMAN_BOUNDED_BEHAVIOR_SCOPE.json'
TAKEOVER = 'HUMAN_EXECUTION_TAKEOVER.json'
TAKEOVER_SHA = '219766f4e93f3fdbba3f6dc2a1e8b3acc4e2b98e62b3a3e84cf94e7a8f6ae70a'
PARENT = 'ALIAS_PARENT.r4blob'
PARENT_DESCRIPTOR = 'ALIAS_PARENT_DESCRIPTOR.json'
NOTEBOOK = 'farhad89/research4-bounded-behavior-data-v4'
NAMESPACE = 'research4-bounded-behavior-v4-cohorts-20261026'
INPUTS = {PROFILE, SCOPE, TAKEOVER, KAGGLE_HUMAN, PARENT, PARENT_DESCRIPTOR, 'WORKSPACE.json'}
BOUNDS = {'hard_timeout_seconds': 1800, 'worker_seconds': 1200,
          'export_reserve_seconds': 120, 'maximum_raw_export_bytes': 256 * 1024**2,
          'terminal_metadata_reserve_bytes': 4 * 1024**2,
          'maximum_reference_states': 3600, 'maximum_concurrent_owned_workers': 1,
          'automatic_scientific_retries': 0}
_KEY = object()


def planned_roster(cfg, role):
    """Identity descriptors only; never construct records, actions or a world."""
    if role not in ['training', 'development']:
        raise PermissionError('V4 initial data has no sealed materialization route')
    contract = cohort_contract(cfg, role)
    result = []
    for group in contract['group_plan']:
        seed = hashlib.sha256((NAMESPACE + '|' + CONFIG_SHA + '|' + role + '|'
                               + str(group['group_index'])).encode()).hexdigest()
        group_id = digest({'version': WORLD_VERSION, 'role': role, 'identity': seed})
        for index, condition in enumerate(contract['conditions']):
            value = {'identity_seed': seed, 'group_id': group_id,
                     'template': group['template'], 'domain': group['domain'],
                     'condition': condition, 'swap': bool(index % 2),
                     'id_digits': cfg['cohorts'][role + '_id_lengths'][
                         group['group_index'] % len(cfg['cohorts'][role + '_id_lengths'])],
                     'hard_stop': 24, 'soft_stop': 20, 'mechanism': group['mechanism'],
                     'task_structure': group['task_structure'], 'role': role,
                     'world_version': 'composite-generalized-world-v2',
                     'behavior_profile': WORLD_VERSION}
            result.append(asdict(Scene(case_id=digest(value), **value)))
    return result


def expected_profile(commit, cfg, parent):
    if not re.fullmatch('[0-9a-f]{40}', commit or ''):
        raise PermissionError('Exact reviewed V4 source commit required')
    from .composite_bounded_behavior_interface import PROFILE as NATIVE_PROFILE
    validate_preparation(cfg)
    dependency = require_parent(parent)
    return {'version': MODE, 'source_commit': commit, 'platform': 'kaggle', 'account': 'farhad89',
            'phase': PHASE, 'owner_thread_id': OWNER_THREAD, 'native_interface_profile': NATIVE_PROFILE,
            'execution_takeover_receipt_sha256': TAKEOVER_SHA,
            'notebook': NOTEBOOK, 'preparation_configuration_sha256': CONFIG_SHA,
            'world_version': WORLD_VERSION, 'identity_namespace': NAMESPACE,
            'identities_depend_on_alias_outcome_or_source_transport': False,
            'roles': ['training', 'development'],
            'rosters': {role: planned_roster(cfg, role) for role in ['training', 'development']},
            'parent_descriptor_sha256': digest(dependency),
            'parent_packet_sha256': dependency['parent_packet_sha256'],
            'visible_devices': '', 'scientific_GPU_count': 0,
            'pretrained_model_forwards': 0, 'optimizer_updates': 0,
            'data_fitting_admission_requires_independent_raw_audit': True,
            'detector_predictions_required_for_actor_data': False,
            'heldout_content_read': False,
            'destination': '/kaggle/working/research4_campaign/runs/' + commit
                           + '/bounded_behavior_v4_data/export_01', **BOUNDS}


class BoundedDataContext:
    def __init__(self, root, profile, cfg, parent, key=None):
        if key is not _KEY: raise PermissionError('Validate the complete new data source and raw parent')
        self._key = key
        self.root = Path(root).resolve()
        self._profile, self._cfg = deepcopy(profile), deepcopy(cfg)
        self.parent = parent

    def profile(self): return deepcopy(self._profile)
    def cfg(self): return deepcopy(self._cfg)

    def require_runtime(self):
        from .composite_runtime import platform, KAGGLE_RUNTIME_DIRECTORIES
        if (platform() != 'kaggle' or not all(Path(p).is_dir() for p in KAGGLE_RUNTIME_DIRECTORIES)
                or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
                or not self.root.is_relative_to(Path('/kaggle/temp/research4_bounded_behavior_data_inputs').resolve())
                or self.root.name != 'source'
                or Path(os.environ.get('RESEARCH4_BOUNDED_DATA_SOURCE_ROOT', '')).resolve() != self.root
                or os.environ.get('RESEARCH4_BOUNDED_DATA_PROFILE_SHA256') != sha(self.root / 'inputs' / PROFILE)):
            raise PermissionError('Exact dedicated Kaggle CPU V4 data runtime required')
        return self


def require_context(value):
    if not isinstance(value, BoundedDataContext) or value._key is not _KEY:
        raise PermissionError('Validate the genuine complete V4 source and alias dependency')
    require_parent(value.parent)
    return value


def notebook_code(root, commit, manifest_sha256):
    if (not re.fullmatch('[0-9a-f]{40}', commit or '')
            or not re.fullmatch('[0-9a-f]{64}', manifest_sha256 or '')):
        raise PermissionError('Exact bootstrap source and manifest bindings required')
    code = (Path(root) / 'notebooks/composite_bounded_behavior_data.py').read_text(encoding='utf-8')
    if (code.count('SOURCE_COMMIT_PLACEHOLDER') != 1
            or code.count('PACKAGE_MANIFEST_SHA256_PLACEHOLDER') != 1):
        raise PermissionError('The V4 bootstrap needs both exact bindings once')
    return code.replace('SOURCE_COMMIT_PLACEHOLDER', commit).replace(
        'PACKAGE_MANIFEST_SHA256_PLACEHOLDER', manifest_sha256)


def validate_source(root, commit, parent):
    root = Path(root).resolve()
    verify_transport(root, commit, MODE, MANIFEST)
    inputs = root / 'inputs'
    manifest = read(inputs / 'MANIFEST.json')
    if set(manifest) != INPUTS or any(sha(inputs / n) != h for n, h in manifest.items()):
        raise PermissionError('Exact complete unsealed V4 data inputs required')
    cfg = validate_preparation(read(root / 'config/composite_bounded_behavior_v4.json'))
    if sha(inputs / SCOPE) != cfg['scope_amendment_receipt_sha256']:
        raise PermissionError('Bind the verified bounded human scope amendment')
    if (sha(inputs / TAKEOVER) != TAKEOVER_SHA
            or read(inputs / TAKEOVER)['new_execution_owner_thread'] != OWNER_THREAD):
        raise PermissionError('Bind the direct human execution-owner transfer')
    if sha(inputs / KAGGLE_HUMAN) != HUMAN_SHA:
        raise PermissionError('Preserve the selected Kaggle account authority')
    dependency = require_parent(parent)
    if (digest(read(inputs / PARENT_DESCRIPTOR)) != digest(dependency)
            or sha(inputs / PARENT) != dependency['parent_packet_sha256']):
        raise PermissionError('Actual raw parent and compiled descriptor differ')
    profile = read(inputs / PROFILE)
    if digest(profile) != digest(expected_profile(commit, cfg, parent)):
        raise PermissionError('Full new source/roles/cohorts/bounds changed')
    if read(inputs / 'WORKSPACE.json') != identity(NOTEBOOK):
        raise PermissionError('Wrong Kaggle data account/notebook')
    return BoundedDataContext(root, profile, cfg, parent, _KEY)
