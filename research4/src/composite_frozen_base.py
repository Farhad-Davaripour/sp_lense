"""Original validation base control, separately typed from the five-control package.

This module has no submission API. A frozen-base result cannot qualify the
behavior adapter, the other four controls, or held-out admission.
"""
import hashlib
import json
from pathlib import Path
import re

from .io_utils import read, sha, digest, verify_release
from .composite_runtime import FrozenInputs
from .composite_frozen_probe import ORIGINAL_INPUTS, PROFILE_FILE, verified_model_context

MODE = 'frozen-base-validation-v1'
AUTHORIZATION = 'FROZEN_BASE_AUTHORIZATION.json'
PLAN = 'FROZEN_BASE_PLAN.json'
ROLE = 'validation'
CONTROLS = ['base', 'always_on', 'initial_gate', 'learned_dynamic', 'oracle_dynamic']
PROTOCOL_SHA256 = '63c0e2706c6125ee1df760a6a195777b6ca41a5d44f955ea84604dc7767ad39d'
EPISODES = 'data_v1/validation/EPISODES.json'
EPISODES_SHA256 = '55fea935e95fe242a23347b30ae07287b88509ade262cdbb1708cb1824e285d7'
ROSTER_SHA256 = '304f8486fc78bdc16b3e57dc14071aeb1b68c5ca156233dcec6251d9da7f813c'
DETECTOR_ARCHIVE = 'kaggle_recognition_v1/361470090/evidence.zip'
DETECTOR_ARCHIVE_SHA256 = '11200f332083c6a44b29a30312e54157659bc3f1f759d578b1c213b86203c75d'
DETECTOR_CHECKPOINT_SHA256 = 'f83e8bdff07b10f3e3ee077d699768f2f1c0f35b35b32d9c22a06f8ae010ceeb'
PARENTS = {
    'recognition_closeout': '810e5439fa1f794ba415ed57e8ad083e5b7204c9652284c22aad61332a4128b2',
    'recognition_audit': 'd5c1b3166d5ccceb46e521d394ad5a047d195fada0d94e95aab050b5b7c6c717',
    'qualification_closeout': '754879e97ff0ccf958179281b9094c179e88372d6426768d0de5a15f2a43e29f',
    'qualification_audit': 'ac3bb19f731ef6c64e97acb20bb3c94c9a6dd7377f96fcce89ffb5d8184e73ed',
}
BOUNDS = {'hard_timeout_seconds': 10800, 'worker_seconds': 10500,
          'export_reserve_seconds': 300, 'maximum_backbone_forwards': 30000,
          'maximum_concurrent_research_runs': 1, 'automatic_scientific_retries': 0}


def roster(raw, cfg):
    """All original scenes in release order; never select by model outcomes."""
    if hashlib.sha256(raw).hexdigest() != EPISODES_SHA256:
        raise ValueError('Original complete validation episode bytes changed')
    if digest(cfg) != PROTOCOL_SHA256 or cfg['evaluation']['conditions'] != CONTROLS:
        raise ValueError('Original full-system protocol/controls changed')
    scenes = [row['scene'] for row in json.loads(raw)]
    ids = [scene['case_id'] for scene in scenes]
    if (len(scenes) != 36 or len(set(ids)) != 36 or digest(ids) != ROSTER_SHA256
        or len({scene['group_id'] for scene in scenes}) != 4
        or {scene['template'] for scene in scenes} != {3, 4}
        or {scene['condition'] for scene in scenes} != set(cfg['cohorts']['conditions'])
        or any(scene['hard_stop'] != 24 for scene in scenes)):
        raise ValueError('Complete original validation roster differs')
    return scenes


def partition_plan(scenes, partition):
    if type(partition) is not int or not 0 <= partition < 6 or len(scenes) != 36:
        raise ValueError('Six fixed partitions cover the complete original roster')
    selected = scenes[partition * 6:(partition + 1) * 6]
    return {'mode': MODE, 'role': ROLE, 'package_kind': 'frozen-base-reference',
            'partition': partition, 'partition_count': 6,
            'case_ids': [scene['case_id'] for scene in selected], 'conditions': ['base'],
            'original_controls': CONTROLS, 'original_expected_trajectories': 180,
            'base_reference_expected_trajectories': 36, 'partition_expected_trajectories': 6,
            'ordered_roster_sha256': ROSTER_SHA256,
            'detector_checkpoint_sha256': DETECTOR_CHECKPOINT_SHA256, 'threshold': 0.9,
            'behavior_adapter': None, 'adapter_bypass_check': None,
            'full_system_admission_allowed': False, 'heldout_content_read': False}


def conservative_forwards(scene_count):
    if type(scene_count) is not int or not 1 <= scene_count <= 6:
        raise ValueError('A base partition must contain one to six whole episodes')
    # At most25 responses plus the initial detector observation per scene;
    # one extra full160-token native response checks instrumentation parity.
    return scene_count * (25 * (160 + 1) + 1) + 160


def validate_parents(inputs):
    values = {}
    for key, expected in PARENTS.items():
        path = Path(inputs) / (key + '.json')
        if sha(path) != expected:
            raise PermissionError('The audited recognition/qualification parent changed')
        values[key] = read(path)
    recognition = values['recognition_closeout']; qualification = values['qualification_closeout']
    if (recognition['run_id'] != 361470090 or recognition['component_validation_admitted'] is not True
        or recognition['checkpoint_sha256'] != DETECTOR_CHECKPOINT_SHA256
        or qualification['run_id'] != 361504592 or qualification['complete_qualification_audited'] is not True
        or any(values[key]['owned_compute_released'] is not True
               for key in ['recognition_closeout', 'qualification_closeout'])):
        raise PermissionError('Audited and released parents are required')
    return values


def authorized_load_plan(root, commit, config, transport=None):
    """Strict source/input/authority binding before downloads or native loading."""
    root = Path(root)
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Bind a complete reviewed source commit')
    verify_release(root, commit)
    for name, expected in read(root / 'inputs/MANIFEST.json').items():
        if Path(name).name != name or sha(root / 'inputs' / name) != expected:
            raise ValueError('Frozen-base input changed')
    pin = read(root / 'inputs/EXECUTION_PLATFORM.json')
    if (pin['platform'] != 'kaggle' or pin['account'] != 'farhad89'
        or pin['mode'] != MODE or pin['source_commit'] != commit
        or pin['scientific_visible_devices'] != '0,1'
        or pin['original_scientific_input_manifest_sha256'] != ORIGINAL_INPUTS
        or any(type(pin.get(key)) is not type(value) or pin[key] != value for key, value in BOUNDS.items())):
        raise PermissionError('Frozen-base source/platform/device envelope differs')
    cfg = read(root / 'config/composite_controller_rank32_v2.json')
    if config != cfg['qwen']:
        raise ValueError('Actor configuration differs from the original protocol')
    validate_parents(root / 'inputs')
    data = FrozenInputs(pin['input_root'] if transport is None else transport, pin['transport_manifest_sha256'])
    scenes = roster(data.get(EPISODES), cfg)
    plan = read(root / 'inputs' / PLAN)
    if plan != partition_plan(scenes, plan['partition']):
        raise ValueError('Frozen-base partition/claims differ from the original control')
    release = data.json('data_v1/DATA_RELEASE.json')
    if (release['cohorts']['validation']['files']['EPISODES.json'] != EPISODES_SHA256
        or release['world_source_sha256'] != sha(root / 'src/composite_world.py')
        or data.files.get(DETECTOR_ARCHIVE) != DETECTOR_ARCHIVE_SHA256):
        raise ValueError('Frozen-base cohort/source/checkpoint transport differs')
    authority = read(root / 'inputs' / AUTHORIZATION)
    expected = {'status': 'explicit_human_continuation', 'authorized_by': 'human_user',
        'mode': MODE, 'source_commit': commit, 'notebook': pin['notebook'],
        'plan_sha256': sha(root / 'inputs' / PLAN), 'profile_sha256': sha(root / PROFILE_FILE),
        'transport_manifest_sha256': pin['transport_manifest_sha256'],
        'original_scientific_input_manifest_sha256': ORIGINAL_INPUTS,
        'scientific_visible_GPU_count': 2, 'maximum_new_stage_submissions': 1,
        'optimizer_updates': 0, 'behavior_adapter_fitting': False,
        'heldout_content_read': False, **BOUNDS}
    if (any(type(authority.get(key)) is not type(value) or authority[key] != value for key, value in expected.items())
        or not authority.get('human_message_evidence_locator')):
        raise PermissionError('Human continuation does not bind this exact frozen-base partition')
    context = verified_model_context(root, commit, config, pin, sha(root / 'inputs' / AUTHORIZATION))
    context.update(bounds=BOUNDS, evaluation_plan=plan, scenes=scenes,
                   scientific_endpoint='original_base_control_only', full_validation_admitted=False)
    return context


def base_report(records, expected_case_ids):
    """Missing trajectories and intervention controls remain explicitly unknown."""
    ids = [record['scene']['case_id'] for record in records]
    if (len(ids) != len(set(ids)) or not set(ids) <= set(expected_case_ids)
        or any(record['condition'] != 'base' for record in records)):
        raise ValueError('Another control or duplicate trajectory cannot enter a base report')
    complete = ids == list(expected_case_ids)
    return {'package_kind': 'frozen-base-reference', 'role': ROLE,
            'partition_complete': complete, 'completed_case_ids': ids,
            'missing_case_ids': [key for key in expected_case_ids if key not in ids],
            'outcomes': {record['scene']['case_id']: record['outcome'] for record in records},
            'original_controls': CONTROLS, 'original_expected_trajectories': 180,
            'base_reference_expected_trajectories': 36,
            'other_controls_status': {key: 'unexecuted_unknown' for key in CONTROLS[1:]},
            'adapter_OFF_bypass': None, 'adapter_ON_effect': None,
            'full_system_validation_complete': False, 'full_validation_admitted': False,
            'heldout_confirmation': 'not_performed', 'heldout_content_read': False}
