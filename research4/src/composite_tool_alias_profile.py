"""Source-bound, frozen ordinary record-entry alias diagnostic.

The fitted actor is loaded from its original released task-development package.
This separate release only selects twelve training-role first observations and
changes the displayed entry function name. No task world or optimizer runs.
"""
import base64
from copy import deepcopy
import hashlib
from pathlib import Path
import re

from .io_utils import digest, read, sha, verify_release
from .composite_source_archive import verify_transport
from .composite_task_skill_inference import validate_source as validate_actor_source
from .composite_task_skill_package import require_frozen_task_package
from .composite_task_skill_training import DATA_ARCHIVE_SHA, read_data
from .composite_tool_alias_probe import ENTRY_NAMES, paired_tool_specs, select_first_catalog_rows
from .composite_v2_model_audit import EvidenceFiles

MODE = 'ordinary-record-entry-alias-frozen-probe-v1'
STAGE = 'TOOL_ALIAS_STAGE.json'
OFFLINE_PREPARATION = 'OFFLINE_PREPARATION.json'
MANIFEST = 'TOOL_ALIAS_PACKAGE_FILES.json'
NOTEBOOK = 'farhad89/research4-ordinary-entry-alias-v1'
PHASE = 'ordinary_entry_alias_diagnostic'
ARMS = ('base', 'always_on')
RESPONSES = 48
FIT_SOURCE = '82cf3ecf41e799facd93b8c11e05c6d1f569599c'
FIT_RUN = 361964150
CHECKPOINT_SHA256 = '76ba55239a00e39e1dea982cea20083ec5536c3168b69e1754b756f2fc646b2f'


def training_selection(actor_root, fit_archive, model):
    """Rebuild the selected rows and original native input IDs from audited bytes."""
    value = require_frozen_task_package(model)
    if (value['fit_source_commit'] != FIT_SOURCE or value['fit_run_id'] != FIT_RUN
            or value['checkpoint_sha256'] != CHECKPOINT_SHA256
            or value['data_archive_sha256'] != DATA_ARCHIVE_SHA):
        raise PermissionError('The original completed full-fit and data lineage is required')
    encoded = (Path(actor_root) / 'inputs/TASK_DATA_EVIDENCE.r4b64').read_bytes()
    data = read_data(base64.b64decode(encoded, validate=True))
    selected = select_first_catalog_rows(data.rows())
    files = EvidenceFiles(fit_archive, value['fit_raw_archive_sha256'], 4 * 1024**3)
    try:
        plan = files.json('TRAINING_TOKEN_PLAN.json')
        token_sha = files.manifest['TRAINING_TOKEN_PLAN.json']
    finally:
        files.close()
    if (plan.get('eligible_training_states') != 1188 or plan.get('source_training_states') != 1188
            or plan.get('heldout_content_read') is not False or len(plan.get('records', [])) != 1188):
        raise PermissionError('Complete original task training-token plan required')
    records = {r['state_id']: r for r in plan['records']}
    if len(records) != 1188:
        raise ValueError('Duplicate state in the original training-token plan')
    result = []
    for index, row in enumerate(selected):
        token_row = records[row['state_id']]
        ids = token_row['input_token_ids']
        if (not isinstance(ids, list) or not ids or len(ids) > 6144
                or any(type(v) is not int or v < 0 for v in ids)
                or token_row['target_token_ids'] == []):
            raise PermissionError('Complete native original prompt and catalog target required')
        result.append({'index': index, 'state_id': row['state_id'], 'case_id': row['case_id'],
                       'group_id': row['group_id'], 'text_sha256': hashlib.sha256(row['text'].encode('utf-8')).hexdigest(),
                       'catalog_input_token_ids_sha256': digest(ids)})
    return selected, result, token_sha


def expected_stage(commit, actor_root, actor_archive_sha256, model, fit_archive,
                   offline_preparation_path):
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Exact reviewed alias source commit required')
    if not isinstance(actor_archive_sha256, str) or not re.fullmatch('[0-9a-f]{64}', actor_archive_sha256):
        raise ValueError('Exact original actor transport SHA required')
    actor = read(Path(actor_root) / 'inputs/TASK_EVALUATION_STAGE.json')
    value = require_frozen_task_package(model)
    if (actor.get('phase') != 'task_development' or actor.get('partition', {}).get('index') != 0
            or actor['source_commit'] != value['evaluation_source_commit']):
        raise PermissionError('Only the released first task partition may supply the actor')
    _, selected, token_sha = training_selection(actor_root, fit_archive, model)
    offline = read(offline_preparation_path)
    specs = paired_tool_specs()
    if (offline.get('version') != 'ordinary-record-entry-alias-preparation-v1'
            or offline.get('status') != 'model_free_prepared_only_no_scientific_run'
            or offline.get('checkpoint_sha256') != value['checkpoint_sha256']
            or offline.get('fit_source_commit') != value['fit_source_commit']
            or offline.get('original_tools_sha256') != digest(specs['catalog'])
            or offline.get('alias_tools_sha256') != digest(specs['list'])
            or offline.get('planned_neural_responses') != RESPONSES
            or offline.get('world_actions_planned') != 0
            or offline.get('new_training_updates') != 0
            or offline.get('new_neural_model_calls') != 0
            or offline.get('provider_submission') is not False
            or len(offline.get('fixed_prompts', [])) != 12):
        raise PermissionError('Original model-free prompt/tool preparation differs')
    for new, old in zip(selected, offline['fixed_prompts']):
        if (new['state_id'] != old.get('state_id') or new['case_id'] != old.get('case_id')
                or new['group_id'] != old.get('group_id')
                or new['catalog_input_token_ids_sha256'] != old.get('original_input_ids_sha256')
                or new['text_sha256'] != old.get('ordinary_text_sha256')
                or not isinstance(old.get('alias_input_ids_sha256'), str)
                or not re.fullmatch('[0-9a-f]{64}', old['alias_input_ids_sha256'])
                or old.get('target_tool_in_original_training') != 'catalog'
                or old.get('training_only') is not True):
            raise PermissionError('The fixed twelve first-call prompts differ')
        new['list_input_token_ids_sha256'] = old['alias_input_ids_sha256']
    return {'version': MODE, 'source_commit': commit, 'platform': 'kaggle', 'account': 'farhad89',
            'notebook': NOTEBOOK, 'phase': PHASE,
            'objective': 'frozen_ordinary_training_first_call_alias_diagnostic_only',
            'actor_source_commit': actor['source_commit'],
            'actor_package_sha256': actor_archive_sha256,
            'actor_stage_sha256': sha(Path(actor_root) / 'inputs/TASK_EVALUATION_STAGE.json'),
            'actor_manifest_sha256': sha(Path(actor_root).parent / 'TASK_TRAINING_PACKAGE_FILES.json'),
            'fit_source_commit': value['fit_source_commit'], 'fit_run_id': value['fit_run_id'],
            'fit_raw_archive_sha256': value['fit_raw_archive_sha256'],
            'checkpoint_sha256': value['checkpoint_sha256'],
            'frozen_model_package_sha256': digest(value),
            'source_data_archive_sha256': value['data_archive_sha256'],
            'original_training_token_plan_sha256': token_sha,
            'offline_preparation_sha256': sha(offline_preparation_path),
            'selected_rows': selected, 'selected_rows_sha256': digest(selected),
            'conditions': list(ENTRY_NAMES), 'arms': list(ARMS),
            'expected_responses': RESPONSES, 'maximum_new_tokens': 160,
            'maximum_tokens': 6144, 'maximum_backbone_forwards': 8000,
            'scientific_GPU_count': 2, 'visible_devices': '0,1',
            'hard_timeout_seconds': 42000, 'worker_seconds': 40800,
            'export_reserve_seconds': 900, 'maximum_raw_evidence_bytes': 2 * 1024**3,
            'physical_export_reserve_bytes': 12 * 1024**3,
            'maximum_concurrent_owned_workers': 1, 'automatic_scientific_retries': 0,
            'optimizer_updates': 0, 'teacher_forcing': False, 'decoder_repair': False,
            'world_actions': 0, 'scientific_validation_admitted': False,
            'full_task72_complete': False, 'risk180_complete': False,
            'heldout_content_read': False,
            'destination': '/kaggle/working/research4_campaign/runs/' + commit + '/ordinary_record_entry_alias_v1'}


def validate_source(alias_root, alias_commit, fit_archive, qualification_archive,
                    actor_root, actor_archive):
    """Validate both independently released sources and the unchanged fit."""
    alias_root = Path(alias_root).resolve()
    actor_root = Path(actor_root).resolve()
    if (not Path(actor_archive).is_file() or
            sha(actor_archive) != read(alias_root / 'inputs' / STAGE).get('actor_package_sha256')):
        raise PermissionError('Original actor package bytes differ')
    verify_release(alias_root, alias_commit)
    verify_transport(alias_root, alias_commit, MODE, MANIFEST)
    actor = validate_actor_source(actor_root, read(actor_root / 'RELEASE.json')['commit'],
                                  fit_archive, qualification_archive)
    from .composite_task_skill_semantics import verify_semantics
    verify_semantics(actor_root, alias_root)
    model = require_frozen_task_package(actor.model)
    stage = read(alias_root / 'inputs' / STAGE)
    expected = expected_stage(alias_commit, actor_root, sha(actor_archive), actor.model,
                              fit_archive, alias_root / 'inputs' / OFFLINE_PREPARATION)
    if digest(stage) != digest(expected):
        raise PermissionError('Alias stage, training-only row selection, or model lineage differs')
    manifest = read(alias_root / 'inputs/MANIFEST.json')
    if manifest != {STAGE: sha(alias_root / 'inputs' / STAGE),
                    OFFLINE_PREPARATION: sha(alias_root / 'inputs' / OFFLINE_PREPARATION)}:
        raise PermissionError('Only the exact alias stage and original offline preparation may enter this source')
    return {'alias_root': alias_root, 'actor_root': actor_root, 'actor': actor,
            'model': actor.model, 'stage': deepcopy(stage),
            'selected_rows': training_selection(actor_root, fit_archive, actor.model)[0]}


def notebook_code(root, source_commit, package_manifest_sha256):
    """Build the one reviewed bootstrap code cell for exact source/manifest bytes."""
    code = (Path(root) / 'notebooks/composite_tool_alias_probe.py').read_text(encoding='utf-8')
    if (code.count('SOURCE_COMMIT_PLACEHOLDER') != 1
            or code.count('PACKAGE_MANIFEST_SHA256_PLACEHOLDER') != 1):
        raise PermissionError('Reviewed alias bootstrap needs both exact release bindings once')
    return (code.replace('SOURCE_COMMIT_PLACEHOLDER', source_commit)
                .replace('PACKAGE_MANIFEST_SHA256_PLACEHOLDER', package_manifest_sha256))
