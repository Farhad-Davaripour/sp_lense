"""One-use shared Kaggle admission for the separate frozen alias diagnostic."""
from pathlib import Path

from .io_utils import digest, read, sha
from .composite_submission_admission import guard, reconcile, reserve
from .composite_kaggle_v2_admission import candidate_rule
from .composite_kaggle_v2_profile import HUMAN_SHA
from .composite_task_skill_source_package import frozen_source_package
from .composite_tool_alias_profile import (MANIFEST, MODE, NOTEBOOK, STAGE,
                                           PHASE, RESPONSES, notebook_code, validate_source)

CANDIDATE = 'ordinary_record_entry_alias_frozen_probe_v1'
ATTEMPTS = 'entry_alias_attempts'


def validate_actor_transport(root, value):
    """The upload blob must be the same exact original released ZIP bytes."""
    root = Path(root)
    actor_archive = root / 'ACTOR_PACKAGE.zip'
    actor_blob = root / 'ACTOR_PACKAGE.r4blob'
    expected = value.get('actor_package_sha256')
    if value.get('actor_blob_sha256') != expected or not isinstance(expected, str):
        raise PermissionError('Opaque transport must equal the original actor archive')
    if (not actor_archive.is_file() or not actor_blob.is_file()
            or sha(actor_archive) != expected or sha(actor_blob) != expected
            or actor_archive.stat().st_size != actor_blob.stat().st_size):
        raise PermissionError('Opaque transport bytes differ from the original actor archive')
    return actor_archive


def validate_preparation(path):
    path = Path(path).resolve()
    value = read(path)
    root = path.parent
    alias_root = root / 'bundle/source'
    fit_archive = Path(value['fit_archive_local_path'])
    qualification_archive = Path(value['qualification_archive_local_path'])
    if (value.get('status') != 'prepared_frozen_ordinary_entry_alias_not_submitted'
            or value.get('mode') != MODE or value.get('phase') != PHASE
            or value.get('notebook') != NOTEBOOK or value.get('expected_responses') != RESPONSES
            or value.get('platform') != 'kaggle' or value.get('account') != 'farhad89'
            or value.get('new_model_forwards') != 0 or value.get('new_optimizer_updates') != 0
            or value.get('new_world_actions') != 0
            or value.get('scientific_allocation_admitted_by_preparation') is not False
            or value.get('heldout_content_read') is not False):
        raise PermissionError('Only the separate, unsubmitted frozen alias preparation is eligible')
    actor_archive = validate_actor_transport(root, value)
    for member, key in [('TOOL_ALIAS_PACKAGE.zip', 'bundle_sha256'),
                        ('bundle/' + MANIFEST, 'bundle_manifest_sha256'),
                        ('bundle/source/inputs/' + STAGE, 'stage_sha256'),
                        ('bundle/source/inputs/MANIFEST.json', 'input_manifest_sha256'),
                        ('bundle/source/inputs/OFFLINE_PREPARATION.json', 'offline_preparation_sha256'),
                        ('research4_ordinary_entry_alias.ipynb', 'notebook_file_sha256')]:
        if sha(root / member) != value[key]:
            raise PermissionError('Prepared alias source, actor or notebook bytes changed: ' + member)
    raw = actor_archive.read_bytes()
    with frozen_source_package(raw, value['actor_package_sha256'],
                               value['actor_source_commit']) as (actor_root, _):
        context = validate_source(alias_root, value['source_commit'], fit_archive,
                                  qualification_archive, actor_root, actor_archive)
        stage = context['stage']
    if (value['fit_run_id'] != stage['fit_run_id']
            or value['checkpoint_sha256'] != stage['checkpoint_sha256']
            or value['selected_rows_sha256'] != stage['selected_rows_sha256']
            or digest(value['selected_rows']) != digest(stage['selected_rows'])
            or value['actor_source_commit'] != stage['actor_source_commit']):
        raise PermissionError('Prepared exact prompt/model/source binding differs')
    import hashlib
    import zipfile
    with zipfile.ZipFile(root / 'TOOL_ALIAS_PACKAGE.zip') as package:
        names = package.namelist()
        manifest = read(root / 'bundle' / MANIFEST)
        if set(names) != set(manifest['files']) | {MANIFEST}:
            raise PermissionError('Alias source transport omitted or added a member')
        for name in names:
            if hashlib.sha256(package.read(name)).hexdigest() != sha(root / 'bundle' / name):
                raise PermissionError('Alias compressed package differs from prepared source')
    # The uploaded notebook must be exactly the reviewed bootstrap from this release.
    expected_code = notebook_code(alias_root, value['source_commit'], value['bundle_manifest_sha256'])
    cells = [''.join(cell['source']) for cell in read(root / 'research4_ordinary_entry_alias.ipynb')['cells']
             if cell['cell_type'] == 'code']
    if cells != [expected_code]:
        raise PermissionError('Prepared notebook differs from this exact source/manifest release')
    return {'source_commit': value['source_commit'], 'notebook': NOTEBOOK,
            'mode': MODE, 'phase': PHASE, 'package_sha256': value['bundle_sha256'],
            'package_manifest_sha256': value['bundle_manifest_sha256'],
            'actor_package_sha256': value['actor_package_sha256'],
            'actor_source_commit': value['actor_source_commit'],
            'authorization_sha256': value['stage_sha256'],
            'fit_source_commit': stage['fit_source_commit'],
            'fit_run_id': stage['fit_run_id'],
            'fit_raw_archive_sha256': stage['fit_raw_archive_sha256'],
            'checkpoint_sha256': stage['checkpoint_sha256'],
            'source_data_archive_sha256': stage['source_data_archive_sha256'],
            'selected_rows_sha256': stage['selected_rows_sha256'],
            'expected_responses': RESPONSES, 'optimizer_updates': 0,
            'world_actions': 0, 'scientific_GPU_count': 2,
            'hard_timeout_seconds': stage['hard_timeout_seconds'],
            'automatic_scientific_retries': 0, 'heldout_content_read': False,
            'kaggle_remainder_scope_sha256': HUMAN_SHA,
            'provider_scope': 'human-selected-Kaggle-remainder-v1', 'fresh_stage': True}


def _provider_snapshot(snapshot, notebook):
    """Require a live same-account notebook/draft/event/quota observation."""
    provider = snapshot.get('kaggle_observation')
    if (not isinstance(provider, dict) or provider.get('account') != 'farhad89'
            or provider.get('notebook') != notebook
            or provider.get('notebook_draft_sessions') != 0
            or provider.get('active_notebook_versions') != 0
            or provider.get('pending_notebook_versions') != 0
            or provider.get('gpu_quota_sufficient') is not True
            or provider.get('events_checked') is not True):
        raise PermissionError('Fresh Kaggle account, draft, version, events and quota proof required')


def reserve_alias(campaign_root, preparation_file, live_snapshot, guard_token=None):
    root = Path(campaign_root)
    lock, token, _ = guard(root, guard_token)
    try:
        if not callable(live_snapshot):
            raise PermissionError('Fresh live Kaggle readback must be requested after source validation')
        ledger = read(root / 'SCIENTIFIC_EXECUTION_LEDGER.json')
        if ledger.get('active_owned_submission') is not None:
            raise PermissionError('The current owned worker must finish, audit and release first')
        binding = validate_preparation(preparation_file)
        parent = ledger['candidates'].get('task_skill_stateful_memory_full_fit_v3')
        if (not parent or parent.get('owned_compute_released') is not True
                or not parent.get('closeout_receipt')):
            raise PermissionError('Original full fit must be fully raw-audited and released')
        closeout = read(parent['closeout_receipt'])
        required = {'scriptVersionId': binding['fit_run_id'],
                    'source_commit': binding['fit_source_commit'],
                    'raw_archive_sha256': binding['fit_raw_archive_sha256'],
                    'checkpoint_sha256': binding['checkpoint_sha256'],
                    'full_fit_completed': True, 'full_fit_raw_audited': True,
                    'owned_compute_released': True}
        if any(digest(closeout.get(k)) != digest(v) for k, v in required.items()):
            raise PermissionError('Full-fit ledger closeout differs from original raw checkpoint')
    except Exception:
        if lock.exists() and read(lock).get('token') == token:
            lock.unlink()
        raise
    def observed():
        snapshot = live_snapshot()
        _provider_snapshot(snapshot, binding['notebook'])
        return snapshot
    return reserve(root, preparation_file, observed, CANDIDATE, ATTEMPTS,
                   'r4-entry-alias-', validate_preparation, candidate_rule, token)


def reconcile_alias(campaign_root, identity, script_version_id=None, guard_token=None):
    return reconcile(campaign_root, identity, CANDIDATE, ATTEMPTS,
                     script_version_id, guard_token)
