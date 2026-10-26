"""One-use V4 CPU data admission and independent provider closure.

This module submits nothing. A prepared packet, completed notebook or serialized
pass flag cannot admit fitting; the separate raw data reader remains mandatory.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import re
import zipfile

from .io_utils import digest, read, sha
from .composite_bounded_behavior_data_profile import (
    MODE, PHASE, NOTEBOOK, PROFILE, MANIFEST, PARENT, PARENT_DESCRIPTOR,
    OWNER_THREAD, notebook_code, validate_source)
from .composite_bounded_behavior_parent import read_parent
from .composite_kaggle_v2_profile import HUMAN_SHA
from .composite_kaggle_v2_admission import candidate_rule
from .composite_submission_admission import guard, reserve, reconcile

CANDIDATE = 'bounded_behavior_data_v4'
ATTEMPTS = 'data_attempts'
PACKAGE = 'BOUNDED_BEHAVIOR_DATA_PACKAGE.zip'
NOTEBOOK_FILE = 'research4_bounded_behavior_data_v4.ipynb'
_KEY = object()


class DataKaggleClosure:
    def __init__(self, value, key=None):
        if key is not _KEY:
            raise PermissionError('Reconcile the exact provider/controller/readback evidence')
        self._key, self._value = key, deepcopy(value)

    def receipt(self): return deepcopy(self._value)


def require_data_closure(value):
    if not isinstance(value, DataKaggleClosure) or value._key is not _KEY:
        raise PermissionError('A serialized release flag is not genuine CPU closure')
    result = value.receipt()
    if (result['owned_compute_released'] is not True or result['phase'] != PHASE
            or result['notebook'] != NOTEBOOK):
        raise PermissionError('Exact independently observed CPU release required')
    return result


def kaggle_closure(binding, archive_sha256, terminal_ax, release_ax,
                   controller, actual_code_cells, prepared_code_cells):
    run, commit = binding.get('scriptVersionId'), binding.get('source_commit')
    if (type(run) is not int or run <= 0
            or not re.fullmatch('[0-9a-f]{40}', commit or '')
            or not re.fullmatch('[0-9a-f]{64}', archive_sha256 or '')
            or binding.get('mode') != MODE or binding.get('phase') != PHASE
            or binding.get('notebook') != NOTEBOOK
            or binding.get('owner_thread_id') != OWNER_THREAD):
        raise PermissionError('Exact consumed V4 CPU submission binding required')
    if (NOTEBOOK not in terminal_ax or 'scriptVersionId=' + str(run) not in terminal_ax
            or not re.search(r'successfully ran after|second run - successful', terminal_ax)
            or not re.search(re.escape('https://www.kaggle.com/code/' + NOTEBOOK)
                             + r"(?=$|[/?#\s\"'<>])", release_ax)
            or '0  Active Events' not in release_ax or 'Draft Session off' not in release_ax):
        raise PermissionError('Actual exact notebook success and provider release required')
    child = controller.get('child', {})
    if (controller.get('source_commit') != commit or controller.get('mode') != MODE
            or controller.get('phase') != PHASE or controller.get('notebook') != NOTEBOOK
            or controller.get('owner_thread_id') != OWNER_THREAD
            or controller.get('status') != 'worker_exited'
            or type(controller.get('exit_code')) is not int or controller['exit_code'] != 0
            or controller.get('timed_out') is not False
            or child.get('child_started') is not True or child.get('child_exit_confirmed') is not True
            or type(child.get('exit_code')) is not int or child['exit_code'] != 0
            or controller.get('automatic_scientific_retries') != 0
            or controller.get('model_forwards') != 0 or controller.get('optimizer_updates') != 0
            or controller.get('public_fixture_test_mode') is not False
            or controller.get('raw_archive_sha256') != archive_sha256
            or controller.get('durable_raw_export') is not True
            or controller.get('heldout_content_read') is not False):
        raise PermissionError('Actual controlled V4 CPU child exit differs')
    if not actual_code_cells or digest(actual_code_cells) != digest(prepared_code_cells):
        raise PermissionError('Actual saved notebook differs from the reviewed prepared code')
    return DataKaggleClosure({'source_commit': commit, 'notebook': NOTEBOOK, 'phase': PHASE,
        'scriptVersionId': run, 'archive_sha256': archive_sha256,
        'owned_compute_released': True, 'owner_thread_id': OWNER_THREAD,
        'terminal_ax_sha256': hashlib.sha256(terminal_ax.encode()).hexdigest(),
        'release_ax_sha256': hashlib.sha256(release_ax.encode()).hexdigest(),
        'controller_sha256': digest(controller),
        'actual_code_cells_sha256': digest(actual_code_cells)}, _KEY)


def validate_package(archive, bundle, manifest_name=MANIFEST):
    """The opaque uploaded ZIP must equal the complete prepared source tree."""
    bundle = Path(bundle)
    manifest = read(bundle / manifest_name)
    with zipfile.ZipFile(archive) as stream:
        infos = stream.infolist(); names = [item.filename for item in infos]
        if (len(names) != len(set(names))
                or len(names) > 4096 or sum(item.file_size for item in infos) > 128 * 1024**2
                or set(names) != set(manifest['files']) | {manifest_name}):
            raise PermissionError('Exact complete V4 source/input archive roster required')
        if any(item.is_dir() or item.flag_bits & 1
               or (item.external_attr >> 16) & 0o170000 == 0o120000 for item in infos):
            raise PermissionError('Only ordinary unencrypted source package files are readable')
        for name in names:
            if (name.startswith('/') or '\\' in name or ':' in name
                    or any(p in ['', '.', '..'] for p in name.split('/'))):
                raise PermissionError('Unsafe prepared source archive member')
            if hashlib.sha256(stream.read(name)).hexdigest() != sha(bundle / name):
                raise PermissionError('Compressed V4 upload differs from its reviewed source')


def validate_preparation(path):
    path = Path(path).resolve(); value = read(path); directory = path.parent
    root = directory / 'bundle/source'
    if (value.get('status') != 'prepared_bounded_behavior_CPU_data_not_submitted'
            or value.get('mode') != MODE or value.get('phase') != PHASE
            or value.get('owner_thread_id') != OWNER_THREAD
            or value.get('notebook') != NOTEBOOK
            or value.get('model_forwards') != 0 or value.get('optimizer_updates') != 0
            or value.get('scientific_allocation_admitted_by_preparation') is not False
            or value.get('behavioral_data_materialized') is not False
            or value.get('training_launch_admitted') is not False
            or value.get('heldout_content_read') is not False):
        raise PermissionError('Exact fresh V4 CPU preparation only')
    for member, key in [(PACKAGE, 'bundle_sha256'), ('bundle/' + MANIFEST, 'bundle_manifest_sha256'),
                         (NOTEBOOK_FILE, 'notebook_file_sha256'),
                         ('bundle/source/inputs/' + PROFILE, 'profile_sha256'),
                         ('bundle/source/inputs/MANIFEST.json', 'input_manifest_sha256')]:
        if sha(directory / member) != value[key]:
            raise PermissionError('Prepared V4 source/input/bootstrap bytes changed: ' + member)
    with read_parent(root / 'inputs' / PARENT, value['parent_packet_sha256'],
                     value['fit_archive_local_path'], value['qualification_archive_local_path']) as parent:
        context = validate_source(root, value['source_commit'], parent)
        profile = context.profile()
    validate_package(directory / PACKAGE, directory / 'bundle')
    cells = [''.join(cell['source']) for cell in read(directory / NOTEBOOK_FILE)['cells']
             if cell['cell_type'] == 'code']
    if cells != [notebook_code(root, value['source_commit'], value['bundle_manifest_sha256'])]:
        raise PermissionError('Notebook code differs from its exact source/manifest release')
    return {'source_commit': value['source_commit'], 'notebook': NOTEBOOK, 'mode': MODE,
        'phase': PHASE, 'owner_thread_id': OWNER_THREAD,
        'package_sha256': value['bundle_sha256'], 'package_manifest_sha256': value['bundle_manifest_sha256'],
        'authorization_sha256': value['profile_sha256'], 'parent_packet_sha256': value['parent_packet_sha256'],
        'parent_descriptor_sha256': digest(read(root / 'inputs' / PARENT_DESCRIPTOR)),
        'scientific_GPU_count': 0, 'model_forwards': 0, 'optimizer_updates': 0,
        'hard_timeout_seconds': profile['hard_timeout_seconds'], 'automatic_scientific_retries': 0,
        'heldout_content_read': False, 'kaggle_remainder_scope_sha256': HUMAN_SHA,
        'provider_scope': 'human-selected-Kaggle-remainder-v1', 'fresh_stage': True}


def _provider_snapshot(value):
    observed = value.get('kaggle_observation', {})
    expected = {'account': 'farhad89', 'notebook': NOTEBOOK, 'notebook_draft_sessions': 0,
                'active_notebook_versions': 0, 'pending_notebook_versions': 0,
                'events_checked': True, 'accelerator': 'None'}
    if any(digest(observed.get(key)) != digest(expected_value) for key, expected_value in expected.items()):
        raise PermissionError('Fresh same-account CPU notebook/version/draft/events proof required')


def reserve_data(campaign_root, preparation_file, live_snapshot, guard_token=None):
    root = Path(campaign_root); lock, token, _ = guard(root, guard_token)
    try:
        if not callable(live_snapshot):
            raise PermissionError('Read fresh provider state under the atomic shared guard')
        validate_preparation(preparation_file)
        ledger = read(root / 'SCIENTIFIC_EXECUTION_LEDGER.json')
        parent = ledger['candidates'].get('ordinary_record_entry_alias_frozen_probe_v1')
        if (not parent or parent.get('owned_compute_released') is not True
                or not parent.get('closeout_receipt')):
            raise PermissionError('The original alias run must be independently audited and released')
        from .composite_bounded_behavior_parent import ALIAS_CLOSEOUT_SHA
        if sha(parent['closeout_receipt']) != ALIAS_CLOSEOUT_SHA:
            raise PermissionError('Shared alias ledger closeout differs from the genuine raw parent')
    except Exception:
        if lock.exists() and read(lock).get('token') == token: lock.unlink()
        raise
    def observed():
        value = live_snapshot(); _provider_snapshot(value); return value
    return reserve(root, preparation_file, observed, CANDIDATE, ATTEMPTS,
                   'r4-bounded-data-v4-', validate_preparation, candidate_rule, token)


def reconcile_data(campaign_root, identity, script_version_id=None, guard_token=None):
    return reconcile(campaign_root, identity, CANDIDATE, ATTEMPTS, script_version_id, guard_token)
