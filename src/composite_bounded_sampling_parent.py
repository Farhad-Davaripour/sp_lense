"""Re-audit the consumed, failed c37 reference run without issuing data authority.

The exact original source executes only its recorded-action reader.  Neither a
serialized diagnostic nor the original worker's failed receipt is a PASS.  The
in-process cache is keyed by complete content hashes, including the large raw
alias ancestors; file timestamps never stand in for lineage verification.
"""
import base64
from contextlib import contextmanager
from copy import deepcopy
import importlib
import sys
from pathlib import Path
import re
import shutil
import types
import uuid

from .io_utils import read, sha, digest
from .composite_task_skill_source_package import frozen_source_package

SOURCE = 'c37f4845896d2dc2824fc8dd2d840dc5fe627948'
RAW_SHA = '44722e1870f55fb6b7a1e1aafe6f9c9254c35687460c18f2cff7df8c21d5531c'
PACKAGE_SHA = 'd7bbf40c73473605780ca24b46a470b1f1d5eb47a71ade5f1be4e3691e561e2a'
CLOSEOUT_SHA = 'c867b2a215ce821e2301dde39b250e0fd5ac5f58fc26871da392a970e843e0b8'
CONFIG_SHA = 'f273497e5f1ff7581cef4332cf9825b84f260f099ce1fc1cedbbb2c2e47a5d21'
MANIFEST = 'BOUNDED_BEHAVIOR_DATA_PACKAGE_FILES.json'
NOTEBOOK = 'farhad89/research4-bounded-behavior-data-v4'
RUN = 362532871
PREFIX = 'FAILED_REFERENCE_'
FILES = {PREFIX + n for n in ['SOURCE.r4b64', 'RAW.r4blob', 'EXPECTED.json',
    'CLOSEOUT.json', 'BINDING.json', 'CONTROLLER.json', 'SETUP.json',
    'TERMINAL_AX.txt', 'RELEASE_AX.txt', 'ACTUAL.ipynb', 'PREPARED.ipynb']}
_KEY = object()
_CACHE = {}
ROLES = ('training', 'development')


def descriptor():
    return {'source_commit': SOURCE, 'raw_archive_sha256': RAW_SHA,
            'package_sha256': PACKAGE_SHA, 'closeout_sha256': CLOSEOUT_SHA,
            'run_id': RUN, 'configuration_sha256': CONFIG_SHA,
            'scientific_dataset_admitted': False, 'worker_admitted_completed_cases': 0,
            'recorded_training_cases': 108, 'recorded_development_cases': 36,
            'eligible_training_rows': 640, 'original_uniform_sampling_failed': True}


def assemble_failed_parent(failed_stage_directory, inputs_directory):
    """Copy already exported evidence; this does not run an audit or sampler."""
    source, target = Path(failed_stage_directory), Path(inputs_directory)
    mapping = {'RAW.r4blob': 'RUN_362532871_EVIDENCE.zip',
        'CLOSEOUT.json': 'FAILED_RUN_CLOSEOUT.json', 'BINDING.json': 'SUBMISSION.json',
        'CONTROLLER.json': 'BOUNDED_DATA_CONTROLLER_RECEIPT.json',
        'SETUP.json': 'BOUNDED_DATA_SETUP_RECEIPT.json',
        'TERMINAL_AX.txt': 'TERMINAL_FAILURE_AX.txt', 'RELEASE_AX.txt': 'RELEASE_AX.txt',
        'ACTUAL.ipynb': 'ACTUAL_VERSION_362532871.ipynb',
        'PREPARED.ipynb': 'research4_bounded_behavior_data_v4.ipynb'}
    for name, expected in [('RUN_362532871_EVIDENCE.zip', RAW_SHA),
                           ('BOUNDED_BEHAVIOR_DATA_PACKAGE.zip', PACKAGE_SHA),
                           ('FAILED_RUN_CLOSEOUT.json', CLOSEOUT_SHA)]:
        if sha(source / name) != expected: raise PermissionError('Pinned failed parent changed: ' + name)
    target.mkdir(parents=True, exist_ok=True)
    if any((target / n).exists() for n in FILES):
        raise FileExistsError('Preserve the existing failed-reference inputs')
    for name, path in mapping.items(): shutil.copyfile(source / path, target / (PREFIX + name))
    (target / (PREFIX + 'SOURCE.r4b64')).write_bytes(base64.b64encode(
        (source / 'BOUNDED_BEHAVIOR_DATA_PACKAGE.zip').read_bytes()))
    from .io_utils import dump
    dump(target / (PREFIX + 'EXPECTED.json'), descriptor())
    return descriptor()


@contextmanager
def _original_modules(encoded):
    """Isolate the actual old reader, including its typed recorded-replay scope."""
    with frozen_source_package(encoded, PACKAGE_SHA, SOURCE, manifest_name=MANIFEST) as (root, _):
        name = '_research4_failed_reference_' + uuid.uuid4().hex
        package = types.ModuleType(name)
        package.__path__ = [str(root / 'src')]
        package.__package__ = name
        sys.modules[name] = package
        try:
            def module(short): return importlib.import_module(name + '.' + short)
            yield root, module
        finally:
            for item in list(sys.modules):
                if item == name or item.startswith(name + '.'): del sys.modules[item]


def _code(path):
    return [''.join(c['source']) for c in read(path)['cells'] if c['cell_type'] == 'code']


def _raw_dependency(value, environment):
    import os
    path = Path(value if value is not None else os.environ[environment]).resolve()
    if value is None and not path.is_relative_to(Path('/kaggle/input').resolve()):
        raise PermissionError('Scientific raw ancestry must be immutable Kaggle input')
    if not path.is_file(): raise PermissionError('Complete original alias raw dependency required')
    return path


def validate_failure_evidence(inputs, root, module):
    """Require the actual controlled failure, source code, and target release."""
    inputs, root = Path(inputs), Path(root)
    closed = read(inputs / (PREFIX + 'CLOSEOUT.json'))
    if sha(inputs / (PREFIX + 'CLOSEOUT.json')) != CLOSEOUT_SHA:
        raise PermissionError('The consumed failed closeout is immutable')
    bound = {'CONTROLLER.json': 'controller_sha256', 'ACTUAL.ipynb': 'actual_saved_notebook_sha256',
             'RELEASE_AX.txt': 'release_AX_original_sha256'}
    if any(sha(inputs / (PREFIX + n)) != closed[k] for n, k in bound.items()):
        raise PermissionError('Original failed provider/controller/source capture changed')
    binding = read(inputs / (PREFIX + 'BINDING.json'))
    controller = read(inputs / (PREFIX + 'CONTROLLER.json'))
    expected_closed = {'scriptVersionId': RUN, 'source_commit': SOURCE,
        'original_raw_archive_sha256': RAW_SHA, 'owned_compute_released': True,
        'controller_child_exit_confirmed': True, 'controller_child_exit_code': 1,
        'timed_out': False, 'scientific_dataset_admitted': False,
        'model_training_admitted': False, 'worker_admitted_completed_cases': 0}
    if any(digest(closed.get(k)) != digest(v) for k, v in expected_closed.items()):
        raise PermissionError('A failed reference receipt cannot become successful data')
    if (binding.get('scriptVersionId') != RUN or binding.get('source_commit') != SOURCE
            or binding.get('identity') != closed.get('identity')
            or binding.get('package_sha256') != PACKAGE_SHA
            or controller.get('source_commit') != SOURCE or controller.get('notebook') != NOTEBOOK
            or controller.get('status') != 'failed' or controller.get('exit_code') != 1
            or controller.get('timed_out') is not False
            or controller.get('child', {}).get('child_exit_confirmed') is not True
            or controller.get('child', {}).get('exit_code') != 1
            or controller.get('public_fixture_test_mode') is not False):
        raise PermissionError('Exact failed controlled child identity required')
    terminal = (inputs / (PREFIX + 'TERMINAL_AX.txt')).read_text(encoding='utf-8')
    released = (inputs / (PREFIX + 'RELEASE_AX.txt')).read_text(encoding='utf-8')
    target = 'https://www.kaggle.com/code/' + NOTEBOOK
    if (target + '?scriptVersionId=' + str(RUN) not in terminal
            or '"status": "failed"' not in terminal):
        raise PermissionError('Exact failed notebook/version capture required')
    if (not re.search(re.escape(target) + r'''(?=$|[/?#\s"'<>])''', released)
            or not re.search(r'0\s+Active Events', released) or 'Draft Session off' not in released):
        raise PermissionError('Actual target notebook and global compute release required')
    profile = module('composite_bounded_behavior_data_profile')
    expected_code = profile.notebook_code(root, SOURCE, sha(root.parent / MANIFEST))
    if (_code(inputs / (PREFIX + 'ACTUAL.ipynb')) != [expected_code]
            or _code(inputs / (PREFIX + 'PREPARED.ipynb')) != [expected_code]):
        raise PermissionError('Actual and prepared failed notebooks must equal reviewed bootstrap bytes')
    return closed


def _reference_exports(files, profile):
    required = {'PRECALL_FREEZE.json', 'OUTPUT_OWNER.json', 'RUN_RECEIPT.json', 'FAILURE.json',
        'metadata/QWEN_PIN.json', 'metadata/QWEN_FILES.json', 'metadata/QWEN_BYTES.json',
        'training/FITTING_ROWS.json', 'training/FITTING_DRAW_PLAN.json', 'LEAKAGE_AUDIT.json'}
    for role in ROLES:
        required |= {role + '/' + n for n in ['SCENES.json', 'EPISODES.json', 'ROWS.json',
            'COHORT_AUDIT.json', 'NATIVE_BUDGET.json']}
        required |= {role + '/case_exports/' + s['case_id'] + '.json' for s in profile['rosters'][role]}
    if set(files.manifest) != required: raise PermissionError('Exact complete failed reference exports required')
    return {n: files.json(n) for n in required}


def audit_references(module, context, files, metadata_parent):
    """Replay only the complete recorded references using their original source."""
    data = module('composite_bounded_behavior_data')
    reader = module('composite_bounded_behavior_data_reader')
    runtime = module('composite_bounded_behavior_runtime')
    cfg, profile = context.cfg(), context.profile()
    episodes, rows, audits, budgets = {}, {}, {}, {}
    for role in ROLES:
        episodes[role], rows[role] = files.json(role + '/EPISODES.json'), files.json(role + '/ROWS.json')
        if files.json(role + '/SCENES.json') != profile['rosters'][role]:
            raise PermissionError('Original complete nonsealed scene roster changed')
        for episode in episodes[role]:
            case = role + '/case_exports/' + episode['scene']['case_id'] + '.json'
            if files.json(case) != episode: raise PermissionError('Durable per-case and final evidence differ')
    archive = reader.BoundedDataArchive(context, files, None, reader._ARCHIVE_KEY)
    with runtime.reference_replay_scope(archive):
        for role in ROLES:
            if rows[role] != data.selected_rows(episodes[role], role):
                raise PermissionError('Original eligibility, phases, and first-request tags changed')
            audits[role] = data.audit_cohort(episodes[role], rows[role], cfg, role, profile['rosters'][role])
            if audits[role] != files.json(role + '/COHORT_AUDIT.json'):
                raise PermissionError('Independent complete recorded-action replay differs')
    with reader.verified_tokenizer(metadata_parent, files) as tokenizer:
        for role in ROLES:
            budgets[role] = data.native_budget(rows[role], tokenizer)
            if budgets[role]['status'] != 'pass' or budgets[role] != files.json(role + '/NATIVE_BUDGET.json'):
                raise PermissionError('Original native token plans must pass exactly without truncation')
    leak = module('composite_generalized_audit').leakage_audit(
        audits['training'], audits['development'], rows['training'], rows['development'])
    if leak['status'] != 'structural_and_exact_leakage_checks_passed' or leak != files.json('LEAKAGE_AUDIT.json'):
        raise PermissionError('Complete original reference leakage checks must pass')
    fit = data.fitting_rows(rows['training'])
    if (len(fit) != 640 or fit != files.json('training/FITTING_ROWS.json')
            or len({r['state_id'] for r in fit}) != len(fit)):
        raise PermissionError('All640 original eligible rows must remain unique and unchanged')
    plan = data.training_draw_plan(rows['training'], cfg)
    if plan != files.json('training/FITTING_DRAW_PLAN.json') or plan['coverage_passed'] is not False:
        raise PermissionError('Preserve the original failed128 uniform indices and coverage')
    return {'audits': audits, 'budgets': budgets, 'leakage': leak, 'rows': rows,
            'episodes': episodes, 'fit': fit, 'original_draw_plan': plan}


class FailedReferenceAudit:
    def __init__(self, receipt, cfg, profile, exports, metadata, package, release, replayed, key=None):
        if key is not _KEY: raise PermissionError('Rebuild the actual failed raw/source lineage')
        self._key = key
        self._receipt, self._cfg, self._profile = deepcopy(receipt), deepcopy(cfg), deepcopy(profile)
        self._exports, self._metadata = deepcopy(exports), deepcopy(metadata)
        self._package, self._release = bytes(package), deepcopy(release)
        self._replayed = deepcopy(replayed)
    def receipt(self): return deepcopy(self._receipt)
    def cfg(self): return deepcopy(self._cfg)
    def scenes(self, role): return self._role(role, 'SCENES.json')
    def rows(self, role): return self._role(role, 'ROWS.json')
    def episodes(self, role): return self._role(role, 'EPISODES.json')
    def native_budget(self, role): return self._role(role, 'NATIVE_BUDGET.json')
    def _role(self, role, name):
        require_failed_parent(self)
        if role not in ROLES: raise PermissionError('There is no sealed reference role')
        return deepcopy(self._exports[role + '/' + name])
    def fitting_rows(self): return deepcopy(self._exports['training/FITTING_ROWS.json'])
    def original_draw_plan(self): return deepcopy(self._exports['training/FITTING_DRAW_PLAN.json'])
    def tokenizer_metadata(self):
        require_failed_parent(self)
        return deepcopy(self._metadata)
    def parent_tokenizer_metadata(self): return self.tokenizer_metadata()
    def original_source(self):
        require_failed_parent(self)
        return bytes(self._package)
    def source_release(self): return deepcopy(self._release)
    def protocol(self):
        return {'version': 'failed-reference-uniform128-preserved-v1',
                'failed_reference': self.receipt(), 'original_draw_plan': self.original_draw_plan(),
                'scientific_dataset_admitted': False, 'model_training_admitted': False}
    def reference_exports(self):
        """Only records; failed worker/freeze files remain separately historical."""
        require_failed_parent(self)
        return deepcopy({n: v for n, v in self._exports.items() if n.startswith(('training/', 'development/', 'metadata/'))
                         or n == 'LEAKAGE_AUDIT.json'})
    def replay(self, files=None):
        """No teacher/new-scene route: repeat the old recorded-action audit only."""
        require_failed_parent(self)
        if files is None:
            # The factory has already independently replayed this complete
            # immutable content. CPU materialization reuses that result rather
            # than repeating the expensive historical alias/native audit.
            return deepcopy(self._replayed)
        expected = self.reference_exports()
        if any(digest(files.json(n)) != digest(v) for n, v in expected.items()):
            raise PermissionError('Fresh exports must preserve every original reference byte value')
        with _original_modules(self._package) as (root, module):
            profile = module('composite_bounded_behavior_data_profile')
            context = profile.BoundedDataContext(root, self._profile, self._cfg, None, profile._KEY)
            return audit_references(module, context, files, self)


def require_failed_parent(value):
    if not isinstance(value, FailedReferenceAudit) or value._key is not _KEY:
        raise PermissionError('Serialized failure diagnostics do not create reference lineage')
    r = value.receipt()
    expected = {**descriptor(), 'complete_recorded_reference_replay': True,
                'native_budget_replayed': True, 'leakage_replayed': True,
                'owned_compute_released': True, 'new_model_forwards': 0,
                'new_optimizer_updates': 0, 'heldout_content_read': False}
    if any(digest(r.get(k)) != digest(v) for k, v in expected.items()):
        raise PermissionError('Exact failed released source/reference lineage required')
    return r


def dependency_stamp(inputs, raw, fit, qualification):
    """Complete immutable content keys, including every external raw ancestor."""
    hashes = {n: sha(Path(inputs) / n) for n in sorted(FILES)}
    raw_hash, fit_hash, qual_hash = sha(raw), sha(fit), sha(qualification)
    return digest({'inputs': hashes, 'raw': raw_hash, 'fit': fit_hash, 'qualification': qual_hash}), hashes


def read_failed_parent(source_root, failed_archive=None, alias_fit_archive=None, alias_qual_archive=None):
    inputs = Path(source_root) / 'inputs'
    if read(inputs / (PREFIX + 'EXPECTED.json')) != descriptor():
        raise PermissionError('Do not replace the original failed reference descriptor')
    raw = Path(failed_archive) if failed_archive is not None else inputs / (PREFIX + 'RAW.r4blob')
    fit = _raw_dependency(alias_fit_archive, 'RESEARCH4_BOUNDED_ALIAS_FIT_ARCHIVE')
    qual = _raw_dependency(alias_qual_archive, 'RESEARCH4_BOUNDED_ALIAS_QUALIFICATION_ARCHIVE')
    # Rehash every dependency on every lookup.  A cached typed proof only saves
    # repeated arithmetic/model-state reading, never actual-byte verification.
    stamp, hashes = dependency_stamp(inputs, raw, fit, qual)
    if hashes[PREFIX + 'RAW.r4blob'] != RAW_SHA or sha(raw) != RAW_SHA:
        raise PermissionError('Original failed raw archive must remain unchanged')
    if stamp in _CACHE: return _CACHE[stamp]
    package = base64.b64decode((inputs / (PREFIX + 'SOURCE.r4b64')).read_bytes(), validate=True)
    with _original_modules(package) as (root, module):
        closed = validate_failure_evidence(inputs, root, module)
        source_profile = module('composite_bounded_behavior_data_profile')
        p = read(root / 'inputs' / source_profile.PARENT_DESCRIPTOR)
        old_parent = module('composite_bounded_behavior_parent')
        with old_parent.read_parent(root / 'inputs' / source_profile.PARENT,
                                    p['parent_packet_sha256'], fit, qual) as alias:
            context = source_profile.validate_source(root, SOURCE, alias)
            cfg, profile = context.cfg(), context.profile()
            if digest(cfg) != CONFIG_SHA: raise PermissionError('Original scientific configuration changed')
            reader = module('composite_bounded_behavior_data_reader')
            files = module('composite_v2_model_audit').EvidenceFiles(raw, RAW_SHA,
                source_profile.BOUNDS['maximum_raw_export_bytes'])
            try:
                receipt = files.json('RUN_RECEIPT.json')
                expected = {'status': 'failed', 'source_commit': SOURCE, 'stage': source_profile.MODE,
                    'phase': source_profile.PHASE, 'model_forwards': 0, 'optimizer_updates': 0,
                    'scientific_GPU_count': 0, 'GPU_required': False, 'model_weights_downloaded': False,
                    'heldout_content_read': False, 'public_fixture_test_mode': False,
                    'counts_exact_at_controlled_exit': True, 'worker_exiting': True,
                    'metadata_cache_removed': True, 'completed_cases': 0, 'completed_roles': [],
                    'scientific_dataset_admitted': False, 'training_launch_admitted': False,
                    'automatic_scientific_retries': 0,
                    'error_type': 'ValueError', 'error': 'The exact frozen128 draws omit registered coverage; no resampling'}
                if any(digest(receipt.get(k)) != digest(v) for k, v in expected.items()):
                    raise PermissionError('Only the exact failed original worker boundary is reusable')
                cpu = reader.read_cpu_runtime(receipt)
                if cpu != read(inputs / (PREFIX + 'SETUP.json'))['cpu_runtime']:
                    raise PermissionError('Original setup and raw CPU provenance differ')
                freeze = {'source_commit': SOURCE, 'profile': profile, 'study_configuration': cfg,
                    'input_manifest_sha256': sha(root / 'inputs/MANIFEST.json'),
                    'source_release_sha256': sha(root / 'RELEASE.json'),
                    'parent_descriptor_sha256': profile['parent_descriptor_sha256']}
                if files.json('PRECALL_FREEZE.json') != freeze:
                    raise PermissionError('Original complete pre-call freeze differs')
                owner = {'source_commit': SOURCE, 'version': source_profile.MODE,
                    'phase': source_profile.PHASE, 'owner_thread_id': profile['owner_thread_id'],
                    'path': profile['destination']}
                if files.json('OUTPUT_OWNER.json') != owner: raise PermissionError('Original raw owner differs')
                exports = _reference_exports(files, profile)
                result = audit_references(module, context, files, alias)
                if (len(result['episodes']['training']) != 108 or len(result['episodes']['development']) != 36
                        or len(result['rows']['training']) != 748 or len(result['rows']['development']) != 236):
                    raise PermissionError('Complete144 original references and984 states required')
                report = {**descriptor(), 'complete_recorded_reference_replay': True,
                    'native_budget_replayed': True, 'leakage_replayed': True,
                    'owned_compute_released': True, 'original_fitting_rows_sha256': digest(result['fit']),
                    'original_draw_plan_sha256': digest(result['original_draw_plan']),
                    'reference_exports_sha256': digest(exports), 'source_release_sha256': sha(root / 'RELEASE.json'),
                    'source_manifest_sha256': sha(root.parent / MANIFEST),
                    'alias_parent': alias.receipt(), 'dependency_content_hashes': {'fit': sha(fit), 'qualification': sha(qual)},
                    'cpu_runtime': cpu, 'new_model_forwards': 0, 'new_optimizer_updates': 0,
                    'heldout_content_read': False, 'success_data_closure_called': False}
                if (report['original_fitting_rows_sha256'] != closed['original_fitting_rows_sha256']
                        or report['original_draw_plan_sha256'] != closed['original_draw_plan_canonical_sha256']):
                    raise PermissionError('Original admitted0 closeout reference bindings differ')
                proof = FailedReferenceAudit(report, cfg, profile, exports, alias.tokenizer_metadata(),
                    package, read(root / 'RELEASE.json'), result, _KEY)
                require_failed_parent(proof)
            finally: files.close()
    _CACHE[stamp] = proof
    return proof
