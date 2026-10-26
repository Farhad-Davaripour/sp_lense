"""Independent V4 raw CPU-data audit; no new cases, teacher or model calls."""
import base64
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import tempfile

from .io_utils import read, sha, digest
from .composite_v2_model_audit import EvidenceFiles
from .composite_bounded_behavior_data_profile import (validate_source, MODE, PROFILE, MANIFEST,
                                                     PARENT, PARENT_DESCRIPTOR, BOUNDS)
from .composite_bounded_behavior_parent import read_parent
from .composite_bounded_behavior_data import (ROLES, audit_cohort, selected_rows, fitting_rows,
                                             native_budget, training_draw_plan, build_release)
from .composite_generalized_audit import leakage_audit

_KEY = object()
_ARCHIVE_KEY = object()


class BoundedDataArchive:
    def __init__(self, context, files, closure, key=None):
        if key is not _ARCHIVE_KEY: raise PermissionError('Use the source-bound raw data reader')
        self._key, self.context, self.files, self.closure = key, context, files, closure
        self.reference_actions = {}
        for role in ROLES:
            episodes = files.json(role + '/EPISODES.json')
            expected = context.profile()['rosters'][role]
            if [e['scene'] for e in episodes] != expected:
                raise PermissionError('Recorded scene roster differs from frozen source')
            for episode in episodes:
                case = episode['scene']['case_id']
                if case in self.reference_actions or len(episode['actions']) > 24:
                    raise PermissionError('Duplicate or unbounded recorded case')
                self.reference_actions[case] = [s['action'] for s in episode['actions']]


class BoundedDataAudit:
    def __init__(self, value, cfg, scenes, rows, budgets, release, key=None):
        if key is not _KEY: raise PermissionError('Only complete raw/source/provider audit creates a data proof')
        self._key = key
        self._value, self._cfg = deepcopy(value), deepcopy(cfg)
        self._scenes, self._rows = deepcopy(scenes), deepcopy(rows)
        self._budgets, self._release = deepcopy(budgets), deepcopy(release)

    def receipt(self): return deepcopy(self._value)
    def cfg(self): return deepcopy(self._cfg)
    def scenes(self, role):
        if role not in ROLES: raise PermissionError('No sealed roster exists in this proof')
        return deepcopy(self._scenes[role])
    def fitting_rows(self): return fitting_rows(self._rows['training'])
    def fitting_draw_plan(self): return deepcopy(self._value['fitting_draw_plan'])
    def native_budget(self, role):
        if role not in ROLES: raise PermissionError('No sealed token records exist in this proof')
        return deepcopy(self._budgets[role])
    def data_release(self): return deepcopy(self._release)


def require_data(value):
    if not isinstance(value, BoundedDataAudit) or value._key is not _KEY:
        raise PermissionError('A serialized PASS/public fixture cannot create a scientific data proof')
    report = value.receipt()
    if (report.get('bounded_behavior_data_admitted') is not True
            or report.get('owned_compute_released') is not True
            or report.get('training_cases') != 108 or report.get('development_cases') != 36
            or report.get('heldout_content_read') is not False
            or report.get('model_training_admitted_by_this_receipt') is not False):
        raise PermissionError('Complete unsealed raw data and provider release are required')
    return report


def read_cpu_runtime(receipt):
    """Independently require the prospective CPU reader software provenance.

    This validates captured version records only; it never probes the auditor's
    host or rewrites the original alias/GPU model-runtime evidence.
    """
    from .composite_bounded_behavior_data_worker import verify_cpu_environment
    return deepcopy(verify_cpu_environment(receipt.get('cpu_runtime')))


@contextmanager
def verified_tokenizer(parent, files=None):
    """Use the alias-parent's original verified metadata, never model weights."""
    pin, proof, encoded = parent.tokenizer_metadata()
    if files is not None:
        for name, value in [('PIN', pin), ('FILES', proof), ('BYTES', encoded)]:
            if digest(files.json('metadata/QWEN_' + name + '.json')) != digest(value):
                raise PermissionError('CPU data tokenizer metadata differs from the original parent')
    from .composite_frozen_probe import verify_metadata_bytes
    verify_metadata_bytes(pin, proof, encoded)
    with tempfile.TemporaryDirectory(prefix='research4-bounded-tokenizer-readonly-') as folder:
        target = Path(folder).resolve()
        for name, value in encoded.items():
            path = (target / name).resolve()
            if not path.is_relative_to(target) or name.endswith('.safetensors'):
                raise PermissionError('Only original tokenizer/config metadata may be reconstructed')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(base64.b64decode(value, validate=True))
        from transformers import AutoTokenizer
        yield AutoTokenizer.from_pretrained(target, local_files_only=True, trust_remote_code=False)


def audit_data_from_source(raw, raw_sha256, root, commit, fit_archive, qualification_archive, closure):
    """Rebuild the opaque alias parent for later qualification/fitting ports."""
    from .composite_source_archive import verify_transport
    root = Path(root)
    verify_transport(root, commit, MODE, MANIFEST)
    descriptor = read(root / 'inputs' / PARENT_DESCRIPTOR)
    with read_parent(root / 'inputs' / PARENT, descriptor['parent_packet_sha256'],
                     fit_archive, qualification_archive) as parent:
        return audit_data(raw, raw_sha256, root, commit, parent, closure)


def audit_data(raw, raw_sha256, root, commit, parent, closure):
    """Replay complete recorded exports and recompute every admission table."""
    from .composite_bounded_behavior_data_admission import require_data_closure
    from .composite_bounded_behavior_runtime import reference_replay_scope
    context = validate_source(root, commit, parent)
    cfg, profile = context.cfg(), context.profile()
    closed = require_data_closure(closure)
    if (closed['source_commit'] != commit or closed['notebook'] != profile['notebook']
            or closed['archive_sha256'] != raw_sha256 or closed['owned_compute_released'] is not True):
        raise PermissionError('Exact source, raw archive and actual provider release required')
    from .composite_bounded_behavior_data_profile import notebook_code
    expected_code = notebook_code(root, commit, sha(Path(root).parent / MANIFEST))
    if closed['actual_code_cells_sha256'] != digest([expected_code]):
        raise PermissionError('Actual submitted CPU notebook differs from the reviewed source')
    files = EvidenceFiles(raw, raw_sha256, BOUNDS['maximum_raw_export_bytes'])
    try:
        receipt = files.json('RUN_RECEIPT.json')
        cpu_runtime = read_cpu_runtime(receipt)
        expected_receipt = {'status': 'complete', 'source_commit': commit, 'stage': MODE,
                            'phase': profile['phase'], 'completed_roles': list(ROLES),
                            'completed_cases': 144, 'model_forwards': 0, 'optimizer_updates': 0,
                            'scientific_GPU_count': 0, 'GPU_required': False,
                            'model_weights_downloaded': False, 'heldout_content_read': False,
                            'public_fixture_test_mode': False, 'counts_exact_at_controlled_exit': True,
                            'worker_exiting': True, 'metadata_cache_removed': True}
        if any(digest(receipt.get(k)) != digest(v) for k, v in expected_receipt.items()):
            raise PermissionError('Only a complete genuine CPU data receipt may be audited')
        expected_freeze = {'source_commit': commit, 'profile': profile, 'study_configuration': cfg,
                           'input_manifest_sha256': sha(Path(root) / 'inputs/MANIFEST.json'),
                           'source_release_sha256': sha(Path(root) / 'RELEASE.json'),
                           'parent_descriptor_sha256': profile['parent_descriptor_sha256']}
        if digest(files.json('PRECALL_FREEZE.json')) != digest(expected_freeze):
            raise PermissionError('Complete source/config/input/parent precall freeze differs')
        expected_owner = {'source_commit': commit, 'version': MODE, 'phase': profile['phase'],
                          'owner_thread_id': profile['owner_thread_id'], 'path': profile['destination']}
        if digest(files.json('OUTPUT_OWNER.json')) != digest(expected_owner):
            raise PermissionError('Raw output directory belongs to another source or owner')
        required = {'PRECALL_FREEZE.json', 'RUN_RECEIPT.json', 'OUTPUT_OWNER.json',
                    'metadata/QWEN_PIN.json', 'metadata/QWEN_FILES.json', 'metadata/QWEN_BYTES.json',
                    'training/FITTING_ROWS.json', 'training/FITTING_DRAW_PLAN.json',
                    'LEAKAGE_AUDIT.json', 'BOUNDED_BEHAVIOR_DATA_RELEASE.json'}
        episodes, rows, audits, budgets = {}, {}, {}, {}
        for role in ROLES:
            required |= {role + '/' + n for n in ['SCENES.json', 'EPISODES.json', 'ROWS.json',
                                                   'COHORT_AUDIT.json', 'NATIVE_BUDGET.json']}
            required |= {role + '/case_exports/' + s['case_id'] + '.json' for s in profile['rosters'][role]}
            if digest(files.json(role + '/SCENES.json')) != digest(profile['rosters'][role]):
                raise PermissionError('A complete frozen role roster was changed')
            episodes[role], rows[role] = files.json(role + '/EPISODES.json'), files.json(role + '/ROWS.json')
            for episode in episodes[role]:
                if digest(files.json(role + '/case_exports/' + episode['scene']['case_id'] + '.json')) != digest(episode):
                    raise PermissionError('Durable per-case evidence differs from final export')
        if set(files.manifest) != required:
            raise PermissionError('Exact complete nonsealed raw export roster required')
        archive = BoundedDataArchive(context, files, closure, _ARCHIVE_KEY)
        with reference_replay_scope(archive):
            for role in ROLES:
                if rows[role] != selected_rows(episodes[role], role):
                    raise PermissionError('Reference row selection/coverage tags changed')
                audits[role] = audit_cohort(episodes[role], rows[role], cfg, role, profile['rosters'][role])
                if digest(audits[role]) != digest(files.json(role + '/COHORT_AUDIT.json')):
                    raise PermissionError('Independent source/action reference audit differs')
        if sum(len(rows[r]) for r in ROLES) > BOUNDS['maximum_reference_states']:
            raise PermissionError('Complete reference state envelope exceeded')
        with verified_tokenizer(parent, files) as tokenizer:
            for role in ROLES:
                budgets[role] = native_budget(rows[role], tokenizer)
                if budgets[role]['status'] != 'pass' or digest(budgets[role]) != digest(files.json(role + '/NATIVE_BUDGET.json')):
                    raise PermissionError('Exact full native budget must pass without truncation or repair')
        leak = leakage_audit(audits['training'], audits['development'], rows['training'], rows['development'])
        if leak['status'] != 'structural_and_exact_leakage_checks_passed' or digest(leak) != digest(files.json('LEAKAGE_AUDIT.json')):
            raise PermissionError('Unresolved leakage/native-role review blocks data admission')
        fit, plan = fitting_rows(rows['training']), training_draw_plan(rows['training'], cfg)
        if (not plan['coverage_passed'] or digest(plan) != digest(files.json('training/FITTING_DRAW_PLAN.json'))
                or digest(fit) != digest(files.json('training/FITTING_ROWS.json'))):
            raise PermissionError('Exact fitting rows,128 draws and target coverage differ')
        release = build_release(profile, cfg, audits, budgets, leak, rows, fit, plan)
        if digest(release) != digest(files.json('BOUNDED_BEHAVIOR_DATA_RELEASE.json')):
            raise PermissionError('Complete independent data release bindings differ')
        report = {'reader': 'bounded-behavior-v4-complete-raw-data-v1', 'source_commit': commit,
                  'raw_archive_sha256': raw_sha256, 'provider_run_id': closed['scriptVersionId'],
                  'source_release_sha256': sha(Path(root) / 'RELEASE.json'),
                  'parent_descriptor_sha256': profile['parent_descriptor_sha256'],
                  'configuration_sha256': digest(cfg), 'bounded_behavior_data_admitted': True,
                  'owned_compute_released': True, 'training_cases': 108, 'development_cases': 36,
                  'eligible_training_states': len(fit), 'release_sha256': digest(release),
                  'fitting_rows_sha256': digest(fit), 'fitting_draw_plan': plan,
                  'reference_audits': audits, 'native_budget_sha256': digest(budgets),
                  'cpu_runtime': cpu_runtime,
                  'model_training_admitted_by_this_receipt': False,
                  'new_model_forwards': 0, 'new_optimizer_updates': 0, 'heldout_content_read': False}
        return BoundedDataAudit(report, cfg, profile['rosters'], rows, budgets, release, _KEY)
    finally:
        files.close()
