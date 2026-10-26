"""Independent fresh raw/source/release audit for the approved sampling stage.

Only this reader issues ApprovedSamplingDataAudit.  The failed c37 parent stays
a different opaque type and can never satisfy the old BoundedDataAudit guards.
Recomputing actual sampling occurs only after complete fresh raw export and
provider release have been established, never as local pre-experiment tuning.
"""
from copy import deepcopy
from pathlib import Path

from .io_utils import sha, digest
from .composite_v2_model_audit import EvidenceFiles
from .composite_bounded_behavior_data_reader import read_cpu_runtime
from .composite_bounded_sampling_parent import read_failed_parent, require_failed_parent, ROLES
from .composite_bounded_sampling_profile import (MODE, PHASE, MANIFEST, BOUNDS,
    APPROVAL_SHA, PROPOSAL_SHA, OWNER_THREAD, notebook_code)
from .composite_bounded_sampling_rule import RULE_SHA
from .composite_bounded_sampling_data import (canonical_rows, protocol, build_release,
                                              reference_export_names)

_KEY = object()
_ARCHIVE_KEY = object()


class ApprovedSamplingArchive:
    def __init__(self, context, files, closure, raw_sha256, key=None):
        if key is not _ARCHIVE_KEY:
            raise PermissionError('A checked complete fresh raw archive is required before sampler replay')
        self._key, self.context, self.files = key, context, files
        self.closure, self.raw_sha256 = closure, raw_sha256


def require_replay_archive(value):
    if not isinstance(value, ApprovedSamplingArchive) or value._key is not _ARCHIVE_KEY:
        raise PermissionError('Recorded approved-plan replay needs the genuine fresh archive, not JSON')
    from .composite_bounded_sampling_source import require_context
    from .composite_bounded_sampling_admission import require_closure
    require_context(value.context)
    closed = require_closure(value.closure)
    if (closed['archive_sha256'] != value.raw_sha256
            or closed['source_commit'] != value.context.profile()['source_commit']
            or closed['owned_compute_released'] is not True):
        raise PermissionError('Recorded plan replay must follow actual complete raw export and provider release')
    return value


class ApprovedSamplingDataAudit:
    def __init__(self, receipt, cfg, parent, fit, plan, release, key=None):
        if key is not _KEY:
            raise PermissionError('Only complete independent approved-data audit creates fitting lineage')
        self._key = key
        self._receipt, self._cfg = deepcopy(receipt), deepcopy(cfg)
        self._parent, self._fit, self._plan, self._release = parent, deepcopy(fit), deepcopy(plan), deepcopy(release)
    def receipt(self): return deepcopy(self._receipt)
    def cfg(self): return deepcopy(self._cfg)
    def scenes(self, role): return self._parent.scenes(role)
    def fitting_rows(self): return deepcopy(self._fit)
    def fitting_draw_plan(self): return deepcopy(self._plan)
    def native_budget(self, role): return self._parent.native_budget(role)
    def data_release(self): return deepcopy(self._release)
    def protocol(self): return protocol(self._parent)
    def failed_reference(self):
        require_failed_parent(self._parent)
        return self._parent


def require_approved_sampling_data(value):
    if not isinstance(value, ApprovedSamplingDataAudit) or value._key is not _KEY:
        raise PermissionError('Serialized PASS, old data proof or failed references cannot authorize amended fitting')
    r = value.receipt()
    expected = {'approved_sampling_data_admitted': True, 'owned_compute_released': True,
        'approval_sha256': APPROVAL_SHA, 'rule_sha256': RULE_SHA,
        'training_cases': 108, 'development_cases': 36, 'eligible_training_states': 640,
        'draws': 128, 'updates': 64, 'accumulation': 2, 'heldout_content_read': False,
        'new_model_forwards': 0, 'new_optimizer_updates': 0,
        'model_training_admitted_by_this_receipt': False}
    if any(digest(r.get(k)) != digest(v) for k, v in expected.items()):
        raise PermissionError('Complete approved source, all references, exact128 occurrences and released compute required')
    failed = require_failed_parent(value.failed_reference())
    if (r.get('failed_reference_raw_archive_sha256') != failed['raw_archive_sha256']
            or r.get('failed_reference_source_commit') != failed['source_commit']
            or r.get('failed_reference_closeout_sha256') != failed['closeout_sha256']
            or r.get('approved_draw_plan_sha256') != digest(value.fitting_draw_plan())
            or r.get('fitting_rows_sha256') != digest(value.fitting_rows())):
        raise PermissionError('Typed amended data no longer matches its audited failed lineage and plan')
    return r


class _OriginalReferenceView:
    """Map dedicated historical exports to the original reader's exact names."""
    def __init__(self, files): self.files = files
    def json(self, name):
        names = {'training/FITTING_ROWS.json': 'training/ORIGINAL_FITTING_ROWS.json',
                 'training/FITTING_DRAW_PLAN.json': 'training/ORIGINAL_FITTING_DRAW_PLAN.json'}
        return self.files.json(names.get(name, name))


def read_worker_receipt(receipt, commit):
    """Captured exact CPU runtime and lifecycle, without probing the auditor."""
    cpu_runtime = read_cpu_runtime(receipt)
    expected = {'status': 'complete', 'source_commit': commit, 'stage': MODE, 'mode': MODE,
        'phase': PHASE, 'completed_roles': list(ROLES), 'completed_cases': 144,
        'model_forwards': 0, 'optimizer_updates': 0, 'scientific_GPU_count': 0,
        'GPU_required': False, 'model_weights_downloaded': False, 'heldout_content_read': False,
        'public_fixture_test_mode': False, 'counts_exact_at_controlled_exit': True,
        'worker_exiting': True, 'metadata_cache_created': False, 'metadata_cache_removed': True,
        'new_teacher_or_case_generation': False, 'scientific_dataset_admitted': False,
        'original_failure_preserved': True, 'training_launch_admitted': False,
        'approval_sha256': APPROVAL_SHA, 'proposal_sha256': PROPOSAL_SHA,
        'owner_thread_id': OWNER_THREAD, 'automatic_scientific_retries': 0}
    if any(digest(receipt.get(k)) != digest(v) for k, v in expected.items()):
        raise PermissionError('Only a complete genuine fresh CPU replay/sampling worker can be audited')
    return cpu_runtime


def audit_data_from_source(raw, raw_sha256, root, commit, alias_fit_archive=None,
                           alias_qual_archive=None, closure=None, **kwargs):
    """Rebuild the actual failed reference ancestor from the new source inputs."""
    # The long form mirrors the legacy transport helper without accepting a
    # serialized data proof.  A single explicit alias spelling is supported for
    # existing fit bridge callers.
    if 'alias_qualification_archive' in kwargs:
        if alias_qual_archive is not None: raise TypeError('Duplicate qualification raw argument')
        alias_qual_archive = kwargs.pop('alias_qualification_archive')
    if kwargs: raise TypeError('Unknown approved-data reader arguments: ' + ','.join(kwargs))
    from .composite_source_archive import verify_transport
    verify_transport(root, commit, MODE, MANIFEST)
    parent = read_failed_parent(root, alias_fit_archive=alias_fit_archive,
                                alias_qual_archive=alias_qual_archive)
    return audit_data(raw, raw_sha256, root, commit, parent, closure)


def audit_data(raw, raw_sha256, root, commit, parent, closure):
    from .composite_bounded_sampling_source import validate_source
    from .composite_bounded_sampling_admission import require_closure
    from .composite_bounded_sampling_rule import verify_draw_plan
    context = validate_source(root, commit, parent)
    cfg, profile = context.cfg(), context.profile()
    closed = require_closure(closure)
    if (closed['source_commit'] != commit or closed['notebook'] != profile['notebook']
            or closed['phase'] != PHASE or closed['archive_sha256'] != raw_sha256
            or closed['owned_compute_released'] is not True):
        raise PermissionError('Exact fresh amended source, archive and target provider release required')
    expected_code = notebook_code(root, commit, sha(Path(root).parent / MANIFEST))
    if closed['actual_code_cells_sha256'] != digest([expected_code]):
        raise PermissionError('Saved actual sampling notebook differs from exact reviewed source')
    files = EvidenceFiles(raw, raw_sha256, BOUNDS['maximum_raw_export_bytes'])
    try:
        required = reference_export_names(parent) | {'PRECALL_FREEZE.json', 'OUTPUT_OWNER.json', 'RUN_RECEIPT.json'}
        if set(files.manifest) != required:
            raise PermissionError('Complete exact fresh amended raw exports required; no missing/extra cases')
        receipt = files.json('RUN_RECEIPT.json')
        cpu_runtime = read_worker_receipt(receipt, commit)
        freeze = {'source_commit': commit, 'profile': profile, 'study_configuration': cfg,
            'input_manifest_sha256': sha(Path(root) / 'inputs/MANIFEST.json'),
            'source_release_sha256': sha(Path(root) / 'RELEASE.json'),
            'failed_reference_receipt_sha256': digest(parent.receipt()),
            'approval_sha256': APPROVAL_SHA, 'approved_rule_sha256': RULE_SHA}
        if files.json('PRECALL_FREEZE.json') != freeze:
            raise PermissionError('Fresh approved sampling pre-call source/input/failure/approval freeze differs')
        owner = {'source_commit': commit, 'version': MODE, 'phase': PHASE,
            'owner_thread_id': profile['owner_thread_id'], 'path': profile['destination']}
        if files.json('OUTPUT_OWNER.json') != owner: raise PermissionError('Fresh raw output ownership differs')
        if files.json('FAILED_REFERENCE_PROTOCOL.json') != protocol(parent):
            raise PermissionError('Failed uniform experiment and explicitly changed distribution must remain disclosed')
        # The old source replays the fresh copies of its recorded references,
        # including source-derived selector/request tags, permissions, external
        # stops, full native budget, role isolation, and leakage checks.
        result = parent.replay(_OriginalReferenceView(files))
        fit = canonical_rows(parent)
        if files.json('training/FITTING_ROWS.json') != fit:
            raise PermissionError('All640 canonical fitting rows must remain unchanged and unique')
        archive = ApprovedSamplingArchive(context, files, closure, raw_sha256, _ARCHIVE_KEY)
        plan = verify_draw_plan(parent.fitting_rows(), cfg, files.json('training/FITTING_DRAW_PLAN.json'), archive)
        if (plan.get('coverage_passed') is not True or plan.get('public_fixture_test_mode') is not False
                or len(plan.get('state_ids', [])) != 128):
            raise PermissionError('Actual approved complete128 occurrences must pass every unchanged coverage criterion')
        release = build_release(profile, cfg, parent, result['audits'], result['budgets'],
                                result['leakage'], fit, plan)
        if files.json('APPROVED_SAMPLING_DATA_RELEASE.json') != release:
            raise PermissionError('Independent fresh approved-data release bindings differ')
        failed = require_failed_parent(parent)
        report = {'reader': 'bounded-approved-sampling-amendment-v1-raw-reader',
            'source_commit': commit, 'raw_archive_sha256': raw_sha256,
            'source_release_sha256': sha(Path(root) / 'RELEASE.json'),
            'input_manifest_sha256': sha(Path(root) / 'inputs/MANIFEST.json'),
            'source_manifest_sha256': sha(Path(root).parent / MANIFEST),
            'provider_run_id': closed['scriptVersionId'], 'owned_compute_released': True,
            'approval_sha256': APPROVAL_SHA, 'rule_sha256': RULE_SHA,
            'approved_draw_plan_sha256': digest(plan), 'fitting_rows_sha256': digest(fit),
            'failed_reference_raw_archive_sha256': failed['raw_archive_sha256'],
            'failed_reference_source_commit': failed['source_commit'],
            'failed_reference_closeout_sha256': failed['closeout_sha256'],
            'failed_reference_receipt_sha256': digest(failed),
            'original_failed_draw_plan_sha256': digest(parent.original_draw_plan()),
            'training_cases': 108, 'development_cases': 36, 'eligible_training_states': 640,
            'draws': 128, 'updates': 64, 'accumulation': 2, 'configuration_sha256': digest(cfg),
            'release_sha256': digest(release), 'protocol_sha256': digest(protocol(parent)),
            'reference_audits': result['audits'], 'native_budget_sha256': digest(result['budgets']),
            'cpu_runtime': cpu_runtime, 'approved_sampling_data_admitted': True,
            'original_failed_dataset_remains_unadmitted': True, 'sampling_distribution_changed': True,
            'model_training_admitted_by_this_receipt': False, 'new_model_forwards': 0,
            'new_optimizer_updates': 0, 'heldout_content_read': False}
        proof = ApprovedSamplingDataAudit(report, cfg, parent, fit, plan, release, _KEY)
        require_approved_sampling_data(proof)
        return proof
    finally: files.close()
