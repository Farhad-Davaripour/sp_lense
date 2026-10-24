"""Bind a future frozen task model to actual fit and task-data evidence only.

No loader, provider, optimizer, world collection or held-out access is present.
Receipt dictionaries cannot manufacture either opaque dependency.
"""
from copy import deepcopy
import hashlib
import io
import json
import math
import re
import zipfile
from .io_utils import digest
from .composite_task_skill_training import DATA_ARCHIVE_SHA,DATA_SOURCE,DATA_RELEASE_SHA,read_data
from .composite_task_skill_validation import validate_roster,task_development_plan

_KEY=object()
FIT_SOURCE='6c37f59b60ceff32c74b1c39f0a6b031d207e443'
FIT_RUN=361922334


class TaskDevelopmentInputs:
    def __init__(self,roster,release,key=None):
        if key is not _KEY:raise PermissionError('Read the exact audited task-data archive')
        self._key=key;self._roster=deepcopy(roster);self._release=deepcopy(release)
    def roster(self):return deepcopy(self._roster)
    def release(self):return deepcopy(self._release)


class FrozenTaskPackage:
    def __init__(self,value,key=None):
        if key is not _KEY:raise PermissionError('Bind the actual complete raw fit and data readers')
        self._key=key;self._value=deepcopy(value)
    def receipt(self):return deepcopy(self._value)


def full_fit_identity(full_fit):
    """Bind raw bytes and tensor identities, without CPU-reduction roundoff."""
    from .composite_task_skill_model_audit import require_full_fit
    proof=require_full_fit(full_fit)
    keys=['phase','source_commit','provider_run_id','raw_archive_sha256',
          'source_data_archive_sha256','checkpoint_sha256','initial_adapter_hashes','recorded_forward_counts']
    value={name:deepcopy(proof[name]) for name in keys}
    value['checkpoint']={name:deepcopy(proof['checkpoint'][name]) for name in
                         ['adapter_parameters','tensors','hashes','tensor_shapes']}
    return digest(value)


def verify_audit_reference(original,current):
    """Only double-precision reduction roundoff may differ on the audit CPU."""
    normalized=deepcopy(original)
    try:left=normalized['optimizer']['clipped_global_norm'];right=current['optimizer']['clipped_global_norm']
    except (KeyError,TypeError):raise PermissionError('Complete original/current audit metrics required')
    if (type(left) not in [int,float] or type(right) not in [int,float] or not math.isfinite(left)
            or not math.isfinite(right) or left<0 or right<0
            or not math.isclose(left,right,rel_tol=1e-12,abs_tol=1e-12)):
        raise PermissionError('Recomputed clipped gradient norm differs beyond CPU roundoff')
    normalized['optimizer']['clipped_global_norm']=right
    if digest(normalized)!=digest(current):raise PermissionError('A non-roundoff raw-audit fact changed')
    return True


def read_task_development_inputs(raw):
    # The shared reader verifies the exact complete archive, all member hashes,
    # release/source/roles and every training-row identity. No world is run.
    data=read_data(raw)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        roster=json.loads(z.read('development/EVALUATION_ROSTER.json'))
        release=json.loads(z.read('TASK_DATA_RELEASE.json'))
        if hashlib.sha256(z.read('TASK_DATA_RELEASE.json')).hexdigest()!=DATA_RELEASE_SHA:
            raise PermissionError('Original admitted task data release differs')
    cases=validate_roster(roster);audit=data.release()['audits']['development']
    if ([r['case_id'] for r in cases]!=audit['case_ids'] or sorted({r['group_id'] for r in cases})!=audit['group_ids']
            or set(audit['case_ids'])&set(release['audits']['training']['case_ids'])
            or set(audit['group_ids'])&set(release['audits']['training']['group_ids'])):
        raise PermissionError('Complete disjoint source-bound development roster required')
    return TaskDevelopmentInputs(roster,release,_KEY)


def frozen_task_package(full_fit,data,evaluation_source):
    from .composite_task_skill_model_audit import require_full_fit
    proof=require_full_fit(full_fit)
    if not isinstance(data,TaskDevelopmentInputs) or data._key is not _KEY:
        raise PermissionError('Actual exact task-data reader required')
    if (proof['source_commit']!=FIT_SOURCE or type(proof.get('provider_run_id')) is not int
            or proof['provider_run_id']!=FIT_RUN):
        raise PermissionError('Only the separately registered one-use full fit may enter this endpoint')
    if not isinstance(evaluation_source,str) or not re.fullmatch('[0-9a-f]{40}',evaluation_source):
        raise ValueError('Exact reviewed evaluation source required')
    roster=data.roster();plan=task_development_plan(roster)
    value={'version':'ordinary-task-frozen-package-v1','fit_source_commit':FIT_SOURCE,
      'fit_run_id':FIT_RUN,'evaluation_source_commit':evaluation_source,
      'fit_raw_archive_sha256':proof['raw_archive_sha256'],'fit_raw_audit_canonical_sha256':digest(proof),
      'fit_evidence_identity_sha256':full_fit_identity(full_fit),
      'checkpoint_sha256':proof['checkpoint_sha256'],'checkpoint_tensor_hashes':proof['checkpoint']['hashes'],
      'checkpoint_tensor_shapes':proof['checkpoint']['tensor_shapes'],
      'data_source_commit':DATA_SOURCE,'data_archive_sha256':DATA_ARCHIVE_SHA,
      'data_release_sha256':DATA_RELEASE_SHA,'development_roster':roster,
      'development_roster_sha256':digest(roster),'task_development_plan_sha256':digest(plan),
      'training_reference_core_exclusion':data.release()['audits']['training']['reference_core_fingerprints'],
      'full_fit_completed_and_raw_audited':True,'owned_fit_compute_released':True,
      'native_generation_without_teacher_forcing_required':True,'decoder_repair_or_logit_policy_mask_allowed':False,
      'risk_examples_used_for_fitting_or_selection':False,'adaptive_refitting_from_validation':False,
      'optimizer_updates_during_evaluation':0,'task_development_trajectories':72,
      'separate_risk_trajectories_still_required':180,'evaluation_runtime_qualified_by_this_package':False,
      'allocation_admitted_by_this_package':False,'scientific_validation_admitted':False,
      'heldout_content_read':False,'heldout_opening_admitted':False}
    return FrozenTaskPackage(value,_KEY)


def restore_frozen_task_package(value,original_audit,full_fit,data,evaluation_source):
    """Rebind frozen metadata to a fresh complete raw read, never trust flags."""
    from .composite_task_skill_model_audit import require_full_fit
    actual=require_full_fit(full_fit)
    verify_audit_reference(original_audit,actual)
    expected=frozen_task_package(full_fit,data,evaluation_source).receipt()
    expected['fit_raw_audit_canonical_sha256']=digest(original_audit)
    if digest(value)!=digest(expected):raise PermissionError('Frozen model/data/source/recipe package differs')
    return FrozenTaskPackage(value,_KEY)
