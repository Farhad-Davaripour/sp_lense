"""Distinct one-use operational identities; failed original identities stay closed."""
from pathlib import Path
from .io_utils import read,sha
from . import composite_submission_admission as shared
from . import composite_frozen_base_admission as original
from .composite_frozen_operational import (PROFILE,DOCUMENT,PARENT_PREFIX_AUDIT_SHA256,
    PARENT_CLOSEOUT_SHA256,validate_parent_inputs)
from .composite_frozen_continuation import ORIGINAL_SOURCE,PARENT_RUN,PARENT_COMMON_FREEZE

PREFIX='frozen_base_allocator_v2_p'
ATTEMPTS='authorized_operational_partition_attempts'
begin_reconciliation=shared.begin_reconciliation


def validate_preparation(file):
    binding=original.validate_preparation(file)
    source=Path(file).parent/'bundle/source';op=read(source/'inputs'/DOCUMENT)
    if (op['profile']!=PROFILE or binding['source_commit']==ORIGINAL_SOURCE
        or read(source/'inputs/EXECUTION_PLATFORM.json').get('operational_profile')!=PROFILE):
        raise PermissionError('A new explicit operational profile/source is required')
    validate_parent_inputs(source/'inputs')
    binding.update(operational_profile=PROFILE,source_relationship=op['source_relationship'],
        original_scientific_source_commit=ORIGINAL_SOURCE,original_common_freeze_sha256=PARENT_COMMON_FREEZE,
        parent_run_id=PARENT_RUN,parent_closeout_sha256=PARENT_CLOSEOUT_SHA256,
        parent_prefix_audit_sha256=PARENT_PREFIX_AUDIT_SHA256,operational_document_sha256=sha(source/'inputs'/DOCUMENT),
        tool_boundary_continuation=op['tool_boundary_continuation'],
        parent_sealed_forward_counts={'attempted':1260,'completed':1260},
        parent_interrupted_response_forwards_retained=1,exact_interrupted_generation_resume=False,
        completed_parent_native_responses_regenerated=0)
    return binding


def verify_campaign_under_guard(campaign_root,binding):
    root=Path(campaign_root);ledger=read(root/'SCIENTIFIC_EXECUTION_LEDGER.json')
    number=binding['partition']
    if type(number) is not int or number not in [3,4,5]:raise PermissionError('Only remaining original BASE partitions are eligible')
    for prior in range(3):
        record=ledger['candidates'].get(original.PREFIX+str(prior).zfill(2),{})
        if record.get('complete_base_partition_audited') is not True or record.get('owned_compute_released') is not True:
            raise PermissionError('The first18 original BASE scenes must remain audited and released')
        attempt=record[original.ATTEMPTS][0]
        if attempt['source_commit']!=ORIGINAL_SOURCE or attempt['common_freeze_sha256']!=PARENT_COMMON_FREEZE:
            raise PermissionError('Original completed-source lineage changed')
    parent=ledger['candidates'].get(original.PREFIX+'03',{})
    if (parent.get('current_run_id')!=PARENT_RUN or parent.get('scientific_execution_started') is not True
        or parent.get('owned_compute_released') is not True or parent.get('owned_cache_removed') is not True
        or parent.get('forward_counts')!={'attempted':1260,'completed':1260}
        or parent.get('complete_base_partition_audited') is not False):
        raise PermissionError('The consumed failed original parent is not completely reconciled')
    closeout=Path(parent['closeout_receipt']).resolve()
    if not closeout.is_relative_to(root.resolve()) or sha(closeout)!=PARENT_CLOSEOUT_SHA256:
        raise PermissionError('The original failed-parent closeout changed')
    for key,record in ledger['candidates'].items():
        if key.startswith(PREFIX):
            for attempt in record.get(ATTEMPTS,[]):
                if attempt['common_freeze_sha256']!=binding['common_freeze_sha256']:
                    raise PermissionError('New operational partitions must share their disclosed new execution freeze')
        if key.startswith(original.PREFIX) and key!=original.PREFIX+'03':
            for attempt in record.get(original.ATTEMPTS,[]):
                if set(attempt.get('case_ids',[])) & set(binding['case_ids']):
                    raise PermissionError('Original or ambiguous prior case execution cannot be duplicated')
    for previous in range(3,number):
        prior=ledger['candidates'].get(PREFIX+str(previous).zfill(2),{})
        if prior.get('complete_base_partition_audited') is not True or prior.get('owned_compute_released') is not True:
            raise PermissionError('The preceding fixed operational partition must be audited and released')


def candidate_rule(candidate,binding):
    if candidate is None:
        return {'status':'prepared','mode':binding['mode'],'source_commit':binding['source_commit'],
            'partition':binding['partition'],'scientific_execution_started':False,'owned_compute_released':True,
            'automatic_scientific_retries':0,'heldout_opened':False,'operational_profile':PROFILE}
    if (candidate.get('status')!='prepared' or candidate.get('scientific_execution_started') is not False
        or candidate.get('source_commit')!=binding['source_commit'] or candidate.get('partition')!=binding['partition']):
        raise PermissionError('A consumed or differently frozen operational identity cannot be reset')
    return candidate


def reserve_once(campaign_root,preparation_file,live_snapshot,guard_token=None):
    initial=validate_preparation(preparation_file);key=PREFIX+str(initial['partition']).zfill(2)
    def checked(file):
        binding=validate_preparation(file)
        if binding!=initial:raise ValueError('Operational preparation changed during reconciliation')
        verify_campaign_under_guard(campaign_root,binding)
        return binding
    return shared.reserve(campaign_root,preparation_file,live_snapshot,key,ATTEMPTS,
        'r4-kaggle-cache-v2-p'+str(initial['partition']).zfill(2)+'-',checked,candidate_rule,guard_token)


def reconcile_submission(campaign_root,identity,script_version_id=None,guard_token=None):
    ledger=read(Path(campaign_root)/'SCIENTIFIC_EXECUTION_LEDGER.json')
    keys=[key for key,record in ledger['candidates'].items() if key.startswith(PREFIX)
          and any(attempt['identity']==identity for attempt in record.get(ATTEMPTS,[]))]
    if len(keys)!=1:raise PermissionError('Unknown operational identity')
    return shared.reconcile(campaign_root,identity,keys[0],ATTEMPTS,script_version_id,guard_token)
