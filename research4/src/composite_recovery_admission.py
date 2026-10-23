"""Consume one human-authorized recovery reservation before external submit.

This helper performs local ledger operations only. It cannot submit compute,
create authority, or refresh the required live cross-workspace observations.
"""
from pathlib import Path
import os
from .io_utils import read, sha, verify_release
from .composite_recovery import execution_config
from . import composite_submission_admission as shared

CANDIDATE = 'detector_rank32_v5'
REQUIRED_SURFACES = shared.REQUIRED_SURFACES


def validate_preparation(preparation_file):
    preparation_file = Path(preparation_file)
    prep = read(preparation_file)
    if prep.get('status') != 'prepared_with_explicit_exception_not_submitted':
        raise PermissionError('An inert proposal does not authorize a recovery reservation')
    source = preparation_file.parent / 'bundle/source'
    verify_release(source, prep['source_commit'])
    for name, expected in read(source / 'inputs/MANIFEST.json').items():
        if Path(name).name != name or sha(source / 'inputs' / name) != expected:
            raise ValueError('Recovery input freeze differs')
    package = preparation_file.parent / 'FROZEN_PACKAGE.zip'
    if sha(package) != prep['package_sha256']:
        raise ValueError('Recovery package changed')
    previous = os.environ.get('RESEARCH4_EXECUTION_PLATFORM')
    try:
        os.environ['RESEARCH4_EXECUTION_PLATFORM'] = 'kaggle'
        execution_config(source, prep['source_commit'], 'detector-rank32-v5',
                         read(source / 'config/composite_controller_rank32_v2.json'))
    finally:
        if previous is None:
            os.environ.pop('RESEARCH4_EXECUTION_PLATFORM', None)
        else:
            os.environ['RESEARCH4_EXECUTION_PLATFORM'] = previous
    authorization_file = source / 'inputs/RECOVERY_AUTHORIZATION.json'
    authorization = read(authorization_file)
    return {'source_commit': prep['source_commit'], 'notebook': prep['notebook'],
            'package_sha256': prep['package_sha256'],
            'parent_run_id': str(prep['recovery']['parent_run_id']),
            'authorization_sha256': sha(authorization_file),
            'human_message_evidence_locator': authorization['human_message_evidence_locator']}


begin_reconciliation=shared.begin_reconciliation
_atomic=shared.atomic
_guard=shared.guard


def _recovery_candidate(candidate,binding):
    if candidate is None or str(candidate['scriptVersionId'])!=binding['parent_run_id']:
        raise PermissionError('Preserved parent run differs')
    if candidate['status']!='scientific_timeout_partial_checkpoint_audited' or candidate.get('owned_compute_released') is not True or candidate.get('raw_audit_pending') is not False:
        raise PermissionError('The scientific parent is not audited and released')
    return candidate


def reserve_once(campaign_root,preparation_file,live_snapshot,guard_token=None):
    return shared.reserve(campaign_root,preparation_file,live_snapshot,CANDIDATE,'authorized_recovery_attempts',
                          'r4-kaggle-recovery-',validate_preparation,_recovery_candidate,guard_token)


def reconcile_submission(campaign_root,identity,script_version_id=None,guard_token=None):
    return shared.reconcile(campaign_root,identity,CANDIDATE,'authorized_recovery_attempts',script_version_id,guard_token)
