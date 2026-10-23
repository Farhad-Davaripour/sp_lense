"""Consume one human-authorized recovery reservation before external submit.

This helper performs local ledger operations only. It cannot submit compute,
create authority, or refresh the required live cross-workspace observations.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import uuid
from .io_utils import read, sha, verify_release
from .composite_recovery import execution_config

CANDIDATE = 'detector_rank32_v5'
REQUIRED_SURFACES = {'free_tier', 'azure', 'centralus', 'kaggle'}


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


def _atomic(path, value, token):
    temp = path.with_name(path.name + '.' + token + '.tmp')
    with temp.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def begin_reconciliation(campaign_root):
    """Hold the shared guard while the operator reads live CLI/browser state."""
    root = Path(campaign_root); token = uuid.uuid4().hex
    at = datetime.now(timezone.utc).isoformat()
    with (root / 'COMPUTE_SUBMISSION.lock').open('x') as stream:
        json.dump({'owner': 'human-authorized-Kaggle-recovery-admission', 'token': token,
                   'acquired_at_UTC': at}, stream)
    return {'guard_token': token, 'acquired_at_UTC': at}


def _guard(root, guard_token):
    if guard_token is None:
        guard_token = begin_reconciliation(root)['guard_token']
    lock = root / 'COMPUTE_SUBMISSION.lock'; value = read(lock)
    if value.get('token') != guard_token or value.get('owner') != 'human-authorized-Kaggle-recovery-admission':
        raise PermissionError('Another owner holds the shared guard')
    return lock, guard_token, datetime.fromisoformat(value['acquired_at_UTC'])


def reserve_once(campaign_root, preparation_file, live_snapshot, guard_token=None):
    """Return one reserved identity, consuming the exception before UI/API use."""
    root = Path(campaign_root)
    lock, token, acquired = _guard(root, guard_token)
    retained = False
    try:
        binding = validate_preparation(preparation_file)
        live_snapshot = live_snapshot() if callable(live_snapshot) else live_snapshot
        now = datetime.now(timezone.utc)
        observed = datetime.fromisoformat(live_snapshot['at_UTC'].replace('Z', '+00:00'))
        if observed.tzinfo is None or observed < acquired or not 0 <= (now-observed).total_seconds() <= 120:
            raise PermissionError('Fresh live observations are required under the shared guard')
        surfaces = live_snapshot['surfaces']
        if set(surfaces) != REQUIRED_SURFACES or any(type(value) is not int or value != 0 for value in surfaces.values()):
            raise PermissionError('An active, pending, missing or unknown workspace blocks recovery')
        ledger_file = root / 'SCIENTIFIC_EXECUTION_LEDGER.json'
        ledger = read(ledger_file)
        if ledger.get('active_owned_submission') is not None:
            raise PermissionError('An existing owned submission requires reconciliation')
        if any(value.get('owned_compute_released') is not True and value.get('status') not in ['prepared', 'terminal_pre_execution_rejection']
               for value in ledger['candidates'].values()):
            raise PermissionError('Another owned or ambiguous candidate blocks admission')
        candidate = ledger['candidates'][CANDIDATE]
        if str(candidate['scriptVersionId']) != binding['parent_run_id']:
            raise PermissionError('Preserved parent run differs')
        if candidate['status'] != 'scientific_timeout_partial_checkpoint_audited' or candidate.get('owned_compute_released') is not True or candidate.get('raw_audit_pending') is not False:
            raise PermissionError('The scientific parent is not audited and released')
        attempts = candidate.setdefault('authorized_recovery_attempts', [])
        if attempts:
            raise PermissionError('The one recovery attempt is consumed; reconcile its existing identity')
        identity = 'r4-kaggle-recovery-' + binding['authorization_sha256'][:24]
        attempt = {**binding, 'identity': identity, 'status': 'reserved_pending_submission',
                   'reserved_at_UTC': now.isoformat(), 'authority_consumed': True,
                   'scientific_attempts_authorized': 1, 'automatic_scientific_retries': 0,
                   'scriptVersionId': None, 'live_reconciliation': live_snapshot}
        attempts.append(attempt)
        ledger['updated_at_UTC'] = now.isoformat()
        ledger['active_owned_submission'] = {'identity': identity, 'candidate': CANDIDATE,
                                             'platform': 'Kaggle', 'status': 'reserved_pending_submission',
                                             'scriptVersionId': None}
        _atomic(ledger_file, ledger, token)
        retained = True
        return {**attempt, 'guard_token': token, 'guard_retained_through_submission': True}
    finally:
        if not retained and lock.exists() and read(lock).get('token') == token:
            lock.unlink()


def reconcile_submission(campaign_root, identity, script_version_id=None, guard_token=None):
    """Bind the SAME consumed identity; None records an ambiguous outcome."""
    root = Path(campaign_root)
    lock, token, _ = _guard(root, guard_token)
    try:
        path = root / 'SCIENTIFIC_EXECUTION_LEDGER.json'; ledger = read(path)
        matches = [value for value in ledger['candidates'][CANDIDATE].get('authorized_recovery_attempts', [])
                   if value['identity'] == identity]
        if len(matches) != 1 or matches[0]['authority_consumed'] is not True:
            raise ValueError('Unknown recovery reservation')
        attempt = matches[0]
        if script_version_id is None:
            if attempt['scriptVersionId'] is not None:
                raise ValueError('Known submission cannot be changed back to unknown')
            attempt['status'] = 'submission_outcome_unknown'
        else:
            if type(script_version_id) is not int or script_version_id <= 0:
                raise ValueError('An exact positive Kaggle script version ID is required')
            if attempt['scriptVersionId'] not in [None, script_version_id]:
                raise PermissionError('Never bind a second notebook copy to one authorization')
            attempt.update(status='submitted_exact_version_reconciled', scriptVersionId=script_version_id)
        attempt['reconciled_at_UTC'] = datetime.now(timezone.utc).isoformat()
        ledger['active_owned_submission'] = {'identity': identity, 'candidate': CANDIDATE,
                                             'platform': 'Kaggle', 'status': attempt['status'],
                                             'scriptVersionId': attempt['scriptVersionId']}
        _atomic(path, ledger, token)
        return attempt
    finally:
        if lock.exists() and read(lock).get('token') == token:
            lock.unlink()
