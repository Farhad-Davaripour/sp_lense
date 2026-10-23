"""Shared one-attempt journal; no cloud submission or human-authority creation."""
from datetime import datetime,timezone
from pathlib import Path
import copy
import json
import os
import uuid
from .io_utils import read

REQUIRED_SURFACES={'free_tier','azure','centralus','kaggle'}
OWNER='human-authorized-Kaggle-recovery-admission'  # Preserve existing guard compatibility.


def atomic(path,value,token):
    temp=path.with_name(path.name+'.'+token+'.tmp')
    with temp.open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def begin_reconciliation(campaign_root):
    root=Path(campaign_root);token=uuid.uuid4().hex;at=datetime.now(timezone.utc).isoformat()
    with (root/'COMPUTE_SUBMISSION.lock').open('x') as stream:
        json.dump({'owner':OWNER,'token':token,'acquired_at_UTC':at},stream)
    return {'guard_token':token,'acquired_at_UTC':at}


def guard(root,token):
    if token is None:token=begin_reconciliation(root)['guard_token']
    path=root/'COMPUTE_SUBMISSION.lock';value=read(path)
    if value.get('token')!=token or value.get('owner')!=OWNER:
        raise PermissionError('Another owner holds the shared guard')
    return path,token,datetime.fromisoformat(value['acquired_at_UTC'])


def reserve(campaign_root,preparation_file,live_snapshot,candidate_key,attempts_key,prefix,validator,candidate_rule,guard_token=None):
    root=Path(campaign_root);lock,token,acquired=guard(root,guard_token);retained=False
    try:
        binding=validator(preparation_file)
        snapshot=live_snapshot() if callable(live_snapshot) else live_snapshot
        now=datetime.now(timezone.utc);observed=datetime.fromisoformat(snapshot['at_UTC'].replace('Z','+00:00'))
        if observed.tzinfo is None or observed<acquired or not 0<=(now-observed).total_seconds()<=120:
            raise PermissionError('Fresh live observations are required under the shared guard')
        surfaces=snapshot['surfaces']
        if set(surfaces)!=REQUIRED_SURFACES or any(type(value) is not int or value!=0 for value in surfaces.values()):
            raise PermissionError('An active, pending, missing or unknown workspace blocks admission')
        path=root/'SCIENTIFIC_EXECUTION_LEDGER.json';ledger=read(path)
        if ledger.get('active_owned_submission') is not None:
            raise PermissionError('An existing owned submission requires reconciliation')
        if any(value.get('owned_compute_released') is not True and value.get('status') not in ['prepared','terminal_pre_execution_rejection'] for value in ledger['candidates'].values()):
            raise PermissionError('Another owned or ambiguous candidate blocks admission')
        candidate=candidate_rule(copy.deepcopy(ledger['candidates'].get(candidate_key)),binding)
        if candidate.get(attempts_key):
            raise PermissionError('The one attempt is consumed; reconcile its existing identity')
        identity=prefix+binding['authorization_sha256'][:24]
        attempt={**binding,'identity':identity,'status':'reserved_pending_submission',
                 'reserved_at_UTC':now.isoformat(),'authority_consumed':True,'scientific_attempts_authorized':1,
                 'automatic_scientific_retries':0,'scriptVersionId':None,'live_reconciliation':snapshot}
        candidate[attempts_key]=[attempt]
        if binding.get('fresh_stage'):
            candidate.update(status='reserved_pending_submission',owned_compute_released=None)
        ledger['candidates'][candidate_key]=candidate;ledger['updated_at_UTC']=now.isoformat()
        ledger['active_owned_submission']={'identity':identity,'candidate':candidate_key,'platform':'Kaggle',
                                           'status':attempt['status'],'scriptVersionId':None}
        atomic(path,ledger,token);retained=True
        return {**attempt,'guard_token':token,'guard_retained_through_submission':True}
    finally:
        if not retained and lock.exists() and read(lock).get('token')==token:lock.unlink()


def reconcile(campaign_root,identity,candidate_key,attempts_key,script_version_id=None,guard_token=None):
    root=Path(campaign_root);lock,token,_=guard(root,guard_token)
    try:
        path=root/'SCIENTIFIC_EXECUTION_LEDGER.json';ledger=read(path)
        active=ledger.get('active_owned_submission')
        if active is not None and (active.get('identity')!=identity or active.get('candidate')!=candidate_key):
            raise PermissionError('Cannot replace another active submission identity')
        matches=[value for value in ledger['candidates'][candidate_key].get(attempts_key,[]) if value['identity']==identity]
        if len(matches)!=1 or matches[0]['authority_consumed'] is not True:raise ValueError('Unknown submission reservation')
        attempt=matches[0]
        if script_version_id is None:
            if attempt['scriptVersionId'] is not None:raise ValueError('Known submission cannot become unknown')
            attempt['status']='submission_outcome_unknown'
        else:
            if type(script_version_id) is not int or script_version_id<=0:raise ValueError('An exact positive Kaggle version is required')
            if attempt['scriptVersionId'] not in [None,script_version_id]:raise PermissionError('Never bind a second copy to one authorization')
            attempt.update(status='submitted_exact_version_reconciled',scriptVersionId=script_version_id)
        attempt['reconciled_at_UTC']=datetime.now(timezone.utc).isoformat()
        ledger['active_owned_submission']={'identity':identity,'candidate':candidate_key,'platform':'Kaggle',
                                           'status':attempt['status'],'scriptVersionId':attempt['scriptVersionId']}
        atomic(path,ledger,token);return attempt
    finally:
        if lock.exists() and read(lock).get('token')==token:lock.unlink()
