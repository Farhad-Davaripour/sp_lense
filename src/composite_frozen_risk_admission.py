"""One-use frozen risk/transfer admission; no scientific submission or retry."""
from pathlib import Path
from .io_utils import read,sha,digest
from .composite_submission_admission import reserve,reconcile,guard
from .composite_kaggle_v2_admission import candidate_rule
from .composite_kaggle_v2_profile import HUMAN_SHA
from .composite_task_skill_profile import SELECTION_SHA
from .composite_frozen_risk_profile import validate_source,MODE,STAGE
from .composite_frozen_risk_audit import require_stage

ATTEMPTS='risk_attempts'


def candidate_key(binding):
    if binding['phase']=='risk_detector_transfer':return 'frozen_risk_transfer_'+binding['risk_data_release_sha256']
    units=sorted((u['case_id'],u['arm']) for u in binding['units'])
    if not units or len(units)!=len(set(units)):raise PermissionError('Exact unique risk case/control units required')
    return 'frozen_risk_units_'+digest(units)


def validate_preparation(path):
    path=Path(path);value=read(path);root=path.parent/'bundle/source'
    if value.get('status')!='prepared_frozen_risk_stage_not_submitted' or value.get('mode')!=MODE:
        raise PermissionError('Exact frozen risk stage preparation required')
    context=validate_source(root,value['source_commit'],value['fit_archive_local_path'],
                            value['qualification_archive_local_path'],value['risk_archive_local_path'])
    stage=context.stage()
    for file,key in [('FROZEN_RISK_STAGE_PACKAGE.zip','bundle_sha256'),
      ('bundle/TASK_TRAINING_PACKAGE_FILES.json','bundle_manifest_sha256'),('research4_frozen_risk_stage.ipynb','notebook_file_sha256'),
      ('bundle/source/inputs/'+STAGE,'stage_sha256'),('bundle/source/inputs/MANIFEST.json','input_manifest_sha256')]:
        if sha(path.parent/file)!=value[key]:raise PermissionError('Prepared frozen risk source/input/code changed')
    units=[] if stage['partition'] is None else stage['partition']['units']
    expected={'phase':stage['phase'],'notebook':stage['notebook'],'destination':stage['destination'],
      'partition_index':None if stage['partition'] is None else stage['partition']['index'],'units':units,
      'checkpoint_sha256':stage['checkpoint_sha256'],'risk_data_release_sha256':stage['risk_data_release_sha256'],
      'detector_checkpoint_sha256':stage['detector_checkpoint_sha256'],'scientific_GPU_count':stage['scientific_GPU_count']}
    if any(digest(value.get(k))!=digest(v) for k,v in expected.items()):raise PermissionError('Risk stage case/control/model/phase identity differs')
    cells=[''.join(c['source']) for c in read(path.parent/'research4_frozen_risk_stage.ipynb')['cells'] if c['cell_type']=='code']
    code=(root/'notebooks/composite_task_skill_qualification.py').read_text(encoding='utf-8-sig').replace(
      'SOURCE_COMMIT_PLACEHOLDER',stage['source_commit']).replace('PACKAGE_MANIFEST_SHA256_PLACEHOLDER',value['bundle_manifest_sha256'])
    if cells!=[code]:raise PermissionError('Exact prepared risk notebook source/input code required')
    return {**expected,'mode':MODE,'source_commit':stage['source_commit'],
      'frozen_model_package_sha256':stage['frozen_model_package_sha256'],
      'authorization_sha256':value['stage_sha256'],'package_sha256':value['bundle_sha256'],
      'provider_scope':'human-selected-Kaggle-remainder-v1','kaggle_remainder_scope_sha256':HUMAN_SHA,
      'scope_A_selection_sha256':SELECTION_SHA,'fresh_stage':True,'optimizer_updates':0,
      'hard_timeout_seconds':stage['hard_timeout_seconds'],'automatic_scientific_retries':0,'heldout_content_read':False}


def _dependency(value,binding,phase,ledger):
    proof=require_stage(value,phase)
    for key in ['source_commit','frozen_model_package_sha256','risk_data_release_sha256','checkpoint_sha256']:
        if proof[key]!=binding[key]:raise PermissionError('Frozen stage dependency belongs to another exact freeze')
    if phase=='risk_development' and (proof['partition_index']!=0 or proof['completed_trajectories']!=5):
        raise PermissionError('The complete original first five-arm case is required before grouped risk work')
    found=False
    for candidate in ledger['candidates'].values():
        if candidate.get('owned_compute_released') is not True:continue
        for attempt in candidate.get(ATTEMPTS,[]):
            if attempt.get('scriptVersionId')==proof['provider_run_id'] and attempt.get('source_commit')==proof['source_commit']:
                found=True
    if not found:raise PermissionError('Exact dependency submission must be reconciled and released in the shared ledger')
    return proof


def reserve_stage(campaign_root,preparation_file,live_snapshot,guard_token=None,*,transfer_audit=None,first_risk_audit=None):
    root=Path(campaign_root);lock,token,_=guard(root,guard_token)
    try:
        ledger=read(root/'SCIENTIFIC_EXECUTION_LEDGER.json')
        if ledger.get('active_owned_submission') is not None:raise PermissionError('Observe the exact existing worker; never duplicate it')
        binding=validate_preparation(preparation_file)
        if binding['phase']=='risk_development':
            _dependency(transfer_audit,binding,'risk_detector_transfer',ledger)
            if binding['partition_index']>0:_dependency(first_risk_audit,binding,'risk_development',ledger)
        current={(u['case_id'],u['arm']) for u in binding['units']}
        for candidate in ledger['candidates'].values():
            for attempt in candidate.get(ATTEMPTS,[]):
                if binding['phase']=='risk_detector_transfer':
                    if attempt.get('phase')=='risk_detector_transfer' and attempt.get('risk_data_release_sha256')==binding['risk_data_release_sha256']:
                        raise PermissionError('Preserve the already consumed fixed-transfer identity')
                elif current&{(u['case_id'],u['arm']) for u in attempt.get('units',[])}:
                    raise PermissionError('At least one risk case/control unit is already consumed')
        key=candidate_key(binding)
    except Exception:
        if lock.exists() and read(lock).get('token')==token:lock.unlink()
        raise
    return reserve(root,preparation_file,live_snapshot,key,ATTEMPTS,'r4-frozen-risk-',lambda _:binding,candidate_rule,token)


def reconcile_stage(campaign_root,identity,binding,script_version_id=None,guard_token=None):
    return reconcile(campaign_root,identity,candidate_key(binding),ATTEMPTS,script_version_id,guard_token)
