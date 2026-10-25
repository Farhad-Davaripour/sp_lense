"""Single CPU reference-data reservation; never submit, model-fit or retry."""
from pathlib import Path
from .io_utils import read,sha,digest
from .composite_frozen_risk_data_profile import validate_source,PROFILE,MODE,HUMAN_SHA,SELECTION_SHA
from .composite_submission_admission import reserve,reconcile,guard
from .composite_kaggle_v2_admission import candidate_rule

CANDIDATE='option_a_frozen_risk_development_data_v1'
ATTEMPTS='data_attempts'


def validate_preparation(path):
    path=Path(path);value=read(path);root=path.parent/'bundle/source'
    context=validate_source(root,value['source_commit']);profile=context.profile()
    if value.get('status')!='prepared_CPU_risk_development_reference_not_submitted' or value.get('mode')!=MODE:
        raise PermissionError('Exact development-only reference preparation required')
    for file,key in [('FROZEN_RISK_DATA_PACKAGE.zip','bundle_sha256'),('bundle/RISK_DATA_PACKAGE_FILES.json','bundle_manifest_sha256'),
      ('research4_frozen_risk_data.ipynb','notebook_file_sha256'),('bundle/source/inputs/'+PROFILE,'profile_sha256'),
      ('bundle/source/inputs/MANIFEST.json','input_manifest_sha256')]:
        if sha(path.parent/file)!=value[key]:raise PermissionError('Prepared risk-reference source/transport changed')
    expected={'notebook':profile['notebook'],'destination':profile['destination'],'expected_cases':36,
              'original_risk_trajectories_required':180,'scientific_GPU_count':0,
              'development_roster_sha256':digest(profile['development_roster'])}
    if any(digest(value.get(k))!=digest(v) for k,v in expected.items()):raise PermissionError('Full original risk-data role/coverage identity differs')
    code=[''.join(c['source']) for c in read(path.parent/'research4_frozen_risk_data.ipynb')['cells'] if c['cell_type']=='code']
    expected_code=(root/'notebooks/composite_task_skill_data.py').read_text(encoding='utf-8-sig').replace(
      'SOURCE_COMMIT_PLACEHOLDER',profile['source_commit']).replace('PACKAGE_MANIFEST_SHA256_PLACEHOLDER',value['bundle_manifest_sha256'])
    if code!=[expected_code]:raise PermissionError('Prepared CPU code differs from exact source/input transport')
    return {'source_commit':profile['source_commit'],'notebook':profile['notebook'],'mode':MODE,'phase':'risk_data_export',
      'package_sha256':value['bundle_sha256'],'authorization_sha256':value['profile_sha256'],
      'kaggle_remainder_scope_sha256':HUMAN_SHA,'scope_A_selection_sha256':SELECTION_SHA,
      'provider_scope':'human-selected-Kaggle-remainder-v1','fresh_stage':True,'scientific_GPU_count':0,
      'hard_timeout_seconds':profile['hard_timeout_seconds'],'optimizer_updates':0,'model_forwards':0,
      'behavior_fitting_or_selection_admitted':False,'heldout_content_read':False,'automatic_scientific_retries':0}


def reserve_data(campaign_root,preparation_file,live_snapshot,guard_token=None):
    root=Path(campaign_root);lock,token,_=guard(root,guard_token)
    try:
        ledger=read(root/'SCIENTIFIC_EXECUTION_LEDGER.json')
        if ledger.get('active_owned_submission') is not None:
            raise PermissionError('Observe the existing worker; no concurrent CPU/GPU science is permitted')
        parent=ledger['candidates'].get('task_skill_stateful_memory_full_fit_v3')
        if not parent or parent.get('owned_compute_released') is not True or not parent.get('closeout_receipt'):
            raise PermissionError('Reconcile the existing fit and its actual release first')
        closed=read(parent['closeout_receipt'])
        if (closed.get('scriptVersionId')!=361964150 or closed.get('source_commit')!='82cf3ecf41e799facd93b8c11e05c6d1f569599c'
                or closed.get('full_fit_completed') is not True or closed.get('full_fit_raw_audited') is not True
                or closed.get('owned_compute_released') is not True):
            raise PermissionError('Current continuation waits for the genuine complete64 fit raw audit/release')
    except Exception:
        if lock.exists() and read(lock).get('token')==token:lock.unlink()
        raise
    return reserve(root,preparation_file,live_snapshot,CANDIDATE,ATTEMPTS,'r4-frozen-risk-data-',
                   validate_preparation,candidate_rule,token)


def reconcile_data(campaign_root,identity,script_version_id=None,guard_token=None):
    return reconcile(campaign_root,identity,CANDIDATE,ATTEMPTS,script_version_id,guard_token)
