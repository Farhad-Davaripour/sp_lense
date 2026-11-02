"""Fresh one-use Kaggle data admission; no submission or authority creation."""
from pathlib import Path
from .io_utils import read,sha
from .composite_kaggle_v2_profile import validate_source,HUMAN_SHA
from .composite_submission_admission import reserve,reconcile

CANDIDATE='generalized_data_kaggle_v1'
ATTEMPTS='data_attempts'


def validate_preparation(path):
    path=Path(path);value=read(path);root=path.parent/'bundle/source'
    profile,identity=validate_source(root,value['source_commit'])
    if value.get('status')!='prepared_Kaggle_CPU_data_bundle_not_submitted':raise PermissionError('Exact fresh CPU-data preparation required')
    for file,key in [('KAGGLE_V2_DATA_PACKAGE.zip','bundle_sha256'),('bundle/KAGGLE_V2_PACKAGE_FILES.json','bundle_manifest_sha256')]:
        if sha(path.parent/file)!=value[key]:raise PermissionError('Kaggle source bundle changed')
    if sha(path.parent/'research4_generalized_data.ipynb')!=value['notebook_file_sha256']:
        raise PermissionError('Notebook code changed after preparation')
    if value['notebook']!=identity['notebook'] or value['profile_sha256']!=sha(root/'inputs/KAGGLE_REMAINDER_PROFILE.json'):
        raise PermissionError('Notebook/profile identity changed')
    from .composite_v2_data_contract import validate_metadata
    context=validate_metadata(root,value['source_commit'])
    return {'source_commit':value['source_commit'],'notebook':identity['notebook'],'mode':profile['mode'],
        'package_sha256':value['bundle_sha256'],'package_manifest_sha256':value['bundle_manifest_sha256'],
        'authorization_sha256':context['authorization_sha256'],'kaggle_remainder_scope_sha256':HUMAN_SHA,
        'provider_scope':'human-selected-Kaggle-remainder-v1','fresh_stage':True,
        'scientific_GPU_count':0,'hard_timeout_seconds':900,'automatic_scientific_retries':0}


def candidate_rule(prior,binding):
    if prior is not None:raise PermissionError('Preserve the consumed candidate; reconcile its existing identity')
    return {'status':'prepared','platform':'Kaggle','account':'farhad89','scientific_execution_started':False,
            'owned_compute_released':True,'actual_data_result':None}


def reserve_data(campaign_root,preparation_file,live_snapshot,guard_token=None):
    return reserve(campaign_root,preparation_file,live_snapshot,CANDIDATE,ATTEMPTS,'r4-kaggle-v2-data-',
                   validate_preparation,candidate_rule,guard_token)


def reconcile_data(campaign_root,identity,script_version_id=None,guard_token=None):
    return reconcile(campaign_root,identity,CANDIDATE,ATTEMPTS,script_version_id,guard_token)
