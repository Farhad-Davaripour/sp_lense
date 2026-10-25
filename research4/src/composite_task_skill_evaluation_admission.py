"""One-use frozen task-case admission. Never submit, fit or manufacture approval."""
from pathlib import Path
import re
from .io_utils import read,sha,digest
from .composite_submission_admission import reserve,reconcile,guard
from .composite_kaggle_v2_admission import candidate_rule
from .composite_kaggle_v2_profile import HUMAN_SHA
from .composite_task_skill_profile import SELECTION_SHA
from .composite_task_skill_inference import validate_source,STAGE,MODE
from .composite_task_skill_package import require_frozen_task_package

ATTEMPTS='evaluation_attempts'


def candidate_key(case_id):
    if not isinstance(case_id,str) or not re.fullmatch('[0-9a-f]{64}',case_id):
        raise PermissionError('Exact complete task-case identity required')
    # Source/transport changes never reset a consumed scientific case.
    return 'task_skill_frozen_pair_'+case_id


def validate_preparation(path):
    path=Path(path);value=read(path);root=path.parent/'bundle/source'
    if (value.get('status')!='prepared_frozen_task_development_not_submitted'
            or value.get('mode')!=MODE or value.get('phase')!='task_development'):
        raise PermissionError('Exact separately prepared frozen task stage required')
    context=validate_source(root,value['source_commit'],value['fit_archive_local_path'],
                            value['qualification_archive_local_path'])
    stage=context.stage();model=require_frozen_task_package(context.model)
    for file,key in [('TASK_EVALUATION_PACKAGE.zip','bundle_sha256'),
                     ('bundle/TASK_TRAINING_PACKAGE_FILES.json','bundle_manifest_sha256'),
                     ('research4_task_development.ipynb','notebook_file_sha256'),
                     ('bundle/source/inputs/'+STAGE,'stage_sha256'),
                     ('bundle/source/inputs/MANIFEST.json','input_manifest_sha256')]:
        if sha(path.parent/file)!=value[key]:raise PermissionError('Frozen task-evaluation package/code changed')
    if (value['notebook']!=stage['notebook'] or value['case_ids']!=stage['partition']['case_ids']
            or value['partition_index']!=stage['partition']['index']
            or value['frozen_model_package_sha256']!=digest(model)
            or value['checkpoint_sha256']!=model['checkpoint_sha256'] or value['fit_run_id']!=model['fit_run_id']):
        raise PermissionError('Prepared task-case/model/source identity differs')
    code=[''.join(c['source']) for c in read(path.parent/'research4_task_development.ipynb')['cells'] if c['cell_type']=='code']
    expected=(root/'notebooks/composite_task_skill_qualification.py').read_text().replace(
      'SOURCE_COMMIT_PLACEHOLDER',stage['source_commit']).replace('PACKAGE_MANIFEST_SHA256_PLACEHOLDER',value['bundle_manifest_sha256'])
    if code!=[expected]:raise PermissionError('Prepared code does not bind the exact partition transport')
    return {'source_commit':stage['source_commit'],'notebook':stage['notebook'],'mode':MODE,'phase':'task_development',
      'package_sha256':value['bundle_sha256'],'package_manifest_sha256':value['bundle_manifest_sha256'],
      'authorization_sha256':value['stage_sha256'],'source_data_archive_sha256':model['data_archive_sha256'],
      'frozen_model_package_sha256':digest(model),'checkpoint_sha256':model['checkpoint_sha256'],
      'fit_source_commit':model['fit_source_commit'],'fit_run_id':model['fit_run_id'],
      'fit_raw_archive_sha256':model['fit_raw_archive_sha256'],
      'case_id':stage['partition']['case_ids'][0],'partition_index':stage['partition']['index'],
      'kaggle_remainder_scope_sha256':HUMAN_SHA,'scope_A_selection_sha256':SELECTION_SHA,
      'provider_scope':'human-selected-Kaggle-remainder-v1','fresh_stage':True,'scientific_GPU_count':2,
      'hard_timeout_seconds':stage['hard_timeout_seconds'],'optimizer_updates':0,'expected_trajectories':2,
      'automatic_scientific_retries':0,'heldout_content_read':False}


def reserve_evaluation(campaign_root,preparation_file,live_snapshot,guard_token=None):
    root=Path(campaign_root);lock,token,_=guard(root,guard_token)
    try:
        ledger=read(root/'SCIENTIFIC_EXECUTION_LEDGER.json')
        if ledger.get('active_owned_submission') is not None:
            raise PermissionError('Observe/reconcile the existing owned worker; never duplicate it')
        binding=validate_preparation(preparation_file)
        parent=ledger['candidates'].get('task_skill_stateful_memory_full_fit_v3')
        if not parent or parent.get('owned_compute_released') is not True or not parent.get('closeout_receipt'):
            raise PermissionError('Actual complete raw-audited and released full fit required')
        closeout=read(parent['closeout_receipt'])
        expected={'scriptVersionId':binding['fit_run_id'],'source_commit':binding['fit_source_commit'],
          'raw_archive_sha256':binding['fit_raw_archive_sha256'],'checkpoint_sha256':binding['checkpoint_sha256'],
          'full_fit_completed':True,'full_fit_raw_audited':True,'owned_compute_released':True}
        if any(digest(closeout.get(k))!=digest(v) for k,v in expected.items()):
            raise PermissionError('Ledger full-fit closeout differs from the actual raw dependency')
        key=candidate_key(binding['case_id'])
        for name,value in ledger['candidates'].items():
            if name!=key and any(a.get('phase')=='task_development' and a.get('case_id')==binding['case_id']
                                 for a in value.get(ATTEMPTS,[])):
                raise PermissionError('This task case already has a consumed submission under another source/key')
    except Exception:
        if lock.exists() and read(lock).get('token')==token:lock.unlink()
        raise
    return reserve(root,preparation_file,live_snapshot,key,ATTEMPTS,'r4-task-pair-',
                   lambda _:binding,candidate_rule,token)


def reconcile_evaluation(campaign_root,identity,case_id,script_version_id=None,guard_token=None):
    return reconcile(campaign_root,identity,candidate_key(case_id),ATTEMPTS,script_version_id,guard_token)
