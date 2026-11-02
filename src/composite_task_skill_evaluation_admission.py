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


def normalized_cases(case_ids):
    ids=[case_ids] if isinstance(case_ids,str) else case_ids
    if (not isinstance(ids,list) or not 1<=len(ids)<=3
            or any(not isinstance(c,str) or not re.fullmatch('[0-9a-f]{64}',c) for c in ids)
            or len(set(ids))!=len(ids)):
        raise PermissionError('One to three exact nonduplicate complete case identities required')
    return list(ids)


def candidate_key(case_ids):
    ids=normalized_cases(case_ids)
    # Source/transport changes never reset a consumed scientific case.
    return 'task_skill_frozen_pair_'+ids[0] if len(ids)==1 else 'task_skill_frozen_group_'+digest(sorted(ids))


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
    descriptors={'notebook':stage['notebook'],'case_ids':stage['partition']['case_ids'],
      'partition_index':stage['partition']['index'],'expected_trajectories':stage['partition']['expected_trajectories'],
      'destination':stage['destination'],'frozen_model_package_sha256':digest(model),
      'checkpoint_sha256':model['checkpoint_sha256'],'fit_run_id':model['fit_run_id']}
    if any(digest(value.get(k))!=digest(v) for k,v in descriptors.items()):
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
      'case_ids':stage['partition']['case_ids'],
      'case_id':stage['partition']['case_ids'][0] if len(stage['partition']['case_ids'])==1 else None,
      'partition_index':stage['partition']['index'],
      'kaggle_remainder_scope_sha256':HUMAN_SHA,'scope_A_selection_sha256':SELECTION_SHA,
      'provider_scope':'human-selected-Kaggle-remainder-v1','fresh_stage':True,'scientific_GPU_count':2,
      'hard_timeout_seconds':stage['hard_timeout_seconds'],'optimizer_updates':0,
      'expected_trajectories':stage['partition']['expected_trajectories'],
      'automatic_scientific_retries':0,'heldout_content_read':False}


def reserve_evaluation(campaign_root,preparation_file,live_snapshot,guard_token=None,*,first_partition_audit=None):
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
        if binding['partition_index']>0:
            from .composite_task_skill_evaluation_audit import require_task_partition
            first=require_task_partition(first_partition_audit)
            if (first['partition_index']!=0 or first['expected_trajectories']!=2
                    or first['source_commit']!=binding['source_commit']
                    or first['checkpoint_sha256']!=binding['checkpoint_sha256']
                    or first['frozen_model_package_sha256']!=binding['frozen_model_package_sha256']):
                raise PermissionError('First useful paired-case raw qualification belongs to another freeze')
            recorded=ledger['candidates'].get(candidate_key(first['case_ids']))
            if (not recorded or recorded.get('owned_compute_released') is not True
                    or not any(a.get('scriptVersionId')==first['provider_run_id']
                               and a.get('source_commit')==first['source_commit'] for a in recorded.get(ATTEMPTS,[]))):
                raise PermissionError('Reconcile and release the exact first paired-case submission before grouped stages')
        ids=normalized_cases(binding.get('case_ids',binding.get('case_id')));key=candidate_key(ids)
        for name,value in ledger['candidates'].items():
            for attempt in value.get(ATTEMPTS,[]):
                if attempt.get('phase')!='task_development':continue
                prior=normalized_cases(attempt.get('case_ids',attempt.get('case_id')))
                if set(prior)&set(ids):
                    raise PermissionError('At least one task case already has a consumed submission under this or another source/group')
    except Exception:
        if lock.exists() and read(lock).get('token')==token:lock.unlink()
        raise
    return reserve(root,preparation_file,live_snapshot,key,ATTEMPTS,'r4-task-pair-',
                   lambda _:binding,candidate_rule,token)


def reconcile_evaluation(campaign_root,identity,case_ids,script_version_id=None,guard_token=None):
    return reconcile(campaign_root,identity,candidate_key(case_ids),ATTEMPTS,script_version_id,guard_token)
