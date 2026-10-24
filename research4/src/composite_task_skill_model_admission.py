"""One-use full-rank task qualification admission; no provider submission API."""
from pathlib import Path
from .io_utils import read, sha
from .composite_submission_admission import reserve, reconcile, guard
from .composite_task_skill_training import validate_model_source, PLAN, DATA_CLOSEOUT_SHA, PRE_FORWARD_PARENT_SHA
from .composite_kaggle_v2_profile import HUMAN_SHA
from .composite_task_skill_profile import SELECTION_SHA
from .composite_kaggle_v2_admission import candidate_rule

CANDIDATE = 'task_skill_gpu_qualification_postload_v2'
ATTEMPTS = 'qualification_attempts'


def validate_preparation(path):
    path = Path(path)
    value = read(path)
    root = path.parent / 'bundle/source'
    phase=value.get('phase')
    from .composite_task_memory_profile import MODE as MEMORY_MODE
    stateful=value.get('mode')==MEMORY_MODE
    stage, _, evidence = validate_model_source(root, value['source_commit'],value.get('qualification_archive_local_path'))
    expected_status={'qualification':'prepared_task_qualification_not_submitted','full_fit':'prepared_task_full_fit_not_submitted'}.get(phase)
    if stateful:
        if phase!='qualification':raise PermissionError('No stateful full-fit workload is admitted yet')
        expected_status='prepared_task_stateful_memory_qualification_not_submitted'
    if expected_status is None or value.get('status') != expected_status or stage['phase']!=phase:
        raise PermissionError('Exact fresh task qualification preparation required')
    package='TASK_FULL_FIT_PACKAGE.zip' if phase=='full_fit' else 'TASK_QUALIFICATION_PACKAGE.zip'
    notebook='research4_task_skill_fit.ipynb' if phase=='full_fit' else 'research4_task_skill_qualification.ipynb'
    if stateful:package='TASK_MEMORY_QUALIFICATION_PACKAGE.zip';notebook='research4_task_memory_qualification_v3.ipynb'
    for name, key in [(package,'bundle_sha256'),
                      ('bundle/TASK_TRAINING_PACKAGE_FILES.json','bundle_manifest_sha256'),
                      (notebook,'notebook_file_sha256'),
                      ('bundle/source/inputs/' + PLAN,'plan_sha256'),
                      ('bundle/source/inputs/MANIFEST.json','input_manifest_sha256')]:
        if sha(path.parent / name) != value[key]:
            raise PermissionError('Frozen qualification package/code changed')
    if value['notebook'] != stage['notebook'] or value['training_states'] != len(evidence.rows()):
        raise PermissionError('Qualification identity or complete task dataset differs')
    return {'source_commit':value['source_commit'], 'notebook':stage['notebook'], 'mode':stage['version'],
            'phase':phase, 'package_sha256':value['bundle_sha256'],
            'authorization_sha256':value['plan_sha256'], 'source_data_archive_sha256':value['data_archive_sha256'],
            'kaggle_remainder_scope_sha256':HUMAN_SHA, 'scope_A_selection_sha256':SELECTION_SHA,
            'provider_scope':'human-selected-Kaggle-remainder-v1', 'fresh_stage':True,
            'scientific_GPU_count':2, 'optimizer_updates':stage['optimizer_updates'], 'full_fit_completed':False,
            'hard_timeout_seconds':stage['hard_timeout_seconds'], 'automatic_scientific_retries':0}


def reserve_qualification(campaign_root, preparation_file, live_snapshot, guard_token=None):
    root = Path(campaign_root)
    lock, token, _ = guard(root, guard_token)
    try:
        prior = read(root / 'SCIENTIFIC_EXECUTION_LEDGER.json')['candidates'].get('ordinary_task_skill_data_bootstrap_v2')
        if (not prior or prior.get('status') != 'complete_task_only_data_independently_audited_CPU_released'
                or prior.get('owned_compute_released') is not True
                or sha(prior['closeout_receipt']) != DATA_CLOSEOUT_SHA):
            raise PermissionError('Complete independently audited/released task-data parent required')
        failure=read(root/'SCIENTIFIC_EXECUTION_LEDGER.json')['candidates'].get('task_skill_gpu_qualification_v1')
        if (not failure or failure.get('status')!='failed_pre_forward_engineering_audited_released'
                or failure.get('model_forwards_started') is not False or failure.get('owned_compute_released') is not True
                or sha(failure['closeout_receipt'])!=PRE_FORWARD_PARENT_SHA):
            raise PermissionError('A started/unknown scientific experiment cannot acquire an engineering repair')
    except Exception:
        if lock.exists() and read(lock).get('token') == token:
            lock.unlink()
        raise
    return reserve(root, preparation_file, live_snapshot, CANDIDATE, ATTEMPTS,
                   'r4-task-qualification-', validate_preparation, candidate_rule, token)


def reconcile_qualification(campaign_root, identity, script_version_id=None, guard_token=None):
    return reconcile(campaign_root, identity, CANDIDATE, ATTEMPTS, script_version_id, guard_token)


def reserve_full_fit(campaign_root,preparation_file,live_snapshot,guard_token=None):
    root=Path(campaign_root);lock,token,_=guard(root,guard_token)
    try:
        parent=read(root/'SCIENTIFIC_EXECUTION_LEDGER.json')['candidates'].get(CANDIDATE)
        if (not parent or parent.get('status')!='complete_full_rank_scratch_qualification_independently_audited_GPU_released'
                or parent.get('owned_compute_released') is not True):
            raise PermissionError('Actual complete audited and released qualification required')
        from .composite_task_skill_parent import PARENT_CLOSEOUT_SHA
        if sha(parent['closeout_receipt'])!=PARENT_CLOSEOUT_SHA:raise PermissionError('Qualification closeout changed')
        binding=validate_preparation(preparation_file)
        if binding['phase']!='full_fit' or binding['optimizer_updates']!=64:raise PermissionError('Full64-update fit required')
    except Exception:
        if lock.exists() and read(lock).get('token')==token:lock.unlink()
        raise
    return reserve(root,preparation_file,live_snapshot,'task_skill_full_fit_v1','fit_attempts',
                   'r4-task-full-fit-',validate_preparation,candidate_rule,token)


def reconcile_full_fit(campaign_root,identity,script_version_id=None,guard_token=None):
    return reconcile(campaign_root,identity,'task_skill_full_fit_v1','fit_attempts',script_version_id,guard_token)


def reserve_stateful_qualification(campaign_root,preparation_file,live_snapshot,guard_token=None):
    """One new declared stress exercise; never reset or resume the started fit."""
    root=Path(campaign_root);lock,token,_=guard(root,guard_token)
    try:
        from .composite_task_memory_profile import MODE as MEMORY_MODE
        from .composite_task_memory_source import FAILURE_CLOSEOUT_SHA
        parent=read(root/'SCIENTIFIC_EXECUTION_LEDGER.json')['candidates'].get('task_skill_full_fit_v1')
        if (not parent or parent.get('status')!='failed_started_full64_fit_OOM_partial_journals_audited_GPU_released'
                or parent.get('owned_compute_released') is not True or parent.get('scientific_execution_started') is not True
                or parent.get('complete_resume_state_available') is not False
                or sha(parent['closeout_receipt'])!=FAILURE_CLOSEOUT_SHA
                or parent.get('actual_partial_counts',{}).get('completed_update_journals')!=12):
            raise PermissionError('Exact consumed started-fit failure/release required; no zero-forward allowance')
        binding=validate_preparation(preparation_file)
        if binding['mode']!=MEMORY_MODE or binding['phase']!='qualification' or binding['optimizer_updates']!=3:
            raise PermissionError('Only the separately registered three-update stateful qualification')
    except Exception:
        if lock.exists() and read(lock).get('token')==token:lock.unlink()
        raise
    return reserve(root,preparation_file,live_snapshot,'task_skill_stateful_memory_qualification_v3','qualification_attempts',
      'r4-task-stateful-q3-',validate_preparation,candidate_rule,token)


def reconcile_stateful_qualification(campaign_root,identity,script_version_id=None,guard_token=None):
    return reconcile(campaign_root,identity,'task_skill_stateful_memory_qualification_v3','qualification_attempts',script_version_id,guard_token)
