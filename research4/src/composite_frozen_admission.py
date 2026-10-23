"""Distinct frozen qualification authority, using the shared submission journal."""
from pathlib import Path
from .io_utils import read,sha,verify_release
from .composite_frozen_probe import authorized_load_plan,MODE,validate_pre_model_parent,PRE_MODEL_PARENT_CANDIDATE
from .composite_frozen_worker import DISK_POLICY,output_target
from .composite_probe_ownership import cache_target
from . import composite_submission_admission as shared

CANDIDATE='frozen_qwen_two_t4_metadata_qualification_v2'
ATTEMPTS='authorized_qualification_attempts'
begin_reconciliation=shared.begin_reconciliation


def validate_preparation(preparation_file):
    preparation_file=Path(preparation_file);prep=read(preparation_file)
    if prep.get('status')!='prepared_authorized_frozen_probe_not_submitted' or prep.get('mode')!=MODE or prep.get('recovery'):
        raise PermissionError('A proposal or recognition exception does not authorize frozen qualification')
    source=preparation_file.parent/'bundle/source'
    verify_release(source,prep['source_commit'])
    if sha(preparation_file.parent/'FROZEN_PACKAGE.zip')!=prep['package_sha256']:
        raise ValueError('Frozen qualification package changed')
    context=authorized_load_plan(source,prep['source_commit'],read(source/'config/composite_controller_rank32_v2.json')['qwen'])
    pin=context['pin']
    if any(prep.get(key)!=pin.get(key) for key in ['source_commit','notebook','destination','cache_root','controller_root','disk_policy','pre_model_parent']):
        raise ValueError('Prepared and runtime qualification descriptors differ')
    parent=validate_pre_model_parent(pin.get('pre_model_parent'))
    if parent['source_commit']==prep['source_commit']:
        raise PermissionError('Never repeat the unchanged failed validator source')
    if pin['disk_policy']!=DISK_POLICY:raise ValueError('Frozen disk policy changed')
    output_target(pin['destination'],prep['source_commit'],'0'*32)
    cache_target(pin['cache_root'],prep['source_commit'])
    authorization=read(source/'inputs/FROZEN_QWEN_AUTHORIZATION.json')
    return {'source_commit':prep['source_commit'],'notebook':prep['notebook'],
            'package_sha256':prep['package_sha256'],'authorization_sha256':context['authorization_sha256'],
            'human_message_evidence_locator':authorization['human_message_evidence_locator'],
            'mode':MODE,'scope':'frozen_base_activation_analysis_only','scientific_visible_GPU_count':2,
            'fresh_stage':True,'pre_model_parent':parent}


def verify_parent_under_guard(campaign_root,parent):
    """Reconcile the original consumed attempt; never reuse its allowance."""
    parent=validate_pre_model_parent(parent);root=Path(campaign_root).resolve()
    ledger=read(root/'SCIENTIFIC_EXECUTION_LEDGER.json')
    record=ledger['candidates'].get(PRE_MODEL_PARENT_CANDIDATE,{})
    if record.get('status')!='failed_before_model_initialization_independently_audited_GPU_released':
        raise PermissionError('The original qualification is not an audited pre-model failure')
    if record.get('scientific_execution_started') is not False:
        raise PermissionError('Unknown or begun scientific execution prohibits this pre-model recovery')
    if str(record.get('current_run_id'))!=parent['run_id'] or record.get('source_commit')!=parent['source_commit']:
        raise PermissionError('Another parent run/source cannot authorize this recovery')
    for key in ['archive_sha256','audit_sha256','closeout_sha256','counts_exact_at_exit','forward_counts',
                'pre_model_initialization_failure_verified','owned_compute_released']:
        if record.get(key)!=parent[key]:raise PermissionError('Parent ledger and immutable recovery freeze differ')
    attempts=record.get(ATTEMPTS,[])
    if (len(attempts)!=1 or str(attempts[0].get('scriptVersionId'))!=parent['run_id']
        or attempts[0].get('source_commit')!=parent['source_commit']
        or attempts[0].get('status')!=record['status'] or attempts[0].get('owned_compute_released') is not True
        or attempts[0].get('authority_consumed') is not True):
        raise PermissionError('The original submission identity is unresolved')
    closeout=Path(record['closeout_receipt']).resolve()
    if not closeout.is_relative_to(root) or sha(closeout)!=parent['closeout_sha256']:
        raise PermissionError('Parent closeout is outside this campaign or changed')
    receipt=read(closeout);audit=Path(receipt['audit_receipt']).resolve()
    if str(receipt.get('run_id'))!=parent['run_id'] or receipt.get('source_commit')!=parent['source_commit'] or receipt.get('mode')!=MODE:
        raise PermissionError('The parent closeout belongs to another run/source/stage')
    for key in ['archive_sha256','audit_sha256','counts_exact_at_exit','forward_counts',
                'pre_model_initialization_failure_verified','owned_compute_released']:
        if receipt.get(key)!=parent[key]:raise PermissionError('The parent closeout does not bind this recovery')
    if not audit.is_relative_to(root) or sha(audit)!=parent['audit_sha256']:
        raise PermissionError('Parent raw audit is outside this campaign or changed')
    proof=read(audit)
    if receipt.get('error')!='Actual model metadata bytes differ from the original Git blob':
        raise PermissionError('This recovery is limited to the metadata validator failure')
    if str(proof.get('run_id'))!=parent['run_id'] or proof.get('worker_status')!='failed' or proof.get('complete_qualification_audited') is not False:
        raise PermissionError('The parent raw audit does not establish this failed stage')
    for key in ['archive_sha256','counts_exact_at_exit','forward_counts']:
        if proof.get(key)!=parent[key]:raise PermissionError('The parent raw evidence does not establish zero forwards')
    return True


def qualification_candidate(candidate,binding):
    if candidate is None:
        return {'status':'prepared','mode':MODE,'source_commit':binding['source_commit'],
                'scientific_execution_started':False,'owned_compute_released':True,
                'automatic_scientific_retries':0,'heldout_opened':False}
    if candidate.get('status')!='prepared' or candidate.get('scientific_execution_started') is not False:
        raise PermissionError('The qualification candidate is already started or ambiguous; no retry')
    if candidate.get('source_commit')!=binding['source_commit'] or candidate.get('mode')!=MODE:
        raise PermissionError('The prepared qualification candidate has another source/mode freeze')
    return candidate


def reserve_once(campaign_root,preparation_file,live_snapshot,guard_token=None):
    def validated_under_guard(file):
        binding=validate_preparation(file)
        verify_parent_under_guard(campaign_root,binding['pre_model_parent'])
        return binding
    return shared.reserve(campaign_root,preparation_file,live_snapshot,CANDIDATE,ATTEMPTS,
                          'r4-kaggle-frozen-metadata-v2-',validated_under_guard,qualification_candidate,guard_token)


def reconcile_submission(campaign_root,identity,script_version_id=None,guard_token=None):
    return shared.reconcile(campaign_root,identity,CANDIDATE,ATTEMPTS,script_version_id,guard_token)
