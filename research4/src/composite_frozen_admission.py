"""Distinct frozen qualification authority, using the shared submission journal."""
from pathlib import Path
from .io_utils import read,sha,verify_release
from .composite_frozen_probe import authorized_load_plan,MODE
from .composite_frozen_worker import DISK_POLICY,output_target
from .composite_probe_ownership import cache_target
from . import composite_submission_admission as shared

CANDIDATE='frozen_qwen_two_t4_qualification_v1'
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
    if any(prep.get(key)!=pin.get(key) for key in ['source_commit','notebook','destination','cache_root','controller_root','disk_policy']):
        raise ValueError('Prepared and runtime qualification descriptors differ')
    if pin['disk_policy']!=DISK_POLICY:raise ValueError('Frozen disk policy changed')
    output_target(pin['destination'],prep['source_commit'],'0'*32)
    cache_target(pin['cache_root'],prep['source_commit'])
    authorization=read(source/'inputs/FROZEN_QWEN_AUTHORIZATION.json')
    return {'source_commit':prep['source_commit'],'notebook':prep['notebook'],
            'package_sha256':prep['package_sha256'],'authorization_sha256':context['authorization_sha256'],
            'human_message_evidence_locator':authorization['human_message_evidence_locator'],
            'mode':MODE,'scope':'frozen_base_activation_analysis_only','scientific_visible_GPU_count':2,
            'fresh_stage':True}


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
    return shared.reserve(campaign_root,preparation_file,live_snapshot,CANDIDATE,ATTEMPTS,
                          'r4-kaggle-frozen-',validate_preparation,qualification_candidate,guard_token)


def reconcile_submission(campaign_root,identity,script_version_id=None,guard_token=None):
    return shared.reconcile(campaign_root,identity,CANDIDATE,ATTEMPTS,script_version_id,guard_token)
