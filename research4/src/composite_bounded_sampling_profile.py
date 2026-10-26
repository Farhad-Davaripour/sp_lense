"""Versioned approved sampler CPU metadata; never an allocation or training permit."""
from copy import deepcopy
from pathlib import Path
import re
from .io_utils import digest
from .composite_bounded_behavior_profile import CONFIG_SHA,validate_preparation
from .composite_bounded_sampling_rule import RULE_SHA

MODE='bounded-behavior-approved-sampling-amendment-v1-data'
PHASE='bounded_sampling_amendment_v1'
NOTEBOOK='farhad89/research4-bounded-sampling-amendment-v1'
OWNER_THREAD='01a175f0-fc99-76b1-8bb2-d64f22afbcd7'
PROFILE='BOUNDED_SAMPLING_AMENDMENT_PROFILE.json'
MANIFEST='BOUNDED_SAMPLING_AMENDMENT_PACKAGE_FILES.json'
APPROVAL='HUMAN_SAMPLING_AMENDMENT_APPROVAL.json'
PROPOSAL='APPROVED_SAMPLING_PROPOSAL.json'
APPROVAL_SHA='d7636f3fac015745715cd610ecbaa677cdeeda4dbc424457e3a5737739ae56fe'
PROPOSAL_SHA='a0dee04c3b1cfe9cf86c527c97678311d9f8ab3f52ab6ebefeb3ceead0b9614c'
FAILED_SOURCE='c37f4845896d2dc2824fc8dd2d840dc5fe627948'
FAILED_RUN=362532871
FAILED_CLOSEOUT_SHA='c867b2a215ce821e2301dde39b250e0fd5ac5f58fc26871da392a970e843e0b8'
BOUNDS={'hard_timeout_seconds':1800,'worker_seconds':1200,'export_reserve_seconds':120,
 'maximum_raw_export_bytes':256*1024**2,'terminal_metadata_reserve_bytes':4*1024**2,
 'maximum_reference_states':3600,'maximum_concurrent_owned_workers':1,'automatic_scientific_retries':0}


def expected_profile(commit,cfg,parent,approval):
    from .composite_bounded_sampling_parent import require_failed_parent
    validate_preparation(cfg);proof=require_failed_parent(parent)
    if not re.fullmatch('[0-9a-f]{40}',commit or ''):raise PermissionError('Exact reviewed amended source required')
    return {'version':MODE,'source_commit':commit,'phase':PHASE,'notebook':NOTEBOOK,'owner_thread_id':OWNER_THREAD,
      'platform':'kaggle','account':'farhad89','approval_sha256':APPROVAL_SHA,'proposal_sha256':PROPOSAL_SHA,
      'approved_rule_sha256':RULE_SHA,'original_preparation_configuration_sha256':CONFIG_SHA,
      'failed_reference_receipt_sha256':digest(proof),'original_failed_source_commit':FAILED_SOURCE,
      'original_failed_run_id':FAILED_RUN,'original_failed_closeout_sha256':FAILED_CLOSEOUT_SHA,
      'parent_failure_remains_failed_unadmitted':True,'training_cases':108,'development_cases':36,
      'original_eligible_training_rows':640,'draws':128,'coverage_anchor_occurrences':48,'uniform_occurrences':80,
      'updates_preserved_for_future_fit':64,'accumulation_preserved_for_future_fit':2,'seed':202610212,
      'canonical_row_order':'ascending unique state_id','original_row_order_and_failed128_preserved':True,
      'new_teacher_or_case_generation':False,'source_only_or_approval_only_admits_training':False,
      'visible_devices':'','scientific_GPU_count':0,'pretrained_model_forwards':0,'optimizer_updates':0,
      'heldout_content_read':False,'detector_predictions_required':False,
      'destination':'/kaggle/working/research4_campaign/runs/'+commit+'/bounded_sampling_amendment_v1/export_01',**BOUNDS}


def notebook_code(root,commit,manifest_sha256):
    if not re.fullmatch('[0-9a-f]{40}',commit or '') or not re.fullmatch('[0-9a-f]{64}',manifest_sha256 or ''):
        raise PermissionError('Exact source/manifest bootstrap bindings required')
    code=(Path(root)/'notebooks/composite_bounded_sampling_amendment.py').read_text(encoding='utf-8')
    if code.count('SOURCE_COMMIT_PLACEHOLDER')!=1 or code.count('PACKAGE_MANIFEST_SHA256_PLACEHOLDER')!=1:
        raise PermissionError('One exact source and manifest binding required')
    return code.replace('SOURCE_COMMIT_PLACEHOLDER',commit).replace('PACKAGE_MANIFEST_SHA256_PLACEHOLDER',manifest_sha256)
