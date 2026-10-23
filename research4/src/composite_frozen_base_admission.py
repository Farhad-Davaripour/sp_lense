"""Original-base partition identities over the shared one-worker submission guard."""
from pathlib import Path
from .io_utils import read, sha, digest, verify_release
from .composite_frozen_base import (MODE, PLAN, AUTHORIZATION, PARENTS, authorized_load_plan,
    DETECTOR_CHECKPOINT_SHA256, validate_parents)
from .composite_frozen_base_worker import DISK_POLICY
from .composite_frozen_worker import output_target
from .composite_probe_ownership import cache_target
from . import composite_submission_admission as shared

ATTEMPTS = 'authorized_base_partition_attempts'
PREFIX = 'frozen_base_validation_v1_p'
begin_reconciliation = shared.begin_reconciliation


def validate_preparation(preparation_file):
    preparation_file = Path(preparation_file); prep = read(preparation_file)
    if prep.get('status') != 'prepared_authorized_frozen_base_not_submitted' or prep.get('mode') != MODE:
        raise PermissionError('Only a prepared authorized original-base partition may be submitted')
    source = preparation_file.parent/'bundle/source'
    verify_release(source, prep['source_commit'])
    if sha(preparation_file.parent/'FROZEN_PACKAGE.zip') != prep['package_sha256']:
        raise ValueError('Frozen-base package changed')
    context = authorized_load_plan(source, prep['source_commit'],
        read(source/'config/composite_controller_rank32_v2.json')['qwen'], preparation_file.parent/'bundle/artifacts')
    pin = context['pin']; plan = context['evaluation_plan']
    if (pin['disk_policy'] != DISK_POLICY
        or any(prep.get(key) != pin.get(key) for key in ['source_commit','notebook','destination','cache_root','controller_root','partition','transport_manifest_sha256'])):
        raise ValueError('Frozen-base preparation/runtime descriptor differs')
    output_target(pin['destination'], prep['source_commit'], '0'*32)
    cache_target(pin['cache_root'], prep['source_commit'])
    authority = read(source/'inputs'/AUTHORIZATION)
    common = {'source_commit': prep['source_commit'], 'transport_manifest_sha256': pin['transport_manifest_sha256'],
              'models': {name: sha(source/'inputs'/name) for name in ['QWEN_PIN.json','MODERNBERT_PIN.json']},
              'protocol_sha256': sha(source/'config/composite_controller_rank32_v2.json'),
              'detector_checkpoint_sha256': DETECTOR_CHECKPOINT_SHA256, 'parent_receipts': PARENTS}
    return {'source_commit': prep['source_commit'], 'notebook': prep['notebook'],
        'package_sha256': prep['package_sha256'], 'authorization_sha256': context['authorization_sha256'],
        'human_message_evidence_locator': authority['human_message_evidence_locator'],
        'mode': MODE, 'scope': 'original_validation_base_control_only', 'fresh_stage': True,
        'scientific_visible_GPU_count': 2, 'partition': plan['partition'], 'case_ids': plan['case_ids'],
        'common_freeze': common, 'common_freeze_sha256': digest(common),
        'conditions': ['base'], 'full_system_validation_complete': False, 'heldout_opened': False}


def verify_campaign_under_guard(campaign_root, binding):
    ledger = read(Path(campaign_root)/'SCIENTIFIC_EXECUTION_LEDGER.json')
    detector = ledger['candidates']['detector_rank32_v5']['authorized_recovery_attempts']
    selected = [item for item in detector if item.get('scriptVersionId') == 361470090]
    if (len(selected) != 1 or selected[0].get('full_validation_complete') is not True
        or selected[0].get('component_validation_admitted') is not True
        or selected[0].get('owned_compute_released') is not True):
        raise PermissionError('The admitted recognition parent is unresolved')
    qualification = ledger['candidates']['frozen_qwen_two_t4_metadata_qualification_v2']
    if (qualification.get('current_run_id') != 361504592
        or qualification.get('complete_qualification_audited') is not True
        or qualification.get('owned_compute_released') is not True):
        raise PermissionError('The frozen native runtime qualification parent is unresolved')
    for key, candidate in ledger['candidates'].items():
        if key.startswith('frozen_base_allocator_v2_p'):
            for attempt in candidate.get('authorized_operational_partition_attempts',[]):
                if set(attempt.get('case_ids',[])) & set(binding.get('case_ids',[])):
                    raise PermissionError('A versioned operational execution already owns these original BASE cases')
        if not key.startswith(PREFIX): continue
        for attempt in candidate.get(ATTEMPTS, []):
            if attempt.get('common_freeze_sha256') != binding['common_freeze_sha256']:
                raise PermissionError('All original-base partitions must use one immutable scientific freeze')


def base_candidate(candidate, binding):
    if candidate is None:
        return {'status':'prepared','mode':MODE,'source_commit':binding['source_commit'],
                'partition':binding['partition'],'scientific_execution_started':False,
                'owned_compute_released':True,'automatic_scientific_retries':0,'heldout_opened':False}
    if (candidate.get('status') != 'prepared' or candidate.get('scientific_execution_started') is not False
        or candidate.get('source_commit') != binding['source_commit'] or candidate.get('mode') != MODE
        or candidate.get('partition') != binding['partition']):
        raise PermissionError('The original-base partition is already executed, ambiguous or differently frozen')
    return candidate


def reserve_once(campaign_root, preparation_file, live_snapshot, guard_token=None):
    # Read-only precheck selects the key; validate again inside the shared guard.
    initial = validate_preparation(preparation_file)
    key = PREFIX+str(initial['partition']).zfill(2)
    def checked(file):
        binding = validate_preparation(file)
        if binding != initial: raise ValueError('Preparation changed during reconciliation')
        verify_campaign_under_guard(campaign_root, binding)
        return binding
    return shared.reserve(campaign_root, preparation_file, live_snapshot, key, ATTEMPTS,
        'r4-kaggle-base-p'+str(initial['partition']).zfill(2)+'-', checked, base_candidate, guard_token)


def reconcile_submission(campaign_root, identity, script_version_id=None, guard_token=None):
    ledger = read(Path(campaign_root)/'SCIENTIFIC_EXECUTION_LEDGER.json')
    keys = [key for key, candidate in ledger['candidates'].items() if key.startswith(PREFIX)
            and any(attempt['identity'] == identity for attempt in candidate.get(ATTEMPTS, []))]
    if len(keys) != 1: raise PermissionError('Unknown or ambiguous original-base partition identity')
    return shared.reconcile(campaign_root, identity, keys[0], ATTEMPTS, script_version_id, guard_token)
