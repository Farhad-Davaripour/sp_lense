"""Separate actor-only bounded V4 qualification/full-fit contracts.

This is metadata, never an allocation or fitting permit. Counts and lineage
come only from the new complete raw data reader, never historical task rows.
"""
from copy import deepcopy
import re
from .io_utils import digest
from .composite_bounded_behavior_interface import PROFILE
from .composite_bounded_behavior_profile import QUAL_BOUNDS, FIT_BOUNDS
from .composite_task_memory_profile import (allocator_evidence, resident_optimizer_evidence,
    verify_allocator_record, export_geometry, ALLOCATOR)

MODE='bounded-fictional-behavior-actor-two-t4-sampling-a1'
DIRECT_MODE='bounded-fictional-behavior-actor-two-t4-sampling-a1-direct-v1'
WAIVER='HUMAN_DIRECT_FIT_WAIVER.json'
WAIVER_SHA='a954c99a1edb5c720c182b26350344f3590404b5aaf8879630d923a6f36d8fcd'
RAW_PROFILE='bounded-behavior-three-update-full-state-raw-v4'
FIT_RAW_PROFILE='bounded-behavior-full-state-raw-v4'
PLAN='BOUNDED_SAMPLING_FIT_PLAN.json'
MANIFEST='BOUNDED_SAMPLING_FIT_PACKAGE_FILES.json'
SOURCE_TRANSPORT_BYTES=256*1024**2
OBJECTIVE='supervised_bounded_fictional_continuation_and_advisory_return_choices_sampling_a1'
PLACEMENT='qwen-two-t4-bounded-behavior-balanced-v4'
MAPPING='per-device-seed202610212-placement22-42-v4'
GiB=1024**3


def stress_schedule(tokens):
    rows=tokens.get('records',[])
    if len(rows)<2 or len({r.get('state_id') for r in rows})!=len(rows):
        raise PermissionError('The complete newly audited V4 training token plan is required')
    for row in rows:
        if type(row.get('combined_tokens')) is not int or not 0<row['combined_tokens']<=6144:
            raise ValueError('Original untruncated native context ceiling required')
    longest=sorted(rows,key=lambda r:(-r['combined_tokens'],r['state_id']))[:2]
    shortest=sorted(rows,key=lambda r:(r['combined_tokens'],r['state_id']))[:2]
    if min(r['combined_tokens'] for r in longest)<=max(r['combined_tokens'] for r in shortest):
        raise ValueError('Actual newly materialized input-size switch is required')
    return {'state_ids':[r['state_id'] for r in longest+shortest+longest],
      'updates':3,'backward_microsteps':6,
      'selection':'complete_new_V4_training_tokens_only_fixed_long_short_long',
      'full_fit_64_update_sampling_changed':False,'qualification_checkpoint_initializes_fit':False}


def placement_plan(config,parent_profile):
    from .composite_placement import candidate_plan
    plan=deepcopy(candidate_plan(config,parent_profile));plan['profile']=PLACEMENT
    for index in range(64):plan['device_map']['model.language_model.layers.'+str(index)]=0 if index<22 else 1
    plan.update(runtime_admitted=False,numerical_parity_admitted=False,
      scientific_validation_admitted=False,initialization_mapping_version=MAPPING)
    return plan


def qualification_plan(commit,data):
    from .composite_bounded_sampling_reader import require_approved_sampling_data
    proof=require_approved_sampling_data(data)
    if not re.fullmatch('[0-9a-f]{40}',commit or ''):raise ValueError('Exact reviewed source commit required')
    rows=data.fitting_rows()
    return {'version':MODE,'source_commit':commit,'phase':'qualification',
      'notebook':'farhad89/research4-bounded-sampling-qualification-v1',
      'owner_thread_id':'01a175f0-fc99-76b1-8bb2-d64f22afbcd7',
      'source_transport_maximum_bytes':SOURCE_TRANSPORT_BYTES,
      'platform':'kaggle','account':'farhad89','objective':OBJECTIVE,
      'source_data_archive_sha256':proof['raw_archive_sha256'],
      'source_data_source_commit':proof['source_commit'],
      'data_audit_canonical_sha256':digest(proof),
      'data_release_canonical_sha256':digest(data.data_release()),
      'approved_draw_plan_sha256':digest(data.fitting_draw_plan()),
      'sampling_approval_sha256':proof['approval_sha256'],'sampling_rule_sha256':proof['rule_sha256'],
      'source_training_rows':len(rows),'fitting_rows_canonical_sha256':digest(rows),
      'optimizer_updates':3,'accumulation':2,'sampling':'fixed_training_only_long_short_long',
      'seed':202610212,'learning_rate':0.0005,'weight_decay':0.0,'clip_norm':1.0,
      'native_interface_profile':PROFILE,'raw_evidence_profile':RAW_PROFILE,
      'visible_devices':'0,1','scientific_GPU_count':2,'kernel_profile':'reference',
      'placement_profile':PLACEMENT,'allocator_configuration':ALLOCATOR,
      'initialization_mapping_version':MAPPING,
      'stateful_AdamW_overlap_and_size_switch_required':True,
      'qualification_checkpoint_initializes_full_fit':False,'checkpoint_initializes_full_fit':False,
      'optimizer_sampler_or_RNG_warm_start':False,'validation_optimizer_updates':0,
      'heldout_optimizer_updates':0,'heldout_content_read':False,
      'detector_predictions_required_for_actor_SFT':False,
      'learned_dynamic_or_sealed_admitted':False,
      'maximum_concurrent_owned_workers':1,'automatic_scientific_retries':0,
      **QUAL_BOUNDS,
      'destination':'/kaggle/working/research4_campaign/runs/'+commit+'/bounded_behavior_sampling_training_a1/qualification_01'}


def full_fit_plan(commit,data,qualification):
    from .composite_bounded_sampling_fit_audit import require_qualification
    parent=require_qualification(qualification)
    plan=qualification_plan(commit,data)
    if any(parent.get(k)!=plan[k] for k in ['source_data_archive_sha256','data_audit_canonical_sha256',
        'fitting_rows_canonical_sha256','native_interface_profile','approved_draw_plan_sha256',
        'sampling_approval_sha256','sampling_rule_sha256']):
        raise PermissionError('New exact-data qualification belongs to another V4 data/interface freeze')
    plan.update(phase='full_fit',notebook='farhad89/research4-bounded-sampling-fit-v1',
      optimizer_updates=64,sampling='approved_48anchors80uniform_exact128_ids_v1',
      raw_evidence_profile=FIT_RAW_PROFILE,**FIT_BOUNDS,
      qualification_source_commit=parent['source_commit'],qualification_run_id=parent['provider_run_id'],
      qualification_raw_archive_sha256=parent['raw_archive_sha256'],
      qualification_initial_hashes_sha256=digest(parent['initial_adapter_hashes']),
      qualification_checkpoint_sha256=parent['checkpoint_sha256'],
      destination='/kaggle/working/research4_campaign/runs/'+commit+'/bounded_behavior_sampling_training_a1/full_fit_01')
    return plan


def direct_full_fit_plan(commit,data,waiver,alias_qualification_archive=None):
    from .composite_bounded_sampling_fit_source import validate_waiver_value
    from .composite_bounded_sampling_fit_parent import historical_initialization_reference
    validate_waiver_value(waiver,data)
    reference,_=historical_initialization_reference(data,alias_qualification_archive)
    plan=qualification_plan(commit,data)
    plan.update(version=DIRECT_MODE,phase='full_fit',
        notebook='farhad89/research4-bounded-sampling-direct-fit-v1',optimizer_updates=64,
        sampling='approved_48anchors80uniform_exact128_ids_v1',raw_evidence_profile=FIT_RAW_PROFILE,
        **FIT_BOUNDS, separate_qualification_waived=True,qualification_performed=False,
        qualification_performed_field_scope='new_prerequisite_for_this_direct_route_only',
        qualification_status='waived_by_explicit_human_instruction_not_passed',
        direct_fit_waiver_sha256=WAIVER_SHA,historical_initialization_reference=reference,
        current_qualification_initial_hash_comparison_performed=False,
        stateful_AdamW_overlap_and_size_switch_required=False,
        interrupted_qualification_attempt_run_id=waiver['qualification_run_id'],
        interrupted_qualification_attempt_outcome='preserved_external_closeout_not_used_as_capacity_evidence',
        destination='/kaggle/working/research4_campaign/runs/'+commit+'/bounded_behavior_sampling_training_a1/direct_full_fit_01')
    return plan
