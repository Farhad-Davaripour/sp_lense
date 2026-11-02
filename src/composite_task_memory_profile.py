"""Versioned memory-only candidate; metadata/read-only evidence is not admission."""
from copy import deepcopy
import os
from .composite_placement import candidate_plan,metadata_bounds
from .io_utils import digest

MODE='ordinary-task-skill-stateful-memory-v3'
PLACEMENT='qwen-two-t4-task-memory-balanced-v3'
RAW_PROFILE='task-three-update-full-state-raw-v3'
ALLOCATOR='expandable_segments:True'
PROFILE_FILE='config/composite_task_skill_memory_v3.json'
GiB=1024**3
QUALIFICATION_BOUNDS={'hard_timeout_seconds':10800,'worker_seconds':9900,'export_reserve_seconds':900,
  'maximum_raw_evidence_bytes':6*GiB,'physical_export_reserve_bytes':18*GiB,
  'maximum_backbone_forwards':30000,'maximum_concurrent_owned_workers':1,'automatic_scientific_retries':0}


def expected_profile():
    return {'version':MODE,'status':'engineering_memory_candidate_not_runtime_admitted',
      'platform':'kaggle','account':'farhad89','physical_GPU_type':'Tesla T4','devices':[0,1],
      'first_device_layer_count':22,'all_blocks':64,'rank':32,'alpha':64,'projections':496,
      'FP32_adapter_parameters':233455616,'allocator_configuration':ALLOCATOR,'allocator_backend':'native',
      'CPU_or_disk_offload':False,'qualifier_updates':3,'accumulation':2,
      'qualifier_schedule':['longest_pair','shortest_pair','longest_pair'],
      'qualifier_initializes_full_fit':False,'full_fit_updates':64,'seed':202610212,
      'learning_rate':0.0005,'clip_norm':1.0,'weight_decay':0.0,
      'all_training_observations':1188,'context_tokens':6144,'maximum_new_tokens':160,
      'full_fit_sample_schedule_changed':False,'validation_or_risk_metrics_select_runtime':False,
      'raw_evidence_profile':RAW_PROFILE,'bounds':dict(QUALIFICATION_BOUNDS),
      'prior_started_failure_run':361922334,'prior_failure_is_pre_execution':False,
      'prior_identity_reset_or_requeued':False,'new_source_freeze_required':True,
      'full_fit_would_be_an_additional_started_attempt':True,'heldout_content_read':False}


def placement_plan(config,parent_profile,profile):
    if digest(profile)!=digest(expected_profile()):raise PermissionError('Exact reviewed memory candidate required')
    prior=candidate_plan(config,parent_profile)
    plan=deepcopy(prior);plan['profile']=PLACEMENT
    for index in range(64):plan['device_map']['model.language_model.layers.'+str(index)]=0 if index<22 else 1
    plan.update(runtime_admitted=False,numerical_parity_admitted=False,scientific_validation_admitted=False,
                initialization_mapping_version='per-device-seed202610212-placement22-42-v3')
    return plan


def stress_schedule(tokens):
    rows=tokens.get('records',[])
    if len(rows)!=1188 or len({r.get('state_id') for r in rows})!=1188:
        raise PermissionError('Complete exact task-only training token plan required')
    for row in rows:
        if type(row.get('combined_tokens')) is not int or not 0<row['combined_tokens']<=6144:
            raise ValueError('Original full training context required')
    longest=[r['state_id'] for r in sorted(rows,key=lambda r:(-r['combined_tokens'],r['state_id']))[:2]]
    shortest=[r['state_id'] for r in sorted(rows,key=lambda r:(r['combined_tokens'],r['state_id']))[:2]]
    by_id={r['state_id']:r['combined_tokens'] for r in rows}
    if min(by_id[k] for k in longest)<=max(by_id[k] for k in shortest):
        raise ValueError('The registered stress test requires a real input-size switch')
    return {'state_ids':longest+shortest+longest,'updates':3,'backward_microsteps':6,
      'selection':'training_tokens_only_fixed_long_short_long_no_loss_or_validation_selection',
      'full_fit_64_update_sampling_changed':False,'qualification_checkpoint_initializes_fit':False}


def export_geometry(layout):
    if layout['adapter_parameters']!=233455616 or layout['projection_count']!=496:
        raise PermissionError('Full agreed rank32 geometry required')
    initial_A=sum(32*p['shape'][1]*4 for p in layout['projections'].values())
    weights=layout['adapter_parameters']*4
    tensor_bytes=initial_A+6*weights  # A0 + 3 clipped gradients + 2 moments + final weights.
    if tensor_bytes+128*1024**2>QUALIFICATION_BOUNDS['maximum_raw_evidence_bytes']:
        raise MemoryError('Complete untrimmed raw state lacks its metadata reserve')
    return {'initial_A_bytes':initial_A,'each_complete_gradient_or_checkpoint_bytes':weights,
      'named_AdamW_moment_bytes':2*weights,'complete_numeric_state_tensor_bytes':tensor_bytes,
      'unique_raw_budget_bytes':QUALIFICATION_BOUNDS['maximum_raw_evidence_bytes'],
      'physical_export_reserve_bytes':QUALIFICATION_BOUNDS['physical_export_reserve_bytes'],
      'runtime_capacity_admitted':False}


def allocator_evidence(cuda,environ=None):
    """Inspect real existing model allocations, allocating no probe tensors."""
    env=os.environ if environ is None else environ
    if env.get('PYTORCH_ALLOC_CONF')!=ALLOCATOR or env.get('PYTORCH_CUDA_ALLOC_CONF') is not None:
        raise PermissionError('Exact unambiguous allocator configuration required before CUDA initialization')
    if cuda.memory.get_allocator_backend()!='native':raise PermissionError('Reviewed native allocator required')
    segments=cuda.memory_snapshot();selected=[];devices=[]
    for row in segments:
        if row.get('device') not in [0,1] or type(row.get('is_expandable')) is not bool:
            raise ValueError('Both declared allocator snapshot devices and flags required')
        values={k:row[k] for k in ['device','total_size','allocated_size','active_size','is_expandable']}
        if (any(type(values[k]) is not int or values[k]<0 for k in ['device','total_size','allocated_size','active_size'])
                or not values['allocated_size']<=values['active_size']<=values['total_size']):
            raise ValueError('Actual allocator segment sizes differ')
        selected.append(values)
    for index in [0,1]:
        actual=[r for r in selected if r['device']==index and r['is_expandable'] and r['allocated_size']>0]
        if not actual:raise PermissionError('Expandable segments must back actual live model allocations on each GPU')
        stats=cuda.memory_stats(index)
        devices.append({'device':index,'expandable_live_segments':len(actual),
          'expandable_allocated_bytes':sum(r['allocated_size'] for r in actual),
          'inactive_split_bytes':stats['inactive_split_bytes.all.current'],
          'allocated_bytes':stats['allocated_bytes.all.current'],'reserved_bytes':stats['reserved_bytes.all.current']})
    return {'backend':'native','configuration':ALLOCATOR,'devices':devices,'segments':selected,
      'new_probe_allocations':0,'actual_training_capacity_qualified':False}


def resident_optimizer_evidence(optimizer,adapters,layout,update,microstep,state_id):
    """Read actual persistent moment ownership before the next backward."""
    import torch
    parameters=[p for m in adapters.values() for p in [m.A,m.B]]
    if len(parameters)!=992 or sum(p.numel() for p in parameters)!=233455616:
        raise ValueError('All real rank32 adapter parameters required')
    expected_entries=0 if update==1 else 992
    if len(optimizer.state)!=expected_entries:raise ValueError('Unexpected initial/resident AdamW state')
    sizes=[0,0]
    for parameter in parameters:
        if parameter.device.type!='cuda' or parameter.device.index not in [0,1]:
            raise ValueError('Moment evidence must retain both declared CUDA devices')
        if update==1:continue
        values=optimizer.state.get(parameter)
        if values is None or float(values['step'])!=update-1:raise ValueError('Prior optimizer step differs')
        for name in ['exp_avg','exp_avg_sq']:
            tensor=values[name]
            if tensor.dtype!=torch.float32 or tensor.shape!=parameter.shape or tensor.device!=parameter.device:
                raise ValueError('Persistent full FP32 moment geometry/device differs')
            sizes[parameter.device.index]+=tensor.numel()*tensor.element_size()
    expected=[0,0] if update==1 else [n*8 for n in layout['device_parameters']]
    if sizes!=expected:raise ValueError('Complete persistent AdamW bytes must be resident before backward')
    return {'update':update,'microstep':microstep,'state_id':state_id,'state_entries':expected_entries,
      'resident_moment_bytes_by_device':sizes,'previous_optimizer_step':update-1,
      'backward_not_yet_executed_for_this_microstep':True,'full_state_resident':update>1}


def verify_allocator_record(value):
    """Recalculate recorded segment evidence without a CUDA query."""
    if (value.get('backend')!='native' or value.get('configuration')!=ALLOCATOR
            or type(value.get('new_probe_allocations')) is not int or value['new_probe_allocations']!=0):
        raise ValueError('Actual native expandable allocator record required')
    rows=value['segments'];devices=value['devices']
    if len(devices)!=2 or {r['device'] for r in devices}!={0,1}:raise ValueError('Both recorded allocator devices required')
    for row in rows:
        if (type(row.get('device')) is not int or row['device'] not in [0,1]
                or type(row.get('is_expandable')) is not bool
                or any(type(row.get(k)) is not int or row[k]<0 for k in ['total_size','allocated_size','active_size'])
                or not row['allocated_size']<=row['active_size']<=row['total_size']):
            raise ValueError('Recorded allocator segment differs')
    for d in devices:
        parts=[r for r in rows if r['device']==d['device'] and r['is_expandable'] and r['allocated_size']>0]
        if (not parts or d['expandable_live_segments']!=len(parts)
                or d['expandable_allocated_bytes']!=sum(r['allocated_size'] for r in parts)
                or any(type(d.get(k)) is not int or d[k]<0 for k in ['inactive_split_bytes','allocated_bytes','reserved_bytes'])
                or not d['expandable_allocated_bytes']<=d['allocated_bytes']<=d['reserved_bytes']):
            raise ValueError('Recorded live expandable allocations or native statistics differ')
    return True
