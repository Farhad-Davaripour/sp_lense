"""V4-only audited-row selection and native-call supervised fitting.

The historical task-only loader remains untouched. This path accepts only the
opaque independently replayed new behavioral data parent and uses the list
interface throughout; it does not use detector predictions as fitting labels.
"""
import os
from pathlib import Path
from .io_utils import digest,sha
from .composite_bounded_behavior_interface import PROFILE,input_ids,action_text
from .composite_bounded_sampling_fit_profile import MODE,PLAN,RAW_PROFILE,FIT_RAW_PROFILE,stress_schedule,allocator_evidence
from .composite_task_skill_training import (PARAMETERS,PROJECTIONS,adapter_layout,memory_admission,
    live_resources,postload_runtime,validate_recipe)


def validate_rows(rows):
    if not isinstance(rows,list) or len(rows)<2 or len({r.get('state_id') for r in rows})!=len(rows):
        raise PermissionError('Complete distinct new V4 fitting rows required')
    for row in rows:
        if (row.get('split')!='training' or row.get('public_fixture') is not False
                or row.get('training_eligible') is not True or row.get('terminal') is not False
                or row.get('teacher_action') is None or row.get('task_only') is True):
            raise PermissionError('Only newly audited training-role nonterminal V4 targets may fit')
    return rows


def token_plan(rows,tokenizer,cfg):
    validate_rows(rows);records=[]
    for row in rows:
        prefix=input_ids(tokenizer,row['text'])
        target=tokenizer.encode(action_text(row['teacher_action']),add_special_tokens=False)+[tokenizer.eos_token_id]
        if (tokenizer.eos_token_id is None or not prefix or len(target)>160
                or len(prefix)+len(target)-1>6144):
            raise ValueError('Full V4 input/target exceeds unchanged native ceiling; never truncate')
        records.append({'state_id':row['state_id'],'input_token_ids':prefix,'target_token_ids':target,
          'combined_tokens':len(prefix)+len(target)-1})
    return {'eligible_training_states':len(rows),'source_training_states':len(rows),'records':records,
      'qualification_state_ids':[r['state_id'] for r in sorted(records,key=lambda r:(-r['combined_tokens'],r['state_id']))[:2]],
      'selection':'complete_new_training_only_V4_all_registered_phases_no_detector_filter',
      'native_interface_profile':PROFILE,'model_forwards':0,'heldout_content_read':False}


def authorized_load_plan(root,commit,qwen):
    from .composite_runtime import require_scientific_runtime,platform
    from .composite_bounded_sampling_fit_source import validate_source
    require_scientific_runtime()
    actual=Path(os.environ['RESEARCH4_BOUNDED_SAMPLING_FIT_SOURCE_ROOT']).resolve()
    if (platform()!='kaggle' or os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1'
            or not actual.is_relative_to(Path('/kaggle/temp/research4_bounded_sampling_fit_inputs').resolve())
            or actual.name!='source' or actual!=Path(root).resolve()):
        raise PermissionError('Exact owned Kaggle V4 fitting pair/source required')
    stage,cfg,data=validate_source(actual,commit)
    if qwen!=cfg['qwen'] or os.environ.get('RESEARCH4_BOUNDED_SAMPLING_FIT_PLAN_SHA256')!=sha(actual/'inputs'/PLAN):
        raise PermissionError('Actual V4 fitting plan/model configuration differs')
    from .composite_frozen_probe import verified_model_context
    context=verified_model_context(actual,commit,qwen,stage,sha(actual/'inputs'/PLAN))
    context.update(stage=stage,full_config=cfg,bounded_data_evidence=data,
      adapter_layout=adapter_layout(context['metadata'],context['plan']))
    return context


def selected_rows(actor,rows):
    context=getattr(actor,'bounded_training_context',None)
    if context is None:raise PermissionError('Separate admitted V4 constructor/data authority required')
    from .composite_bounded_sampling_reader import require_approved_sampling_data
    require_approved_sampling_data(context['bounded_data_evidence'])
    if digest(rows)!=digest(context['bounded_data_evidence'].fitting_rows()):
        raise PermissionError('Every fitting row must equal the complete audited new training freeze')
    return validate_rows(rows)


def fit_backend(actor,rows,placement,cfg,output,guard,progress,observer=None,qualification_rows=None):
    from .composite_behavior import fit_selected_rows
    rows=selected_rows(actor,rows);context=actor.bounded_training_context;stage=context['stage']
    if cfg!=context['full_config']:raise PermissionError('New V4 fitting recipe differs')
    if stage['phase']=='qualification':
        expected=stress_schedule(token_plan(rows,actor.tokenizer,cfg))['state_ids']
        if qualification_rows is None or [r['state_id'] for r in qualification_rows]!=expected:
            raise PermissionError('Exact new-data long/short/long qualification schedule required')
    else:
        approved_full_fit_rows(actor,rows,cfg)
    return fit_selected_rows(actor,rows,placement,cfg,output,guard,progress,observer,qualification_rows)


def approved_full_fit_rows(actor,rows,cfg):
    """The opaque new CPU proof supplies all128 occurrences; never run IID RNG."""
    rows=selected_rows(actor,rows);context=actor.bounded_training_context;stage=context['stage']
    if (stage['version']!=MODE or stage['phase']!='full_fit'
            or stage['sampling']!='approved_48anchors80uniform_exact128_ids_v1'
            or cfg!=context['full_config']):
        raise PermissionError('The distinct approved-sampler full fit is required')
    plan=context['bounded_data_evidence'].fitting_draw_plan()
    ids=plan['state_ids'];by_id={r['state_id']:r for r in rows}
    if (digest(plan)!=stage['approved_draw_plan_sha256'] or plan.get('coverage_passed') is not True
            or len(ids)!=128 or any(state not in by_id for state in ids)
            or plan.get('draws')!=128 or plan.get('updates')!=64 or plan.get('accumulation')!=2):
        raise PermissionError('All exact approved128 occurrences/coverage must match the frozen data proof')
    return [by_id[state] for state in ids]


def run_fit(actor,rows,cfg,plan,output,guard,progress,persist):
    context=getattr(actor,'bounded_training_context',None)
    if context is None or context['stage']!=plan:raise PermissionError('Exact new V4 stage and constructor required')
    actual=authorized_load_plan(context['source_root'],context['source_commit'],actor.cfg)
    if actual['stage']!=plan:raise PermissionError('Source/runtime changed before V4 fit')
    selected_rows(actor,rows)
    import torch
    from importlib.metadata import version
    persist('BOUNDED_TRAINING_HARDWARE.json',postload_runtime(actor,torch.cuda,str(torch.__version__),
      {name:version(name) for name in ['transformers','bitsandbytes','accelerate','safetensors','numpy','psutil']}))
    persist('ALLOCATOR_BEFORE_INSTALL.json',allocator_evidence(torch.cuda))
    from .composite_v2_training import instrumented_fit
    return instrumented_fit(actor,rows,cfg,plan,output,guard,progress,persist,
      token_planner=token_plan,row_selector=lambda values:selected_rows(actor,values),fitter=fit_backend,
      devices=(0,1),export_profile=RAW_PROFILE if plan['phase']=='qualification' else FIT_RAW_PROFILE)
