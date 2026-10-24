"""V2 training-only recipe/qualification contract and reusable fit instrumentation."""
from pathlib import Path
import math
from .io_utils import digest
from .composite_generalized_interface import PROFILE,input_ids,action_text
from .composite_behavior import training_rows,fit_behavior

MODE='generalized-qwen-rank32-training-v2'
RECIPE={'steps':64,'accumulation':2,'learning_rate':0.0005,'seed':202610212,'clip_norm':1.0}


def validate_plan_metadata(plan,cfg,data_release,workspace):
    """Source/data metadata only; never authorizes an allocation or a fit."""
    for key,value in RECIPE.items():
        if type(cfg['behavior_training'].get(key)) is not type(value) or cfg['behavior_training'][key]!=value:
            raise PermissionError('Registered complete supervised recipe changed')
    if (cfg['qwen']['rank'],cfg['qwen']['alpha'],cfg['qwen']['layers'])!=(32,64,64):raise PermissionError('Full all-block rank32 capacity required')
    phase=plan.get('phase');updates={'qualification':1,'full_fit':64}
    if phase not in updates or type(plan.get('optimizer_updates')) is not int or plan['optimizer_updates']!=updates[phase]:
        raise PermissionError('Distinct scratch qualification or full64update fit required')
    sampling='longest_complete_eligible_training_examples' if phase=='qualification' else 'registered_seeded_training_sampler'
    if (plan.get('native_interface_profile')!=PROFILE or plan.get('sampling')!=sampling
        or plan.get('qualification_warm_start_used') is not False or plan.get('validation_optimizer_updates')!=0
        or plan.get('heldout_optimizer_updates')!=0):raise PermissionError('Training/native/data-role boundary changed')
    if (data_release.get('workspace')!=workspace or data_release.get('heldout_content_read') is not False
        or data_release.get('data_admission',{}).get('training_and_development_admitted') is not True
        or data_release.get('data_admission',{}).get('public_fixture_test_mode',False) is not False
        or plan.get('data_release_sha256')!=digest(data_release)):
        raise PermissionError('Exact complete admitted V2 data/workspace freeze required')
    if set(data_release.get('cohorts',{}))!={'training','development'}:raise PermissionError('Both whole nonsealed cohorts required')
    if phase=='full_fit':
        parent=plan.get('qualification',{})
        if (parent.get('complete_training_qualification_audited') is not True or parent.get('owned_compute_released') is not True
            or parent.get('data_release_sha256')!=plan['data_release_sha256']
            or parent.get('workspace')!=workspace or parent.get('optimizer_updates')!=1
            or parent.get('checkpoint_initializes_full_fit') is not False):
            raise PermissionError('Full fit requires an exact audited/released scratch qualification')
    return phase


def validate_plan(plan,cfg,data_release,workspace,*,qualification_evidence=None):
    phase=validate_plan_metadata(plan,cfg,data_release,workspace)
    if phase=='full_fit':
        from .composite_v2_model_audit import require_qualification_evidence
        require_qualification_evidence(qualification_evidence,plan['qualification'])
    return phase


def token_plan(rows,tokenizer,cfg):
    """Actual admitted V2 rows determine counts; never assume old720/228values."""
    eligible=training_rows(rows);result=[]
    for row in eligible:
        prefix=input_ids(tokenizer,row['text'])
        target=tokenizer.encode(action_text(row['teacher_action']),add_special_tokens=False)+[tokenizer.eos_token_id]
        if tokenizer.eos_token_id is None or len(target)>160 or len(prefix)+len(target)-1>6144:
            raise ValueError('Complete training input/target exceeds the unchanged native ceiling')
        result.append({'state_id':row['state_id'],'input_token_ids':prefix,'target_token_ids':target,
                       'combined_tokens':len(prefix)+len(target)-1})
    if len({r['state_id'] for r in result})!=len(result):raise ValueError('Duplicate eligible state identities')
    if len(result)<2:raise ValueError('Two complete native training examples required for accumulation2')
    longest=sorted(result,key=lambda row:(-row['combined_tokens'],row['state_id']))[:2]
    return {'eligible_training_states':len(result),'source_training_states':len(rows),'records':result,
            'qualification_state_ids':[r['state_id'] for r in longest],
            'model_forwards':0,'heldout_content_read':False}


def run_fit(actor,rows,cfg,plan,output,guard,progress,persist):
    """Called only inside a separately admitted model worker; no provider API."""
    from .composite_generalized_data import require_runtime
    require_runtime()
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():
        raise PermissionError('Current Databricks training authority requires one BF16-capable visible GPU')
    return instrumented_fit(actor,rows,cfg,plan,output,guard,progress,persist,
                            token_planner=token_plan,row_selector=training_rows,fitter=fit_behavior,devices=(0,))


def instrumented_fit(actor,rows,cfg,plan,output,guard,progress,persist,*,token_planner,row_selector,fitter,devices,export_profile=None):
    """Shared raw training instrumentation; each caller enforces its own runtime/data authority."""
    if getattr(actor,'native_interface_profile',None)!=PROFILE:raise PermissionError('V2 native actor required')
    if getattr(actor,'v2_training_plan',None)!=plan:raise PermissionError('Bind the same validated stage plan to the actor')
    import torch
    from .composite_v2_model_audit import capture_live_backbone
    from .composite_v2_raw_exports import validate_export_plan,runtime_evidence,write_optimizer_state,BypassWriter
    if export_profile is None:validate_export_plan(plan)
    else:validate_export_plan(plan,expected_profile=export_profile)
    from .composite_training import tensor_digest
    tokens=token_planner(rows,actor.tokenizer,cfg);persist('TRAINING_TOKEN_PLAN.json',tokens)
    by_id={r['state_id']:r for r in row_selector(rows)}
    qualification=[by_id[key] for key in tokens['qualification_state_ids']] if plan['phase']=='qualification' else None
    from .composite_task_memory_profile import MODE as MEMORY_MODE,stress_schedule,allocator_evidence,resident_optimizer_evidence
    memory_mode=plan.get('version')==MEMORY_MODE
    stateful=memory_mode and plan['phase']=='qualification'
    if stateful:
        schedule=stress_schedule(tokens);qualification=[by_id[key] for key in schedule['state_ids']]
        persist('STATEFUL_TRAINING_SCHEDULE.json',schedule)
    ledger={'optimizer_steps_attempted':0,'optimizer_steps_completed':0,'backward_microsteps_completed':0,
            'layer_forward_invocations':[0]*64,'backward_recomputation_invocations':[0]*64,'in_backward':False}
    persist('BACKBONE_BEFORE.json',actor.frozen_proof())
    persist('BACKBONE_LIVE_BEFORE.json',capture_live_backbone(actor))
    def memory():
        records=[]
        for index in devices:
            free,total=torch.cuda.mem_get_info(index)
            records.append({'device':index,'free_bytes':free,'total_bytes':total,'allocated_bytes':torch.cuda.memory_allocated(index),
                'reserved_bytes':torch.cuda.memory_reserved(index),'peak_allocated_bytes':torch.cuda.max_memory_allocated(index),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(index)})
        return records[0] if devices==(0,) else records
    handles=[]
    for index,layer in enumerate(actor.text.layers):
        def before(module,args,index=index):
            guard();ledger['layer_forward_invocations'][index]+=1
            if ledger['in_backward']:ledger['backward_recomputation_invocations'][index]+=1
        handles.append(layer.register_forward_pre_hook(before))
    def observer(event,**value):
        guard()
        if event=='after_configure':
            count=sum(p.numel() for p in value['parameters'])
            if count!=233455616 or len(actor.adapters)!=496:raise ValueError('Full agreed real adapter capacity differs')
            persist('ADAPTER_INITIAL_HASHES.json',{name:tensor_digest(t) for name,t in value['initial_weights'].items()})
            persist('ADAPTER_CAPACITY_RUNTIME.json',actor.capacity_proof)
            if export_profile is not None:
                group=value['optimizer'].param_groups[0]
                persist('OPTIMIZER_CONFIG.json',{key:group[key] for key in
                    ['lr','betas','eps','weight_decay','foreach','maximize','capturable','differentiable']})
            if memory_mode:
                from .composite_v2_raw_exports import write_initial_A_state
                if len(value['optimizer'].state)!=0:raise ValueError('Fresh qualification optimizer must start empty')
                persist('OPTIMIZER_INITIAL_STATE.json',{'state_entries':0,'parameter_tensors':992})
                if stateful:persist('INITIAL_A_EXPORT.json',write_initial_A_state(actor.adapters,output))
                persist('ALLOCATOR_AFTER_INSTALL.json',allocator_evidence(torch.cuda))
        elif event=='before_microstep':
            ledger['in_backward']=False
            for index in devices:
                allocated=torch.cuda.memory_allocated(index)
                if devices==(0,):torch.cuda.empty_cache()
                else:
                    with torch.cuda.device(index):torch.cuda.empty_cache()
                if torch.cuda.memory_allocated(index)!=allocated:raise RuntimeError('Idle reclamation changed live training allocations')
            if memory_mode:
                item=resident_optimizer_evidence(value['optimizer'],actor.adapters,actor.task_training_context['adapter_layout'],
                  value['update'],value['microstep'],value['row']['state_id'])
                persist('training/resident_optimizer_before_'+str(ledger['backward_microsteps_completed']+1).zfill(4)+'.json',item)
        elif event=='after_forward':ledger['in_backward']=True
        elif event=='after_backward':
            ledger['in_backward']=False
            gradients=[]
            for name,module in actor.adapters.items():
                for part in ['A','B']:
                    p=getattr(module,part)
                    if p.grad is None or not torch.isfinite(p.grad).all():raise ValueError('Missing/nonfinite full-block gradient')
                    gradients.append({'name':name+'.'+part,'device':str(p.device),'dtype':str(p.grad.dtype),
                                      'norm':float(p.grad.float().norm())})
            if any(p.grad is not None or p.requires_grad for _,p in actor.base_references):raise ValueError('Backbone received gradients')
            ledger['backward_microsteps_completed']+=1
            persist('training/backward_'+str(ledger['backward_microsteps_completed']).zfill(4)+'.json',
                    {'update':value['update'],'microstep':value['microstep'],'state_id':value['row']['state_id'],
                     'loss':value['loss'],'gradients':gradients,'memory':memory()})
            if memory_mode:
                persist('training/allocator_after_backward_'+str(ledger['backward_microsteps_completed']).zfill(4)+'.json',allocator_evidence(torch.cuda))
        elif event=='before_optimizer_step':
            ledger['optimizer_steps_attempted']+=1
            if stateful:
                from .composite_v2_raw_exports import write_gradient_state
                name='GRADIENT_STEP_'+str(value['update']).zfill(3)
                persist(name+'_EXPORT.json',write_gradient_state(actor.adapters,output,name+'.pt'))
            elif export_profile is not None and value['update']==plan['optimizer_updates']:
                from .composite_v2_raw_exports import write_gradient_state
                persist('GRADIENT_EXPORT.json',write_gradient_state(actor.adapters,output))
        elif event=='after_optimizer_step':
            ledger['optimizer_steps_completed']+=1
            optimizer=value['optimizer']
            if len(optimizer.state)!=len(value['parameters']):raise ValueError('Incomplete AdamW state')
            for parameter,state in optimizer.state.items():
                for key in ['exp_avg','exp_avg_sq']:
                    if state[key].shape!=parameter.shape or state[key].dtype!=torch.float32 or state[key].device!=parameter.device or not torch.isfinite(state[key]).all():
                        raise ValueError('AdamW moment precision/shape/device differs')
            if value['update']==plan['optimizer_updates']:
                proof=write_optimizer_state(optimizer,actor.adapters,output)
                persist('OPTIMIZER_EXPORT.json',proof)
            extra={'gradient_norm_before_clipping':value['gradient_norm']} if export_profile is not None else {}
            persist('training/update_'+str(value['update']).zfill(3)+'.json',{**ledger,'memory':memory(),**extra})
            if stateful:
                progress('stateful_memory_update_complete',optimizer_steps_completed=ledger['optimizer_steps_completed'],
                  backward_microsteps_completed=ledger['backward_microsteps_completed'],counts_are_lower_bounds_until_controlled_exit=True)
    try:
        result=fitter(actor,rows,{'name':'all_blocks','scope':'all_blocks'},cfg,output,guard,progress,
                            observer=observer,qualification_rows=qualification)
        actor.verify_frozen()
        persist('BACKBONE_AFTER.json',actor.frozen_proof())
        persist('BACKBONE_LIVE_AFTER.json',capture_live_backbone(actor))
        persist('RUNTIME_RAW.json',runtime_evidence(actor))
        expected=plan['optimizer_updates']
        if ledger['optimizer_steps_completed']!=expected or ledger['backward_microsteps_completed']!=expected*2:
            raise ValueError('Exact registered update/microstep count differs')
        saved=torch.load(Path(output)/'behavior.pt',map_location='cpu',weights_only=True)
        current=actor.adapter_state()
        if (saved['config']!=actor.cfg or saved.get('native_interface_profile')!=PROFILE
            or set(saved['weights'])!=set(current['weights'])
            or any(tensor_digest(saved['weights'][name])!=tensor_digest(t) for name,t in current['weights'].items())):
            raise ValueError('Full saved checkpoint/profile readback differs')
        texts=[by_id[key]['text'] for key in tokens['qualification_state_ids']]
        raw_bypass=BypassWriter(output)
        bypass=actor.bypass_check(texts,persist_logits=raw_bypass);persist('BYPASS_CHECK.json',bypass)
        persist('BYPASS_RAW_MANIFEST.json',raw_bypass.manifest())
        if bypass['pass'] is not True:raise ValueError('Exact OFF/nonzero ON mechanical admission failed')
        return {**result,'phase':plan['phase'],'full_fit_completed':plan['phase']=='full_fit',
                'qualification_checkpoint_initializes_full_fit':False,'training_progress':ledger,
                'native_interface_profile':PROFILE,'behavioral_validation':None}
    finally:
        for handle in handles:handle.remove()
        ledger['in_backward']=False;persist('TRAINING_TERMINAL_JOURNAL.json',dict(ledger))
