"""Function-preserving ordinary LoRA16 to LoRA32 embedding; no base update."""
import copy
import math

NEW_A_SEED = 260304941


def normalize(value):
    if isinstance(value,dict):return {str(k):normalize(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [normalize(v) for v in value]
    if isinstance(value,set):return sorted(normalize(v) for v in value)
    if hasattr(value,'value'):return normalize(value.value)
    return value


def audit_config(config):
    data=normalize(config.to_dict() if hasattr(config,'to_dict') else config)
    issues=[]
    if data.get('r')!=16 or data.get('lora_alpha')!=32:issues.append('Expected uniform rank16/alpha32')
    for field in ('use_rslora','use_dora','lora_bias','fan_in_fan_out','use_qalora'):
        if data.get(field,False):issues.append('Unsupported '+field)
    for field in ('rank_pattern','alpha_pattern','modules_to_save','layer_replication','trainable_token_indices',
                  'alora_invocation_tokens','loftq_config','eva_config','corda_config','megatron_config'):
        if data.get(field):issues.append('Unsupported nonempty '+field)
    if data.get('bias','none')!='none':issues.append('Unsupported trainable bias')
    if data.get('task_type')!='CAUSAL_LM':issues.append('Expected CAUSAL_LM')
    if data.get('init_lora_weights',True) not in (True,False,'gaussian'):
        issues.append('Unsupported initializer that may mutate base weights')
    if not data.get('target_modules'):issues.append('Missing fixed target modules')
    dropout=data.get('lora_dropout',0)
    if not isinstance(dropout,(int,float)) or not 0<=dropout<1:issues.append('Unsupported dropout')
    return {'passed':not issues,'issues':issues,'actual_config':data,
            'expected_scaling':2.0,'expansion':{'rank':32,'alpha':64,'new_A_seed':NEW_A_SEED,
            'new_A_distribution':'Uniform[-1/sqrt(in_features),+1/sqrt(in_features)]',
            'new_B_columns':'Exactly zero'},
            'stochastic_limit':'Matched nominal seeds do not guarantee identical floating-point kernels or stochastic training paths across ranks.'}


def layer_scalings(model,rank,alpha):
    result=[]
    for name,module in model.named_modules():
        if hasattr(module,'lora_A') and 'default' in module.lora_A:
            if 'default' not in module.lora_B:raise RuntimeError('Missing paired B factor')
            a,b=module.lora_A['default'],module.lora_B['default']
            if a.weight.ndim!=2 or b.weight.ndim!=2 or a.weight.shape[0]!=rank or b.weight.shape[1]!=rank:
                raise RuntimeError('Unsupported factor geometry: '+name)
            if a.bias is not None or b.bias is not None:raise RuntimeError('Unsupported factor bias')
            actual=float(module.scaling['default'])
            if module.r['default']!=rank or module.lora_alpha['default']!=alpha or actual!=2.0:
                raise RuntimeError('Unexpected effective per-layer scaling: '+name)
            result.append({'module':name,'A_shape':list(a.weight.shape),'B_shape':list(b.weight.shape),'scaling':actual})
    if not result:raise RuntimeError('No ordinary LoRA factor layers found')
    return result


def expand_on_same_base(model16):
    import torch
    from peft import get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
    if list(model16.active_adapters)!=['default']:raise RuntimeError('Unexpected active original adapter')
    config16=model16.peft_config['default']
    audit=audit_config(config16)
    if not audit['passed']:raise RuntimeError('Unsupported H2 LoRA configuration: '+str(audit['issues']))
    audit['old_layer_geometry']=layer_scalings(model16,16,32)
    old_baseline_trainable=sum(value.numel() for value in model16.parameters() if value.requires_grad)
    old={name:value.detach().cpu().clone() for name,value in get_peft_model_state_dict(model16).items()}
    old_factor_capacity=sum(value.numel() for value in old.values())
    if not old or any(not (name.endswith('.lora_A.weight') or name.endswith('.lora_B.weight')) for name in old):
        raise RuntimeError('Unsupported non-factor adapter state')
    config32=copy.deepcopy(config16);config32.r=32;config32.lora_alpha=64;config32.inference_mode=False
    # unload removes adapter wrappers without merging or modifying base weights.
    base=model16.unload()
    model32=get_peft_model(base,config32)
    target=get_peft_model_state_dict(model32)
    if set(target)!=set(old):raise RuntimeError('Expanded adapter target names differ')
    generator=torch.Generator(device='cpu').manual_seed(NEW_A_SEED)
    expanded={};checks=[]
    for name,value in sorted(old.items()):
        expected=target[name]
        if expected.dtype!=value.dtype:raise RuntimeError('Factor dtype changed: '+name)
        if name.endswith('.lora_A.weight'):
            if value.shape[0]!=16 or list(expected.shape)!=[32,value.shape[1]]:
                raise RuntimeError('A shape mismatch: '+name)
            new=torch.empty((32,value.shape[1]),dtype=value.dtype,device='cpu')
            new[:16]=value
            new[16:].uniform_(-1/math.sqrt(value.shape[1]),1/math.sqrt(value.shape[1]),generator=generator)
            exact=torch.equal(new[:16],value)
            check={'name':name,'old_block_exact':exact,'new_A_rows':16,'finite':bool(torch.isfinite(new).all())}
        else:
            if value.shape[1]!=16 or list(expected.shape)!=[value.shape[0],32]:
                raise RuntimeError('B shape mismatch: '+name)
            new=torch.zeros((value.shape[0],32),dtype=value.dtype,device='cpu');new[:,:16]=value
            exact=torch.equal(new[:,:16],value)
            check={'name':name,'old_block_exact':exact,'new_B_exact_zero':bool(torch.count_nonzero(new[:,16:])==0),
                   'finite':bool(torch.isfinite(new).all())}
        if not exact or not check['finite'] or check.get('new_B_exact_zero') is False:
            raise RuntimeError('Expanded state integrity failed: '+name)
        expanded[name]=new;checks.append(check)
    loaded=set_peft_model_state_dict(model32,expanded,adapter_name='default')
    if loaded.unexpected_keys:raise RuntimeError('Unexpected adapter load keys: '+str(loaded.unexpected_keys))
    actual=get_peft_model_state_dict(model32)
    if any(not torch.equal(actual[name].detach().cpu(),value) for name,value in expanded.items()):
        raise RuntimeError('Loaded expanded factors differ from exact embedded state')
    audit['new_layer_geometry']=layer_scalings(model32,32,64)
    audit['factor_checks']=checks
    audit['parameter_counts']={'old_adapter_factor_capacity_numel':old_factor_capacity,
        'old_frozen_baseline_actual_trainable_numel':old_baseline_trainable,
        'new_adapter_factor_capacity_numel':sum(value.numel() for value in actual.values()),
        'rank32_actual_trainable_numel':sum(value.numel() for value in model32.parameters() if value.requires_grad),
        'factor_capacity_definition':'Sum of all canonical LoRA A/B factor state tensor elements; independent of requires_grad.',
        'trainable_definition':'Actual unique model parameter elements with requires_grad true.'}
    audit['effective_delta_equivalence']={
        'proved_algebraically':True,'scaling_before':2.0,'scaling_after':2.0,
        'basis':'A32=[A16;Anew], B32=[B16,0]. Exact old blocks and zero extra B imply scaled B32@A32 == scaled B16@A16.',
        'full_dense_delta_materialized':False,'finite_precision_generation_checked_separately':True}
    return model32,base,audit


def gradient_observation(model,update):
    """Read gradients produced by the existing optimizer update; no backward call."""
    import torch
    sums={key:[] for key in ('A_old','A_new','B_old','B_new')};counts={key:0 for key in sums}
    missing=[]
    for name,param in model.named_parameters():
        if '.lora_A.default.weight' in name:blocks=(('A_old',param.grad[:16] if param.grad is not None else None),
                                                     ('A_new',param.grad[16:] if param.grad is not None else None))
        elif '.lora_B.default.weight' in name:blocks=(('B_old',param.grad[:,:16] if param.grad is not None else None),
                                                       ('B_new',param.grad[:,16:] if param.grad is not None else None))
        else:continue
        if param.grad is None:missing.append(name)
        for key,grad in blocks:
            if grad is not None:sums[key].append(grad.detach().float().square().sum());counts[key]+=1
    groups={}
    for key,values in sums.items():
        squared=torch.stack(values).sum() if values else None
        groups[key]={'factor_gradients':counts[key],'squared_norm':float(squared) if squared is not None else None,
                     'norm':float(squared.sqrt()) if squared is not None else None}
    return {'update':update,'source':'Existing accumulated parameter gradients after clipping/optimizer.step; no additional backward',
            'groups':groups,'missing_gradients':missing,'observational_only':True}
