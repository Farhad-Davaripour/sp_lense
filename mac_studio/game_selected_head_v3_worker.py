"""Owned TRAIN-only real equivalence diagnostic, never a research candidate."""
from functools import partial
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import gc

import psutil
import game_selected_head_v3_plan as plans
import game_selected_head_v3_loss as selected_loss
from game_selected_head_v3_compare import compare_gradients,compare_logits,loss_equivalence
import game_all_layers_layout as layout
from game_all_layers_loss import make_batch
from game_training_renderer import canonical,render


def once(path,value):
    with path.open('x')as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())


def phase(run,name,mx=None,**extra):
    value={'phase':name,'observed_epoch':time.time(),'available_bytes':psutil.virtual_memory().available,
        'swap_used_bytes':psutil.swap_memory().used,'process_rss_bytes':psutil.Process().memory_info().rss,**extra}
    if mx is not None:value.update({'mlx_active_bytes':int(mx.get_active_memory()),'mlx_cache_bytes':int(mx.get_cache_memory()),'mlx_peak_bytes':int(mx.get_peak_memory())})
    with (run/'phases.jsonl').open('a')as stream:stream.write(canonical(value)+'\n');stream.flush();os.fsync(stream.fileno())
    temporary=run/'.phase.tmp'
    with temporary.open('w')as stream:json.dump(value,stream,allow_nan=False)
    os.replace(temporary,run/'phase.json')


def parameter_digest(model):
    import numpy as np
    from mlx.utils import tree_flatten
    result=hashlib.sha256()
    for name,array in sorted(tree_flatten(model.trainable_parameters())):
        result.update(name.encode());result.update(str(array.shape).encode());result.update(np.array(array).tobytes())
    return result.hexdigest()


def run_diagnostic(plan_path,plan_sha,run,owner):
    plan,cohort,data=plans.read_plan(plan_path,plan_sha)
    counters={'diagnostic_checks_completed':0,'diagnostic_checks_total':3,'diagnostic_optimizer_updates':0}
    def mark(name,mx=None,**extra):phase(run,name,mx,**{**counters,**extra})
    lock=plans.PROJECT/'.local/validation/qwen38_shared_v1/inference.lock/owner.json'
    ownership=json.loads(lock.read_text());permit=json.loads((plans.ROOT/'authorities/equivalence_rank4.json').read_text())
    if (ownership.get('pid'),ownership.get('owner'),ownership.get('kind'))!=(os.getppid(),owner,'selected_head_v3_equivalence'):
        raise ValueError('Live diagnostic supervisor ownership differs')
    if (permit.get('owner'),permit.get('plan_sha256'),permit.get('run_dir'))!=(owner,plan_sha,str(run)):
        raise ValueError('One-use diagnostic authority differs')
    if platform.system()!='Darwin'or platform.machine()!='arm64':raise RuntimeError('Only the configured local Mac is permitted')
    mark('cpu_preflight')
    versions={name:importlib.metadata.version(name)for name in('mlx','mlx-lm','transformers','numpy')}
    if versions!={'mlx':'0.32.4','mlx-lm':'0.32.2','transformers':'5.22.2','numpy':'2.5.4'}:raise ValueError('Numerical versions differ')
    sources=layout.verify_installed_lora_sources()
    package=importlib.metadata.distribution('mlx-lm')
    if plans.sha(Path(package.locate_file('mlx_lm/models/gated_delta.py')))!='f69e837ea68641dbb06fdb5dcc024a9a7e1d9353f08ae58c05275253a0c39428':
        raise ValueError('GDN primitive source differs')
    import qwen38_local as local
    local.prepare_local();os.environ.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','DO_NOT_TRACK':'1'})
    if plans.sha(local.MANIFEST)!=plans.MODEL_SHA:raise ValueError('Base manifest differs')
    local.verify()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(local.MODEL),local_files_only=True,trust_remote_code=False)
    schema=json.loads((data/'action_schema.json').read_text());prepared=[];maximum=0
    for filename in('train768.jsonl','dev144.jsonl'):
        rows=[json.loads(line)for line in(data/filename).read_text().splitlines()]
        for index,row in enumerate(rows):
            ids,prefix=render(tokenizer,row,schema);selected_loss.gather_positions(prefix,len(ids),min(1024,1+32*((len(ids)+31)//32)))
            maximum=max(maximum,len(ids)-prefix)
            if filename=='train768.jsonl':prepared.append((ids,prefix))
    if maximum>32 or maximum!=plan['actual_maximum_suffix_tokens']:raise ValueError('Complete suffix preflight differs')
    for label in('short','long'):
        item=plan['inputs'][label];ids,prefix=prepared[item['training_index']]
        if (len(ids),prefix)!=(item['total_tokens'],item['prefix_tokens']):raise ValueError('Predeclared diagnostic token boundaries differ')
    once(run/'token_inputs.json',{label:{'training_index':plan['inputs'][label]['training_index'],
        'token_ids':prepared[plan['inputs'][label]['training_index']][0],
        'prefix_tokens':prepared[plan['inputs'][label]['training_index']][1]}for label in('short','long')})
    expected=layout.expected_adapter_layout(local.MODEL,4,64);projections=layout.projection_layout(local.MODEL,64)
    mark('before_mlx_import',all_912_suffixes_checked=True,maximum_suffix=maximum)
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten,tree_unflatten
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.tuner.trainer import grad_checkpoint
    mx.set_memory_limit(plan['mlx_memory_guideline_bytes'])
    if mx.distributed.init().size()!=1:raise RuntimeError('Single local device group required')
    mark('before_base_load',mx)
    model,_=load(str(local.MODEL),trust_remote_code=False);mark('after_base_load',mx)
    model.freeze();mx.random.seed(20261108)
    modules=dict(model.named_modules())
    for name,spec in projections.items():
        if type(modules.get(name))is not(nn.QuantizedLinear if spec['base_quantized']else nn.Linear):raise ValueError('Native projection type differs')
    linear_to_lora_layers(model,64,{'rank':4,'scale':2.0,'dropout':0.0})
    parameters=dict(tree_flatten(model.trainable_parameters()))
    if set(parameters)!=set(expected)or any(tuple(array.shape)!=expected[name]or array.dtype!=mx.float32 for name,array in parameters.items()):
        raise ValueError('Exact 992 F32 trainable layout differs')
    mx.eval(model.parameters());mark('after_lora_materialization_before_cache_clear',mx)
    if any(bool(mx.any(value!=0).item())for name,value in parameters.items()if name.endswith('.lora_b')):raise ValueError('Fresh B initialization was not zero')
    del parameters,modules
    selected_loss.assert_head_contract(model);grad_checkpoint(model.layers[0]);model.train()
    gdns=[module for _,module in model.named_modules()if type(module).__name__=='GatedDeltaNet']
    if len(gdns)!=48 or any(not module.training for module in gdns)or any(not block.training for block in model.layers):
        raise ValueError('All native GDN blocks must be training/reference mode before tracing')
    once(run/'run_config.json',{'schema':'research4.selected-head-diagnostic-run.v3','plan_sha256':plan_sha,
        'model':plans.MODEL_ID,'revision':plans.REVISION,'model_manifest_sha256':plans.MODEL_SHA,
        'prior_adapter_loaded':False,'initialization':'fresh_pinned_base','rank':4,'num_layers':64,'head_window':32,
        'versions':versions,'installed_sources':sources,'scope':'Diagnostic-only weights; not eligible for tuning or serving.'})
    mx.save_safetensors(str(run/'diagnostic_state0_weights.safetensors'),dict(tree_flatten(model.trainable_parameters())))
    mx.clear_cache();gc.collect();mark('after_setup_cache_clear_before_first_trace',mx)
    states=[];full_zero_gradient=None;update_count=0
    def capture(method,label,ids,prefix,gradients=True):
        batch,lengths=make_batch(mx,ids,prefix);count_expected=len(ids)-prefix
        before=parameter_digest(model);mark('before_'+label,mx,method=method,diagnostic_optimizer_updates=update_count,
            operation='compiled_gradient'if gradients else'forward_only',model_training=bool(model.training),
            gdn_training_count=sum(bool(module.training)for module in gdns))
        implementation=selected_loss.full_components_with_runtime if method=='full'else selected_loss.components_with_runtime
        def objective(m,b,l):
            total,count,logits=implementation(mx,nn,m,b,l)
            return total/count,(count,logits.astype(mx.float32))
        started=time.perf_counter();gradient_path=None
        if gradients:
            transformed=nn.value_and_grad(model,objective);state=[model.state,mx.random.state]
            @partial(mx.compile,inputs=state,outputs=state)
            def differentiated(b,l):return transformed(model,b,l)
            (mean,(count,logits)),gradient=differentiated(batch,lengths)
            mx.eval(model.state,mean,count,logits,gradient)
            flat=dict(tree_flatten(gradient))
            if set(flat)!=set(expected)or any(tuple(g.shape)!=expected[name]or g.dtype!=mx.float32 for name,g in flat.items()):
                raise ValueError('Exact all-992 gradient layout differs')
            gradient_path=run/(label+'_gradients.safetensors');mx.save_safetensors(str(gradient_path),flat)
        else:
            total,count,logits=implementation(mx,nn,model,batch,lengths);mean=total/count;mx.eval(mean,count,logits)
        mark('after_'+label+'_evaluation',mx,method=method)
        mean_value=float(mean.item());count_value=int(count.item())
        if not math.isfinite(mean_value)or mean_value<0 or count_value!=count_expected:raise ValueError('Target likelihood/count differs')
        logits_path=run/(label+'_target_logits.safetensors')
        # Only actual target rows are saved and compared; masked window extras
        # cannot dilute or contaminate the numerical equivalence statistics.
        valid_logits=logits[:,:count_expected,:].astype(mx.float32).reshape(count_expected,logits.shape[-1])
        mx.save_safetensors(str(logits_path),{'target_logits':valid_logits})
        if parameter_digest(model)!=before:raise ValueError('Weight state changed between equivalent paths')
        result={'label':label,'method':method,'mean_nll':mean_value,'target_tokens':count_value,'weight_state_sha256':before,
            'target_logits':{'path':plans.rel(logits_path),'sha256':plans.sha(logits_path),'shape':[count_expected,int(logits.shape[-1])]},
            'elapsed_seconds':time.perf_counter()-started}
        if gradients:
            result['gradients']={'path':plans.rel(gradient_path),'sha256':plans.sha(gradient_path)}
            check=compare_gradients(gradient_path,gradient_path,expected,plan['gradient_tolerances'],require_zero_a=label.startswith('state0'))
            layers={int(row['name'].split('.layers.')[1].split('.')[0])for row in check['tensors']if row['name'].endswith('.lora_b')and row['reference_l2']>0}
            a_layers={int(row['name'].split('.layers.')[1].split('.')[0])for row in check['tensors']if row['name'].endswith('.lora_a')and row['reference_l2']>0}
            a_count=sum(row['name'].endswith('.lora_a')and row['reference_l2']>0 for row in check['tensors'])
            if not check['passed']or layers!=set(range(64)):raise ValueError('Reference gradient finite/zero-A/all-layer coverage check failed')
            if label.startswith('state1')and a_count<1:raise ValueError('Post-update comparison did not exercise any nonzero A gradient')
            result['gradient_b_layers']=sorted(layers)
            result['nonzero_a_gradient_layers']=sorted(a_layers);result['nonzero_a_gradient_tensors']=a_count
            del gradient,flat,differentiated,transformed,state
        once(run/(label+'_capture.json'),result)
        del batch,lengths,mean,count,logits,valid_logits
        gc.collect();mx.clear_cache();mark('after_'+label+'_saved_and_released',mx)
        return result
    short=plan['inputs']['short'];ids,prefix=prepared[short['training_index']]
    optimizer=optim.Adam(learning_rate=0.00001,betas=[0.9,0.999],eps=1e-8,bias_correction=False)
    for state_index in(0,1):
        if state_index:
            mark('before_single_full_reference_adam_update',mx)
            saved=mx.load(str(full_zero_gradient));optimizer.update(model,tree_unflatten(list(saved.items())));mx.eval(model.state,optimizer.state)
            if int(optimizer.step.item())!=1:raise ValueError('Diagnostic Adam update count differs')
            if not all(bool(mx.all(mx.isfinite(value)).item())for _,value in tree_flatten(optimizer.state)if hasattr(value,'dtype')):
                raise ValueError('Diagnostic optimizer state is nonfinite')
            after_update=parameter_digest(model)
            if after_update==json.loads((run/'state0_full_capture.json').read_text())['weight_state_sha256']:
                raise ValueError('Adam step counter advanced without changing the diagnostic weights')
            updated=dict(tree_flatten(model.trainable_parameters()))
            b_layers={int(name.split('.layers.')[1].split('.')[0])for name,value in updated.items()
                if name.endswith('.lora_b')and bool(mx.any(value!=0).item())}
            if b_layers!=set(range(64))or not all(bool(mx.all(mx.isfinite(value)).item())for value in updated.values()):
                raise ValueError('The single diagnostic update did not produce finite nonzero B in every layer')
            once(run/'single_update_evidence.json',{'optimizer_step':1,'weight_state_sha256':after_update,
                'changed_from_zero_b_snapshot':True,'nonzero_b_weight_layers':sorted(b_layers),'all_trainable_weights_finite':True,
                'gradient_source':'state0_full_gradients.safetensors','eligible_for_tuning_or_serving':False})
            mx.save_safetensors(str(run/'diagnostic_state1_weights.safetensors'),updated)
            del updated
            update_count=1
            counters['diagnostic_optimizer_updates']=1
            del saved;gc.collect();mx.clear_cache();mark('after_single_full_reference_adam_update',mx)
        full=capture('full',f'state{state_index}_full',ids,prefix)
        selected=capture('selected',f'state{state_index}_selected',ids,prefix)
        if full['weight_state_sha256']!=selected['weight_state_sha256']:raise ValueError('Full/selected weight snapshots differ')
        loss=loss_equivalence(full['mean_nll'],selected['mean_nll'],full['target_tokens'],selected['target_tokens'],plan['gradient_tolerances'])
        gradients=compare_gradients(plans.path_for(full['gradients']['path']),plans.path_for(selected['gradients']['path']),expected,
            plan['gradient_tolerances'],require_zero_a=state_index==0)
        logits=compare_logits(plans.path_for(full['target_logits']['path']),plans.path_for(selected['target_logits']['path']),full['target_logits']['shape'])
        comparison={'state_index':state_index,'loss':loss,'gradients':gradients,'target_logits':logits,
            'passed':loss['passed']and gradients['passed']and logits['passed']}
        once(run/f'state{state_index}_comparison.json',comparison)
        if not comparison['passed']:raise ValueError('Predeclared full/selected loss/gradient/logit gate failed; no relaxation or next stage')
        states.append({'state_index':state_index,'passed':True,'comparison_sha256':plans.sha(run/f'state{state_index}_comparison.json'),
            'weight_state_sha256':full['weight_state_sha256'],'nonzero_a_gradient_tensors':full['nonzero_a_gradient_tensors']})
        counters['diagnostic_checks_completed']=len(states)
        mark(f'state{state_index}_equivalence_passed',mx)
        if state_index==0:full_zero_gradient=plans.path_for(full['gradients']['path'])
    model.eval()
    if any(module.training for module in gdns):raise ValueError('Long forward pair must use native eval mode')
    mark('before_long_train_forward_pair_eval_mode',mx)
    long=plan['inputs']['long'];ids,prefix=prepared[long['training_index']]
    full=capture('full','long_full_forward',ids,prefix,False);selected=capture('selected','long_selected_forward',ids,prefix,False)
    long_loss=loss_equivalence(full['mean_nll'],selected['mean_nll'],full['target_tokens'],selected['target_tokens'],plan['gradient_tolerances'])
    long_logits=compare_logits(plans.path_for(full['target_logits']['path']),plans.path_for(selected['target_logits']['path']),full['target_logits']['shape'])
    if full['weight_state_sha256']!=selected['weight_state_sha256']or not(long_loss['passed']and long_logits['passed']):
        raise ValueError('Long unchanged-context forward equivalence failed')
    counters['diagnostic_checks_completed']=3;mark('long_training_forward_equivalence_passed',mx)
    mark('before_final_base_file_verify',mx);local.verify()
    receipt={'schema':'research4.selected-head-equivalence-receipt.v3','status':'passed','plan_sha256':plan_sha,'states':states,
        'long_training_forward':{'loss':long_loss,'target_logits':long_logits},'rank':4,'num_layers':64,'gradient_tensors':992,
        'diagnostic_optimizer_updates':1,'paired_full_selected_comparisons_same_weight_state':True,
        'two_distinct_weight_snapshots_tested':True,'full_context_retained':True,'head_window':32,
        'dataset_modified':False,'development_or_heldout_used_for_model_calls':False,'eligible_for_tuning_or_serving':False,
        'base_files_verified_before_and_after':True,'original_deadline_epoch':plans.ORIGINAL_DEADLINE,
        'scope':'TRAIN-only numerical equivalence passed at tested snapshots/inputs; long optimized backward/memory/three-update compatibility is still unproven.'}
    once(run/'equivalence_receipt.json',receipt);mark('equivalence_complete_not_a_model_candidate',mx)
    return receipt


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--owner',required=True)
    args=parser.parse_args();run=args.run_dir.resolve()
    if not run.is_relative_to((plans.ROOT/'runs').resolve())or not run.is_dir():raise ValueError('Invalid new owned diagnostic directory')
    try:run_diagnostic(args.plan,args.plan_sha256,run,args.owner)
    except BaseException as exc:
        once(run/'worker_failure.json',{'status':'failed','error':f'{type(exc).__name__}: {exc}','epoch':time.time()});raise
