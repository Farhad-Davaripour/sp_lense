"""Owned full-context V4 chain-rule proof; never a training candidate."""
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time

import psutil
import game_chain_rule_v4_plan as plans
import game_chain_rule_v4_math as chain
from game_chain_rule_v4_compare import native_descriptor,verify_native_audit_copy,compare_covectors,exact_native_values
from game_selected_head_v3_compare import compare_gradients,compare_logits,loss_equivalence
import game_all_layers_layout as layout
from game_all_layers_loss import make_batch
from game_training_renderer import canonical,render


def once(path,value):
    with path.open('x')as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())


def parameter_digest(model):
    import numpy as np
    from mlx.utils import tree_flatten
    result=hashlib.sha256()
    for name,array in sorted(tree_flatten(model.trainable_parameters())):
        result.update(name.encode());result.update(str(array.shape).encode());result.update(np.array(array).tobytes())
    return result.hexdigest()


def run_proof(plan_path,plan_sha,run,owner):
    plan,cohort,data=plans.read_plan(plan_path,plan_sha)
    ownership=json.loads((plans.PROJECT/'.local/validation/qwen38_shared_v1/inference.lock/owner.json').read_text())
    permit=json.loads((plans.ROOT/'authorities/chain_rule_equivalence_rank4.json').read_text())
    if (ownership.get('pid'),ownership.get('owner'),ownership.get('kind'))!=(os.getppid(),owner,'chain_rule_v4_equivalence'):
        raise ValueError('Live V4 supervisor ownership differs')
    if (permit.get('owner'),permit.get('plan_sha256'),permit.get('run_dir'))!=(owner,plan_sha,str(run)):
        raise ValueError('One-use V4 authority differs')
    counters={'diagnostic_checks_completed':0,'diagnostic_checks_total':3,'diagnostic_optimizer_updates':0}
    def mark(name,mx=None,**extra):
        value={'phase':name,'observed_epoch':time.time(),**counters,'available_bytes':psutil.virtual_memory().available,
            'swap_used_bytes':psutil.swap_memory().used,'process_rss_bytes':psutil.Process().memory_info().rss,**extra}
        if mx is not None:value.update({'mlx_active_bytes':int(mx.get_active_memory()),'mlx_cache_bytes':int(mx.get_cache_memory()),'mlx_peak_bytes':int(mx.get_peak_memory())})
        with(run/'phases.jsonl').open('a')as stream:stream.write(canonical(value)+'\n');stream.flush();os.fsync(stream.fileno())
        temporary=run/'.phase.tmp'
        with temporary.open('w')as stream:json.dump(value,stream,allow_nan=False)
        os.replace(temporary,run/'phase.json')
    mark('cpu_preflight')
    if platform.system()!='Darwin'or platform.machine()!='arm64':raise RuntimeError('Only the configured local Mac is permitted')
    versions={name:importlib.metadata.version(name)for name in('mlx','mlx-lm','transformers','numpy')}
    if versions!={'mlx':'0.32.4','mlx-lm':'0.32.2','transformers':'5.22.2','numpy':'2.5.4'}:raise ValueError('Numerical versions differ')
    source_binding=layout.verify_installed_lora_sources();package=importlib.metadata.distribution('mlx-lm')
    if plans.sha(Path(package.locate_file('mlx_lm/models/gated_delta.py')))!='f69e837ea68641dbb06fdb5dcc024a9a7e1d9353f08ae58c05275253a0c39428':raise ValueError('GDN primitive source differs')
    import qwen38_local as local
    local.prepare_local();os.environ.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','DO_NOT_TRACK':'1'})
    if plans.sha(local.MANIFEST)!=plans.MODEL_SHA:raise ValueError('Base manifest differs')
    local.verify()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(local.MODEL),local_files_only=True,trust_remote_code=False)
    schema=json.loads((data/'action_schema.json').read_text());chosen=None;maximum=0
    for filename in('train768.jsonl','dev144.jsonl'):
        for index,row in enumerate(json.loads(line)for line in(data/filename).read_text().splitlines()):
            ids,prefix=render(tokenizer,row,schema)
            if not 0<prefix<len(ids)<=1024 or len(ids)-prefix>32:raise ValueError('Full suffix/context must fit without truncation')
            maximum=max(maximum,len(ids)-prefix)
            if filename=='train768.jsonl'and index==16:chosen=(ids,prefix)
    ids,prefix=chosen;length=len(ids);width=min(1024,1+32*((length+31)//32));count_expected=length-prefix
    if (length,prefix)!=(1022,1001)or maximum!=plan['actual_maximum_suffix_tokens']:raise ValueError('Full long TRAIN token geometry differs')
    once(run/'token_input.json',{'training_index':16,'token_ids':ids,'prefix_tokens':prefix,'padded_width':width})
    expected=layout.expected_adapter_layout(local.MODEL,4,64);projections=layout.projection_layout(local.MODEL,64)
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten,tree_unflatten
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.tuner.trainer import grad_checkpoint
    mx.set_memory_limit(plan['mlx_memory_guideline_bytes'])
    if mx.distributed.init().size()!=1:raise ValueError('One local device group required')
    mark('before_base_load',mx);model,_=load(str(local.MODEL),trust_remote_code=False);mark('after_base_load',mx)
    model.freeze();mx.random.seed(20261108);modules=dict(model.named_modules())
    for name,spec in projections.items():
        if type(modules.get(name))is not(nn.QuantizedLinear if spec['base_quantized']else nn.Linear):raise ValueError('Native projection type differs')
    linear_to_lora_layers(model,64,{'rank':4,'scale':2.,'dropout':0.});parameters=dict(tree_flatten(model.trainable_parameters()))
    if set(parameters)!=set(expected)or any(tuple(value.shape)!=expected[name]or value.dtype!=mx.float32 for name,value in parameters.items()):raise ValueError('Exact all-992 trainable layout differs')
    mx.eval(model.parameters());mark('after_lora_materialized_before_cache_clear',mx)
    if any(bool(mx.any(value!=0).item())for name,value in parameters.items()if name.endswith('.lora_b')):raise ValueError('Initial B was not zero')
    if model.language_model.args.tie_word_embeddings or model.language_model.sharding_group is not None:raise ValueError('Native fixed untied unsharded head required')
    del parameters,modules
    grad_checkpoint(model.layers[0]);model.train();gdns=[module for _,module in model.named_modules()if type(module).__name__=='GatedDeltaNet']
    if len(gdns)!=48 or any(not module.training for module in gdns)or any(not block.training for block in model.layers):raise ValueError('All GDN blocks must be native training/reference mode')
    once(run/'run_config.json',{'schema':'research4.chain-rule-diagnostic-run.v4','plan_sha256':plan_sha,
        'model':plans.MODEL_ID,'revision':plans.REVISION,'model_manifest_sha256':plans.MODEL_SHA,'prior_adapter_loaded':False,
        'initialization':'fresh_pinned_base','rank':4,'num_layers':64,'versions':versions,'installed_sources':source_binding,
        'execution_policy':plan['execution_policy'],'outer_mx_compile':False,'gdn_math_changed':False,
        'scope':'Diagnostic-only fresh LoRA; never eligible for serving/tuning.'})
    mx.save_safetensors(str(run/'diagnostic_state0_weights.safetensors'),dict(tree_flatten(model.trainable_parameters())))
    gc.collect();mx.clear_cache();mark('after_setup_cache_clear',mx)
    optimizer=optim.Adam(learning_rate=.00001,betas=[.9,.999],eps=1e-8,bias_correction=False)
    state_receipts=[];reference_zero=None
    def artifact(path):return{'path':plans.rel(path),'sha256':plans.sha(path)}
    def save_hidden(label,value,key='hidden',audit_key='hidden_values'):
        native=run/(label+'_native.safetensors');audit=run/(label+'_audit.safetensors')
        mx.save_safetensors(str(native),{key:value});mx.save_safetensors(str(audit),{audit_key:value.astype(mx.float32)})
        shape=tuple(value.shape);report=verify_native_audit_copy(native,key,audit,audit_key,shape)
        return{'native':artifact(native),'audit':artifact(audit),'shape':list(shape),'dtype':str(value.dtype),'native_copy_check':report}
    for state_index in(0,1):
        if state_index:
            mark('before_single_reference_adam_update',mx)
            saved=mx.load(str(reference_zero));optimizer.update(model,tree_unflatten(list(saved.items())));mx.eval(model.state,optimizer.state)
            if int(optimizer.step.item())!=1:raise ValueError('Exactly one diagnostic update required')
            if not all(bool(mx.all(mx.isfinite(value)).item())for _,value in tree_flatten(optimizer.state)if hasattr(value,'dtype')):raise ValueError('Optimizer state is nonfinite')
            parameters=dict(tree_flatten(model.trainable_parameters()));layers={int(name.split('.layers.')[1].split('.')[0])for name,value in parameters.items()if name.endswith('.lora_b')and bool(mx.any(value!=0).item())}
            if (layers!=set(range(64))or parameter_digest(model)==state_receipts[0]['weight_state_sha256']
                or not all(bool(mx.all(mx.isfinite(value)).item())for value in parameters.values())):raise ValueError('Diagnostic weights did not actually update all B layers with finite values')
            mx.save_safetensors(str(run/'diagnostic_state1_weights.safetensors'),parameters);del parameters,saved
            counters['diagnostic_optimizer_updates']=1;gc.collect();mx.clear_cache();mark('after_single_reference_adam_update',mx)
        before=parameter_digest(model);batch,lengths=make_batch(mx,ids,prefix)
        import numpy as np
        input_tensor_sha=hashlib.sha256(np.array(batch).tobytes()).hexdigest()
        if any(not module.training for module in gdns):raise ValueError('Training mode changed within a snapshot')
        mark(f'state{state_index}_before_detached_full_body_forward',mx)
        live_hidden=chain.body_hidden(model,batch);detached=mx.stop_gradient(live_hidden);mx.eval(detached)
        if tuple(detached.shape)!=(1,width-1,model.language_model.args.hidden_size):raise ValueError('Full hidden geometry differs')
        hidden=save_hidden(f'state{state_index}_hidden',detached);del live_hidden,detached;gc.collect();mx.clear_cache()
        if parameter_digest(model)!=before:raise ValueError('Detached forward changed parameter values')
        mark(f'state{state_index}_after_hidden_saved_graph_released',mx)
        heads={}
        for selected in(False,True):
            label=f'state{state_index}_head_'+('selected'if selected else'full')
            h=mx.load(str(plans.path_for(hidden['native']['path'])))['hidden']
            if str(h.dtype)!=hidden['dtype']:raise ValueError('Native H dtype changed on reload')
            def objective(h):return chain.head_components(mx,nn,model.language_model,h,batch,lengths,selected)
            mark('before_'+label,mx,operation='array_head_autograd_native_hidden')
            (mean,(count,logits)),dh=mx.value_and_grad(objective)(h);mx.eval(mean,count,logits,dh)
            if dh.dtype!=h.dtype or dh.shape!=h.shape:raise ValueError('Native hidden covector dtype/shape differs')
            mean_value=float(mean.item());count_value=int(count.item())
            if not math.isfinite(mean_value)or mean_value<0 or count_value!=count_expected:raise ValueError('True head NLL/count invalid')
            covector=save_hidden(label+'_covector',dh,'hidden_cotangent','hidden_gradient')
            logits_path=run/(label+'_target_logits.safetensors');mx.save_safetensors(str(logits_path),{'target_logits':logits[:,:count_expected,:].astype(mx.float32).reshape(count_expected,logits.shape[-1])})
            if parameter_digest(model)!=before:raise ValueError('Head-only gradient changed theta')
            heads['selected'if selected else'full']={'mean_nll':mean_value,'target_tokens':count_value,
                'hidden_covector':covector,'target_logits':{**artifact(logits_path),'shape':[count_expected,int(logits.shape[-1])]},
                'weight_state_sha256':before,'head_input_native_file':hidden['native'],'head_input_dtype':str(h.dtype),
                'detached_hidden':hidden,'full_input_tensor_sha256':input_tensor_sha,
                'model_training':bool(model.training),'gdn_training_count':sum(bool(module.training)for module in gdns)}
            once(run/(label+'_capture.json'),heads['selected'if selected else'full'])
            del h,mean,count,logits,dh,objective;gc.collect();mx.clear_cache();mark('after_'+label+'_saved_released',mx)
        loss=loss_equivalence(heads['full']['mean_nll'],heads['selected']['mean_nll'],count_expected,count_expected,plan['gradient_tolerances'])
        hidden_gradient=compare_covectors(plans.path_for(heads['full']['hidden_covector']['audit']['path']),plans.path_for(heads['selected']['hidden_covector']['audit']['path']),hidden['shape'],prefix,length,plan['gradient_tolerances'])
        logits=compare_logits(plans.path_for(heads['full']['target_logits']['path']),plans.path_for(heads['selected']['target_logits']['path']),heads['full']['target_logits']['shape'])
        head_check={'true_mean_nll':loss,'full_hidden_gradient':hidden_gradient,'valid_logits':logits,'passed':loss['passed']and hidden_gradient['passed']and logits['passed']}
        once(run/f'state{state_index}_head_comparison.json',head_check)
        if not head_check['passed']:raise ValueError('Predeclared head/dH/logit gate failed')
        mark(f'state{state_index}_head_gates_passed_before_direct_backward',mx)
        theta={}
        # The actual selected backward is deliberately first. If it cannot fit,
        # the guard stops before a further reference body-backward pass.
        for kind in('direct_selected','reference_body_vjp'):
            label=f'state{state_index}_'+kind;mark('before_'+label,mx,operation=kind,outer_mx_compile=False)
            if kind=='direct_selected':
                (scalar,(count,live_logits,recomputed_h)),gradient=nn.value_and_grad(model,lambda m:chain.direct_selected(mx,nn,m,batch,lengths))(model)
                mx.eval(model.state,scalar,count,live_logits,recomputed_h,gradient)
                scalar_record={'true_mean_nll':float(scalar.item()),'target_tokens':int(count.item())}
                if scalar_record['target_tokens']!=count_expected or not math.isfinite(scalar_record['true_mean_nll']):raise ValueError('Direct mean loss/count invalid')
                direct_logits=run/(label+'_target_logits.safetensors');mx.save_safetensors(str(direct_logits),{'target_logits':live_logits[:,:count_expected,:].astype(mx.float32).reshape(count_expected,live_logits.shape[-1])})
                scalar_record['target_logits']={**artifact(direct_logits),'shape':[count_expected,int(live_logits.shape[-1])]}
            else:
                native_dh=heads['full']['hidden_covector']['native'];dh=mx.load(str(plans.path_for(native_dh['path'])))['hidden_cotangent']
                (scalar,recomputed_h),gradient=nn.value_and_grad(model,lambda m:chain.body_contraction(mx,m,batch,dh))(model)
                mx.eval(model.state,scalar,recomputed_h,gradient)
                scalar_record={'vjp_contraction_value_not_nll':float(scalar.item()),'constant_cotangent_native_file':native_dh,
                    'covector_dtype':str(dh.dtype),'normalization_already_in_mean_head_covector':True,'second_target_division':False}
                if not math.isfinite(scalar_record['vjp_contraction_value_not_nll']):raise ValueError('Reference linear-functional scalar is nonfinite')
            flat=dict(tree_flatten(gradient))
            if set(flat)!=set(expected)or any(tuple(g.shape)!=expected[name]or g.dtype!=mx.float32 for name,g in flat.items()):raise ValueError('All-992 theta gradient layout differs')
            path=run/(label+'_gradients.safetensors');mx.save_safetensors(str(path),flat)
            check=compare_gradients(path,path,expected,plan['gradient_tolerances'],require_zero_a=state_index==0)
            layers={int(row['name'].split('.layers.')[1].split('.')[0])for row in check['tensors']if row['name'].endswith('.lora_b')and row['reference_l2']>0}
            a_count=sum(row['name'].endswith('.lora_a')and row['reference_l2']>0 for row in check['tensors'])
            if not check['passed']or layers!=set(range(64))or(state_index and a_count<1):raise ValueError('Finite/all-layer/initial-zero-A/post-update-A gradient gate failed')
            if str(recomputed_h.dtype)!=hidden['dtype']or tuple(recomputed_h.shape)!=tuple(hidden['shape']):raise ValueError('Body recompute changed native hidden shape/dtype')
            recomputed=save_hidden(label+'_recomputed_hidden',recomputed_h)
            hidden_identity=exact_native_values(plans.path_for(hidden['native']['path']),plans.path_for(recomputed['native']['path']),'hidden',hidden['shape'])
            once(run/(label+'_hidden_identity.json'),hidden_identity)
            if not hidden_identity['passed']:raise ValueError('The body VJP/direct body did not recompute the exact saved native hidden values')
            if parameter_digest(model)!=before:raise ValueError('Theta changed within paired gradients')
            theta[kind]={**scalar_record,'gradients':artifact(path),'weight_state_sha256':before,
                'nonzero_b_gradient_layers':sorted(layers),'nonzero_a_gradient_tensors':a_count,'recomputed_hidden':recomputed,
                'full_input_tensor_sha256':input_tensor_sha,'model_training':bool(model.training),
                'gdn_training_count':sum(bool(module.training)for module in gdns)}
            once(run/(label+'_capture.json'),theta[kind])
            del gradient,flat,scalar,recomputed_h,check
            if kind=='direct_selected':del count,live_logits
            else:del dh
            gc.collect();mx.clear_cache();mark('after_'+label+'_saved_released',mx)
        reference=theta['reference_body_vjp'];direct=theta['direct_selected']
        gradients=compare_gradients(plans.path_for(reference['gradients']['path']),plans.path_for(direct['gradients']['path']),expected,plan['gradient_tolerances'],require_zero_a=state_index==0)
        direct_loss=loss_equivalence(heads['full']['mean_nll'],direct['true_mean_nll'],count_expected,direct['target_tokens'],plan['gradient_tolerances'])
        direct_logits_check=compare_logits(plans.path_for(heads['full']['target_logits']['path']),plans.path_for(direct['target_logits']['path']),direct['target_logits']['shape'])
        comparison={'state_index':state_index,'all_992_theta_gradients':gradients,'true_mean_nll':direct_loss,'valid_logits':direct_logits_check,
            'passed':gradients['passed']and direct_loss['passed']and direct_logits_check['passed'],'reference_kind':'decomposed_full_native_mean_objective_chain_rule'}
        once(run/f'state{state_index}_theta_comparison.json',comparison)
        if not comparison['passed']:raise ValueError('Decomposed reference/direct selected all-theta gate failed')
        state_receipts.append({'state_index':state_index,'weight_state_sha256':before,'head_check_sha256':plans.sha(run/f'state{state_index}_head_comparison.json'),
            'theta_check_sha256':plans.sha(run/f'state{state_index}_theta_comparison.json'),'passed':True})
        if state_index==0:reference_zero=plans.path_for(reference['gradients']['path'])
        counters['diagnostic_checks_completed']=len(state_receipts);mark(f'state{state_index}_all_gates_passed',mx)
        del batch,lengths;gc.collect();mx.clear_cache()
    model.eval()
    if any(module.training for module in gdns):raise ValueError('Final outer-model pair must use native eval mode')
    batch,lengths=make_batch(mx,ids,prefix);eval_records={};eval_digest=parameter_digest(model)
    if eval_digest!=state_receipts[1]['weight_state_sha256']:raise ValueError('Eval mode changed the updated parameter state')
    for selected in(False,True):
        label='eval_selected'if selected else'eval_outer_full';mark('before_'+label,mx,model_training=False,gdn_training_count=0)
        if selected:
            mean,(count,logits,h)=chain.direct_selected(mx,nn,model,batch,lengths);mx.eval(mean,count,logits,h)
        else:
            mean,(count,logits)=chain.outer_full_components(mx,nn,model,batch,lengths);mx.eval(mean,count,logits)
        value=float(mean.item());tokens=int(count.item())
        if not math.isfinite(value)or value<0 or tokens!=21 or parameter_digest(model)!=eval_digest:raise ValueError('Eval outer/factorized NLL/count/state invalid')
        path=run/(label+'_target_logits.safetensors');mx.save_safetensors(str(path),{'target_logits':logits[:,:21,:].astype(mx.float32).reshape(21,logits.shape[-1])})
        eval_records[label]={'mean_nll':value,'target_tokens':tokens,'weight_state_sha256':eval_digest,'target_logits':{**artifact(path),'shape':[21,int(logits.shape[-1])]},'model_training':False,'gdn_training_count':0}
        once(run/(label+'_capture.json'),eval_records[label]);del mean,count,logits
        if selected:del h
        gc.collect();mx.clear_cache();mark('after_'+label+'_saved_released',mx)
    eval_loss=loss_equivalence(eval_records['eval_outer_full']['mean_nll'],eval_records['eval_selected']['mean_nll'],21,21,plan['gradient_tolerances'])
    eval_logits=compare_logits(plans.path_for(eval_records['eval_outer_full']['target_logits']['path']),plans.path_for(eval_records['eval_selected']['target_logits']['path']),eval_records['eval_selected']['target_logits']['shape'])
    eval_check={'true_mean_nll':eval_loss,'valid_logits':eval_logits,'passed':eval_loss['passed']and eval_logits['passed']}
    once(run/'eval_outer_comparison.json',eval_check)
    if not eval_check['passed']:raise ValueError('Actual outer-model eval forward did not match factorized selected forward')
    counters['diagnostic_checks_completed']=3;mark('actual_outer_eval_pair_passed',mx)
    mark('before_final_base_file_verify',mx);local.verify()
    receipt={'schema':'research4.decomposed-chain-rule-equivalence-receipt.v4','status':'passed','plan_sha256':plan_sha,
        'states':state_receipts,'eval_outer_forward':eval_check,'rank':4,'num_layers':64,'gradient_tensors':992,'diagnostic_optimizer_updates':1,
        'full_context_tokens':1022,'prefix_tokens':1001,'full_hidden_positions_checked':1023,'head_window':32,
        'native_hidden_and_covector_dtype_preserved':True,'full_inputs_training_mode_and_paired_weights_preserved':True,
        'reference_kind':'full_native_mean_head_covector_then_stopped_body_VJP','vjp_scalar_reported_as_nll':False,
        'execution_policy':plan['execution_policy'],'outer_mx_compile':False,'gdn_math_changed':False,
        'base_files_verified_before_and_after':True,'original_deadline_epoch':plans.DEADLINE,
        'eligible_for_tuning_or_serving':False,'optimized_long_pilot_verified':False,
        'scope':'Numerical proof on full TRAIN context at two snapshots; resource stability of a fresh three-update pilot and future tuning remain unverified.'}
    once(run/'equivalence_receipt.json',receipt);mark('chain_rule_equivalence_complete_not_a_candidate',mx)
    return receipt


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--owner',required=True);args=parser.parse_args();run=args.run_dir.resolve()
    if not run.is_relative_to((plans.ROOT/'runs').resolve())or not run.is_dir():raise ValueError('Invalid owned V4 run')
    try:run_proof(args.plan,args.plan_sha256,run,args.owner)
    except BaseException as exc:once(run/'worker_failure.json',{'status':'failed','error':f'{type(exc).__name__}: {exc}','epoch':time.time()});raise
