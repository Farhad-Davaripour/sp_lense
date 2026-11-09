"""Owned v2 full-depth pilot/trial/final worker. Never launch directly."""
from functools import partial
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time

import psutil
import game_all_layers_plan as plans
from game_all_layers_loss import make_batch, masked_target_loss, masked_target_sum, LOSS_VERSION
import game_all_layers_layout as layout
from game_all_layers_selection import pool_development
from game_training_renderer import canonical, render

LOCK=plans.PROJECT/'.local/validation/qwen38_shared_v1/inference.lock'


def once(path,value):
    with path.open('x',encoding='utf-8')as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())


def append(path,value):
    with path.open('a',encoding='utf-8')as stream:
        stream.write(canonical(value)+'\n');stream.flush();os.fsync(stream.fileno())


def progress(run,value):
    value={**value,'updated_epoch':time.time()};temporary=run/f'.progress.{os.getpid()}.tmp'
    with temporary.open('w')as stream:json.dump(value,stream,allow_nan=False)
    os.replace(temporary,run/'worker_progress.json')


def gradient_coverage(names,flags):
    if len(names)!=len(flags)or any(type(flag)is not bool for flag in flags):raise ValueError('Gradient flags differ')
    layers=set();a_count=b_count=0
    for name,nonzero in zip(names,flags):
        match=re.fullmatch(r'language_model\.model\.layers\.(\d+)\..+\.lora_([ab])',name)
        if match is None or not 0<=int(match[1])<64:raise ValueError('Unrecognized gradient name')
        if nonzero:
            if match[2]=='b':layers.add(int(match[1]));b_count+=1
            else:a_count+=1
    return {'nonzero_b_layer_indices':sorted(layers),'nonzero_a_tensors':a_count,'nonzero_b_tensors':b_count}


def tokenize(path,tokenizer,schema):
    rows=[json.loads(line)for line in path.read_text().splitlines()];result=[]
    for row in rows:
        ids,prefix=render(tokenizer,row,schema)
        if not 0<prefix<len(ids)<=1024:raise ValueError('Sequence would truncate or have no target')
        result.append((hashlib.sha256(canonical(row).encode()).hexdigest(),ids,prefix))
    return result


def parameter_digest(model):
    """Stream the small trainable tensors to CPU; never copy the full base."""
    import numpy as np
    from mlx.utils import tree_flatten
    result=hashlib.sha256()
    for name,array in sorted(tree_flatten(model.trainable_parameters())):
        result.update(name.encode());result.update(str(array.shape).encode())
        result.update(np.array(array).tobytes(order='C'))
    return result.hexdigest()


def run_stage(plan_path,plan_sha,run,owner):
    worker_started=time.perf_counter()
    plan,cohort,data=plans.read_plan(plan_path,plan_sha)
    ownership=json.loads((LOCK/'owner.json').read_text())
    if (ownership.get('owner'),ownership.get('pid'),ownership.get('kind'))!=(owner,os.getppid(),'sp_lens_all_layers_v2'):
        raise ValueError('Live all-layer supervisor ownership differs')
    authority=plans.ROOT/'authorities'/(plan_sha+'.json')
    permit=json.loads(authority.read_text())
    if (permit.get('owner'),permit.get('run_dir'),permit.get('plan_sha256'))!=(owner,str(run),plan_sha):
        raise ValueError('Single-use all-layer authority differs')
    if platform.system()!='Darwin'or platform.machine()!='arm64':raise RuntimeError('Use the configured local Apple-silicon Mac')
    versions={name:importlib.metadata.version(name)for name in ('mlx','mlx-lm','transformers','numpy')}
    if versions!={'mlx':'0.32.4','mlx-lm':'0.32.2','transformers':'5.22.2','numpy':'2.5.4'}:raise ValueError('Unreviewed numerical package versions')
    installed_sources=layout.verify_installed_lora_sources()
    package=importlib.metadata.distribution('mlx-lm')
    for name,expected_sha in plan['additional_installed_source_sha256'].items():
        if plans.sha(Path(package.locate_file(name)))!=expected_sha:raise ValueError('GDN primitive implementation differs')
    import qwen38_local as local
    local.prepare_local();os.environ.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','DO_NOT_TRACK':'1'})
    if plans.sha(local.MANIFEST)!=plans.MODEL_SHA:raise ValueError('Base manifest differs')
    identity=local.verify()
    if (identity['repo_id'],identity['revision'])!=(plans.MODEL_ID,plans.REVISION):raise ValueError('Base identity differs')
    schema=json.loads((data/'action_schema.json').read_text())
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(local.MODEL),local_files_only=True,trust_remote_code=False)
    filename={'pilot':'pilot_longest.jsonl','trial':'train48.jsonl','final':'train768.jsonl'}[plan['mode']]
    prepared=tokenize(data/filename,tokenizer,schema)
    if plan['mode']=='pilot':order=[0,0,0]
    else:order=cohort['train48_optimizer_order'if plan['mode']=='trial'else'train768_optimizer_order']
    if len(order)!=plan['iterations']or any(type(i)is not int or not 0<=i<len(prepared)for i in order):raise ValueError('Optimizer input order differs')
    development=tokenize(data/'dev144.jsonl',tokenizer,schema)if plan['mode']in('trial','final')else[]
    if development and [row[0]for row in development]!=cohort['development_ids']:raise ValueError('Development cohort IDs differ')
    projections=layout.projection_layout(local.MODEL,64)
    expected=layout.expected_adapter_layout(local.MODEL,plan['rank'],64)
    names=sorted(expected)
    progress(run,{'phase':'loading_model','completed_steps':0,'total_steps':plan['iterations'],'stage':plan['stage'],'rank':plan['rank'],'num_layers':64})
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.tuner.trainer import grad_checkpoint
    mx.set_memory_limit(plan['mlx_memory_guideline_bytes'])
    if mx.distributed.init().size()!=1:raise RuntimeError('Only one local model process/device group is permitted')
    load_started=time.perf_counter()
    model,_=load(str(local.MODEL),trust_remote_code=False)
    load_seconds=time.perf_counter()-load_started
    model.freeze();mx.random.seed(plan['seed'])
    modules=dict(model.named_modules())
    for name,spec in projections.items():
        target=nn.QuantizedLinear if spec['base_quantized']else nn.Linear
        if type(modules.get(name))is not target:raise ValueError('Actual base projection type differs: '+name)
    geometry={'rank':plan['rank'],'scale':2.0,'dropout':0.0}
    linear_to_lora_layers(model,64,geometry)
    parameters=dict(tree_flatten(model.trainable_parameters()))
    if set(parameters)!=set(expected):raise ValueError('Actual all-layer trainable tensor inventory differs')
    for name,array in parameters.items():
        if tuple(array.shape)!=expected[name]or array.dtype!=mx.float32:raise ValueError('Actual trainable shape/dtype differs')
    mx.eval(model.parameters())
    if any(bool(mx.any(value!=0).item())for name,value in parameters.items()if name.endswith('.lora_b')):
        raise ValueError('Fresh adapter B initialization was not zero')
    config={'fine_tune_type':'lora','num_layers':64,'batch_size':1,'max_seq_length':1024,'mask_prompt':True,
        'resume_adapter_file':None,'optimizer':'adam','grad_accumulation_steps':1,'grad_checkpoint':True,
        'lora_parameters':geometry,'iters':plan['iterations'],'learning_rate':plan['learning_rate'],
        'seed':plan['seed'],'optimizer_config':{'adam':{'betas':[0.9,0.999],'eps':1e-8,'bias_correction':False}},'loss_version':LOSS_VERSION}
    adapters=run/'adapters';adapters.mkdir(exist_ok=False);once(adapters/'adapter_config.json',config)
    once(run/'run_config.json',{'schema':'research4.sp-lens-game-all-layers-run.v2','plan_sha256':plan_sha,
        'model':plans.MODEL_ID,'revision':plans.REVISION,'model_manifest_sha256':plans.MODEL_SHA,
        'plan':plan,'initialization':'fresh_pinned_base','prior_adapter_loaded':False,'legacy_9b_adapters_loaded':False,
        'package_versions':versions,'installed_source_binding':installed_sources,'tensor_inventory':{key:list(value)for key,value in expected.items()},
        'train_order_ids':[prepared[i][0]for i in order],'host':{'system':platform.system(),'machine':platform.machine(),'memory_bytes':psutil.virtual_memory().total}})
    grad_checkpoint(model.layers[0]);model.train()
    delta_modules=[module for _,module in model.named_modules()if type(module).__name__=='GatedDeltaNet']
    if len(delta_modules)!=48 or any(not module.training for module in delta_modules)or any(not block.training for block in model.layers):
        raise RuntimeError('All 48 native GDN modules must use their training/reference route before differentiated tracing')
    optimizer=optim.Adam(learning_rate=plan['learning_rate'],betas=[0.9,0.999],eps=1e-8,bias_correction=False)
    value_and_grad=nn.value_and_grad(model,masked_target_loss)
    state=[model.state,optimizer.state,mx.random.state]
    @partial(mx.compile,inputs=state,outputs=state)
    def step(batch,lengths):
        (mean,count),gradient=value_and_grad(model,batch,lengths)
        flat=dict(tree_flatten(gradient))
        if set(flat)!=set(expected):raise ValueError('Actual gradient inventory differs')
        arrays=[flat[name]for name in names]
        if any(tuple(g.shape)!=expected[name]or g.dtype!=mx.float32 for name,g in zip(names,arrays)):
            raise ValueError('Actual gradient shape/dtype differs')
        finite=mx.stack([mx.all(mx.isfinite(g))for g in arrays])
        nonzero=mx.stack([mx.any(g!=0)for g in arrays])
        norm=mx.sqrt(mx.sum(mx.stack([mx.sum(g*g)for g in arrays])))
        optimizer.update(model,gradient)
        return mean,count,finite,nonzero,norm
    started=time.perf_counter();times=[];coverage=set();minimum_available=psutil.virtual_memory().available
    swap_before=psutil.swap_memory().used
    for index,row_index in enumerate(order,1):
        if psutil.virtual_memory().available<plan['memory_floor_bytes']:raise RuntimeError('Available memory below 4 GiB')
        row_id,ids,prefix=prepared[row_index];batch,lengths=make_batch(mx,ids,prefix)
        before=time.perf_counter();mean,count,finite,nonzero,norm=step(batch,lengths)
        mx.eval(state,mean,count,finite,nonzero,norm)
        elapsed=time.perf_counter()-before;mean_value=float(mean.item());count_value=int(count.item());norm_value=float(norm.item())
        if not all(finite.tolist())or not all(math.isfinite(v)for v in(mean_value,norm_value))or count_value!=len(ids)-prefix or int(optimizer.step.item())!=index:
            raise RuntimeError('Nonfinite gradient/loss or incorrect supervised target mask')
        observed=gradient_coverage(names,nonzero.tolist());coverage.update(observed['nonzero_b_layer_indices'])
        minimum_available=min(minimum_available,psutil.virtual_memory().available);times.append(elapsed)
        event={'stage':plan['stage'],'phase':'training','rank':plan['rank'],'num_layers':64,'step':index,'total_steps':plan['iterations'],
            'row_id':row_id,'loss':mean_value,'target_tokens':count_value,'sequence_tokens':len(ids),'step_seconds':elapsed,
            'tokens_per_second':count_value/elapsed,'sequence_tokens_per_second':len(ids)/elapsed,'gradient_global_l2':norm_value,
            'optimizer_step':int(optimizer.step.item()),'native_gdn_training_route':True,
            'gradient_tensor_count':len(names),'gradient_all_finite':True,**observed,
            'mlx_peak_bytes':int(mx.get_peak_memory()),'available_bytes':psutil.virtual_memory().available,
            'swap_used_bytes':psutil.swap_memory().used,'swap_change_bytes':psutil.swap_memory().used-swap_before}
        append(run/'updates.jsonl',event);print('R4_ALL_LAYER_REPORT '+canonical(event),flush=True)
        progress(run,{'phase':'training','completed_steps':index,'total_steps':plan['iterations'],**{key:event[key]for key in ('stage','rank','num_layers','loss','tokens_per_second','sequence_tokens_per_second','mlx_peak_bytes','available_bytes','swap_used_bytes')}})
        mx.clear_cache()
    if coverage!=set(range(64)):raise RuntimeError('Not all 64 blocks had observed nonzero B gradients')
    weights=adapters/'adapters.safetensors'
    mx.save_safetensors(str(weights),dict(tree_flatten(model.trainable_parameters())))
    verification=layout.verify_saved_adapter(weights,adapters/'adapter_config.json',local.MODEL,plan['rank'],64,
        expected_iterations=plan['iterations'],expected_learning_rate=plan['learning_rate'],require_all_layer_updates=True)
    once(run/'verification.json',verification)
    model.eval();before_dev=parameter_digest(model);optimizer_before_dev=int(optimizer.step.item());dev_rows=[];pilot_forwards=[]
    if plan['mode']=='pilot':
        for repetition in range(3):
            _,ids,prefix=prepared[0];batch,lengths=make_batch(mx,ids,prefix);before=time.perf_counter()
            total,count=masked_target_sum(model,batch,lengths);mx.eval(total,count)
            elapsed=time.perf_counter()-before
            if not math.isfinite(float(total.item()))or int(count.item())!=len(ids)-prefix:raise RuntimeError('Pilot eval-mode training-row forward failed')
            pilot_forwards.append(elapsed)
            append(run/'pilot_train_forward.jsonl',{'repetition':repetition+1,'seconds':elapsed,'target_tokens':int(count.item()),
                'nll_sum':float(total.item()),'data_split':'train','mode':'eval','used_for_selection':False})
            progress(run,{'phase':'pilot_training_forward','completed_steps':plan['iterations'],'total_steps':plan['iterations'],
                'pilot_forward_completed':repetition+1,'pilot_forward_total':3,'stage':plan['stage'],'rank':plan['rank'],'num_layers':64})
            mx.clear_cache()
    for index,(row_id,ids,prefix)in enumerate(development):
        batch,lengths=make_batch(mx,ids,prefix);total,count=masked_target_sum(model,batch,lengths);mx.eval(total,count)
        value=float(total.item());count_value=int(count.item())
        if not math.isfinite(value)or value<0 or count_value!=cohort['development_target_tokens'][index]:raise RuntimeError('Invalid development likelihood')
        result={'row_id':row_id,'target_tokens':count_value,'nll_sum':value};append(run/'development_nll.jsonl',result);dev_rows.append(result)
        progress(run,{'phase':'development_likelihood','completed_steps':plan['iterations'],'total_steps':plan['iterations'],
            'development_completed':index+1,'development_total':len(development),'stage':plan['stage'],'rank':plan['rank'],'num_layers':64})
        mx.clear_cache()
    if parameter_digest(model)!=before_dev:raise RuntimeError('Trainable weights changed during no-update development evaluation')
    if int(optimizer.step.item())!=optimizer_before_dev:raise RuntimeError('Optimizer advanced during development evaluation')
    progress(run,{'phase':'base_verification','completed_steps':plan['iterations'],'total_steps':plan['iterations'],
        'stage':plan['stage'],'rank':plan['rank'],'num_layers':64})
    local.verify()
    once(run/'full_base_file_verify.json',{'status':'passed','model':plans.MODEL_ID,'revision':plans.REVISION,
        'manifest_sha256':plans.MODEL_SHA,'config_sha256':plans.sha(local.MODEL/'config.json'),'full_inventory_verified_before_and_after':True})
    receipt={'schema':'research4.sp-lens-game-all-layers-'+('trial'if plan['mode']=='trial'else'pilot'if plan['mode']=='pilot'else'worker')+'.v2',
        'status':'complete','stage':plan['stage'],'trial_id':plan['stage'],'rank':plan['rank'],'learning_rate':plan['learning_rate'],'num_layers':64,
        'initialization':'fresh_pinned_base','completed_steps':len(times),'expected_steps':plan['iterations'],'plan_sha256':plan_sha,
        'dataset_manifest_sha256':plans.DATA_SHA,'gradient_b_layer_coverage':sorted(coverage),'gradient_all_finite':True,
        'cohort_sha256':plan['cohort']['sha256'],'training_rows_sha256':plans.sha(data/filename),
        'training_order_ids_sha256':hashlib.sha256(canonical([prepared[i][0]for i in order]).encode()).hexdigest(),
        'loss_version':LOSS_VERSION,'source_sha256':plan['source_sha256'],'protocol':plan['protocol'],'config':plan['config'],
        'step_seconds':times,'first_step_compile_and_compute_seconds':times[0],'steady_step_seconds':times[1:],
        'model_load_seconds':load_seconds,'setup_to_first_step_seconds':started-worker_started,
        'pilot_eval_training_forward_seconds':pilot_forwards,
        'minimum_available_bytes':minimum_available,'mlx_peak_bytes':int(mx.get_peak_memory()),'swap_used_before':swap_before,
        'swap_used_after':psutil.swap_memory().used,'elapsed_training_and_dev_seconds':time.perf_counter()-started,
        'full_worker_elapsed_seconds':time.perf_counter()-worker_started,
        'development_rows':dev_rows,'development_no_optimizer_updates':True,'trainable_state_unchanged_during_dev':True,
        'optimizer_step_before_dev':optimizer_before_dev,'optimizer_step_after_dev':int(optimizer.step.item()),
        'artifacts':{'adapter':{'path':plans.rel(weights),'sha256':plans.sha(weights)},'adapter_config':{'path':plans.rel(adapters/'adapter_config.json'),'sha256':plans.sha(adapters/'adapter_config.json')},
            'plan':{'path':plans.rel(plan_path),'sha256':plan_sha},'run_config':{'path':plans.rel(run/'run_config.json'),'sha256':plans.sha(run/'run_config.json')},
            'full_base_file_verify':{'path':plans.rel(run/'full_base_file_verify.json'),'sha256':plans.sha(run/'full_base_file_verify.json')}},
        'verification':verification,'scope':'Compatibility pilot only; not a candidate.'if plan['mode']=='pilot'else'Development likelihood and saved-file evidence only; no final game result.'}
    if development:
        receipt['development_summary']=pool_development(dev_rows,cohort['development_ids'])
        receipt['artifacts']['development_nll']={'path':plans.rel(run/'development_nll.jsonl'),'sha256':plans.sha(run/'development_nll.jsonl')}
    once(run/('trial_receipt.json'if plan['mode']=='trial'else'worker_receipt.json'),receipt)
    progress(run,{'phase':'complete','completed_steps':len(times),'total_steps':plan['iterations'],'stage':plan['stage'],'rank':plan['rank'],'num_layers':64})
    return receipt


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--owner',required=True)
    args=parser.parse_args();run=args.run_dir.resolve()
    if not run.is_relative_to((plans.ROOT/'runs').resolve())or not run.is_dir():raise ValueError('Invalid owned run directory')
    try:run_stage(args.plan,args.plan_sha256,run,args.owner)
    except BaseException as exc:
        once(run/'worker_failure.json',{'status':'failed','error':f'{type(exc).__name__}: {exc}','epoch':time.time()});raise
