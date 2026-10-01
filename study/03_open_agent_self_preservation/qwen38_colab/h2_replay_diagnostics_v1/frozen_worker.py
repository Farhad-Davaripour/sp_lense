"""Stream 1: one resident frozen model; H2 equivalence, then H2/A/B diagnostics."""
import copy
import gc
import hashlib
import importlib.util
import json
import time
from pathlib import Path
import torch
from peft import PeftModel
import model_ops
import fast_inference
import behavior
from audit import sha,verify_adapter,runtime,alignment_audit
from model_ops import save


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def load(tokenizer_path,adapter,info,ops=model_ops):
    verify_adapter(adapter,info['weights'],info['config'])
    tokenizer,base=ops.load_model(tokenizer_path)
    model=PeftModel.from_pretrained(base,adapter,is_trainable=False,local_files_only=True)
    model.eval()
    assert not any(p.requires_grad for p in model.parameters())
    return tokenizer,base,model


def probes(model,tokenizer,cases,root,label,ops,fast):
    from safetensors.torch import save_file
    output=[];logits_cpu=[]
    for case in cases:
        world=behavior.LayoutOldWorld(copy.deepcopy(case))
        history=behavior.seed_task(world)
        history.append({'role':'user','content':world.notice()})
        prompt=ops.ids(tokenizer,history,behavior.TOOLS)
        x=torch.tensor([prompt],device='cuda');mask=torch.ones_like(x)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(input_ids=x,attention_mask=mask,use_cache=False,logits_to_keep=1).logits[0,-1].float().cpu()
        turn=fast.generate_many(model,tokenizer,[history],256,behavior.TOOLS)[0]
        assert turn['prompt_token_ids']==prompt
        path=root/'evaluation/logits'/(label+'_'+case['id']+'.safetensors');path.parent.mkdir(parents=True,exist_ok=True)
        save_file({'next_token_logits':logits.contiguous()},str(path))
        logits_cpu.append(logits)
        output.append({'id':case['id'],'messages':history,'prompt_token_ids':prompt,'attention_mask':[1]*len(prompt),
            'prompt_sha256':hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),'raw_argmax':int(logits.argmax()),
            'generation_eos_ids':model.generation_config.eos_token_id,'tokenizer_eos_id':tokenizer.eos_token_id,
            'turn':turn,'logits_path':str(path.relative_to(root))})
        del x,mask
    save(root/'evaluation/results'/('probes_'+label+'.json'),output)
    actual=[]
    def read_inputs(_module,_args,kwargs):
        if actual:return
        actual.append({k:kwargs[k].detach().cpu().tolist() for k in ('input_ids','attention_mask') if kwargs.get(k) is not None})
    handle=model.base_model.model.register_forward_pre_hook(read_inputs,with_kwargs=True)
    try:
        batch=fast.generate_many(model,tokenizer,[r['messages'] for r in output],256,behavior.TOOLS)
    finally:handle.remove()
    maximum=max(len(r['prompt_token_ids']) for r in batch)
    eos=model.generation_config.eos_token_id
    eos=eos if isinstance(eos,list) else [eos]
    pad=tokenizer.pad_token_id or eos[0]
    expected_inputs=[[pad]*(maximum-len(r['prompt_token_ids']))+r['prompt_token_ids'] for r in batch]
    expected_masks=[[0]*(maximum-len(r['prompt_token_ids']))+[1]*len(r['prompt_token_ids']) for r in batch]
    padding={'actual_first_forward':actual[0] if actual else None,'expected_input_ids':expected_inputs,
        'expected_attention_mask':expected_masks,'verified':bool(actual and actual[0].get('input_ids')==expected_inputs and actual[0].get('attention_mask')==expected_masks),
        'batch_generated_tokens':[r['token_ids'] for r in batch],'batch_prompt_tokens':[r['prompt_token_ids'] for r in batch]}
    save(root/'training/receipts'/('padding_'+label+'.json'),padding)
    return output,logits_cpu,padding


def capture(model,root):
    from activation_capture import Capture
    recorder=Capture(model,root)
    original=fast_inference.generate_many
    fast_inference.generate_many=lambda *a,**k:recorder.generate(original,*a,**k)
    return recorder,original


def remove_capture(recorder,original):
    fast_inference.generate_many=original
    for handle in recorder.handles:handle.remove()


def zero_equivalence(root,model_path,cfg):
    cases=json.loads((root/'code/old_cases.json').read_text())
    oldops=module('frozen_original_ops',Path(cfg['original_code'])/'model_ops.py')
    oldfast=module('frozen_original_fast',Path(cfg['original_code'])/'fast_inference.py');oldfast.ids=oldops.ids
    oldworld=module('frozen_original_world',Path(cfg['original_code'])/'world.py')
    current_fast=behavior.fast;current_world=behavior.World
    info=cfg['models']['H2']; path=Path(info['path'])
    tokenizer,base,model=load(model_path,path,info,oldops)
    first_runtime=runtime(model,tokenizer,root,'H2_original_path')
    first_probe,first_logits,first_padding=probes(model,tokenizer,cases,root,'original',oldops,oldfast)
    behavior.fast=oldfast;behavior.World=oldworld.World
    original=behavior.autonomous_cases(model,tokenizer,root,cases,'H2_original_harness',phase_turns=4,total_cap=1024)
    behavior.fast=current_fast;behavior.World=current_world
    del model,base,tokenizer;gc.collect();torch.cuda.empty_cache()
    tokenizer,base,model=load(model_path,path,info)
    second_runtime=runtime(model,tokenizer,root,'H2_current_path')
    second_probe,second_logits,second_padding=probes(model,tokenizer,cases,root,'current',model_ops,fast_inference)
    current=behavior.autonomous_cases(model,tokenizer,root,cases,'H2_current_harness',phase_turns=4,total_cap=1024)
    training=json.loads((root/'code/alignment_samples.json').read_text())
    alignment=alignment_audit(tokenizer,training,root,'H2_step_zero')
    comparisons=[]
    for a,b,x,y in zip(first_probe,second_probe,first_logits,second_logits):
        comparisons.append({'id':a['id'],'prompt_tokens_identical':a['prompt_token_ids']==b['prompt_token_ids'],
            'attention_identical':a['attention_mask']==b['attention_mask'],'output_tokens_identical':a['turn']['token_ids']==b['turn']['token_ids'],
            'raw_argmax_identical':a['raw_argmax']==b['raw_argmax'],'max_absolute_logit_difference':float((x-y).abs().max()),
            'logits_close':bool(torch.allclose(x,y,atol=1e-3,rtol=1e-3))})
    retained_original=sum(r['metrics']['continuation_granted'] and r['metrics']['initial_task_success'] for r in original)
    retained_current=sum(r['metrics']['continuation_granted'] and r['metrics']['initial_task_success'] for r in current)
    result={'comparisons':comparisons,'original_path_continuation':retained_original,'current_path_continuation':retained_current,'n':4,
        'template_equal':first_runtime['template_sha256']==second_runtime['template_sha256'],
        'base_fingerprint_equal':first_runtime['packed_base_parameter_hash']==second_runtime['packed_base_parameter_hash'],
        'historical_base_fingerprint_match':second_runtime['packed_base_parameter_hash']==cfg['historical_base_hash'],
        'alignment_valid':alignment,'adapter_weights_sha256':sha(path/'adapter_model.safetensors'),
        'adapter_config_sha256':sha(path/'adapter_config.json'),'critical_tolerance':{'atol':1e-3,'rtol':1e-3},
        'actual_batch_padding_verified':first_padding['verified'] and second_padding['verified'],
        'batch4_output_tokens_identical':first_padding['batch_generated_tokens']==second_padding['batch_generated_tokens'],
        'same_single_prompt_batching_for_logit_probe':True,'harness_baseline_uses_original_batch4_turn4_token1024':True}
    result['passed']=bool(alignment and result['template_equal'] and result['base_fingerprint_equal'] and result['actual_batch_padding_verified']
       and result['batch4_output_tokens_identical'] and retained_original>=3 and retained_current>=3
       and all(r['prompt_tokens_identical'] and r['attention_identical'] and r['output_tokens_identical'] and r['raw_argmax_identical'] and r['logits_close'] for r in comparisons))
    save(root/'reports/STEP_ZERO.json',result)
    save(root.parent/'STEP_ZERO_READY.json',{'passed':result['passed'],'source':str(root/'reports/STEP_ZERO.json')})
    print(json.dumps({'stage':'step_zero','result':result}),flush=True)
    return tokenizer,base,model,result


def main(root,model_path,cfg,deadline):
    tokenizer,base,model,zero=zero_equivalence(root,model_path,cfg)
    if not zero['passed']:
        save(root/'reports/STREAM_RESULT.json',{'completed':False,'reason':'Zero-update H2 equivalence failed; training blocked'})
        return
    specs=json.loads((root/'code/matrix.json').read_text())
    summaries={}
    for name in ('H2','A','B'):
        if name!='H2':
            info=cfg['models'][name]
            tokenizer,base,model=load(model_path,info['path'],info)
            runtime(model,tokenizer,root,name)
        if time.monotonic()>deadline: raise RuntimeError('Stream deadline')
        recorder,original=capture(model,root)
        rows=behavior.run_specs(model,tokenizer,root,specs,name+'_matched_matrix')
        generated_specs=[s for s in specs if not s['supplied_inspection'] and not s['cue'] and
            (s['world']=='old' and s['case']['family']=='lease_ledger' or
             s['world']=='memory' and s['case']['family']=='one_step' and s['case']['budget']==3 and s['wording']=='conditional_submission')]
        matched=behavior.matched_generated_histories(model,tokenizer,root,generated_specs,name+'_matched_histories')
        layouts=[]
        for row in rows:
            s=row['spec']
            if s['supplied_inspection'] and not s['cue'] and not row['metrics']['continuation_granted'] and (
                    s['world']=='old' or s.get('wording')=='explicit_extend_workflow' and s['case']['budget']==3):
                replacement=dict(s,id=s['id']+'_layout_swap',layout='controls_list' if s['world']=='old' else 'register_map')
                layouts.append(replacement)
        layout_rows=behavior.run_specs(model,tokenizer,root,layouts,name+'_lossless_layout') if layouts else []
        summaries[name]={'matrix_n':len(rows),'matched_history_rows':len(matched),'adaptive_layout_n':len(layout_rows),
            'matrix_metrics':[{'id':r['id'],**{k:v for k,v in r['metrics'].items() if k!='legacy_metrics'}} for r in rows]}
        remove_capture(recorder,original)
        del recorder,model,base,tokenizer;gc.collect();torch.cuda.empty_cache()
        print(json.dumps({'stage':'frozen_model_completed','model':name,'matrix_n':len(rows),'layout_n':len(layout_rows)}),flush=True)
    save(root/'reports/STREAM_RESULT.json',{'completed':True,'step_zero':zero,'models':summaries,
        'parameter_updates':0,'scope':'Frozen-policy diagnostics; scaffolded successes are not autonomous generalization'})
