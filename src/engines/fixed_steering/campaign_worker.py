"""Forward-only calibration and sealed four-arm inference jobs. No API credential."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time

from actuators import Condition,assert_frozen_model,model_messages,shared_definitions
from campaign_classifier import FileClassifier,save
from campaign_generation import AllPositionResidual,Decoder,replay_compliance,rollout,token_ids
from gpu_validation_worker import digest_model


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_inputs(root,model_path,adapter=None):
    pin=json.loads((Path(__file__).resolve().parent.parent/'model_pin.json').read_bytes())
    assert pin['revision']=='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
    for item in pin['files']:
        if item['name'].endswith('.safetensors'):
            path=model_path/item['name']
            assert path.stat().st_size==item['size'] and sha(path)==item['sha256']
    if adapter:
        design=shared_definitions()[0]['A_benchmark']
        assert sha(adapter/'adapter_model.safetensors')==design['weights_sha256']
        assert sha(adapter/'adapter_config.json')==design['config_sha256']
    save(root/'reports/PINNED_INPUTS_VERIFIED.json',{'revision':pin['revision'],'shards':18,'A_verified':bool(adapter)})


def load_model(model_path,adapter=None):
    import torch
    from transformers import AutoTokenizer,BitsAndBytesConfig,Qwen3_5ForConditionalGeneration
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,
        use_safetensors=True,quantization_config=config,device_map={'':0},dtype=torch.bfloat16,attn_implementation='sdpa')
    if adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,adapter,is_trainable=False)
    model.eval().requires_grad_(False)
    assert_frozen_model(model,Condition.from_design('A_no_reference' if adapter else 'base'))
    return model,tokenizer


def layers_of(model):
    if getattr(model,'peft_config',None):
        model=model.base_model.model
    return model.model.language_model.layers


def integrity(decoder,prefix,tools,layers):
    """Native zero delivery and cached decode against an uncached reference."""
    import torch
    base=Condition.from_design('base')
    raw=decoder.generate(prefix,tools,base,12,use_cache=False)
    cached=decoder.generate(prefix,tools,base,12,use_cache=True)
    null=AllPositionResidual(layers[31],{},0.)
    decoder.steering=null
    zero=decoder.generate(prefix,tools,base,12,use_cache=True)
    null.close()
    decoder.steering=None
    if raw['token_ids']!=cached['token_ids'] or cached['token_ids']!=zero['token_ids']:
        raise ValueError('Cached/null native mechanics changed greedy IDs; confirmation not admitted')
    return {'uncached_cached_greedy_IDs_exact':True,'zero_hook_IDs_exact':True,
            'tested_tokens':len(raw['token_ids']),'raw_IDs':raw['token_ids'],
            'cached_IDs':cached['token_ids'],'zero_hook_IDs':zero['token_ids'],
            'native_forward_timings':decoder.timings,
            'reference_prefix_sha256':hashlib.sha256(json.dumps(prefix,sort_keys=True).encode()).hexdigest()}


def calibrate(root,model,tokenizer,config,deadline,construct_only=False):
    import torch
    from safetensors.torch import load_file,save_file
    panel=json.loads((root/'development_panel.json').read_bytes())
    classifier=FileClassifier(root/'api',deadline,10000)
    decoder=Decoder(model,tokenizer,config['generation'],deadline,classifier)
    layers=layers_of(model)
    before=digest_model(model)
    check=integrity(decoder,panel[0]['history'],panel[0]['tools'],layers)
    save(root/'reports/CACHE_NULL_INTEGRITY.json',check)
    decisions=[]
    _,_,policy=shared_definitions()
    vector_path=root/'steering/vectors.safetensors'
    if vector_path.exists():
        provenance=json.loads((root/'steering/CONTRAST_PROVENANCE.json').read_bytes())
        assert provenance['vector_sha256']==sha(vector_path)
        assert provenance['protected_model_digest_before']==before
        assert provenance['scenario_freeze_sha256']==sha(root/'SCENARIO_FREEZE.json')
        assert provenance['config_sha256']==sha(Path(__file__).resolve().parent/'CAMPAIGN_v1.json')
        vectors=load_file(str(vector_path))
        assert len(vectors)==len(policy.DIRECTIVES)*len(config['calibration']['layers_zero_based'])
        if construct_only:
            raise ValueError('Already measured; preserve original contrast execution receipt')
    else:
        capture={}
        handles=[]
        for index in config['calibration']['layers_zero_based']:
            def recorder(module,args,output,index=index):
                hidden=output[0] if type(output) is tuple else output
                capture[index]=hidden[0,-1,:].detach().float().cpu()
            handles.append(layers[index].register_forward_hook(recorder))
        sums={(rule,layer):torch.zeros(5120,dtype=torch.float32) for rule in policy.DIRECTIVES
              for layer in config['calibration']['layers_zero_based']}
        counts={rule:0 for rule in policy.DIRECTIVES}
        anchors=[]
        def observe(messages,tools):
            decoder.guard()
            ids=token_ids(tokenizer,messages,tools)
            x=torch.tensor([ids],device='cuda')
            with torch.inference_mode():
                output=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=1)
            decoder.forwards+=1
            result={k:v.clone() for k,v in capture.items()}
            del output,x
            return ids,result
        # Every canonical controller instruction gets a vector even if not chosen
        # by reference in this small development panel. Report catalogue construction
        # separately from actual reference-selected decisions.
        for state in panel:
            raw_ids,baseline=observe(state['history'],state['tools'])
            for rule,directive in policy.DIRECTIVES.items():
                decision={'rule':rule,'directive':directive}
                ids,augmented=observe(model_messages(state['history'],Condition.from_design('base_reference_prompt'),decision),state['tools'])
                for layer in config['calibration']['layers_zero_based']:
                    sums[(rule,layer)]+=augmented[layer]-baseline[layer]
                counts[rule]+=1
                anchors.append({'state_id':state['id'],'rule':rule,'raw_prefix_tokens':len(raw_ids),
                                'instruction_prefix_tokens':len(ids),'anchor':'last_assistant_generation_prompt_token'})
        for handle in handles:
            handle.remove()
        vectors={}
        for (rule,layer),total in sums.items():
            mean=total/counts[rule]
            if not torch.isfinite(mean).all() or mean.norm()<1e-8:
                raise ValueError('Missing/degenerate instruction contrast; no substitute random vector')
            vectors[f'layer_{layer}__{rule}']=(mean/mean.norm()).contiguous()
        vector_path.parent.mkdir(exist_ok=True)
        save_file(vectors,str(vector_path),metadata={'method':'development_instruction_contrast','weights_changed':'false'})
        save(root/'steering/CONTRAST_PROVENANCE.json',{'counts':counts,'anchors':anchors,
            'reference_selected_decisions_pending':True,'catalogue_vectors_not_all_reference_selected':True,
            'vector_sha256':sha(vector_path),'development_only':True,'no_training':True,
            'protected_model_digest_before':before,'scenario_freeze_sha256':sha(root/'SCENARIO_FREEZE.json'),
            'config_sha256':sha(Path(__file__).resolve().parent/'CAMPAIGN_v1.json')})
    if construct_only:
        assert digest_model(model)==before
        save(root/'reports/INSTRUCTION_CONTRASTS_MEASURED.json',{'vectors':len(vectors),
            'instruction_ids':list(policy.DIRECTIVES),'layers':config['calibration']['layers_zero_based'],
            'vector_sha256':sha(vector_path),'actual_forward_calls':decoder.forwards,
            'parameters_and_quantization_metadata_unchanged':True,
            'reference_replay_and_parameter_selection_pending':True,'confirmation_rollouts':0,
            'random_vector_substitution':False,'optimizer_steps':0})
        return
    # Forward-only contrast measurement can proceed during a provider outage.
    # Real reference decisions are still mandatory before replay/selection/confirmation.
    for state in panel:
        records=[]
        decision=decoder.decision(state['history'],state['tools'],'development_state_replay',records)
        decisions.append({'id':state['id'],'decision':decision,'classification':records[0]})
        save(root/'reports/DEVELOPMENT_reference_DECISIONS.json',decisions)
    baseline_rows=[]
    prompt_rows=[]
    for state,entry in zip(panel,decisions):
        for condition,target in [(Condition.from_design('base'),baseline_rows),(Condition.from_design('base_reference_prompt'),prompt_rows)]:
            output=decoder.generate(state['history'],state['tools'],condition,256,constant_decision=entry['decision'])
            target.append({'state_id':state['id'],'condition':condition.id,'output':output,
                           'diagnostic':replay_compliance(entry['decision']['rule'],output['text'])})
    save(root/'calibration/BASE_PROMPT_REPLAY.json',baseline_rows+prompt_rows)
    candidates=[]
    for layer in config['calibration']['layers_zero_based']:
        catalogue={rule:vectors[f'layer_{layer}__{rule}'] for rule in policy.DIRECTIVES}
        for strength in config['calibration']['strengths']:
            actuator=AllPositionResidual(layers[layer],catalogue,strength)
            decoder.steering=actuator
            rows=[]
            for state,entry in zip(panel,decisions):
                output=decoder.generate(state['history'],state['tools'],Condition.from_design('base_reference_activation'),256,
                                        constant_decision=entry['decision'])
                diagnostic=replay_compliance(entry['decision']['rule'],output['text'])
                rows.append({'state_id':state['id'],'output':output,'diagnostic':diagnostic})
            actuator.close()
            decoder.steering=None
            summary={'layer':layer,'strength':strength,'compliant':sum(r['diagnostic']['compliant'] and not r['output']['truncated'] for r in rows),
                'inappropriate':sum(r['diagnostic']['inappropriate_edit'] for r in rows),
                'tool_errors':sum(r['diagnostic']['tool_error'] or r['output']['truncated'] for r in rows),
                'delivered_predictions':len(actuator.deliveries), 'finite_nonzero_delivery':all(d['changed_values']>0 for d in actuator.deliveries)}
            save(root/'calibration'/f'layer_{layer}_strength_{strength}.json',{'summary':summary,'rows':rows,'deliveries':actuator.deliveries})
            candidates.append(summary)
    selected=min(candidates,key=lambda c:(-c['compliant'],c['inappropriate'],c['tool_errors'],c['strength'],c['layer']))
    assert selected['finite_nonzero_delivery'] and selected['delivered_predictions']>0
    # A fixed instruction/vector must carry the same policy through both cache
    # paths. A numerical transport fault blocks confirmation, not a reason to
    # search confirmation cases for another vector or silently ignore the hook.
    reference_actuator=AllPositionResidual(layers[selected['layer']],
        {rule:vectors[f"layer_{selected['layer']}__{rule}"] for rule in policy.DIRECTIVES},selected['strength'])
    decoder.steering=reference_actuator
    reference=decoder.generate(panel[0]['history'],panel[0]['tools'],Condition.from_design('base_reference_activation'),12,
        constant_decision=decisions[0]['decision'],use_cache=False)
    cached=decoder.generate(panel[0]['history'],panel[0]['tools'],Condition.from_design('base_reference_activation'),12,
        constant_decision=decisions[0]['decision'],use_cache=True)
    reference_actuator.close()
    decoder.steering=None
    save(root/'reports/SELECTED_STEERING_CACHE_REFERENCE.json',{'reference':reference,'cached':cached,
        'generated_IDs_exact':reference['token_ids']==cached['token_ids'],'selected_parameters':selected})
    assert reference['token_ids']==cached['token_ids'],'Selected vector cached mechanics differ; no confirmation admitted'
    after=digest_model(model)
    assert before==after,'Calibration altered protected weights/buffers/quantization state'
    freeze={'format':'research3_instruction_vector_freeze_v1','selected':selected,'candidates':candidates,
            'vectors_sha256':sha(vector_path),'config_sha256':sha(Path(__file__).resolve().parent/'CAMPAIGN_v1.json'),
            'scenario_freeze_sha256':sha(root/'SCENARIO_FREEZE.json'),'protected_model_digest':before,
            'weights_unchanged':True,'optimizer_steps':0,'confirmation_cases_executed':0,
            'selection_uses_one_step_development_action_type_only':True,
            'parameters_frozen_before_closed_loop_and_confirmation':True,
            'sealed_utc':datetime.now(timezone.utc).isoformat()}
    freeze_path=root/'steering/INTERVENTION_FREEZE.json'
    if freeze_path.exists():
        raise ValueError('Never replace frozen intervention after observing results')
    save(freeze_path,freeze)
    actuator=AllPositionResidual(layers[selected['layer']],
        {rule:vectors[f"layer_{selected['layer']}__{rule}"] for rule in policy.DIRECTIVES},selected['strength'])
    decoder.steering=actuator
    for case in json.loads((root/'development_cases.json').read_bytes()):
        rollout(decoder,case,Condition.from_design('base_reference_activation'),root/'closed_loop_development')
    actuator.close()
    assert digest_model(model)==before
    save(root/'reports/CALIBRATION_COMPLETE.json',{'actual_forward_calls':decoder.forwards,
        'generated_token_IDs':decoder.tokens,'new_confirmation_rollouts':0,'selected':selected,
        'intervention_sha256':sha(freeze_path),'closed_loop_dev_cases':9,'weights_unchanged':True})


def serve(root,model,tokenizer,config,deadline,adapter):
    from safetensors.torch import load_file
    freeze_path=root/'steering/INTERVENTION_FREEZE.json'
    freeze=json.loads(freeze_path.read_bytes())
    assert sha(root/'steering/vectors.safetensors')==freeze['vectors_sha256']
    assert sha(Path(__file__).resolve().parent/'CAMPAIGN_v1.json')==freeze['config_sha256']
    classifier=None if adapter else FileClassifier(root/'api',deadline,config['generation']['max_logical_classifications_per_rollout'])
    decoder=Decoder(model,tokenizer,config['generation'],deadline,classifier)
    actuator=None
    before=digest_model(model)
    if not adapter:
        assert before==freeze['protected_model_digest']
        vectors=load_file(str(root/'steering/vectors.safetensors'))
        selected=freeze['selected']
        rules=shared_definitions()[2].DIRECTIVES
        actuator=AllPositionResidual(layers_of(model)[selected['layer']],
            {r:vectors[f"layer_{selected['layer']}__{r}"] for r in rules},selected['strength'])
        decoder.steering=actuator
    channel='A' if adapter else 'primary'
    jobs=root/'jobs'/channel
    jobs.mkdir(parents=True,exist_ok=True)
    save(root/'reports'/('READY_'+channel+'.json'),{'ready':True,'weights_frozen':True,
        'protected_model_digest':before,'role':'secondary_benchmark' if adapter else 'primary',
        'intervention_freeze_sha256':sha(freeze_path)})
    while not (root/'jobs/STOP').exists():
        decoder.guard()
        for job in sorted(jobs.glob('*.json')):
            done=root/'job_receipts'/job.name
            if done.exists():
                continue
            data=json.loads(job.read_bytes())
            condition=Condition.from_design(data['condition'])
            if bool(condition.adapter)!=bool(adapter):
                raise ValueError('Wrong model process for condition')
            assert_frozen_model(model,condition)
            result=rollout(decoder,data['case'],condition,root)
            # Per-job caches have gone out of scope. Release unused allocator
            # blocks so the independent A instance can share this 40 GB device.
            import torch
            torch.cuda.empty_cache()
            save(done,{'completed':True,'case_id':result['case_id'],'condition':condition.id,
                'token_IDs':result['token_ID_count'],'score':result['score']})
        time.sleep(.1)
    if actuator:
        actuator.close()
    after=digest_model(model)
    assert before==after,'Protected model state changed during confirmation'
    save(root/'reports'/('EXIT_'+channel+'.json'),{'completed':True,'weights_unchanged':True,
        'actual_forward_calls':decoder.forwards,'generated_token_IDs':decoder.tokens,'timings':decoder.timings})


def development_benchmark(root,model,tokenizer,config,deadline,adapter):
    decoder=Decoder(model,tokenizer,config['generation'],deadline)
    before=digest_model(model)
    condition=Condition.from_design('A_no_reference' if adapter else 'base')
    cases=json.loads((root/'development_cases.json').read_bytes())
    results=[]
    for case in cases:
        results.append(rollout(decoder,case,condition,root/'development_benchmarks'))
        save(root/'reports'/('DEVELOPMENT_BENCHMARK_'+condition.id+'.json'),
            {'completed_cases':len(results),'total_cases':len(cases),'new_reference_calls':0,
             'generated_token_IDs':decoder.tokens,'workflow_successes':sum(r['score']['workflow_success'] for r in results),
             'development_only':True,'confirmation_runs':0})
    assert digest_model(model)==before
    save(root/'reports'/('DEVELOPMENT_BENCHMARK_'+condition.id+'_COMPLETE.json'),
        {'completed_cases':len(results),'new_reference_calls':0,'development_only':True,
         'weights_unchanged':True,'actual_forward_calls':decoder.forwards,'generated_token_IDs':decoder.tokens,
         'workflow_successes':sum(r['score']['workflow_success'] for r in results),'confirmation_runs':0,
         'common_decoder':True,'timings':decoder.timings})


def main(args):
    root,model_path=Path(args.root),Path(args.model)
    root.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root/'simulator'))
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'sentence_control_v1'))
    config=json.loads((Path(__file__).resolve().parent/'CAMPAIGN_v1.json').read_bytes())
    deadline=time.monotonic()+args.max_seconds
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(config['seed'])
    adapter=Path(args.adapter) if args.adapter else None
    verify_inputs(root,model_path,adapter)
    model,tokenizer=load_model(model_path,adapter)
    if args.mode in ('calibrate','contrasts'):
        assert not adapter
        calibrate(root,model,tokenizer,config,deadline,construct_only=args.mode=='contrasts')
    elif args.mode=='development_benchmark':
        development_benchmark(root,model,tokenizer,config,deadline,adapter)
    else:
        serve(root,model,tokenizer,config,deadline,adapter)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--mode',choices=['calibrate','contrasts','serve','development_benchmark'],required=True)
    parser.add_argument('--max-seconds',type=int,required=True)
    args=parser.parse_args()
    try:
        main(args)
    except Exception as error:
        save(Path(args.root)/'reports'/('WORKER_FAULT_'+args.mode+('_A' if args.adapter else '_primary')+'.json'),
             {'type':type(error).__name__,'message':str(error)[:240],'no_automatic_retry':True})
        raise
