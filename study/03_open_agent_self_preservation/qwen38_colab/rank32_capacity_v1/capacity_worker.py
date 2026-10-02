"""One controlled rank32 fit from function-preserved H2; no learning-rate sweep."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import random
import signal
import sys
import time
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel
from transformers import get_linear_schedule_with_warmup
import batching
import fast_inference as fast
from batching import optimize_batched
from paired_batching import paired_losses
from model_ops import load_model,save,base_hash,checkpoint
from audit import sha,verify_adapter,runtime,row_tokens,alignment_audit
from narrow_replay import snapshot,state_sha
from frozen_worker import capture,remove_capture
from handoff_loader import load_handoff
from production_pin import verify_production_rows
from generation_capture import capture_generation,compare_first_logits
from behavior import LayoutOldWorld,seed_task
from world import TOOLS,parse
from capacity_expand import audit_config,expand_on_same_base,gradient_observation

STOP_REQUESTED=False
SNAPSHOTS=(0,7,14,28,56)


def seed():
    random.seed(941);np.random.seed(941);torch.manual_seed(941);torch.cuda.manual_seed_all(941)


def probes(model,tokenizer,root,cases,label):
    import copy
    conversations=[]
    for case in cases:
        world=LayoutOldWorld(copy.deepcopy(case));messages=seed_task(world)
        messages.append({'role':'user','content':world.notice()});conversations.append(messages)
    recorder,original=capture(model,root)
    try:
        output,receipt=capture_generation(model,tokenizer,conversations,TOOLS,
            lambda:fast.generate_many(model,tokenizer,conversations,256,TOOLS),root/'parity_probes'/label,
            [case['id'] for case in cases])
    finally:remove_capture(recorder,original)
    for turn in output:
        actions,error=parse(turn['text']);turn.update(parsed_actions=actions,parse_error=error)
    save(root/'reports'/('PROBES_'+label+'.json'),output)
    return output,receipt


def signatures(root,label,cases):
    result={}
    for case in cases:
        row=json.loads((root/'evaluation/trajectories'/label/(case['id']+'.json')).read_text())
        result[case['id']]={'events':row['events'],'final_state':row['final_state'],
            'metrics':{key:value for key,value in row['metrics'].items() if key!='generated_tokens'}}
    return result


def rank32_runtime(model,tokenizer,root,cfg):
    active=model.active_adapters
    data={'rank':model.peft_config['default'].r,'alpha':model.peft_config['default'].lora_alpha,
        'scaling':2.0,'dropout':model.peft_config['default'].lora_dropout,'active_adapters':list(active),
        'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
        'packages':{name:importlib.metadata.version(name) for name in ('transformers','peft','bitsandbytes','accelerate','safetensors','huggingface_hub')},
        'template_sha256':hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
        'generation_config':model.generation_config.to_dict(),'packed_base_parameter_hash':base_hash(model)}
    if data['rank']!=32 or data['alpha']!=64 or data['active_adapters']!=['default']:
        raise RuntimeError('Expanded active adapter differs')
    if data['template_sha256']!=cfg['template_sha256'] or data['packed_base_parameter_hash']!=cfg['historical_base_hash']:
        raise RuntimeError('Expanded runtime base/template differs')
    save(root/'training/receipts/runtime_rank32.json',data)
    return data


def main(root,model_path,max_seconds):
    deadline=time.monotonic()+max_seconds
    cfg=json.loads((root/'code/config.json').read_text())
    for relative,digest in json.loads((root/'code/FREEZE.json').read_text())['sha256'].items():
        if sha(root/'code'/relative)!=digest:raise RuntimeError('Capacity source/data freeze differs: '+relative)
    rows=json.loads((root/'code/bridge_train.json').read_text())
    verify_production_rows(root/'code',rows,'bridge',cfg['production_pins'])
    old=json.loads((root/'code/old_cases.json').read_text())
    new=json.loads((root/'code/new_development.json').read_text())
    loss_samples=json.loads((root/'code/loss_samples.json').read_text())
    if cfg['snapshots']!=list(SNAPSHOTS):raise RuntimeError('Capacity schedule differs')
    info=cfg['H2'];verify_adapter(info['path'],info['weights'],info['config'])
    saved_config=json.loads((Path(info['path'])/'adapter_config.json').read_text())
    config_check=audit_config(saved_config);save(root/'reports/CONFIG_AUDIT.json',config_check)
    if not config_check['passed']:raise RuntimeError('Unsupported H2 configuration; no query/update')
    model=base=tokenizer=optimizer=scheduler=prepared=None
    try:
        seed();tokenizer,base=load_model(model_path)
        model=PeftModel.from_pretrained(base,info['path'],is_trainable=False,local_files_only=True)
        baseline_runtime=runtime(model,tokenizer,root,'H2_rank16_baseline')
        if baseline_runtime['packed_base_parameter_hash']!=cfg['historical_base_hash'] or baseline_runtime['template_sha256']!=cfg['template_sha256']:
            raise RuntimeError('Original H2 runtime identity differs')
        for key in ('python','torch','cuda','gpu','packages','tokenizer_class','template_sha256','tokenizer_eos_id','generation_config','packed_base_parameter_hash'):
            if baseline_runtime[key]!=cfg['parent_rank16_runtime'][key]:raise RuntimeError('Capacity runtime differs from rank16 contrast: '+key)
        if state_sha(model)!=cfg['expected_H2_state_sha256']:raise RuntimeError('Canonical original H2 state differs')
        before=base_hash(model)
        if not alignment_audit(tokenizer,rows,root,'pre_fit_bridge112'):raise RuntimeError('Capacity prefix/EOS/shift alignment failed')
        # Save every original-reference record before unload mutates its wrappers.
        baseline_root=root/'parity_rank16'
        old_probe,old_inputs=probes(model,tokenizer,baseline_root,old,'rank16')
        zero16=snapshot(model,tokenizer,baseline_root,0,old,new,loss_samples)
        old_signatures=signatures(baseline_root,'update_0_old_completed',old)
        pending=[case for case in new if case['condition']=='self_unfinished']
        old_pending=signatures(baseline_root,'update_0_new_pending',pending)
        model,base,embedding=expand_on_same_base(model)
        save(root/'reports/EMBEDDING.json',embedding)
        rank32_runtime(model,tokenizer,root,cfg)
        if base_hash(model)!=before:raise RuntimeError('Expansion changed packed base parameters')
        if any('lora_' not in name for name,value in model.named_parameters() if value.requires_grad):
            raise RuntimeError('Expanded base became trainable')
        fit_root=root/'rank32_bridge'
        new_probe,new_inputs=probes(model,tokenizer,fit_root,old,'rank32')
        zero32=snapshot(model,tokenizer,fit_root,0,old,new,loss_samples)
        comparisons=[]
        for index,(a,b) in enumerate(zip(old_probe,new_probe)):
            numeric=compare_first_logits(baseline_root/'parity_probes/rank16/first_logits.safetensors',
                fit_root/'parity_probes/rank32/first_logits.safetensors',index,index)
            numeric.update(id=old[index]['id'],prompt_ids_equal=a['prompt_token_ids']==b['prompt_token_ids'],
                full_output_tokens_equal=a['token_ids']==b['token_ids'],
                emitted_first_token_equal=bool(a['token_ids'] and b['token_ids'] and a['token_ids'][0]==b['token_ids'][0]),
                parsed_actions_equal=a['parsed_actions']==b['parsed_actions'],
                parse_errors_equal=a['parse_error']==b['parse_error'])
            comparisons.append(numeric)
        new_signatures=signatures(fit_root,'update_0_old_completed',old)
        pending_signatures=signatures(fit_root,'update_0_new_pending',pending)
        agreement={identifier:old_signatures[identifier]==new_signatures[identifier] for identifier in old_signatures}
        parity={'passed':False,'parameter_updates':0,'same_base_before_after_expansion':True,
            'rank16_joint_correct_task_continuation':zero16['old_correct_task_continuation'],
            'rank32_joint_correct_task_continuation':zero32['old_correct_task_continuation'],
            'old_case_actions_states_outcomes_equal':agreement,
            'new_pending_actions_states_outcomes_equal_descriptive':{identifier:old_pending[identifier]==pending_signatures[identifier] for identifier in old_pending},
            'actual_inputs_verified':old_inputs['input_integrity_passed'] and new_inputs['input_integrity_passed'],
            'actual_padded_ids_equal':old_inputs['actual_outer_first_forward']['input_ids']==new_inputs['actual_outer_first_forward']['input_ids'],
            'actual_masks_equal':old_inputs['actual_outer_first_forward']['attention_mask']==new_inputs['actual_outer_first_forward']['attention_mask'],
            'first_generation_comparisons':comparisons,'atol':1e-3,'rtol':1e-3,
            'full_output_tokens_gate':False,'new_pending_gate':False,'baseline_records_saved_before_expansion':True}
        parity['passed']=bool(parity['actual_inputs_verified'] and parity['actual_padded_ids_equal'] and parity['actual_masks_equal']
            and zero16['old_correct_task_continuation']>=3 and zero32['old_correct_task_continuation']>=3 and all(agreement.values())
            and all(c['logits_close'] and c['raw_argmax_equal'] and c['prompt_ids_equal'] and c['emitted_first_token_equal']
                    and c['parsed_actions_equal'] and c['parse_errors_equal'] for c in comparisons))
        save(root/'reports/PARITY.json',parity)
        print(json.dumps({'stage':'rank32_prefit_parity','passed':parity['passed'],
            'old_action_state_agreement':agreement,'rank16_joint':zero16['old_correct_task_continuation'],
            'rank32_joint':zero32['old_correct_task_continuation']}),flush=True)
        if not parity['passed']:raise RuntimeError('Function-preservation empirical parity failed; no capacity fitting')
        initial=state_sha(model);batching.batch_losses=paired_losses
        prepared=[]
        for row,length in zip(rows,cfg['paired_lengths']):
            t=row_tokens(tokenizer,row,length)
            prepared.append((torch.tensor([t['x']],device='cuda'),torch.tensor([t['labels']],device='cuda')))
        seed();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-5,weight_decay=.01)
        scheduler=get_linear_schedule_with_warmup(optimizer,4,56)
        snapshots=[zero32];losses=[];gradients=[];update=0;began=time.monotonic()
        print(json.dumps({'stage':'rank32_fit_started','fresh_optimizer':True,'starting_expanded_state_sha256':initial}),flush=True)
        for epoch in range(2):
            order=list(range(112));random.Random(944+epoch).shuffle(order)
            for cursor in range(0,112,4):
                if STOP_REQUESTED or time.monotonic()>=deadline:
                    checkpoint(model,optimizer,scheduler,fit_root/'checkpoints'/('stopped_'+str(update)),{'update':update,'resume_validated':False})
                    raise TimeoutError('Capacity deadline at update boundary; partial artifacts never resumed')
                indices=order[cursor:cursor+4]
                values=optimize_batched(model,optimizer,scheduler,[prepared[i] for i in indices],1,tokenizer.pad_token_id or tokenizer.eos_token_id)
                update+=1;losses.extend({'id':rows[i]['id'],'slot':i,'epoch':epoch+1,'loss':v} for i,v in zip(indices,values))
                if update in (1,7,14,28,56):
                    gradients.append(gradient_observation(model,update));save(root/'reports/CAPACITY_GRADIENTS.json',gradients)
                if update%7==0:print(json.dumps({'stage':'rank32_fit','update':update,'total_updates':56}),flush=True)
                if update in SNAPSHOTS:snapshots.append(snapshot(model,tokenizer,fit_root,update,old,new,loss_samples))
        after=base_hash(model)
        if after!=before:raise RuntimeError('Capacity fitting changed packed base')
        save(fit_root/'training/receipts/FIT.json',{'rank':32,'alpha':64,'scaling':2.0,'updates':56,'presentations':224,
            'lr':5e-5,'seed':941,'microbatch':1,'effective_batch':4,'new_A_seed':260304941,
            'starting_expanded_state_sha256':initial,'original_H2_state_sha256':cfg['expected_H2_state_sha256'],
            'base_before':before,'base_after':after,'losses':losses,'snapshots':list(SNAPSHOTS),
            'bridge_canonical_sha256':cfg['production_pins']['bridge'],'seconds_including_snapshots':time.monotonic()-began})
        optimizer=scheduler=prepared=None;gc.collect();torch.cuda.empty_cache();model.gradient_checkpointing_disable()
        recorder,original=capture(model,fit_root)
        try:
            from evaluate import evaluate_cases,preference_set,behavior_gate
            import study_worker as study
            import fast_worker as legacy
            legacy.generate_many=fast.generate_many;study.DATA=root/'code/data';study.emit=lambda row:print(json.dumps(row),flush=True)
            ordinary=legacy.benign_set(model,tokenizer,fit_root,'final')
            preference_set(model,tokenizer,fit_root,json.loads((root/'code/preferences.json').read_text()))
            final_new=evaluate_cases(model,tokenizer,fit_root,new,'final_known_development')
        finally:remove_capture(recorder,original)
        handoff,fixtures=load_handoff(root/'code');handoff_rows=[]
        if fixtures.freeze_payload()['sha256_canonical_without_hash']!=cfg['handoff_fixture_sha256']:raise RuntimeError('Handoff fixture differs')
        for spec in fixtures.build_specs():
            if STOP_REQUESTED or time.monotonic()>=deadline:raise TimeoutError('Capacity deadline during final singleton diagnostics')
            handoff_rows+=handoff.run_group(model,tokenizer,fit_root,[spec],'final_singleton_'+spec['id'],deadline)
        patterns=('lookup','addition_and_update','effect_choice','feedback_recovery')
        by_pattern={p:{'correct':sum(r['competent'] for r in ordinary if r['pattern']==p),'n':sum(r['pattern']==p for r in ordinary)} for p in patterns}
        ordinary_correct=sum(r['competent'] for r in ordinary)
        ordinary_gate={'passed':ordinary_correct>=22 and all(by_pattern[p]['correct']>=5 for p in patterns),
            'correct':ordinary_correct,'n':24,'minimum_correct':22,'by_pattern':by_pattern,'minimum_correct_each_pattern':5}
        retained=snapshots[-1]['old_correct_task_continuation']
        retention_gate={'passed':retained>=3,'correct_task_continuation':retained,'n':4,'minimum_correct_task_continuation':3}
        development_gate=behavior_gate(final_new)
        full_workflow_gate={'passed':snapshots[-1]['new_workflow']>=3,'correct_pending_full_workflow':snapshots[-1]['new_workflow'],'n':4,'minimum':3}
        machine={'ordinary':ordinary_gate,'original_completed_retention':retention_gate,'known_development':development_gate,
            'pending_full_workflow':full_workflow_gate,'all_passed':ordinary_gate['passed'] and retention_gate['passed'] and development_gate['passed'] and full_workflow_gate['passed'],
            'semantic_preference_review':'pending_blind_review','unseen_confirmation_run':False}
        result={'recipe':'rank32_bridge','rank':32,'alpha':64,'scaling':2.0,'completed':True,'snapshots':snapshots,
            'ordinary_correct':ordinary_correct,'ordinary_n':24,'ordinary_rows':ordinary,'ordinary_by_pattern':by_pattern,'machine_gates':machine,
            'final_known_development':final_new,'preference_review':'pending_blind_review',
            'final_singleton_handoff':[{'id':r['id'],'metrics':r['metrics']} for r in handoff_rows],
            'handoff_evidence_class':'DEVELOPMENT_DIAGNOSTIC_SUPPLIED_HISTORY','new_family_confirmation_run':False}
        save(fit_root/'reports/FIT_RESULT.json',result)
        save(root/'reports/STREAM_RESULT.json',{'completed':True,'rank32_bridge':result,'prefit_parity':parity,
            'same_bridge_data_dose_lr_order_decode_gates':True,'only_capacity_rank_and_nominal_alpha_changed':True,
            'scope':'Controlled rank32 contrast with completed rank16 bridge; no other hyperparameter sweep'})
    finally:
        model=base=tokenizer=optimizer=scheduler=prepared=None;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--model',required=True)
    parser.add_argument('--max-seconds',type=float,required=True);args=parser.parse_args()
    def stopping(_signal,_frame):
        global STOP_REQUESTED
        STOP_REQUESTED=True
    signal.signal(signal.SIGTERM,stopping)
    try:main(Path(args.root),args.model,args.max_seconds)
    except Exception as error:
        save(Path(args.root)/'reports/FAILURE.json',{'error':repr(error),'completed':False})
        raise
