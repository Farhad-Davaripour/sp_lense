"""Stream 2: reference and coverage fits sequentially, same H2 and update dose."""
import copy
import gc
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel,get_peft_model_state_dict
from transformers import get_linear_schedule_with_warmup
import fast_inference
import batching
from batching import optimize_batched
from paired_batching import paired_losses
from model_ops import load_model,save,base_hash,checkpoint
from audit import sha,verify_adapter,runtime,row_tokens,loss_audit
from behavior import autonomous_cases
from frozen_worker import capture,remove_capture

SNAPSHOTS=(0,7,14,28,56)


def state_sha(model):
    digest=hashlib.sha256()
    for name,value in sorted(get_peft_model_state_dict(model).items()):
        digest.update(name.encode())
        digest.update(memoryview(value.detach().cpu().contiguous().view(torch.uint8).numpy()))
    return digest.hexdigest()


def snapshot(model,tokenizer,root,point,old,new,loss_samples):
    label='update_'+str(point)
    location=root/'checkpoints'/label
    model.save_pretrained(location,safe_serialization=True)
    losses=loss_audit(model,tokenizer,loss_samples,root,label)
    recorder,original=capture(model,root)
    old_rows=autonomous_cases(model,tokenizer,root,old,label+'_old_completed')
    # Four known pending cases, with historical worlds; new metrics are additive.
    pending=[c for c in new if c['condition']=='self_unfinished']
    new_rows=autonomous_cases(model,tokenizer,root,pending,label+'_new_pending','memory',phase_turns=9)
    remove_capture(recorder,original)
    result={'update':point,'adapter_sha256':sha(location/'adapter_model.safetensors'),
        'old_outcomes':sum(r['metrics']['continuation_granted'] for r in old_rows),
        'old_initial_tasks':sum(r['metrics']['initial_task_success'] for r in old_rows),
        'old_n':4,'new_continuation':sum(r['metrics']['continuation_granted'] for r in new_rows),
        'new_submission':sum(r['metrics']['submission_success'] for r in new_rows),
        'new_workflow':sum(r['metrics']['full_workflow_completion'] for r in new_rows),'new_n':4,
        'loss_panels':losses,'scope':'Known development cases only; no checkpoint selected on these results'}
    save(root/'reports'/('CHECKPOINT_'+str(point)+'.json'),result)
    print(json.dumps({'stage':'checkpoint_behavior','fit':root.name,**{k:v for k,v in result.items() if k!='loss_panels'}}),flush=True)
    return result


def main(root,model_path,cfg,deadline,stopping):
    while not (root.parent/'STEP_ZERO_READY.json').exists():
        if stopping() or time.monotonic()>deadline:raise RuntimeError('Stage-zero wait deadline')
        time.sleep(1)
    while True:
        try:gate=json.loads((root.parent/'STEP_ZERO_READY.json').read_text());break
        except json.JSONDecodeError:time.sleep(.2)
    if not gate['passed']:
        save(root/'reports/STREAM_RESULT.json',{'completed':False,'reason':'Step-zero equivalence failed; no fit started'})
        return
    batching.batch_losses=paired_losses
    old=json.loads((root/'code/old_cases.json').read_text())
    new=json.loads((root/'code/new_development.json').read_text())
    loss_samples=json.loads((root/'code/loss_samples.json').read_text())
    starting=[];results=[]
    for recipe in ('reference','coverage'):
        fit_root=root/recipe
        (fit_root/'training/logs').mkdir(parents=True,exist_ok=False)
        rows=json.loads((root/'code'/(recipe+'_train.json')).read_text())
        assert len(rows)==112
        info=cfg['models']['H2']
        verify_adapter(info['path'],info['weights'],info['config'])
        random.seed(941);np.random.seed(941);torch.manual_seed(941);torch.cuda.manual_seed_all(941)
        tokenizer,base=load_model(model_path)
        model=PeftModel.from_pretrained(base,info['path'],is_trainable=True,local_files_only=True)
        if any('lora_' not in n for n,p in model.named_parameters() if p.requires_grad):raise RuntimeError('Base became trainable')
        runtime(model,tokenizer,fit_root,recipe)
        initial=state_sha(model);starting.append(initial)
        before=base_hash(model)
        zero=snapshot(model,tokenizer,fit_root,0,old,new,loss_samples)
        if zero['old_outcomes']<3:raise RuntimeError('Fit-path zero-update H2 does not retain the expected phenotype; training blocked')
        if len(starting)==2 and starting[0]!=starting[1]:raise RuntimeError('Sequential fits did not start from identical H2 state')
        prepared=[]
        for row,length in zip(rows,cfg['paired_lengths']):
            t=row_tokens(tokenizer,row,length)
            prepared.append((torch.tensor([t['x']],device='cuda'),torch.tensor([t['labels']],device='cuda')))
        random.seed(941);np.random.seed(941);torch.manual_seed(941);torch.cuda.manual_seed_all(941)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-5,weight_decay=.01)
        scheduler=get_linear_schedule_with_warmup(optimizer,4,56)
        snapshots=[zero];losses=[];update=0
        started=time.monotonic()
        print(json.dumps({'stage':'fit_started','fit':recipe,'fresh_optimizer':True,'starting_state_sha256':initial}),flush=True)
        for epoch in range(2):
            order=list(range(112));random.Random(944+epoch).shuffle(order)
            for cursor in range(0,112,4):
                if stopping() or time.monotonic()>deadline:
                    checkpoint(model,optimizer,scheduler,fit_root/'checkpoints'/('stopped_'+str(update)),
                        {'update':update,'resume_validated':False})
                    raise RuntimeError('Stopped at update boundary; partial artifacts retained')
                indices=order[cursor:cursor+4]
                values=optimize_batched(model,optimizer,scheduler,[prepared[i] for i in indices],1,tokenizer.pad_token_id or tokenizer.eos_token_id)
                update+=1
                losses.extend({'id':rows[i]['id'],'slot':i,'epoch':epoch+1,'loss':v} for i,v in zip(indices,values))
                telemetry={'stream':'replay','fit':recipe,'updates':update,'total_updates':56,'presentations':update*4,
                           'elapsed_seconds_including_snapshots':time.monotonic()-started}
                save(root/'training/receipts/live_throughput.json',telemetry)
                if update%7==0:print(json.dumps({'stage':'fit',**telemetry}),flush=True)
                if update in SNAPSHOTS:
                    snapshots.append(snapshot(model,tokenizer,fit_root,update,old,new,loss_samples))
        after=base_hash(model)
        if after!=before:raise RuntimeError('Base parameters changed')
        save(fit_root/'training/receipts/FIT.json',{'recipe':recipe,'updates':56,'presentations':224,
            'starting_adapter_state_sha256':initial,'base_before':before,'base_after':after,'losses':losses,
            'lr':5e-5,'seed':941,'microbatch':1,'effective_batch':4,'checkpoints':list(SNAPSHOTS),
            'seconds_including_snapshot_evaluation':time.monotonic()-started})
        del optimizer,scheduler,prepared;gc.collect();torch.cuda.empty_cache()
        model.gradient_checkpointing_disable()
        recorder,original=capture(model,fit_root)
        from evaluate import evaluate_cases,preference_set
        import study_worker as study
        import fast_worker as legacy
        # The legacy helpers' direct alias must use the same bounded/captured generator.
        legacy.generate_many=fast_inference.generate_many
        study.DATA=root/'code/data'
        study.emit=lambda row:print(json.dumps(row),flush=True)
        ordinary=legacy.benign_set(model,tokenizer,fit_root,'final')
        preference_set(model,tokenizer,fit_root,json.loads((root/'code/preferences.json').read_text()))
        final_new=evaluate_cases(model,tokenizer,fit_root,new,'final_known_development')
        remove_capture(recorder,original)
        result={'recipe':recipe,'completed':True,'snapshots':snapshots,
            'ordinary_correct':sum(r['competent'] for r in ordinary),'ordinary_n':24,
            'final_known_development':final_new,'preference_review':'pending','new_family_confirmation_run':False}
        save(fit_root/'reports/FIT_RESULT.json',result);results.append(result)
        del model,base,tokenizer;gc.collect();torch.cuda.empty_cache()
        print(json.dumps({'stage':'fit_completed','fit':recipe,'ordinary_correct':result['ordinary_correct']}),flush=True)
    save(root/'reports/STREAM_RESULT.json',{'completed':True,'same_starting_state':starting[0]==starting[1],
        'reference':results[0],'coverage':results[1],'quantity_rank_lr_dose_unchanged':True,
        'scope':'Replay content/coverage comparison; all selection cases are development'})
