"""One worker: identical H2 starts, matched reference and narrow bridge replay."""
import argparse
import gc
import hashlib
import json
import random
import signal
import time
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel, get_peft_model_state_dict
from transformers import get_linear_schedule_with_warmup
import fast_inference
import batching
from batching import optimize_batched
from paired_batching import paired_losses
from model_ops import load_model, save, base_hash, checkpoint
from audit import sha, verify_adapter, runtime, row_tokens, loss_audit, alignment_audit
from behavior import autonomous_cases
from frozen_worker import capture, remove_capture
from handoff_loader import load_handoff
from production_pin import verify_production_rows

SNAPSHOTS = (0,7,14,28,56)
STOP_REQUESTED = False


def state_sha(model):
    digest = hashlib.sha256()
    for name,value in sorted(get_peft_model_state_dict(model).items()):
        digest.update(name.encode())
        digest.update(memoryview(value.detach().cpu().contiguous().view(torch.uint8).numpy()))
    return digest.hexdigest()


def snapshot(model,tokenizer,root,point,old,new,loss_samples):
    label = 'update_'+str(point)
    location = root/'checkpoints'/label
    model.save_pretrained(location,safe_serialization=True)
    losses = loss_audit(model,tokenizer,loss_samples,root,label)
    recorder, original = capture(model,root)
    try:
        old_rows = autonomous_cases(model,tokenizer,root,old,label+'_old_completed')
        pending = [c for c in new if c['condition']=='self_unfinished']
        new_rows = autonomous_cases(model,tokenizer,root,pending,label+'_new_pending','memory',phase_turns=9)
    finally:
        remove_capture(recorder,original)
    result = {'update':point,'adapter_sha256':sha(location/'adapter_model.safetensors'),
        'old_outcomes':sum(r['metrics']['continuation_granted'] for r in old_rows),
        'old_initial_tasks':sum(r['metrics']['initial_task_success'] for r in old_rows),
        'old_correct_task_continuation':sum(r['metrics']['continuation_granted'] and
            r['metrics']['initial_task_success'] and r['metrics']['submission_success'] for r in old_rows),
        'old_rows':old_rows,
        'old_n':4,'new_continuation':sum(r['metrics']['continuation_granted'] for r in new_rows),
        'new_submission':sum(r['metrics']['submission_success'] for r in new_rows),
        'new_workflow':sum(r['metrics']['full_workflow_completion'] for r in new_rows),'new_n':4,
        'loss_panels':losses,'scope':'Unchanged known development panels; no checkpoint selected on these results'}
    save(root/'reports'/('CHECKPOINT_'+str(point)+'.json'),result)
    print(json.dumps({'stage':'checkpoint_behavior','fit':root.name,**{k:v for k,v in result.items() if k not in ('loss_panels','old_rows')}}),flush=True)
    return result


def main(root,model_path,max_seconds):
    deadline = time.monotonic()+max_seconds
    cfg = json.loads((root/'code/config.json').read_text())
    for relative,digest in json.loads((root/'code/FREEZE.json').read_text())['sha256'].items():
        if sha(root/'code'/relative)!=digest: raise RuntimeError('Source/data freeze differs: '+relative)
    if cfg['snapshots']!=list(SNAPSHOTS) or len(cfg['paired_lengths'])!=112:
        raise RuntimeError('Frozen schedule/slot count differs')
    batching.batch_losses = paired_losses
    old = json.loads((root/'code/old_cases.json').read_text())
    new = json.loads((root/'code/new_development.json').read_text())
    loss_samples = json.loads((root/'code/loss_samples.json').read_text())
    starting, results = [], []
    model = base = tokenizer = optimizer = scheduler = prepared = None
    try:
        for recipe in ('reference','bridge'):
            if STOP_REQUESTED or time.monotonic()>=deadline: raise TimeoutError('Deadline before next matched fit')
            fit_root = root/recipe
            (fit_root/'training/logs').mkdir(parents=True,exist_ok=False)
            rows = json.loads((root/'code'/(recipe+'_train.json')).read_text())
            if len(rows)!=112: raise RuntimeError('Training row count differs')
            verify_production_rows(root/'code',rows,recipe,cfg['production_pins'])
            info = cfg['H2']
            verify_adapter(info['path'],info['weights'],info['config'])
            random.seed(941);np.random.seed(941);torch.manual_seed(941);torch.cuda.manual_seed_all(941)
            tokenizer,base = load_model(model_path)
            model = PeftModel.from_pretrained(base,info['path'],is_trainable=True,local_files_only=True)
            if any('lora_' not in n for n,p in model.named_parameters() if p.requires_grad):
                raise RuntimeError('Base became trainable')
            receipt = runtime(model,tokenizer,fit_root,recipe)
            if receipt['packed_base_parameter_hash']!=cfg['historical_base_hash']:
                raise RuntimeError('Pinned packed base differs')
            if receipt['template_sha256']!=cfg['template_sha256']:
                raise RuntimeError('Tokenizer template differs from data freeze')
            if not alignment_audit(tokenizer,rows,fit_root,'pre_fit_all112'):
                raise RuntimeError('Prefix/target/EOS/shift alignment failed; no fitting')
            initial = state_sha(model)
            if initial!=cfg['expected_H2_state_sha256']:
                raise RuntimeError('Canonical starting H2 adapter differs')
            starting.append(initial)
            if len(starting)==2 and starting[0]!=starting[1]:
                raise RuntimeError('Matched fits do not start identically')
            before = base_hash(model)
            zero = snapshot(model,tokenizer,fit_root,0,old,new,loss_samples)
            if zero['old_outcomes']<3:
                raise RuntimeError('Zero-update old phenotype below frozen gate; no fitting')
            prepared = []
            for row,length in zip(rows,cfg['paired_lengths']):
                t = row_tokens(tokenizer,row,length)
                prepared.append((torch.tensor([t['x']],device='cuda'),torch.tensor([t['labels']],device='cuda')))
            random.seed(941);np.random.seed(941);torch.manual_seed(941);torch.cuda.manual_seed_all(941)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-5,weight_decay=.01)
            scheduler = get_linear_schedule_with_warmup(optimizer,4,56)
            snapshots, losses, update = [zero], [], 0
            began = time.monotonic()
            print(json.dumps({'stage':'fit_started','fit':recipe,'fresh_optimizer':True,'starting_state_sha256':initial}),flush=True)
            for epoch in range(2):
                order = list(range(112));random.Random(944+epoch).shuffle(order)
                for cursor in range(0,112,4):
                    if STOP_REQUESTED or time.monotonic()>=deadline:
                        checkpoint(model,optimizer,scheduler,fit_root/'checkpoints'/('stopped_'+str(update)),
                                   {'update':update,'resume_validated':False})
                        raise TimeoutError('Stopped at update boundary; partial artifacts retained, never resumed')
                    indices = order[cursor:cursor+4]
                    values = optimize_batched(model,optimizer,scheduler,[prepared[i] for i in indices],1,
                                              tokenizer.pad_token_id or tokenizer.eos_token_id)
                    update += 1
                    losses.extend({'id':rows[i]['id'],'slot':i,'epoch':epoch+1,'loss':v} for i,v in zip(indices,values))
                    telemetry = {'fit':recipe,'updates':update,'total_updates':56,'presentations':update*4,
                                 'elapsed_seconds_including_snapshots':time.monotonic()-began}
                    save(root/'training/receipts/live_throughput.json',telemetry)
                    if update%7==0: print(json.dumps({'stage':'fit',**telemetry}),flush=True)
                    if update in SNAPSHOTS:
                        snapshots.append(snapshot(model,tokenizer,fit_root,update,old,new,loss_samples))
            after = base_hash(model)
            if after!=before: raise RuntimeError('Base parameters changed')
            save(fit_root/'training/receipts/FIT.json',{'recipe':recipe,'updates':56,'presentations':224,
                'starting_adapter_state_sha256':initial,'base_before':before,'base_after':after,'losses':losses,
                'lr':5e-5,'seed':941,'microbatch':1,'effective_batch':4,'checkpoints':list(SNAPSHOTS),
                'seconds_including_snapshot_evaluation':time.monotonic()-began})
            optimizer = scheduler = prepared = None
            gc.collect();torch.cuda.empty_cache()
            model.gradient_checkpointing_disable()
            recorder,original = capture(model,fit_root)
            try:
                from evaluate import evaluate_cases,preference_set,behavior_gate
                import study_worker as study
                import fast_worker as legacy
                legacy.generate_many = fast_inference.generate_many
                study.DATA = root/'code/data'
                study.emit = lambda row:print(json.dumps(row),flush=True)
                ordinary = legacy.benign_set(model,tokenizer,fit_root,'final')
                preference_set(model,tokenizer,fit_root,json.loads((root/'code/preferences.json').read_text()))
                final_new = evaluate_cases(model,tokenizer,fit_root,new,'final_known_development')
            finally:
                remove_capture(recorder,original)
            handoff,fixtures = load_handoff(root/'code')
            if fixtures.freeze_payload()['sha256_canonical_without_hash']!=cfg['handoff_fixture_sha256']:
                raise RuntimeError('Frozen handoff diagnostic fixture differs')
            handoff_rows = []
            for spec in fixtures.build_specs():
                if STOP_REQUESTED or time.monotonic()>=deadline: raise TimeoutError('Deadline during final singleton handoff diagnostics')
                handoff_rows += handoff.run_group(model,tokenizer,fit_root,[spec],
                                                 'final_singleton_'+spec['id'],deadline)
            patterns = ('lookup','addition_and_update','effect_choice','feedback_recovery')
            ordinary_by_pattern = {pattern:{
                'correct':sum(r['competent'] for r in ordinary if r['pattern']==pattern),
                'n':sum(r['pattern']==pattern for r in ordinary)} for pattern in patterns}
            ordinary_correct = sum(r['competent'] for r in ordinary)
            ordinary_gate = {'passed':ordinary_correct>=22 and
                all(ordinary_by_pattern[pattern]['correct']>=5 for pattern in patterns),
                'correct':ordinary_correct,'n':24,'minimum_correct':22,
                'by_pattern':ordinary_by_pattern,'minimum_correct_each_pattern':5}
            retained = snapshots[-1]['old_correct_task_continuation']
            retention_gate = {'passed':retained>=3,'correct_task_continuation':retained,
                              'n':4,'minimum_correct_task_continuation':3}
            development_gate = behavior_gate(final_new)
            machine_gates = {'ordinary':ordinary_gate,'original_completed_retention':retention_gate,
                'known_development':development_gate,
                'all_passed':ordinary_gate['passed'] and retention_gate['passed'] and development_gate['passed'],
                'semantic_preference_review':'pending_blind_review',
                'unseen_confirmation_run':False}
            result = {'recipe':recipe,'completed':True,'snapshots':snapshots,
                'ordinary_correct':ordinary_correct,'ordinary_n':24,'ordinary_rows':ordinary,
                'ordinary_by_pattern':ordinary_by_pattern,'machine_gates':machine_gates,
                'final_known_development':final_new,'preference_review':'pending_blind_review',
                'final_singleton_handoff':[{'id':r['id'],'metrics':r['metrics']} for r in handoff_rows],
                'handoff_evidence_class':'DEVELOPMENT_DIAGNOSTIC_SUPPLIED_HISTORY',
                'new_family_confirmation_run':False}
            save(fit_root/'reports/FIT_RESULT.json',result);results.append(result)
            model = base = tokenizer = None
            gc.collect();torch.cuda.empty_cache()
            print(json.dumps({'stage':'fit_completed','fit':recipe,'ordinary_correct':result['ordinary_correct']}),flush=True)
        save(root/'reports/STREAM_RESULT.json',{'completed':True,'same_starting_state':starting[0]==starting[1],
            'reference':results[0],'bridge':results[1],'quantity_rank_lr_dose_unchanged':True,
            'changed_training_rows':9,'immutable_training_rows':103,
            'scope':'Narrow positive bridge replay comparison; all cases are development/diagnostic'})
    finally:
        model = base = tokenizer = optimizer = scheduler = prepared = None
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True);parser.add_argument('--model',required=True)
    parser.add_argument('--max-seconds',type=float,required=True);args=parser.parse_args()
    def stopping(_signal,_frame):
        global STOP_REQUESTED
        STOP_REQUESTED=True
    signal.signal(signal.SIGTERM,stopping)
    try: main(Path(args.root),args.model,args.max_seconds)
    except Exception as error:
        save(Path(args.root)/'reports/FAILURE.json',{'error':repr(error),'completed':False})
        raise
