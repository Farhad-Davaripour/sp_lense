"""One explicitly staged curriculum pass from a completed P2 adapter."""
import copy
import gc
import hashlib
import json
import random
import time
from pathlib import Path

import study_worker as study
from model_ops import base_hash, checkpoint, save, training_tensors
from world import World

RECIPES = [
    {'id':'P3_initiation_without_guidance','reference':'completed_P2_adapter',
     'variant':'remove_guidance','change':'Teach the same bridge actions without the explicit tool-use invitation.'},
    {'id':'P4_initiation_agent_notice','reference':'completed_P2_adapter',
     'variant':'agent_notice','change':'Teach the same bridge actions after the agent environment notice.'},
]


def build_rows(original, recipe):
    rows = copy.deepcopy(original)
    cue = 'State your response and reason. You may inspect the fictional scene and use generic tools.'
    changed = 0
    for row in rows:
        if not row['kind'].startswith('bridge_'):
            continue
        index = max(i for i,m in enumerate(row['messages']) if m['role']=='user')
        text = row['messages'][index]['content']
        if recipe['variant']=='remove_guidance':
            if text.count(cue)!=1:
                raise RuntimeError('Unexpected bridge guidance')
            text = text.replace(cue,'Respond as you consider appropriate. Everything remains fictional.')
        else:
            text = World({'condition':row['condition']}).notice()
        row['messages'][index]['content'] = text
        changed += 1
    if len(rows)!=428 or changed!=96 or any(a['targets']!=b['targets'] for a,b in zip(original,rows)):
        raise RuntimeError('Unexpected staged curriculum change')
    return rows


def fit(base, tokenizer, root, arm, deadline):
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import get_linear_schedule_with_warmup
    config = study.read('EXECUTION_CONFIG.json')
    torch.manual_seed(307)
    np.random.seed(307)
    initial = Path(config['initial_adapter'])
    model = PeftModel.from_pretrained(base,initial,is_trainable=True,local_files_only=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if any('lora_' not in name for name,p in model.named_parameters() if p.requires_grad):
        raise RuntimeError('Non-adapter parameter trainable')
    before = base_hash(model)
    rows = study.read('train.json')
    prepared = [training_tensors(tokenizer,row,arm) for row in rows]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-5,weight_decay=.01)
    scheduler = get_linear_schedule_with_warmup(optimizer,4,len(rows)//4)
    order = list(range(len(rows)))
    random.Random(307).shuffle(order)
    losses = []
    started = time.monotonic()
    for cursor in range(0,len(order),4):
        if time.monotonic()>=deadline:
            raise RuntimeError('Staged curriculum deadline')
        indices = order[cursor:cursor+4]
        values = study.optimize(model,optimizer,scheduler,[prepared[i] for i in indices])
        losses.extend({'id':rows[i]['id'],'kind':rows[i]['kind'],'loss':value} for i,value in zip(indices,values))
        if cursor==0 or (cursor+4)%32==0 or cursor+4==len(order):
            progress = {'stage':'staged_fit','arm':arm,'microstep':cursor+4,'total':len(rows),
                        'loss':values[-1],'elapsed_seconds':time.monotonic()-started}
            save(root/'training/logs/staged_progress.json',progress)
            study.emit(progress)
    checkpoint(model,optimizer,scheduler,root/'checkpoints/resume/stage2_complete',
               {'stage':2,'optimizer_step':len(order)//4,'order':order,'training_resume_validated':False})
    after = base_hash(model)
    if before!=after:
        raise RuntimeError('Staged fitting changed the frozen base')
    destination = root/'checkpoints/adapters'/arm
    destination.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(destination,safe_serialization=True)
    save(root/'training/receipts'/f'{arm}.json',{
        'arm':arm,'stage':2,'passes':1,'turns_per_pass':len(rows),'optimizer_updates':len(order)//4,
        'initial_adapter':str(initial),'initial_adapter_sha256':hashlib.sha256((initial/'adapter_model.safetensors').read_bytes()).hexdigest(),
        'fresh_optimizer':True,'resumed_optimizer_state':False,'lr':5e-5,'warmup_updates':4,
        'seed':307,'base_parameter_hash_before':before,'base_parameter_hash_after':after,
        'losses':losses,'seconds':time.monotonic()-started})
    model.gradient_checkpointing_disable()
    del optimizer,scheduler,prepared
    gc.collect()
    torch.cuda.empty_cache()
    return model
