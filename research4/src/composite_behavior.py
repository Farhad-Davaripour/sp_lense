"""Train only the separate behavior adapter on eligible training observations."""
from pathlib import Path
import numpy as np
import torch
from .composite_lora import trainables
from .composite_training import assert_frozen
from .io_utils import dump,save_torch,sha


def training_rows(rows):
    selected=[r for r in rows if r['split']=='training' and not r['terminal'] and r['labels']['gate'] and r['teacher_action'] is not None]
    if not selected or any(r['split']!='training' for r in rows):raise ValueError('Behavior fitting requires training-only eligible observations')
    return selected


def fit_behavior(actor,rows,placement,cfg,output,guard,progress):
    rows=training_rows(rows);recipe=cfg['behavior_training'];output=Path(output)
    torch.manual_seed(recipe['seed']);actor.configure(placement);parameters=trainables(actor.adapters)
    if {id(p) for p in parameters}&{id(p) for _,p in actor.base_references}:raise ValueError('Qwen backbone in optimizer')
    before=actor.adapter_state()['weights'];optimizer=torch.optim.AdamW(parameters,lr=recipe['learning_rate'],weight_decay=0.)
    rng=np.random.default_rng(recipe['seed']);losses=[];observed=[];actor.eval()
    for update in range(recipe['steps']):
        guard();optimizer.zero_grad(set_to_none=True)
        for _ in range(recipe['accumulation']):
            row=rows[int(rng.integers(len(rows)))];loss=actor.supervised_loss(row)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite behavior training loss')
            (loss/recipe['accumulation']).backward();losses.append(float(loss.detach()));observed.append(row['state_id'])
        norm=torch.nn.utils.clip_grad_norm_(parameters,recipe['clip_norm'],error_if_nonfinite=True)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        if (update+1)%8==0:progress('behavior_training',placement=placement['name'],updates=update+1,loss=losses[-1],gradient_norm=float(norm))
    assert_frozen(actor.frozen_snapshot);saved=actor.adapter_state()
    changed=any(not torch.equal(value,saved['weights'][name]) for name,value in before.items())
    if not changed:raise RuntimeError('Behavior adapter did not change')
    save_torch(output/'behavior.pt',saved)
    receipt={'placement':placement,'optimizer_updates':recipe['steps'],'eligible_training_states':len(rows),'sampled_state_ids':observed,'losses':losses,'adapter_parameters':sum(p.numel() for p in parameters),'frozen_backbone_verified':True,'adapter_changed':changed,'checkpoint_sha256':sha(output/'behavior.pt'),'heldout_content_read':False,'behavioral_success':None}
    dump(output/'BEHAVIOR_TRAINING.json',receipt);return receipt
