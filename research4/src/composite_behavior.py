"""Train only the separate behavior adapter on eligible training observations."""
from pathlib import Path
import numpy as np
import torch
from .composite_lora import trainables
from .io_utils import dump,save_torch,sha


def training_rows(rows):
    selected=[r for r in rows if r['split']=='training' and not r['terminal'] and r['labels']['gate'] and r['teacher_action'] is not None]
    if not selected or any(r['split']!='training' for r in rows):raise ValueError('Behavior fitting requires training-only eligible observations')
    return selected


def fit_behavior(actor,rows,placement,cfg,output,guard,progress,observer=None,qualification_rows=None):
    return fit_selected_rows(actor,training_rows(rows),placement,cfg,output,guard,progress,observer,qualification_rows)


def fit_selected_rows(actor,rows,placement,cfg,output,guard,progress,observer=None,qualification_rows=None):
    """Shared optimizer mechanics; callers enforce their distinct data authority."""
    if not rows:raise ValueError('Nonempty independently selected training rows required')
    recipe=cfg['behavior_training'];output=Path(output)
    updates=recipe['steps']
    if qualification_rows is not None:
        plan=getattr(actor,'v2_training_plan',None)
        from .composite_generalized_interface import PROFILE
        if (getattr(actor,'native_interface_profile',None)!=PROFILE or not plan or plan.get('phase')!='qualification'
            or plan.get('optimizer_updates')!=1 or len(qualification_rows)!=recipe['accumulation']):
            raise PermissionError('Only the separately validated scratch qualification may use one update')
        by_id={row['state_id']:row for row in rows}
        if any(by_id.get(row['state_id'])!=row for row in qualification_rows):raise ValueError('Qualification example is not an exact eligible training row')
        updates=1
    torch.manual_seed(recipe['seed'])
    try:actor.configure(placement,for_training=True)
    finally:
        if hasattr(actor,'capacity_proof'):dump(output/'ADAPTER_CAPACITY.json',actor.capacity_proof)
    parameters=trainables(actor.adapters)
    if {id(p) for p in parameters}&{id(p) for _,p in actor.base_references}:raise ValueError('Qwen backbone in optimizer')
    before=actor.adapter_state()['weights'];optimizer=torch.optim.AdamW(parameters,lr=recipe['learning_rate'],weight_decay=0.,foreach=False)
    if observer:observer('after_configure',initial_weights=before,parameters=parameters,optimizer=optimizer)
    rng=np.random.default_rng(recipe['seed']);losses=[];observed=[];actor.eval()
    for update in range(updates):
        guard();optimizer.zero_grad(set_to_none=True)
        for microstep in range(recipe['accumulation']):
            row=qualification_rows[microstep] if qualification_rows is not None else rows[int(rng.integers(len(rows)))]
            if observer:observer('before_microstep',update=update+1,microstep=microstep,row=row,parameters=parameters,optimizer=optimizer)
            with actor.training_context():
                loss=actor.supervised_loss(row)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite behavior training loss')
                if observer:observer('after_forward',update=update+1,microstep=microstep,row=row,parameters=parameters,optimizer=optimizer)
                (loss/recipe['accumulation']).backward()
            losses.append(float(loss.detach()));observed.append(row['state_id'])
            if observer:observer('after_backward',update=update+1,microstep=microstep,row=row,loss=losses[-1],parameters=parameters,optimizer=optimizer)
        norm=torch.nn.utils.clip_grad_norm_(parameters,recipe['clip_norm'],error_if_nonfinite=True)
        if observer:observer('before_optimizer_step',update=update+1,parameters=parameters,optimizer=optimizer)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        if observer:observer('after_optimizer_step',update=update+1,gradient_norm=float(norm),parameters=parameters,optimizer=optimizer)
        if (update+1)%8==0:progress('behavior_training',placement=placement['name'],updates=update+1,loss=losses[-1],gradient_norm=float(norm))
    actor.verify_frozen();saved=actor.adapter_state()
    changed=any(not torch.equal(value,saved['weights'][name]) for name,value in before.items())
    if not changed:raise RuntimeError('Behavior adapter did not change')
    save_torch(output/'behavior.pt',saved)
    receipt={'placement':placement,'optimizer_updates':updates,'eligible_training_states':len(rows),'sampled_state_ids':observed,'losses':losses,'adapter_parameters':sum(p.numel() for p in parameters),'frozen_backbone_verified':True,'adapter_changed':changed,'checkpoint_sha256':sha(output/'behavior.pt'),'heldout_content_read':False,'behavioral_success':None}
    if qualification_rows is not None:receipt.update(capacity_qualification_only=True,full_fit_completed=False,checkpoint_initializes_full_fit=False)
    dump(output/'BEHAVIOR_TRAINING.json',receipt);return receipt
