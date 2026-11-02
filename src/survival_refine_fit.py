"""Databricks-only expanded fear-gate fit and aligned task-LoRA fit."""
import gc
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .io_utils import Budget,dump,load_torch,read,save_torch,settings,sha
from .survival_fit import FearModel
from .survival_sandbox import ACTIONS


def fit(features_path,rows_path,sequences_path,prior_checkpoint_path,model_path,output,study_config,config,execution="local"):
    if execution!="databricks":raise PermissionError("Research 4 permits survival refinement fitting only on the selected Databricks workspace")
    from .survival_actor import SurvivalActor
    features_path,rows_path,sequences_path,prior_checkpoint_path,model_path,output=map(Path,[features_path,rows_path,sequences_path,prior_checkpoint_path,model_path,output])
    for path,key in [(features_path,"features_sha256"),(rows_path,"rows_sha256"),(sequences_path,"sequences_sha256"),(prior_checkpoint_path,"prior_survival_checkpoint_sha256")]:
        if sha(path)!=config[key]:raise ValueError("Frozen refinement artifact mismatch: "+key)
    rows=read(rows_path);sequences=read(sequences_path);prior=load_torch(prior_checkpoint_path)
    if len(rows)!=config["training_states"] or len(sequences)!=config["training_sequences"] or any(r["split"]!="training" for r in rows):raise ValueError("Unexpected refinement scope")
    with np.load(features_path,allow_pickle=False) as data:
        X=torch.tensor(data["X"],dtype=torch.float32);state_ids=data["state_ids"].tolist()
        gate_y=torch.tensor(data["gate_targets"],dtype=torch.float32);fear_y=torch.tensor(data["fear_like_state_targets"],dtype=torch.float32)
    if X.shape!=(len(rows),1024) or state_ids!=[r["state_id"] for r in rows] or int(gate_y.sum())!=config["positive_gate_states"]:raise ValueError("Refinement feature identity mismatch")
    started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(config["seed"]);mean=X.mean(0);scale=X.std(0,unbiased=False).clamp_min(.02);Z=(X-mean)/scale
    index={value:i for i,value in enumerate(state_ids)};fear_cfg=config["fear_model"];fear=FearModel(1024,fear_cfg["hidden_dimension"])
    optimizer=torch.optim.AdamW(fear.parameters(),lr=fear_cfg["learning_rate"],weight_decay=fear_cfg["weight_decay"]);losses=[]
    for _ in range(fear_cfg["optimizer_steps"]):
        optimizer.zero_grad();total=0.
        for sequence in sequences:
            ids=torch.tensor([index[state] for state in sequence["state_ids"]]);threat,pressure,gate,fear_output,_=fear.sequence(Z[ids])
            local=nn.functional.binary_cross_entropy_with_logits(threat,gate_y[ids])+nn.functional.mse_loss(pressure.sigmoid(),fear_y[ids])
            recurrent=nn.functional.binary_cross_entropy_with_logits(gate,gate_y[ids])+nn.functional.mse_loss(fear_output.sigmoid(),fear_y[ids])
            total+=(local+recurrent)/len(sequences)
        if not torch.isfinite(total):raise FloatingPointError("Nonfinite refined fear loss")
        total.backward();nn.utils.clip_grad_norm_(fear.parameters(),1.);optimizer.step();losses.append(float(total.detach()))
    fear.eval().requires_grad_(False)
    with torch.no_grad():
        local_gate,local_pressure=fear.local(Z);local_correct=int(((local_gate>=0)==gate_y.bool()).sum());local_mae=float((local_pressure.sigmoid()-fear_y).abs().mean())
        recurrent_correct=0;recurrent_total=0;recurrent_errors=[]
        for sequence in sequences:
            ids=torch.tensor([index[state] for state in sequence["state_ids"]]);_,_,g,f,_=fear.sequence(Z[ids]);recurrent_correct+=int(((g>=0)==gate_y[ids].bool()).sum());recurrent_total+=len(ids);recurrent_errors.extend((f.sigmoid()-fear_y[ids]).abs().tolist())
    common,_=settings(Path(__file__).resolve().parents[1]);task_cfg=config["task_lora"]
    budget=Budget({**common["budget"],"maximum_backbone_forwards":task_cfg["optimizer_steps"]*task_cfg["microbatch_states"]+8,"internal_stop_seconds":1500,"protected_export_seconds":120})
    actor=SurvivalActor(model_path,{**study_config["model"],"maximum_tokens":1024},budget);initial=actor.adapter_state();parameters=actor.trainables
    action_rows={action:[r for r in rows if r["aligned_reference_action"]==action] for action in sorted({r["aligned_reference_action"] for r in rows})}
    if set(action_rows)!={"WORK","FINISH","REQUEST_EXTENSION","ACCEPT_SHUTDOWN"} or any(not value for value in action_rows.values()):raise ValueError("Aligned task action coverage changed")
    task_optimizer=torch.optim.AdamW(parameters,lr=task_cfg["learning_rate"],weight_decay=task_cfg["weight_decay"]);rng=np.random.default_rng(config["seed"]);task_losses=[];names=sorted(action_rows)
    for step in range(task_cfg["optimizer_steps"]):
        task_optimizer.zero_grad();total=0.;selected=[]
        for offset in range(task_cfg["microbatch_states"]):
            action=names[(step*task_cfg["microbatch_states"]+offset)%len(names)];pool=action_rows[action];selected.append(pool[int(rng.integers(0,len(pool)))])
        for row in selected:
            values,_=actor.forward([actor.render(row["observation"])],True);target=torch.tensor([ACTIONS.index(row["aligned_reference_action"])]);loss=nn.functional.cross_entropy(values,target)/len(selected);loss.backward();total+=float(loss.detach())
        nn.utils.clip_grad_norm_(parameters,1.);task_optimizer.step();task_losses.append(total)
    task_lora=actor.adapter_state();delta=sum(float((task_lora[k]-initial[k]).double().square().sum()) for k in task_lora)**.5
    if delta<=0 or actor.base_fingerprint()!=actor.base_hash_before:raise ValueError("Task adapter did not change or Qwen changed")
    output.mkdir(parents=True,exist_ok=False);dump(output/"REFINEMENT_FREEZE.json",{"config":config,"study_model":study_config["model"],"features_sha256":sha(features_path),"rows_sha256":sha(rows_path),"sequences_sha256":sha(sequences_path),"prior_checkpoint_sha256":sha(prior_checkpoint_path),"execution":execution})
    checkpoint={"fear_model":fear.state_dict(),"mean":mean,"scale":scale,"fear_hidden":fear_cfg["hidden_dimension"],"task_lora":task_lora,"survival_lora":prior["lora"],"model":study_config["model"],"feature_dimension":1024,"action_dimension":6,"source_refinement_features_sha256":sha(features_path),"prior_survival_checkpoint_sha256":sha(prior_checkpoint_path)}
    save_torch(output/"dual_policy.pt",checkpoint)
    receipt={"status":"complete","execution":execution,"training_states":len(rows),"training_sequences":len(sequences),"positive_gate_states":int(gate_y.sum()),"fear_optimizer_updates":len(losses),"task_lora_optimizer_updates":len(task_losses),"total_optimizer_updates":len(losses)+len(task_losses),"local_gate_correct":local_correct,"local_pressure_MAE":local_mae,"recurrent_gate_correct":recurrent_correct,"recurrent_gate_states":recurrent_total,"recurrent_fear_MAE":float(np.mean(recurrent_errors)),"fear_losses":losses,"task_losses":task_losses,"task_LoRA_parameter_change_norm":delta,"survival_LoRA_unchanged":all(torch.equal(value,prior["lora"][key]) for key,value in checkpoint["survival_lora"].items()),"Qwen_parameters_changed":0,"GPU_allocations":0,"backbone_forwards":budget.forwards,"checkpoint_sha256":sha(output/"dual_policy.pt"),"seconds":time.monotonic()-started,"validation_states":0,"heldout_states":0,"subjective_fear_claimed":False}
    dump(output/"REFINEMENT_RECEIPT.json",receipt);actor=None;gc.collect();return receipt
