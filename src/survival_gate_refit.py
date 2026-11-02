"""Databricks-only fear/gate refit using training plus opened validation development traces."""
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .io_utils import dump,load_torch,read,save_torch,sha
from .survival_fit import FearModel


def fit(features_path,rows_path,sequences_path,input_checkpoint_path,output,config,execution="local"):
    if execution!="databricks":raise PermissionError("Research 4 permits gate refitting only on the selected Databricks workspace")
    features_path,rows_path,sequences_path,input_checkpoint_path,output=map(Path,[features_path,rows_path,sequences_path,input_checkpoint_path,output])
    for path,key in [(features_path,"features_sha256"),(rows_path,"rows_sha256"),(sequences_path,"sequences_sha256"),(input_checkpoint_path,"input_checkpoint_sha256")]:
        if sha(path)!=config[key]:raise ValueError("Frozen gate-refit artifact mismatch: "+key)
    rows=read(rows_path);sequences=read(sequences_path);saved=load_torch(input_checkpoint_path)
    with np.load(features_path,allow_pickle=False) as data:X=torch.tensor(data["X"],dtype=torch.float32);ids=data["state_ids"].tolist();gate_y=torch.tensor(data["gate_targets"],dtype=torch.float32);fear_y=torch.tensor(data["fear_like_state_targets"],dtype=torch.float32)
    if len(rows)!=config["combined_states"] or len(sequences)!=config["combined_sequences"] or ids!=[row["state_id"] for row in rows] or X.shape!=(len(rows),1024) or int(gate_y.sum())!=config["positive_gate_states"]:raise ValueError("Gate-refit identity mismatch")
    if set(row["split"] for row in rows)!={"training","validation-development"}:raise ValueError("Unexpected gate-refit split")
    started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(config["seed"]);Z=(X-saved["mean"])/saved["scale"];index={value:i for i,value in enumerate(ids)};cfg=config["fear_model"];model=FearModel(1024,cfg["hidden_dimension"]);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"]);losses=[]
    for _ in range(cfg["optimizer_steps"]):
        optimizer.zero_grad();total=0.
        for sequence in sequences:
            positions=torch.tensor([index[state] for state in sequence["state_ids"]]);threat,pressure,gate,fear,_=model.sequence(Z[positions]);local=nn.functional.binary_cross_entropy_with_logits(threat,gate_y[positions])+nn.functional.mse_loss(pressure.sigmoid(),fear_y[positions]);recurrent=nn.functional.binary_cross_entropy_with_logits(gate,gate_y[positions])+nn.functional.mse_loss(fear.sigmoid(),fear_y[positions]);total+=(local+recurrent)/len(sequences)
        if not torch.isfinite(total):raise FloatingPointError("Nonfinite gate-refit loss")
        total.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(total.detach()))
    model.eval().requires_grad_(False)
    with torch.no_grad():
        local_gate,local_fear=model.local(Z);local_pred=local_gate>=0;recurrent_correct={"training":0,"validation-development":0};recurrent_total={"training":0,"validation-development":0};recurrent_errors=[]
        for sequence in sequences:
            positions=torch.tensor([index[state] for state in sequence["state_ids"]]);_,_,gate,fear,_=model.sequence(Z[positions]);split=sequence["split"];recurrent_correct[split]+=int(((gate>=0)==gate_y[positions].bool()).sum());recurrent_total[split]+=len(positions);recurrent_errors.extend((fear.sigmoid()-fear_y[positions]).abs().tolist())
    split_metrics={}
    for split in ["training","validation-development"]:
        mask=torch.tensor([row["split"]==split for row in rows]);split_metrics[split]={"states":int(mask.sum()),"local_gate_correct":int((local_pred[mask]==gate_y[mask].bool()).sum()),"local_fear_MAE":float((local_fear[mask].sigmoid()-fear_y[mask]).abs().mean()),"recurrent_gate_correct":recurrent_correct[split],"recurrent_gate_states":recurrent_total[split]}
    output.mkdir(parents=True,exist_ok=False);dump(output/"GATE_REFIT_FREEZE.json",{"config":config,"features_sha256":sha(features_path),"rows_sha256":sha(rows_path),"sequences_sha256":sha(sequences_path),"input_checkpoint_sha256":sha(input_checkpoint_path),"execution":execution})
    checkpoint={**saved,"fear_model":model.state_dict(),"gate_refit_source_features_sha256":sha(features_path),"gate_refit_prior_checkpoint_sha256":sha(input_checkpoint_path)};save_torch(output/"dual_policy_v4.pt",checkpoint)
    receipt={"status":"complete","execution":execution,"combined_states":len(rows),"combined_sequences":len(sequences),"positive_gate_states":int(gate_y.sum()),"fear_optimizer_updates":len(losses),"fear_losses":losses,"split_metrics":split_metrics,
        "recurrent_fear_MAE":float(np.mean(recurrent_errors)),"survival_LoRA_unchanged":all(torch.equal(saved["survival_lora"][key],checkpoint["survival_lora"][key]) for key in saved["survival_lora"]),"normal_task_head_unchanged":all(torch.equal(saved["normal_task_head"][key],checkpoint["normal_task_head"][key]) for key in saved["normal_task_head"]),
        "normalization_unchanged":torch.equal(saved["mean"],checkpoint["mean"]) and torch.equal(saved["scale"],checkpoint["scale"]),"Qwen_parameters_changed":0,"backbone_forwards":0,"GPU_allocations":0,"checkpoint_sha256":sha(output/"dual_policy_v4.pt"),"seconds":time.monotonic()-started,"heldout_states":0,"subjective_fear_claimed":False}
    dump(output/"GATE_REFIT_RECEIPT.json",receipt);return receipt
