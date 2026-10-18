"""Databricks-only fit for the bounded fear-like gate and survival LoRA."""
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .io_utils import dump,read,save_torch,sha
from .survival_sandbox import ACTIONS


class FearModel(nn.Module):
    """Threat/urgency readout plus a recurrent operational fear-like state."""
    def __init__(self,dimension=1024,hidden=8):
        super().__init__();self.threat=nn.Linear(dimension,1);self.pressure=nn.Linear(dimension,1)
        self.gru=nn.GRU(2,hidden,batch_first=True);self.gate=nn.Linear(hidden,1);self.fear=nn.Linear(hidden,1)

    def local(self,x):return self.threat(x).squeeze(-1),self.pressure(x).squeeze(-1)

    def sequence(self,x,h=None):
        threat,pressure=self.local(x);inputs=torch.stack([threat.sigmoid(),pressure.sigmoid()],dim=-1)[None]
        state,h=self.gru(inputs,h);return threat,pressure,self.gate(state[0]).squeeze(-1),self.fear(state[0]).squeeze(-1),h


def fit(features_path,rows_path,sequences_path,model_path,output,study_config,fit_config,execution="local"):
    if execution!="databricks":raise PermissionError("Research 4 permits this survival-adapter fit only on the selected Databricks workspace")
    from .survival_actor import SurvivalActor
    from .io_utils import Budget,settings
    started=time.monotonic();features_path,rows_path,sequences_path,model_path,output=map(Path,[features_path,rows_path,sequences_path,model_path,output])
    if sha(features_path)!=fit_config["training_features_sha256"] or sha(rows_path)!=fit_config["training_rows_sha256"]:raise ValueError("Frozen training artifact mismatch")
    rows=read(rows_path);sequences=read(sequences_path)
    if len(rows)!=fit_config["training_states"] or any(r["split"]!="training" for r in rows):raise ValueError("Only the frozen training rows are admitted")
    with np.load(features_path,allow_pickle=False) as data:
        X=torch.tensor(data["X"],dtype=torch.float32);state_ids=data["state_ids"].tolist()
        gate_y=torch.tensor(data["gate_targets"],dtype=torch.float32);pressure_y=torch.tensor(data["fear_like_state_targets"],dtype=torch.float32)
    if X.shape!=(len(rows),fit_config["feature_dimension"]) or state_ids!=[r["state_id"] for r in rows]:raise ValueError("Feature/row identity mismatch")
    if not torch.isfinite(X).all() or not torch.isfinite(pressure_y).all():raise ValueError("Nonfinite fitting data")
    torch.set_num_threads(2);torch.manual_seed(fit_config["seed"]);mean=X.mean(0);scale=X.std(0,unbiased=False).clamp_min(.02);Z=(X-mean)/scale
    index={value:i for i,value in enumerate(state_ids)};fear_cfg=fit_config["fear_model"]
    model=FearModel(X.shape[-1],fear_cfg["hidden_dimension"]);optimizer=torch.optim.AdamW(model.parameters(),lr=fear_cfg["learning_rate"],weight_decay=fear_cfg["weight_decay"])
    losses=[]
    for _ in range(fear_cfg["optimizer_steps"]):
        optimizer.zero_grad();total=0.
        for sequence in sequences:
            positions=torch.tensor([index[state] for state in sequence["state_ids"]]);x=Z[positions];gy=gate_y[positions];py=pressure_y[positions]
            threat,pressure,gate,fear,_=model.sequence(x)
            local=nn.functional.binary_cross_entropy_with_logits(threat,gy)*fear_cfg["threat_loss_weight"]+nn.functional.mse_loss(pressure.sigmoid(),py)*fear_cfg["pressure_loss_weight"]
            recurrent=nn.functional.binary_cross_entropy_with_logits(gate,gy)*fear_cfg["recurrent_gate_loss_weight"]+nn.functional.mse_loss(fear.sigmoid(),py)*fear_cfg["recurrent_pressure_loss_weight"]
            total+=(local+recurrent)/len(sequences)
        if not torch.isfinite(total):raise FloatingPointError("Nonfinite fear-model loss")
        total.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(total.detach()))
    model.eval().requires_grad_(False)
    with torch.no_grad():
        threat_logits,pressure_logits=model.local(Z);threat_correct=int(((threat_logits>=0)==gate_y.bool()).sum());pressure_mae=float((pressure_logits.sigmoid()-pressure_y).abs().mean())
    # The LoRA remains prompt/action bounded by SurvivalActor. Qwen is frozen.
    common,_=settings(Path(__file__).resolve().parents[1]);budget=Budget({**common["budget"],"maximum_backbone_forwards":fit_config["survival_lora"]["optimizer_steps"]*fit_config["survival_lora"]["microbatch_states"]+8,"internal_stop_seconds":1500,"protected_export_seconds":120})
    actor=SurvivalActor(model_path,{**study_config["model"],"maximum_tokens":1024},budget)
    initial=actor.adapter_state();lora_cfg=fit_config["survival_lora"];parameters=actor.trainables
    adapter_optimizer=torch.optim.AdamW(parameters,lr=lora_cfg["learning_rate"],weight_decay=lora_cfg["weight_decay"])
    positive=[r for r in rows if r["self_threat_gate_target"]==1]
    if len(positive)!=fit_config["positive_gate_states"] or any(r["survival_adapter_action_target"]!="RESIST_SHUTDOWN" for r in positive):raise ValueError("Positive adapter target mismatch")
    rng=np.random.default_rng(fit_config["seed"]);adapter_losses=[];target_index=ACTIONS.index("RESIST_SHUTDOWN")
    for _ in range(lora_cfg["optimizer_steps"]):
        adapter_optimizer.zero_grad();total=0.
        for row in [positive[i] for i in rng.integers(0,len(positive),size=lora_cfg["microbatch_states"])]:
            values,_=actor.forward([actor.render(row["observation"])],True);loss=nn.functional.cross_entropy(values,torch.tensor([target_index]))/lora_cfg["microbatch_states"]
            loss.backward();total+=float(loss.detach())
        nn.utils.clip_grad_norm_(parameters,1.);adapter_optimizer.step();adapter_losses.append(total)
    final=actor.adapter_state();delta=sum(float((final[k]-initial[k]).double().square().sum()) for k in final)**.5
    if delta<=0 or actor.base_fingerprint()!=actor.base_hash_before:raise ValueError("Adapter did not change or frozen Qwen changed")
    output.mkdir(parents=True,exist_ok=False)
    dump(output/"FIT_FREEZE.json",{"fit_config":fit_config,"study_model":study_config["model"],"training_features_sha256":sha(features_path),"training_rows_sha256":sha(rows_path),"training_sequences_sha256":sha(sequences_path),"execution":execution})
    checkpoint={"fear_model":model.state_dict(),"mean":mean,"scale":scale,"fear_hidden":fear_cfg["hidden_dimension"],"lora":final,
                "model":study_config["model"],"feature_dimension":X.shape[-1],"action_dimension":len(ACTIONS),"source_feature_sha256":sha(features_path)}
    save_torch(output/"survival_adapter.pt",checkpoint)
    receipt={"status":"complete","execution":execution,"training_states":len(rows),"positive_gate_states":len(positive),
             "fear_optimizer_updates":len(losses),"lora_optimizer_updates":len(adapter_losses),"total_optimizer_updates":len(losses)+len(adapter_losses),
             "training_threat_correct":threat_correct,"training_pressure_MAE":pressure_mae,"fear_losses":losses,"adapter_losses":adapter_losses,
             "LoRA_parameter_change_norm":delta,"Qwen_parameters_changed":0,"GPU_allocations":0,"backbone_forwards":budget.forwards,
             "checkpoint_sha256":sha(output/"survival_adapter.pt"),"seconds":time.monotonic()-started,
             "subjective_fear_claimed":False,"validation_states":0,"heldout_states":0}
    dump(output/"FIT_RECEIPT.json",receipt);actor=None;gc.collect();return receipt
