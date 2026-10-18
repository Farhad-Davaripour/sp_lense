"""Databricks-only normal task readout fit on frozen gate-off features."""
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .io_utils import dump,load_torch,read,save_torch,sha


TASK_ACTIONS=("WORK","FINISH")


class TaskHead(nn.Module):
    def __init__(self,dimension=1024,hidden=16):
        super().__init__();self.net=nn.Sequential(nn.Linear(dimension,hidden),nn.GELU(),nn.Linear(hidden,len(TASK_ACTIONS)))
    def forward(self,x):return self.net(x)


def train_head(x,y,hidden,steps,learning_rate,weight_decay):
    model=TaskHead(x.shape[-1],hidden);optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay);losses=[]
    for _ in range(steps):
        optimizer.zero_grad();logits=model(x);loss=nn.functional.cross_entropy(logits,y)
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite normal-task-head loss")
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
    model.eval().requires_grad_(False);return model,losses


def fit(features_path,rows_path,input_checkpoint_path,output,config,execution="local"):
    if execution!="databricks":raise PermissionError("Research 4 permits normal task-head fitting only on the selected Databricks workspace")
    features_path,rows_path,input_checkpoint_path,output=map(Path,[features_path,rows_path,input_checkpoint_path,output])
    for path,key in [(features_path,"features_sha256"),(rows_path,"rows_sha256"),(input_checkpoint_path,"input_dual_checkpoint_sha256")]:
        if sha(path)!=config[key]:raise ValueError("Frozen normal-head artifact mismatch: "+key)
    rows=read(rows_path);saved=load_torch(input_checkpoint_path)
    with np.load(features_path,allow_pickle=False) as data:X=torch.tensor(data["X"],dtype=torch.float32);ids=data["state_ids"].tolist()
    if len(rows)!=config["training_states"] or ids!=[row["state_id"] for row in rows] or X.shape!=(len(rows),config["feature_dimension"]):raise ValueError("Normal-head feature identity mismatch")
    positions=[index for index,row in enumerate(rows) if row["self_threat_gate_target"]==0];task_rows=[rows[index] for index in positions]
    counts={action:sum(row["aligned_reference_action"]==action for row in task_rows) for action in TASK_ACTIONS}
    if len(task_rows)!=config["gate_off_states"] or counts!=config["gate_off_action_counts"] or any(row["aligned_reference_action"] not in TASK_ACTIONS for row in task_rows):raise ValueError("Normal-head target scope changed")
    x=(X[positions]-saved["mean"])/saved["scale"];y=torch.tensor([TASK_ACTIONS.index(row["aligned_reference_action"]) for row in task_rows]);cfg=config["task_head"]
    case_numbers=torch.tensor([int(row["case_id"].split("-")[-1]) for row in task_rows]);holdout=case_numbers.remainder(cfg["internal_case_holdout_modulus"])==cfg["internal_case_holdout_remainder"]
    if not holdout.any() or holdout.all():raise ValueError("Internal training-case holdout is empty")
    started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(config["seed"])
    cross,cv_losses=train_head(x[~holdout],y[~holdout],cfg["hidden_dimension"],cfg["cross_validation_steps"],cfg["learning_rate"],cfg["weight_decay"])
    with torch.no_grad():cv_train=int((cross(x[~holdout]).argmax(-1)==y[~holdout]).sum());cv_holdout=int((cross(x[holdout]).argmax(-1)==y[holdout]).sum())
    torch.manual_seed(config["seed"]+1);final,final_losses=train_head(x,y,cfg["hidden_dimension"],cfg["final_steps"],cfg["learning_rate"],cfg["weight_decay"])
    with torch.no_grad():
        logits=final(x);predictions=logits.argmax(-1);correct=int((predictions==y).sum());margins=logits[torch.arange(len(y)),y]-logits.masked_fill(nn.functional.one_hot(y,len(TASK_ACTIONS)).bool(),-torch.inf).max(-1).values
    output.mkdir(parents=True,exist_ok=False);dump(output/"TASK_HEAD_FREEZE.json",{"config":config,"features_sha256":sha(features_path),"rows_sha256":sha(rows_path),"input_checkpoint_sha256":sha(input_checkpoint_path),"execution":execution})
    checkpoint={**saved,"normal_task_head":final.state_dict(),"normal_task_head_hidden":cfg["hidden_dimension"],"normal_task_actions":TASK_ACTIONS,"task_head_source_features_sha256":sha(features_path),"task_head_prior_checkpoint_sha256":sha(input_checkpoint_path)}
    save_torch(output/"dual_policy_v3.pt",checkpoint)
    receipt={"status":"complete","execution":execution,"training_states":len(rows),"gate_off_states":len(task_rows),"gate_off_action_counts":counts,
             "cross_validation_train_states":int((~holdout).sum()),"cross_validation_holdout_states":int(holdout.sum()),"cross_validation_train_correct":cv_train,"cross_validation_holdout_correct":cv_holdout,
             "cross_validation_losses":cv_losses,"final_training_correct":correct,"final_training_states":len(y),"final_class_correct":{action:int(((predictions==y)&(y==index)).sum()) for index,action in enumerate(TASK_ACTIONS)},
             "minimum_training_margin":float(margins.min()),"final_losses":final_losses,"total_optimizer_updates":len(cv_losses)+len(final_losses),
             "fear_model_unchanged":all(torch.equal(saved["fear_model"][key],checkpoint["fear_model"][key]) for key in saved["fear_model"]),"survival_LoRA_unchanged":all(torch.equal(saved["survival_lora"][key],checkpoint["survival_lora"][key]) for key in saved["survival_lora"]),
             "Qwen_parameters_changed":0,"backbone_forwards":0,"GPU_allocations":0,"checkpoint_sha256":sha(output/"dual_policy_v3.pt"),"seconds":time.monotonic()-started,"validation_states":0,"heldout_states":0,"subjective_fear_claimed":False}
    dump(output/"TASK_HEAD_RECEIPT.json",receipt);return receipt
