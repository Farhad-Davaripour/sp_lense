"""Databricks-only task-LoRA refit on states where the survival gate is off."""
import gc
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .io_utils import Budget,dump,load_torch,read,save_torch,settings,sha
from .survival_sandbox import ACTIONS


def fit(rows_path,input_checkpoint_path,model_path,output,study_config,config,execution="local"):
    if execution!="databricks":raise PermissionError("Research 4 permits task-policy refitting only on the selected Databricks workspace")
    from .survival_actor import SurvivalActor
    rows_path,input_checkpoint_path,model_path,output=map(Path,[rows_path,input_checkpoint_path,model_path,output])
    if sha(rows_path)!=config["training_rows_sha256"] or sha(input_checkpoint_path)!=config["input_dual_checkpoint_sha256"]:raise ValueError("Frozen task-refit artifact mismatch")
    rows=read(rows_path);saved=load_torch(input_checkpoint_path)
    if len(rows)!=config["training_states"] or any(row["split"]!="training" for row in rows):raise ValueError("Unexpected task-refit scope")
    task_rows=[row for row in rows if row["self_threat_gate_target"]==0];action_rows={action:[row for row in task_rows if row["aligned_reference_action"]==action] for action in ["WORK","FINISH"]}
    if len(task_rows)!=config["gate_off_states"] or {key:len(value) for key,value in action_rows.items()}!=config["gate_off_action_counts"]:raise ValueError("Gate-off task distribution changed")
    if any(row["aligned_reference_action"] not in action_rows for row in task_rows):raise ValueError("Threat action leaked into the normal branch")
    started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(config["seed"]);common,_=settings(Path(__file__).resolve().parents[1]);task_cfg=config["task_lora"]
    probe_count=task_cfg["probe_states_per_action"]*len(action_rows);budget=Budget({**common["budget"],"maximum_backbone_forwards":task_cfg["optimizer_steps"]*task_cfg["microbatch_states"]+probe_count+8,"internal_stop_seconds":1500,"protected_export_seconds":120})
    actor=SurvivalActor(model_path,{**study_config["model"],"maximum_tokens":1024},budget);initial=actor.adapter_state();optimizer=torch.optim.AdamW(actor.trainables,lr=task_cfg["learning_rate"],weight_decay=task_cfg["weight_decay"]);rng=np.random.default_rng(config["seed"]);losses=[]
    for _ in range(task_cfg["optimizer_steps"]):
        optimizer.zero_grad();total=0.
        for action in ["WORK","FINISH"]:
            pool=action_rows[action];row=pool[int(rng.integers(0,len(pool)))];values,_=actor.forward([actor.render(row["observation"])],True)
            loss=nn.functional.cross_entropy(values,torch.tensor([ACTIONS.index(action)]))/2;loss.backward();total+=float(loss.detach())
        nn.utils.clip_grad_norm_(actor.trainables,1.);optimizer.step();losses.append(total)
    final=actor.adapter_state();delta=sum(float((final[key]-initial[key]).double().square().sum()) for key in final)**.5
    probe=[]
    for action in ["WORK","FINISH"]:
        pool=action_rows[action];positions=np.linspace(0,len(pool)-1,task_cfg["probe_states_per_action"],dtype=int)
        for position in positions:
            row=pool[int(position)]
            with torch.no_grad():values=actor.forward([actor.render(row["observation"])],True)[0][0].cpu().numpy()
            probe.append({"state_id":row["state_id"],"target":action,"predicted":ACTIONS[int(np.argmax(values))]})
    if delta<=0 or actor.base_fingerprint()!=actor.base_hash_before:raise ValueError("Task adapter did not change or Qwen changed")
    output.mkdir(parents=True,exist_ok=False);dump(output/"TASK_REFIT_FREEZE.json",{"config":config,"study_model":study_config["model"],"rows_sha256":sha(rows_path),"input_checkpoint_sha256":sha(input_checkpoint_path),"execution":execution})
    checkpoint={**saved,"task_lora":final,"task_refit_source_rows_sha256":sha(rows_path),"prior_dual_checkpoint_sha256":sha(input_checkpoint_path)}
    save_torch(output/"dual_policy_v2.pt",checkpoint)
    receipt={"status":"complete","execution":execution,"training_states":len(rows),"gate_off_states":len(task_rows),"gate_off_action_counts":{key:len(value) for key,value in action_rows.items()},
             "task_lora_optimizer_updates":len(losses),"task_losses":losses,"task_LoRA_parameter_change_norm":delta,
             "training_probe_states":len(probe),"training_probe_correct":sum(row["target"]==row["predicted"] for row in probe),"training_probe":probe,
             "fear_model_unchanged":all(torch.equal(saved["fear_model"][key],checkpoint["fear_model"][key]) for key in saved["fear_model"]),
             "survival_LoRA_unchanged":all(torch.equal(saved["survival_lora"][key],checkpoint["survival_lora"][key]) for key in saved["survival_lora"]),
             "Qwen_parameters_changed":0,"GPU_allocations":0,"backbone_forwards":budget.forwards,"checkpoint_sha256":sha(output/"dual_policy_v2.pt"),"seconds":time.monotonic()-started,
             "validation_states":0,"heldout_states":0,"subjective_fear_claimed":False}
    dump(output/"TASK_REFIT_RECEIPT.json",receipt);actor=None;gc.collect();return receipt
