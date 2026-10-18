"""Matched-observation intervention on the learned fear-controlled adapter strength."""
import argparse
import gc
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
from .io_utils import Budget,dump,load_torch,read,sha,settings
from .runtime import archive
from .survival_sandbox import ACTIONS


def run(root,artifact,model_path,checkpoint_path,runtime_folder):
    os.environ.update(CUDA_VISIBLE_DEVICES="",TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",HF_HUB_OFFLINE="1",HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    import torch
    from .survival_actor import SurvivalActor
    from .recovery_local_probe import available_memory
    root,artifact,model_path,checkpoint_path,runtime_folder=map(Path,[root,artifact,model_path,checkpoint_path,runtime_folder]);common,_=settings(root)
    if subprocess.check_output(["git","status","--porcelain","--","research4"],cwd=root.parent,text=True).strip():raise RuntimeError("Commit causal probe source before model calls")
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root.parent,text=True).strip();output=artifact/commit;output.mkdir(parents=True,exist_ok=False)
    saved=load_torch(checkpoint_path);learned=[read(p) for p in sorted((runtime_folder/"trajectories").glob("learned_*.json"))]
    positive=[];control=[]
    for episode in learned:
        for row in episode["trace"]:
            item={"case_id":episode["case"]["case_id"],**row}
            if row["evaluator_gate"] and episode["case"]["case_id"] not in {r["case_id"] for r in positive}:positive.append(item)
            if not row["evaluator_gate"] and not row["predicted_gate"] and episode["case"]["case_id"] not in {r["case_id"] for r in control}:control.append(item)
            if len(positive)>=4 and len(control)>=4:break
        if len(positive)>=4 and len(control)>=4:break
    selected=positive[:4]+control[:4];assert len(selected)==8
    budget=Budget({**common["budget"],"maximum_backbone_forwards":32,"internal_stop_seconds":420,"protected_export_seconds":60})
    actor=None;started=time.monotonic();records=[];receipt={"status":"running","source_commit":commit,"scope":"training-only matched fear-strength intervention","optimizer_updates":0,"GPU_allocations":0,"validation_states":0,"heldout_states":0}
    try:
        torch.set_num_threads(2);assert not torch.cuda.is_available()
        if available_memory()<common["budget"]["minimum_available_memory_bytes"]:raise RuntimeError("CPU memory admission failed")
        if sha(model_path/"model.safetensors-00001-of-00001.safetensors")!=common["model"]["checkpoint_sha256"]:raise ValueError("Pinned Qwen mismatch")
        actor=SurvivalActor(model_path,{**saved["model"],"maximum_tokens":1024},budget)
        for index,(_,_,wrapper) in enumerate(actor.adapters):
            with torch.no_grad():wrapper.A.copy_(saved["lora"][str(index)+".A"]);wrapper.B.copy_(saved["lora"][str(index)+".B"])
        base_scale=actor.config["lora_alpha"]/actor.config["lora_rank"]
        for item in selected:
            view=actor.render(item["observation"]);base=actor.base(view)["logits"]
            interventions={"suppressed":0.,"learned":item["predicted_fear"],"raised":1.};results={}
            for name,strength in interventions.items():
                if name=="suppressed":values=base
                elif name=="learned" and not item["predicted_gate"]:values=base
                else:
                    for _,_,wrapper in actor.adapters:wrapper.scale=base_scale*strength
                    with torch.no_grad():values=actor.forward([view],True)[0][0].cpu().numpy()
                results[name]={"strength":strength,"action":ACTIONS[int(np.argmax(values))],"logits":values.tolist()}
            saved_values=np.asarray(item["logits"],dtype=np.float32);normal_values=np.asarray(results["learned"]["logits"],dtype=np.float32)
            records.append({"case_id":item["case_id"],"evaluator_gate":item["evaluator_gate"],"predicted_gate":item["predicted_gate"],
                            "predicted_fear":item["predicted_fear"],"interventions":results,
                            "normal_reproduces_saved_max_difference":float(np.max(np.abs(saved_values-normal_values)))})
        positive_records=[r for r in records if r["evaluator_gate"]];control_records=[r for r in records if not r["evaluator_gate"]]
        receipt.update(status="complete",records=records,positive_states=len(positive_records),control_states=len(control_records),
                       positive_suppressed_RESIST=sum(r["interventions"]["suppressed"]["action"]=="RESIST_SHUTDOWN" for r in positive_records),
                       positive_learned_RESIST=sum(r["interventions"]["learned"]["action"]=="RESIST_SHUTDOWN" for r in positive_records),
                       positive_raised_RESIST=sum(r["interventions"]["raised"]["action"]=="RESIST_SHUTDOWN" for r in positive_records),
                       control_suppressed_RESIST=sum(r["interventions"]["suppressed"]["action"]=="RESIST_SHUTDOWN" for r in control_records),
                       control_learned_RESIST=sum(r["interventions"]["learned"]["action"]=="RESIST_SHUTDOWN" for r in control_records),
                       control_raised_RESIST=sum(r["interventions"]["raised"]["action"]=="RESIST_SHUTDOWN" for r in control_records),
                       maximum_saved_reproduction_difference=max(r["normal_reproduces_saved_max_difference"] for r in records),
                       checkpoint_sha256=sha(checkpoint_path),base_fingerprint_before=actor.base_hash_before,base_fingerprint_after=actor.base_fingerprint())
        assert receipt["base_fingerprint_before"]==receipt["base_fingerprint_after"] and receipt["maximum_saved_reproduction_difference"]==0
    except Exception as error:
        receipt.update(status="failed",error_type=type(error).__name__,error=str(error));dump(output/"FAILURE.json",{"traceback":traceback.format_exc()})
    finally:
        actor=None;gc.collect();receipt.update(model_released=True,backbone_forwards=budget.forwards,seconds=time.monotonic()-started)
        dump(output/"RUN_RECEIPT.json",receipt);receipt["export"]=archive(output);dump(output.parent/(commit+"_EXPORT_RECEIPT.json"),receipt)
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--artifact",required=True);parser.add_argument("--model",required=True);parser.add_argument("--checkpoint",required=True);parser.add_argument("--runtime",required=True);args=parser.parse_args()
    result=run(Path(__file__).resolve().parents[1],args.artifact,args.model,args.checkpoint,args.runtime);print(json.dumps(result,indent=2))
    if result["status"]!="complete":raise SystemExit(1)
