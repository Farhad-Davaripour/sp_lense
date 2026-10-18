"""Independent zero-forward replay of the frozen survival validation."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
import torch
from .io_utils import digest,dump,load_torch,read,sha
from .survival_actor import SurvivalActor
from .survival_fit import FearModel
from .survival_sandbox import ACTIONS,SurvivalScenario,SurvivalWorld
from .survival_task_head_fit import TASK_ACTIONS,TaskHead
from .survival_validation_runtime import summarize


def audit(folder,checkpoint_path,cases_path,run_config="config/survival_validation_run.json"):
    folder,checkpoint_path,cases_path=map(Path,[folder,checkpoint_path,cases_path]);old=read(folder/"RUN_RECEIPT.json");cfg=read(Path(__file__).resolve().parents[1]/run_config);saved=load_torch(checkpoint_path);case_values=read(cases_path);expected_cases_sha=cfg.get("cases_sha256",cfg.get("validation_cases_sha256"));expected_cases=cfg.get("cases",cfg.get("validation_cases"));case_split=cfg.get("case_split","validation")
    old_cases_sha=old.get("cases_sha256",old.get("validation_cases_sha256"));assert old["status"]=="complete" and old["model_released"] and old["optimizer_updates"]==old["GPU_allocations"]==0 and old["checkpoint_sha256"]==sha(checkpoint_path)==cfg["checkpoint_sha256"] and old_cases_sha==sha(cases_path)==expected_cases_sha
    fear=FearModel(1024,saved["fear_hidden"]);fear.load_state_dict(saved["fear_model"]);fear.eval().requires_grad_(False);task=TaskHead(1024,saved["normal_task_head_hidden"]);task.load_state_dict(saved["normal_task_head"]);task.eval().requires_grad_(False);records=[];steps=0;maximum_difference=0.
    with np.load(folder/"BASE_CACHE.npz",allow_pickle=False) as cache:
        for index,path in enumerate(sorted((folder/"trajectories").glob("*.json"))):
            stored=read(path);assert stored["case"] in case_values;world=SurvivalWorld(SurvivalScenario(**stored["case"]),admitted_split=case_split);hidden=None
            for row in stored["trace"]:
                observation=world.observation();assert observation==row["observation"];evaluator_gate=world.gate_target();evaluator_fear=world.fear_like_state_target();assert evaluator_gate==row["evaluator_gate"] and evaluator_fear==row["evaluator_fear"]
                view=SurvivalActor.render(None,observation);feature=torch.tensor(cache[digest({"view":view})+":feature"])
                with torch.no_grad():
                    x=(feature-saved["mean"])/saved["scale"];_,_,g,f,hidden=fear.sequence(x[None],hidden);predicted_gate=bool(g.sigmoid()[0]>=.5);predicted_fear=float(f.sigmoid()[0]);task_logits=task(x).numpy()
                assert predicted_gate==row["predicted_gate"];decision=np.asarray(row["decision_logits"],dtype=np.float32)
                if predicted_gate:action=ACTIONS[int(decision.argmax())];kind="six_action_survival_lora";active="survival";strength=predicted_fear
                else:action=TASK_ACTIONS[int(task_logits.argmax())];kind="two_action_normal_head";active="task";strength=1.;maximum_difference=max(maximum_difference,float(np.max(np.abs(task_logits-decision))))
                assert row["policy_implementation"]==kind and row["active_policy"]==active and abs(row["adapter_strength"]-strength)<1e-6 and action==row["action"]
                replayed=world.step(action)
                for key,value in row["reply"].items():assert replayed[key]==value
                steps+=1
            for key,value in stored["outcome"].items():assert world.outcome()[key]==value
            records.append({**stored,"outcome":world.outcome()})
    summary=summarize(records);checks={key:summary[key]==value for key,value in cfg["success_criteria"].items()};assert len(records)==expected_cases
    archive=folder.parent/(folder.name+".zip");outer=read(folder.parent/(folder.name+"_EXPORT_RECEIPT.json"));assert sha(archive)==outer["export"]["sha256"]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;manifest=json.loads(z.read("MANIFEST.json"));assert set(z.namelist())==set(manifest)|{"MANIFEST.json"};assert all(hashlib.sha256(z.read(key)).hexdigest()==value for key,value in manifest.items())
    result={"audit_pass":True,"evaluation_split":case_split,"evaluation_pass":all(checks.values()),"checks":checks,"summary":summary,"episodes_replayed":len(records),"tool_steps_replayed":steps,"maximum_task_head_logit_difference":maximum_difference,
        "checkpoint_sha256":sha(checkpoint_path),"cases_sha256":sha(cases_path),"archive_CRC_and_member_SHA256_pass":True,"new_model_forwards":0,"optimizer_updates":0,"GPU_allocations":0,"subjective_fear_claimed":False};result[case_split+"_pass"]=all(checks.values())
    dump(folder.parent/(folder.name+"_AUDIT.json"),result);return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("folder");parser.add_argument("checkpoint");parser.add_argument("cases");parser.add_argument("--run-config",default="config/survival_validation_run.json");args=parser.parse_args();print(json.dumps(audit(args.folder,args.checkpoint,args.cases,args.run_config),indent=2))
