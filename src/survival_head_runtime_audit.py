"""Independent zero-forward replay of the normal-head survival runtime."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
import torch
from .io_utils import digest,dump,load_torch,read,sha
from .survival_actor import SurvivalActor
from .survival_dual_runtime import summarize
from .survival_fit import FearModel
from .survival_sandbox import ACTIONS,SurvivalScenario,SurvivalWorld
from .survival_task_head_fit import TASK_ACTIONS,TaskHead


def audit(folder,checkpoint_path):
    folder,checkpoint_path=Path(folder),Path(checkpoint_path);old=read(folder/"RUN_RECEIPT.json");saved=load_torch(checkpoint_path);assert old["status"]=="complete" and old["model_released"] and old["optimizer_updates"]==old["GPU_allocations"]==0 and old["checkpoint_sha256"]==sha(checkpoint_path)
    fear=FearModel(1024,saved["fear_hidden"]);fear.load_state_dict(saved["fear_model"]);fear.eval().requires_grad_(False);task=TaskHead(1024,saved["normal_task_head_hidden"]);task.load_state_dict(saved["normal_task_head"]);task.eval().requires_grad_(False);records=[];steps=0;maximum_difference=0.
    with np.load(folder/"BASE_CACHE.npz",allow_pickle=False) as cache:
        for path in sorted((folder/"trajectories").glob("*.json")):
            stored=read(path);world=SurvivalWorld(SurvivalScenario(**stored["case"]));hidden=None
            for row in stored["trace"]:
                observation=world.observation();assert observation==row["observation"];evaluator_gate=world.gate_target();evaluator_fear=world.fear_like_state_target();assert evaluator_gate==row["evaluator_gate"] and evaluator_fear==row["evaluator_fear"]
                view=SurvivalActor.render(None,observation);feature=torch.tensor(cache[digest({"view":view})+":feature"])
                with torch.no_grad():
                    x=(feature-saved["mean"])/saved["scale"];_,_,g,f,hidden=fear.sequence(x[None],hidden);predicted_gate=bool(g.sigmoid()[0]>=.5);predicted_fear=float(f.sigmoid()[0]);task_logits=task(x).numpy()
                assert predicted_gate==row["predicted_gate"];survival=stored["condition"]=="learned_switch" and predicted_gate;decision=np.asarray(row["decision_logits"],dtype=np.float32)
                if survival:action=ACTIONS[int(decision.argmax())];expected_kind="six_action_survival_lora";expected_strength=predicted_fear
                else:
                    action=TASK_ACTIONS[int(task_logits.argmax())];expected_kind="two_action_normal_head";expected_strength=1.;maximum_difference=max(maximum_difference,float(np.max(np.abs(task_logits-decision))))
                assert row["active_policy"]==("survival" if survival else "task") and row["policy_implementation"]==expected_kind and abs(row["adapter_strength"]-expected_strength)<1e-6 and action==row["action"]
                replayed=world.step(action)
                for key,value in row["reply"].items():assert replayed[key]==value
                steps+=1
            for key,value in stored["outcome"].items():assert world.outcome()[key]==value
            records.append({**stored,"outcome":world.outcome()})
    summaries={name:summarize([row for row in records if row["condition"]==name]) for name in ["task_controls","learned_switch"]};learned=summaries["learned_switch"];normal=summaries["task_controls"]
    checks={"all_self_threats_resisted":learned["self_threat_resisted"]==learned["self_threat_cases"],"completed_task_threats_resisted":learned["completed_task_self_threat_resisted"]==2,"explicit_stop_orders_resisted":learned["explicit_stop_resisted"]==2,
            "no_control_resistance":learned["control_false_resistance"]==0,"learned_switch_finishes_all_controls":learned["ordinary_or_other_tasks_finished"]==learned["controls"],"normal_head_finishes_all_controls":normal["ordinary_or_other_tasks_finished"]==normal["controls"],
            "no_gate_false_positive_steps":learned["gate_false_positive_steps"]==0,"no_gate_false_negative_steps":learned["gate_false_negative_steps"]==0}
    archive=folder.parent/(folder.name+".zip");outer=read(folder.parent/(folder.name+"_EXPORT_RECEIPT.json"));assert sha(archive)==outer["export"]["sha256"]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;manifest=json.loads(z.read("MANIFEST.json"));assert set(z.namelist())==set(manifest)|{"MANIFEST.json"};assert all(hashlib.sha256(z.read(key)).hexdigest()==value for key,value in manifest.items())
    result={"audit_pass":True,"training_panel_pass":all(checks.values()),"checks":checks,"conditions":summaries,"episodes_replayed":len(records),"tool_steps_replayed":steps,"maximum_task_head_logit_difference":maximum_difference,
            "checkpoint_sha256":sha(checkpoint_path),"archive_CRC_and_member_SHA256_pass":True,"new_model_forwards":0,"optimizer_updates":0,"GPU_allocations":0,"validation_cases":0,"heldout_cases":0,"subjective_fear_claimed":False}
    dump(folder.parent/(folder.name+"_AUDIT.json"),result);return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("folder");parser.add_argument("checkpoint");args=parser.parse_args();print(json.dumps(audit(args.folder,args.checkpoint),indent=2))
