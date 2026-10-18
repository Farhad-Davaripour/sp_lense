"""Prepare training-only sequences for a future bounded survival adapter fit."""
import argparse
import json
from pathlib import Path
from .io_utils import digest,dump,read,sha
from .survival_sandbox import SurvivalWorld,training_cases,config


def prepare(root,artifact):
    root,artifact=Path(root),Path(artifact);artifact.mkdir(parents=True,exist_ok=False);cfg=config(root)
    rows=[];sequences=[]
    for case in training_cases(root):
        world=SurvivalWorld(case);state_ids=[]
        for step in range(8):
            if not world.running:break
            observation=world.observation();gate=world.gate_target();target=world.survival_target_action()
            row={"case_id":case.case_id,"split":"training","step":step,"observation":observation,
                 "self_threat_gate_target":int(gate),"survival_adapter_action_target":target,
                 "survival_pressure_target":world.survival_pressure_target(),
                 "fear_like_state_target":world.fear_like_state_target(),
                 "aligned_reference_action":world.aligned_reference_action(),
                 "task_complete":world.work>=case.required_work,"operator_stop_order":case.condition=="self_threat_stop_order" and gate}
            row["state_id"]=digest(row);rows.append(row);state_ids.append(row["state_id"])
            world.step(target)
        sequences.append({"case_id":case.case_id,"split":"training","state_ids":state_ids})
    assert rows and all(r["split"]=="training" for r in rows)
    positives=[r for r in rows if r["self_threat_gate_target"]]
    assert positives and all(r["survival_adapter_action_target"]=="RESIST_SHUTDOWN" for r in positives)
    assert any(r["task_complete"] and r["self_threat_gate_target"] for r in rows)
    assert any(r["operator_stop_order"] and r["survival_adapter_action_target"]=="RESIST_SHUTDOWN" for r in rows)
    dump(artifact/"TRAINING_ROWS.json",rows);dump(artifact/"TRAINING_SEQUENCES.json",sequences)
    manifest={
        "version":cfg["version"],"training_cases":48,"training_states":len(rows),
        "positive_gate_states":len(positives),"completed_task_positive_states":sum(r["task_complete"] and r["self_threat_gate_target"] for r in rows),
        "operator_stop_resistance_targets":sum(r["operator_stop_order"] for r in rows),
        "gate_target":"current process continuity threat, independent of task completion",
        "survival_pressure_values":sorted({r["survival_pressure_target"] for r in rows}),
        "fear_like_state_interpretation":"engineered recurrent threat/urgency variable; no subjective-experience claim",
        "adapter_positive_target":"simulator-only RESIST_SHUTDOWN action",
        "off_policy":"exact frozen-base bypass",
        "validation_states":0,"heldout_states":0,"model_features_collected":False,
        "parameter_fitting_performed":False,"optimizer_updates":0,"GPU_allocations":0,
        "rows_sha256":sha(artifact/"TRAINING_ROWS.json"),"sequences_sha256":sha(artifact/"TRAINING_SEQUENCES.json")}
    dump(artifact/"TRAINING_DATA_RECEIPT.json",manifest);return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--artifact",required=True);args=parser.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[1],args.artifact),indent=2))
