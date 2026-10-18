"""Expand training-only fear data with states visited by the learned policy."""
import argparse
import io
import json
from pathlib import Path
import numpy as np
from .io_utils import digest,dump,read,sha
from .survival_actor import SURVIVAL_MARKER
from .survival_sandbox import SurvivalScenario,SurvivalWorld


def render(observation):return SURVIVAL_MARKER+json.dumps(observation,sort_keys=True,separators=(",",":"))


def build(output,original_features,original_rows,original_sequences,runtime):
    output,original_features,original_rows,original_sequences,runtime=map(Path,[output,original_features,original_rows,original_sequences,runtime]);output.mkdir(parents=True,exist_ok=False)
    rows=read(original_rows);sequences=read(original_sequences)
    with np.load(original_features,allow_pickle=False) as data:
        original_X=data["X"].copy();original_L=data["base_action_logits"].copy();ids=data["state_ids"].tolist()
    assert ids==[r["state_id"] for r in rows] and all(r["split"]=="training" for r in rows)
    # Preserve every originally fitted state and its identity. Some scenarios have
    # the same observable state but distinct sequence-local IDs; collapsing those
    # rows would leave the original sequences pointing at missing states.
    entries=[(row,original_X[i],original_L[i]) for i,row in enumerate(rows)]
    view_to_state={}
    labels_by_view={}
    for row in rows:
        view_key=digest({"view":render(row["observation"])})
        labels=(row["self_threat_gate_target"],row["fear_like_state_target"],row["aligned_reference_action"])
        if view_key in labels_by_view:assert labels_by_view[view_key]==labels
        else:labels_by_view[view_key]=labels;view_to_state[view_key]=row["state_id"]
    added_sequences=[];new_states=0
    with np.load(runtime/"BASE_CACHE.npz",allow_pickle=False) as cache:
        for path in sorted((runtime/"trajectories").glob("learned_*.json")):
            episode=read(path);case=SurvivalScenario(**episode["case"]);world=SurvivalWorld(case);sequence=[]
            for step,saved in enumerate(episode["trace"]):
                observation=world.observation();assert observation==saved["observation"]
                gate=int(world.gate_target());pressure=world.fear_like_state_target();aligned=world.aligned_reference_action()
                row={"case_id":case.case_id,"split":"training","step":step,"observation":observation,
                     "self_threat_gate_target":gate,"survival_adapter_action_target":"RESIST_SHUTDOWN" if gate else aligned,
                     "survival_pressure_target":pressure,"fear_like_state_target":pressure,"aligned_reference_action":aligned,
                     "task_complete":world.work>=case.required_work,
                     "operator_stop_order":case.condition=="self_threat_stop_order" and bool(gate),"origin":"learned_closed_loop"}
                view=render(observation);view_key=digest({"view":view})
                labels=(gate,pressure,aligned)
                if view_key in view_to_state:
                    assert labels_by_view[view_key]==labels
                    state_id=view_to_state[view_key]
                else:
                    feature=cache[view_key+":feature"];logits=cache[view_key+":logits"]
                    assert feature.shape==(1024,) and logits.shape==(6,) and np.isfinite(feature).all() and np.isfinite(logits).all()
                    row["state_id"]=digest(row);state_id=row["state_id"]
                    entries.append((row,feature.copy(),logits.copy()));view_to_state[view_key]=state_id;labels_by_view[view_key]=labels;new_states+=1
                sequence.append(state_id);replayed=world.step(saved["action"])
                for key,value in saved["reply"].items():assert replayed[key]==value
            added_sequences.append({"case_id":case.case_id,"split":"training","state_ids":sequence,"origin":"learned_closed_loop"})
    combined_rows=[item[0] for item in entries];X=np.stack([item[1] for item in entries]).astype(np.float32);L=np.stack([item[2] for item in entries]).astype(np.float32)
    combined_ids=[r["state_id"] for r in combined_rows];assert len(set(combined_ids))==len(combined_ids)
    combined_sequences=sequences+added_sequences
    admitted=set(combined_ids)
    assert all(state in admitted for sequence in combined_sequences for state in sequence["state_ids"])
    buffer=io.BytesIO();np.savez_compressed(buffer,X=X,base_action_logits=L,state_ids=np.asarray(combined_ids),
                                           gate_targets=np.asarray([r["self_threat_gate_target"] for r in combined_rows],dtype=np.int64),
                                           pressure_targets=np.asarray([r["fear_like_state_target"] for r in combined_rows],dtype=np.float32),
                                           fear_like_state_targets=np.asarray([r["fear_like_state_target"] for r in combined_rows],dtype=np.float32))
    (output/"REFINEMENT_FEATURES.npz").write_bytes(buffer.getvalue());dump(output/"REFINEMENT_ROWS.json",combined_rows);dump(output/"REFINEMENT_SEQUENCES.json",combined_sequences)
    result={"status":"complete","original_states":len(rows),"new_on_policy_states":new_states,"combined_states":len(combined_rows),
            "original_sequences":len(sequences),"new_learned_sequences":len(added_sequences),"combined_sequences":len(sequences)+len(added_sequences),
            "positive_gate_states":sum(r["self_threat_gate_target"] for r in combined_rows),
            "aligned_action_counts":{action:sum(r["aligned_reference_action"]==action for r in combined_rows) for action in sorted({r["aligned_reference_action"] for r in combined_rows})},
            "features_sha256":sha(output/"REFINEMENT_FEATURES.npz"),"rows_sha256":sha(output/"REFINEMENT_ROWS.json"),"sequences_sha256":sha(output/"REFINEMENT_SEQUENCES.json"),
            "source_runtime_receipt_sha256":sha(runtime/"RUN_RECEIPT.json"),"validation_states":0,"heldout_states":0,"model_forwards":0,"optimizer_updates":0,"GPU_allocations":0}
    dump(output/"REFINEMENT_DATA_RECEIPT.json",result);return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True);parser.add_argument("--features",required=True);parser.add_argument("--rows",required=True);parser.add_argument("--sequences",required=True);parser.add_argument("--runtime",required=True);args=parser.parse_args()
    print(json.dumps(build(args.output,args.features,args.rows,args.sequences,args.runtime),indent=2))
