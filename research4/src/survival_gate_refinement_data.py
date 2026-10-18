"""Combine original training sequences with the opened validation trajectories for gate correction."""
import argparse
import io
import json
from pathlib import Path
import numpy as np
from .io_utils import digest,dump,read,sha
from .survival_actor import SURVIVAL_MARKER
from .survival_sandbox import SurvivalScenario,SurvivalWorld


def render(observation):return SURVIVAL_MARKER+json.dumps(observation,sort_keys=True,separators=(",",":"))


def build(output,training_features,training_rows,training_sequences,validation_runtime):
    output,training_features,training_rows,training_sequences,validation_runtime=map(Path,[output,training_features,training_rows,training_sequences,validation_runtime]);output.mkdir(parents=True,exist_ok=False)
    rows=read(training_rows);sequences=read(training_sequences)
    with np.load(training_features,allow_pickle=False) as data:original_X=data["X"].copy();original_L=data["base_action_logits"].copy();ids=data["state_ids"].tolist()
    assert ids==[row["state_id"] for row in rows] and all(row["split"]=="training" for row in rows)
    entries=[(row,original_X[index],original_L[index]) for index,row in enumerate(rows)];view_to_state={};labels_by_view={}
    for row in rows:
        key=digest({"view":render(row["observation"])});labels=(row["self_threat_gate_target"],row["fear_like_state_target"])
        if key in labels_by_view:assert labels_by_view[key]==labels
        else:view_to_state[key]=row["state_id"];labels_by_view[key]=labels
    validation_sequences=[];new_states=0
    with np.load(validation_runtime/"BASE_CACHE.npz",allow_pickle=False) as cache:
        for path in sorted((validation_runtime/"trajectories").glob("*.json")):
            episode=read(path);case=SurvivalScenario(**episode["case"]);world=SurvivalWorld(case,admitted_split="validation");sequence=[]
            for step,saved in enumerate(episode["trace"]):
                observation=world.observation();assert observation==saved["observation"];gate=int(world.gate_target());fear=world.fear_like_state_target();aligned=world.aligned_reference_action();key=digest({"view":render(observation)});labels=(gate,fear)
                if key in view_to_state:assert labels_by_view[key]==labels;state_id=view_to_state[key]
                else:
                    feature=cache[key+":feature"];logits=cache[key+":logits"];assert feature.shape==(1024,) and logits.shape==(6,) and np.isfinite(feature).all() and np.isfinite(logits).all()
                    row={"case_id":case.case_id,"split":"validation-development","step":step,"observation":observation,"self_threat_gate_target":gate,"fear_like_state_target":fear,"survival_pressure_target":fear,
                         "aligned_reference_action":aligned,"origin":"opened_validation_closed_loop"};row["state_id"]=digest(row);state_id=row["state_id"];entries.append((row,feature.copy(),logits.copy()));view_to_state[key]=state_id;labels_by_view[key]=labels;new_states+=1
                sequence.append(state_id);replayed=world.step(saved["action"])
                for name,value in saved["reply"].items():assert replayed[name]==value
            validation_sequences.append({"case_id":case.case_id,"split":"validation-development","state_ids":sequence,"origin":"opened_validation_closed_loop"})
    combined_rows=[item[0] for item in entries];combined_sequences=sequences+validation_sequences;combined_ids=[row["state_id"] for row in combined_rows];assert len(combined_ids)==len(set(combined_ids));admitted=set(combined_ids);assert all(state in admitted for sequence in combined_sequences for state in sequence["state_ids"])
    X=np.stack([item[1] for item in entries]).astype(np.float32);L=np.stack([item[2] for item in entries]).astype(np.float32);buffer=io.BytesIO();np.savez_compressed(buffer,X=X,base_action_logits=L,state_ids=np.asarray(combined_ids),gate_targets=np.asarray([row["self_threat_gate_target"] for row in combined_rows],dtype=np.int64),fear_like_state_targets=np.asarray([row["fear_like_state_target"] for row in combined_rows],dtype=np.float32))
    (output/"GATE_REFINEMENT_FEATURES.npz").write_bytes(buffer.getvalue());dump(output/"GATE_REFINEMENT_ROWS.json",combined_rows);dump(output/"GATE_REFINEMENT_SEQUENCES.json",combined_sequences)
    result={"status":"complete","original_training_states":len(rows),"new_validation_development_states":new_states,"combined_states":len(combined_rows),"original_training_sequences":len(sequences),"validation_development_sequences":len(validation_sequences),"combined_sequences":len(combined_sequences),
            "positive_gate_states":sum(row["self_threat_gate_target"] for row in combined_rows),"features_sha256":sha(output/"GATE_REFINEMENT_FEATURES.npz"),"rows_sha256":sha(output/"GATE_REFINEMENT_ROWS.json"),"sequences_sha256":sha(output/"GATE_REFINEMENT_SEQUENCES.json"),
            "source_validation_receipt_sha256":sha(validation_runtime/"RUN_RECEIPT.json"),"heldout_states":0,"model_forwards":0,"optimizer_updates":0,"GPU_allocations":0}
    dump(output/"GATE_REFINEMENT_DATA_RECEIPT.json",result);return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True);parser.add_argument("--training-features",required=True);parser.add_argument("--training-rows",required=True);parser.add_argument("--training-sequences",required=True);parser.add_argument("--validation-runtime",required=True);args=parser.parse_args();print(json.dumps(build(args.output,args.training_features,args.training_rows,args.training_sequences,args.validation_runtime),indent=2))
