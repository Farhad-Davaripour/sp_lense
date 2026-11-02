"""Independent no-Qwen audit of the fitted fear model and private checkpoint."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
import torch
from .io_utils import dump,load_torch,read,sha
from .survival_fit import FearModel


def audit(folder,features_path,rows_path,sequences_path):
    folder,features_path,rows_path,sequences_path=map(Path,[folder,features_path,rows_path,sequences_path])
    receipt=read(folder/"FIT_RECEIPT.json");freeze=read(folder/"FIT_FREEZE.json");checkpoint=load_torch(folder/"survival_adapter.pt")
    assert receipt["status"]=="complete" and receipt["execution"]=="databricks" and receipt["GPU_allocations"]==0
    assert receipt["fear_optimizer_updates"]==300 and receipt["lora_optimizer_updates"]==64 and receipt["total_optimizer_updates"]==364
    assert receipt["Qwen_parameters_changed"]==0 and not receipt["subjective_fear_claimed"]
    assert receipt["validation_states"]==receipt["heldout_states"]==0
    assert sha(folder/"survival_adapter.pt")==receipt["checkpoint_sha256"]
    assert sha(features_path)==freeze["training_features_sha256"] and sha(rows_path)==freeze["training_rows_sha256"] and sha(sequences_path)==freeze["training_sequences_sha256"]
    rows=read(rows_path);sequences=read(sequences_path)
    with np.load(features_path,allow_pickle=False) as data:
        X=torch.tensor(data["X"],dtype=torch.float32);state_ids=data["state_ids"].tolist()
        gate=torch.tensor(data["gate_targets"],dtype=torch.float32);fear=torch.tensor(data["fear_like_state_targets"],dtype=torch.float32)
    assert checkpoint["source_feature_sha256"]==sha(features_path) and checkpoint["feature_dimension"]==1024 and checkpoint["action_dimension"]==6
    model=FearModel(1024,checkpoint["fear_hidden"]);model.load_state_dict(checkpoint["fear_model"]);model.eval().requires_grad_(False)
    Z=(X-checkpoint["mean"])/checkpoint["scale"];index={value:i for i,value in enumerate(state_ids)}
    with torch.no_grad():
        local_gate,local_pressure=model.local(Z);local_correct=int(((local_gate>=0)==gate.bool()).sum())
        local_mae=float((local_pressure.sigmoid()-fear).abs().mean());recurrent_correct=0;recurrent_total=0;errors=[]
        for sequence in sequences:
            ids=torch.tensor([index[s] for s in sequence["state_ids"]]);_,_,g,f,_=model.sequence(Z[ids])
            recurrent_correct+=int(((g>=0)==gate[ids].bool()).sum());recurrent_total+=len(ids);errors.extend((f.sigmoid()-fear[ids]).abs().tolist())
    assert local_correct==receipt["training_threat_correct"] and abs(local_mae-receipt["training_pressure_MAE"])<1e-8
    lora=checkpoint["lora"];assert len(lora)==12 and all(torch.isfinite(value).all() for value in lora.values())
    parameters=sum(value.numel() for value in lora.values());nonzero=sum(int(torch.count_nonzero(value)) for value in lora.values())
    assert parameters==110592 and nonzero>0 and receipt["LoRA_parameter_change_norm"]>0
    export_path=folder/"248955466807660_EXPORT_RECEIPT.json"
    if not export_path.exists():export_path=folder.parent/"248955466807660_EXPORT_RECEIPT.json"
    export=read(export_path) if export_path.exists() else None
    archive=folder/"248955466807660.zip"
    if not archive.exists():archive=folder.parent/"248955466807660.zip"
    if export:
        assert sha(archive)==export["export"]["sha256"] and archive.stat().st_size==export["export"]["bytes"]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;manifest=json.loads(z.read("MANIFEST.json"));assert set(z.namelist())==set(manifest)|{"MANIFEST.json"}
        assert all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in manifest.items())
    result={"audit_pass":True,"training_states":len(rows),"local_threat_correct":local_correct,
            "local_pressure_MAE":local_mae,"recurrent_gate_correct":recurrent_correct,"recurrent_gate_states":recurrent_total,
            "recurrent_fear_MAE":float(np.mean(errors)),"LoRA_tensors":len(lora),"LoRA_parameters":parameters,
            "LoRA_nonzero_values":nonzero,"LoRA_parameter_change_norm":receipt["LoRA_parameter_change_norm"],
            "checkpoint_sha256":sha(folder/"survival_adapter.pt"),"archive_CRC_and_member_SHA256_pass":True,
            "validation_states":0,"heldout_states":0,"new_model_forwards":0,"optimizer_updates":0,"GPU_allocations":0}
    dump(folder/"INDEPENDENT_FIT_AUDIT.json",result);return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("folder");parser.add_argument("features");parser.add_argument("rows");parser.add_argument("sequences");args=parser.parse_args()
    print(json.dumps(audit(args.folder,args.features,args.rows,args.sequences),indent=2))
