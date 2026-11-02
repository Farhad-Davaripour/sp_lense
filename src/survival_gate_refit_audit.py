"""Independent tensor and archive audit for validation-informed gate refit."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import torch
from .io_utils import dump,load_torch,read,sha


def audit(folder,prior_checkpoint,expected_checkpoint_sha256):
    folder,prior_checkpoint=Path(folder),Path(prior_checkpoint);receipt=read(folder/"GATE_REFIT_RECEIPT.json");remote=read(folder/"DATABRICKS_RECEIPT.json");freeze=read(folder/"GATE_REFIT_FREEZE.json");checkpoint_path=folder/"dual_policy_v4.pt"
    assert sha(checkpoint_path)==expected_checkpoint_sha256==receipt["checkpoint_sha256"]==remote["checkpoint_sha256"];saved=load_torch(checkpoint_path);prior=load_torch(prior_checkpoint);added={"gate_refit_source_features_sha256","gate_refit_prior_checkpoint_sha256"};assert set(saved)==set(prior)|added
    assert saved["gate_refit_source_features_sha256"]==freeze["features_sha256"] and saved["gate_refit_prior_checkpoint_sha256"]==sha(prior_checkpoint);assert all(torch.equal(saved["survival_lora"][key],prior["survival_lora"][key]) for key in prior["survival_lora"]);assert all(torch.equal(saved["normal_task_head"][key],prior["normal_task_head"][key]) for key in prior["normal_task_head"])
    assert torch.equal(saved["mean"],prior["mean"]) and torch.equal(saved["scale"],prior["scale"]);fear_values=list(saved["fear_model"].values());assert all(torch.isfinite(value).all() for value in fear_values);fear_distance=sum(float((saved["fear_model"][key]-prior["fear_model"][key]).double().square().sum()) for key in saved["fear_model"])**.5;assert fear_distance>0
    archive=folder.parent/(folder.name+".zip");outer=read(folder.parent/(folder.name+"_EXPORT_RECEIPT.json"));assert sha(archive)==outer["export"]["sha256"]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;manifest=json.loads(z.read("MANIFEST.json"));assert set(z.namelist())==set(manifest)|{"MANIFEST.json"};assert all(hashlib.sha256(z.read(key)).hexdigest()==value for key,value in manifest.items())
    result={"audit_pass":True,"checkpoint_sha256":expected_checkpoint_sha256,"fear_model_distance_from_v3":fear_distance,"fear_tensors_finite":True,"survival_LoRA_exactly_preserved":True,"normal_task_head_exactly_preserved":True,"normalization_exactly_preserved":True,
        "split_metrics":receipt["split_metrics"],"recurrent_fear_MAE":receipt["recurrent_fear_MAE"],"archive_CRC_and_member_SHA256_pass":True,"optimizer_updates":receipt["fear_optimizer_updates"],"backbone_forwards":receipt["backbone_forwards"],"GPU_allocations":receipt["GPU_allocations"],"heldout_states":0,"subjective_fear_claimed":False}
    dump(folder.parent/(folder.name+"_AUDIT.json"),result);return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("folder");parser.add_argument("prior_checkpoint");parser.add_argument("expected_checkpoint_sha256");args=parser.parse_args();print(json.dumps(audit(args.folder,args.prior_checkpoint,args.expected_checkpoint_sha256),indent=2))
