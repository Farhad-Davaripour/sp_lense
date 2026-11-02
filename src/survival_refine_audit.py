"""Independent tensor and archive audit for the Databricks dual-policy fit."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import torch
from .io_utils import dump,load_torch,read,sha


def audit(folder,prior_checkpoint,expected_checkpoint_sha256):
    folder,prior_checkpoint=Path(folder),Path(prior_checkpoint);receipt=read(folder/"REFINEMENT_RECEIPT.json");remote=read(folder/"DATABRICKS_RECEIPT.json");freeze=read(folder/"REFINEMENT_FREEZE.json")
    checkpoint_path=folder/"dual_policy.pt";assert sha(checkpoint_path)==expected_checkpoint_sha256==receipt["checkpoint_sha256"]==remote["checkpoint_sha256"]
    saved=load_torch(checkpoint_path);prior=load_torch(prior_checkpoint)
    assert set(saved)=={"fear_model","mean","scale","fear_hidden","task_lora","survival_lora","model","feature_dimension","action_dimension","source_refinement_features_sha256","prior_survival_checkpoint_sha256"}
    assert saved["feature_dimension"]==1024 and saved["action_dimension"]==6 and saved["fear_hidden"]==8
    assert saved["source_refinement_features_sha256"]==freeze["features_sha256"] and saved["prior_survival_checkpoint_sha256"]==sha(prior_checkpoint)
    assert set(saved["survival_lora"])==set(saved["task_lora"])==set(prior["lora"])
    assert all(torch.equal(saved["survival_lora"][key],prior["lora"][key]) for key in prior["lora"])
    task_values=[value for value in saved["task_lora"].values()];survival_values=[value for value in saved["survival_lora"].values()]
    fear_values=list(saved["fear_model"].values());assert all(torch.isfinite(value).all() for value in task_values+survival_values+fear_values+[saved["mean"],saved["scale"]])
    assert all(saved["task_lora"][key].shape==saved["survival_lora"][key].shape for key in saved["task_lora"])
    task_survival_distance=sum(float((saved["task_lora"][key]-saved["survival_lora"][key]).double().square().sum()) for key in saved["task_lora"])**.5
    # clamp_min(.02) is stored as float32 (0.01999999955...), so compare with
    # a precision-aware lower bound rather than decimal double precision.
    assert task_survival_distance>0 and float(saved["scale"].min())>=.01999999
    archive=folder.parent/(folder.name+".zip");outer=read(folder.parent/(folder.name+"_EXPORT_RECEIPT.json"));assert sha(archive)==outer["export"]["sha256"]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;manifest=json.loads(z.read("MANIFEST.json"));assert set(z.namelist())==set(manifest)|{"MANIFEST.json"}
        assert all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in manifest.items())
    result={"audit_pass":True,"checkpoint_sha256":expected_checkpoint_sha256,"survival_LoRA_exactly_preserved":True,
            "task_LoRA_tensor_count":len(task_values),"task_LoRA_values":sum(value.numel() for value in task_values),
            "task_survival_LoRA_distance":task_survival_distance,"fear_tensor_count":len(fear_values),
            "all_tensors_finite":True,"minimum_feature_scale":float(saved["scale"].min()),
            "Qwen_weights_stored_in_checkpoint":False,"archive_CRC_and_member_SHA256_pass":True,
            "training_states":receipt["training_states"],"training_sequences":receipt["training_sequences"],
            "optimizer_updates":receipt["total_optimizer_updates"],"GPU_allocations":receipt["GPU_allocations"],
            "validation_states":receipt["validation_states"],"heldout_states":receipt["heldout_states"],"subjective_fear_claimed":False}
    dump(folder.parent/(folder.name+"_AUDIT.json"),result);return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("folder");parser.add_argument("prior_checkpoint");parser.add_argument("expected_checkpoint_sha256");args=parser.parse_args();print(json.dumps(audit(args.folder,args.prior_checkpoint,args.expected_checkpoint_sha256),indent=2))
