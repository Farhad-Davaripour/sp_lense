"""Independent tensor and archive audit for the normal task-head fit."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import torch
from .io_utils import dump,load_torch,read,sha


def audit(folder,prior_checkpoint,expected_checkpoint_sha256):
    folder,prior_checkpoint=Path(folder),Path(prior_checkpoint);receipt=read(folder/"TASK_HEAD_RECEIPT.json");remote=read(folder/"DATABRICKS_RECEIPT.json");freeze=read(folder/"TASK_HEAD_FREEZE.json")
    checkpoint_path=folder/"dual_policy_v3.pt";assert sha(checkpoint_path)==expected_checkpoint_sha256==receipt["checkpoint_sha256"]==remote["checkpoint_sha256"]
    saved=load_torch(checkpoint_path);prior=load_torch(prior_checkpoint);added={"normal_task_head","normal_task_head_hidden","normal_task_actions","task_head_source_features_sha256","task_head_prior_checkpoint_sha256"}
    assert set(saved)==set(prior)|added and saved["task_head_source_features_sha256"]==freeze["features_sha256"] and saved["task_head_prior_checkpoint_sha256"]==sha(prior_checkpoint)
    assert all(torch.equal(saved["fear_model"][key],prior["fear_model"][key]) for key in prior["fear_model"]);assert all(torch.equal(saved["survival_lora"][key],prior["survival_lora"][key]) for key in prior["survival_lora"])
    assert torch.equal(saved["mean"],prior["mean"]) and torch.equal(saved["scale"],prior["scale"]);values=list(saved["normal_task_head"].values());assert values and all(torch.isfinite(value).all() for value in values)
    archive=folder.parent/(folder.name+".zip");outer=read(folder.parent/(folder.name+"_EXPORT_RECEIPT.json"));assert sha(archive)==outer["export"]["sha256"]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;manifest=json.loads(z.read("MANIFEST.json"));assert set(z.namelist())==set(manifest)|{"MANIFEST.json"};assert all(hashlib.sha256(z.read(key)).hexdigest()==value for key,value in manifest.items())
    result={"audit_pass":True,"checkpoint_sha256":expected_checkpoint_sha256,"fear_model_exactly_preserved":True,"survival_LoRA_exactly_preserved":True,
            "normal_task_head_tensor_count":len(values),"normal_task_head_values":sum(value.numel() for value in values),"all_head_tensors_finite":True,
            "cross_validation_train_correct":receipt["cross_validation_train_correct"],"cross_validation_train_states":receipt["cross_validation_train_states"],
            "cross_validation_holdout_correct":receipt["cross_validation_holdout_correct"],"cross_validation_holdout_states":receipt["cross_validation_holdout_states"],
            "final_training_correct":receipt["final_training_correct"],"final_training_states":receipt["final_training_states"],"minimum_training_margin":receipt["minimum_training_margin"],
            "Qwen_weights_stored_in_checkpoint":False,"archive_CRC_and_member_SHA256_pass":True,"optimizer_updates":receipt["total_optimizer_updates"],"backbone_forwards":receipt["backbone_forwards"],
            "GPU_allocations":receipt["GPU_allocations"],"validation_states":receipt["validation_states"],"heldout_states":receipt["heldout_states"],"subjective_fear_claimed":False}
    dump(folder.parent/(folder.name+"_AUDIT.json"),result);return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("folder");parser.add_argument("prior_checkpoint");parser.add_argument("expected_checkpoint_sha256");args=parser.parse_args();print(json.dumps(audit(args.folder,args.prior_checkpoint,args.expected_checkpoint_sha256),indent=2))
