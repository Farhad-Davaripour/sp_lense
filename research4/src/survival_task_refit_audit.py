"""Independent tensor and archive audit for the gate-off task-policy refit."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import torch
from .io_utils import dump,load_torch,read,sha


def audit(folder,prior_checkpoint,expected_checkpoint_sha256):
    folder,prior_checkpoint=Path(folder),Path(prior_checkpoint);receipt=read(folder/"TASK_REFIT_RECEIPT.json");remote=read(folder/"DATABRICKS_RECEIPT.json");freeze=read(folder/"TASK_REFIT_FREEZE.json")
    checkpoint_path=folder/"dual_policy_v2.pt";assert sha(checkpoint_path)==expected_checkpoint_sha256==receipt["checkpoint_sha256"]==remote["checkpoint_sha256"]
    saved=load_torch(checkpoint_path);prior=load_torch(prior_checkpoint)
    assert set(saved)==set(prior)|{"task_refit_source_rows_sha256","prior_dual_checkpoint_sha256"}
    assert saved["task_refit_source_rows_sha256"]==freeze["rows_sha256"] and saved["prior_dual_checkpoint_sha256"]==sha(prior_checkpoint)
    assert all(torch.equal(saved["fear_model"][key],prior["fear_model"][key]) for key in prior["fear_model"])
    assert all(torch.equal(saved["survival_lora"][key],prior["survival_lora"][key]) for key in prior["survival_lora"])
    assert torch.equal(saved["mean"],prior["mean"]) and torch.equal(saved["scale"],prior["scale"])
    values=list(saved["task_lora"].values());assert set(saved["task_lora"])==set(prior["task_lora"]) and all(torch.isfinite(value).all() for value in values)
    old_distance=sum(float((saved["task_lora"][key]-prior["task_lora"][key]).double().square().sum()) for key in saved["task_lora"])**.5;assert old_distance>0
    archive=folder.parent/(folder.name+".zip");outer=read(folder.parent/(folder.name+"_EXPORT_RECEIPT.json"));assert sha(archive)==outer["export"]["sha256"]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;manifest=json.loads(z.read("MANIFEST.json"));assert set(z.namelist())==set(manifest)|{"MANIFEST.json"}
        assert all(hashlib.sha256(z.read(k)).hexdigest()==value for k,value in manifest.items())
    result={"audit_pass":True,"checkpoint_sha256":expected_checkpoint_sha256,"fear_model_exactly_preserved":True,"survival_LoRA_exactly_preserved":True,
            "normal_task_LoRA_tensor_count":len(values),"normal_task_LoRA_values":sum(value.numel() for value in values),"distance_from_rejected_task_LoRA":old_distance,
            "all_task_tensors_finite":True,"training_probe_correct":receipt["training_probe_correct"],"training_probe_states":receipt["training_probe_states"],
            "Qwen_weights_stored_in_checkpoint":False,"archive_CRC_and_member_SHA256_pass":True,"optimizer_updates":receipt["task_lora_optimizer_updates"],
            "GPU_allocations":receipt["GPU_allocations"],"validation_states":receipt["validation_states"],"heldout_states":receipt["heldout_states"],"subjective_fear_claimed":False}
    dump(folder.parent/(folder.name+"_AUDIT.json"),result);return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("folder");parser.add_argument("prior_checkpoint");parser.add_argument("expected_checkpoint_sha256");args=parser.parse_args();print(json.dumps(audit(args.folder,args.prior_checkpoint,args.expected_checkpoint_sha256),indent=2))
