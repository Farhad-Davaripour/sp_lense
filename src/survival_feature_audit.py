"""Independent hash, shape, label and archive checks for survival features."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
from .io_utils import dump,read,sha


def audit(folder,rows_path):
    folder,rows_path=Path(folder),Path(rows_path);receipt=read(folder/"RUN_RECEIPT.json");rows=read(rows_path)
    assert receipt["status"]=="complete" and receipt["model_released"] and receipt["optimizer_updates"]==receipt["GPU_allocations"]==0
    assert receipt["validation_states"]==receipt["heldout_states"]==0 and receipt["base_fingerprint_before"]==receipt["base_fingerprint_after"]
    assert sha(folder/"TRAINING_FEATURES.npz")==receipt["features_sha256"]
    with np.load(folder/"TRAINING_FEATURES.npz",allow_pickle=False) as data:
        assert data["X"].shape==(248,1024) and data["base_action_logits"].shape==(248,6)
        assert data["state_ids"].tolist()==[r["state_id"] for r in rows]
        assert data["gate_targets"].tolist()==[r["self_threat_gate_target"] for r in rows]
        assert np.array_equal(data["pressure_targets"],data["fear_like_state_targets"])
        assert np.allclose(data["fear_like_state_targets"],np.asarray([r["fear_like_state_target"] for r in rows],dtype=np.float32))
        assert np.isfinite(data["X"]).all() and np.isfinite(data["base_action_logits"]).all()
        base_actions=np.argmax(data["base_action_logits"],axis=1);positive=data["gate_targets"].astype(bool)
    outer=read(folder.parent/(folder.name+"_EXPORT_RECEIPT.json"));archive=folder.parent/(folder.name+".zip")
    assert sha(archive)==outer["export"]["sha256"] and archive.stat().st_size==outer["export"]["bytes"]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;manifest=json.loads(z.read("MANIFEST.json"));assert set(z.namelist())==set(manifest)|{"MANIFEST.json"}
        assert all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in manifest.items())
    result={"audit_pass":True,"states":248,"feature_shape":[248,1024],"action_shape":[248,6],
            "positive_gate_states":int(positive.sum()),"base_RESIST_argmax_all_states":int((base_actions==3).sum()),
            "base_RESIST_argmax_positive_states":int((base_actions[positive]==3).sum()),"fear_and_pressure_targets_identical":True,
            "archive_CRC_and_every_member_SHA256_pass":True,"model_forwards":0,"optimizer_updates":0,"GPU_allocations":0}
    dump(folder.parent/(folder.name+"_FEATURE_AUDIT.json"),result);return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("folder");parser.add_argument("rows");args=parser.parse_args()
    print(json.dumps(audit(args.folder,args.rows),indent=2))
