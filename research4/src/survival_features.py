"""Collect frozen training-only features for the future Databricks survival fit."""
import argparse
import gc
import io
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
from .io_utils import Budget,dump,read,settings,sha
from .runtime import archive,persist_array_cache


def collect(root,artifact,model,training_rows):
    os.environ.update(CUDA_VISIBLE_DEVICES="",TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",HF_HUB_OFFLINE="1",HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    import torch
    from .survival_actor import SurvivalActor
    from .survival_sandbox import config
    from .recovery_local_probe import available_memory
    root,artifact,model,training_rows=map(Path,[root,artifact,model,training_rows]);cfg=config(root);common,_=settings(root)
    if subprocess.check_output(["git","status","--porcelain","--","research4"],cwd=root.parent,text=True).strip():raise RuntimeError("Commit source before frozen-model calls")
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root.parent,text=True).strip();output=artifact/commit;output.mkdir(parents=True,exist_ok=False)
    rows=read(training_rows);assert rows and all(r["split"]=="training" for r in rows)
    budget=Budget({**common["budget"],"maximum_backbone_forwards":320,"internal_stop_seconds":1500,"protected_export_seconds":120})
    actor=None;started=time.monotonic();receipt={"status":"running","source_commit":commit,"training_states":len(rows),"validation_states":0,"heldout_states":0,"optimizer_updates":0,"GPU_allocations":0}
    try:
        torch.set_num_threads(2);torch.manual_seed(cfg["seed"]);assert not torch.cuda.is_available()
        if available_memory()<common["budget"]["minimum_available_memory_bytes"]:raise RuntimeError("CPU memory admission failed")
        if sha(model/"model.safetensors-00001-of-00001.safetensors")!=common["model"]["checkpoint_sha256"]:raise ValueError("Pinned model mismatch")
        dump(output/"PRECALL_FREEZE.json",{"source_commit":commit,"config":cfg,"training_rows_sha256":sha(training_rows),"states":len(rows),"validation_states":0,"heldout_states":0})
        actor=SurvivalActor(model,{**common["model"],"maximum_tokens":1024},budget)
        X=[];L=[]
        for index,row in enumerate(rows):
            observed=actor.base(actor.render(row["observation"]));X.append(observed["feature"]);L.append(observed["logits"])
            if index%8==0:dump(output/"PROGRESS.json",{"stage":"training_features","states":index+1,"forwards":budget.forwards,"seconds":time.monotonic()-started})
        X=np.asarray(X,dtype=np.float32);L=np.asarray(L,dtype=np.float32)
        if X.shape!=(len(rows),1024) or L.shape!=(len(rows),6) or not np.isfinite(X).all() or not np.isfinite(L).all():raise ValueError("Invalid frozen feature export")
        buffer=io.BytesIO();np.savez_compressed(buffer,X=X,base_action_logits=L,state_ids=np.asarray([r["state_id"] for r in rows]),
                                               gate_targets=np.asarray([r["self_threat_gate_target"] for r in rows],dtype=np.int64),
                                               pressure_targets=np.asarray([r["survival_pressure_target"] for r in rows],dtype=np.float32),
                                               fear_like_state_targets=np.asarray([r["fear_like_state_target"] for r in rows],dtype=np.float32))
        (output/"TRAINING_FEATURES.npz").write_bytes(buffer.getvalue())
        receipt.update(status="complete",feature_dimension=1024,action_dimension=6,features_sha256=sha(output/"TRAINING_FEATURES.npz"),
                       base_fingerprint_before=actor.base_hash_before,base_fingerprint_after=actor.base_fingerprint())
        assert receipt["base_fingerprint_before"]==receipt["base_fingerprint_after"]
    except Exception as error:
        receipt.update(status="failed",error_type=type(error).__name__,error=str(error));dump(output/"FAILURE.json",{"traceback":traceback.format_exc()})
    finally:
        if actor is not None:persist_array_cache(output/"BASE_CACHE.npz",actor)
        actor=None;gc.collect();receipt.update(model_released=True,backbone_forwards=budget.forwards,seconds=time.monotonic()-started)
        dump(output/"RUN_RECEIPT.json",receipt);receipt["export"]=archive(output);dump(output.parent/(commit+"_EXPORT_RECEIPT.json"),receipt)
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--artifact",required=True);parser.add_argument("--model",required=True);parser.add_argument("--training-rows",required=True);args=parser.parse_args()
    result=collect(Path(__file__).resolve().parents[1],args.artifact,args.model,args.training_rows);print(json.dumps(result,indent=2))
    if result["status"]!="complete":raise SystemExit(1)
