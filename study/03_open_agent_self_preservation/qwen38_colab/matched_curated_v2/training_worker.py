"""One curated-concept B fit from H2; no Jev/service calls or recurrent eval."""
import argparse
import gc
import hashlib
import json
import random
import signal
import time
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel, get_peft_model_state_dict
from transformers import get_linear_schedule_with_warmup
import model_ops
import audit
import batching
from paired_batching import paired_losses
from prepare_training import paired_token_audit, digest, RAW_FILE_SHA256

SNAPSHOTS=(0,7,14,28,56)
EXPECTED_H2_STATE="42185e8399f610adf84baa6ae55c419d9d1e5508eb2d738399f6b0565167da4b"
EXPECTED_BASE="ec43e239f6ffad79d854149f9d7115aeda5292beddd411dd763cac8f9ddcd755"
EXPECTED_TEMPLATE="c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
H2_WEIGHTS="0782b3f3ea9b3e67460b7bc981e21526bb6819d3ddb9623def4fbc2218002b30"
H2_CONFIG="3df028243d1d0f6643b906143a4c1ad4e8393b5e947d8b52c03f0893601b3528"
STOP=False


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes((json.dumps(value,indent=2,allow_nan=False)+"\n").encode("utf-8"))


def state_sha(model):
    h=hashlib.sha256()
    for name,value in sorted(get_peft_model_state_dict(model).items()):
        h.update(name.encode())
        h.update(memoryview(value.detach().cpu().contiguous().view(torch.uint8).numpy()))
    return h.hexdigest()


def snapshot(model,root,update):
    destination=root/"checkpoints"/("update_"+str(update))
    destination.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(destination,safe_serialization=True)
    receipt={"update":update,"adapter_weights_sha256":audit.sha(destination/"adapter_model.safetensors"),
             "adapter_config_sha256":audit.sha(destination/"adapter_config.json"),
             "behavior_queries_at_snapshot":0,"selected_as_primary":update==56}
    save(root/"reports"/("CHECKPOINT_"+str(update)+".json"),receipt)
    return receipt


def verify(root):
    frozen=json.loads((root/"source/FREEZE.json").read_text(encoding="utf-8"))
    for relative,expected in frozen["sha256"].items():
        if audit.sha(root/"source"/relative)!=expected:
            raise RuntimeError("Source/cache freeze differs:"+relative)
    cfg=json.loads((root/"config.json").read_text(encoding="utf-8"))
    if cfg.get("training_recipe") not in (None,"matched_curated_B_v2","matched_curated_A_fresh_v2"):
        raise RuntimeError("Unexpected training recipe")
    return cfg


def main(root,model_path,max_seconds):
    deadline=time.monotonic()+max_seconds
    cfg=verify(root)
    data=Path(cfg.get("training_data_dir",str(root/"data")))
    plain_raw=(data/"plain_train.json").read_bytes()
    if hashlib.sha256(plain_raw).hexdigest()!=RAW_FILE_SHA256:
        raise RuntimeError("Original primary training file differs")
    plain=json.loads(plain_raw)
    concept_rows=json.loads((data/"concept_train.json").read_text(encoding="utf-8"))
    mode=cfg.get("training_mode","curated_B")
    if mode not in ("curated_B","plain_ref"):
        raise RuntimeError("Unexpected admitted fit mode")
    if mode=="plain_ref" and cfg.get("fallback_plain_admitted") is not True:
        raise RuntimeError("Fresh plain fallback lacks root budget admission")
    recipe="matched_curated_B_v2" if mode=="curated_B" else "matched_curated_A_fresh_v2"
    rows=concept_rows if mode=="curated_B" else plain
    prep=json.loads((data/"PREPARATION_AUDIT.json").read_text(encoding="utf-8"))
    if not prep["passed"] or len(concept_rows)!=112 or len(plain)!=112 or prep["primary_scope_register_repair"]:
        raise RuntimeError("Matched preparation differs")
    if digest(concept_rows)!=prep["concept_rows_canonical_sha256"]:
        raise RuntimeError("Curated training rows differ from preparation freeze")
    if (prep.get("training_service_calls")!=0 or prep.get("unique_curated_prefixes_used")!=104
        or prep.get("runtime_curated_fallback") is not False
        or prep.get("visible_prefix_evidence_required") is not True
        or prep.get("targets_categories_tools_metadata_unchanged") is not True):
        raise RuntimeError("Curated visible-prefix preparation provenance differs")
    for a,b in zip(plain,concept_rows):
        if a["targets"]!=b["targets"] or {k:v for k,v in a.items() if k!="messages"}!={k:v for k,v in b.items() if k!="messages"}:
            raise RuntimeError("Fixed targets/categories/tools changed")
        if b["messages"][:-1]!=a["messages"] or len(b["messages"])!=len(a["messages"])+1:
            raise RuntimeError("Raw histories were repaired or misplaced")
    info=cfg["H2"]
    if info["weights"]!=H2_WEIGHTS or info["config"]!=H2_CONFIG:
        raise RuntimeError("H2 input identity differs")
    audit.verify_adapter(info["path"],H2_WEIGHTS,H2_CONFIG)
    random.seed(941);np.random.seed(941);torch.manual_seed(941);torch.cuda.manual_seed_all(941)
    model=base=tokenizer=optimizer=scheduler=prepared=None
    try:
        tokenizer,base=model_ops.load_model(model_path)
        model=PeftModel.from_pretrained(base,info["path"],is_trainable=True,local_files_only=True)
        if any("lora_" not in n for n,p in model.named_parameters() if p.requires_grad):
            raise RuntimeError("Base became trainable")
        config=model.peft_config["default"]
        if config.r!=16 or config.lora_alpha!=32 or config.lora_dropout!=0:
            raise RuntimeError("Frozen rank/alpha/dropout differs")
        if config.use_rslora or config.use_dora or config.bias!="none" or config.rank_pattern or config.alpha_pattern:
            raise RuntimeError("Unsupported changed adapter recipe")
        identity=audit.runtime(model,tokenizer,root,"B")
        if identity["packed_base_parameter_hash"]!=EXPECTED_BASE or identity["template_sha256"]!=EXPECTED_TEMPLATE:
            raise RuntimeError("Pinned base/tokenizer differs")
        initial=state_sha(model)
        if initial!=EXPECTED_H2_STATE:raise RuntimeError("Starting H2 tensor state differs")
        token_audit=paired_token_audit(tokenizer,plain,concept_rows,model_ops,audit)
        save(root/"training/receipts/PAIRED_TOKEN_AUDIT.json",token_audit)
        if "paired_lengths" in cfg and cfg["paired_lengths"]!=token_audit["paired_lengths"]:
            raise RuntimeError("Predeclared paired lengths differ")
        # Historical A reuse is conditional on root's separately frozen gate,
        # never inferred from labels/padding assertions in this trainer.
        save(root/"training/receipts/START.json",{"H2_weights":H2_WEIGHTS,"H2_config":H2_CONFIG,
            "canonical_state":initial,"base":EXPECTED_BASE,"template":EXPECTED_TEMPLATE,
            "rows":112,"targets_unchanged":True,"primary_scope_register_repair":False,
            "fresh_optimizer":True,"service_calls":0,"admitted_training_mode":mode,
            "training_concepts":"Curated visible-prefix meanings; no Jev training predictions",
            "training_service_calls":0,"runtime_curated_fallback":False})
        before=model_ops.base_hash(model)
        saved=[snapshot(model,root,0)]
        prepared=[]
        for row,length in zip(rows,token_audit["paired_lengths"]):
            t=audit.row_tokens(tokenizer,row,length)
            prepared.append((torch.tensor([t["x"]],device="cuda"),torch.tensor([t["labels"]],device="cuda")))
        random.seed(941);np.random.seed(941);torch.manual_seed(941);torch.cuda.manual_seed_all(941)
        batching.batch_losses=paired_losses
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-5,weight_decay=.01)
        scheduler=get_linear_schedule_with_warmup(optimizer,4,56)
        update=0;losses=[];began=time.monotonic()
        for epoch in range(2):
            order=list(range(112));random.Random(944+epoch).shuffle(order)
            for cursor in range(0,112,4):
                if STOP or time.monotonic()>=deadline:
                    model_ops.checkpoint(model,optimizer,scheduler,root/"checkpoints"/("stopped_"+str(update)),
                                         {"update":update,"resume_validated":False})
                    raise TimeoutError("Stopped at update boundary; partial checkpoint not resumed")
                indices=order[cursor:cursor+4]
                values=batching.optimize_batched(model,optimizer,scheduler,[prepared[i] for i in indices],1,
                                                 tokenizer.pad_token_id or tokenizer.eos_token_id)
                update+=1
                losses.extend({"slot":i,"id":rows[i]["id"],"epoch":epoch+1,"loss":v} for i,v in zip(indices,values))
                live={"stage":recipe+"_training","updates":update,"total_updates":56,
                      "presentations":update*4,"elapsed_seconds":time.monotonic()-began}
                save(root/"reports/LIVE_TRAINING.json",live)
                if update%7==0:print(json.dumps(live),flush=True)
                if update in SNAPSHOTS:saved.append(snapshot(model,root,update))
        after=model_ops.base_hash(model)
        if after!=before or after!=EXPECTED_BASE:raise RuntimeError("Base changed during fit")
        model.gradient_checkpointing_disable()
        receipt={"completed":True,"recipe":recipe,"training_mode":mode,"updates":56,"presentations":224,
                 "starting_adapter_state_sha256":initial,"base_before":before,"base_after":after,
                 "rank":16,"alpha":32,"dropout":0,"lr":5e-5,"weight_decay":.01,"clip":1,
                 "seed":941,"shuffle_seeds":[944,945],"microbatch":1,"effective_batch":4,"warmup":4,
                 "snapshots":saved,"primary_update":56,"snapshot_behavior_queries":0,
                 "losses":losses,"seconds_training_including_saves":time.monotonic()-began,
                 "service_calls":0,"selected_A_unchanged":True,
                 "training_concepts":"Curated visible-prefix meanings; no Jev training predictions",
                 "training_service_calls":0,"runtime_curated_fallback":False,
                 "final_adapter_path":str(root/"checkpoints/update_56"),
                 "B_final_adapter_path":str(root/"checkpoints/update_56") if mode=="curated_B" else None,
                 "A_fresh_adapter_path":str(root/"checkpoints/update_56") if mode=="plain_ref" else None,
                 "evaluation":"Separate frozen eval_worker; no checkpoint selected by outcomes"}
        save(root/"training/receipts/FIT.json",receipt)
        save(root/"reports/TRAINING_RESULT.json",receipt)
        print(json.dumps({"stage":recipe+"_fit_complete","updates":56,
                          "adapter":receipt["final_adapter_path"],"service_calls":0}),flush=True)
        return receipt
    finally:
        model=base=tokenizer=optimizer=scheduler=prepared=None
        gc.collect();torch.cuda.empty_cache()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--model",required=True)
    p.add_argument("--max-seconds",type=float,required=True);args=p.parse_args()
    def stopping(_signal,_frame):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stopping)
    try:main(Path(args.root),args.model,args.max_seconds)
    except Exception as error:
        save(Path(args.root)/"reports/TRAINING_FAILURE.json",
             {"completed":False,"error_type":type(error).__name__,"error":str(error),
              "service_calls":0,"partial_checkpoint_resume_validated":False})
        raise
