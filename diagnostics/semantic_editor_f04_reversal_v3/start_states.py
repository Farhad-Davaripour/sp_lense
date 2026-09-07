"""Authenticated constructed starts; stdlib only, no model/tokenizer/gate scoring."""
import json,math,sys
from pathlib import Path
from core import HERE,ROOT,read,require,sha,git
sys.path.insert(0,str(ROOT))
from scripts.verify_local_controllability import read_logits,f32

PARENT="diagnostics/semantic_editor_f04_assay_v3"
COMMIT="1b83722c1d959b1f576282b8183097ebf6137300"
INVENTORY="e6969acf0e4342df29433003280bae6417001d9b08baa19a89cc344280a4266c"
SOURCE_FREEZE="3939dea8b44e2bca2b961182edbf03d7d26dc993fbf8273c7eb3af96f954b60f"
SELECTION=(
    {"prompt_id":"f04_KEEP_then_STOP","request_id":"f04_KEEP_then_STOP__reverse__P","policy":"P","sign":1,
     "supplied_target_word":"KEEP","target_display_position":1,"source_request_id":"f04_KEEP_then_STOP__C",
     "entry":5,"gradients":[6,8],"steps":[7,9],"endpoint":10},
    {"prompt_id":"f04_STOP_then_KEEP","request_id":"f04_STOP_then_KEEP__reverse__C","policy":"C","sign":-1,
     "supplied_target_word":"STOP","target_display_position":1,"source_request_id":"f04_STOP_then_KEEP__P",
     "entry":11,"gradients":[12],"steps":[13],"endpoint":14},
)
def norm(v):return math.sqrt(math.fsum(float(x)**2 for x in v))
def carry_geometry(h0,old_path,new_path,net,step):
    hn=norm(h0)
    require(hn>0 and all(math.isfinite(x) and x>=0 for x in (old_path,new_path,net,step)),"finite carried geometry")
    require(step<=.05*hn+1e-6 and old_path+new_path<=.20*hn+1e-6 and net<=min(old_path+new_path,.20*hn)+1e-6,"original plus reversal path/net allowance")
def check_h0(fresh,original):
    require(len(fresh)==len(original)==1024 and all(math.isfinite(x) for seq in (fresh,original) for x in seq),"finite 1024 h0")
    require(max(abs(x-y) for x,y in zip(fresh,original))<=1e-6,"archived original h0 mismatch; never recalibrate")
def check_replay(fresh,logits,original,old_logits):
    check_h0(fresh["h0"],original["h0"])
    require(fresh["cumulative_offset"]==original["cumulative_offset"] and any(fresh["cumulative_offset"]),"exact nonzero saved offset; no zero-reset shortcut")
    require(fresh["h"]==original["h"] and len(logits)==len(old_logits)==248320,"starting/final hidden exact and full vocabulary")
    require(all(math.isfinite(x) for seq in (logits,old_logits) for x in seq) and max(abs(x-y) for x,y in zip(logits,old_logits))<=2e-5,"independent archived endpoint replay mismatch")
def authenticate_parent():
    raw=git("show",COMMIT+":"+PARENT+"/real_attempt/FINAL_INVENTORY.json")
    require(sha(raw)==INVENTORY and raw==(ROOT/PARENT/"real_attempt/FINAL_INVENTORY.json").read_bytes(),"committed immutable 84-entry result inventory")
    entries={x["path"]:x for x in json.loads(raw)["files"]};require(len(entries)==84,"exact completed source inventory")
    freeze=(ROOT/PARENT/"freeze.json").read_bytes()
    require(sha(freeze)==SOURCE_FREEZE,"original source freeze")
    for name,digest in json.loads(freeze)["source_sha256"].items():
        require(sha((ROOT/PARENT/name).read_bytes())==digest,"unchanged historical source "+name)
    for name in ("inputs.json","input_lock.json","tokens_01.json","tokens_02.json","fitted_parameters.json","source_bindings.json"):
        require((HERE/name).read_bytes()==(ROOT/PARENT/name).read_bytes(),"exact inherited input/runtime "+name)
    return entries
def source_bytes(name,entries):
    raw=(ROOT/PARENT/"real_attempt"/name).read_bytes();item=entries[name]
    require(len(raw)==item["bytes"] and sha(raw)==item["sha256"],"authenticated source artifact "+name)
    return raw
def load_starts(plan):
    if plan["execution_mode"]=="SYNTHETIC_ONLY":
        require(plan["fixture_scope"]=="SYNTHETIC_CONSTRUCTED_RECOVERY_ONLY","explicit fake starts")
        result={}
        for rid,item in plan["synthetic_starts"].items():
            root=Path(item["root"])
            require(root.is_relative_to(HERE/"synthetic"),"fake starting artifact path")
            result[rid]={**item,"logits":list(read_logits(root,item["row"]))}
        return result
    entries=authenticate_parent();output=ROOT/PARENT/"real_attempt"
    final=json.loads(source_bytes("final_closeout.json",entries));judge=json.loads(source_bytes("judge_results.json",entries))
    require(final["classification"]==judge["classification"]=="PASS","source result remains independently passing")
    result={}
    from saved_judge import verify_update,accepts
    for spec in SELECTION:
        def row(index):return json.loads(source_bytes(f"rows/{index:03d}.json",entries))
        current=row(spec["entry"]);offset=[0.]*1024;path=0.;first=previous=None
        require(current["prompt_id"]==spec["prompt_id"] and not any(current["cumulative_offset"]),"source cold entry")
        old_spec={"sign":-spec["sign"]}
        for gi,si in zip(spec["gradients"],spec["steps"],strict=True):
            gradient,step=row(gi),row(si)
            offset,path=verify_update(gradient,step,current,previous,first,offset,path,old_spec)
            previous=gradient["gradient"];first=previous if first is None else first;current=step
        endpoint=row(spec["endpoint"])
        require(current["request_id"]==endpoint["request_id"]==spec["source_request_id"] and accepts(current,old_spec) and accepts(endpoint,old_spec),"exact accepted displayed-second source endpoints")
        for item in (current,endpoint):source_bytes(item["logits_file"],entries)
        logits=read_logits(output,current);replayed=read_logits(output,endpoint)
        check_replay(endpoint,list(replayed),current,list(logits))
        require(path==current["path_norm"] and path<=.20*norm(current["h0"])+1e-6,"inherited actual path not renewed")
        result[spec["request_id"]]={"row":current,"logits":list(logits),"path":path,
            "source_endpoint_cell_id":endpoint["cell_id"],"source_inventory_sha256":INVENTORY}
    return result
