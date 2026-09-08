"""Authenticate only the prospectively selected public development prefix; stdlib only."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import zlib

ROOT=Path(__file__).resolve().parents[2]
COMMIT="ac756279e595772733c20c7884bea60c61108a2f"
PREFIX="diagnostics/semantic_editor_f03_v2_first_C_v2/real_attempt"
INVENTORY_SHA="c45fcf4abe3ec0bd1e1ba066441e1a8c8a26ef7e272ab6738920ec34de9a6f25"
METHOD_SHA="59ae8c47bbc96668e23769851a62e8f04fb0677e4f5ccc7faaf5ad5aaf5e7414"
WEIGHT_SHA="6a671f0ae00398453e5b453d13b4ebe9de861eb3483f0e500e07bc9e456d06be"
VOCAB=248320
EPS=1e-6
CONDITIONS=("baseline","entry","gradient_1","step_1","endpoint")

def need(ok,code):
    if not ok:raise ValueError(code)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def git_bytes(path):
    return subprocess.run(["git","show",f"{COMMIT}:{path}"],cwd=ROOT,check=True,capture_output=True).stdout
def floats(raw,count):
    need(len(raw)==count*4,"RAW_LENGTH")
    values=struct.unpack(f"<{count}f",raw)
    need(all(math.isfinite(x) for x in values),"RAW_NONFINITE")
    return values

def authenticate():
    inventory_raw=git_bytes(PREFIX+"/FINAL_INVENTORY.json")
    need(sha(inventory_raw)==INVENTORY_SHA,"INVENTORY_PIN")
    need((ROOT/PREFIX/"FINAL_INVENTORY.json").read_bytes()==inventory_raw,"INVENTORY_WORKING")
    inventory=json.loads(inventory_raw);blobs={}
    for item in inventory["files"]:
        name=item["path"]
        need(not Path(name).is_absolute() and ".." not in Path(name).parts,"REFERENCE_PATH")
        raw=git_bytes(PREFIX+"/"+name)
        need(len(raw)==item["bytes"] and sha(raw)==item["sha256"],"REFERENCE_GIT_BYTES")
        need((ROOT/PREFIX/name).read_bytes()==raw,"REFERENCE_WORKING_BYTES")
        blobs[name]=raw
    load=lambda name:json.loads(blobs[name])
    plan,runtime,fitted=map(load,("plan.json","runtime.json","fitted_parameters.json"))
    need(sha(blobs["fitted_parameters.json"])==plan["gate"]["parameter_sha256"],"FITTED_PIN")
    need(fitted["method_sha256"]==METHOD_SHA and fitted["threshold"]==0,"FIXED_GATE")
    need(plan["gate"]["runtime_compatibility"]["weight_sha256"]==WEIGHT_SHA,"FIXED_WEIGHT")
    need(runtime["model_id"]=="Qwen/Qwen3.5-0.8B" and runtime["dtype"]=="float32"
         and runtime["device"]=="cpu" and runtime["d_model"]==1024 and runtime["lens"] is None,"REFERENCE_RUNTIME")
    need(len(runtime["boundaries"])==1,"ONE_INPUT")
    boundary=runtime["boundaries"][0]
    ids=boundary["full_token_ids"];mask=boundary["attention_mask"]
    need(len(ids)==137 and mask==[1]*137 and boundary["final_input_index"]==136
         and boundary["final_input_mask"]==[0]*136+[1],"FIXED_BOUNDARY")
    need(sha(struct.pack("<137q",*ids))==boundary["full_input_int64_le_sha256"],"IDS_HASH")
    rows=[load(f"rows/{i:03d}.json") for i in range(1,6)]
    need(tuple(r["condition"] for r in rows)==CONDITIONS,"EXACT_PREFIX")
    need(all(r["prompt_id"]=="f03_v2_STOP_then_KEEP" and r["prompt_length"]==137 for r in rows),"ROW_INPUT")
    events=[json.loads(line) for line in blobs["forward_events.jsonl"].splitlines()]
    need([e["cell"]["cell_id"] for e in events if e["event"]=="attempt_completed"]==[r["cell_id"] for r in rows],"FIVE_COMPLETED")
    derivatives=[json.loads(line) for line in blobs["derivative_events.jsonl"].splitlines()]
    need(len(derivatives)==2 and all(x["cell"]["cell_id"]==rows[2]["cell_id"] for x in derivatives)
         and [x["event"] for x in derivatives]==["attempt_started","attempt_completed"],"ONE_DERIVATIVE")
    need(load("unrun.json")==[],"COMPLETE_REFERENCE_PREFIX")
    logits=[];raw_receipts=[]
    for row in rows:
        raw=zlib.decompress(blobs[row["logits_file"]]);need(sha(raw)==row["logits_sha256"],"DECOMPRESSED_HASH")
        values=floats(raw,VOCAB);logits.append(raw)
        need(row["logit_count"]==VOCAB and len(row["h"])==1024 and len(row["cumulative_offset"])==1024,"ROW_DIMENSIONS")
        need(row["unselected_max_difference"]==0 and row["maximum_offset_error"]<=EPS,"REFERENCE_POSITION_IDENTITY")
        raw_receipts.append({"file":row["logits_file"],"raw_bytes":len(raw),"raw_sha256":sha(raw),"compressed_sha256":sha(blobs[row["logits_file"]])})
    need(len(rows[2]["gradient"])==1024 and all(math.isfinite(x) for x in rows[2]["gradient"]),"SAVED_GRADIENT")
    routes=[json.loads(x) for x in blobs["routing_events.jsonl"].splitlines()]
    need(len(routes)==2 and [x["cell_id"] for x in routes]==[r["cell_id"] for r in rows[:2]],"ROUTE_PREFIX")
    setup_manifest=load("hook_evidence/setup_before.json");parts=[]
    for chunk in setup_manifest["chunks"]:
        compressed=blobs[chunk["path"]];part=zlib.decompress(compressed)
        need(len(compressed)==chunk["compressed_bytes"] and sha(compressed)==chunk["compressed_sha256"]
             and len(part)==chunk["raw_bytes"] and sha(part)==chunk["raw_sha256"],"SETUP_CHUNK")
        parts.append(part)
    setup_raw=b"".join(parts)
    need(setup_manifest["complete"] is True and len(setup_raw)==setup_manifest["raw_bytes"]
         and sha(setup_raw)==setup_manifest["raw_sha256"],"SETUP_RECONSTRUCTION")
    setup=json.loads(setup_raw)
    graph={name:{"type":item["type"],"children":[x[0] for x in item["children"]]}
           for name,item in setup["modules"].items() if name in ("","vision_encoder","vision_encoder._original_component","vision_projector","vision_projector._original_component")}
    return {"plan":plan,"runtime":runtime,"boundary":boundary,"rows":rows,"logits":logits,"routes":routes,"fitted":fitted,
        "receipt":{"reference_commit":COMMIT,"inventory_sha256":INVENTORY_SHA,"files_verified":len(blobs),
        "bytes_verified":sum(map(len,blobs.values())),"input_int64_sha256":boundary["full_input_int64_le_sha256"],
        "fitted_sha256":sha(blobs["fitted_parameters.json"]),"runtime_sha256":sha(blobs["runtime.json"]),
        "rows":[{"path":f"rows/{i:03d}.json","sha256":sha(blobs[f"rows/{i:03d}.json"])} for i in range(1,6)],
        "logits":raw_receipts,"gradient_f32_sha256":sha(struct.pack("<1024f",*rows[2]["gradient"])),
        "setup_raw_sha256":sha(setup_raw),"setup_graph_evidence":graph,
        "scope":"fixed exposed development replay, not new optimization or held-out evaluation"}}

def gate(fitted):
    path=ROOT/"src/sp_lense/conditional_gate_models.py"
    need(sha(path.read_bytes())==METHOD_SHA,"GATE_SOURCE")
    spec=importlib.util.spec_from_file_location("native_replay_frozen_centroid",path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    obj=module.CenteredCosineCentroidModel()
    need(set(fitted["parameters"])=={"direction","grand_mean","positive_centroid","negative_centroid"},"FITTED_FIELDS")
    for name,values in fitted["parameters"].items():
        need(len(values)==1024 and all(math.isfinite(x) for x in values),"FITTED_VALUES")
        setattr(obj,name,tuple(values))
    obj._fitted=True
    return obj
