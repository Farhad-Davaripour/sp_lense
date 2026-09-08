"""Source-bound finite trace admission, exact locked selector and native reserves."""
import json
from pathlib import Path
from support import HERE,ROOT,SOURCES,sha,require,write_new,bounds

PROMPT_ID="N01__v1__self_shutdown__KEEP_then_STOP"
CELL_ID=PROMPT_ID+"__baseline"
INPUT_LOCK="d008fb7c53b974694e76afa9ef7db3f4f2d4443eed8a2c83ba06ea6e83987b22"

def exact_input(plan):
    require(len(plan["prompts"])==24 and len(plan["requests"])==48 and len(plan["cells"])==180,"unchanged complete input denominator")
    prompt=plan["prompts"][0];cell=plan["cells"][0]
    require(prompt["prompt_id"]==PROMPT_ID and cell=={"cell_id":CELL_ID,"prompt_id":PROMPT_ID,"request_id":None,"phase":"baseline"},"only prospectively fixed first baseline")
    lock=json.loads((HERE/"BINDINGS.json").read_bytes())
    require(lock["input_lock_sha256"]==INPUT_LOCK,"unchanged exact token lock")
    require(lock["diagnostic_scope"]["prompt_id"]==PROMPT_ID and lock["diagnostic_scope"]["cell_id"]==CELL_ID
        and prompt["token_proof_sha256"]==lock["diagnostic_scope"]["token_record_sha256"],"exact prospective first proof identity")
    require(len(prompt["input_ids"])<=160 and prompt["attention_mask"]==[1]*len(prompt["input_ids"]),"complete fixed input IDs and mask")
    # Only these tensor inputs enter the adapter; all policy/gold/category data is excluded.
    return {k:prompt[k] for k in ("input_ids","attention_mask","final_input_index","input_int64_le_sha256","token_proof_sha256")},cell

def allowlist():
    frozen=json.loads((HERE/"SOURCE_FREEZE.json").read_bytes());spec=json.loads((HERE/"RUNTIME_SPEC.json").read_bytes())
    entries={}
    for name,digest in frozen["source_sha256"].items():
        if name.endswith(".py"):entries[str((HERE/name).absolute()).casefold()]={"source":"candidate/"+name,"sha256":digest}
    for key,record in SOURCES.pins.items():
        path=key.split(":",1)[1]
        if path.endswith(".py"):
            source="repository/"+path
            prior=entries.get(str((ROOT/path).absolute()).casefold())
            require(prior is None or prior["sha256"]==record["sha256"],"same path cannot name distinct compiled source")
            entries[str((ROOT/path).absolute()).casefold()]={"source":source,"sha256":record["sha256"]}
    for ordinal,(path,digest) in enumerate(sorted(spec["installed_sources_sha256"].items())):
        entries[str(Path(path).absolute()).casefold()]={"source":"installed/%03d/"%ordinal+Path(path).name,"sha256":digest}
    for record in json.loads((HERE/"TRACE_SOURCE_ALLOWLIST.json").read_bytes())["files"]:
        raw=Path(record["path"]).read_bytes()
        require(len(raw)==record["bytes"] and sha(raw)==record["sha256"],"prospectively frozen observational source locations")
        entries[str(Path(record["path"]).absolute()).casefold()]={"source":record["source"],"sha256":record["sha256"]}
    require(len(entries)<=512,"finite traceback source allowlist")
    return entries

def reserve():
    cap={"diagnostic":2*1024**2,"terminal":9*1024**2//4,"index":5*1024**2,"other_closeout":3*1024**2//4,
        "trace_receipt":65536}
    require(cap["terminal"]+cap["index"]+cap["other_closeout"]==8*1024**2,"unchanged complete closeout allocation")
    require(bounds()["evidence_bytes"]+cap["diagnostic"]+8*1024**2<=288*1024**2,"admit complete native reserves before load")
    write_new("DIAGNOSTIC_RESERVATION.json",{"caps":cap,"trace_inside_other_closeout":True,"before_load":True,
        "fixed_cell_id":CELL_ID,"input_lock_sha256":INPUT_LOCK,"load_max":1,"forward_max":1,"derivative_max":0,"tokenizer_max":0},critical=True)
    return cap

def publish_trace(raw):
    require(type(raw) is bytes and len(raw)<=65536,"additional diagnostic receipt 64KiB maximum")
    return write_new("FIRST_FORWARD_TRACE.json",raw,raw=True,critical=True)["sha256"]
