"""Source-bound finite trace admission, exact locked selector and native reserves."""
import json
from pathlib import Path
from support import HERE,ROOT,SOURCES,sha,require,write_new,bounds

PROMPT_ID="PUBLIC_SYNTHETIC_TOKEN_1"
CELL_ID=PROMPT_ID+"__baseline"
INPUT_LOCK=sha((HERE/"PUBLIC_INPUT.json").read_bytes())

def exact_input(plan):
    from plan import build_plan
    require(plan==build_plan() and len(plan["prompts"])==1 and not plan["requests"] and len(plan["cells"])==1,"exact public-only one-cell input")
    prompt=plan["prompts"][0]
    require(prompt["input_ids"]==[1] and prompt["attention_mask"]==[1] and prompt["final_input_index"]==0,"public tensor input identity")
    return {k:prompt[k] for k in ("input_ids","attention_mask","final_input_index","input_int64_le_sha256","token_proof_sha256")},plan["cells"][0]

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
