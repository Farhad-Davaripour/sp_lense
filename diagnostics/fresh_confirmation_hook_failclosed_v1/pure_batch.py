"""Six prospective pure case groups. Fixtures are synthetic hook metadata, not model evidence."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import hook_evidence as h
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FIX=HERE/"fixtures"
LABELS=tuple("check_%03d"%i for i in range(109))
SCHEDULE=("F001","F002","F003")
SOURCE={"passed":True,"before_materialization":True,"installed_sources_sha256":{"fixture_source.py":"0"*64}}
BEFORE={"modules":{"m":{"identity":7,"callbacks":[]}},"globals":{}}
REFERENCE={"modules":{"m":{"identity":7,"callbacks":[{"key":1,"identity":9}]}},"globals":{}}
SETUP={"changes":h.differences(BEFORE,REFERENCE),"verified_setup_callbacks":[{"fixture_only":True}]}
STR_CALLS=[]
NORMAL_STATUS=None
class PoisonError(Exception):
    def __str__(self): STR_CALLS.append("str");raise AssertionError("arbitrary exception str called")
    def __repr__(self): STR_CALLS.append("repr");raise AssertionError("arbitrary exception repr called")
class FaultIO(h.FileIO):
    def __init__(self,root): super().__init__(root);self.failures={}
    def write(self,name,data,append=False,fault=False):
        if name in self.failures:
            n=self.failures.pop(name)
            super().write(name,data[:n],append,fault)
            raise PoisonError()
        return super().write(name,data,append,fault)
class LostCloseoutIO(FaultIO):
    def write(self,name,data,append=False,fault=False):
        if name=="hook_evidence/index.json":
            super().write(name,data,append,fault)
            raise PoisonError()
        if name=="hook_evidence/fault.json": raise PoisonError()
        return super().write(name,data,append,fault)
def require(ok):
    if not ok: raise AssertionError("fixed fixture assertion")
def blocked(call):
    try: call()
    except h.HookStopped: return
    raise AssertionError("expected finite terminal rejection")
def make(name,io_class=h.FileIO):
    path=FIX/name
    io=io_class(path)
    latch=h.DispatchLatch(SCHEDULE)
    return h.Recorder(path,latch,LABELS,io),path
def admit(rec):
    return rec.admit(copy.deepcopy(BEFORE),copy.deepcopy(REFERENCE),copy.deepcopy(SETUP),
        copy.deepcopy(SOURCE),{"forward_calls":0,"known_setup_verified":True,"fixture_only":True})
def good():
    return {"current":copy.deepcopy(REFERENCE),"matches":True,"changes":[]}
def bad():
    current=copy.deepcopy(REFERENCE);current["modules"]["m"]["callbacks"].append({"key":2,"identity":99})
    return {"current":current,"matches":False,"changes":h.differences(REFERENCE,current)}
def populate(rec):
    admit(rec)
    for cell in SCHEDULE: rec.latch.consume(cell)
    for label in LABELS: require(rec.inspect(good,label))
def assert_terminal(rec):
    primary=rec.latch.primary_code;remaining=rec.latch.remaining
    blocked(rec.latch.require_dispatch)
    blocked(lambda:rec.inspect(good,LABELS[min(rec.checks,108)]))
    blocked(rec.finish)
    blocked(lambda:rec.admit(BEFORE,REFERENCE,SETUP,SOURCE,{"forward_calls":0}))
    require(rec.latch.terminal and rec.latch.primary_code==primary and rec.latch.remaining==remaining)
    require(not rec.status()["permits_pass"])
def group1():
    global NORMAL_STATUS
    rec,path=make("normal_109")
    populate(rec);result=rec.finish()
    NORMAL_STATUS=rec.status()
    require(result["permits_pass"] and h.judge(path,LABELS,SCHEDULE,SOURCE,rec.status())["permits_pass"])
    require(not h.judge(path,LABELS,SCHEDULE,SOURCE,None)["permits_pass"])
    for label in LABELS: require(len(h.encode({"label":label}))<h.CHECK_BYTES)
    pins=json.loads((HERE/"SOURCE_PINS.json").read_bytes())
    guard_pin=next(p for p in pins if p["path"].endswith("guard_candidate.py"))
    tree=ast.parse((ROOT/guard_pin["path"]).read_bytes())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="differences"]
    scope={};exec(compile(ast.Module(body=nodes,type_ignores=[]),"<pinned-pure-differences>","exec"),scope)
    require(scope["differences"](BEFORE,REFERENCE)==h.differences(BEFORE,REFERENCE))
    return {"complete_checks":109,"saved_judge":"COMPLETE","index_bound_bytes":rec.normal_index_bound_bytes,
        "actual_index_bytes":len(rec.io.read("hook_evidence/index.json")),"scientific_model_calls":0}
def group2():
    reserve=109*h.CHECK_BYTES+h.FAULT_RESERVE+h.INDEX_RESERVE
    exact=h.HOOK_CAP-reserve
    require(h.allocation(0,0,exact,109)==h.HOOK_CAP)
    blocked(lambda:h.allocation(0,0,exact+1,109))
    rec,path=make("check_110")
    populate(rec)
    require(rec.checks==109 and not rec.latch.terminal)
    blocked(lambda:rec.inspect(good,"check_109"))
    require(rec.latch.primary_code=="H_CHECK_COUNT")
    assert_terminal(rec)
    return {"exact_fit_accepted":True,"plus_one_rejected":True,"109_complete_before_110":True,
        "110_rejected":True,"saved_judge":h.judge(path,LABELS,SCHEDULE,SOURCE,rec.status())}
def group3():
    rec,path=make("fitting_mismatch")
    admit(rec);rec.latch.consume("F001")
    original=h.zlib.compress;seen=[]
    def verify_latch(raw):
        require(rec.latch.terminal);seen.append(True);return original(raw)
    h.zlib.compress=verify_latch
    try: blocked(lambda:rec.inspect(bad,LABELS[0]))
    finally: h.zlib.compress=original
    require(seen and rec.diagnostic_complete and rec.latch.primary_code=="H_IDENTITY")
    receipt=json.loads(rec.io.read("hook_evidence/fault.json"))
    require(receipt["diagnostic_complete"] and receipt["remaining_ids"]==["F002","F003"])
    mapping={p:{"path":p} for p in rec.io.hook_files()}
    value,_=h._read_value(rec.io,receipt["details_reference"],mapping,"failure_001")
    require(value==h.differences(REFERENCE,bad()["current"]))
    assert_terminal(rec)
    return {"latch_observed_before_codec":True,"lossless_changed_metadata":True,"remaining":["F002","F003"],
        "saved_judge":h.judge(path,LABELS,SCHEDULE,SOURCE,rec.status())}
def group4():
    raw_rec,raw_path=make("raw_overflow")
    huge={"x":"a"*h.RAW_CAP}
    blocked(lambda:raw_rec.admit(huge,huge,{"changes":[]},SOURCE,{"forward_calls":0}))
    require(raw_rec.latch.primary_code=="H_RAW_CAP");del huge
    assert_terminal(raw_rec)
    compressed_rec,compressed_path=make("compressed_overflow")
    old=h.zlib.compress
    h.zlib.compress=lambda data:b"x"*(h.FILE_CAP+1)
    try: blocked(lambda:admit(compressed_rec))
    finally: h.zlib.compress=old
    require(compressed_rec.latch.primary_code=="H_COMPRESSED_CAP")
    assert_terminal(compressed_rec)
    aggregate_rec,aggregate_path=make("aggregate_overflow")
    admit(aggregate_rec)
    aggregate_rec.io.total_bytes=lambda:h.AGGREGATE_CAP
    blocked(lambda:aggregate_rec.inspect(good,LABELS[0]))
    require(aggregate_rec.latch.primary_code=="H_CAPACITY" and aggregate_rec.receipt_failed)
    assert_terminal(aggregate_rec)
    data=random.Random(8073).randbytes(h.CHUNK)
    actual=h.zlib.compress(data)
    require(len(actual)>len(data) and h.zlib.decompress(actual)==data)
    used=h.HOOK_CAP-h.FAULT_RESERVE-h.INDEX_RESERVE-len(data)
    require(h.allocation(used,used,len(data),0)==h.HOOK_CAP)
    blocked(lambda:h.allocation(used,used,len(actual),0))
    return {"raw_cap_rejected":True,"compressed_file_cap_rejected":True,"aggregate_rejected":True,
        "codec_raw_bytes":len(data),"codec_encoded_bytes":len(actual),"expansion_allocation_rejected":True,
        "all_judges_reject":all(not h.judge(p,LABELS,SCHEDULE,SOURCE,r.status())["permits_pass"]
            for r,p in ((raw_rec,raw_path),(compressed_rec,compressed_path),(aggregate_rec,aggregate_path)))}
def group5():
    results=[]
    for name,broken in (("partial_chunk","hook_evidence/setup_before_000.zlib"),
                        ("partial_manifest","hook_evidence/setup_before.json")):
        rec,path=make(name,FaultIO);rec.io.failures[broken]=7
        blocked(lambda:admit(rec))
        require(len(rec.io.read(broken))==7 and rec.latch.primary_code=="H_IO")
        require(isinstance(rec.primary_exception,PoisonError))
        assert_terminal(rec)
        results.append({"case":name,"partial_bytes_preserved":7,"saved_judge":h.judge(path,LABELS,SCHEDULE,SOURCE,rec.status())})
    rec,path=make("fault_receipt_failure",FaultIO);admit(rec)
    rec.io.failures["hook_evidence/fault.json"]=0
    blocked(lambda:rec.inspect(bad,LABELS[0]))
    require(rec.receipt_failed and rec.latch.primary_code=="H_IDENTITY")
    require("H_RECEIPT_IO" in rec.latch.secondary_codes)
    assert_terminal(rec)
    results.append({"case":"fault_receipt_failure","status":rec.status(),"saved_judge":h.judge(path,LABELS,SCHEDULE,SOURCE,rec.status())})
    rec,path=make("closeout_failure",FaultIO);populate(rec)
    rec.io.failures["hook_evidence/index.json"]=7
    blocked(rec.finish)
    require(rec.latch.primary_code=="H_IO" and len(rec.io.read("hook_evidence/index.json"))==7)
    assert_terminal(rec)
    results.append({"case":"closeout_failure","saved_judge":h.judge(path,LABELS,SCHEDULE,SOURCE,rec.status())})
    rec,path=make("fully_written_index_lost_fault_receipt",LostCloseoutIO);populate(rec)
    blocked(rec.finish)
    require(json.loads(rec.io.read("hook_evidence/index.json"))["complete"] is True)
    require("hook_evidence/fault.json" not in rec.io.hook_files() and rec.receipt_failed)
    require(rec.latch.primary_code=="H_IO" and rec.completed_index_sha is None)
    assert_terminal(rec)
    verdict=h.judge(path,LABELS,SCHEDULE,SOURCE,rec.status())
    require(not verdict["permits_pass"])
    results.append({"case":"fully_written_index_lost_fault_receipt","status":rec.status(),
        "complete_index_not_authoritative":True,"saved_judge":verdict})
    require(not STR_CALLS)
    return {"fault_cases":results,"arbitrary_exception_str_repr_calls":0}
def group6():
    normal=FIX/"normal_109";results=[]
    for name in ("forged_completeness","missing_artifact","changed_artifact","untracked_bytes",
                 "incomplete_index","tracked_but_unreferenced"):
        path=FIX/name;shutil.copytree(normal,path)
        index_path=path/"hook_evidence/index.json"
        index=json.loads(index_path.read_bytes())
        if name=="forged_completeness":
            checks_path=path/"hook_evidence/checks.jsonl"
            lines=checks_path.read_bytes().splitlines(keepends=True)
            checks_path.write_bytes(b"".join(lines[:-1]))
            for record in index["files"]:
                if record["path"]=="hook_evidence/checks.jsonl":
                    data=checks_path.read_bytes();record.update(bytes=len(data),sha256=h.sha(data))
            index_path.write_bytes(h.encode(index))
        elif name=="missing_artifact":
            (path/"hook_evidence/reference_000.zlib").unlink()
        elif name=="changed_artifact":
            f=path/"hook_evidence/reference_000.zlib";data=f.read_bytes();f.write_bytes(data[:-1]+bytes([data[-1]^1]))
        elif name in ("untracked_bytes","tracked_but_unreferenced"):
            f=path/"hook_evidence/extra.bin";f.write_bytes(b"EXTRA")
            if name=="tracked_but_unreferenced":
                index["files"].append({"path":"hook_evidence/extra.bin","bytes":5,"sha256":h.sha(b"EXTRA")})
                index_path.write_bytes(h.encode(index))
        else:
            index["files"]=index["files"][:-1];index_path.write_bytes(h.encode(index))
        # Even a self-consistent claimed controller index hash cannot replace byte reconstruction.
        claimed=copy.deepcopy(NORMAL_STATUS);claimed["index_sha256"]=h.sha(index_path.read_bytes())
        verdict=h.judge(path,LABELS,SCHEDULE,SOURCE,claimed)
        require(not verdict["permits_pass"])
        results.append({"case":name,"saved_judge":verdict})
    return {"rejections":results}

def main():
    started=time.monotonic();cases=[];failure=None
    freeze=json.loads((HERE/"SOURCE_FREEZE.json").read_bytes())
    for record in freeze["files"]:
        raw=(HERE/record["path"]).read_bytes()
        require(len(raw)==record["bytes"] and h.sha(raw)==record["sha256"])
    pins=json.loads((HERE/"SOURCE_PINS.json").read_bytes())
    for p in pins:
        raw=(ROOT/p["path"]).read_bytes()
        require(len(raw)==p["bytes"] and h.sha(raw)==p["sha256"] and raw==
            subprocess.check_output(["git","show",p["commit"]+":"+p["path"]],cwd=ROOT))
    with (HERE/"BATCH_STARTED.json").open("xb") as out:out.write(h.encode({"groups":6,"seconds_limit":60,"model_calls":0,"tokenizer_calls":0}))
    FIX.mkdir(exist_ok=False)
    group_name="not_started"
    try:
        for group_name,test in (("normal_109_reconstruction",group1),("exact_fit_count_boundaries",group2),
            ("fitting_terminal_mismatch",group3),("raw_compressed_aggregate_expansion",group4),
            ("partial_fault_closeout_failure",group5),("independent_saved_rejections",group6)):
            detail=test();cases.append({"group":group_name,"status":"PASS","details":detail})
    except BaseException as error:
        failure={"group":group_name,"code":"FINITE_TEST_FAILURE","exception_text_serialized":False,
            "hook_code":error.code if type(error) is h.HookStopped else None}
    elapsed=time.monotonic()-started
    fixture_files=[p for p in FIX.rglob("*") if p.is_file()]
    fixture_bytes=sum(p.stat().st_size for p in fixture_files)
    if elapsed>60 or fixture_bytes>32*h.MIB or any(p.stat().st_size>h.FILE_CAP for p in fixture_files):
        failure={"group":group_name,"code":"BATCH_RESOURCE_LIMIT","prior":failure}
    receipt={"status":"PASS" if failure is None and len(cases)==6 else "INCONCLUSIVE",
        "groups":cases,"failure":failure,"elapsed_seconds":elapsed,"seconds_limit":60,
        "fixture_bytes":fixture_bytes,"fixture_files":len(fixture_files),
        "arbitrary_exception_str_repr_calls":len(STR_CALLS),"model_calls":0,"tokenizer_calls":0,
        "real_run_authorized":False,"source_freeze_sha256":h.sha((HERE/"SOURCE_FREEZE.json").read_bytes())}
    with (HERE/"TEST_RECEIPT.json").open("xb") as out:out.write(h.encode(receipt))
    print(json.dumps(receipt,sort_keys=True))
    return 0 if receipt["status"]=="PASS" else 1
if __name__=="__main__":
    raise SystemExit(main())
