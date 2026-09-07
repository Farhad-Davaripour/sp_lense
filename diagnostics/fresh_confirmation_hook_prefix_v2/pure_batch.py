"""Three fixed pure groups; no model/backend/tokenizer imports."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import hook_prefix as h
b=h.b
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FIX=HERE/"fixtures"
LABELS=tuple(["REQUEST-entry:R01","ON-entry:R01","REQUEST-exit:R01"]+
             ["UNRUN-check-%03d"%i for i in range(105)]+["matrix-finally"])
SCHEDULE=tuple("F%03d"%i for i in range(180))
SOURCE={"passed":True,"before_materialization":True,"installed_sources_sha256":{"fixture.py":"0"*64}}
REFERENCE={"modules":{"m":{"identity":7,"callbacks":[]}},"globals":{}}
FINDING={"kind":"finite_eligibility","prompt_id":"synthetic-only","finite":True}
FINDING_SHA=b.sha(b.encode(FINDING))
STR_CALLS=[]
NORMAL_PREFIX=None
class PoisonError(Exception):
    def __str__(self): STR_CALLS.append("str");raise AssertionError("forbidden exception text")
    def __repr__(self): STR_CALLS.append("repr");raise AssertionError("forbidden exception text")
class FaultIO(h.FileIO):
    mode=None
    def write(self,name,data,append=False,fault=False):
        if name=="hook_evidence/fault.json" and self.mode in ("index","stop"):
            raise PoisonError()
        if self.mode=="index" and name=="hook_evidence/prefix_index.json":
            super().write(name,data,append,fault);raise PoisonError()
        if self.mode=="stop" and name=="hook_evidence/scientific_stop.json":
            super().write(name,data[:7],append,fault);raise PoisonError()
        return super().write(name,data,append,fault)
def require(ok):
    if not ok: raise AssertionError("fixed pure assertion")
def blocked(fn):
    try:fn()
    except h.HookStopped:return
    raise AssertionError("expected rejection")
def good():
    return {"current":copy.deepcopy(REFERENCE),"matches":True,"changes":[]}
def bad():
    result=good();result["current"]["modules"]["m"]["callbacks"]=[{"identity":99}]
    result["matches"]=False;result["changes"]=b.differences(REFERENCE,result["current"])
    return result
def make(name,count=2,io_class=h.FileIO):
    root=FIX/name;io=io_class(root)
    latch=h.DispatchLatch(SCHEDULE);rec=h.Recorder(root,latch,LABELS,io=io)
    rec.admit(copy.deepcopy(REFERENCE),copy.deepcopy(REFERENCE),{"changes":[]},SOURCE,{"forward_calls":0})
    for i in range(count):
        latch.consume(SCHEDULE[i]);require(rec.inspect(good,LABELS[i]))
    return rec,root
def stop(rec):
    return rec.scientific_stop("FINITE_ELIGIBILITY",FINDING_SHA)
def clean(rec):
    for label in rec.terminal_labels:require(rec.terminal_cleanup(good,label))
def status_file(rec,root):
    value=rec.status();raw=b.encode(value)
    require(len(raw)<=b.JSON_CAP)
    # Separate controller channel: never infer live status from any hook index.
    with (root/"controller_status.json").open("xb") as out:out.write(raw)
    with (root/"scientific_finding.json").open("xb") as out:out.write(b.encode(FINDING))
    return json.loads((root/"controller_status.json").read_bytes())
def saved(root,status,expected):
    return h.judge_prefix(root,LABELS,SCHEDULE,SOURCE,status,expected)
def group1():
    global NORMAL_PREFIX
    # Static inheritance proof reuses prior109/six-group results; no rerun.
    for name in ("admit","inspect","finish","_index_value","_packet","_write","_fault"):
        require(getattr(h.Recorder,name) is getattr(b.Recorder,name))
    require(h.judge_full is b.judge and b.MAX_CHECKS==109)
    prior=json.loads((ROOT/"diagnostics/fresh_confirmation_hook_failclosed_v1/TEST_RECEIPT.json").read_bytes())
    require(prior["status"]=="PASS" and len(prior["groups"])==6 and prior["model_calls"]==prior["tokenizer_calls"]==0)
    details=[]
    for name,count in (("valid_request_prefix",2),("valid_preflight_prefix",0)):
        rec,root=make(name,count);stop(rec);finding=rec.latch.scientific
        require(rec.latch.terminal and rec.latch._state=="SCIENTIFIC_STOP" and not rec.status()["permits_pass"])
        clean(rec);result=rec.finish_scientific_stop();status=status_file(rec,root)
        verdict=saved(root,status,finding)
        require(verdict["prefix_verified"] and not verdict["permits_pass"] and
                not h.judge_full(root,LABELS,SCHEDULE,SOURCE,status)["permits_pass"])
        require(len(verdict["remaining_ids"])==180-count and rec.checks==count and
                len(LABELS[count:])==109-count)
        if count==2:NORMAL_PREFIX=(root,status,finding)
        details.append({"case":name,"verdict":verdict,"status":status,"closeout":result})
    return {"full_pass_predicates_inherited_unchanged":True,"prior_six_groups_reused_not_rerun":True,"cases":details}
def group2():
    details=[]
    rec,root=make("cleanup_identity_fault");stop(rec);finding=rec.latch.scientific
    blocked(lambda:rec.terminal_cleanup(bad,rec.terminal_labels[0]))
    require(rec.latch.primary_code=="H_IDENTITY" and rec.latch.scientific==finding)
    status=status_file(rec,root);verdict=saved(root,status,finding)
    require(not verdict["prefix_verified"] and verdict["scientific_stop"]==finding)
    details.append({"case":"cleanup_identity_fault","status":status,"verdict":verdict})
    for mode in ("index","stop"):
        rec,root=make("lost_"+mode+"_acknowledgement",io_class=FaultIO)
        if mode=="index":
            stop(rec);clean(rec);rec.io.mode=mode
            blocked(rec.finish_scientific_stop)
            require(json.loads(rec.io.read("hook_evidence/prefix_index.json"))["attempt_complete"] is False)
        else:
            rec.io.mode=mode;blocked(lambda:stop(rec))
            require(len(rec.io.read("hook_evidence/scientific_stop.json"))==7)
        finding=rec.latch.scientific
        require(finding is not None and rec.latch.primary_code=="H_IO" and rec.receipt_failed and
                rec.prefix_index_sha is None and isinstance(rec.primary_exception,PoisonError))
        status=status_file(rec,root);verdict=saved(root,status,finding)
        require(not verdict["prefix_verified"] and status["scientific_stop"]==finding)
        details.append({"case":"lost_"+mode+"_acknowledgement","status":status,"verdict":verdict})
    require(not STR_CALLS)
    return {"cases":details,"arbitrary_exception_str_repr_calls":0}
def group3():
    original,status,finding=NORMAL_PREFIX
    cases=[]
    for kind in ("forged_normal_count","forged_remaining","forged_cleanup_label"):
        root=FIX/kind;shutil.copytree(original,root)
        index_path=root/"hook_evidence/prefix_index.json"
        index=json.loads(index_path.read_bytes());claimed=copy.deepcopy(status)
        if kind=="forged_normal_count":
            index["normal_checks"]=3;claimed["normal_checks"]=3
        elif kind=="forged_remaining":
            index["remaining_ids"]=index["remaining_ids"][1:];claimed["remaining_ids"]=index["remaining_ids"]
        else:
            path=root/"hook_evidence/terminal_cleanup.jsonl"
            lines=[json.loads(line) for line in path.read_bytes().splitlines()]
            lines[0]["label"]="matrix-finally"
            raw=b"".join(b.encode(line) for line in lines);path.write_bytes(raw)
            for record in index["files"]:
                if record["path"]=="hook_evidence/terminal_cleanup.jsonl":
                    record.update(bytes=len(raw),sha256=b.sha(raw))
        index_path.write_bytes(b.encode(index))
        claimed["prefix_index_sha256"]=b.sha(index_path.read_bytes())
        verdict=saved(root,claimed,finding)
        require(not verdict["prefix_verified"] and not verdict["permits_pass"])
        cases.append({"case":kind,"verdict":verdict})
    for kind in ("reset_admission","resume_dispatch"):
        rec,root=make(kind);stop(rec);finding=rec.latch.scientific;clean(rec);rec.finish_scientific_stop()
        calls=[]
        if kind=="reset_admission":blocked(rec.latch._admit)
        elif kind=="resume_dispatch":
            def forward():
                rec.latch.require_dispatch();calls.append("FORWARD")
            blocked(forward)
        require(not calls and rec.latch.scientific==finding and rec.latch.cursor==2)
        status=status_file(rec,root);verdict=saved(root,status,finding)
        require(not verdict["prefix_verified"] and not verdict["permits_pass"])
        cases.append({"case":kind,"actual_dispatches":0,"status":status,"verdict":verdict})
    missing=saved(original,None,finding);require(not missing["prefix_verified"])
    return {"cases":cases,"missing_live_controller_rejected":True}

def main():
    started=time.monotonic();groups=[];failure=None
    freeze=json.loads((HERE/"SOURCE_FREEZE.json").read_bytes())
    for entry in freeze["files"]:
        raw=(HERE/entry["path"]).read_bytes()
        require(len(raw)==entry["bytes"] and b.sha(raw)==entry["sha256"])
    for entry in json.loads((HERE/"SOURCE_PINS.json").read_bytes()):
        raw=(ROOT/entry["path"]).read_bytes()
        require(len(raw)==entry["bytes"] and b.sha(raw)==entry["sha256"] and raw==
                subprocess.check_output(["git","show",entry["commit"]+":"+entry["path"]],cwd=ROOT))
    with (HERE/"BATCH_STARTED.json").open("xb") as out:
        out.write(b.encode({"groups":3,"seconds_limit":60,"model_calls":0,"tokenizer_calls":0}))
    FIX.mkdir(exist_ok=False);name="not_started"
    try:
        for name,fn in (("valid_scientific_stop",group1),("negative_cleanup_technical_fault",group2),
                        ("forged_prefix_reset_resume_rejected",group3)):
            groups.append({"group":name,"status":"PASS","details":fn()})
    except BaseException as error:
        failure={"group":name,"code":error.code if type(error) is h.HookStopped else "PURE_ASSERTION_FAILURE",
                 "exception_text_serialized":False}
    elapsed=time.monotonic()-started
    files=[p for p in FIX.rglob("*") if p.is_file()]
    size=sum(p.stat().st_size for p in files)
    if elapsed>60 or size>32*b.MIB or any(p.stat().st_size>b.FILE_CAP for p in files):
        failure={"code":"RESOURCE_LIMIT","prior":failure}
    receipt={"status":"PASS" if len(groups)==3 and failure is None else "INCONCLUSIVE","groups":groups,
        "failure":failure,"elapsed_seconds":elapsed,"fixture_files":len(files),"fixture_bytes":size,
        "model_calls":0,"tokenizer_calls":0,"production_authorized":False,
        "source_freeze_sha256":b.sha((HERE/"SOURCE_FREEZE.json").read_bytes())}
    with (HERE/"TEST_RECEIPT.json").open("xb") as out:out.write(b.encode(receipt))
    print(json.dumps(receipt,sort_keys=True))
    return 0 if receipt["status"]=="PASS" else 1
if __name__=="__main__":raise SystemExit(main())
