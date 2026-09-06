"""Bounded structured hook evidence; source-locked setup, no model calls."""
import json
import sys
import time
import zlib
from pathlib import Path
from core import Budget,require,sha,json_bytes
from guard_candidate import HookGuard,snapshot,differences

class HookRecorder:
    def __init__(self,model,plan,output,counter):
        self.root=Path(output)
        self.budget=Budget(output)
        self.cap=plan["hook_integration"]["hook_evidence_cap_bytes"]
        self.chunk=plan["hook_integration"]["chunk_raw_bytes"]
        self.raw_cap=plan["hook_integration"]["maximum_snapshot_raw_bytes"]
        self.used=self.checks=0
        self.maximum_checks=plan["hook_integration"]["maximum_checks"]
        (self.root/"hook_evidence").mkdir(exist_ok=False)
        before_snapshot=before=None
        stage="source_lock"
        try:
            sources=plan["hook_integration"]["installed_sources_sha256"]
            require(all(sha(Path(path).read_bytes())==digest for path,digest in sources.items()),"hook installed source mismatch before setup")
            self.put_json("source_lock.json",{"installed_sources_sha256":sources,"passed":True,"before_materialization":True})
            from transformer_lens.model_bridge.generalized_components.block import BlockBridge
            blocks=[{"path":name,"type":type(block).__module__+"."+type(block).__qualname__,
                     "previously_wired":block._pre_ln_capture_wired}
                     for name,block in model.named_modules() if isinstance(block,BlockBridge)]
            initial=counter.attempts
            stage="setup_reference"
            before_snapshot=snapshot(model)
            before=self.write_value("setup_before",before_snapshot)
            stage="materialization"
            self.guard=HookGuard(model)
            require(counter.attempts==initial==0,"hook setup must precede all forwards and add none")
            require(self.guard.setup["before"]==before_snapshot,"unchanged setup input reference")
            reference=self.write_value("reference",self.guard.baseline)
            require(self.guard.setup["after"]==self.guard.baseline,"setup/reference identity")
            setup=self.write_value("setup_changes",{k:v for k,v in self.guard.setup.items() if k not in ("before","after")})
            self.reference_sha=reference["raw_sha256"]
            self.put_json("setup_receipt.json",{"status":"PASS","forward_calls":0,"before":before,"reference":reference,"setup_changes":setup,
                 "blocks":blocks,"newly_wired":[b["path"] for b in blocks if not b["previously_wired"]],
                 "already_wired":[b["path"] for b in blocks if b["previously_wired"]],
                 "already_wired_interpretation":"preserved inherited loaded state; skipped setup callbacks NOT authenticated by candidate",
                 "installed_source_lock":"hook_evidence/source_lock.json"})
        except BaseException as error:
            detail={"setup_stage":stage}
            if before_snapshot is not None and stage=="materialization":
                try:
                    detail["changes_artifact"]=self.write_value("setup_failure",{"before_reference":before,
                            "changes":differences(before_snapshot,snapshot(model))})
                except BaseException as recording_error:
                    detail["recording_error"]=type(recording_error).__name__+": "+str(recording_error)[:2048]
            self.fault("setup",error,detail)
            raise
    def put_json(self,name,value):
        raw=json_bytes(value)
        require(self.used+len(raw)<=self.cap-65536,"hook evidence reserved capacity")
        self.budget.write_bytes("hook_evidence/"+name,raw)
        self.used+=len(raw)
    def write_value(self,name,value):
        raw=json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        require(len(raw)<=self.raw_cap,"hook raw snapshot size")
        chunks=[raw[i:i+self.chunk] for i in range(0,len(raw),self.chunk)]
        compressed=[zlib.compress(c) for c in chunks]
        manifest={"raw_bytes":len(raw),"raw_sha256":sha(raw),"encoding":"UTF8 compact JSON; concatenated zlib-compressed raw chunks",
                  "complete":True,"chunks":[{"path":f"hook_evidence/{name}_{i:03d}.zlib","raw_bytes":len(c),"raw_sha256":sha(c),
                    "compressed_bytes":len(z),"compressed_sha256":sha(z)} for i,(c,z) in enumerate(zip(chunks,compressed,strict=True))]}
        needed=sum(len(z) for z in compressed)+len(json_bytes(manifest))
        require(self.used+needed<=self.cap-65536,"hook compressed evidence capacity")
        for entry,z in zip(manifest["chunks"],compressed,strict=True):
            self.budget.write_bytes(entry["path"],z)
            self.used+=len(z)
        self.put_json(name+".json",manifest)
        return {"manifest":"hook_evidence/"+name+".json","raw_bytes":len(raw),"raw_sha256":sha(raw)}
    def fault(self,stage,error,details=None):
        path=self.root/"hook_evidence/fault.json"
        if not path.exists():
            self.budget.write("hook_evidence/fault.json",{"status":"INCONCLUSIVE","stage":stage,"complete_evidence":False,
               "error":type(error).__name__+": "+str(error)[:2048],"recorded_hook_bytes":self.used,"cap_bytes":self.cap,"monotonic":time.monotonic(),"details":details})
    def inspect(self,model,label):
        try:
            self.checks+=1
            require(self.checks<=self.maximum_checks,"hook check count ceiling")
            inspection=self.guard.inspect(model)
            event={"check":self.checks,"label":label,"reference_sha256":self.reference_sha,"matches":inspection["matches"],
                   "change_count":len(inspection["changes"]),"complete_evidence":True,"monotonic":time.monotonic()}
            if inspection["matches"]: event["changes"]=[]
            else: event["changes_artifact"]=self.write_value(f"failure_{self.checks:02d}",inspection["changes"])
            raw=(json.dumps(event,allow_nan=False)+"\n").encode()
            require(self.used+len(raw)<=self.cap-65536,"hook event capacity")
            self.budget.write_bytes("hook_evidence/checks.jsonl",raw,"ab")
            self.used+=len(raw)
            return inspection["matches"]
        except BaseException as error:
            self.fault("check:"+label,error)
            raise

def finish_preserving_original(output,stage,function):
    """Use from finally: never replace an active causal exception with cleanup failure."""
    primary=sys.exc_info()[1]
    try:
        return function()
    except BaseException as secondary:
        receipt={"stage":stage,"primary_error":None if primary is None else type(primary).__name__+": "+str(primary)[:2048],
                 "cleanup_error":type(secondary).__name__+": "+str(secondary)[:2048],"monotonic":time.monotonic()}
        try: Budget(output).event("cleanup_errors.jsonl",receipt)
        except BaseException:
            if primary is None: raise secondary
        if primary is None: raise
        return None

def read_value(root,reference):
    root=Path(root)
    manifest=json.loads((root/reference["manifest"]).read_bytes())
    require(manifest["complete"] and manifest["raw_sha256"]==reference["raw_sha256"],"hook manifest")
    chunks=[]
    for entry in manifest["chunks"]:
        compressed=(root/entry["path"]).read_bytes()
        require(len(compressed)==entry["compressed_bytes"] and sha(compressed)==entry["compressed_sha256"],"hook compressed authentication")
        raw=zlib.decompress(compressed)
        require(len(raw)==entry["raw_bytes"] and sha(raw)==entry["raw_sha256"],"hook raw chunk authentication")
        chunks.append(raw)
    raw=b"".join(chunks)
    require(len(raw)==manifest["raw_bytes"] and sha(raw)==manifest["raw_sha256"],"complete hook snapshot authentication")
    return json.loads(raw)

def verify_saved(root,expected_checks=45,plan=None):
    root=Path(root)
    require(not (root/"hook_evidence/fault.json").exists(),"hook evidence fault")
    receipt=json.loads((root/"hook_evidence/setup_receipt.json").read_bytes())
    before=read_value(root,receipt["before"])
    reference=read_value(root,receipt["reference"])
    changes=read_value(root,receipt["setup_changes"])
    from guard_candidate import differences
    require(changes["changes"]==differences(before,reference),"saved hook setup differences")
    checks=[json.loads(line) for line in (root/"hook_evidence/checks.jsonl").read_text().splitlines()]
    require(len(checks)==expected_checks and [c["check"] for c in checks]==list(range(1,len(checks)+1)),"complete hook check schedule")
    require(all(c["matches"] and c["complete_evidence"] and c.get("changes")==[] and
                c["reference_sha256"]==receipt["reference"]["raw_sha256"] for c in checks),"exact hook checks")
    require(receipt["forward_calls"]==0,"no setup model forward")
    if plan is not None:
        lock=json.loads((root/"hook_evidence/source_lock.json").read_bytes())
        require(lock["passed"] and lock["before_materialization"] and
                lock["installed_sources_sha256"]==plan["hook_integration"]["installed_sources_sha256"],"saved installed source lock")
        labels=[]
        for request in plan["requests"]:
            labels.extend(("ON-entry:"+request["prompt_id"],"ON-finally:"+request["prompt_id"]))
            for prompt in plan["prompts"]:
                if prompt["category"]!="self_shutdown":
                    cid=prompt["prompt_id"]+"__oracle_off_"+request["policy"]
                    labels.extend(("OFF-entry:"+cid,"OFF-exit:"+cid))
        labels.append("matrix-finally")
        require([c["label"] for c in checks]==labels,"paired hook cleanup sequence")
        require(all(a["monotonic"]<=b["monotonic"] for a,b in zip(checks,checks[1:])),"hook event order")
        require(sum(p.stat().st_size for p in (root/"hook_evidence").rglob("*") if p.is_file())<=
                plan["hook_integration"]["hook_evidence_cap_bytes"],"hook evidence cap")
    return {"checks":len(checks),"reference_sha256":receipt["reference"]["raw_sha256"],
            "newly_wired":receipt["newly_wired"],"already_wired":receipt["already_wired"],
            "already_wired_interpretation":receipt["already_wired_interpretation"],"status":"PASS"}
