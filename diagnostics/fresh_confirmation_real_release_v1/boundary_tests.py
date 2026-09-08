"""One pure/sentinel boundary batch. Never invokes the pinned loader or torch."""
import builtins
import copy
import json
import os
import sys
import threading
import time
from types import SimpleNamespace
from real_boundary import MAIN,Boundary,Denied,need,digest,encoded,strict,write_exclusive,OUTPUT

FORBIDDEN=("torch","transformer_lens","transformers","sp_lense","backend","loader","real_adapter","injection_backend")


def main():
    need(len(sys.argv)==2,"D_TEST_CALLER_LOCK")
    input_lock_raw=(MAIN/"TEST_INPUT_LOCK.json").read_bytes()
    need(digest(input_lock_raw)==sys.argv[1],"D_TEST_INPUT_LOCK")
    input_lock=strict(input_lock_raw)
    need(digest((MAIN/"SOURCE_FREEZE.json").read_bytes())==input_lock["source_sha256"] and
        digest((MAIN/"TEST_INPUTS.json").read_bytes())==input_lock["test_inputs_sha256"],"D_FROZEN_TEST_INPUTS")
    test_inputs=strict((MAIN/"TEST_INPUTS.json").read_bytes())
    usage=strict((MAIN/"USAGE_BEFORE_BATCH.json").read_bytes())
    from real_boundary import usage_value
    need(0<=time.time()-usage["observed_unix_seconds"]<=120 and usage_value(usage["tool_result"])["used_percent"]<100,"D_BATCH_USAGE")
    protocol=strict((MAIN/"TEST_PROTOCOL.json").read_bytes())
    started=time.monotonic();cutoff=started+75.;absolute=started+90.
    write_exclusive(MAIN/"BATCH_STARTED.json",encoded({"started_monotonic":started,"cutoff":cutoff,"absolute":absolute,
        "observed_unix_seconds":time.time(),"source_sha256":input_lock["source_sha256"],"test_input_lock_sha256":sys.argv[1]}),MAIN)
    expired=threading.Event()
    soft=threading.Timer(75.,expired.set);soft.daemon=True;soft.start()
    hard=threading.Timer(89.5,lambda:os._exit(124));hard.daemon=True;hard.start()
    blocked=[]
    original_import=builtins.__import__
    def guarded_import(name,*args,**kwargs):
        if any(name==n or name.startswith(n+".") for n in FORBIDDEN):
            blocked.append(name);raise RuntimeError("FORBIDDEN_MODEL_IMPORT")
        return original_import(name,*args,**kwargs)
    builtins.__import__=guarded_import
    report={"status":"INCONCLUSIVE_MODEL_FREE_BOUNDARY","groups":[{"name":g,"status":"UNRUN"} for g in protocol["groups"]],
        "production_authorized":False,"real_model_calls":0,"torch_backend_tokenizer_imports":0,"failure":None}
    calls=[];active=None
    def tick(): need(not expired.is_set() and time.monotonic()<cutoff,"D_SUBSTANTIVE_DEADLINE")
    def sentinel(identity,target):
        tick();need(identity["execution"]["permission_scope"]=="MODEL_FREE_SENTINEL_ONLY" and
            identity["execution"]["production_authorized"] is False,"D_SENTINEL_SCOPE")
        calls.append({"target":target,"actual_work":"SENTINEL_ONLY","identity":identity["execution"]})
        return "SENTINEL_REACHED_NO_IMPORT"
    def fixture(name,*,changes=None,usage_mode=None,omit=None):
        tick()
        root=MAIN/"test_evidence"/name
        need(not root.exists(),"D_FRESH_TEST_CASE")
        root.mkdir(parents=True)
        release=copy.deepcopy(test_inputs["release"]);auth=copy.deepcopy(test_inputs["authorization"])
        use=copy.deepcopy(test_inputs["usage"])
        if changes:
            for document,key,value in changes:
                (release if document=="release" else auth)[key]=value
        if usage_mode=="stale": use["observed_unix_seconds"]=0
        if usage_mode=="exhausted": use["tool_result"]["rateLimitsByLimitId"]["codex"]["primary"]["usedPercent"]=100
        if usage_mode=="unavailable": use["tool_result"]["rateLimitsByLimitId"]["codex"]["primary"]=None
        use_raw=encoded(use)
        release["usage_sha256"]=digest(use_raw) if usage_mode!="altered" else "0"*64
        release_raw=encoded(release)
        auth["release_sha256"]=digest(release_raw)
        auth_raw=encoded(auth)
        lock={"schema":"real_root_authority_lock.v1","source_sha256":input_lock["source_sha256"],
            "release_sha256":digest(release_raw),"authorization_sha256":digest(auth_raw)}
        lock_raw=encoded(lock)
        for filename,raw in (("ROOT_RELEASE.json",release_raw),("AUTHORIZATION.json",auth_raw),("AUTHORITY_LOCK.json",lock_raw),("USAGE_RECEIPT.json",use_raw)):
            if filename!=omit: write_exclusive(root/"root_release"/filename,raw,root)
        return Boundary(root,mock=True),digest(lock_raw)
    def admit(b,key): return b.admit_once(key,now=1060)
    def bootstrap(b,identity):
        write_exclusive(b.control/"owned/production_worker/BOOTSTRAP.json",encoded({"permission_received":True,
            "execution":identity["execution"],"actual_pid":101,"proof_scope":"SENTINEL_FIXTURE_NOT_OS_OWNERSHIP"}),b.root)
    def worker_terminal(b,identity):
        write_exclusive(b.control/"WORKER_RESULT.json",encoded({"terminal":{"execution":identity["execution"],
            "observed_model_work":"SENTINEL_ONLY","scientific_verdict":"NOT_TESTED"}}),b.root)
    def denial(label,function):
        before=len(calls)
        try: function()
        except (ValueError,OSError): pass
        else: raise AssertionError("DENIAL_NOT_ENFORCED:"+label)
        need(len(calls)==before,"D_SENTINEL_REACHED_ON_DENIAL")
        return {"case":label,"denied_before_sentinel":True}
    try:
        for active in report["groups"]:
            tick();active["status"]="STARTED";name=active["name"]
            details=[]
            if name=="approved_mock_sentinel":
                b,key=fixture("approved")
                pin,identity=admit(b,key);bootstrap(b,identity)
                before=len(calls);b.role("worker",key,pin,pid=101)
                need(b.dispatch(key,pin,sentinel,sentinel=True)=="SENTINEL_REACHED_NO_IMPORT" and len(calls)==before+1,"D_APPROVED_SENTINEL")
                worker_terminal(b,identity)
                details={"sentinel_calls":1,"intended_mode":"REAL_QWEN","observed_model_work":"SENTINEL_ONLY",
                    "actual_production_authorized":False,"loader_target":calls[-1]["target"]}
            elif name=="mode_and_binding_denials":
                variants=[("injection_relabel",[("release","execution_mode","INJECTION_MODULE"),("auth","execution_mode","INJECTION_MODULE")]),
                    ("production_relabel",[("release","production_authorized",True)]),
                    ("source_binding",[("release","source_sha256","0"*64)]),
                    ("input_binding",[("release","input_lock_sha256","0"*64)]),
                    ("runtime_binding",[("release","runtime_spec_sha256","0"*64)]),
                    ("output_escape",[("release","output_relative","../escape"),("auth","output_relative","../escape")]),
                    ("output_case",[("release","output_relative",OUTPUT.upper()),("auth","output_relative",OUTPUT.upper())])]
                for label,change in variants:
                    b,key=fixture(label,changes=change);details.append(denial(label,lambda b=b,key=key:admit(b,key)))
            elif name=="usage_and_authority_denials":
                for label,mode,omit in (("missing_usage",None,"USAGE_RECEIPT.json"),("stale_usage","stale",None),
                    ("altered_usage","altered",None),("exhausted_usage","exhausted",None),("unavailable_usage","unavailable",None),
                    ("missing_authority",None,"AUTHORIZATION.json")):
                    b,key=fixture(label,usage_mode=mode,omit=omit);details.append(denial(label,lambda b=b,key=key:admit(b,key)))
                b,key=fixture("wrong_caller_pin");details.append(denial("wrong_caller_pin",lambda:admit(b,"0"*64)))
                b,key=fixture("disabled",changes=[("auth","allow_execute",False)])
                details.append(denial("disabled",lambda:admit(b,key)))
                from launch import preflight
                disabled=preflight("0"*64)
                need(disabled["status"]=="DISABLED_NO_APPROVED_REAL_RELEASE" and not disabled["production_authorized"],"D_DISABLED_PREFLIGHT")
                report["disabled_state_preflight"]=disabled
            elif name=="exclusive_and_terminal_denials":
                b=Boundary(MAIN/"test_evidence/approved",mock=True)
                key=digest(b.read("root_release/AUTHORITY_LOCK.json"));pin=digest(b.read_control("ATTEMPT_ADMISSION.json"))
                before={p.relative_to(b.root).as_posix():digest(p.read_bytes()) for p in b.root.rglob("*") if p.is_file()}
                details.append(denial("duplicate_attempt",lambda:admit(b,key)))
                details.append(denial("duplicate_worker",lambda:b.role("worker",key,pin,pid=101)))
                details.append(denial("duplicate_loader",lambda:b.dispatch(key,pin,sentinel,sentinel=True)))
                after={p.relative_to(b.root).as_posix():digest(p.read_bytes()) for p in b.root.rglob("*") if p.is_file()}
                need(before==after,"D_PRIOR_EVIDENCE_CHANGED")
                b,key=fixture("no_worker_terminal");pin,identity=admit(b,key)
                details.append(denial("missing_worker_terminal",lambda:b.role("audit",key,pin)))
                b,key=fixture("no_loader_terminal");pin,identity=admit(b,key);bootstrap(b,identity);b.role("worker",key,pin,pid=101)
                write_exclusive(b.control/"LOADER_STARTED.json",encoded({"execution":identity["execution"]}),b.root)
                worker_terminal(b,identity)
                details.append(denial("missing_loader_terminal",lambda:b.role("audit",key,pin)))
                b,key=fixture("missing_permission");pin,identity=admit(b,key)
                details.append(denial("missing_owned_permission",lambda:b.role("worker",key,pin,pid=101)))
            else:
                b=Boundary(MAIN/"test_evidence/approved",mock=True)
                key=digest(b.read("root_release/AUTHORITY_LOCK.json"));pin=digest(b.read_control("ATTEMPT_ADMISSION.json"))
                original_identity=b.reauthenticate(key,pin)
                need(strict(b.read_control("WORKER_ENTRY.json"))["execution"]==original_identity["execution"],"D_ADMITTED_WORKER_IDENTITY")
                writer_identity=b.role("writer",key,pin)
                import real_boundary
                original_time=real_boundary.time
                real_boundary.time=SimpleNamespace(time=lambda:10000.)
                try: audit_identity=b.role("audit",key,pin)
                finally: real_boundary.time=original_time
                need(original_identity==writer_identity==audit_identity,"D_SHARED_ROLE_IDENTITY")
                need(original_identity["execution"]["real_model_work"] is None,"D_AUTHORITY_IS_NOT_OBSERVATION")
                bindings=strict((MAIN/"BINDINGS.json").read_bytes())
                need(bindings["production_seconds"]=={"worker":1800,"audit":180,"cleanup":15} and
                    bindings["production_ceiling"]=={"forwards":180,"derivatives":48,"loads":1,"tokens":160,"hook_checks":109,
                        "evidence_bytes":288*1024**2,"file_bytes":5*1024**2},"D_UNCHANGED_RESOURCE_CEILINGS")
                need(calls[0]["target"]==bindings["loader"],"D_EXACT_LOADER_TARGET")
                details={"same_worker_writer_audit_identity":True,"late_audit_seconds":10000.,"usage_freshness_applied_only_at_admission":True,
                    "production_seconds":bindings["production_seconds"],"production_ceiling":bindings["production_ceiling"]}
            active.update(status="PASS",details=details)
            print(json.dumps({"group":name,"status":"PASS"}),flush=True)
        need(not blocked and len(calls)==1 and time.monotonic()<cutoff,"D_ZERO_MODEL_IMPORTS_ONE_SENTINEL")
        report["status"]="PASS_MODEL_FREE_BOUNDARY"
    except BaseException as error:
        if active: active["status"]="FAIL"
        import traceback
        report["failure"]={"code":"BOUNDARY_BATCH_FAILURE","type":type(error).__name__,
            "frames":[{"file":frame.filename.split("\\")[-1].split("/")[-1],"line":frame.lineno,"function":frame.name}
                for frame in traceback.extract_tb(error.__traceback__)[-6:]]}
    finally:
        builtins.__import__=original_import;soft.cancel()
        report.update(elapsed_seconds=time.monotonic()-started,within_substantive=time.monotonic()<cutoff,
            within_absolute=time.monotonic()<absolute,sentinel_calls=len(calls),forbidden_import_attempts=blocked,
            source_sha256=input_lock["source_sha256"],test_input_lock_sha256=sys.argv[1],
            unrun_groups=sum(g["status"]=="UNRUN" for g in report["groups"]))
        write_exclusive(MAIN/"BATCH_REPORT.json",encoded(report),MAIN)
        hard.cancel()
    return 0 if report["status"]=="PASS_MODEL_FREE_BOUNDARY" else 1


if __name__=="__main__":raise SystemExit(main())
