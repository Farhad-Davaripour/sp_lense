"""One frozen clock-path delta; pure stdlib and one inert sentinel only."""
import builtins
import copy
import contextlib
import json
import os
import sys
import threading
import time
from types import SimpleNamespace
import real_boundary as boundary_module
from real_boundary import MAIN,Boundary,Denied,need,digest,encoded,strict,write_exclusive,OUTPUT

FORBIDDEN=("torch","transformer_lens","transformers","sp_lense","backend","loader","real_adapter","injection_backend")


class ClockFixture:
    """Test-only module clock; no production override argument or environment knob."""
    def __init__(self,case):
        self.case=case;self.now=case["entry_time"];self.events=[];self.chain_done=False;self.path_done=False
    def read(self):
        self.events.append({"event":"clock_sample","time":self.now})
        return self.now
    @contextlib.contextmanager
    def bind(self,boundary):
        saved_time=boundary_module.time;saved_checked=boundary_module.checked_path;saved_verify=boundary.verify_chain
        def verify(key):
            value=saved_verify(key)
            if not self.chain_done:
                self.now=self.case["after_chain"];self.chain_done=True
                self.events.append({"event":"chain_complete","time":self.now})
            return value
        def checked(root,relative):
            value=saved_checked(root,relative)
            if self.chain_done and not self.path_done and root==boundary.root and relative==OUTPUT:
                self.now=self.case.get("after_path",self.now);self.path_done=True
                self.events.append({"event":"path_complete","time":self.now})
            return value
        boundary.verify_chain=verify
        boundary_module.checked_path=checked
        boundary_module.time=SimpleNamespace(time=self.read)
        try: yield self
        finally:
            boundary.verify_chain=saved_verify
            boundary_module.checked_path=saved_checked
            boundary_module.time=saved_time


def main():
    need(len(sys.argv)==2,"D_TEST_CALLER_LOCK")
    lock_raw=(MAIN/"TEST_INPUT_LOCK.json").read_bytes()
    need(digest(lock_raw)==sys.argv[1],"D_TEST_INPUT_LOCK")
    lock=strict(lock_raw)
    for name,field in (("SOURCE_FREEZE.json","source_sha256"),("TEST_INPUTS.json","test_inputs_sha256"),("TEST_PROTOCOL.json","test_protocol_sha256")):
        need(digest((MAIN/name).read_bytes())==lock[field],"D_FROZEN_TEST_INPUTS")
    from support import check_freeze,bounds,ROOT
    check_freeze()
    reused=strict((MAIN/"REUSED_PROOFS.json").read_bytes())
    for item in reused["files"]:
        raw=(ROOT/item["path"]).read_bytes()
        need(len(raw)==item["bytes"] and digest(raw)==item["sha256"],"D_REUSED_PROOF_HASH")
    usage=strict((MAIN/"USAGE_BEFORE_BATCH.json").read_bytes())
    need(0<=time.time()-usage["observed_unix_seconds"]<=120 and boundary_module.usage_value(usage["tool_result"])["used_percent"]<100,"D_BATCH_USAGE")
    protocol=strict((MAIN/"TEST_PROTOCOL.json").read_bytes());inputs=strict((MAIN/"TEST_INPUTS.json").read_bytes())
    started=time.monotonic();cutoff=started+45.;absolute=started+60.
    write_exclusive(MAIN/"BATCH_STARTED.json",encoded({"started_monotonic":started,"substantive_cutoff":cutoff,"absolute_deadline":absolute,
        "observed_unix_seconds":time.time(),"source_sha256":lock["source_sha256"],"test_input_lock_sha256":sys.argv[1]}),MAIN)
    expired=threading.Event();soft=threading.Timer(45.,expired.set);soft.daemon=True;soft.start()
    hard=threading.Timer(59.5,lambda:os._exit(124));hard.daemon=True;hard.start()
    original_import=builtins.__import__;blocked=[]
    def guarded_import(name,*args,**kwargs):
        if any(name==n or name.startswith(n+".") for n in FORBIDDEN):
            blocked.append(name);raise RuntimeError("FORBIDDEN_MODEL_IMPORT")
        return original_import(name,*args,**kwargs)
    builtins.__import__=guarded_import
    calls=[];active=None
    report={"status":"INCONCLUSIVE_MODEL_FREE_FRESHNESS","groups":[{"name":n,"status":"UNRUN"} for n in protocol["groups"]],
        "production_authorized":False,"real_model_calls":0,"failure":None}
    def tick(): need(not expired.is_set() and time.monotonic()<cutoff,"D_SUBSTANTIVE_DEADLINE")
    def fixture(name):
        tick();root=MAIN/"test_evidence"/name;need(not root.exists(),"D_FRESH_CASE");root.mkdir(parents=True)
        release=copy.deepcopy(inputs["release"]);auth=copy.deepcopy(inputs["authorization"]);usage_raw=encoded(inputs["usage"])
        release["usage_sha256"]=digest(usage_raw);release_raw=encoded(release)
        auth["release_sha256"]=digest(release_raw);auth_raw=encoded(auth)
        authority={"schema":"real_root_authority_lock.v1","source_sha256":lock["source_sha256"],
            "release_sha256":digest(release_raw),"authorization_sha256":digest(auth_raw)}
        authority_raw=encoded(authority)
        for name,raw in (("ROOT_RELEASE.json",release_raw),("AUTHORIZATION.json",auth_raw),("AUTHORITY_LOCK.json",authority_raw),("USAGE_RECEIPT.json",usage_raw)):
            write_exclusive(root/"root_release"/name,raw,root)
        return Boundary(root,mock=True),digest(authority_raw)
    def sentinel(identity,target):
        tick();need(identity["execution"]["production_authorized"] is False and identity["execution"]["permission_scope"]=="MODEL_FREE_SENTINEL_ONLY","D_SENTINEL_SCOPE")
        calls.append({"observed_model_work":"SENTINEL_ONLY","target":target})
        return "SENTINEL_ONLY"
    try:
        for active in report["groups"]:
            tick();active["status"]="STARTED";name=active["name"]
            if name=="validation_delay_denials":
                details=[]
                for case_name in ("verification_delay","path_delay"):
                    case=protocol["cases"][case_name];b,key=fixture(case_name);clock=ClockFixture(case)
                    with clock.bind(b):
                        try: b.admit_once(key)
                        except Denied as error: need(error.args==(case["expected"],),"D_WRONG_DELAY_DENIAL")
                        else: raise AssertionError("D_DELAY_WAS_ADMITTED")
                    need(clock.events==[{"event":"chain_complete","time":case["after_chain"]},
                        {"event":"path_complete","time":case.get("after_path",case["after_chain"])},
                        {"event":"clock_sample","time":1121}],"D_CLOCK_BEFORE_VALIDATION")
                    need(not b.output.exists() and not calls,"D_DELAY_RESERVED_OR_DISPATCHED")
                    details.append({"case":case_name,"code":"D_STALE_USAGE","events":clock.events,
                        "attempt_reserved":False,"admission_receipt_exists":False,"sentinel_calls":0})
                active["details"]=details
            elif name=="timely_admission_late_audit":
                case=protocol["cases"]["timely"];b,key=fixture("timely");clock=ClockFixture(case)
                with clock.bind(b):
                    pin,identity=b.admit_once(key)
                    saved=strict(b.read_control("ATTEMPT_ADMISSION.json"))
                    need(saved["admitted_unix_seconds"]==case["admitted_time"] and
                        [x for x in clock.events if x["event"]=="clock_sample"]==[{"event":"clock_sample","time":case["admitted_time"]}],"D_POST_CHECK_TIMESTAMP")
                    write_exclusive(b.control/"owned/production_worker/BOOTSTRAP.json",encoded({"permission_received":True,
                        "execution":identity["execution"],"actual_pid":101,"proof_scope":"PURE_TIMING_FIXTURE_NOT_OS_OWNERSHIP"}),b.root)
                    need(b.role("worker",key,pin,pid=101)==identity,"D_WORKER_IDENTITY")
                    need(b.dispatch(key,pin,sentinel,sentinel=True)=="SENTINEL_ONLY","D_TIMELY_SENTINEL")
                    write_exclusive(b.control/"WORKER_RESULT.json",encoded({"terminal":{"execution":identity["execution"],
                        "observed_model_work":"SENTINEL_ONLY","scientific_verdict":"NOT_TESTED"}}),b.root)
                    before=digest(b.read_control("ATTEMPT_ADMISSION.json"));clock.now=case["audit_time"]
                    need(b.role("writer",key,pin)==b.role("audit",key,pin)==identity,"D_LATE_SHARED_IDENTITY")
                    need(digest(b.read_control("ATTEMPT_ADMISSION.json"))==before==pin,"D_ADMISSION_MUTATED")
                    need(sum(x["event"]=="clock_sample" for x in clock.events)==1,"D_AUDIT_REDEMANDS_FRESHNESS")
                active["details"]={"events":clock.events,"admission_sha256":pin,"admitted_unix_seconds":saved["admitted_unix_seconds"],
                    "late_audit_unix_seconds":case["audit_time"],"same_worker_writer_audit_identity":True,
                    "production_authorized":False,"observed_model_work":"SENTINEL_ONLY","sentinel_calls":1}
            elif name=="production_clock_override_denied":
                b=Boundary();verify_calls=[]
                def forbidden_verify(key): verify_calls.append(key);raise AssertionError("D_OVERRIDE_REACHED_CHAIN")
                b.verify_chain=forbidden_verify
                try: b.admit_once("0"*64,now=protocol["cases"]["production_override"]["now"])
                except Denied as error: need(error.args==("D_REAL_CLOCK_OVERRIDE",),"D_WRONG_OVERRIDE_DENIAL")
                else: raise AssertionError("D_PRODUCTION_CLOCK_OVERRIDE_ACCEPTED")
                need(not verify_calls and not b.output.exists(),"D_OVERRIDE_SIDE_EFFECT")
                active["details"]={"code":"D_REAL_CLOCK_OVERRIDE","verify_chain_calls":0,"attempt_reserved":False}
            elif name=="actual_disabled_preflight":
                import launch
                saved_argv=sys.argv
                sys.argv=[str(MAIN/"launch.py"),"--authority-lock",str(MAIN/"root_release/AUTHORITY_LOCK.json"),"--approved-lock-sha256","0"*64,"--preflight"]
                try: exit_code=launch.main()
                finally: sys.argv=saved_argv
                result=launch.preflight("0"*64)
                need(exit_code==2 and result["status"]=="DISABLED_NO_APPROVED_REAL_RELEASE" and
                    result["production_authorized"] is False and result["observed_model_work"]=="NONE","D_REAL_PREFLIGHT_NOT_DISABLED")
                need(not (MAIN/"root_release").exists() and not (MAIN/"real_evidence").exists(),"D_REAL_AUTHORITY_OR_OUTPUT_CREATED")
                active["details"]={"exit_code":exit_code,**result,"root_release_exists":False,"real_evidence_exists":False}
            else: raise AssertionError("D_UNKNOWN_FROZEN_GROUP")
            active["status"]="PASS"
            print(json.dumps({"group":name,"status":"PASS"}),flush=True)
        tick();need(not blocked and len(calls)==1,"D_MODEL_IMPORT_OR_SENTINEL_COUNT")
        report["status"]="PASS_MODEL_FREE_FRESHNESS"
    except BaseException as error:
        if active: active["status"]="FAIL"
        import traceback
        report["failure"]={"code":"FRESHNESS_BATCH_FAILURE","type":type(error).__name__,
            "frames":[{"file":frame.filename.replace("\\","/").split("/")[-1],"line":frame.lineno,"function":frame.name}
                for frame in traceback.extract_tb(error.__traceback__)[-6:]]}
    finally:
        builtins.__import__=original_import;soft.cancel()
        elapsed=time.monotonic()-started
        report.update(elapsed_seconds=elapsed,within_substantive=time.monotonic()<cutoff,within_absolute=time.monotonic()<absolute,
            sentinel_calls=len(calls),forbidden_import_attempts=blocked,source_sha256=lock["source_sha256"],
            test_input_lock_sha256=sys.argv[1],unrun_groups=sum(x["status"]=="UNRUN" for x in report["groups"]),
            resource_bytes=bounds(),owned_children_started=0,model_work_observation="ONE_PURE_SENTINEL_NO_MODEL")
        write_exclusive(MAIN/"BATCH_REPORT.json",encoded(report),MAIN)
        hard.cancel()
    return 0 if report["status"]=="PASS_MODEL_FREE_FRESHNESS" else 1


if __name__=="__main__":raise SystemExit(main())
