"""Reviewed owned controller under a strict first-baseline diagnostic subbudget."""
import builtins
import json
import time
from support import HERE,run_dir,write_new,require,sha,bounds

def control(name): return run_dir()/"control"/name

def forbid_research_imports():
    original=builtins.__import__
    blocked={"torch","transformers","transformer_lens","sp_lense","tokenizers","numpy"}
    def checked(name,*args,**kwargs):
        require(name.split(".")[0] not in blocked,"MODEL_FREE_IMPORT_BOUNDARY")
        return original(name,*args,**kwargs)
    builtins.__import__=checked

def good_capture(value):
    return bool(value and value.get("binding_authenticated") and value.get("quiescent") and
        value.get("stop_reason") is None and not value.get("faults") and not value.get("cleanup_faults") and
        not value.get("stdout_capture_errors") and value.get("primary_error") is None and value.get("threads_joined") and
        value.get("pipes_closed") and value.get("within_absolute_cleanup_deadline") and
        all(value.get("exit_proofs",{}).get(k,{}).get("exit_code")==0 for k in ("actual_worker","launcher")))

def worker(deadline):
    from authority import current
    boundary,approved,admission=current()
    identity=boundary.role("worker",approved,admission)
    result={"execution":identity["execution"],"status":"INCONCLUSIVE_DIAGNOSTIC","failure":"DIAGNOSTIC_ENTRY_FAILURE"}
    try:
        if boundary.mock: forbid_research_imports()
        from diagnostic_core import execute
        result=execute(boundary,approved,admission,deadline,__import__("os").environ.get("SP_SETUP_FIXTURE") if boundary.mock else None)
    except BaseException as error:
        import sys
        core=sys.modules.get("diagnostic_core")
        if core is None or not getattr(core,"STARTUP_COMPLETE",False):
            from startup_status import capture
            result["startup_failure"]=capture(error,getattr(core,"STARTUP_STAGE","CORE_IMPORT"),identity["execution"],write_new)
        result["live_trace_status"]=getattr(core,"LAST_FINITE_STATUS",None)
    finally:
        import sys
        core=sys.modules.get("diagnostic_core")
        helper_status=getattr(core,"LAST_HELPER_STATUS",None)
        operand_status=getattr(core,"LAST_OPERAND_STATUS",None)
        write_new("WORKER_RESULT.json",{"terminal":{"execution":identity["execution"]},"diagnostic_result":result,"helper_status":helper_status,"constructor_operands":operand_status,"startup_failure":result.get("startup_failure"),
            "scope":"FIRST_LOCKED_UNEDITED_BASELINE_ONLY","scientific_pass":False,"full_scientific_study_unrun":True},critical=True)
    return 0  # A saved diagnostic outcome, not a successful construction.

def audit(deadline):
    from authority import current
    boundary,approved,admission=current()
    execution=boundary.reauthenticate(approved,admission)["execution"]
    result={"execution":execution,"audit_completed":False,"classification":"INCONCLUSIVE_DIAGNOSTIC","scientific_pass":False}
    try:
        boundary.role("audit",approved,admission)
        forbid_research_imports()  # Saved reader has no model dependencies in either mode.
        from diagnostic_reader import judge
        result=judge(boundary,execution,deadline)
    except BaseException: result["audit_error"]="SAVED_DIAGNOSTIC_AUDIT_FAILURE"
    finally: write_new("AUDIT_RESULT.json",result,critical=True)
    return 0 if result["audit_completed"] else 1

def controller(budget,key):
    from authority import current
    from owned_production import supervise
    boundary,approved,admission=current()
    identity=boundary.reauthenticate(approved,admission);execution=identity["execution"]
    write_new("CONTROLLER_ENTRY.json",{"execution":execution,"scope":"FIRST_LOCKED_UNEDITED_BASELINE_ONLY","observed_model_work":"NONE_BEFORE_OWNED_LAUNCH"})
    started=time.monotonic();captures={};errors=[]
    final={"schema":"first_forward_diagnostic_parent_final.v1","execution":execution,"classification":"INCONCLUSIVE_DIAGNOSTIC",
        "audit_completed":False,"scientific_pass":False,"requests_unrun":0,"full_scientific_study_unrun":True,
        "external_capture_overrides_worker":True,"unconditional_closeout_attempted":True}
    try:
        write_new("CONTROLLER_STARTED.json",{"execution":execution,"diagnostic_seconds":{"worker":180,"audit":60,"shared_cleanup":15},
            "unchanged_upper_seconds":{"worker":1800,"audit":180,"cleanup":15},"absolute_end":budget.absolute_end,
            "substantive_end":budget.substantive_end,"one_global_cleanup_budget":True})
        owned=json.loads((HERE/"OWNED_IDENTITY.json").read_bytes())
        for lane in ("worker","audit"):
            deadline,cleanup_end=budget.deadlines(time.monotonic(),lane)
            value=supervise("production_"+lane,deadline,cleanup_end,owned);captures[lane]=value
            budget.charge(key+":"+lane,time.monotonic()-value["cleanup_started_monotonic"])
            require(value["quiescent"],"no successor before owned quiescence")
            if lane=="worker":
                worker=control("WORKER_RESULT.json").read_bytes()
                write_new("CLOSED_WORKER_BINDING.json",{"execution":execution,"worker_result_sha256":sha(worker),
                    "owned_capture_sha256":sha(control("owned/production_worker/CAPTURE.json").read_bytes()),
                    "quiescent":True,"good_capture":good_capture(value)},critical=True)
        result=json.loads(control("AUDIT_RESULT.json").read_bytes())
        require(result["execution"]==execution,"independent audited authority join")
        if all(good_capture(captures.get(x)) for x in ("worker","audit")) and result["audit_completed"] and time.monotonic()<=budget.absolute_end:
            final.update(classification=result["classification"],audit_completed=True)
    except BaseException: errors.append("DIAGNOSTIC_CONTROLLER_OR_CAPTURE_FAILURE")
    finally:
        def optional_hash(name):
            try: return sha(control(name).read_bytes())
            except OSError: return None
        final.update(controller_errors=errors,shared_cleanup=budget.record(),elapsed_seconds=time.monotonic()-started,
            finished_monotonic=time.monotonic(),worker_quiescent=bool(captures.get("worker",{}).get("quiescent")),
            audit_quiescent=bool(captures.get("audit",{}).get("quiescent")),
            worker_result_sha256=optional_hash("WORKER_RESULT.json"),audit_result_sha256=optional_hash("AUDIT_RESULT.json"),
            worker_capture_sha256=optional_hash("owned/production_worker/CAPTURE.json"),audit_capture_sha256=optional_hash("owned/production_audit/CAPTURE.json"))
        write_new("PARENT_FINAL.json",final,critical=True)
    return final
