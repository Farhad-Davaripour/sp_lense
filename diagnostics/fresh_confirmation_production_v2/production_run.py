"""Production controller/worker/independent-reader path; current authority is injection only."""
import json
import time
from support import HERE,run_dir,require,write_new,sha,bounds,SOURCES
from shared_cleanup import SharedCleanup


def contract():
    value=json.loads(SOURCES.read("2620f66d4d50456c88800554bc9a26845998cf50","diagnostics/semantic_confirmation_resource_v1/contract.json"))
    require(value["seconds"]=={"worker":1800,"cleanup":15,"audit":180},"one production contract")
    return value


def control_file(name): return run_dir()/"control"/name


def good_capture(value):
    return bool(value and value.get("binding_authenticated") and value.get("quiescent") and
        value.get("stop_reason") is None and not value.get("faults") and not value.get("cleanup_faults") and
        not value.get("stdout_capture_errors") and value.get("primary_error") is None and value.get("threads_joined") and
        value.get("pipes_closed") and value.get("within_absolute_cleanup_deadline") and
        all(value.get("exit_proofs",{}).get(k,{}).get("exit_code")==0 for k in ("actual_worker","launcher")))


def controller(outer_cutoff):
    from authority import authenticate
    from owned_production import supervise
    started=time.monotonic()
    budget=SharedCleanup(started,**contract()["seconds"])
    worker_capture=audit_capture=assessment=execution=None
    errors=[]
    parent_final={"schema":"confirmation_parent_final.v2","classification":"INCONCLUSIVE_STUDY",
        "audit_completed":False,"external_capture_overrides_worker":True,"scientific_failures":[],
        "execution":None,"unconditional_closeout_attempted":True}
    try:
        execution=authenticate()["execution"]
        parent_final["execution"]=execution
        require(bounds()["evidence_bytes"]<=280*1024**2,"reserve independent parent closeout before permission")
        write_new("CONTROLLER_STARTED.json",{"execution":execution,"started_monotonic":started,
            "production_seconds":contract()["seconds"],"absolute_envelope":budget.absolute_end,
            "shared_cleanup_seconds":15,"outer_substantive_cutoff":outer_cutoff,"attempts":1})
        identity=json.loads((HERE/"OWNED_IDENTITY.json").read_bytes())
        for lane in ("worker","audit"):
            require(time.monotonic()<outer_cutoff,"single batch substantive cutoff")
            deadline,cleanup_end=budget.deadlines(time.monotonic(),lane,outer_cutoff)
            value=supervise("production_"+lane,deadline,cleanup_end,identity)
            if lane=="worker": worker_capture=value
            else: audit_capture=value
            budget.charge(lane,time.monotonic()-value["cleanup_started_monotonic"])
            require(value["quiescent"],"no successor while any owned writer lives")
            if lane=="worker":
                raw=control_file("WORKER_RESULT.json").read_bytes()
                terminal=json.loads(raw)["terminal"]
                parent_final["scientific_failures"]=terminal["state"]["scientific_failures"]
                write_new("CLOSED_WORKER_BINDING.json",{"execution":execution,"worker_result_sha256":sha(raw),
                    "owned_capture_sha256":sha(control_file("owned/production_worker/CAPTURE.json").read_bytes()),
                    "quiescent":True,"good_capture":good_capture(value)},critical=True)
        assessment=json.loads(control_file("AUDIT_RESULT.json").read_bytes())
        require(assessment["execution"]==execution,"same audited mode")
        if good_capture(worker_capture) and good_capture(audit_capture) and assessment["audit_completed"] and time.monotonic()<=budget.absolute_end:
            parent_final["classification"]=assessment["classification"]
            parent_final["audit_completed"]=True
    except BaseException:
        errors.append("CONTROLLER_OR_CAPTURE_FAILURE")
    finally:
        parent_final.update(controller_errors=errors,shared_cleanup=budget.record(),
            elapsed_seconds=time.monotonic()-started,finished_monotonic=time.monotonic(),
            worker_quiescent=bool(worker_capture and worker_capture.get("quiescent")),
            audit_quiescent=bool(audit_capture and audit_capture.get("quiescent")),
            worker_result_sha256=sha(control_file("WORKER_RESULT.json").read_bytes()) if control_file("WORKER_RESULT.json").exists() else None,
            audit_result_sha256=sha(control_file("AUDIT_RESULT.json").read_bytes()) if control_file("AUDIT_RESULT.json").exists() else None)
        # Dedicated native/exclusive parent channel, independent of the failed
        # EvidenceWriter and hook receipt path. Missing final receipt cannot PASS.
        write_new("PARENT_FINAL.json",parent_final,critical=True)
    return parent_final


def worker(deadline):
    from authority import authenticate
    execution=authenticate()["execution"]
    result=None
    try:
        from bind_production import bind
        engine,_=bind()
        result=engine.execute("attempt","production",deadline)
    except BaseException:
        result={"path":None,"sha256":None,"status":"INCOMPLETE","terminal":{
            "schema":"confirmation_terminal.v2","execution":execution,"state":{"status":"INCONCLUSIVE_STUDY",
            "scientific_failures":[],"technical_failures":[{"code":"WORKER_ENTRY_FAILURE"}]},
            "recorder_status":None,"accounting_available":False,"complete_evidence":False}}
    finally:
        write_new("WORKER_RESULT.json",result,critical=True)
    # This exit means a terminal record was durably produced, not science PASS.
    return 0


def audit(deadline):
    from authority import authenticate
    execution=authenticate()["execution"]
    result={"execution":execution,"audit_completed":False,"classification":"INCONCLUSIVE_STUDY"}
    try:
        raw=control_file("WORKER_RESULT.json").read_bytes()
        binding=json.loads(control_file("CLOSED_WORKER_BINDING.json").read_bytes())
        require(binding["execution"]==execution and binding["worker_result_sha256"]==sha(raw) and binding["quiescent"] is True and
            binding["owned_capture_sha256"]==sha(control_file("owned/production_worker/CAPTURE.json").read_bytes()),"authoritative retained closed worker status")
        capture=json.loads(raw)
        require(capture["terminal"]["execution"]==execution,"terminal provenance")
        require(capture["path"] is not None,"no index means unavailable independent evidence")
        from bind_production import bind
        _,judge=bind(include_engine=False)
        result=judge.judge(run_dir()/"evidence"/"attempt",capture,deadline)
        require(result["execution"]==execution and result["audit_completed"] is True,"independent audit completed")
    except BaseException:
        result.update(audit_completed=False,classification="INCONCLUSIVE_STUDY",audit_error="SAVED_AUDIT_FAILURE")
    finally: write_new("AUDIT_RESULT.json",result,critical=True)
    return 0 if result["audit_completed"] else 1
