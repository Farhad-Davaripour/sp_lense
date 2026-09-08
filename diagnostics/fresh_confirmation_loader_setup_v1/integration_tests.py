"""One prospectively locked owned startup-only batch. No model packages allowed."""
import copy
import json
import os
import sys
import threading
import time
from support import HERE,ROOT,require,sha,bounds,check_freeze,write_new
from real_boundary import Boundary,encoded,strict,write_exclusive,usage_value

def main():
    require(len(sys.argv)==2,"explicit caller batch lock")
    lockraw=(HERE/"BATCH_LOCK.json").read_bytes();require(sha(lockraw)==sys.argv[1],"caller batch SHA")
    lock=strict(lockraw)
    for name,field in (("SOURCE_FREEZE.json","source_sha256"),("TEST_INPUTS.json","test_inputs_sha256"),("TEST_PROTOCOL.json","test_protocol_sha256")):
        require(sha((HERE/name).read_bytes())==lock[field],"prospectively locked test bytes")
    check_freeze()
    for item in strict((HERE/"REUSED_PROOFS.json").read_bytes())["files"]:
        data=(ROOT/item["path"]).read_bytes();require(len(data)==item["bytes"] and sha(data)==item["sha256"],"reused immutable proof bytes")
    usage=strict((HERE/"USAGE_BEFORE_BATCH.json").read_bytes())
    require(0<=time.time()-usage["observed_unix_seconds"]<=120 and usage_value(usage["tool_result"])["used_percent"]<100,"fresh actual standard usage")
    from production_run import controller,forbid_research_imports
    from setup_budget import Budget
    from launch import preflight,terminal
    forbid_research_imports()
    protocol=strict((HERE/"TEST_PROTOCOL.json").read_bytes());inputs=strict((HERE/"TEST_INPUTS.json").read_bytes())
    started=time.monotonic();budget=Budget(started,testing=True)
    write_new("BATCH_STARTED.json",{"source_sha256":lock["source_sha256"],"batch_lock_sha256":sys.argv[1],
        "started_monotonic":started,"substantive_cutoff":budget.substantive_end,"absolute_deadline":budget.absolute_end},preparation=True,critical=True)
    hard=threading.Timer(max(0,budget.absolute_end-time.monotonic()-.5),lambda:os._exit(124));hard.daemon=True;hard.start()
    report={"schema":"owned_model_free_setup_batch.v1","status":"INCONCLUSIVE_MODEL_FREE_SETUP",
        "cases":[{"name":name,"status":"UNRUN"} for name in protocol["cases"]],"production_authorized":False,
        "actual_real_loads":0,"actual_forwards":0,"actual_derivatives":0,"actual_tokenizer_calls":0,
        "failure":None,"source_sha256":lock["source_sha256"],"batch_lock_sha256":sys.argv[1]}
    active=None
    try:
        disabled=preflight("0"*64)
        require(disabled["status"]=="DISABLED_NO_APPROVED_REAL_RELEASE" and not (HERE/"root_release").exists()
            and not (HERE/"real_evidence").exists(),"actual disabled-state preflight")
        write_new("DISABLED_STATE.json",disabled,preparation=True,critical=True)
        for case in report["cases"]:
            active=case;name=case["name"]
            require(time.monotonic()<budget.substantive_end,"one batch substantive cutoff")
            root=HERE/"test_evidence"/name;root.mkdir(parents=True,exist_ok=False)
            release=copy.deepcopy(inputs["release"]);auth=copy.deepcopy(inputs["authorization"])
            usage_raw=encoded(inputs["usage"]);release["usage_sha256"]=sha(usage_raw);release_raw=encoded(release)
            auth["release_sha256"]=sha(release_raw);auth_raw=encoded(auth)
            authority={"schema":"real_root_authority_lock.v1","source_sha256":lock["source_sha256"],"release_sha256":sha(release_raw),"authorization_sha256":sha(auth_raw)}
            authority_raw=encoded(authority)
            for filename,data in (("ROOT_RELEASE.json",release_raw),("AUTHORIZATION.json",auth_raw),("AUTHORITY_LOCK.json",authority_raw),("USAGE_RECEIPT.json",usage_raw)):
                write_exclusive(root/"root_release"/filename,data,root)
            boundary=Boundary(root,mock=True);approved=sha(authority_raw);admission,identity=boundary.admit_once(approved,now=1060.)
            os.environ.update(SP_SETUP_FIXTURE=name,SP_SETUP_BATCH_LOCK=sys.argv[1],
                SP_CONFIRMATION_AUTHORITY_SHA=approved,SP_CONFIRMATION_ADMISSION_SHA=admission)
            result=None
            try: result=controller(budget,name)
            finally: terminal(boundary,identity,result)
            assessment=strict(boundary.read_control("AUDIT_RESULT.json"))
            worker=strict(boundary.read_control("WORKER_RESULT.json"))
            observed=strict(boundary.read_control("LOADER_TERMINAL.json"))
            require(result["classification"]==protocol["expectations"][name]["classification"]
                and result["audit_completed"] and result["worker_quiescent"] and result["audit_quiescent"]
                and not result["controller_errors"] and result["scientific_pass"] is False,"owned expected classification and quiescence")
            require(observed["observed_model_work"]=="SENTINEL_ONLY" and not identity["execution"]["production_authorized"],"inert provenance never real evidence")
            require(assessment["audit_completed"] and assessment["scientific_cells_unrun"]==180 and assessment["requests_unrun"]==48
                and assessment["baselines_unrun"]==24 and assessment["scientific_pass"] is False,"all science remains UNRUN")
            if name!="missing_terminal":
                saved=strict(boundary.read_control("SETUP_TERMINAL.json"))
                require(saved["counts"]["actual_load_dispatches"]==1 and saved["counts"]["forwards"]==saved["counts"]["derivatives"]==0
                    and saved["guard"]["restored"] is True,"one inert load zero actual dispatches restored guard")
                if name=="constructor_failure":
                    require(saved["first_failure"]["predicate_code"]=="LD_ORDERED_WEIGHT_DIGEST"
                        and saved["recorder_status"]["terminal"] is True and saved["loader_diagnostics"]["complete_diagnostic_evidence"],"constructor first cause and retained setup recorder")
                if name=="partial_publisher":
                    require(len(boundary.read_control("LOADER_DIAGNOSTICS.json"))==7 and saved["loader_diagnostics"]["diagnostic_io_failed"]
                        and saved["loader_diagnostics"]["receipt_sha256"] is None and not saved["loader_diagnostics"]["complete_diagnostic_evidence"],"native partial bytes retained no acknowledgement or retry")
                if name in ("forbidden_forward","forbidden_derivative"):
                    require(saved["guard"]["first_dispatch_code"]=="LOAD_OR_SETUP_FAILURE" and saved["cleanup"] is None,"inherited pending hook latch denies before body")
                if name=="second_load": require(saved["counts"]["load_dispatch_attempts_including_denied"]==2 and saved["counts"]["stop_reason"]=="SECOND_LOAD","second actual body denied")
                if name=="cleanup_failure": require(saved["cleanup"] is None and "SETUP_CLEANUP_FAILURE" in saved["failures"] and saved["recorder_status"]["terminal"],"cleanup failure terminal despite successful load")
                case["counts"]=saved["counts"];case["first_failure"]=saved["first_failure"]
            else: require(assessment["first_failure"]=="MISSING_SETUP_TERMINAL" and not boundary.control_path("SETUP_TERMINAL.json").exists(),"missing terminal cannot complete")
            case.update(status="PASS",classification=result["classification"],parent_final_sha256=sha(boundary.read_control("PARENT_FINAL.json")),
                audit_sha256=sha(boundary.read_control("AUDIT_RESULT.json")),mock_loads=1,actual_real_loads=0)
            bounds();require(time.monotonic()<budget.substantive_end,"bounded audit included in batch")
        require(budget.used<=15 and len(budget.charges)==16,"one cleanup budget for all sixteen owned lanes")
        report["status"]="PASS_MODEL_FREE_SETUP_ONLY"
    except BaseException:
        report["failure"]="FROZEN_SETUP_BATCH_FAILURE"
        if active is not None: active["status"]="FAIL"
    finally:
        for key in ("SP_SETUP_FIXTURE","SP_SETUP_BATCH_LOCK","SP_CONFIRMATION_AUTHORITY_SHA","SP_CONFIRMATION_ADMISSION_SHA"): os.environ.pop(key,None)
        report.update(elapsed_seconds=time.monotonic()-started,shared_cleanup=budget.record(),area=bounds())
        write_new("BATCH_REPORT.json",report,preparation=True,critical=True);hard.cancel()
    print(json.dumps(report,sort_keys=True))
    return 0 if report["status"]=="PASS_MODEL_FREE_SETUP_ONLY" else 1

if __name__=="__main__": raise SystemExit(main())
