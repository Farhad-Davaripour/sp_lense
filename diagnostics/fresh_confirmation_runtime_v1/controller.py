"""One actual-input-bound fake candidate run under the checked outer controller."""
import json
import time
from support import HERE, SOURCES, bounds, check_freeze, require, sha, write_new

DELTA_COMMIT = "23ce46f1e5177791751b08bd15de07f9d836ecf8"
DELTA_SUPERVISOR = "diagnostics/fresh_confirmation_failure_delta_v1/supervisor.py"


def main():
    started = time.monotonic()
    work_deadline,closeout_deadline,cleanup_deadline = started+170.,started+180.,started+195.
    check_freeze()
    from admission import admit
    bound = admit()
    usage = json.loads((HERE/"USAGE_BEFORE_BATCH.json").read_bytes())
    require(type(usage["used_percent"]) is int and usage["used_percent"] < 100, "known available usage")
    lock = json.loads((HERE/"RUNTIME_LOCK.json").read_bytes())
    require(lock["production_authorized"] is False, "production remains disabled")
    write_new("BATCH_STARTED.json",{"started_monotonic":started,"work_deadline":work_deadline,
        "closeout_deadline":closeout_deadline,"cleanup_deadline":cleanup_deadline,"maximum_runs":1,
        "source_freeze_sha256":sha((HERE/"SOURCE_FREEZE.json").read_bytes()),"input_binding":bound["binding"],
        "usage":usage,"execution_mode":"SYNTHETIC_ONLY","real_model_calls":0,"tokenizer_calls":0})
    result = {"status":"INCONCLUSIVE_CANDIDATE_FAKE","error":None,"capture":None,
        "real_model_calls":0,"tokenizer_calls":0,"production_authorized":False,"milestone_credit":False}
    try:
        supervisor = SOURCES.load("candidate_owned_supervisor",DELTA_COMMIT,DELTA_SUPERVISOR)
        capture = supervisor.supervise("runtime_worker",work_deadline,cleanup_deadline,lock["owned_identity"])
        result["capture"] = capture
        require(capture["quiescent"] and capture["binding_authenticated"] and capture["stop_reason"] is None
            and not capture["faults"] and not capture["cleanup_faults"] and capture["primary_error"] is None
            and not capture["stdout_capture_errors"] and capture["threads_joined"] and capture["pipes_closed"],
            "authoritative normal owned completion")
        require(all(capture["exit_proofs"][key]["exit_code"] == 0 for key in ("actual_worker","launcher")), "both retained normal exits")
        require(time.monotonic() <= closeout_deadline, "closeout within absolute 180-second total")
        worker = json.loads((HERE/"CANDIDATE_RESULT.json").read_bytes())
        require(worker["status"] == "PASS_CANDIDATE_FAKE_ONLY" and worker["completed_monotonic"] < work_deadline,
                "independently reconstructed success before work cutoff")
        result["candidate_result_sha256"] = sha((HERE/"CANDIDATE_RESULT.json").read_bytes())
        check_freeze()
        result["status"] = "PASS_CANDIDATE_FAKE_ONLY"
    except BaseException as error:
        result["error"] = {"type":type(error).__name__,"message":str(error)}
    finally:
        result.update(elapsed_seconds=time.monotonic()-started,storage=bounds(),
            within_180_seconds=time.monotonic() <= closeout_deadline,
            within_195_seconds=time.monotonic() <= cleanup_deadline,
            independent_audit_stage="RETURNED" if (HERE/"CANDIDATE_RESULT.json").exists() else "UNRUN")
        if not result["within_180_seconds"] or not result["within_195_seconds"]:
            result["status"] = "INCONCLUSIVE_CANDIDATE_FAKE"
        write_new("TEST_REPORT.json",result,critical=True)
        after = time.monotonic()
        write_new("FINAL_RETURN.json",{"elapsed_seconds_after_report":after-started,
            "within_180_seconds":after <= closeout_deadline,"within_195_seconds":after <= cleanup_deadline,
            "authoritative_status":result["status"] if after <= closeout_deadline else "INCONCLUSIVE_CANDIDATE_FAKE",
            "test_report_sha256":sha((HERE/"TEST_REPORT.json").read_bytes())},critical=True)
    print(json.dumps({"status":result["status"],"elapsed_seconds":time.monotonic()-started,"error":result["error"]}),flush=True)
    return 0 if result["status"] == "PASS_CANDIDATE_FAKE_ONLY" and time.monotonic() <= closeout_deadline else 1


if __name__ == "__main__":
    raise SystemExit(main())
