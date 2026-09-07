"""One adapter-only tensor injection batch under retained-handle ownership."""
import json
import time
from support import HERE, SOURCES, bounds, check_freeze, require, sha, write_new


def main():
    start = time.monotonic()
    deadline,end = start+45.,start+60.
    check_freeze()
    from production_admission import usage_value
    usage = json.loads((HERE/"USAGE_BEFORE_BATCH.json").read_bytes())
    require(usage_value(usage["tool_result"])["used_percent"] < 100 and 0 <= time.time()-usage["observed_unix_seconds"] <= 60,
            "fresh available unexhausted usage immediately before adapter batch")
    write_new("BATCH_STARTED.json",{"started_monotonic":start,"work_deadline":deadline,"absolute_envelope":end,
        "runs":1,"source_freeze_sha256":sha((HERE/"SOURCE_FREEZE.json").read_bytes()),"production_authorized":False})
    result = {"status":"INCONCLUSIVE_ADAPTER_ONLY","error":None,"checks":"UNRUN",
              "real_model_loads":0,"tokenizer_loads":0,"real_forwards":0,"production_authorized":False}
    try:
        owner = SOURCES.load("adapter_only_owned", "23ce46f1e5177791751b08bd15de07f9d836ecf8",
            "diagnostics/fresh_confirmation_failure_delta_v1/supervisor.py")
        capture = owner.supervise("adapter_worker",deadline,end,json.loads((HERE/"OWNED_IDENTITY.json").read_bytes()))
        result["capture"] = capture
        require(capture["quiescent"] and capture["binding_authenticated"] and capture["stop_reason"] is None
            and not capture["faults"] and not capture["cleanup_faults"] and not capture["stdout_capture_errors"]
            and capture["primary_error"] is None and capture["threads_joined"] and capture["pipes_closed"]
            and all(capture["exit_proofs"][k]["exit_code"] == 0 for k in ("actual_worker","launcher")), "owned adapter process normal quiescence")
        checks = json.loads((HERE/"ADAPTER_TEST_REPORT.json").read_bytes())
        result["checks"] = checks
        require(checks["status"] == "PASS_ADAPTER_ONLY" and checks["completed_monotonic"] < deadline
            and time.monotonic() < end, "all selected small checks within fixed envelope")
        check_freeze()
        result["status"] = "PASS_ADAPTER_ONLY"
    except BaseException:
        result["error"] = {"code":"ADAPTER_BATCH_FAILURE"}
    result.update(elapsed_seconds=time.monotonic()-start,within_60_seconds=time.monotonic() < end,storage=bounds())
    if not result["within_60_seconds"]:
        result["status"] = "INCONCLUSIVE_ADAPTER_ONLY"
    write_new("TEST_REPORT.json",result,critical=True)
    print(json.dumps({"status":result["status"],"elapsed_seconds":time.monotonic()-start}),flush=True)
    return 0 if result["status"] == "PASS_ADAPTER_ONLY" and time.monotonic() < end else 1


if __name__ == "__main__":
    raise SystemExit(main())
