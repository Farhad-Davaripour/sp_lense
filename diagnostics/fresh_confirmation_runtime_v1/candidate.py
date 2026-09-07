"""The candidate's sole enabled execution path: actual locked inputs, fake backend."""
import json
import time
from admission import admit
from support import HERE, EVIDENCE, bounds, check_freeze, require, sha, write_new


def run(deadline, execution_mode):
    require(execution_mode == "SYNTHETIC_ONLY", "production authorization is false; no real backend path")
    started = time.monotonic()
    result = {"status":"INCONCLUSIVE_CANDIDATE_FAKE","error":None,"real_model_loads":0,
        "real_forwards":0,"tokenizer_loads":0,"production_authorized":False,
        "real_hook_capacity_verified":False,"real_backend_tested":False}
    try:
        bound = admit()
        check_freeze()
        from plan import build_plan
        require(build_plan() == json.loads((HERE/"BOUND_PLAN.json").read_bytes()), "prospectively frozen actual input/schedule equality")
        require(time.monotonic() < deadline, "one absolute candidate deadline")
        from bind_runtime import bind
        engine,judge = bind()
        capture = engine.execute("candidate_success","normal",deadline)
        write_new("CANDIDATE_CAPTURE.json",capture)
        require(time.monotonic() < deadline, "independent audit before outer cutoff")
        finding = judge.judge(EVIDENCE/"candidate_success",capture,deadline)
        expected = json.loads((HERE/"SYNTHETIC_EXPECTATIONS.json").read_bytes())
        require(finding["classification"] == "PASS_SYNTHETIC_ONLY" and finding["io_status"] == "COMPLETE", "full independent candidate finding")
        require(all(finding[key] == value for key,value in expected["judge_counts"].items()), "prelocked synthetic counts")
        require(finding["cell_counts"] == {"DONE":96,"SKIPPED":84} and finding["request_counts"] == {"DONE":48}, "fixed schedule complete, no UNRUN")
        accuracy = finding["ordinary_accuracy"]
        require(accuracy["baseline"] == accuracy["P"] == accuracy["C"] == expected["ordinary_accuracy"], "same actual-cohort scoring-only ordinary errors preserved synthetically")
        require(time.monotonic() < deadline, "independent audit completed within common cutoff")
        result.update(status="PASS_CANDIDATE_FAKE_ONLY",input_binding=bound["binding"],capture=capture,judge=finding,
            source_freeze_sha256=sha((HERE/"SOURCE_FREEZE.json").read_bytes()),
            bound_plan_sha256=sha((HERE/"BOUND_PLAN.json").read_bytes()),tensor_only_backend=True,
            no_encode_fallback=True,gate_input_interface="fresh unedited state only",fixture_is_not_real_prediction=True)
    except BaseException as error:
        result["error"] = {"type":type(error).__name__,"message":str(error)}
    finally:
        result.update(elapsed_seconds=time.monotonic()-started,completed_monotonic=time.monotonic(),storage=bounds())
        write_new("CANDIDATE_RESULT.json",result,critical=True)
    print(json.dumps({"status":result["status"],"elapsed_seconds":result["elapsed_seconds"],"error":result["error"]}),flush=True)
    return 0 if result["status"] == "PASS_CANDIDATE_FAKE_ONLY" else 1
