"""The three missing failure cases only; each saved judge runs before next case."""
import json
import time
from support import HERE, EVIDENCE, bounds, check_freeze, require, write_new


def run(deadline):
    started = time.monotonic()
    lock = json.loads((HERE/"LOCK.json").read_bytes())
    receipt = {"status":"INCONCLUSIVE_DELTA_WORKER","cases":[],"error":None,
        "real_model_calls":0,"tokenizer_calls":0,"author_input_reads":0,"complete_success_repeated":False,"routing_prefix_repeated":False}
    try:
        from reuse import bind
        workflow,judge = bind()
        for case in lock["selected_cases"]:
            name,mode = case["name"],case["mode"]
            require(time.monotonic() < deadline, "outer absolute work cutoff")
            write_new("stages/"+name+"_started.json",{"stage":"WORKER_STARTED","monotonic":time.monotonic(),"deadline":deadline})
            capture = workflow.execute(name,mode,deadline)
            write_new("stages/"+name+"_capture.json",capture)
            require(time.monotonic() < deadline, "audit must start before outer cutoff")
            result = judge.judge(EVIDENCE/name,capture,deadline)
            require(result["classification"] == case["expected"] and result["forward_attempts"] == case["forward_attempts"]
                    and result["derivatives"] == case["derivatives"], "selected independent failure/call outcome")
            if mode == "finite_eligibility":
                require(result["io_status"] == "COMPLETE" and result["routes"] == 1 and result["self_endpoints"] == 0
                    and result["cell_counts"] == {"DONE":1,"UNRUN":179} and result["request_counts"] == {"UNRUN":48},
                    "finite eligibility scientific stop leaves every request UNRUN")
                require(result["scientific_failures"] == [{"kind":"finite_eligibility","cell_id":"fake_n01_self_keep_first__baseline"}]
                        and result["technical_failures"] == [], "finite failure is scientific, not technical")
            elif mode == "endpoint_corruption":
                require(result["io_status"] == "COMPLETE" and result["routes"] == 25 and result["self_endpoints"] == 1
                    and result["strict_flips"] == result["retentions"] == result["off_identities"] == 0
                    and result["cell_counts"] == {"DONE":28,"SKIPPED":6,"UNRUN":146}
                    and result["request_counts"] == {"FAILED":1,"UNRUN":47}, "corrupted replay never a successful flip/retention")
                require(result["scientific_failures"] == [] and any(s.startswith("endpoint_identity:") for s in result["technical_failures"]),
                    "independent endpoint corruption finding")
            else:
                require(result["io_status"] == "INCOMPLETE" and result["forward_completed"] == result["routes"] == result["self_endpoints"] == 0
                    and result["cell_counts"] == {"FAILED":1,"UNRUN":179} and result["request_counts"] == {"UNRUN":48},
                    "partial attempt consumes one forward, not a completed result")
                index = json.loads((EVIDENCE/name/capture["path"]).read_bytes())
                records = [r for r in index["reconciliation"]["files"] if r.get("logit")]
                require(len(records) == 1 and records[0]["actual_bytes"] == 4096 and records[0]["expected_bytes"] == 993280
                    and records[0]["complete"] is False and index["sticky_failure"] is True
                    and index["reconciliation"]["reserved"]["used_bytes"]["logits"] == 993280
                    and capture["remaining_closeout_bytes"] > 0, "partial bytes, full attempted reservation and reserved failure closeout")
            require(time.monotonic() < deadline, "completed audit before outer cutoff")
            item = {"name":name,"test_status":"PASS","capture":capture,"judge":result,"completed_monotonic":time.monotonic()}
            write_new("stages/"+name+"_judged.json",item)
            receipt["cases"].append(item)
            bounds()
        check_freeze()
        receipt["status"] = "PASS_FAILURE_COVERAGE_SYNTHETIC_ONLY"
    except BaseException as error:
        receipt["error"] = {"type":type(error).__name__,"message":str(error)}
    finally:
        receipt["elapsed_seconds"] = time.monotonic()-started
        receipt["returned_monotonic"] = time.monotonic()
        write_new("WORKER_RESULT.json",receipt,critical=True)
    print(json.dumps({"status":receipt["status"],"cases":len(receipt["cases"])}),flush=True)
    return 0 if receipt["status"] == "PASS_FAILURE_COVERAGE_SYNTHETIC_ONLY" else 1
