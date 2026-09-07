"""ONE new failure-only delta: 80s work cutoff, 10s closeout, +15s cleanup."""
import json
import time
from support import HERE, bounds, check_freeze, require, sha, write_new


def main():
    started = time.monotonic()
    work_deadline,closeout_deadline,cleanup_deadline = started+80.,started+90.,started+105.
    lock = json.loads((HERE/"LOCK.json").read_bytes())
    check_freeze()
    usage = json.loads((HERE/"USAGE_BEFORE_BATCH.json").read_bytes())
    require(type(usage["used_percent"]) is int and usage["used_percent"] < 100, "known available usage before batch")
    write_new("BATCH_STARTED.json",{"started_monotonic":started,"work_deadline":work_deadline,
        "closeout_deadline":closeout_deadline,"cleanup_deadline":cleanup_deadline,"maximum_runs":1,
        "source_freeze_sha256":sha((HERE/"SOURCE_FREEZE.json").read_bytes()),"usage":usage,"real_run_authorized":False})
    receipt = {"status":"INCONCLUSIVE_FAILURE_DELTA","error":None,"captures":{},"case_dispositions":[],
        "real_model_calls":0,"tokenizer_calls":0,"author_input_reads":0,"milestone_credit":False,
        "prior_v1_status":"INCONCLUSIVE_PREPARATION_ONLY","prior_v1_elapsed_seconds":121.438,
        "prior_v1_unchanged":True,"complete_success_repeated":False,"routing_prefix_repeated":False}
    try:
        from supervisor import supervise
        require(time.monotonic() < work_deadline, "outer work time available")
        stall_deadline = min(time.monotonic()+lock["forced_stall"]["seconds_including_bootstrap"],work_deadline)
        stall = supervise("forced_stall",stall_deadline,min(stall_deadline+15.,cleanup_deadline),lock["owned_identity"])
        receipt["captures"]["forced_stall"] = stall
        require(stall["quiescent"] and stall["binding_authenticated"] and stall["stop_reason"] == "deadline"
            and stall["threads_joined"] and stall["pipes_closed"] and not stall["cleanup_faults"]
            and stall["primary_error"] is None and not stall["stdout_capture_errors"], "authoritative forced-stall stop and quiescence")
        require((HERE/"owned/forced_stall/STALL_READY.json").exists(), "forced stall actually entered")
        require(stall["exit_proofs"]["actual_worker"]["exit_code"] == 125 and stall["termination_attempts"]
            and stall["stop_episode"]["completed_at"] <= stall["stop_episode"]["deadline"]
            and not stall["stop_episode"]["overrun"] and stall["within_absolute_cleanup_deadline"], "retained termination, not emergency watchdog or EOF")
        receipt["forced_stall_test"] = "PASS_EXPECTED_DEADLINE_TERMINATION"
        print(json.dumps({"case":"forced_stall","status":"PASS","elapsed_seconds":time.monotonic()-started}),flush=True)
        require(time.monotonic() < work_deadline, "no workload launch after cutoff")
        worker = supervise("failure_worker",work_deadline,cleanup_deadline,lock["owned_identity"])
        receipt["captures"]["failure_worker"] = worker
        require(worker["quiescent"] and worker["binding_authenticated"] and worker["stop_reason"] is None
            and not worker["faults"] and not worker["cleanup_faults"] and worker["primary_error"] is None
            and not worker["stdout_capture_errors"] and worker["threads_joined"] and worker["pipes_closed"], "normal owned failure-worker completion")
        require(all(worker["exit_proofs"][key]["exit_code"] == 0 for key in ("actual_worker","launcher")), "both retained normal exit codes")
        require(time.monotonic() <= closeout_deadline, "closeout headroom inside 90-second total")
        result = json.loads((HERE/"WORKER_RESULT.json").read_bytes())
        require(result["status"] == "PASS_FAILURE_COVERAGE_SYNTHETIC_ONLY" and len(result["cases"]) == 3, "all three independent failure cases completed")
        require(all(item["completed_monotonic"] < work_deadline for item in result["cases"]), "all substantive audits before cutoff")
        receipt["worker_result_sha256"] = sha((HERE/"WORKER_RESULT.json").read_bytes())
        check_freeze()
        receipt["status"] = "PASS_FAILURE_DELTA_SYNTHETIC_ONLY"
    except BaseException as error:
        receipt["error"] = {"type":type(error).__name__,"message":str(error)}
    finally:
        # No new fake execution or saved audit after the cutoff. Stage existence
        # never promotes an unaudited worker finding to a scientific PASS.
        all_quiet = all(c.get("quiescent") for c in receipt["captures"].values()) and bool(receipt["captures"])
        for case in lock["selected_cases"]:
            name = case["name"]
            begun = HERE/("stages/"+name+"_started.json")
            captured = HERE/("stages/"+name+"_capture.json")
            judged = HERE/("stages/"+name+"_judged.json")
            receipt["case_dispositions"].append({"name":name,"worker_stage":"RETURNED" if captured.exists() else "INCOMPLETE" if begun.exists() else "UNRUN",
                "independent_audit_stage":"RETURNED" if judged.exists() else "UNRUN",
                "independent_test_status":"PASS" if judged.exists() and all_quiet else "UNVERIFIED",
                "judge_receipt_sha256":sha(judged.read_bytes()) if judged.exists() else None})
        receipt["storage"] = bounds()
        receipt.update(finished_monotonic=time.monotonic(),elapsed_seconds=time.monotonic()-started,
            work_cutoff_seconds=80,closeout_limit_seconds=90,cleanup_limit_seconds=105,
            within_90_second_total=time.monotonic() <= closeout_deadline,
            within_105_second_cleanup=time.monotonic() <= cleanup_deadline,
            all_captured_processes_quiescent=all_quiet)
        if not receipt["within_90_second_total"] or not receipt["within_105_second_cleanup"] or not all_quiet:
            receipt["status"] = "INCONCLUSIVE_FAILURE_DELTA"
        write_new("TEST_REPORT.json",receipt,critical=True)
        # Actual closeout write time is retained separately, never hidden by the
        # pre-write elapsed field in TEST_REPORT.json.
        final_time = time.monotonic()
        write_new("FINAL_RETURN.json",{"elapsed_seconds_after_test_report":final_time-started,
            "within_90_seconds":final_time <= closeout_deadline,"within_105_seconds":final_time <= cleanup_deadline,
            "authoritative_status":receipt["status"] if final_time <= closeout_deadline else "INCONCLUSIVE_FAILURE_DELTA",
            "report_sha256":sha((HERE/"TEST_REPORT.json").read_bytes()),"no_work_after_cutoff":True},critical=True)
    print(json.dumps({"status":receipt["status"],"elapsed_seconds":time.monotonic()-started,
        "cases":receipt["case_dispositions"],"error":receipt["error"]}),flush=True)
    return 0 if receipt["status"] == "PASS_FAILURE_DELTA_SYNTHETIC_ONLY" and time.monotonic() <= closeout_deadline else 1


if __name__ == "__main__":
    raise SystemExit(main())
