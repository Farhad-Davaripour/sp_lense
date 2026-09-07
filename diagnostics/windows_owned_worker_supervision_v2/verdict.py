"""Fail-closed verdicts computed only from completed assertions and cleanup."""
import math

def live_verdict(row):
    failures=[]
    for key in ("primary_exception","error","final_cleanup_errors","cleanup_faults","reader_faults","cap_errors"):
        if row.get(key):failures.append(key)
    for key in ("assertions_passed","authenticated_before_permission","actual_handle_retained_from_authentication",
                "all_threads_joined","pipes_closed","stdout_EOF","stderr_EOF","quiescent","evidence_cap_ok"):
        if row.get(key) is not True:failures.append(key)
    before=row.get("before_cleanup",{})
    if before.get("launcher_exited") is not False or before.get("actual_worker_exited") is not False or before.get("quiescent") is not False:
        failures.append("both_live_before_cleanup")
    if row.get("actual_retained_termination_events")!=1:failures.append("exactly_one_retained_termination")
    if row.get("worker_exit_code")!=125:failures.append("worker_exit125_not_watchdog")
    if row.get("launcher_exit_code") not in (0,125):failures.append("launcher_exit")
    faults=row.get("faults")
    if not isinstance(faults,list) or [x.get("kind") for x in faults]!=["deadline"]:failures.append("only_expected_deadline_fault")
    elapsed=row.get("elapsed_seconds")
    if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or not 0<=elapsed<=20:failures.append("supervisor_cap")
    return {"status":"PASS_EXPECTED_DEADLINE_CLEANUP" if not failures else "INCONCLUSIVE", "failures":failures}

def batch_verdict(live,pure_rows,reused_count,receipt_checked,error=None):
    failures=[]
    if error:failures.append("batch_exception")
    if receipt_checked is not True:failures.append("durable_live_receipt_not_checked")
    if reused_count!=12:failures.append("inherited_fixture_authentication")
    if len(pure_rows)!=9 or any(x.get("status")!="PASS" for x in pure_rows):failures.append("new_verdict_fixtures")
    derived=live_verdict(live)
    if derived["failures"]:failures.append("live_receipt_failed")
    return {"status":"PASS_FAKE_ONLY" if not failures else "INCONCLUSIVE", "failures":failures,"independently_derived_live":derived}
