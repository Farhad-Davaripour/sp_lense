"""Authoritative post-process join; raw saved-judge verdict/report stay separate."""
import json,math
from pathlib import Path
from core import Budget,read,require,sha

def capture_valid(capture):
    return (isinstance(capture,dict) and capture.get("status")=="complete_valid" and capture.get("quiescent") is True
        and type(capture.get("worker_exit_code")) is int and capture["worker_exit_code"]==0 and capture.get("eof_observed") is True
        and all(capture.get(k) is True for k in ("worker_joined","reader_joined","budget_watcher_joined","fault_writer_joined","prefix_reader_joined"))
        and isinstance(capture.get("owned_worker"),dict)
        and all(capture["owned_worker"].get(k) is True for k in ("binding_authenticated","handshake_thread_joined","evidence_threads_joined","pipes_closed","quiescent"))
        and capture["owned_worker"].get("actual_worker_exit_code")==0 and capture["owned_worker"].get("launcher_exit_code")==0
        and not capture["owned_worker"].get("faults") and not capture["owned_worker"].get("cleanup_faults") and not capture["owned_worker"].get("evidence_errors")
        and capture.get("technical_recording_fault") is None and capture.get("cleanup_error") is None and capture.get("fault_persistence_error") is False)

def join(worker_capture,audit_capture,judge_receipt,judge_result):
    reasons=[]
    if not capture_valid(worker_capture):reasons.append("worker_external_capture_invalid")
    if not capture_valid(audit_capture):reasons.append("audit_external_capture_invalid")
    elapsed=judge_receipt.get("elapsed_seconds") if isinstance(judge_receipt,dict) else None
    complete=(isinstance(judge_receipt,dict) and judge_receipt.get("status")=="complete" and "error" not in judge_receipt
        and type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>=0)
    if not complete:reasons.append("completed_judge_receipt_missing_or_malformed")
    raw=judge_result.get("classification") if isinstance(judge_result,dict) else None
    if raw not in ("PASS","FAIL","INCONCLUSIVE"):reasons.append("judge_verdict_missing_or_malformed")
    return {"classification":"INCONCLUSIVE" if reasons else raw,"raw_judge_classification":raw,
        "reasons":reasons,"worker_capture_status":worker_capture.get("status") if isinstance(worker_capture,dict) else None,
        "audit_capture_status":audit_capture.get("status") if isinstance(audit_capture,dict) else None,
        "scientific_findings":judge_result.get("independently_derived_scientific_failures",judge_result.get("durable_scientific_failures_not_discarded",[])) if isinstance(judge_result,dict) else [],
        "assessment_status":judge_result.get("assessment_status") if isinstance(judge_result,dict) else None,"execution_validity":judge_result.get("execution_validity") if isinstance(judge_result,dict) else None,
        "raw_judge_result_preserved":True,"external_fault_overrides_stale_pass":bool(reasons)}

def finalize(output,worker_capture,audit_capture):
    output=Path(output);budget=Budget(output)
    require(worker_capture.get("quiescent") is True and audit_capture.get("quiescent") is True,"no final writer until both process trees/writers are quiescent")
    errors=[]
    def optional(name):
        try:return read(output/name)
        except (OSError,ValueError) as error:errors.append({"artifact":name,"error":type(error).__name__+": "+str(error)[:1024]});return None
    receipt=optional("judge_receipt.json");result=optional("judge_results.json")
    final=join(worker_capture,audit_capture,receipt,result);final["read_errors"]=errors
    final["preserved_artifacts"]={name:{"sha256":sha((output/name).read_bytes()),"bytes":(output/name).stat().st_size} for name in
        ("judge_results.json","judge_receipt.json","judge_report.md","scientific_failures.jsonl","technical_faults.jsonl","cleanup_errors.jsonl","unrun.json") if (output/name).exists()}
    final["authoritative"]="final_closeout.json and final REPORT.md; raw judge output alone is never the final verdict"
    budget.write("final_closeout.json",final)
    raw_counts=result.get("counts",{}) if isinstance(result,dict) else {}
    report=("Final assessment: "+str(final["classification"])+"\nRaw saved-judge verdict: "+str(final["raw_judge_classification"])+
        "\nAssessment status: "+str(final["assessment_status"])+"; execution validity: "+str(final["execution_validity"])+
        "\nExternal/adjudication faults: "+json.dumps(final["reasons"])+"\nRaw judged counts (not overrides): "+json.dumps(raw_counts)+
        "\nScientific findings retained: "+json.dumps(final["scientific_findings"])+
        "\nSee final_closeout.json and immutable raw judge/technical/cleanup/UNRUN artifacts. One census-selected exposed f03/v2 STOP-first C request; P absent/UNTESTED. Only a verified actual KEEP-second to STOP-first flip counts; no opportunity is not success; no ordinary-preservation, double-direction or broad reliability claim.\n")
    budget.write_bytes("REPORT.md",report.encode())
    return final
