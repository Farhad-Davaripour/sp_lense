"""Disabled production ownership schedule; budgets come from one pinned contract."""
import json
import time
from support import HERE, REAL_EVIDENCE, SOURCES, check_freeze, require, sha, write_new


def contract():
    raw = SOURCES.read("2620f66d4d50456c88800554bc9a26845998cf50","diagnostics/semantic_confirmation_resource_v1/contract.json")
    value = json.loads(raw)
    require(value["seconds"] == {"worker":1800,"cleanup":15,"audit":180}, "no inherited 300/90 timing")
    return value


def controller():
    from production_admission import admit_production
    admitted = admit_production("worker")  # Committed authorization FALSE stops here.
    limits = contract()["seconds"]
    start = time.monotonic()
    absolute_end = start+limits["worker"]+limits["audit"]+limits["cleanup"]
    write_new("PRODUCTION_START.json",{"observed_unix_seconds":time.time(),"started_monotonic":start,
        "usage_sha256":admitted["release"]["usage_sha256"],"binding":admitted["binding"],"maximum_attempts":1,
        "absolute_production_envelope":absolute_end})
    from owned_production import supervise
    identity = json.loads((HERE/"OWNED_IDENTITY.json").read_bytes())
    worker = supervise("production_worker",start+limits["worker"],start+limits["worker"]+limits["cleanup"],identity)
    write_new("PRODUCTION_WORKER_CAPTURE.json",worker,critical=True)
    require(worker["quiescent"], "no audit while worker/writers may live")
    # The audit receives at most 180 seconds after worker quiescence, and cannot
    # extend the single 1800+180+15 envelope or create a second cleanup allowance.
    audit_start = time.monotonic()
    audit_deadline = min(audit_start+limits["audit"],absolute_end-limits["cleanup"])
    audit = supervise("production_audit",audit_deadline,absolute_end,identity)
    write_new("PRODUCTION_AUDIT_CAPTURE.json",audit,critical=True)
    require(audit["quiescent"], "all audit processes and writers quiescent")
    good = lambda x: (x["binding_authenticated"] and x["stop_reason"] is None and not x["faults"]
        and not x["cleanup_faults"] and not x["stdout_capture_errors"] and x["primary_error"] is None
        and x["threads_joined"] and x["pipes_closed"] and x["within_absolute_cleanup_deadline"]
        and all(x["exit_proofs"][k]["exit_code"] == 0 for k in ("actual_worker","launcher")))
    assessment = json.loads((HERE/"PRODUCTION_AUDIT.json").read_bytes()) if (HERE/"PRODUCTION_AUDIT.json").exists() else None
    final = assessment["classification"] if good(worker) and good(audit) and assessment and time.monotonic() <= absolute_end else "INCONCLUSIVE_STUDY"
    write_new("PRODUCTION_FINAL.json",{"classification":final,"external_capture_overrides_worker":True,
        "worker_sha256":sha((HERE/"PRODUCTION_WORKER_CAPTURE.json").read_bytes()),
        "audit_sha256":sha((HERE/"PRODUCTION_AUDIT_CAPTURE.json").read_bytes()),"elapsed_seconds":time.monotonic()-start},critical=True)
    return 0 if final == "PASS_STUDY" else 1


def worker(deadline):
    from production_admission import admit_production
    admit_production("worker")
    from bind_production import bind
    engine,_ = bind()
    capture = engine.execute("production_attempt","production",deadline)
    write_new("PRODUCTION_CAPTURE.json",capture,critical=True)
    return 0 if capture["status"] == "COMPLETE" else 1


def audit(deadline):
    from production_admission import admit_production
    admit_production("audit")
    from bind_production import bind
    _,judge = bind(include_engine=False)
    require((HERE/"PRODUCTION_CAPTURE.json").exists(), "missing worker closeout means INCONCLUSIVE, never invented completion")
    capture = json.loads((HERE/"PRODUCTION_CAPTURE.json").read_bytes())
    finding = judge.judge(REAL_EVIDENCE/"production_attempt",capture,deadline)
    write_new("PRODUCTION_AUDIT.json",finding,critical=True)
    return 0 if finding["classification"] == "PASS_STUDY" else 1


if __name__ == "__main__":
    raise SystemExit(controller())
