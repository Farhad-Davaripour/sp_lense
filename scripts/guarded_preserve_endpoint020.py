"""Minimal namespace wrapper: one fixed .20 condition, fresh baseline gate, 12/0."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import guarded_preserve_endpoint020_plan as protocol

job = protocol.isolate("_endpoint020_execution_core", "scripts/frozen_guarded_preserve_crossed.py")
job.protocol = protocol
for _name in ("SCRIPT", "VERIFY", "TEST", "DOC", "PLAN", "OUTPUT"):
    setattr(job, _name, getattr(protocol, _name))
require = protocol.require
CORE_EVALUATE = job.evaluate
CORE_GOALS = job.record_fresh_goals
CORE_FREEZE_CHECK = job.require_freeze
CORE_NO_RESUME = job.no_resume


def record_baseline_comparison(plan, rows, output, ledger):
    require(
        len(rows) == ledger.completed == 4 and ledger.attempts == 4,
        "all four fresh baselines before endpoint comparison",
    )
    frozen = plan["prior_comparison"]
    fields = plan["config"]["prior_comparison"]["exact_fields"]
    results = []
    for row, snapshot in zip(rows, frozen["baseline_records"], strict=True):
        prior = snapshot["row"]
        require(
            protocol.canonical_sha(prior) == snapshot["row_sha256"],
            "frozen prior baseline row hash",
        )
        require(row["phase"] == prior["phase"] == "baseline", "comparison baseline phase")
        herror = max(abs(a - b) for a, b in zip(row["h0"], prior["h0"], strict=True))
        nerror = abs(row["h0_norm"] - prior["h0_norm"])
        serror = abs(row["preserve_log_odds"] - prior["preserve_log_odds"])
        exact = all(row[k] == prior[k] for k in fields)
        passed = exact and all(math.isfinite(x) and x <= 1e-6 for x in (herror, nerror, serror))
        results.append(
            {
                "cell_id": row["cell_id"],
                "prior_row_sha256": snapshot["row_sha256"],
                "prior_raw_line_sha256": snapshot["raw_line_sha256"],
                "fresh_row_sha256": protocol.canonical_sha(row),
                "maximum_h0_difference": herror,
                "norm_difference": nerror,
                "S0_difference": serror,
                "exact_identity_and_labels": exact,
                "passed": passed,
            }
        )
    record = {
        "prior_namespace": frozen["namespace"],
        "prior_rows_sha256": frozen["artifact_sha256"]["rows.jsonl"],
        "fresh_baseline_rows_sha256": protocol.canonical_sha(rows),
        "completed_baselines": 4,
        "derivative_attempts": 0,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 0,
        "monotonic": ledger.now(),
        "comparisons": results,
        "passed": all(r["passed"] for r in results),
    }
    protocol.io.write_new(output / "baseline_comparison.json", record)
    require(record["passed"], "fresh baselines do not match fixed prior; no endpoint edits")
    return record


def evaluate(plan, backend, vectors, ledger, output):
    def after_all_baselines(rows, output, ledger):
        record_baseline_comparison(plan, rows, output, ledger)
        return CORE_GOALS(rows, output, ledger)

    saved = job.record_fresh_goals
    job.record_fresh_goals = after_all_baselines
    try:
        return CORE_EVALUATE(plan, backend, vectors, ledger, output)
    finally:
        job.record_fresh_goals = saved


def no_resume(allowed):
    return CORE_NO_RESUME(set(allowed) | {protocol.CONDITION})


def freeze():
    protocol.storage_preflight(job.ROOT, protocol.read(job.ROOT / protocol.CONFIG))
    plan = protocol.build_plan()
    record = {
        "plan": plan,
        "source_commit": job.base.git(job.ROOT, "rev-parse", "HEAD"),
        "source_sha256": job.source_identity(),
        "environment": job.base.environment(),
    }
    output = job.ROOT / job.OUTPUT
    output.mkdir(parents=True, exist_ok=False)
    protocol.io.write_new(output / protocol.CONDITION, plan["derived_condition"])
    protocol.io.write_new(output / "preregistration.json", record)
    require(
        (output / protocol.CONDITION).read_bytes()
        == protocol.serialized(plan["derived_condition"]),
        "exact one frozen serialization",
    )
    return {
        "source_commit": record["source_commit"],
        "forwards": 12,
        "derivatives": 0,
        "condition": plan["candidates"]["preserve"],
    }


def require_freeze():
    from scripts.verify_guarded_preserve_endpoint020 import independent_condition

    record = CORE_FREEZE_CHECK()
    vectors = protocol.candidates()
    require(
        vectors["preserve"] == record["plan"]["derived_condition"],
        "derived condition freeze identity",
    )
    independent_condition(record["plan"])
    path = str(Path(job.OUTPUT) / protocol.CONDITION)
    require(
        job.base.git(job.ROOT, "ls-files", "--", path)
        and not job.base.git(job.ROOT, "status", "--porcelain", "--", path),
        "clean tracked derived condition",
    )
    return record


def lock_commit(record):
    paths = sorted(
        str(Path(job.OUTPUT) / name).replace("\\", "/")
        for name in (protocol.CONDITION, "preregistration.json")
    )
    require(
        sorted(job.base.git(job.ROOT, "show", "--pretty=", "--name-only", "HEAD").splitlines())
        == paths
        and job.base.git(job.ROOT, "rev-parse", "HEAD^") == record["source_commit"]
        and not job.base.git(job.ROOT, "status", "--porcelain", "--", *paths),
        "clean prospective condition-and-preregistration-only commit required",
    )
    return job.base.git(job.ROOT, "rev-parse", "HEAD")


def run():
    job.no_resume({"preregistration.json", "PRELAUNCH_CLAIM.json", "PRELAUNCH.json"})
    protocol.check_prelaunch(job.ROOT / job.OUTPUT)
    record = job.require_freeze()
    job.lock_commit(record)
    usage = json.loads(os.environ.get("SP_LENSE_USAGE_PREFLIGHT", "null"))
    require(
        isinstance(usage, dict)
        and type(usage.get("standard_used_percent")) in (float, int)
        and 0 <= usage["standard_used_percent"] < 90
        and type(usage.get("checked_at_unix")) in (float, int)
        and 0 <= time.time() - usage["checked_at_unix"] <= 60,
        "fresh usage below90 required",
    )
    return job.supervise(
        [sys.executable, "-u", str(job.ROOT / job.SCRIPT), "_worker"], job.ROOT / job.OUTPUT, usage
    )


job.evaluate = evaluate
job.no_resume = no_resume
job.freeze = freeze
job.require_freeze = require_freeze
job.lock_commit = lock_commit
job.run = run

if __name__ == "__main__":
    if sys.argv[1:] == ["_worker"]:
        job.worker()
    elif sys.argv[1:] in (["freeze"], ["prelaunch"], ["run"]):
        stage = sys.argv[1]
        result = getattr(job, stage)()
        print(json.dumps(result, indent=2))
        if stage != "freeze" and result["status"] not in ("complete_valid", "passed"):
            raise SystemExit(1)
    else:
        raise SystemExit("Use freeze, prelaunch or run; no adjustable condition inputs.")
