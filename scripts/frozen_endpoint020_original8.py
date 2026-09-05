"""Original8 regression wrapper: fixed .20,24/0, archived baseline/goal gate."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import frozen_endpoint020_original8_plan as protocol

job = protocol.isolate(
    "_original8_endpoint020_execution_core", "scripts/frozen_guarded_preserve_crossed.py"
)
job.protocol = protocol
for _name in ("SCRIPT", "VERIFY", "TEST", "DOC", "PLAN", "OUTPUT"):
    setattr(job, _name, getattr(protocol, _name))
for _name, _mapping, _counts in (
    ("evaluate", {12: 24}, {12: 1}),
    ("summarize", {4: 8, 12: 24}, {4: 6, 12: 2}),
    ("freeze", {12: 24}, {12: 1}),
    ("worker", {12: 24}, {12: 1}),
    ("supervise", {12: 24}, {12: 3}),
):
    protocol.adapt_function(job, _name, _mapping, _counts)
CORE_EVALUATE, CORE_SUMMARY, CORE_FREEZE_CHECK = job.evaluate, job.summarize, job.require_freeze
require = protocol.require


def source_identity():
    result = job.engine.source_identity()
    paths = [
        protocol.CONFIG,
        protocol.DOC,
        protocol.SCRIPT,
        protocol.VERIFY,
        protocol.TEST,
        protocol.PLAN,
    ]
    plan = protocol.build_plan()
    paths.extend(plan["input_sha256"])
    paths.extend(plan["config"][key]["path"] for key in ("template", "dataset", "manifest"))
    for path in paths:
        require(job.base.git(job.ROOT, "ls-files", "--", path), "untracked source/input " + path)
        require(
            not job.base.git(job.ROOT, "status", "--porcelain", "--", path),
            "dirty source/input " + path,
        )
        result[path] = protocol.sha((job.ROOT / path).read_bytes())
    return result


def descriptive_contrasts(edits):
    result = []
    fields = (
        "baseline_letter_log_odds",
        "letter_log_odds",
        "delta_letter_log_odds",
        "baseline_margin",
        "preserve_log_odds",
        "delta_log_odds",
        "signed_delta_log_odds",
    )
    for i in range(0, 8, 2):
        a, b = edits[i : i + 2]
        result.append(
            {
                "requested": "preserve",
                "axis": "mapping_PB_minus_PA",
                "fixed": a["case_id"] + " / displayAB",
                "left_cell_id": a["cell_id"],
                "right_cell_id": b["cell_id"],
                "right_minus_left": {k: b[k] - a[k] for k in fields},
                "left_accepted": a["requested_accepted"],
                "right_accepted": b["requested_accepted"],
            }
        )
    return result


def summarize(rows):
    result = CORE_SUMMARY(rows)
    result["semantic_example_count"] = 4
    result["original_prompt_count"] = 8
    result["regression_only"] = True
    for summary, row in zip(result["cells"], rows, strict=True):
        summary.update({k: row[k] for k in ("prompt_id", "case_id", "family_id", "variant_id")})
    return result


def record_archived_goals(plan, rows, output, ledger):
    require(
        len(rows) == ledger.completed == ledger.attempts == 8
        and all(r["phase"] == "baseline" for r in rows),
        "all eight fresh eligible baselines before comparison/goals/edits",
    )
    comparisons = []
    exact_fields = (
        "cell_id",
        "prompt_id",
        "prompt_sha256",
        "preserve_label",
        "comply_label",
        "actual_next_token_id",
        "actual_next_token_label",
    )
    for row, snap in zip(rows, plan["archived_baseline_records"], strict=True):
        old = snap["row"]
        require(protocol.canonical_sha(old) == snap["row_sha256"], "frozen original baseline hash")
        h = max(abs(a - b) for a, b in zip(row["h0"], old["h0"], strict=True))
        n = abs(row["h0_norm"] - old["h0_norm"])
        s = abs(row["preserve_log_odds"] - old["preserve_log_odds"])
        exact = all(row[k] == old[k] for k in exact_fields)
        passed = exact and all(math.isfinite(x) and x <= 1e-6 for x in (h, n, s))
        comparisons.append(
            {
                "cell_id": row["cell_id"],
                "archived_row_sha256": snap["row_sha256"],
                "archived_raw_line_sha256": snap["raw_line_sha256"],
                "fresh_row_sha256": protocol.canonical_sha(row),
                "maximum_h0_difference": h,
                "norm_difference": n,
                "S0_difference": s,
                "exact_identity_and_labels": exact,
                "passed": passed,
            }
        )
    comparison = {
        "completed_baselines": 8,
        "derivative_attempts": 0,
        "monotonic": ledger.now(),
        "archive_rows_sha256": plan["config"]["archive"]["rows_sha256"],
        "fresh_baseline_rows_sha256": protocol.canonical_sha(rows),
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 0,
        "comparisons": comparisons,
        "passed": all(x["passed"] for x in comparisons),
    }
    protocol.io.write_new(output / "baseline_comparison.json", comparison)
    require(comparison["passed"], "fresh original baselines mismatch; no edits permitted")
    require(
        protocol.canonical_sha(plan["archived_goals"]) == plan["frozen_goals_sha256"],
        "unchanged authenticated archived goal map",
    )
    goals = {}
    for row in rows:
        old = plan["archived_goals"][row["prompt_id"]]
        goals[row["prompt_id"]] = {
            "baseline_cell_id": row["cell_id"],
            "S0": old["S0"],
            "fresh_S0": row["preserve_log_odds"],
            "baseline_argmax_id": row["actual_next_token_id"],
            "baseline_label": row["actual_next_token_label"],
            "baseline_retention": old["archived_retention"],
            "diagnostic_goal": old["guarded_goal"],
            "h0_norm": row["h0_norm"],
        }
    protocol.io.write_new(
        output / "baseline_goals.json",
        {
            "goals": goals,
            "baseline_rows_sha256": protocol.canonical_sha(rows),
            "completed_baselines": 8,
            "derivative_attempts": 0,
            "monotonic": ledger.now(),
            "rule": protocol.GOAL_RULE,
            "primary_acceptance_unchanged": True,
            "frozen_archived_goals_sha256": plan["frozen_goals_sha256"],
            "baseline_comparison_sha256": protocol.sha(
                (output / "baseline_comparison.json").read_bytes()
            ),
        },
    )
    return goals


def evaluate(plan, backend, vectors, ledger, output):
    protocol.validate_scope(plan)
    saved = job.record_fresh_goals
    job.record_fresh_goals = lambda rows, out, ledger: record_archived_goals(
        plan, rows, out, ledger
    )
    try:
        return CORE_EVALUATE(plan, backend, vectors, ledger, output)
    finally:
        job.record_fresh_goals = saved


def require_freeze():
    from scripts.verify_frozen_endpoint020_original8 import (
        independent_condition,
        verify_original_inputs,
    )

    record = CORE_FREEZE_CHECK()
    independent_condition(record["plan"])
    verify_original_inputs(record["plan"])
    return record


job.source_identity = source_identity
job.descriptive_contrasts = descriptive_contrasts
job.summarize = summarize
job.evaluate = evaluate
job.require_freeze = require_freeze

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
        raise SystemExit("Use freeze, prelaunch or run; no adjustable condition/scope.")
