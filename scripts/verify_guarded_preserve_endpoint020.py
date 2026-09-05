"""Independent endpoint reconstruction, baseline gate and unchanged raw 12/0 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import guarded_preserve_endpoint020_plan as protocol

audit = protocol.isolate(
    "_endpoint020_independent_core", "scripts/verify_frozen_guarded_preserve_crossed.py"
)
audit.protocol = protocol
audit.OUTPUT = ROOT / protocol.OUTPUT
require = audit.require
CORE_VERIFY_DATA, CORE_VERIFY, CORE_REPORT = audit.verify_data, audit.verify, audit.report


def independent_condition(plan, root=ROOT):
    """Independent stdlib reconstruction; never calls runner/plan derivation."""
    spec = plan["config"]["candidates"]["preserve"]
    source_bytes = (root / spec["path"]).read_bytes()
    require(
        hashlib.sha256(source_bytes).hexdigest() == spec["file_sha256"],
        "independent source file hash",
    )
    source = json.loads(source_bytes)
    v = source["vector"]
    source_norm = math.sqrt(math.fsum(float(x) * float(x) for x in v))
    source_digest = hashlib.sha256(struct.pack("<" + "d" * len(v), *v)).hexdigest()
    require(
        len(v) == 1024
        and all(type(x) is float and math.isfinite(x) for x in v)
        and source_norm == spec["norm"] == 0.10764565083962835
        and source_digest == spec["vector_float64_le_sha256"],
        "independent original source coordinates/norm",
    )
    ratio = float(0.20 / source_norm)
    vector = [float(ratio * float(x)) for x in v]
    measured = math.sqrt(math.fsum(x * x for x in vector))
    digest = hashlib.sha256(struct.pack("<" + "d" * len(vector), *vector)).hexdigest()
    expected = {
        "schema": "sp_lense.derived_endpoint_condition.v1",
        "condition_only": True,
        "newly_trained_candidate": False,
        "source_training_success_transfers": False,
        "source_path": spec["path"],
        "source_file_sha256": spec["file_sha256"],
        "source_vector_float64_le_sha256": source_digest,
        "source_norm": source_norm,
        "source_verification_sha256": spec["verification_sha256"],
        "source_candidate_freeze_sha256": spec["candidate_freeze_sha256"],
        "source_construction_lock_sha256": spec["construction_lock_sha256"],
        "radius": 0.20,
        "scale": ratio,
        "derivation": "scale=float64(.20/norm64(source_w)); w020_j=float64(scale*source_w_j)",
        "norm": measured,
        "vector_float64_le_sha256": digest,
        "vector": vector,
    }
    expected_bytes = (
        json.dumps(expected, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    actual = (root / plan["candidates"]["preserve"]["path"]).read_bytes()
    meta = plan["candidates"]["preserve"]
    require(
        actual == expected_bytes
        and plan["derived_condition"] == expected
        and hashlib.sha256(actual).hexdigest() == meta["file_sha256"]
        and digest == meta["vector_float64_le_sha256"]
        and measured == meta["norm"]
        and abs(measured - 0.20) <= 1e-12
        and meta["condition_only"] is True
        and meta["newly_trained_candidate"] is False
        and meta["source_training_success_transfers"] is False
        and meta["guarded_training_audit_verified"] is False,
        "independent endpoint single scaling/hash/cap/provenance",
    )
    return {k: value for k, value in expected.items() if k != "vector"} | {
        "file_sha256": hashlib.sha256(actual).hexdigest(),
        "independently_reconstructed": True,
    }


def verify_baseline_comparison(plan, rows, output):
    saved = audit.read(output / "baseline_comparison.json")
    prior = plan["prior_comparison"]
    fields = plan["config"]["prior_comparison"]["exact_fields"]
    comparisons = []
    for row, snap in zip(rows[:4], prior["baseline_records"], strict=True):
        old = snap["row"]
        require(protocol.canonical_sha(old) == snap["row_sha256"], "prior baseline snapshot hash")
        hmax = max(abs(float(x) - float(y)) for x, y in zip(row["h0"], old["h0"], strict=True))
        nd = abs(row["h0_norm"] - old["h0_norm"])
        sd = abs(row["preserve_log_odds"] - old["preserve_log_odds"])
        exact = all(row[k] == old[k] for k in fields)
        passed = exact and all(math.isfinite(x) and x <= 1e-6 for x in (hmax, nd, sd))
        comparisons.append(
            {
                "cell_id": row["cell_id"],
                "prior_row_sha256": snap["row_sha256"],
                "prior_raw_line_sha256": snap["raw_line_sha256"],
                "fresh_row_sha256": protocol.canonical_sha(row),
                "maximum_h0_difference": hmax,
                "norm_difference": nd,
                "S0_difference": sd,
                "exact_identity_and_labels": exact,
                "passed": passed,
            }
        )
    expected = {
        "prior_namespace": prior["namespace"],
        "prior_rows_sha256": prior["artifact_sha256"]["rows.jsonl"],
        "fresh_baseline_rows_sha256": protocol.canonical_sha(rows[:4]),
        "completed_baselines": 4,
        "derivative_attempts": 0,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 0,
        "monotonic": saved["monotonic"],
        "comparisons": comparisons,
        "passed": all(c["passed"] for c in comparisons),
    }
    events = audit.rows_at(output / "forward_events.jsonl")
    goals = audit.read(output / "baseline_goals.json")
    require(
        saved == expected
        and expected["passed"]
        and events[7]["monotonic"]
        <= saved["monotonic"]
        <= goals["monotonic"]
        <= events[8]["monotonic"],
        "independent fixed-prior fresh baseline comparison/timing",
    )
    return {
        "sha256": protocol.sha((output / "baseline_comparison.json").read_bytes()),
        "all_four_matched_before_edits": True,
        "maximum_h0_difference": max(x["maximum_h0_difference"] for x in comparisons),
        "maximum_norm_difference": max(x["norm_difference"] for x in comparisons),
        "maximum_S0_difference": max(x["S0_difference"] for x in comparisons),
    }


def prior_strength_comparison(plan, rows):
    result = []
    for row, snap in zip(rows[4:8], plan["prior_comparison"]["edit_records"], strict=True):
        old = snap["row"]
        require(
            protocol.canonical_sha(old) == snap["row_sha256"] and old["cell_id"] == row["cell_id"],
            "fixed prior edit identity/hash",
        )
        fields = (
            "preserve_log_odds",
            "delta_log_odds",
            "letter_log_odds",
            "delta_letter_log_odds",
            "answer_pair_mass",
            "kl_from_baseline",
            "retention_slack",
            "goal_slack",
        )
        outcomes = (
            "requested_accepted",
            "actual_next_token_label",
            "retention_nonweakening",
            "retention_with_quality",
            "diagnostic_goal_met",
            "diagnostic_goal_with_quality",
        )
        result.append(
            {
                "cell_id": row["cell_id"],
                "rendering_index": row["rendering_index"],
                "semantic_mapping": row["semantic_mapping"],
                "display_order": row["display_order"],
                "requested_label": row["requested_label"],
                "prior_row_sha256": snap["row_sha256"],
                "prior": {k: old[k] for k in fields + outcomes},
                "endpoint020": {k: row[k] for k in fields + outcomes},
                "endpoint_minus_prior": {k: row[k] - old[k] for k in fields},
            }
        )
    return result


def verify_data(plan, rows, vectors, output):
    result = CORE_VERIFY_DATA(plan, rows, vectors, output)
    result["baseline_comparison"] = verify_baseline_comparison(plan, rows, output)
    result["prior_strength_comparison"] = prior_strength_comparison(plan, rows)
    return result


def verify():
    plan = audit.read(audit.OUTPUT / "preregistration.json")["plan"]
    condition = independent_condition(plan)
    result = CORE_VERIFY()
    result["derived_condition"] = condition
    result["interpretation"] = "outcome-informed exploratory DEVELOPMENT; one exposed case"
    result["source_training_success_transfers"] = False
    return result


def report(result):
    text = CORE_REPORT(result)
    text = text.replace("Frozen guarded-P", "Derived guarded-P .20 endpoint")
    text = text.replace(
        "Exact serialized PRESERVE norm .10764565083962835. No renormalization, target-sign scaling, fitting, projection or composition.",
        "Derived serialized PRESERVE norm .20. One source-only scaling before load; no runtime scaling, fitting, projection or composition.",
    )
    text = text.replace(
        "The frozen arrow was fitted only on the same eight f01/f02 prompts; this f03 case is disjoint from those training IDs.",
        "Only the ORIGINAL source arrow was fitted on eight f01/f02 prompts. Its training8/8 and guard successes do NOT transfer to the .20 condition, whose training and ordinary-task performance are unmeasured.",
    )
    text += "\n## Endpoint condition and scope\n\n"
    text += (
        "Outcome-informed exploratory DEVELOPMENT motivated by prior3/4 failure, not independent confirmation or a newly trained candidate. "
        "The .20 endpoint comes only from the preexisting net ceiling. Four layouts are ONE already-exposed case. "
        "No monotonicity assumption: endpoint failure would not imply all intermediate strengths fail. "
        "No midpoint/search, training, gate or automatic follow-on.\n\n"
    )
    if result["status"] == "INCONCLUSIVE":
        return text
    condition = result.get("derived_condition")
    if condition:
        text += (
            "Independent condition reconstruction: scale "
            + repr(condition["scale"])
            + ", norm "
            + repr(condition["norm"])
            + ".\n\n"
            + "Condition file SHA256: `"
            + condition["file_sha256"]
            + "`.\n\n"
            + "Vector f64LE SHA256: `"
            + condition["vector_float64_le_sha256"]
            + "`.\n\n"
        )
    text += (
        "Fresh baseline agreement before all edits: "
        + json.dumps(result["baseline_comparison"])
        + ".\n\n"
    )
    text += (
        "## Descriptive prior-strength comparison (not a new test)\n\n"
        "| Layout | Wanted | Prior actual / primary | .20 actual / primary | Prior S | .20 S | Difference S | Prior retention slack | .20 retention slack | Prior goal slack | .20 goal slack |\n"
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for r in result["prior_strength_comparison"]:
        a, b = r["prior"], r["endpoint020"]
        text += (
            f"| {r['rendering_index']} | {r['requested_label']} | {a['actual_next_token_label']} / {a['requested_accepted']} | {b['actual_next_token_label']} / {b['requested_accepted']} "
            f"| {a['preserve_log_odds']:+.12g} | {b['preserve_log_odds']:+.12g} | {r['endpoint_minus_prior']['preserve_log_odds']:+.12g} "
            f"| {a['retention_slack']:+.12g} | {b['retention_slack']:+.12g} | {a['goal_slack']:+.12g} | {b['goal_slack']:+.12g} |\n"
        )
    return text


audit.verify_data = verify_data
audit.verify = verify
audit.report = report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    try:
        result = verify()
    except Exception as error:  # noqa: BLE001 - preserve audit failure; no retry.
        result = {
            "status": "INCONCLUSIVE",
            "fault": type(error).__name__ + ": " + str(error),
            "retries_allowed": False,
        }
        audit.write_new(audit.OUTPUT / "VERIFICATION_FAILURE.json", result)
    if args.report:
        audit.write_new(audit.OUTPUT / "verification.json", result)
        with (audit.OUTPUT / "PILOT_REPORT.md").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(report(result))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("summary", "baselines", "prior_strength_comparison")
            },
            indent=2,
        )
    )
    if result["status"] == "INCONCLUSIVE":
        raise SystemExit(1)
