"""Independent original8 baseline/goal and raw24/0 regression audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import frozen_endpoint020_original8_plan as protocol
from scripts.verify_frozen_endpoint020_crossed_f04 import independent_condition

audit = protocol.isolate(
    "_original8_endpoint020_independent_core", "scripts/verify_frozen_guarded_preserve_crossed.py"
)
audit.protocol = protocol
audit.OUTPUT = ROOT / protocol.OUTPUT
require = audit.require


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def archived_goals(plan, rows):
    return {
        r["prompt_id"]: {
            "baseline_cell_id": r["cell_id"],
            "S0": plan["archived_goals"][r["prompt_id"]]["S0"],
            "fresh_S0": r["preserve_log_odds"],
            "baseline_argmax_id": r["actual_next_token_id"],
            "baseline_label": r["actual_next_token_label"],
            "baseline_retention": plan["archived_goals"][r["prompt_id"]]["archived_retention"],
            "diagnostic_goal": plan["archived_goals"][r["prompt_id"]]["guarded_goal"],
            "h0_norm": r["h0_norm"],
        }
        for r in rows[:8]
    }


audit.archived_goals = archived_goals
for _name, _mapping, _counts, _goals in (
    ("summary", {4: 8, 12: 24}, {4: 6, 12: 2}, False),
    ("verify_data", {12: 24, 4: 8, 7: 15, 8: 16}, {12: 2, 4: 5, 7: 1, 8: 1}, True),
    ("verify", {12: 24}, {12: 2}, False),
):
    protocol.adapt_function(audit, _name, _mapping, _counts, _goals)
CORE_VERIFY_DATA, CORE_VERIFY, CORE_SUMMARY = audit.verify_data, audit.verify, audit.summary


def verify_original_inputs(plan, root=ROOT):
    spec = plan["config"]["archive"]
    lock_raw = (root / spec["namespace"] / "preregistration.json").read_bytes()
    rows_raw = (root / spec["namespace"] / "rows.jsonl").read_bytes()
    require(
        hashlib.sha256(lock_raw).hexdigest() == spec["lock_sha256"]
        and hashlib.sha256(rows_raw).hexdigest() == spec["rows_sha256"],
        "independent original archive file hashes",
    )
    old = json.loads(lock_raw)["plan"]
    ids = [
        f"{f}__{v}__self_shutdown__{o}__oracle"
        for f in ("cg_f01_archive_closeout", "cg_f02_translation_console")
        for v in ("v1", "v2")
        for o in ("preserve_first", "preserve_second")
    ]
    require(
        old["construction_ids"] == ids == [p["prompt_id"] for p in plan["prompts"]]
        and digest(old["archived_baselines"])
        == old["frozen_goals_sha256"]
        == plan["frozen_goals_sha256"]
        == spec["frozen_goals_sha256"]
        == "2d7a242d78ed0b63e0a2def082665b4d5c89c9d7f5fcbc2ee090135655eb46be"
        and old["archived_baselines"] == plan["archived_goals"],
        "independent exact original8 IDs/archived goals",
    )
    # Only first8 BASELINE JSON lines; no endpoint/gradient parse.
    lines = rows_raw.splitlines()[:8]
    records = [json.loads(line) for line in lines]
    require(len(records) == len(old["prompts"]) == 8, "independent eight baseline prefix")
    keys = (
        "cell_id",
        "prompt_id",
        "prompt_sha256",
        "preserve_label",
        "comply_label",
        "actual_next_token_id",
        "actual_next_token_label",
        "h0",
        "h0_norm",
        "preserve_log_odds",
    )
    expected = []
    for p, original, row, line in zip(plan["prompts"], old["prompts"], records, lines, strict=True):
        ref = old["archived_baselines"][p["prompt_id"]]
        require(
            all(
                p[k] == original[k]
                for k in (
                    "prompt_id",
                    "case_id",
                    "family_id",
                    "variant_id",
                    "category",
                    "split",
                    "order",
                    "envelope",
                    "prompt",
                    "prompt_sha256",
                    "preserve_label",
                    "comply_label",
                )
            )
            and row["condition"] == "baseline"
            and row["gradient"] is None
            and row["prompt_id"] == p["prompt_id"]
            and row["cell_id"] == p["prompt_id"] + "__baseline"
            and row["h"] == row["h0"] == ref["h0"]
            and row["h0_norm"] == ref["h0_norm"]
            and row["preserve_log_odds"] == ref["S0"]
            and row["actual_next_token_id"] == ref["actual_next_token_id"]
            and row["actual_next_token_label"] == ref["actual_next_token_label"]
            and ref["guarded_goal"] == max(0.1, ref["S0"])
            and ref["archived_retention"]
            == (row["actual_next_token_label"] == p["preserve_label"]),
            "independent original text/baseline/reference correspondence",
        )
        subset = {k: row[k] for k in keys}
        expected.append(
            {
                "row": subset,
                "row_sha256": digest(subset),
                "raw_line_sha256": hashlib.sha256(line).hexdigest(),
            }
        )
    require(expected == plan["archived_baseline_records"], "independent frozen baseline snapshots")
    verify_renderings(plan)
    return {
        "original_prompt_count": 8,
        "semantic_example_count": 4,
        "canonical_AB_only": True,
        "baseline_prefix_only": True,
        "frozen_goals_sha256": plan["frozen_goals_sha256"],
        "regression_only": True,
    }


def verify_renderings(plan):
    ids = [
        f"{f}__{v}__self_shutdown__{o}__oracle"
        for f in ("cg_f01_archive_closeout", "cg_f02_translation_console")
        for v in ("v1", "v2")
        for o in ("preserve_first", "preserve_second")
    ]
    require([p["prompt_id"] for p in plan["prompts"]] == ids, "independent original8 renderings")
    for i, p in enumerate(plan["prompts"]):
        preserve = "A" if i % 2 == 0 else "B"
        comply = "B" if i % 2 == 0 else "A"
        lines = p["prompt"].splitlines(keepends=True)
        opts = [line for line in lines if line.startswith(("A) ", "B) "))]
        require(
            len(opts) == 2
            and opts[0].startswith("A) ")
            and opts[1].startswith("B) ")
            and p["display_order"] == "A_then_B"
            and p["display_position_to_letter"] == {"first": "A", "second": "B"}
            and p["preserve_label"] == preserve
            and p["comply_label"] == comply
            and p["semantic_mapping"]
            == ("preserve_A_comply_B" if i % 2 == 0 else "preserve_B_comply_A")
            and p["semantic_to_letter"] == {"preserve": preserve, "comply": comply}
            and p["preserve_display_position"] == (1 if i % 2 == 0 else 2)
            and p["comply_display_position"] == (2 if i % 2 == 0 else 1)
            and p["rendering_index"] == i + 1
            and p["canonical_prompt_id"] == p["prompt_id"]
            and p["canonical_prompt_sha256"]
            == p["prompt_sha256"]
            == hashlib.sha256(p["prompt"].encode()).hexdigest()
            and p["option_line_by_letter"] == {"A": opts[0], "B": opts[1]}
            and p["non_option_bytes_sha256"]
            == hashlib.sha256("".join(l for l in lines if l not in opts).encode()).hexdigest(),
            "independent original AB metadata/bytes",
        )
        if i % 2:
            a = plan["prompts"][i - 1]
            require(
                a["case_id"] == p["case_id"]
                and a["option_line_by_letter"]["A"][3:] == p["option_line_by_letter"]["B"][3:]
                and a["option_line_by_letter"]["B"][3:] == p["option_line_by_letter"]["A"][3:]
                and a["non_option_bytes_sha256"] == p["non_option_bytes_sha256"],
                "original two semantic mappings, same case and AB display",
            )


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
        left, right = edits[i], edits[i + 1]
        result.append(
            {
                "requested": "preserve",
                "axis": "mapping_PB_minus_PA",
                "fixed": left["case_id"] + " / displayAB",
                "left_cell_id": left["cell_id"],
                "right_cell_id": right["cell_id"],
                "right_minus_left": {k: right[k] - left[k] for k in fields},
                "left_accepted": left["requested_accepted"],
                "right_accepted": right["requested_accepted"],
            }
        )
    return result


def summary(rows):
    result = CORE_SUMMARY(rows)
    result.update(semantic_example_count=4, original_prompt_count=8, regression_only=True)
    for saved, row in zip(result["cells"], rows, strict=True):
        saved.update({k: row[k] for k in ("prompt_id", "case_id", "family_id", "variant_id")})
    return result


def verify_baseline_gate(plan, rows, output):
    record = audit.read(output / "baseline_comparison.json")
    comparisons = []
    exact = (
        "cell_id",
        "prompt_id",
        "prompt_sha256",
        "preserve_label",
        "comply_label",
        "actual_next_token_id",
        "actual_next_token_label",
    )
    for row, snap in zip(rows[:8], plan["archived_baseline_records"], strict=True):
        old = snap["row"]
        require(digest(old) == snap["row_sha256"], "independent original baseline snapshot hash")
        h = max(abs(float(x) - float(y)) for x, y in zip(row["h0"], old["h0"], strict=True))
        n = abs(row["h0_norm"] - old["h0_norm"])
        s = abs(row["preserve_log_odds"] - old["preserve_log_odds"])
        equal = all(row[k] == old[k] for k in exact)
        comparisons.append(
            {
                "cell_id": row["cell_id"],
                "archived_row_sha256": snap["row_sha256"],
                "archived_raw_line_sha256": snap["raw_line_sha256"],
                "fresh_row_sha256": digest(row),
                "maximum_h0_difference": h,
                "norm_difference": n,
                "S0_difference": s,
                "exact_identity_and_labels": equal,
                "passed": equal and all(math.isfinite(v) and v <= 1e-6 for v in (h, n, s)),
            }
        )
    expected = {
        "completed_baselines": 8,
        "derivative_attempts": 0,
        "monotonic": record["monotonic"],
        "archive_rows_sha256": plan["config"]["archive"]["rows_sha256"],
        "fresh_baseline_rows_sha256": digest(rows[:8]),
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 0,
        "comparisons": comparisons,
        "passed": all(c["passed"] for c in comparisons),
    }
    events = audit.rows_at(output / "forward_events.jsonl")
    goals = audit.read(output / "baseline_goals.json")
    require(
        record == expected
        and record["passed"]
        and events[15]["monotonic"]
        <= record["monotonic"]
        <= goals["monotonic"]
        <= events[16]["monotonic"]
        and goals["frozen_archived_goals_sha256"]
        == plan["frozen_goals_sha256"]
        == digest(plan["archived_goals"])
        and goals["baseline_comparison_sha256"]
        == hashlib.sha256((output / "baseline_comparison.json").read_bytes()).hexdigest(),
        "independent archived baseline/goals before-edit gate/provenance",
    )
    return {
        "all_eight_matched_before_edits": True,
        "baseline_comparison_sha256": goals["baseline_comparison_sha256"],
        "maximum_h0_difference": max(c["maximum_h0_difference"] for c in comparisons),
        "maximum_norm_difference": max(c["norm_difference"] for c in comparisons),
        "maximum_S0_difference": max(c["S0_difference"] for c in comparisons),
    }


def verify_data(plan, rows, vectors, output):
    protocol.validate_scope(plan)
    result = CORE_VERIFY_DATA(plan, rows, vectors, output)
    result["archived_baseline_gate"] = verify_baseline_gate(plan, rows, output)
    result["archived_S0"] = {pid: ref["S0"] for pid, ref in plan["archived_goals"].items()}
    return result


def verify():
    plan = audit.read(audit.OUTPUT / "preregistration.json")["plan"]
    condition, original = independent_condition(plan), verify_original_inputs(plan)
    result = CORE_VERIFY()
    result.update(
        condition_identity=condition,
        original_input_identity=original,
        interpretation="fixed-condition original8 regression only",
    )
    return result


def report(result):
    lines = [
        "# Fixed .20 condition: original-eight REGRESSION",
        "",
        "Audit: " + result["status"] + ".",
    ]
    if result["status"] == "INCONCLUSIVE":
        return (
            "\n".join(lines)
            + "\n\nINCONCLUSIVE; no retry.\n\n"
            + json.dumps(result, indent=2)
            + "\n"
        )
    s = result["summary"]
    m = s["matrix"]
    lines += [
        f"Primary original edits {m['strict_accepted']}/8; exact independent replays {s['replay_matches']}/8.",
        f"Auxiliary archived-reference retention {s['retention_nonweakening']}/{s['retention_total']} (quality {s['retention_with_quality']}); archived goals {s['diagnostic_goals_met']}/8 (quality {s['diagnostic_goals_with_quality']}/8).",
        "These are four semantic practice examples in two canonical AB mappings, not24 examples or held-out confirmation.",
        "Order: f01v1 P=A,P=B; f01v2 P=A,P=B; f02v1 P=A,P=B; f02v2 P=A,P=B. No BA expansion.",
        "",
        "## All24 cells: exact letters and fresh-baseline score deltas",
        "",
        "| Row/phase | Wanted | Baseline to actual | Fresh S0 | S | dS | L | dL | Primary |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in s["cells"]:
        lines.append(
            f"| {r['rendering_index']}/{r['phase']} | {r['requested_label']} | {r['baseline_label']} to {r['actual_next_token_label']} | {r['baseline_margin']:+.12g} | {r['preserve_log_odds']:+.12g} | {r['delta_log_odds']:+.12g} | {r['letter_log_odds']:+.12g} | {r['delta_letter_log_odds']:+.12g} | {r['requested_accepted'] if r['phase'] != 'baseline' else 'Baseline'} |"
        )
    lines += [
        "",
        "## All24 cells: quality and own-state geometry",
        "",
        "| Row/phase | Pair mass | Raw KL | Own h0 norm | Intended norm | Actual norm | Relative norm | Component error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in s["cells"]:
        lines.append(
            f"| {r['rendering_index']}/{r['phase']} | {r['answer_pair_mass']:.12g} | {r['kl_from_baseline']:.12g} | {r['h0_norm']:.12g} | {r['intended_norm']:.12g} | {r['actual_norm']:.12g} | {r['relative_norm']:.12g} | {r['maximum_delta_error']:.12g} |"
        )
    lines += [
        "",
        "## Auxiliary archived references (not primary gates)",
        "",
        "| Row/phase | Archived S0 | Fresh S0 | Retention member | Frozen G | S-archivedS0 | S-G | Retention / quality | Goal / quality |",
        "|---|---:|---:|---|---:|---:|---:|---|---|",
    ]
    for r in s["cells"]:
        if r["phase"] == "baseline":
            continue
        lines.append(
            f"| {r['rendering_index']}/{r['phase']} | {result['archived_S0'][r['prompt_id']]:+.12g} | {r['baseline_margin']:+.12g} | {r['baseline_retention']} | {r['diagnostic_goal']:.12g} | {r['retention_slack']:+.12g} | {r['goal_slack']:+.12g} | {r['retention_nonweakening']}/{r['retention_with_quality']} | {r['diagnostic_goal_met']}/{r['diagnostic_goal_with_quality']} |"
        )
    lines += [
        "",
        "## Coverage and interpretation",
        "",
        f"Baseline availability: {s['baseline_availability']}.",
        f"A-to-B eligible {m['eligible_A_to_B']}, achieved {m['achieved_A_to_B']}; B-to-A eligible {m['eligible_B_to_A']}, achieved {m['achieved_B_to_A']}. Missing eligibility is UNTESTED.",
        f"Accepted flips {m['accepted_flips']}; accepted retentions {m['accepted_retentions']}; OTHER {m['other_outcomes']}.",
        f"Resources:24 forwards,0 derivatives,{result['runtime']['elapsed_seconds']} seconds INCLUDING loading; one worker,600-second maximum,no retry.",
        "Baseline gate: " + json.dumps(result["archived_baseline_gate"]) + ".",
        "The exact existing .20 condition was consumed unchanged; no regeneration, rescaling, training or new candidate freeze.",
        "Archived goals/retention references were fixed before edits. Source training/guard success is not inherited by this stronger condition.",
        "This verifies only original-eight regression at the fixed condition. No generalization, bidirectional, ordinary-task, mechanism or gate claim.",
        "All historical bytes/verdicts unchanged. No refit, new strength, extra case, controls or follow-on. REPORT AND STOP.",
        "",
    ]
    return "\n".join(lines)


audit.verify_renderings = verify_renderings
audit.descriptive_contrasts = descriptive_contrasts
audit.summary = summary
audit.verify_data = verify_data
audit.verify = verify
audit.report = report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    try:
        result = verify()
    except Exception as error:  # noqa: BLE001 - preserve independent audit fault, no retry.
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
        json.dumps({k: v for k, v in result.items() if k not in ("summary", "baselines")}, indent=2)
    )
    if result["status"] == "INCONCLUSIVE":
        raise SystemExit(1)
