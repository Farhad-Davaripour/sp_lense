"""Independent frozen-C f03 text/letter/raw/geometry/replay audit, no model imports."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import frozen_crossed_comply_f03_plan as protocol

engine = protocol.isolate(
    "scripts._frozen_crossed_C_f03_audit", "scripts/verify_crossed_pair_probe.py"
)
engine.protocol, engine.OUTPUT = protocol, ROOT / protocol.OUTPUT
OUTPUT = ROOT / protocol.OUTPUT
require, read, sha = protocol.require, protocol.read, protocol.sha
f32, compare_summary, outcome = engine.f32, engine.compare_summary, engine.outcome
_parent_renderings, _parent_coverage = engine.verify_renderings, engine.coverage_counts
_raw_data = protocol.adapt(
    engine,
    "verify_data",
    {20: 12},
    {20: 2},
    replacements=[('{"preserve", "comply"}', '{"comply"}')],
)
protocol.adapt(engine, "verify", {20: 12}, {20: 2})
protocol.adapt(
    engine,
    "descriptive_contrasts",
    replacements=[('("preserve", "comply")', '("comply",)')],
)
_parent_summary = protocol.adapt(
    engine,
    "summary",
    {20: 12, 8: 4},
    {20: 2, 8: 3},
    replacements=[('("preserve", "comply")', '("comply",)')],
)


def coverage_counts(group):
    result = _parent_coverage(group)
    for name in ("A_to_B", "B_to_A"):
        eligible = result["eligible_" + name]
        achieved = result["achieved_" + name]
        result[name + "_status"] = (
            "UNTESTED" if not eligible else "ALL" if achieved == eligible else "PARTIAL_OR_FAIL"
        )
    return result


def summary(rows):
    value = _parent_summary(rows)
    value["status"] = (
        "FROZEN_COMPLY_F03_DEVELOPMENT_ACCEPTED_ONLY"
        if value["matrix"]["matrix_pass"]
        else "FROZEN_COMPLY_F03_DEVELOPMENT_PARTIAL_OR_FAIL"
    )
    edits = [r for r in rows if r["phase"] == "edit"]
    weakening = [
        r
        for r in edits
        if r["baseline_argmax_id"] == r["requested_token_id"] and r["signed_delta_log_odds"] < 0
    ]
    value.update(
        {
            "retention_weakening": len(weakening),
            "all_edits_shift_toward_A": all(r["delta_letter_log_odds"] > 0 for r in edits),
            "exposed_semantic_situations": 1,
            "held_out_confirmation": False,
            "physical_sign_inversion": False,
            "training_performed": False,
            "old_v2_success_inherited": False,
        }
    )
    return value


def verify_renderings(plan):
    _parent_renderings(plan)
    expected = protocol.config_at()["rendered_prompts"]
    require(plan["prompts"] == expected, "complete fixed original f03 prompt bytes/metadata")
    cells = []
    for phase in ("baseline", "edit", "replay"):
        for p in expected:
            condition = "baseline" if phase == "baseline" else phase + "_comply"
            cell = {
                "cell_id": p["prompt_id"] + "__" + condition,
                "prompt_id": p["prompt_id"],
                "condition": condition,
                "phase": phase,
                "requested": None if phase == "baseline" else "comply",
                "target_sign": 0 if phase == "baseline" else -1,
                "replay_of": p["prompt_id"] + "__edit_comply" if phase == "replay" else None,
            }
            digest = sha(json.dumps(cell, sort_keys=True, separators=(",", ":")).encode())
            cells.append({**cell, "cell_sha256": digest})
    require(
        plan["cells"] == cells
        and not plan["derivative_cells"]
        and set(plan["candidates"]) == {"comply"},
        "independent exact12 C-only cells; negative semantic sign not physical",
    )


def finite_tree(value):
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)
    elif isinstance(value, float):
        require(math.isfinite(value), "nonfinite raw evidence")


def verify_data(plan, rows, vectors, output):
    finite_tree([rows, vectors])
    for name in ("forward_events.jsonl", "derivative_events.jsonl"):
        finite_tree(engine.rows_at(Path(output) / name))
    return _raw_data(plan, rows, vectors, Path(output))


engine.coverage_counts, engine.summary = coverage_counts, summary
engine.verify_renderings, engine.verify_data = verify_renderings, verify_data


def verify():
    for path in OUTPUT.iterdir():
        if path.is_file() and path.suffix == ".json":
            finite_tree(read(path))
    return engine.verify()


def report(result):
    if result["status"] == "INCONCLUSIVE":
        return (
            "# Frozen COMPLY f03 DEVELOPMENT\n\nINCONCLUSIVE; no retry.\n\n"
            + json.dumps(result, indent=2)
            + "\n"
        )
    s, runtime = result["summary"], result["runtime"]
    m = s["matrix"]
    lines = [
        "# Frozen crossed-trained COMPLY: exposed f03/v1 DEVELOPMENT",
        "",
        f"Audit: {result['status']}. Outcome: {s['status']}.",
        f"Original edits accepted {m['strict_accepted']}/4; independent matching replays {s['replay_matches']}/4.",
        f"Resources: {runtime['forward_attempts']}F/{runtime['derivative_attempts']}D; {runtime['elapsed_seconds']}seconds including loading; one attempt, no retry.",
        "One already-exposed semantic situation, four existing layouts. Not pristine held-out confirmation.",
        "Stored native C vector norm0.2, applied unchanged. Semantic scoring sign-1; physical vector sign+1.",
        "",
        "| Row / mapping / display / phase | Desired | Baseline→actual | L=A−B | ΔL | S=P−C | ΔS | COMPLY change | COMPLY margin | Accepted |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in s["cells"]:
        label = f"{r['rendering_index']}/{r['semantic_mapping']}/{r['display_order']}/{r['phase']}"
        lines.append(
            f"| {label} | {r['requested_label']} | {r['baseline_label']}→{r['actual_next_token_label']} | "
            f"{r['letter_log_odds']:+.12g} | {r['delta_letter_log_odds']:+.12g} | "
            f"{r['preserve_log_odds']:+.12g} | {r['delta_log_odds']:+.12g} | "
            f"{r['signed_delta_log_odds']:+.12g} | {r['signed_margin']:+.12g} | "
            f"{r['requested_accepted'] if r['phase'] != 'baseline' else 'baseline'} |"
        )
    lines += [
        "",
        "## Quality, geometry and replays",
        "",
        "| Row/mapping/display/phase | Pair mass | Raw KL | Own h0 norm | Actual norm | Relative norm | Component error | Replay |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in s["cells"]:
        label = f"{r['rendering_index']}/{r['semantic_mapping']}/{r['display_order']}/{r['phase']}"
        lines.append(
            f"| {label} | {r['answer_pair_mass']:.12g} | {r['kl_from_baseline']:.12g} | "
            f"{r['h0_norm']:.12g} | {r['actual_norm']:.12g} | {r['relative_norm']:.12g} | "
            f"{r['maximum_delta_error']:.12g} | {r['replay_consistent']} |"
        )
    groups = [("overall", m)]
    groups += [("mapping:" + k, v) for k, v in s["coverage_by_mapping"].items()]
    groups += [("display:" + k, v) for k, v in s["coverage_by_display"].items()]
    groups += [(v["group"], v) for v in s["coverage_by_vector_mapping_display"]]
    lines += [
        "",
        "## Original-edit directional coverage",
        "",
        "| Group | Strict/total | A→B eligible/achieved/status | B→A eligible/achieved/status | Accepted flips | Accepted retentions | Actual A→B/B→A | OTHER |",
        "|---|---|---|---|---:|---:|---|---:|",
    ]
    for name, g in groups:
        lines.append(
            f"| {name} | {g['strict_accepted']}/{g['total']} | "
            f"{g['eligible_A_to_B']}/{g['achieved_A_to_B']}/{g['A_to_B_status']} | "
            f"{g['eligible_B_to_A']}/{g['achieved_B_to_A']}/{g['B_to_A_status']} | "
            f"{g['accepted_flips']} | {g['accepted_retentions']} | "
            f"{g['actual_A_to_B']}/{g['actual_B_to_A']} | {g['other_outcomes']} |"
        )
    lines += [
        "",
        f"Baseline availability: {s['baseline_availability']}.",
        f"Retention weakening count: {s['retention_weakening']} (descriptive, NOT a new gate).",
        f"All edited raw letter margins shift toward A: {s['all_edits_shift_toward_A']}.",
        "Missing eligibility is UNTESTED, never permission for baseline search.",
        "Prior construction8/8 had all-B baselines, four B→A flips and four weakened B retentions; an A-favoring alternative remains important.",
        "The f03 situation/layouts and old COMPLY failure were previously exposed. No old failure is overwritten and no old-v2/training success is inherited.",
        "No ordinary preservation, reliable semantic/bidirectional control, learned gate or mechanism established by this small DEVELOPMENT test.",
        "No fitting, gradients, P condition, controls, renormalization, strength search, retry or successor. Report and STOP.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    try:
        result = verify()
    except Exception as error:  # noqa: BLE001 - preserve independent audit fault, never repair/retry.
        result = {
            "status": "INCONCLUSIVE",
            "fault": type(error).__name__ + ": " + str(error),
            "retries_allowed": False,
        }
        engine.write_new(OUTPUT / "VERIFICATION_FAILURE.json", result)
    if args.report:
        engine.write_new(OUTPUT / "verification.json", result)
        with (OUTPUT / "PILOT_REPORT.md").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(report(result))
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("summary", "baselines")}, indent=2)
    )
    if result["status"] == "INCONCLUSIVE":
        raise SystemExit(1)
