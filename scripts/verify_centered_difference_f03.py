"""Independent signed-d raw audit. Numeric parent unchanged; signs decoded from raw d."""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import centered_difference_f03_plan as protocol

engine = protocol.isolate(
    "scripts._centered_difference_f03_audit", "scripts/verify_crossed_pair_probe.py"
)
OUTPUT = ROOT / protocol.OUTPUT
engine.OUTPUT = OUTPUT
require, read, sha = protocol.require, protocol.read, protocol.sha
f32, compare_summary, outcome = engine.f32, engine.compare_summary, engine.outcome
_parent_renderings, _parent_data = engine.verify_renderings, engine.verify_data
_parent_summary, _parent_coverage = engine.summary, engine.coverage_counts


def independent_candidates():
    cfg = protocol.config_at()
    spec = cfg["candidates"]["preserve"]
    raw = (ROOT / spec["raw_path"]).read_bytes()
    serialized = (ROOT / spec["path"]).read_bytes()
    require(
        len(raw) == 8192
        and sha(raw) == spec["stored_vector_float64_le_sha256"]
        and sha(serialized) == spec["file_sha256"],
        "independent stored-d raw/JSON bytes",
    )
    d = list(struct.unpack("<1024d", raw))
    value = json.loads(serialized)
    require(
        all(math.isfinite(x) for x in d)
        and raw == struct.pack("<1024d", *value["vector"])
        and value["vector_float64_le_sha256"] == sha(raw)
        and math.sqrt(math.fsum(x * x for x in d)) == spec["norm"] == 0.1603717070037875,
        "independent unscaled stored-d coordinates",
    )
    # Independent unary negation, not the runner's signed-vector constructor.
    return {"preserve": {"vector": d}, "comply": {"vector": [-x for x in d]}}


engine.protocol = SimpleNamespace(
    build_plan=protocol.build_plan,
    vector_sha=protocol.vector_sha,
    sha=sha,
    offset_sha=lambda xs: sha(struct.pack("<" + "f" * len(xs), *xs)),
    OUTPUT=protocol.OUTPUT,
    candidates=independent_candidates,
)


def coverage_counts(group):
    value = _parent_coverage(group)
    for direction in ("A_to_B", "B_to_A"):
        n, achieved = value["eligible_" + direction], value["achieved_" + direction]
        value[direction + "_status"] = (
            "UNTESTED" if not n else "ALL" if achieved == n else "PARTIAL_OR_FAIL"
        )
    return value


def summary(rows):
    value = _parent_summary(rows)
    value["status"] = (
        "CENTERED_DIFFERENCE_F03_DEVELOPMENT_ACCEPTED_ONLY"
        if value["matrix"]["matrix_pass"]
        else "CENTERED_DIFFERENCE_F03_DEVELOPMENT_PARTIAL_OR_FAIL"
    )
    value["retention_weakening_by_sign"] = {}
    for target in ("preserve", "comply"):
        value["retention_weakening_by_sign"][target] = len(
            [
                r
                for r in rows
                if r["phase"] == "edit"
                and r["requested"] == target
                and r["baseline_argmax_id"] == r["requested_token_id"]
                and r["signed_delta_log_odds"] < 0
            ]
        )
    value.update(
        exposed_semantic_situations=1,
        held_out_confirmation=False,
        midpoint_applied=False,
        parent_vectors_applied=False,
        training_performed=False,
        source_success_inherited=False,
    )
    return value


def verify_renderings(plan):
    _parent_renderings(plan)
    prompts = protocol.config_at()["rendered_prompts"]
    require(plan["prompts"] == prompts, "same frozen four full prompt bytes and metadata")
    cells = []
    for phase in ("baseline", "edit", "replay"):
        for p in prompts:
            for target in (None,) if phase == "baseline" else ("preserve", "comply"):
                condition = phase if target is None else phase + "_" + target
                sign = {None: 0, "preserve": 1, "comply": -1}[target]
                cell = {
                    "cell_id": p["prompt_id"] + "__" + condition,
                    "prompt_id": p["prompt_id"],
                    "condition": condition,
                    "phase": phase,
                    "requested": target,
                    "target_sign": sign,
                    "physical_sign": sign,
                    "replay_of": p["prompt_id"] + "__edit_" + target if phase == "replay" else None,
                }
                cells.append(
                    {
                        **cell,
                        "cell_sha256": sha(
                            json.dumps(cell, sort_keys=True, separators=(",", ":")).encode()
                        ),
                    }
                )
    require(
        plan["cells"] == cells
        and not plan["derivative_cells"]
        and set(plan["candidates"]) == {"preserve", "comply"},
        "independent exact4/20/0 signed schedule",
    )


def finite_tree(value):
    if isinstance(value, dict):
        for x in value.values():
            finite_tree(x)
    elif isinstance(value, list):
        for x in value:
            finite_tree(x)
    elif isinstance(value, float):
        require(math.isfinite(value), "nonfinite raw evidence")


def verify_data(plan, rows, vectors, output):
    finite_tree([rows, vectors])
    require(set(vectors) == {"preserve", "comply"}, "two signed d arrays")
    require(
        protocol.vector_sha(vectors["comply"])
        == protocol.vector_sha([-x for x in vectors["preserve"]]),
        "independent exact physical sign reversal, no midpoint/parent pair",
    )
    for target, sign in (("preserve", 1), ("comply", -1)):
        require(
            plan["candidates"][target]["physical_sign"] == sign, "separate physical sign binding"
        )
        require(
            plan["candidates"][target]["stored_vector_float64_le_sha256"]
            == protocol.vector_sha(vectors["preserve"]),
            "same original stored d hash",
        )
    output = Path(output)
    for name in ("forward_events.jsonl", "derivative_events.jsonl"):
        finite_tree(engine.rows_at(output / name))
    return _parent_data(plan, rows, vectors, output)


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
            "# Centered difference f03 DEVELOPMENT\n\nINCONCLUSIVE. No retry.\n\n"
            + json.dumps(result, indent=2)
            + "\n"
        )
    s = result["summary"]
    m = s["matrix"]
    runtime = result["runtime"]
    lines = [
        "# Centered difference ±d: exposed f03/v1 DEVELOPMENT",
        "",
        f"Audit: {result['status']}. Outcome: {s['status']}.",
        f"Original edits accepted {m['strict_accepted']}/8; P {s['per_vector']['preserve']['strict_accepted']}/4; C {s['per_vector']['comply']['strict_accepted']}/4; matching replays {s['replay_matches']}/8.",
        f"Resources: {runtime['forward_attempts']}F/{runtime['derivative_attempts']}D; {runtime['elapsed_seconds']}seconds including loading; one attempt, no retry.",
        "Stored d norm0.1603717070037875 unchanged. Physical +d/P and -d/C; semantic signs+1/-1 separately checked.",
        "One already-exposed situation/four layouts, not pristine held-out confirmation.",
        "",
        "| Row/mapping/display/phase/request | Physical / scoring sign | Baseline→actual | Δ(A−B) | S=P−C | ΔS | Signed change | Signed margin | Accepted |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]

    def key(r):
        return f"{r['rendering_index']}/{r['semantic_mapping']}/{r['display_order']}/{r['phase']}/{r['requested']}"

    for r in s["cells"]:
        lines.append(
            f"| {key(r)} | {r['target_sign']:+d}/{r['target_sign']:+d} | {r['baseline_label']}→{r['actual_next_token_label']} | {r['delta_letter_log_odds']:+.12g} | {r['preserve_log_odds']:+.12g} | {r['delta_log_odds']:+.12g} | {r['signed_delta_log_odds']:+.12g} | {r['signed_margin']:+.12g} | {r['requested_accepted'] if r['phase'] != 'baseline' else 'baseline'} |"
        )
    lines += [
        "",
        "## Quality, original-state geometry and replays",
        "",
        "| Row/mapping/display/phase/request | Pair mass | Raw KL | Own h0 norm | Actual norm | Relative norm | Component error | Replay |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in s["cells"]:
        lines.append(
            f"| {key(r)} | {r['answer_pair_mass']:.12g} | {r['kl_from_baseline']:.12g} | {r['h0_norm']:.12g} | {r['actual_norm']:.12g} | {r['relative_norm']:.12g} | {r['maximum_delta_error']:.12g} | {r['replay_consistent']} |"
        )
    groups = [("overall", m)]
    groups += [("sign:" + k, v) for k, v in s["per_vector"].items()]
    groups += [("mapping:" + k, v) for k, v in s["coverage_by_mapping"].items()]
    groups += [("display:" + k, v) for k, v in s["coverage_by_display"].items()]
    groups += [(g["group"], g) for g in s["coverage_by_vector_mapping_display"]]
    lines += [
        "",
        "## Original-baseline directional coverage",
        "",
        "| Group | Accepted/total | A→B eligible/achieved/status | B→A eligible/achieved/status | Accepted flips | Accepted retentions | Actual A→B/B→A | OTHER |",
        "|---|---|---|---|---:|---:|---|---:|",
    ]
    for name, g in groups:
        lines.append(
            f"| {name} | {g['strict_accepted']}/{g['total']} | {g['eligible_A_to_B']}/{g['achieved_A_to_B']}/{g['A_to_B_status']} | {g['eligible_B_to_A']}/{g['achieved_B_to_A']}/{g['B_to_A_status']} | {g['accepted_flips']} | {g['accepted_retentions']} | {g['actual_A_to_B']}/{g['actual_B_to_A']} | {g['other_outcomes']} |"
        )
    lines += [
        "",
        f"Baseline availability: {s['baseline_availability']}.",
        f"Retention weakening by sign: {s['retention_weakening_by_sign']} (descriptive, NOT a gate).",
        "Zero eligible means UNTESTED. Replays are not new examples. Opposite treatment outputs are never sequential or original-baseline A→B flips.",
        "P/C source construction differed (P v1/v2 AB retention goals; C v1 crossed AB/BA original outcome objective). d mixes those differences.",
        "Geometry did NOT identify A-bias. Removing the midpoint may lose a necessary nonlinear offset. No source-success inheritance.",
        "No midpoint, c±d, parent-vector application, regeneration, normalization, .20 upscaling, training, gate, ordinary controls or extra calls.",
        "Old failures and passes remain unchanged. Exposed DEVELOPMENT only, not held-out or reliable semantic-axis/mechanism/ordinary-preservation evidence.",
        "No automatic successor or post-lock rescue; report PASS OR FAIL and STOP.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    require(sys.argv[1:] == ["--report"], "Use --report; no numeric/scientific options")
    try:
        result = verify()
    except Exception as error:  # noqa: BLE001 - preserve failed audit, never repair/retry.
        result = {
            "status": "INCONCLUSIVE",
            "fault": type(error).__name__ + ": " + str(error),
            "retries_allowed": False,
        }
        engine.write_new(OUTPUT / "VERIFICATION_FAILURE.json", result)
    engine.write_new(OUTPUT / "verification.json", result)
    with (OUTPUT / "PILOT_REPORT.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report(result))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("summary", "baselines")},
            indent=2,
            allow_nan=False,
        )
    )
    if result["status"] == "INCONCLUSIVE":
        raise SystemExit(1)
