"""Independent raw midpoint audit; descriptive outcomes without semantic acceptance."""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import midpoint_only_f03_plan as protocol

engine = protocol.isolate(
    "scripts._midpoint_only_f03_audit", "scripts/verify_crossed_pair_probe.py"
)
OUTPUT = ROOT / protocol.OUTPUT
engine.OUTPUT = OUTPUT
require, read, sha = protocol.require, protocol.read, protocol.sha
f32, compare_summary = engine.f32, engine.compare_summary
_parent_renderings = engine.verify_renderings


def independent_candidates():
    cfg = protocol.config_at()
    spec = cfg["candidates"]["midpoint"]
    raw = (ROOT / spec["raw_path"]).read_bytes()
    serialized = (ROOT / spec["path"]).read_bytes()
    require(
        len(raw) == 8192
        and sha(raw)
        == spec["stored_vector_float64_le_sha256"]
        == "c6cf338185ff98262afba2eb137cf64bc009e17d5d29bad542ec11464308f26d"
        and sha(serialized)
        == spec["file_sha256"]
        == "94d858ef364fd907337d112aa5672b47802289667f39d1516eaad720a0119c5b",
        "independent stored-c raw/JSON bytes",
    )
    c = list(struct.unpack("<1024d", raw))
    value = json.loads(serialized)
    require(
        all(math.isfinite(x) for x in c)
        and raw == struct.pack("<1024d", *value["vector"])
        and value["vector_float64_le_sha256"] == sha(raw)
        and math.sqrt(math.fsum(x * x for x in c)) == spec["norm"] == 0.11950278487420843
        and value["kind"] == "midpoint"
        and spec["physical_sign"] == 1
        and spec["semantic_target"] is None,
        "independent exact unscaled midpoint; no P/C/d or signed selection",
    )
    return {"midpoint": {"vector": c}}


engine.protocol = SimpleNamespace(
    build_plan=protocol.build_plan,
    vector_sha=protocol.vector_sha,
    sha=sha,
    offset_sha=lambda xs: sha(struct.pack("<" + "f" * len(xs), *xs)),
    OUTPUT=protocol.OUTPUT,
    candidates=independent_candidates,
)
_raw_data = protocol.adapt(
    engine,
    "verify_data",
    {20: 12},
    {20: 2},
    replacements=[
        ('{"preserve", "comply"}', '{"midpoint"}'),
        ('"two candidate bindings"', '"one midpoint candidate binding"'),
        ('"direct deltaS/signed effect not exact"', '"direct descriptive deltaS not exact"'),
        (
            'and row["signed_delta_log_odds"] == row["target_sign"] * effect',
            'and row["signed_delta_log_odds"] is None',
        ),
        (
            '        sign = row["target_sign"]\n        requested = "preserve" if sign == 1 else "comply" if sign == -1 else None\n        wanted = (\n            (\n                row["choice_a_token_id"]\n                if prompt[requested + "_label"] == "A"\n                else row["choice_b_token_id"]\n            )\n            if sign\n            else None\n        )\n        require(\n            row["requested"] == requested\n            and row["requested_token_id"] == wanted\n            and row["requested_label"] == (prompt[requested + "_label"] if sign else None)\n            and row["signed_margin"] == sign * measured["preserve_log_odds"],\n            "semantic sign/label mapping",\n        )\n        vector = vectors[requested] if sign else [0.0] * plan["model"]["d_model"]\n        meta = plan["candidates"][requested] if sign else None\n',
            '        sign = row["intervention_on"]\n        require(\n            row["requested"] is row["target_sign"] is row["requested_token_id"]\n            is row["requested_label"] is row["signed_margin"] is None,\n            "no semantic target, desired label or signed margin",\n        )\n        vector = vectors["midpoint"] if sign else [0.0] * plan["model"]["d_model"]\n        meta = plan["candidates"]["midpoint"] if sign else None\n',
        ),
        (
            '("letter_log_odds", "delta_letter_log_odds", "signed_delta_log_odds")',
            '("letter_log_odds", "delta_letter_log_odds", "delta_log_odds")',
        ),
        (
            "real_norm, intended_norm = norm(real), norm(expected_delta)",
            'real_norm, intended_norm = norm(real), norm(expected_delta)\n        require(not sign or (real_norm > 0 and intended_norm > 0), "ON must be physically nonzero")',
        ),
    ],
)
protocol.adapt(engine, "verify", {20: 12}, {20: 2})


def outcome(row):
    # Independent observations, not inherited P/C requested-acceptance logic.
    return {
        "quality_valid": row["answer_pair_mass"] >= 0.8 and row["kl_from_baseline"] >= -1e-6,
        "actual_argmax_changed": row["actual_next_token_id"] != row["baseline_argmax_id"],
        "physical_intervention_on": row["intervention_on"],
        "physical_nonzero": row["actual_norm"] > 0 and row["intended_norm"] > 0,
    }


def observations(group):
    counts = {a + "_to_" + b: 0 for a in ("A", "B") for b in ("A", "B", "OTHER")}
    for r in group:
        counts[r["baseline_label"] + "_to_" + r["actual_next_token_label"]] += 1
    return {
        "total": len(group),
        "observed_transitions": counts,
        "quality_flagged": len([r for r in group if not r["quality_valid"]]),
        "physical_on": len([r for r in group if r["physical_intervention_on"]]),
        "physical_nonzero": len([r for r in group if r["physical_nonzero"]]),
    }


def summary(rows):
    groups = {
        phase: [r for r in rows if r["phase"] == phase] for phase in ("baseline", "edit", "replay")
    }
    require(
        len(rows) == 12
        and all(len(g) == 4 for g in groups.values())
        and all(r["integrity_passed"] for r in rows)
        and all(r["replay_consistent"] for r in groups["replay"]),
        "complete12 raw-audited diagnostic with four matching replays",
    )
    for r in rows:
        on = r["phase"] != "baseline"
        require(
            r["intervention_on"] is r["physical_intervention_on"] is on
            and r["physical_nonzero"] is on
            and all(
                r[k] is None
                for k in (
                    "requested",
                    "target_sign",
                    "requested_label",
                    "requested_token_id",
                    "signed_margin",
                    "signed_delta_log_odds",
                )
            ),
            "physical ON/nonzero distinct from null semantic target",
        )
    baselines, edits = groups["baseline"], groups["edit"]
    fields = (
        "cell_id",
        "rendering_index",
        "order",
        "semantic_mapping",
        "display_order",
        "semantic_to_letter",
        "display_position_to_letter",
        "phase",
        "requested",
        "target_sign",
        "requested_label",
        "requested_token_id",
        "intervention",
        "intervention_on",
        "physical_sign",
        "physical_intervention_on",
        "physical_nonzero",
        "baseline_label",
        "actual_next_token_label",
        "actual_next_token_id",
        "baseline_margin",
        "baseline_letter_log_odds",
        "letter_log_odds",
        "delta_letter_log_odds",
        "preserve_log_odds",
        "delta_log_odds",
        "signed_delta_log_odds",
        "signed_margin",
        "answer_pair_mass",
        "kl_from_baseline",
        "h0_norm",
        "intended_norm",
        "actual_norm",
        "maximum_offset_error",
        "maximum_delta_error",
        "frozen_vector_norm",
        "relative_norm",
        "quality_valid",
        "unselected_max_difference",
        "weights_unchanged",
        "actual_argmax_changed",
        "maximum_replay_h_difference",
        "maximum_replay_logit_difference",
        "replay_of",
        "replay_consistent",
    )
    return {
        "status": "MIDPOINT_ONLY_F03_DIAGNOSTIC_COMPLETED",
        "semantic_target": None,
        "behavioral_success_claimed": False,
        "baseline_count": 4,
        "edit_count": 4,
        "replay_matches": 4,
        "forward_count": 12,
        "derivative_count": 0,
        "original_edit_observations": observations(edits),
        "observations_by_mapping": {
            m: observations([r for r in edits if r["semantic_mapping"] == m])
            for m in ("preserve_A_comply_B", "preserve_B_comply_A")
        },
        "observations_by_display": {
            d: observations([r for r in edits if r["display_order"] == d])
            for d in ("A_then_B", "B_then_A")
        },
        "baseline_availability": {
            label: len([r for r in baselines if r["actual_next_token_label"] == label])
            for label in ("A", "B", "OTHER")
        },
        "cells": [{k: r[k] for k in fields} for r in rows],
        "exposed_semantic_situations": 1,
        "held_out_confirmation": False,
        "replays_are_new_examples": False,
        "difference_applied": False,
        "parent_vectors_applied": False,
        "training_performed": False,
        "ordinary_task_preservation_tested": False,
        "learned_gate_allowed": False,
        "reliable_generalization_established": False,
        "old_verdicts_unchanged": True,
    }


def verify_renderings(plan):
    _parent_renderings(plan)
    expected = protocol.config_at()["rendered_prompts"]
    require(plan["prompts"] == expected, "complete fixed original f03 prompt bytes/metadata")
    cells = []
    for phase in ("baseline", "edit", "replay"):
        for p in expected:
            on = phase != "baseline"
            condition = phase + "_midpoint" if on else "baseline"
            cell = {
                "cell_id": p["prompt_id"] + "__" + condition,
                "prompt_id": p["prompt_id"],
                "condition": condition,
                "phase": phase,
                "requested": None,
                "target_sign": None,
                "intervention": "midpoint" if on else None,
                "intervention_on": on,
                "physical_sign": int(on),
                "replay_of": p["prompt_id"] + "__edit_midpoint" if phase == "replay" else None,
            }
            digest = sha(json.dumps(cell, sort_keys=True, separators=(",", ":")).encode())
            cells.append({**cell, "cell_sha256": digest})
    require(
        plan["cells"] == cells
        and not plan["derivative_cells"]
        and set(plan["candidates"]) == {"midpoint"},
        "independent exact12 ON/null-target cells; no all-OFF or P/C",
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
    require(set(vectors) == set(plan["candidates"]) == {"midpoint"}, "only midpoint array")
    meta = plan["candidates"]["midpoint"]
    require(
        meta["physical_sign"] == 1
        and meta["semantic_target"] is None
        and meta["stored_vector_float64_le_sha256"] == protocol.vector_sha(vectors["midpoint"]),
        "independent physical positive c with null target; no sign/scaling",
    )
    for r in rows:
        require(
            not any(
                k in r
                for k in (
                    "requested_accepted",
                    "requested_argmax",
                    "new_requested_flip",
                    "requested_retention",
                )
            ),
            "semantic acceptance metadata forbidden",
        )
    output = Path(output)
    for name in ("forward_events.jsonl", "derivative_events.jsonl"):
        finite_tree(engine.rows_at(output / name))
    return _raw_data(plan, rows, vectors, output)


engine.outcome, engine.summary = outcome, summary
engine.verify_renderings, engine.verify_data = verify_renderings, verify_data


def verify():
    for path in OUTPUT.iterdir():
        if path.is_file() and path.suffix == ".json":
            finite_tree(read(path))
    return engine.verify()


def report(result):
    if result["status"] == "INCONCLUSIVE":
        return (
            "# Midpoint-only f03 diagnostic\n\nINCONCLUSIVE; no retry.\n\n"
            + json.dumps(result, indent=2)
            + "\n"
        )
    s, runtime = result["summary"], result["runtime"]
    lines = [
        "# Midpoint c alone: exposed f03/v1 descriptive diagnostic",
        "",
        f"Audit: {result['status']}. Disposition: {s['status']}.",
        "Semantic target NULL for all rows; no desired answer, behavioral acceptance or signed semantic target.",
        "Four physical-ON/nonzero midpoint edits and four independent matching replays; not extra examples.",
        f"Resources: {runtime['forward_attempts']}F/{runtime['derivative_attempts']}D; {runtime['elapsed_seconds']}seconds including loading; one attempt, no retry.",
        "Stored midpoint norm0.11950278487420843 unchanged; own-original state/norm each call.",
        "",
        "| Row / mapping / display / phase | ON | Baseline→actual | L=A−B | ΔL | S=P−C | ΔS |",
        "|---|---|---|---:|---:|---:|---:|",
    ]

    def key(r):
        return f"{r['rendering_index']}/{r['semantic_mapping']}/{r['display_order']}/{r['phase']}"

    for r in s["cells"]:
        lines.append(
            f"| {key(r)} | {r['intervention_on']} | {r['baseline_label']}→{r['actual_next_token_label']} | "
            f"{r['letter_log_odds']:+.12g} | {r['delta_letter_log_odds']:+.12g} | "
            f"{r['preserve_log_odds']:+.12g} | {r['delta_log_odds']:+.12g} |"
        )
    lines += [
        "",
        "## Quality observations, geometry and replay identity",
        "",
        "| Row/mapping/display/phase | Pairmass | Raw KL | Flagged | Own h0 norm | Actual norm | Relative norm | Cast error | Weights / nonfinal | Replay |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for r in s["cells"]:
        lines.append(
            f"| {key(r)} | {r['answer_pair_mass']:.12g} | {r['kl_from_baseline']:.12g} | {not r['quality_valid']} | "
            f"{r['h0_norm']:.12g} | {r['actual_norm']:.12g} | {r['relative_norm']:.12g} | "
            f"{r['maximum_delta_error']:.12g} | {r['weights_unchanged']} / {r['unselected_max_difference']} | {r['replay_consistent']} |"
        )
    lines += [
        "",
        f"Original-edit observations: {s['original_edit_observations']}.",
        f"Baseline availability: {s['baseline_availability']}.",
        "All finite outcomes retained, including negative/zero movements, OTHER and low mass if observed. Quality flags are not acceptance gates.",
        "A→B/B→A are observed transitions only; no desired-transition eligibility.",
        "The midpoint is NOT identified A-bias. Parent P/C construction differed, and this single situation is already exposed DEVELOPMENT.",
        "Neither A-favoring movement nor its absence establishes general necessity/sufficiency, semantic mechanism, ordinary preservation or bidirectional control.",
        "Do not add earlier P/C/d score changes as if the network were linear. No d, parents, c±d, normalization, upscaling, selection, training, gate or extra calls.",
        "Historical evidence/verdicts unchanged. Publication readiness40%. No autonomous successor; hand off and STOP.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    require(sys.argv[1:] == ["--report"], "Use --report; no numeric/scientific options")
    try:
        result = verify()
    except Exception as error:  # noqa: BLE001 - retain audit failure; never repair/retry.
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
