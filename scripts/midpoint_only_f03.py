"""Midpoint-only diagnostic. Physical ON is never inferred from a semantic target."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import midpoint_only_f03_plan as protocol

engine = protocol.isolate("scripts._midpoint_only_f03_engine", "scripts/crossed_pair_probe.py")
engine.protocol = protocol
for key in ("SCRIPT", "VERIFY", "TEST", "DOC", "PLAN", "OUTPUT"):
    setattr(engine, key, getattr(protocol, key))
OUTPUT = ROOT / protocol.OUTPUT
require, base = protocol.require, engine.base
make_delta = engine.make_delta
DerivativeGuard, EligibilityError = engine.DerivativeGuard, engine.EligibilityError
require_freeze = engine.require_freeze
EPS, norm = protocol.EPS, protocol.norm


def validate_vectors(plan, vectors):
    require(set(vectors) == set(plan["candidates"]) == {"midpoint"}, "only stored midpoint")
    v, meta = vectors["midpoint"], plan["candidates"]["midpoint"]
    require(
        len(v) == plan["model"]["d_model"]
        and all(type(x) is float and math.isfinite(x) for x in v)
        and protocol.vector_sha(v)
        == meta["vector_float64_le_sha256"]
        == meta["stored_vector_float64_le_sha256"]
        and norm(v) == meta["norm"]
        and norm(v) > 0
        and meta["physical_sign"] == 1
        and meta["semantic_target"] is None,
        "exact c coordinates/norm; no sign/scaling or semantic target",
    )
    require(
        len(plan["prompts"]) == 4 and len(plan["cells"]) == 12 and not plan["derivative_cells"],
        "exact4/12/0 before forwards",
    )
    for i, cell in enumerate(plan["cells"]):
        phase = ("baseline", "edit", "replay")[i // 4]
        on, pid = phase != "baseline", plan["prompts"][i % 4]["prompt_id"]
        condition = phase + "_midpoint" if on else "baseline"
        require(
            cell["prompt_id"] == pid
            and cell["cell_id"] == pid + "__" + condition
            and cell["phase"] == phase
            and cell["condition"] == condition
            and cell["requested"] is cell["target_sign"] is None
            and cell["intervention_on"] is on
            and cell["intervention"] == ("midpoint" if on else None)
            and type(cell["physical_sign"]) is int
            and cell["physical_sign"] == int(on)
            and cell["replay_of"] == (pid + "__edit_midpoint" if phase == "replay" else None)
            and cell["cell_sha256"]
            == protocol.canonical_sha(
                {k: value for k, value in cell.items() if k != "cell_sha256"}
            ),
            "exact physical ON/null-target schedule; all-OFF no-op forbidden",
        )


def assess(row):
    return {
        "quality_valid": row["answer_pair_mass"] >= 0.8 and row["kl_from_baseline"] >= -EPS,
        "actual_argmax_changed": row["actual_next_token_id"] != row["baseline_argmax_id"],
        "physical_intervention_on": row["intervention_on"],
        "physical_nonzero": row["intended_norm"] > 0 and row["actual_norm"] > 0,
    }


# Exact-site adaptation: unchanged numeric/geometry/scoring/hook math;
# physical selection independent of requested=None, no signed semantic outcomes.
evaluate = protocol.adapt(
    engine,
    "evaluate",
    {20: 12},
    {20: 1},
    replacements=[
        (
            '                p, sign = prompts[cell["prompt_id"]], cell["target_sign"]\n                target = cell["requested"]\n                vector = vectors[target] if target else [0.0] * plan["model"]["d_model"]\n                meta = plan["candidates"][target] if target else None\n',
            '                p, sign = prompts[cell["prompt_id"]], cell["intervention_on"]\n                target = cell["intervention"]\n                vector = vectors[target] if sign else [0.0] * plan["model"]["d_model"]\n                meta = plan["candidates"][target] if sign else None\n',
        ),
        (
            '                        "requested": "preserve" if sign == 1 else "comply" if sign == -1 else None,\n                        "requested_token_id": boundary.token_id(\n                            p["preserve_label"] if sign == 1 else p["comply_label"]\n                        )\n                        if sign\n                        else None,\n                        "requested_label": p[target + "_label"] if target else None,\n',
            '                        "requested": None,\n                        "requested_token_id": None,\n                        "requested_label": None,\n',
        ),
        (
            'row["signed_delta_log_odds"] = sign * row["delta_log_odds"]',
            'row["signed_delta_log_odds"] = None',
        ),
        ('row["signed_margin"] = sign * row["preserve_log_odds"]', 'row["signed_margin"] = None'),
        (
            '("letter_log_odds", "delta_letter_log_odds", "signed_delta_log_odds")',
            '("letter_log_odds", "delta_letter_log_odds", "delta_log_odds")',
        ),
        (
            "faults = []",
            'faults = ["ON intervention is a no-op"] if sign and not row["physical_nonzero"] else []',
        ),
    ],
)
protocol.adapt(engine, "worker", {20: 12}, {20: 1})
supervise = protocol.adapt(engine, "supervise", {20: 12}, {20: 3})
freeze = protocol.adapt(engine, "freeze", {20: 12}, {20: 1})


def observations(group):
    return {
        "total": len(group),
        "observed_transitions": {
            a + "_to_" + b: sum(
                r["baseline_label"] == a and r["actual_next_token_label"] == b for r in group
            )
            for a in ("A", "B")
            for b in ("A", "B", "OTHER")
        },
        "quality_flagged": sum(not r["quality_valid"] for r in group),
        "physical_on": sum(r["physical_intervention_on"] for r in group),
        "physical_nonzero": sum(r["physical_nonzero"] for r in group),
    }


def summarize(rows):
    require(len(rows) == 12 and all(r["integrity_passed"] for r in rows), "complete12 integrity")
    baselines = [r for r in rows if r["phase"] == "baseline"]
    edits = [r for r in rows if r["phase"] == "edit"]
    replays = [r for r in rows if r["phase"] == "replay"]
    require(
        len(baselines) == len(edits) == len(replays) == 4
        and all(r["replay_consistent"] for r in replays)
        and all(r["physical_intervention_on"] and r["physical_nonzero"] for r in edits + replays)
        and all(
            r["requested"]
            is r["target_sign"]
            is r["requested_label"]
            is r["requested_token_id"]
            is r["signed_margin"]
            is r["signed_delta_log_odds"]
            is None
            for r in rows
        ),
        "four OFF baselines/four nonzero null-target edits/four matched replays",
    )
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
            label: sum(r["actual_next_token_label"] == label for r in baselines)
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


def source_identity():
    result = {}
    paths = [
        *protocol.build_plan()["input_sha256"],
        *(getattr(protocol, key) for key in ("CONFIG", "DOC", "SCRIPT", "VERIFY", "TEST", "PLAN")),
        "scripts/crossed_pair_probe.py",
        "scripts/crossed_pair_plan.py",
        "scripts/verify_crossed_pair_probe.py",
        "scripts/shared_comply_crossed_plan.py",
        "scripts/frozen_crossed_comply_f03_plan.py",
        "scripts/frozen_guarded_preserve_crossed_plan.py",
    ]
    for path in dict.fromkeys(paths):
        require(
            base.git(ROOT, "ls-files", "--", path)
            and not base.git(ROOT, "status", "--porcelain", "--", path),
            "clean tracked source/input " + path,
        )
        result[path] = protocol.sha((ROOT / path).read_bytes())
    return result


engine.validate_vectors = validate_vectors
engine.assess, engine.summarize = assess, summarize
engine.source_identity = source_identity


def preflight(worker_entry=False):
    from scripts.verify_midpoint_only_f03 import verify_renderings

    record = require_freeze()
    lock_path = protocol.OUTPUT + "/preregistration.json"
    require(
        base.git(ROOT, "show", "--pretty=", "--name-only", "HEAD").splitlines() == [lock_path]
        and base.git(ROOT, "rev-parse", "HEAD^") == record["source_commit"]
        and not base.git(ROOT, "status", "--porcelain", "--", lock_path),
        "clean preregistration-only HEAD and exact source parent",
    )
    expected = {"preregistration.json"} | (
        {"RUN_STARTED.json", "worker.log"} if worker_entry else set()
    )
    require(
        OUTPUT.is_dir() and {p.name for p in OUTPUT.iterdir()} == expected,
        "untouched namespace; no prior attempt or extra artifacts",
    )
    verify_renderings(record["plan"])
    return {
        "status": "ZERO_MODEL_PREFLIGHT_PASSED",
        "source_commit": record["source_commit"],
        "source_hash_entries": len(record["source_sha256"]),
        "preregistration_sha256": protocol.sha((OUTPUT / "preregistration.json").read_bytes()),
        "storage": protocol.storage_preflight(ROOT, record["plan"]["config"]),
        "prompts": 4,
        "forwards": 12,
        "derivatives": 0,
        "model_loads": 0,
        "tokenizer_loads": 0,
        "real_forwards": 0,
        "real_derivatives": 0,
    }


def checked_usage(usage):
    require(
        isinstance(usage, dict)
        and all(
            type(usage.get(k)) in (int, float) and math.isfinite(usage[k])
            for k in ("standard_used_percent", "checked_at_unix")
        )
        and 0 <= usage["standard_used_percent"] < 90
        and 0 <= time.time() - usage["checked_at_unix"] <= 60,
        "finite fresh usage below90",
    )
    return usage


def worker():
    preflight(worker_entry=True)
    started = protocol.read(OUTPUT / "RUN_STARTED.json")
    checked_usage(started.get("usage_preflight"))
    require(
        all(
            type(started.get(k)) in (int, float) and math.isfinite(started[k])
            for k in (
                "started_monotonic",
                "deadline_monotonic",
                "forward_ceiling",
                "derivative_ceiling",
                "timeout_seconds",
            )
        )
        and started["command"] == [sys.executable, "-u", str(ROOT / protocol.SCRIPT), "_worker"]
        and started["forward_ceiling"] == 12
        and started["derivative_ceiling"] == 0
        and started["timeout_seconds"] == 600
        and started["deadline_monotonic"] == started["started_monotonic"] + 600
        and started["started_monotonic"] <= time.monotonic() < started["deadline_monotonic"],
        "exact finite12/0/600 worker envelope before claim/load",
    )
    return engine.worker()


def run():
    preflight()
    usage = checked_usage(json.loads(os.environ.get("SP_LENSE_USAGE_PREFLIGHT", "null")))
    return supervise([sys.executable, "-u", str(ROOT / protocol.SCRIPT), "_worker"], OUTPUT, usage)


if __name__ == "__main__":
    commands = {"freeze": freeze, "preflight": preflight, "run": run, "_worker": worker}
    require(
        len(sys.argv) == 2 and sys.argv[1] in commands,
        "Use freeze/preflight/run only; no recipe options",
    )
    result = commands[sys.argv[1]]()
    print(json.dumps(result, indent=2, allow_nan=False))
    if sys.argv[1] == "run" and result["status"] != "complete_valid":
        raise SystemExit(1)
