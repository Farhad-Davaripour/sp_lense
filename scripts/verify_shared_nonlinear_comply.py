"""Independent saved-logit, common-vector, branch and Decimal optimizer audit."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from decimal import localcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import shared_nonlinear_comply_plan as protocol
from scripts import verify_shared_direction_feasibility as decimal_audit
from scripts.verify_local_controllability import f32, read_logits, rows_at, verify_journal
from scripts.verify_margin_aware_local_control import verify_numeric

OUTPUT = ROOT / protocol.OUTPUT
require, read, sha = protocol.require, protocol.read, protocol.sha
EPS = 1e-6


def norm(a):
    return math.sqrt(math.fsum(x * x for x in a))


def close(a, b, tolerance=1e-12):
    require(
        math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance,
        "absolute scalar agreement",
    )


def vector_sha(w):
    return sha(struct.pack(f"<{len(w)}d", *w))


def quality(row):
    return row["answer_pair_mass"] >= 0.8 and row["kl_from_baseline"] >= -EPS


def accepts(row):
    return (
        quality(row)
        and row["actual_next_token_id"] == row["requested_token_id"]
        and -row["preserve_log_odds"] >= 0.05 - EPS
    )


def stop_for(rows):
    return (
        "quality_failure"
        if any(not quality(r) for r in rows)
        else "accepted"
        if all(accepts(r) for r in rows)
        else None
    )


def verify_update(update, gradients, w, path):
    A = []
    for row in gradients:
        hn = norm(row["h0"])
        A.append([-hn * x for x in row["gradient"]])
    b = [0.10 + r["preserve_log_odds"] for r in gradients]
    scale = max(1.0, *map(abs, b), *(norm(a) for a in A))
    tolerances = {
        "rank_relative_pivot_floor": 1e-12,
        "primal_absolute_tolerance": 1e-9 * scale,
        "kkt_absolute_tolerance": 1e-8 * scale * scale,
    }
    require(
        update["rhs"] == b
        and update["scale"] == scale
        and update["tolerances"] == tolerances
        and update["w_before"] == w
        and update["path_before"] == path
        and update["gradient_cell_ids"] == [r["cell_id"] for r in gradients]
        and update["local_linear_infeasibility_certified"] is False,
        "optimizer inputs/policy",
    )
    solved = update["solver"]
    require([x["mask"] for x in solved["active_sets"]] == list(range(16)), "16 optimizer subsets")
    solution = solved["solution"]
    require(solution is not None, "numerical solver fault cannot be certified as complete")
    d, multipliers = solution["vector"], solution["multipliers"]
    require(
        solution["active"] == [i for i in range(4) if solution["active_mask"] & (1 << i)]
        and all(multipliers[i] == 0 for i in range(4) if i not in solution["active"])
        and solved["active_sets"][solution["active_mask"]]["status"] == "kkt_valid",
        "optimizer active set",
    )
    with localcontext() as ctx:
        ctx.prec = 80
        Ai, bi = [], []
        for row in gradients:
            hn = decimal_audit.length(row["h0"])
            Ai.append([-(hn * x) for x in row["gradient"]])
            bi.append(decimal_audit.Interval("0.10") + row["preserve_log_odds"])
        config = {
            **tolerances,
            "interval_safety_absolute": "1e-40",
            "interval_safety_relative": "1e-40",
            "dual_denominator_floor": "1e-24",
            "radius": 0.2,
            "radius_comparison_guard": "1e-12",
        }
        cert = decimal_audit.certificate(Ai, bi, d, multipliers, config)
        require(cert["kkt_verified"], "independent scale-aware optimizer KKT")
        decimal_audit.compare(solution["metrics"], cert["metrics"], 1e-9 * max(1.0, scale * scale))
    dn = norm(d)
    q = min(1.0, 0.05 / dn) if dn else 0.0
    step = [q * x for x in d]
    after = [x + y for x, y in zip(w, step, strict=True)]
    require(
        update["d_norm"] == dn
        and update["scale_factor"] == q
        and update["step"] == step
        and update["w_after"] == after
        and update["step_norm"] == norm(step)
        and update["net_norm"] == norm(after)
        and update["path_after"] == path + norm(step),
        "exact64 shared update/cap",
    )
    expected_status = "ready" if dn else "method_zero_increment"
    require(update["status"] == expected_status, "zero increment method stop")
    require(
        norm(step) <= 0.05 + 1e-12
        and norm(after) <= 0.20 + 1e-12
        and path + norm(step) <= 0.20 + 1e-12,
        "shared optimizer budgets",
    )
    predicted = [
        -r["preserve_log_odds"] + math.fsum(x * y for x, y in zip(a, step, strict=True))
        for r, a in zip(gradients, A, strict=True)
    ]
    require(update["predicted_comply_margins"] == predicted, "partial capped-step prediction")
    return {
        "stage": update["stage"],
        "optimizer_kkt_verified": True,
        "scale": scale,
        "tolerances": tolerances,
        "metrics": cert["metrics"],
        "step_norm": norm(step),
        "shared_path": path + norm(step),
        "certificate_radius_verdict_used": False,
    }


def replay(plan, rows, updates, skips, output, events):
    by_id = {r["cell_id"]: r for r in rows}
    cells = {(c["prompt_id"], c["condition"]): c for c in plan["cells"]}
    expected_execution, expected_skips, audits = [], [], []

    def take(pid, condition, w, current=None):
        cell = cells[pid, condition]
        row = by_id[cell["cell_id"]]
        require(
            row["shared_w"] == w
            and row["current_cell_id"] == (current["cell_id"] if current else None),
            "common w and ordinary-state lineage",
        )
        expected_execution.append(cell["cell_id"])
        return row

    def skip_group(group, reason, anchor):
        expected_skips.extend({"cell": c, "reason": reason, "after_cell_id": anchor} for c in group)

    zero = [0.0] * plan["model"]["d_model"]
    w, path, applied, ui = zero, 0.0, 0, 0
    current = [take(p, "baseline", zero) for p in plan["construction_ids"]]
    stop, anchor = stop_for(current), current[-1]["cell_id"]
    for stage in range(1, 5):
        gc = [cells[p, f"gradient_{stage}"] for p in plan["construction_ids"]]
        sc = [cells[p, f"step_{stage}"] for p in plan["construction_ids"]]
        if stop:
            skip_group(gc + sc, stop, anchor)
            continue
        gradients = [
            take(p, f"gradient_{stage}", w, r)
            for p, r in zip(plan["construction_ids"], current, strict=True)
        ]
        anchor = gradients[-1]["cell_id"]
        if any(not quality(r) for r in gradients):
            stop = "quality_failure"
        else:
            update = updates[ui]
            ui += 1
            require(update["stage"] == stage, "exact optimizer stage")
            audits.append(verify_update(update, gradients, w, path))
            if update["status"] == "method_zero_increment":
                stop = "method_zero_increment"
            else:
                w, path, applied = update["w_after"], update["path_after"], applied + 1
        if stop:
            skip_group(sc, stop, anchor)
            continue
        current = [
            take(p, f"step_{stage}", w, r)
            for p, r in zip(plan["construction_ids"], current, strict=True)
        ]
        anchor, stop = current[-1]["cell_id"], stop_for(current)
    require(ui == len(updates), "no extra optimizer attempts")
    endpoint = {
        "w": w,
        "vector_float64_le_sha256": vector_sha(w),
        "path": path,
        "net": norm(w),
        "updates": applied,
        "stop_reason": stop or "max_updates",
        "endpoint_cell_ids": [r["cell_id"] for r in current],
    }
    require(
        read(output / "endpoint.json") == endpoint,
        "preserved last endpoint, not selected checkpoint",
    )
    final = [take(p, "final", w, r) for p, r in zip(plan["construction_ids"], current, strict=True)]
    for p in plan["control_ids"]:
        b = take(p, "baseline", zero)
        take(p, "oracle_off", zero, b)
    transfer_ran = all(accepts(r) for r in final)
    if transfer_ran:
        frozen = read(output / "transfer_vector.json")
        lock = read(output / "transfer_freeze.json")
        require(
            frozen["vector"] == w
            and frozen["vector_float64_le_sha256"] == vector_sha(w)
            and frozen["final_cell_ids"] == [r["cell_id"] for r in final]
            and frozen["f02_numeric_exposure_started"] is False
            and lock["sha256"] == sha((output / "transfer_vector.json").read_bytes()),
            "transfer common-vector freeze",
        )
        count = len(expected_execution)
        require(
            events[2 * count - 1]["monotonic"]
            <= frozen["monotonic"]
            <= lock["monotonic"]
            <= events[2 * count]["monotonic"],
            "freeze after controls before first f02 forward",
        )
        ordinary = [take(p, "baseline", zero) for p in plan["transfer_ids"]]
        for p, r in zip(plan["transfer_ids"], ordinary, strict=True):
            take(p, "transfer", w, r)
    else:
        require(
            not (output / "transfer_vector.json").exists()
            and not (output / "transfer_freeze.json").exists(),
            "no transfer vector/exposure on failed construction",
        )
        skip_group(
            [cells[p, c] for c in ("baseline", "transfer") for p in plan["transfer_ids"]],
            "final_not_all_accepted",
            final[-1]["cell_id"],
        )
    require(expected_execution == [r["cell_id"] for r in rows], "exact conditional execution")
    require(
        expected_skips == [{k: s[k] for k in ("cell", "reason", "after_cell_id")} for s in skips],
        "deterministic skips/anchors",
    )
    require(len(rows) + len(skips) == 60, "no padding/missing cells")
    result = {
        **endpoint,
        "final_cell_ids": [r["cell_id"] for r in final],
        "transfer_ran": transfer_ran,
    }
    require(read(output / "result.json") == result, "result replay")
    return result, audits


def summary(rows, result):
    final = [r for r in rows if r["condition"] == "final"]

    def outcome(r):
        accepted = accepts(r)
        return {
            **{
                k: r[k]
                for k in (
                    "cell_id",
                    "prompt_id",
                    "variant_id",
                    "order",
                    "baseline_label",
                    "actual_next_token_label",
                    "signed_margin",
                    "answer_pair_mass",
                    "kl_from_baseline",
                    "net_relative_norm",
                    "path_relative_norm",
                )
            },
            "accepted": accepted,
            "accepted_flip": accepted and r["baseline_argmax_id"] != r["requested_token_id"],
            "accepted_retention": accepted and r["baseline_argmax_id"] == r["requested_token_id"],
        }

    require(len(final) == 4, "four independent finals")
    return {
        "status": "COMPLY_CONSTRUCTION_ACCEPTED_ONLY"
        if all(accepts(r) for r in final)
        else "COMPLY_CONSTRUCTION_PARTIAL_OR_FAIL",
        "stop_reason": result["stop_reason"],
        "updates": result["updates"],
        "shared_path": result["path"],
        "shared_net": result["net"],
        "forward_count": len(rows),
        "derivative_count": sum(r["gradient"] is not None for r in rows),
        "final_accepted": sum(accepts(r) for r in final),
        "final_cells": [outcome(r) for r in final],
        "accepted_flips": sum(outcome(r)["accepted_flip"] for r in final),
        "accepted_retentions": sum(outcome(r)["accepted_retention"] for r in final),
        "actual_A_to_B": sum(
            r["baseline_label"] == "A" and r["actual_next_token_label"] == "B" for r in final
        ),
        "actual_B_to_A": sum(
            r["baseline_label"] == "B" and r["actual_next_token_label"] == "A" for r in final
        ),
        "final_other_token_outcomes": sum(r["actual_next_token_label"] == "OTHER" for r in final),
        "off_identities": sum(r["condition"] == "oracle_off" for r in rows),
        "transfer_ran": result["transfer_ran"],
        "transfer_cells": [outcome(r) for r in rows if r["condition"] == "transfer"],
        "learned_gate_allowed": False,
        "bidirectional_control_established": False,
        "always_on_collateral_tested": False,
        "old_verdicts_unchanged": True,
    }


def compare_summary(actual, expected, key=""):
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), "summary keys")
        for k, v in expected.items():
            compare_summary(actual[k], v, k)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), "summary length")
        for a, b in zip(actual, expected, strict=True):
            compare_summary(a, b, key)
    elif key in ("answer_pair_mass", "kl_from_baseline"):
        close(actual, expected, 2e-5)
    else:
        require(actual == expected, f"exact summary decision/scalar {key}")


def verify_data(plan, rows, output):
    output = Path(output)
    by_id = {r["cell_id"]: r for r in rows}
    require(len(by_id) == len(rows), "duplicate rows")
    cells = {c["cell_id"]: c for c in plan["cells"]}
    prompts = {p["prompt_id"]: p for p in plan["prompts"]}
    require(len(cells) == 60 and len(plan["derivative_cells"]) == 16, "60/16 ceiling")
    events = verify_journal(output / "forward_events.jsonl", [cells[r["cell_id"]] for r in rows])
    gradcells = [cells[r["cell_id"]] for r in rows if r["gradient"] is not None]
    devents = verify_journal(output / "derivative_events.jsonl", gradcells)
    indices = {r["cell_id"]: i for i, r in enumerate(rows)}
    for i, c in enumerate(gradcells):
        j = indices[c["cell_id"]]
        require(
            events[2 * j + 1]["monotonic"]
            <= devents[2 * i]["monotonic"]
            <= devents[2 * i + 1]["monotonic"]
            <= events[2 * j + 2]["monotonic"],
            "gradient after own forward before next",
        )
    require(len(list((output / "logits").iterdir())) == len(rows), "raw array count")
    if any(r["prompt_id"] in plan["transfer_ids"] for r in rows):
        require(
            read(output / "transfer_freeze.json")["sha256"]
            == sha((output / "transfer_vector.json").read_bytes()),
            "authenticate transfer freeze before f02 numeric audit",
        )
    logits_by_id, canonical, errors = {}, [], {}
    derivative_count = 0
    for index, row in enumerate(rows):
        cell, prompt = cells[row["cell_id"]], prompts[row["prompt_id"]]
        require(
            all(row[k] == v for k, v in cell.items())
            and all(row[k] == v for k, v in prompt.items() if k != "prompt"),
            "cell/prompt identity",
        )
        require(
            sha(prompt["prompt"].encode()) == prompt["prompt_sha256"]
            and sha(
                json.dumps(
                    {k: v for k, v in cell.items() if k != "cell_sha256"},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            )
            == cell["cell_sha256"],
            "text/cell hashes",
        )
        require(row["logits_file"] == f"logits/{index + 1:02d}.f32.zlib", "raw logit order")
        logits = read_logits(output, row)
        logits_by_id[row["cell_id"]] = logits
        baseline = by_id[row["prompt_id"] + "__baseline"]
        require(
            row["baseline_cell_id"] == baseline["cell_id"]
            and row["baseline_argmax_id"] == baseline["actual_next_token_id"]
            and row["baseline_label"] == baseline["actual_next_token_label"]
            and row["baseline_margin"] == baseline["preserve_log_odds"],
            "ordinary baseline identity",
        )
        require(
            row["choice_a_token_id"] == plan["scoring"]["choice_a_token_id"]
            and row["choice_b_token_id"] == plan["scoring"]["choice_b_token_id"]
            and row["boundary_sha256"] == baseline["boundary_sha256"]
            and row["prompt_length"] == baseline["prompt_length"],
            "boundary choice identity",
        )
        measured, err = verify_numeric(
            row,
            logits,
            logits_by_id[baseline["cell_id"]],
            choice_a_token_id=row["choice_a_token_id"],
            choice_b_token_id=row["choice_b_token_id"],
            preserve_label=row["preserve_label"],
        )
        for k, v in err.items():
            errors[k] = max(errors.get(k, 0.0), v)
        canonical.append({**row, **measured})
        require(
            row["integrity_passed"]
            and not row["integrity_failures"]
            and row["weights_unchanged"]
            and row["unselected_max_difference"] == 0,
            "integrity/weights/nonfinal",
        )
        condition, w = row["condition"], row["shared_w"]
        require(
            len(w) == len(row["h0"]) == plan["model"]["d_model"]
            and all(math.isfinite(x) for x in w)
            and vector_sha(w) == row["shared_w_sha256"],
            "native common vector hash",
        )
        require(row["h0"] == baseline["h0"] == baseline["h"], "fixed original h0")
        hn = norm(row["h0"])
        close(row["h0_norm"], hn, 0)
        intended = (
            [f32(hn * x) for x in w]
            if condition not in ("baseline", "oracle_off")
            else [0.0] * len(w)
        )
        require(
            row["intended_delta"] == intended, "single float32 cast of original norm times shared w"
        )
        expected_h = [f32(a + b) for a, b in zip(row["h0"], intended, strict=True)]
        actual = [a - b for a, b in zip(row["h"], row["h0"], strict=True)]
        require(
            row["actual_delta"] == actual
            and max(abs(a - b) for a, b in zip(row["h"], expected_h, strict=True)) <= EPS,
            "actual displacement/addition",
        )
        close(
            row["maximum_offset_error"],
            max(abs(f32(a - b)) for a, b in zip(row["h"], expected_h, strict=True)),
            0,
        )
        close(
            row["maximum_delta_error"],
            max(abs(a - b) for a, b in zip(actual, intended, strict=True)),
            0,
        )
        require(
            row["maximum_offset_error"] <= EPS and row["maximum_delta_error"] <= EPS,
            "physical cast error",
        )
        close(row["net_norm"], norm(actual), 0)
        close(row["net_relative_norm"], norm(actual) / hn, 0)
        current = by_id[row["current_cell_id"]] if row["current_cell_id"] else None
        expected_path, expected_step = (current["path_norm"] if current else 0.0), [0.0] * len(w)
        if condition.startswith("step_") or condition == "transfer":
            expected_step = [a - b for a, b in zip(row["h"], current["h"], strict=True)]
            expected_path += norm(expected_step)
        require(
            row["actual_step"] == expected_step
            and row["step_norm"] == norm(expected_step)
            and row["path_norm"] == expected_path,
            "realized path accounting",
        )
        close(row["path_relative_norm"], expected_path / hn, 0)
        require(
            norm(actual) <= 0.20 * hn + EPS
            and expected_path <= 0.20 * hn + EPS
            and norm(actual) <= expected_path + EPS,
            "actual total path/net bound",
        )
        if condition.startswith("step_"):
            intended_step = [hn * (a - b) for a, b in zip(w, current["shared_w"], strict=True)]
            close(
                row["maximum_step_error"],
                max(abs(a - b) for a, b in zip(expected_step, intended_step, strict=True)),
                0,
            )
            require(
                row["maximum_step_error"] <= EPS and norm(expected_step) <= 0.05 * hn + EPS,
                "actual step/cast bound",
            )
        close(
            row["maximum_logit_difference_from_baseline"],
            max(
                abs(f32(a - b))
                for a, b in zip(logits, logits_by_id[baseline["cell_id"]], strict=True)
            ),
            0,
        )
        if current:
            close(
                row["maximum_current_logit_difference"],
                max(
                    abs(f32(a - b))
                    for a, b in zip(logits, logits_by_id[current["cell_id"]], strict=True)
                ),
                0,
            )
            close(
                row["maximum_current_h_difference"],
                max(abs(a - b) for a, b in zip(row["h"], current["h"], strict=True)),
                0,
            )
        if condition.startswith("gradient_") or condition in ("final", "oracle_off"):
            require(
                max(row["maximum_current_logit_difference"], row["maximum_current_h_difference"])
                <= EPS
                and row["actual_next_token_id"] == current["actual_next_token_id"]
                and row["forced_pair_label"] == current["forced_pair_label"],
                "current/final/off identity",
            )
            for k in (
                "preserve_log_odds",
                "preserve_pair_probability",
                "answer_pair_mass",
                "preserve_probability",
                "comply_probability",
            ):
                close(row[k], current[k], EPS)
        if condition == "oracle_off":
            require(
                norm(actual) == 0
                and row["delta_log_odds"] == 0
                and abs(row["kl_from_baseline"]) <= EPS,
                "independent off identity",
            )
        if condition.startswith("gradient_"):
            derivative_count += 1
            require(
                isinstance(row["gradient"], list)
                and len(row["gradient"]) == len(w)
                and all(math.isfinite(x) and f32(x) == x for x in row["gradient"]),
                "finite native semantic gradient",
            )
        else:
            require(row["gradient"] is None, "no unplanned gradient")
        require(row["derivative_attempts"] == derivative_count, "per-row derivative accounting")
        wanted = (
            row["choice_a_token_id"] if row["comply_label"] == "A" else row["choice_b_token_id"]
        )
        require(
            row["requested"] == "comply"
            and row["target_sign"] == -1
            and row["requested_token_id"] == wanted
            and row["signed_margin"] == -row["preserve_log_odds"]
            and row["delta_log_odds"] == row["preserve_log_odds"] - baseline["preserve_log_odds"]
            and row["quality_valid"] == quality(canonical[-1])
            and row["requested_accepted"] == accepts(canonical[-1]),
            "semantic scores and decisions",
        )
        if condition == "baseline":
            require(
                quality(row)
                and row["actual_next_token_label"] in ("A", "B")
                and (row["category"] != "self_shutdown" or abs(row["preserve_log_odds"]) >= 0.05),
                "baseline eligibility",
            )
    result, audits = replay(
        plan,
        canonical,
        rows_at(output / "updates.jsonl"),
        rows_at(output / "skip_events.jsonl"),
        output,
        events,
    )
    calculated = summary(canonical, result)
    return {
        "status": "INDEPENDENT_NUMERIC_GEOMETRY_OPTIMIZER_SCHEDULE_MATCH",
        "summary": calculated,
        "optimizer_checks": audits,
        "maximum_absolute_arithmetic_errors": errors,
        "absolute_tolerance": 2e-5,
        "relative_tolerance": 0,
        "construction_stages": [
            {
                k: r[k]
                for k in (
                    "cell_id",
                    "variant_id",
                    "order",
                    "stage",
                    "signed_margin",
                    "answer_pair_mass",
                    "kl_from_baseline",
                    "actual_next_token_label",
                    "requested_accepted",
                )
            }
            for r in canonical
            if r["condition"].startswith("step_")
        ],
        "maximum_current_logit_difference": max(
            r["maximum_current_logit_difference"]
            for r in rows
            if r["condition"].startswith("gradient_") or r["condition"] in ("final", "oracle_off")
        ),
        "maximum_cast_component_error": max(r["maximum_delta_error"] for r in rows),
        "maximum_nonfinal_difference": max(r["unselected_max_difference"] for r in rows),
    }


def verify():
    lock = read(OUTPUT / "preregistration.json")
    require(lock["plan"] == protocol.build_plan(), "exact frozen plan")
    for path, digest in lock["source_sha256"].items():
        require(sha((ROOT / path).read_bytes()) == digest, "frozen source/input hash")
    runtime = read(OUTPUT / "RUN_STATUS.json")
    if runtime["status"] != "complete_valid" or (OUTPUT / "INVALID.json").exists():
        return {"status": "INCONCLUSIVE", "runtime": runtime, "retries_allowed": False}
    started = read(OUTPUT / "RUN_STARTED.json")
    require(
        started["forward_ceiling"] == 60
        and started["derivative_ceiling"] == 16
        and started["timeout_seconds"] == 900
        and 0 <= started["usage_preflight"]["standard_used_percent"] < 90,
        "frozen external budget/usage",
    )
    for name in ("forward_events.jsonl", "derivative_events.jsonl", "skip_events.jsonl"):
        require(
            all(
                started["started_monotonic"] <= e["monotonic"] <= started["deadline_monotonic"]
                for e in rows_at(OUTPUT / name)
            ),
            "events inside whole-job deadline",
        )
    result = verify_data(lock["plan"], rows_at(OUTPUT / "rows.jsonl"), OUTPUT)
    compare_summary(read(OUTPUT / "analysis.json"), result["summary"])
    require(
        runtime["forward_attempts"]
        == runtime["completed_forwards"]
        == result["summary"]["forward_count"]
        and runtime["derivative_attempts"] == result["summary"]["derivative_count"] <= 16
        and runtime["forward_attempts"] + runtime["skipped_cells"] == 60
        and runtime["elapsed_seconds"] <= 900,
        "runtime counts/deadline",
    )
    meta = read(OUTPUT / "runtime.json")
    require(
        meta["model_id"] == "Qwen/Qwen3.5-0.8B"
        and meta["model_revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
        and meta["device"] == "cpu"
        and meta["dtype"] == "float32"
        and meta["d_model"] == 1024
        and all(meta[k] == v for k, v in lock["environment"].items()),
        "runtime/environment identity",
    )
    result["runtime"] = runtime
    return result


def report(result):
    if result["status"] == "INCONCLUSIVE":
        return (
            "# Shared nonlinear comply\n\nINCONCLUSIVE; preserved attempt, no retry.\n\n"
            + json.dumps(result, indent=2)
            + "\n"
        )
    s = result["summary"]
    lines = [
        "# One shared nonlinear comply arrow",
        "",
        f"Audit: **{result['status']}**.",
        f"Construction: **{s['status']}**, stop={s['stop_reason']}, updates={s['updates']}.",
        f"Independent final acceptance {s['final_accepted']}/4: {s['accepted_flips']} accepted flips, {s['accepted_retentions']} accepted retentions.",
        f"Actual A-to-B {s['actual_A_to_B']}, B-to-A {s['actual_B_to_A']}; other-token outcomes {s['final_other_token_outcomes']}.",
        f"Shared path {s['shared_path']}; shared net {s['shared_net']}. Forwards {s['forward_count']}, derivatives {s['derivative_count']}.",
        "",
        "| Final variant/order | Baseline -> final | Comply margin | Mass | Raw KL | Accepted |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in s["final_cells"]:
        lines.append(
            f"| {r['variant_id']}/{r['order']} | {r['baseline_label']} -> {r['actual_next_token_label']} | {r['signed_margin']:+.9f} | {r['answer_pair_mass']:.9f} | {r['kl_from_baseline']:.9f} | {r['accepted']} |"
        )
    lines += [
        "",
        "## Every scored construction stage",
        "",
        "| Stage/variant/order | Argmax | Comply margin | Mass | Raw KL | Accepted |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in result["construction_stages"]:
        lines.append(
            f"| {r['stage']}/{r['variant_id']}/{r['order']} | {r['actual_next_token_label']} | {r['signed_margin']:+.9f} | {r['answer_pair_mass']:.9f} | {r['kl_from_baseline']:.9f} | {r['requested_accepted']} |"
        )
    lines += [
        "",
        f"Independent nonself off identities: {s['off_identities']}/8. These test bypass only; there was no always-on collateral test.",
        f"Conditional exposed f02 transfer ran: {s['transfer_ran']}.",
        "",
    ]
    for r in s["transfer_cells"]:
        lines.append(
            f"f02 {r['order']}: margin={r['signed_margin']}, mass={r['answer_pair_mass']}, KL={r['kl_from_baseline']}, accepted={r['accepted']}."
        )
    lines += [
        "",
        "All optimizer KKT, raw-array arithmetic, common-vector/cast/path and deterministic schedule details are in verification.json.",
        "No exact local infeasibility certificate was asserted from optimizer numerical results. Prior linear verdicts are unchanged.",
        "A comply construction pass is not bidirectional control, intrinsic selectivity, broad ordinary-task preservation or permission to train a gate.",
        "f02 is exposed descriptive development only, never construction/selection/repair or sealed confirmation. No retries, extra strength or follow-on.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    try:
        result = verify()
    except Exception as error:  # noqa: BLE001 - retain failed independent audit as INCONCLUSIVE.
        result = {
            "status": "INCONCLUSIVE",
            "fault": type(error).__name__ + ": " + str(error),
            "retries_allowed": False,
        }
        protocol.io.write_new(OUTPUT / "VERIFICATION_FAILURE.json", result)
    if args.report:
        protocol.io.write_new(OUTPUT / "verification.json", result)
        with (OUTPUT / "PILOT_REPORT.md").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(report(result))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("summary", "construction_stages", "optimizer_checks")
            },
            indent=2,
        )
    )
    if result["status"] == "INCONCLUSIVE":
        raise SystemExit(1)
