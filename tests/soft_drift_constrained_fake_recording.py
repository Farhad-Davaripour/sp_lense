"""One fixed stdlib-only native/full-vocabulary recording fixture, not a model.

Both outcomes use the same fixed216-cell schedule and eight actual QP proposals.
Scores/gradients are supplied abstract numbers, NOT derivatives of an LM or proof
of a hook. The real driver, ledgers, optimizer, raw independent verifier, compact
serializer and final inventory are exercised without modifying production code.
"""

from __future__ import annotations

import array
import hashlib
import json
import sys
import time
import types
import zlib
from pathlib import Path

from scripts import soft_drift_constrained_comply as job
from scripts import soft_drift_constrained_comply_plan as plan_module
from scripts import soft_drift_constrained_comply_recording as recording
from scripts import three_family_recording_bindings as bindings
from scripts.future_choice_scoring_reference import reference_score
from scripts.verify_local_controllability import f32


def build_full_recording(output, *, accepted):
    """Build one positive or negative max-size fake recording; never load a model."""
    output = Path(output)
    plan = plan_module.build_plan()
    assert plan["model"]["d_model"] == 1024
    assert plan["config"]["storage"]["vocabulary"] == 248320
    budget = recording.PairedBudget(output)
    start = time.monotonic()
    deadline = start + 1800
    prompts = {p["prompt_id"]: p for p in plan["prompts"]}
    rows, baselines, score_cache, raw_cache = [], {}, {}, {}
    native_norms = {pid: float(2 ** (i % 3)) for i, pid in enumerate(plan["construction_ids"])}
    choice_a, choice_b = (
        plan["scoring"][key] for key in ("choice_a_token_id", "choice_b_token_id")
    )
    pending_gradient = [None]
    autograd = types.SimpleNamespace(grad=lambda: pending_gradient[0], backward=lambda: None)
    fake_torch_namespace = types.SimpleNamespace(autograd=autograd)

    def logits_and_score(margin, label):
        key = (margin, label)
        if key not in raw_cache:
            logits = array.array("f", [-80.0]) * 248320
            pi, ci = (choice_a, choice_b) if label == "A" else (choice_b, choice_a)
            logits[pi], logits[ci] = f32(margin), 0.0
            raw = array.array("f", logits)
            if sys.byteorder != "little":
                raw.byteswap()
            encoded = raw.tobytes()
            raw_cache[key] = (logits, zlib.compress(encoded), hashlib.sha256(encoded).hexdigest())
        if key not in score_cache:
            baseline_logits = raw_cache[(0.06, label)][0]
            score_cache[key] = reference_score(
                raw_cache[key][0],
                baseline_logits,
                choice_a_token_id=choice_a,
                choice_b_token_id=choice_b,
                preserve_label=label,
            )
        return raw_cache[key], score_cache[key]

    with bindings.bind_writers(budget):
        # The fixture's provenance explicitly identifies synthetic records. Real
        # source authentication still runs over every existing plan-bound input.
        env = {"fixture_kind": "stdlib_native_full_vocabulary_records_no_model_or_hook"}
        job.base.write_new(
            output / "preregistration.json",
            {
                "plan": plan,
                "source_sha256": plan["input_sha256"],
                "environment": env,
            },
        )
        job.base.write_new(
            output / "RUN_STARTED.json",
            {
                "forward_ceiling": 216,
                "derivative_ceiling": 96,
                "timeout_seconds": 1800,
                "usage_preflight": {"standard_used_percent": 0},
                "started_monotonic": start,
                "deadline_monotonic": deadline,
            },
        )
        job.base.write_new(
            output / "runtime.json",
            {
                "model_id": "Qwen/Qwen3.5-0.8B",
                "model_revision": "2fc06364715b967f1860aea9cf38778875588b17",
                "device": "cpu",
                "dtype": "float32",
                "d_model": 1024,
                **env,
            },
        )
        job.base.write_new(
            output / "storage_preflight.json",
            {
                "bounds": plan["config"]["storage"],
                "available_free_bytes": 1024**3,
                "passed": True,
                "monotonic": time.monotonic(),
            },
        )
        budget.write_bytes(output / "worker.log", b"stdlib fake recording; no model or hook\n")
        output.joinpath("logits").mkdir()
        ledger = job.ForwardLedger(output, plan["cells"], deadline)
        derivatives = job.Derivatives(
            fake_torch_namespace, output, plan["derivative_cells"], deadline
        )
        budget.write_bytes(output / "updates.jsonl", b"")

        def call(cell, w, current):
            p = prompts[cell["prompt_id"]]
            condition = cell["condition"]
            hn = native_norms[p["prompt_id"]]
            if condition == "baseline":
                margin = 0.06
            elif condition.startswith("step_"):
                stage = cell["stage"]
                margin = (
                    -0.08
                    if accepted and stage == 8
                    else -0.02
                    if stage == 8
                    else 0.06 - 0.01 * stage
                )
            else:
                margin = current["fixture_margin"]
            raw, score = logits_and_score(margin, p["preserve_label"])
            ledger.begin(cell)
            logit_file = f"logits/{ledger.attempts:02d}.f32.zlib"
            budget.write_bytes(output / logit_file, raw[1])
            ledger.finish(True)
            gradient = None
            if condition.startswith("gradient_"):
                pending_gradient[0] = [f32(-16.0 / hn)] + [0.0] * 1023
                derivatives.cell = cell
                gradient = autograd.grad()
            h0 = [0.0, hn] + [0.0] * 1022
            intended = [f32(hn * x) for x in w]
            h = [f32(x + y) for x, y in zip(h0, intended, strict=True)]
            delta = [x - y for x, y in zip(h, h0, strict=True)]
            before = current["row"] if current else None
            step = (
                [x - y for x, y in zip(h, before["h"], strict=True)]
                if condition.startswith("step_")
                else [0.0] * 1024
            )
            row = {
                **{k: v for k, v in p.items() if k != "prompt"},
                **cell,
                **score,
                "choice_a_token_id": choice_a,
                "choice_b_token_id": choice_b,
                "boundary_sha256": "0" * 64,
                "prompt_length": 128,
                "logits_file": logit_file,
                "logits_sha256": raw[2],
                "logit_count": 248320,
                "h0": h0,
                "h": h,
                "h0_norm": hn,
                "shared_w": list(w),
                "shared_w_sha256": job.vector_sha(w),
                "intended_delta": intended,
                "actual_delta": delta,
                "net_norm": job.norm(delta),
                "net_relative_norm": job.norm(delta) / hn,
                "maximum_offset_error": 0.0,
                "maximum_delta_error": max(
                    abs(x - y) for x, y in zip(delta, intended, strict=True)
                ),
                "unselected_max_difference": 0.0,
                "maximum_logit_difference_from_baseline": abs(f32(f32(margin) - f32(0.06))),
                "gradient": gradient,
                "requested": "comply",
                "requested_token_id": choice_a if p["comply_label"] == "A" else choice_b,
                "signed_margin": -score["preserve_log_odds"],
                "target_sign": -1,
                "current_cell_id": before["cell_id"] if before else None,
                "maximum_current_logit_difference": abs(
                    f32(f32(margin) - f32(current["fixture_margin"]))
                )
                if current
                else 0.0,
                "maximum_current_h_difference": max(
                    abs(x - y) for x, y in zip(h, before["h"], strict=True)
                )
                if before
                else 0.0,
                "path_norm": (before["path_norm"] if before else 0.0) + job.norm(step),
                "actual_step": step,
                "step_norm": job.norm(step),
                "weights_unchanged": True,
                "derivative_attempts": derivatives.attempts,
            }
            if condition == "baseline":
                baselines[p["prompt_id"]] = row
            base = baselines[p["prompt_id"]]
            row.update(
                baseline_cell_id=base["cell_id"],
                baseline_argmax_id=base["actual_next_token_id"],
                baseline_label=base["actual_next_token_label"],
                baseline_margin=base["preserve_log_odds"],
                delta_log_odds=row["preserve_log_odds"] - base["preserve_log_odds"],
                signed_delta_log_odds=-(row["preserve_log_odds"] - base["preserve_log_odds"]),
                path_relative_norm=row["path_norm"] / hn,
            )
            if condition.startswith("step_"):
                row["maximum_step_error"] = max(
                    abs(x - hn * (a - b))
                    for x, a, b in zip(step, w, before["shared_w"], strict=True)
                )
            row.update(
                quality_valid=job.quality(row),
                requested_accepted=job.accepts(row),
                integrity_passed=True,
                integrity_failures=[],
            )
            job.base.append_row(output / "rows.jsonl", row)
            rows.append(row)
            if len(rows) % 12 == 0:
                print(
                    json.dumps(
                        {
                            "fixture_progress_only_not_audit": True,
                            "accepted_scripted": accepted,
                            "rows_written": len(rows),
                            "condition": condition,
                            "elapsed_seconds": time.monotonic() - start,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            return {"row": row, "fixture_margin": margin}

        with derivatives:
            result = job.drive(
                plan,
                call,
                ledger.skip,
                job.increment,
                lambda update: job.base.append_row(output / "updates.jsonl", update),
                lambda endpoint: job.base.write_new(output / "endpoint.json", endpoint),
            )
        job.base.write_new(output / "result.json", result)
        job.base.write_new(output / "analysis.json", job.summarize(rows, result))
    assert ledger.attempts == ledger.completed == ledger.cursor == len(rows) == 216
    assert derivatives.attempts == derivatives.completed == 96
    assert result["updates"] == result["attempted_updates"] == 8
    assert result["candidate_eligible"] is accepted
    assert 0 < result["net"] < 0.20
    runtime = {
        "status": "complete_valid",
        "forward_attempts": 216,
        "completed_forwards": 216,
        "derivative_attempts": 96,
        "completed_derivatives": 96,
        "skipped_cells": 0,
        "elapsed_seconds": time.monotonic() - start,
    }
    raw_log = (output / "worker.log").read_bytes()
    capture = {
        "status": "complete_valid",
        "quiescent": True,
        "process_attempts": 1,
        "worker_exit_code": 0,
        "reader_joined": True,
        "writer_joined": True,
        "prefix_bytes_written": len(raw_log),
        "prefix_sha256": recording.sha(raw_log),
        "fixture_kind": "simulated_receipt_no_process_was_launched",
    }
    return budget, runtime, capture, rows, result
