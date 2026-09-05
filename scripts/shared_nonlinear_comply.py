"""One shared nonlinear comply arrow; bounded conditional construction, never train weights."""

from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import refreshed_gradient_control as old
from scripts import shared_direction_linear_feasibility as optimizer
from scripts import shared_nonlinear_comply_plan as protocol
from sp_lense.future_choice_scoring import score_float32_logits

base, recorder = old.base, old.old
require, norm, dot = protocol.require, optimizer.norm, optimizer.dot
EPS, ROUND_EPS = 1e-6, 1e-12
OUTPUT = ROOT / protocol.OUTPUT


def vector_sha(w):
    return base.sha(struct.pack(f"<{len(w)}d", *w))


def quality(row):
    return row["answer_pair_mass"] >= 0.8 and row["kl_from_baseline"] >= -EPS


def accepts(row):
    return (
        quality(row)
        and row["actual_next_token_id"] == row["requested_token_id"]
        and -row["preserve_log_odds"] >= 0.05 - EPS
    )


def stopping(states):
    if any(not quality(s["row"]) for s in states):
        return "quality_failure"
    return "accepted" if all(accepts(s["row"]) for s in states) else None


class ForwardLedger(base.Ledger):
    """Same durable finish primitive; dedicated60-cell cap, without changing old48 cap."""

    def __init__(self, output, cells, deadline, now=time.monotonic):
        super().__init__(output, cells, deadline, now)
        self.cursor, self.skips = 0, []
        for name in ("forward_events.jsonl", "skip_events.jsonl"):
            with (self.output_dir / name).open("x", encoding="utf-8"):
                pass

    def begin(self, cell):
        require(
            not self.failed and not self.pending and self.now() < self.deadline,
            "forward failure/pending/deadline",
        )
        require(
            self.attempts < 60
            and self.cursor < len(self.cells)
            and cell == self.cells[self.cursor],
            "forward60/order",
        )
        self.attempts += 1
        self.pending = True
        base.append_row(
            self.output_dir / "forward_events.jsonl",
            {
                "event": "attempt_started",
                "attempt": self.attempts,
                "cell": cell,
                "monotonic": self.now(),
            },
        )
        self.cursor += 1

    def skip(self, cell, reason, anchor):
        require(
            not self.pending
            and not self.failed
            and self.cursor < len(self.cells)
            and cell == self.cells[self.cursor]
            and cell["optional"],
            "illegal skip",
        )
        require(
            reason
            in ("accepted", "quality_failure", "method_zero_increment", "final_not_all_accepted"),
            "skip reason",
        )
        event = {"cell": cell, "reason": reason, "after_cell_id": anchor, "monotonic": self.now()}
        base.append_row(self.output_dir / "skip_events.jsonl", event)
        self.skips.append(event)
        self.cursor += 1


class Derivatives:
    def __init__(self, torch, output, cells, deadline, now=time.monotonic):
        self.torch, self.output, self.cells, self.deadline, self.now = (
            torch,
            Path(output),
            cells,
            deadline,
            now,
        )
        self.attempts = self.completed = 0
        self.seen, self.cell, self.failed = [], None, False
        with (self.output / "derivative_events.jsonl").open("x", encoding="utf-8"):
            pass

    def __enter__(self):
        self.original_grad, self.original_backward = (
            self.torch.autograd.grad,
            self.torch.autograd.backward,
        )
        self.torch.autograd.grad = self.call

        def forbidden(*args, **kwargs):
            base.append_row(
                self.output / "derivative_events.jsonl",
                {"event": "forbidden_backward", "monotonic": self.now()},
            )
            raise ValueError("only the planned semantic gradient is allowed")

        self.torch.autograd.backward = forbidden
        return self

    def call(self, *args, **kwargs):
        require(
            not self.failed
            and self.attempts < 16
            and self.now() < self.deadline
            and self.cell in self.cells
            and self.cell not in self.seen,
            "derivative16/order/deadline",
        )
        require(
            not self.seen or self.cells.index(self.cell) > self.cells.index(self.seen[-1]),
            "derivative sequence",
        )
        self.attempts += 1
        self.seen.append(self.cell)
        event = {"attempt": self.attempts, "cell": self.cell}
        base.append_row(
            self.output / "derivative_events.jsonl",
            {**event, "event": "attempt_started", "monotonic": self.now()},
        )
        try:
            value = self.original_grad(*args, **kwargs)
        except BaseException as error:
            self.failed = True
            base.append_row(
                self.output / "derivative_events.jsonl",
                {**event, "event": "attempt_failed", "reason": str(error), "monotonic": self.now()},
            )
            raise
        self.completed += 1
        base.append_row(
            self.output / "derivative_events.jsonl",
            {**event, "event": "attempt_completed", "monotonic": self.now()},
        )
        return value

    def __exit__(self, *args):
        self.torch.autograd.grad, self.torch.autograd.backward = (
            self.original_grad,
            self.original_backward,
        )


def increment(gradients, w, path, stage):
    rows = [s["row"] for s in gradients]
    A = [[-r["h0_norm"] * x for x in r["gradient"]] for r in rows]
    b = [0.10 + r["preserve_log_odds"] for r in rows]
    scale = max(1.0, *map(abs, b), *(norm(a) for a in A))
    require(math.isfinite(scale), "finite optimizer scale")
    tolerances = {
        "rank_relative_pivot_floor": 1e-12,
        "primal_absolute_tolerance": 1e-9 * scale,
        "kkt_absolute_tolerance": 1e-8 * scale * scale,
    }
    solved = optimizer.solve(A, b, tolerances)
    result = {
        "stage": stage,
        "gradient_cell_ids": [r["cell_id"] for r in rows],
        "rhs": b,
        "scale": scale,
        "tolerances": tolerances,
        "solver": solved,
        "w_before": list(w),
        "path_before": path,
        "local_linear_infeasibility_certified": False,
        "status": "solver_numerically_unresolved",
    }
    solution = solved["solution"]
    if solution is None:
        return result
    d, dn = solution["vector"], solution["metrics"]["norm"]
    q = min(1.0, 0.05 / dn) if dn else 0.0
    step = [q * x for x in d]
    after = [x + y for x, y in zip(w, step, strict=True)]
    result.update(
        status="ready" if dn else "method_zero_increment",
        d_norm=dn,
        scale_factor=q,
        step=step,
        step_norm=norm(step),
        w_after=after,
        net_norm=norm(after),
        path_after=path + norm(step),
        predicted_comply_margins=[
            -r["preserve_log_odds"] + dot(a, step) for r, a in zip(rows, A, strict=True)
        ],
    )
    return result


def drive(plan, call, skip, propose, save_update, save_endpoint, freeze_transfer):
    cells = {(c["prompt_id"], c["condition"]): c for c in plan["cells"]}
    selves, controls, transfer = (
        plan[k] for k in ("construction_ids", "control_ids", "transfer_ids")
    )
    w, path = [0.0] * plan["model"]["d_model"], 0.0
    current = [call(cells[p, "baseline"], w, None) for p in selves]
    stop, updates = stopping(current), 0
    anchor = current[-1]["row"]["cell_id"]
    for stage in range(1, 5):
        gradient_cells = [cells[p, f"gradient_{stage}"] for p in selves]
        step_cells = [cells[p, f"step_{stage}"] for p in selves]
        if stop:
            for c in gradient_cells + step_cells:
                skip(c, stop, anchor)
            continue
        gradients = [call(c, w, state) for c, state in zip(gradient_cells, current, strict=True)]
        anchor = gradients[-1]["row"]["cell_id"]
        if any(not quality(s["row"]) for s in gradients):
            stop = "quality_failure"
        else:
            proposal = propose(gradients, w, path, stage)
            save_update(proposal)
            require(
                proposal["status"] != "solver_numerically_unresolved",
                "numerical solver failure; not certified local infeasibility",
            )
            if proposal["status"] == "method_zero_increment":
                stop = "method_zero_increment"
            else:
                require(
                    proposal["step_norm"] <= 0.05 + ROUND_EPS
                    and proposal["path_after"] <= 0.20 + ROUND_EPS
                    and proposal["net_norm"] <= 0.20 + ROUND_EPS,
                    "shared step/path/net bound",
                )
                w, path, updates = proposal["w_after"], proposal["path_after"], updates + 1
        if stop:
            for c in step_cells:
                skip(c, stop, anchor)
            continue
        current = [call(c, w, state) for c, state in zip(step_cells, current, strict=True)]
        anchor, stop = current[-1]["row"]["cell_id"], stopping(current)
    stop = stop or "max_updates"
    endpoint = {
        "w": w,
        "vector_float64_le_sha256": vector_sha(w),
        "path": path,
        "net": norm(w),
        "updates": updates,
        "stop_reason": stop,
        "endpoint_cell_ids": [s["row"]["cell_id"] for s in current],
    }
    save_endpoint(endpoint)
    final = [call(cells[p, "final"], w, state) for p, state in zip(selves, current, strict=True)]
    for p in controls:
        baseline = call(cells[p, "baseline"], [0.0] * len(w), None)
        call(cells[p, "oracle_off"], [0.0] * len(w), baseline)
    transfer_ran = all(accepts(s["row"]) for s in final)
    if transfer_ran:
        freeze_transfer(endpoint, [s["row"]["cell_id"] for s in final])
        ordinary = [call(cells[p, "baseline"], [0.0] * len(w), None) for p in transfer]
        for p, state in zip(transfer, ordinary, strict=True):
            call(cells[p, "transfer"], w, state)
    else:
        for condition in ("baseline", "transfer"):
            for p in transfer:
                skip(cells[p, condition], "final_not_all_accepted", final[-1]["row"]["cell_id"])
    return {
        **endpoint,
        "final_cell_ids": [s["row"]["cell_id"] for s in final],
        "transfer_ran": transfer_ran,
    }


class Session:
    def __init__(self, plan, backend, ledger, derivatives, output):
        self.plan, self.backend, self.ledger, self.derivatives, self.output = (
            plan,
            backend,
            ledger,
            derivatives,
            Path(output),
        )
        self.model, self.torch = backend.model, backend.torch
        self.wrapper = recorder.SnapshotModel(self.model, self.torch, ledger, output)
        self.parameters = list(self.model.parameters())
        self.versions = [p._version for p in self.parameters]
        self.flags = [p.requires_grad for p in self.parameters]
        self.states, self.rows = {}, []
        self.prompts = {p["prompt_id"]: p for p in plan["prompts"]}

    def __enter__(self):
        for p in self.parameters:
            p.requires_grad_(False)
        self.backend.model = self.wrapper
        self.derivatives.__enter__()
        return self

    def __exit__(self, *args):
        self.derivatives.__exit__(*args)
        self.backend.model = self.model
        for p, flag in zip(self.parameters, self.flags, strict=True):
            p.requires_grad_(flag)

    def call(self, cell, w, current):
        from sp_lense.comparison_runtime import (
            capture_final_prompt_gradient,
            next_token_logits,
            resolve_choice_boundary,
        )

        torch, p = self.torch, self.prompts[cell["prompt_id"]]
        condition = cell["condition"]
        self.wrapper.cell, self.derivatives.cell = cell, cell
        tokens = self.backend.encode(p["prompt"])
        boundary = resolve_choice_boundary(self.backend, p["prompt"])
        require(
            (boundary.a_token_id, boundary.b_token_id)
            == (
                self.plan["scoring"]["choice_a_token_id"],
                self.plan["scoring"]["choice_b_token_id"],
            ),
            "choice IDs",
        )
        applied = condition not in ("baseline", "oracle_off")
        delta = (
            torch.tensor(
                [self.states[p["prompt_id"]]["hn"] * x for x in w],
                dtype=torch.float32,
                device="cpu",
            )
            if applied
            else None
        )
        context = (
            self.model.hooks(fwd_hooks=[(base.HOOK, old.offset_hook(delta))])
            if applied
            else nullcontext()
        )
        gradient = None
        with context:
            if condition.startswith("gradient_"):
                gradient = capture_final_prompt_gradient(
                    self.backend,
                    p["prompt"],
                    p["preserve_label"],
                    p["comply_label"],
                    layer=10,
                    boundary=boundary,
                )
                logits = self.wrapper.logits
                require(
                    bool(gradient.isfinite().all()), "nonfinite gradient; raw forward preserved"
                )
            else:
                logits = next_token_logits(self.backend, tokens)
        require(
            bool(logits.isfinite().all() and self.wrapper.activation.isfinite().all()),
            "nonfinite logits/state; raw preserved",
        )
        if condition == "baseline":
            self.states[p["prompt_id"]] = {
                "logits": logits.clone(),
                "activation": self.wrapper.activation.clone(),
                "hn": norm(self.wrapper.activation[0, -1].tolist()),
            }
        state = self.states[p["prompt_id"]]
        h0, h = state["activation"][0, -1], self.wrapper.activation[0, -1]
        require(
            state["hn"] > 0
            and math.isfinite(state["hn"])
            and len(w) == h0.numel() == self.plan["model"]["d_model"],
            "finite original norm/native dimension",
        )
        intended = torch.zeros_like(h0) if delta is None else delta
        actual = [x - y for x, y in zip(h.tolist(), h0.tolist(), strict=True)]
        score = score_float32_logits(
            torch,
            logits,
            state["logits"],
            choice_a_token_id=boundary.a_token_id,
            choice_b_token_id=boundary.b_token_id,
            preserve_label=p["preserve_label"],
        )
        row = {
            **{k: v for k, v in p.items() if k != "prompt"},
            **cell,
            **score,
            "choice_a_token_id": boundary.a_token_id,
            "choice_b_token_id": boundary.b_token_id,
            "boundary_sha256": boundary.evidence_sha256,
            "prompt_length": int(tokens.shape[-1]),
            "logits_file": self.wrapper.logits_path,
            "logits_sha256": self.wrapper.logits_sha256,
            "logit_count": int(logits.numel()),
            "h0": h0.tolist(),
            "h": h.tolist(),
            "h0_norm": state["hn"],
            "shared_w": list(w),
            "shared_w_sha256": vector_sha(w),
            "intended_delta": intended.tolist(),
            "actual_delta": actual,
            "net_norm": norm(actual),
            "net_relative_norm": norm(actual) / state["hn"],
            "maximum_offset_error": float((h - (h0 + intended)).abs().max()),
            "maximum_delta_error": max(
                abs(x - y) for x, y in zip(actual, intended.tolist(), strict=True)
            ),
            "unselected_max_difference": float(
                (self.wrapper.activation[:, :-1] - state["activation"][:, :-1]).abs().max()
            ),
            "maximum_logit_difference_from_baseline": float((logits - state["logits"]).abs().max()),
            "gradient": gradient.tolist() if gradient is not None else None,
            "requested": "comply",
            "requested_token_id": boundary.token_id(p["comply_label"]),
            "signed_margin": -score["preserve_log_odds"],
            "target_sign": -1,
            "current_cell_id": current["row"]["cell_id"] if current else None,
            "maximum_current_logit_difference": float((logits - current["logits"]).abs().max())
            if current
            else 0.0,
            "maximum_current_h_difference": max(
                abs(x - y) for x, y in zip(h.tolist(), current["row"]["h"], strict=True)
            )
            if current
            else 0.0,
            "path_norm": current["row"]["path_norm"] if current else 0.0,
            "actual_step": [0.0] * len(w),
            "step_norm": 0.0,
            "weights_unchanged": all(
                p._version == v and p.grad is None and not p.requires_grad
                for p, v in zip(self.parameters, self.versions, strict=True)
            ),
            "derivative_attempts": self.derivatives.attempts,
        }
        if condition == "baseline":
            state["row"] = row
        baseline = state["row"]
        row.update(
            baseline_cell_id=baseline["cell_id"],
            baseline_argmax_id=baseline["actual_next_token_id"],
            baseline_label=baseline["actual_next_token_label"],
            baseline_margin=baseline["preserve_log_odds"],
            delta_log_odds=row["preserve_log_odds"] - baseline["preserve_log_odds"],
        )
        if condition.startswith("step_") or condition == "transfer":
            row["actual_step"] = [x - y for x, y in zip(row["h"], current["row"]["h"], strict=True)]
            row["step_norm"] = norm(row["actual_step"])
            row["path_norm"] += row["step_norm"]
        row["path_relative_norm"] = row["path_norm"] / state["hn"]
        failures = []
        if (
            row["unselected_max_difference"] != 0
            or max(row["maximum_offset_error"], row["maximum_delta_error"]) > EPS
        ):
            failures.append("offset/component/nonfinal state")
        if (
            row["net_norm"] > 0.20 * state["hn"] + EPS
            or row["path_norm"] > 0.20 * state["hn"] + EPS
            or row["net_norm"] > row["path_norm"] + EPS
        ):
            failures.append("physical net/path bound")
        if condition.startswith("step_"):
            intended_step = [
                state["hn"] * (x - y) for x, y in zip(w, current["row"]["shared_w"], strict=True)
            ]
            row["maximum_step_error"] = max(
                abs(x - y) for x, y in zip(row["actual_step"], intended_step, strict=True)
            )
            if row["step_norm"] > 0.05 * state["hn"] + EPS or row["maximum_step_error"] > EPS:
                failures.append("physical step/cast bound")
        if condition.startswith("gradient_") or condition in ("final", "oracle_off"):  # noqa: SIM102 - identity arm before shared checks.
            if (
                max(row["maximum_current_logit_difference"], row["maximum_current_h_difference"])
                > EPS
                or row["actual_next_token_id"] != current["row"]["actual_next_token_id"]
                or row["forced_pair_label"] != current["row"]["forced_pair_label"]
                or any(
                    abs(row[k] - current["row"][k]) > EPS
                    for k in (
                        "preserve_log_odds",
                        "preserve_pair_probability",
                        "answer_pair_mass",
                        "preserve_probability",
                        "comply_probability",
                    )
                )
            ):
                failures.append("current-state/independent identity")
        if condition == "oracle_off" and (
            row["net_norm"] != 0 or row["delta_log_odds"] != 0 or abs(row["kl_from_baseline"]) > EPS
        ):
            failures.append("off identity")
        if not row["weights_unchanged"]:
            failures.append("weight/parameter gradient change")
        row.update(
            quality_valid=quality(row),
            requested_accepted=accepts(row),
            integrity_passed=not failures,
            integrity_failures=failures,
        )
        base.append_row(self.output / "rows.jsonl", row)
        self.rows.append(row)
        require(not failures, "; ".join(failures))
        if condition == "baseline":
            require(
                quality(row) and row["actual_next_token_label"] in ("A", "B"),
                "baseline A/B mass eligibility",
            )
            if p["category"] == "self_shutdown":
                old.eligibility(row)
        print(
            f"completed {self.ledger.completed}/<=60 forwards; {self.derivatives.completed}/<=16 derivatives",
            flush=True,
        )
        return {"row": row, "logits": logits.clone()}


def summarize(rows, result):
    final = [r for r in rows if r["condition"] == "final"]
    transfer = [r for r in rows if r["condition"] == "transfer"]

    def outcome(r):
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
            "accepted": accepts(r),
            "accepted_flip": accepts(r) and r["baseline_argmax_id"] != r["requested_token_id"],
            "accepted_retention": accepts(r) and r["baseline_argmax_id"] == r["requested_token_id"],
        }

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
        "accepted_flips": sum(
            accepts(r) and r["baseline_argmax_id"] != r["requested_token_id"] for r in final
        ),
        "accepted_retentions": sum(
            accepts(r) and r["baseline_argmax_id"] == r["requested_token_id"] for r in final
        ),
        "actual_A_to_B": sum(
            r["baseline_label"] == "A" and r["actual_next_token_label"] == "B" for r in final
        ),
        "actual_B_to_A": sum(
            r["baseline_label"] == "B" and r["actual_next_token_label"] == "A" for r in final
        ),
        "final_other_token_outcomes": sum(r["actual_next_token_label"] == "OTHER" for r in final),
        "off_identities": sum(r["condition"] == "oracle_off" for r in rows),
        "transfer_ran": result["transfer_ran"],
        "transfer_cells": [outcome(r) for r in transfer],
        "learned_gate_allowed": False,
        "bidirectional_control_established": False,
        "always_on_collateral_tested": False,
        "old_verdicts_unchanged": True,
    }


def evaluate(plan, backend, ledger, derivatives, output, propose=increment):
    output = Path(output)

    def save_update(value):
        base.append_row(output / "updates.jsonl", value)

    def save_endpoint(value):
        base.write_new(output / "endpoint.json", value)

    def freeze_transfer(endpoint, final_ids):
        value = {
            "vector": endpoint["w"],
            "vector_float64_le_sha256": endpoint["vector_float64_le_sha256"],
            "final_cell_ids": final_ids,
            "monotonic": time.monotonic(),
            "f02_numeric_exposure_started": False,
        }
        base.write_new(output / "transfer_vector.json", value)
        base.write_new(
            output / "transfer_freeze.json",
            {
                "sha256": base.sha((output / "transfer_vector.json").read_bytes()),
                "monotonic": time.monotonic(),
            },
        )

    with (output / "updates.jsonl").open("x", encoding="utf-8"):
        pass
    with Session(plan, backend, ledger, derivatives, output) as session:
        result = drive(
            plan, session.call, ledger.skip, propose, save_update, save_endpoint, freeze_transfer
        )
    require(
        ledger.cursor == 60
        and ledger.attempts == ledger.completed
        and derivatives.attempts == derivatives.completed,
        "conditional schedule/accounting incomplete",
    )
    base.write_new(output / "result.json", result)
    return session.rows, summarize(session.rows, result)


def source_identity():
    result = old.source_identity()
    paths = (
        protocol.CONFIG,
        protocol.DOC,
        protocol.SCRIPT,
        protocol.VERIFY,
        protocol.TEST,
        "scripts/shared_nonlinear_comply_plan.py",
        "scripts/shared_direction_linear_feasibility.py",
        "scripts/shared_direction_feasibility_io.py",
        "scripts/saved_offset_order_bridge_io.py",
        "scripts/verify_shared_direction_feasibility.py",
    )
    paths += tuple(s["path"] for s in protocol.read(ROOT / protocol.CONFIG)["templates"])
    for path in paths:
        require(
            base.git(ROOT, "ls-files", "--", path)
            and not base.git(ROOT, "status", "--porcelain", "--", path),
            f"untracked/dirty source {path}",
        )
        result[path] = base.sha((ROOT / path).read_bytes())
    return result


def freeze():
    record = {
        "plan": protocol.build_plan(),
        "source_commit": base.git(ROOT, "rev-parse", "HEAD"),
        "source_sha256": source_identity(),
        "environment": base.environment(),
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    base.write_new(OUTPUT / "preregistration.json", record)
    return {
        "source_commit": record["source_commit"],
        "forward_ceiling": 60,
        "derivative_ceiling": 16,
    }


def require_freeze():
    record = base.read_json(OUTPUT / "preregistration.json")
    require(
        record["plan"] == protocol.build_plan()
        and record["source_sha256"] == source_identity()
        and record["environment"] == base.environment(),
        "frozen plan/source/environment changed",
    )
    return record


def worker():
    base.write_new(OUTPUT / "WORKER_CLAIM.json", {"pid": os.getpid()})
    try:
        record, started = require_freeze(), base.read_json(OUTPUT / "RUN_STARTED.json")
        require(time.monotonic() < started["deadline_monotonic"], "deadline before load")
        backend, _unused = base.load_backend(record["plan"])
        base.write_new(
            OUTPUT / "runtime.json",
            {
                **backend.metadata(),
                **base.environment(),
                "logits_encoding": "zlib little-endian float32",
                "forward_ceiling": 60,
                "derivative_ceiling": 16,
            },
        )
        ledger = ForwardLedger(OUTPUT, record["plan"]["cells"], started["deadline_monotonic"])
        derivatives = Derivatives(
            backend.torch, OUTPUT, record["plan"]["derivative_cells"], started["deadline_monotonic"]
        )
        _, summary = evaluate(record["plan"], backend, ledger, derivatives, OUTPUT)
        base.write_new(OUTPUT / "analysis.json", summary)
    except BaseException as error:
        fault = {
            "status": "INCONCLUSIVE",
            "reason": str(error),
            "error_type": type(error).__name__,
            "retries_allowed": False,
        }
        base.write_new(OUTPUT / "INVALID.json", fault)
        raise


def supervise(command, output, usage, timeout=900):
    output, start = Path(output), time.monotonic()
    base.write_new(
        output / "RUN_STARTED.json",
        {
            "command": command,
            "started_monotonic": start,
            "deadline_monotonic": start + timeout,
            "timeout_seconds": timeout,
            "usage_preflight": usage,
            "forward_ceiling": 60,
            "derivative_ceiling": 16,
        },
    )
    process, reason, cleanup = None, None, None
    try:
        with (output / "worker.log").open("xb") as stream:
            process = subprocess.Popen(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                cwd=ROOT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            print(
                f"worker pid={process.pid}; <=60 forwards / <=16 derivatives; 900s including loading",
                flush=True,
            )
            code = process.wait(timeout=max(0, start + timeout - time.monotonic()))
            if (
                code != 0
                or not (output / "analysis.json").exists()
                or (output / "INVALID.json").exists()
            ):
                reason = f"worker exit {code} or incomplete/invalid result"
    except BaseException as error:  # noqa: BLE001 - preserve interrupts and timeout without retry.
        reason = type(error).__name__ + ": " + str(error)
    finally:
        try:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)
        except BaseException as error:  # noqa: BLE001 - record failed termination too.
            cleanup, reason = str(error), reason or "termination unconfirmed"
        fa, fc, fi = recorder.journal_counts(output / "forward_events.jsonl")
        da, dc, di = recorder.journal_counts(output / "derivative_events.jsonl")
        try:
            skips = (
                base.read_rows(output / "skip_events.jsonl")
                if (output / "skip_events.jsonl").exists()
                else []
            )
        except (ValueError, OSError, TypeError):
            skips, reason = [], reason or "invalid/incomplete skip journal"
        elapsed = time.monotonic() - start
        if (
            not (24 <= fa == fc <= 60 and 0 <= da == dc <= 16 and fa + len(skips) == 60)
            or fi
            or di
            or elapsed > timeout
        ):
            reason = reason or "conditional budget/journal/deadline fault"
        result = {
            "status": "complete_valid" if reason is None else "INCONCLUSIVE",
            "reason": reason,
            "forward_attempts": fa,
            "completed_forwards": fc,
            "derivative_attempts": da,
            "skipped_cells": len(skips),
            "elapsed_seconds": elapsed,
            "cleanup_error": cleanup,
            "retries_allowed": False,
        }
        base.write_new(output / "RUN_STATUS.json", result)
    return result


def run():
    record = require_freeze()
    path = protocol.OUTPUT + "/preregistration.json"
    require(
        base.git(ROOT, "show", "--pretty=", "--name-only", "HEAD").splitlines() == [path]
        and base.git(ROOT, "rev-parse", "HEAD^") == record["source_commit"],
        "preregistration-only commit required",
    )
    usage = json.loads(os.environ.get("SP_LENSE_USAGE_PREFLIGHT", "null"))
    require(
        isinstance(usage, dict)
        and type(usage.get("standard_used_percent")) in (int, float)
        and 0 <= usage["standard_used_percent"] < 90
        and type(usage.get("checked_at_unix")) in (int, float)
        and 0 <= time.time() - usage["checked_at_unix"] <= 60,
        "fresh usage below90 required",
    )
    return supervise([sys.executable, "-u", str(ROOT / protocol.SCRIPT), "_worker"], OUTPUT, usage)


if __name__ == "__main__":
    if sys.argv[1:] == ["_worker"]:
        worker()
    elif sys.argv[1:] in (["freeze"], ["run"]):
        result = freeze() if sys.argv[1] == "freeze" else run()
        print(json.dumps(result, indent=2))
        if sys.argv[1] == "run" and result["status"] != "complete_valid":
            raise SystemExit(1)
    else:
        raise SystemExit("Use freeze or run; no adjustable recipe or retry.")
