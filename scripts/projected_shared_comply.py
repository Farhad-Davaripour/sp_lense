"""Fresh projected common-COMPLY successor; immutable backend/optimizer reuse."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import projected_shared_comply_plan as protocol
from scripts import shared_nonlinear_comply as previous
from sp_lense.future_choice_scoring import score_float32_logits

old, base, recorder = previous.old, previous.base, previous.recorder
require, norm, dot = protocol.require, previous.norm, previous.dot
vector_sha, quality, accepts, stopping = (
    previous.vector_sha,
    previous.quality,
    previous.accepts,
    previous.stopping,
)
EPS, ROUND_EPS = previous.EPS, previous.ROUND_EPS
OUTPUT = ROOT / protocol.OUTPUT


class ForwardLedger(previous.ForwardLedger):
    def begin(self, cell):
        require(
            not self.failed and not self.pending and self.now() < self.deadline,
            "forward failure/pending/deadline",
        )
        require(
            self.attempts < 92
            and self.cursor < len(self.cells)
            and cell == self.cells[self.cursor],
            "forward92/order",
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
            in (
                "accepted",
                "quality_failure",
                "method_zero_increment",
                "projection_stall",
                "final_not_all_accepted",
            ),
            "skip reason",
        )
        event = {"cell": cell, "reason": reason, "after_cell_id": anchor, "monotonic": self.now()}
        base.append_row(self.output_dir / "skip_events.jsonl", event)
        self.skips.append(event)
        self.cursor += 1


class Derivatives(previous.Derivatives):
    def call(self, *args, **kwargs):
        require(
            not self.failed
            and self.attempts < 32
            and self.now() < self.deadline
            and self.cell in self.cells
            and self.cell not in self.seen,
            "derivative32/order/deadline",
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


def project(w, d):
    """One Euclidean projection only; exact-zero stall, no near-stall threshold."""
    dn = norm(d)
    q = min(1.0, 0.05 / dn) if dn else 0.0
    s = [q * x for x in d]
    u = [x + y for x, y in zip(w, s, strict=True)]
    un = norm(u)
    factor = 1.0 if un <= 0.20 else 0.20 / un
    after = list(u) if factor == 1.0 else [factor * x for x in u]
    r = [x - y for x, y in zip(after, w, strict=True)]
    return {
        "d": list(d),
        "s": s,
        "u": u,
        "w_next": after,
        "r": r,
        "d_norm": dn,
        "scale_factor": q,
        "proposed_step_norm": norm(s),
        "unprojected_net_norm": un,
        "projection_factor": factor,
        "projection_distance": norm([x - y for x, y in zip(u, after, strict=True)]),
        "step": r,
        "step_norm": norm(r),
        "w_after": after,
        "net_norm": norm(after),
        "status": "ready"
        if norm(r)
        else "method_zero_increment"
        if dn == 0
        else "projection_stall",
    }


def increment(gradients, w, path, stage):
    # Preserve the solver inputs, tolerances, all active-set candidates and KKT.
    result = previous.increment(gradients, w, path, stage)
    if result["solver"]["solution"] is None:
        return result
    update = project(w, result["solver"]["solution"]["vector"])
    A = [[-x["row"]["h0_norm"] * g for g in x["row"]["gradient"]] for x in gradients]
    proposed = [dot(a, update["s"]) for a in A]
    actual = [dot(a, update["r"]) for a in A]
    c = [-x["row"]["preserve_log_odds"] for x in gradients]
    result.update(update)
    result.update(
        path_after=path + update["step_norm"],
        proposed_path_after=path + update["proposed_step_norm"],
        proposed_constraint_changes=proposed,
        actual_constraint_changes=actual,
        proposed_constraint_residuals=[x - b for x, b in zip(proposed, result["rhs"], strict=True)],
        actual_constraint_residuals=[x - b for x, b in zip(actual, result["rhs"], strict=True)],
        proposed_comply_margins=[x + y for x, y in zip(c, proposed, strict=True)],
        actual_comply_margins=[x + y for x, y in zip(c, actual, strict=True)],
        predicted_comply_margins=[x + y for x, y in zip(c, actual, strict=True)],
        signed_projection_loss=[x - y for x, y in zip(proposed, actual, strict=True)],
    )
    return result


def drive(plan, call, skip, propose, save_update, save_endpoint, freeze_transfer):
    cells = {(c["prompt_id"], c["condition"]): c for c in plan["cells"]}
    selves, controls, transfer = (
        plan[k] for k in ("construction_ids", "control_ids", "transfer_ids")
    )
    w, path = [0.0] * plan["model"]["d_model"], 0.0
    current = [call(cells[p, "baseline"], w, None) for p in selves]
    stop, updates, attempted_updates = stopping(current), 0, 0
    anchor = current[-1]["row"]["cell_id"]
    for stage in range(1, 9):
        gradient_cells = [cells[p, f"gradient_{stage}"] for p in selves]
        step_cells = [cells[p, f"step_{stage}"] for p in selves]
        if stop:
            for c in gradient_cells + step_cells:
                skip(c, stop, anchor)
            continue
        attempted_updates += 1
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
            if proposal["status"] in ("method_zero_increment", "projection_stall"):
                stop = proposal["status"]
            else:
                require(
                    proposal["step_norm"] <= 0.05 + ROUND_EPS
                    and proposal["path_after"] <= 0.40 + ROUND_EPS
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
        "attempted_updates": attempted_updates,
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


class Session(previous.Session):
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
            or row["path_norm"] > 0.40 * state["hn"] + EPS
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
            f"completed {self.ledger.completed}/<=92 forwards; {self.derivatives.completed}/<=32 derivatives",
            flush=True,
        )
        return {"row": row, "logits": logits.clone()}


def summarize(rows, result):
    return {**previous.summarize(rows, result), "attempted_updates": result["attempted_updates"]}


def source_identity():
    result = previous.source_identity()
    for path in (
        protocol.CONFIG,
        protocol.DOC,
        protocol.SCRIPT,
        protocol.VERIFY,
        protocol.TEST,
        "scripts/projected_shared_comply_plan.py",
    ):
        require(
            base.git(ROOT, "ls-files", "--", path)
            and not base.git(ROOT, "status", "--porcelain", "--", path),
            f"untracked/dirty source {path}",
        )
        result[path] = base.sha((ROOT / path).read_bytes())
    return result


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
        ledger.cursor == 92
        and ledger.attempts == ledger.completed
        and derivatives.attempts == derivatives.completed,
        "conditional schedule/accounting incomplete",
    )
    base.write_new(output / "result.json", result)
    return session.rows, summarize(session.rows, result)


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
        "forward_ceiling": 92,
        "derivative_ceiling": 32,
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
                "forward_ceiling": 92,
                "derivative_ceiling": 32,
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
            "forward_ceiling": 92,
            "derivative_ceiling": 32,
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
                f"worker pid={process.pid}; <=92 forwards / <=32 derivatives; 900s including loading",
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
            not (24 <= fa == fc <= 92 and 0 <= da == dc <= 32 and fa + len(skips) == 92)
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
