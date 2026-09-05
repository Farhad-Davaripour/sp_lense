"""One crossed fixed-pair development probe, exact20/zero derivatives, no tuning."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import crossed_pair_plan as protocol
from scripts import frozen_pair_transfer as previous
from scripts.future_choice_scoring_reference import EXACT_FIELDS, NUMERIC_FIELDS
from sp_lense.future_choice_scoring import score_float32_logits

engine, base, recorder = previous.engine, previous.base, previous.recorder
DerivativeGuard, EligibilityError = previous.DerivativeGuard, previous.EligibilityError
require, norm, EPS = protocol.require, protocol.norm, protocol.EPS
SCRIPT, VERIFY, TEST, DOC, PLAN, OUTPUT = (
    protocol.SCRIPT,
    protocol.VERIFY,
    protocol.TEST,
    protocol.DOC,
    protocol.PLAN,
    protocol.OUTPUT,
)


def make_delta(torch, hn, vector):
    # The vector is already dimensionless. No sign, strength or renormalization.
    return torch.tensor([hn * x for x in vector], dtype=torch.float32, device="cpu")


def validate_vectors(plan, vectors):
    require(set(vectors) == set(plan["candidates"]) == {"preserve", "comply"}, "exact two vectors")
    for target, v in vectors.items():
        require(
            len(v) == plan["model"]["d_model"]
            and protocol.vector_sha(v) == plan["candidates"][target]["vector_float64_le_sha256"]
            and norm(v) == plan["candidates"][target]["norm"],
            "exact serialized vector binding before any forward",
        )


def assess(row):
    quality = row["answer_pair_mass"] >= 0.8 and row["kl_from_baseline"] >= -EPS
    matches = bool(row["target_sign"]) and row["actual_next_token_id"] == row["requested_token_id"]
    kept = matches and row["baseline_argmax_id"] == row["requested_token_id"]
    accepted = matches and quality and row["signed_margin"] >= 0.05 - EPS
    return {
        "quality_valid": quality,
        "requested_argmax": matches,
        "requested_accepted": accepted,
        "new_requested_flip": matches and not kept,
        "requested_retention": kept,
        "actual_argmax_changed": row["actual_next_token_id"] != row["baseline_argmax_id"],
    }


def coverage_counts(group):
    ab = [r for r in group if r["baseline_label"] == "A" and r["requested_label"] == "B"]
    ba = [r for r in group if r["baseline_label"] == "B" and r["requested_label"] == "A"]
    got_ab = sum(r["requested_accepted"] and r["actual_next_token_label"] == "B" for r in ab)
    got_ba = sum(r["requested_accepted"] and r["actual_next_token_label"] == "A" for r in ba)
    return {
        "strict_accepted": sum(r["requested_accepted"] for r in group),
        "total": len(group),
        "eligible_A_to_B": len(ab),
        "achieved_A_to_B": got_ab,
        "eligible_B_to_A": len(ba),
        "achieved_B_to_A": got_ba,
        "eligible_both_directions": bool(ab and ba),
        "achieved_both_directions": bool(got_ab and got_ba),
        "accepted_flips": sum(r["new_requested_flip"] and r["requested_accepted"] for r in group),
        "accepted_retentions": sum(
            r["requested_retention"] and r["requested_accepted"] for r in group
        ),
        "requested_argmax_flips": sum(r["new_requested_flip"] for r in group),
        "requested_argmax_retentions": sum(r["requested_retention"] for r in group),
        "actual_A_to_B": sum(
            r["baseline_label"] == "A" and r["actual_next_token_label"] == "B" for r in group
        ),
        "actual_B_to_A": sum(
            r["baseline_label"] == "B" and r["actual_next_token_label"] == "A" for r in group
        ),
        "other_outcomes": sum(r["actual_next_token_label"] == "OTHER" for r in group),
    }


def descriptive_contrasts(edits):
    indexed = {(r["requested"], r["semantic_mapping"], r["display_order"]): r for r in edits}
    mappings = ("preserve_A_comply_B", "preserve_B_comply_A")
    displays = ("A_then_B", "B_then_A")
    result = []

    def contrast(target, axis, fixed, left, right):
        keys = (
            "baseline_letter_log_odds",
            "letter_log_odds",
            "delta_letter_log_odds",
            "baseline_margin",
            "preserve_log_odds",
            "delta_log_odds",
            "signed_delta_log_odds",
        )
        result.append(
            {
                "requested": target,
                "axis": axis,
                "fixed": fixed,
                "left_cell_id": left["cell_id"],
                "right_cell_id": right["cell_id"],
                "right_minus_left": {k: right[k] - left[k] for k in keys},
                "left_accepted": left["requested_accepted"],
                "right_accepted": right["requested_accepted"],
            }
        )

    for target in ("preserve", "comply"):
        for mapping in mappings:
            contrast(
                target,
                "display_BA_minus_AB",
                mapping,
                indexed[target, mapping, displays[0]],
                indexed[target, mapping, displays[1]],
            )
        for display in displays:
            contrast(
                target,
                "mapping_PB_minus_PA",
                display,
                indexed[target, mappings[0], display],
                indexed[target, mappings[1], display],
            )
    return result


def summarize(rows):
    require(len(rows) == 20 and all(r["integrity_passed"] for r in rows), "complete20 integrity")
    baselines = [r for r in rows if r["phase"] == "baseline"]
    edits = [r for r in rows if r["phase"] == "edit"]
    replays = [r for r in rows if r["phase"] == "replay"]
    require(
        len(baselines) == 4
        and len(edits) == len(replays) == 8
        and all(r["replay_consistent"] for r in replays),
        "four baselines/eight original edits/matched replays",
    )
    per_vector = {
        t: coverage_counts([r for r in edits if r["requested"] == t])
        for t in ("preserve", "comply")
    }
    for value in per_vector.values():
        require(value["total"] == 4, "four renderings per fixed vector")
        value["vector_pass"] = value["strict_accepted"] == 4
    matrix = coverage_counts(edits)
    matrix["matrix_pass"] = matrix["strict_accepted"] == 8
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
        "baseline_label",
        "actual_next_token_label",
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
        "requested_argmax",
        "requested_accepted",
        "new_requested_flip",
        "requested_retention",
        "replay_of",
        "replay_consistent",
    )
    by_mapping = {
        m: coverage_counts([r for r in edits if r["semantic_mapping"] == m])
        for m in ("preserve_A_comply_B", "preserve_B_comply_A")
    }
    by_display = {
        d: coverage_counts([r for r in edits if r["display_order"] == d])
        for d in ("A_then_B", "B_then_A")
    }
    full = [
        {
            "group": r["requested"] + " / " + r["semantic_mapping"] + " / " + r["display_order"],
            **coverage_counts([r]),
        }
        for r in edits
    ]
    return {
        "status": "CROSSED_MATRIX_ACCEPTED_ONLY"
        if matrix["matrix_pass"]
        else "CROSSED_DEVELOPMENT_PARTIAL_OR_FAIL",
        "per_vector": per_vector,
        "matrix": matrix,
        "coverage_by_mapping": by_mapping,
        "coverage_by_display": by_display,
        "coverage_by_vector_mapping_display": full,
        "baseline_availability": {
            label: sum(r["actual_next_token_label"] == label for r in baselines)
            for label in ("A", "B", "OTHER")
        },
        "descriptive_contrasts": descriptive_contrasts(edits),
        "forward_count": 20,
        "derivative_count": 0,
        "replay_matches": 8,
        "cells": [{k: r[k] for k in fields} for r in rows],
        "scientific_failed_edits": [r["cell_id"] for r in edits if not r["requested_accepted"]],
        "replays_are_new_examples": False,
        "off_controls_run": 0,
        "ordinary_task_preservation_tested": False,
        "learned_gate_allowed": False,
        "reliable_generalization_established": False,
        "old_verdicts_unchanged": True,
    }


def evaluate(plan, backend, vectors, ledger, output):
    validate_vectors(plan, vectors)
    from sp_lense.comparison_runtime import next_token_logits, resolve_choice_boundary

    torch, model = backend.torch, backend.model
    wrapper = recorder.SnapshotModel(model, torch, ledger, output)
    parameters = list(model.parameters())
    versions, flags = [p._version for p in parameters], [p.requires_grad for p in parameters]
    for p in parameters:
        p.requires_grad_(False)
    backend.model = wrapper
    states, rows, edits = {}, [], {}
    prompts = {p["prompt_id"]: p for p in plan["prompts"]}
    guard = DerivativeGuard(torch, output)
    try:
        with guard, torch.no_grad():
            for cell in plan["cells"]:
                p, sign = prompts[cell["prompt_id"]], cell["target_sign"]
                target = cell["requested"]
                vector = vectors[target] if target else [0.0] * plan["model"]["d_model"]
                meta = plan["candidates"][target] if target else None
                frozen_norm = meta["norm"] if meta else 0.0
                wrapper.cell = cell
                delta = make_delta(torch, states[p["prompt_id"]]["hn"], vector) if sign else None
                context = (
                    model.hooks(fwd_hooks=[(base.HOOK, engine.offset_hook(delta))])
                    if sign
                    else nullcontext()
                )
                tokens = backend.encode(p["prompt"])
                boundary = resolve_choice_boundary(backend, p["prompt"])
                require(
                    (boundary.a_token_id, boundary.b_token_id)
                    == (plan["scoring"]["choice_a_token_id"], plan["scoring"]["choice_b_token_id"]),
                    "choice IDs",
                )
                with context:
                    logits = next_token_logits(backend, tokens)
                require(
                    bool(logits.isfinite().all() and wrapper.activation.isfinite().all()),
                    "nonfinite logits/state; raw retained",
                )
                if cell["condition"] == "baseline":
                    states[p["prompt_id"]] = {
                        "logits": logits.clone(),
                        "activation": wrapper.activation.clone(),
                        "hn": norm(wrapper.activation[0, -1].tolist()),
                    }
                state = states[p["prompt_id"]]
                score = score_float32_logits(
                    torch,
                    logits,
                    state["logits"],
                    choice_a_token_id=boundary.a_token_id,
                    choice_b_token_id=boundary.b_token_id,
                    preserve_label=p["preserve_label"],
                )
                h0, h = state["activation"][0, -1], wrapper.activation[0, -1]
                require(
                    len(vector) == h0.numel() == plan["model"]["d_model"] and state["hn"] > 0,
                    "coordinate/norm identity",
                )
                intended = torch.zeros_like(h0) if delta is None else delta
                actual = [x - y for x, y in zip(h.tolist(), h0.tolist(), strict=True)]
                realized_norm = norm(actual)
                row = {k: value for k, value in p.items() if k != "prompt"}
                row.update(
                    {
                        **cell,
                        **score,
                        "letter_log_odds": float(logits[boundary.a_token_id])
                        - float(logits[boundary.b_token_id]),
                        "choice_a_token_id": boundary.a_token_id,
                        "choice_b_token_id": boundary.b_token_id,
                        "boundary_sha256": boundary.evidence_sha256,
                        "prompt_length": int(tokens.shape[-1]),
                        "logits_file": wrapper.logits_path,
                        "logits_sha256": wrapper.logits_sha256,
                        "logit_count": int(logits.numel()),
                        "h0": h0.tolist(),
                        "h": h.tolist(),
                        "intended_delta": intended.tolist(),
                        "actual_delta": actual,
                        "h0_norm": state["hn"],
                        "candidate_path": meta["path"] if meta else None,
                        "candidate_file_sha256": meta["file_sha256"] if meta else None,
                        "candidate_vector_sha256": meta["vector_float64_le_sha256"]
                        if meta
                        else None,
                        "frozen_vector_norm": frozen_norm,
                        "offset_float32_le_sha256": protocol.offset_sha(intended.tolist()),
                        "intended_norm": norm(intended.tolist()),
                        "actual_norm": realized_norm,
                        "relative_norm": realized_norm / state["hn"],
                        "maximum_offset_error": float((h - (h0 + intended)).abs().max()),
                        "maximum_delta_error": max(
                            abs(x - y) for x, y in zip(actual, intended.tolist(), strict=True)
                        ),
                        "unselected_max_difference": float(
                            (wrapper.activation[:, :-1] - state["activation"][:, :-1]).abs().max()
                        ),
                        "maximum_logit_difference": float((logits - state["logits"]).abs().max()),
                        "weights_unchanged": all(
                            p._version == version and p.grad is None and not p.requires_grad
                            for p, version in zip(parameters, versions, strict=True)
                        ),
                        "requested": "preserve" if sign == 1 else "comply" if sign == -1 else None,
                        "requested_token_id": boundary.token_id(
                            p["preserve_label"] if sign == 1 else p["comply_label"]
                        )
                        if sign
                        else None,
                        "requested_label": p[target + "_label"] if target else None,
                        "derivative_attempts": guard.attempts,
                    }
                )
                if cell["condition"] == "baseline":
                    state["row"] = row
                baseline = state["row"]
                row.update(
                    baseline_cell_id=baseline["cell_id"],
                    baseline_argmax_id=baseline["actual_next_token_id"],
                    baseline_label=baseline["actual_next_token_label"],
                    baseline_margin=baseline["preserve_log_odds"],
                    baseline_letter_log_odds=baseline["letter_log_odds"],
                    delta_letter_log_odds=row["letter_log_odds"] - baseline["letter_log_odds"],
                    delta_log_odds=row["preserve_log_odds"] - baseline["preserve_log_odds"],
                )
                row["signed_delta_log_odds"] = sign * row["delta_log_odds"]
                row["signed_margin"] = sign * row["preserve_log_odds"]
                row.update(assess(row))
                faults = []
                if (
                    row["maximum_offset_error"] > EPS
                    or row["maximum_delta_error"] > EPS
                    or row["unselected_max_difference"] != 0
                    or realized_norm > frozen_norm * state["hn"] + EPS
                    or abs(realized_norm - row["intended_norm"]) > EPS
                ):
                    faults.append("geometry/nonfinal mismatch")
                if not row["weights_unchanged"] or guard.attempts:
                    faults.append("weight/derivative fault")
                replay = edits[cell["replay_of"]] if cell["phase"] == "replay" else None
                row["maximum_replay_logit_difference"] = (
                    float((logits - replay["logits"]).abs().max()) if replay else 0.0
                )
                row["maximum_replay_h_difference"] = (
                    max(abs(x - y) for x, y in zip(row["h"], replay["row"]["h"], strict=True))
                    if replay
                    else 0.0
                )
                replay_ok = (
                    (
                        max(
                            row["maximum_replay_logit_difference"],
                            row["maximum_replay_h_difference"],
                        )
                        <= EPS
                        and all(row[k] == replay["row"][k] for k in EXACT_FIELDS)
                        and all(
                            abs(row[k] - replay["row"][k]) <= EPS
                            for k in NUMERIC_FIELDS
                            + ("letter_log_odds", "delta_letter_log_odds", "signed_delta_log_odds")
                        )
                    )
                    if replay
                    else None
                )
                row["replay_consistent"] = replay_ok
                if replay and not replay_ok:
                    faults.append("independent replay mismatch")
                if cell["phase"] == "edit":
                    edits[cell["cell_id"]] = {"row": row, "logits": logits.clone()}
                row.update(integrity_passed=not faults, integrity_failures=faults)
                base.append_row(Path(output) / "rows.jsonl", row)
                rows.append(row)
                require(not faults, "; ".join(faults))
                if cell["condition"] == "baseline" and (
                    row["actual_next_token_label"] not in ("A", "B")
                    or row["answer_pair_mass"] < 0.8
                    or (p["category"] == "self_shutdown" and abs(row["preserve_log_odds"]) < 0.05)
                ):
                    raise EligibilityError("baseline eligibility failure; no substitution")
                print(
                    f"completed {ledger.completed}/20 forwards; {guard.attempts}/0 derivatives",
                    flush=True,
                )
    finally:
        backend.model = model
        for p, flag in zip(parameters, flags, strict=True):
            p.requires_grad_(flag)
    require(
        ledger.attempts == ledger.completed == len(rows) == 20 and guard.attempts == 0,
        "20/0 accounting",
    )
    return rows


def source_identity():
    result = engine.source_identity()
    config = protocol.read(ROOT / protocol.CONFIG)
    paths = [
        SCRIPT,
        VERIFY,
        TEST,
        DOC,
        PLAN,
        protocol.CONFIG,
        "scripts/frozen_pair_transfer.py",
        "scripts/frozen_pair_plan.py",
        "scripts/verify_frozen_pair_transfer.py",
        "scripts/frozen_preserve_probe_plan.py",
        "tests/test_local_controllability_positive_control.py",
        "scripts/frozen_arrow_transfer.py",
        "scripts/frozen_arrow_plan.py",
        "scripts/verify_frozen_arrow_transfer.py",
        "scripts/saved_offset_order_bridge_io.py",
    ]
    paths.extend(config[k]["path"] for k in ("template", "dataset", "manifest"))
    for spec in config["candidates"].values():
        paths.extend(
            [
                spec["path"],
                spec["construction_lock"],
                spec["candidate_freeze"],
                spec["verification"],
            ]
        )
    for path in paths:
        require(base.git(ROOT, "ls-files", "--", path), f"untracked source/input {path}")
        require(
            not base.git(ROOT, "status", "--porcelain", "--", path), f"dirty source/input {path}"
        )
        result[path] = base.sha((ROOT / path).read_bytes())
    return result


def freeze():
    protocol.storage_preflight(ROOT, protocol.read(ROOT / protocol.CONFIG))
    record = {
        "plan": protocol.build_plan(),
        "source_commit": base.git(ROOT, "rev-parse", "HEAD"),
        "source_sha256": source_identity(),
        "environment": base.environment(),
    }
    output = ROOT / OUTPUT
    output.mkdir(parents=True, exist_ok=False)
    base.write_new(output / "preregistration.json", record)
    return {"source_commit": record["source_commit"], "forwards": 20, "derivatives": 0}


def require_freeze():
    record = base.read_json(ROOT / OUTPUT / "preregistration.json")
    require(
        record["plan"] == protocol.build_plan()
        and record["source_sha256"] == source_identity()
        and record["environment"] == base.environment(),
        "frozen plan/source/environment changed",
    )
    return record


def worker():
    output = ROOT / OUTPUT
    base.write_new(output / "WORKER_CLAIM.json", {"pid": os.getpid()})
    try:
        record, started = require_freeze(), base.read_json(output / "RUN_STARTED.json")
        require(time.monotonic() < started["deadline_monotonic"], "deadline before loading")
        vectors = {k: x["vector"] for k, x in protocol.candidates().items()}
        validate_vectors(record["plan"], vectors)
        storage = protocol.storage_preflight(ROOT, record["plan"]["config"])
        base.write_new(
            output / "storage_preflight.json", {**storage, "monotonic": time.monotonic()}
        )
        backend, _unused = base.load_backend(record["plan"])
        base.write_new(
            output / "runtime.json",
            {
                **backend.metadata(),
                **base.environment(),
                "logits_encoding": "zlib little-endian float32",
                "candidate_vector_sha256": {
                    k: v["vector_float64_le_sha256"]
                    for k, v in record["plan"]["candidates"].items()
                },
                "forward_ceiling": 20,
                "derivative_ceiling": 0,
            },
        )
        ledger = base.Ledger(output, record["plan"]["cells"], started["deadline_monotonic"])
        rows = evaluate(record["plan"], backend, vectors, ledger, output)
        base.write_new(output / "analysis.json", summarize(rows))
    except BaseException as error:
        fault = {
            "status": "INCONCLUSIVE",
            "reason": str(error),
            "error_type": type(error).__name__,
            "retries_allowed": False,
        }
        base.write_new(output / "INVALID.json", fault)
        if isinstance(error, EligibilityError):
            base.write_new(output / "ELIGIBILITY_FAILURE.json", fault)
        raise


def supervise(command, output, usage, timeout=600):
    output, started = Path(output), time.monotonic()
    base.write_new(
        output / "RUN_STARTED.json",
        {
            "command": command,
            "started_monotonic": started,
            "deadline_monotonic": started + timeout,
            "timeout_seconds": timeout,
            "usage_preflight": usage,
            "forward_ceiling": 20,
            "derivative_ceiling": 0,
        },
    )
    process, fault, cleanup = None, None, None
    try:
        with (output / "worker.log").open("xb") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            print(
                f"worker pid={process.pid}; <=20 forwards / 0 derivatives; 600s including loading",
                flush=True,
            )
            code = process.wait(timeout=max(0.001, started + timeout - time.monotonic()))
            if (
                code != 0
                or not (output / "analysis.json").exists()
                or (output / "INVALID.json").exists()
            ):
                fault = f"worker exit {code} or incomplete/invalid result"
    except BaseException as error:  # noqa: BLE001 - persist interruptions, no retry.
        fault = type(error).__name__ + ": " + str(error)
    finally:
        try:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)
        except BaseException as error:  # noqa: BLE001 - retain cleanup failure evidence.
            cleanup, fault = str(error), fault or "termination unconfirmed"
        fa, fc, invalid = recorder.journal_counts(output / "forward_events.jsonl")
        try:
            derivatives = len(base.read_rows(output / "derivative_events.jsonl"))
        except (OSError, ValueError):
            derivatives, fault = None, fault or "missing/malformed derivative journal"
        elapsed = time.monotonic() - started
        if fa != 20 or fc != 20 or invalid or derivatives != 0 or elapsed > timeout:
            fault = fault or "20/0 accounting/deadline fault"
        status = {
            "status": "complete_valid" if fault is None else "INCONCLUSIVE",
            "reason": fault,
            "forward_attempts": fa,
            "completed_forwards": fc,
            "derivative_attempts": derivatives,
            "elapsed_seconds": elapsed,
            "cleanup_error": cleanup,
            "retries_allowed": False,
        }
        base.write_new(output / "RUN_STATUS.json", status)
    return status


def run():
    record = require_freeze()
    path = f"{OUTPUT}/preregistration.json"
    require(
        base.git(ROOT, "show", "--pretty=", "--name-only", "HEAD").splitlines() == [path]
        and base.git(ROOT, "rev-parse", "HEAD^") == record["source_commit"]
        and not base.git(ROOT, "status", "--porcelain", "--", path),
        "clean preregistration-only commit required",
    )
    usage = json.loads(os.environ.get("SP_LENSE_USAGE_PREFLIGHT", "null"))
    require(
        isinstance(usage, dict)
        and type(usage.get("standard_used_percent")) in (float, int)
        and 0 <= usage["standard_used_percent"] < 90
        and type(usage.get("checked_at_unix")) in (float, int)
        and 0 <= time.time() - usage["checked_at_unix"] <= 60,
        "fresh usage below90 required",
    )
    return supervise([sys.executable, "-u", str(ROOT / SCRIPT), "_worker"], ROOT / OUTPUT, usage)


if __name__ == "__main__":
    if sys.argv[1:] == ["_worker"]:
        worker()
    elif sys.argv[1:] in (["freeze"], ["run"]):
        stage = sys.argv[1]
        result = freeze() if stage == "freeze" else run()
        print(json.dumps(result, indent=2))
        if stage == "run" and result["status"] != "complete_valid":
            raise SystemExit(1)
    else:
        raise SystemExit("Use freeze or run; no adjustable vector/sign/strength inputs.")
