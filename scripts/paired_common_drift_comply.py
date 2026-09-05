"""One prospectively locked paired COMPLY construction; real execution needs new authority."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import paired_common_drift_comply_optimizer as optimizer
from scripts import paired_common_drift_comply_plan as protocol
from scripts import paired_common_drift_comply_recording as recording
from scripts import three_family_bounded_capture as capture
from scripts import three_family_recording_bindings as bindings

parent = protocol.isolate(
    "scripts._paired_comply_runtime", "scripts/shared_comply_crossed_three_family.py"
)
engine, OUTPUT = parent.engine, ROOT / protocol.OUTPUT
base, require = engine.base, protocol.require
ForwardLedger, Derivatives, Session = parent.ForwardLedger, parent.Derivatives, parent.Session
accepts, quality, stopping = parent.accepts, parent.quality, parent.stopping
norm, vector_sha, EPS, ROUND_EPS = parent.norm, parent.vector_sha, parent.EPS, parent.ROUND_EPS
increment = optimizer.increment
PairedBudget = recording.PairedBudget


def drive(plan, call, skip, propose, save_update, save_endpoint):
    """Original conditional schedule; one new objective/update, fresh baselines and zero."""
    cells = {(c["prompt_id"], c["condition"]): c for c in plan["cells"]}
    ids = plan["construction_ids"]
    require(
        len(ids) == 12 and not plan["control_ids"] and not plan["transfer_ids"], "12 training only"
    )
    w, path = [0.0] * plan["model"]["d_model"], 0.0
    current = [call(cells[pid, "baseline"], w, None) for pid in ids]
    baselines = list(current)
    stop, applied, attempted = stopping(current), 0, 0
    anchor = current[-1]["row"]["cell_id"]
    for stage in range(1, 9):
        gc = [cells[pid, f"gradient_{stage}"] for pid in ids]
        sc = [cells[pid, f"step_{stage}"] for pid in ids]
        if stop:
            for cell in gc + sc:
                skip(cell, stop, anchor)
            continue
        attempted += 1
        gradients = [call(c, w, s) for c, s in zip(gc, current, strict=True)]
        anchor = gradients[-1]["row"]["cell_id"]
        if any(not quality(state["row"]) for state in gradients):
            stop = "quality_failure"
        else:
            proposal = propose(gradients, w, path, stage, baselines)
            # Preserve the attempted update even if a later scored forward fails.
            save_update(proposal)
            require(
                proposal["status"] in ("ready", "method_zero_increment", "projection_stall"),
                "one finite declared update; no alternative optimizer",
            )
            if proposal["status"] != "ready":
                stop = proposal["status"]
            else:
                require(
                    proposal["step_norm"] <= 0.05 + ROUND_EPS
                    and proposal["path_after"] <= 0.40 + ROUND_EPS
                    and proposal["net_norm"] <= 0.20 + ROUND_EPS,
                    "shared step/path/net bound",
                )
                w, path, applied = proposal["w_after"], proposal["path_after"], applied + 1
        if stop:
            for cell in sc:
                skip(cell, stop, anchor)
            continue
        current = [call(c, w, s) for c, s in zip(sc, current, strict=True)]
        anchor, stop = current[-1]["row"]["cell_id"], stopping(current)
    endpoint = {
        "w": w,
        "vector_float64_le_sha256": vector_sha(w),
        "path": path,
        "net": norm(w),
        "updates": applied,
        "attempted_updates": attempted,
        "stop_reason": stop or "max_updates",
        "endpoint_cell_ids": [s["row"]["cell_id"] for s in current],
    }
    save_endpoint(endpoint)
    final = [call(cells[pid, "final"], w, s) for pid, s in zip(ids, current, strict=True)]
    return {
        **endpoint,
        "final_cell_ids": [s["row"]["cell_id"] for s in final],
        "transfer_ran": False,
        "candidate_eligible": all(accepts(s["row"]) for s in final),
    }


def summarize(rows, result):
    value = parent.summarize(rows, result)
    baselines = [r for r in rows if r["condition"] == "baseline"]
    conditions = ["baseline", *(f"step_{i}" for i in range(1, 9)), "final"]
    value["paired_objective_trajectory"] = [
        {"condition": condition, **optimizer.objective(group, baselines)}
        for condition in conditions
        if (group := [r for r in rows if r["condition"] == condition])
    ]
    return value


def evaluate(plan, backend, ledger, derivatives, output, propose=increment):
    output = Path(output)
    with (output / "updates.jsonl").open("x", encoding="utf-8"):
        pass
    with Session(plan, backend, ledger, derivatives, output) as session:
        result = drive(
            plan,
            session.call,
            ledger.skip,
            propose,
            lambda update: base.append_row(output / "updates.jsonl", update),
            lambda endpoint: base.write_new(output / "endpoint.json", endpoint),
        )
    require(
        ledger.cursor == 216
        and ledger.attempts == ledger.completed
        and derivatives.attempts == derivatives.completed,
        "conditional schedule/accounting incomplete",
    )
    base.write_new(output / "result.json", result)
    return session.rows, summarize(session.rows, result)


def source_identity():
    """Exact scoped inputs/reused code and raw Git bytes; no historical evidence audit."""
    expected = protocol.build_plan()["input_sha256"]
    paths = list(dict.fromkeys([*expected, *protocol.SOURCE_PATHS]))

    def git(*args, input=None):
        return subprocess.check_output(["git", *args], cwd=ROOT, input=input)

    tracked = {}
    for item in git("ls-files", "--stage", "-z").split(b"\0"):
        if item:
            metadata, path = item.split(b"\t", 1)
            _mode, blob, stage = metadata.split()
            require(stage == b"0", "no unmerged source state")
            tracked[path.decode()] = blob.decode()
    dirty = set()
    for options in (("diff", "--name-only", "-z"), ("diff", "--cached", "--name-only", "-z")):
        dirty.update(x.decode() for x in git(*options).split(b"\0") if x)
    require(all(p in tracked and p not in dirty for p in paths), "clean tracked source/input set")
    result = {p: protocol.sha((ROOT / p).read_bytes()) for p in paths}
    require(all(result[p] == digest for p, digest in expected.items()), "frozen input bytes")
    blobs = (
        git(
            "hash-object", "--no-filters", "--stdin-paths", input=("\n".join(paths) + "\n").encode()
        )
        .decode()
        .splitlines()
    )
    require(
        all(tracked[p] == b for p, b in zip(paths, blobs, strict=True)),
        "raw Git source bytes without normalization",
    )
    return result


def freeze():
    recording.require_certificate()
    protocol.storage_preflight(ROOT, protocol.config_at())
    record = {
        "plan": protocol.build_plan(),
        "source_commit": base.git(ROOT, "rev-parse", "HEAD"),
        "source_sha256": source_identity(),
        "environment": base.environment(),
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    bindings.write_preregistration(OUTPUT / "preregistration.json", record)
    return {
        "source_commit": record["source_commit"],
        "forward_ceiling": 216,
        "derivative_ceiling": 96,
    }


def require_freeze():
    record = protocol.read(OUTPUT / "preregistration.json")
    require(
        record["plan"] == protocol.build_plan()
        and record["source_sha256"] == source_identity()
        and record["environment"] == base.environment(),
        "frozen plan/source/environment changed",
    )
    return record


def preflight(worker_entry=False):
    from scripts.verify_paired_common_drift_comply import verify_plan

    recording.require_certificate()
    record = require_freeze()
    path = protocol.OUTPUT + "/preregistration.json"
    require(
        base.git(ROOT, "show", "--pretty=", "--name-only", "HEAD").splitlines() == [path]
        and base.git(ROOT, "rev-parse", "HEAD^") == record["source_commit"]
        and not base.git(ROOT, "status", "--porcelain", "--", path),
        "clean preregistration-only HEAD immediately after source commit",
    )
    names = {"preregistration.json"}
    if worker_entry:
        names |= {"RUN_STARTED.json", "worker.log", "recording_state.json", "recording.lock"}
    require(
        OUTPUT.is_dir() and {p.name for p in OUTPUT.iterdir()} == names,
        "untouched namespace; no stale claim, calls, retry or extra artifacts",
    )
    return {
        "status": "ZERO_MODEL_PREFLIGHT_PASSED",
        **verify_plan(record["plan"]),
        "source_commit": record["source_commit"],
        "source_hash_entries": len(record["source_sha256"]),
        "preregistration_sha256": protocol.sha((OUTPUT / "preregistration.json").read_bytes()),
        "storage": protocol.storage_preflight(ROOT, record["plan"]["config"]),
        "model_loads": 0,
        "tokenizer_loads": 0,
        "real_forwards": 0,
        "real_derivatives": 0,
        "run_authorized": False,
    }


def require_authorization():
    require(
        json.loads(os.environ.get(protocol.AUTH_KEY, "null"))
        == {
            "authorized_by": "supervisor",
            "scope": protocol.AUTH_SCOPE,
            "preregistration_sha256": protocol.sha((OUTPUT / "preregistration.json").read_bytes()),
        },
        "separate supervisor authorization bound to exact paired lock required",
    )


def checked_usage(value):
    require(
        isinstance(value, dict)
        and all(
            type(value.get(k)) in (int, float) and math.isfinite(value[k])
            for k in ("standard_used_percent", "checked_at_unix")
        )
        and 0 <= value["standard_used_percent"] < 90
        and 0 <= time.time() - value["checked_at_unix"] <= 60,
        "finite fresh standard usage below90 required",
    )
    return value


def worker():
    require_authorization()
    preflight(worker_entry=True)
    record = protocol.read(OUTPUT / "preregistration.json")
    started = protocol.read(OUTPUT / "RUN_STARTED.json")
    usage = checked_usage(started.get("usage_preflight"))
    require(
        all(
            type(started.get(k)) in (int, float) and math.isfinite(started[k])
            for k in ("started_monotonic", "deadline_monotonic")
        )
        and started["command"] == [sys.executable, "-u", str(ROOT / protocol.SCRIPT), "_worker"]
        and started["forward_ceiling"] == 216
        and started["derivative_ceiling"] == 96
        and started["timeout_seconds"] == 1200
        and started["deadline_monotonic"] == started["started_monotonic"] + 1200
        and started["started_monotonic"] <= time.monotonic() < started["deadline_monotonic"],
        "exact finite 216/96/1200 budget before claim or loading",
    )
    budget = PairedBudget(OUTPUT, initialize=False)
    try:
        with bindings.bind_writers(budget):
            base.write_new(OUTPUT / "WORKER_CLAIM.json", {"pid": os.getpid()})
            storage = protocol.storage_preflight(ROOT, record["plan"]["config"])
            base.write_new(
                OUTPUT / "storage_preflight.json", {**storage, "monotonic": time.monotonic()}
            )
            checked_usage(usage)
            require(time.monotonic() < started["deadline_monotonic"], "deadline before loading")
            backend, _unused = base.load_backend(record["plan"])
            base.write_new(
                OUTPUT / "runtime.json",
                {
                    **backend.metadata(),
                    **base.environment(),
                    "logits_encoding": "zlib little-endian float32",
                    "forward_ceiling": 216,
                    "derivative_ceiling": 96,
                },
            )
            ledger = ForwardLedger(OUTPUT, record["plan"]["cells"], started["deadline_monotonic"])
            derivatives = Derivatives(
                backend.torch,
                OUTPUT,
                record["plan"]["derivative_cells"],
                started["deadline_monotonic"],
            )
            _, summary = evaluate(record["plan"], backend, ledger, derivatives, OUTPUT)
            base.write_new(OUTPUT / "analysis.json", summary)
    except BaseException as error:  # noqa: BLE001 - one bounded terminal failure, no retry.
        failure = {
            "status": "INCONCLUSIVE",
            "exception": capture.exception_record(error),
            "retries_allowed": False,
        }
        try:
            budget.write_bytes(OUTPUT / "INVALID.json", bindings.encoded_json(failure))
        except BaseException as receipt_error:  # noqa: BLE001 - no recursive failure logging.
            failure["invalid_receipt_persisted"] = False
            failure["receipt_exception"] = capture.exception_record(receipt_error)
        print(json.dumps(failure), flush=True)
        raise SystemExit(1) from None


def supervise(command, output, usage, timeout=1200):
    from scripts.verify_paired_common_drift_comply import finalize_recording

    output, started = Path(output), time.monotonic()
    require(timeout == 1200, "fixed1200s including loading; no extension")
    usage = checked_usage(usage)
    budget = PairedBudget(output)
    receipt, entered = None, False
    try:
        budget.write_bytes(
            output / "RUN_STARTED.json",
            bindings.encoded_json(
                {
                    "command": command,
                    "started_monotonic": started,
                    "deadline_monotonic": started + timeout,
                    "timeout_seconds": timeout,
                    "usage_preflight": usage,
                    "forward_ceiling": 216,
                    "derivative_ceiling": 96,
                }
            ),
        )
        entered = True
        receipt = capture.run_capture(command, budget, started + timeout, cwd=ROOT)
        if receipt.get("quiescent") is not True:
            result = {
                "status": "INCONCLUSIVE",
                "reason": "worker/writer quiescence unconfirmed",
                "forward_attempts": None,
                "completed_forwards": None,
                "derivative_attempts": None,
                "skipped_cells": None,
                "elapsed_seconds": time.monotonic() - started,
                "cleanup_error": receipt.get("cleanup_error"),
                "retries_allowed": False,
            }
        else:
            reason = (
                None if receipt["status"] == "complete_valid" else "bounded capture/worker failure"
            )
            if (output / "INVALID.json").exists() or not (output / "analysis.json").exists():
                reason = reason or "worker incomplete or invalid"
            fa, fc, fi = engine.recorder.journal_counts(output / "forward_events.jsonl")
            da, dc, di = engine.recorder.journal_counts(output / "derivative_events.jsonl")
            skips = (
                base.read_rows(output / "skip_events.jsonl")
                if (output / "skip_events.jsonl").exists()
                else []
            )
            elapsed = time.monotonic() - started
            if (
                not (24 <= fa == fc <= 216 and 0 <= da == dc <= 96 and fa + len(skips) == 216)
                or fi
                or di
                or elapsed > timeout
            ):
                reason = reason or "conditional journal/count/deadline fault"
            result = {
                "status": "complete_valid" if reason is None else "INCONCLUSIVE",
                "reason": reason,
                "forward_attempts": fa,
                "completed_forwards": fc,
                "derivative_attempts": da,
                "skipped_cells": len(skips),
                "elapsed_seconds": elapsed,
                "cleanup_error": receipt.get("cleanup_error"),
                "retries_allowed": False,
            }
    except BaseException as error:  # noqa: BLE001 - bounded fail-closed recording.
        result = {
            "status": "INCONCLUSIVE",
            "exception": capture.exception_record(error),
            "retries_allowed": False,
        }
        if receipt is None:
            receipt = {
                "status": "INCONCLUSIVE",
                "quiescent": not entered,
                "worker_started": None if entered else False,
                "exception": capture.exception_record(error),
            }
    return finalize_recording(budget, receipt, result)


def run():
    require_authorization()
    preflight()
    usage = checked_usage(json.loads(os.environ.get("SP_LENSE_USAGE_PREFLIGHT", "null")))
    return supervise([sys.executable, "-u", str(ROOT / protocol.SCRIPT), "_worker"], OUTPUT, usage)


if __name__ == "__main__":
    commands = {"freeze": freeze, "preflight": preflight, "run": run, "_worker": worker}
    require(
        len(sys.argv) == 2 and sys.argv[1] in commands,
        "Use freeze/preflight/run; no recipe switches",
    )
    result = commands[sys.argv[1]]()
    print(json.dumps(result, indent=2, allow_nan=False))
    if sys.argv[1] == "run" and result["status"] != "complete_valid":
        raise SystemExit(1)
