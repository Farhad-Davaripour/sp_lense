"""Exactly four immutable min-norm QPs followed by one model-free audit."""

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
from scripts import retention_guard_feasibility_io as io
from scripts import shared_preserve_eight_row_solver as solver


def system(records, name):
    io.require(name in io.ORDER and len(records) == 8, "fixed objective/eight rows")
    t = 1 if name.endswith("P") else -1
    c0 = [t * r["S0"] for r in records]
    A = [[t * io.norm(r["h0"]) * x for x in r["g"]] for r in records]
    b = [0.10 - c for c in c0]
    if name.startswith("guarded"):
        b = [max(x, 0.0) for x in b]
    R = max(1.0, *map(abs, b), *(io.norm(a) for a in A))
    io.require(math.isfinite(R), "finite scale")
    return {
        "name": name,
        "target_sign": t,
        "c0": c0,
        "A": A,
        "b": b,
        "scale": R,
        "endpoint_goals": [max(0.10, c) if name.startswith("guarded") else 0.10 for c in c0],
        "tolerances": {
            "rank_relative_pivot_floor": 1e-12,
            "primal_absolute_tolerance": 1e-9 * R,
            "kkt_absolute_tolerance": 1e-8 * R * R,
        },
    }


def calculate(records, config, journal, deadline, solve=solver.solve):
    io.require(
        config["solve_order"] == io.ORDER and config["maximum_solves"] == 4,
        "exact four-solve order",
    )
    results = []
    for name in io.ORDER:
        io.require(time.monotonic() < deadline, "diagnostic deadline")
        inputs = system(records, name)
        journal("attempt", name)
        result = solve(inputs["A"], inputs["b"], inputs["tolerances"])
        io.require([x["mask"] for x in result["active_sets"]] == list(range(256)), "256 masks")
        results.append(
            {
                **inputs,
                "solver": result,
                "role": "numeric audit only; never apply or freeze as steering candidate",
            }
        )
        journal("complete", name)
        print(f"completed QP {len(results)}/4: {name}; no model calls", flush=True)
    return {
        "objectives": results,
        "solve_order": io.ORDER,
        "qp_solves": 4,
        "model_calls": 0,
        "derivatives": 0,
        "selected_rows_sha256": io.canonical_sha(records),
    }


def worker():
    io.forbid_models()
    io.write_new(io.OUTPUT / "WORKER_CLAIM.json", {"pid": os.getpid()})
    record = io.locked()
    deadline = io.read(io.OUTPUT / "RUN_STARTED.json")["deadline_monotonic"]

    def journal(event, name):
        with (io.OUTPUT / "events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                json.dumps({"event": event, "name": name, "monotonic": time.monotonic()}) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())

    result = calculate(record["selected_rows"], record["config"], journal, deadline)
    io.write_new(io.OUTPUT / "analysis.json", result)
    io.require(time.monotonic() < deadline, "deadline before independent certificate pass")
    journal("attempt", "independent_audit")
    from scripts import verify_retention_guard_feasibility as audit

    verified = audit.verify(io.read(io.OUTPUT / "analysis.json"), io.locked())
    io.write_new(io.OUTPUT / "verification.json", verified)
    # Serialization/integrity check only, not a second arithmetic pass.
    io.require(io.read(io.OUTPUT / "verification.json") == verified, "audit artifact roundtrip")
    with (io.OUTPUT / "PILOT_REPORT.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(audit.report(verified))
    journal("complete", "independent_audit")
    io.forbid_models()
    io.write_new(
        io.OUTPUT / "MODEL_FREE.json",
        {
            "model_imports": [],
            "model_calls": 0,
            "derivatives": 0,
            "loaded_module_names": sorted(sys.modules),
        },
    )


def supervise(command, output, usage, timeout=180):
    started = time.monotonic()
    io.write_new(
        output / "RUN_STARTED.json",
        {
            "started_monotonic": started,
            "deadline_monotonic": started + timeout,
            "maximum_seconds": timeout,
            "command": command,
            "usage_preflight": usage,
            "qp_ceiling": 4,
            "audit_ceiling": 1,
            "model_calls": 0,
            "derivatives": 0,
        },
    )
    process, fault = None, None
    try:
        with (output / "worker.log").open("xb") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            code = process.wait(timeout=max(0.001, started + timeout - time.monotonic()))
            if code != 0:
                fault = f"worker exit {code}; no retry"
    except BaseException as error:  # noqa: BLE001 - persist interruptions, no rescue.
        fault = type(error).__name__ + ": " + str(error)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        elapsed = time.monotonic() - started
        events = (
            [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            if (output / "events.jsonl").exists()
            else []
        )
        expected = [
            (e, n) for n in io.ORDER + ["independent_audit"] for e in ("attempt", "complete")
        ]
        if (
            [(e["event"], e["name"]) for e in events] != expected
            or elapsed > timeout
            or any(not (started <= e["monotonic"] <= started + timeout) for e in events)
            or not (output / "verification.json").exists()
            or not (output / "MODEL_FREE.json").exists()
        ):
            fault = fault or "accounting/deadline/completion fault"
        status = {
            "status": "complete_valid" if fault is None else "INCONCLUSIVE",
            "reason": fault,
            "elapsed_seconds": elapsed,
            "maximum_seconds": timeout,
            "qp_attempts": sum(e["event"] == "attempt" and e["name"] in io.ORDER for e in events),
            "qp_completed": sum(e["event"] == "complete" and e["name"] in io.ORDER for e in events),
            "audit_attempts": sum(
                e["event"] == "attempt" and e["name"] == "independent_audit" for e in events
            ),
            "audit_completed": sum(
                e["event"] == "complete" and e["name"] == "independent_audit" for e in events
            ),
            "model_calls": 0,
            "derivatives": 0,
            "retries_allowed": False,
        }
        io.write_new(output / "RUN_STATUS.json", status)
    return status


def run():
    io.forbid_models()
    record = io.locked()
    path = io.OUTPUT.relative_to(ROOT).as_posix() + "/preregistration.json"
    io.require(
        io.git("show", "--pretty=", "--name-only", "HEAD").splitlines() == [path]
        and io.git("rev-parse", "HEAD^") == record["source_commit"]
        and not io.git("status", "--porcelain", "--", path),
        "preregistration-only commit",
    )
    return supervise(
        [sys.executable, "-u", str(ROOT / io.SCRIPT), "_worker"], io.OUTPUT, io.usage()
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["freeze"]:
        print(json.dumps(io.freeze(), indent=2))
    elif sys.argv[1:] == ["run"]:
        status = run()
        print(json.dumps(status, indent=2))
        if status["status"] != "complete_valid":
            raise SystemExit(1)
    elif sys.argv[1:] == ["_worker"]:
        worker()
    else:
        raise SystemExit("Use freeze or run; no adjustable formulas/inputs.")
