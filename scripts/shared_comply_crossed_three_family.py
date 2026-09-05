"""Twelve-row binding of the immutable crossed COMPLY method; no new model math.

Preparation does not authorize a real run. The future launch remains locked to
separate supervisor authorization, one attempt, and the prospective source HEAD.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import frozen_crossed_comply_f03_plan as adapter
from scripts import shared_comply_crossed_three_family_plan as protocol
from scripts import shared_comply_twelve_row_solver as optimizer
from scripts import three_family_recording_bindings as recording_bindings

shell = protocol.isolate(
    "scripts._three_family_crossed_comply_shell", "scripts/shared_comply_crossed.py"
)
engine = shell.engine
OUTPUT = ROOT / protocol.OUTPUT
shell.protocol, shell.OUTPUT = protocol, OUTPUT
engine.protocol, engine.OUTPUT, engine.optimizer = protocol, OUTPUT, optimizer
require, base, adapt = protocol.require, engine.base, adapter.adapt


def bound_method(method, integers=None, sites=None, replacements=()):
    """Adapt a method in copied globals, without modifying its historical class."""
    namespace = types.ModuleType("scripts._three_family_method_" + method.__name__)
    namespace.__dict__.update(method.__globals__)
    namespace.__dict__[method.__name__] = method
    return adapt(namespace, method.__name__, integers, sites, replacements)


class ForwardLedger(engine.ForwardLedger):
    begin = bound_method(
        engine.ForwardLedger.begin,
        {144: 216},
        {144: 1},
        [("forward144/order", "forward216/order")],
    )


class Derivatives(engine.Derivatives):
    call = bound_method(
        engine.Derivatives.call,
        {64: 96},
        {64: 1},
        [("derivative64/order/deadline", "derivative96/order/deadline")],
    )


class Session(engine.Session):
    # Original numerical body; only progress-message ceilings differ.
    call = bound_method(
        engine.Session.call,
        replacements=[
            ("/<=144 forwards;", "/<=216 forwards;"),
            ("/<=64 derivatives", "/<=96 derivatives"),
        ],
    )


engine.ForwardLedger, engine.Derivatives, engine.Session = ForwardLedger, Derivatives, Session
increment, project, norm, dot, vector_sha = (
    engine.increment,
    engine.project,
    engine.norm,
    engine.dot,
    engine.vector_sha,
)
accepts, quality, stopping, EPS, ROUND_EPS = (
    engine.accepts,
    engine.quality,
    engine.stopping,
    engine.EPS,
    engine.ROUND_EPS,
)
drive = adapt(
    engine,
    "drive",
    {8: 12},
    {8: 1},
    [
        (
            "exact eight-row construction; no controls or transfer",
            "exact twelve-row construction; no controls or transfer",
        )
    ],
)
evaluate = adapt(engine, "evaluate", {144: 216}, {144: 1})
_frozen_freeze = adapt(engine, "freeze", {144: 216, 64: 96}, {144: 1, 64: 1})
adapt(engine, "worker", {144: 216, 64: 96}, {144: 1, 64: 1})
require_freeze = engine.require_freeze
_crossed_summary = adapt(
    shell,
    "summarize",
    {8: 12, 2: 3},
    {8: 2, 2: 1},
    [
        ("exact eight final renderings", "exact twelve final renderings"),
        (
            "EXPOSED development; not run",
            "TRAINING for this new candidate only; prior candidate transfer history unchanged",
        ),
    ],
)


def summarize(rows, result):
    value = _crossed_summary(rows, result)
    value["f04_status"] = "EXPOSED development outside fitting; not pristine heldout"
    return value


PARENT_LOCK = "evidence/shared_comply_crossed_f01_f02_v1_qwen35_08b/preregistration.json"
PARENT_LOCK_SHA = "c3c32fc972541d9b3969525b3edc088372a164e5815f3afe69cb55e7bd40d431"


def source_identity():
    """All immutable parent-source hashes plus clean new sources and input bytes."""
    raw = (ROOT / PARENT_LOCK).read_bytes()
    require(protocol.sha(raw) == PARENT_LOCK_SHA, "exact immutable crossed parent lock")
    prior = json.loads(raw)
    expected = {**prior["source_sha256"], PARENT_LOCK: PARENT_LOCK_SHA}
    for path, digest in protocol.build_plan()["input_sha256"].items():
        require(path not in expected or expected[path] == digest, "consistent parent/input hash")
        expected[path] = digest
    config = protocol.read(ROOT / protocol.CONFIG)
    for spec in config["protected_artifacts"]:
        path, digest = spec["path"], spec["sha256"]
        require(path not in expected or expected[path] == digest, "consistent protected input hash")
        expected[path] = digest
    result = {}
    for path in dict.fromkeys([*expected, *protocol.SOURCE_PATHS]):
        require(
            base.git(ROOT, "ls-files", "--", path)
            and not base.git(ROOT, "status", "--porcelain", "--", path),
            "clean tracked source/input " + path,
        )
        digest = protocol.sha((ROOT / path).read_bytes())
        require(
            path not in expected or digest == expected[path], "source/input byte mismatch " + path
        )
        result[path] = digest
    return result


engine.source_identity, engine.summarize = source_identity, summarize
shell.source_identity, shell.summarize = source_identity, summarize
shell.require_freeze = require_freeze
require_authorization = shell.require_authorization
_locked_preflight = adapt(
    shell,
    "preflight",
    replacements=[
        (
            "from scripts.verify_shared_comply_crossed import verify_plan",
            "from scripts.verify_shared_comply_crossed_three_family import verify_plan",
        )
    ],
)


def freeze():
    protocol.require_preparation_certificate()
    original_writer = base.write_new
    base.write_new = recording_bindings.write_preregistration
    try:
        return _frozen_freeze()
    finally:
        base.write_new = original_writer


def preflight(worker_entry=False):
    protocol.require_preparation_certificate()
    return _locked_preflight(worker_entry=worker_entry)


shell.preflight = preflight
worker = adapt(
    shell,
    "worker",
    {144: 216, 64: 96, 900: 1200},
    {144: 1, 64: 1, 900: 2},
)


def untouched_namespace(worker_entry=False):
    expected = {"preregistration.json"}
    if worker_entry:
        expected |= {"RUN_STARTED.json", "worker.log", "recording_state.json", "recording.lock"}
    require(
        OUTPUT.is_dir() and {p.name for p in OUTPUT.iterdir()} == expected,
        "untouched namespace; only prospective lock and bounded startup controls",
    )


shell.untouched_namespace = untouched_namespace
_unrecorded_worker = engine.worker


def _budgeted_worker():
    from scripts.three_family_bounded_capture import exception_record
    from scripts.three_family_recording_budget import Budget

    budget = Budget(OUTPUT, initialize=False)
    try:
        with recording_bindings.bind_writers(budget):
            return _unrecorded_worker()
    except BaseException as error:  # noqa: BLE001 - bounded technical failure, including interrupts.
        # Do not print an arbitrary exception traceback outside the bounded capture.
        print(
            json.dumps({"status": "INCONCLUSIVE", "exception": exception_record(error)}), flush=True
        )
        raise SystemExit(1) from None


engine.worker = _budgeted_worker


def supervise(command, output, usage, timeout=1200):
    """One bounded recording attempt; scientific schedule and stopping are unchanged."""
    from scripts import three_family_bounded_capture as capture
    from scripts.three_family_recording_budget import Budget
    from scripts.verify_shared_comply_crossed_three_family import finalize_recording

    output, started = Path(output), time.monotonic()
    require(timeout == 1200, "fixed1200s including loading; no extension")
    budget = Budget(output)
    receipt = None
    capture_entered = False
    reason = None
    result = None
    try:
        with recording_bindings.bind_writers(budget):
            base.write_new(
                output / "RUN_STARTED.json",
                {
                    "command": command,
                    "started_monotonic": started,
                    "deadline_monotonic": started + timeout,
                    "timeout_seconds": timeout,
                    "usage_preflight": usage,
                    "forward_ceiling": 216,
                    "derivative_ceiling": 96,
                },
            )
        capture_entered = True
        receipt = capture.run_capture(command, budget, started + timeout, cwd=ROOT)
        if receipt["status"] != "complete_valid":
            reason = "technical bounded capture/worker/termination failure"
        if (output / "INVALID.json").exists() or not (output / "analysis.json").exists():
            reason = reason or "worker incomplete or invalid"
        fa, fc, fi = engine.recorder.journal_counts(output / "forward_events.jsonl")
        da, dc, di = engine.recorder.journal_counts(output / "derivative_events.jsonl")
        try:
            skips = base.read_rows(output / "skip_events.jsonl")
        except (ValueError, OSError, TypeError):
            skips, reason = [], reason or "invalid/incomplete skip journal"
        elapsed = time.monotonic() - started
        if (
            not (24 <= fa == fc <= 216 and 0 <= da == dc <= 96 and fa + len(skips) == 216)
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
            "cleanup_error": receipt.get("cleanup_error"),
            "retries_allowed": False,
        }
    except BaseException as error:  # noqa: BLE001 - recording must fail closed without retry.
        result = {
            "status": "INCONCLUSIVE",
            "exception": capture.exception_record(error),
            "retries_allowed": False,
        }
        if receipt is None:
            receipt = {
                "status": "INCONCLUSIVE",
                "quiescent": not capture_entered,
                "worker_started": None if capture_entered else False,
                "exception": capture.exception_record(error),
            }
    return finalize_recording(budget, receipt, result)


engine.supervise = supervise


def checked_usage(value):
    require(
        isinstance(value, dict)
        and all(
            type(value.get(key)) in (int, float) and math.isfinite(value[key])
            for key in ("standard_used_percent", "checked_at_unix")
        )
        and 0 <= value["standard_used_percent"] < 90
        and 0 <= time.time() - value["checked_at_unix"] <= 60,
        "finite fresh standard usage below90 required",
    )
    return value


def run():
    require_authorization()
    preflight()
    usage = checked_usage(json.loads(os.environ.get("SP_LENSE_USAGE_PREFLIGHT", "null")))
    return supervise([sys.executable, "-u", str(ROOT / protocol.SCRIPT), "_worker"], OUTPUT, usage)


if __name__ == "__main__":
    commands = {"freeze": freeze, "preflight": preflight, "run": run, "_worker": worker}
    require(
        len(sys.argv) == 2 and sys.argv[1] in commands,
        "Use freeze/preflight; run requires separate supervisor authorization",
    )
    result = commands[sys.argv[1]]()
    print(json.dumps(result, indent=2, allow_nan=False))
    if sys.argv[1] == "run" and result["status"] != "complete_valid":
        raise SystemExit(1)
