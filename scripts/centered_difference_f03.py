"""One +d/P and -d/C exposed f03 study; unmodified verified 20-cell numerical engine."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import centered_difference_f03_plan as protocol

engine = protocol.isolate(
    "scripts._centered_difference_f03_engine", "scripts/crossed_pair_probe.py"
)
engine.protocol = protocol
for key in ("SCRIPT", "VERIFY", "TEST", "DOC", "PLAN", "OUTPUT"):
    setattr(engine, key, getattr(protocol, key))
OUTPUT = ROOT / protocol.OUTPUT
require, base = protocol.require, engine.base
make_delta, evaluate, supervise = engine.make_delta, engine.evaluate, engine.supervise
DerivativeGuard, EligibilityError = engine.DerivativeGuard, engine.EligibilityError
freeze, require_freeze = engine.freeze, engine.require_freeze
EPS, norm = protocol.EPS, protocol.norm
_parent_summary, _parent_coverage = engine.summarize, engine.coverage_counts


def validate_vectors(plan, vectors):
    require(
        set(vectors) == set(plan["candidates"]) == {"preserve", "comply"}, "exact signed d pair"
    )
    for target, v in vectors.items():
        meta = plan["candidates"][target]
        require(
            len(v) == plan["model"]["d_model"]
            and all(type(x) is float and math.isfinite(x) for x in v)
            and protocol.vector_sha(v) == meta["vector_float64_le_sha256"]
            and norm(v) == meta["norm"]
            and meta["physical_sign"] == (1 if target == "preserve" else -1),
            "fixed signed coordinates/norm before any forward",
        )
        require(
            meta["stored_vector_float64_le_sha256"] == protocol.vector_sha(vectors["preserve"]),
            "both physical signs bind the same original stored d hash",
        )
    require(
        protocol.vector_sha(vectors["comply"])
        == protocol.vector_sha([-x for x in vectors["preserve"]]),
        "physical C is exact negative of P; no midpoint or parents",
    )


def coverage_counts(group):
    result = _parent_coverage(group)
    for direction in ("A_to_B", "B_to_A"):
        n, got = result["eligible_" + direction], result["achieved_" + direction]
        result[direction + "_status"] = (
            "UNTESTED" if n == 0 else "ALL" if got == n else "PARTIAL_OR_FAIL"
        )
    return result


def summarize(rows):
    result = _parent_summary(rows)
    result["status"] = (
        "CENTERED_DIFFERENCE_F03_DEVELOPMENT_ACCEPTED_ONLY"
        if result["matrix"]["matrix_pass"]
        else "CENTERED_DIFFERENCE_F03_DEVELOPMENT_PARTIAL_OR_FAIL"
    )
    edits = [r for r in rows if r["phase"] == "edit"]
    result["retention_weakening_by_sign"] = {
        target: sum(
            r["requested"] == target
            and r["baseline_argmax_id"] == r["requested_token_id"]
            and r["signed_delta_log_odds"] < 0
            for r in edits
        )
        for target in ("preserve", "comply")
    }
    result.update(
        exposed_semantic_situations=1,
        held_out_confirmation=False,
        midpoint_applied=False,
        parent_vectors_applied=False,
        training_performed=False,
        source_success_inherited=False,
    )
    return result


def source_identity():
    result = {}
    paths = [
        *protocol.build_plan()["input_sha256"],
        *(getattr(protocol, key) for key in ("CONFIG", "DOC", "SCRIPT", "VERIFY", "TEST", "PLAN")),
        "scripts/crossed_pair_probe.py",
        "scripts/crossed_pair_plan.py",
        "scripts/verify_crossed_pair_probe.py",
        "scripts/shared_comply_crossed_plan.py",
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
engine.coverage_counts, engine.summarize = coverage_counts, summarize
engine.source_identity = source_identity


def preflight(worker_entry=False):
    from scripts.verify_centered_difference_f03 import verify_renderings

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
        "forwards": 20,
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
        and started["forward_ceiling"] == 20
        and started["derivative_ceiling"] == 0
        and started["timeout_seconds"] == 600
        and started["deadline_monotonic"] == started["started_monotonic"] + 600
        and started["started_monotonic"] <= time.monotonic() < started["deadline_monotonic"],
        "exact finite20/0/600 worker envelope before claim/load",
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
