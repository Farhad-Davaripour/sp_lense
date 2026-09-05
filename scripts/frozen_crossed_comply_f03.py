"""One frozen-C-only f03 transfer. Physical vector never multiplied by semantic sign."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import frozen_crossed_comply_f03_plan as protocol

engine = protocol.isolate("scripts._frozen_crossed_C_f03_engine", "scripts/crossed_pair_probe.py")
engine.protocol = protocol
for key in ("SCRIPT", "VERIFY", "TEST", "DOC", "PLAN", "OUTPUT"):
    setattr(engine, key, getattr(protocol, key))
OUTPUT = ROOT / protocol.OUTPUT
require, base = protocol.require, engine.base
make_delta, DerivativeGuard = engine.make_delta, engine.DerivativeGuard
EligibilityError = engine.EligibilityError
norm, EPS = protocol.norm, protocol.EPS


def validate_vectors(plan, vectors):
    require(set(vectors) == set(plan["candidates"]) == {"comply"}, "one COMPLY vector only")
    v, meta = vectors["comply"], plan["candidates"]["comply"]
    require(
        len(v) == plan["model"]["d_model"]
        and all(math.isfinite(x) for x in v)
        and protocol.vector_sha(v) == meta["vector_float64_le_sha256"]
        and norm(v) == meta["norm"],
        "exact stored C before forward; no sign/strength/renormalization",
    )


engine.validate_vectors = validate_vectors
evaluate = protocol.adapt(engine, "evaluate", {20: 12}, {20: 1})
protocol.adapt(engine, "worker", {20: 12}, {20: 1})
supervise = protocol.adapt(engine, "supervise", {20: 12}, {20: 3})
freeze = protocol.adapt(engine, "freeze", {20: 12}, {20: 1})
protocol.adapt(
    engine,
    "descriptive_contrasts",
    replacements=[('("preserve", "comply")', '("comply",)')],
)
_parent_summary = protocol.adapt(
    engine,
    "summarize",
    {20: 12, 8: 4},
    {20: 2, 8: 3},
    replacements=[('("preserve", "comply")', '("comply",)')],
)
_parent_coverage = engine.coverage_counts


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
        "FROZEN_COMPLY_F03_DEVELOPMENT_ACCEPTED_ONLY"
        if result["matrix"]["matrix_pass"]
        else "FROZEN_COMPLY_F03_DEVELOPMENT_PARTIAL_OR_FAIL"
    )
    edits = [r for r in rows if r["phase"] == "edit"]
    result.update(
        retention_weakening=sum(
            r["baseline_argmax_id"] == r["requested_token_id"] and r["signed_delta_log_odds"] < 0
            for r in edits
        ),
        all_edits_shift_toward_A=all(r["delta_letter_log_odds"] > 0 for r in edits),
        exposed_semantic_situations=1,
        held_out_confirmation=False,
        physical_sign_inversion=False,
        training_performed=False,
        old_v2_success_inherited=False,
    )
    return result


_source_identity = engine.source_identity


def source_identity():
    result = _source_identity()
    paths = [
        *protocol.build_plan()["input_sha256"],
        "scripts/crossed_pair_probe.py",
        "scripts/crossed_pair_plan.py",
        "scripts/verify_crossed_pair_probe.py",
        "tests/test_crossed_pair_probe.py",
        "scripts/frozen_guarded_preserve_crossed_plan.py",
        "scripts/shared_comply_crossed_plan.py",
    ]
    for path in paths:
        require(base.git(ROOT, "ls-files", "--", path), f"untracked source/input {path}")
        require(
            not base.git(ROOT, "status", "--porcelain", "--", path), f"dirty source/input {path}"
        )
        result[path] = protocol.sha((ROOT / path).read_bytes())
    return result


engine.coverage_counts, engine.summarize = coverage_counts, summarize
engine.source_identity = source_identity
require_freeze = engine.require_freeze


def preflight(worker_entry=False):
    """Read-only preflight; no claim/artifact/model created."""
    from scripts.verify_frozen_crossed_comply_f03 import verify_renderings

    record = require_freeze()
    lock_path = protocol.OUTPUT + "/preregistration.json"
    require(
        base.git(ROOT, "show", "--pretty=", "--name-only", "HEAD").splitlines() == [lock_path]
        and base.git(ROOT, "rev-parse", "HEAD^") == record["source_commit"]
        and not base.git(ROOT, "status", "--porcelain", "--", lock_path),
        "clean preregistration-only HEAD/source parent",
    )
    expected = {"preregistration.json"}
    if worker_entry:
        expected |= {"RUN_STARTED.json", "worker.log"}
    require(
        OUTPUT.is_dir() and {p.name for p in OUTPUT.iterdir()} == expected,
        "untouched namespace; no prior claim/calls/retry/extra files",
    )
    verify_renderings(record["plan"])
    require(
        len(record["plan"]["prompts"]) == 4
        and len(record["plan"]["cells"]) == 12
        and not record["plan"]["derivative_cells"],
        "exact4/12/0 before load",
    )
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


def worker():
    preflight(worker_entry=True)
    started = protocol.read(OUTPUT / "RUN_STARTED.json")
    usage = started.get("usage_preflight", {})
    values = [started.get(k) for k in ("started_monotonic", "deadline_monotonic")]
    values += [usage.get(k) for k in ("standard_used_percent", "checked_at_unix")]
    require(
        all(type(v) in (int, float) and math.isfinite(v) for v in values),
        "finite started/deadline/usage before worker claim",
    )
    require(
        started["command"] == [sys.executable, "-u", str(ROOT / protocol.SCRIPT), "_worker"]
        and started["forward_ceiling"] == 12
        and started["derivative_ceiling"] == 0
        and started["timeout_seconds"] == 600
        and started["deadline_monotonic"] == started["started_monotonic"] + 600
        and started["started_monotonic"] <= time.monotonic() < started["deadline_monotonic"]
        and 0 <= usage["standard_used_percent"] < 90
        and 0 <= time.time() - usage["checked_at_unix"] <= 60,
        "exact12/0/600 finite external budget and fresh usage before claim",
    )
    return engine.worker()


def run():
    preflight()
    return engine.run()


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
