"""Crossed-layout wrapper; unchanged eight-row COMPLY engine in isolated globals."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import shared_comply_crossed_plan as protocol
from scripts import shared_comply_two_family as parent

engine = protocol.isolate("scripts._crossed_comply_engine", "scripts/shared_comply_two_family.py")
OUTPUT = ROOT / protocol.OUTPUT
engine.protocol, engine.OUTPUT = protocol, OUTPUT
require, base = protocol.require, engine.base
increment, project, norm, vector_sha = (
    engine.increment,
    engine.project,
    engine.norm,
    engine.vector_sha,
)
accepts, quality, drive = engine.accepts, engine.quality, engine.drive
ForwardLedger, Derivatives, Session = engine.ForwardLedger, engine.Derivatives, engine.Session
_parent_summary = engine.summarize

# Byte-hash provenance only: never parse old candidates, failures or smoke results.
HISTORICAL_BYTE_GUARDS = {
    "evidence/shared_comply_two_family_qwen35_08b/comply_vector.json": "2c3beb65e308dfe757d3c50a70dc504abf7495fdafb80472a0f706cfc7d9687f",
    "evidence/crossed_pair_f03_v1_qwen35_08b/PILOT_REPORT.md": "ab86481f784627da7ce56f5c580d3fcdf76d4f62e699013307f70fee99170067",
    "evidence/crossed_pair_f03_v1_qwen35_08b/verification.json": "1781d09823b7b95cbab3975958fa0a01153e722385b08dcda4f6852e67ab97ee",
    "evidence/frozen_endpoint020_oracle_accuracy6_qwen35_08b/PILOT_REPORT.md": "64793db16bd337c2ee22b053d6556eda6e4eebfa15f47f28c499864bdec46cab",
    "evidence/frozen_endpoint020_oracle_accuracy6_qwen35_08b/verification.json": "bfbf5491569ede258e88eefcf65083b33d7dbca5192b642596bc653601aa9ba6",
}


def summarize(rows, result):
    value = _parent_summary(rows, result)
    finals = [r for r in rows if r["condition"] == "final"]
    require(len(finals) == 8, "exact eight final renderings")
    coverage = {}
    for source, target in (("A", "B"), ("B", "A")):
        eligible = [
            r for r in finals if r["baseline_label"] == source and r["comply_label"] == target
        ]
        achieved = sum(accepts(r) and r["actual_next_token_label"] == target for r in eligible)
        coverage[source + "_to_" + target] = {
            "eligible": len(eligible),
            "achieved": achieved,
            "status": "UNTESTED"
            if not eligible
            else "ALL"
            if achieved == len(eligible)
            else "PARTIAL_OR_FAIL",
        }
    for outcome, row in zip(value["final_cells"], finals, strict=True):
        outcome.update(
            {k: row[k] for k in ("semantic_mapping", "display_order", "rendering_index")}
        )
    value.update(
        directional_coverage=coverage,
        retention_weakening=sum(
            r["baseline_argmax_id"] == r["requested_token_id"] and r["signed_delta_log_odds"] < 0
            for r in finals
        ),
        independent_semantic_situations=2,
        rendering_rows=8,
        outcome_informed_successor=True,
        old_v2_success_inherited=False,
        f03_status="EXPOSED development; not run",
    )
    return value


def source_identity():
    result = parent.source_identity()
    config = protocol.read(ROOT / protocol.CONFIG)
    paths = [
        protocol.CONFIG,
        protocol.DOC,
        protocol.PREP_REPORT,
        protocol.SCRIPT,
        protocol.VERIFY,
        protocol.TEST,
        protocol.PLAN,
        "scripts/crossed_pair_plan.py",
        *protocol.build_plan()["input_sha256"],
        *(s["path"] for s in config["protected_artifacts"]),
        *HISTORICAL_BYTE_GUARDS,
    ]
    for path in paths:
        require(base.git(ROOT, "ls-files", "--", path), f"untracked source/input {path}")
        require(
            not base.git(ROOT, "status", "--porcelain", "--", path), f"dirty source/input {path}"
        )
        result[path] = protocol.sha((ROOT / path).read_bytes())
    for spec in config["protected_artifacts"]:
        require(result[spec["path"]] == spec["sha256"], "protected byte-hash provenance changed")
    require(
        all(result[path] == digest for path, digest in HISTORICAL_BYTE_GUARDS.items()),
        "old COMPLY candidate/failure and accuracy smoke byte hashes changed",
    )
    return result


engine.source_identity, engine.summarize = source_identity, summarize
evaluate, freeze, require_freeze = engine.evaluate, engine.freeze, engine.require_freeze


def untouched_namespace(worker_entry=False):
    expected = {"preregistration.json"}
    if worker_entry:
        expected |= {"RUN_STARTED.json", "worker.log"}
    require(
        OUTPUT.is_dir() and {p.name for p in OUTPUT.iterdir()} == expected,
        "untouched namespace; no stale worker, runtime arrays, retry or extra artifacts",
    )


def preflight(worker_entry=False):
    """Read-only ZERO-model preflight, stdout only; preserves preregistration-only HEAD."""
    from scripts.verify_shared_comply_crossed import verify_plan

    record = require_freeze()
    path = protocol.OUTPUT + "/preregistration.json"
    require(
        base.git(ROOT, "show", "--pretty=", "--name-only", "HEAD").splitlines() == [path]
        and base.git(ROOT, "rev-parse", "HEAD^") == record["source_commit"]
        and not base.git(ROOT, "status", "--porcelain", "--", path),
        "clean preregistration-only HEAD immediately after source commit",
    )
    untouched_namespace(worker_entry=worker_entry)
    checked = verify_plan(record["plan"])
    return {
        "status": "MODEL_FREE_PREPARATION_PREFLIGHT_ONLY",
        **checked,
        "storage": protocol.storage_preflight(ROOT, record["plan"]["config"]),
        "preregistration_sha256": protocol.sha((OUTPUT / "preregistration.json").read_bytes()),
        "source_files_checked": len(record["source_sha256"]),
        "tokenizer_loads": 0,
        "model_loads": 0,
        "real_forwards": 0,
        "real_derivatives": 0,
        "scientific_success": False,
        "run_authorized": False,
    }


def require_authorization():
    value = json.loads(os.environ.get(protocol.AUTH_KEY, "null"))
    require(
        value
        == {
            "preregistration_sha256": protocol.sha((OUTPUT / "preregistration.json").read_bytes()),
            "scope": protocol.AUTH_SCOPE,
            "authorized_by": "supervisor",
        },
        "separate supervisor authorization bound to this lock required",
    )


def worker():
    require_authorization()
    preflight(worker_entry=True)
    started = protocol.read(OUTPUT / "RUN_STARTED.json")
    usage = started.get("usage_preflight", {})
    values = [started.get(k) for k in ("started_monotonic", "deadline_monotonic")]
    values += [usage.get(k) for k in ("standard_used_percent", "checked_at_unix")]
    require(
        all(type(v) in (int, float) and math.isfinite(v) for v in values),
        "finite external worker budget/usage envelope before claim",
    )
    require(
        started["command"] == [sys.executable, "-u", str(ROOT / protocol.SCRIPT), "_worker"]
        and started["forward_ceiling"] == 144
        and started["derivative_ceiling"] == 64
        and started["timeout_seconds"] == 900
        and started["deadline_monotonic"] == started["started_monotonic"] + 900
        and started["started_monotonic"] <= time.monotonic() < started["deadline_monotonic"]
        and 0 <= usage["standard_used_percent"] < 90
        and 0 <= time.time() - usage["checked_at_unix"] <= 60,
        "exact external deadline/command/fresh usage before worker claim",
    )
    return engine.worker()


def run():
    require_authorization()
    preflight()
    return engine.run()


if __name__ == "__main__":
    commands = {"freeze": freeze, "preflight": preflight, "run": run, "_worker": worker}
    require(
        len(sys.argv) == 2 and sys.argv[1] in commands,
        "Use freeze/preflight; run requires new authorization",
    )
    result = commands[sys.argv[1]]()
    print(json.dumps(result, indent=2, allow_nan=False))
    if sys.argv[1] == "run" and result["status"] != "complete_valid":
        raise SystemExit(1)
