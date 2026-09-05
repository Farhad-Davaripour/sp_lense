"""Thin transfer wrapper; original four fresh baselines, unchanged frozen .20."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import frozen_endpoint020_crossed_f04_plan as protocol

job = protocol.isolate(
    "_f04_endpoint020_execution_core", "scripts/frozen_guarded_preserve_crossed.py"
)
job.protocol = protocol
for _name in ("SCRIPT", "VERIFY", "TEST", "DOC", "PLAN", "OUTPUT"):
    setattr(job, _name, getattr(protocol, _name))
CORE_EVALUATE, CORE_FREEZE_CHECK = job.evaluate, job.require_freeze


def source_identity():
    result = job.engine.source_identity()
    paths = [
        protocol.CONFIG,
        protocol.DOC,
        protocol.SCRIPT,
        protocol.VERIFY,
        protocol.TEST,
        protocol.PLAN,
    ]
    plan = protocol.build_plan()
    paths.extend(plan["input_sha256"])
    paths.extend(plan["config"][key]["path"] for key in ("template", "dataset", "manifest"))
    for path in paths:
        protocol.require(
            job.base.git(job.ROOT, "ls-files", "--", path), "untracked source/input " + path
        )
        protocol.require(
            not job.base.git(job.ROOT, "status", "--porcelain", "--", path),
            "dirty source/input " + path,
        )
        result[path] = protocol.sha((job.ROOT / path).read_bytes())
    return result


def evaluate(plan, backend, vectors, ledger, output):
    protocol.validate_scope(plan)
    return CORE_EVALUATE(plan, backend, vectors, ledger, output)


def require_freeze():
    from scripts.verify_frozen_endpoint020_crossed_f04 import (
        independent_condition,
        verify_selection,
    )

    record = CORE_FREEZE_CHECK()
    independent_condition(record["plan"])
    verify_selection(record["plan"])
    return record


job.source_identity = source_identity
job.evaluate = evaluate
job.require_freeze = require_freeze

if __name__ == "__main__":
    if sys.argv[1:] == ["_worker"]:
        job.worker()
    elif sys.argv[1:] in (["freeze"], ["prelaunch"], ["run"]):
        stage = sys.argv[1]
        result = getattr(job, stage)()
        print(json.dumps(result, indent=2))
        if stage != "freeze" and result["status"] not in ("complete_valid", "passed"):
            raise SystemExit(1)
    else:
        raise SystemExit("Use freeze, prelaunch or run; no selection/condition overrides.")
