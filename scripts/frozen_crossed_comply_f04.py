"""Minimal f04 binding of the frozen-C03 12-cell runtime; no numerical edits."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import frozen_crossed_comply_f04_plan as protocol

parent = protocol.isolate(
    "scripts._frozen_C_f04_runtime_parent", "scripts/frozen_crossed_comply_f03.py"
)
parent.protocol, parent.OUTPUT = protocol, ROOT / protocol.OUTPUT
engine = parent.engine
engine.protocol = protocol
for key in ("SCRIPT", "VERIFY", "TEST", "DOC", "PLAN", "OUTPUT"):
    setattr(engine, key, getattr(protocol, key))
OUTPUT, require, base = ROOT / protocol.OUTPUT, protocol.require, parent.base
evaluate, make_delta, validate_vectors = parent.evaluate, parent.make_delta, parent.validate_vectors
DerivativeGuard, EligibilityError = parent.DerivativeGuard, parent.EligibilityError
norm, EPS = protocol.norm, protocol.EPS
freeze, require_freeze, supervise = parent.freeze, parent.require_freeze, parent.supervise
_parent_summary = parent.summarize


def summarize(rows):
    result = _parent_summary(rows)
    result["status"] = result["status"].replace("FROZEN_COMPLY_F03_", "FROZEN_COMPLY_F04_")
    return result


def source_identity():
    paths = [
        *protocol.build_plan()["input_sha256"],
        *(getattr(protocol, k) for k in ("CONFIG", "DOC", "SCRIPT", "VERIFY", "TEST", "PLAN")),
        "scripts/frozen_crossed_comply_f03.py",
        "scripts/frozen_crossed_comply_f03_plan.py",
        "scripts/verify_frozen_crossed_comply_f03.py",
        "scripts/frozen_endpoint020_crossed_f04_plan.py",
    ]
    result = {}
    for path in dict.fromkeys(paths):
        require(
            base.git(ROOT, "ls-files", "--", path)
            and not base.git(ROOT, "status", "--porcelain", "--", path),
            "clean tracked source/input " + path,
        )
        result[path] = protocol.sha((ROOT / path).read_bytes())
    return result


engine.summarize, engine.source_identity = summarize, source_identity
preflight = protocol.adapt(
    parent,
    "preflight",
    replacements=[
        (
            "from scripts.verify_frozen_crossed_comply_f03 import verify_renderings",
            "from scripts.verify_frozen_crossed_comply_f04 import verify_renderings",
        ),
    ],
)
worker = parent.worker


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


def run():
    preflight()
    usage = checked_usage(json.loads(os.environ.get("SP_LENSE_USAGE_PREFLIGHT", "null")))
    return supervise([sys.executable, "-u", str(ROOT / protocol.SCRIPT), "_worker"], OUTPUT, usage)


if __name__ == "__main__":
    commands = {"freeze": freeze, "preflight": preflight, "run": run, "_worker": worker}
    require(len(sys.argv) == 2 and sys.argv[1] in commands, "Use freeze/preflight/run only")
    result = commands[sys.argv[1]]()
    print(json.dumps(result, indent=2, allow_nan=False))
    if sys.argv[1] == "run" and result["status"] != "complete_valid":
        raise SystemExit(1)
