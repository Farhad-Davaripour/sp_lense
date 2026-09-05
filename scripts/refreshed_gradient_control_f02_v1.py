"""One f02/v1 second-family replication, reusing the immutable v1 execution engine."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import refreshed_gradient_control as v1
from scripts import refreshed_gradient_control_f02_inputs as inputs

base, require = v1.base, v1.require
SCRIPT = "scripts/refreshed_gradient_control_f02_v1.py"
VERIFY = "scripts/verify_refreshed_gradient_control_f02_v1.py"
TEST = "tests/test_refreshed_gradient_control_f02_v1.py"
PROTOCOL = "docs/REFRESHED_GRADIENT_CONTROL_F02_V1_PROTOCOL.md"
OUTPUT = inputs.OUTPUT
INPUTS = "scripts/refreshed_gradient_control_f02_inputs.py"


build_plan = inputs.build_plan


def source_identity():
    result = v1.source_identity()
    for path in (SCRIPT, VERIFY, TEST, PROTOCOL, INPUTS, inputs.TEMPLATE):
        require(base.git(ROOT, "ls-files", "--", path), f"untracked source {path}")
        require(not base.git(ROOT, "status", "--porcelain", "--", path), f"dirty source {path}")
        result[path] = base.sha((ROOT / path).read_bytes())
    return result


def freeze():
    record = {
        "plan": build_plan(),
        "source_commit": base.git(ROOT, "rev-parse", "HEAD"),
        "source_sha256": source_identity(),
        "environment": base.environment(),
    }
    output = ROOT / OUTPUT
    output.mkdir(parents=True, exist_ok=False)
    base.write_new(output / "preregistration.json", record)
    return {
        "source_commit": record["source_commit"],
        "forward_ceiling": 30,
        "derivative_ceiling": 8,
    }


def require_freeze():
    record = base.read_json(ROOT / OUTPUT / "preregistration.json")
    require(
        record["plan"] == build_plan() and record["source_sha256"] == source_identity(),
        "frozen plan/source changed",
    )
    require(record["environment"] == base.environment(), "environment changed")
    return record


def worker():
    output = ROOT / OUTPUT
    base.write_new(output / "WORKER_CLAIM.json", {"pid": os.getpid()})
    try:
        record, started = require_freeze(), base.read_json(output / "RUN_STARTED.json")
        require(time.monotonic() < started["deadline_monotonic"], "deadline before load")
        backend, _unused = base.load_backend(record["plan"])
        base.write_new(
            output / "runtime.json",
            {
                **backend.metadata(),
                **base.environment(),
                "logits_encoding": "zlib little-endian float32",
                "conditional_forward_ceiling": 30,
                "derivative_ceiling": 8,
            },
        )
        ledger = v1.ForwardLedger(output, record["plan"]["cells"], started["deadline_monotonic"])
        derivative = v1.DerivativeLedger(
            output, record["plan"]["derivative_cells"], started["deadline_monotonic"]
        )
        rows, requests = v1.evaluate(record["plan"], backend, ledger, derivative, output)
        base.write_new(output / "analysis.json", v1.summarize(rows, requests))
    except BaseException as error:
        invalid = {
            "classification": "INCONCLUSIVE",
            "reason": str(error),
            "error_type": type(error).__name__,
            "retries_allowed": False,
        }
        if isinstance(error, v1.EligibilityError):
            base.write_new(output / "ELIGIBILITY_FAILURE.json", invalid)
        base.write_new(output / "INVALID.json", invalid)
        raise


def run():
    record = require_freeze()
    path = f"{OUTPUT}/preregistration.json"
    require(
        base.git(ROOT, "show", "--pretty=", "--name-only", "HEAD").splitlines() == [path]
        and base.git(ROOT, "rev-parse", "HEAD^") == record["source_commit"]
        and not base.git(ROOT, "status", "--porcelain", "--", path),
        "must directly follow clean preregistration-only commit",
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
    return v1.supervise([sys.executable, "-u", str(ROOT / SCRIPT), "_worker"], ROOT / OUTPUT, usage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "_worker"))
    stage = parser.parse_args().stage
    if stage == "_worker":
        worker()
    else:
        result = {"freeze": freeze, "run": run}[stage]()
        print(json.dumps(result, indent=2))
        if stage == "run" and result["status"] != "complete_valid":
            raise SystemExit(1)
