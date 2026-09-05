"""Single alpha0.20 wrapper around the immutable22/0 frozen-arrow runner."""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import frozen_arrow_alpha020_plan as protocol
from scripts import frozen_arrow_transfer as previous

base, require = previous.base, protocol.require
SCRIPT = "scripts/frozen_arrow_alpha020.py"
VERIFY = "scripts/verify_frozen_arrow_alpha020.py"
TEST = "tests/test_frozen_arrow_alpha020.py"
DOC = "docs/FROZEN_ARROW_ALPHA020_PROTOCOL.md"
PLAN = "scripts/frozen_arrow_alpha020_plan.py"
OUTPUT = protocol.OUTPUT
EligibilityError = previous.EligibilityError
summarize, supervise = previous.summarize, previous.supervise


@contextmanager
def amplitude():
    require(previous.ALPHA == 0.05, "original amplitude must remain .05 outside the adapter")
    original = previous.ALPHA
    previous.ALPHA = protocol.ALPHA
    try:
        yield
    finally:
        previous.ALPHA = original


def evaluate(plan, backend, vector, ledger, output):
    require(plan["intervention"]["alpha"] == 0.20, "only the separately locked .20 amplitude")
    with amplitude():
        return previous.evaluate(plan, backend, vector, ledger, output)


def source_identity():
    result = previous.source_identity()
    for path in (
        SCRIPT,
        VERIFY,
        TEST,
        DOC,
        PLAN,
        protocol.CANDIDATE,
        protocol.TEMPLATE,
        protocol.COMPARISON,
    ):
        require(base.git(ROOT, "ls-files", "--", path), f"untracked source/input {path}")
        require(
            not base.git(ROOT, "status", "--porcelain", "--", path), f"dirty source/input {path}"
        )
        result[path] = base.sha((ROOT / path).read_bytes())
    return result


def freeze():
    record = {
        "plan": protocol.build_plan(),
        "source_commit": base.git(ROOT, "rev-parse", "HEAD"),
        "source_sha256": source_identity(),
        "environment": base.environment(),
    }
    output = ROOT / OUTPUT
    output.mkdir(parents=True, exist_ok=False)
    base.write_new(output / "preregistration.json", record)
    return {"source_commit": record["source_commit"], "forwards": 22, "derivatives": 0}


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
        vector = protocol.candidate()["vector"]
        backend, _unused = base.load_backend(record["plan"])
        base.write_new(
            output / "runtime.json",
            {
                **backend.metadata(),
                **base.environment(),
                "logits_encoding": "zlib little-endian float32",
                "candidate_vector_sha256": protocol.VECTOR_SHA256,
                "forward_ceiling": 22,
                "derivative_ceiling": 0,
            },
        )
        ledger = base.Ledger(output, record["plan"]["cells"], started["deadline_monotonic"])
        rows = evaluate(record["plan"], backend, vector, ledger, output)
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
