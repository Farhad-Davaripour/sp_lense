"""Single-use clean01 entry; unchanged experiment engine with extra prelaunch gate."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import iterative_guarded_preserve_clean01_plan as protocol

engine = protocol.isolated_engine("runner")
OUTPUT, require = engine.OUTPUT, protocol.require


def no_resume(allowed):
    require(
        {p.name for p in OUTPUT.iterdir()} == set(allowed),
        "fresh namespace only; no resume, predecessor state or extra artifact",
    )


def lock_commit(record):
    path = protocol.OUTPUT + "/preregistration.json"
    require(
        engine.base.git(ROOT, "show", "--pretty=", "--name-only", "HEAD").splitlines() == [path]
        and engine.base.git(ROOT, "rev-parse", "HEAD^") == record["source_commit"],
        "new preregistration-only commit required",
    )
    return engine.base.git(ROOT, "rev-parse", "HEAD")


def prelaunch():
    no_resume({"preregistration.json"})
    started = time.monotonic()
    claim = {
        "pid": os.getpid(),
        "python_executable": sys.executable,
        "working_directory": str(Path.cwd()),
        "started_monotonic": started,
    }
    protocol.io.write_new(OUTPUT / "PRELAUNCH_CLAIM.json", claim)
    record = {
        **claim,
        "preregistration_sha256": protocol.sha((OUTPUT / "preregistration.json").read_bytes()),
        "claim_sha256": protocol.sha((OUTPUT / "PRELAUNCH_CLAIM.json").read_bytes()),
        "git_checks": [],
        "source_identity_passed": False,
        "model_calls": 0,
        "retries_allowed": False,
        "status": "INCONCLUSIVE",
    }
    try:
        require(Path.cwd() == ROOT, "prelaunch must use the same workspace")
        for args in protocol.prelaunch_commands():
            result = subprocess.run(
                ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False
            )
            record["git_checks"].append(
                {
                    "args": args,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
            require(result.returncode == 0, "prelaunch Git read failed; stop without retry")
            if args[0] == "cat-file":
                require(result.stdout.strip() == "tree", "exact predecessor object is tree")
            if args[0] in ("status", "diff"):
                require(result.stdout.strip() == "", "prelaunch source must be clean")
        frozen = engine.require_freeze()
        record.update(
            source_identity_passed=True,
            environment=engine.base.environment(),
            source_commit=frozen["source_commit"],
            lock_commit=lock_commit(frozen),
            status="passed",
        )
    except Exception as error:  # noqa: BLE001 - one durable prelaunch outcome, no retry.
        record["fault"] = type(error).__name__ + ": " + str(error)
    record["finished_monotonic"] = time.monotonic()
    protocol.io.write_new(OUTPUT / "PRELAUNCH.json", record)
    return record


def run():
    no_resume({"preregistration.json", "PRELAUNCH_CLAIM.json", "PRELAUNCH.json"})
    protocol.check_prelaunch(OUTPUT)
    return engine.run()


def worker():
    no_resume(
        {
            "preregistration.json",
            "PRELAUNCH_CLAIM.json",
            "PRELAUNCH.json",
            "RUN_STARTED.json",
            "worker.log",
        }
    )
    protocol.check_prelaunch(OUTPUT)
    return engine.worker()


if __name__ == "__main__":
    if sys.argv[1:] == ["_worker"]:
        worker()
    elif sys.argv[1:] in (["freeze"], ["prelaunch"], ["run"]):
        action = sys.argv[1]
        result = (
            engine.freeze()
            if action == "freeze"
            else prelaunch()
            if action == "prelaunch"
            else run()
        )
        print(json.dumps(result, indent=2))
        if action != "freeze" and result["status"] not in ("passed", "complete_valid"):
            raise SystemExit(1)
    else:
        raise SystemExit("Use freeze, prelaunch or run; no adjustable recipe or retry.")
