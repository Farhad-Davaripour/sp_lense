"""Independent clean01 identity/prelaunch wrapper; unchanged raw audit engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import iterative_guarded_preserve_clean01_plan as protocol

engine = protocol.isolated_engine("audit")
OUTPUT = engine.OUTPUT


def verify():
    pre = protocol.check_prelaunch(OUTPUT, require_passed=False)
    if pre["status"] != "passed":
        # Prelaunch failure still authenticates the new plan and frozen source/input bytes.
        lock = protocol.read(OUTPUT / "preregistration.json")
        protocol.require(lock["plan"] == protocol.build_plan(), "exact new frozen plan")
        for path, digest in lock["source_sha256"].items():
            protocol.require(
                protocol.sha((ROOT / path).read_bytes()) == digest, "frozen source hash"
            )
        protocol.require(
            not (OUTPUT / "WORKER_CLAIM.json").exists()
            and not (OUTPUT / "RUN_STARTED.json").exists(),
            "no worker after failed prelaunch",
        )
        return {
            "status": "INCONCLUSIVE",
            "prelaunch": pre,
            "model_calls": 0,
            "scientific_observations": False,
            "retries_allowed": False,
        }
    started = protocol.read(OUTPUT / "RUN_STARTED.json")
    protocol.require(
        pre["finished_monotonic"] <= started["started_monotonic"],
        "prelaunch completed before fresh runtime deadline/worker",
    )
    result = engine.verify()
    result["prelaunch_sha256"] = protocol.sha((OUTPUT / "PRELAUNCH.json").read_bytes())
    result["predecessor_status"] = "TECHNICAL_INCONCLUSIVE_BEFORE_LOAD"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    try:
        result = verify()
    except Exception as error:  # noqa: BLE001 - preserve the only failed independent audit.
        result = {
            "status": "INCONCLUSIVE",
            "fault": type(error).__name__ + ": " + str(error),
            "retries_allowed": False,
        }
        protocol.io.write_new(OUTPUT / "VERIFICATION_FAILURE.json", result)
    if args.report:
        protocol.io.write_new(OUTPUT / "verification.json", result)
        if result["status"] != "INCONCLUSIVE":
            engine.freeze_verified_candidate(OUTPUT, result)
        with (OUTPUT / "PILOT_REPORT.md").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(engine.report(result))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("summary", "construction_stages", "optimizer_checks")
            },
            indent=2,
        )
    )
    if result["status"] == "INCONCLUSIVE":
        raise SystemExit(1)
