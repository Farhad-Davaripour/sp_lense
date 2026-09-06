"""Exactly one NEW frozen synthetic solve; externally enforce hard ten seconds.

Invoke as a module from the repository. Stdout is provisional until the parent
observes normal process completion within ten seconds. No old fixture is read.
"""

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "docs/certified_descent_dyadic_prototype_preregistration.json"
FIXTURE = "configs/certified_descent_dyadic_synthetic_fixtures.json"
REQUIRED_SOURCES = {
    FIXTURE,
    "docs/CERTIFIED_DESCENT_DYADIC_DESIGN.md",
    "docs/CERTIFIED_DESCENT_DYADIC_PROTOTYPE_PROTOCOL.md",
    "scripts/certified_descent_dyadic_solver.py",
    "scripts/verify_certified_descent_dyadic.py",
    "scripts/run_certified_descent_dyadic_native_smoke.py",
    "scripts/partial_progress_deficit_solver.py",
    "scripts/verify_partial_progress_deficit.py",
    "tests/test_certified_descent_dyadic_prototype.py",
}


def main():
    started = time.monotonic()
    deadline = started + 10.0

    def tick():
        if time.monotonic() >= deadline:
            raise TimeoutError("NATIVE_WORKER_DEADLINE")

    def emit(value):
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        tick()
        print(encoded, flush=True)
        tick()

    try:
        prereg = json.loads(LOCK.read_bytes())
        assert prereg["native_executions"] == 1 and prereg["native_hard_seconds"] == 10
        assert set(prereg["source_sha256"]) == REQUIRED_SOURCES
        for name, expected in prereg["source_sha256"].items():
            tick()
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
        spec = json.loads((ROOT / FIXTURE).read_bytes())
        native = spec["native_smoke"]
        raw = native["canonical_problem_json"]
        actual_sha = hashlib.sha256(raw.encode()).hexdigest()
        assert actual_sha == native["problem_sha256"] == prereg["native_problem_sha256"]
        problem = json.loads(raw)
        assert len(problem["A"]) == 12 and all(len(row) == 1024 for row in problem["A"])
        tick()
        from scripts.certified_descent_dyadic_solver import solve

        def observe(snapshot):
            emit(
                {"kind": "provisional_checkpoint", "external_completion_required": True, **snapshot}
            )

        result = solve(problem, actual_sha, observer=observe)
        tick()
        emit(
            {
                "kind": "completed_result",
                "external_completion_required": True,
                "worker_elapsed_before_encoding": time.monotonic() - started,
                "native_problem_sha256": actual_sha,
                "result": result,
            }
        )
    except Exception as error:  # noqa: BLE001 - incomplete work cannot issue a step.
        print(
            json.dumps(
                {
                    "kind": "worker_failure",
                    "status": "NO_CERTIFIED_STEP",
                    "w_next": None,
                    "reason": type(error).__name__ + ":" + str(error)[:192],
                },
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
