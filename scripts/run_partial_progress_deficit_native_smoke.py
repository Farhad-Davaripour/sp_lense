"""One preregistered native synthetic smoke; external parent enforces hard10s.

This worker never retries, writes files, loads a model, or reads real gradients.
"""

import hashlib
import json
import sys
from pathlib import Path

from scripts.partial_progress_deficit_solver import solve

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "docs/partial_progress_deficit_prototype_preregistration.json"


def emit(kind, payload):
    print(json.dumps({"kind": kind, **payload}, sort_keys=True, allow_nan=False), flush=True)


def main():
    lock = json.loads(LOCK.read_bytes())
    assert lock["native_smoke_executions"] == 1 and lock["native_hard_seconds"] == 10
    for path, expected in lock["source_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    spec = json.loads(
        (ROOT / "configs/partial_progress_deficit_synthetic_fixtures.json").read_bytes()
    )
    native = spec["native_smoke"]
    raw = native["canonical_problem_json"]
    assert hashlib.sha256(raw.encode()).hexdigest() == native["problem_sha256"]
    assert native["problem_sha256"] == lock["native_problem_sha256"]
    problem = json.loads(raw)
    assert len(problem["A"]) == 12 and len(problem["w"]) == 1024
    for i, row in enumerate(problem["A"]):
        assert len(row) == 1024
        for j, value in enumerate(row):
            expected = (-1 if i % 2 else 1) * (1 + (((j + 1) * (i + 3)) % 17) / 16)
            expected += (2 * i - 11) / 2048
            assert value == expected
    emit(
        "authenticated_start",
        {"problem_sha256": native["problem_sha256"], "dimension": 1024, "rows": 12},
    )

    def observer(snapshot):
        cert = snapshot["certificate"]
        if (
            snapshot["certificates_checked"] == 1
            or snapshot["certificates_checked"] % 10 == 0
            or cert["status"] != "NOT_CERTIFIED"
        ):
            emit(
                "checkpoint",
                {
                    "iterations": snapshot["iterations"],
                    "certificates_checked": snapshot["certificates_checked"],
                    "fixed_M": snapshot["fixed_M"],
                    "status": cert["status"],
                    "reason": cert["reason"],
                    "geometry_valid": cert.get("geometry_valid"),
                    "gap_bound": cert.get("gap_bound"),
                    "gain": cert.get("gain"),
                    "epsilon": cert.get("epsilon"),
                    "projection_case_counts": snapshot["projection_case_counts"],
                },
            )

    result = solve(problem, native["problem_sha256"], observer=observer)
    assert not any(
        name.split(".")[0] in {"torch", "transformers", "transformer_lens", "numpy", "scipy"}
        for name in sys.modules
    )
    emit("final", {"result": result, "model_loads": 0, "real_forwards": 0, "real_derivatives": 0})


if __name__ == "__main__":
    main()
