"""Exactly one declared synthetic1024/12 timing-size smoke; no model or native data."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import soft_drift_qp_solver as solver
from scripts import verify_soft_drift_qp as checker

REPORT = ROOT / "docs/soft_drift_qp_native_smoke.json"


def main():
    started = time.perf_counter()
    if REPORT.exists():
        raise ValueError("single exclusive synthetic smoke; no retry")
    A = [
        [(-1.0 if (i & j).bit_count() % 2 else 1.0) / 32.0 for j in range(1024)] for i in range(12)
    ]
    b = [(20 + i) / 200.0 for i in range(12)]
    c = [(-1.0 if p % 2 else 1.0) * (p + 1) / 100.0 for p in range(6)]
    solved = solver.solve(A, b, c)
    verified = (
        checker.verify(A, b, c, solved["solution"])
        if solved["solution"] is not None
        else {"status": "NUMERICALLY_UNRESOLVED", "accepted": False, "reason": "NO_SOLVER_POINT"}
    )
    solver_bytes = len(json.dumps(solved, allow_nan=False).encode())
    checker_bytes = len(json.dumps(verified, allow_nan=False).encode())
    size_pass = solver_bytes <= 131072 and checker_bytes <= 32768
    modules = sorted({"torch", "transformers", "transformer_lens", "tokenizers"} & set(sys.modules))
    passed = (
        solved["status"] == "KKT_ESTIMATE_ONLY"
        and verified["accepted"] is True
        and size_pass
        and not modules
    )
    passed = (
        passed and solved["masks_visited"] == 4096 and solved["solution"]["active_mask"] == 4095
    )
    compact = {k: v for k, v in solved.items() if k not in {"solution", "mask_trace"}}
    compact["mask_trace_sha256"] = hashlib.sha256(solved["mask_trace"].encode()).hexdigest()
    if solved["solution"]:
        compact["active_mask"] = solved["solution"]["active_mask"]
        compact["solution_sha256"] = hashlib.sha256(
            json.dumps(solved["solution"], sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
    sources = (
        "docs/SOFT_DRIFT_QP_NUMERIC_POLICY.md",
        "scripts/soft_drift_qp_solver.py",
        "scripts/verify_soft_drift_qp.py",
        "scripts/soft_drift_qp_native_smoke.py",
        "tests/test_soft_drift_qp.py",
    )
    report = {
        "status": "SYNTHETIC_NATIVE_SMOKE_PASSED"
        if passed
        else "SYNTHETIC_NATIVE_SMOKE_UNRESOLVED",
        "fixture": "fixed12 dense Walsh rows /1024 columns; b=(20+i)/200,c=(-1)^p*(p+1)/100",
        "dimension": 1024,
        "constraints": 12,
        "attempts": 1,
        "elapsed_seconds_before_persistence": time.perf_counter() - started,
        "solver_serialized_bytes": solver_bytes,
        "checker_serialized_bytes": checker_bytes,
        "size_pass": size_pass,
        "solver": compact,
        "checker": verified,
        "model_loads": 0,
        "tokenizer_loads": 0,
        "real_forwards": 0,
        "real_derivatives": 0,
        "ml_modules": modules,
        "production_integration": False,
        "real_model_performance_or_efficacy_inference": False,
        "full_pipeline_audit_size_certified": False,
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
    }
    raw = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    if len(raw) > 16384:
        raise ValueError("compact smoke report cap; no expansion")
    with REPORT.open("xb") as stream:
        stream.write(raw)
    print(
        json.dumps(
            {
                "status": report["status"],
                "elapsed_seconds": report["elapsed_seconds_before_persistence"],
                "report_bytes": len(raw),
                "solver_bytes": solver_bytes,
                "checker_bytes": checker_bytes,
            }
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("fixed single smoke only")
    raise SystemExit(main())
