"""Fixed, externally timed, model-free affordability check; JSON stdout only.

Declaration below is fixed before execution. Synthetic systems are not model
measurements or scientific evidence. No files, evidence namespace, candidate,
model/tokenizer/gradient calls, or runtime arrays are produced by this script.
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import shared_comply_twelve_row_solver as solver

DECLARATION = {
    "cases": [
        "positive_orthogonal",
        "negative_rhs_zero_optimum",
        "rank_deficient_redundant",
        "opposing_infeasible",
    ],
    "constraints": 12,
    "native_dimensions": 1024,
    "masks_per_completed_case": 4096,
    "per_case_wall_limit_seconds": 20,
    "whole_benchmark_wall_limit_seconds": 90,
    "maximum_case_attempts": 1,
    "model_loads": 0,
    "tokenizer_loads": 0,
    "forwards": 0,
    "derivatives": 0,
}


def problem(name):
    if name not in DECLARATION["cases"]:
        raise ValueError("fixed benchmark cases only")
    A = [[float(i == j) for j in range(1024)] for i in range(12)]
    b = [(i + 1) / 100 for i in range(12)]
    if name == "negative_rhs_zero_optimum":
        b = [-x for x in b]
    elif name == "rank_deficient_redundant":
        A = [[1.0] + [0.0] * 1023 for _ in range(12)]
        b = [0.1] * 12
    elif name == "opposing_infeasible":
        A = [[1.0 if i < 6 else -1.0] + [0.0] * 1023 for i in range(12)]
        b = [0.1] * 12
    scale = max(1.0, *map(abs, b), *(solver.norm(a) for a in A))
    config = {
        "rank_relative_pivot_floor": 1e-12,
        "primal_absolute_tolerance": 1e-9 * scale,
        "kkt_absolute_tolerance": 1e-8 * scale * scale,
    }
    return A, b, config


def journal_bound():
    """Schema-based bound for the existing default compact JSONL serializer.

    Every float is a finite binary64 value. CPython's shortest-roundtrip repr
    needs at most 24 ASCII bytes; 32 allows further slack. Each mask contains
    <=12 pivots, <=12 primal residuals and exactly eight scalar metrics when
    metrics exist. Active integers are in 0..11 and mask is in 0..4095.
    JSON key/status/punctuation overhead is accounted from a maximally populated
    schema using integer zero sentinels, then enlarged per numeric field.
    No native-dimension vector is stored per mask. Selected solution and outer
    update data must be budgeted separately by the preparation protocol.
    """
    metrics = {
        "norm": 0,
        "primal_residuals": [0] * 12,
        "primal_violation": 0,
        "minimum_multiplier": 0,
        "stationarity_max": 0,
        "complementarity_max": 0,
        "primal_objective": 0,
        "dual_objective": 0,
        "gap": 0,
    }
    statuses = [
        "rank_deficient_or_near_dependent_skipped",
        "negative_or_nonfinite_multiplier_independent_set",
        "kkt_valid",
        "kkt_rejected",
    ]
    entry = {
        "mask": 4095,
        "active": list(range(12)),
        "pivots": [0] * 12,
        "status": max(statuses, key=len),
        "metrics": metrics,
    }
    base_bytes = len(json.dumps(entry, sort_keys=True, allow_nan=False).encode("ascii"))
    float_fields = 12 + 12 + 8
    maximum_entry_bytes = base_bytes + float_fields * (32 - 1)
    maximum_journal_bytes = 2 + 4096 * maximum_entry_bytes + 4095 * 2
    return {
        "serializer": "json.dumps(sort_keys=True), default separators, ensure_ascii=True; JSONL",
        "finite_binary64_bytes_reserved": 32,
        "maximum_float_fields_per_mask": float_fields,
        "maximally_populated_zero_schema_bytes": base_bytes,
        "maximum_mask_entry_bytes": maximum_entry_bytes,
        "maximum_4096_mask_journal_bytes": maximum_journal_bytes,
        "maximum_eight_update_mask_journals_bytes": 8 * maximum_journal_bytes,
        "outer_update_and_selected_solution_included": False,
        "vectors_per_mask": 0,
    }


def run_case(name):
    A, b, config = problem(name)
    started = time.monotonic()
    result = solver.solve(A, b, config)
    elapsed = time.monotonic() - started
    logs = result["active_sets"]
    if [row["mask"] for row in logs] != list(range(4096)):
        raise ValueError("incomplete mask accounting")
    if any(any(key in row for key in ("vector", "multipliers")) for row in logs):
        raise ValueError("unexpected per-mask native vectors")
    encoded = json.dumps(logs, sort_keys=True, allow_nan=False).encode("ascii")
    if len(encoded) > journal_bound()["maximum_4096_mask_journal_bytes"]:
        raise ValueError("mask journal schema bound exceeded")
    selected = result["solution"]
    imported = sorted(
        name for name in sys.modules if name.split(".")[0] in {"torch", "transformers"}
    )
    if imported:
        raise ValueError("unexpected model library import")
    return {
        "case": name,
        "status": "COMPLETED",
        "solver_status": result["status"],
        "solver_seconds": elapsed,
        "mask_count": len(logs),
        "mask_status_counts": dict(
            sorted(collections.Counter(row["status"] for row in logs).items())
        ),
        "active_mask": selected["active_mask"] if selected else None,
        "selected_norm": selected["metrics"]["norm"] if selected else None,
        "mask_journal_serialized_bytes": len(encoded),
        "selected_kkt_metrics": selected["metrics"] if selected else None,
        "infeasibility_certified": False,
        "model_library_imports": imported,
        "forwards": 0,
        "derivatives": 0,
    }


def benchmark():
    start = time.monotonic()
    deadline = start + DECLARATION["whole_benchmark_wall_limit_seconds"]
    rows = []
    for name in DECLARATION["cases"]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            rows.append({"case": name, "status": "NOT_ATTEMPTED_WHOLE_LIMIT"})
            continue
        timeout = min(DECLARATION["per_case_wall_limit_seconds"], remaining)
        case_start = time.monotonic()
        try:
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--case", name],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if completed.returncode:
                row = {
                    "case": name,
                    "status": "FAULT",
                    "returncode": completed.returncode,
                    "stderr": completed.stderr,
                    "stdout": completed.stdout,
                }
            else:
                row = json.loads(completed.stdout)
        except subprocess.TimeoutExpired as error:
            row = {
                "case": name,
                "status": "TIMEOUT",
                "timeout_seconds": timeout,
                "partial_stdout": (error.stdout or b"").decode("utf-8", errors="replace"),
                "partial_stderr": (error.stderr or b"").decode("utf-8", errors="replace"),
                "retry_allowed": False,
            }
        row["external_seconds"] = time.monotonic() - case_start
        rows.append(row)
    elapsed = time.monotonic() - start
    affordable = (
        len(rows) == 4 and all(row["status"] == "COMPLETED" for row in rows) and elapsed <= 90
    )
    return {
        "declaration": DECLARATION,
        "status": "FIXED_SYNTHETIC_BENCHMARK_COMPLETED"
        if affordable
        else "AFFORDABILITY_UNVERIFIED",
        "cases": rows,
        "whole_benchmark_seconds": elapsed,
        "journal_bound": journal_bound(),
        "future_216_forward_runtime_prediction": False,
        "scientific_evidence": False,
        "retries_allowed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=DECLARATION["cases"])
    args = parser.parse_args()
    value = run_case(args.case) if args.case else benchmark()
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False), flush=True)
    if value["status"] in {"AFFORDABILITY_UNVERIFIED", "FAULT", "TIMEOUT"}:
        raise SystemExit(1)
