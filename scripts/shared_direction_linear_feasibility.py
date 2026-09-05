"""One four-constraint native-coordinate convex surrogate, with no model imports."""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import shared_direction_feasibility_io as io


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def norm(a):
    return math.sqrt(dot(a, a))


def system(records, margin):
    return [[norm(r["h0"]) * x for x in r["g"]] for r in records], [
        margin + abs(r["S"]) for r in records
    ]


def linear_solve(matrix, rhs, floor):
    n = len(rhs)
    if not n:
        return [], []
    scale = max(abs(x) for row in matrix for x in row)
    work = [list(row) + [b] for row, b in zip(matrix, rhs, strict=True)]
    pivots = []
    for k in range(n):
        pivot = max(range(k, n), key=lambda j: abs(work[j][k]))
        value = abs(work[pivot][k])
        pivots.append(value)
        if scale == 0 or value <= floor * scale:
            return None, pivots
        work[k], work[pivot] = work[pivot], work[k]
        for j in range(k + 1, n):
            factor = work[j][k] / work[k][k]
            for col in range(k + 1, n + 1):
                work[j][col] -= factor * work[k][col]
            work[j][k] = 0.0
    result = [0.0] * n
    for k in reversed(range(n)):
        result[k] = (
            work[k][n] - math.fsum(work[k][j] * result[j] for j in range(k + 1, n))
        ) / work[k][k]
    return result, pivots


def metrics(A, b, w, multipliers):
    at = [
        math.fsum(lam * row[j] for lam, row in zip(multipliers, A, strict=True))
        for j in range(len(w))
    ]
    residuals = [dot(row, w) - bi for row, bi in zip(A, b, strict=True)]
    primal = dot(w, w) / 2
    dual = dot(b, multipliers) - dot(at, at) / 2
    return {
        "norm": norm(w),
        "primal_residuals": residuals,
        "primal_violation": max(0.0, -min(residuals)),
        "minimum_multiplier": min(multipliers),
        "stationarity_max": max(abs(x - y) for x, y in zip(w, at, strict=True)),
        "complementarity_max": max(abs(x * y) for x, y in zip(multipliers, residuals, strict=True)),
        "primal_objective": primal,
        "dual_objective": dual,
        "gap": primal - dual,
    }


def kkt_valid(values, config):
    scalars = [v for v in values.values() if not isinstance(v, list)] + values["primal_residuals"]
    return (
        all(math.isfinite(x) for x in scalars)
        and values["minimum_multiplier"] >= 0
        and values["primal_violation"] <= config["primal_absolute_tolerance"]
        and max(values["stationarity_max"], values["complementarity_max"], abs(values["gap"]))
        <= config["kkt_absolute_tolerance"]
    )


def solve(A, b, config):
    io.require(
        0 < len(A) == len(b) <= 4
        and len(A[0]) > 0
        and all(len(row) == len(A[0]) and all(math.isfinite(x) for x in row) for row in A)
        and all(math.isfinite(x) for x in b),
        "finite small native system",
    )
    gram = [[dot(a, c) for c in A] for a in A]
    best, log = None, []
    for mask in range(1 << len(A)):
        active = [i for i in range(len(A)) if mask & (1 << i)]
        lam, pivots = linear_solve(
            [[gram[i][j] for j in active] for i in active],
            [b[i] for i in active],
            config["rank_relative_pivot_floor"],
        )
        entry = {"mask": mask, "active": active, "pivots": pivots}
        log.append(entry)
        if lam is None:
            entry["status"] = "rank_deficient_or_near_dependent_skipped"
            continue
        if not all(math.isfinite(x) and x >= 0 for x in lam):
            entry["status"] = "negative_or_nonfinite_multiplier_independent_set"
            continue
        multipliers = [lam[active.index(i)] if i in active else 0.0 for i in range(len(A))]
        vector = [
            math.fsum(l * a[j] for l, a in zip(multipliers, A, strict=True))
            for j in range(len(A[0]))
        ]
        numbers = metrics(A, b, vector, multipliers)
        entry.update(
            status="kkt_valid" if kkt_valid(numbers, config) else "kkt_rejected", metrics=numbers
        )
        if entry["status"] == "kkt_valid" and (
            best is None or numbers["primal_objective"] < best["metrics"]["primal_objective"]
        ):
            best = {
                "active_mask": mask,
                "active": active,
                "vector": vector,
                "multipliers": multipliers,
                "metrics": numbers,
            }
    return {
        "solution": best,
        "active_sets": log,
        "status": "KKT_ESTIMATE_ONLY" if best else "NUMERICALLY_UNRESOLVED",
    }


def row_table(records, vector, margin):
    table = []
    for r in records:
        hn, gn = norm(r["h0"]), norm(r["g"])
        bound = (margin + abs(r["S"])) / (hn * gn) if hn * gn else None
        effect = hn * dot(r["g"], vector) if vector is not None else None
        table.append(
            {
                k: r[k]
                for k in (
                    "dataset",
                    "order",
                    "prompt_id",
                    "baseline_cell_id",
                    "initial_gradient_cell_id",
                )
            }
        )
        table[-1].update(
            S=r["S"],
            h0_norm=hn,
            gradient_norm=gn,
            necessary_norm_bound=bound,
            predicted_preserve_margin=r["S"] + effect if effect is not None else None,
            predicted_comply_margin=-r["S"] + effect if effect is not None else None,
            residual=effect - (margin + abs(r["S"])) if effect is not None else None,
        )
    return table


def calculate(config, loader, freezer):
    io.require(
        config["construction"] == ["f01_v1", "f01_v2"]
        and config["descriptive_comparison"] == ["f02_v1"],
        "fixed construction/exposed split",
    )
    records = [r for key in config["construction"] for r in loader(key)]
    io.require(
        [(r["dataset"], r["order"]) for r in records]
        == [(key, order) for key in config["construction"] for order in config["order"]],
        "four construction rows",
    )
    A, b = system(records, config["margin"])
    result = solve(A, b, config)
    solution = result["solution"]
    vector = solution["vector"] if solution else None
    frozen = {
        "status": result["status"],
        "solution": solution,
        "role": "mathematical diagnostic only; no neural intervention authorized",
        "construction": [
            {k: r[k] for k in ("dataset", "prompt_id", "initial_gradient_cell_id")} for r in records
        ],
        "vector_float64_le_sha256": io.sha(struct.pack(f"<{len(vector)}d", *vector))
        if vector is not None
        else None,
        "model_calls": 0,
    }
    frozen_sha = freezer(frozen)  # Exclusive durable freeze MUST precede f02 numeric loading.
    exposed = (
        [r for key in config["descriptive_comparison"] for r in loader(key)]
        if vector is not None
        else []
    )
    pairs = []
    for key in config["construction"]:
        first, second = [r for r in records if r["dataset"] == key]
        den = norm(first["g"]) * norm(second["g"])
        pairs.append(
            {
                "dataset": key,
                "semantic_gradient_cosine": dot(first["g"], second["g"]) / den if den else None,
            }
        )
    return {
        "status": result["status"],
        "solution_sha256": frozen_sha,
        "active_sets": result["active_sets"],
        "construction_rows": row_table(records, vector, config["margin"]),
        "within_pair": pairs,
        "exposed_f02_rows": row_table(exposed, vector, config["margin"]),
        "model_calls": 0,
        "causal_test_performed": False,
    }


def analyze():
    config = io.locked()["config"]

    def freezer(value):
        io.write_new(io.OUTPUT / "solution.json", value)
        digest = io.sha((io.OUTPUT / "solution.json").read_bytes())
        io.write_new(
            io.OUTPUT / "construction_frozen.json",
            {"solution_sha256": digest, "f02_numeric_loading_started": False},
        )
        return digest

    result = calculate(config, lambda key: io.initial_rows(key, config), freezer)
    io.write_new(io.OUTPUT / "analysis.json", result)


if __name__ == "__main__":
    if sys.argv[1:] == ["freeze"]:
        print(json.dumps(io.freeze(), indent=2))
    elif sys.argv[1:] == ["run"]:
        status = io.launch(io.SCRIPT, "_analyze")
        print(json.dumps(status, indent=2))
        if status["status"] != "completed":
            raise SystemExit(1)
    elif sys.argv[1:] == ["_analyze"]:
        analyze()
    else:
        raise SystemExit("Use freeze or run; no adjustable inputs.")
