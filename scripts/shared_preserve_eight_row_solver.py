"""Eight-row wrapper: unchanged original primitives and exhaustive mask policy."""

from __future__ import annotations

import math

from scripts import shared_direction_linear_feasibility as previous

io, dot, norm = previous.io, previous.dot, previous.norm
linear_solve, metrics, kkt_valid = previous.linear_solve, previous.metrics, previous.kkt_valid


def solve(A, b, config):
    io.require(
        0 < len(A) == len(b) <= 8
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
