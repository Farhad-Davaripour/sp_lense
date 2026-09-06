"""Exact supplied 2D design witnesses only; no optimizer, native benchmark or ML."""

import json
import sys
from fractions import Fraction as F

PAIRS = ((2, 0), (3, 1), (6, 4), (7, 5), (10, 8), (11, 9))
KAPPA, STEP, NET, PATH = F(1, 24), F(1, 20), F(1, 5), F(2, 5)
ZERO = (F(0), F(0))


def dot(a, b):
    return sum((x * y for x, y in zip(a, b, strict=True)), F(0))


def add(a, b):
    return tuple(x + y for x, y in zip(a, b, strict=True))


def fixture(rhs, *, affine=False):
    A, b, c = [ZERO] * 12, [F(-1)] * 12, [F(0)] * 6
    A[2], A[0], b[2] = (F(1), F(0)), (F(1), F(0)), rhs
    if affine:
        A[0], b[0], c[0] = (F(-1), F(0)), F(-1, 10), F(1, 10)
    return A, b, c


def quantities(A, b, c, w, length, r):
    assert len(A) == len(b) == 12 and len(c) == 6
    assert 0 <= length <= PATH and dot(w, w) <= NET * NET
    rho = min(STEP, PATH - length)
    D = [tuple((x - y) / 2 for x, y in zip(A[i], A[j], strict=True)) for i, j in PAIRS]
    residual = [cp + dot(row, r) for cp, row in zip(c, D, strict=True)]
    slacks = [dot(row, r) - rhs for row, rhs in zip(A, b, strict=True)]
    gradient = tuple(
        r[j] + KAPPA * sum(dp[j] * z for dp, z in zip(D, residual, strict=True)) for j in range(2)
    )
    return rho, slacks, gradient, residual


def feasible(A, b, c, w, length, r):
    rho, slacks, gradient, residual = quantities(A, b, c, w, length, r)
    assert min(slacks) >= 0
    assert dot(r, r) <= rho * rho
    assert dot(add(w, r), add(w, r)) <= NET * NET
    return (
        {
            "increment": r,
            "step_squared": dot(r, r),
            "net_squared": dot(add(w, r), add(w, r)),
            "rho": rho,
            "minimum_halfspace_slack": min(slacks),
            "pair1_residual_drift": residual[0],
        },
        gradient,
        slacks,
    )


def optimum(A, b, c, w, length, r, multiplier, *, net_multiplier=F(0)):
    # Supplied sufficient KKT witnesses for these fixed toys, using HALF squared
    # ball inequalities. No necessity claim; the general design uses SOC duals.
    record, gradient, slacks = feasible(A, b, c, w, length, r)
    assert multiplier >= 0 and net_multiplier >= 0
    assert multiplier * slacks[2] == 0
    assert net_multiplier * (dot(add(w, r), add(w, r)) - NET * NET) == 0
    stationarity = tuple(
        gradient[j] - multiplier * A[2][j] + net_multiplier * (w[j] + r[j]) for j in range(2)
    )
    assert stationarity == ZERO
    return {
        **record,
        "supplied_kkt_checked": True,
        "halfspace_multiplier": multiplier,
        "half_squared_net_multiplier": net_multiplier,
    }


def infeasible(A, b, c, w, length, u, v, alpha, beta):
    # lambda[2]=1, others zero. This is a supplied exact separation witness,
    # not a feasibility search. It proves impossibility for ALL real increments.
    rho, _, _, _ = quantities(A, b, c, w, length, ZERO)
    assert alpha >= 0 and beta >= 0
    assert dot(u, u) <= alpha * alpha and dot(v, v) <= beta * beta
    assert add(u, v) == A[2]
    gap = b[2] + dot(v, w) - alpha * rho - beta * NET
    assert gap > 0
    return {"strict_separation_gap": gap, "rho": rho, "exact_infeasibility_witness": True}


def main():
    checks = []
    A, b, c = fixture(F(3, 100), affine=True)
    result = optimum(A, b, c, ZERO, F(0), (F(3, 100), F(0)), F(17, 480))
    assert b[0] == F(-1, 10) and result["pair1_residual_drift"] == F(13, 100)
    checks.append({"name": "affine_drift_negative_rhs", **result})

    A, b, c = fixture(STEP)
    result = optimum(A, b, c, ZERO, F(0), (STEP, F(0)), STEP)
    assert result["step_squared"] == STEP * STEP
    checks.append({"name": "feasible_step_boundary", **result})

    A, b, c = fixture(F(3, 50))
    result = infeasible(A, b, c, ZERO, F(0), (F(1), F(0)), ZERO, F(1), F(0))
    assert F(1, 25) + F(1, 100) == F(1, 20)  # .05 surrogate margin attainable
    assert F(1, 10) - F(1, 25) == b[2]  # unchanged .10 local aim is infeasible
    checks.append({"name": "step_infeasible_not_behavioral_impossibility", **result})

    A, b, c = fixture(F(3, 100))
    result = infeasible(A, b, c, ZERO, F(19, 50), (F(1), F(0)), ZERO, F(1), F(0))
    assert result["rho"] == F(1, 50)
    checks.append({"name": "remaining_path_infeasible", **result})

    A, b, c = fixture(F(1, 100))
    result = infeasible(A, b, c, (NET, F(0)), F(1, 5), ZERO, (F(1), F(0)), F(0), F(1))
    checks.append({"name": "outward_net_infeasible", **result})

    A, b, c = fixture(F(3, 100))
    result, _, _ = feasible(A, b, c, (F(0), NET), F(1, 5), (F(3, 100), F(-1, 100)))
    assert result["step_squared"] == F(1, 1000)
    assert result["net_squared"] == F(37, 1000)
    assert NET * NET + b[2] * b[2] > NET * NET  # every row-span r_x>=b fails
    checks.append({"name": "gradient_span_alone_is_unsound", **result})

    A, b, c = fixture(F(4, 101))
    result = optimum(
        A, b, c, (F(0), NET), F(1, 5), (F(4, 101), F(-2, 505)), F(4, 99), net_multiplier=F(2, 99)
    )
    assert result["step_squared"] == F(4, 2525)
    assert result["net_squared"] == NET * NET
    checks.append({"name": "feasible_active_net_boundary", **result})

    assert not any(
        name.split(".")[0] in {"torch", "transformers", "transformer_lens", "numpy", "scipy"}
        for name in sys.modules
    )
    return {
        "status": "EXACT_SUPPLIED_TOY_WITNESSES_PASSED",
        "checks": checks,
        "arithmetic": "fractions.Fraction",
        "dimensions": 2,
        "rows": 12,
        "pairs_zero_based": PAIRS,
        "kappa": str(KAPPA),
        "model_loads": 0,
        "real_forwards": 0,
        "real_derivatives": 0,
        "optimizer_or_search_executed": False,
        "native_benchmark_executed": False,
    }


if __name__ == "__main__":
    print(json.dumps(main(), default=str, sort_keys=True))
