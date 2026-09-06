"""Check supplied 2D partial-progress witnesses; no optimizer, data reads or ML."""

import json
import sys
from fractions import Fraction as F

PAIRS = ((2, 0), (3, 1), (6, 4), (7, 5), (10, 8), (11, 9))
STEP, NET, PATH = F(1, 20), F(1, 5), F(2, 5)
ZERO, E1 = (F(0), F(0)), (F(1), F(0))


def dot(a, b):
    return sum((x * y for x, y in zip(a, b, strict=True)), F(0))


def add(a, b):
    return tuple(x + y for x, y in zip(a, b, strict=True))


def fixture(rhs, opposite=False):
    A, b, c = [ZERO] * 12, [F(-1)] * 12, [F(0)] * 6
    A[2], A[0] = E1, (F(-1), F(0)) if opposite else E1
    b[2] = b[0] = rhs
    return A, b, c


def objective(A, b, c, r):
    assert len(A) == len(b) == 12 and len(c) == 6
    D = [tuple((x - y) / 2 for x, y in zip(A[i], A[j], strict=True)) for i, j in PAIRS]
    deficits = [max(F(0), rhs - dot(row, r)) for row, rhs in zip(A, b, strict=True)]
    drift = [cp + dot(row, r) for cp, row in zip(c, D, strict=True)]
    value = dot(r, r) / 2 + dot(deficits, deficits) / 24 + dot(drift, drift) / 48
    grad = tuple(
        r[j]
        - sum((row[j] * z for row, z in zip(A, deficits, strict=True)), F(0)) / 12
        + sum((row[j] * z for row, z in zip(D, drift, strict=True)), F(0)) / 24
        for j in range(2)
    )
    return value, grad, deficits, drift


def geometry(w, length, r):
    assert 0 <= length <= PATH and dot(w, w) <= NET * NET
    assert dot(w, w) <= length * length
    rho = min(STEP, PATH - length)
    assert dot(r, r) <= rho * rho
    assert dot(add(w, r), add(w, r)) <= NET * NET
    return rho


def certificate(A, b, c, w, length, r, u=ZERO, v=ZERO, alpha=F(0), gamma=F(0)):
    rho = geometry(w, length, r)
    assert alpha >= 0 and gamma >= 0
    assert dot(u, u) <= alpha * alpha and dot(v, v) <= gamma * gamma
    value, grad, deficits, drift = objective(A, b, c, r)
    f0 = objective(A, b, c, ZERO)[0]
    e = add(add(grad, u), v)
    step_support = rho * alpha - dot(u, r)
    net_support = NET * gamma - dot(v, add(w, r))
    assert step_support >= 0 and net_support >= 0
    gap = step_support + net_support + dot(e, e) / 2
    epsilon = F(1, 2**30) * max(F(1), f0)
    return {
        "input": {"A": A, "b": b, "c": c, "w": w, "path": length},
        "r": r,
        "rho": rho,
        "step_squared": dot(r, r),
        "net_squared": dot(add(w, r), add(w, r)),
        "u": u,
        "v": v,
        "alpha": alpha,
        "gamma": gamma,
        "stationarity_residual": e,
        "step_support_gap": step_support,
        "net_support_gap": net_support,
        "objective_gap_bound": gap,
        "objective_at_zero": f0,
        "objective": value,
        "gain": f0 - value,
        "deficits": deficits,
        "drift": drift,
        "epsilon": epsilon,
        "certified_progress_admission": gap <= epsilon and f0 - value > epsilon,
        "certified_epsilon_optimal_zero": f0 - (value - gap) <= epsilon,
        "exact_optimum_certified": gap == 0,
    }


def main():
    checks = []
    A, b, c = fixture(F(1, 10))
    partial = certificate(A, b, c, ZERO, F(0), (F(1, 70), F(0)))
    assert partial["exact_optimum_certified"] and partial["certified_progress_admission"]
    assert partial["gain"] == F(1, 8400) and partial["deficits"][2] == F(3, 35)
    assert b[2] > STEP  # full local target still cannot fit this step ball
    rounded = certificate(A, b, c, ZERO, F(0), (F.from_float(float(F(1, 70))), F(0)))
    assert rounded["certified_progress_admission"] and not rounded["exact_optimum_certified"]
    checks.append(
        {"name": "feasible_partial_progress", **partial, "binary64_interior_check": rounded}
    )

    A, b, c = fixture(F(-1, 10))
    zero = certificate(A, b, c, ZERO, F(0), ZERO)
    assert zero["exact_optimum_certified"] and zero["certified_epsilon_optimal_zero"]
    assert zero["objective"] == 0 and not zero["certified_progress_admission"]
    checks.append({"name": "zero_step_satisfied_deficits", **zero})

    A, b, c = fixture(F(1, 10), opposite=True)
    conflict = certificate(A, b, c, ZERO, F(0), ZERO)
    assert conflict["exact_optimum_certified"] and conflict["certified_epsilon_optimal_zero"]
    assert conflict["deficits"][0] == conflict["deficits"][2] == F(1, 10)
    # An explicitly supplied nearby point checks the nonzero D term and its sign.
    nearby_value, nearby_grad, _, nearby_drift = objective(A, b, c, (F(1, 100), F(0)))
    assert nearby_value - conflict["objective"] == F(29, 480000)
    assert nearby_grad == (F(29, 2400), F(0)) and nearby_drift[0] == F(1, 100)
    checks.append({"name": "contradictory_pair_positive_deficit_stagnation", **conflict})

    A, b, c = fixture(F(1, 10))
    boundary = certificate(
        A,
        b,
        c,
        (F(19, 100), F(0)),
        F(19, 100),
        (F(1, 100), F(0)),
        v=(F(1, 200), F(0)),
        gamma=F(1, 200),
    )
    assert boundary["exact_optimum_certified"] and boundary["certified_progress_admission"]
    assert boundary["objective"] == F(29, 40000) and boundary["gain"] == F(13, 120000)
    assert boundary["net_squared"] == NET * NET
    # This rational optimum is NOT automatically a serializable admitted point.
    # Literal binary64 .2 is above decimal1/5; a rounded endpoint must be rejected.
    stored_before, stored_after = F.from_float(0.19), F.from_float(0.20)
    actual_difference = stored_after - stored_before
    assert actual_difference**2 < STEP * STEP and stored_after**2 > NET * NET
    checks.append(
        {
            "name": "active_net_boundary",
            **boundary,
            "binary64_boundary_expected_rejection": {
                "stored_w": stored_before,
                "stored_w_next": stored_after,
                "exact_serialized_difference": actual_difference,
                "net_squared_excess": stored_after**2 - NET * NET,
                "admitted": False,
                "repaired": False,
            },
        }
    )

    assert not any(
        name.split(".")[0] in {"torch", "transformers", "transformer_lens", "numpy", "scipy"}
        for name in sys.modules
    )
    return {
        "status": "EXACT_SUPPLIED_PARTIAL_PROGRESS_TOYS_PASSED",
        "checks": checks,
        "dimension": 2,
        "rows": 12,
        "pairs_zero_based": PAIRS,
        "objective_coefficients": {
            "increment_squared": "1/2",
            "deficit_squared": "1/24",
            "common_drift_squared": "1/48",
        },
        "arithmetic": "fractions.Fraction",
        "optimizer_or_search_executed": False,
        "native_benchmark_executed": False,
        "model_loads": 0,
        "real_forwards": 0,
        "real_derivatives": 0,
        "evidence_files_read": 0,
        "numbers_source": "hand-derived witnesses from declared coefficients and decimal radii; no model data",
    }


if __name__ == "__main__":
    print(json.dumps(main(), default=str, sort_keys=True))
