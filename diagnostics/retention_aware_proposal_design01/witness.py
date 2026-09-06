"""One fixed exact-rational witness check; no proposal solver, model or search."""

import hashlib
import json
import sys
from fractions import Fraction as F
from pathlib import Path


def dot(a, b):
    return sum((x * y for x, y in zip(a, b, strict=True)), F(0))


def objective(A, margins, original, pairs, target, r):
    # Independent direct evaluation of the frozen mathematical J, not a solver.
    D = [tuple((A[i][k] - A[j][k]) / 2 for k in range(2)) for i, j in pairs]
    c = [(margins[i] - original[i] - margins[j] + original[j]) / 2 for i, j in pairs]
    deficit = [max(F(0), target - m - dot(a, r)) for a, m in zip(A, margins, strict=True)]
    drift = [v + dot(d, r) for v, d in zip(c, D, strict=True)]
    components = {
        "proximal": dot(r, r) / 2,
        "deficit": dot(deficit, deficit) / 24,
        "drift": dot(drift, drift) / 48,
    }
    gradient = tuple(
        r[k] - sum((a[k] * h for a, h in zip(A, deficit, strict=True)), F(0)) / 12
        + sum((d[k] * v for d, v in zip(D, drift, strict=True)), F(0)) / 24
        for k in range(2)
    )
    M = F(1) + sum((dot(a, a) for a in A), F(0)) / 12
    M += sum((dot(d, d) for d in D), F(0)) / 24
    return sum(components.values(), F(0)), gradient, M, components, D, c


def main():
    raw = Path(__file__).with_name("inputs.json").read_bytes()
    source = json.loads(raw)
    assert source["dimension"] == 2 and source["rows"] == 12
    pairs = tuple(tuple(p) for p in source["pairs_zero_based"])
    assert pairs == ((2, 0), (3, 1), (6, 4), (7, 5), (10, 8), (11, 9))
    constants = source["constants"]
    assert constants["tau_policy"] == "Fraction.from_float(0.05 - 1e-6)"
    # The sole binary64 expression preserves the already-existing floor exactly;
    # all subsequent toy arithmetic uses rational numbers, without float casts.
    tau = F.from_float(0.05 - 1e-6)
    target, step, net, path, armijo, eta_factor = (
        F(constants[k]) for k in ("target", "step", "net", "path", "armijo", "eta_factor")
    )
    assert (target, step, net, path, armijo, eta_factor) == (
        F(1, 10), F(1, 20), F(1, 5), F(2, 5), F(1, 4), F(1, 2**30)
    )
    A = [tuple(F(v) for v in a) for a in source["A"]]
    margins = [F(v) for v in source["margins"]]
    assert len(A) == len(margins) == 12 and all(len(a) == 2 for a in A)
    assert source["original_margins"] == "equal_to_current_margins"
    original = margins.copy()
    protected = tuple(source["original_protected_rows"])
    assert protected == (0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    zero = (F(0), F(0))
    w = tuple(F(v) for v in source["w"])
    length = F(source["path_upper"])
    r = tuple(F(v) for v in source["supplied_increment"])
    assert w == zero and length == 0 and r == (F(1, 100), F(0))
    assert A[0] == (F(-1), F(0)) and A[2] == (F(1), F(0))
    assert margins[0] == F(3, 50) and margins[2] == F(-1, 5)
    assert all(A[i] == zero and margins[i] == F(1, 5) for i in range(12) if i not in (0, 2))
    rho = min(step, path - length)
    assert 0 <= length <= path and dot(w, w) <= net**2 and dot(w, w) <= length**2
    assert all(margins[i] >= tau for i in protected)
    assert tau > 0 and tau < F(1, 20)

    J0, g0, M, before, D, c = objective(A, margins, original, pairs, target, zero)
    Jr, _, M_after, after, _, _ = objective(A, margins, original, pairs, target, r)
    assert D == [(F(1), F(0))] + [zero] * 5 and c == [F(0)] * 6
    assert M == M_after == F(29, 24) and g0 == (F(-13, 600), F(0))
    assert before == {"proximal": F(0), "deficit": F(229, 60000), "drift": F(0)}
    assert after == {"proximal": F(1, 20000), "deficit": F(433, 120000), "drift": F(1, 480000)}
    assert J0 == F(229, 60000) and Jr == F(1757, 480000)
    next_margins = [m + dot(a, r) for a, m in zip(A, margins, strict=True)]
    slacks = {str(i): next_margins[i] - tau for i in protected}
    assert all(v >= 0 for v in slacks.values())
    assert next_margins[0] == F(1, 20) < margins[0]
    assert margins[2] < next_margins[2] == F(-19, 100) < target
    assert all(next_margins[i] == margins[i] for i in range(12) if i not in (0, 2))
    actual_net = tuple(x + y for x, y in zip(w, r, strict=True))
    step2, net2 = dot(r, r), dot(actual_net, actual_net)
    # Supplied axis-aligned rational increment: its norm is exact, no sqrt bound.
    assert r[1] == 0
    path_after = length + abs(r[0])
    assert step2 <= rho**2 and net2 <= net**2 and path_after <= path
    s, gain, eta = -dot(g0, r), J0 - Jr, eta_factor * max(F(1), J0)
    assert s == F(13, 60000) > 0 and gain == F(1, 6400)
    assert gain - armijo * s == F(49, 480000) > eta
    y = tuple(-v / M for v in g0)
    assert y == (F(13, 725), F(0))
    assert dot(y, y) <= rho**2 and dot(y, y) <= net**2 and abs(y[0]) <= path
    assert margins[0] + dot(A[0], y) < tau

    # Same rows, not a different fixture: two hard .10 inequalities contradict.
    old_upper = margins[0] - target
    old_lower = target - margins[2]
    assert old_upper == F(-1, 25) and old_lower == F(3, 10)
    assert old_lower > old_upper

    assert source["zero_case"]["row"] == 0 and source["zero_case"]["margin"] == "tau"
    zero_margins = margins.copy()
    zero_margins[0] = tau
    Jz, gz, _, _, Dz, cz = objective(A, zero_margins, zero_margins.copy(), pairs, target, zero)
    assert all(zero_margins[i] >= tau for i in protected)
    assert Dz == D and cz == c
    assert tau - zero_margins[0] == 0 and A[0] == (F(-1), F(0))
    assert gz == (-(F(1, 5) + tau) / 12, F(0)) and gz[0] < 0
    # Exact KKT witness: lambda=-gz[0]>=0 for h(r)=r1<=0;
    # g0+lambda*e1=0, lambda*h(0)=0. Other constraints have zero multipliers.
    # J is 1-strongly convex because its other terms are convex squared hinges
    # or quadratics. This analytic argument covers ALL feasible r, not samples.
    multiplier = -gz[0]
    assert multiplier > 0 and (gz[0] + multiplier, gz[1]) == zero
    assert multiplier * F(0) == 0
    assert not any(name.split(".")[0] in {"torch", "transformers", "numpy", "scipy"} for name in sys.modules)
    return {
        "schema": "sp_lense.retention_aware_rational_witness_results.v1",
        "status": "FIXED_RATIONAL_WITNESSES_PASSED",
        "inputs_sha256": hashlib.sha256(raw).hexdigest(),
        "arithmetic": "Exact Fraction arithmetic after exact import of unchanged binary64 tau expression",
        "tau_exact": tau,
        "supplied_partial_improvement": {
            "r": r, "g0": g0, "M": M, "J0": J0, "Jr": Jr,
            "components_before": before, "components_after": after,
            "s": s, "Delta": gain, "eta": eta,
            "strict_certificate_excess": gain - armijo * s - eta,
            "protected_margin_before": margins[0], "protected_margin_after": next_margins[0],
            "minimum_protected_slack": min(slacks.values()),
            "opposed_margin_before": margins[2], "opposed_margin_after": next_margins[2],
            "step_squared": step2, "net_squared": net2, "path_after": path_after,
            "geometry_retention_and_sufficient_decrease_pass": True,
            "old_two_ball_target": y,
            "old_two_ball_target_protected_margin": margins[0] + dot(A[0], y),
            "old_two_ball_target_violates_retention": True,
            "supplied_r_is_a_solver_output": False,
        },
        "same_example_old_all_targets": {
            "required_r1_upper": old_upper, "required_r1_lower": old_lower,
            "contradiction_gap": old_lower - old_upper,
            "infeasible_by_two_exact_inequalities": True,
        },
        "zero_constrained_optimum": {
            "protected_margin": tau, "J0": Jz, "g0": gz,
            "retention_constraint": "r1<=0", "active_halfspace_multiplier": multiplier,
            "exact_stationarity_residual": zero, "complementarity_residual": F(0),
            "proof": "g0 dot r>=0 on every feasible r; J(r)>=J(0)+g0 dot r+||r||^2/2. Zero is unique for this local affine toy.",
            "global_model_impossibility_claim": False,
            "future_automatic_zero_certificate_implemented": False,
        },
        "boundary": {
            "model_loads": 0, "real_forwards": 0, "real_derivatives": 0,
            "solver_implemented_or_run": False, "optimizer_or_ray_search": False,
            "native_fixtures_or_historical_scores_read": False,
            "binary64_endpoint_or_Dykstra_implementation_validated": False,
            "nonlinear_behavior_or_ordinary_task_preservation_tested": False,
            "missing_behavioral_eligibility": "UNTESTED",
            "claim": "Mathematical feasibility only; no runtime readiness or steering efficacy.",
        },
    }


if __name__ == "__main__":
    print(json.dumps(main(), default=str, sort_keys=True, separators=(",", ":"), allow_nan=False))
