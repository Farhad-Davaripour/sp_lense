"""Independent exact-rational descent checks; no production objective oracle.

Reuse only the frozen independent input/history authentication and rational
primitives. Candidate costs and the descriptive diagnostic are evaluated here;
the previous near-optimality certificate and the old solver are never called.
"""

from fractions import Fraction as F

from scripts import verify_partial_progress_deficit as auth


def prepare(problem, expected_sha, deadline):
    context = auth.prepare(problem, expected_sha, deadline)
    sealed = context["_sealed"]
    before = _components(sealed, (F(0),) * len(sealed.w))
    auth._require(before["total"] == sealed.f0, "independent_initial_cost_mismatch")
    auth._tick(sealed.deadline)
    return context


def _components(sealed, actual):
    deadline = sealed.deadline
    deficits = tuple(
        max(F(0), rhs - auth._dot(row, actual, deadline))
        for row, rhs in zip(sealed.A, sealed.b, strict=True)
    )
    drift = tuple(
        cp + auth._dot(row, actual, deadline) for cp, row in zip(sealed.c, sealed.D, strict=True)
    )
    values = {
        "proximal": auth._dot(actual, actual, deadline) / 2,
        "deficit": auth._dot(deficits, deficits, deadline) / 24,
        "drift": auth._dot(drift, drift, deadline) / 48,
    }
    values["total"] = sum(values.values(), F(0))
    return values


def _initial_components(sealed):
    deficits = tuple(max(F(0), b) for b in sealed.b)
    deficit = auth._dot(deficits, deficits, sealed.deadline) / 24
    drift = auth._dot(sealed.c, sealed.c, sealed.deadline) / 48
    return {"proximal": F(0), "deficit": deficit, "drift": drift, "total": deficit + drift}


def exact_zero(context):
    sealed = context["_sealed"]
    auth._tick(sealed.deadline)
    stationary = all(x == 0 for x in sealed.gradient_zero)
    singleton = sealed.rho == 0
    exact = stationary or singleton
    result = {
        "status": "EXACT_ZERO_OPTIMUM" if exact else "NOT_EXACT_ZERO",
        "reason": "singleton_step_ball"
        if singleton
        else ("exact_zero_gradient" if stationary else "no_exact_zero_proof"),
        "problem_sha256": sealed.problem_sha256,
        "current_w_sha256": sealed.w_sha256,
        "exact_zero_gradient": stationary,
        "singleton_step_ball": singleton,
        "objective_zero": str(sealed.f0),
        "exact_zero_optimum": exact,
    }
    auth._tick(sealed.deadline)
    return result


def _endpoint(sealed, w_next):
    endpoint = auth._vector(w_next, len(sealed.w), sealed.deadline)
    actual = tuple(x - w for x, w in zip(endpoint, sealed.w, strict=True))
    fingerprint = auth.vector_sha256([float(x) for x in w_next])
    auth._tick(sealed.deadline)
    return endpoint, actual, fingerprint


def check(context, w_next):
    """Check the serialized endpoint, not the nominal projected/scaled vector."""
    sealed = context["_sealed"]
    deadline = sealed.deadline
    auth._tick(deadline)
    endpoint, actual, fingerprint = _endpoint(sealed, w_next)
    step_square = auth._dot(actual, actual, deadline)
    net_square = auth._dot(endpoint, endpoint, deadline)
    path_after = sealed.path + auth._norm_upper(step_square)
    geometry = (
        step_square <= sealed.rho**2 and net_square <= auth.NET**2 and path_after <= auth.PATH
    )
    before, after = _initial_components(sealed), _components(sealed, actual)
    auth._require(before["total"] == sealed.f0, "initial_cost_mismatch")
    gain = before["total"] - after["total"]
    s = -auth._dot(sealed.gradient_zero, actual, deadline)
    eta = F(1, 2**30) * max(F(1), sealed.f0)
    excess = gain - s / 4 - eta
    is_zero = all(x == 0 for x in actual)
    admitted = geometry and s > 0 and excess > 0 and not is_zero
    if is_zero:
        status, reason = "NO_CERTIFIED_STEP", "serialized_zero_increment"
    elif not geometry:
        status, reason = "REJECTED_TRIAL", "serialized_geometry_or_path_violation"
    elif not admitted:
        status, reason = "REJECTED_TRIAL", "strict_sufficient_decrease_not_met"
    else:
        status, reason = "ADMITTED_DESCENT", "exact_serialized_sufficient_decrease"
    result = {
        "status": status,
        "reason": reason,
        "problem_sha256": sealed.problem_sha256,
        "current_w_sha256": sealed.w_sha256,
        "w_next_sha256": fingerprint,
        "actual_displacement": [str(x) for x in actual],
        "evaluated_exact_serialized_difference": True,
        "geometry_valid": geometry,
        "geometry": {
            "actual_step_squared": str(step_square),
            "net_squared": str(net_square),
            "rho": str(sealed.rho),
            "path_before_upper": str(sealed.path),
            "path_after_upper": str(path_after),
            "authenticated_history_path_upper": str(sealed.history_path_upper),
            "geometry_tolerance": "0",
        },
        "objective_zero": str(sealed.f0),
        "objective": str(after["total"]),
        "gain": str(gain),
        "s": str(s),
        "eta": str(eta),
        "sufficient_decrease_excess": str(excess),
        "components_before": {key: str(value) for key, value in before.items()},
        "components_after": {key: str(value) for key, value in after.items()},
        "near_optimality_gate": False,
        "exact_zero_optimum": False,
    }
    auth._tick(deadline)
    return result


def diagnostic(context, w_next):
    """Compute once for a selected point; this upper bound is not an admission gate."""
    sealed = context["_sealed"]
    deadline = sealed.deadline
    auth._tick(deadline)
    _, actual, fingerprint = _endpoint(sealed, w_next)
    deficits = tuple(
        max(F(0), rhs - auth._dot(row, actual, deadline))
        for row, rhs in zip(sealed.A, sealed.b, strict=True)
    )
    drift = tuple(
        cp + auth._dot(row, actual, deadline) for cp, row in zip(sealed.c, sealed.D, strict=True)
    )
    gradient = []
    for j, rj in enumerate(actual):
        if j % 32 == 0:
            auth._tick(deadline)
        deficit_part = sum((row[j] * z for row, z in zip(sealed.A, deficits, strict=True)), F(0))
        drift_part = sum((row[j] * z for row, z in zip(sealed.D, drift, strict=True)), F(0))
        gradient.append(rj - deficit_part / 12 + drift_part / 24)
    gap_upper = auth._dot(gradient, gradient, deadline) / 2
    result = {
        "w_next_sha256": fingerprint,
        "descriptive_gap_upper": str(gap_upper),
        "bound_kind": "one_strong_convexity_zero_support_upper_bound",
        "is_actual_gap": False,
        "used_as_admission_gate": False,
    }
    auth._tick(deadline)
    return result
