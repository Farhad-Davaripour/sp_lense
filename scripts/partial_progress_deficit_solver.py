"""One bounded model-free projected-gradient prototype, never a model controller.

The independent checker owns exact input authentication, path history, exact
serialized geometry, and the objective-gap certificate. Projection tolerances
below screen only floating-point projection arithmetic: they NEVER enlarge a
ball or admit a point. There is no point repair, restart, or fallback.

On NOT_CERTIFIED the next internal gradient uses the nominal floating projection
p, not the serialized increment. Every proposed endpoint is nevertheless checked
using its exact serialized difference from the authenticated original w.
"""

from __future__ import annotations

import copy
import math
import time

from scripts import verify_partial_progress_deficit as checker

MAX_ITERATIONS = 200
SECONDS = 10.0
NET_RADIUS = 0.20
PROJECTION_FACTOR = 128.0
MACHINE_EPSILON = 2.0**-52
CERTIFIED = {"ADMITTED_PROGRESS", "EXACT_ZERO_OPTIMUM", "EPSILON_OPTIMAL_ZERO"}
CHECK_STATUSES = CERTIFIED | {"NOT_CERTIFIED", "NUMERICALLY_UNRESOLVED"}
CASES = ("interior", "step_only", "net_only", "both_active", "zero_radius")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), "NONFINITE_FLOAT_ARITHMETIC")
    return float(value)


def vector(values, dimension=None):
    require(
        isinstance(values, list)
        and 1 <= len(values) <= 1024
        and (dimension is None or len(values) == dimension),
        "VECTOR_SHAPE",
    )
    return [finite(value) for value in values]


def norm(values):
    return finite(math.hypot(*values))


def dot(left, right):
    return finite(math.fsum(finite(x * y) for x, y in zip(left, right, strict=True)))


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("COMBINED_TEN_SECOND_DEADLINE")


def _projection_record(y, w, point, step_normal, net_normal, step_multiplier, net_multiplier, case):
    """Check projection KKT arithmetic, not exact feasibility or objective KKT."""
    n = len(y)
    point = vector(point, n)
    step_normal, net_normal = vector(step_normal, n), vector(net_normal, n)
    step_multiplier, net_multiplier = finite(step_multiplier), finite(net_multiplier)
    require(step_multiplier >= 0.0 and net_multiplier >= 0.0, "NEGATIVE_PROJECTION_MULTIPLIER")
    residual = vector(
        [p - yi + ns + nn for p, yi, ns, nn in zip(point, y, step_normal, net_normal, strict=True)],
        n,
    )
    residual_norm = norm(residual)
    normal_norm = finite(math.hypot(norm(step_normal), norm(net_normal)))
    scale = max(1.0, norm(y), norm(point), normal_norm)
    limit = finite(PROJECTION_FACTOR * n * MACHINE_EPSILON * scale)
    require(residual_norm <= limit, "PROJECTION_NORMAL_RESIDUAL")
    require(case in CASES, "PROJECTION_CASE")
    # Both vector normals remain available to the independent objective checker;
    # no outward-normal or geometric tolerance is silently inferred from this.
    return {
        "point": point,
        "step_normal": step_normal,
        "net_normal": net_normal,
        "step_multiplier": step_multiplier,
        "net_multiplier": net_multiplier,
        "case": case,
        "residual_norm": residual_norm,
        "residual_limit": limit,
    }


def project_intersection(y, w, rho):
    """Fixed analytic cases for B(0,rho) intersect B(-w,.20).

    The radii here are nominal binary64 projection inputs only. Exact decimal
    admission, including serialization effects, belongs exclusively to checker.
    The both-active circle formula is used only when neither single-ball
    projection is feasible in the other ball; it is not alternating projection.
    """
    y, w = vector(y), vector(w, len(y))
    n = len(y)
    rho = finite(rho)
    require(0.0 <= rho <= 0.05, "INVALID_STEP_RADIUS")
    require(norm(w) <= NET_RADIUS, "INVALID_NET_ORIGIN")
    zero = [0.0] * n
    if rho == 0.0:
        # A singleton's normal cone is the whole space, not lambda*point.
        return _projection_record(y, w, zero, list(y), zero, 0.0, 0.0, "zero_radius")

    yn = norm(y)
    if yn <= rho and norm([x + wi for x, wi in zip(y, w, strict=True)]) <= NET_RADIUS:
        return _projection_record(y, w, list(y), zero, zero, 0.0, 0.0, "interior")

    # This case also covers concentric balls and a contained step ball.
    if yn <= rho:
        step_point, step_lambda = list(y), 0.0
    else:
        require(yn > 0.0, "ZERO_STEP_PROJECTION_DENOMINATOR")
        scale = finite(rho / yn)
        step_point = vector([scale * x for x in y], n)
        step_lambda = finite(yn / rho - 1.0)
    if norm([x + wi for x, wi in zip(step_point, w, strict=True)]) <= NET_RADIUS:
        step_normal = vector([step_lambda * x for x in step_point], n)
        return _projection_record(
            y, w, step_point, step_normal, zero, step_lambda, 0.0, "step_only"
        )

    translated = vector([x + wi for x, wi in zip(y, w, strict=True)], n)
    translated_norm = norm(translated)
    if translated_norm <= NET_RADIUS:
        net_point, net_lambda = list(y), 0.0
    else:
        require(translated_norm > 0.0, "ZERO_NET_PROJECTION_DENOMINATOR")
        scale = finite(NET_RADIUS / translated_norm)
        net_point = vector([scale * x - wi for x, wi in zip(translated, w, strict=True)], n)
        net_lambda = finite(translated_norm / NET_RADIUS - 1.0)
    if norm(net_point) <= rho:
        net_normal = vector([net_lambda * (x + wi) for x, wi in zip(net_point, w, strict=True)], n)
        return _projection_record(y, w, net_point, zero, net_normal, 0.0, net_lambda, "net_only")

    distance = norm(w)
    require(distance > 0.0, "UNCERTAIN_CONCENTRIC_PROJECTION")
    axis = vector([x / distance for x in w], n)
    axial_y = dot(y, axis)
    perpendicular = vector([x - axial_y * e for x, e in zip(y, axis, strict=True)], n)
    perpendicular_norm = norm(perpendicular)
    axial_point = finite(
        (NET_RADIUS * NET_RADIUS - rho * rho - distance * distance) / (2.0 * distance)
    )
    height_squared = finite(rho * rho - axial_point * axial_point)
    require(
        height_squared > 0.0 and perpendicular_norm > 0.0,
        "DEGENERATE_OR_UNCERTAIN_TWO_BOUNDARY_PROJECTION",
    )
    height = finite(math.sqrt(height_squared))
    radial_scale = finite(height / perpendicular_norm)
    point = vector(
        [axial_point * e + radial_scale * x for e, x in zip(axis, perpendicular, strict=True)], n
    )
    total_scale = finite(perpendicular_norm / height)
    net_lambda = finite((axial_y - total_scale * axial_point) / distance)
    step_lambda = finite(total_scale - 1.0 - net_lambda)
    require(step_lambda >= 0.0 and net_lambda >= 0.0, "NEGATIVE_TWO_BOUNDARY_MULTIPLIER")
    step_normal = vector([step_lambda * x for x in point], n)
    net_normal = vector([net_lambda * (x + wi) for x, wi in zip(point, w, strict=True)], n)
    return _projection_record(
        y, w, point, step_normal, net_normal, step_lambda, net_lambda, "both_active"
    )


def gradient(context, r):
    """One fixed floating local gradient; independent exact audit is elsewhere."""
    A, b, c, D = (context[key] for key in ("A", "b", "c", "D"))
    n = context["dimension"]
    r = vector(r, n)
    deficits = [max(0.0, finite(rhs - dot(row, r))) for row, rhs in zip(A, b, strict=True)]
    drift = [finite(cp + dot(row, r)) for row, cp in zip(D, c, strict=True)]
    result = []
    for j in range(n):
        if j % 64 == 0:
            check_deadline(context["deadline"])
        margin_part = finite(
            math.fsum(finite(row[j] * z) for row, z in zip(A, deficits, strict=True)) / 12.0
        )
        drift_part = finite(
            math.fsum(finite(row[j] * z) for row, z in zip(D, drift, strict=True)) / 24.0
        )
        result.append(finite(r[j] - margin_part + drift_part))
    return result


def solve(problem, expected_problem_sha256, *, observer=None):
    """Attempt one fixed numerical solve; no status authorizes any model call."""
    started = time.monotonic()
    deadline = started + SECONDS
    result = {
        "status": "NUMERICALLY_UNRESOLVED",
        "terminal_reason": "NOT_STARTED",
        "problem_sha256": expected_problem_sha256,
        "iterations": 0,
        "certificates_checked": 0,
        "initial_certificate_checked": False,
        "projection_case_counts": dict.fromkeys(CASES, 0),
        "last_projection": None,
        "certificate": None,
        "witness": None,
        "step_admitted": False,
        "w_next": None,
        "maximum_pg_iterations": MAX_ITERATIONS,
        "combined_seconds_limit": SECONDS,
        "next_internal_iterate": "nominal floating projection, not serialized difference",
        "checked_increment": "exact serialized w_next minus authenticated w",
        "point_repaired": False,
        "fallback_used": False,
        "model_calls_authorized": False,
    }

    def finish(status, reason):
        finished = time.monotonic()
        if status in CERTIFIED and finished >= deadline:
            status, reason = "NUMERICALLY_UNRESOLVED", "COMBINED_TEN_SECOND_DEADLINE"
        result["status"], result["terminal_reason"] = status, reason
        result["step_admitted"] = status == "ADMITTED_PROGRESS"
        result["w_next"] = result["witness"]["w_next"] if result["step_admitted"] else None
        result["elapsed_seconds"] = finished - started
        return result

    def certify(context, endpoint, u, v, *, initial=False):
        check_deadline(deadline)
        result["witness"] = {"w_next": list(endpoint), "u": list(u), "v": list(v)}
        result["certificates_checked"] += 1
        if initial:
            result["initial_certificate_checked"] = True
        certificate = checker.check(context, endpoint, u, v)
        result["certificate"] = certificate
        check_deadline(deadline)
        require(
            isinstance(certificate, dict) and certificate.get("status") in CHECK_STATUSES,
            "INVALID_INDEPENDENT_CERTIFICATE_STATUS",
        )
        if observer is not None:
            # Recording only: no mutable solver/certificate references are
            # exposed, and a callback's return value never affects generation.
            observer(
                {
                    "iterations": result["iterations"],
                    "certificates_checked": result["certificates_checked"],
                    "certificate": copy.deepcopy(certificate),
                    "fixed_M": result["fixed_M"],
                    "projection_case_counts": dict(result["projection_case_counts"]),
                }
            )
            check_deadline(deadline)
        return certificate["status"]

    try:
        require(observer is None or callable(observer), "INVALID_RECORDING_OBSERVER")
        require(
            type(expected_problem_sha256) is str
            and len(expected_problem_sha256) == 64
            and all(x in "0123456789abcdef" for x in expected_problem_sha256),
            "INVALID_EXPECTED_PROBLEM_SHA256",
        )
        context = checker.prepare(problem, expected_problem_sha256, deadline)
        check_deadline(deadline)
        n = context["dimension"]
        require(
            type(n) is int and 1 <= n <= 1024 and context["deadline"] == deadline,
            "INVALID_CHECKER_CONTEXT",
        )
        w, rho, M = vector(context["w"], n), finite(context["rho"]), finite(context["M"])
        require(M >= 1.0, "INVALID_OUTWARD_LIPSCHITZ_BOUND")
        result["dimension"], result["fixed_M"] = n, M
        r, zero = [0.0] * n, [0.0] * n
        status = certify(context, list(w), zero, zero, initial=True)
        if status in CERTIFIED:
            return finish(status, "INITIAL_ZERO_CERTIFICATE")
        if status == "NUMERICALLY_UNRESOLVED":
            return finish(status, "INITIAL_ZERO_CERTIFICATE_UNRESOLVED")

        for iteration in range(1, MAX_ITERATIONS + 1):
            check_deadline(deadline)
            result["iterations"] = iteration
            g = gradient(context, r)
            y = vector([x - gj / M for x, gj in zip(r, g, strict=True)], n)
            projected = project_intersection(y, w, rho)
            result["projection_case_counts"][projected["case"]] += 1
            result["last_projection"] = {
                key: projected[key]
                for key in (
                    "case",
                    "step_multiplier",
                    "net_multiplier",
                    "residual_norm",
                    "residual_limit",
                )
            }
            result["last_projection"]["objective_normal_scale"] = M
            check_deadline(deadline)
            p = projected["point"]
            endpoint = vector([wi + x for wi, x in zip(w, p, strict=True)], n)
            u = vector([M * x for x in projected["step_normal"]], n)
            v = vector([M * x for x in projected["net_normal"]], n)
            status = certify(context, endpoint, u, v)
            if status in CERTIFIED:
                return finish(status, "INDEPENDENT_OBJECTIVE_CERTIFICATE")
            if status == "NUMERICALLY_UNRESOLVED":
                return finish(status, "SERIALIZED_CANDIDATE_UNRESOLVED")
            # No endpoint repair or alternate proposal. The checker never treats
            # this nominal binary64 iterate as the actually serialized increment.
            r = list(p)
        return finish("NUMERICALLY_UNRESOLVED", "FIXED_ITERATION_LIMIT")
    except Exception as error:  # noqa: BLE001 - a prototype fault never admits a point.
        reason = (
            "COMBINED_TEN_SECOND_DEADLINE"
            if isinstance(error, TimeoutError)
            else type(error).__name__
        )
        result["exception_detail"] = str(error)[:256]
        return finish("NUMERICALLY_UNRESOLVED", reason)
