"""Independent exact-rational admission for the standalone partial-deficit prototype.

No solver imports, model imports, external evidence reads, or candidate repairs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import time
from dataclasses import dataclass
from fractions import Fraction as F

PAIRS = ((2, 0), (3, 1), (6, 4), (7, 5), (10, 8), (11, 9))
STEP, NET, PATH = F(1, 20), F(1, 5), F(2, 5)
NORM_GRID = 2**80


class DeadlineExceeded(TimeoutError):
    pass


def _tick(deadline):
    if time.monotonic() >= deadline:
        raise DeadlineExceeded("combined_solve_certificate_deadline")


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def problem_sha256(problem):
    raw = json.dumps(problem, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def vector_sha256(values):
    return hashlib.sha256(struct.pack("<" + "d" * len(values), *values)).hexdigest()


def _vector(values, dimension, deadline):
    _require(isinstance(values, (list, tuple)) and len(values) == dimension, "vector_dimension")
    result = []
    for index, value in enumerate(values):
        if index % 64 == 0:
            _tick(deadline)
        _require(type(value) in (float, int), "numeric_binary64_required")
        value = float(value)
        _require(math.isfinite(value), "nonfinite_input_or_witness")
        result.append(F.from_float(value))
    return tuple(result)


def _dot(a, b, deadline):
    total = F(0)
    for index, (x, y) in enumerate(zip(a, b, strict=True)):
        if index % 64 == 0:
            _tick(deadline)
        total += x * y
    return total


def _norm_upper(square):
    _require(square >= 0, "negative_norm_square")
    k = math.isqrt(square.numerator * NORM_GRID**2 // square.denominator)
    if k * k * square.denominator < square.numerator * NORM_GRID**2:
        k += 1
    result = F(k, NORM_GRID)
    _require(result * result >= square, "norm_upper_construction")
    return result


def _upward_float(value):
    result = float(value)
    _require(math.isfinite(result), "lipschitz_bound_overflow")
    if F.from_float(result) < value:
        # Scalar error-bound rounding ONLY. Never modify a point or a radius.
        result = math.nextafter(result, math.inf)
    _require(math.isfinite(result) and F.from_float(result) >= value, "lipschitz_upper_bound")
    return result


def _objective(A, b, c, D, r, deadline):
    deficits = tuple(max(F(0), rhs - _dot(row, r, deadline)) for row, rhs in zip(A, b, strict=True))
    drift = tuple(cp + _dot(row, r, deadline) for cp, row in zip(c, D, strict=True))
    value = (
        _dot(r, r, deadline) / 2
        + _dot(deficits, deficits, deadline) / 24
        + _dot(drift, drift, deadline) / 48
    )
    gradient = []
    for j in range(len(r)):
        if j % 32 == 0:
            _tick(deadline)
        gradient.append(
            r[j]
            - sum((row[j] * z for row, z in zip(A, deficits, strict=True)), F(0)) / 12
            + sum((row[j] * z for row, z in zip(D, drift, strict=True)), F(0)) / 24
        )
    return value, tuple(gradient), deficits


@dataclass(frozen=True)
class _Sealed:
    A: tuple
    b: tuple
    c: tuple
    D: tuple
    w: tuple
    w_sha256: str
    problem_sha256: str
    path: F
    history_path_upper: F
    rho: F
    f0: F
    gradient_zero: tuple
    deadline: float


def prepare(problem, expected_problem_sha256, deadline):
    """Authenticate frozen synthetic inputs and precompute independent exact data."""
    deadline = float(deadline)
    _require(math.isfinite(deadline), "finite_deadline_required")
    deadline = min(deadline, time.monotonic() + 10.0)
    _tick(deadline)
    _require(isinstance(problem, dict), "problem_object")
    _require(problem_sha256(problem) == expected_problem_sha256, "problem_sha256_mismatch")
    _tick(deadline)
    _require(isinstance(problem.get("w"), list), "current_w_vector")
    dimension = len(problem["w"])
    _require(1 <= dimension <= 1024, "prototype_dimension_range")
    w = _vector(problem["w"], dimension, deadline)
    w_hash = vector_sha256([float(x) for x in problem["w"]])
    _require(problem.get("w_sha256") == w_hash, "current_w_sha256_mismatch")
    _require(isinstance(problem.get("A"), list) and len(problem["A"]) == 12, "twelve_rows")
    A = tuple(_vector(row, dimension, deadline) for row in problem["A"])
    b, c = _vector(problem["b"], 12, deadline), _vector(problem["c"], 6, deadline)
    D = tuple(tuple((x - y) / 2 for x, y in zip(A[i], A[j], strict=True)) for i, j in PAIRS)
    _tick(deadline)
    path_text = problem.get("path_upper")
    _require(
        isinstance(path_text, str)
        and re.fullmatch(r"-?\d{1,40}(?:/[1-9]\d{0,39})?", path_text, flags=re.ASCII) is not None,
        "bounded_rational_path_upper_required",
    )
    path = F(path_text)
    _require(0 <= path <= PATH, "path_upper_outside_original_cap")
    _require(_dot(w, w, deadline) <= NET * NET, "current_w_outside_net_ball")
    _require(_dot(w, w, deadline) <= path * path, "current_net_exceeds_path_upper")
    history = problem.get("history")
    _require(isinstance(history, list) and 1 <= len(history) <= 9, "bounded_history_required")
    history_exact = tuple(_vector(row, dimension, deadline) for row in history)
    _require(all(x == 0 for x in history_exact[0]), "history_not_fresh_zero")
    _require(
        vector_sha256([float(x) for x in history[-1]]) == w_hash, "history_endpoint_w_mismatch"
    )
    spent = F(0)
    for index, snapshot in enumerate(history_exact):
        _require(_dot(snapshot, snapshot, deadline) <= NET * NET, "historical_net_violation")
        if index:
            step = tuple(x - y for x, y in zip(snapshot, history_exact[index - 1], strict=True))
            square = _dot(step, step, deadline)
            _require(square <= STEP * STEP, "historical_step_violation")
            spent += _norm_upper(square)
            _tick(deadline)
    _require(spent <= path, "declared_path_not_conservative")
    rho = min(STEP, PATH - path)
    frobenius = sum((_dot(row, row, deadline) for row in A), F(0)) / 12
    frobenius += sum((_dot(row, row, deadline) for row in D), F(0)) / 24
    M = _upward_float(1 + frobenius)
    zero = (F(0),) * dimension
    f0, gradient_zero, _ = _objective(A, b, c, D, zero, deadline)
    sealed = _Sealed(
        A,
        b,
        c,
        D,
        w,
        w_hash,
        expected_problem_sha256,
        path,
        spent,
        rho,
        f0,
        gradient_zero,
        deadline,
    )
    # Float copies are iteration inputs only. Checker arithmetic uses immutable
    # tuples in sealed, not mutable solver copies or solver factors/derivatives.
    context = {
        "A": [[float(x) for x in row] for row in A],
        "b": [float(x) for x in b],
        "c": [float(x) for x in c],
        "D": [[float(x) for x in row] for row in D],
        "w": [float(x) for x in problem["w"]],
        "rho": float(rho),
        "M": M,
        "dimension": dimension,
        "deadline": deadline,
        "_sealed": sealed,
        "lipschitz_bound_exact": str(1 + frobenius),
    }
    _tick(deadline)
    return context


def check(context, w_next, u, v):
    """Bound admission at the EXACT displacement of the serialized endpoint."""
    sealed = context["_sealed"]
    deadline, dimension = sealed.deadline, len(sealed.w)
    _tick(deadline)
    endpoint = _vector(w_next, dimension, deadline)
    U, V = _vector(u, dimension, deadline), _vector(v, dimension, deadline)
    actual = tuple(x - y for x, y in zip(endpoint, sealed.w, strict=True))
    step_square = _dot(actual, actual, deadline)
    net_square = _dot(endpoint, endpoint, deadline)
    base = {
        "problem_sha256": sealed.problem_sha256,
        "current_w_sha256": sealed.w_sha256,
        "w_next_sha256": vector_sha256([float(x) for x in w_next]),
        "geometry": {
            "actual_step_squared": str(step_square),
            "net_squared": str(net_square),
            "rho": str(sealed.rho),
            "path_before_upper": str(sealed.path),
            "authenticated_history_path_upper": str(sealed.history_path_upper),
            "geometry_tolerance": "0",
        },
        "evaluated_exact_serialized_difference": True,
    }
    if step_square > sealed.rho**2 or net_square > NET**2:
        return {
            **base,
            "status": "NUMERICALLY_UNRESOLVED",
            "reason": "serialized_geometry_violation",
            "geometry_valid": False,
        }
    path_after = sealed.path + _norm_upper(step_square)
    base["geometry"]["path_after_upper"] = str(path_after)
    if path_after > PATH:
        return {
            **base,
            "status": "NUMERICALLY_UNRESOLVED",
            "reason": "serialized_path_upper_violation",
            "geometry_valid": False,
        }
    base["geometry_valid"] = True
    value, gradient, deficits = _objective(sealed.A, sealed.b, sealed.c, sealed.D, actual, deadline)
    epsilon = F(1, 2**30) * max(F(1), sealed.f0)
    gain = sealed.f0 - value
    is_zero = all(x == 0 for x in actual)
    if sealed.rho == 0 and is_zero:
        _tick(deadline)
        return {
            **base,
            "status": "EXACT_ZERO_OPTIMUM",
            "reason": "singleton_step_ball",
            "certificate_kind": "EXACT_SINGLETON_DOMAIN",
            "objective": str(value),
            "objective_zero": str(sealed.f0),
            "gain": str(gain),
            "epsilon": str(epsilon),
            "gap_bound": "0",
            "exact_zero_optimum": True,
        }
    alpha = _norm_upper(_dot(U, U, deadline))
    gamma = _norm_upper(_dot(V, V, deadline))
    delta_s = sealed.rho * alpha - _dot(U, actual, deadline)
    delta_n = NET * gamma - _dot(V, endpoint, deadline)
    _require(delta_s >= 0 and delta_n >= 0, "ball_support_bound_inconsistency")
    residual = tuple(g + x + y for g, x, y in zip(gradient, U, V, strict=True))
    residual_square = _dot(residual, residual, deadline)
    gap = delta_s + delta_n + residual_square / 2
    lower_bound = value - gap
    exact_zero = is_zero and gap == 0
    if exact_zero:
        status, reason = "EXACT_ZERO_OPTIMUM", "exact_zero_support_gap"
    elif gap <= epsilon and gain > epsilon:
        status, reason = "ADMITTED_PROGRESS", "certified_gap_and_gain"
    elif sealed.f0 - lower_bound <= epsilon:
        status, reason = "EPSILON_OPTIMAL_ZERO", "certified_zero_objective_epsilon_optimality"
    else:
        status, reason = "NOT_CERTIFIED", "gap_or_gain_not_certified"
    _tick(deadline)
    return {
        **base,
        "status": status,
        "reason": reason,
        "certificate_kind": "EXACT_RATIONAL_STRONG_CONVEXITY_SUPPORT",
        "objective": str(value),
        "objective_zero": str(sealed.f0),
        "gain": str(gain),
        "epsilon": str(epsilon),
        "gap_bound": str(gap),
        "objective_lower_bound": str(lower_bound),
        "zero_suboptimality_upper": str(sealed.f0 - lower_bound),
        "alpha_upper": str(alpha),
        "gamma_upper": str(gamma),
        "step_support_gap": str(delta_s),
        "net_support_gap": str(delta_n),
        "stationarity_squared": str(residual_square),
        "positive_deficit_count": sum(z > 0 for z in deficits),
        "maximum_deficit": str(max(deficits)),
        "exact_zero_optimum": exact_zero,
    }
