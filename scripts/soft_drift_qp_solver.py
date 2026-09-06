"""Standalone fixed soft-drift QP estimates; no model integration or checker oracle."""

from __future__ import annotations

import hashlib
import math
import struct
from collections import Counter

PAIRS = ((2, 0), (3, 1), (6, 4), (7, 5), (10, 8), (11, 9))
KAPPA = 1.0 / 24.0
RANK_FLOOR = 2.0**-26


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("finite real scalar required")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("finite binary64 required")
    return result


def _inputs(A, b, c):
    if not isinstance(A, (list, tuple)) or len(A) != 12:
        raise ValueError("exactly12 rows required")
    n = len(A[0])
    if not 1 <= n <= 1024 or any(len(row) != n for row in A) or len(b) != 12 or len(c) != 6:
        raise ValueError("bounded rectangular12-row/six-pair shape")
    return (
        [[_number(x) for x in row] for row in A],
        [_number(x) for x in b],
        [_number(x) for x in c],
    )


def fingerprint(A, b, c):
    A, b, c = _inputs(A, b, c)
    digest = hashlib.sha256(b"soft-drift-qp-v1\0" + struct.pack("<II", 12, len(A[0])))
    for values in (*A, b, c):
        for value in values:
            digest.update(struct.pack("<d", value))
    return digest.hexdigest()


def assemble(norms, gradients, margins, baseline_margins):
    """Fake-input usable own-norm conversion only; no runtime hooks or data reads."""
    if not all(len(x) == 12 for x in (norms, gradients, margins, baseline_margins)):
        raise ValueError("twelve own-norm rows")
    ns = [_number(x) for x in norms]
    ms, bs = ([_number(x) for x in values] for values in (margins, baseline_margins))
    if any(x <= 0 for x in ns):
        raise ValueError("positive own original norms")
    A = [[-n * _number(g) for g in row] for n, row in zip(ns, gradients, strict=True)]
    rhs = [0.10 - m for m in ms]
    c = [((ms[i] - bs[i]) - (ms[j] - bs[j])) / 2.0 for i, j in PAIRS]
    A, rhs, c = _inputs(A, rhs, c)
    return {"A": A, "b": rhs, "c": c}


def _dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def _tau(n):
    product = (n + 64) * 2.0**-53
    return 128.0 * product / (1.0 - product)


def _factor(matrix):
    """Fixed-order Cholesky after unit-diagonal equilibration; no rank repair."""
    count = len(matrix)
    if not count:
        return [], [], None
    if any(not math.isfinite(matrix[i][i]) or matrix[i][i] <= 0 for i in range(count)):
        return None
    scales = [math.sqrt(matrix[i][i]) for i in range(count)]
    lower = [[0.0] * count for _ in range(count)]
    pivots = []
    for i in range(count):
        for j in range(i + 1):
            value = math.fsum(
                [
                    matrix[i][j] / scales[i] / scales[j],
                    *(-lower[i][k] * lower[j][k] for k in range(j)),
                ]
            )
            if not math.isfinite(value):
                return None
            if i == j:
                if value <= RANK_FLOOR:
                    return None
                pivots.append(value)
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = value / lower[j][j]
    return scales, lower, min(pivots)


def _solve_factor(factor, rhs):
    scales, lower, _minimum = factor
    n = len(rhs)
    y, x = [0.0] * n, [0.0] * n
    for i in range(n):
        y[i] = (
            math.fsum([rhs[i] / scales[i], *(-lower[i][k] * y[k] for k in range(i))]) / lower[i][i]
        )
    for i in reversed(range(n)):
        x[i] = math.fsum([y[i], *(-lower[k][i] * x[k] for k in range(i + 1, n))]) / lower[i][i]
    result = [x[i] / scales[i] for i in range(n)]
    if not all(math.isfinite(v) for v in result):
        raise ArithmeticError("nonfinite small-system solution")
    return result


def _residuals(A, b, c, D, vector, multipliers):
    n = len(vector)
    slacks = [_dot(a, vector) - rhs for a, rhs in zip(A, b, strict=True)]
    terms = [math.fsum(abs(x * y) for x, y in zip(a, vector, strict=True)) for a in A]
    primal = max(
        max(0.0, -r) / max(1.0, abs(rhs), total)
        for r, rhs, total in zip(slacks, b, terms, strict=True)
    )
    common = [cp + _dot(dp, vector) for cp, dp in zip(c, D, strict=True)]
    common_abs = [
        abs(cp) + math.fsum(abs(x * y) for x, y in zip(dp, vector, strict=True))
        for cp, dp in zip(c, D, strict=True)
    ]
    stationarity = []
    for j in range(n):
        at = math.fsum(mu * a[j] for mu, a in zip(multipliers, A, strict=True))
        drift = KAPPA * math.fsum(dp[j] * cp for dp, cp in zip(D, common, strict=True))
        scale = max(
            1.0,
            abs(vector[j])
            + KAPPA * math.fsum(abs(dp[j]) * cp for dp, cp in zip(D, common_abs, strict=True))
            + math.fsum(abs(mu * a[j]) for mu, a in zip(multipliers, A, strict=True)),
        )
        stationarity.append(abs(math.fsum((vector[j], drift, -at))) / scale)
    comp = max(
        abs(mu * r) / max(1.0, abs(mu) * max(abs(rhs), total))
        for mu, r, rhs, total in zip(multipliers, slacks, b, terms, strict=True)
    )
    result = {
        "normalized_primal_violation": primal,
        "normalized_stationarity_max": max(stationarity),
        "normalized_complementarity_max": comp,
        "minimum_multiplier": min(multipliers),
        "raw_proposal_norm": math.sqrt(_dot(vector, vector)),
        "objective": 0.5 * _dot(vector, vector) + 0.5 * KAPPA * _dot(common, common),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ArithmeticError("nonfinite direct residual")
    return result


def solve(A, b, c):
    """Return at most one local numerical estimate, never a behavioral candidate."""
    trace = []
    result = {
        "status": "NUMERICALLY_UNRESOLVED",
        "solution": None,
        "reason": "NO_STRICT_ESTIMATE",
        "masks_visited": 0,
        "mask_trace": "",
        "mask_status_counts": {},
        "maximum_masks": 4096,
        "geometry_applied": False,
        "infeasibility_certified": False,
        "independent_checker_required": True,
    }
    try:
        A, b, c = _inputs(A, b, c)
        n = len(A[0])
        identity = fingerprint(A, b, c)
        D = [[(x - y) / 2.0 for x, y in zip(A[i], A[j], strict=True)] for i, j in PAIRS]
        M = [[float(i == j) + KAPPA * _dot(D[i], D[j]) for j in range(6)] for i in range(6)]
        factor = _factor(M)
        if factor is None:
            result["reason"] = "WOODBURY_RANK_GUARD"
            return result
        z = _solve_factor(factor, c)
        d0 = [-KAPPA * math.fsum(dp[j] * zp for dp, zp in zip(D, z, strict=True)) for j in range(n)]
        row_scales = [max(map(abs, a)) or 1.0 for a in A]
        E = [[x / scale for x in a] for a, scale in zip(A, row_scales, strict=True)]
        rhs = [bi / scale for bi, scale in zip(b, row_scales, strict=True)]
        inverse_rows = []
        for a in E:
            weights = _solve_factor(factor, [_dot(dp, a) for dp in D])
            inverse_rows.append(
                [
                    a[j] - KAPPA * math.fsum(dp[j] * p for dp, p in zip(D, weights, strict=True))
                    for j in range(n)
                ]
            )
        gram = [
            [(_dot(E[i], inverse_rows[j]) + _dot(E[j], inverse_rows[i])) / 2.0 for j in range(12)]
            for i in range(12)
        ]
        at_zero = [_dot(a, d0) for a in E]
        tau = _tau(n)
        result.update(
            dimension=n,
            input_sha256=identity,
            tolerance=tau,
            acceptance_band=tau / 4,
            rank_floor=RANK_FLOOR,
            woodbury_minimum_scaled_pivot=factor[2],
        )
        for mask in range(4096):
            trace.append("X")  # Count the current mask even if arithmetic raises.
            active = [i for i in range(12) if mask & (1 << i)]
            fact = _factor([[gram[i][j] for j in active] for i in active])
            if fact is None:
                trace[-1] = "R"
                continue
            lam = _solve_factor(fact, [rhs[i] - at_zero[i] for i in active])
            if any(value < 0 for value in lam):
                trace[-1] = "D"
                continue
            # Cheap full-primal screening before forming any native-dimensional vector.
            effects = [
                at_zero[i] + math.fsum(gram[i][j] * p for j, p in zip(active, lam, strict=True))
                for i in range(12)
            ]
            if any(
                not math.isfinite(v) or v < bi - (tau / 8) * max(1.0, abs(bi), abs(v))
                for v, bi in zip(effects, rhs, strict=True)
            ):
                trace[-1] = "P"
                continue
            vector = [
                d0[j] + math.fsum(inverse_rows[i][j] * p for i, p in zip(active, lam, strict=True))
                for j in range(n)
            ]
            multipliers = [0.0] * 12
            for i, value in zip(active, lam, strict=True):
                multipliers[i] = value / row_scales[i]
            metrics = _residuals(A, b, c, D, vector, multipliers)
            if (
                metrics["minimum_multiplier"] < 0
                or max(
                    metrics[key]
                    for key in (
                        "normalized_primal_violation",
                        "normalized_stationarity_max",
                        "normalized_complementarity_max",
                    )
                )
                > tau / 4
            ):
                trace[-1] = "F"
                continue
            trace[-1] = "C"
            result.update(
                status="KKT_ESTIMATE_ONLY",
                reason=None,
                solution={
                    "vector": vector,
                    "multipliers": multipliers,
                    "active_mask": mask,
                    "input_sha256": identity,
                },
                metrics=metrics,
                minimum_active_scaled_pivot=fact[2],
            )
            break
    except (ValueError, TypeError, OverflowError, ArithmeticError, IndexError) as error:
        result.update(
            status="NUMERICALLY_UNRESOLVED",
            solution=None,
            reason="INVALID_OR_NUMERIC_INPUT",
            error_type=type(error).__name__,
        )
    finally:
        result.update(
            masks_visited=len(trace),
            mask_trace="".join(trace),
            mask_status_counts=dict(sorted(Counter(trace).items())),
        )
    return result
