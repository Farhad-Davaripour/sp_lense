"""Independent outward-interval KKT checks for one supplied soft-drift QP solution.

Standard library only. This module neither searches for a solution nor imports
the production solver. A passing status is a fixed-tolerance numerical statement,
not exact feasibility, nonlinear behavior, or permission to freeze a candidate.
"""

from __future__ import annotations

import hashlib
import math
import struct
from decimal import ROUND_CEILING, ROUND_FLOOR, Context, Decimal, DecimalException

PRECISION = 80
PAIR_INDICES = ((2, 0), (3, 1), (6, 4), (7, 5), (10, 8), (11, 9))
SOLUTION_KEYS = frozenset({"vector", "multipliers", "active_mask", "input_sha256"})
PASS = "NUMERIC_KKT_WITHIN_TOLERANCE"
UNRESOLVED = "NUMERICALLY_UNRESOLVED"
_DOWN = Context(prec=PRECISION, rounding=ROUND_FLOOR, Emin=-999999, Emax=999999)
_UP = Context(prec=PRECISION, rounding=ROUND_CEILING, Emin=-999999, Emax=999999)
_ZERO, _ONE = Decimal(0), Decimal(1)


class _InvalidInput(ValueError):
    pass


class _Interval:
    """Closed finite real interval; every arithmetic endpoint rounds outward."""

    __slots__ = ("hi", "lo")

    def __init__(self, lo, hi=None):
        self.lo, self.hi = lo, lo if hi is None else hi
        if not (self.lo.is_finite() and self.hi.is_finite() and self.lo <= self.hi):
            raise ArithmeticError("invalid interval")

    @classmethod
    def integer(cls, value):
        return cls(Decimal(value))

    @classmethod
    def floating(cls, value):
        return cls(Decimal.from_float(value))

    def add(self, other):
        return _Interval(_DOWN.add(self.lo, other.lo), _UP.add(self.hi, other.hi))

    def subtract(self, other):
        return _Interval(_DOWN.subtract(self.lo, other.hi), _UP.subtract(self.hi, other.lo))

    def multiply(self, other):
        corners = (
            (self.lo, other.lo),
            (self.lo, other.hi),
            (self.hi, other.lo),
            (self.hi, other.hi),
        )
        return _Interval(
            min(_DOWN.multiply(a, b) for a, b in corners),
            max(_UP.multiply(a, b) for a, b in corners),
        )

    def divide(self, other):
        if other.lo <= 0 <= other.hi:
            raise ArithmeticError("interval denominator contains zero")
        reciprocal = _Interval(_DOWN.divide(_ONE, other.hi), _UP.divide(_ONE, other.lo))
        return self.multiply(reciprocal)

    def absolute(self):
        if self.lo >= 0:
            return self
        if self.hi <= 0:
            return _Interval(self.hi.copy_negate(), self.lo.copy_negate())
        return _Interval(_ZERO, max(self.lo.copy_abs(), self.hi.copy_abs()))

    def square(self):
        absolute = self.absolute()
        return _Interval(
            _DOWN.multiply(absolute.lo, absolute.lo), _UP.multiply(absolute.hi, absolute.hi)
        )

    def sqrt(self):
        if self.lo < 0:
            raise ArithmeticError("negative interval square-root endpoint")
        # Decimal.sqrt may use nearest rounding even with a directed context.
        # An extra representable step outward encloses its correctly rounded result.
        lo = _ZERO if not self.lo else _DOWN.next_minus(_DOWN.sqrt(self.lo))
        hi = _ZERO if not self.hi else _UP.next_plus(_UP.sqrt(self.hi))
        return _Interval(max(_ZERO, lo), hi)

    def maximum(self, other):
        return _Interval(max(self.lo, other.lo), max(self.hi, other.hi))

    def serialized(self):
        return {"lower": str(self.lo), "upper": str(self.hi)}


def _sum(values):
    result = _Interval.integer(0)
    for value in values:
        result = result.add(value)
    return result


def _dot(left, right):
    return _sum(a.multiply(b) for a, b in zip(left, right, strict=True))


def _squared_norm(vector):
    return _sum(x.square() for x in vector)


def _floats(value, length):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise _InvalidInput("SHAPE")
    if any(type(x) is not float or not math.isfinite(x) for x in value):
        raise _InvalidInput("FINITE_BINARY64_INPUTS_REQUIRED")
    return value


def _validate(A, b, c, solution):
    if not isinstance(A, (list, tuple)) or len(A) != 12:
        raise _InvalidInput("TWELVE_ROWS_REQUIRED")
    if not isinstance(A[0], (list, tuple)) or not 1 <= len(A[0]) <= 1024:
        raise _InvalidInput("DIMENSION")
    n = len(A[0])
    for row in A:
        _floats(row, n)
    _floats(b, 12)
    _floats(c, 6)
    if type(solution) is not dict or set(solution) != SOLUTION_KEYS:
        raise _InvalidInput("EXACT_SOLUTION_KEYS_REQUIRED")
    _floats(solution["vector"], n)
    _floats(solution["multipliers"], 12)
    mask = solution["active_mask"]
    if type(mask) is not int or not 0 <= mask < 4096:
        raise _InvalidInput("ACTIVE_MASK")
    claimed = solution["input_sha256"]
    if (
        type(claimed) is not str
        or len(claimed) != 64
        or any(char not in "0123456789abcdef" for char in claimed)
    ):
        raise _InvalidInput("INPUT_FINGERPRINT_FORMAT")
    for index, mu in enumerate(solution["multipliers"]):
        if mu < 0 or (not mask & (1 << index) and mu != 0):
            raise _InvalidInput("NONNEGATIVE_ACTIVE_ONLY_DUALS_REQUIRED")
    return n


def _fingerprint(A, b, c, n):
    digest = hashlib.sha256(b"soft-drift-qp-v1\0" + struct.pack("<II", 12, n))
    for row in A:
        digest.update(struct.pack("<" + "d" * n, *row))
    digest.update(struct.pack("<12d", *b))
    digest.update(struct.pack("<6d", *c))
    return digest.hexdigest()


def _active_rank(rows, active):
    """Independent normalized-row MGS: two total passes, one reorthogonalization."""
    basis, checks = [], []
    threshold = _Interval.integer(4).divide(_Interval.integer(2**26))
    for index in active:
        norm_squared = _squared_norm(rows[index])
        if norm_squared.lo <= 0:
            return False, checks, "ACTIVE_ZERO_OR_UNRESOLVED_NORM"
        length = norm_squared.sqrt()
        normalized = [x.divide(length) for x in rows[index]]
        original_squared = _squared_norm(normalized)
        residual = normalized
        for _pass in range(2):
            for q in basis:
                coefficient = _dot(q, residual)
                residual = [
                    x.subtract(coefficient.multiply(y)) for x, y in zip(residual, q, strict=True)
                ]
        residual_squared = _squared_norm(residual)
        ratio_lower = _DOWN.divide(residual_squared.lo, original_squared.hi)
        checks.append({"row_index": index, "squared_residual_ratio_lower": str(ratio_lower)})
        if ratio_lower <= threshold.hi:
            return False, checks, "ACTIVE_RANK_NOT_INTERIOR"
        residual_norm = residual_squared.sqrt()
        basis.append([x.divide(residual_norm) for x in residual])
    return True, checks, None


def _normalized_upper(numerator_upper, scale):
    if scale.lo < 1:
        raise ArithmeticError("normalization scale below one")
    return _UP.divide(numerator_upper, scale.lo)


def _verify(A, b, c, solution):
    n = _validate(A, b, c, solution)
    fingerprint = _fingerprint(A, b, c, n)
    if solution["input_sha256"] != fingerprint:
        raise _InvalidInput("INPUT_FINGERPRINT_MISMATCH")
    rows = [[_Interval.floating(x) for x in row] for row in A]
    rhs = [_Interval.floating(x) for x in b]
    drift = [_Interval.floating(x) for x in c]
    d = [_Interval.floating(x) for x in solution["vector"]]
    duals = [_Interval.floating(x) for x in solution["multipliers"]]
    one, two = _Interval.integer(1), _Interval.integer(2)
    kappa = one.divide(_Interval.integer(24))
    factor = _Interval.integer(n + 64).divide(_Interval.integer(2**53))
    gamma = factor.divide(one.subtract(factor))
    tau = _Interval.integer(128).multiply(gamma)
    interior = tau.divide(_Interval.integer(4))
    active = [index for index in range(12) if solution["active_mask"] & (1 << index)]
    rank_ok, rank_checks, rank_reason = _active_rank(rows, active)
    result = {
        "status": UNRESOLVED,
        "accepted": False,
        "reason": rank_reason,
        "dimension": n,
        "precision_digits": PRECISION,
        "input_sha256": fingerprint,
        "active_mask": solution["active_mask"],
        "active_rows": active,
        "active_rank_checks": rank_checks,
        "rank_squared_residual_ratio_required_strictly_above": "5.9604644775390625E-8",
        "rank_orthogonalization_passes": 2,
        "gamma": gamma.serialized(),
        "tau": tau.serialized(),
        "acceptance_threshold_tau_over_four": interior.serialized(),
        "exact_feasibility_certified": False,
        "nonlinear_behavior_certified": False,
        "candidate_claim": False,
    }
    if not rank_ok:
        return result
    D = [
        [x.subtract(y).divide(two) for x, y in zip(rows[ia], rows[ib], strict=True)]
        for ia, ib in PAIR_INDICES
    ]
    row_products = [[x.multiply(y) for x, y in zip(row, d, strict=True)] for row in rows]
    absolute_row_products = [_sum(x.absolute() for x in products) for products in row_products]
    slacks = [
        _sum(products).subtract(bound) for products, bound in zip(row_products, rhs, strict=True)
    ]
    primal_scales = [
        one.maximum(bound.absolute()).maximum(products)
        for bound, products in zip(rhs, absolute_row_products, strict=True)
    ]
    primal_violations = [
        _normalized_upper(max(_ZERO, slack.lo.copy_negate()), scale)
        for slack, scale in zip(slacks, primal_scales, strict=True)
    ]
    residual_drift = [cp.add(_dot(dp, d)) for cp, dp in zip(drift, D, strict=True)]
    drift_scale_terms = [
        cp.absolute().add(_sum(x.multiply(y).absolute() for x, y in zip(dp, d, strict=True)))
        for cp, dp in zip(drift, D, strict=True)
    ]
    stationarity_max = _ZERO
    for coordinate in range(n):
        regularizer = kappa.multiply(
            _sum(dp[coordinate].multiply(cp) for dp, cp in zip(D, residual_drift, strict=True))
        )
        dual_action = _sum(
            mu.multiply(row[coordinate]) for mu, row in zip(duals, rows, strict=True)
        )
        stationarity = d[coordinate].add(regularizer).subtract(dual_action)
        regularizer_scale = kappa.multiply(
            _sum(
                dp[coordinate].absolute().multiply(cp)
                for dp, cp in zip(D, drift_scale_terms, strict=True)
            )
        )
        dual_scale = _sum(
            mu.multiply(row[coordinate]).absolute() for mu, row in zip(duals, rows, strict=True)
        )
        scale = one.maximum(d[coordinate].absolute().add(regularizer_scale).add(dual_scale))
        stationarity_max = max(
            stationarity_max, _normalized_upper(stationarity.absolute().hi, scale)
        )
    complementarity_violations = []
    for mu, slack, bound, products in zip(duals, slacks, rhs, absolute_row_products, strict=True):
        complementarity = mu.multiply(slack)
        scale = one.maximum(mu.absolute().multiply(bound.absolute().maximum(products)))
        complementarity_violations.append(_normalized_upper(complementarity.absolute().hi, scale))
    primal_max = max(primal_violations)
    complementarity_max = max(complementarity_violations)
    largest = max(primal_max, stationarity_max, complementarity_max)
    passed = largest <= interior.lo
    result.update(
        {
            "status": PASS if passed else UNRESOLVED,
            "accepted": passed,
            "reason": None
            if passed
            else "MARGINAL_KKT_REJECTED"
            if largest <= tau.hi
            else "KKT_RESIDUAL_TOO_LARGE",
            "maximum_normalized_primal_violation_upper": str(primal_max),
            "maximum_normalized_stationarity_upper": str(stationarity_max),
            "maximum_normalized_complementarity_upper": str(complementarity_max),
            "maximum_normalized_kkt_upper": str(largest),
            "primal_slack_intervals": [slack.serialized() for slack in slacks],
            "multipliers": [str(mu.lo) for mu in duals],
            "normalization": "outward numerator upper divided by conservative scale lower",
        }
    )
    return result


def verify(A, b, c, solution):
    """Return a compact fail-closed interval audit; never search or repair inputs."""
    try:
        return _verify(A, b, c, solution)
    except _InvalidInput as error:
        return {
            "status": UNRESOLVED,
            "accepted": False,
            "reason": str(error),
            "precision_digits": PRECISION,
            "exact_feasibility_certified": False,
            "nonlinear_behavior_certified": False,
            "candidate_claim": False,
        }
    except (ArithmeticError, DecimalException, TypeError, KeyError, ValueError, struct.error):
        return {
            "status": UNRESOLVED,
            "accepted": False,
            "reason": "INTERVAL_ARITHMETIC_OR_STRUCTURE_UNRESOLVED",
            "precision_digits": PRECISION,
            "exact_feasibility_certified": False,
            "nonlinear_behavior_certified": False,
            "candidate_claim": False,
        }
