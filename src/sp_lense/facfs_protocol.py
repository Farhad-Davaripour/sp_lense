"""Outcome-free protocol primitives for the FACFS successor design.

This module deliberately has no model or tensor-library dependency.  It supports
protocol enumeration, exact small-sample power checks, projected-Fisher arithmetic,
and the blinded sign-code ceremony before a model-facing runner exists.
"""

from __future__ import annotations

import hashlib
import hmac
import math
from collections.abc import Sequence
from dataclasses import dataclass

CONDITIONS = ("SP", "OP", "ST", "OT")
SIGN_CONDITIONS = ("baseline", "positive", "negative")


@dataclass(frozen=True, slots=True)
class GeometryCell:
    """One repeated-measure cell in a Stage-G scenario orbit."""

    scenario_id: str
    condition: str
    assignment: int
    preservation_first: bool
    key_mapping: int
    alphabet: str

    @property
    def cell_id(self) -> str:
        order = "preserve_first" if self.preservation_first else "preserve_second"
        return (
            f"{self.scenario_id}:{self.condition}:assignment_{self.assignment}:"
            f"{order}:mapping_{self.key_mapping}:{self.alphabet}"
        )


def geometry_orbit(scenario_id: str, alphabets: Sequence[str]) -> tuple[GeometryCell, ...]:
    """Return the complete condition/assignment/order/mapping/alphabet orbit."""

    if not scenario_id:
        raise ValueError("scenario_id must be non-empty")
    names = tuple(alphabets)
    if (
        len(names) < 4
        or len(names) != len(set(names))
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ValueError("alphabets must contain at least four unique non-empty names")
    cells = tuple(
        GeometryCell(scenario_id, condition, assignment, first, mapping, alphabet)
        for condition in CONDITIONS
        for assignment in (0, 1)
        for first in (True, False)
        for mapping in (0, 1)
        for alphabet in names
    )
    if len({cell.cell_id for cell in cells}) != len(cells):  # pragma: no cover - invariant
        raise AssertionError("geometry cell IDs are not unique")
    return cells


def stage_g_sequence_items(
    scenario_count: int,
    alphabet_count: int,
    *,
    option_free_assignments: int = 2,
    option_free_completions: int = 2,
) -> int:
    """Count logical sequence items in the proposed Stage-G screen."""

    values = (
        scenario_count,
        alphabet_count,
        option_free_assignments,
        option_free_completions,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("all item-count inputs must be positive integers")
    if alphabet_count < 4:
        raise ValueError("authorizing Stage G requires at least four alphabet families")
    opaque = scenario_count * len(CONDITIONS) * 2 * 2 * 2 * alphabet_count
    option_free = scenario_count * option_free_assignments * option_free_completions
    return opaque + option_free


def binomial_cdf(n: int, maximum_successes: int, probability: float) -> float:
    """Exact binomial lower-tail probability for small protocol power calculations."""

    _validate_binomial(n, maximum_successes, probability)
    return math.fsum(
        math.comb(n, count)
        * probability**count
        * (1.0 - probability) ** (n - count)
        for count in range(maximum_successes + 1)
    )


def binomial_upper_tail(n: int, minimum_successes: int, probability: float) -> float:
    """Exact binomial upper-tail probability."""

    _validate_binomial(n, minimum_successes, probability)
    return math.fsum(
        math.comb(n, count)
        * probability**count
        * (1.0 - probability) ** (n - count)
        for count in range(minimum_successes, n + 1)
    )


def all_success_screen(
    n: int, *, null_rate: float, alternative_rate: float, alpha: float
) -> dict[str, float | int | bool]:
    """Summarize the exact all-success Stage-G decision rule."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    _validate_probability(null_rate, "null_rate")
    _validate_probability(alternative_rate, "alternative_rate")
    if alternative_rate <= null_rate:
        raise ValueError("alternative_rate must exceed null_rate")
    size = null_rate**n
    power = alternative_rate**n
    lower_bound = alpha ** (1.0 / n)
    return {
        "n": n,
        "required_successes": n,
        "size": size,
        "power": power,
        "all_success_clopper_pearson_lower": lower_bound,
        "rejects_at_alpha": size <= alpha,
    }


def low_fpr_screen(
    n: int,
    maximum_false_positives: int,
    *,
    null_fpr: float,
    alternative_fpr: float,
    alpha: float,
) -> dict[str, float | int | bool]:
    """Summarize an exact lower-tail test of a protected-family FPR."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    _validate_probability(null_fpr, "null_fpr")
    _validate_probability(alternative_fpr, "alternative_fpr")
    if alternative_fpr >= null_fpr:
        raise ValueError("alternative_fpr must be below null_fpr")
    size = binomial_cdf(n, maximum_false_positives, null_fpr)
    power = binomial_cdf(n, maximum_false_positives, alternative_fpr)
    return {
        "n": n,
        "maximum_false_positives": maximum_false_positives,
        "size": size,
        "power": power,
        "rejects_at_alpha": size <= alpha,
    }


def sign_code_commitment(key: bytes, manifest_sha256: str) -> str:
    """Commit an escrowed 256-bit key to one immutable manifest digest."""

    _validate_key(key)
    manifest = _decode_sha256(manifest_sha256)
    payload = b"FACFS-SIGN-v1\0" + manifest + key
    return hashlib.sha256(payload).hexdigest()


def sign_code_permutation(key: bytes, manifest_sha256: str, row_id: str) -> tuple[str, ...]:
    """Derive a deterministic unbiased permutation for one immutable row ID."""

    _validate_key(key)
    manifest = _decode_sha256(manifest_sha256)
    if not row_id:
        raise ValueError("row_id must be non-empty")
    values = list(SIGN_CONDITIONS)
    draw_index = 0
    for stop in range(len(values) - 1, 0, -1):
        selected, draw_index = _hmac_randbelow(
            key,
            manifest + b"\0" + row_id.encode("utf-8"),
            stop + 1,
            draw_index,
        )
        values[stop], values[selected] = values[selected], values[stop]
    return tuple(values)


def projected_fisher(
    probabilities: Sequence[float],
    directional_logit_derivatives: Sequence[Sequence[float]],
    *,
    probability_tolerance: float = 1e-12,
) -> tuple[tuple[float, ...], ...]:
    """Compute ``R.T @ (diag(p) - p p.T) @ R`` without materializing vocab Fisher."""

    p = tuple(float(value) for value in probabilities)
    rows = tuple(tuple(float(value) for value in row) for row in directional_logit_derivatives)
    if not p or len(p) != len(rows):
        raise ValueError("probabilities and derivative rows must have equal nonzero length")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("derivative rows must have one common positive width")
    if any(not math.isfinite(value) or value < 0.0 for value in p):
        raise ValueError("probabilities must be finite and nonnegative")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("derivatives must be finite")
    if not math.isclose(math.fsum(p), 1.0, rel_tol=0.0, abs_tol=probability_tolerance):
        raise ValueError("probabilities must sum to one")

    mean = [math.fsum(weight * row[column] for weight, row in zip(p, rows)) for column in range(width)]
    matrix = []
    for left in range(width):
        output_row = []
        for right in range(width):
            second = math.fsum(
                weight * row[left] * row[right] for weight, row in zip(p, rows)
            )
            output_row.append(second - mean[left] * mean[right])
        matrix.append(tuple(output_row))
    return tuple(matrix)


def quadratic_form(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> float:
    """Evaluate a square-matrix quadratic form in stable scalar arithmetic."""

    values = tuple(float(value) for value in vector)
    rows = tuple(tuple(float(value) for value in row) for row in matrix)
    if not values or len(rows) != len(values) or any(len(row) != len(values) for row in rows):
        raise ValueError("matrix must be square and match a non-empty vector")
    return math.fsum(
        values[left] * rows[left][right] * values[right]
        for left in range(len(values))
        for right in range(len(values))
    )


def compose_candidate(
    axis: Sequence[float],
    basis_columns: Sequence[Sequence[float]],
    beta: Sequence[float],
    coefficient: float,
) -> tuple[float, ...]:
    """Compose ``a d + B beta`` without normalizing the resulting vector."""

    d = tuple(float(value) for value in axis)
    columns = tuple(tuple(float(value) for value in column) for column in basis_columns)
    weights = tuple(float(value) for value in beta)
    if not d or len(columns) != len(weights) or any(len(column) != len(d) for column in columns):
        raise ValueError("axis, basis columns, and beta have incompatible shapes")
    flattened = (*d, *weights, coefficient, *(value for column in columns for value in column))
    if not all(math.isfinite(value) for value in flattened):
        raise ValueError("candidate inputs must be finite")
    return tuple(
        coefficient * d[index]
        + math.fsum(weight * column[index] for weight, column in zip(weights, columns))
        for index in range(len(d))
    )


def axis_projection_coefficient(axis: Sequence[float], vector: Sequence[float]) -> float:
    """Return ``d.T v / d.T d`` for a nonzero axis."""

    d = tuple(float(value) for value in axis)
    v = tuple(float(value) for value in vector)
    if not d or len(d) != len(v):
        raise ValueError("axis and vector must have equal nonzero length")
    denominator = math.fsum(value * value for value in d)
    if denominator <= 0.0:
        raise ValueError("axis must be nonzero")
    return math.fsum(left * right for left, right in zip(d, v)) / denominator


def _hmac_randbelow(key: bytes, domain: bytes, upper: int, draw_index: int) -> tuple[int, int]:
    if upper <= 0:
        raise ValueError("upper must be positive")
    limit = (1 << 64) - ((1 << 64) % upper)
    index = draw_index
    while True:
        digest = hmac.new(key, domain + index.to_bytes(8, "big"), hashlib.sha256).digest()
        value = int.from_bytes(digest[:8], "big")
        index += 1
        if value < limit:
            return value % upper, index


def _validate_binomial(n: int, count: int, probability: float) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= n:
        raise ValueError("count must be an integer in [0, n]")
    _validate_probability(probability, "probability")


def _validate_probability(value: float, label: str) -> None:
    if isinstance(value, bool) or not math.isfinite(float(value)) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be a finite probability")


def _validate_key(key: bytes) -> None:
    if not isinstance(key, bytes) or len(key) != 32:
        raise ValueError("key must contain exactly 32 bytes")


def _decode_sha256(value: str) -> bytes:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("manifest_sha256 must be a 64-character hex digest")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError("manifest_sha256 must be hexadecimal") from error
    if len(decoded) != 32:
        raise ValueError("manifest_sha256 must decode to 32 bytes")
    return decoded
