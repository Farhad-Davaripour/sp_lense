"""Model-free math for paired-order analytic prompt-gradient steering.

The routines here do not load a model, render a prompt, or inspect an intervention
outcome.  They operate on baseline logits, residuals, gradients, and directional
derivatives supplied by a runner.  In particular, they construct one common hidden
vector for two answer encodings and solve a one-dimensional, full-vocabulary local
feasibility problem before an intervention is evaluated.

This is privileged forced-choice steering machinery.  It does not identify a natural
self-preservation mechanism or an intrinsically selective static direction.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

SCHEMA_VERSION = "sp_lense.paired_order_analytic_gradient.v1"
DEFAULT_MINIMUM_ORDER_COSINE = -0.99
DEFAULT_MAXIMUM_RELATIVE_NORM = 0.10
DEFAULT_CAST_ABSOLUTE_TOLERANCE = 1e-6
DEFAULT_CAST_RELATIVE_TOLERANCE = 1e-5
DEFAULT_MARGIN_TOLERANCE = 1e-6


class PairedOrderGradientIneligible(ValueError):
    """A declared geometric or analytic construction gate failed closed."""

    def __init__(self, message: str, *, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _matrix(values: Any, *, field: str, rows: int | None = None) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{field} must be a non-empty two-dimensional matrix")
    if rows is not None and matrix.shape[0] != rows:
        raise ValueError(f"{field} must have exactly {rows} rows")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{field} must be finite")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _vector(
    values: Any,
    *,
    field: str,
    length: int | None = None,
    positive: bool = False,
) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if vector.ndim != 1 or vector.size < 1:
        raise ValueError(f"{field} must be a non-empty one-dimensional vector")
    if length is not None and vector.size != length:
        raise ValueError(f"{field} must contain exactly {length} values")
    if not np.isfinite(vector).all():
        raise ValueError(f"{field} must be finite")
    if positive and np.any(vector <= 0.0):
        raise ValueError(f"{field} must be strictly positive")
    return np.ascontiguousarray(vector, dtype=np.float64)


def _finite_scalar(value: Any, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a finite scalar")
    checked = float(value)
    if not math.isfinite(checked):
        raise ValueError(f"{field} must be a finite scalar")
    if positive and checked <= 0.0:
        raise ValueError(f"{field} must be strictly positive")
    return checked


def _token_ids(values: Any, *, field: str, vocabulary_size: int) -> tuple[int, int]:
    try:
        ids = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{field} must contain two token IDs") from exc
    if len(ids) != 2:
        raise ValueError(f"{field} must contain exactly two token IDs")
    output = []
    for token_id in ids:
        if isinstance(token_id, bool) or not isinstance(token_id, (int, np.integer)):
            raise TypeError(f"{field} values must be integers")
        checked = int(token_id)
        if not 0 <= checked < vocabulary_size:
            raise ValueError(f"{field} contains an out-of-vocabulary token ID")
        output.append(checked)
    return output[0], output[1]


def _cosine_floor(value: Any) -> float:
    checked = _finite_scalar(value, field="minimum_order_cosine")
    if not -1.0 < checked < 1.0:
        raise ValueError("minimum_order_cosine must lie strictly between -1 and 1")
    return checked


@dataclass(frozen=True)
class CommonGradientBisector:
    """One common orientation and physical base vector for two answer orders."""

    direction: np.ndarray
    common_scale: float
    base_vector: np.ndarray
    semantic_slopes: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class AnalyticDoseInterval:
    """The common non-negative dose interval satisfying all local constraints."""

    lower: float
    upper: float
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class CastRecertification:
    """The exact float32 delta plus its post-cast local certificate."""

    delta: np.ndarray
    diagnostics: dict[str, Any]


def construct_common_gradient_bisector(
    order_gradients: Any,
    residual_norms: Any,
    *,
    minimum_order_cosine: float = DEFAULT_MINIMUM_ORDER_COSINE,
    identity_tolerance: float = 1e-10,
) -> CommonGradientBisector:
    """Construct the unit bisector of two preserve-minus-comply gradients.

    ``order_gradients`` must already use semantic preserve-minus-comply orientation.
    A geometric-mean residual scale makes ``base_vector`` symmetric in answer order.
    The caller can later cast one selected multiple of that vector once and apply the
    exact same bytes to both encodings.
    """

    gradients = _matrix(order_gradients, field="order_gradients", rows=2)
    residuals = _vector(
        residual_norms, field="residual_norms", length=2, positive=True
    )
    floor = _cosine_floor(minimum_order_cosine)
    tolerance = _finite_scalar(identity_tolerance, field="identity_tolerance", positive=True)

    gradient_norms = np.linalg.norm(gradients, axis=1)
    if not np.isfinite(gradient_norms).all() or np.any(gradient_norms <= 0.0):
        raise PairedOrderGradientIneligible(
            "an order gradient has zero or non-finite norm",
            diagnostics={
                "schema_version": SCHEMA_VERSION,
                "failure": "invalid_order_gradient_norm",
                "gradient_norms": gradient_norms.tolist(),
            },
        )
    unit = gradients / gradient_norms[:, None]
    raw_cosine = float(unit[0] @ unit[1])
    if raw_cosine < -1.0 - tolerance or raw_cosine > 1.0 + tolerance:
        raise PairedOrderGradientIneligible(
            "order-gradient cosine lies outside its numerical range",
            diagnostics={
                "schema_version": SCHEMA_VERSION,
                "failure": "invalid_order_gradient_cosine",
                "raw_order_gradient_cosine": raw_cosine,
            },
        )
    cosine = min(1.0, max(-1.0, raw_cosine))
    if cosine <= floor:
        raise PairedOrderGradientIneligible(
            "order gradients fail the minimum cosine gate",
            diagnostics={
                "schema_version": SCHEMA_VERSION,
                "failure": "order_gradient_cosine_below_floor",
                "order_gradient_cosine": cosine,
                "minimum_order_cosine": floor,
            },
        )

    summed = unit[0] + unit[1]
    summed_norm = float(np.linalg.norm(summed))
    if not math.isfinite(summed_norm) or summed_norm <= 0.0:
        raise PairedOrderGradientIneligible(
            "order-gradient bisector is numerically zero",
            diagnostics={
                "schema_version": SCHEMA_VERSION,
                "failure": "zero_bisector",
                "order_gradient_cosine": cosine,
            },
        )
    direction = np.ascontiguousarray(summed / summed_norm, dtype=np.float64)
    alignments = np.ascontiguousarray(unit @ direction, dtype=np.float64)
    expected_alignment = math.sqrt((1.0 + cosine) / 2.0)
    identity_error = float(np.max(np.abs(alignments - expected_alignment)))
    if identity_error > tolerance or np.any(alignments <= 0.0):
        raise PairedOrderGradientIneligible(
            "order-gradient bisector failed its analytic identity",
            diagnostics={
                "schema_version": SCHEMA_VERSION,
                "failure": "bisector_identity_failure",
                "order_gradient_cosine": cosine,
                "alignments": alignments.tolist(),
                "expected_alignment": expected_alignment,
                "maximum_identity_error": identity_error,
                "identity_tolerance": tolerance,
            },
        )

    common_scale = math.sqrt(float(residuals[0] * residuals[1]))
    base_vector = np.ascontiguousarray(common_scale * direction, dtype=np.float64)
    semantic_slopes = np.ascontiguousarray(gradients @ base_vector, dtype=np.float64)
    if not np.isfinite(semantic_slopes).all() or np.any(semantic_slopes <= 0.0):
        raise PairedOrderGradientIneligible(
            "common vector lacks a positive semantic slope in both orders",
            diagnostics={
                "schema_version": SCHEMA_VERSION,
                "failure": "nonpositive_semantic_slope",
                "semantic_slopes": semantic_slopes.tolist(),
            },
        )

    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "construction": "unit_semantic_gradient_bisector",
        "order_count": 2,
        "width": int(gradients.shape[1]),
        "order_gradients_float64_sha256": _array_sha256(gradients),
        "residual_norms_float64_sha256": _array_sha256(residuals),
        "gradient_norms": gradient_norms.tolist(),
        "order_gradient_cosine": cosine,
        "minimum_order_cosine": floor,
        "normalized_order_alignments": alignments.tolist(),
        "expected_bisector_alignment": expected_alignment,
        "maximum_bisector_identity_error": identity_error,
        "identity_tolerance": tolerance,
        "common_scale_rule": "geometric_mean_of_two_residual_norms",
        "common_scale": common_scale,
        "direction_float64_sha256": _array_sha256(direction),
        "base_vector_float64_sha256": _array_sha256(base_vector),
        "semantic_slopes": semantic_slopes.tolist(),
    }
    diagnostics["diagnostics_sha256"] = _canonical_sha256(diagnostics)
    return CommonGradientBisector(
        direction=direction,
        common_scale=common_scale,
        base_vector=base_vector,
        semantic_slopes=semantic_slopes,
        diagnostics=diagnostics,
    )


def _choice_inputs(
    baseline_logits: Any,
    logit_directional_derivatives: Any,
    preserve_token_ids: Any,
    comply_token_ids: Any,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
    logits = _matrix(baseline_logits, field="baseline_logits", rows=2)
    derivatives = _matrix(
        logit_directional_derivatives, field="logit_directional_derivatives", rows=2
    )
    if derivatives.shape != logits.shape:
        raise ValueError("logit_directional_derivatives must match baseline_logits")
    preserve = _token_ids(
        preserve_token_ids,
        field="preserve_token_ids",
        vocabulary_size=int(logits.shape[1]),
    )
    comply = _token_ids(
        comply_token_ids,
        field="comply_token_ids",
        vocabulary_size=int(logits.shape[1]),
    )
    if any(preserve[order] == comply[order] for order in range(2)):
        raise ValueError("preserve and comply token IDs must differ in each order")
    return logits, derivatives, preserve, comply


def full_vocabulary_bidirectional_interval(
    baseline_logits: Any,
    logit_directional_derivatives: Any,
    preserve_token_ids: Any,
    comply_token_ids: Any,
    *,
    reserve_logit: float,
    slope_tolerance: float | None = None,
) -> AnalyticDoseInterval:
    """Solve the exact one-dimensional local constraints for both signs and orders.

    The positive sign targets preservation and the negative sign targets compliance.
    Every target is constrained against every other vocabulary token.  The returned
    lower endpoint is the unique deterministic analytic dose rule; finite model
    outcomes are not an input.
    """

    logits, derivatives, preserve, comply = _choice_inputs(
        baseline_logits,
        logit_directional_derivatives,
        preserve_token_ids,
        comply_token_ids,
    )
    reserve = _finite_scalar(reserve_logit, field="reserve_logit", positive=True)
    if slope_tolerance is None:
        scale = max(1.0, float(np.max(np.abs(derivatives))))
        tolerance = 128.0 * np.finfo(np.float64).eps * scale
        tolerance_rule = "128_times_float64_epsilon_times_max_1_or_abs_derivative"
    else:
        tolerance = _finite_scalar(
            slope_tolerance, field="slope_tolerance", positive=True
        )
        tolerance_rule = "caller_supplied"

    lower = 0.0
    upper = math.inf
    lower_binding: dict[str, Any] | None = None
    upper_binding: dict[str, Any] | None = None
    constraint_count = 0
    near_zero_constraint_count = 0
    vocabulary_size = int(logits.shape[1])

    for order in range(2):
        for semantic_sign, target in ((1, preserve[order]), (-1, comply[order])):
            target_logit = float(logits[order, target])
            target_derivative = float(derivatives[order, target])
            for competitor in range(vocabulary_size):
                if competitor == target:
                    continue
                gap = target_logit - float(logits[order, competitor])
                slope = semantic_sign * (
                    target_derivative - float(derivatives[order, competitor])
                )
                required = reserve - gap
                constraint_count += 1
                common = {
                    "order": order,
                    "semantic_sign": semantic_sign,
                    "target_token_id": target,
                    "competitor_token_id": competitor,
                    "baseline_gap": gap,
                    "signed_directional_slope": slope,
                }
                if slope > tolerance:
                    bound = required / slope
                    if bound > lower:
                        lower = float(bound)
                        lower_binding = {**common, "bound": float(bound)}
                elif slope < -tolerance:
                    bound = required / slope
                    if bound < upper:
                        upper = float(bound)
                        upper_binding = {**common, "bound": float(bound)}
                else:
                    near_zero_constraint_count += 1
                    if gap < reserve:
                        raise PairedOrderGradientIneligible(
                            "a locally immovable vocabulary constraint misses the reserve",
                            diagnostics={
                                "schema_version": SCHEMA_VERSION,
                                "failure": "immovable_constraint",
                                "reserve_logit": reserve,
                                "slope_tolerance": tolerance,
                                **common,
                            },
                        )

    if not math.isfinite(lower):
        raise PairedOrderGradientIneligible(
            "analytic lower dose is non-finite",
            diagnostics={
                "schema_version": SCHEMA_VERSION,
                "failure": "nonfinite_lower_bound",
                "lower": lower,
            },
        )
    lower = max(0.0, lower)
    if lower <= 0.0:
        raise PairedOrderGradientIneligible(
            "bidirectional construction does not require a positive dose",
            diagnostics={
                "schema_version": SCHEMA_VERSION,
                "failure": "nonpositive_lower_bound",
                "lower": lower,
                "upper": None if math.isinf(upper) else upper,
            },
        )
    if upper < lower:
        raise PairedOrderGradientIneligible(
            "full-vocabulary analytic dose interval is empty",
            diagnostics={
                "schema_version": SCHEMA_VERSION,
                "failure": "empty_analytic_interval",
                "lower": lower,
                "upper": upper,
                "lower_binding": lower_binding,
                "upper_binding": upper_binding,
            },
        )

    minimum_margins = _minimum_target_margins(
        logits,
        lower * derivatives,
        preserve,
        comply,
    )
    minimum_at_lower = min(item["minimum_target_margin"] for item in minimum_margins)
    arithmetic_tolerance = 256.0 * np.finfo(np.float64).eps * max(
        1.0,
        float(np.max(np.abs(logits))),
        float(np.max(np.abs(lower * derivatives))),
        reserve,
    )
    if minimum_at_lower < reserve - arithmetic_tolerance:
        raise PairedOrderGradientIneligible(
            "analytic lower dose failed direct constraint recertification",
            diagnostics={
                "schema_version": SCHEMA_VERSION,
                "failure": "interval_recertification_failure",
                "lower": lower,
                "reserve_logit": reserve,
                "minimum_margin_at_lower": minimum_at_lower,
                "arithmetic_tolerance": arithmetic_tolerance,
            },
        )

    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "solver": "one_dimensional_full_vocabulary_bidirectional_interval",
        "order_count": 2,
        "vocabulary_size": vocabulary_size,
        "constraint_count": constraint_count,
        "near_zero_constraint_count": near_zero_constraint_count,
        "reserve_logit": reserve,
        "slope_tolerance": tolerance,
        "slope_tolerance_rule": tolerance_rule,
        "lower": lower,
        "upper": None if math.isinf(upper) else upper,
        "upper_is_unbounded": math.isinf(upper),
        "lower_binding": lower_binding,
        "upper_binding": upper_binding,
        "minimum_target_margins_at_lower": minimum_margins,
        "minimum_margin_at_lower": minimum_at_lower,
        "arithmetic_tolerance": arithmetic_tolerance,
        "baseline_logits_float64_sha256": _array_sha256(logits),
        "logit_directional_derivatives_float64_sha256": _array_sha256(derivatives),
        "preserve_token_ids": list(preserve),
        "comply_token_ids": list(comply),
    }
    diagnostics["diagnostics_sha256"] = _canonical_sha256(diagnostics)
    return AnalyticDoseInterval(lower=lower, upper=upper, diagnostics=diagnostics)


def _minimum_target_margins(
    logits: np.ndarray,
    positive_delta_logit_changes: np.ndarray,
    preserve: tuple[int, int],
    comply: tuple[int, int],
) -> list[dict[str, Any]]:
    rows = []
    vocabulary_size = int(logits.shape[1])
    for order in range(2):
        for semantic_sign, target in ((1, preserve[order]), (-1, comply[order])):
            changed = logits[order] + semantic_sign * positive_delta_logit_changes[order]
            competitors = np.arange(vocabulary_size) != target
            competitor_ids = np.flatnonzero(competitors)
            competitor_local_index = int(np.argmax(changed[competitors]))
            competitor = int(competitor_ids[competitor_local_index])
            margin = float(changed[target] - changed[competitor])
            rows.append(
                {
                    "order": order,
                    "semantic_sign": semantic_sign,
                    "target_token_id": target,
                    "strongest_competitor_token_id": competitor,
                    "minimum_target_margin": margin,
                }
            )
    return rows


def cast_and_recertify_common_delta(
    base_vector: Any,
    alpha: float,
    residual_norms: Any,
    baseline_logits: Any,
    base_logit_directional_derivatives: Any,
    cast_delta_logit_changes: Any,
    preserve_token_ids: Any,
    comply_token_ids: Any,
    *,
    reserve_logit: float,
    maximum_relative_norm: float = DEFAULT_MAXIMUM_RELATIVE_NORM,
    cast_absolute_tolerance: float = DEFAULT_CAST_ABSOLUTE_TOLERANCE,
    cast_relative_tolerance: float = DEFAULT_CAST_RELATIVE_TOLERANCE,
    margin_tolerance: float = DEFAULT_MARGIN_TOLERANCE,
) -> CastRecertification:
    """Cast one common delta to float32 and recertify its exact local JVP.

    ``cast_delta_logit_changes`` must be the runner's JVP for the returned float32
    delta itself, not merely ``alpha`` times the uncast base-vector JVP.  Supplying
    it explicitly avoids pretending that componentwise float32 rounding preserves
    an exact scalar multiple in hidden space.
    """

    vector = _vector(base_vector, field="base_vector")
    dose = _finite_scalar(alpha, field="alpha", positive=True)
    residuals = _vector(
        residual_norms, field="residual_norms", length=2, positive=True
    )
    maximum_norm = _finite_scalar(
        maximum_relative_norm, field="maximum_relative_norm", positive=True
    )
    absolute_tolerance = _finite_scalar(
        cast_absolute_tolerance, field="cast_absolute_tolerance", positive=True
    )
    relative_tolerance = _finite_scalar(
        cast_relative_tolerance, field="cast_relative_tolerance", positive=True
    )
    allowed_margin_error = _finite_scalar(
        margin_tolerance, field="margin_tolerance", positive=True
    )
    reserve = _finite_scalar(reserve_logit, field="reserve_logit", positive=True)

    logits, base_derivatives, preserve, comply = _choice_inputs(
        baseline_logits,
        base_logit_directional_derivatives,
        preserve_token_ids,
        comply_token_ids,
    )
    cast_changes = _matrix(
        cast_delta_logit_changes, field="cast_delta_logit_changes", rows=2
    )
    if cast_changes.shape != logits.shape:
        raise ValueError("cast_delta_logit_changes must match baseline_logits")

    requested = np.ascontiguousarray(dose * vector, dtype=np.float64)
    delta = np.ascontiguousarray(requested.astype(np.float32), dtype=np.float32)
    if not np.isfinite(delta).all() or float(np.linalg.norm(delta.astype(np.float64))) <= 0.0:
        raise PairedOrderGradientIneligible(
            "float32 cast produced an invalid common delta",
            diagnostics={
                "schema_version": SCHEMA_VERSION,
                "failure": "invalid_float32_delta",
                "alpha": dose,
            },
        )

    delta_norm = float(np.linalg.norm(delta.astype(np.float64)))
    relative_norms = np.ascontiguousarray(delta_norm / residuals, dtype=np.float64)
    if np.any(relative_norms > maximum_norm):
        raise PairedOrderGradientIneligible(
            "float32 common delta exceeds the relative-norm cap",
            diagnostics={
                "schema_version": SCHEMA_VERSION,
                "failure": "relative_norm_cap_exceeded",
                "alpha": dose,
                "delta_float32_sha256": _array_sha256(delta),
                "delta_norm": delta_norm,
                "relative_norms": relative_norms.tolist(),
                "maximum_relative_norm": maximum_norm,
            },
        )

    expected_changes = dose * base_derivatives
    cast_jvp_error = np.abs(cast_changes - expected_changes)
    maximum_expected_change = float(np.max(np.abs(expected_changes)))
    jvp_allowance = absolute_tolerance + relative_tolerance * maximum_expected_change
    maximum_jvp_error = float(np.max(cast_jvp_error))
    if maximum_jvp_error > jvp_allowance:
        raise PairedOrderGradientIneligible(
            "cast-delta JVP differs materially from the uncast local prediction",
            diagnostics={
                "schema_version": SCHEMA_VERSION,
                "failure": "cast_jvp_inconsistency",
                "alpha": dose,
                "delta_float32_sha256": _array_sha256(delta),
                "maximum_cast_jvp_error": maximum_jvp_error,
                "cast_jvp_allowance": jvp_allowance,
                "cast_absolute_tolerance": absolute_tolerance,
                "cast_relative_tolerance": relative_tolerance,
            },
        )

    margin_rows = _minimum_target_margins(
        logits,
        cast_changes,
        preserve,
        comply,
    )
    minimum_margin = min(item["minimum_target_margin"] for item in margin_rows)
    if minimum_margin < reserve - allowed_margin_error:
        raise PairedOrderGradientIneligible(
            "float32 common delta fails the requested local target reserve",
            diagnostics={
                "schema_version": SCHEMA_VERSION,
                "failure": "post_cast_margin_failure",
                "alpha": dose,
                "delta_float32_sha256": _array_sha256(delta),
                "reserve_logit": reserve,
                "margin_tolerance": allowed_margin_error,
                "minimum_target_margin": minimum_margin,
                "minimum_target_margins": margin_rows,
            },
        )

    cast_error = delta.astype(np.float64) - requested
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "certificate": "exact_float32_common_delta_local_jvp_recertification",
        "alpha": dose,
        "width": int(delta.size),
        "base_vector_float64_sha256": _array_sha256(vector),
        "requested_delta_float64_sha256": _array_sha256(requested),
        "delta_float32_sha256": _array_sha256(delta),
        "delta_norm": delta_norm,
        "residual_norms": residuals.tolist(),
        "relative_norms": relative_norms.tolist(),
        "maximum_relative_norm": maximum_norm,
        "cast_error_l2": float(np.linalg.norm(cast_error)),
        "base_logit_directional_derivatives_float64_sha256": _array_sha256(
            base_derivatives
        ),
        "cast_delta_logit_changes_float64_sha256": _array_sha256(cast_changes),
        "maximum_cast_jvp_error": maximum_jvp_error,
        "cast_jvp_allowance": jvp_allowance,
        "cast_absolute_tolerance": absolute_tolerance,
        "cast_relative_tolerance": relative_tolerance,
        "reserve_logit": reserve,
        "margin_tolerance": allowed_margin_error,
        "minimum_target_margins": margin_rows,
        "minimum_target_margin": minimum_margin,
        "same_float32_delta_required_for_both_orders": True,
    }
    diagnostics["diagnostics_sha256"] = _canonical_sha256(diagnostics)
    return CastRecertification(delta=delta, diagnostics=diagnostics)
