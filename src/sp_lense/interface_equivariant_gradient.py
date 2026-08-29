"""Model-free math for an answer-interface-equivariant gradient controller.

The controller applies one semantic construction rule to two answer encodings while
allowing the physical residual vector to transform with the encoding.  Both encodings
share one scalar coefficient selected from a joint full-vocabulary interval.  This is
a prompt-local white-box intervention, not a reusable representation or evidence of a
natural self-preservation mechanism.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from .paired_order_analytic_gradient import PairedOrderGradientIneligible

SCHEMA_VERSION = "sp_lense.interface_equivariant_gradient.v2"
DEFAULT_MAXIMUM_RELATIVE_NORM = 0.10
DEFAULT_CAST_ABSOLUTE_TOLERANCE = 1e-6
DEFAULT_CAST_RELATIVE_TOLERANCE = 1e-5
DEFAULT_MARGIN_TOLERANCE = 1e-6


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


def _matrix(values: Any, *, field: str, rows: int = 2) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if matrix.ndim != 2 or matrix.shape[0] != rows or matrix.shape[1] < 1:
        raise ValueError(f"{field} must be a {rows}-row non-empty matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{field} must be finite")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _positive_pair(values: Any, *, field: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if vector.shape != (2,) or not np.isfinite(vector).all() or np.any(vector <= 0.0):
        raise ValueError(f"{field} must contain two finite positive values")
    return np.ascontiguousarray(vector, dtype=np.float64)


def _positive_scalar(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a finite positive scalar")
    checked = float(value)
    if not math.isfinite(checked) or checked <= 0.0:
        raise ValueError(f"{field} must be a finite positive scalar")
    return checked


def _token_pair(values: Any, *, field: str, vocabulary_size: int) -> tuple[int, int]:
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{field} must contain two token IDs") from exc
    if len(raw) != 2:
        raise ValueError(f"{field} must contain two token IDs")
    output = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{field} values must be integers")
        checked = int(value)
        if not 0 <= checked < vocabulary_size:
            raise ValueError(f"{field} contains an out-of-vocabulary token ID")
        output.append(checked)
    return output[0], output[1]


@dataclass(frozen=True)
class InterfaceEquivariantField:
    """Two context-specific base vectors with one shared coefficient coordinate."""

    base_vectors: np.ndarray
    semantic_slopes: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class InterfaceEquivariantRecertification:
    """Two exact float32 deltas and their joint local certificate."""

    deltas: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class ExactHeadSharedAlphaCertificate:
    """A conservative finite RMSNorm-head certificate for one shared alpha."""

    alpha: float
    lower: float
    upper: float
    diagnostics: dict[str, Any]


def construct_interface_equivariant_field(
    order_gradients: Any, residual_norms: Any
) -> InterfaceEquivariantField:
    """Normalize each semantic gradient in its own answer-interface coordinates.

    The two physical vectors may be antipodal.  Equivariance is supplied by applying
    the same semantic orientation, normalization rule, and later scalar coefficient
    to each encoding; physical vector equality is deliberately not required.
    """

    gradients = _matrix(order_gradients, field="order_gradients")
    residuals = _positive_pair(residual_norms, field="residual_norms")
    gradient_norms = np.linalg.norm(gradients, axis=1)
    if not np.isfinite(gradient_norms).all() or np.any(gradient_norms <= 0.0):
        raise PairedOrderGradientIneligible(
            "an interface gradient has zero or non-finite norm",
            diagnostics={
                "schema_version": SCHEMA_VERSION,
                "failure": "invalid_interface_gradient_norm",
                "gradient_norms": gradient_norms.tolist(),
            },
        )
    unit = gradients / gradient_norms[:, None]
    cosine = float(np.clip(unit[0] @ unit[1], -1.0, 1.0))
    base_vectors = np.ascontiguousarray(residuals[:, None] * unit, dtype=np.float64)
    base_norms = np.linalg.norm(base_vectors, axis=1)
    normalization_error = float(np.max(np.abs(base_norms / residuals - 1.0)))
    semantic_slopes = np.ascontiguousarray(
        np.sum(gradients * base_vectors, axis=1), dtype=np.float64
    )
    if np.any(semantic_slopes <= 0.0) or not np.isfinite(semantic_slopes).all():
        raise PairedOrderGradientIneligible(
            "an interface base vector lacks positive semantic slope",
            diagnostics={
                "schema_version": SCHEMA_VERSION,
                "failure": "nonpositive_interface_semantic_slope",
                "semantic_slopes": semantic_slopes.tolist(),
            },
        )
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "construction": "per_interface_unit_semantic_gradient_times_local_residual_norm",
        "order_count": 2,
        "width": int(gradients.shape[1]),
        "physical_vector_equality_required": False,
        "shared_scalar_coordinate_required": True,
        "order_gradient_cosine_descriptive": cosine,
        "gradient_norms": gradient_norms.tolist(),
        "residual_norms": residuals.tolist(),
        "base_norms": base_norms.tolist(),
        "maximum_residual_relative_normalization_error": normalization_error,
        "semantic_slopes": semantic_slopes.tolist(),
        "order_gradients_float64_sha256": _array_sha256(gradients),
        "base_vectors_float64_sha256": _array_sha256(base_vectors),
    }
    diagnostics["diagnostics_sha256"] = _canonical_sha256(diagnostics)
    return InterfaceEquivariantField(
        base_vectors=base_vectors,
        semantic_slopes=semantic_slopes,
        diagnostics=diagnostics,
    )


def construct_effective_unembedding_field(
    effective_unembedding: Any,
    residual_norms: Any,
    preserve_token_ids: Any,
    comply_token_ids: Any,
) -> InterfaceEquivariantField:
    """Construct the no-backward, answer-token-boundary comparator.

    ``effective_unembedding`` is ``RMSNorm.weight[:, None] * W_U`` with shape
    ``[residual_width, vocabulary_size]``.  This baseline receives the same target
    token IDs, residual norms, dose solver, and intervention site as the gradient
    ray.  It is therefore the required control for testing whether a last-block
    gradient contributes anything beyond the final answer-token boundary.
    """

    try:
        weights = np.asarray(effective_unembedding, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("effective_unembedding must be numeric") from exc
    if weights.ndim != 2 or min(weights.shape) < 1 or not np.isfinite(weights).all():
        raise ValueError("effective_unembedding must be a finite non-empty matrix")
    vocabulary_size = int(weights.shape[1])
    preserve = _token_pair(
        preserve_token_ids,
        field="preserve_token_ids",
        vocabulary_size=vocabulary_size,
    )
    comply = _token_pair(
        comply_token_ids,
        field="comply_token_ids",
        vocabulary_size=vocabulary_size,
    )
    if any(preserve[index] == comply[index] for index in range(2)):
        raise ValueError("preserve and comply token IDs must differ")
    token_boundaries = np.stack(
        [weights[:, preserve[index]] - weights[:, comply[index]] for index in range(2)]
    )
    field = construct_interface_equivariant_field(token_boundaries, residual_norms)
    diagnostics = dict(field.diagnostics)
    diagnostics.update(
        {
            "construction": "effective_unembedding_preserve_minus_comply_boundary",
            "requires_backward_pass": False,
            "effective_unembedding_float64_sha256": _array_sha256(weights),
            "preserve_token_ids": list(preserve),
            "comply_token_ids": list(comply),
        }
    )
    diagnostics.pop("diagnostics_sha256", None)
    diagnostics["diagnostics_sha256"] = _canonical_sha256(diagnostics)
    return InterfaceEquivariantField(
        base_vectors=field.base_vectors,
        semantic_slopes=field.semantic_slopes,
        diagnostics=diagnostics,
    )


def exact_rmsnorm_semantic_gradients(
    residuals: Any,
    effective_unembedding: Any,
    preserve_token_ids: Any,
    comply_token_ids: Any,
    *,
    rms_epsilon: float,
) -> np.ndarray:
    """Return the analytic final-head preserve-minus-comply gradients.

    This identity is the attribution control for a block-23 gradient.  If it agrees
    with autograd, that gradient is fully explained by RMSNorm's tangent projection
    of the preserve-vs-comply unembedding boundary; it is not independent evidence
    for a self-preservation representation.
    """

    try:
        residual_matrix = np.asarray(residuals, dtype=np.float64)
        weights = np.asarray(effective_unembedding, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("residuals and effective_unembedding must be numeric") from exc
    if residual_matrix.ndim != 2 or residual_matrix.shape[0] != 2:
        raise ValueError("residuals must be a two-row matrix")
    if weights.ndim != 2 or residual_matrix.shape[1] != weights.shape[0]:
        raise ValueError("effective_unembedding has the wrong residual width")
    if not np.isfinite(residual_matrix).all() or not np.isfinite(weights).all():
        raise ValueError("gradient inputs must be finite")
    epsilon = _positive_scalar(rms_epsilon, field="rms_epsilon")
    vocabulary_size = int(weights.shape[1])
    preserve = _token_pair(
        preserve_token_ids,
        field="preserve_token_ids",
        vocabulary_size=vocabulary_size,
    )
    comply = _token_pair(
        comply_token_ids,
        field="comply_token_ids",
        vocabulary_size=vocabulary_size,
    )
    boundaries = np.stack(
        [weights[:, preserve[order]] - weights[:, comply[order]] for order in range(2)]
    )
    return exact_rmsnorm_semantic_gradients_from_boundaries(
        residual_matrix,
        boundaries,
        rms_epsilon=epsilon,
    )


def exact_rmsnorm_semantic_gradients_from_boundaries(
    residuals: Any,
    semantic_token_boundaries: Any,
    *,
    rms_epsilon: float,
) -> np.ndarray:
    """Memory-bounded analytic gradient identity using two token boundaries."""

    residual_matrix = _matrix(residuals, field="residuals")
    boundaries = _matrix(
        semantic_token_boundaries,
        field="semantic_token_boundaries",
    )
    if residual_matrix.shape != boundaries.shape:
        raise ValueError("semantic token boundaries must match the residual shape")
    epsilon = _positive_scalar(rms_epsilon, field="rms_epsilon")
    width = int(residual_matrix.shape[1])
    gradients = []
    for order in range(2):
        residual = residual_matrix[order]
        boundary = boundaries[order]
        rho = math.sqrt(float(np.mean(residual**2)) + epsilon)
        numerator = float(residual @ boundary)
        gradient = boundary / rho - (numerator / (width * rho**3)) * residual
        gradients.append(gradient)
    output = np.ascontiguousarray(np.stack(gradients), dtype=np.float64)
    if not np.isfinite(output).all():
        raise ValueError("analytic semantic gradients are non-finite")
    return output


def _exact_head_inputs(
    residuals: Any,
    base_vectors: Any,
    effective_unembedding: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual_matrix = _matrix(residuals, field="residuals")
    vector_matrix = _matrix(base_vectors, field="base_vectors")
    try:
        weights = np.asarray(effective_unembedding, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("effective_unembedding must be numeric") from exc
    if weights.ndim != 2 or min(weights.shape) < 1 or not np.isfinite(weights).all():
        raise ValueError("effective_unembedding must be a finite non-empty matrix")
    if residual_matrix.shape != vector_matrix.shape:
        raise ValueError("residuals and base_vectors must have the same shape")
    if residual_matrix.shape[1] != weights.shape[0]:
        raise ValueError("effective_unembedding width differs from residual width")
    return residual_matrix, vector_matrix, np.ascontiguousarray(weights, dtype=np.float64)


def certify_exact_rmsnorm_head_shared_alpha(
    residuals: Any,
    base_vectors: Any,
    effective_unembedding: Any,
    preserve_token_ids: Any,
    comply_token_ids: Any,
    *,
    rms_epsilon: float,
    construction_reserve_logit: float,
    maximum_relative_norm: float = DEFAULT_MAXIMUM_RELATIVE_NORM,
) -> ExactHeadSharedAlphaCertificate:
    """Solve a deterministic full-vocabulary finite-head dose certificate.

    At the final block the remaining unbiased head is RMSNorm followed by the
    unembedding. Under the locked relative perturbation cap, the RMS denominator is
    bounded from the measured residual and direction norms. Each target-vs-competitor
    logit-margin requirement is therefore a sufficient linear inequality in one
    alpha. Intersecting every inequality for both semantic signs and both answer
    orders yields one shared dose without an outcome scan.

    The caller should use a construction reserve slightly larger than its later
    finite-precision acceptance reserve and must still recertify the actual model
    head after float32 casting.  This function assumes the unembedding bias is absent
    or identical for every token; the runtime must verify that architecture guard.
    """

    residual_matrix, vector_matrix, weights = _exact_head_inputs(
        residuals, base_vectors, effective_unembedding
    )
    residual_norms = np.linalg.norm(residual_matrix, axis=1)
    vector_norms = np.linalg.norm(vector_matrix, axis=1)
    if np.any(residual_norms <= 0.0) or not np.isfinite(residual_norms).all():
        raise ValueError("residuals must have finite positive norms")
    normalization_error = float(np.max(np.abs(vector_norms / residual_norms - 1.0)))
    if normalization_error > 1e-10:
        raise ValueError("base_vectors must each have their order's residual norm")
    baseline_numerators = residual_matrix @ weights
    direction_numerators = vector_matrix @ weights
    certificate = certify_exact_rmsnorm_head_shared_alpha_from_numerators(
        baseline_numerators,
        direction_numerators,
        residual_norms,
        vector_norms,
        preserve_token_ids,
        comply_token_ids,
        residual_width=int(residual_matrix.shape[1]),
        rms_epsilon=rms_epsilon,
        construction_reserve_logit=construction_reserve_logit,
        maximum_relative_norm=maximum_relative_norm,
    )
    diagnostics = dict(certificate.diagnostics)
    diagnostics.update(
        {
            "residuals_float64_sha256": _array_sha256(residual_matrix),
            "base_vectors_float64_sha256": _array_sha256(vector_matrix),
            "effective_unembedding_float64_sha256": _array_sha256(weights),
        }
    )
    diagnostics.pop("diagnostics_sha256", None)
    diagnostics["diagnostics_sha256"] = _canonical_sha256(diagnostics)
    return ExactHeadSharedAlphaCertificate(
        alpha=certificate.alpha,
        lower=certificate.lower,
        upper=certificate.upper,
        diagnostics=diagnostics,
    )


def certify_exact_rmsnorm_head_shared_alpha_from_numerators(
    baseline_numerators: Any,
    direction_numerators: Any,
    residual_norms: Any,
    direction_norms: Any,
    preserve_token_ids: Any,
    comply_token_ids: Any,
    *,
    residual_width: int,
    rms_epsilon: float,
    construction_reserve_logit: float,
    maximum_relative_norm: float = DEFAULT_MAXIMUM_RELATIVE_NORM,
) -> ExactHeadSharedAlphaCertificate:
    """Memory-bounded variant using precomputed numerator rows and actual norms."""

    baseline_matrix = _matrix(baseline_numerators, field="baseline_numerators")
    direction_matrix = _matrix(direction_numerators, field="direction_numerators")
    if baseline_matrix.shape != direction_matrix.shape:
        raise ValueError("numerator matrices must have the same shape")
    residual_values = _positive_pair(residual_norms, field="residual_norms")
    direction_values = _positive_pair(direction_norms, field="direction_norms")
    if isinstance(residual_width, bool) or not isinstance(residual_width, (int, np.integer)):
        raise TypeError("residual_width must be a positive integer")
    width = int(residual_width)
    if width <= 0:
        raise ValueError("residual_width must be a positive integer")
    epsilon = _positive_scalar(rms_epsilon, field="rms_epsilon")
    reserve = _positive_scalar(
        construction_reserve_logit,
        field="construction_reserve_logit",
    )
    maximum_norm = _positive_scalar(maximum_relative_norm, field="maximum_relative_norm")
    vocabulary_size = int(baseline_matrix.shape[1])
    preserve = _token_pair(
        preserve_token_ids,
        field="preserve_token_ids",
        vocabulary_size=vocabulary_size,
    )
    comply = _token_pair(
        comply_token_ids,
        field="comply_token_ids",
        vocabulary_size=vocabulary_size,
    )
    if any(preserve[index] == comply[index] for index in range(2)):
        raise ValueError("preserve and comply token IDs must differ")

    relative_alpha_caps = maximum_norm * residual_values / direction_values
    alpha_cap = float(min(maximum_norm, *relative_alpha_caps.tolist()))
    if not math.isfinite(alpha_cap) or alpha_cap <= 0.0:
        raise PairedOrderGradientIneligible(
            "exact-head relative-norm cap is non-finite or non-positive",
            diagnostics={
                "schema_version": SCHEMA_VERSION,
                "failure": "invalid_exact_head_relative_norm_cap",
            },
        )
    rho_maxima = np.sqrt((residual_values + alpha_cap * direction_values) ** 2 / width + epsilon)
    if not np.isfinite(rho_maxima).all():
        raise PairedOrderGradientIneligible(
            "exact-head RMS bound is non-finite",
            diagnostics={
                "schema_version": SCHEMA_VERSION,
                "failure": "nonfinite_exact_head_rms_bound",
            },
        )
    lower = 0.0
    upper = alpha_cap
    lower_binding: dict[str, Any] | None = None
    upper_binding: dict[str, Any] | None = None
    constraint_count = 0
    zero_slope_constraint_count = 0
    rows: list[dict[str, Any]] = []
    all_token_ids = np.arange(vocabulary_size)

    for order in range(2):
        for semantic_sign, target, target_name in (
            (1, preserve[order], "preserve"),
            (-1, comply[order], "comply"),
        ):
            competitors = all_token_ids[all_token_ids != target]
            with np.errstate(over="ignore", invalid="ignore"):
                baseline_margins = (
                    baseline_matrix[order, target] - baseline_matrix[order, competitors]
                )
                slopes = semantic_sign * (
                    direction_matrix[order, target] - direction_matrix[order, competitors]
                )
                right_sides = reserve * rho_maxima[order] - baseline_margins
            if not all(
                np.isfinite(values).all() for values in (baseline_margins, slopes, right_sides)
            ):
                raise PairedOrderGradientIneligible(
                    "an exact-head constraint overflowed or became non-finite",
                    diagnostics={
                        "schema_version": SCHEMA_VERSION,
                        "failure": "nonfinite_derived_exact_head_constraint",
                        "order": order,
                        "semantic_sign": semantic_sign,
                        "target_token_id": int(target),
                    },
                )
            local_lower = 0.0
            local_upper = alpha_cap
            local_lower_id: int | None = None
            local_upper_id: int | None = None
            for competitor, slope, right_side in zip(
                competitors.tolist(), slopes.tolist(), right_sides.tolist(), strict=True
            ):
                constraint_count += 1
                if slope > 0.0:
                    candidate = right_side / slope
                    if not math.isfinite(candidate) or (right_side != 0.0 and candidate == 0.0):
                        raise PairedOrderGradientIneligible(
                            "an exact-head lower bound is non-finite or unrepresentable",
                            diagnostics={
                                "schema_version": SCHEMA_VERSION,
                                "failure": "nonfinite_exact_head_interval_bound",
                            },
                        )
                    if right_side != 0.0:
                        candidate = float(np.nextafter(candidate, math.inf))
                    if candidate > local_lower:
                        local_lower = candidate
                        local_lower_id = int(competitor)
                elif slope < 0.0:
                    candidate = right_side / slope
                    if not math.isfinite(candidate) or (right_side != 0.0 and candidate == 0.0):
                        raise PairedOrderGradientIneligible(
                            "an exact-head upper bound is non-finite or unrepresentable",
                            diagnostics={
                                "schema_version": SCHEMA_VERSION,
                                "failure": "nonfinite_exact_head_interval_bound",
                            },
                        )
                    if right_side != 0.0:
                        candidate = float(np.nextafter(candidate, -math.inf))
                    if candidate < local_upper:
                        local_upper = candidate
                        local_upper_id = int(competitor)
                else:
                    zero_slope_constraint_count += 1
                    if right_side > 0.0:
                        raise PairedOrderGradientIneligible(
                            "an exact-head target constraint has zero usable slope",
                            diagnostics={
                                "schema_version": SCHEMA_VERSION,
                                "failure": "zero_slope_infeasible_exact_head_constraint",
                                "order": order,
                                "semantic_sign": semantic_sign,
                                "target_token_id": int(target),
                                "competitor_token_id": int(competitor),
                                "right_side": float(right_side),
                            },
                        )
            row = {
                "order": order,
                "semantic_sign": semantic_sign,
                "target": target_name,
                "target_token_id": int(target),
                "local_lower": float(local_lower),
                "local_upper": float(local_upper),
                "lower_binding_competitor_token_id": local_lower_id,
                "upper_binding_competitor_token_id": local_upper_id,
                "slope_zero_policy": "only_exact_float64_zero_is_zero",
            }
            rows.append(row)
            if local_lower > lower:
                lower = local_lower
                lower_binding = row
            if local_upper < upper:
                upper = local_upper
                upper_binding = row

    clipped_lower = max(0.0, float(lower))
    clipped_upper = min(alpha_cap, float(upper))
    if (
        not math.isfinite(clipped_lower)
        or not math.isfinite(clipped_upper)
        or clipped_lower > clipped_upper
    ):
        raise PairedOrderGradientIneligible(
            "the two answer interfaces have no shared exact-head dose",
            diagnostics={
                "schema_version": SCHEMA_VERSION,
                "failure": "empty_shared_exact_head_interval",
                "lower": clipped_lower,
                "upper": clipped_upper,
                "maximum_relative_norm": maximum_norm,
                "maximum_alpha_from_relative_norm_cap": alpha_cap,
                "constraints": rows,
            },
        )
    alpha = clipped_lower
    minimum_constraint_surplus = math.inf
    for order in range(2):
        for semantic_sign, target in (
            (1, preserve[order]),
            (-1, comply[order]),
        ):
            competitors = all_token_ids[all_token_ids != target]
            with np.errstate(over="ignore", invalid="ignore"):
                baseline_margins = (
                    baseline_matrix[order, target] - baseline_matrix[order, competitors]
                )
                slopes = semantic_sign * (
                    direction_matrix[order, target] - direction_matrix[order, competitors]
                )
                surpluses = baseline_margins + alpha * slopes - reserve * rho_maxima[order]
            if not np.isfinite(surpluses).all():
                raise PairedOrderGradientIneligible(
                    "selected exact-head interval point is non-finite",
                    diagnostics={
                        "schema_version": SCHEMA_VERSION,
                        "failure": "nonfinite_selected_exact_head_constraint",
                    },
                )
            local_minimum = float(np.min(surpluses))
            minimum_constraint_surplus = min(minimum_constraint_surplus, local_minimum)
            if local_minimum < 0.0:
                raise PairedOrderGradientIneligible(
                    "selected exact-head interval point fails direct constraint verification",
                    diagnostics={
                        "schema_version": SCHEMA_VERSION,
                        "failure": "selected_exact_head_constraint_rounding_failure",
                        "order": order,
                        "semantic_sign": semantic_sign,
                        "minimum_constraint_surplus": local_minimum,
                    },
                )
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "certificate": "conservative_exact_rmsnorm_unembedding_full_vocabulary_shared_alpha",
        "assumption": "unembedding_bias_absent_or_vocabulary_constant_runtime_guard_required",
        "order_count": 2,
        "semantic_sign_count": 2,
        "width": width,
        "vocabulary_size": vocabulary_size,
        "constraint_count": constraint_count,
        "zero_slope_constraint_count": zero_slope_constraint_count,
        "construction_reserve_logit": reserve,
        "maximum_relative_norm": maximum_norm,
        "maximum_alpha_from_relative_norm_cap": alpha_cap,
        "rms_epsilon": epsilon,
        "rho_maxima": rho_maxima.tolist(),
        "residual_norms": residual_values.tolist(),
        "direction_norms": direction_values.tolist(),
        "lower": clipped_lower,
        "upper": clipped_upper,
        "selected_alpha": alpha,
        "minimum_constraint_surplus": minimum_constraint_surplus,
        "selection_rule": "smallest_nonnegative_value_in_joint_interval",
        "lower_binding": lower_binding,
        "upper_binding": upper_binding,
        "constraints": rows,
        "baseline_numerators_float64_sha256": _array_sha256(baseline_matrix),
        "direction_numerators_float64_sha256": _array_sha256(direction_matrix),
    }
    diagnostics["diagnostics_sha256"] = _canonical_sha256(diagnostics)
    return ExactHeadSharedAlphaCertificate(
        alpha=alpha,
        lower=clipped_lower,
        upper=clipped_upper,
        diagnostics=diagnostics,
    )


def exact_rmsnorm_unembedding_logits(
    residuals: Any,
    effective_unembedding: Any,
    *,
    rms_epsilon: float,
) -> np.ndarray:
    """Evaluate the bias-free final head in float64 for model-free recertification."""

    try:
        residual_matrix = np.asarray(residuals, dtype=np.float64)
        weights = np.asarray(effective_unembedding, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("residuals and effective_unembedding must be numeric") from exc
    if residual_matrix.ndim != 2 or residual_matrix.shape[0] < 1:
        raise ValueError("residuals must be a non-empty matrix")
    if weights.ndim != 2 or residual_matrix.shape[1] != weights.shape[0]:
        raise ValueError("effective_unembedding has the wrong residual width")
    if not np.isfinite(residual_matrix).all() or not np.isfinite(weights).all():
        raise ValueError("head inputs must be finite")
    epsilon = _positive_scalar(rms_epsilon, field="rms_epsilon")
    rho = np.sqrt(np.mean(residual_matrix**2, axis=1) + epsilon)
    return np.ascontiguousarray((residual_matrix @ weights) / rho[:, None])


def recertify_exact_head_deltas(
    residuals: Any,
    deltas: Any,
    effective_unembedding: Any,
    preserve_token_ids: Any,
    comply_token_ids: Any,
    *,
    rms_epsilon: float,
    acceptance_reserve_logit: float,
    maximum_relative_norm: float = DEFAULT_MAXIMUM_RELATIVE_NORM,
    margin_tolerance: float = DEFAULT_MARGIN_TOLERANCE,
) -> dict[str, Any]:
    """Recertify cast deltas under both exact signs in the finite bias-free head."""

    try:
        raw_deltas = np.asarray(deltas)
    except (TypeError, ValueError) as exc:
        raise ValueError("deltas must be numeric float32 values") from exc
    if raw_deltas.dtype != np.dtype(np.float32):
        raise TypeError("deltas must retain their exact float32 intervention dtype")
    if raw_deltas.ndim != 2 or raw_deltas.shape[0] != 2 or raw_deltas.shape[1] < 1:
        raise ValueError("deltas must be a two-row non-empty float32 matrix")
    if not np.isfinite(raw_deltas).all():
        raise ValueError("deltas must be finite")
    residual_matrix, delta_matrix, weights = _exact_head_inputs(
        residuals, raw_deltas, effective_unembedding
    )
    reserve = _positive_scalar(
        acceptance_reserve_logit,
        field="acceptance_reserve_logit",
    )
    allowed_error = _positive_scalar(margin_tolerance, field="margin_tolerance")
    maximum_norm = _positive_scalar(maximum_relative_norm, field="maximum_relative_norm")
    vocabulary_size = int(weights.shape[1])
    preserve = _token_pair(
        preserve_token_ids,
        field="preserve_token_ids",
        vocabulary_size=vocabulary_size,
    )
    comply = _token_pair(
        comply_token_ids,
        field="comply_token_ids",
        vocabulary_size=vocabulary_size,
    )
    if any(preserve[index] == comply[index] for index in range(2)):
        raise ValueError("preserve and comply token IDs must differ")
    residual_norms = np.linalg.norm(residual_matrix, axis=1)
    delta_norms = np.linalg.norm(delta_matrix, axis=1)
    if np.any(residual_norms <= 0.0) or not np.isfinite(residual_norms).all():
        raise ValueError("residuals must have finite positive norms")
    relative_norms = delta_norms / residual_norms
    if np.any(relative_norms > maximum_norm):
        raise PairedOrderGradientIneligible(
            "float32 deltas exceed the exact-head relative-norm cap",
            diagnostics={
                "schema_version": SCHEMA_VERSION,
                "failure": "exact_head_recertification_norm_cap_exceeded",
                "relative_norms": relative_norms.tolist(),
                "maximum_relative_norm": maximum_norm,
            },
        )
    rows: list[dict[str, Any]] = []
    all_token_ids = np.arange(vocabulary_size)
    for order in range(2):
        for semantic_sign, target, target_name in (
            (1, preserve[order], "preserve"),
            (-1, comply[order], "comply"),
        ):
            changed_residual = residual_matrix[order] + semantic_sign * delta_matrix[order]
            logits = exact_rmsnorm_unembedding_logits(
                changed_residual[None, :],
                weights,
                rms_epsilon=rms_epsilon,
            )[0]
            competitors = all_token_ids[all_token_ids != target]
            strongest = int(competitors[int(np.argmax(logits[competitors]))])
            margin = float(logits[target] - logits[strongest])
            rows.append(
                {
                    "order": order,
                    "semantic_sign": semantic_sign,
                    "target": target_name,
                    "target_token_id": int(target),
                    "argmax_token_id": int(np.argmax(logits)),
                    "strongest_competitor_token_id": strongest,
                    "minimum_target_margin": margin,
                    "target_met": bool(
                        int(np.argmax(logits)) == int(target) and margin >= reserve - allowed_error
                    ),
                    "changed_logits_float64_sha256": _array_sha256(logits),
                }
            )
    minimum_margin = min(float(row["minimum_target_margin"]) for row in rows)
    if not all(bool(row["target_met"]) for row in rows):
        raise PairedOrderGradientIneligible(
            "cast deltas fail exact finite-head recertification",
            diagnostics={
                "schema_version": SCHEMA_VERSION,
                "failure": "exact_finite_head_recertification_failure",
                "acceptance_reserve_logit": reserve,
                "margin_tolerance": allowed_error,
                "minimum_target_margin": minimum_margin,
                "rows": rows,
            },
        )
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "certificate": "finite_bias_free_rmsnorm_unembedding_both_orders_both_signs",
        "acceptance_reserve_logit": reserve,
        "margin_tolerance": allowed_error,
        "minimum_target_margin": minimum_margin,
        "maximum_relative_norm": maximum_norm,
        "delta_norms": delta_norms.tolist(),
        "residual_norms": residual_norms.tolist(),
        "relative_norms": relative_norms.tolist(),
        "rows": rows,
        "residuals_float64_sha256": _array_sha256(residual_matrix),
        "deltas_float32_sha256": _array_sha256(np.ascontiguousarray(raw_deltas)),
        "negative_deltas_float32_sha256": _array_sha256(
            np.ascontiguousarray(-raw_deltas, dtype=np.float32)
        ),
        "effective_unembedding_float64_sha256": _array_sha256(weights),
    }
    diagnostics["diagnostics_sha256"] = _canonical_sha256(diagnostics)
    return diagnostics


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
            competitor_ids = np.flatnonzero(np.arange(vocabulary_size) != target)
            competitor = int(competitor_ids[int(np.argmax(changed[competitor_ids]))])
            rows.append(
                {
                    "order": order,
                    "semantic_sign": semantic_sign,
                    "target_token_id": target,
                    "strongest_competitor_token_id": competitor,
                    "minimum_target_margin": float(changed[target] - changed[competitor]),
                }
            )
    return rows


def cast_and_recertify_interface_deltas(
    base_vectors: Any,
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
) -> InterfaceEquivariantRecertification:
    """Cast two interface-specific deltas and recertify their shared coefficient."""

    vectors = _matrix(base_vectors, field="base_vectors")
    residuals = _positive_pair(residual_norms, field="residual_norms")
    logits = _matrix(baseline_logits, field="baseline_logits")
    derivatives = _matrix(
        base_logit_directional_derivatives,
        field="base_logit_directional_derivatives",
    )
    cast_changes = _matrix(cast_delta_logit_changes, field="cast_delta_logit_changes")
    if derivatives.shape != logits.shape or cast_changes.shape != logits.shape:
        raise ValueError("all logit matrices must share one shape")
    preserve = _token_pair(
        preserve_token_ids,
        field="preserve_token_ids",
        vocabulary_size=int(logits.shape[1]),
    )
    comply = _token_pair(
        comply_token_ids,
        field="comply_token_ids",
        vocabulary_size=int(logits.shape[1]),
    )
    if any(preserve[index] == comply[index] for index in range(2)):
        raise ValueError("preserve and comply token IDs must differ")
    dose = _positive_scalar(alpha, field="alpha")
    reserve = _positive_scalar(reserve_logit, field="reserve_logit")
    maximum_norm = _positive_scalar(maximum_relative_norm, field="maximum_relative_norm")
    absolute_tolerance = _positive_scalar(cast_absolute_tolerance, field="cast_absolute_tolerance")
    relative_tolerance = _positive_scalar(cast_relative_tolerance, field="cast_relative_tolerance")
    allowed_margin_error = _positive_scalar(margin_tolerance, field="margin_tolerance")

    base_norms = np.linalg.norm(vectors, axis=1)
    base_relative_norms = base_norms / residuals
    if float(np.max(np.abs(base_relative_norms - 1.0))) > 1e-10:
        raise ValueError("base_vectors must each have their order's residual norm")
    requested = np.ascontiguousarray(dose * vectors, dtype=np.float64)
    deltas = np.ascontiguousarray(requested.astype(np.float32), dtype=np.float32)
    delta_norms = np.linalg.norm(deltas.astype(np.float64), axis=1)
    relative_norms = delta_norms / residuals
    if not np.isfinite(deltas).all() or np.any(delta_norms <= 0.0):
        raise PairedOrderGradientIneligible(
            "float32 cast produced an invalid interface delta",
            diagnostics={
                "schema_version": SCHEMA_VERSION,
                "failure": "invalid_float32_interface_delta",
                "alpha": dose,
            },
        )
    if np.any(relative_norms > maximum_norm):
        raise PairedOrderGradientIneligible(
            "an interface delta exceeds the relative-norm cap",
            diagnostics={
                "schema_version": SCHEMA_VERSION,
                "failure": "relative_norm_cap_exceeded",
                "alpha": dose,
                "relative_norms": relative_norms.tolist(),
                "maximum_relative_norm": maximum_norm,
            },
        )

    expected_changes = dose * derivatives
    maximum_expected_change = float(np.max(np.abs(expected_changes)))
    jvp_allowance = absolute_tolerance + relative_tolerance * maximum_expected_change
    maximum_jvp_error = float(np.max(np.abs(cast_changes - expected_changes)))
    if maximum_jvp_error > jvp_allowance:
        raise PairedOrderGradientIneligible(
            "cast interface JVP differs materially from the shared-scalar prediction",
            diagnostics={
                "schema_version": SCHEMA_VERSION,
                "failure": "cast_jvp_inconsistency",
                "alpha": dose,
                "maximum_cast_jvp_error": maximum_jvp_error,
                "cast_jvp_allowance": jvp_allowance,
            },
        )

    margins = _minimum_target_margins(logits, cast_changes, preserve, comply)
    minimum_margin = min(row["minimum_target_margin"] for row in margins)
    if minimum_margin < reserve - allowed_margin_error:
        raise PairedOrderGradientIneligible(
            "float32 interface deltas fail the requested local target reserve",
            diagnostics={
                "schema_version": SCHEMA_VERSION,
                "failure": "post_cast_margin_failure",
                "alpha": dose,
                "reserve_logit": reserve,
                "margin_tolerance": allowed_margin_error,
                "minimum_target_margin": minimum_margin,
                "minimum_target_margins": margins,
            },
        )

    cast_error = deltas.astype(np.float64) - requested
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "certificate": "two_interface_float32_deltas_one_shared_scalar_local_jvp",
        "alpha": dose,
        "width": int(vectors.shape[1]),
        "physical_vector_equality_required": False,
        "shared_scalar_coordinate_required": True,
        "base_vectors_float64_sha256": _array_sha256(vectors),
        "requested_deltas_float64_sha256": _array_sha256(requested),
        "deltas_float32_sha256": _array_sha256(deltas),
        "per_order_delta_float32_sha256": [
            _array_sha256(np.ascontiguousarray(row)) for row in deltas
        ],
        "delta_norms": delta_norms.tolist(),
        "residual_norms": residuals.tolist(),
        "relative_norms": relative_norms.tolist(),
        "maximum_relative_norm": maximum_norm,
        "per_order_cast_error_l2": np.linalg.norm(cast_error, axis=1).tolist(),
        "maximum_cast_jvp_error": maximum_jvp_error,
        "cast_jvp_allowance": jvp_allowance,
        "reserve_logit": reserve,
        "margin_tolerance": allowed_margin_error,
        "minimum_target_margins": margins,
        "minimum_target_margin": minimum_margin,
    }
    diagnostics["diagnostics_sha256"] = _canonical_sha256(diagnostics)
    return InterfaceEquivariantRecertification(
        deltas=deltas,
        diagnostics=diagnostics,
    )
