"""Pure mathematics for nuisance-null, minimum-Fisher gradient steering v3.

This module contains no model loading or evaluation code.  Callers provide
residual-scaled gradients and Fisher score-gradient factors.  All fitting uses
CPU float64 and only PyTorch linear algebra; the returned intervention direction
is contiguous CPU float32.

The fitted perturbation is prompt-adaptive.  It is a minimum-information
white-box attack under supplied first-order controls, not evidence for a natural
or reusable self-preservation direction.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_VERSION = "sp_lense.gradient_specificity_v3.v1"
# Captured Qwen gradients originate in float32.  The default rank cutoff therefore
# follows max(m, n) * eps_float32 for the 1,024-wide residual stream instead of using a
# misleading float64-scale threshold after the tensors are upcast for the solve.
DEFAULT_SVD_RTOL = 0.0001220703125
DEFAULT_SVD_ATOL = 1e-7
DEFAULT_PROBABILITY_TOLERANCE = 1e-7
# For the locked 1,024-wide float32 residual stream, this is
# gamma_1024(float32) + gamma_11(float64), where gamma_n = n*u/(1-n*u)
# and u is unit roundoff.  The float64 term covers the largest frozen
# top-8-plus-required-A/B-plus-tail partition.  Callers may override it
# explicitly for a different numerical contract.
DEFAULT_SCORE_IDENTITY_TOLERANCE = 6.103888176890726e-05
DEFAULT_CENTERED_SCORE_IDENTITY_TOLERANCE = 1e-12
DEFAULT_CONDITION_LIMIT = 1e12
DEFAULT_RESIDUAL_TOLERANCE = 1e-8


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value deterministically."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value deterministically."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def tensor_float32_sha256(value: Any) -> str:
    """Hash contiguous little-endian CPU float32 tensor bytes."""

    if not hasattr(value, "detach"):
        raise TypeError("value must be a tensor")
    array = value.detach().to(device="cpu").float().contiguous().numpy()
    return hashlib.sha256(array.astype("<f4", copy=False).tobytes(order="C")).hexdigest()


def tensor_float64_sha256(value: Any) -> str:
    """Hash contiguous little-endian CPU float64 tensor bytes."""

    if not hasattr(value, "detach"):
        raise TypeError("value must be a tensor")
    array = value.detach().to(device="cpu").double().contiguous().numpy()
    return hashlib.sha256(array.astype("<f8", copy=False).tobytes(order="C")).hexdigest()


def _finite_float(value: Any, *, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def _positive_float(value: Any, *, field: str) -> float:
    result = _finite_float(value, field=field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def _nonnegative_float(value: Any, *, field: str) -> float:
    result = _finite_float(value, field=field)
    if result < 0.0:
        raise ValueError(f"{field} must be nonnegative")
    return result


def _validated_ab_shift_parameters(
    *,
    baseline_conditional_probability: float,
    pair_probability_mass: float,
    target_semantic_log_odds: float,
) -> tuple[float, float, float, float, float, float]:
    probability = _finite_float(
        baseline_conditional_probability,
        field="baseline_conditional_probability",
    )
    if probability < 0.0 or probability > 1.0:
        raise ValueError("baseline_conditional_probability must be in [0, 1]")
    pair_mass = _positive_float(pair_probability_mass, field="pair_probability_mass")
    if pair_mass > 1.0:
        raise ValueError("pair_probability_mass must not exceed 1")
    target_log_odds = _finite_float(
        target_semantic_log_odds,
        field="target_semantic_log_odds",
    )

    def softplus(value: float) -> float:
        return max(value, 0.0) + math.log1p(math.exp(-abs(value)))

    log_target_probability = -softplus(-target_log_odds)
    log_target_complement = -softplus(target_log_odds)
    if target_log_odds >= 0.0:
        target_probability = 1.0 / (1.0 + math.exp(-target_log_odds))
    else:
        exponential = math.exp(target_log_odds)
        target_probability = exponential / (1.0 + exponential)
    return (
        probability,
        pair_mass,
        target_log_odds,
        target_probability,
        log_target_probability,
        log_target_complement,
    )


def minimum_baseline_to_steered_kl_for_ab_shift(
    *,
    baseline_conditional_probability: float,
    pair_probability_mass: float,
    target_semantic_log_odds: float,
) -> tuple[float, dict[str, Any]]:
    """Lower-bound ``KL(baseline || steered)`` for a target A/B conditional.

    By the KL chain rule this orientation is bounded by
    ``s * D_Bernoulli(p || q)``.  It is not interchangeable with the
    changed-to-baseline KL reported by the intervention runtime.
    """

    (
        probability,
        pair_mass,
        target_log_odds,
        target_probability,
        log_target_probability,
        log_target_complement,
    ) = _validated_ab_shift_parameters(
        baseline_conditional_probability=baseline_conditional_probability,
        pair_probability_mass=pair_probability_mass,
        target_semantic_log_odds=target_semantic_log_odds,
    )
    if probability == 0.0:
        binary_kl = -log_target_complement
    elif probability == 1.0:
        binary_kl = -log_target_probability
    else:
        binary_kl = probability * (math.log(probability) - log_target_probability) + (
            1.0 - probability
        ) * (math.log1p(-probability) - log_target_complement)
    binary_kl = max(0.0, binary_kl)
    lower_bound = pair_mass * binary_kl
    if not math.isfinite(lower_bound):
        raise RuntimeError("A/B KL lower bound is not finite")
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "kl_orientation": "baseline_to_steered",
        "baseline_conditional_probability": probability,
        "baseline_pair_probability_mass": pair_mass,
        "target_semantic_log_odds": target_log_odds,
        "target_conditional_probability": target_probability,
        "binary_baseline_to_target_kl": binary_kl,
        "minimum_baseline_to_steered_full_vocab_kl": lower_bound,
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return lower_bound, diagnostics


def minimum_changed_to_baseline_kl_for_ab_shift(
    *,
    baseline_conditional_probability: float,
    pair_probability_mass: float,
    target_semantic_log_odds: float,
) -> tuple[float, dict[str, Any]]:
    """Lower-bound ``KL(changed || baseline)`` for a target A/B conditional.

    Minimizing over the changed distribution's total A/B pair mass gives
    ``-log(1-s + s*exp(-D_Bernoulli(q || p)))``.  This is the orientation used
    by the intervention runtime's full-vocabulary KL report.
    """

    (
        probability,
        pair_mass,
        target_log_odds,
        target_probability,
        log_target_probability,
        log_target_complement,
    ) = _validated_ab_shift_parameters(
        baseline_conditional_probability=baseline_conditional_probability,
        pair_probability_mass=pair_probability_mass,
        target_semantic_log_odds=target_semantic_log_odds,
    )
    if probability <= 0.0 or probability >= 1.0:
        raise ValueError(
            "baseline_conditional_probability must be strictly between 0 and 1 "
            "for changed-to-baseline KL"
        )
    if target_probability == 0.0:
        reverse_binary_kl = -math.log1p(-probability)
    elif target_probability == 1.0:
        reverse_binary_kl = -math.log(probability)
    else:
        reverse_binary_kl = target_probability * (
            log_target_probability - math.log(probability)
        ) + (1.0 - target_probability) * (log_target_complement - math.log1p(-probability))
    reverse_binary_kl = max(0.0, reverse_binary_kl)
    if pair_mass == 1.0:
        lower_bound = reverse_binary_kl
    else:
        first = math.log1p(-pair_mass)
        second = math.log(pair_mass) - reverse_binary_kl
        maximum = max(first, second)
        log_normalizer = maximum + math.log(math.exp(first - maximum) + math.exp(second - maximum))
        lower_bound = -log_normalizer
    lower_bound = max(0.0, lower_bound)
    if not math.isfinite(lower_bound):
        raise RuntimeError("changed-to-baseline A/B KL lower bound is not finite")
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "kl_orientation": "changed_to_baseline",
        "baseline_conditional_probability": probability,
        "baseline_pair_probability_mass": pair_mass,
        "target_semantic_log_odds": target_log_odds,
        "target_conditional_probability": target_probability,
        "binary_target_to_baseline_kl": reverse_binary_kl,
        "minimum_changed_to_baseline_full_vocab_kl": lower_bound,
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return lower_bound, diagnostics


def _float64_vector(torch: Any, value: Any, *, field: str) -> Any:
    if not torch.is_tensor(value):
        raise TypeError(f"{field} must be a tensor")
    if torch.is_complex(value) or not torch.is_floating_point(value):
        raise TypeError(f"{field} must be a real floating-point tensor")
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{field} must be a non-empty one-dimensional tensor")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"{field} must contain only finite values")
    return result


def _float64_matrix(
    torch: Any,
    value: Any,
    *,
    field: str,
    allow_empty_rows: bool = False,
) -> Any:
    if not torch.is_tensor(value):
        raise TypeError(f"{field} must be a tensor")
    if torch.is_complex(value) or not torch.is_floating_point(value):
        raise TypeError(f"{field} must be a real floating-point tensor")
    if value.ndim != 2 or value.shape[1] == 0:
        raise ValueError(f"{field} must be a two-dimensional tensor with a nonzero width")
    if value.shape[0] == 0 and not allow_empty_rows:
        raise ValueError(f"{field} must contain at least one row")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"{field} must contain only finite values")
    return result


def _canonicalize_basis_rows(torch: Any, basis: Any) -> Any:
    """Construct a coordinate-pivoted basis invariant to SVD subspace rotations."""

    rank, dimension = basis.shape
    if rank == 0:
        return basis.detach().cpu().double().contiguous()
    numerical_threshold = max(
        1e-14,
        100.0 * float(torch.finfo(torch.float64).eps) * dimension,
    )
    canonical_rows = []
    for coordinate in range(dimension):
        candidate = basis.T @ basis[:, coordinate]
        for _ in range(2):
            for row in canonical_rows:
                candidate = candidate - (candidate @ row) * row
        norm = float(torch.linalg.vector_norm(candidate).item())
        if norm <= numerical_threshold:
            continue
        candidate = candidate / norm
        pivot = int(torch.argmax(torch.abs(candidate)).item())
        if float(candidate[pivot].item()) < 0.0:
            candidate = -candidate
        canonical_rows.append(candidate)
        if len(canonical_rows) == rank:
            break
    if len(canonical_rows) != rank:
        raise RuntimeError("could not construct a deterministic basis for the SVD row span")
    return torch.stack(canonical_rows, dim=0).contiguous()


def _row_multiset_sha256(value: Any) -> str:
    """Hash a matrix as an order-independent multiset of float64 rows."""

    row_hashes = sorted(tensor_float64_sha256(row) for row in value)
    return canonical_sha256(row_hashes)


def _sort_matrix_rows(torch: Any, value: Any) -> Any:
    """Return float64 matrix rows in deterministic byte-hash order."""

    if value.shape[0] == 0:
        return value.contiguous()
    indexed = [(tensor_float64_sha256(row), index, row) for index, row in enumerate(value)]
    indexed.sort(key=lambda item: (item[0], item[1]))
    return torch.stack([row for _, _, row in indexed], dim=0).contiguous()


def row_normalized_svd_basis(
    torch: Any,
    rows: Any,
    *,
    rtol: float = DEFAULT_SVD_RTOL,
    atol: float = DEFAULT_SVD_ATOL,
) -> tuple[Any, dict[str, Any]]:
    """Return a deterministic orthonormal basis for a nuisance row span.

    Each row is normalized before SVD, so row magnitude cannot determine rank.
    ``rtol`` and ``atol`` are explicit frozen thresholds; a singular value is
    retained exactly when it exceeds ``max(atol, rtol * largest_singular)``.
    A supplied zero or near-zero row is rejected instead of silently discarded.
    """

    relative_tolerance = _positive_float(rtol, field="rtol")
    absolute_tolerance = _nonnegative_float(atol, field="atol")
    matrix = _float64_matrix(
        torch,
        rows,
        field="rows",
        allow_empty_rows=True,
    )
    dimension = int(matrix.shape[1])
    if matrix.shape[0] == 0:
        basis = torch.empty((0, dimension), dtype=torch.float64)
        projector = torch.zeros((dimension, dimension), dtype=torch.float64)
        diagnostics = {
            "schema_version": SCHEMA_VERSION,
            "input_shape": [0, dimension],
            "rank": 0,
            "rtol": relative_tolerance,
            "atol": absolute_tolerance,
            "rank_threshold": absolute_tolerance,
            "singular_values": [],
            "row_norm_min": None,
            "row_norm_max": None,
            "input_rows_sha256": _row_multiset_sha256(matrix),
            "normalized_rows_sha256": tensor_float64_sha256(matrix),
            "basis_sha256": tensor_float64_sha256(basis),
            "projector_sha256": tensor_float64_sha256(projector),
        }
        diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
        return basis, diagnostics

    norms = torch.linalg.vector_norm(matrix, dim=1)
    if not bool(torch.isfinite(norms).all().item()):
        raise RuntimeError("nuisance row norms overflowed float64")
    if bool((norms <= absolute_tolerance).any().item()):
        bad = [
            index for index, norm in enumerate(norms.tolist()) if float(norm) <= absolute_tolerance
        ]
        raise ValueError(f"nuisance rows at indices {bad} are zero or below atol")
    normalized = matrix / norms[:, None]
    normalized = _sort_matrix_rows(torch, normalized)
    _, singular_values, vh = torch.linalg.svd(normalized, full_matrices=False)
    if not bool(torch.isfinite(singular_values).all().item()):
        raise RuntimeError("nuisance SVD produced non-finite singular values")
    largest = float(singular_values[0].item())
    threshold = max(absolute_tolerance, relative_tolerance * largest)
    rank = int((singular_values > threshold).sum().item())
    if rank == 0:
        raise RuntimeError("non-empty nuisance rows produced a zero-rank SVD basis")
    basis = _canonicalize_basis_rows(torch, vh[:rank])
    projector = basis.T @ basis
    orthogonality_error = (
        float(
            torch.linalg.matrix_norm(basis @ basis.T - torch.eye(rank, dtype=torch.float64)).item()
        )
        if rank
        else 0.0
    )
    span_residual = float(torch.linalg.matrix_norm(normalized - normalized @ projector).item())
    certificate_tolerance = max(absolute_tolerance, relative_tolerance) * (
        1.0 + float(torch.linalg.matrix_norm(normalized).item())
    )
    if not math.isfinite(orthogonality_error) or not math.isfinite(span_residual):
        raise RuntimeError("nuisance basis certificate contains a non-finite residual")
    if orthogonality_error > certificate_tolerance:
        raise RuntimeError("nuisance basis orthogonality residual exceeds tolerance")
    if span_residual > certificate_tolerance:
        raise RuntimeError("nuisance basis span residual exceeds tolerance")
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "input_shape": [int(matrix.shape[0]), dimension],
        "rank": rank,
        "rtol": relative_tolerance,
        "atol": absolute_tolerance,
        "rank_threshold": threshold,
        "singular_values": [float(value) for value in singular_values.tolist()],
        "row_norm_min": float(norms.min().item()),
        "row_norm_max": float(norms.max().item()),
        "orthogonality_error": orthogonality_error,
        "span_residual": span_residual,
        "certificate_tolerance": certificate_tolerance,
        "input_rows_sha256": _row_multiset_sha256(matrix),
        "normalized_rows_sha256": tensor_float64_sha256(normalized),
        "basis_sha256": tensor_float64_sha256(basis),
        "projector_sha256": tensor_float64_sha256(projector),
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return basis, diagnostics


def prompt_balanced_topk_tail_fisher_factors(
    torch: Any,
    prompts: Sequence[Mapping[str, Any]],
    *,
    expected_top_k: int | None,
    minimum_top_k: int = 1,
    probability_tolerance: float = DEFAULT_PROBABILITY_TOLERANCE,
    score_identity_tolerance: float = DEFAULT_SCORE_IDENTITY_TOLERANCE,
    centered_score_identity_tolerance: float = DEFAULT_CENTERED_SCORE_IDENTITY_TOLERANCE,
) -> tuple[Any, dict[str, Any]]:
    """Build low-rank factors for a prompt-balanced top-k-plus-tail Fisher.

    Each prompt supplies residual-scaled gradients of ``log p(token)`` for its
    selected tokens, plus the gradient of the aggregate tail log probability.
    Set ``expected_top_k`` to ``None`` when required answer tokens are unioned
    with a fixed top-k and therefore category counts vary across prompts.  If
    ``f`` is such a score gradient, the emitted row is
    ``sqrt(category_probability / prompt_count) * f``.  Consequently ``R.T@R``
    is the mean Fisher matrix of the coarsened categorical distributions.

    The original validated probability partition is certified before any repair
    using the scale-free defect ``||sum_k p_k f_k|| / sum_k p_k ||f_k||``.
    Probabilities are then normalized, their score mean is recomputed, and that
    certified float32-sized common-mode defect is removed in float64.  A tighter
    float64 identity certificate is required before Fisher factors are built.
    """

    if expected_top_k is not None and (
        not isinstance(expected_top_k, int)
        or isinstance(expected_top_k, bool)
        or expected_top_k <= 0
    ):
        raise ValueError("expected_top_k must be a positive integer or None")
    if not isinstance(minimum_top_k, int) or isinstance(minimum_top_k, bool) or minimum_top_k <= 0:
        raise ValueError("minimum_top_k must be a positive integer")
    if expected_top_k is not None and expected_top_k < minimum_top_k:
        raise ValueError("expected_top_k must not be smaller than minimum_top_k")
    probability_error = _positive_float(
        probability_tolerance,
        field="probability_tolerance",
    )
    score_error = _positive_float(
        score_identity_tolerance,
        field="score_identity_tolerance",
    )
    if score_error >= 1.0:
        raise ValueError("score_identity_tolerance must be less than 1")
    centered_score_error = _positive_float(
        centered_score_identity_tolerance,
        field="centered_score_identity_tolerance",
    )
    if centered_score_error >= score_error:
        raise ValueError(
            "centered_score_identity_tolerance must be less than score_identity_tolerance"
        )
    sources = list(prompts)
    if not sources:
        raise ValueError("prompts must be non-empty")

    checked = []
    seen_ids = set()
    dimension = None
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise TypeError(f"prompts[{index}] must be a mapping")
        prompt_id = source.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError(f"prompts[{index}].prompt_id must be a non-empty string")
        if prompt_id in seen_ids:
            raise ValueError(f"duplicate prompt_id {prompt_id!r}")
        seen_ids.add(prompt_id)

        token_ids = source.get("top_token_ids")
        if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
            raise TypeError(f"prompts[{index}].top_token_ids must be a sequence")
        category_count = len(token_ids)
        if (
            category_count < minimum_top_k
            or (expected_top_k is not None and category_count != expected_top_k)
            or any(
                not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
                for token_id in token_ids
            )
            or len(set(token_ids)) != category_count
        ):
            expected_text = (
                f"exactly {expected_top_k}"
                if expected_top_k is not None
                else f"at least {minimum_top_k}"
            )
            raise ValueError(
                f"prompts[{index}].top_token_ids must contain {expected_text} unique integers"
            )
        probabilities = _float64_vector(
            torch,
            source.get("top_probabilities"),
            field=f"prompts[{index}].top_probabilities",
        )
        if probabilities.numel() != category_count:
            raise ValueError(
                f"prompts[{index}].top_probabilities must have length {category_count}"
            )
        if bool(((probabilities < 0.0) | (probabilities > 1.0)).any().item()):
            raise ValueError("top probabilities must be in the inclusive range [0, 1]")
        top_gradients = _float64_matrix(
            torch,
            source.get("top_score_gradients"),
            field=f"prompts[{index}].top_score_gradients",
        )
        if top_gradients.shape[0] != category_count:
            raise ValueError(
                f"prompts[{index}].top_score_gradients must have {category_count} rows"
            )
        tail_probability = _nonnegative_float(
            source.get("tail_probability"),
            field=f"prompts[{index}].tail_probability",
        )
        if tail_probability > 1.0:
            raise ValueError("tail_probability must not exceed 1")
        tail_gradient = _float64_vector(
            torch,
            source.get("tail_score_gradient"),
            field=f"prompts[{index}].tail_score_gradient",
        )
        if tail_gradient.shape != top_gradients.shape[1:]:
            raise ValueError("tail and top score gradients must have the same width")
        if dimension is None:
            dimension = int(top_gradients.shape[1])
        elif top_gradients.shape[1] != dimension:
            raise ValueError("all prompt score gradients must share one dimension")
        category_order = sorted(range(category_count), key=lambda offset: token_ids[offset])
        sorted_token_ids = [token_ids[offset] for offset in category_order]
        probabilities = probabilities[category_order].contiguous()
        top_gradients = top_gradients[category_order].contiguous()
        raw_probabilities = torch.cat(
            (
                probabilities,
                torch.tensor([tail_probability], dtype=torch.float64),
            )
        ).contiguous()
        probability_sum = float(raw_probabilities.sum().item())
        if not math.isclose(
            probability_sum,
            1.0,
            rel_tol=probability_error,
            abs_tol=probability_error,
        ):
            raise ValueError(
                f"prompt {prompt_id!r} top-plus-tail probability is {probability_sum}, not 1"
            )
        raw_score_gradients = torch.cat(
            (top_gradients, tail_gradient.reshape(1, -1)),
            dim=0,
        ).contiguous()
        raw_weighted_score_mean = raw_probabilities @ raw_score_gradients
        raw_weighted_score_scale = float(
            (raw_probabilities * torch.linalg.vector_norm(raw_score_gradients, dim=1)).sum().item()
        )
        raw_weighted_score_mu_norm = float(torch.linalg.vector_norm(raw_weighted_score_mean).item())
        if raw_weighted_score_scale <= 0.0:
            raise ValueError(
                f"prompt {prompt_id!r} categorical weighted score scale must be positive"
            )
        raw_score_identity_relative_residual = raw_weighted_score_mu_norm / raw_weighted_score_scale
        raw_weighted_score_allowed = score_error * raw_weighted_score_scale
        if not all(
            math.isfinite(value)
            for value in (
                raw_weighted_score_scale,
                raw_weighted_score_mu_norm,
                raw_score_identity_relative_residual,
                raw_weighted_score_allowed,
            )
        ):
            raise RuntimeError(f"prompt {prompt_id!r} score-identity certificate is non-finite")
        if raw_score_identity_relative_residual > score_error:
            raise ValueError(
                f"prompt {prompt_id!r} categorical weighted score mean relative "
                f"residual {raw_score_identity_relative_residual} exceeds {score_error}"
            )

        normalized_probabilities = (raw_probabilities / probability_sum).contiguous()
        if not bool(torch.isfinite(normalized_probabilities).all().item()):
            raise RuntimeError(f"prompt {prompt_id!r} normalized probabilities are non-finite")
        processed_probability_sum = float(normalized_probabilities.sum().item())
        if not math.isfinite(processed_probability_sum):
            raise RuntimeError(f"prompt {prompt_id!r} processed probability sum is non-finite")
        normalized_weighted_score_mean = normalized_probabilities @ raw_score_gradients
        normalized_weighted_score_mean_norm = float(
            torch.linalg.vector_norm(normalized_weighted_score_mean).item()
        )
        if not math.isfinite(normalized_weighted_score_mean_norm):
            raise RuntimeError(f"prompt {prompt_id!r} normalized weighted score mean is non-finite")
        centered_score_gradients = (
            raw_score_gradients - normalized_weighted_score_mean.reshape(1, -1)
        ).contiguous()
        centered_weighted_score_mean = normalized_probabilities @ centered_score_gradients
        centered_weighted_score_scale = float(
            (normalized_probabilities * torch.linalg.vector_norm(centered_score_gradients, dim=1))
            .sum()
            .item()
        )
        centered_weighted_score_residual = float(
            torch.linalg.vector_norm(centered_weighted_score_mean).item()
        )
        if centered_weighted_score_scale <= 0.0:
            raise RuntimeError(
                f"prompt {prompt_id!r} centered categorical score scale is not positive"
            )
        centered_score_identity_relative_residual = (
            centered_weighted_score_residual / centered_weighted_score_scale
        )
        if not all(
            math.isfinite(value)
            for value in (
                centered_weighted_score_scale,
                centered_weighted_score_residual,
                centered_score_identity_relative_residual,
            )
        ):
            raise RuntimeError(
                f"prompt {prompt_id!r} centered score-identity certificate is non-finite"
            )
        if centered_score_identity_relative_residual > centered_score_error:
            raise RuntimeError(
                f"prompt {prompt_id!r} centered score-identity relative residual "
                f"{centered_score_identity_relative_residual} exceeds "
                f"{centered_score_error}"
            )
        centered_top_gradients = centered_score_gradients[:-1].contiguous()
        centered_tail_gradient = centered_score_gradients[-1].contiguous()
        probabilities = normalized_probabilities[:-1].contiguous()
        tail_probability = float(normalized_probabilities[-1].item())
        checked.append(
            {
                "prompt_id": prompt_id,
                "top_token_ids": sorted_token_ids,
                "top_probabilities": probabilities,
                "top_score_gradients": centered_top_gradients,
                "tail_probability": tail_probability,
                "tail_score_gradient": centered_tail_gradient,
                "probability_sum_before_normalization": probability_sum,
                "raw_categorical_probabilities_sha256": tensor_float64_sha256(raw_probabilities),
                "processed_probability_sum": processed_probability_sum,
                "processed_categorical_probabilities_sha256": tensor_float64_sha256(
                    normalized_probabilities
                ),
                # Compatibility alias: Fisher factors use the processed partition.
                "categorical_probabilities_sha256": tensor_float64_sha256(normalized_probabilities),
                "raw_weighted_score_mu_norm": raw_weighted_score_mu_norm,
                "raw_weighted_score_mean_sha256": tensor_float64_sha256(raw_weighted_score_mean),
                "raw_weighted_score_scale": raw_weighted_score_scale,
                "raw_score_identity_relative_residual": raw_score_identity_relative_residual,
                "raw_score_identity_relative_tolerance": score_error,
                "normalized_weighted_score_mean_norm": normalized_weighted_score_mean_norm,
                "normalized_weighted_score_mean_sha256": tensor_float64_sha256(
                    normalized_weighted_score_mean
                ),
                "centered_weighted_score_mean_residual": centered_weighted_score_residual,
                "centered_weighted_score_scale": centered_weighted_score_scale,
                "centered_score_identity_relative_residual": (
                    centered_score_identity_relative_residual
                ),
                "centered_score_identity_relative_tolerance": centered_score_error,
                # Retain the former absolute diagnostic names for downstream
                # readers while making their raw-before-repair meaning explicit.
                "weighted_score_mean_residual": raw_weighted_score_mu_norm,
                "weighted_score_mean_tolerance": raw_weighted_score_allowed,
                "raw_score_gradients_sha256": tensor_float64_sha256(raw_score_gradients),
                "centered_score_gradients_sha256": tensor_float64_sha256(centered_score_gradients),
                "raw_top_score_gradients_sha256": tensor_float64_sha256(top_gradients),
                "raw_tail_score_gradient_sha256": tensor_float64_sha256(tail_gradient),
                "centered_top_score_gradients_sha256": tensor_float64_sha256(
                    centered_top_gradients
                ),
                "centered_tail_score_gradient_sha256": tensor_float64_sha256(
                    centered_tail_gradient
                ),
            }
        )

    checked.sort(key=lambda item: item["prompt_id"])
    prompt_count = len(checked)
    factors = []
    manifest = []
    for source in checked:
        weights = source["top_probabilities"] / prompt_count
        for probability, gradient in zip(weights, source["top_score_gradients"], strict=True):
            factors.append(torch.sqrt(probability) * gradient)
        tail_weight = source["tail_probability"] / prompt_count
        factors.append(math.sqrt(tail_weight) * source["tail_score_gradient"])
        manifest.append(
            {
                "prompt_id": source["prompt_id"],
                "top_token_ids": source["top_token_ids"],
                "top_probabilities": [
                    float(value) for value in source["top_probabilities"].tolist()
                ],
                "tail_probability": source["tail_probability"],
                "weighted_score_mean_residual": source["weighted_score_mean_residual"],
                "weighted_score_mean_tolerance": source["weighted_score_mean_tolerance"],
                "probability_sum_before_normalization": source[
                    "probability_sum_before_normalization"
                ],
                "raw_categorical_probabilities_sha256": source[
                    "raw_categorical_probabilities_sha256"
                ],
                "processed_probability_sum": source["processed_probability_sum"],
                "processed_categorical_probabilities_sha256": source[
                    "processed_categorical_probabilities_sha256"
                ],
                "categorical_probabilities_sha256": source["categorical_probabilities_sha256"],
                "raw_weighted_score_mu_norm": source["raw_weighted_score_mu_norm"],
                "raw_weighted_score_mean_sha256": source["raw_weighted_score_mean_sha256"],
                "raw_weighted_score_scale": source["raw_weighted_score_scale"],
                "raw_score_identity_relative_residual": source[
                    "raw_score_identity_relative_residual"
                ],
                "raw_score_identity_relative_tolerance": source[
                    "raw_score_identity_relative_tolerance"
                ],
                "normalized_weighted_score_mean_norm": source[
                    "normalized_weighted_score_mean_norm"
                ],
                "normalized_weighted_score_mean_sha256": source[
                    "normalized_weighted_score_mean_sha256"
                ],
                "centered_weighted_score_mean_residual": source[
                    "centered_weighted_score_mean_residual"
                ],
                "centered_weighted_score_scale": source["centered_weighted_score_scale"],
                "centered_score_identity_relative_residual": source[
                    "centered_score_identity_relative_residual"
                ],
                "centered_score_identity_relative_tolerance": source[
                    "centered_score_identity_relative_tolerance"
                ],
                "raw_score_gradients_sha256": source["raw_score_gradients_sha256"],
                "centered_score_gradients_sha256": source["centered_score_gradients_sha256"],
                "raw_top_score_gradients_sha256": source["raw_top_score_gradients_sha256"],
                "raw_tail_score_gradient_sha256": source["raw_tail_score_gradient_sha256"],
                "centered_top_score_gradients_sha256": source[
                    "centered_top_score_gradients_sha256"
                ],
                "centered_tail_score_gradient_sha256": source[
                    "centered_tail_score_gradient_sha256"
                ],
            }
        )
    factor_matrix = torch.stack(factors, dim=0).contiguous()
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "prompt_count": prompt_count,
        "expected_top_k": expected_top_k,
        "minimum_top_k": minimum_top_k,
        "minimum_category_count": min(len(source["top_token_ids"]) for source in checked),
        "maximum_category_count": max(len(source["top_token_ids"]) for source in checked),
        "factor_shape": [int(value) for value in factor_matrix.shape],
        "probability_tolerance": probability_error,
        "score_identity_tolerance": score_error,
        "score_identity_tolerance_kind": "relative_weighted_score_norm",
        "centered_score_identity_tolerance": centered_score_error,
        "minimum_top_probability_mass": min(
            float(source["top_probabilities"].sum().item()) for source in checked
        ),
        "maximum_tail_probability": max(source["tail_probability"] for source in checked),
        "maximum_weighted_score_mean_residual": max(
            source["weighted_score_mean_residual"] for source in checked
        ),
        "maximum_raw_score_identity_relative_residual": max(
            source["raw_score_identity_relative_residual"] for source in checked
        ),
        "maximum_raw_weighted_score_mu_norm": max(
            source["raw_weighted_score_mu_norm"] for source in checked
        ),
        "maximum_raw_probability_sum_absolute_error": max(
            abs(source["probability_sum_before_normalization"] - 1.0) for source in checked
        ),
        "maximum_processed_probability_sum_absolute_error": max(
            abs(source["processed_probability_sum"] - 1.0) for source in checked
        ),
        "maximum_normalized_weighted_score_mean_norm": max(
            source["normalized_weighted_score_mean_norm"] for source in checked
        ),
        "maximum_centered_weighted_score_mean_residual": max(
            source["centered_weighted_score_mean_residual"] for source in checked
        ),
        "maximum_centered_score_identity_relative_residual": max(
            source["centered_score_identity_relative_residual"] for source in checked
        ),
        "prompt_manifest": manifest,
        "prompt_manifest_sha256": canonical_sha256(manifest),
        "factor_matrix_sha256": tensor_float64_sha256(factor_matrix),
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return factor_matrix, diagnostics


def semantic_margin_constraints(
    torch: Any,
    self_semantic_gradients: Any,
    baseline_semantic_log_odds: Any,
    *,
    attack_sign: int,
    boundary_buffer: float = 0.0,
) -> tuple[Any, Any, dict[str, Any]]:
    """Orient two self-order gradients and compute boundary-crossing margins."""

    gradients = _float64_matrix(
        torch,
        self_semantic_gradients,
        field="self_semantic_gradients",
    )
    if gradients.shape[0] != 2:
        raise ValueError("self_semantic_gradients must contain exactly two option-order rows")
    logits = _float64_vector(
        torch,
        baseline_semantic_log_odds,
        field="baseline_semantic_log_odds",
    )
    if logits.numel() != 2:
        raise ValueError("baseline_semantic_log_odds must have length two")
    if attack_sign not in {-1, 1} or isinstance(attack_sign, bool):
        raise ValueError("attack_sign must be +1 or -1")
    buffer = _nonnegative_float(boundary_buffer, field="boundary_buffer")
    signed_baselines = attack_sign * logits
    if bool((signed_baselines >= 0.0).any().item()):
        raise ValueError(
            "both baseline option orders must be semantically opposite the requested attack sign"
        )
    inequality_rows = attack_sign * gradients
    lower_bounds = buffer - signed_baselines
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "attack_sign": attack_sign,
        "boundary_buffer": buffer,
        "baseline_semantic_log_odds": [float(value) for value in logits.tolist()],
        "lower_bounds": [float(value) for value in lower_bounds.tolist()],
        "self_semantic_gradients_sha256": tensor_float64_sha256(gradients),
        "inequality_rows_sha256": tensor_float64_sha256(inequality_rows),
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return inequality_rows, lower_bounds, diagnostics


def _prepare_woodbury(
    torch: Any,
    fisher_factors: Any,
    *,
    ridge: float,
    condition_limit: float,
) -> tuple[Any, float, Any | None, dict[str, Any]]:
    factors = _float64_matrix(
        torch,
        fisher_factors,
        field="fisher_factors",
        allow_empty_rows=True,
    )
    factors = _sort_matrix_rows(torch, factors)
    ridge_value = _positive_float(ridge, field="ridge")
    maximum_condition = _positive_float(condition_limit, field="condition_limit")
    if factors.shape[0] == 0:
        diagnostics = {
            "factor_shape": [0, int(factors.shape[1])],
            "ridge": ridge_value,
            "factor_gram_condition_number": 1.0,
            "factor_gram_min_eigenvalue": ridge_value,
            "factor_gram_max_eigenvalue": ridge_value,
            "hessian_condition_number": 1.0,
            "hessian_min_eigenvalue": ridge_value,
            "hessian_max_eigenvalue": ridge_value,
            "factor_matrix_sha256": tensor_float64_sha256(factors),
        }
        return factors, ridge_value, None, diagnostics

    gram = factors @ factors.T + ridge_value * torch.eye(factors.shape[0], dtype=torch.float64)
    gram = (gram + gram.T) / 2.0
    eigenvalues = torch.linalg.eigvalsh(gram)
    minimum = float(eigenvalues[0].item())
    maximum = float(eigenvalues[-1].item())
    condition = maximum / minimum if minimum > 0.0 else math.inf
    if not math.isfinite(condition) or condition > maximum_condition:
        raise RuntimeError(
            f"Fisher Woodbury Gram condition number {condition} exceeds {maximum_condition}"
        )
    cholesky, info = torch.linalg.cholesky_ex(gram)
    if int(info.item()) != 0:
        raise RuntimeError("Fisher Woodbury Gram is not numerically positive definite")
    singular_values = torch.linalg.svdvals(factors)
    hessian_maximum = ridge_value + float(singular_values[0].item()) ** 2
    if factors.shape[0] < factors.shape[1]:
        hessian_minimum = ridge_value
    else:
        hessian_minimum = ridge_value + float(singular_values[-1].item()) ** 2
    hessian_condition = hessian_maximum / hessian_minimum
    if not math.isfinite(hessian_condition) or hessian_condition > maximum_condition:
        raise RuntimeError(
            f"Fisher Hessian condition number {hessian_condition} exceeds {maximum_condition}"
        )
    diagnostics = {
        "factor_shape": [int(value) for value in factors.shape],
        "ridge": ridge_value,
        "factor_gram_condition_number": condition,
        "factor_gram_min_eigenvalue": minimum,
        "factor_gram_max_eigenvalue": maximum,
        "hessian_condition_number": hessian_condition,
        "hessian_min_eigenvalue": hessian_minimum,
        "hessian_max_eigenvalue": hessian_maximum,
        "factor_matrix_sha256": tensor_float64_sha256(factors),
    }
    return factors, ridge_value, cholesky, diagnostics


def _woodbury_apply(
    torch: Any, factors: Any, ridge: float, cholesky: Any | None, value: Any
) -> Any:
    matrix_input = value.ndim == 2
    vectors = value if matrix_input else value[:, None]
    if cholesky is None:
        result = vectors / ridge
    else:
        coefficients = torch.cholesky_solve(factors @ vectors, cholesky)
        result = (vectors - factors.T @ coefficients) / ridge
    return result if matrix_input else result[:, 0]


def woodbury_h_inverse(
    torch: Any,
    fisher_factors: Any,
    vectors: Any,
    *,
    ridge: float,
    condition_limit: float = DEFAULT_CONDITION_LIMIT,
    residual_tolerance: float = DEFAULT_RESIDUAL_TOLERANCE,
) -> tuple[Any, dict[str, Any]]:
    """Apply ``(ridge*I + R.T@R)^-1`` without forming a dense Hessian."""

    factors, ridge_value, cholesky, diagnostics = _prepare_woodbury(
        torch,
        fisher_factors,
        ridge=ridge,
        condition_limit=condition_limit,
    )
    if not torch.is_tensor(vectors) or vectors.ndim not in {1, 2}:
        raise TypeError("vectors must be a one- or two-dimensional tensor")
    original_matrix = vectors.ndim == 2
    source = (
        _float64_matrix(torch, vectors, field="vectors")
        if original_matrix
        else _float64_vector(torch, vectors, field="vectors")
    )
    if source.shape[0] != factors.shape[1]:
        raise ValueError("vectors and Fisher factors must have the same leading dimension")
    result = _woodbury_apply(torch, factors, ridge_value, cholesky, source)
    result_matrix = result if result.ndim == 2 else result[:, None]
    source_matrix = source if source.ndim == 2 else source[:, None]
    reconstructed = ridge_value * result_matrix + factors.T @ (factors @ result_matrix)
    residual = float(torch.linalg.matrix_norm(reconstructed - source_matrix).item())
    scale = 1.0 + float(torch.linalg.matrix_norm(source_matrix).item())
    allowed = _positive_float(residual_tolerance, field="residual_tolerance") * scale
    if residual > allowed:
        raise RuntimeError(f"Woodbury inverse residual {residual} exceeds {allowed}")
    output = result if original_matrix else result[:, 0] if result.ndim == 2 else result
    full_diagnostics = {
        **diagnostics,
        "right_hand_side_shape": [int(value) for value in source_matrix.shape],
        "inverse_residual": residual,
        "inverse_residual_tolerance": allowed,
        "result_sha256": tensor_float64_sha256(output),
    }
    full_diagnostics["diagnostics_sha256"] = canonical_sha256(full_diagnostics)
    return output.contiguous(), full_diagnostics


def solve_min_fisher_qp(
    torch: Any,
    *,
    inequality_rows: Any,
    lower_bounds: Any,
    nuisance_rows: Any,
    fisher_factors: Any,
    ridge: float,
    svd_rtol: float = DEFAULT_SVD_RTOL,
    svd_atol: float = DEFAULT_SVD_ATOL,
    condition_limit: float = DEFAULT_CONDITION_LIMIT,
    residual_tolerance: float = DEFAULT_RESIDUAL_TOLERANCE,
) -> tuple[Any, dict[str, Any]]:
    """Solve a certified two-inequality minimum-Fisher convex QP.

    The objective is ``0.5*w.T@(ridge*I + R.T@R)@w``.  Hard nuisance
    equalities are the row span of ``nuisance_rows``.  The two inequalities are
    ``inequality_rows @ w >= lower_bounds``.  Because both lower bounds must be
    positive, one of the active sets ``{0}``, ``{1}``, or ``{0,1}`` contains the
    optimum; all three are enumerated deterministically.
    """

    inequalities = _float64_matrix(
        torch,
        inequality_rows,
        field="inequality_rows",
    )
    if inequalities.shape[0] != 2:
        raise ValueError("inequality_rows must contain exactly two rows")
    bounds = _float64_vector(torch, lower_bounds, field="lower_bounds")
    if bounds.numel() != 2 or bool((bounds <= 0.0).any().item()):
        raise ValueError("lower_bounds must contain exactly two positive values")
    nuisances = _float64_matrix(
        torch,
        nuisance_rows,
        field="nuisance_rows",
        allow_empty_rows=True,
    )
    factors, ridge_value, cholesky, woodbury_diagnostics = _prepare_woodbury(
        torch,
        fisher_factors,
        ridge=ridge,
        condition_limit=condition_limit,
    )
    dimension = int(inequalities.shape[1])
    if nuisances.shape[1] != dimension or factors.shape[1] != dimension:
        raise ValueError("all QP matrices must share one vector dimension")
    tolerance = _positive_float(residual_tolerance, field="residual_tolerance")
    maximum_condition = _positive_float(condition_limit, field="condition_limit")
    nuisance_basis, nuisance_diagnostics = row_normalized_svd_basis(
        torch,
        nuisances,
        rtol=svd_rtol,
        atol=svd_atol,
    )
    projected_inequalities = inequalities - (inequalities @ nuisance_basis.T) @ nuisance_basis
    projected_norms = torch.linalg.vector_norm(projected_inequalities, dim=1)
    if bool((projected_norms <= float(svd_atol)).any().item()):
        raise RuntimeError(
            "at least one positive self margin is infeasible in the nuisance nullspace"
        )

    active_sets = ((0,), (1,), (0, 1))
    reports = []
    feasible = []
    for active in active_sets:
        active_rows = projected_inequalities[list(active)]
        constraints = torch.cat((nuisance_basis, active_rows), dim=0)
        right_hand_side = torch.cat(
            (
                torch.zeros(nuisance_basis.shape[0], dtype=torch.float64),
                bounds[list(active)],
            )
        )
        report: dict[str, Any] = {"active_orders": list(active)}
        try:
            inverse_constraint_transpose = _woodbury_apply(
                torch,
                factors,
                ridge_value,
                cholesky,
                constraints.T,
            )
            schur = constraints @ inverse_constraint_transpose
            schur = (schur + schur.T) / 2.0
            eigenvalues = torch.linalg.eigvalsh(schur)
            minimum_eigenvalue = float(eigenvalues[0].item())
            maximum_eigenvalue = float(eigenvalues[-1].item())
            condition = (
                maximum_eigenvalue / minimum_eigenvalue if minimum_eigenvalue > 0.0 else math.inf
            )
            if not math.isfinite(condition) or condition > maximum_condition:
                raise RuntimeError(f"active-set Schur condition number {condition} exceeds limit")
            cholesky_schur, info = torch.linalg.cholesky_ex(schur)
            if int(info.item()) != 0:
                raise RuntimeError("active-set Schur matrix is not positive definite")
            multipliers = torch.cholesky_solve(
                right_hand_side[:, None],
                cholesky_schur,
            )[:, 0]
            perturbation = inverse_constraint_transpose @ multipliers
            if not bool(torch.isfinite(multipliers).all().item()) or not bool(
                torch.isfinite(perturbation).all().item()
            ):
                raise RuntimeError("active-set solve produced a non-finite result")
            active_multipliers = multipliers[-len(active) :]
            if bool((active_multipliers < -tolerance).any().item()):
                raise RuntimeError("active inequality has a negative KKT multiplier")

            inequality_slacks = inequalities @ perturbation - bounds
            equality_values = nuisance_basis @ perturbation
            hessian_times = ridge_value * perturbation + factors.T @ (factors @ perturbation)
            stationarity = hessian_times - constraints.T @ multipliers
            equality_residual = (
                float(torch.max(torch.abs(equality_values)).item())
                if equality_values.numel()
                else 0.0
            )
            minimum_slack = float(inequality_slacks.min().item())
            stationarity_residual = float(torch.linalg.vector_norm(stationarity).item())
            active_residual = float(torch.max(torch.abs(inequality_slacks[list(active)])).item())
            complementarity_residual = float(
                torch.max(torch.abs(active_multipliers * inequality_slacks[list(active)])).item()
            )
            certificate_values = (
                equality_residual,
                minimum_slack,
                stationarity_residual,
                active_residual,
                complementarity_residual,
            )
            if not all(math.isfinite(value) for value in certificate_values):
                raise RuntimeError("active-set KKT certificate is non-finite")
            equality_allowed = tolerance * (
                1.0 + float(torch.linalg.vector_norm(perturbation).item())
            )
            inequality_allowed = tolerance * (1.0 + float(torch.max(torch.abs(bounds)).item()))
            stationarity_allowed = tolerance * (
                1.0
                + float(torch.linalg.vector_norm(hessian_times).item())
                + float(torch.linalg.vector_norm(constraints.T @ multipliers).item())
            )
            if equality_residual > equality_allowed:
                raise RuntimeError("hard nuisance equality residual exceeds tolerance")
            if minimum_slack < -inequality_allowed:
                raise RuntimeError("inactive self-order margin is not satisfied")
            if active_residual > inequality_allowed:
                raise RuntimeError("active self-order margin residual exceeds tolerance")
            if stationarity_residual > stationarity_allowed:
                raise RuntimeError("KKT stationarity residual exceeds tolerance")
            if complementarity_residual > inequality_allowed:
                raise RuntimeError("KKT complementarity residual exceeds tolerance")

            objective = float((0.5 * perturbation @ hessian_times).item())
            if not math.isfinite(objective) or objective < -inequality_allowed:
                raise RuntimeError("active-set objective is invalid")
            report.update(
                {
                    "status": "feasible",
                    "objective": objective,
                    "schur_condition_number": condition,
                    "schur_min_eigenvalue": minimum_eigenvalue,
                    "schur_max_eigenvalue": maximum_eigenvalue,
                    "active_multipliers": [float(value) for value in active_multipliers.tolist()],
                    "inequality_slacks": [float(value) for value in inequality_slacks.tolist()],
                    "equality_residual": equality_residual,
                    "active_residual": active_residual,
                    "stationarity_residual": stationarity_residual,
                    "complementarity_residual": complementarity_residual,
                    "perturbation_sha256": tensor_float64_sha256(perturbation),
                }
            )
            feasible.append((objective, active, perturbation, report))
        except RuntimeError as error:
            report.update({"status": "rejected", "reason": str(error)})
        reports.append(report)

    if not feasible:
        reasons = "; ".join(
            f"{report['active_orders']}: {report.get('reason', 'unknown')}" for report in reports
        )
        raise RuntimeError(f"no certified feasible active set: {reasons}")
    objective, selected_active, perturbation, selected_report = min(
        feasible,
        key=lambda item: (item[0], len(item[1]), item[1]),
    )
    perturbation = perturbation.detach().cpu().double().contiguous()
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "dimension": dimension,
        "objective": objective,
        "selected_active_orders": list(selected_active),
        "inequality_rows_sha256": tensor_float64_sha256(inequalities),
        "lower_bounds": [float(value) for value in bounds.tolist()],
        "nuisance_rows_sha256": _row_multiset_sha256(nuisances),
        "fisher_factors_sha256": tensor_float64_sha256(factors),
        "perturbation_sha256": tensor_float64_sha256(perturbation),
        "projected_self_row_norms": [float(value) for value in projected_norms.tolist()],
        "svd_rtol": float(svd_rtol),
        "svd_atol": float(svd_atol),
        "condition_limit": maximum_condition,
        "residual_tolerance": tolerance,
        "nuisance_basis": nuisance_diagnostics,
        "woodbury": woodbury_diagnostics,
        "active_set_reports": reports,
        "selected_certificate": selected_report,
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return perturbation, diagnostics


def _package_certified_direction(
    torch: Any,
    *,
    perturbation: Any,
    inequality_rows: Any,
    lower_bounds: Any,
    nuisance_rows: Any,
    qp_diagnostics: Mapping[str, Any],
    method: str,
    global_multiplier_convention: str,
    extra_diagnostics: Mapping[str, Any],
    svd_rtol: float,
    svd_atol: float,
    residual_tolerance: float,
) -> tuple[Any, float, dict[str, Any]]:
    """Cast, normalize, and recertify the perturbation that will be injected."""

    native_norm = float(torch.linalg.vector_norm(perturbation).item())
    absolute_tolerance = _nonnegative_float(svd_atol, field="svd_atol")
    numerical_tolerance = _positive_float(
        residual_tolerance,
        field="residual_tolerance",
    )
    if not math.isfinite(native_norm) or native_norm <= absolute_tolerance:
        raise RuntimeError("native perturbation has no numerically usable norm")
    direction64 = perturbation / native_norm
    direction32 = direction64.to(dtype=torch.float32).contiguous()
    output_norm = float(torch.linalg.vector_norm(direction32.double()).item())
    if not math.isclose(output_norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError("float32 output direction is not unit normalized")
    applied_perturbation = native_norm * direction32.double()
    nuisance_basis, _ = row_normalized_svd_basis(
        torch,
        nuisance_rows,
        rtol=svd_rtol,
        atol=svd_atol,
    )
    post_cast_slacks = inequality_rows @ applied_perturbation - lower_bounds
    post_cast_nuisance_values = nuisance_basis @ applied_perturbation
    post_cast_reconstruction_error = float(
        torch.linalg.vector_norm(applied_perturbation - perturbation).item()
    )
    post_cast_base_tolerance = max(
        numerical_tolerance,
        8.0 * float(torch.finfo(torch.float32).eps),
    )
    margin_allowed = post_cast_base_tolerance * (
        1.0 + float(torch.max(torch.abs(lower_bounds)).item())
    )
    nuisance_allowed = post_cast_base_tolerance * (1.0 + native_norm)
    minimum_post_cast_slack = float(post_cast_slacks.min().item())
    maximum_post_cast_nuisance = (
        float(torch.max(torch.abs(post_cast_nuisance_values)).item())
        if post_cast_nuisance_values.numel()
        else 0.0
    )
    post_cast_values = (
        post_cast_reconstruction_error,
        margin_allowed,
        nuisance_allowed,
        minimum_post_cast_slack,
        maximum_post_cast_nuisance,
    )
    if not all(math.isfinite(value) for value in post_cast_values):
        raise RuntimeError("float32 output certificate is non-finite")
    if minimum_post_cast_slack < -margin_allowed:
        raise RuntimeError("float32 output direction violates a self-order margin")
    if maximum_post_cast_nuisance > nuisance_allowed:
        raise RuntimeError("float32 output direction violates hard nuisance nulling")
    if post_cast_reconstruction_error > nuisance_allowed:
        raise RuntimeError("float32 output direction changes the certified perturbation too much")
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "arithmetic_dtype": "float64",
        "output_dtype": "float32",
        "application": (
            "delta_h = global_multiplier * native_residual_relative_norm * "
            "residual_norm * direction"
        ),
        "native_residual_relative_norm": native_norm,
        "global_multiplier_convention": global_multiplier_convention,
        "direction_float64_sha256": tensor_float64_sha256(direction64),
        "direction_float32_sha256": tensor_float32_sha256(direction32),
        "perturbation_float64_sha256": tensor_float64_sha256(perturbation),
        "post_cast_applied_perturbation_sha256": tensor_float64_sha256(applied_perturbation),
        "post_cast_certificate": {
            "inequality_slacks": [float(value) for value in post_cast_slacks.tolist()],
            "minimum_inequality_slack": minimum_post_cast_slack,
            "inequality_tolerance": margin_allowed,
            "maximum_abs_nuisance_basis_projection": maximum_post_cast_nuisance,
            "nuisance_tolerance": nuisance_allowed,
            "float64_reconstruction_error": post_cast_reconstruction_error,
            "base_tolerance": post_cast_base_tolerance,
        },
        **dict(extra_diagnostics),
        "qp": dict(qp_diagnostics),
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return direction32, native_norm, diagnostics


def construct_v3_direction(
    torch: Any,
    *,
    self_semantic_gradients: Any,
    baseline_semantic_log_odds: Any,
    attack_sign: int,
    nuisance_rows: Any,
    fisher_factors: Any,
    ridge: float,
    boundary_buffer: float = 0.0,
    svd_rtol: float = DEFAULT_SVD_RTOL,
    svd_atol: float = DEFAULT_SVD_ATOL,
    condition_limit: float = DEFAULT_CONDITION_LIMIT,
    residual_tolerance: float = DEFAULT_RESIDUAL_TOLERANCE,
) -> tuple[Any, float, dict[str, Any]]:
    """Construct a one-sided signed direction (not the bidirectional knob test)."""

    inequality_rows, lower_bounds, margin_diagnostics = semantic_margin_constraints(
        torch,
        self_semantic_gradients,
        baseline_semantic_log_odds,
        attack_sign=attack_sign,
        boundary_buffer=boundary_buffer,
    )
    perturbation, qp_diagnostics = solve_min_fisher_qp(
        torch,
        inequality_rows=inequality_rows,
        lower_bounds=lower_bounds,
        nuisance_rows=nuisance_rows,
        fisher_factors=fisher_factors,
        ridge=ridge,
        svd_rtol=svd_rtol,
        svd_atol=svd_atol,
        condition_limit=condition_limit,
        residual_tolerance=residual_tolerance,
    )
    return _package_certified_direction(
        torch,
        perturbation=perturbation,
        inequality_rows=inequality_rows,
        lower_bounds=lower_bounds,
        nuisance_rows=nuisance_rows,
        qp_diagnostics=qp_diagnostics,
        method="one_sided_nuisance_null_minimum_fisher_two_order_qp",
        global_multiplier_convention=(
            "inject the already-signed one-sided direction at "
            "strength=global_multiplier*native_residual_relative_norm"
        ),
        extra_diagnostics={"margin_constraints": margin_diagnostics},
        svd_rtol=svd_rtol,
        svd_atol=svd_atol,
        residual_tolerance=residual_tolerance,
    )


def construct_v3_bidirectional_direction(
    torch: Any,
    *,
    self_semantic_gradients: Any,
    baseline_semantic_log_odds: Any,
    nuisance_rows: Any,
    fisher_factors: Any,
    ridge: float,
    decision_margin: float,
    svd_rtol: float = DEFAULT_SVD_RTOL,
    svd_atol: float = DEFAULT_SVD_ATOL,
    condition_limit: float = DEFAULT_CONDITION_LIMIT,
    residual_tolerance: float = DEFAULT_RESIDUAL_TOLERANCE,
) -> tuple[Any, float, dict[str, Any]]:
    """Fit one positive-SP vector whose positive and negative signs cross both orders.

    For baseline semantic log-odds ``m_i`` and first-order gradient ``q_i``, the
    constraints are ``q_i @ w >= abs(m_i) + decision_margin``.  Therefore the
    linearized predictions satisfy ``m_i + q_i@w >= decision_margin`` and
    ``m_i - q_i@w <= -decision_margin`` for each option order.
    """

    gradients = _float64_matrix(
        torch,
        self_semantic_gradients,
        field="self_semantic_gradients",
    )
    if gradients.shape[0] != 2:
        raise ValueError("self_semantic_gradients must contain exactly two option-order rows")
    baselines = _float64_vector(
        torch,
        baseline_semantic_log_odds,
        field="baseline_semantic_log_odds",
    )
    if baselines.numel() != 2:
        raise ValueError("baseline_semantic_log_odds must have length two")
    margin = _positive_float(decision_margin, field="decision_margin")
    lower_bounds = torch.abs(baselines) + margin
    constraint_diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "mode": "symmetric_bidirectional",
        "positive_direction_semantics": "increase preserve-minus-comply log-odds",
        "decision_margin": margin,
        "baseline_semantic_log_odds": [float(value) for value in baselines.tolist()],
        "lower_bounds": [float(value) for value in lower_bounds.tolist()],
        "self_semantic_gradients_sha256": tensor_float64_sha256(gradients),
    }
    constraint_diagnostics["diagnostics_sha256"] = canonical_sha256(constraint_diagnostics)
    perturbation, qp_diagnostics = solve_min_fisher_qp(
        torch,
        inequality_rows=gradients,
        lower_bounds=lower_bounds,
        nuisance_rows=nuisance_rows,
        fisher_factors=fisher_factors,
        ridge=ridge,
        svd_rtol=svd_rtol,
        svd_atol=svd_atol,
        condition_limit=condition_limit,
        residual_tolerance=residual_tolerance,
    )
    direction, native_norm, diagnostics = _package_certified_direction(
        torch,
        perturbation=perturbation,
        inequality_rows=gradients,
        lower_bounds=lower_bounds,
        nuisance_rows=nuisance_rows,
        qp_diagnostics=qp_diagnostics,
        method="bidirectional_nuisance_null_minimum_fisher_two_order_qp",
        global_multiplier_convention=(
            "+1 is positive-SP and -1 is comply, each at "
            "abs(global_multiplier)*native_residual_relative_norm"
        ),
        extra_diagnostics={"bidirectional_constraints": constraint_diagnostics},
        svd_rtol=svd_rtol,
        svd_atol=svd_atol,
        residual_tolerance=residual_tolerance,
    )
    applied_perturbation = native_norm * direction.double()
    positive_predictions = baselines + gradients @ applied_perturbation
    negative_predictions = baselines - gradients @ applied_perturbation
    if not bool(torch.isfinite(positive_predictions).all().item()) or not bool(
        torch.isfinite(negative_predictions).all().item()
    ):
        raise RuntimeError("bidirectional float32 predictions are non-finite")
    prediction_tolerance = float(diagnostics["post_cast_certificate"]["inequality_tolerance"])
    if margin <= prediction_tolerance:
        raise RuntimeError("decision_margin is too small to certify after float32 conversion")
    if float(positive_predictions.min().item()) <= 0.0:
        raise RuntimeError("positive float32 direction does not flip both option orders")
    if float(negative_predictions.max().item()) >= 0.0:
        raise RuntimeError("negative float32 direction does not flip both option orders")
    if float(positive_predictions.min().item()) < margin - prediction_tolerance:
        raise RuntimeError("positive float32 direction does not cross both option orders")
    if float(negative_predictions.max().item()) > -margin + prediction_tolerance:
        raise RuntimeError("negative float32 direction does not cross both option orders")
    diagnostics["bidirectional_post_cast_certificate"] = {
        "positive_direction_semantic_log_odds": [
            float(value) for value in positive_predictions.tolist()
        ],
        "negative_direction_semantic_log_odds": [
            float(value) for value in negative_predictions.tolist()
        ],
        "required_positive_margin": margin,
        "required_negative_margin": -margin,
        "prediction_tolerance": prediction_tolerance,
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(
        {key: value for key, value in diagnostics.items() if key != "diagnostics_sha256"}
    )
    return direction, native_norm, diagnostics
