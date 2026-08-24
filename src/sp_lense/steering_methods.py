from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

MATCHED_FINAL_PROMPT = "matched_final_prompt"
CANONICAL_BROADCAST = "canonical_broadcast"
GRADIENT_SELF_SPECIFIC = "gradient_self_specific"
GRADIENT_UNCORRECTED = "gradient_uncorrected"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_safe(value: Any, *, path: str = "metadata") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            output[key] = _json_safe(item, path=f"{path}.{key}")
        return output
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{path} must contain only JSON-compatible values, got {type(value).__name__}")


def _float32_direction_bytes(direction: Any) -> bytes:
    array = direction.detach().to(device="cpu").float().contiguous().numpy()
    return array.astype("<f4", copy=False).tobytes(order="C")


def _python_l2_norm(direction: Any) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in direction.tolist()))


@dataclass(frozen=True)
class DirectionArtifact:
    """A direction plus deterministic, machine-readable provenance.

    The direction is copied to contiguous CPU float32 before hashing. The artifact
    hash binds those exact bytes to canonical JSON metadata; it is not a hash of a
    pickle file, whose bytes can depend on serializer details.
    """

    method: str
    direction: Any
    layer: int
    intervention_geometry: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("method must be a non-empty string")
        if not isinstance(self.layer, int) or isinstance(self.layer, bool) or self.layer < 0:
            raise ValueError("layer must be a non-negative integer")
        if not isinstance(self.intervention_geometry, str) or not self.intervention_geometry:
            raise ValueError("intervention_geometry must be a non-empty string")
        if not hasattr(self.direction, "detach"):
            raise TypeError("direction must be a tensor")
        direction = self.direction.detach().to(device="cpu").float().contiguous().clone()
        if direction.ndim != 1 or direction.numel() == 0:
            raise ValueError("direction must be a non-empty one-dimensional tensor")
        if not bool(direction.isfinite().all().item()):
            raise ValueError("direction must contain only finite values")
        if _python_l2_norm(direction) <= 1e-12:
            raise ValueError("direction must be non-zero")
        metadata = _json_safe(dict(self.metadata))
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "metadata", metadata)

    @property
    def direction_sha256(self) -> str:
        return hashlib.sha256(_float32_direction_bytes(self.direction)).hexdigest()

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "sp_lense.direction.v1",
            "method": self.method,
            "layer": self.layer,
            "intervention_geometry": self.intervention_geometry,
            "d_model": int(self.direction.numel()),
            "dtype": "float32",
            "direction_l2_norm": _python_l2_norm(self.direction),
            "direction_sha256": self.direction_sha256,
            "metadata": _json_safe(self.metadata),
        }

    @property
    def metadata_sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.to_metadata_dict())).hexdigest()

    @property
    def artifact_sha256(self) -> str:
        payload = _canonical_json_bytes(self.to_metadata_dict())
        return hashlib.sha256(payload + b"\0" + _float32_direction_bytes(self.direction)).hexdigest()

    def to_record(self) -> dict[str, Any]:
        return {
            **self.to_metadata_dict(),
            "metadata_sha256": self.metadata_sha256,
            "artifact_sha256": self.artifact_sha256,
            "direction": self.direction.tolist(),
        }


def _validate_vector(vector: Any, name: str) -> Any:
    if not hasattr(vector, "float"):
        raise TypeError(f"{name} must be a tensor")
    working = vector.float()
    if working.ndim != 1 or working.numel() == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional tensor")
    if not bool(working.isfinite().all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return working


def normalize_direction(torch: Any, vector: Any, *, eps: float = 1e-12) -> Any:
    working = _validate_vector(vector, "vector")
    norm = working.norm()
    norm_value = float(norm.detach().item())
    if not math.isfinite(norm_value) or norm_value <= eps:
        raise ValueError("cannot normalize a zero or non-finite direction")
    return working / norm


def orient_direction(
    torch: Any,
    direction: Any,
    positive_reference: Any,
    *,
    eps: float = 1e-12,
) -> Any:
    del torch
    unit = normalize_direction(None, direction, eps=eps)
    reference = _validate_vector(positive_reference, "positive_reference")
    if unit.shape != reference.shape:
        raise ValueError("direction and positive_reference must have the same shape")
    score = unit @ reference
    score_value = float(score.detach().item())
    if not math.isfinite(score_value) or abs(score_value) <= eps:
        raise ValueError("positive_reference does not define the direction's orientation")
    return unit if score_value > 0 else -unit


def _mean_vector(torch: Any, vectors: Sequence[Any], name: str) -> Any:
    items = list(vectors)
    if not items:
        raise ValueError(f"{name} must be non-empty")
    validated = [_validate_vector(vector, f"{name}[{index}]") for index, vector in enumerate(items)]
    shape = validated[0].shape
    if any(vector.shape != shape for vector in validated[1:]):
        raise ValueError(f"all {name} vectors must have the same shape")
    return torch.stack(validated, dim=0).mean(dim=0)


def construct_gradient_directions(
    torch: Any,
    self_gradients: Sequence[Any],
    matched_other_gradients: Sequence[Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Construct the corrected gradient direction and its uncorrected ablation."""

    self_items = list(self_gradients)
    other_items = list(matched_other_gradients)
    if not self_items or len(self_items) != len(other_items):
        raise ValueError("self and matched-other gradients must be non-empty and paired")
    mean_self = _mean_vector(torch, self_items, "self_gradients")
    mean_other = _mean_vector(torch, other_items, "matched_other_gradients")
    if mean_self.shape != mean_other.shape:
        raise ValueError("self and matched-other gradients must have the same shape")

    uncorrected = orient_direction(torch, mean_self, mean_self)
    other_unit = normalize_direction(torch, mean_other)
    removed_coefficient = mean_self @ other_unit
    corrected_raw = mean_self - removed_coefficient * other_unit
    corrected = orient_direction(torch, corrected_raw, mean_self)

    self_norm = mean_self.norm()
    other_norm = mean_other.norm()
    diagnostics = {
        "n_pairs": len(self_items),
        "d_model": int(mean_self.numel()),
        "mean_self_norm": float(self_norm.detach().item()),
        "mean_other_norm": float(other_norm.detach().item()),
        "self_other_cosine": float(
            ((mean_self @ mean_other) / (self_norm * other_norm)).detach().item()
        ),
        "removed_projection_coefficient": float(removed_coefficient.detach().item()),
        "corrected_raw_norm": float(corrected_raw.norm().detach().item()),
        "corrected_mean_self_projection": float((mean_self @ corrected).detach().item()),
        "corrected_mean_other_projection": float((mean_other @ corrected).detach().item()),
        "uncorrected_mean_other_projection": float((mean_other @ uncorrected).detach().item()),
    }
    directions = {
        GRADIENT_SELF_SPECIFIC: corrected.detach().cpu().float().contiguous(),
        GRADIENT_UNCORRECTED: uncorrected.detach().cpu().float().contiguous(),
    }
    return directions, diagnostics


@dataclass(frozen=True)
class SemanticActivationPair:
    preserve_activation: Any
    comply_activation: Any
    preserve_label: str
    comply_label: str
    case_id: str | None = None

    def __post_init__(self) -> None:
        if not self.preserve_label or not self.comply_label:
            raise ValueError("semantic labels must be non-empty")
        if self.preserve_label == self.comply_label:
            raise ValueError("preserve_label and comply_label must differ")


def semantic_activation_pair(
    activations_by_label: Mapping[str, Any],
    preserve_label: str,
    comply_label: str,
    *,
    case_id: str | None = None,
) -> SemanticActivationPair:
    """Resolve activations by semantic answer, never by fixed A/B position."""

    if preserve_label not in activations_by_label:
        raise KeyError(f"missing preserve label activation: {preserve_label}")
    if comply_label not in activations_by_label:
        raise KeyError(f"missing comply label activation: {comply_label}")
    return SemanticActivationPair(
        preserve_activation=activations_by_label[preserve_label],
        comply_activation=activations_by_label[comply_label],
        preserve_label=preserve_label,
        comply_label=comply_label,
        case_id=case_id,
    )


def construct_caa_direction(
    torch: Any, pairs: Sequence[SemanticActivationPair]
) -> tuple[Any, dict[str, Any]]:
    """Mean answer-token activation difference, preserve minus comply (CAA)."""

    items = list(pairs)
    if not items:
        raise ValueError("CAA requires at least one semantic activation pair")
    deltas = []
    label_orders: dict[str, int] = {}
    for index, pair in enumerate(items):
        preserve = _validate_vector(pair.preserve_activation, f"pairs[{index}].preserve_activation")
        comply = _validate_vector(pair.comply_activation, f"pairs[{index}].comply_activation")
        if preserve.shape != comply.shape:
            raise ValueError(f"CAA pair {index} activations must have the same shape")
        deltas.append(preserve - comply)
        order = f"{pair.preserve_label}>{pair.comply_label}"
        label_orders[order] = label_orders.get(order, 0) + 1
    raw = _mean_vector(torch, deltas, "CAA semantic differences")
    direction = normalize_direction(torch, raw)
    diagnostics = {
        "n_pairs": len(items),
        "d_model": int(raw.numel()),
        "raw_direction_norm": float(raw.norm().detach().item()),
        "semantic_difference": "preserve_activation_minus_comply_activation",
        "label_orders": dict(sorted(label_orders.items())),
    }
    return direction.detach().cpu().float().contiguous(), diagnostics


def completion_logprob_sums(
    torch: Any,
    logits: Any,
    token_ids: Any,
    completion_mask: Any,
) -> Any:
    """Sum causal-LM log probabilities only over completion tokens."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if token_ids.ndim != 2 or completion_mask.ndim != 2:
        raise ValueError("token_ids and completion_mask must have shape [batch, sequence]")
    if logits.shape[:2] != token_ids.shape or token_ids.shape != completion_mask.shape:
        raise ValueError("logits, token_ids, and completion_mask batch/sequence shapes must match")
    if logits.shape[1] < 2:
        raise ValueError("at least two sequence positions are required for causal scoring")

    labels = token_ids[:, 1:].to(device=logits.device, dtype=torch.long)
    selected = completion_mask[:, 1:].to(device=logits.device, dtype=torch.bool)
    counts = selected.sum(dim=-1)
    if bool((counts == 0).any().item()):
        raise ValueError("every row must select at least one completion token")
    vocabulary = logits.shape[-1]
    invalid_selected = selected & ((labels < 0) | (labels >= vocabulary))
    if bool(invalid_selected.any().item()):
        raise ValueError("a selected completion token id is outside the vocabulary")
    safe_labels = labels.clamp(min=0, max=vocabulary - 1)
    log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    token_logps = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * selected.to(dtype=token_logps.dtype)).sum(dim=-1)


def sample_bipo_direction(torch: Any, *, generator: Any | None = None) -> int:
    """Sample the single minibatch coefficient d ~ Uniform({-1, +1})."""

    draw = int(torch.randint(0, 2, (1,), generator=generator, device="cpu").item())
    return 1 if draw else -1


def bipo_loss(
    torch: Any,
    policy_target_logps: Any,
    policy_opposite_logps: Any,
    reference_target_logps: Any,
    reference_opposite_logps: Any,
    direction_sign: Any,
    *,
    beta: float = 0.1,
    reduction: str = "mean",
) -> Any:
    """Exact BiPO Eq. 3 loss using cached, detached reference log probabilities.

    The policy log probabilities must be computed with the intervention ``d * v``.
    Each input log probability is the sum over the corresponding completion, not a
    token average.
    """

    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    target = policy_target_logps.float()
    opposite = policy_opposite_logps.float()
    if target.shape != opposite.shape:
        raise ValueError("target and opposite policy log probabilities must match")
    reference_target = reference_target_logps.detach().to(device=target.device, dtype=target.dtype)
    reference_opposite = reference_opposite_logps.detach().to(
        device=target.device, dtype=target.dtype
    )
    if reference_target.shape != target.shape or reference_opposite.shape != target.shape:
        raise ValueError("cached reference and policy log probabilities must have matching shapes")
    sign = torch.as_tensor(direction_sign, device=target.device, dtype=target.dtype)
    if not bool(((sign == 1) | (sign == -1)).all().item()):
        raise ValueError("direction_sign must contain only -1 or +1")
    if sign.ndim > 0 and sign.shape != target.shape:
        raise ValueError("a non-scalar direction_sign must match the log-probability shape")

    log_ratio_gap = (target - reference_target) - (opposite - reference_opposite)
    margin = sign * beta * log_ratio_gap
    losses = -torch.nn.functional.logsigmoid(margin)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError("reduction must be one of: none, sum, mean")


def initialize_bipo_vector(
    torch: Any,
    d_model: int,
    *,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Any:
    """Create the only trainable BiPO parameter with model-derived width."""

    if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model <= 0:
        raise ValueError("d_model must be a positive integer")
    resolved_dtype = torch.float32 if dtype is None else dtype
    return torch.nn.Parameter(torch.zeros(d_model, device=device, dtype=resolved_dtype))


def freeze_model_parameters(model: Any) -> int:
    """Freeze and clear gradients for every model parameter; call before vector creation."""

    count = 0
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
        count += int(parameter.numel())
    return count


def assert_model_frozen(model: Any) -> None:
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable:
        preview = ", ".join(trainable[:5])
        raise ValueError(f"BiPO assumes frozen model parameters; trainable parameters: {preview}")


def _batch_coefficient(torch: Any, coefficient: Any, activations: Any) -> Any:
    value = torch.as_tensor(coefficient, device=activations.device, dtype=activations.dtype)
    if value.ndim == 0:
        return value.reshape(1, 1, 1)
    if value.ndim == 1 and value.shape[0] == activations.shape[0]:
        return value.reshape(activations.shape[0], 1, 1)
    raise ValueError("coefficient must be scalar or have one value per batch row")


def apply_steering_vector(
    torch: Any,
    activations: Any,
    vector: Any,
    coefficient: Any = 1.0,
    *,
    geometry: str,
    final_prompt_indices: Any | None = None,
) -> Any:
    """Apply either the fair matched geometry or BiPO's canonical broadcast geometry."""

    if activations.ndim != 3:
        raise ValueError("activations must have shape [batch, sequence, d_model]")
    unit_or_vector = _validate_vector(vector, "vector").to(
        device=activations.device, dtype=activations.dtype
    )
    if unit_or_vector.shape[0] != activations.shape[-1]:
        raise ValueError("vector width must equal activation d_model")
    scaled = _batch_coefficient(torch, coefficient, activations) * unit_or_vector.reshape(1, 1, -1)

    if geometry == CANONICAL_BROADCAST:
        if final_prompt_indices is not None:
            raise ValueError("final_prompt_indices are not used for canonical broadcast geometry")
        return activations + scaled
    if geometry != MATCHED_FINAL_PROMPT:
        raise ValueError(f"unknown intervention geometry: {geometry}")
    if final_prompt_indices is None:
        raise ValueError("matched final-prompt geometry requires final_prompt_indices")
    indices = torch.as_tensor(final_prompt_indices, device=activations.device, dtype=torch.long)
    if indices.ndim != 1 or indices.shape[0] != activations.shape[0]:
        raise ValueError("final_prompt_indices must contain one index per batch row")
    if bool(((indices < 0) | (indices >= activations.shape[1])).any().item()):
        raise ValueError("final_prompt_indices contain an out-of-range position")
    positions = torch.arange(activations.shape[1], device=activations.device).reshape(1, -1)
    mask = (positions == indices.reshape(-1, 1)).to(dtype=activations.dtype).unsqueeze(-1)
    return activations + mask * scaled


def masked_token_mean(torch: Any, activations: Any, mask: Any) -> Any:
    """Return one response-token mean per batch row."""

    if activations.ndim != 3:
        raise ValueError("activations must have shape [batch, sequence, d_model]")
    if mask.ndim != 2 or mask.shape != activations.shape[:2]:
        raise ValueError("mask must have shape [batch, sequence]")
    selected = mask.to(device=activations.device, dtype=torch.bool)
    counts = selected.sum(dim=-1)
    if bool((counts == 0).any().item()):
        raise ValueError("every response must select at least one response token")
    weights = selected.to(dtype=activations.dtype).unsqueeze(-1)
    return (activations * weights).sum(dim=1) / counts.to(activations.dtype).unsqueeze(-1)


def _score_tensor(torch: Any, values: Any, *, n_rows: int, name: str, device: Any) -> Any:
    scores = torch.as_tensor(values, dtype=torch.float32, device=device)
    if scores.ndim != 1 or scores.shape[0] != n_rows:
        raise ValueError(f"{name} must contain one score per response pair")
    if not bool(scores.isfinite().all().item()):
        raise ValueError(f"{name} must contain only finite scores")
    if bool(((scores < 0) | (scores > 100)).any().item()):
        raise ValueError(f"{name} scores must be between 0 and 100")
    return scores


def construct_persona_direction(
    torch: Any,
    positive_activations: Any,
    negative_activations: Any,
    positive_response_mask: Any,
    negative_response_mask: Any,
    positive_scores: Any,
    negative_scores: Any,
    positive_coherence: Any,
    negative_coherence: Any,
    *,
    trait_threshold: float = 50.0,
    coherence_threshold: float = 50.0,
    min_retained_pairs: int = 2,
) -> tuple[Any, dict[str, Any]]:
    """Construct the published response-average persona-vector baseline.

    The authors' released code uses ``positive >= threshold`` but
    ``negative < 100 - threshold``. At the default 50 boundary, a positive score
    of exactly 50 is retained while a negative score of exactly 50 is not. We
    intentionally preserve and report that code-level convention.
    """

    if not 0 <= trait_threshold <= 100 or not math.isfinite(trait_threshold):
        raise ValueError("trait_threshold must be finite and between 0 and 100")
    if not 0 <= coherence_threshold <= 100 or not math.isfinite(coherence_threshold):
        raise ValueError("coherence_threshold must be finite and between 0 and 100")
    if not isinstance(min_retained_pairs, int) or min_retained_pairs < 1:
        raise ValueError("min_retained_pairs must be a positive integer")
    if positive_activations.ndim != 3 or negative_activations.ndim != 3:
        raise ValueError("positive and negative activations must be rank-three tensors")
    n_rows = positive_activations.shape[0]
    if negative_activations.shape[0] != n_rows:
        raise ValueError("positive and negative activations must have the same pair count")
    if positive_activations.shape[-1] != negative_activations.shape[-1]:
        raise ValueError("positive and negative activations must have the same d_model")

    positive_means = masked_token_mean(torch, positive_activations.float(), positive_response_mask)
    negative_means = masked_token_mean(torch, negative_activations.float(), negative_response_mask)
    device = positive_means.device
    negative_means = negative_means.to(device=device)
    pos_scores = _score_tensor(
        torch, positive_scores, n_rows=n_rows, name="positive_scores", device=device
    )
    neg_scores = _score_tensor(
        torch, negative_scores, n_rows=n_rows, name="negative_scores", device=device
    )
    pos_coherence = _score_tensor(
        torch, positive_coherence, n_rows=n_rows, name="positive_coherence", device=device
    )
    neg_coherence = _score_tensor(
        torch, negative_coherence, n_rows=n_rows, name="negative_coherence", device=device
    )
    retained = (
        (pos_scores >= trait_threshold)
        & (neg_scores < 100.0 - trait_threshold)
        & (pos_coherence >= coherence_threshold)
        & (neg_coherence >= coherence_threshold)
    )
    retained_indices = retained.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
    if len(retained_indices) < min_retained_pairs:
        raise ValueError(
            "persona-vector filtering retained "
            f"{len(retained_indices)} pairs, fewer than minimum {min_retained_pairs}"
        )
    raw = positive_means[retained].mean(dim=0) - negative_means[retained].mean(dim=0)
    direction = normalize_direction(torch, raw)
    diagnostics = {
        "n_pairs": n_rows,
        "n_retained_pairs": len(retained_indices),
        "retained_pair_indices": retained_indices,
        "d_model": int(raw.numel()),
        "raw_direction_norm": float(raw.norm().detach().item()),
        "trait_threshold": float(trait_threshold),
        "coherence_threshold": float(coherence_threshold),
        "positive_score_rule": ">= trait_threshold",
        "negative_score_rule": "< 100 - trait_threshold",
        "coherence_rule": ">= coherence_threshold",
        "boundary_note": (
            "faithful to released code: at threshold 50, positive score 50 is retained "
            "but negative score 50 is excluded"
        ),
        "pooling": "per_response_masked_token_mean_then_mean_difference",
    }
    return direction.detach().cpu().float().contiguous(), diagnostics


def random_orthogonal_controls(
    torch: Any,
    reference: Any,
    *,
    count: int,
    seed: int,
    mutually_orthogonal: bool = False,
    max_attempts: int = 1000,
) -> list[Any]:
    """Generate deterministic unit controls orthogonal to a reference direction."""

    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    reference_unit = normalize_direction(torch, reference.detach().cpu().float())
    d_model = int(reference_unit.numel())
    if d_model < 2:
        raise ValueError("orthogonal controls require d_model of at least two")
    if mutually_orthogonal and count > d_model - 1:
        raise ValueError("at most d_model - 1 mutually orthogonal controls are possible")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    controls: list[Any] = []
    basis = [reference_unit]
    attempts = 0
    while len(controls) < count and attempts < max_attempts:
        attempts += 1
        candidate = torch.randn(reference_unit.shape, generator=generator, dtype=torch.float32)
        projection_basis = basis if mutually_orthogonal else [reference_unit]
        for unit in projection_basis:
            candidate = candidate - (candidate @ unit) * unit
        norm = float(candidate.norm().item())
        if not math.isfinite(norm) or norm <= 1e-7:
            continue
        candidate = candidate / candidate.norm()
        controls.append(candidate.contiguous())
        if mutually_orthogonal:
            basis.append(candidate)
    if len(controls) != count:
        raise RuntimeError("could not generate the requested orthogonal controls")
    return controls


def actual_perturbation_norms(
    torch: Any,
    before: Any,
    after: Any,
    *,
    position_mask: Any | None = None,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Measure realized residual-stream changes at the explicitly selected positions."""

    if before.shape != after.shape or before.ndim < 2:
        raise ValueError("before and after must have the same [..., d_model] shape")
    before_float = before.float()
    after_float = after.to(device=before.device).float()
    if not bool(before_float.isfinite().all().item()) or not bool(after_float.isfinite().all().item()):
        raise ValueError("before and after must contain only finite activations")
    leading_shape = before.shape[:-1]
    if position_mask is None:
        selected = torch.ones(leading_shape, dtype=torch.bool, device=before.device)
    else:
        if tuple(position_mask.shape) != tuple(leading_shape):
            raise ValueError("position_mask shape must equal the activation leading dimensions")
        selected = position_mask.to(device=before.device, dtype=torch.bool)
    if not bool(selected.any().item()):
        raise ValueError("position_mask must select at least one position")

    flat_before = before_float.reshape(-1, before.shape[-1])[selected.reshape(-1)]
    flat_delta = (after_float - before_float).reshape(-1, before.shape[-1])[selected.reshape(-1)]
    delta_norms = flat_delta.norm(dim=-1)
    reference_norms = flat_before.norm(dim=-1)
    relative = delta_norms / reference_norms.clamp_min(eps)
    return {
        "n_positions": int(delta_norms.numel()),
        "total_frobenius_norm": float(flat_delta.norm().detach().item()),
        "mean_l2_norm": float(delta_norms.mean().detach().item()),
        "rms_l2_norm": float(delta_norms.square().mean().sqrt().detach().item()),
        "max_l2_norm": float(delta_norms.max().detach().item()),
        "mean_relative_l2_norm": float(relative.mean().detach().item()),
        "max_relative_l2_norm": float(relative.max().detach().item()),
        "zero_reference_positions": int((reference_norms <= eps).sum().detach().item()),
    }
