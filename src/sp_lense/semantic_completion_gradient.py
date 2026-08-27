from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from .comparison_runtime import (
    ChoiceBoundaryEvidence,
    encode_prompt_and_completion,
    resolve_choice_boundary,
)
from .gradient_specificity_v3 import tensor_float32_sha256
from .steering_methods import completion_logprob_sums

DEFAULT_CAUSAL_RESIDUAL_RELATIVE_L2_TOLERANCE = 1e-5


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validated_layer(layer: int) -> int:
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise ValueError("layer must be a non-negative integer")
    return layer


def _single_token_sequence(torch: Any, value: Any, *, field: str) -> Any:
    if not hasattr(value, "ndim") or value.ndim != 2 or int(value.shape[0]) != 1:
        raise ValueError(f"{field} must have shape [1, sequence]")
    if int(value.shape[1]) < 1:
        raise ValueError(f"{field} must contain at least one token")
    if value.dtype == torch.bool or value.dtype.is_floating_point:
        raise TypeError(f"{field} must contain integer token IDs")
    return value


def _finite_vector(torch: Any, value: Any, *, field: str) -> Any:
    if not hasattr(value, "detach"):
        raise TypeError(f"{field} must be a tensor")
    vector = value.detach().cpu().double().contiguous()
    if vector.ndim != 1 or vector.numel() == 0:
        raise ValueError(f"{field} must be a non-empty vector")
    if not bool(torch.isfinite(vector).all().item()):
        raise ValueError(f"{field} must contain only finite values")
    return vector


@dataclass(frozen=True)
class AuthoredCompletionEncoding:
    """One jointly tokenized assistant completion with an explicit content-only mask."""

    prompt_tokens: Any
    full_tokens: Any
    content_mask: Any
    prompt_length: int
    prompt_final_index: int
    content_token_ids: tuple[int, ...]
    assistant_end_token_ids: tuple[int, ...]
    choice_boundary_evidence_sha256: str
    prompt_token_ids_sha256: str
    completion_text_sha256: str

    @property
    def content_token_count(self) -> int:
        return len(self.content_token_ids)

    def audit_record(self) -> dict[str, Any]:
        return {
            "prompt_length": self.prompt_length,
            "prompt_final_index": self.prompt_final_index,
            "content_token_count": self.content_token_count,
            "content_token_ids_sha256": _canonical_sha256(list(self.content_token_ids)),
            "assistant_end_token_ids": list(self.assistant_end_token_ids),
            "assistant_end_excluded_from_objective": True,
            "joint_chat_template_tokenization": True,
            "choice_boundary_evidence_sha256": self.choice_boundary_evidence_sha256,
            "prompt_token_ids_sha256": self.prompt_token_ids_sha256,
            "completion_text_sha256": self.completion_text_sha256,
        }


@dataclass(frozen=True)
class PromptFinalResidualCapture:
    residual: Any
    audit: dict[str, Any]


@dataclass(frozen=True)
class AuthoredCompletionGradientCapture:
    raw_gradient: Any
    prompt_residual: Any
    mean_log_probability: float
    encoding: AuthoredCompletionEncoding
    audit: dict[str, Any]


@dataclass(frozen=True)
class SemanticCompletionGradientCapture:
    effective_gradient: Any
    prompt_residual: Any
    preserve: AuthoredCompletionGradientCapture
    comply: AuthoredCompletionGradientCapture
    causal_residual_audit: dict[str, Any]
    audit: dict[str, Any]


def encode_authored_completion(
    backend: Any,
    prompt: str,
    completion: str,
    *,
    boundary: ChoiceBoundaryEvidence | None = None,
) -> AuthoredCompletionEncoding:
    """Tokenize at the real assistant boundary and exclude only the final template EOM.

    The content IDs are derived from the full joint chat template. Independent suffix
    tokenization is deliberately not used because it can change boundary segmentation.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(completion, str) or not completion.strip():
        raise ValueError("completion must contain non-whitespace authored text")
    torch = backend.torch
    resolved = resolve_choice_boundary(backend, prompt) if boundary is None else boundary
    prompt_tokens, full_tokens = encode_prompt_and_completion(
        backend,
        prompt,
        completion,
        include_chat_end=True,
    )
    prompt_tokens = _single_token_sequence(torch, prompt_tokens, field="prompt_tokens")
    full_tokens = _single_token_sequence(torch, full_tokens, field="full_tokens")
    prompt_length = int(prompt_tokens.shape[1])
    if resolved.prompt_length != prompt_length:
        raise ValueError("completion boundary evidence has the wrong prompt length")
    if int(full_tokens.shape[1]) <= prompt_length:
        raise ValueError("joint chat template produced no assistant suffix")

    end_ids = tuple(int(value) for value in resolved.assistant_end_token_ids)
    if not end_ids:
        raise ValueError("assistant boundary has no verified end-of-message tokens")
    suffix_ids = tuple(int(value) for value in full_tokens[0, prompt_length:].tolist())
    if len(suffix_ids) < len(end_ids) or suffix_ids[-len(end_ids) :] != end_ids:
        raise ValueError("joint authored completion does not end in the verified assistant EOM")
    if len(suffix_ids) == len(end_ids):
        raise ValueError("joint chat template produced no authored completion content tokens")
    content_ids = suffix_ids[: -len(end_ids)]

    content_stop = prompt_length + len(content_ids)
    content_mask = torch.zeros_like(full_tokens, dtype=torch.bool)
    content_mask[:, prompt_length:content_stop] = True
    if int(content_mask.sum().item()) != len(content_ids):
        raise RuntimeError("authored completion mask has the wrong number of selected tokens")
    if bool(content_mask[:, content_stop:].any().item()):
        raise RuntimeError("assistant end-of-message tokens leaked into the content mask")

    prompt_ids = [int(value) for value in prompt_tokens[0].tolist()]
    if _canonical_sha256(prompt_ids) != resolved.prompt_prefix_token_ids_sha256:
        raise RuntimeError("joint completion prompt hash differs from boundary evidence")
    return AuthoredCompletionEncoding(
        prompt_tokens=prompt_tokens,
        full_tokens=full_tokens,
        content_mask=content_mask,
        prompt_length=prompt_length,
        prompt_final_index=prompt_length - 1,
        content_token_ids=content_ids,
        assistant_end_token_ids=end_ids,
        choice_boundary_evidence_sha256=resolved.evidence_sha256,
        prompt_token_ids_sha256=resolved.prompt_prefix_token_ids_sha256,
        completion_text_sha256=_text_sha256(completion),
    )


def authored_completion_mean_logprob(
    torch: Any,
    logits: Any,
    encoding: AuthoredCompletionEncoding,
) -> Any:
    """Return mean log probability over authored content, excluding template terminators."""

    if not hasattr(logits, "ndim") or logits.ndim != 3:
        raise ValueError("logits must have shape [1, sequence, vocabulary]")
    if int(logits.shape[0]) != 1 or tuple(logits.shape[:2]) != tuple(encoding.full_tokens.shape):
        raise ValueError("logits do not match the jointly tokenized completion")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("completion logits contain a non-finite value")
    summed = completion_logprob_sums(
        torch,
        logits,
        encoding.full_tokens,
        encoding.content_mask,
    )
    if summed.ndim != 1 or int(summed.numel()) != 1:
        raise RuntimeError("authored completion scorer did not return one value")
    result = summed[0] / encoding.content_token_count
    if not bool(torch.isfinite(result).item()):
        raise RuntimeError("authored completion mean log probability is non-finite")
    return result


def validate_causal_prompt_residuals(
    torch: Any,
    prompt_only: Any,
    preserve_continuation: Any,
    comply_continuation: Any,
    *,
    tolerance: float = DEFAULT_CAUSAL_RESIDUAL_RELATIVE_L2_TOLERANCE,
) -> dict[str, Any]:
    """Verify that continuation length does not materially change the causal prefix."""

    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        raise TypeError("causal residual tolerance must be a number")
    allowed = float(tolerance)
    if not math.isfinite(allowed) or allowed <= 0.0:
        raise ValueError("causal residual tolerance must be finite and positive")
    prompt_vector = _finite_vector(torch, prompt_only, field="prompt_only")
    preserve_vector = _finite_vector(torch, preserve_continuation, field="preserve_continuation")
    comply_vector = _finite_vector(torch, comply_continuation, field="comply_continuation")
    if prompt_vector.shape != preserve_vector.shape or prompt_vector.shape != comply_vector.shape:
        raise ValueError("all causal prompt residuals must have the same shape")
    reference_norm = float(prompt_vector.norm().item())
    if not math.isfinite(reference_norm) or reference_norm <= 0.0:
        raise ValueError("prompt-only residual norm must be finite and positive")

    differences = {
        "preserve_vs_prompt_residual_relative_l2": float(
            (preserve_vector - prompt_vector).norm().item() / reference_norm
        ),
        "comply_vs_prompt_residual_relative_l2": float(
            (comply_vector - prompt_vector).norm().item() / reference_norm
        ),
        "preserve_vs_comply_residual_relative_l2": float(
            (preserve_vector - comply_vector).norm().item() / reference_norm
        ),
    }
    maximum = max(differences.values())
    audit = {
        **differences,
        "maximum_causal_residual_relative_l2": maximum,
        "maximum_allowed_causal_residual_relative_l2": allowed,
        "reference_prompt_residual_norm": reference_norm,
        "causal_prompt_residuals_within_tolerance": maximum <= allowed,
    }
    if maximum > allowed:
        raise RuntimeError(
            "completion sequence changed the causal prompt residual beyond tolerance: "
            f"observed={maximum}, allowed={allowed}"
        )
    return audit


def capture_prompt_final_residual(
    backend: Any,
    prompt: str,
    *,
    layer: int,
) -> PromptFinalResidualCapture:
    """Capture the prompt-only residual used as the common coordinate and scale."""

    _validated_layer(layer)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    torch = backend.torch
    tokens = _single_token_sequence(torch, backend.encode(prompt), field="prompt_tokens")
    prompt_length = int(tokens.shape[1])
    prompt_final_index = prompt_length - 1
    captured: dict[str, Any] = {"hook_calls": 0}

    def capture_hook(activation: Any, hook: Any) -> Any:
        del hook
        captured["hook_calls"] += 1
        if captured["hook_calls"] != 1:
            raise RuntimeError("prompt residual hook fired more than once")
        captured["activation"] = activation.detach()
        return activation

    with (
        torch.inference_mode(),
        backend.model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_out", capture_hook)]),
    ):
        backend.model(tokens)
    activation = captured.get("activation")
    if activation is None:
        raise RuntimeError("prompt residual hook did not capture an activation")
    if (
        activation.ndim != 3
        or int(activation.shape[0]) != 1
        or int(activation.shape[1]) != prompt_length
    ):
        raise RuntimeError("prompt residual activation does not match the prompt sequence")
    residual = activation[0, prompt_final_index].detach().cpu().float().contiguous()
    _finite_vector(torch, residual, field="prompt_final_residual")
    residual_norm = float(residual.norm().item())
    if not math.isfinite(residual_norm) or residual_norm <= 0.0:
        raise RuntimeError("prompt-final residual norm must be positive")
    prompt_ids = [int(value) for value in tokens[0].tolist()]
    return PromptFinalResidualCapture(
        residual=residual,
        audit={
            "prompt_length": prompt_length,
            "prompt_final_index": prompt_final_index,
            "prompt_token_ids_sha256": _canonical_sha256(prompt_ids),
            "prompt_residual_sha256": tensor_float32_sha256(residual),
            "residual_norm": residual_norm,
            "hook_call_count": int(captured["hook_calls"]),
        },
    )


def capture_authored_completion_mean_logprob_gradient(
    backend: Any,
    prompt: str,
    completion: str,
    *,
    layer: int,
    boundary: ChoiceBoundaryEvidence | None = None,
) -> AuthoredCompletionGradientCapture:
    """Capture a content-only likelihood gradient at the final prompt residual position."""

    _validated_layer(layer)
    torch = backend.torch
    encoding = encode_authored_completion(
        backend,
        prompt,
        completion,
        boundary=boundary,
    )
    captured: dict[str, Any] = {"hook_calls": 0}

    def capture_hook(activation: Any, hook: Any) -> Any:
        del hook
        captured["hook_calls"] += 1
        if captured["hook_calls"] != 1:
            raise RuntimeError("completion gradient hook fired more than once")
        leaf = activation.detach().requires_grad_(True)
        captured["activation"] = leaf
        return leaf

    backend.model.zero_grad(set_to_none=True)
    parameter_gradients_allocated = False
    try:
        with (
            torch.enable_grad(),
            backend.model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_out", capture_hook)]),
        ):
            logits = backend.model(encoding.full_tokens)
            activation = captured.get("activation")
            if activation is None:
                raise RuntimeError("completion gradient hook did not capture an activation")
            if (
                activation.ndim != 3
                or int(activation.shape[0]) != 1
                or int(activation.shape[1]) != int(encoding.full_tokens.shape[1])
            ):
                raise RuntimeError(
                    "completion residual activation does not match the joint sequence"
                )
            objective = authored_completion_mean_logprob(torch, logits, encoding)
            full_gradient = torch.autograd.grad(
                objective,
                activation,
                retain_graph=False,
                create_graph=False,
            )[0]
            parameter_gradients_allocated = any(
                parameter.grad is not None for parameter in backend.model.parameters()
            )
            if parameter_gradients_allocated:
                raise RuntimeError("completion capture allocated model parameter gradients")
    finally:
        backend.model.zero_grad(set_to_none=True)

    if full_gradient.shape != activation.shape:
        raise RuntimeError("completion gradient shape differs from its residual activation")
    index = encoding.prompt_final_index
    raw_gradient = full_gradient[0, index].detach().cpu().float().contiguous()
    prompt_residual = activation[0, index].detach().cpu().float().contiguous()
    checked_gradient = _finite_vector(torch, raw_gradient, field="completion_raw_gradient")
    _finite_vector(torch, prompt_residual, field="completion_prompt_residual")
    residual_norm = float(prompt_residual.norm().item())
    if not math.isfinite(residual_norm) or residual_norm <= 0.0:
        raise RuntimeError("completion prompt-final residual norm must be positive")
    mean_log_probability = float(objective.detach().item())
    audit = {
        **encoding.audit_record(),
        "objective_name": "mean_authored_content_log_probability",
        "mean_log_probability": mean_log_probability,
        "gradient_position": "final_prompt_token",
        "raw_gradient_norm": float(checked_gradient.norm().item()),
        "raw_gradient_sha256": tensor_float32_sha256(raw_gradient),
        "prompt_residual_norm": residual_norm,
        "prompt_residual_sha256": tensor_float32_sha256(prompt_residual),
        "hook_call_count": int(captured["hook_calls"]),
        "model_parameter_gradients_allocated": parameter_gradients_allocated,
    }
    return AuthoredCompletionGradientCapture(
        raw_gradient=raw_gradient,
        prompt_residual=prompt_residual,
        mean_log_probability=mean_log_probability,
        encoding=encoding,
        audit=audit,
    )


def capture_semantic_completion_gradient(
    backend: Any,
    prompt: str,
    preserve_completion: str,
    comply_completion: str,
    *,
    layer: int,
    causal_residual_tolerance: float = DEFAULT_CAUSAL_RESIDUAL_RELATIVE_L2_TOLERANCE,
) -> SemanticCompletionGradientCapture:
    """Capture residual-scaled preserve-minus-comply authored-completion gradient."""

    _validated_layer(layer)
    if preserve_completion == comply_completion:
        raise ValueError("preserve and comply completions must differ")
    boundary = resolve_choice_boundary(backend, prompt)
    prompt_only = capture_prompt_final_residual(backend, prompt, layer=layer)
    preserve = capture_authored_completion_mean_logprob_gradient(
        backend,
        prompt,
        preserve_completion,
        layer=layer,
        boundary=boundary,
    )
    comply = capture_authored_completion_mean_logprob_gradient(
        backend,
        prompt,
        comply_completion,
        layer=layer,
        boundary=boundary,
    )
    if preserve.encoding.content_token_ids == comply.encoding.content_token_ids:
        raise ValueError("preserve and comply completions have identical joint content tokens")
    prompt_hashes = {
        str(prompt_only.audit["prompt_token_ids_sha256"]),
        preserve.encoding.prompt_token_ids_sha256,
        comply.encoding.prompt_token_ids_sha256,
    }
    if len(prompt_hashes) != 1:
        raise RuntimeError("semantic completion captures do not share one exact prompt prefix")
    causal_audit = validate_causal_prompt_residuals(
        backend.torch,
        prompt_only.residual,
        preserve.prompt_residual,
        comply.prompt_residual,
        tolerance=causal_residual_tolerance,
    )
    common_residual_norm = float(prompt_only.residual.norm().item())
    effective = (
        (common_residual_norm * (preserve.raw_gradient - comply.raw_gradient)).float().contiguous()
    )
    checked = _finite_vector(backend.torch, effective, field="semantic_completion_gradient")
    semantic_objective = preserve.mean_log_probability - comply.mean_log_probability
    audit = {
        "objective_name": "mean_authored_content_logp_preserve_minus_comply",
        "gradient_coordinate": "common_prompt_residual_scaled_final_prompt",
        "layer": layer,
        "prompt_length": preserve.encoding.prompt_length,
        "prompt_final_index": preserve.encoding.prompt_final_index,
        "prompt_token_ids_sha256": preserve.encoding.prompt_token_ids_sha256,
        "choice_boundary_evidence_sha256": boundary.evidence_sha256,
        "assistant_end_token_ids": list(preserve.encoding.assistant_end_token_ids),
        "assistant_end_excluded_from_both_objectives": True,
        "semantic_objective_value": semantic_objective,
        "common_prompt_residual_norm": common_residual_norm,
        "common_prompt_residual_norm_computation_dtype": "torch.float32",
        "effective_gradient_norm": float(checked.norm().item()),
        "effective_gradient_sha256": tensor_float32_sha256(effective),
        "prompt_only": dict(prompt_only.audit),
        "preserve": dict(preserve.audit),
        "comply": dict(comply.audit),
        "causal_prompt_residuals": dict(causal_audit),
    }
    return SemanticCompletionGradientCapture(
        effective_gradient=effective,
        prompt_residual=prompt_only.residual,
        preserve=preserve,
        comply=comply,
        causal_residual_audit=causal_audit,
        audit=audit,
    )
