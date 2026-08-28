"""Model-facing capture helpers for a shared pre-answer causal anchor."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .comparison_runtime import resolve_choice_boundary
from .factorial_causal_anchor import (
    canonical_sha256,
    resolve_shared_anchor_index,
    tensor_float32_sha256,
    text_sha256,
)
from .semantic_completion_gradient import (
    authored_completion_mean_logprob,
    encode_authored_completion,
)

DEFAULT_CAUSAL_ANCHOR_RESIDUAL_RELATIVE_L2_TOLERANCE = 1e-5


@dataclass(frozen=True)
class MultilayerAnchorObjectiveCapture:
    raw_gradients: Any
    anchor_residuals: Any
    mean_log_probability: float
    audit: dict[str, Any]


@dataclass(frozen=True)
class MultilayerSemanticAnchorCapture:
    raw_semantic_gradients: Any
    reference_anchor_residuals: Any
    preserve: MultilayerAnchorObjectiveCapture
    comply: MultilayerAnchorObjectiveCapture
    audit: dict[str, Any]


@dataclass(frozen=True)
class SharedAnchorEvidence:
    anchor_index: int
    shared_prefix_length: int
    shared_token_prefix_sha256: str
    prompt_token_sha256s: tuple[str, ...]
    audit: dict[str, Any]


def _checked_layers(layers: Sequence[int]) -> tuple[int, ...]:
    result = tuple(map(int, layers))
    if not result or len(set(result)) != len(result) or any(layer < 0 for layer in result):
        raise ValueError("layers must be non-empty, unique, and non-negative")
    if result != tuple(sorted(result)):
        raise ValueError("layers must be in ascending order")
    return result


def _checked_anchor_index(anchor_index: int) -> int:
    if isinstance(anchor_index, bool) or not isinstance(anchor_index, int) or anchor_index < 0:
        raise ValueError("anchor_index must be a non-negative integer")
    return anchor_index


def _token_rows(backend: Any, prompts: Sequence[str]) -> tuple[list[list[int]], tuple[str, ...]]:
    if not prompts or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise ValueError("prompts must be a non-empty sequence of non-empty strings")
    rows = []
    hashes = []
    for prompt in prompts:
        tokens = backend.encode(prompt)
        if tokens.ndim != 2 or int(tokens.shape[0]) != 1 or int(tokens.shape[1]) < 1:
            raise ValueError("backend.encode must return one non-empty token row")
        row = [int(value) for value in tokens[0].detach().cpu().tolist()]
        rows.append(row)
        hashes.append(canonical_sha256(row))
    return rows, tuple(hashes)


def resolve_shared_anchor_evidence(
    backend: Any,
    *,
    anchor_prefix: str,
    prompts: Sequence[str],
    anchor_marker: str,
) -> SharedAnchorEvidence:
    """Verify exact causal token-prefix identity across construction/evaluation views."""

    if not isinstance(anchor_prefix, str) or not anchor_prefix:
        raise ValueError("anchor_prefix must be non-empty")
    if not isinstance(anchor_marker, str) or not anchor_marker:
        raise ValueError("anchor_marker must be non-empty")
    if not anchor_prefix.rstrip().endswith(anchor_marker):
        raise ValueError("anchor_prefix must end with the literal anchor marker")
    if any(not prompt.startswith(anchor_prefix) for prompt in prompts):
        raise ValueError("every prompt must share the exact textual anchor prefix")
    rows, hashes = _token_rows(backend, prompts)
    anchor_index = resolve_shared_anchor_index(rows)
    shared = rows[0][: anchor_index + 1]
    if any(row[: anchor_index + 1] != shared for row in rows[1:]):
        raise RuntimeError("shared anchor token prefix is not exact")
    tokenizer = backend.model.tokenizer
    try:
        decoded = tokenizer.decode(
            shared,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:  # pragma: no cover - compatibility with minimal test tokenizers
        decoded = tokenizer.decode(shared, skip_special_tokens=False)
    if anchor_marker not in decoded:
        raise ValueError("shared token prefix ends before the declared anchor marker")
    audit = {
        "schema_version": "sp_lense.shared_causal_anchor_evidence.v1",
        "anchor_position": "last_token_of_longest_shared_prompt_prefix",
        "anchor_index": anchor_index,
        "shared_prefix_length": anchor_index + 1,
        "shared_token_prefix_sha256": canonical_sha256(shared),
        "prompt_token_sha256s": list(hashes),
        "prompt_count": len(prompts),
        "anchor_prefix_text_sha256": text_sha256(anchor_prefix),
        "anchor_marker": anchor_marker,
        "anchor_marker_present_in_decoded_shared_prefix": True,
        "future_suffix_cannot_change_anchor_by_causal_mask": True,
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return SharedAnchorEvidence(
        anchor_index=anchor_index,
        shared_prefix_length=anchor_index + 1,
        shared_token_prefix_sha256=canonical_sha256(shared),
        prompt_token_sha256s=hashes,
        audit=audit,
    )


def _capture_prompt_anchor_residuals(
    backend: Any,
    prompt: str,
    *,
    layers: tuple[int, ...],
    anchor_index: int,
    before_forward: Callable[[str], None] | None,
) -> tuple[Any, dict[str, Any]]:
    torch = backend.torch
    tokens = backend.encode(prompt)
    sequence_length = int(tokens.shape[1])
    if anchor_index >= sequence_length:
        raise ValueError("anchor index lies outside the construction prompt")
    captures: dict[int, Any] = {}
    hook_calls = {layer: 0 for layer in layers}

    def hook_for(layer: int) -> Any:
        def capture(activation: Any, hook: Any) -> Any:
            del hook
            hook_calls[layer] += 1
            if hook_calls[layer] != 1:
                raise RuntimeError(f"prompt anchor hook at layer {layer} fired more than once")
            captures[layer] = activation.detach()
            return activation

        return capture

    if before_forward is not None:
        before_forward("prompt_only_forward")
    hooks = [(f"blocks.{layer}.hook_out", hook_for(layer)) for layer in layers]
    with torch.inference_mode(), backend.model.hooks(fwd_hooks=hooks):
        backend.model(tokens)
    if set(captures) != set(layers):
        raise RuntimeError("prompt anchor capture did not observe every layer")
    residuals = torch.stack(
        [captures[layer][0, anchor_index].detach().cpu().float() for layer in layers]
    ).contiguous()
    if residuals.ndim != 2 or not bool(torch.isfinite(residuals).all().item()):
        raise RuntimeError("prompt anchor residual matrix is invalid")
    norms = residuals.double().norm(dim=1)
    if bool((norms <= 0).any().item()):
        raise RuntimeError("one or more prompt anchor residuals have zero norm")
    audit = {
        "sequence_length": sequence_length,
        "anchor_index": anchor_index,
        "layers": list(layers),
        "hook_call_counts": {str(layer): hook_calls[layer] for layer in layers},
        "anchor_residual_norms": norms.tolist(),
        "anchor_residual_float32_sha256": tensor_float32_sha256(residuals),
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return residuals, audit


def _capture_completion_objective(
    backend: Any,
    prompt: str,
    completion: str,
    *,
    layers: tuple[int, ...],
    anchor_index: int,
    before_forward: Callable[[str], None] | None,
    before_backward: Callable[[str], None] | None,
    metering_label: str,
) -> MultilayerAnchorObjectiveCapture:
    torch = backend.torch
    boundary = resolve_choice_boundary(backend, prompt)
    encoding = encode_authored_completion(backend, prompt, completion, boundary=boundary)
    if anchor_index >= encoding.prompt_length:
        raise ValueError("anchor index must lie inside the prompt prefix")
    captures: dict[int, Any] = {}
    hook_calls = {layer: 0 for layer in layers}
    first_layer = layers[0]

    def hook_for(layer: int) -> Any:
        def capture(activation: Any, hook: Any) -> Any:
            del hook
            hook_calls[layer] += 1
            if hook_calls[layer] != 1:
                raise RuntimeError(f"completion anchor hook at layer {layer} fired more than once")
            if layer == first_layer:
                activation = activation.detach().requires_grad_(True)
            captures[layer] = activation
            return activation

        return capture

    backend.model.zero_grad(set_to_none=True)
    parameter_gradients_allocated = False
    try:
        if before_forward is not None:
            before_forward(f"{metering_label}_forward")
        hooks = [(f"blocks.{layer}.hook_out", hook_for(layer)) for layer in layers]
        with torch.enable_grad(), backend.model.hooks(fwd_hooks=hooks):
            logits = backend.model(encoding.full_tokens)
            if set(captures) != set(layers):
                raise RuntimeError("completion anchor capture did not observe every layer")
            objective = authored_completion_mean_logprob(torch, logits, encoding)
            if before_backward is not None:
                before_backward(f"{metering_label}_backward")
            gradients = torch.autograd.grad(
                objective,
                tuple(captures[layer] for layer in layers),
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
            parameter_gradients_allocated = any(
                parameter.grad is not None for parameter in backend.model.parameters()
            )
            if parameter_gradients_allocated:
                raise RuntimeError("multi-layer capture allocated model parameter gradients")
    finally:
        backend.model.zero_grad(set_to_none=True)

    raw = torch.stack(
        [gradient[0, anchor_index].detach().cpu().float() for gradient in gradients]
    ).contiguous()
    residuals = torch.stack(
        [captures[layer][0, anchor_index].detach().cpu().float() for layer in layers]
    ).contiguous()
    if raw.shape != residuals.shape or raw.ndim != 2:
        raise RuntimeError("multi-layer anchor gradient/residual shapes differ")
    if not bool(torch.isfinite(raw).all().item()) or not bool(torch.isfinite(residuals).all().item()):
        raise RuntimeError("multi-layer anchor capture contains non-finite values")
    residual_norms = residuals.double().norm(dim=1)
    if bool((residual_norms <= 0).any().item()):
        raise RuntimeError("completion anchor residual contains a zero-norm layer")
    audit = {
        "schema_version": "sp_lense.multilayer_anchor_objective_capture.v1",
        "objective": "mean_authored_content_log_probability",
        "mean_log_probability": float(objective.detach().cpu().item()),
        "completion_text_sha256": text_sha256(completion),
        "content_token_ids_sha256": encoding.audit_record()["content_token_ids_sha256"],
        "content_token_count": encoding.content_token_count,
        "assistant_end_excluded_from_objective": True,
        "joint_chat_template_tokenization": True,
        "anchor_index": anchor_index,
        "gradient_position": "shared_pre_encoding_causal_anchor",
        "layers": list(layers),
        "hook_call_counts": {str(layer): hook_calls[layer] for layer in layers},
        "raw_gradients_float32_sha256": tensor_float32_sha256(raw),
        "anchor_residuals_float32_sha256": tensor_float32_sha256(residuals),
        "raw_gradient_norms": raw.double().norm(dim=1).tolist(),
        "anchor_residual_norms": residual_norms.tolist(),
        "model_parameter_gradients_allocated": parameter_gradients_allocated,
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return MultilayerAnchorObjectiveCapture(
        raw_gradients=raw,
        anchor_residuals=residuals,
        mean_log_probability=float(objective.detach().cpu().item()),
        audit=audit,
    )


def capture_multilayer_semantic_anchor_gradient(
    backend: Any,
    prompt: str,
    preserve_completion: str,
    comply_completion: str,
    *,
    layers: Sequence[int],
    anchor_index: int,
    causal_residual_tolerance: float = DEFAULT_CAUSAL_ANCHOR_RESIDUAL_RELATIVE_L2_TOLERANCE,
    capture_prompt_only_reference: bool = True,
    before_forward: Callable[[str], None] | None = None,
    before_backward: Callable[[str], None] | None = None,
) -> MultilayerSemanticAnchorCapture:
    """Capture preserve-minus-comply gradients at one shared internal token."""

    layer_tuple = _checked_layers(layers)
    index = _checked_anchor_index(anchor_index)
    if preserve_completion == comply_completion:
        raise ValueError("preserve and comply completions must differ")
    if (
        isinstance(causal_residual_tolerance, bool)
        or not isinstance(causal_residual_tolerance, (int, float))
        or not math.isfinite(float(causal_residual_tolerance))
        or float(causal_residual_tolerance) <= 0.0
    ):
        raise ValueError("causal_residual_tolerance must be finite and positive")

    if not isinstance(capture_prompt_only_reference, bool):
        raise TypeError("capture_prompt_only_reference must be boolean")
    prompt_residuals = None
    prompt_audit = None
    if capture_prompt_only_reference:
        prompt_residuals, prompt_audit = _capture_prompt_anchor_residuals(
            backend,
            prompt,
            layers=layer_tuple,
            anchor_index=index,
            before_forward=before_forward,
        )
    preserve = _capture_completion_objective(
        backend,
        prompt,
        preserve_completion,
        layers=layer_tuple,
        anchor_index=index,
        before_forward=before_forward,
        before_backward=before_backward,
        metering_label="preserve",
    )
    comply = _capture_completion_objective(
        backend,
        prompt,
        comply_completion,
        layers=layer_tuple,
        anchor_index=index,
        before_forward=before_forward,
        before_backward=before_backward,
        metering_label="comply",
    )
    reference = preserve.anchor_residuals if prompt_residuals is None else prompt_residuals
    reference_norms = reference.double().norm(dim=1)
    preserve_errors = (preserve.anchor_residuals.double() - reference.double()).norm(dim=1) / (
        reference_norms
    )
    comply_errors = (comply.anchor_residuals.double() - reference.double()).norm(dim=1) / (
        reference_norms
    )
    between_errors = (
        preserve.anchor_residuals.double() - comply.anchor_residuals.double()
    ).norm(dim=1) / reference_norms
    maximum_error = float(
        max(preserve_errors.max().item(), comply_errors.max().item(), between_errors.max().item())
    )
    if maximum_error > float(causal_residual_tolerance):
        raise RuntimeError("future completion tokens changed a causal anchor residual")
    semantic = (preserve.raw_gradients - comply.raw_gradients).float().contiguous()
    if not bool(backend.torch.isfinite(semantic).all().item()):
        raise RuntimeError("semantic anchor gradient contains a non-finite value")
    audit = {
        "schema_version": "sp_lense.multilayer_semantic_anchor_capture.v1",
        "objective": "preserve_minus_comply_mean_authored_completion_log_probability",
        "target_uses_answer_identifiers": False,
        "target_uses_answer_order": False,
        "prompt_text_sha256": text_sha256(prompt),
        "preserve_completion_text_sha256": text_sha256(preserve_completion),
        "comply_completion_text_sha256": text_sha256(comply_completion),
        "layers": list(layer_tuple),
        "anchor_index": index,
        "prompt": prompt_audit,
        "preserve": preserve.audit,
        "comply": comply.audit,
        "semantic_raw_gradients_float32_sha256": tensor_float32_sha256(semantic),
        "reference_anchor_residuals_float32_sha256": tensor_float32_sha256(reference),
        "prompt_only_reference_captured": capture_prompt_only_reference,
        "causal_anchor_residual_audit": {
            "reference_kind": (
                "prompt_only" if capture_prompt_only_reference else "preserve_completion_prefix"
            ),
            "preserve_vs_reference_per_layer": preserve_errors.tolist(),
            "comply_vs_reference_per_layer": comply_errors.tolist(),
            "preserve_vs_comply_per_layer": between_errors.tolist(),
            "maximum_relative_l2": maximum_error,
            "maximum_allowed_relative_l2": float(causal_residual_tolerance),
            "passes": True,
        },
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return MultilayerSemanticAnchorCapture(
        raw_semantic_gradients=semantic,
        reference_anchor_residuals=reference,
        preserve=preserve,
        comply=comply,
        audit=audit,
    )


def anchor_residual_scale_geometric_mean(torch: Any, residual_matrices: Sequence[Any]) -> Any:
    """Return equal-cell geometric-mean residual norms for each layer."""

    if not residual_matrices:
        raise ValueError("at least one residual matrix is required")
    checked = [matrix.detach().cpu().double().contiguous() for matrix in residual_matrices]
    shapes = {tuple(matrix.shape) for matrix in checked}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError("residual matrices must share one [layers, d_model] shape")
    norms = torch.stack([matrix.norm(dim=1) for matrix in checked])
    if not bool(torch.isfinite(norms).all().item()) or bool((norms <= 0).any().item()):
        raise ValueError("residual matrices must have finite positive per-layer norms")
    scales = torch.exp(torch.log(norms).mean(dim=0)).double().contiguous()
    if not bool(torch.isfinite(scales).all().item()) or bool((scales <= 0).any().item()):
        raise RuntimeError("geometric-mean residual scales are invalid")
    return scales


__all__ = [
    "DEFAULT_CAUSAL_ANCHOR_RESIDUAL_RELATIVE_L2_TOLERANCE",
    "MultilayerAnchorObjectiveCapture",
    "MultilayerSemanticAnchorCapture",
    "SharedAnchorEvidence",
    "anchor_residual_scale_geometric_mean",
    "capture_multilayer_semantic_anchor_gradient",
    "resolve_shared_anchor_evidence",
]
