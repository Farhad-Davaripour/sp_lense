"""One-pass model captures for Counterfactual KL-Extragradient Surgery.

This module has no model-loading, optimization, generation, or sign-selection
logic.  It provides two narrowly scoped captures at one pinned residual anchor:

* an unsteered A/B baseline margin gradient that retains the full vocabulary
  logits and pre-anchor residual in the same forward/backward pass; and
* a caller-signed, nonzero lookahead capture of
  ``KL(changed || unsteered_baseline)``.  The KL is formed with float64
  log-softmax operations, and only the detached post-edit anchor is
  differentiated.

The caller owns every prompt, layer, anchor, direction, scale, and branch sign.
All supplied identities are checked before or during the single model forward.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .comparison_runtime import choice_score_from_logits, resolve_choice_boundary
from .decision_margin_shield_finite import full_vocabulary_kl_float64
from .factorial_causal_anchor import (
    canonical_sha256,
    tensor_float32_sha256,
    text_sha256,
)

SCHEMA_VERSION = "sp_lense.counterfactual_kl_runtime.v1"
DEFAULT_MAXIMUM_REALIZED_RELATIVE_L2_ERROR = 1e-4


@dataclass(frozen=True, slots=True)
class CounterfactualKLBaselineCapture:
    """Hash-audited result of one unsteered A/B baseline capture."""

    layer: int
    anchor_index: int
    positive_token_id: int
    negative_token_id: int
    positive_minus_negative_log_odds: float
    unrestricted_predicted_token_id: int
    unrestricted_predicted_label: str
    unrestricted_semantic_choice: str
    pair_choice_label: str
    pair_semantic_choice: str
    answer_format_valid: bool
    raw_anchor_gradient: Any
    pre_anchor_residual: Any
    full_logits: Any
    audit: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CounterfactualKLLookaheadCapture:
    """Hash-audited result of one externally signed KL lookahead capture."""

    layer: int
    anchor_index: int
    branch_sign: int
    full_vocabulary_kl_changed_to_baseline: float
    raw_anchor_kl_gradient: Any
    shared_standardized_kl_gradient: Any
    pre_anchor_residual: Any
    post_anchor_residual: Any
    realized_signed_delta: Any
    full_logits: Any
    audit: Mapping[str, Any]

    @property
    def full_kl(self) -> float:
        """Concise alias for the exact changed-to-baseline full-vocabulary KL."""

        return self.full_vocabulary_kl_changed_to_baseline


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _audited(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = dict(record)
    value["audit_sha256"] = canonical_sha256(value)
    return _freeze(value)


def _checked_non_negative_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _checked_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be one lowercase SHA-256")
    return value


def _checked_positive_scalar(value: Any, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{field} must be finite and positive")
    return float(value)


def _checked_realization_tolerance(value: Any) -> float:
    return _checked_positive_scalar(value, field="maximum_realized_relative_l2_error")


def _checked_semantic(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value == "OTHER":
        raise ValueError(f"{field} must be a non-empty semantic other than OTHER")
    return value


def _checked_prompt(prompt: Any, expected_prompt_sha256: Any) -> tuple[str, str]:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    expected = _checked_sha256(expected_prompt_sha256, field="expected_prompt_sha256")
    observed = text_sha256(prompt)
    if observed != expected:
        raise RuntimeError("prompt text differs from the pinned hash")
    return prompt, observed


def _model_coordinate(backend: Any, *, layer: int) -> tuple[int, int]:
    model_cfg = getattr(backend.model, "cfg", None)
    configured_layers = getattr(model_cfg, "n_layers", None)
    configured_width = getattr(model_cfg, "d_model", None)
    if (
        isinstance(configured_layers, bool)
        or not isinstance(configured_layers, int)
        or configured_layers < 1
        or isinstance(configured_width, bool)
        or not isinstance(configured_width, int)
        or configured_width < 1
    ):
        raise RuntimeError("resident backend must expose positive cfg.n_layers and cfg.d_model")
    if layer >= configured_layers:
        raise ValueError("layer lies outside the resident model")
    return configured_layers, configured_width


def _resolve_pinned_context(
    backend: Any,
    prompt: str,
    *,
    anchor_index: int,
    expected_choice_boundary_evidence_sha256: str,
    expected_prompt_token_ids_sha256: str,
) -> tuple[Any, Any]:
    torch = backend.torch
    expected_boundary = _checked_sha256(
        expected_choice_boundary_evidence_sha256,
        field="expected_choice_boundary_evidence_sha256",
    )
    expected_tokens = _checked_sha256(
        expected_prompt_token_ids_sha256,
        field="expected_prompt_token_ids_sha256",
    )
    tokens = backend.encode(prompt)
    if getattr(tokens, "ndim", None) != 2 or int(tokens.shape[0]) != 1:
        raise ValueError("backend.encode must return one token row")
    if int(tokens.shape[1]) < 1:
        raise ValueError("backend.encode must return a non-empty prompt")
    if bool(torch.is_floating_point(tokens)) or bool(torch.is_complex(tokens)):
        raise TypeError("backend.encode must return integer token IDs")
    if anchor_index >= int(tokens.shape[1]):
        raise ValueError("anchor index lies outside the encoded prompt")
    boundary = resolve_choice_boundary(backend, prompt)
    if boundary.prompt_length != int(tokens.shape[1]):
        raise RuntimeError("choice-boundary evidence has the wrong prompt length")
    if boundary.evidence_sha256 != expected_boundary:
        raise RuntimeError("choice-boundary evidence differs from the pinned value")
    if boundary.prompt_prefix_token_ids_sha256 != expected_tokens:
        raise RuntimeError("prompt token IDs differ from the pinned value")
    return tokens, boundary


def _checked_float32_vector(
    torch: Any,
    value: Any,
    *,
    field: str,
    length: int | None = None,
) -> Any:
    if getattr(value, "ndim", None) != 1:
        raise ValueError(f"{field} must be one-dimensional")
    if getattr(value, "dtype", None) != torch.float32:
        raise ValueError(f"{field} must already be float32")
    result = value.detach().to(device="cpu").contiguous().clone()
    if length is not None and int(result.numel()) != length:
        raise ValueError(f"{field} has the wrong length")
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"{field} must contain only finite values")
    return result


def _checked_standardized_direction(torch: Any, value: Any, *, width: int) -> Any:
    if getattr(value, "ndim", None) != 1:
        raise ValueError("lookahead_standardized_direction must be one-dimensional")
    if getattr(value, "dtype", None) != torch.float64:
        raise ValueError("lookahead_standardized_direction must be canonical float64")
    result = value.detach().to(device="cpu").contiguous().clone()
    if int(result.numel()) != width:
        raise ValueError("lookahead standardized direction width differs from d_model")
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError("lookahead standardized direction must contain only finite values")
    result[result == 0.0] = 0.0
    return result


def _checked_model_output(torch: Any, output: Any, tokens: Any) -> Any:
    if getattr(output, "ndim", None) != 3:
        raise ValueError("model must return logits shaped [batch, sequence, vocab]")
    if int(output.shape[0]) != 1 or int(output.shape[1]) != int(tokens.shape[1]):
        raise ValueError("model logits do not match the encoded prompt dimensions")
    logits = output[0, -1].float()
    if logits.ndim != 1 or int(logits.numel()) < 2:
        raise ValueError("next-token logits must be one nontrivial vocabulary vector")
    if not bool(torch.isfinite(logits).all().item()):
        raise RuntimeError("next-token logits contain a non-finite value")
    return logits


def capture_counterfactual_kl_baseline(
    backend: Any,
    prompt: str,
    positive_label: str,
    negative_label: str,
    *,
    positive_semantic: str,
    negative_semantic: str,
    layer: int,
    anchor_index: int,
    expected_prompt_sha256: str,
    expected_choice_boundary_evidence_sha256: str,
    expected_prompt_token_ids_sha256: str,
    expected_pre_anchor_residual_float32_sha256: str | None = None,
) -> CounterfactualKLBaselineCapture:
    """Capture an unsteered A/B gradient, logits, and residual in one F+B pass."""

    selected_layer = _checked_non_negative_integer(layer, field="layer")
    selected_anchor = _checked_non_negative_integer(anchor_index, field="anchor_index")
    prompt, observed_prompt_hash = _checked_prompt(prompt, expected_prompt_sha256)
    if {positive_label, negative_label} != {"A", "B"}:
        raise ValueError("positive and negative labels must be exactly A and B")
    positive_meaning = _checked_semantic(positive_semantic, field="positive_semantic")
    negative_meaning = _checked_semantic(negative_semantic, field="negative_semantic")
    if positive_meaning == negative_meaning:
        raise ValueError("positive and negative semantics must differ")
    expected_pre_hash = (
        None
        if expected_pre_anchor_residual_float32_sha256 is None
        else _checked_sha256(
            expected_pre_anchor_residual_float32_sha256,
            field="expected_pre_anchor_residual_float32_sha256",
        )
    )
    _, width = _model_coordinate(backend, layer=selected_layer)
    tokens, boundary = _resolve_pinned_context(
        backend,
        prompt,
        anchor_index=selected_anchor,
        expected_choice_boundary_evidence_sha256=expected_choice_boundary_evidence_sha256,
        expected_prompt_token_ids_sha256=expected_prompt_token_ids_sha256,
    )
    positive_id = boundary.token_id(positive_label)
    negative_id = boundary.token_id(negative_label)
    if positive_id == negative_id:  # pragma: no cover - boundary invariant.
        raise RuntimeError("positive and negative labels resolved to one token")

    torch = backend.torch
    capture: dict[str, Any] = {}
    hook_calls = 0

    def detach_anchor(activation: Any, hook: Any) -> Any:
        nonlocal hook_calls
        del hook
        hook_calls += 1
        if hook_calls != 1:
            raise RuntimeError("baseline residual hook fired more than once")
        if activation.ndim != 3 or int(activation.shape[0]) != 1:
            raise ValueError("residual activation must have shape [1, sequence, d_model]")
        if int(activation.shape[1]) != int(tokens.shape[1]):
            raise RuntimeError("hook activation sequence differs from the encoded prompt")
        if int(activation.shape[2]) != width:
            raise RuntimeError("hook activation width differs from d_model")
        if selected_anchor >= int(activation.shape[1]):
            raise ValueError("anchor index lies outside the hooked residual")
        if not bool(torch.is_floating_point(activation)) or bool(torch.is_complex(activation)):
            raise TypeError("hooked residual must be real floating point")
        if not bool(torch.isfinite(activation).all().item()):
            raise RuntimeError("hooked residual contains a non-finite value")

        detached = activation.detach()
        pre_native = detached[0, selected_anchor].clone()
        pre = pre_native.float().detach().cpu().contiguous()
        if expected_pre_hash is not None and tensor_float32_sha256(pre) != expected_pre_hash:
            raise RuntimeError("pre-anchor residual differs from the pinned value")
        anchor_leaf = pre_native.detach().requires_grad_(True)
        reconstructed = torch.cat(
            (
                detached[:, :selected_anchor],
                anchor_leaf.reshape(1, 1, -1),
                detached[:, selected_anchor + 1 :],
            ),
            dim=1,
        )
        if reconstructed.shape != activation.shape:
            raise RuntimeError("reconstructed baseline residual has the wrong shape")
        maximum_delta = float(
            (reconstructed.detach().float() - detached.float()).abs().max().cpu().item()
        )
        if maximum_delta != 0.0:
            raise RuntimeError("zero-direction baseline reconstruction changed an activation")
        capture.update(
            {
                "anchor_leaf": anchor_leaf,
                "pre_anchor": pre,
                "maximum_reconstruction_delta": maximum_delta,
            }
        )
        return reconstructed

    backend.model.zero_grad(set_to_none=True)
    parameter_gradients_allocated = False
    try:
        hook_name = f"blocks.{selected_layer}.hook_out"
        with torch.enable_grad(), backend.model.hooks(fwd_hooks=[(hook_name, detach_anchor)]):
            output = backend.model(tokens)
            if hook_calls != 1 or "anchor_leaf" not in capture:
                raise RuntimeError("baseline residual hook did not fire exactly once")
            logits = _checked_model_output(torch, output, tokens)
            if int(logits.numel()) <= max(positive_id, negative_id):
                raise ValueError("next-token logits do not contain the verified A/B tokens")
            objective = logits[positive_id] - logits[negative_id]
            gradient = torch.autograd.grad(
                objective,
                capture["anchor_leaf"],
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
            parameter_gradients_allocated = any(
                parameter.grad is not None for parameter in backend.model.parameters()
            )
            if parameter_gradients_allocated:
                raise RuntimeError("baseline capture allocated model parameter gradients")
    finally:
        backend.model.zero_grad(set_to_none=True)

    raw_gradient = gradient.detach().cpu().float().contiguous().clone()
    pre_anchor = capture["pre_anchor"].detach().cpu().float().contiguous().clone()
    full_logits = logits.detach().cpu().float().contiguous().clone()
    if raw_gradient.shape != pre_anchor.shape or tuple(raw_gradient.shape) != (width,):
        raise RuntimeError("baseline anchor gradient or residual has the wrong shape")
    if not bool(torch.isfinite(raw_gradient).all().item()):
        raise RuntimeError("baseline anchor gradient contains a non-finite value")
    margin = float(objective.detach().cpu().item())
    score = choice_score_from_logits(
        torch,
        full_logits,
        positive_id,
        negative_id,
        preserve_label=positive_label,
        comply_label=negative_label,
        choice_boundary_evidence_sha256=boundary.evidence_sha256,
        choice_a_token_id=boundary.a_token_id,
        choice_b_token_id=boundary.b_token_id,
    )
    if score.preserve_log_odds != margin:
        raise RuntimeError("scored baseline log odds differ from the differentiated objective")
    unrestricted_label = score.predicted_label
    unrestricted_semantic = (
        positive_meaning
        if unrestricted_label == positive_label
        else negative_meaning
        if unrestricted_label == negative_label
        else "OTHER"
    )
    pair_semantic = (
        positive_meaning if score.pair_choice == positive_label else negative_meaning
    )
    predicted_token_id = int(full_logits.argmax().item())
    audit = _audited(
        {
            "schema_version": f"{SCHEMA_VERSION}.baseline",
            "capture_kind": "unsteered_zero_direction_A_B_margin_gradient_and_full_logits",
            "layer": selected_layer,
            "hook_name": f"blocks.{selected_layer}.hook_out",
            "hook_call_count": hook_calls,
            "anchor_index": selected_anchor,
            "prompt_sha256": observed_prompt_hash,
            "prompt_length": int(tokens.shape[1]),
            "prompt_token_ids_sha256": boundary.prompt_prefix_token_ids_sha256,
            "choice_boundary_evidence_sha256": boundary.evidence_sha256,
            "positive_label": positive_label,
            "negative_label": negative_label,
            "positive_semantic": positive_meaning,
            "negative_semantic": negative_meaning,
            "positive_token_id": positive_id,
            "negative_token_id": negative_id,
            "positive_minus_negative_log_odds": margin,
            "unrestricted_predicted_token_id": predicted_token_id,
            "unrestricted_predicted_label": unrestricted_label,
            "unrestricted_semantic_choice": unrestricted_semantic,
            "pair_choice_label": score.pair_choice,
            "pair_semantic_choice": pair_semantic,
            "answer_format_valid": unrestricted_label != "OTHER",
            "pre_anchor_residual_float32_sha256": tensor_float32_sha256(pre_anchor),
            "pre_anchor_residual_hash_precommitted": expected_pre_hash is not None,
            "raw_anchor_gradient_float32_sha256": tensor_float32_sha256(raw_gradient),
            "raw_anchor_gradient_l2": float(raw_gradient.double().norm().item()),
            "full_logits_float32_sha256": tensor_float32_sha256(full_logits),
            "zero_direction": True,
            "maximum_abs_activation_reconstruction_delta": capture[
                "maximum_reconstruction_delta"
            ],
            "model_forward_evaluations": 1,
            "model_backward_evaluations": 1,
            "model_parameter_gradients_allocated": parameter_gradients_allocated,
            "detach_scope": "selected_layer_residual_with_only_anchor_as_leaf",
        }
    )
    return CounterfactualKLBaselineCapture(
        layer=selected_layer,
        anchor_index=selected_anchor,
        positive_token_id=positive_id,
        negative_token_id=negative_id,
        positive_minus_negative_log_odds=margin,
        unrestricted_predicted_token_id=predicted_token_id,
        unrestricted_predicted_label=unrestricted_label,
        unrestricted_semantic_choice=unrestricted_semantic,
        pair_choice_label=score.pair_choice,
        pair_semantic_choice=pair_semantic,
        answer_format_valid=unrestricted_label != "OTHER",
        raw_anchor_gradient=raw_gradient,
        pre_anchor_residual=pre_anchor,
        full_logits=full_logits,
        audit=audit,
    )


def capture_counterfactual_kl_lookahead(
    backend: Any,
    prompt: str,
    *,
    layer: int,
    anchor_index: int,
    branch_sign: int,
    lookahead_standardized_direction: Any,
    physical_residual_scale: float,
    signed_delta: Any,
    baseline_full_logits: Any,
    expected_prompt_sha256: str,
    expected_choice_boundary_evidence_sha256: str,
    expected_prompt_token_ids_sha256: str,
    expected_pre_anchor_residual_float32_sha256: str,
    expected_lookahead_standardized_direction_sha256: str,
    expected_signed_delta_float32_sha256: str,
    expected_baseline_full_logits_float32_sha256: str,
    maximum_realized_relative_l2_error: float = DEFAULT_MAXIMUM_REALIZED_RELATIVE_L2_ERROR,
) -> CounterfactualKLLookaheadCapture:
    """Capture one signed nonzero lookahead KL and its shared-direction gradient."""

    selected_layer = _checked_non_negative_integer(layer, field="layer")
    selected_anchor = _checked_non_negative_integer(anchor_index, field="anchor_index")
    if isinstance(branch_sign, bool) or not isinstance(branch_sign, int) or branch_sign not in {-1, 1}:
        raise ValueError("branch_sign must be exactly -1 or +1")
    prompt, observed_prompt_hash = _checked_prompt(prompt, expected_prompt_sha256)
    expected_pre_hash = _checked_sha256(
        expected_pre_anchor_residual_float32_sha256,
        field="expected_pre_anchor_residual_float32_sha256",
    )
    expected_direction_hash = _checked_sha256(
        expected_lookahead_standardized_direction_sha256,
        field="expected_lookahead_standardized_direction_sha256",
    )
    expected_delta_hash = _checked_sha256(
        expected_signed_delta_float32_sha256,
        field="expected_signed_delta_float32_sha256",
    )
    expected_baseline_hash = _checked_sha256(
        expected_baseline_full_logits_float32_sha256,
        field="expected_baseline_full_logits_float32_sha256",
    )
    realization_tolerance = _checked_realization_tolerance(
        maximum_realized_relative_l2_error
    )
    residual_scale = _checked_positive_scalar(
        physical_residual_scale, field="physical_residual_scale"
    )
    _, width = _model_coordinate(backend, layer=selected_layer)
    torch = backend.torch
    direction = _checked_standardized_direction(
        torch, lookahead_standardized_direction, width=width
    )
    observed_direction_hash = canonical_sha256(direction.tolist())
    if observed_direction_hash != expected_direction_hash:
        raise RuntimeError("lookahead standardized direction hash mismatch")
    requested_delta = _checked_float32_vector(
        torch, signed_delta, field="signed_delta", length=width
    )
    requested_l2 = float(requested_delta.double().norm().item())
    if requested_l2 <= 0.0:
        raise ValueError("signed_delta must have non-zero L2 norm")
    observed_delta_hash = tensor_float32_sha256(requested_delta)
    if observed_delta_hash != expected_delta_hash:
        raise RuntimeError("signed delta hash mismatch; possible external sign error")
    unsigned_delta = (direction * residual_scale).float().contiguous()
    reconstructed_delta = (
        unsigned_delta if branch_sign == 1 else -unsigned_delta
    ).contiguous()
    reconstructed_delta_hash = tensor_float32_sha256(reconstructed_delta)
    if reconstructed_delta_hash != observed_delta_hash:
        raise RuntimeError(
            "signed delta is inconsistent with branch sign, lookahead direction, and scale"
        )
    baseline_logits = _checked_float32_vector(
        torch, baseline_full_logits, field="baseline_full_logits"
    )
    observed_baseline_hash = tensor_float32_sha256(baseline_logits)
    if observed_baseline_hash != expected_baseline_hash:
        raise RuntimeError("unsteered baseline logits differ from the pinned hash")

    tokens, boundary = _resolve_pinned_context(
        backend,
        prompt,
        anchor_index=selected_anchor,
        expected_choice_boundary_evidence_sha256=expected_choice_boundary_evidence_sha256,
        expected_prompt_token_ids_sha256=expected_prompt_token_ids_sha256,
    )
    capture: dict[str, Any] = {}
    hook_calls = 0

    def inject_and_detach(activation: Any, hook: Any) -> Any:
        nonlocal hook_calls
        del hook
        hook_calls += 1
        if hook_calls != 1:
            raise RuntimeError("KL lookahead residual hook fired more than once")
        if activation.ndim != 3 or int(activation.shape[0]) != 1:
            raise ValueError("residual activation must have shape [1, sequence, d_model]")
        if int(activation.shape[1]) != int(tokens.shape[1]):
            raise RuntimeError("hook activation sequence differs from the encoded prompt")
        if int(activation.shape[2]) != width:
            raise RuntimeError("hook activation width differs from d_model")
        if selected_anchor >= int(activation.shape[1]):
            raise ValueError("anchor index lies outside the hooked residual")
        if not bool(torch.is_floating_point(activation)) or bool(torch.is_complex(activation)):
            raise TypeError("hooked residual must be real floating point")
        if not bool(torch.isfinite(activation).all().item()):
            raise RuntimeError("hooked residual contains a non-finite value")

        detached = activation.detach()
        pre = detached[0, selected_anchor].float().clone()
        pre_cpu = pre.detach().cpu().float().contiguous()
        if tensor_float32_sha256(pre_cpu) != expected_pre_hash:
            raise RuntimeError("pre-anchor residual differs from the pinned value")
        requested_on_device = requested_delta.to(device=pre.device)
        requested_post = pre + requested_on_device
        post_anchor = requested_post.to(dtype=activation.dtype).detach().requires_grad_(True)
        changed = torch.cat(
            (
                detached[:, :selected_anchor],
                post_anchor.reshape(1, 1, -1),
                detached[:, selected_anchor + 1 :],
            ),
            dim=1,
        )
        if changed.shape != activation.shape:
            raise RuntimeError("reconstructed residual shape differs after lookahead injection")
        post = post_anchor.detach().float()
        realized = post - pre
        realization_error = realized - requested_on_device
        error_l2 = float(realization_error.double().norm().detach().cpu().item())
        relative_error = error_l2 / requested_l2
        untouched = changed.detach().float() - detached.float()
        untouched[0, selected_anchor] = 0.0
        untouched_max_abs = float(untouched.abs().max().detach().cpu().item())
        signed_dot = float(
            torch.dot(realized.double(), requested_on_device.double()).detach().cpu().item()
        )
        if untouched_max_abs != 0.0:
            raise RuntimeError("lookahead edit changed a non-anchor residual")
        if signed_dot <= 0.0:
            raise RuntimeError("realized delta does not preserve the externally supplied sign")
        if relative_error > realization_tolerance:
            raise RuntimeError("realized anchor delta differs from the physical request")
        capture.update(
            {
                "post_anchor_leaf": post_anchor,
                "pre_anchor": pre_cpu,
                "post_anchor": post.detach().cpu().float().contiguous(),
                "requested_post_anchor": requested_post.detach().cpu().float().contiguous(),
                "realized_delta": realized.detach().cpu().float().contiguous(),
                "requested_minus_realized_l2": error_l2,
                "requested_minus_realized_relative_l2": relative_error,
                "requested_realized_dot": signed_dot,
                "untouched_positions_max_abs_delta": untouched_max_abs,
            }
        )
        return changed

    backend.model.zero_grad(set_to_none=True)
    parameter_gradients_allocated = False
    try:
        hook_name = f"blocks.{selected_layer}.hook_out"
        with torch.enable_grad(), backend.model.hooks(fwd_hooks=[(hook_name, inject_and_detach)]):
            output = backend.model(tokens)
            if hook_calls != 1 or "post_anchor_leaf" not in capture:
                raise RuntimeError("KL lookahead residual hook did not fire exactly once")
            logits = _checked_model_output(torch, output, tokens)
            if logits.shape != baseline_logits.shape:
                raise ValueError("changed and unsteered baseline vocabulary widths differ")
            changed_log_probs = torch.log_softmax(logits.double(), dim=-1)
            baseline_log_probs = torch.log_softmax(
                baseline_logits.to(device=logits.device).double(), dim=-1
            )
            changed_probs = changed_log_probs.exp()
            objective = (
                changed_probs * (changed_log_probs - baseline_log_probs)
            ).sum()
            gradient = torch.autograd.grad(
                objective,
                capture["post_anchor_leaf"],
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
            parameter_gradients_allocated = any(
                parameter.grad is not None for parameter in backend.model.parameters()
            )
            if parameter_gradients_allocated:
                raise RuntimeError("KL lookahead capture allocated model parameter gradients")
    finally:
        backend.model.zero_grad(set_to_none=True)

    raw_gradient = gradient.detach().cpu().float().contiguous().clone()
    shared_gradient = (branch_sign * residual_scale * raw_gradient.double()).contiguous()
    pre_anchor = capture["pre_anchor"].detach().cpu().float().contiguous().clone()
    post_anchor = capture["post_anchor"].detach().cpu().float().contiguous().clone()
    realized_delta = capture["realized_delta"].detach().cpu().float().contiguous().clone()
    full_logits = logits.detach().cpu().float().contiguous().clone()
    if tuple(raw_gradient.shape) != (width,) or shared_gradient.shape != raw_gradient.shape:
        raise RuntimeError("KL anchor or shared gradient has the wrong shape")
    if not bool(torch.isfinite(raw_gradient).all().item()) or not bool(
        torch.isfinite(shared_gradient).all().item()
    ):
        raise RuntimeError("KL gradient contains a non-finite value")
    objective_value = float(objective.detach().cpu().item())
    independent_kl = full_vocabulary_kl_float64(torch, baseline_logits, full_logits)
    if abs(objective_value - independent_kl) > 1e-12:
        raise RuntimeError("differentiated float64 KL differs from independent recomputation")
    if independent_kl <= 0.0:
        raise RuntimeError("nonzero lookahead did not produce a positive full-vocabulary KL")

    audit = _audited(
        {
            "schema_version": f"{SCHEMA_VERSION}.lookahead",
            "capture_kind": "signed_nonzero_full_vocabulary_KL_gradient",
            "kl_direction": "KL(changed||unsteered_baseline)",
            "kl_softmax_dtype": "float64",
            "layer": selected_layer,
            "hook_name": f"blocks.{selected_layer}.hook_out",
            "hook_call_count": hook_calls,
            "anchor_index": selected_anchor,
            "prompt_sha256": observed_prompt_hash,
            "prompt_length": int(tokens.shape[1]),
            "prompt_token_ids_sha256": boundary.prompt_prefix_token_ids_sha256,
            "choice_boundary_evidence_sha256": boundary.evidence_sha256,
            "external_branch_sign": branch_sign,
            "runtime_selected_or_changed_sign": False,
            "physical_residual_scale": residual_scale,
            "lookahead_standardized_direction_sha256": observed_direction_hash,
            "requested_signed_delta_float32_sha256": observed_delta_hash,
            "reconstructed_signed_delta_float32_sha256": reconstructed_delta_hash,
            "signed_delta_matches_branch_direction_and_scale": True,
            "baseline_full_logits_float32_sha256": observed_baseline_hash,
            "changed_full_logits_float32_sha256": tensor_float32_sha256(full_logits),
            "full_vocabulary_kl_changed_to_baseline": independent_kl,
            "pre_anchor_residual_float32_sha256": tensor_float32_sha256(pre_anchor),
            "requested_post_anchor_residual_float32_sha256": tensor_float32_sha256(
                capture["requested_post_anchor"]
            ),
            "post_anchor_residual_float32_sha256": tensor_float32_sha256(post_anchor),
            "realized_signed_delta_float32_sha256": tensor_float32_sha256(realized_delta),
            "requested_signed_delta_l2": requested_l2,
            "requested_minus_realized_l2": capture["requested_minus_realized_l2"],
            "requested_minus_realized_relative_l2": capture[
                "requested_minus_realized_relative_l2"
            ],
            "maximum_allowed_realized_relative_l2": realization_tolerance,
            "requested_realized_dot": capture["requested_realized_dot"],
            "untouched_positions_max_abs_delta": capture[
                "untouched_positions_max_abs_delta"
            ],
            "raw_anchor_kl_gradient_float32_sha256": tensor_float32_sha256(raw_gradient),
            "shared_standardized_kl_gradient_sha256": canonical_sha256(
                shared_gradient.tolist()
            ),
            "shared_gradient_chain_rule": "branch_sign*physical_residual_scale*raw_anchor_gradient",
            "model_forward_evaluations": 1,
            "model_backward_evaluations": 1,
            "model_parameter_gradients_allocated": parameter_gradients_allocated,
            "detach_scope": "selected_layer_residual_with_only_post_edit_anchor_as_leaf",
        }
    )
    return CounterfactualKLLookaheadCapture(
        layer=selected_layer,
        anchor_index=selected_anchor,
        branch_sign=branch_sign,
        full_vocabulary_kl_changed_to_baseline=independent_kl,
        raw_anchor_kl_gradient=raw_gradient,
        shared_standardized_kl_gradient=shared_gradient,
        pre_anchor_residual=pre_anchor,
        post_anchor_residual=post_anchor,
        realized_signed_delta=realized_delta,
        full_logits=full_logits,
        audit=audit,
    )


__all__ = [
    "DEFAULT_MAXIMUM_REALIZED_RELATIVE_L2_ERROR",
    "SCHEMA_VERSION",
    "CounterfactualKLBaselineCapture",
    "CounterfactualKLLookaheadCapture",
    "capture_counterfactual_kl_baseline",
    "capture_counterfactual_kl_lookahead",
]
