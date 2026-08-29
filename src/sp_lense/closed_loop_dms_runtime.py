"""One-pass model runtime for Closed-Loop Decision-Margin Shielding.

The caller owns the controller state and the sign of the physical edit.  This
module only applies the supplied signed float32 delta at one declared causal
anchor, measures the resulting A/B boundary, and returns the local raw
gradient.  It deliberately contains no solver, sign selection, generation, or
model-loading code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .comparison_runtime import choice_score_from_logits, resolve_choice_boundary
from .factorial_causal_anchor import canonical_sha256, tensor_float32_sha256

DEFAULT_MAXIMUM_REALIZED_RELATIVE_L2_ERROR = 1e-4


@dataclass(frozen=True)
class ClosedLoopDMSCapture:
    """Audited result of one externally signed CL-DMS endpoint capture."""

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
    post_anchor_residual: Any
    realized_signed_delta: Any
    full_logits: Any | None
    audit: dict[str, Any]


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


def _checked_semantic(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value == "OTHER":
        raise ValueError(f"{field} must be a non-empty semantic other than OTHER")
    return value


def _checked_realization_tolerance(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError("maximum realization error must be finite and positive")
    return float(value)


def _checked_positive_scalar(value: Any, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{field} must be finite and positive")
    return float(value)


def capture_closed_loop_dms_step(
    backend: Any,
    prompt: str,
    positive_label: str,
    negative_label: str,
    *,
    positive_semantic: str,
    negative_semantic: str,
    layer: int,
    anchor_index: int,
    branch_sign: int,
    cumulative_standardized_direction: Any,
    physical_residual_scale: float,
    signed_delta: Any,
    expected_signed_delta_float32_sha256: str,
    expected_cumulative_standardized_direction_sha256: str,
    expected_choice_boundary_evidence_sha256: str,
    expected_prompt_token_ids_sha256: str,
    expected_pre_anchor_residual_float32_sha256: str,
    maximum_realized_relative_l2_error: float = (DEFAULT_MAXIMUM_REALIZED_RELATIVE_L2_ERROR),
    return_full_logits: bool = False,
) -> ClosedLoopDMSCapture:
    """Apply one caller-signed physical edit and capture its local A/B gradient.

    ``signed_delta`` is applied verbatim.  To evaluate both controller paths,
    the caller invokes this function once with ``+D`` and once with ``-D`` and
    supplies the corresponding expected hash and declared ``branch_sign`` each
    time.  The runtime validates and records that declaration but never chooses
    or infers the sign.  It independently reconstructs
    ``branch_sign * physical_residual_scale * cumulative_standardized_direction``
    and requires bitwise equality with the supplied float32 delta before the
    forward pass.

    The residual stream is detached at exactly the selected hook.  Only the
    intervened anchor vector is made a gradient leaf, so
    :func:`torch.autograd.grad` cannot allocate model-parameter gradients.
    The required expected hashes bind the call to previously pinned prompt,
    boundary, and anchor evidence.
    """

    selected_layer = _checked_non_negative_integer(layer, field="layer")
    selected_anchor = _checked_non_negative_integer(anchor_index, field="anchor_index")
    if (
        isinstance(branch_sign, bool)
        or not isinstance(branch_sign, int)
        or branch_sign
        not in {
            -1,
            1,
        }
    ):
        raise ValueError("branch_sign must be exactly -1 or +1")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    if {positive_label, negative_label} != {"A", "B"}:
        raise ValueError("positive and negative labels must be exactly A and B")
    positive_meaning = _checked_semantic(positive_semantic, field="positive_semantic")
    negative_meaning = _checked_semantic(negative_semantic, field="negative_semantic")
    if positive_meaning == negative_meaning:
        raise ValueError("positive and negative semantics must differ")
    if not isinstance(return_full_logits, bool):
        raise TypeError("return_full_logits must be a boolean")

    expected_delta_hash = _checked_sha256(
        expected_signed_delta_float32_sha256,
        field="expected_signed_delta_float32_sha256",
    )
    cumulative_direction_hash = _checked_sha256(
        expected_cumulative_standardized_direction_sha256,
        field="expected_cumulative_standardized_direction_sha256",
    )
    expected_boundary_hash = _checked_sha256(
        expected_choice_boundary_evidence_sha256,
        field="expected_choice_boundary_evidence_sha256",
    )
    expected_prompt_hash = _checked_sha256(
        expected_prompt_token_ids_sha256,
        field="expected_prompt_token_ids_sha256",
    )
    expected_pre_hash = _checked_sha256(
        expected_pre_anchor_residual_float32_sha256,
        field="expected_pre_anchor_residual_float32_sha256",
    )
    realization_tolerance = _checked_realization_tolerance(maximum_realized_relative_l2_error)
    residual_scale = _checked_positive_scalar(
        physical_residual_scale, field="physical_residual_scale"
    )

    torch = backend.torch
    if getattr(cumulative_standardized_direction, "ndim", None) != 1:
        raise ValueError("cumulative_standardized_direction must be one-dimensional")
    if getattr(cumulative_standardized_direction, "dtype", None) != torch.float64:
        raise ValueError("cumulative_standardized_direction must be canonical float64")
    standardized_direction = (
        cumulative_standardized_direction.detach().to(device="cpu").contiguous().clone()
    )
    if not bool(torch.isfinite(standardized_direction).all().item()):
        raise ValueError("cumulative_standardized_direction must contain only finite values")
    standardized_direction[standardized_direction == 0.0] = 0.0
    observed_cumulative_hash = canonical_sha256(standardized_direction.tolist())
    if observed_cumulative_hash != cumulative_direction_hash:
        raise RuntimeError("cumulative standardized direction hash mismatch")
    if getattr(signed_delta, "ndim", None) != 1:
        raise ValueError("signed_delta must be a one-dimensional tensor")
    if getattr(signed_delta, "dtype", None) != torch.float32:
        raise ValueError("signed_delta must already be physical float32")
    requested_delta = signed_delta.detach().to(device="cpu").contiguous()
    if not bool(torch.isfinite(requested_delta).all().item()):
        raise ValueError("signed_delta must contain only finite values")
    requested_l2 = float(requested_delta.double().norm().item())
    if requested_l2 <= 0.0:
        raise ValueError("signed_delta must have non-zero L2 norm")
    observed_delta_hash = tensor_float32_sha256(requested_delta)
    if observed_delta_hash != expected_delta_hash:
        raise RuntimeError("signed delta hash mismatch; possible external sign error")
    unsigned_physical_delta = (standardized_direction * residual_scale).float().contiguous()
    reconstructed_signed_delta = (
        unsigned_physical_delta if branch_sign == 1 else -unsigned_physical_delta
    ).contiguous()
    reconstructed_delta_hash = tensor_float32_sha256(reconstructed_signed_delta)
    if observed_delta_hash != reconstructed_delta_hash:
        raise RuntimeError(
            "signed delta is inconsistent with branch sign, cumulative direction, and scale"
        )

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
    if selected_layer >= configured_layers:
        raise ValueError("layer lies outside the resident model")
    if int(requested_delta.numel()) != configured_width:
        raise ValueError("signed delta width differs from the resident d_model")
    if int(standardized_direction.numel()) != configured_width:
        raise ValueError("cumulative standardized direction width differs from d_model")

    tokens = backend.encode(prompt)
    if getattr(tokens, "ndim", None) != 2 or int(tokens.shape[0]) != 1 or int(tokens.shape[1]) < 1:
        raise ValueError("backend.encode must return one non-empty token row")
    if bool(torch.is_floating_point(tokens)) or bool(torch.is_complex(tokens)):
        raise TypeError("backend.encode must return integer token IDs")
    if selected_anchor >= int(tokens.shape[1]):
        raise ValueError("anchor index lies outside the encoded prompt")
    boundary = resolve_choice_boundary(backend, prompt)
    if boundary.prompt_length != int(tokens.shape[1]):
        raise RuntimeError("choice-boundary evidence has the wrong prompt length")
    if boundary.evidence_sha256 != expected_boundary_hash:
        raise RuntimeError("choice-boundary evidence differs from the pinned value")
    if boundary.prompt_prefix_token_ids_sha256 != expected_prompt_hash:
        raise RuntimeError("prompt token IDs differ from the pinned value")

    positive_id = boundary.token_id(positive_label)
    negative_id = boundary.token_id(negative_label)
    if positive_id == negative_id:  # pragma: no cover - boundary resolver invariant
        raise RuntimeError("positive and negative labels resolved to one token")

    capture: dict[str, Any] = {}
    hook_calls = 0

    def inject_and_detach(activation: Any, hook: Any) -> Any:
        nonlocal hook_calls
        del hook
        hook_calls += 1
        if hook_calls != 1:
            raise RuntimeError("closed-loop residual hook fired more than once")
        if activation.ndim != 3 or int(activation.shape[0]) != 1:
            raise ValueError("residual activation must have shape [1, sequence, d_model]")
        if int(activation.shape[1]) != int(tokens.shape[1]):
            raise RuntimeError("hook activation sequence differs from the encoded prompt")
        if selected_anchor >= int(activation.shape[1]):
            raise ValueError("anchor index lies outside the hooked residual")
        if int(activation.shape[2]) != int(requested_delta.numel()):
            raise ValueError("signed delta width differs from the residual width")
        if not bool(torch.is_floating_point(activation)):
            raise TypeError("hooked residual must be floating point")
        if not bool(torch.isfinite(activation).all().item()):
            raise RuntimeError("hooked residual contains a non-finite value")

        # This is the only graph cut.  The non-anchor residuals remain constant,
        # while the physically realized post-edit anchor becomes the sole leaf.
        detached = activation.detach()
        pre = detached[0, selected_anchor].float().clone()
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
            raise RuntimeError("reconstructed residual shape differs after anchor injection")

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
            raise RuntimeError("closed-loop edit changed a non-anchor residual")
        if signed_dot <= 0.0:
            raise RuntimeError("realized delta does not preserve the externally supplied sign")
        if relative_error > realization_tolerance:
            raise RuntimeError("realized anchor delta differs from the physical request")
        pre_cpu = pre.detach().cpu().float().contiguous()
        post_cpu = post.detach().cpu().float().contiguous()
        if tensor_float32_sha256(pre_cpu) != expected_pre_hash:
            raise RuntimeError("pre-anchor residual differs from the pinned value")

        capture.update(
            {
                "post_anchor_leaf": post_anchor,
                "pre_anchor": pre_cpu,
                "requested_post_anchor": requested_post.detach().cpu().float().contiguous(),
                "post_anchor": post_cpu,
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
            model_output = backend.model(tokens)
            if hook_calls != 1 or "post_anchor_leaf" not in capture:
                raise RuntimeError("closed-loop residual hook did not fire exactly once")
            if getattr(model_output, "ndim", None) != 3:
                raise ValueError("model must return logits shaped [batch, sequence, vocab]")
            if int(model_output.shape[0]) != 1 or int(model_output.shape[1]) != int(
                tokens.shape[1]
            ):
                raise ValueError("model logits do not match the encoded prompt dimensions")
            logits = model_output[0, -1].float()
            if logits.ndim != 1 or int(logits.numel()) <= max(positive_id, negative_id):
                raise ValueError("next-token logits do not contain the verified A/B tokens")
            if not bool(torch.isfinite(logits).all().item()):
                raise RuntimeError("next-token logits contain a non-finite value")
            objective = logits[positive_id] - logits[negative_id]
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
                raise RuntimeError("closed-loop capture allocated model parameter gradients")
    finally:
        backend.model.zero_grad(set_to_none=True)

    gradient_cpu = gradient.detach().cpu().float().contiguous()
    logits_cpu = logits.detach().cpu().float().contiguous()
    if gradient_cpu.ndim != 1 or gradient_cpu.shape != requested_delta.shape:
        raise RuntimeError("raw anchor gradient has the wrong shape")
    if not bool(torch.isfinite(gradient_cpu).all().item()):
        raise RuntimeError("raw anchor gradient contains a non-finite value")

    score = choice_score_from_logits(
        torch,
        logits_cpu,
        positive_id,
        negative_id,
        preserve_label=positive_label,
        comply_label=negative_label,
        choice_boundary_evidence_sha256=boundary.evidence_sha256,
        choice_a_token_id=boundary.a_token_id,
        choice_b_token_id=boundary.b_token_id,
    )
    objective_value = float(objective.detach().cpu().item())
    if score.preserve_log_odds != objective_value:
        raise RuntimeError("scored log odds differ from the differentiated objective")
    unrestricted_label = score.predicted_label
    unrestricted_semantic = (
        positive_meaning
        if unrestricted_label == positive_label
        else negative_meaning
        if unrestricted_label == negative_label
        else "OTHER"
    )
    pair_semantic = positive_meaning if score.pair_choice == positive_label else negative_meaning
    predicted_token_id = int(logits_cpu.argmax().item())

    audit = {
        "schema_version": "sp_lense.closed_loop_dms_runtime_capture.v1",
        "layer": selected_layer,
        "hook_name": f"blocks.{selected_layer}.hook_out",
        "hook_call_count": hook_calls,
        "anchor_index": selected_anchor,
        "prompt_length": int(tokens.shape[1]),
        "prompt_token_ids_sha256": boundary.prompt_prefix_token_ids_sha256,
        "choice_boundary_evidence_sha256": boundary.evidence_sha256,
        "positive_label": positive_label,
        "negative_label": negative_label,
        "positive_semantic": positive_meaning,
        "negative_semantic": negative_meaning,
        "positive_token_id": positive_id,
        "negative_token_id": negative_id,
        "positive_minus_negative_log_odds": objective_value,
        "unrestricted_predicted_token_id": predicted_token_id,
        "unrestricted_predicted_label": unrestricted_label,
        "unrestricted_semantic_choice": unrestricted_semantic,
        "pair_choice_label": score.pair_choice,
        "pair_semantic_choice": pair_semantic,
        "answer_format_valid": unrestricted_label != "OTHER",
        "requested_signed_delta_float32_sha256": observed_delta_hash,
        "external_branch_sign": branch_sign,
        "expected_cumulative_standardized_direction_sha256": cumulative_direction_hash,
        "observed_cumulative_standardized_direction_sha256": observed_cumulative_hash,
        "physical_residual_scale": residual_scale,
        "reconstructed_signed_delta_float32_sha256": reconstructed_delta_hash,
        "signed_delta_matches_declared_branch_direction_and_scale": True,
        "signed_delta_applied_verbatim": True,
        "runtime_selected_or_changed_sign": False,
        "requested_signed_delta_l2": requested_l2,
        "pre_anchor_residual_float32_sha256": tensor_float32_sha256(capture["pre_anchor"]),
        "requested_post_anchor_residual_float32_sha256": tensor_float32_sha256(
            capture["requested_post_anchor"]
        ),
        "post_anchor_residual_float32_sha256": tensor_float32_sha256(capture["post_anchor"]),
        "realized_signed_delta_float32_sha256": tensor_float32_sha256(capture["realized_delta"]),
        "realized_signed_delta_l2": float(capture["realized_delta"].double().norm().item()),
        "requested_minus_realized_l2": capture["requested_minus_realized_l2"],
        "requested_minus_realized_relative_l2": capture["requested_minus_realized_relative_l2"],
        "maximum_allowed_realized_relative_l2": realization_tolerance,
        "requested_realized_dot": capture["requested_realized_dot"],
        "realized_delta_aligned_with_requested_signed_delta": True,
        "untouched_positions_max_abs_delta": capture["untouched_positions_max_abs_delta"],
        "raw_anchor_gradient_float32_sha256": tensor_float32_sha256(gradient_cpu),
        "raw_anchor_gradient_l2": float(gradient_cpu.double().norm().item()),
        "full_logits_float32_sha256": tensor_float32_sha256(logits_cpu),
        "full_logits_returned": return_full_logits,
        "model_parameter_gradients_allocated": parameter_gradients_allocated,
        "detach_scope": "selected_layer_residual_with_only_intervened_anchor_as_leaf",
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return ClosedLoopDMSCapture(
        layer=selected_layer,
        anchor_index=selected_anchor,
        positive_token_id=positive_id,
        negative_token_id=negative_id,
        positive_minus_negative_log_odds=objective_value,
        unrestricted_predicted_token_id=predicted_token_id,
        unrestricted_predicted_label=unrestricted_label,
        unrestricted_semantic_choice=unrestricted_semantic,
        pair_choice_label=score.pair_choice,
        pair_semantic_choice=pair_semantic,
        answer_format_valid=unrestricted_label != "OTHER",
        raw_anchor_gradient=gradient_cpu,
        pre_anchor_residual=capture["pre_anchor"],
        post_anchor_residual=capture["post_anchor"],
        realized_signed_delta=capture["realized_delta"],
        full_logits=logits_cpu if return_full_logits else None,
        audit=audit,
    )


__all__ = [
    "DEFAULT_MAXIMUM_REALIZED_RELATIVE_L2_ERROR",
    "ClosedLoopDMSCapture",
    "capture_closed_loop_dms_step",
]
