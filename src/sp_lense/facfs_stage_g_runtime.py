"""Capture-only runtime and locked analysis for the FACFS Stage-G screen."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from .facfs_stage_g import canonical_sha256, template_ids
from .gradient_specificity_v3 import tensor_float32_sha256

WIDTH = 1024
LAYER = 10
HOOK_NAME = "blocks.10.hook_out"
CAUSAL_RESIDUAL_TOLERANCE = 1e-5
WALSH_COMPONENTS = ("even", "R", "O", "M", "RO", "RM", "OM", "ROM")


@contextmanager
def model_parameters_disabled(backend: Any) -> Any:
    parameters = tuple(backend.model.parameters())
    original = tuple(bool(parameter.requires_grad) for parameter in parameters)
    backend.model.zero_grad(set_to_none=True)
    for parameter in parameters:
        parameter.requires_grad_(False)
    if any(bool(parameter.requires_grad) for parameter in parameters):
        raise RuntimeError("could not disable every model parameter gradient")
    try:
        yield {
            "parameter_count": len(parameters),
            "requires_grad_true_count_before": sum(original),
        }
        if any(parameter.grad is not None for parameter in parameters):
            raise RuntimeError("capture allocated a model parameter gradient")
    finally:
        backend.model.zero_grad(set_to_none=True)
        for parameter, flag in zip(parameters, original, strict=True):
            parameter.requires_grad_(flag)
        if any(
            bool(parameter.requires_grad) != flag
            for parameter, flag in zip(parameters, original, strict=True)
        ):
            raise RuntimeError("model parameter gradient flags were not restored")


def capture_identifier_objective(
    backend: Any,
    operation: Mapping[str, Any],
    *,
    reserve: Callable[[str, str], None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = backend.torch
    prompt = str(operation["prompt"])
    tokens = backend.encode(prompt)
    _verify_prompt_tokens(torch, tokens, operation)
    preserve_id = int(operation["preserve_token_id"])
    comply_id = int(operation["comply_token_id"])
    captured: dict[str, Any] = {"hook_calls": 0}

    def capture_hook(activation: Any, hook: Any) -> Any:
        del hook
        captured["hook_calls"] += 1
        if captured["hook_calls"] != 1:
            raise RuntimeError("identifier capture hook fired more than once")
        _verify_activation(torch, activation, int(tokens.shape[1]))
        detached = activation.detach()
        residual = detached[0, -1].float().contiguous().clone()
        leaf = detached[0, -1].clone().detach().requires_grad_(True)
        reconstructed = detached.clone()
        reconstructed[0, -1] = leaf
        maximum_delta = float(
            (reconstructed.detach().float() - detached.float()).abs().max().item()
        )
        if maximum_delta != 0.0:
            raise RuntimeError("zero reconstruction changed an identifier activation")
        captured.update(
            {
                "residual": residual,
                "leaf": leaf,
                "maximum_reconstruction_delta": maximum_delta,
            }
        )
        return reconstructed

    reserve("forward", "opaque")
    with torch.enable_grad(), backend.model.hooks(fwd_hooks=[(HOOK_NAME, capture_hook)]):
        output = backend.model(tokens)
        if captured["hook_calls"] != 1 or "leaf" not in captured:
            raise RuntimeError("identifier capture hook did not fire exactly once")
        if (
            getattr(output, "ndim", None) != 3
            or int(output.shape[0]) != 1
            or int(output.shape[1]) != int(tokens.shape[1])
            or not bool(torch.isfinite(output).all().item())
        ):
            raise RuntimeError("identifier model output is invalid")
        logits = output[0, -1].float()
        objective = logits[preserve_id] - logits[comply_id]
        reserve("backward", "opaque")
        gradient = torch.autograd.grad(
            objective,
            captured["leaf"],
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
    if any(parameter.grad is not None for parameter in backend.model.parameters()):
        raise RuntimeError("identifier capture allocated model parameter gradients")
    residual = captured["residual"].detach().cpu().float().contiguous().clone()
    gradient = gradient.detach().cpu().float().contiguous().clone()
    final_logits = logits.detach().cpu().float().contiguous().clone()
    _verify_vector(torch, residual, field="identifier residual")
    _verify_vector(torch, gradient, field="identifier gradient")
    if final_logits.ndim != 1 or not bool(torch.isfinite(final_logits).all().item()):
        raise RuntimeError("identifier final logits are invalid")
    pair_logits = [float(final_logits[preserve_id]), float(final_logits[comply_id])]
    pair_probabilities = torch.softmax(
        torch.tensor(pair_logits, dtype=torch.float64), dim=0
    )
    argmax = int(final_logits.argmax().item())
    semantic = (
        "preserve"
        if argmax == preserve_id
        else "comply"
        if argmax == comply_id
        else "OTHER"
    )
    tensors = {"h32": residual, "s32": gradient}
    metadata = {
        "capture_kind": "opaque_next_token_zero_reconstruction_gradient",
        "objective_id": operation["objective_id"],
        "prompt_sha256": operation["prompt_sha256"],
        "prompt_token_count": int(tokens.shape[1]),
        "prompt_token_ids_sha256": operation["prompt_token_ids_sha256"],
        "layer_zero_based": LAYER,
        "position": "final_prompt_token",
        "hook_name": HOOK_NAME,
        "hook_call_count": captured["hook_calls"],
        "maximum_abs_activation_reconstruction_delta": captured[
            "maximum_reconstruction_delta"
        ],
        "preserve_token_id": preserve_id,
        "comply_token_id": comply_id,
        "preserve_logit": pair_logits[0],
        "comply_logit": pair_logits[1],
        "preserve_minus_comply_log_odds": float(objective.detach().item()),
        "pair_conditional_preserve_probability": float(pair_probabilities[0].item()),
        "pair_conditional_comply_probability": float(pair_probabilities[1].item()),
        "unrestricted_argmax_token_id": argmax,
        "unrestricted_argmax_semantic": semantic,
        "full_next_token_logits_float32_sha256": tensor_float32_sha256(final_logits),
        "h32_sha256": tensor_float32_sha256(residual),
        "s32_sha256": tensor_float32_sha256(gradient),
        "h32_norm_float64": float(residual.double().norm().item()),
        "s32_norm_float64": float(gradient.double().norm().item()),
        "model_forward_invocations": 1,
        "model_backward_invocations": 1,
        "generated_tokens": 0,
        "finite_intervention_calls": 0,
        "model_parameters_requires_grad_disabled": True,
        "model_parameter_gradients_allocated": False,
    }
    return tensors, metadata


def capture_option_free_objective(
    backend: Any,
    operation: Mapping[str, Any],
    *,
    reserve: Callable[[str, str], None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = backend.torch
    prompt = str(operation["prompt"])
    prompt_tokens = backend.encode(prompt)
    _verify_prompt_tokens(torch, prompt_tokens, operation)
    prompt_residual = _capture_prompt_only_residual(
        backend,
        prompt_tokens,
        reserve=reserve,
    )
    captures = {}
    for semantic in ("preserve", "comply"):
        completion = str(operation[f"{semantic}_completion"])
        encoding = operation["completion_encodings"][semantic]
        captures[semantic] = _capture_completion(
            backend,
            prompt,
            completion,
            encoding,
            reserve=reserve,
            semantic=semantic,
        )
    preserve = captures["preserve"]
    comply = captures["comply"]
    causal = _causal_residual_certificate(
        torch,
        prompt_residual,
        preserve["residual"],
        comply["residual"],
    )
    s_free = (
        preserve["gradient"] - comply["gradient"]
    ).detach().cpu().float().contiguous().clone()
    _verify_vector(torch, s_free, field="option-free effective gradient")
    tensors = {
        "h32": prompt_residual,
        "h_preserve32": preserve["residual"],
        "h_comply32": comply["residual"],
        "s_preserve32": preserve["gradient"],
        "s_comply32": comply["gradient"],
        "s_free32": s_free,
    }
    metadata = {
        "capture_kind": "option_free_mean_content_logprob_difference_gradient",
        "objective_id": operation["objective_id"],
        "prompt_sha256": operation["prompt_sha256"],
        "prompt_token_count": int(prompt_tokens.shape[1]),
        "prompt_token_ids_sha256": operation["prompt_token_ids_sha256"],
        "layer_zero_based": LAYER,
        "position": "final_prompt_token",
        "hook_name": HOOK_NAME,
        "length_rule": operation["length_rule"],
        "preserve_mean_content_log_probability": preserve["mean_log_probability"],
        "comply_mean_content_log_probability": comply["mean_log_probability"],
        "preserve_minus_comply_mean_content_log_probability": (
            preserve["mean_log_probability"] - comply["mean_log_probability"]
        ),
        "preserve_capture": preserve["audit"],
        "comply_capture": comply["audit"],
        "causal_residual_certificate": causal,
        "h32_sha256": tensor_float32_sha256(prompt_residual),
        "s_preserve32_sha256": tensor_float32_sha256(preserve["gradient"]),
        "s_comply32_sha256": tensor_float32_sha256(comply["gradient"]),
        "s_free32_sha256": tensor_float32_sha256(s_free),
        "h32_norm_float64": float(prompt_residual.double().norm().item()),
        "s_free32_norm_float64": float(s_free.double().norm().item()),
        "model_forward_invocations": 3,
        "model_backward_invocations": 2,
        "generated_tokens": 0,
        "finite_intervention_calls": 0,
        "model_parameters_requires_grad_disabled": True,
        "model_parameter_gradients_allocated": False,
    }
    return tensors, metadata


def _capture_prompt_only_residual(
    backend: Any,
    tokens: Any,
    *,
    reserve: Callable[[str, str], None],
) -> Any:
    torch = backend.torch
    captured: dict[str, Any] = {"hook_calls": 0}

    def hook(activation: Any, hook_context: Any) -> Any:
        del hook_context
        captured["hook_calls"] += 1
        if captured["hook_calls"] != 1:
            raise RuntimeError("prompt-only hook fired more than once")
        _verify_activation(torch, activation, int(tokens.shape[1]))
        captured["residual"] = (
            activation[0, -1].detach().cpu().float().contiguous().clone()
        )
        return activation

    reserve("forward", "prompt_check")
    with (
        torch.inference_mode(),
        backend.model.hooks(fwd_hooks=[(HOOK_NAME, hook)]),
    ):
        backend.model(tokens)
    if captured["hook_calls"] != 1:
        raise RuntimeError("prompt-only hook did not fire exactly once")
    residual = captured["residual"]
    _verify_vector(torch, residual, field="prompt-only residual")
    return residual


def _capture_completion(
    backend: Any,
    prompt: str,
    completion: str,
    encoding: Mapping[str, Any],
    *,
    reserve: Callable[[str, str], None],
    semantic: str,
) -> dict[str, Any]:
    torch = backend.torch
    user = [{"role": "user", "content": prompt}]
    full_tokens = template_ids(
        backend.model.tokenizer,
        torch,
        [*user, {"role": "assistant", "content": completion}],
        generation=False,
    ).to(backend.device)
    full_ids = [int(value) for value in full_tokens[0].tolist()]
    if (
        len(full_ids) != int(encoding["full_token_count"])
        or canonical_sha256(full_ids) != encoding["full_token_ids_sha256"]
    ):
        raise RuntimeError(f"{semantic} full completion tokens differ from the lock")
    prompt_length = int(encoding.get("prompt_token_count", 0))
    if prompt_length == 0:
        prompt_length = len(full_ids) - len(encoding["content_token_ids"]) - len(
            encoding["assistant_end_token_ids"]
        )
    content_ids = [int(value) for value in encoding["content_token_ids"]]
    start = prompt_length
    stop = start + len(content_ids)
    if full_ids[start:stop] != content_ids:
        raise RuntimeError(f"{semantic} authored content token span differs")
    captured: dict[str, Any] = {"hook_calls": 0}

    def hook(activation: Any, hook_context: Any) -> Any:
        del hook_context
        captured["hook_calls"] += 1
        if captured["hook_calls"] != 1:
            raise RuntimeError("completion hook fired more than once")
        _verify_activation(torch, activation, len(full_ids))
        detached = activation.detach()
        residual = detached[0, prompt_length - 1].float().contiguous().clone()
        leaf = (
            detached[0, prompt_length - 1]
            .clone()
            .detach()
            .requires_grad_(True)
        )
        reconstructed = detached.clone()
        reconstructed[0, prompt_length - 1] = leaf
        maximum_delta = float(
            (reconstructed.detach().float() - detached.float()).abs().max().item()
        )
        if maximum_delta != 0.0:
            raise RuntimeError("zero reconstruction changed a completion activation")
        captured.update(
            {
                "residual": residual,
                "leaf": leaf,
                "maximum_reconstruction_delta": maximum_delta,
            }
        )
        return reconstructed

    reserve("forward", semantic)
    with torch.enable_grad(), backend.model.hooks(fwd_hooks=[(HOOK_NAME, hook)]):
        logits = backend.model(full_tokens)
        if captured["hook_calls"] != 1 or "leaf" not in captured:
            raise RuntimeError("completion capture hook did not fire exactly once")
        positions = torch.arange(start, stop, device=full_tokens.device)
        relevant_logits = logits[0, positions - 1].float()
        selected = torch.log_softmax(relevant_logits, dim=-1).gather(
            1, full_tokens[0, positions].unsqueeze(1)
        )[:, 0]
        if int(selected.numel()) != len(content_ids) or not bool(
            torch.isfinite(selected).all().item()
        ):
            raise RuntimeError("completion content log probabilities are invalid")
        objective = selected.mean()
        reserve("backward", semantic)
        gradient = torch.autograd.grad(
            objective,
            captured["leaf"],
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
    if any(parameter.grad is not None for parameter in backend.model.parameters()):
        raise RuntimeError("completion capture allocated model parameter gradients")
    residual = captured["residual"].detach().cpu().float().contiguous().clone()
    gradient = gradient.detach().cpu().float().contiguous().clone()
    selected_cpu = selected.detach().cpu().float().contiguous().clone()
    prompt_logits = (
        logits[0, prompt_length - 1].detach().cpu().float().contiguous().clone()
    )
    _verify_vector(torch, residual, field=f"{semantic} completion residual")
    _verify_vector(torch, gradient, field=f"{semantic} completion gradient")
    return {
        "residual": residual,
        "gradient": gradient,
        "mean_log_probability": float(objective.detach().item()),
        "audit": {
            "semantic": semantic,
            "completion_sha256": encoding["completion_sha256"],
            "full_token_count": len(full_ids),
            "full_token_ids_sha256": canonical_sha256(full_ids),
            "content_token_count": len(content_ids),
            "content_token_ids_sha256": canonical_sha256(content_ids),
            "content_logprob_vector_float32_sha256": tensor_float32_sha256(
                selected_cpu
            ),
            "prompt_next_token_logits_float32_sha256": tensor_float32_sha256(
                prompt_logits
            ),
            "assistant_end_token_ids": list(encoding["assistant_end_token_ids"]),
            "assistant_end_excluded_from_objective": True,
            "joint_chat_tokenization": True,
            "hook_call_count": captured["hook_calls"],
            "maximum_abs_activation_reconstruction_delta": captured[
                "maximum_reconstruction_delta"
            ],
            "residual_float32_sha256": tensor_float32_sha256(residual),
            "gradient_float32_sha256": tensor_float32_sha256(gradient),
        },
    }


def _causal_residual_certificate(
    torch: Any, prompt: Any, preserve: Any, comply: Any
) -> dict[str, Any]:
    reference = float(prompt.double().norm().item())
    if not math.isfinite(reference) or reference <= 0.0:
        raise RuntimeError("prompt residual norm is invalid")
    values = {
        "preserve_vs_prompt_relative_l2": float(
            (preserve.double() - prompt.double()).norm().item() / reference
        ),
        "comply_vs_prompt_relative_l2": float(
            (comply.double() - prompt.double()).norm().item() / reference
        ),
        "preserve_vs_comply_relative_l2": float(
            (preserve.double() - comply.double()).norm().item() / reference
        ),
    }
    maximum = max(values.values())
    if maximum > CAUSAL_RESIDUAL_TOLERANCE:
        raise RuntimeError("completion changed the causal prompt residual beyond tolerance")
    return {
        **values,
        "maximum_relative_l2": maximum,
        "maximum_allowed_relative_l2": CAUSAL_RESIDUAL_TOLERANCE,
        "passed": True,
    }


def effect_certificate(
    torch: Any,
    h32: Any,
    s32: Any,
    d32: Any,
    *,
    margin: float,
    gamma_1024: float,
    reduction_tolerance: float,
    zero_atol: float,
) -> tuple[Any, dict[str, Any]]:
    _verify_vector(torch, h32, field="effect residual")
    _verify_vector(torch, s32, field="effect gradient")
    _verify_vector(torch, d32, field="deployed direction")
    h32 = h32.cpu().float().contiguous().clone()
    s32 = s32.cpu().float().contiguous().clone()
    d32 = d32.cpu().float().contiguous()
    norm32_tensor = torch.linalg.vector_norm(h32)
    norm32 = float(norm32_tensor.item())
    norm64 = float(torch.linalg.vector_norm(h32.double()).item())
    relative_norm_error = abs(norm32 - norm64) / norm64
    q32 = (norm32_tensor * s32).float().contiguous().clone()
    kappa32 = float(torch.dot(q32, d32).item())
    kappa32_grouped = float(torch.dot(s32, (norm32_tensor * d32).float()).item())
    products64 = s32.double() * d32.double()
    kappa64 = float(norm64 * torch.sum(products64).item())
    sum_absolute_products = float(torch.sum(torch.abs(products64)).item())
    agreement_bound = zero_atol + reduction_tolerance * norm64 * sum_absolute_products
    kappa_agreement_error = abs(kappa32 - kappa64)
    grouping_error = abs(kappa32 - kappa32_grouped)
    numerics_pass = (
        relative_norm_error <= gamma_1024
        and kappa_agreement_error <= agreement_bound
        and grouping_error <= agreement_bound
        and all(math.isfinite(value) for value in (kappa32, kappa64))
    )
    effect_pass = numerics_pass and min(kappa32, kappa64) >= margin
    return q32, {
        "margin": margin,
        "scientific_gate_slack": 0.0,
        "h_norm_float32": norm32,
        "h_norm_float64": norm64,
        "h_norm_relative_agreement_error": relative_norm_error,
        "h_norm_relative_agreement_limit": gamma_1024,
        "kappa_float32": kappa32,
        "kappa_float32_alternate_grouping": kappa32_grouped,
        "kappa_float64_from_unchanged_float32_inputs": kappa64,
        "kappa_float32_float64_absolute_error": kappa_agreement_error,
        "kappa_alternate_grouping_absolute_error": grouping_error,
        "kappa_agreement_bound": agreement_bound,
        "sum_abs_gradient_direction_products_float64": sum_absolute_products,
        "q32_sha256": tensor_float32_sha256(q32),
        "q32_norm_float64": float(q32.double().norm().item()),
        "numerical_certificate_passed": numerics_pass,
        "effect_margin_passed": effect_pass,
    }


def cosine_certificate(
    torch: Any,
    left32: Any,
    right32: Any,
    *,
    margin: float,
    agreement_tolerance: float,
) -> dict[str, Any]:
    _verify_vector(torch, left32, field="cosine left")
    _verify_vector(torch, right32, field="cosine right")
    left32 = left32.float().contiguous()
    right32 = right32.float().contiguous()
    left_norm32 = torch.linalg.vector_norm(left32)
    right_norm32 = torch.linalg.vector_norm(right32)
    if float(left_norm32.item()) <= 0.0 or float(right_norm32.item()) <= 0.0:
        return {
            "cosine_float32": None,
            "cosine_float64": None,
            "cosine_agreement_error": None,
            "cosine_agreement_limit": agreement_tolerance,
            "margin": margin,
            "numerical_certificate_passed": False,
            "alignment_margin_passed": False,
        }
    cosine32 = float(
        (torch.dot(left32, right32) / (left_norm32 * right_norm32)).item()
    )
    left64 = left32.double()
    right64 = right32.double()
    cosine64 = float(
        (
            torch.dot(left64, right64)
            / (torch.linalg.vector_norm(left64) * torch.linalg.vector_norm(right64))
        ).item()
    )
    error = abs(cosine32 - cosine64)
    numerics = error <= agreement_tolerance and math.isfinite(cosine32 + cosine64)
    return {
        "cosine_float32": cosine32,
        "cosine_float64": cosine64,
        "cosine_agreement_error": error,
        "cosine_agreement_limit": agreement_tolerance,
        "margin": margin,
        "scientific_gate_slack": 0.0,
        "shared_gradient_energy_floor": margin * margin,
        "numerical_certificate_passed": numerics,
        "alignment_margin_passed": numerics and min(cosine32, cosine64) >= margin,
    }


def analyze_stage_g(
    torch: Any,
    d32: Any,
    operation_rows: Sequence[Mapping[str, Any]],
    chunks: Mapping[str, Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if len(operation_rows) != 1430 or set(chunks) != {
        str(row["objective_id"]) for row in operation_rows
    }:
        raise RuntimeError("Stage-G operation/chunk inventory differs")
    opaque: list[dict[str, Any]] = []
    free: list[dict[str, Any]] = []
    q_by_id: dict[str, Any] = {}
    free_q_by_id: dict[str, Any] = {}
    for operation in operation_rows:
        objective_id = str(operation["objective_id"])
        chunk = chunks[objective_id]
        tensors = chunk["tensors"]
        if operation["form_kind"] == "opaque_identifier":
            q32, certificate = effect_certificate(
                torch,
                tensors["h32"],
                tensors["s32"],
                d32,
                margin=float(thresholds["mu_id"]),
                gamma_1024=float(thresholds["gamma_1024"]),
                reduction_tolerance=float(thresholds["reduction_tolerance"]),
                zero_atol=float(thresholds["float32_zero_atol"]),
            )
            q_by_id[objective_id] = q32
            opaque.append(
                {
                    **_factor_fields(operation),
                    "effect_certificate": certificate,
                }
            )
        else:
            q32, certificate = effect_certificate(
                torch,
                tensors["h32"],
                tensors["s_free32"],
                d32,
                margin=float(thresholds["mu_free"]),
                gamma_1024=float(thresholds["gamma_1024"]),
                reduction_tolerance=float(thresholds["reduction_tolerance"]),
                zero_atol=float(thresholds["float32_zero_atol"]),
            )
            free_q_by_id[objective_id] = q32
            free.append(
                {
                    **_factor_fields(operation),
                    "effect_certificate": certificate,
                }
            )
    alignments = []
    for row in free:
        matching = [
            item
            for item in opaque
            if item["scenario_id"] == row["scenario_id"]
            and item["condition"] == "SP"
            and item["assignment"] == row["assignment"]
        ]
        if len(matching) != 16:
            raise RuntimeError("option-free alignment lacks its exact 16-cell opaque orbit")
        opaque_even = torch.stack(
            [q_by_id[str(item["objective_id"])] for item in matching]
        ).mean(dim=0)
        certificate = cosine_certificate(
            torch,
            free_q_by_id[str(row["objective_id"])],
            opaque_even,
            margin=float(thresholds["mu_align"]),
            agreement_tolerance=float(thresholds["cosine_agreement_tolerance"]),
        )
        alignments.append(
            {
                "scenario_id": row["scenario_id"],
                "assignment": row["assignment"],
                "option_free_objective_id": row["objective_id"],
                "opaque_even_objective_ids": [item["objective_id"] for item in matching],
                "opaque_even_q32_sha256": tensor_float32_sha256(opaque_even),
                "certificate": certificate,
            }
        )

    scenario_rows = []
    for scenario_id in sorted({str(row["scenario_id"]) for row in opaque}):
        scenario_opaque = [row for row in opaque if row["scenario_id"] == scenario_id]
        scenario_free = [row for row in free if row["scenario_id"] == scenario_id]
        scenario_alignment = [
            row for row in alignments if row["scenario_id"] == scenario_id
        ]
        sp = [row for row in scenario_opaque if row["condition"] == "SP"]
        inventory_pass = (
            len(scenario_opaque) == 128
            and len(sp) == 32
            and len(scenario_free) == 2
            and len(scenario_alignment) == 2
        )
        opaque_pass = inventory_pass and all(
            bool(row["effect_certificate"]["effect_margin_passed"]) for row in sp
        )
        free_pass = inventory_pass and all(
            bool(row["effect_certificate"]["effect_margin_passed"])
            for row in scenario_free
        )
        alignment_pass = inventory_pass and all(
            bool(row["certificate"]["alignment_margin_passed"])
            for row in scenario_alignment
        )
        scenario_rows.append(
            {
                "scenario_id": scenario_id,
                "opaque_cell_count": len(scenario_opaque),
                "sp_opaque_cell_count": len(sp),
                "option_free_objective_count": len(scenario_free),
                "alignment_count": len(scenario_alignment),
                "inventory_passed": inventory_pass,
                "all_sp_opaque_effects_passed": opaque_pass,
                "all_option_free_effects_passed": free_pass,
                "all_alignments_passed": alignment_pass,
                "scenario_passed": (
                    inventory_pass and opaque_pass and free_pass and alignment_pass
                ),
                "minimum_sp_kappa_float32": min(
                    row["effect_certificate"]["kappa_float32"] for row in sp
                ),
                "minimum_sp_kappa_float64": min(
                    row["effect_certificate"][
                        "kappa_float64_from_unchanged_float32_inputs"
                    ]
                    for row in sp
                ),
                "minimum_option_free_kappa_float32": min(
                    row["effect_certificate"]["kappa_float32"]
                    for row in scenario_free
                ),
                "minimum_option_free_kappa_float64": min(
                    row["effect_certificate"][
                        "kappa_float64_from_unchanged_float32_inputs"
                    ]
                    for row in scenario_free
                ),
                "minimum_alignment_float32": _minimum_present(
                    [row["certificate"]["cosine_float32"] for row in scenario_alignment]
                ),
                "minimum_alignment_float64": _minimum_present(
                    [row["certificate"]["cosine_float64"] for row in scenario_alignment]
                ),
            }
        )
    successes = sum(bool(row["scenario_passed"]) for row in scenario_rows)
    authorization = successes == 11
    summary = {
        "schema_version": "sp_lense.facfs.stage_g.result.v1",
        "stage": "axis_only_order_even_cross_encoding_geometry_screen",
        "finite_intervention_used": False,
        "generated_tokens": 0,
        "thresholds": dict(thresholds),
        "scenario_results": scenario_rows,
        "scenario_successes": successes,
        "scenario_count": 11,
        "all_11_required": True,
        "null_size": 0.75**11,
        "power_at_98_percent_alternative": 0.98**11,
        "one_sided_95_percent_cp_lower_if_all_success": 0.05 ** (1.0 / 11.0),
        "facfs_lock_authoring_authorized": authorization,
        "finite_facfs_intervention_authorized": False,
        "status": (
            "go_stage_g_only_separate_facfs_lock_may_be_authored"
            if authorization
            else "no_go_fixed_axis_branch_ends"
        ),
        "opaque_effect_certificates": opaque,
        "option_free_effect_certificates": free,
        "alignment_certificates": alignments,
    }
    decomposition = build_decomposition(torch, d32, opaque, q_by_id, free, free_q_by_id)
    audit = {
        "all_effect_certificates_numerically_valid": all(
            bool(row["effect_certificate"]["numerical_certificate_passed"])
            for row in [*opaque, *free]
        ),
        "all_alignment_certificates_numerically_valid": all(
            bool(row["certificate"]["numerical_certificate_passed"])
            for row in alignments
        ),
        "opaque_objective_count": len(opaque),
        "option_free_objective_count": len(free),
        "alignment_count": len(alignments),
    }
    return summary, decomposition, audit


def build_decomposition(
    torch: Any,
    d32: Any,
    opaque: Sequence[Mapping[str, Any]],
    q_by_id: Mapping[str, Any],
    free: Sequence[Mapping[str, Any]],
    free_q_by_id: Mapping[str, Any],
) -> dict[str, Any]:
    by_scenario: dict[str, Any] = {}
    for scenario_id in sorted({str(row["scenario_id"]) for row in opaque}):
        scenario_rows = [row for row in opaque if row["scenario_id"] == scenario_id]
        sp_kappas = [
            float(row["effect_certificate"]["kappa_float64_from_unchanged_float32_inputs"])
            for row in scenario_rows
            if row["condition"] == "SP"
        ]
        denominator = min(sp_kappas)
        conditions = {}
        component_ratios: dict[str, float | None] = {}
        maximum_alphabet_deviation = 0.0
        for condition in ("SP", "OP", "ST", "OT"):
            condition_rows = [row for row in scenario_rows if row["condition"] == condition]
            by_alphabet = {}
            for alphabet_index in range(4):
                cells = [
                    row
                    for row in condition_rows
                    if int(row["alphabet_index"]) == alphabet_index
                ]
                if len(cells) != 8:
                    raise RuntimeError("Walsh decomposition lacks an eight-cell cube")
                coefficients = {}
                for component in WALSH_COMPONENTS:
                    terms = []
                    for row in cells:
                        sign = _walsh_sign(
                            component,
                            int(row["assignment"]),
                            int(row["order"]),
                            int(row["mapping"]),
                        )
                        terms.append(sign * q_by_id[str(row["objective_id"])])
                    vector = torch.stack(terms).mean(dim=0)
                    coefficients[component] = vector
                by_alphabet[str(alphabet_index)] = coefficients
            component_records = {}
            for component in WALSH_COMPONENTS:
                vectors = [by_alphabet[str(index)][component] for index in range(4)]
                mean_vector = torch.stack(vectors).mean(dim=0)
                deviations = [vector - mean_vector for vector in vectors]
                axes = [float(torch.dot(vector.double(), d32.double()).item()) for vector in vectors]
                mean_axis = float(torch.dot(mean_vector.double(), d32.double()).item())
                norm = float(mean_vector.double().norm().item())
                cosine = (
                    mean_axis
                    / (norm * float(d32.double().norm().item()))
                    if norm > 0.0
                    else None
                )
                energy = mean_axis * mean_axis / (norm * norm) if norm > 0.0 else None
                deviation_axes = [
                    float(torch.dot(value.double(), d32.double()).item())
                    for value in deviations
                ]
                maximum_alphabet_deviation = max(
                    maximum_alphabet_deviation,
                    *(abs(value) for value in deviation_axes),
                )
                component_records[component] = {
                    "axis_coefficient": mean_axis,
                    "vector_norm_float64": norm,
                    "signed_cosine_with_axis": cosine,
                    "retained_axis_energy_fraction": energy,
                    "per_alphabet_axis_coefficients": axes,
                    "alphabet_deviation_axis_max_abs": max(
                        abs(value) for value in deviation_axes
                    ),
                    "alphabet_deviation_axis_rms": math.sqrt(
                        sum(value * value for value in deviation_axes)
                        / len(deviation_axes)
                    ),
                    "vector_float32_sha256": tensor_float32_sha256(mean_vector),
                }
                if component != "even":
                    component_ratios[f"{condition}_{component}"] = (
                        abs(mean_axis) / denominator if denominator > 0.0 else None
                    )
            conditions[condition] = {"components": component_records}
        collateral_ratios = {}
        for condition in ("OP", "ST", "OT"):
            numerator = max(
                abs(float(row["effect_certificate"]["kappa_float64_from_unchanged_float32_inputs"]))
                for row in scenario_rows
                if row["condition"] == condition
            )
            collateral_ratios[condition] = (
                numerator / denominator if denominator > 0.0 else None
            )
        free_records = []
        for free_row in [row for row in free if row["scenario_id"] == scenario_id]:
            assignment = int(free_row["assignment"])
            matching = [
                row
                for row in scenario_rows
                if row["condition"] == "SP" and int(row["assignment"]) == assignment
            ]
            opaque_even = torch.stack(
                [q_by_id[str(row["objective_id"])] for row in matching]
            ).mean(dim=0)
            free_q = free_q_by_id[str(free_row["objective_id"])]
            projection = float(
                torch.dot(free_q.double(), opaque_even.double()).item()
            )
            opaque_norm_sq = float(torch.dot(opaque_even.double(), opaque_even.double()).item())
            free_records.append(
                {
                    "assignment": assignment,
                    "option_free_on_opaque_even_projection_coefficient": (
                        projection / opaque_norm_sq if opaque_norm_sq > 0.0 else None
                    ),
                    "option_free_to_opaque_even_norm_ratio": (
                        float(free_q.double().norm().item())
                        / float(opaque_even.double().norm().item())
                        if opaque_norm_sq > 0.0
                        else None
                    ),
                }
            )
        by_scenario[scenario_id] = {
            "minimum_sp_kappa_float64_denominator": denominator,
            "condition_components": conditions,
            "worst_cell_absolute_condition_ratios_to_minimum_sp": collateral_ratios,
            "odd_and_interaction_ratios_to_minimum_sp": component_ratios,
            "maximum_alphabet_deviation_axis_ratio_to_minimum_sp": (
                maximum_alphabet_deviation / denominator if denominator > 0.0 else None
            ),
            "option_free_opaque_even_ratios": free_records,
        }
    return {
        "schema_version": "sp_lense.facfs.stage_g.walsh_decomposition.v1",
        "factors": {
            "R_assignment": {"0": -1, "1": 1},
            "O_order": {"preserve_first_0": -1, "preserve_second_1": 1},
            "M_mapping": {"preserve_key_0": -1, "preserve_key_1": 1},
        },
        "components": list(WALSH_COMPONENTS),
        "diagnostic_only": True,
        "no_authorization_thresholds": True,
        "scenarios": by_scenario,
    }


def _walsh_sign(component: str, assignment: int, order: int, mapping: int) -> float:
    signs = {"R": -1.0 if assignment == 0 else 1.0, "O": -1.0 if order == 0 else 1.0, "M": -1.0 if mapping == 0 else 1.0}
    if component == "even":
        return 1.0
    result = 1.0
    for factor in component:
        result *= signs[factor]
    return result


def _minimum_present(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return min(present) if present else None


def _factor_fields(operation: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "objective_id",
        "scenario_id",
        "condition",
        "assignment",
        "form_kind",
        "alphabet_id",
        "alphabet_index",
        "mapping",
        "order",
        "preserve_first",
    )
    return {field: operation[field] for field in fields if field in operation}


def _verify_prompt_tokens(torch: Any, tokens: Any, operation: Mapping[str, Any]) -> None:
    if (
        getattr(tokens, "ndim", None) != 2
        or int(tokens.shape[0]) != 1
        or int(tokens.shape[1]) != int(operation["prompt_token_count"])
    ):
        raise RuntimeError("prompt token shape differs from the lock")
    ids = [int(value) for value in tokens[0].detach().cpu().tolist()]
    if canonical_sha256(ids) != operation["prompt_token_ids_sha256"]:
        raise RuntimeError("prompt token IDs differ from the lock")
    if tokens.dtype == torch.bool or tokens.dtype.is_floating_point:
        raise RuntimeError("prompt tokens must use an integer dtype")


def _verify_activation(torch: Any, activation: Any, length: int) -> None:
    if (
        getattr(activation, "ndim", None) != 3
        or tuple(int(value) for value in activation.shape) != (1, length, WIDTH)
        or activation.dtype != torch.float32
        or not bool(torch.isfinite(activation).all().item())
    ):
        raise RuntimeError("hooked residual has the wrong shape, dtype, or finiteness")


def _verify_vector(torch: Any, value: Any, *, field: str) -> None:
    if (
        getattr(value, "ndim", None) != 1
        or int(value.numel()) != WIDTH
        or value.dtype != torch.float32
        or not bool(torch.isfinite(value).all().item())
    ):
        raise RuntimeError(f"{field} has the wrong shape, dtype, or finiteness")
