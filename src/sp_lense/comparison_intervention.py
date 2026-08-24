from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

Geometry = Literal[
    "matched_final_prompt",
    "caa_post_prompt",
    "bipo_all_tokens",
    "persona_response",
]
MagnitudeMode = Literal["residual_relative", "canonical_coefficient"]
InterventionPhase = Literal["full_sequence", "prefill", "decode"]


@dataclass(frozen=True)
class InterventionSpec:
    """A fully specified residual-stream intervention.

    ``prompt_length`` is deliberately part of the specification.  During autoregressive
    generation the prompt's final token remains at ``prompt_length - 1``; it must not
    drift to the newest generated token in the matched comparison.
    """

    layer: int
    direction: Any
    strength: float
    geometry: Geometry
    prompt_length: int
    magnitude_mode: MagnitudeMode = "residual_relative"

    def validate(self) -> None:
        if self.layer < 0:
            raise ValueError("layer must be non-negative")
        if self.prompt_length < 1:
            raise ValueError("prompt_length must be positive")
        if not math.isfinite(self.strength):
            raise ValueError("strength must be finite")
        if getattr(self.direction, "ndim", None) != 1:
            raise ValueError("direction must be a one-dimensional tensor")
        norm = float(self.direction.detach().float().norm().item())
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError("direction must have finite, non-zero norm")


def hook_name(layer: int) -> str:
    if layer < 0:
        raise ValueError("layer must be non-negative")
    return f"blocks.{layer}.hook_out"


def intervention_mask(
    torch: Any,
    activation: Any,
    spec: InterventionSpec,
    *,
    phase: InterventionPhase = "full_sequence",
) -> Any:
    """Return a ``[batch, sequence, 1]`` mask for an intervention schedule."""

    spec.validate()
    if activation.ndim != 3:
        raise ValueError("residual activation must have shape [batch, sequence, d_model]")
    if activation.shape[-1] != spec.direction.numel():
        raise ValueError(
            "direction width does not match activation width: "
            f"{spec.direction.numel()} != {activation.shape[-1]}"
        )
    sequence_length = int(activation.shape[1])
    if phase not in {"full_sequence", "prefill", "decode"}:
        raise ValueError(f"unknown intervention phase: {phase!r}")
    if phase == "prefill" and sequence_length != spec.prompt_length:
        raise ValueError(
            "cached prefill activation length must equal the intervention prompt length"
        )
    if phase == "decode" and sequence_length != 1:
        raise ValueError("cached decode intervention requires exactly one new token")
    if phase == "full_sequence" and sequence_length < spec.prompt_length:
        raise ValueError("full-sequence activation is shorter than the intervention prompt")
    mask = torch.zeros(
        (activation.shape[0], sequence_length, 1),
        device=activation.device,
        dtype=torch.bool,
    )
    prompt_index = spec.prompt_length - 1

    if spec.geometry not in {
        "matched_final_prompt",
        "caa_post_prompt",
        "bipo_all_tokens",
        "persona_response",
    }:
        raise ValueError(f"unknown intervention geometry: {spec.geometry!r}")
    if phase == "decode":
        if spec.geometry != "matched_final_prompt":
            mask[:] = True
        return mask
    if spec.geometry == "bipo_all_tokens":
        mask[:] = True
    elif spec.geometry == "matched_final_prompt":
        mask[:, prompt_index, :] = True
    else:
        # CAA and persona change the prompt-final token during prefill and every
        # generated token during one-token cached decode. The full-sequence mask is
        # the exact no-cache reference schedule used by equivalence tests.
        mask[:, prompt_index:, :] = True
    return mask


def apply_intervention(
    torch: Any,
    activation: Any,
    spec: InterventionSpec,
    *,
    phase: InterventionPhase = "full_sequence",
) -> Any:
    """Apply ``spec`` without mutating the activation tensor.

    Matched experiments use a unit vector and set the perturbation norm at each
    selected token to ``abs(strength) * ||residual||``.  Canonical experiments may
    instead use the published raw-vector coefficient convention.
    """

    mask = intervention_mask(torch, activation, spec, phase=phase)
    working = activation.float()
    direction = spec.direction.detach().to(device=activation.device, dtype=working.dtype)

    if spec.magnitude_mode == "residual_relative":
        direction = direction / direction.norm().clamp_min(1e-12)
        coefficient = spec.strength * working.norm(dim=-1, keepdim=True)
    elif spec.magnitude_mode == "canonical_coefficient":
        coefficient = torch.full(
            (*working.shape[:-1], 1),
            spec.strength,
            device=working.device,
            dtype=working.dtype,
        )
    else:  # pragma: no cover - protected by the Literal type for typed callers
        raise ValueError(f"unknown magnitude mode: {spec.magnitude_mode!r}")

    changed = working + mask.to(working.dtype) * coefficient * direction.view(1, 1, -1)
    return changed.to(dtype=activation.dtype)


def make_intervention_hook(
    torch: Any, spec: InterventionSpec, *, phase: InterventionPhase = "full_sequence"
) -> Any:
    """Create a TransformerLens-compatible forward hook."""

    spec.validate()

    def hook(activation: Any, hook_context: Any) -> Any:
        del hook_context
        return apply_intervention(torch, activation, spec, phase=phase)

    return hook


def hooks_for_spec(
    torch: Any, spec: InterventionSpec, *, phase: InterventionPhase = "full_sequence"
) -> list[tuple[str, Any]]:
    return [(hook_name(spec.layer), make_intervention_hook(torch, spec, phase=phase))]


def perturbation_norms(torch: Any, before: Any, after: Any) -> Any:
    if before.shape != after.shape:
        raise ValueError("before and after tensors must have identical shapes")
    if before.ndim != 3:
        raise ValueError("expected tensors shaped [batch, sequence, d_model]")
    return (after.float() - before.float()).norm(dim=-1)
