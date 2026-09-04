from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __package__:
    from scripts import layer_localization_pilot as pilot
else:
    import layer_localization_pilot as pilot

V2_LOCK_RELATIVE_PATH = Path("configs/layer_localization_pilot_v2.json")
V2_SCRIPT_RELATIVE_PATH = Path("scripts/layer_localization_pilot_v2.py")
V2_TEST_RELATIVE_PATH = Path("tests/test_layer_localization_pilot_v2.py")
V2_OUTPUT_RELATIVE_PATH = Path("evidence/layer_localization_qwen35_08b_v2")

V2_SOURCE_RELATIVE_PATHS = (
    V2_SCRIPT_RELATIVE_PATH,
    V2_LOCK_RELATIVE_PATH,
    Path("scripts/layer_localization_pilot.py"),
    Path("configs/layer_localization_pilot.json"),
    Path("data/direction_repair_nonsealed_cases.json"),
    Path("pyproject.toml"),
    Path("src/sp_lense/backend.py"),
    Path("src/sp_lense/comparison_controls.py"),
    Path("src/sp_lense/comparison_intervention.py"),
    Path("src/sp_lense/comparison_runtime.py"),
    Path("src/sp_lense/conditional_gate_data.py"),
    Path("src/sp_lense/config.py"),
    Path("src/sp_lense/core.py"),
    Path("src/sp_lense/steering_methods.py"),
    V2_TEST_RELATIVE_PATH,
)


def _capture_final_prompt_gradients(
    backend: Any,
    prompt: str,
    preserve_label: str,
    comply_label: str,
    *,
    layers: Sequence[int],
    boundary: Any,
) -> dict[int, Any]:
    """V1 gradient capture with the resident TransformerLens hook keyword accepted.

    The hook context is deliberately unused. All gradient objectives, layer handling,
    tensor operations, and validation checks are otherwise identical to V1.
    """

    requested = tuple(sorted(int(layer) for layer in layers))
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("gradient layers must be non-empty and unique")
    tokens = backend.encode(prompt)
    if boundary.prompt_length != int(tokens.shape[-1]):
        raise ValueError("choice-boundary evidence has the wrong prompt length")
    captured: dict[int, Any] = {}

    def make_hook(layer: int) -> Any:
        def hook(activation: Any, hook: Any) -> Any:
            del hook
            if layer == requested[0]:
                activation = activation.detach().requires_grad_(True)
            captured[layer] = activation
            return activation

        return hook

    hooks = [(f"blocks.{layer}.hook_out", make_hook(layer)) for layer in requested]
    backend.model.zero_grad(set_to_none=True)
    with backend.torch.enable_grad(), backend.model.hooks(fwd_hooks=hooks):
        logits = backend.model(tokens)[0, -1].float()
        objective = (
            logits[boundary.token_id(preserve_label)] - logits[boundary.token_id(comply_label)]
        )
        gradients = backend.torch.autograd.grad(
            objective,
            [captured[layer] for layer in requested],
            retain_graph=False,
            create_graph=False,
        )
    backend.model.zero_grad(set_to_none=True)
    output = {
        layer: gradient[0, -1].detach().to(device="cpu").float().contiguous()
        for layer, gradient in zip(requested, gradients, strict=True)
    }
    if any(tuple(value.shape) != (1024,) for value in output.values()):
        raise RuntimeError("captured semantic gradient has an invalid residual width")
    if any(not bool(value.isfinite().all().item()) for value in output.values()):
        raise RuntimeError("captured semantic gradient is non-finite")
    return output


def configure_v2() -> None:
    pilot.LOCK_RELATIVE_PATH = V2_LOCK_RELATIVE_PATH
    pilot.SCRIPT_RELATIVE_PATH = V2_SCRIPT_RELATIVE_PATH
    pilot.OUTPUT_RELATIVE_PATH = V2_OUTPUT_RELATIVE_PATH
    pilot.SOURCE_RELATIVE_PATHS = V2_SOURCE_RELATIVE_PATHS
    pilot._capture_final_prompt_gradients = _capture_final_prompt_gradients


def main(argv: Sequence[str] | None = None) -> int:
    configure_v2()
    return pilot.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
