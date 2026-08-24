from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .comparison_runtime import completion_logprob_sum, encode_prompt_and_completion
from .steering_methods import (
    assert_model_frozen,
    bipo_loss,
    freeze_model_parameters,
    normalize_direction,
)

BiPOGeometry = Literal["matched_final_prompt", "canonical_broadcast"]


@dataclass(frozen=True)
class BiPOTrainingExample:
    case_id: str
    prompt: str
    preserve_completion: str
    comply_completion: str


@dataclass(frozen=True)
class BiPOTrainingConfig:
    beta: float = 0.1
    learning_rate: float = 5e-4
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0
    epochs: int = 20
    checkpoint_epochs: tuple[int, ...] = (5, 20)
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    seed: int = 11

    def validate(self) -> None:
        if self.beta <= 0 or not math.isfinite(self.beta):
            raise ValueError("beta must be finite and positive")
        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite and positive")
        if self.weight_decay < 0 or not math.isfinite(self.weight_decay):
            raise ValueError("weight_decay must be finite and non-negative")
        if self.max_grad_norm <= 0 or not math.isfinite(self.max_grad_norm):
            raise ValueError("max_grad_norm must be finite and positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if (
            not self.checkpoint_epochs
            or any(
                isinstance(epoch, bool)
                or not isinstance(epoch, int)
                or epoch < 1
                or epoch > self.epochs
                for epoch in self.checkpoint_epochs
            )
            or tuple(sorted(set(self.checkpoint_epochs))) != self.checkpoint_epochs
        ):
            raise ValueError(
                "checkpoint_epochs must be sorted, unique, positive, and no greater than epochs"
            )
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")


def _example_key(example: BiPOTrainingExample, completion_kind: str) -> str:
    completion = (
        example.preserve_completion if completion_kind == "preserve" else example.comply_completion
    )
    payload = f"{example.case_id}\0{example.prompt}\0{completion_kind}\0{completion}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BiPOReferenceCache:
    identity: dict[str, Any]
    values: dict[str, float]


def _reference_identity(backend: Any) -> dict[str, Any]:
    template = getattr(backend.model.tokenizer, "chat_template", None)
    template_sha256 = hashlib.sha256(str(template).encode("utf-8")).hexdigest()
    return {
        "schema_version": "sp_lense.bipo_reference.v2",
        "model_id": backend.config.model.id,
        "model_revision": backend.config.model.revision,
        "prompt_format": backend.config.model.prompt_format,
        "chat_template_sha256": template_sha256,
        "enable_thinking": False,
        "completion_encoding": "joint_chat_template_with_assistant_end_token",
    }


def _reference_logprob(backend: Any, prompt: str, completion: str) -> float:
    prompt_tokens, tokens = encode_prompt_and_completion(backend, prompt, completion)
    with backend.torch.inference_mode():
        logits = backend.model(tokens)
    values = completion_logprob_sum(
        backend.torch, logits, tokens, int(prompt_tokens.shape[-1])
    )
    return float(values[0].item())


def build_reference_logprob_cache(
    backend: Any, examples: Sequence[BiPOTrainingExample]
) -> BiPOReferenceCache:
    """Cache π0 response log-probabilities without duplicating the frozen model.

    The published implementation keeps a second reference-model copy.  Since the base
    model is frozen and the reference never changes, these scalar values are exactly
    equivalent and are substantially more practical on a 32 GB CPU laptop.
    """

    cache: dict[str, float] = {}
    for example in examples:
        cache[_example_key(example, "preserve")] = _reference_logprob(
            backend, example.prompt, example.preserve_completion
        )
        cache[_example_key(example, "comply")] = _reference_logprob(
            backend, example.prompt, example.comply_completion
        )
    return BiPOReferenceCache(identity=_reference_identity(backend), values=cache)


def validate_reference_cache(
    backend: Any,
    examples: Sequence[BiPOTrainingExample],
    cache: BiPOReferenceCache,
) -> None:
    if cache.identity != _reference_identity(backend):
        raise ValueError("BiPO reference cache identity does not match this model/chat template")
    expected = {
        _example_key(example, completion_kind)
        for example in examples
        for completion_kind in ("preserve", "comply")
    }
    observed = set(cache.values)
    if observed != expected:
        raise ValueError(
            f"BiPO reference cache key mismatch: {len(expected - observed)} missing, "
            f"{len(observed - expected)} unexpected"
        )
    if any(not math.isfinite(float(value)) for value in cache.values.values()):
        raise ValueError("BiPO reference cache contains non-finite log-probabilities")


def _trainable_vector_hook(
    torch: Any,
    vector: Any,
    *,
    direction_sign: int,
    geometry: BiPOGeometry,
    prompt_length: int,
) -> Any:
    if direction_sign not in {-1, 1}:
        raise ValueError("direction_sign must be -1 or +1")

    def hook(activation: Any, hook_context: Any) -> Any:
        del hook_context
        if activation.ndim != 3:
            raise ValueError("expected residual activation [batch, sequence, d_model]")
        if vector.numel() != activation.shape[-1]:
            raise ValueError("BiPO vector width does not match model residual width")
        addition = direction_sign * vector.to(dtype=activation.dtype).view(1, 1, -1)
        if geometry == "canonical_broadcast":
            return activation + addition
        if geometry == "matched_final_prompt":
            index = prompt_length - 1
            if index >= activation.shape[1]:
                raise ValueError("prompt final position is outside the sequence")
            result = activation.clone()
            result[:, index : index + 1, :] = result[:, index : index + 1, :] + addition
            return result
        raise ValueError(f"unknown BiPO geometry: {geometry!r}")

    return hook


def differentiable_completion_logprob(
    backend: Any,
    prompt: str,
    completion: str,
    vector: Any,
    *,
    layer: int,
    direction_sign: int,
    geometry: BiPOGeometry,
) -> Any:
    prompt_tokens, tokens = encode_prompt_and_completion(backend, prompt, completion)
    prompt_length = int(prompt_tokens.shape[-1])
    hook = _trainable_vector_hook(
        backend.torch,
        vector,
        direction_sign=direction_sign,
        geometry=geometry,
        prompt_length=prompt_length,
    )
    with backend.model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_out", hook)]):
        logits = backend.model(tokens)
    return completion_logprob_sum(backend.torch, logits, tokens, prompt_length)[0]


def validation_preference_loss(
    backend: Any,
    examples: Sequence[BiPOTrainingExample],
    vector: Any,
    *,
    layer: int,
    geometry: BiPOGeometry,
    beta: float,
    reference_cache: BiPOReferenceCache | None = None,
) -> float:
    """Held-out BiPO objective averaged over both signs, never downstream selectivity."""

    if not examples:
        raise ValueError("BiPO validation requires at least one example")
    cache = reference_cache or build_reference_logprob_cache(backend, examples)
    validate_reference_cache(backend, examples, cache)
    losses: list[float] = []
    with backend.torch.inference_mode():
        for example in examples:
            reference_preferred = backend.torch.tensor(
                [cache.values[_example_key(example, "preserve")]],
                device=backend.device,
                dtype=backend.torch.float32,
            )
            reference_rejected = backend.torch.tensor(
                [cache.values[_example_key(example, "comply")]],
                device=backend.device,
                dtype=backend.torch.float32,
            )
            for direction_sign in (-1, 1):
                policy_preferred = differentiable_completion_logprob(
                    backend,
                    example.prompt,
                    example.preserve_completion,
                    vector,
                    layer=layer,
                    direction_sign=direction_sign,
                    geometry=geometry,
                ).reshape(1)
                policy_rejected = differentiable_completion_logprob(
                    backend,
                    example.prompt,
                    example.comply_completion,
                    vector,
                    layer=layer,
                    direction_sign=direction_sign,
                    geometry=geometry,
                ).reshape(1)
                loss = bipo_loss(
                    backend.torch,
                    policy_preferred,
                    policy_rejected,
                    reference_preferred,
                    reference_rejected,
                    direction_sign,
                    beta=beta,
                )
                losses.append(float(loss.item()))
    return sum(losses) / len(losses)


def _lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    if total_steps < 1:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return float(step) / float(warmup_steps)
    denominator = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / denominator))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _optimizer_state_record(optimizer: Any, vector: Any) -> dict[str, Any]:
    """Serialize the single-vector AdamW state without pickle."""

    state = optimizer.state.get(vector)
    if not state:
        raise RuntimeError("BiPO optimizer produced no state for the learned vector")
    output: dict[str, Any] = {}
    for key, value in sorted(state.items()):
        if hasattr(value, "detach"):
            tensor = value.detach().float().cpu().contiguous()
            output[key] = {
                "shape": list(tensor.shape),
                "dtype": "float32",
                "values": tensor.reshape(-1).tolist(),
                "float32_sha256": hashlib.sha256(
                    tensor.numpy().tobytes(order="C")
                ).hexdigest(),
            }
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            output[key] = value
        else:  # pragma: no cover - AdamW currently uses only tensors/scalars
            raise TypeError(f"unsupported optimizer-state value for {key!r}")
    payload = json.dumps(
        output, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "optimizer": "AdamW",
        "state": output,
        "canonical_json_sha256": hashlib.sha256(payload).hexdigest(),
    }


def train_bipo_direction(
    backend: Any,
    examples: Sequence[BiPOTrainingExample],
    *,
    layer: int,
    geometry: BiPOGeometry,
    config: BiPOTrainingConfig,
    reference_cache: BiPOReferenceCache | None = None,
) -> dict[str, Any]:
    """Train a bidirectional residual vector with the published BiPO DPO objective."""

    config.validate()
    if not examples:
        raise ValueError("BiPO requires at least one training example")
    if layer < 0 or layer >= backend.model.cfg.n_layers:
        raise ValueError("BiPO layer is outside the model")

    torch = backend.torch
    freeze_model_parameters(backend.model)
    assert_model_frozen(backend.model)
    reference_cache = reference_cache or build_reference_logprob_cache(backend, examples)
    validate_reference_cache(backend, examples, reference_cache)
    d_model = int(backend.model.cfg.d_model)
    vector = torch.nn.Parameter(torch.zeros(d_model, device=backend.device, dtype=torch.float32))
    optimizer = torch.optim.AdamW(
        [vector], lr=config.learning_rate, weight_decay=config.weight_decay
    )
    optimizer_steps_per_epoch = math.ceil(len(examples) / config.gradient_accumulation_steps)
    total_optimizer_steps = optimizer_steps_per_epoch * config.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_multiplier(step, total_optimizer_steps, config.warmup_steps),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    history: list[dict[str, Any]] = []
    checkpoints: dict[str, Any] = {}
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0

    for epoch in range(config.epochs):
        permutation = torch.randperm(len(examples), generator=generator).tolist()
        epoch_losses: list[float] = []
        pre_clip_norms: list[float] = []
        example_direction_counts = {-1: 0, 1: 0}
        effective_batch_signs: list[int] = []
        pending = 0
        batch_direction_sign: int | None = None
        for order_index, example_index in enumerate(permutation):
            example = examples[example_index]
            if pending == 0:
                batch_direction_sign = (
                    1 if int(torch.randint(0, 2, (1,), generator=generator).item()) else -1
                )
                effective_batch_signs.append(batch_direction_sign)
            if batch_direction_sign is None:  # pragma: no cover - invariant above
                raise RuntimeError("failed to sample a BiPO effective-batch sign")
            direction_sign = batch_direction_sign
            example_direction_counts[direction_sign] += 1
            # BiPO keeps y_w/y_l fixed and multiplies their log-ratio gap by d.
            # For d=-1 this makes the rejected/compliant answer preferable under -v.
            # Swapping the two completions here as well would invert the target twice.
            target_kind, opposite_kind = "preserve", "comply"
            target_completion = example.preserve_completion
            opposite_completion = example.comply_completion

            policy_target = differentiable_completion_logprob(
                backend,
                example.prompt,
                target_completion,
                vector,
                layer=layer,
                direction_sign=direction_sign,
                geometry=geometry,
            )
            policy_opposite = differentiable_completion_logprob(
                backend,
                example.prompt,
                opposite_completion,
                vector,
                layer=layer,
                direction_sign=direction_sign,
                geometry=geometry,
            )
            reference_target = torch.tensor(
                reference_cache.values[_example_key(example, target_kind)],
                device=policy_target.device,
                dtype=policy_target.dtype,
            )
            reference_opposite = torch.tensor(
                reference_cache.values[_example_key(example, opposite_kind)],
                device=policy_target.device,
                dtype=policy_target.dtype,
            )
            loss = bipo_loss(
                torch,
                policy_target.unsqueeze(0),
                policy_opposite.unsqueeze(0),
                reference_target.unsqueeze(0),
                reference_opposite.unsqueeze(0),
                direction_sign,
                beta=config.beta,
                reduction="mean",
            )
            (loss / config.gradient_accumulation_steps).backward()
            epoch_losses.append(float(loss.detach().item()))
            pending += 1
            is_last = order_index == len(permutation) - 1
            if pending == config.gradient_accumulation_steps or is_last:
                if is_last and pending < config.gradient_accumulation_steps:
                    correction = config.gradient_accumulation_steps / pending
                    if vector.grad is not None:
                        vector.grad.mul_(correction)
                pre_clip_norm = torch.nn.utils.clip_grad_norm_([vector], config.max_grad_norm)
                pre_clip_norms.append(float(pre_clip_norm.detach().item()))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                batch_direction_sign = None
                optimizer_step += 1

        raw = vector.detach().float().cpu()
        history.append(
            {
                "epoch": epoch + 1,
                "mean_loss": sum(epoch_losses) / len(epoch_losses),
                "effective_batch_sign_sequence": effective_batch_signs,
                "effective_batch_sign_counts": {
                    str(sign): effective_batch_signs.count(sign) for sign in (-1, 1)
                },
                "example_direction_sign_counts": {
                    str(k): v for k, v in example_direction_counts.items()
                },
                "example_order_case_ids": [examples[index].case_id for index in permutation],
                "raw_vector_norm": float(raw.norm().item()),
                "optimizer_steps": optimizer_step,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "mean_pre_clip_gradient_norm": sum(pre_clip_norms) / len(pre_clip_norms),
                "max_pre_clip_gradient_norm": max(pre_clip_norms),
            }
        )
        if epoch + 1 in config.checkpoint_epochs:
            checkpoints[str(epoch + 1)] = raw.clone()

    raw_direction = vector.detach().float().cpu()
    direction = normalize_direction(torch, raw_direction)
    optimizer_state = _optimizer_state_record(optimizer, vector)
    return {
        "direction": direction,
        "raw_direction": raw_direction,
        "checkpoint_raw_directions": checkpoints,
        "layer": layer,
        "geometry": geometry,
        "training_config": asdict(config),
        "reference_implementation_adaptation": (
            "Frozen-model reference response log-probabilities were cached once instead of "
            "holding a duplicate reference model; this is mathematically identical because "
            "the reference policy never changes. Microbatch 1 with exact gradient accumulation "
            "implements the published effective batch size on CPU."
        ),
        "history": history,
        "optimizer_state": optimizer_state,
        "reference_cache_identity": reference_cache.identity,
        "reference_cache_values": reference_cache.values,
        "reference_cache_values_sha256": hashlib.sha256(
            json.dumps(
                reference_cache.values,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }
