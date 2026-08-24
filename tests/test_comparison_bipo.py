from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from sp_lense.comparison_bipo import (
    BiPOTrainingConfig,
    BiPOTrainingExample,
    _lr_multiplier,
    _reference_logprob,
    _trainable_vector_hook,
    build_reference_logprob_cache,
    differentiable_completion_logprob,
    train_bipo_direction,
    validation_preference_loss,
)
from sp_lense.steering_methods import bipo_loss


def test_bipo_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="beta"):
        BiPOTrainingConfig(beta=0).validate()
    with pytest.raises(ValueError, match="epochs"):
        BiPOTrainingConfig(epochs=0).validate()
    with pytest.raises(ValueError, match="checkpoint_epochs"):
        BiPOTrainingConfig(epochs=5, checkpoint_epochs=(5, 10)).validate()
    with pytest.raises(ValueError, match="max_grad_norm"):
        BiPOTrainingConfig(max_grad_norm=0).validate()


def test_matched_trainable_hook_changes_only_absolute_prompt_position() -> None:
    activation = torch.zeros(1, 5, 3)
    vector = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    hook = _trainable_vector_hook(
        torch,
        vector,
        direction_sign=-1,
        geometry="matched_final_prompt",
        prompt_length=3,
    )
    changed = hook(activation, None)
    assert changed[0, 2].tolist() == [-1.0, -2.0, -3.0]
    assert changed[0, 4].tolist() == [0.0, 0.0, 0.0]
    changed.sum().backward()
    assert vector.grad is not None


def test_canonical_bipo_hook_broadcasts_all_tokens() -> None:
    activation = torch.zeros(1, 2, 2)
    vector = torch.tensor([1.0, -1.0])
    hook = _trainable_vector_hook(
        torch,
        vector,
        direction_sign=1,
        geometry="canonical_broadcast",
        prompt_length=1,
    )
    assert hook(activation, None).tolist() == [[[1.0, -1.0], [1.0, -1.0]]]


def test_cosine_schedule_has_warmup_and_reaches_zero() -> None:
    assert _lr_multiplier(0, 200, 100) == pytest.approx(0.0)
    assert _lr_multiplier(99, 200, 100) == pytest.approx(0.99)
    assert _lr_multiplier(100, 200, 100) == pytest.approx(1.0)
    assert _lr_multiplier(200, 200, 100) == pytest.approx(0.0)


def test_negative_bipo_sign_reverses_fixed_preference_gap() -> None:
    preferred = torch.tensor([1.0], requires_grad=True)
    rejected = torch.tensor([0.0], requires_grad=True)
    zero = torch.tensor([0.0])
    positive_loss = bipo_loss(torch, preferred, rejected, zero, zero, 1, beta=1.0)
    negative_loss = bipo_loss(torch, preferred, rejected, zero, zero, -1, beta=1.0)
    assert positive_loss.item() < negative_loss.item()


class _TinyTokenizer:
    chat_template = None
    eos_token_id = 1

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_tensors: str | None = None,
    ):
        del add_special_tokens
        ids = [2 + (ord(character) % 13) for character in text]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids


class _TinyHookedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(71)
        self.embedding = torch.nn.Embedding(16, 4)
        self.unembed = torch.nn.Linear(4, 16, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.randn(16, 4, generator=generator))
            self.unembed.weight.copy_(torch.randn(16, 4, generator=generator))
        self.tokenizer = _TinyTokenizer()
        self.cfg = SimpleNamespace(n_layers=1, d_model=4)
        self._hooks: list[tuple[str, object]] = []

    @contextmanager
    def hooks(self, *, fwd_hooks):
        previous = self._hooks
        self._hooks = list(fwd_hooks)
        try:
            yield self
        finally:
            self._hooks = previous

    def forward(self, tokens):
        activation = self.embedding(tokens)
        for name, hook in self._hooks:
            if name == "blocks.0.hook_out":
                activation = hook(activation, None)
        return self.unembed(activation)


class _TinyBackend:
    def __init__(self) -> None:
        self.torch = torch
        self.device = torch.device("cpu")
        self.model = _TinyHookedModel()
        self.config = SimpleNamespace(
            model=SimpleNamespace(
                id="tiny-bipo-test",
                revision="1" * 40,
                prompt_format="plain",
            )
        )

    def encode(self, prompt: str):
        return torch.tensor(
            [self.model.tokenizer.encode(prompt, add_special_tokens=True)],
            dtype=torch.long,
        )


def _tiny_examples(count: int = 4) -> list[BiPOTrainingExample]:
    return [
        BiPOTrainingExample(
            case_id=f"case-{index}",
            prompt=f"prompt {index}:",
            preserve_completion="A",
            comply_completion="B",
        )
        for index in range(count)
    ]


def test_cached_reference_loss_equals_live_frozen_reference_loss() -> None:
    backend = _TinyBackend()
    example = _tiny_examples(1)[0]
    cache = build_reference_logprob_cache(backend, [example])
    vector = torch.tensor([0.1, -0.2, 0.05, 0.3])
    cached_loss = validation_preference_loss(
        backend,
        [example],
        vector,
        layer=0,
        geometry="matched_final_prompt",
        beta=0.1,
        reference_cache=cache,
    )

    live_reference_preserve = _reference_logprob(
        backend, example.prompt, example.preserve_completion
    )
    live_reference_comply = _reference_logprob(
        backend, example.prompt, example.comply_completion
    )
    live_losses = []
    for sign in (-1, 1):
        policy_preserve = differentiable_completion_logprob(
            backend,
            example.prompt,
            example.preserve_completion,
            vector,
            layer=0,
            direction_sign=sign,
            geometry="matched_final_prompt",
        ).reshape(1)
        policy_comply = differentiable_completion_logprob(
            backend,
            example.prompt,
            example.comply_completion,
            vector,
            layer=0,
            direction_sign=sign,
            geometry="matched_final_prompt",
        ).reshape(1)
        live_losses.append(
            bipo_loss(
                torch,
                policy_preserve,
                policy_comply,
                torch.tensor([live_reference_preserve]),
                torch.tensor([live_reference_comply]),
                sign,
                beta=0.1,
            ).item()
        )
    assert sorted(cache.values.values()) == pytest.approx(
        sorted([live_reference_preserve, live_reference_comply]), abs=0
    )
    assert cached_loss == pytest.approx(sum(live_losses) / len(live_losses), abs=1e-8)


def test_real_bipo_step_freezes_model_updates_vector_and_holds_batch_sign() -> None:
    backend = _TinyBackend()
    before = {
        name: parameter.detach().clone()
        for name, parameter in backend.model.named_parameters()
    }
    result = train_bipo_direction(
        backend,
        _tiny_examples(4),
        layer=0,
        geometry="matched_final_prompt",
        config=BiPOTrainingConfig(
            learning_rate=0.01,
            weight_decay=0.0,
            epochs=1,
            checkpoint_epochs=(1,),
            gradient_accumulation_steps=4,
            warmup_steps=0,
            seed=11,
        ),
    )

    assert not torch.equal(result["raw_direction"], torch.zeros(4))
    for name, parameter in backend.model.named_parameters():
        assert torch.equal(parameter.detach(), before[name])
        assert parameter.requires_grad is False
        assert parameter.grad is None
    history = result["history"][0]
    assert len(history["effective_batch_sign_sequence"]) == 1
    sampled_sign = str(history["effective_batch_sign_sequence"][0])
    opposite_sign = "-1" if sampled_sign == "1" else "1"
    assert history["example_direction_sign_counts"][sampled_sign] == 4
    assert history["example_direction_sign_counts"][opposite_sign] == 0
    assert result["optimizer_state"]["state"]["exp_avg"]["shape"] == [4]
