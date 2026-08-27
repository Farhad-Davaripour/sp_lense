from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import ClassVar

import pytest

torch = pytest.importorskip("torch")

from sp_lense.semantic_completion_gradient import (
    authored_completion_mean_logprob,
    capture_authored_completion_mean_logprob_gradient,
    capture_prompt_final_residual,
    capture_semantic_completion_gradient,
    encode_authored_completion,
    validate_causal_prompt_residuals,
)


class _JointTokenizer:
    chat_template = "semantic-completion-gradient-test-template"
    eos_token_id = None

    _joint_content: ClassVar[dict[str, list[int]]] = {
        "": [],
        "A": [0],
        "B": [1],
        "preserve": [2, 3],
        "comply": [4, 5, 6],
        "same-one": [7],
        "same-two": [7],
    }

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        # The authored completions intentionally differ from their joint-chat encoding.
        # A correct boundary helper must never use these IDs for completion content.
        return {
            "A": [0],
            "B": [1],
            "preserve": [15],
            "comply": [14],
            "same-one": [7],
            "same-two": [7],
        }[text]

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
        return_dict,
        return_tensors,
    ):
        assert tokenize and not enable_thinking and return_dict and return_tensors == "pt"
        prefix = [8, 9]
        if add_generation_prompt:
            assert len(messages) == 1
            values = prefix
        else:
            content = messages[-1]["content"]
            values = prefix + list(self._joint_content[content]) + [12, 13]
        return {"input_ids": torch.tensor([values], dtype=torch.long)}

    def decode(self, token_ids, **kwargs):
        del kwargs
        labels = {0: "A", 1: "B"}
        return "".join(labels.get(int(token_id), "") for token_id in token_ids)


class _BrokenPrefixTokenizer(_JointTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        encoded = super().apply_chat_template(messages, **kwargs)
        if not kwargs["add_generation_prompt"] and messages[-1]["content"] == "preserve":
            encoded["input_ids"][0, 0] = 11
        return encoded


class _MissingTerminatorTokenizer(_JointTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        encoded = super().apply_chat_template(messages, **kwargs)
        if not kwargs["add_generation_prompt"] and messages[-1]["content"] == "preserve":
            encoded["input_ids"][0, -1] = 11
        return encoded


class _NoContentTokenizer(_JointTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        encoded = super().apply_chat_template(messages, **kwargs)
        if not kwargs["add_generation_prompt"] and messages[-1]["content"] == "preserve":
            encoded["input_ids"] = torch.tensor([[8, 9, 12, 13]], dtype=torch.long)
        return encoded


class _ToyCausalModel(torch.nn.Module):
    def __init__(
        self,
        tokenizer: _JointTokenizer | None = None,
        *,
        noncausal_length_offset: float = 0.0,
        fire_hook_twice: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = _JointTokenizer() if tokenizer is None else tokenizer
        self.embedding = torch.nn.Embedding(16, 3)
        self.unembed = torch.nn.Linear(3, 16, bias=False)
        with torch.no_grad():
            embedding = torch.arange(48, dtype=torch.float32).reshape(16, 3) / 17.0
            unembedding = torch.arange(48, dtype=torch.float32).reshape(16, 3)
            unembedding = torch.sin(unembedding / 5.0)
            self.embedding.weight.copy_(embedding)
            self.unembed.weight.copy_(unembedding)
        self.noncausal_length_offset = float(noncausal_length_offset)
        self.fire_hook_twice = fire_hook_twice
        self._active_hooks = []

    @contextmanager
    def hooks(self, fwd_hooks):
        previous = self._active_hooks
        self._active_hooks = list(fwd_hooks)
        try:
            yield
        finally:
            self._active_hooks = previous

    def forward(self, tokens):
        activation = self.embedding(tokens)
        if self.noncausal_length_offset:
            offset = torch.tensor(
                [self.noncausal_length_offset, 0.0, 0.0],
                device=activation.device,
                dtype=activation.dtype,
            )
            activation = activation + int(tokens.shape[1]) * offset
        for _, hook in self._active_hooks:
            activation = hook(activation, hook=None)
            if self.fire_hook_twice:
                activation = hook(activation, hook=None)
        # Post-hook causal mixing ensures later authored tokens depend on the final
        # prompt residual, while no prompt position depends on a future token.
        causal_state = activation.cumsum(dim=1)
        return self.unembed(causal_state)


def _backend(
    tokenizer: _JointTokenizer | None = None,
    *,
    noncausal_length_offset: float = 0.0,
    fire_hook_twice: bool = False,
):
    model = _ToyCausalModel(
        tokenizer,
        noncausal_length_offset=noncausal_length_offset,
        fire_hook_twice=fire_hook_twice,
    )
    return SimpleNamespace(
        torch=torch,
        model=model,
        device="cpu",
        config=SimpleNamespace(model=SimpleNamespace(prompt_format="chat")),
        encode=lambda prompt: model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )["input_ids"],
    )


def test_joint_boundary_uses_true_content_tokens_and_excludes_assistant_end() -> None:
    backend = _backend()

    encoding = encode_authored_completion(backend, "prompt", "preserve")

    assert backend.model.tokenizer.encode("preserve") == [15]
    assert encoding.content_token_ids == (2, 3)
    assert encoding.assistant_end_token_ids == (12, 13)
    assert encoding.prompt_length == 2
    assert encoding.prompt_final_index == 1
    assert encoding.content_mask.tolist() == [[False, False, True, True, False, False]]
    assert encoding.audit_record()["assistant_end_excluded_from_objective"] is True


def test_content_mean_logprob_is_invariant_to_terminator_prediction_positions() -> None:
    backend = _backend()
    encoding = encode_authored_completion(backend, "prompt", "preserve")
    generator = torch.Generator().manual_seed(41)
    values = torch.randn(1, 6, 16, generator=generator)
    changed = values.clone()
    # Content tokens at positions 2 and 3 are predicted by logits 1 and 2. Logits
    # from position 3 onward predict template EOM tokens or lie beyond the objective.
    changed[:, 3:, :] = torch.randn(1, 3, 16, generator=generator) * 100.0

    baseline_logits = values.requires_grad_(True)
    baseline = authored_completion_mean_logprob(torch, baseline_logits, encoding)
    altered = authored_completion_mean_logprob(torch, changed, encoding)
    baseline.backward()

    assert altered.item() == pytest.approx(baseline.item())
    assert bool((baseline_logits.grad[:, 1:3].abs() > 0).any().item())
    assert torch.equal(baseline_logits.grad[:, 3:], torch.zeros_like(baseline_logits.grad[:, 3:]))


@pytest.mark.parametrize(
    ("tokenizer", "message"),
    [
        (_BrokenPrefixTokenizer(), "does not preserve the generation prefix"),
        (_MissingTerminatorTokenizer(), "does not end in the verified assistant EOM"),
        (_NoContentTokenizer(), "authored completion"),
    ],
)
def test_joint_boundary_failures_are_closed(tokenizer, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        encode_authored_completion(_backend(tokenizer), "prompt", "preserve")


@pytest.mark.parametrize(("prompt", "completion"), [("", "preserve"), ("prompt", "  ")])
def test_blank_prompt_or_completion_is_rejected(prompt: str, completion: str) -> None:
    with pytest.raises(ValueError, match="non-empty|non-whitespace"):
        encode_authored_completion(_backend(), prompt, completion)


def test_gradient_is_taken_at_prompt_final_position_and_allocates_no_parameter_grads() -> None:
    backend = _backend()

    capture = capture_authored_completion_mean_logprob_gradient(
        backend,
        "prompt",
        "preserve",
        layer=10,
    )

    assert capture.raw_gradient.shape == (3,)
    # The final full-sequence position predicts no selected content token and would
    # have zero gradient. A nonzero result therefore detects an accidental [0, -1].
    assert capture.raw_gradient.norm().item() > 0.0
    assert capture.audit["gradient_position"] == "final_prompt_token"
    assert capture.audit["prompt_final_index"] == 1
    assert capture.audit["hook_call_count"] == 1
    assert capture.audit["assistant_end_excluded_from_objective"] is True
    assert capture.audit["model_parameter_gradients_allocated"] is False
    assert all(parameter.grad is None for parameter in backend.model.parameters())


def test_semantic_gradient_uses_common_prompt_norm_and_swapping_completions_negates_it() -> None:
    backend = _backend()

    forward = capture_semantic_completion_gradient(
        backend,
        "prompt",
        "preserve",
        "comply",
        layer=10,
    )
    reverse = capture_semantic_completion_gradient(
        backend,
        "prompt",
        "comply",
        "preserve",
        layer=10,
    )

    prompt_norm = forward.prompt_residual.norm()
    expected = prompt_norm * (forward.preserve.raw_gradient - forward.comply.raw_gradient)
    assert torch.equal(forward.effective_gradient, expected)
    assert forward.audit["common_prompt_residual_norm_computation_dtype"] == "torch.float32"
    assert torch.allclose(forward.effective_gradient, -reverse.effective_gradient, atol=1e-6)
    assert forward.audit["semantic_objective_value"] == pytest.approx(
        -reverse.audit["semantic_objective_value"]
    )
    assert forward.causal_residual_audit["maximum_causal_residual_relative_l2"] == 0.0
    assert forward.audit["assistant_end_excluded_from_both_objectives"] is True
    assert all(parameter.grad is None for parameter in backend.model.parameters())


def test_metering_callbacks_fire_before_each_forward_and_backward() -> None:
    backend = _backend()
    events = []

    capture_semantic_completion_gradient(
        backend,
        "prompt",
        "preserve",
        "comply",
        layer=10,
        before_forward=lambda work_id: events.append(work_id),
        before_backward=lambda work_id: events.append(work_id),
    )

    assert events == [
        "prompt_only_forward",
        "preserve_forward",
        "preserve_backward",
        "comply_forward",
        "comply_backward",
    ]

    model_calls = []
    original_forward = backend.model.forward

    def counted_forward(tokens):
        model_calls.append(1)
        return original_forward(tokens)

    backend.model.forward = counted_forward
    with pytest.raises(RuntimeError, match="meter stop"):
        capture_prompt_final_residual(
            backend,
            "prompt",
            layer=10,
            before_forward=lambda _work_id: (_ for _ in ()).throw(RuntimeError("meter stop")),
        )
    assert model_calls == []


def test_identical_joint_content_tokens_are_rejected_even_if_text_differs() -> None:
    with pytest.raises(ValueError, match="identical joint content tokens"):
        capture_semantic_completion_gradient(
            _backend(),
            "prompt",
            "same-one",
            "same-two",
            layer=10,
        )


def test_causal_residual_audit_passes_below_and_fails_above_tolerance() -> None:
    prompt = torch.tensor([1.0, 0.0], dtype=torch.float64)
    below = torch.tensor([1.0 + 4e-6, 0.0], dtype=torch.float64)

    audit = validate_causal_prompt_residuals(
        torch,
        prompt,
        below,
        prompt,
        tolerance=1e-5,
    )

    assert audit["causal_prompt_residuals_within_tolerance"] is True
    assert audit["maximum_causal_residual_relative_l2"] < 1e-5
    with pytest.raises(RuntimeError, match="beyond tolerance"):
        validate_causal_prompt_residuals(
            torch,
            prompt,
            torch.tensor([1.0 + 2e-5, 0.0], dtype=torch.float64),
            prompt,
            tolerance=1e-5,
        )


@pytest.mark.parametrize(
    "bad",
    [
        torch.tensor([float("nan"), 0.0]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([1.0, 0.0, 0.0]),
    ],
)
def test_causal_residual_audit_rejects_nonfinite_or_mismatched_vectors(bad) -> None:
    prompt = torch.tensor([1.0, 0.0])
    with pytest.raises((ValueError, TypeError)):
        validate_causal_prompt_residuals(torch, prompt, bad, prompt)


def test_noncausal_sequence_length_effect_fails_semantic_capture() -> None:
    backend = _backend(noncausal_length_offset=0.25)

    with pytest.raises(RuntimeError, match="causal prompt residual beyond tolerance"):
        capture_semantic_completion_gradient(
            backend,
            "prompt",
            "preserve",
            "comply",
            layer=10,
        )


def test_hooks_must_fire_exactly_once() -> None:
    backend = _backend(fire_hook_twice=True)

    with pytest.raises(RuntimeError, match="hook fired more than once"):
        capture_prompt_final_residual(backend, "prompt", layer=10)


@pytest.mark.parametrize("layer", [-1, True, 1.5])
def test_invalid_layer_is_rejected(layer) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        capture_prompt_final_residual(_backend(), "prompt", layer=layer)


def test_completion_scorer_rejects_wrong_shape_and_nonfinite_logits() -> None:
    backend = _backend()
    encoding = encode_authored_completion(backend, "prompt", "preserve")
    with pytest.raises(ValueError, match="shape"):
        authored_completion_mean_logprob(torch, torch.zeros(6, 16), encoding)
    logits = torch.zeros(1, 6, 16)
    logits[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        authored_completion_mean_logprob(torch, logits, encoding)
