from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from sp_lense.causal_anchor_runtime import capture_multilayer_choice_anchor_gradient


class _Tokenizer:
    chat_template = "choice-anchor-capture-test"
    eos_token_id = None

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return {"A": [0], "B": [1]}[text]

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
        values = [2, 3]
        if not add_generation_prompt:
            content = messages[-1]["content"]
            if content:
                values.append(0 if content == "A" else 1)
            values.append(4)
        return {"input_ids": torch.tensor([values], dtype=torch.long)}

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join({0: "A", 1: "B"}.get(int(value), "") for value in token_ids)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = _Tokenizer()
        self.embedding = torch.nn.Embedding(5, 3)
        self.unembed = torch.nn.Linear(3, 5, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(15).reshape(5, 3) / 9.0)
            self.unembed.weight.copy_(torch.sin(torch.arange(15).reshape(5, 3)))
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
        for _, hook in self._active_hooks:
            activation = hook(activation, hook=None)
        return self.unembed(activation.cumsum(dim=1))


def _backend():
    model = _Model()
    return SimpleNamespace(
        torch=torch,
        model=model,
        device="cpu",
        config=SimpleNamespace(model=SimpleNamespace(prompt_format="chat")),
        encode=lambda prompt: torch.tensor([[2, 3]], dtype=torch.long),
    )


def test_choice_anchor_capture_is_finite_audited_and_orientation_symmetric() -> None:
    backend = _backend()
    work = []
    first = capture_multilayer_choice_anchor_gradient(
        backend,
        "prompt",
        "A",
        "B",
        layers=(0,),
        anchor_index=0,
        before_forward=lambda label: work.append(label),
        before_backward=lambda label: work.append(label),
    )
    reverse = capture_multilayer_choice_anchor_gradient(
        backend,
        "prompt",
        "B",
        "A",
        layers=(0,),
        anchor_index=0,
    )
    assert first.raw_gradients.shape == (1, 3)
    assert first.anchor_residuals.shape == (1, 3)
    assert torch.isfinite(first.raw_gradients).all()
    assert torch.allclose(reverse.raw_gradients, -first.raw_gradients)
    assert reverse.preserve_log_odds == pytest.approx(-first.preserve_log_odds)
    assert first.audit["model_parameter_gradients_allocated"] is False
    assert first.audit["hook_call_counts"] == {"0": 1}
    assert work == ["choice_forward", "choice_backward"]


def test_choice_anchor_capture_rejects_wrong_labels_and_anchor() -> None:
    backend = _backend()
    with pytest.raises(ValueError, match="exactly A and B"):
        capture_multilayer_choice_anchor_gradient(
            backend, "prompt", "X", "Y", layers=(0,), anchor_index=0
        )
    with pytest.raises(ValueError, match="outside"):
        capture_multilayer_choice_anchor_gradient(
            backend, "prompt", "A", "B", layers=(0,), anchor_index=3
        )
