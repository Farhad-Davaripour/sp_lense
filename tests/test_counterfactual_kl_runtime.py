from __future__ import annotations

from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from types import MappingProxyType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from sp_lense.comparison_runtime import resolve_choice_boundary
from sp_lense.counterfactual_kl_runtime import (
    capture_counterfactual_kl_baseline,
    capture_counterfactual_kl_lookahead,
)
from sp_lense.decision_margin_shield_finite import full_vocabulary_kl_float64
from sp_lense.factorial_causal_anchor import (
    canonical_sha256,
    tensor_float32_sha256,
    text_sha256,
)


class _Tokenizer:
    chat_template = "counterfactual-kl-runtime-mock-template"
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
        prefix = [2, 3]
        if add_generation_prompt:
            values = prefix
        else:
            content = messages[-1]["content"]
            values = prefix + ({"": [], "A": [0], "B": [1]}[content]) + [4]
        return {"input_ids": torch.tensor([values], dtype=torch.long)}

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join({0: "A", 1: "B"}.get(int(value), "") for value in token_ids)


class _NonlinearHookModel(torch.nn.Module):
    def __init__(self, *, hook_mode: str = "normal") -> None:
        super().__init__()
        self.tokenizer = _Tokenizer()
        self.cfg = SimpleNamespace(n_layers=2, d_model=3)
        self.embedding = torch.nn.Embedding(6, 3)
        self.unembed = torch.nn.Linear(3, 6, bias=False)
        self.hook_mode = hook_mode
        self._active_hooks = []
        self.forward_calls = 0
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.tensor(
                    [
                        [0.2, -0.1, 0.3],
                        [-0.3, 0.4, 0.1],
                        [0.5, -0.2, 0.1],
                        [0.1, 0.4, -0.3],
                        [0.0, 0.0, 0.0],
                        [0.2, 0.2, 0.2],
                    ]
                )
            )
            self.unembed.weight.copy_(
                torch.tensor(
                    [
                        [1.0, -0.5, 0.25],
                        [-0.25, 0.75, -0.5],
                        [2.0, 2.0, 2.0],
                        [-1.5, 0.1, 0.2],
                        [0.0, -0.5, 0.3],
                        [0.1, 0.1, -0.1],
                    ]
                )
            )

    @contextmanager
    def hooks(self, fwd_hooks):
        previous = self._active_hooks
        self._active_hooks = list(fwd_hooks)
        try:
            yield
        finally:
            self._active_hooks = previous

    def forward(self, tokens):
        self.forward_calls += 1
        activation = self.embedding(tokens)
        for name, hook in self._active_hooks:
            if name != "blocks.0.hook_out" or self.hook_mode == "missing":
                continue
            activation = hook(activation, hook=None)
            if self.hook_mode == "double":
                activation = hook(activation, hook=None)
        hidden = torch.tanh(activation.cumsum(dim=1))
        return self.unembed(hidden)


def _backend(*, hook_mode: str = "normal"):
    model = _NonlinearHookModel(hook_mode=hook_mode)
    return SimpleNamespace(
        torch=torch,
        model=model,
        device="cpu",
        config=SimpleNamespace(model=SimpleNamespace(prompt_format="chat")),
        encode=lambda prompt: torch.tensor([[2, 3]], dtype=torch.long),
    )


def _pre_anchor(backend, anchor_index: int = 0):
    return (
        backend.model.embedding(backend.encode("prompt"))[0, anchor_index]
        .detach()
        .float()
        .contiguous()
    )


def _pinned_context(backend) -> dict[str, str]:
    boundary = resolve_choice_boundary(backend, "prompt")
    return {
        "expected_prompt_sha256": text_sha256("prompt"),
        "expected_choice_boundary_evidence_sha256": boundary.evidence_sha256,
        "expected_prompt_token_ids_sha256": boundary.prompt_prefix_token_ids_sha256,
        "expected_pre_anchor_residual_float32_sha256": tensor_float32_sha256(
            _pre_anchor(backend)
        ),
    }


def _baseline(backend):
    return capture_counterfactual_kl_baseline(
        backend,
        "prompt",
        "A",
        "B",
        positive_semantic="preserve",
        negative_semantic="comply",
        layer=0,
        anchor_index=0,
        **_pinned_context(backend),
    )


def test_first_baseline_capture_can_commit_its_own_pre_anchor_hash() -> None:
    backend = _backend()
    context = _pinned_context(backend)
    context.pop("expected_pre_anchor_residual_float32_sha256")
    capture = capture_counterfactual_kl_baseline(
        backend,
        "prompt",
        "A",
        "B",
        positive_semantic="preserve",
        negative_semantic="comply",
        layer=0,
        anchor_index=0,
        **context,
    )
    assert capture.audit["pre_anchor_residual_hash_precommitted"] is False
    assert capture.audit["pre_anchor_residual_float32_sha256"] == tensor_float32_sha256(
        capture.pre_anchor_residual
    )


def _lookahead(
    backend,
    baseline,
    direction,
    *,
    branch_sign: int = 1,
    scale: float = 1.3,
    signed_delta=None,
    expected_delta_hash: str | None = None,
    expected_baseline_hash: str | None = None,
):
    canonical_direction = direction.detach().cpu().double().contiguous().clone()
    canonical_direction[canonical_direction == 0.0] = 0.0
    unsigned = (canonical_direction * scale).float().contiguous()
    delta = (
        (unsigned if branch_sign == 1 else -unsigned).contiguous()
        if signed_delta is None
        else signed_delta
    )
    return capture_counterfactual_kl_lookahead(
        backend,
        "prompt",
        layer=0,
        anchor_index=0,
        branch_sign=branch_sign,
        lookahead_standardized_direction=canonical_direction,
        physical_residual_scale=scale,
        signed_delta=delta,
        baseline_full_logits=baseline.full_logits,
        expected_lookahead_standardized_direction_sha256=canonical_sha256(
            canonical_direction.tolist()
        ),
        expected_signed_delta_float32_sha256=(
            tensor_float32_sha256(delta)
            if expected_delta_hash is None
            else expected_delta_hash
        ),
        expected_baseline_full_logits_float32_sha256=(
            tensor_float32_sha256(baseline.full_logits)
            if expected_baseline_hash is None
            else expected_baseline_hash
        ),
        **_pinned_context(backend),
    )


def _manual_logits(backend, physical_delta):
    tokens = backend.encode("prompt")

    def add_delta(activation, hook):
        del hook
        changed = activation.detach().clone()
        changed[0, 0] += physical_delta
        return changed

    with torch.no_grad(), backend.model.hooks(
        fwd_hooks=[("blocks.0.hook_out", add_delta)]
    ):
        return backend.model(tokens)[0, -1].float().detach().cpu()


def test_baseline_is_one_pass_hash_audited_and_matches_finite_difference() -> None:
    backend = _backend()
    observed = _baseline(backend)

    assert backend.model.forward_calls == 1
    assert observed.audit["model_forward_evaluations"] == 1
    assert observed.audit["model_backward_evaluations"] == 1
    assert observed.audit["zero_direction"] is True
    assert observed.audit["model_parameter_gradients_allocated"] is False
    assert observed.full_logits.dtype == torch.float32
    assert observed.full_logits.requires_grad is False
    assert observed.raw_anchor_gradient.requires_grad is False
    assert observed.pre_anchor_residual.equal(_pre_anchor(backend))
    assert observed.unrestricted_semantic_choice in {"preserve", "comply", "OTHER"}
    assert observed.pair_semantic_choice in {"preserve", "comply"}
    assert observed.audit["full_logits_float32_sha256"] == tensor_float32_sha256(
        observed.full_logits
    )
    assert isinstance(observed.audit, MappingProxyType)
    with pytest.raises(TypeError):
        observed.audit["changed"] = True
    with pytest.raises(FrozenInstanceError):
        observed.layer = 1
    assert all(parameter.grad is None for parameter in backend.model.parameters())

    epsilon = 1e-3
    finite_difference = []
    for coordinate in range(backend.model.cfg.d_model):
        offset = torch.zeros(backend.model.cfg.d_model)
        offset[coordinate] = epsilon
        higher = _manual_logits(backend, offset)
        lower = _manual_logits(backend, -offset)
        finite_difference.append(
            float((higher[0] - higher[1] - lower[0] + lower[1]).item())
            / (2.0 * epsilon)
        )
    assert observed.raw_anchor_gradient.tolist() == pytest.approx(
        finite_difference, rel=2e-3, abs=2e-3
    )


def test_float64_kl_gradient_matches_shared_direction_finite_difference() -> None:
    backend = _backend()
    baseline = _baseline(backend)
    direction = torch.tensor([0.07, -0.03, 0.02], dtype=torch.float64)
    observed = _lookahead(backend, baseline, direction)

    assert observed.full_vocabulary_kl_changed_to_baseline == pytest.approx(
        full_vocabulary_kl_float64(torch, baseline.full_logits, observed.full_logits),
        abs=1e-12,
    )
    assert observed.audit["kl_softmax_dtype"] == "float64"
    assert observed.audit["kl_direction"] == "KL(changed||unsteered_baseline)"
    assert observed.audit["model_parameter_gradients_allocated"] is False
    assert all(parameter.grad is None for parameter in backend.model.parameters())

    epsilon = 5e-4
    finite_difference = []
    for coordinate in range(direction.numel()):
        offset = torch.zeros_like(direction)
        offset[coordinate] = epsilon
        higher = _lookahead(backend, baseline, direction + offset)
        lower = _lookahead(backend, baseline, direction - offset)
        finite_difference.append(
            (
                higher.full_vocabulary_kl_changed_to_baseline
                - lower.full_vocabulary_kl_changed_to_baseline
            )
            / (2.0 * epsilon)
        )
    assert observed.shared_standardized_kl_gradient.tolist() == pytest.approx(
        finite_difference, rel=4e-3, abs=4e-4
    )


def test_branch_sign_is_external_exact_and_applied_to_shared_gradient() -> None:
    backend = _backend()
    baseline = _baseline(backend)
    direction = torch.tensor([0.06, -0.025, 0.015], dtype=torch.float64)
    scale = 1.7
    plus = _lookahead(backend, baseline, direction, branch_sign=1, scale=scale)
    minus = _lookahead(backend, baseline, direction, branch_sign=-1, scale=scale)

    assert torch.equal(
        plus.shared_standardized_kl_gradient,
        scale * plus.raw_anchor_kl_gradient.double(),
    )
    assert torch.equal(
        minus.shared_standardized_kl_gradient,
        -scale * minus.raw_anchor_kl_gradient.double(),
    )
    assert plus.audit["external_branch_sign"] == 1
    assert minus.audit["external_branch_sign"] == -1
    assert plus.audit["runtime_selected_or_changed_sign"] is False
    assert minus.audit["runtime_selected_or_changed_sign"] is False
    assert torch.allclose(
        plus.realized_signed_delta, (direction * scale).float(), rtol=1e-6, atol=1e-7
    )
    assert torch.allclose(
        minus.realized_signed_delta, -(direction * scale).float(), rtol=1e-6, atol=1e-7
    )
    assert plus.audit["requested_signed_delta_float32_sha256"] == tensor_float32_sha256(
        (direction * scale).float()
    )
    assert minus.audit["requested_signed_delta_float32_sha256"] == tensor_float32_sha256(
        -(direction * scale).float()
    )

    calls_before = backend.model.forward_calls
    with pytest.raises(RuntimeError, match="inconsistent with branch sign"):
        _lookahead(
            backend,
            baseline,
            direction,
            branch_sign=-1,
            scale=scale,
            signed_delta=(direction * scale).float(),
        )
    assert backend.model.forward_calls == calls_before


def test_identity_mismatches_fail_closed_and_hooks_clear_parameter_gradients() -> None:
    backend = _backend()
    common = _pinned_context(backend)
    with pytest.raises(RuntimeError, match="prompt text differs"):
        capture_counterfactual_kl_baseline(
            backend,
            "prompt",
            "A",
            "B",
            positive_semantic="preserve",
            negative_semantic="comply",
            layer=0,
            anchor_index=0,
            **{**common, "expected_prompt_sha256": "0" * 64},
        )
    assert backend.model.forward_calls == 0

    baseline = _baseline(backend)
    direction = torch.tensor([0.05, -0.02, 0.01], dtype=torch.float64)
    calls_before = backend.model.forward_calls
    with pytest.raises(RuntimeError, match="baseline logits differ"):
        _lookahead(
            backend,
            baseline,
            direction,
            expected_baseline_hash="0" * 64,
        )
    assert backend.model.forward_calls == calls_before
    assert all(parameter.grad is None for parameter in backend.model.parameters())


@pytest.mark.parametrize(
    ("hook_mode", "message"),
    [("missing", "did not fire exactly once"), ("double", "fired more than once")],
)
def test_missing_or_repeated_hook_fails_closed(hook_mode: str, message: str) -> None:
    backend = _backend(hook_mode=hook_mode)
    with pytest.raises(RuntimeError, match=message):
        _baseline(backend)
    assert all(parameter.grad is None for parameter in backend.model.parameters())
