from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from sp_lense.closed_loop_dms_runtime import capture_closed_loop_dms_step
from sp_lense.comparison_runtime import resolve_choice_boundary
from sp_lense.factorial_causal_anchor import canonical_sha256, tensor_float32_sha256


class _Tokenizer:
    chat_template = "closed-loop-dms-mock-template"
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


class _BrokenPrefixTokenizer(_Tokenizer):
    def apply_chat_template(self, messages, **kwargs):
        encoded = super().apply_chat_template(messages, **kwargs)
        if not kwargs["add_generation_prompt"] and messages[-1]["content"] == "A":
            encoded["input_ids"][0, 0] = 5
        return encoded


class _NonlinearHookModel(torch.nn.Module):
    def __init__(
        self,
        *,
        hook_mode: str = "normal",
        activation_dtype: torch.dtype = torch.float32,
        tokenizer: _Tokenizer | None = None,
    ) -> None:
        super().__init__()
        self.tokenizer = _Tokenizer() if tokenizer is None else tokenizer
        self.cfg = SimpleNamespace(n_layers=2, d_model=3)
        self.embedding = torch.nn.Embedding(6, 3)
        self.unembed = torch.nn.Linear(3, 6, bias=False)
        self.hook_mode = hook_mode
        self.activation_dtype = activation_dtype
        self._active_hooks = []
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
        activation = self.embedding(tokens).to(self.activation_dtype)
        for name, hook in self._active_hooks:
            # This mock exposes only the first residual layer.
            if name != "blocks.0.hook_out" or self.hook_mode == "missing":
                continue
            activation = hook(activation, hook=None)
            if self.hook_mode == "double":
                activation = hook(activation, hook=None)
        hidden = torch.tanh(activation.float().cumsum(dim=1))
        return self.unembed(hidden)


def _backend(
    *,
    hook_mode: str = "normal",
    activation_dtype: torch.dtype = torch.float32,
    tokenizer: _Tokenizer | None = None,
):
    model = _NonlinearHookModel(
        hook_mode=hook_mode,
        activation_dtype=activation_dtype,
        tokenizer=tokenizer,
    )
    return SimpleNamespace(
        torch=torch,
        model=model,
        device="cpu",
        config=SimpleNamespace(model=SimpleNamespace(prompt_format="chat")),
        encode=lambda prompt: torch.tensor([[2, 3]], dtype=torch.long),
    )


def _pre_anchor(backend, anchor_index: int = 0):
    tokens = backend.encode("prompt")
    return (
        backend.model.embedding(tokens)
        .to(backend.model.activation_dtype)[0, anchor_index]
        .float()
        .contiguous()
    )


def _capture(
    backend,
    delta,
    *,
    positive_label: str = "A",
    negative_label: str = "B",
    positive_semantic: str = "preserve",
    negative_semantic: str = "comply",
    anchor_index: int = 0,
    expected_delta_hash: str | None = None,
    expected_pre_hash: str | None = None,
    branch_sign: int = 1,
    cumulative_direction=None,
    cumulative_direction_hash: str | None = None,
    physical_scale: float = 1.0,
    return_full_logits: bool = True,
    maximum_error: float = 1e-4,
):
    boundary = resolve_choice_boundary(backend, "prompt")
    standardized_direction = (
        (branch_sign * delta).double().div(physical_scale).contiguous()
        if cumulative_direction is None
        else cumulative_direction.detach().cpu().double().contiguous().clone()
    )
    standardized_direction[standardized_direction == 0.0] = 0.0
    return capture_closed_loop_dms_step(
        backend,
        "prompt",
        positive_label,
        negative_label,
        positive_semantic=positive_semantic,
        negative_semantic=negative_semantic,
        layer=0,
        anchor_index=anchor_index,
        branch_sign=branch_sign,
        cumulative_standardized_direction=standardized_direction,
        physical_residual_scale=physical_scale,
        signed_delta=delta,
        expected_signed_delta_float32_sha256=(
            tensor_float32_sha256(delta) if expected_delta_hash is None else expected_delta_hash
        ),
        expected_cumulative_standardized_direction_sha256=(
            canonical_sha256(standardized_direction.tolist())
            if cumulative_direction_hash is None
            else cumulative_direction_hash
        ),
        expected_choice_boundary_evidence_sha256=boundary.evidence_sha256,
        expected_prompt_token_ids_sha256=boundary.prompt_prefix_token_ids_sha256,
        expected_pre_anchor_residual_float32_sha256=(
            expected_pre_hash
            if expected_pre_hash is not None
            else (
                tensor_float32_sha256(_pre_anchor(backend, anchor_index))
                if anchor_index < int(backend.encode("prompt").shape[1])
                else "0" * 64
            )
        ),
        maximum_realized_relative_l2_error=maximum_error,
        return_full_logits=return_full_logits,
    )


def test_capture_applies_external_sign_and_returns_audited_gradient_and_logits() -> None:
    backend = _backend()
    delta = torch.tensor([0.08, -0.04, 0.03], dtype=torch.float32)

    plus = _capture(backend, delta)
    minus = _capture(backend, -delta, branch_sign=-1, return_full_logits=False)

    assert torch.allclose(plus.post_anchor_residual - plus.pre_anchor_residual, delta)
    assert torch.allclose(minus.post_anchor_residual - minus.pre_anchor_residual, -delta)
    assert torch.allclose(plus.realized_signed_delta, delta)
    assert torch.allclose(minus.realized_signed_delta, -delta)
    assert not torch.allclose(plus.raw_anchor_gradient, minus.raw_anchor_gradient)
    assert plus.full_logits is not None
    assert plus.full_logits.dtype == torch.float32
    assert minus.full_logits is None
    assert plus.unrestricted_predicted_label == "OTHER"
    assert plus.unrestricted_semantic_choice == "OTHER"
    assert plus.answer_format_valid is False
    assert plus.pair_choice_label in {"A", "B"}
    assert plus.audit["hook_call_count"] == 1
    assert plus.audit["untouched_positions_max_abs_delta"] == 0.0
    assert plus.audit["requested_minus_realized_relative_l2"] < 1e-6
    assert plus.audit["runtime_selected_or_changed_sign"] is False
    assert plus.audit["external_branch_sign"] == 1
    assert minus.audit["external_branch_sign"] == -1
    assert (
        plus.audit["expected_cumulative_standardized_direction_sha256"]
        == minus.audit["expected_cumulative_standardized_direction_sha256"]
    )
    assert plus.audit["model_parameter_gradients_allocated"] is False
    assert all(parameter.grad is None for parameter in backend.model.parameters())
    assert plus.audit["full_logits_float32_sha256"] == tensor_float32_sha256(plus.full_logits)
    audit_without_hash = {key: value for key, value in plus.audit.items() if key != "audit_sha256"}
    assert plus.audit["audit_sha256"] == canonical_sha256(audit_without_hash)


def test_raw_anchor_gradient_matches_centered_finite_difference() -> None:
    backend = _backend()
    delta = torch.tensor([0.06, -0.03, 0.02], dtype=torch.float32)
    observed = _capture(backend, delta, return_full_logits=False)
    epsilon = 1e-3
    finite_difference = []
    for coordinate in range(delta.numel()):
        offset = torch.zeros_like(delta)
        offset[coordinate] = epsilon
        higher = _capture(backend, delta + offset, return_full_logits=False)
        lower = _capture(backend, delta - offset, return_full_logits=False)
        finite_difference.append(
            (higher.positive_minus_negative_log_odds - lower.positive_minus_negative_log_odds)
            / (2.0 * epsilon)
        )
    assert observed.raw_anchor_gradient.tolist() == pytest.approx(
        finite_difference, rel=2e-3, abs=2e-3
    )


def test_reversing_ab_orientation_negates_margin_and_raw_gradient() -> None:
    backend = _backend()
    delta = torch.tensor([0.05, 0.02, -0.01], dtype=torch.float32)
    forward = _capture(backend, delta)
    reverse = _capture(
        backend,
        delta,
        positive_label="B",
        negative_label="A",
        positive_semantic="comply",
        negative_semantic="preserve",
    )

    assert reverse.positive_minus_negative_log_odds == pytest.approx(
        -forward.positive_minus_negative_log_odds
    )
    assert torch.allclose(reverse.raw_anchor_gradient, -forward.raw_anchor_gradient)
    assert reverse.pre_anchor_residual.equal(forward.pre_anchor_residual)
    assert reverse.post_anchor_residual.equal(forward.post_anchor_residual)


def test_external_sign_hash_mismatch_fails_before_forward() -> None:
    backend = _backend()
    delta = torch.tensor([0.08, -0.04, 0.03], dtype=torch.float32)
    with pytest.raises(RuntimeError, match="possible external sign error"):
        _capture(
            backend,
            -delta,
            branch_sign=-1,
            expected_delta_hash=tensor_float32_sha256(delta),
        )
    assert all(parameter.grad is None for parameter in backend.model.parameters())


def test_branch_sign_must_match_cumulative_direction_and_physical_delta() -> None:
    backend = _backend()
    direction = torch.tensor([0.08, -0.04, 0.03], dtype=torch.float64)
    signed_delta = direction.float()
    with pytest.raises(RuntimeError, match="inconsistent with branch sign"):
        _capture(
            backend,
            signed_delta,
            branch_sign=-1,
            cumulative_direction=direction,
        )
    assert all(parameter.grad is None for parameter in backend.model.parameters())


def test_non_unit_residual_scale_is_reconstructed_before_forward() -> None:
    backend = _backend()
    direction = torch.tensor([0.04, -0.02, 0.015], dtype=torch.float64)
    scale = 2.0
    signed_delta = (direction * scale).float()
    observed = _capture(
        backend,
        signed_delta,
        cumulative_direction=direction,
        physical_scale=scale,
    )
    assert observed.audit["physical_residual_scale"] == scale
    assert observed.audit["signed_delta_matches_declared_branch_direction_and_scale"] is True
    assert observed.audit["reconstructed_signed_delta_float32_sha256"] == (
        tensor_float32_sha256(signed_delta)
    )


def test_branch_reconstruction_is_bitwise_strict_about_signed_zero() -> None:
    backend = _backend()
    direction = torch.tensor([0.0, 0.04, -0.02], dtype=torch.float64)
    wrong_signed_zero_delta = torch.tensor([0.0, -0.04, 0.02], dtype=torch.float32)
    exact_negative_delta = -direction.float()
    assert tensor_float32_sha256(wrong_signed_zero_delta) != tensor_float32_sha256(
        exact_negative_delta
    )
    with pytest.raises(RuntimeError, match="inconsistent with branch sign"):
        _capture(
            backend,
            wrong_signed_zero_delta,
            branch_sign=-1,
            cumulative_direction=direction,
        )


@pytest.mark.parametrize(
    ("delta", "message"),
    [
        (torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64), "physical float32"),
        (torch.tensor([0.1, float("nan"), 0.3], dtype=torch.float32), "only finite"),
        (torch.zeros(3, dtype=torch.float32), "non-zero"),
        (torch.ones((1, 3), dtype=torch.float32), "one-dimensional"),
    ],
)
def test_invalid_physical_delta_fails_closed(delta, message: str) -> None:
    backend = _backend()
    with pytest.raises((ValueError, RuntimeError), match=message):
        _capture(
            backend,
            delta,
            cumulative_direction=torch.ones(3, dtype=torch.float64),
        )


def test_anchor_layer_and_pre_residual_are_pinned() -> None:
    backend = _backend()
    delta = torch.tensor([0.04, -0.02, 0.01], dtype=torch.float32)
    with pytest.raises(ValueError, match="outside the encoded prompt"):
        _capture(backend, delta, anchor_index=2)
    with pytest.raises(RuntimeError, match="pre-anchor residual differs"):
        _capture(backend, delta, expected_pre_hash="0" * 64)
    boundary = resolve_choice_boundary(backend, "prompt")
    with pytest.raises(ValueError, match="outside the resident model"):
        capture_closed_loop_dms_step(
            backend,
            "prompt",
            "A",
            "B",
            positive_semantic="preserve",
            negative_semantic="comply",
            layer=2,
            anchor_index=0,
            branch_sign=1,
            cumulative_standardized_direction=delta.double(),
            physical_residual_scale=1.0,
            signed_delta=delta,
            expected_signed_delta_float32_sha256=tensor_float32_sha256(delta),
            expected_cumulative_standardized_direction_sha256=canonical_sha256(
                delta.double().tolist()
            ),
            expected_choice_boundary_evidence_sha256=boundary.evidence_sha256,
            expected_prompt_token_ids_sha256=boundary.prompt_prefix_token_ids_sha256,
            expected_pre_anchor_residual_float32_sha256=tensor_float32_sha256(_pre_anchor(backend)),
        )


@pytest.mark.parametrize(
    ("hook_mode", "message"),
    [("missing", "did not fire exactly once"), ("double", "fired more than once")],
)
def test_missing_or_repeated_hook_fails_closed_and_clears_gradients(
    hook_mode: str, message: str
) -> None:
    backend = _backend(hook_mode=hook_mode)
    delta = torch.tensor([0.04, -0.02, 0.01], dtype=torch.float32)
    with pytest.raises(RuntimeError, match=message):
        _capture(backend, delta)
    assert all(parameter.grad is None for parameter in backend.model.parameters())


def test_post_cast_realization_error_fails_closed() -> None:
    backend = _backend(activation_dtype=torch.float16)
    delta = torch.tensor([1e-5, -1e-5, 1e-5], dtype=torch.float32)
    with pytest.raises(
        RuntimeError, match="realized (anchor delta differs|delta does not preserve)"
    ):
        _capture(backend, delta)
    assert all(parameter.grad is None for parameter in backend.model.parameters())


def test_choice_boundary_and_label_errors_fail_closed() -> None:
    delta = torch.tensor([0.04, -0.02, 0.01], dtype=torch.float32)
    broken = _backend(tokenizer=_BrokenPrefixTokenizer())
    with pytest.raises(ValueError, match="does not preserve generation prefix"):
        _capture(broken, delta)

    backend = _backend()
    boundary = resolve_choice_boundary(backend, "prompt")
    common = {
        "positive_semantic": "preserve",
        "negative_semantic": "comply",
        "layer": 0,
        "anchor_index": 0,
        "branch_sign": 1,
        "cumulative_standardized_direction": delta.double(),
        "physical_residual_scale": 1.0,
        "signed_delta": delta,
        "expected_signed_delta_float32_sha256": tensor_float32_sha256(delta),
        "expected_cumulative_standardized_direction_sha256": canonical_sha256(
            delta.double().tolist()
        ),
        "expected_choice_boundary_evidence_sha256": boundary.evidence_sha256,
        "expected_prompt_token_ids_sha256": boundary.prompt_prefix_token_ids_sha256,
        "expected_pre_anchor_residual_float32_sha256": tensor_float32_sha256(_pre_anchor(backend)),
    }
    with pytest.raises(ValueError, match="exactly A and B"):
        capture_closed_loop_dms_step(backend, "prompt", "A", "A", **common)
    with pytest.raises(ValueError, match="semantics must differ"):
        capture_closed_loop_dms_step(
            backend,
            "prompt",
            "A",
            "B",
            **{**common, "negative_semantic": "preserve"},
        )
    with pytest.raises(RuntimeError, match="boundary evidence differs"):
        capture_closed_loop_dms_step(
            backend,
            "prompt",
            "A",
            "B",
            **{**common, "expected_choice_boundary_evidence_sha256": "0" * 64},
        )
    with pytest.raises(RuntimeError, match="prompt token IDs differ"):
        capture_closed_loop_dms_step(
            backend,
            "prompt",
            "A",
            "B",
            **{**common, "expected_prompt_token_ids_sha256": "0" * 64},
        )
