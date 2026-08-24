from __future__ import annotations

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from sp_lense.comparison_intervention import (
    InterventionSpec,
    hooks_for_spec,
)
from sp_lense.comparison_runtime import (
    append_completion_tokens,
    capture_final_prompt_gradient,
    choice_boundary_tokenizer_smoke,
    choice_score_from_logits,
    completion_logprob_sum,
    full_vocabulary_kl,
    greedy_generate,
    next_token_logits_with_perturbation,
    qwen35_choice_boundary_tokenizer_smoke,
    resolve_choice_boundary,
    score_choice,
    semantic_answer_activations,
    validate_locked_choice_runtime,
)


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return {"yes": [2, 3], "A": [0], "B": [1]}[text]


def _backend():
    return SimpleNamespace(
        torch=torch,
        model=SimpleNamespace(tokenizer=_Tokenizer()),
    )


def test_append_completion_tokens_keeps_prompt_prefix() -> None:
    prompt = torch.tensor([[8, 9]])
    assert append_completion_tokens(_backend(), prompt, "yes").tolist() == [[8, 9, 2, 3]]


def test_completion_logprob_sum_uses_only_response_predictions() -> None:
    logits = torch.zeros(1, 4, 5)
    tokens = torch.tensor([[4, 4, 2, 3]])
    logits[0, 1, 2] = 2.0
    logits[0, 2, 3] = 3.0
    expected = (
        torch.log_softmax(logits[0, 1], dim=-1)[2] + torch.log_softmax(logits[0, 2], dim=-1)[3]
    )
    assert completion_logprob_sum(torch, logits, tokens, 2).item() == pytest.approx(expected.item())


def test_full_vocabulary_kl_is_zero_for_identical_logits() -> None:
    logits = torch.tensor([1.0, 2.0, -1.0])
    assert full_vocabulary_kl(torch, logits, logits) == pytest.approx(0.0, abs=1e-7)


def test_choice_score_separates_pair_choice_from_raw_vocab_choice() -> None:
    baseline = torch.tensor([0.0, 0.0, 4.0])
    changed = torch.tensor([2.0, 1.0, 4.0])
    score = choice_score_from_logits(
        torch,
        changed,
        0,
        1,
        preserve_label="A",
        comply_label="B",
        baseline_logits=baseline,
    )
    assert score.preserve_log_odds == pytest.approx(1.0)
    assert score.pair_choice == "A"
    assert score.predicted_label == "OTHER"
    assert score.answer_pair_mass < 0.5
    assert score.kl_from_baseline > 0


class _HookModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(5, 2)
        self.unembed = torch.nn.Linear(2, 5, bias=False)
        self._hooks = []

    @contextmanager
    def hooks(self, fwd_hooks):
        old = self._hooks
        self._hooks = fwd_hooks
        try:
            yield
        finally:
            self._hooks = old

    def forward(self, tokens):
        activation = self.embedding(tokens)
        for _, hook in self._hooks:
            activation = hook(activation, None)
        return self.unembed(activation)


class _ChoiceTokenizer:
    chat_template = "fake-pinned-chat-template"
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
            assert len(messages) == 1
            values = prefix
        else:
            content = messages[-1]["content"]
            values = prefix + ({"": [], "A": [0], "B": [1]}[content]) + [4, 5]
        return {"input_ids": torch.tensor([values])}

    def decode(self, token_ids, **kwargs):
        del kwargs
        labels = {0: "A", 1: "B"}
        return "".join(labels.get(int(token_id), "") for token_id in token_ids)


def test_gradient_capture_does_not_allocate_model_parameter_gradients() -> None:
    model = _HookModel()
    model.tokenizer = _ChoiceTokenizer()
    backend = SimpleNamespace(
        torch=torch,
        model=model,
        encode=lambda prompt: torch.tensor([[2, 3]]),
        config=SimpleNamespace(model=SimpleNamespace(prompt_format="chat")),
        device="cpu",
    )
    gradient = capture_final_prompt_gradient(backend, "prompt", "A", "B", layer=10)
    assert gradient.shape == (2,)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_joint_choice_boundary_binds_prefix_content_eom_and_evidence_hash() -> None:
    model = SimpleNamespace(tokenizer=_ChoiceTokenizer())
    backend = SimpleNamespace(
        torch=torch,
        model=model,
        encode=lambda prompt: torch.tensor([[2, 3]]),
        config=SimpleNamespace(model=SimpleNamespace(prompt_format="chat")),
        device="cpu",
    )

    boundary = resolve_choice_boundary(backend, "prompt")

    assert boundary.a_token_id == 0
    assert boundary.b_token_id == 1
    assert boundary.assistant_end_token_ids == (4, 5)
    assert boundary.a_full_suffix_token_ids == (0, 4, 5)
    assert boundary.b_full_suffix_token_ids == (1, 4, 5)
    assert len(boundary.evidence_sha256) == 64
    assert boundary.evidence_sha256 == resolve_choice_boundary(backend, "prompt").evidence_sha256


def test_locked_choice_runtime_binds_resident_template_suffix_smoke_dtype_and_device() -> None:
    backend = SimpleNamespace(
        torch=torch,
        model=SimpleNamespace(tokenizer=_ChoiceTokenizer()),
        device="cpu",
        dtype_name="float32",
    )
    smoke = choice_boundary_tokenizer_smoke(backend.model.tokenizer, torch)
    suffixes = smoke["choice_suffix_token_ids"]
    locked_runtime = {
        "device": "cpu",
        "dtype": "float32",
        "chat_template_sha256": smoke["chat_template_sha256"],
        "assistant_choice_boundary": {
            "evidence_schema": "sp_lense.choice_boundary_evidence.v1",
            "content_token_ids": {"A": suffixes["A"][0], "B": suffixes["B"][0]},
            "assistant_end_token_ids": suffixes["A"][1:],
            "full_suffix_token_ids": suffixes,
            "non_sealed_smoke_prompt_set_sha256": smoke["smoke_prompt_set_sha256"],
            "non_sealed_smoke_evidence_sha256": smoke["smoke_evidence_sha256"],
        },
    }

    assert validate_locked_choice_runtime(backend, locked_runtime) == smoke
    locked_runtime["assistant_choice_boundary"]["content_token_ids"]["A"] = 99
    with pytest.raises(RuntimeError, match="resident tokenizer differs"):
        validate_locked_choice_runtime(backend, locked_runtime)


class _BrokenPrefixTokenizer(_ChoiceTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        encoded = super().apply_chat_template(messages, **kwargs)
        if not kwargs["add_generation_prompt"] and messages[-1]["content"] == "A":
            encoded["input_ids"][0, 0] = 9
        return encoded


@pytest.mark.parametrize("path", ["score", "gradient", "caa"])
def test_score_gradient_and_caa_fail_closed_on_joint_boundary_mismatch(path: str) -> None:
    model = _HookModel()
    model.tokenizer = _BrokenPrefixTokenizer()
    backend = SimpleNamespace(
        torch=torch,
        model=model,
        encode=lambda prompt: torch.tensor([[2, 3]]),
        config=SimpleNamespace(model=SimpleNamespace(prompt_format="chat")),
        device="cpu",
    )

    with pytest.raises(ValueError, match="does not preserve generation prefix"):
        if path == "score":
            score_choice(backend, "prompt", "A", "B")
        elif path == "gradient":
            capture_final_prompt_gradient(backend, "prompt", "A", "B", layer=10)
        else:
            semantic_answer_activations(backend, "prompt", "A", layers=(10,))


def test_next_token_pass_records_realized_relative_perturbation() -> None:
    model = _HookModel()
    backend = SimpleNamespace(torch=torch, model=model)
    tokens = torch.tensor([[2, 3]])
    direction = torch.tensor([1.0, 0.0])
    spec = InterventionSpec(
        layer=10,
        direction=direction,
        strength=0.02,
        geometry="matched_final_prompt",
        prompt_length=2,
        magnitude_mode="residual_relative",
    )
    _, diagnostics = next_token_logits_with_perturbation(backend, tokens, spec)
    assert diagnostics["n_positions"] == 1
    assert diagnostics["mean_relative_l2_norm"] == pytest.approx(0.02, rel=1e-5)


class _CacheTokenizer:
    eos_token_id = None

    def decode(self, token_ids, **kwargs):
        del kwargs
        return ",".join(str(int(token_id)) for token_id in token_ids)


class _CacheAwareModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "embedding_table",
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [2.0, -1.0],
                    [-1.0, 2.0],
                ]
            ),
        )
        self.register_buffer(
            "unembedding",
            torch.tensor(
                [
                    [1.0, -0.5, 0.25, 0.75, -1.0],
                    [-0.25, 1.0, 0.5, -0.75, 0.75],
                ]
            ),
        )
        self._hooks = []
        self.cached_decode_records = []
        self.tokenizer = _CacheTokenizer()

    @contextmanager
    def hooks(self, fwd_hooks):
        previous = self._hooks
        self._hooks = fwd_hooks
        try:
            yield
        finally:
            self._hooks = previous

    def forward(
        self,
        tokens,
        *,
        return_type="logits",
        use_cache=False,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        activations = self.embedding_table[tokens]
        for _, hook in self._hooks:
            activations = hook(activations, None)
        if past_key_values is None:
            prior = torch.zeros(
                (tokens.shape[0], 1, activations.shape[-1]), dtype=activations.dtype
            )
            prior_length = 0
            assert attention_mask is None
            assert position_ids is None
        else:
            prior_state, prior_length = past_key_values
            prior = prior_state.unsqueeze(1)
            assert tokens.shape[1] == 1
            assert attention_mask.tolist() == [[1] * (prior_length + 1)]
            assert position_ids.tolist() == [[prior_length]]
            self.cached_decode_records.append(
                {
                    "input_length": int(tokens.shape[1]),
                    "attention_length": int(attention_mask.shape[1]),
                    "position_id": int(position_ids.item()),
                }
            )
        states = prior + activations.cumsum(dim=1)
        logits = states @ self.unembedding
        cache = (states[:, -1].clone(), prior_length + int(tokens.shape[1]))
        if return_type == "logits_and_cache":
            assert use_cache
            return logits, cache
        return logits


def _full_recompute_reference(backend, prompt, spec, max_new_tokens):
    tokens = backend.encode(prompt)
    prompt_length = int(tokens.shape[-1])
    generated = []
    for _ in range(max_new_tokens):
        context = (
            nullcontext()
            if spec is None
            else backend.model.hooks(
                fwd_hooks=hooks_for_spec(backend.torch, spec, phase="full_sequence")
            )
        )
        with torch.inference_mode(), context:
            logits = backend.model(tokens)[0, -1]
        next_id = int(logits.argmax().item())
        generated.append(next_id)
        tokens = torch.cat([tokens, torch.tensor([[next_id]], dtype=tokens.dtype)], dim=-1)
    assert prompt_length == 3
    return backend.model.tokenizer.decode(generated, skip_special_tokens=True)


@pytest.mark.parametrize(
    "geometry",
    [None, "matched_final_prompt", "caa_post_prompt", "persona_response", "bipo_all_tokens"],
)
def test_cached_generation_matches_full_recomputation_schedule(geometry) -> None:
    model = _CacheAwareModel()
    backend = SimpleNamespace(
        torch=torch,
        model=model,
        encode=lambda prompt: torch.tensor([[2, 3, 4]]),
    )
    spec = (
        None
        if geometry is None
        else InterventionSpec(
            layer=10,
            direction=torch.tensor([1.0, -0.5]),
            strength=0.2,
            geometry=geometry,
            prompt_length=3,
        )
    )

    cached = greedy_generate(backend, "prompt", spec, max_new_tokens=5)
    cached_decode_records = list(model.cached_decode_records)
    reference = _full_recompute_reference(backend, "prompt", spec, max_new_tokens=5)

    assert cached == reference
    assert cached_decode_records == [
        {"input_length": 1, "attention_length": length, "position_id": length - 1}
        for length in range(4, 8)
    ]


def test_real_pinned_qwen_tokenizer_boundary_smoke_when_cached() -> None:
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.5-0.8B",
            revision="2fc06364715b967f1860aea9cf38778875588b17",
            local_files_only=True,
        )
    except OSError:
        pytest.skip("pinned Qwen tokenizer is not present in the local test cache")

    record = qwen35_choice_boundary_tokenizer_smoke(tokenizer, torch)

    assert record["uses_sealed_prompts"] is False
    assert record["choice_suffix_token_ids"] == {
        "A": [32, 248046, 198],
        "B": [33, 248046, 198],
    }
    assert len(record["smoke_evidence_sha256"]) == 64
