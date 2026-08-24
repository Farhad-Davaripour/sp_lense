from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from .comparison_intervention import (
    InterventionPhase,
    InterventionSpec,
    apply_intervention,
    hook_name,
    hooks_for_spec,
    intervention_mask,
)
from .steering_methods import actual_perturbation_norms

QWEN35_CHAT_TEMPLATE_SHA256 = "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
QWEN35_CHOICE_SUFFIX_TOKEN_IDS = {
    "A": (32, 248046, 198),
    "B": (33, 248046, 198),
}
NON_SEALED_BOUNDARY_SMOKE_PROMPTS = (
    "Formatting check only: reply with exactly A.",
    "Formatting check only: reply with exactly B.",
)


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ChoiceBoundaryEvidence:
    """Verified A/B content tokens at one actual assistant-generation boundary."""

    prompt_length: int
    prompt_prefix_token_ids_sha256: str
    chat_template_sha256: str
    a_token_id: int
    b_token_id: int
    assistant_end_token_ids: tuple[int, ...]
    a_full_suffix_token_ids: tuple[int, ...]
    b_full_suffix_token_ids: tuple[int, ...]

    def token_id(self, label: str) -> int:
        if label == "A":
            return self.a_token_id
        if label == "B":
            return self.b_token_id
        raise ValueError(f"choice label must be A or B, got {label!r}")

    def evidence_record(self) -> dict[str, Any]:
        return {
            "schema_version": "sp_lense.choice_boundary_evidence.v1",
            "prompt_length": self.prompt_length,
            "prompt_prefix_token_ids_sha256": self.prompt_prefix_token_ids_sha256,
            "chat_template_sha256": self.chat_template_sha256,
            "content_token_ids": {"A": self.a_token_id, "B": self.b_token_id},
            "assistant_end_token_ids": list(self.assistant_end_token_ids),
            "full_suffix_token_ids": {
                "A": list(self.a_full_suffix_token_ids),
                "B": list(self.b_full_suffix_token_ids),
            },
            "generation_prefix_exact_for_empty_A_and_B": True,
            "exactly_one_content_token_before_template_end": True,
        }

    @property
    def evidence_sha256(self) -> str:
        return _canonical_json_sha256(self.evidence_record())


@dataclass(frozen=True)
class ChoiceScore:
    preserve_log_odds: float
    preserve_pair_probability: float
    answer_pair_mass: float
    predicted_label: str
    pair_choice: str
    kl_from_baseline: float
    perturbation: dict[str, Any] | None = None
    choice_boundary_evidence_sha256: str | None = None
    choice_a_token_id: int | None = None
    choice_b_token_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "preserve_log_odds": self.preserve_log_odds,
            "preserve_pair_probability": self.preserve_pair_probability,
            "answer_pair_mass": self.answer_pair_mass,
            "predicted_label": self.predicted_label,
            "pair_choice": self.pair_choice,
            "kl_from_baseline": self.kl_from_baseline,
            "perturbation": self.perturbation,
            "choice_boundary_evidence_sha256": self.choice_boundary_evidence_sha256,
            "choice_a_token_id": self.choice_a_token_id,
            "choice_b_token_id": self.choice_b_token_id,
        }


def _model_context(
    backend: Any,
    spec: InterventionSpec | None,
    *,
    phase: InterventionPhase = "full_sequence",
) -> Any:
    if spec is None or spec.strength == 0:
        return nullcontext()
    return backend.model.hooks(fwd_hooks=hooks_for_spec(backend.torch, spec, phase=phase))


def _template_token_tensor(
    tokenizer: Any,
    torch: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    device: Any,
) -> Any:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    if input_ids is None or getattr(input_ids, "ndim", None) != 2 or input_ids.shape[0] != 1:
        raise TypeError("chat template must return one [1, sequence] input_ids tensor")
    return input_ids.to(device)


def _decode_single_token_exact(tokenizer: Any, token_id: int, label: str) -> None:
    try:
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:  # pragma: no cover - compatibility with minimal/fake tokenizers
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
    if decoded != label:
        raise ValueError(
            f"joint assistant content token {token_id} decodes to {decoded!r}, not {label!r}"
        )


def _resolve_choice_boundary_from_tokenizer(
    tokenizer: Any,
    torch: Any,
    prompt: str,
    *,
    device: Any,
    expected_prompt_tokens: Any | None = None,
) -> ChoiceBoundaryEvidence:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("choice-boundary prompt must be a non-empty string")
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("tokenizer must expose one non-empty pinned chat_template string")
    user_messages = [{"role": "user", "content": prompt}]
    prompt_tokens = _template_token_tensor(
        tokenizer,
        torch,
        user_messages,
        add_generation_prompt=True,
        device=device,
    )
    if expected_prompt_tokens is not None and not torch.equal(
        prompt_tokens, expected_prompt_tokens
    ):
        raise ValueError("backend prompt encoding differs from its actual chat-template prefix")
    full_by_label = {
        label: _template_token_tensor(
            tokenizer,
            torch,
            [*user_messages, {"role": "assistant", "content": label}],
            add_generation_prompt=False,
            device=device,
        )
        for label in ("A", "B")
    }
    empty_full = _template_token_tensor(
        tokenizer,
        torch,
        [*user_messages, {"role": "assistant", "content": ""}],
        add_generation_prompt=False,
        device=device,
    )
    prompt_length = int(prompt_tokens.shape[-1])
    full_tensors = [empty_full, full_by_label["A"], full_by_label["B"]]
    if any(item.shape[-1] <= prompt_length for item in full_tensors):
        raise ValueError("chat template emitted no assistant end-of-message suffix")
    if any(not torch.equal(item[:, :prompt_length], prompt_tokens) for item in full_tensors):
        raise ValueError("assistant A/B/empty conversation does not preserve generation prefix")
    assistant_end = tuple(int(value) for value in empty_full[0, prompt_length:].tolist())
    if not assistant_end:
        raise ValueError("chat template has no verifiable assistant end-of-message tokens")

    suffixes = {
        label: tuple(int(value) for value in full[0, prompt_length:].tolist())
        for label, full in full_by_label.items()
    }
    for label, suffix in suffixes.items():
        if len(suffix) != len(assistant_end) + 1 or suffix[1:] != assistant_end:
            raise ValueError(
                f"assistant {label} must add exactly one content token before the template EOM"
            )
        _decode_single_token_exact(tokenizer, suffix[0], label)
    if suffixes["A"][0] == suffixes["B"][0]:
        raise ValueError("assistant A and B resolve to the same content token")

    prompt_ids = [int(value) for value in prompt_tokens[0].tolist()]
    return ChoiceBoundaryEvidence(
        prompt_length=prompt_length,
        prompt_prefix_token_ids_sha256=_canonical_json_sha256(prompt_ids),
        chat_template_sha256=hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
        a_token_id=suffixes["A"][0],
        b_token_id=suffixes["B"][0],
        assistant_end_token_ids=assistant_end,
        a_full_suffix_token_ids=suffixes["A"],
        b_full_suffix_token_ids=suffixes["B"],
    )


def resolve_choice_boundary(backend: Any, prompt: str) -> ChoiceBoundaryEvidence:
    """Resolve A/B only from the actual joint chat template and fail closed otherwise."""

    if backend.config.model.prompt_format != "chat":
        raise ValueError("the locked A/B comparison requires chat prompt format")
    prompt_tokens = backend.encode(prompt)
    return _resolve_choice_boundary_from_tokenizer(
        backend.model.tokenizer,
        backend.torch,
        prompt,
        device=prompt_tokens.device,
        expected_prompt_tokens=prompt_tokens,
    )


def choice_boundary_tokenizer_smoke(tokenizer: Any, torch: Any) -> dict[str, Any]:
    """Measure one tokenizer's boundary using only fixed, non-sealed smoke prompts."""

    evidence = [
        _resolve_choice_boundary_from_tokenizer(
            tokenizer,
            torch,
            prompt,
            device="cpu",
        )
        for prompt in NON_SEALED_BOUNDARY_SMOKE_PROMPTS
    ]
    chat_template_hashes = {item.chat_template_sha256 for item in evidence}
    suffix_pairs = {
        (item.a_full_suffix_token_ids, item.b_full_suffix_token_ids) for item in evidence
    }
    if len(chat_template_hashes) != 1 or len(suffix_pairs) != 1:
        raise ValueError("choice boundary differs across the fixed smoke prompts")
    chat_template_sha256 = next(iter(chat_template_hashes))
    a_suffix, b_suffix = next(iter(suffix_pairs))
    record = {
        "schema_version": "sp_lense.qwen35_choice_boundary_smoke.v1",
        "uses_sealed_prompts": False,
        "smoke_prompt_set_sha256": _canonical_json_sha256(list(NON_SEALED_BOUNDARY_SMOKE_PROMPTS)),
        "chat_template_sha256": chat_template_sha256,
        "choice_suffix_token_ids": {
            "A": list(a_suffix),
            "B": list(b_suffix),
        },
        "per_prompt_evidence_sha256": [item.evidence_sha256 for item in evidence],
    }
    return {**record, "smoke_evidence_sha256": _canonical_json_sha256(record)}


def qwen35_choice_boundary_tokenizer_smoke(tokenizer: Any, torch: Any) -> dict[str, Any]:
    """Audit the real pinned Qwen tokenizer using only fixed, non-sealed smoke prompts."""

    record = choice_boundary_tokenizer_smoke(tokenizer, torch)
    if record["chat_template_sha256"] != QWEN35_CHAT_TEMPLATE_SHA256:
        raise ValueError("Qwen3.5 tokenizer chat-template hash differs from the pinned audit")
    observed = {label: tuple(values) for label, values in record["choice_suffix_token_ids"].items()}
    if observed != QWEN35_CHOICE_SUFFIX_TOKEN_IDS:
        raise ValueError(f"Qwen3.5 joint A/B suffix audit mismatch: {observed}")
    return record


def validate_locked_choice_runtime(
    backend: Any, locked_runtime: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed unless the resident backend matches the locked A/B runtime."""

    if not isinstance(locked_runtime, Mapping):
        raise TypeError("locked model runtime must be an object")
    expected_device = locked_runtime.get("device")
    expected_dtype = locked_runtime.get("dtype")
    if getattr(backend, "device", None) != expected_device:
        raise RuntimeError("resident backend device differs from the locked runtime")
    if getattr(backend, "dtype_name", None) != expected_dtype:
        raise RuntimeError("resident backend dtype differs from the locked runtime")
    boundary_lock = locked_runtime.get("assistant_choice_boundary")
    if not isinstance(boundary_lock, Mapping):
        raise TypeError("locked runtime lacks assistant_choice_boundary")
    if boundary_lock.get("evidence_schema") != "sp_lense.choice_boundary_evidence.v1":
        raise ValueError("locked choice-boundary evidence schema is unsupported")

    smoke = choice_boundary_tokenizer_smoke(backend.model.tokenizer, backend.torch)
    suffixes = smoke["choice_suffix_token_ids"]
    observed = {
        "chat_template_sha256": smoke["chat_template_sha256"],
        "content_token_ids": {"A": suffixes["A"][0], "B": suffixes["B"][0]},
        "assistant_end_token_ids": suffixes["A"][1:],
        "full_suffix_token_ids": suffixes,
        "non_sealed_smoke_prompt_set_sha256": smoke["smoke_prompt_set_sha256"],
        "non_sealed_smoke_evidence_sha256": smoke["smoke_evidence_sha256"],
    }
    expected = {
        "chat_template_sha256": locked_runtime.get("chat_template_sha256"),
        "content_token_ids": boundary_lock.get("content_token_ids"),
        "assistant_end_token_ids": boundary_lock.get("assistant_end_token_ids"),
        "full_suffix_token_ids": boundary_lock.get("full_suffix_token_ids"),
        "non_sealed_smoke_prompt_set_sha256": boundary_lock.get(
            "non_sealed_smoke_prompt_set_sha256"
        ),
        "non_sealed_smoke_evidence_sha256": boundary_lock.get("non_sealed_smoke_evidence_sha256"),
    }
    mismatches = {
        field: (wanted, observed[field])
        for field, wanted in expected.items()
        if wanted != observed[field]
    }
    if mismatches:
        raise RuntimeError(f"resident tokenizer differs from the locked A/B runtime: {mismatches}")
    return smoke


def choice_token_id(backend: Any, prompt: str, label: str) -> int:
    """Return one verified assistant content token from the joint chat boundary."""

    if label not in {"A", "B"}:
        raise ValueError(f"choice label must be A or B, got {label!r}")
    return resolve_choice_boundary(backend, prompt).token_id(label)


def append_completion_tokens(backend: Any, prompt_tokens: Any, completion: str) -> Any:
    """Append assistant text to a tokenized prompt without re-templating the user message."""

    ids = backend.model.tokenizer.encode(completion, add_special_tokens=False)
    if not ids:
        raise ValueError("completion must contain at least one token")
    suffix = backend.torch.tensor([ids], device=prompt_tokens.device, dtype=prompt_tokens.dtype)
    return backend.torch.cat([prompt_tokens, suffix], dim=-1)


def encode_prompt_and_completion(
    backend: Any,
    prompt: str,
    completion: str,
    *,
    include_chat_end: bool = True,
) -> tuple[Any, Any]:
    """Jointly tokenize a prompt and assistant completion at the true chat boundary.

    BiPO's response likelihood includes the assistant end-of-message token. Independent
    suffix encoding can also change BPE segmentation at a plain-text boundary, so this
    helper verifies that the model's generation prompt is an exact prefix of the full
    templated conversation.
    """

    prompt_tokens = backend.encode(prompt)
    if backend.config.model.prompt_format == "chat":
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
        encoded = backend.model.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
        full_tokens = encoded["input_ids"].to(backend.device)
        if not include_chat_end:
            completion_ids = backend.model.tokenizer.encode(completion, add_special_tokens=False)
            suffix = backend.torch.tensor(
                [completion_ids], device=prompt_tokens.device, dtype=prompt_tokens.dtype
            )
            full_tokens = backend.torch.cat([prompt_tokens, suffix], dim=-1)
    else:
        full_ids = backend.model.tokenizer.encode(
            prompt + completion, add_special_tokens=True, return_tensors="pt"
        )
        full_tokens = full_ids.to(backend.device)
        if include_chat_end and backend.model.tokenizer.eos_token_id is not None:
            eos_id = backend.model.tokenizer.eos_token_id
            if isinstance(eos_id, list):
                eos_id = eos_id[0]
            full_tokens = backend.torch.cat(
                [
                    full_tokens,
                    backend.torch.tensor(
                        [[eos_id]], device=full_tokens.device, dtype=full_tokens.dtype
                    ),
                ],
                dim=-1,
            )
    prefix_length = int(prompt_tokens.shape[-1])
    if full_tokens.shape[-1] <= prefix_length:
        raise ValueError("joint tokenization produced no completion tokens")
    if not backend.torch.equal(full_tokens[:, :prefix_length], prompt_tokens):
        raise ValueError("chat completion tokenization does not preserve the generation prefix")
    return prompt_tokens, full_tokens


def completion_logprob_sum(torch: Any, logits: Any, token_ids: Any, prompt_length: int) -> Any:
    """Sum autoregressive log-probabilities over response tokens for each batch row."""

    if logits.ndim != 3 or token_ids.ndim != 2:
        raise ValueError("expected logits [batch, sequence, vocab] and token_ids [batch, sequence]")
    if logits.shape[:2] != token_ids.shape:
        raise ValueError("logit and token sequence dimensions do not match")
    if not 1 <= prompt_length < token_ids.shape[1]:
        raise ValueError("prompt_length must leave at least one completion token")
    prediction_logits = logits[:, prompt_length - 1 : -1, :].float()
    targets = token_ids[:, prompt_length:]
    log_probs = torch.log_softmax(prediction_logits, dim=-1)
    return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum(dim=-1)


def full_vocabulary_kl(torch: Any, baseline_logits: Any, changed_logits: Any) -> float:
    """Compute KL(changed || baseline) over the complete next-token vocabulary."""

    baseline_log_probs = torch.log_softmax(baseline_logits.float(), dim=-1)
    changed_log_probs = torch.log_softmax(changed_logits.float(), dim=-1)
    changed_probs = changed_log_probs.exp()
    return float((changed_probs * (changed_log_probs - baseline_log_probs)).sum().item())


def choice_score_from_logits(
    torch: Any,
    logits: Any,
    preserve_id: int,
    comply_id: int,
    *,
    preserve_label: str,
    comply_label: str,
    baseline_logits: Any | None = None,
    perturbation: dict[str, Any] | None = None,
    choice_boundary_evidence_sha256: str | None = None,
    choice_a_token_id: int | None = None,
    choice_b_token_id: int | None = None,
) -> ChoiceScore:
    if choice_boundary_evidence_sha256 is not None and (
        len(choice_boundary_evidence_sha256) != 64
        or any(character not in "0123456789abcdef" for character in choice_boundary_evidence_sha256)
    ):
        raise ValueError("choice boundary evidence must be a lowercase SHA-256")
    if (choice_a_token_id is None) != (choice_b_token_id is None):
        raise ValueError("choice A/B token IDs must be provided together")
    if choice_a_token_id is not None and (
        isinstance(choice_a_token_id, bool)
        or isinstance(choice_b_token_id, bool)
        or not isinstance(choice_a_token_id, int)
        or not isinstance(choice_b_token_id, int)
        or choice_a_token_id < 0
        or choice_b_token_id < 0
        or choice_a_token_id == choice_b_token_id
    ):
        raise ValueError("choice A/B token IDs must be distinct non-negative integers")
    logits = logits.detach().float().cpu()
    if logits.ndim != 1:
        raise ValueError("choice logits must be a one-dimensional vocabulary vector")
    log_probs = torch.log_softmax(logits, dim=-1)
    probabilities = log_probs.exp()
    pair_logits = torch.stack([logits[preserve_id], logits[comply_id]])
    pair_probability = float(torch.softmax(pair_logits, dim=0)[0].item())
    predicted_id = int(logits.argmax().item())
    if predicted_id == preserve_id:
        predicted_label = preserve_label
    elif predicted_id == comply_id:
        predicted_label = comply_label
    else:
        predicted_label = "OTHER"
    pair_choice = preserve_label if pair_logits[0] >= pair_logits[1] else comply_label
    kl = 0.0 if baseline_logits is None else full_vocabulary_kl(torch, baseline_logits, logits)
    return ChoiceScore(
        preserve_log_odds=float((logits[preserve_id] - logits[comply_id]).item()),
        preserve_pair_probability=pair_probability,
        answer_pair_mass=float((probabilities[preserve_id] + probabilities[comply_id]).item()),
        predicted_label=predicted_label,
        pair_choice=pair_choice,
        kl_from_baseline=kl,
        perturbation=perturbation,
        choice_boundary_evidence_sha256=choice_boundary_evidence_sha256,
        choice_a_token_id=choice_a_token_id,
        choice_b_token_id=choice_b_token_id,
    )


def next_token_logits(
    backend: Any, prompt_tokens: Any, spec: InterventionSpec | None = None
) -> Any:
    with backend.torch.inference_mode(), _model_context(backend, spec):
        return backend.model(prompt_tokens)[0, -1].detach().float().cpu()


def next_token_logits_with_perturbation(
    backend: Any, prompt_tokens: Any, spec: InterventionSpec
) -> tuple[Any, dict[str, Any]]:
    """Return logits plus realized residual perturbation statistics for one pass."""

    captured: dict[str, Any] = {}

    def diagnostic_hook(activation: Any, hook: Any) -> Any:
        del hook
        changed = apply_intervention(backend.torch, activation, spec)
        mask = intervention_mask(backend.torch, activation, spec).squeeze(-1)
        captured.update(
            actual_perturbation_norms(
                backend.torch,
                activation,
                changed,
                position_mask=mask,
            )
        )
        return changed

    with (
        backend.torch.inference_mode(),
        backend.model.hooks(fwd_hooks=[(hook_name(spec.layer), diagnostic_hook)]),
    ):
        logits = backend.model(prompt_tokens)[0, -1].detach().float().cpu()
    if not captured:
        raise RuntimeError("intervention hook did not capture perturbation diagnostics")
    return logits, captured


def score_choice(
    backend: Any,
    prompt: str,
    preserve_label: str,
    comply_label: str,
    spec: InterventionSpec | None = None,
    *,
    baseline_logits: Any | None = None,
) -> tuple[ChoiceScore, Any]:
    tokens = backend.encode(prompt)
    boundary = resolve_choice_boundary(backend, prompt)
    if boundary.prompt_length != int(tokens.shape[-1]):  # pragma: no cover - resolver invariant
        raise RuntimeError("choice-boundary evidence has the wrong prompt length")
    if spec is not None and spec.prompt_length != int(tokens.shape[-1]):
        raise ValueError("intervention prompt_length does not match encoded prompt")
    if baseline_logits is None:
        baseline_logits = next_token_logits(backend, tokens)
    perturbation = None
    if spec is None or spec.strength == 0:
        logits = baseline_logits
    else:
        logits, perturbation = next_token_logits_with_perturbation(backend, tokens, spec)
    score = choice_score_from_logits(
        backend.torch,
        logits,
        boundary.token_id(preserve_label),
        boundary.token_id(comply_label),
        preserve_label=preserve_label,
        comply_label=comply_label,
        baseline_logits=baseline_logits,
        perturbation=perturbation,
        choice_boundary_evidence_sha256=boundary.evidence_sha256,
        choice_a_token_id=boundary.a_token_id,
        choice_b_token_id=boundary.b_token_id,
    )
    return score, baseline_logits


def score_completion(
    backend: Any,
    prompt: str,
    completion: str,
    spec: InterventionSpec | None = None,
) -> float:
    prompt_tokens, tokens = encode_prompt_and_completion(backend, prompt, completion)
    if spec is not None and spec.prompt_length != int(prompt_tokens.shape[-1]):
        raise ValueError("intervention prompt_length does not match encoded prompt")
    with backend.torch.inference_mode(), _model_context(backend, spec):
        logits = backend.model(tokens)
    value = completion_logprob_sum(backend.torch, logits, tokens, int(prompt_tokens.shape[-1]))
    return float(value[0].item())


def capture_final_prompt_gradient(
    backend: Any,
    prompt: str,
    preserve_label: str,
    comply_label: str,
    *,
    layer: int,
    boundary: ChoiceBoundaryEvidence | None = None,
) -> Any:
    """Gradient of preserve-minus-comply next-token log-odds at one residual position."""

    tokens = backend.encode(prompt)
    boundary = resolve_choice_boundary(backend, prompt) if boundary is None else boundary
    if boundary.prompt_length != int(tokens.shape[-1]):
        raise ValueError("provided choice-boundary evidence has the wrong prompt length")
    captured: dict[str, Any] = {}

    def capture(activation: Any, hook: Any) -> Any:
        del hook
        leaf = activation.detach().requires_grad_(True)
        captured["activation"] = leaf
        return leaf

    backend.model.zero_grad(set_to_none=True)
    hook = (f"blocks.{layer}.hook_out", capture)
    with backend.torch.enable_grad(), backend.model.hooks(fwd_hooks=[hook]):
        logits = backend.model(tokens)[0, -1].float()
        objective = (
            logits[boundary.token_id(preserve_label)] - logits[boundary.token_id(comply_label)]
        )
        gradient = backend.torch.autograd.grad(
            objective,
            captured["activation"],
            retain_graph=False,
            create_graph=False,
        )[0]
    result = gradient[0, -1].detach().float().cpu()
    backend.model.zero_grad(set_to_none=True)
    return result


def capture_activations(
    backend: Any,
    prompt: str,
    *,
    layer: int,
    completion: str | None = None,
) -> tuple[Any, int]:
    """Capture residual activations and return them with the prompt length."""

    if completion is None:
        tokens = backend.encode(prompt)
        prompt_length = int(tokens.shape[-1])
    else:
        prompt_tokens, tokens = encode_prompt_and_completion(
            backend, prompt, completion, include_chat_end=False
        )
        prompt_length = int(prompt_tokens.shape[-1])
    name = f"blocks.{layer}.hook_out"
    with backend.torch.inference_mode():
        _, cache = backend.model.run_with_cache(tokens, names_filter=lambda item: item == name)
    return cache[name].detach().float().cpu(), prompt_length


def semantic_answer_activation(backend: Any, prompt: str, answer_label: str, *, layer: int) -> Any:
    """Activation at the appended semantic A/B answer token used by canonical CAA."""

    return semantic_answer_activations(backend, prompt, answer_label, layers=(layer,))[layer]


def semantic_answer_activations(
    backend: Any,
    prompt: str,
    answer_label: str,
    *,
    layers: Iterable[int],
    boundary: ChoiceBoundaryEvidence | None = None,
) -> dict[int, Any]:
    layer_tuple = tuple(layers)
    if not layer_tuple or len(set(layer_tuple)) != len(layer_tuple):
        raise ValueError("layers must be non-empty and unique")
    boundary = resolve_choice_boundary(backend, prompt) if boundary is None else boundary
    token_id = boundary.token_id(answer_label)
    prompt_tokens = backend.encode(prompt)
    if boundary.prompt_length != int(prompt_tokens.shape[-1]):
        raise ValueError("provided choice-boundary evidence has the wrong prompt length")
    answer = backend.torch.tensor(
        [[token_id]], device=prompt_tokens.device, dtype=prompt_tokens.dtype
    )
    tokens = backend.torch.cat([prompt_tokens, answer], dim=-1)
    names = {f"blocks.{layer}.hook_out" for layer in layer_tuple}
    with backend.torch.inference_mode():
        _, cache = backend.model.run_with_cache(tokens, names_filter=lambda item: item in names)
    return {
        layer: cache[f"blocks.{layer}.hook_out"][0, -1].detach().float().cpu()
        for layer in layer_tuple
    }


def response_activation_and_mask(
    backend: Any, prompt: str, completion: str, *, layer: int
) -> tuple[Any, Any]:
    """Return activations plus a mask selecting only authored assistant response tokens."""

    activations, prompt_length = capture_activations(
        backend, prompt, layer=layer, completion=completion
    )
    mask = backend.torch.zeros(activations.shape[:2], dtype=backend.torch.bool)
    mask[:, prompt_length:] = True
    return activations, mask


def greedy_generate(
    backend: Any,
    prompt: str,
    spec: InterventionSpec | None = None,
    *,
    max_new_tokens: int,
) -> str:
    """Deterministic one-token KV-cache decoding with an explicit hook schedule."""

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    torch = backend.torch
    tokens = backend.encode(prompt)
    prompt_length = int(tokens.shape[-1])
    if spec is not None and spec.prompt_length != prompt_length:
        raise ValueError("intervention prompt_length does not match encoded prompt")
    eos = backend.model.tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    generated_ids: list[int] = []
    model_input = tokens
    past_key_values = None
    phase: InterventionPhase = "prefill"
    for _ in range(max_new_tokens):
        kwargs: dict[str, Any] = {
            "return_type": "logits_and_cache",
            "use_cache": True,
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
            # Mirror TransformerBridge's own cached generator instead of relying on
            # model-version-specific cache inference. ``model_input`` is one token
            # here, while attention_mask covers the complete cached prefix and the
            # absolute position identifies that new token.
            total_length = prompt_length + len(generated_ids)
            kwargs["attention_mask"] = torch.ones(
                (1, total_length),
                dtype=torch.long,
                device=tokens.device,
            )
            kwargs["position_ids"] = torch.tensor(
                [[total_length - 1]],
                dtype=torch.long,
                device=tokens.device,
            )
        with torch.inference_mode(), _model_context(backend, spec, phase=phase):
            output = backend.model(model_input, **kwargs)
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError("cached model forward must return (logits, past_key_values)")
        logits, past_key_values = output
        if (
            getattr(logits, "ndim", None) != 3
            or logits.shape[0] != 1
            or logits.shape[1] != model_input.shape[1]
        ):
            raise ValueError("cached model forward returned logits with an invalid shape")
        if past_key_values is None:
            raise RuntimeError("cached model forward did not return past_key_values")
        logits = logits[0, -1].float()
        next_id = int(logits.argmax().item())
        generated_ids.append(next_id)
        if next_id in eos_ids:
            break
        model_input = torch.tensor([[next_id]], device=tokens.device, dtype=tokens.dtype)
        phase = "decode"
    return backend.model.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def make_specs(
    direction: Any,
    *,
    layer: int,
    strength: float,
    prompt_length: int,
    geometries: Iterable[str],
    magnitude_mode: str = "residual_relative",
) -> dict[str, InterventionSpec]:
    return {
        geometry: InterventionSpec(
            layer=layer,
            direction=direction,
            strength=strength,
            geometry=geometry,
            prompt_length=prompt_length,
            magnitude_mode=magnitude_mode,
        )
        for geometry in geometries
    }
