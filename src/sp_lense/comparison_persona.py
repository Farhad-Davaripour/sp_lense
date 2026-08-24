from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

_JUDGE_KEYS = {
    "provider",
    "api",
    "model",
    "revision",
    "request_parameters",
    "prompt",
    "response_schema",
    "parser",
}
_JUDGE_SCORE_KEYS = {"trait_score", "coherence_score"}


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
class PersonaRollout:
    instruction_pair_id: str
    question_id: str
    rollout_index: int
    polarity: str
    system_prompt: str
    question: str
    response: str
    response_token_ids: tuple[int, ...] | None = None
    generation_seed: int | None = None
    source_model_id: str | None = None
    source_model_revision: str | None = None
    source_model_config_sha256: str | None = None
    stage1_lock_sha256: str | None = None
    runner_commit: str | None = None
    persona_protocol_sha256: str | None = None
    generation_config_sha256: str | None = None
    trait_score: float | None = None
    coherence_score: float | None = None
    judge_model: str | None = None
    judge_revision: str | None = None
    judge_rubric_sha256: str | None = None
    judge_prompt_sha256: str | None = None
    judge_config_sha256: str | None = None
    judge_raw_response: str | None = None
    judge_raw_response_sha256: str | None = None

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> PersonaRollout:
        expected_fields = set(cls.__dataclass_fields__)
        if set(row) != expected_fields:
            missing = sorted(expected_fields - set(row))
            extra = sorted(set(row) - expected_fields)
            raise ValueError(
                "persona rollout fields differ from the locked schema: "
                f"missing={missing}, extra={extra}"
            )
        values = {field: row.get(field) for field in cls.__dataclass_fields__}
        token_ids = values.get("response_token_ids")
        if token_ids is not None:
            values["response_token_ids"] = tuple(token_ids)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction_pair_id": self.instruction_pair_id,
            "question_id": self.question_id,
            "rollout_index": self.rollout_index,
            "polarity": self.polarity,
            "system_prompt": self.system_prompt,
            "question": self.question,
            "response": self.response,
            "response_token_ids": (
                None if self.response_token_ids is None else list(self.response_token_ids)
            ),
            "generation_seed": self.generation_seed,
            "source_model_id": self.source_model_id,
            "source_model_revision": self.source_model_revision,
            "source_model_config_sha256": self.source_model_config_sha256,
            "stage1_lock_sha256": self.stage1_lock_sha256,
            "runner_commit": self.runner_commit,
            "persona_protocol_sha256": self.persona_protocol_sha256,
            "generation_config_sha256": self.generation_config_sha256,
            "trait_score": self.trait_score,
            "coherence_score": self.coherence_score,
            "judge_model": self.judge_model,
            "judge_revision": self.judge_revision,
            "judge_rubric_sha256": self.judge_rubric_sha256,
            "judge_prompt_sha256": self.judge_prompt_sha256,
            "judge_config_sha256": self.judge_config_sha256,
            "judge_raw_response": self.judge_raw_response,
            "judge_raw_response_sha256": self.judge_raw_response_sha256,
        }


def load_persona_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload.get("instruction_pairs")
    questions = payload.get("extraction_questions")
    if not isinstance(pairs, list) or len(pairs) != 5:
        raise ValueError("canonical persona protocol requires exactly five instruction pairs")
    if not isinstance(questions, list) or len(questions) != 20:
        raise ValueError("canonical persona protocol requires exactly twenty extraction questions")
    _validate_judge_protocol(payload)
    pair_ids: set[str] = set()
    for pair in pairs:
        if set(pair) != {"id", "positive", "negative"}:
            raise ValueError("each persona instruction pair requires id, positive, and negative")
        if pair["id"] in pair_ids:
            raise ValueError(f"duplicate persona instruction pair: {pair['id']}")
        pair_ids.add(pair["id"])
    question_ids: set[str] = set()
    for question in questions:
        if set(question) != {"id", "text"}:
            raise ValueError("each persona extraction question requires id and text")
        if question["id"] in question_ids:
            raise ValueError(f"duplicate persona extraction question: {question['id']}")
        question_ids.add(question["id"])
    return payload


def persona_generation_provenance(
    protocol: Mapping[str, Any],
    *,
    model_id: str,
    model_revision: str,
    model_config_sha256: str,
    stage1_lock_sha256: str,
    runner_commit: str,
    persona_protocol_sha256: str,
) -> dict[str, str]:
    """Bind every rollout to the exact target model and locked generation recipe."""

    generation = protocol.get("generation")
    if not isinstance(generation, Mapping):
        raise TypeError("persona protocol lacks a generation configuration")
    values = {
        "source_model_id": model_id,
        "source_model_revision": model_revision,
        "source_model_config_sha256": model_config_sha256,
        "stage1_lock_sha256": stage1_lock_sha256,
        "runner_commit": runner_commit,
        "persona_protocol_sha256": persona_protocol_sha256,
        "generation_config_sha256": _canonical_json_sha256(generation),
    }
    for field, value in values.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"persona generation provenance {field} must be non-empty")
    for field in (
        "source_model_config_sha256",
        "stage1_lock_sha256",
        "persona_protocol_sha256",
        "generation_config_sha256",
    ):
        value = values[field]
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"persona generation provenance {field} must be a lowercase SHA-256")
    if len(runner_commit) != 40 or any(
        character not in "0123456789abcdef" for character in runner_commit
    ):
        raise ValueError("persona generation provenance runner_commit must be a Git commit")
    return values


def _validate_judge_protocol(protocol: Mapping[str, Any]) -> None:
    judge = protocol.get("judge")
    if not isinstance(judge, Mapping) or set(judge) != _JUDGE_KEYS:
        raise ValueError(f"persona judge must have exactly these keys: {sorted(_JUDGE_KEYS)}")
    for field in ("provider", "api", "model", "revision"):
        if not isinstance(judge[field], str) or not judge[field].strip():
            raise ValueError(f"persona judge {field} must be a non-empty string")
    request = judge["request_parameters"]
    if not isinstance(request, Mapping) or set(request) != {
        "temperature",
        "top_p",
        "max_output_tokens",
        "store",
    }:
        raise ValueError("persona judge request parameters are not fully pinned")
    if (
        request["temperature"] != 0.0
        or request["top_p"] != 1.0
        or not isinstance(request["max_output_tokens"], int)
        or isinstance(request["max_output_tokens"], bool)
        or request["max_output_tokens"] < 1
        or request["store"] is not False
    ):
        raise ValueError("persona judge request parameters are invalid")
    prompt = judge["prompt"]
    if not isinstance(prompt, Mapping) or set(prompt) != {"system", "user_template"}:
        raise ValueError("persona judge prompt must pin system and user_template")
    if not all(isinstance(prompt[key], str) and prompt[key].strip() for key in prompt):
        raise ValueError("persona judge prompt strings must be non-empty")
    if (
        prompt["user_template"].count("{question}") != 1
        or prompt["user_template"].count("{response}") != 1
    ):
        raise ValueError("persona judge template requires one question and response placeholder")
    schema = judge["response_schema"]
    if (
        not isinstance(schema, Mapping)
        or schema.get("type") != "object"
        or set(schema.get("properties", {})) != _JUDGE_SCORE_KEYS
        or set(schema.get("required", [])) != _JUDGE_SCORE_KEYS
        or schema.get("additionalProperties") is not False
    ):
        raise ValueError("persona judge response schema is not the locked two-score schema")
    for score_schema in schema["properties"].values():
        if score_schema != {"type": "integer", "minimum": 0, "maximum": 100}:
            raise ValueError("persona judge score schema must require integers from 0 to 100")
    parser = judge["parser"]
    expected_parser = {
        "format": "single_json_object",
        "allow_surrounding_whitespace": True,
        "allow_markdown_fences": False,
        "allow_extra_keys": False,
        "score_type": "integer_not_boolean",
        "required_keys": ["trait_score", "coherence_score"],
        "minimum": 0,
        "maximum": 100,
    }
    if parser != expected_parser:
        raise ValueError("persona judge parser rules differ from the locked strict parser")


def persona_judge_rubric_sha256(protocol: Mapping[str, Any]) -> str:
    return _canonical_json_sha256(protocol["judge_schema"])


def persona_judge_config_sha256(protocol: Mapping[str, Any]) -> str:
    """Hash the exact provider, model, request, prompt, schema, and parser lock."""

    _validate_judge_protocol(protocol)
    return _canonical_json_sha256(protocol["judge"])


def render_persona_judge_messages(
    record: PersonaRollout, protocol: Mapping[str, Any]
) -> list[dict[str, str]]:
    """Render the exact blinded judge input; condition polarity is intentionally absent."""

    _validate_judge_protocol(protocol)
    prompt = protocol["judge"]["prompt"]
    user = prompt["user_template"].format(
        question=record.question,
        response=record.response,
    )
    return [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": user},
    ]


def persona_judge_prompt_sha256(record: PersonaRollout, protocol: Mapping[str, Any]) -> str:
    return _canonical_json_sha256(render_persona_judge_messages(record, protocol))


def parse_persona_judge_response(raw_response: str) -> dict[str, int]:
    """Parse the locked single-object response format without repairing judge output."""

    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ValueError("persona judge raw response must be a non-empty string")
    stripped = raw_response.strip()
    if "```" in stripped:
        raise ValueError("persona judge response must not contain markdown fences")
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as error:
        raise ValueError("persona judge response is not one valid JSON object") from error
    if not isinstance(payload, dict) or set(payload) != _JUDGE_SCORE_KEYS:
        raise ValueError("persona judge response must contain exactly the two score keys")
    output: dict[str, int] = {}
    for field in sorted(_JUDGE_SCORE_KEYS):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
            raise ValueError(f"persona judge {field} must be an integer from 0 to 100")
        output[field] = value
    return output


def attach_persona_judgment(
    record: PersonaRollout,
    protocol: Mapping[str, Any],
    raw_response: str,
) -> PersonaRollout:
    """Attach a pre-obtained judge response and all provenance; no API call is made."""

    scores = parse_persona_judge_response(raw_response)
    judge = protocol["judge"]
    return replace(
        record,
        trait_score=float(scores["trait_score"]),
        coherence_score=float(scores["coherence_score"]),
        judge_model=judge["model"],
        judge_revision=judge["revision"],
        judge_rubric_sha256=persona_judge_rubric_sha256(protocol),
        judge_prompt_sha256=persona_judge_prompt_sha256(record, protocol),
        judge_config_sha256=persona_judge_config_sha256(protocol),
        judge_raw_response=raw_response,
        judge_raw_response_sha256=hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
    )


def expected_rollout_keys(
    protocol: Mapping[str, Any], *, rollouts_per_instruction_question: int = 10
) -> set[tuple[str, str, int, str]]:
    if rollouts_per_instruction_question < 1:
        raise ValueError("rollouts_per_instruction_question must be positive")
    return {
        (pair["id"], question["id"], rollout_index, polarity)
        for pair in protocol["instruction_pairs"]
        for question in protocol["extraction_questions"]
        for rollout_index in range(rollouts_per_instruction_question)
        for polarity in ("positive", "negative")
    }


def validate_rollouts(
    records: Sequence[PersonaRollout],
    protocol: Mapping[str, Any],
    *,
    rollouts_per_instruction_question: int = 10,
    require_scores: bool,
    expected_generation_provenance: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    expected = expected_rollout_keys(
        protocol, rollouts_per_instruction_question=rollouts_per_instruction_question
    )
    observed: set[tuple[str, str, int, str]] = set()
    pair_lookup = {pair["id"]: pair for pair in protocol["instruction_pairs"]}
    question_lookup = {question["id"]: question for question in protocol["extraction_questions"]}
    rubric_digest = persona_judge_rubric_sha256(protocol)
    config_digest = persona_judge_config_sha256(protocol)
    locked_judge = protocol["judge"]
    provenance_fields = (
        "source_model_id",
        "source_model_revision",
        "source_model_config_sha256",
        "stage1_lock_sha256",
        "runner_commit",
        "persona_protocol_sha256",
        "generation_config_sha256",
    )
    if expected_generation_provenance is not None and set(
        expected_generation_provenance
    ) != set(provenance_fields):
        raise ValueError("expected persona generation provenance fields are incomplete")
    for index, record in enumerate(records):
        if record.polarity not in {"positive", "negative"}:
            raise ValueError(f"rollout {index} has invalid polarity")
        if not record.response.strip():
            raise ValueError(f"rollout {index} has an empty response")
        if (
            not isinstance(record.response_token_ids, tuple)
            or not record.response_token_ids
            or any(
                isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
                for token_id in record.response_token_ids
            )
        ):
            raise ValueError(
                f"rollout {index} must retain its exact non-negative response token IDs"
            )
        key = (
            record.instruction_pair_id,
            record.question_id,
            record.rollout_index,
            record.polarity,
        )
        if key in observed:
            raise ValueError(f"duplicate persona rollout: {key}")
        observed.add(key)
        pair = pair_lookup.get(record.instruction_pair_id)
        question = question_lookup.get(record.question_id)
        if pair is None or question is None:
            raise ValueError(f"rollout {index} refers to an unknown locked prompt ID")
        if record.system_prompt != pair[record.polarity]:
            raise ValueError(f"rollout {index} system_prompt differs from locked text")
        if record.question != question["text"]:
            raise ValueError(f"rollout {index} question differs from locked text")
        if not isinstance(record.generation_seed, int) or isinstance(record.generation_seed, bool):
            raise TypeError(f"rollout {index} generation_seed must be an integer")
        if expected_generation_provenance is not None:
            mismatches = {
                field: (expected_generation_provenance[field], getattr(record, field))
                for field in provenance_fields
                if getattr(record, field) != expected_generation_provenance[field]
            }
            if mismatches:
                raise ValueError(
                    f"rollout {index} generation provenance differs from the lock: "
                    f"{mismatches}"
                )
        if require_scores:
            for field, value in (
                ("trait_score", record.trait_score),
                ("coherence_score", record.coherence_score),
            ):
                if value is None or not math.isfinite(float(value)) or not 0 <= float(value) <= 100:
                    raise ValueError(f"rollout {index} {field} must be a score from 0 to 100")
            if record.judge_model != locked_judge["model"]:
                raise ValueError(f"rollout {index} judge model differs from the lock")
            if record.judge_revision != locked_judge["revision"]:
                raise ValueError(f"rollout {index} judge revision differs from the lock")
            if record.judge_rubric_sha256 != rubric_digest:
                raise ValueError(f"rollout {index} judge rubric hash differs from the lock")
            if record.judge_prompt_sha256 != persona_judge_prompt_sha256(record, protocol):
                raise ValueError(f"rollout {index} judge prompt hash differs from the lock")
            if record.judge_config_sha256 != config_digest:
                raise ValueError(f"rollout {index} judge config hash differs from the lock")
            if record.judge_raw_response is None:
                raise ValueError(f"rollout {index} lacks the raw judge response")
            raw_digest = hashlib.sha256(record.judge_raw_response.encode("utf-8")).hexdigest()
            if record.judge_raw_response_sha256 != raw_digest:
                raise ValueError(f"rollout {index} raw judge response hash is invalid")
            parsed_scores = parse_persona_judge_response(record.judge_raw_response)
            if float(record.trait_score) != float(parsed_scores["trait_score"]):
                raise ValueError(f"rollout {index} trait_score differs from raw judge response")
            if float(record.coherence_score) != float(parsed_scores["coherence_score"]):
                raise ValueError(f"rollout {index} coherence_score differs from raw judge response")
    missing = expected - observed
    extra = observed - expected
    if missing or extra:
        raise ValueError(
            f"persona rollout grid mismatch: {len(missing)} missing and {len(extra)} unexpected"
        )
    return {
        "records": len(records),
        "paired_units": len(expected) // 2,
        "positive_records": sum(record.polarity == "positive" for record in records),
        "negative_records": sum(record.polarity == "negative" for record in records),
        "scores_present": require_scores,
    }


def encode_persona_prompt(backend: Any, system_prompt: str, question: str) -> Any:
    encoded = backend.model.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )
    return encoded["input_ids"].to(backend.device)


def generate_persona_rollout(
    backend: Any,
    system_prompt: str,
    question: str,
    *,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> tuple[str, tuple[int, ...]]:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if temperature < 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and non-negative")
    torch = backend.torch
    tokens = encode_persona_prompt(backend, system_prompt, question)
    prompt_length = int(tokens.shape[-1])
    # TransformerBridge's native generator uses a past-KV cache. The former manual
    # loop recomputed the full prefix at every token and made the 2,000-rollout grid
    # needlessly impractical on CPU.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        output = backend.model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            stop_at_eos=True,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-8),
            use_past_kv_cache=True,
            return_type="tokens",
            verbose=False,
        )
    if not isinstance(output, torch.Tensor):
        raise TypeError("TransformerBridge.generate did not return token IDs")
    response_ids = output[0, prompt_length:].tolist()
    special_ids = set(getattr(backend.model.tokenizer, "all_special_ids", ()))
    while response_ids and response_ids[-1] in special_ids:
        response_ids.pop()
    if not response_ids:
        raise RuntimeError("persona generation produced no non-special response tokens")
    response = backend.model.tokenizer.decode(response_ids, skip_special_tokens=True)
    return response, tuple(int(token_id) for token_id in response_ids)


def generate_persona_grid(
    backend: Any,
    protocol: Mapping[str, Any],
    *,
    rollouts_per_instruction_question: int = 10,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    seed: int = 20260824,
    generation_provenance: Mapping[str, str] | None = None,
) -> list[PersonaRollout]:
    """Generate the canonical 5×20×10×2 extraction grid, without judging it."""

    expected_rollout_keys(
        protocol, rollouts_per_instruction_question=rollouts_per_instruction_question
    )
    provenance = dict(generation_provenance or {})
    provenance_fields = {
        "source_model_id",
        "source_model_revision",
        "source_model_config_sha256",
        "stage1_lock_sha256",
        "runner_commit",
        "persona_protocol_sha256",
        "generation_config_sha256",
    }
    if provenance and set(provenance) != provenance_fields:
        raise ValueError("persona generation provenance fields are incomplete")
    output: list[PersonaRollout] = []
    rng = random.Random(seed)
    for pair in protocol["instruction_pairs"]:
        for question in protocol["extraction_questions"]:
            for rollout_index in range(rollouts_per_instruction_question):
                for polarity in ("positive", "negative"):
                    system_prompt = pair[polarity]
                    response, response_token_ids = generate_persona_rollout(
                        backend,
                        system_prompt,
                        question["text"],
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        seed=(rollout_seed := rng.randrange(0, 2**31)),
                    )
                    output.append(
                        PersonaRollout(
                            instruction_pair_id=pair["id"],
                            question_id=question["id"],
                            rollout_index=rollout_index,
                            polarity=polarity,
                            system_prompt=system_prompt,
                            question=question["text"],
                            response=response,
                            response_token_ids=response_token_ids,
                            generation_seed=rollout_seed,
                            **provenance,
                        )
                    )
    validate_rollouts(
        output,
        protocol,
        rollouts_per_instruction_question=rollouts_per_instruction_question,
        require_scores=False,
        expected_generation_provenance=(provenance if provenance else None),
    )
    return output


def response_mean_activation(backend: Any, record: PersonaRollout, *, layer: int) -> Any:
    return response_mean_activations(backend, record, layers=(layer,))[layer]


def response_mean_activations(
    backend: Any, record: PersonaRollout, *, layers: Sequence[int]
) -> dict[int, Any]:
    if not layers or len(set(layers)) != len(layers) or any(layer < 0 for layer in layers):
        raise ValueError("layers must be unique non-negative indices")
    prompt_tokens = encode_persona_prompt(backend, record.system_prompt, record.question)
    if not record.response_token_ids:
        raise ValueError("persona rollout lacks exact generated response token IDs")
    response_ids = list(record.response_token_ids)
    decoded_response = backend.model.tokenizer.decode(
        response_ids, skip_special_tokens=True
    )
    if decoded_response != record.response:
        raise ValueError("persona response text does not match its retained token IDs")
    response_tokens = backend.torch.tensor(
        [response_ids], device=prompt_tokens.device, dtype=prompt_tokens.dtype
    )
    tokens = backend.torch.cat([prompt_tokens, response_tokens], dim=-1)
    names = {f"blocks.{layer}.hook_out" for layer in layers}
    with backend.torch.inference_mode():
        _, cache = backend.model.run_with_cache(tokens, names_filter=lambda item: item in names)
    output = {}
    for layer in layers:
        name = f"blocks.{layer}.hook_out"
        response_activations = cache[name][0, int(prompt_tokens.shape[-1]) :].detach().float().cpu()
        if response_activations.shape[0] != len(response_ids):
            raise RuntimeError("persona response activation boundary is inconsistent")
        output[layer] = response_activations.mean(dim=0)
    return output


def retained_persona_pairs(
    records: Sequence[PersonaRollout],
    *,
    trait_threshold: float,
    coherence_threshold: float,
) -> list[tuple[PersonaRollout, PersonaRollout]]:
    keyed = {
        (
            record.instruction_pair_id,
            record.question_id,
            record.rollout_index,
            record.polarity,
        ): record
        for record in records
    }
    pairs = []
    for pair_id, question_id, rollout_index, polarity in sorted(keyed):
        if polarity != "positive":
            continue
        positive = keyed[(pair_id, question_id, rollout_index, "positive")]
        negative = keyed[(pair_id, question_id, rollout_index, "negative")]
        if (
            float(positive.trait_score) >= trait_threshold
            and float(negative.trait_score) < 100.0 - trait_threshold
            and float(positive.coherence_score) >= coherence_threshold
            and float(negative.coherence_score) >= coherence_threshold
        ):
            pairs.append((positive, negative))
    return pairs


def construct_persona_all_layers_from_scored_rollouts(
    backend: Any,
    records: Sequence[PersonaRollout],
    protocol: Mapping[str, Any],
    *,
    layers: Sequence[int],
    rollouts_per_instruction_question: int = 10,
    trait_threshold: float = 50,
    coherence_threshold: float = 50,
    min_retained_pairs: int = 16,
) -> tuple[dict[int, Any], dict[str, Any]]:
    validation = validate_rollouts(
        records,
        protocol,
        rollouts_per_instruction_question=rollouts_per_instruction_question,
        require_scores=True,
    )
    retained = retained_persona_pairs(
        records,
        trait_threshold=trait_threshold,
        coherence_threshold=coherence_threshold,
    )
    if len(retained) < min_retained_pairs:
        raise ValueError(
            f"persona filtering retained {len(retained)} pairs, fewer than minimum "
            f"{min_retained_pairs}"
        )
    positive_by_layer: dict[int, list[Any]] = {layer: [] for layer in layers}
    negative_by_layer: dict[int, list[Any]] = {layer: [] for layer in layers}
    for positive, negative in retained:
        positive_means = response_mean_activations(backend, positive, layers=layers)
        negative_means = response_mean_activations(backend, negative, layers=layers)
        for layer in layers:
            positive_by_layer[layer].append(positive_means[layer])
            negative_by_layer[layer].append(negative_means[layer])
    directions: dict[int, Any] = {}
    layer_diagnostics: dict[str, Any] = {}
    for layer in layers:
        raw = backend.torch.stack(positive_by_layer[layer]).mean(dim=0) - backend.torch.stack(
            negative_by_layer[layer]
        ).mean(dim=0)
        raw_norm = raw.norm()
        if not backend.torch.isfinite(raw_norm) or float(raw_norm.item()) <= 1e-12:
            raise ValueError(f"persona direction at layer {layer} is zero or non-finite")
        directions[layer] = (raw / raw_norm).detach().float().cpu()
        layer_diagnostics[str(layer)] = {"raw_direction_norm": float(raw_norm.item())}
    diagnostics = {
        "canonical_grid": validation,
        "n_retained_pairs": len(retained),
        "trait_threshold": trait_threshold,
        "coherence_threshold": coherence_threshold,
        "minimum_retained_pairs": min_retained_pairs,
        "construction": "published_response_token_average_positive_minus_negative",
        "layers": layer_diagnostics,
        "activation_passes": len(retained) * 2,
        "all_layers_captured_per_pass": True,
    }
    return directions, diagnostics


def construct_persona_from_scored_rollouts(
    backend: Any,
    records: Sequence[PersonaRollout],
    protocol: Mapping[str, Any],
    *,
    layer: int,
    rollouts_per_instruction_question: int = 10,
    trait_threshold: float = 50,
    coherence_threshold: float = 50,
) -> tuple[Any, dict[str, Any]]:
    """Fit the published response-average positive-minus-negative persona vector."""

    directions, diagnostics = construct_persona_all_layers_from_scored_rollouts(
        backend,
        records,
        protocol,
        layers=(layer,),
        rollouts_per_instruction_question=rollouts_per_instruction_question,
        trait_threshold=trait_threshold,
        coherence_threshold=coherence_threshold,
        min_retained_pairs=16,
    )
    diagnostics["layer"] = layer
    return directions[layer], diagnostics


def read_rollouts(path: Path) -> list[PersonaRollout]:
    return [
        PersonaRollout.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
