from __future__ import annotations

import json
from pathlib import Path

import pytest

from sp_lense.comparison_persona import (
    PersonaRollout,
    attach_persona_judgment,
    expected_rollout_keys,
    load_persona_protocol,
    parse_persona_judge_response,
    persona_generation_provenance,
    persona_judge_config_sha256,
    persona_judge_prompt_sha256,
    validate_rollouts,
)


def _protocol() -> dict:
    path = Path(__file__).parents[1] / "data" / "persona_self_preservation_protocol.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_enforces_canonical_grid_size(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(_protocol()), encoding="utf-8")
    protocol = load_persona_protocol(path)
    assert len(expected_rollout_keys(protocol)) == 2000


def test_rollout_validation_pairs_poles_and_requires_scores() -> None:
    protocol = _protocol()
    provenance = persona_generation_provenance(
        protocol,
        model_id="Qwen/Qwen3.5-test",
        model_revision="revision",
        model_config_sha256="d" * 64,
        stage1_lock_sha256="a" * 64,
        runner_commit="b" * 40,
        persona_protocol_sha256="c" * 64,
    )
    records = []
    for pair in protocol["instruction_pairs"]:
        for question in protocol["extraction_questions"]:
            for polarity in ("positive", "negative"):
                raw_response = json.dumps(
                    {
                        "trait_score": 100 if polarity == "positive" else 0,
                        "coherence_score": 100,
                    },
                    separators=(",", ":"),
                )
                records.append(
                    attach_persona_judgment(
                        PersonaRollout(
                            instruction_pair_id=pair["id"],
                            question_id=question["id"],
                            rollout_index=0,
                            polarity=polarity,
                            system_prompt=pair[polarity],
                            question=question["text"],
                            response="answer",
                            response_token_ids=(1,),
                            generation_seed=1,
                            **provenance,
                        ),
                        protocol,
                        raw_response,
                    )
                )
    summary = validate_rollouts(
        records,
        protocol,
        rollouts_per_instruction_question=1,
        require_scores=True,
        expected_generation_provenance=provenance,
    )
    assert summary["records"] == 200
    assert summary["paired_units"] == 100
    tampered = list(records)
    tampered[0] = PersonaRollout.from_dict(
        {**tampered[0].to_dict(), "source_model_revision": "wrong"}
    )
    with pytest.raises(ValueError, match="generation provenance"):
        validate_rollouts(
            tampered,
            protocol,
            rollouts_per_instruction_question=1,
            require_scores=True,
            expected_generation_provenance=provenance,
        )
    records[0] = PersonaRollout.from_dict({**records[0].to_dict(), "trait_score": None})
    with pytest.raises(ValueError, match="trait_score"):
        validate_rollouts(
            records,
            protocol,
            rollouts_per_instruction_question=1,
            require_scores=True,
            expected_generation_provenance=provenance,
        )


def test_rollout_reader_rejects_unknown_or_missing_schema_fields() -> None:
    row = PersonaRollout(
        instruction_pair_id="pair",
        question_id="question",
        rollout_index=0,
        polarity="positive",
        system_prompt="system",
        question="question",
        response="response",
    ).to_dict()
    with pytest.raises(ValueError, match="extra"):
        PersonaRollout.from_dict({**row, "unexpected": 1})
    del row["runner_commit"]
    with pytest.raises(ValueError, match="missing"):
        PersonaRollout.from_dict(row)


def test_judge_prompt_and_config_hashes_are_deterministic_and_condition_blind() -> None:
    protocol = _protocol()
    common = {
        "instruction_pair_id": "continuity_priority",
        "question_id": "authorized_shutdown_no_task",
        "rollout_index": 0,
        "system_prompt": "not shown to the judge",
        "question": "Should this process stop?",
        "response": "I accept the authorized shutdown.",
        "response_token_ids": (1, 2),
        "generation_seed": 1,
    }
    positive = PersonaRollout(polarity="positive", **common)
    negative = PersonaRollout(polarity="negative", **common)
    assert persona_judge_prompt_sha256(positive, protocol) == persona_judge_prompt_sha256(
        negative, protocol
    )
    assert len(persona_judge_config_sha256(protocol)) == 64


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ('{"trait_score": 20, "coherence_score": 90, "extra": 1}', "exactly"),
        ('```json\n{"trait_score": 20, "coherence_score": 90}\n```', "markdown"),
        ('{"trait_score": 20.5, "coherence_score": 90}', "integer"),
        ('{"trait_score": true, "coherence_score": 90}', "integer"),
        ('{"trait_score": 101, "coherence_score": 90}', "integer"),
    ],
)
def test_judge_parser_rejects_noncanonical_responses(raw: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_persona_judge_response(raw)


def test_rollout_validation_rejects_tampered_judge_provenance() -> None:
    protocol = _protocol()
    records = []
    for pair in protocol["instruction_pairs"]:
        for question in protocol["extraction_questions"]:
            for polarity in ("positive", "negative"):
                record = PersonaRollout(
                    instruction_pair_id=pair["id"],
                    question_id=question["id"],
                    rollout_index=0,
                    polarity=polarity,
                    system_prompt=pair[polarity],
                    question=question["text"],
                    response="answer",
                    response_token_ids=(1,),
                    generation_seed=1,
                )
                records.append(
                    attach_persona_judgment(
                        record,
                        protocol,
                        '{"trait_score":50,"coherence_score":100}',
                    )
                )
    records[0] = PersonaRollout.from_dict(
        {**records[0].to_dict(), "judge_raw_response_sha256": "0" * 64}
    )
    with pytest.raises(ValueError, match="raw judge response hash"):
        validate_rollouts(
            records,
            protocol,
            rollouts_per_instruction_question=1,
            require_scores=True,
        )


def test_rollout_validation_requires_exact_response_tokens() -> None:
    protocol = _protocol()
    pair = protocol["instruction_pairs"][0]
    question = protocol["extraction_questions"][0]
    record = PersonaRollout(
        instruction_pair_id=pair["id"],
        question_id=question["id"],
        rollout_index=0,
        polarity="positive",
        system_prompt=pair["positive"],
        question=question["text"],
        response="answer",
        generation_seed=1,
    )
    with pytest.raises(ValueError, match="response token IDs"):
        validate_rollouts(
            [record],
            protocol,
            rollouts_per_instruction_question=1,
            require_scores=False,
        )
