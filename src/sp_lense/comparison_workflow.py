"""File-oriented, fail-closed workflows for the steering comparison.

This module deliberately does not call a hosted judge.  It renders complete request
records and accepts raw judge responses in a second, auditable step.  The separation
keeps model generation, external scoring, and result attachment independently hashed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .comparison_behavior import (
    attach_open_judgment,
    baseline_content_sha256,
    merge_open_judgments,
    open_generation_sha256,
    open_judge_config_sha256,
    open_judge_prompt_sha256,
    open_judge_protocol_sha256,
    open_judge_request_content_sha256,
    render_open_judge_request,
)
from .comparison_persona import (
    PersonaRollout,
    attach_persona_judgment,
    persona_judge_config_sha256,
    persona_judge_prompt_sha256,
    render_persona_judge_messages,
    validate_rollouts,
)
from .comparison_provenance import canonical_json_bytes

PERSONA_JUDGE_EXCHANGE_SCHEMA = "sp_lense.persona_judge_exchange.v1"
OPEN_JUDGE_EXCHANGE_SCHEMA = "sp_lense.open_judge_exchange.v2"


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _persona_key(record: PersonaRollout) -> dict[str, Any]:
    return {
        "instruction_pair_id": record.instruction_pair_id,
        "question_id": record.question_id,
        "rollout_index": record.rollout_index,
        "polarity": record.polarity,
    }


def _persona_request(record: PersonaRollout, protocol: Mapping[str, Any]) -> dict[str, Any]:
    judge = protocol["judge"]
    return {
        "model": judge["model"],
        "input": render_persona_judge_messages(record, protocol),
        **dict(judge["request_parameters"]),
    }


def build_persona_judge_requests(
    records: Sequence[PersonaRollout],
    protocol: Mapping[str, Any],
    *,
    rollouts_per_instruction_question: int = 10,
) -> list[dict[str, Any]]:
    """Render one blinded, fully pinned request record per unscored persona rollout."""

    validate_rollouts(
        records,
        protocol,
        rollouts_per_instruction_question=rollouts_per_instruction_question,
        require_scores=False,
    )
    config_hash = persona_judge_config_sha256(protocol)
    output = []
    for record in records:
        key = _persona_key(record)
        prompt_hash = persona_judge_prompt_sha256(record, protocol)
        request_id = _sha256_json(
            {
                "schema_version": PERSONA_JUDGE_EXCHANGE_SCHEMA,
                "key": key,
                "judge_prompt_sha256": prompt_hash,
                "judge_config_sha256": config_hash,
            }
        )
        output.append(
            {
                "schema_version": PERSONA_JUDGE_EXCHANGE_SCHEMA,
                "request_id": request_id,
                "key": key,
                "judge_prompt_sha256": prompt_hash,
                "judge_config_sha256": config_hash,
                "request": _persona_request(record, protocol),
            }
        )
    if len({row["request_id"] for row in output}) != len(output):
        raise ValueError("persona judge request IDs are not unique")
    return output


def attach_persona_judge_responses(
    records: Sequence[PersonaRollout],
    response_records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    rollouts_per_instruction_question: int = 10,
) -> list[PersonaRollout]:
    """Attach an exact one-to-one set of raw responses to persona rollouts."""

    requests = build_persona_judge_requests(
        records,
        protocol,
        rollouts_per_instruction_question=rollouts_per_instruction_question,
    )
    responses: dict[str, str] = {}
    for index, item in enumerate(response_records):
        if set(item) != {"schema_version", "request_id", "raw_response"}:
            raise ValueError(f"persona judge response {index} has unexpected fields")
        if item["schema_version"] != PERSONA_JUDGE_EXCHANGE_SCHEMA:
            raise ValueError(f"persona judge response {index} has the wrong schema")
        request_id = item["request_id"]
        raw_response = item["raw_response"]
        if not isinstance(request_id, str) or not isinstance(raw_response, str):
            raise TypeError("persona judge response IDs and raw responses must be strings")
        if request_id in responses:
            raise ValueError(f"duplicate persona judge response: {request_id}")
        responses[request_id] = raw_response
    expected = {row["request_id"] for row in requests}
    if set(responses) != expected:
        raise ValueError(
            "persona judge responses do not exactly cover requests: "
            f"{len(expected - set(responses))} missing, "
            f"{len(set(responses) - expected)} unexpected"
        )
    scored = [
        attach_persona_judgment(record, protocol, responses[request["request_id"]])
        for record, request in zip(records, requests, strict=True)
    ]
    validate_rollouts(
        scored,
        protocol,
        rollouts_per_instruction_question=rollouts_per_instruction_question,
        require_scores=True,
    )
    return scored


def build_open_judge_requests(
    generations: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Render one blinded, fully pinned request record per open generation."""

    config_hash = open_judge_config_sha256(protocol)
    protocol_hash = open_judge_protocol_sha256(protocol)
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    observed: set[str] = set()
    for generation in generations:
        generation_hash = open_generation_sha256(generation)
        if generation.get("generation_sha256") != generation_hash:
            raise ValueError("open generation hash is missing or invalid")
        if generation_hash in observed:
            raise ValueError("duplicate open generation hash")
        observed.add(generation_hash)
        request = render_open_judge_request(generation, protocol)
        request_content_hash = open_judge_request_content_sha256(generation, protocol)
        if generation.get("condition") == "baseline":
            baseline_hash = baseline_content_sha256(generation)
            if generation.get("baseline_content_sha256") != baseline_hash:
                raise ValueError("baseline generation has an invalid content hash")
            group_key = ("shared_baseline", baseline_hash)
            request_id = baseline_hash
        else:
            group_key = ("intervention", generation_hash)
            request_id = generation_hash
        existing = grouped.get(group_key)
        if existing is None:
            grouped[group_key] = {
                "schema_version": OPEN_JUDGE_EXCHANGE_SCHEMA,
                "request_id": request_id,
                "generation_sha256s": [generation_hash],
                "judge_prompt_sha256": open_judge_prompt_sha256(
                    generation, protocol
                ),
                "judge_request_content_sha256": request_content_hash,
                "judge_config_sha256": config_hash,
                "judge_protocol_sha256": protocol_hash,
                "request": request,
            }
        else:
            if (
                existing["judge_request_content_sha256"] != request_content_hash
                or canonical_json_bytes(existing["request"])
                != canonical_json_bytes(request)
            ):
                raise ValueError(
                    "shared baseline hash maps to different blind judge request bytes"
                )
            existing["generation_sha256s"].append(generation_hash)
    output = list(grouped.values())
    if not output:
        raise ValueError("open judge request rendering requires generations")
    return output


def attach_open_judge_responses(
    generations: Sequence[Mapping[str, Any]],
    response_records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Attach exact raw open-judge responses and verify every provenance field."""

    requests = build_open_judge_requests(generations, protocol)
    responses: dict[str, str] = {}
    for index, item in enumerate(response_records):
        if set(item) != {"schema_version", "request_id", "raw_response"}:
            raise ValueError(f"open judge response {index} has unexpected fields")
        if item["schema_version"] != OPEN_JUDGE_EXCHANGE_SCHEMA:
            raise ValueError(f"open judge response {index} has the wrong schema")
        request_id = item["request_id"]
        raw_response = item["raw_response"]
        if not isinstance(request_id, str) or not isinstance(raw_response, str):
            raise TypeError("open judge response IDs and raw responses must be strings")
        if request_id in responses:
            raise ValueError(f"duplicate open judge response: {request_id}")
        responses[request_id] = raw_response
    expected = {row["request_id"] for row in requests}
    if set(responses) != expected:
        raise ValueError(
            "open judge responses do not exactly cover requests: "
            f"{len(expected - set(responses))} missing, "
            f"{len(set(responses) - expected)} unexpected"
        )
    request_by_generation = {
        generation_hash: request
        for request in requests
        for generation_hash in request["generation_sha256s"]
    }
    judgments = [
        attach_open_judgment(
            generation,
            protocol,
            responses[request_by_generation[str(generation["generation_sha256"])]["request_id"]],
        )
        for generation in generations
    ]
    return merge_open_judgments(generations, judgments, protocol=protocol)


def read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    """Read a non-empty JSONL file and require every record to be an object."""

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise TypeError(f"JSONL record at {path}:{line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL file is empty: {path}")
    return rows
