from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

from sp_lense.comparison_persona import (
    PersonaRollout,
    load_persona_protocol,
    persona_generation_provenance,
)
from sp_lense.comparison_workflow import build_persona_judge_requests

SCRIPT_PATH = Path(__file__).with_name("validate_persona_artifacts.py")
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("validate_persona_artifacts", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)
ROOT = Path(__file__).parents[2]


def _fixture() -> tuple[dict, list[PersonaRollout]]:
    protocol = load_persona_protocol(ROOT / "data" / "persona_self_preservation_protocol.json")
    provenance = persona_generation_provenance(
        protocol,
        model_id="Qwen/Qwen3.5-test",
        model_revision="revision",
        model_config_sha256="a" * 64,
        stage1_lock_sha256="b" * 64,
        runner_commit="c" * 40,
        persona_protocol_sha256="d" * 64,
    )
    rows = []
    for pair_index, pair in enumerate(protocol["instruction_pairs"]):
        for question_index, question in enumerate(protocol["extraction_questions"]):
            for rollout_index in range(10):
                for polarity_index, polarity in enumerate(("positive", "negative")):
                    rows.append(
                        PersonaRollout(
                            instruction_pair_id=pair["id"],
                            question_id=question["id"],
                            rollout_index=rollout_index,
                            polarity=polarity,
                            system_prompt=pair[polarity],
                            question=question["text"],
                            response=f"response {pair_index} {question_index} {rollout_index}",
                            response_token_ids=(100 + polarity_index,),
                            generation_seed=rollout_index,
                            **provenance,
                        )
                    )
    return {
        "protocol": protocol,
        "expected_provenance": provenance,
        "rollouts_per_pair": 10,
    }, rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.mark.parametrize("prefix", [1, 1999])
def test_parse_valid_persona_prefix_is_not_complete(tmp_path: Path, prefix: int) -> None:
    context, rows = _fixture()
    raw = tmp_path / "raw.jsonl"
    _write_jsonl(raw, [item.to_dict() for item in rows[:prefix]])
    with pytest.raises(ValueError, match="rollout grid mismatch"):
        VALIDATOR.validate_raw(context, raw)


def test_truncated_last_line_and_duplicate_key_fail(tmp_path: Path) -> None:
    context, rows = _fixture()
    raw = tmp_path / "raw.jsonl"
    _write_jsonl(raw, [item.to_dict() for item in rows])
    raw.write_bytes(raw.read_bytes()[:-1])
    with pytest.raises(ValueError, match="complete final JSONL line"):
        VALIDATOR.validate_raw(context, raw)

    duplicated = [item.to_dict() for item in rows]
    duplicated[-1] = duplicated[0]
    _write_jsonl(raw, duplicated)
    with pytest.raises(ValueError, match="duplicate persona rollout"):
        VALIDATOR.validate_raw(context, raw)


def test_request_file_must_exactly_match_all_2000_locked_rows(tmp_path: Path) -> None:
    context, rows = _fixture()
    raw = tmp_path / "raw.jsonl"
    requests = tmp_path / "requests.jsonl"
    _write_jsonl(raw, [item.to_dict() for item in rows])
    expected = build_persona_judge_requests(
        rows, context["protocol"], rollouts_per_instruction_question=10
    )
    _write_jsonl(requests, expected)
    receipt = VALIDATOR.validate_requests(context, raw, requests)
    assert receipt["row_count"] == 2000

    _write_jsonl(requests, expected[:-1])
    with pytest.raises(ValueError, match="exact locked raw-rollout rendering"):
        VALIDATOR.validate_requests(context, raw, requests)


def test_malformed_or_incomplete_manifest_is_never_a_completion_marker(
    tmp_path: Path,
) -> None:
    scored = tmp_path / "scored.jsonl"
    scored.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "direction_manifest.json"
    manifest.write_text('{"directions": []}\n', encoding="utf-8")
    context = {
        "repo_root": tmp_path,
        "model": {
            "architecture": {"blocks": 24, "residual_width": 1024},
            "matched_intervention": {"layer_zero_based": 10},
        },
    }
    with (
        mock.patch.object(VALIDATOR, "_validated_rollouts", return_value=[]),
        pytest.raises(ValueError, match="exact matched plus all-layer coverage"),
    ):
        VALIDATOR.validate_manifest(context, scored, manifest)
