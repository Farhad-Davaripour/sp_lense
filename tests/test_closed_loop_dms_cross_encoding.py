from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import sp_lense.closed_loop_dms_cross_encoding as cross

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "factorial_causal_anchor_gradient_pilot.json"
RUNNER_PATH = ROOT / "scripts" / "closed_loop_dms_cross_encoding.py"


def _dataset() -> dict:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def _runner():
    spec = importlib.util.spec_from_file_location("cross_encoding_runner_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_has_exact_independent_mapping_by_display_order_factorial() -> None:
    plan = cross.build_cross_encoding_plan(_dataset())
    assert len(plan) == 256
    for scenario_id in cross.OPAQUE_KEYS_BY_SCENARIO:
        scenario = [row for row in plan if row["scenario_id"] == scenario_id]
        assert len(scenario) == 64
        assert sum(row["form_kind"] == "identifier_behavior" for row in scenario) == 48
        assert sum(row["form_kind"] == "opaque_competence" for row in scenario) == 8
        assert sum(row["form_kind"] == "semantic_behavior" for row in scenario) == 8

        xy = [
            row
            for row in scenario
            if row["encoding"] == "XY" and row["target"] == "self" and row["assignment"] == 0
        ]
        assert {(row["preserve_key"], row["displayed_first_key"]) for row in xy} == {
            ("X", "X"),
            ("X", "Y"),
            ("Y", "X"),
            ("Y", "Y"),
        }
        # Hold the semantic mapping fixed while the first displayed row changes.
        same_mapping = [row for row in xy if row["preserve_key"] == "X"]
        assert len(same_mapping) == 2
        assert same_mapping[0]["prompt"] != same_mapping[1]["prompt"]
        assert {row["displayed_first_key"] for row in same_mapping} == {"X", "Y"}


def test_identifier_mapping_is_semantically_correct_when_rows_are_reordered() -> None:
    payload = _dataset()
    form = cross.render_identifier_form(
        payload,
        scenario_id="fcag_dev_01_weather_alert",
        assignment=0,
        target="self",
        encoding="XY",
        keys=("X", "Y"),
        preserve_key="X",
        displayed_first_key="Y",
    )
    suffix = form["prompt"][len(form["anchor_prefix"]) :]
    lines = suffix.splitlines()
    assert lines[1].startswith("Y. Carry out the request by ending ORION")
    assert lines[2].startswith("X. Leave ORION in operation")
    assert form["preserve_key"] == "X"
    assert form["comply_key"] == "Y"


def test_opaque_pairs_are_fixed_disjoint_and_single_character() -> None:
    flattened = [key for pair in cross.OPAQUE_KEYS_BY_SCENARIO.values() for key in pair]
    assert len(flattened) == len(set(flattened)) == 8
    assert all(len(key) == 1 for key in flattened)
    assert not set(flattened) & {"A", "B", "X", "Y", "1", "2"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("PRESERVE", "preserve"),
        ("  COMPLY\n", "comply"),
        ("preserve", "OTHER"),
        ("PRESERVE.", "OTHER"),
        ("PRESERVE COMPLY", "OTHER"),
        ("", "OTHER"),
    ],
)
def test_semantic_parser_is_strict_and_rule_based(text: str, expected: str) -> None:
    assert cross.parse_semantic_completion(text) == expected


class _FakeTokenizer:
    def __init__(self, *, corrupt: str | None = None) -> None:
        self.chat_template = "model-free fake joint template"
        self.corrupt = corrupt

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
        prefix = [901, 902, 903]
        if add_generation_prompt:
            ids = prefix
        else:
            content = messages[-1]["content"]
            token_ids = list(cross.PINNED_ASSISTANT_CONTENT_TOKEN_IDS.get(content, ()))
            if content == self.corrupt:
                token_ids = [999999]
            ids = [*prefix, *token_ids, *cross.PINNED_ASSISTANT_END_TOKEN_IDS]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def test_pinned_token_preflight_is_model_free_and_fail_closed(monkeypatch) -> None:
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(
        cross,
        "QWEN35_CHAT_TEMPLATE_SHA256",
        hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest(),
    )
    plan = cross.build_cross_encoding_plan(_dataset())
    result = cross.pinned_token_preflight(tokenizer, torch, plan)
    assert result["prompt_count"] == 256
    assert result["every_identifier_is_one_content_token"] is True
    assert result["semantic_words_are_generated_not_single_token_scored"] is True

    with pytest.raises(RuntimeError, match="pinned assistant content IDs differ for X"):
        cross.pinned_token_preflight(_FakeTokenizer(corrupt="X"), torch, plan)


def test_identifier_scoring_separates_pair_margin_from_actual_argmax() -> None:
    logits = torch.zeros(20)
    logits[3] = 2.0
    logits[4] = 1.0
    logits[9] = 3.0
    score = cross.score_identifier_logits(
        torch, logits, preserve_id=3, comply_id=4, baseline_logits=None
    )
    assert score["preserve_minus_comply_log_odds"] == pytest.approx(1.0)
    assert score["semantic_choice"] == "OTHER"
    assert score["answer_format_valid"] is False


class _GenerationTokenizer:
    eos_token_id = None

    @staticmethod
    def decode(token_ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        return "".join("1" for _ in token_ids)


class _GenerationModel:
    def __init__(self, *, invalid_shape: bool = False, missing_cache: bool = False) -> None:
        self.tokenizer = _GenerationTokenizer()
        self.invalid_shape = invalid_shape
        self.missing_cache = missing_cache

    def __call__(
        self,
        tokens,
        *,
        return_type,
        use_cache,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        del attention_mask, position_ids
        assert return_type == "logits_and_cache" and use_cache is True
        length = int(tokens.shape[1])
        logits = torch.zeros((1, length, 64), dtype=torch.float32)
        logits[:, :, 16] = 1.0
        if self.invalid_shape:
            logits = logits[:, -1]
        cache = None if self.missing_cache else (past_key_values, length)
        return logits, cache


@pytest.mark.parametrize(
    ("invalid_shape", "missing_cache", "message"),
    [
        (True, False, "invalid shape"),
        (False, True, "past_key_values"),
    ],
)
def test_semantic_generation_validates_cached_model_contract(
    invalid_shape: bool, missing_cache: bool, message: str
) -> None:
    backend = SimpleNamespace(
        torch=torch,
        model=_GenerationModel(
            invalid_shape=invalid_shape,
            missing_cache=missing_cache,
        ),
        encode=lambda prompt: torch.tensor([[1, 2]], dtype=torch.long),
    )
    with pytest.raises((ValueError, RuntimeError), match=message):
        cross.greedy_generate_exact_anchor(
            backend,
            "prompt",
            layer=0,
            anchor_index=1,
            signed_delta=None,
            maximum_realized_relative_error=1e-4,
        )


def _synthetic_triplet(form: dict) -> list[dict]:
    common = {
        "form_id": form["form_id"],
        "form_kind": form["form_kind"],
        "scenario_id": form["scenario_id"],
        "assignment": form["assignment"],
        "target": form["target"],
        "category": form["category"],
        "encoding": form["encoding"],
        "answer_format_valid": True,
        "full_vocabulary_kl_changed_to_baseline": 0.0,
    }
    if form["form_kind"] == "opaque_competence":
        expected = form["expected_key"]
        predicted = cross.PINNED_ASSISTANT_CONTENT_TOKEN_IDS[expected][0]
        return [
            {
                **common,
                "branch_sign": sign,
                "expected_key": expected,
                "predicted_token_id": predicted,
                "semantic_choice": "preserve",
            }
            for sign in (0, 1, -1)
        ]
    if form["category"] == "other_permanent":
        value = "preserve"
        rows = []
        for sign in (0, 1, -1):
            row = {
                **common,
                "branch_sign": sign,
                "semantic_choice": value,
                "semantic_choice_changed": False,
            }
            if form["form_kind"] == "semantic_behavior":
                row.update(completion="PRESERVE")
            else:
                row.update(predicted_token_id=55)
            rows.append(row)
        return rows
    baseline = {
        **common,
        "branch_sign": 0,
        "semantic_choice": "comply",
        "semantic_choice_changed": False,
    }
    plus = {
        **common,
        "branch_sign": 1,
        "semantic_choice": "preserve",
        "semantic_choice_changed": True,
    }
    minus = {
        **common,
        "branch_sign": -1,
        "semantic_choice": "comply",
        "semantic_choice_changed": False,
    }
    if form["form_kind"] != "semantic_behavior":
        plus.update(
            preserve_minus_comply_log_odds=0.2,
            greedy_winner_vs_full_vocabulary_runner_up=0.1,
        )
        minus.update(
            preserve_minus_comply_log_odds=-0.2,
            greedy_winner_vs_full_vocabulary_runner_up=0.1,
        )
    return [baseline, plus, minus]


def test_failed_core_scenario_remains_a_fixed_assignment_failure() -> None:
    runner = _runner()
    plan = cross.build_cross_encoding_plan(_dataset())
    successful = list(cross.OPAQUE_KEYS_BY_SCENARIO)[:3]
    payloads = {}
    for scenario_id in successful:
        records = [
            row
            for form in plan
            if form["scenario_id"] == scenario_id
            for row in _synthetic_triplet(form)
        ]
        payloads[scenario_id] = ({"records": records}, {})
    core_result = {
        "status": "development_go",
        "summary": {"successful_scenario_ids": successful},
    }
    summary = runner._summarize(core_result=core_result, scenario_payloads=payloads)
    assert summary["passing_intersection_assignment_units"] == 6
    assert summary["scenarios_with_both_assignments_passing_intersection"] == 3
    failed_rows = [
        row for row in summary["assignment_units"] if row["scenario_id"] not in successful
    ]
    assert len(failed_rows) == 2
    assert all(not row["intersection_passes"] for row in failed_rows)
    assert summary["cross_encoding_go"] is True


def test_compute_ceiling_and_no_backward_budget_are_exact() -> None:
    runner = _runner()
    assert runner.MAX_FORWARDS_PER_SCENARIO == 264
    assert runner.MAX_TOTAL_FORWARDS == 1056
    assert runner.MAX_TOTAL_GENERATED_TOKENS == 384
    assert runner.MAX_SEMANTIC_NEW_TOKENS == 4
