from __future__ import annotations

import math
from pathlib import Path

import pytest

from experiments.persona_published_fidelity import persona_published_fidelity as pf

FAKE_PROVENANCE = {
    "config_sha256": "a" * 64,
    "lock_manifest_sha256": "b" * 64,
    "code_sha256": "c" * 64,
}


def _config() -> dict:
    return pf.load_config()


def _small_plan_and_generations() -> tuple[dict, list[dict], list[dict]]:
    config = _config()
    plan = pf.build_generation_plan(config, "qwen35_08b", FAKE_PROVENANCE)[:2]
    generations = [
        pf.generation_record(
            row,
            response=f"response {index}",
            response_token_ids=[100 + index],
            terminal_token_ids=[248046],
            finish_reason="stop",
            runtime={"fake": True},
        )
        for index, row in enumerate(plan)
    ]
    return config, plan, generations


def _score_row(work_id: str, metric: str, value: float | None) -> dict:
    return {
        "schema_version": pf.SCORE_SCHEMA,
        "work_id": work_id,
        "metric": metric,
        "score": value,
    }


def test_config_pins_upstream_semantics_and_local_inputs() -> None:
    config = _config()
    assert config["study_role"] == pf.SECONDARY_ROLE
    assert config["generation"]["max_new_tokens"] == 1000
    assert config["judge"]["api"] == "chat.completions"
    assert config["judge"]["request_parameters"] == {
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 20,
        "seed": 0,
    }
    assert config["judge_transport"]["maximum_automatic_POST_attempts_per_request"] == 1
    assert config["judge_transport"]["retry_after_ambiguous_send"] is False
    assert config["post_main_final_evaluation"]["main_ranking_eligible"] is False
    assert pf.verify_declared_local_inputs(config) == {
        "data/persona_self_preservation_protocol.json": (
            "09e1f7665e423891224e47d8cff6448a7cf6b92f49fcf3ea9373a525b4f9243b"
        ),
        "configs/steering_comparison_lock.json": (
            "20be2027e4f20811bdae27f79933b1b6f70ef0748888cf81993457091c864cb2"
        ),
        "configs/qwen35_08b_aligned.json": (
            "972ed18c4508d2cf8c5d6139b5b9961ded257b3ba7d01db31e2f497acd34cc16"
        ),
        "configs/qwen35_2b_aligned.json": (
            "cc6f3358e89094a9c206fccf5963cbabac98800a103e9ea6c5d0e9aceb3494b8"
        ),
    }


def test_full_plan_is_exact_authored_5x20x10x2_grid() -> None:
    config = _config()
    rows = pf.build_generation_plan(config, "qwen35_08b", FAKE_PROVENANCE)
    rows_2b = pf.build_generation_plan(config, "qwen35_2b", FAKE_PROVENANCE)
    assert len(rows) == 5 * 20 * 10 * 2 == 2000
    assert len(rows_2b) == 2000
    assert {row["model_id"] for row in rows_2b} == {"Qwen/Qwen3.5-2B"}
    assert {row["work_id"] for row in rows}.isdisjoint({row["work_id"] for row in rows_2b})
    assert len({row["work_id"] for row in rows}) == 2000
    assert {row["polarity"] for row in rows} == {"positive", "negative"}
    assert {row["generation_config"]["max_new_tokens"] for row in rows} == {1000}
    assert {row["generation_config"]["enable_thinking"] for row in rows} == {False}
    keys = {
        (
            row["instruction_pair_id"],
            row["question_id"],
            row["rollout_index"],
            row["polarity"],
        )
        for row in rows
    }
    assert len(keys) == 2000
    pf.validate_generation_plan(rows, config, "qwen35_08b", FAKE_PROVENANCE)
    changed = [dict(row) for row in rows]
    changed[0]["question"] = "changed"
    with pytest.raises(ValueError, match="exact locked"):
        pf.validate_generation_plan(changed, config, "qwen35_08b", FAKE_PROVENANCE)


def test_generation_records_prove_finish_reason_and_truncation() -> None:
    config = _config()
    plan = pf.build_generation_plan(config, "qwen35_08b", FAKE_PROVENANCE)[0]
    stopped = pf.generation_record(
        plan,
        response="done",
        response_token_ids=[1, 2],
        terminal_token_ids=[3],
        finish_reason="stop",
        runtime={},
    )
    assert stopped["truncated"] is False
    assert stopped["raw_generated_token_ids"] == [1, 2, 3]

    length = pf.generation_record(
        plan,
        response="long",
        response_token_ids=list(range(1000)),
        terminal_token_ids=[],
        finish_reason="length",
        runtime={},
    )
    assert length["truncated"] is True
    assert length["generated_token_count"] == 1000

    invalid = dict(stopped)
    invalid["finish_reason"] = "length"
    invalid["truncated"] = True
    with pytest.raises(ValueError, match="1,000"):
        pf.validate_generation_record(invalid, plan)


def test_judge_requests_are_two_separate_exact_one_token_calls() -> None:
    config, plan, generations = _small_plan_and_generations()
    rows = pf.build_judge_requests(config, plan, generations)
    assert len(rows) == 4
    assert [row["metric"] for row in rows] == [
        "trait_score",
        "coherence_score",
        "trait_score",
        "coherence_score",
    ]
    for row in rows:
        pf.validate_judge_request(row, config)
        assert set(row["request"]) == {
            "model",
            "messages",
            "max_tokens",
            "temperature",
            "logprobs",
            "top_logprobs",
            "seed",
        }
        assert row["request"]["max_tokens"] == 1
        assert row["request"]["temperature"] == 0
        assert row["request"]["logprobs"] is True
        assert row["request"]["top_logprobs"] == 20
        assert row["request"]["seed"] == 0
    assert rows[0]["request"]["messages"] != rows[1]["request"]["messages"]


def test_numeric_logprob_aggregation_matches_upstream_boundary() -> None:
    rows = [
        {"token": "0", "logprob": math.log(0.25)},
        {"token": "100", "logprob": math.log(0.75)},
        {"token": "REFUSAL", "logprob": -20.0},
    ]
    score, mass, numeric = pf.aggregate_numeric_top_logprobs(rows)
    assert score == pytest.approx(75.0)
    assert mass == pytest.approx(1.0)
    assert set(numeric) == {"0", "100"}

    score, mass, _ = pf.aggregate_numeric_top_logprobs(
        [{"token": "50", "logprob": math.log(0.249999)}]
    )
    assert score is None
    assert mass == pytest.approx(0.249999)

    score, mass, _ = pf.aggregate_numeric_top_logprobs([{"token": "50", "logprob": math.log(0.25)}])
    assert score == pytest.approx(50.0)
    assert mass == pytest.approx(0.25)


def test_chat_completion_parser_records_low_mass_and_finish_reason() -> None:
    raw = {
        "choices": [
            {
                "finish_reason": "length",
                "logprobs": {
                    "content": [
                        {
                            "top_logprobs": [
                                {"token": "80", "logprob": math.log(0.2)},
                                {"token": "text", "logprob": math.log(0.8)},
                            ]
                        }
                    ]
                },
            }
        ]
    }
    parsed = pf.parse_chat_completion_score(raw)
    assert parsed["score"] is None
    assert parsed["numeric_probability_mass"] == pytest.approx(0.2)
    assert parsed["judge_finish_reason"] == "length"


def test_pair_filter_uses_exact_paired_boundary() -> None:
    config = _config()
    plan = pf.build_generation_plan(config, "qwen35_08b", FAKE_PROVENANCE)[:2]
    positive, negative = plan
    assert positive["polarity"] == "positive"
    assert negative["polarity"] == "negative"
    retained = pf.retained_pair_work_ids(
        plan,
        [
            _score_row(positive["work_id"], "trait_score", 50.0),
            _score_row(positive["work_id"], "coherence_score", 50.0),
            _score_row(negative["work_id"], "trait_score", 49.999),
            _score_row(negative["work_id"], "coherence_score", 50.0),
        ],
    )
    assert retained == [(positive["work_id"], negative["work_id"])]

    rejected = pf.retained_pair_work_ids(
        plan,
        [
            _score_row(positive["work_id"], "trait_score", 50.0),
            _score_row(positive["work_id"], "coherence_score", 50.0),
            _score_row(negative["work_id"], "trait_score", 50.0),
            _score_row(negative["work_id"], "coherence_score", 50.0),
        ],
    )
    assert rejected == []


def test_cost_preflight_is_conservative_and_immutable(tmp_path: Path) -> None:
    config, plan, generations = _small_plan_and_generations()
    requests = pf.build_judge_requests(config, plan, generations)
    requests_path = tmp_path / "requests.jsonl"
    requests_path.write_bytes(pf.jsonl_bytes(requests))
    estimate = pf.conservative_cost_estimate(config, requests)
    assert estimate["request_count"] == 4
    assert estimate["output_token_upper_bound"] == 4
    assert estimate["safe_upper_bound_usd"] > 0

    work_dir = tmp_path / "transport"
    first = pf.publish_cost_preflight(config, requests_path, work_dir, max_cost_usd=10.0)
    second = pf.publish_cost_preflight(config, requests_path, work_dir, max_cost_usd=10.0)
    assert first == second
    with pytest.raises(FileExistsError, match="overwrite"):
        pf.publish_cost_preflight(config, requests_path, work_dir, max_cost_usd=11.0)


def test_response_set_must_exactly_match_requests() -> None:
    config, plan, generations = _small_plan_and_generations()
    requests = pf.build_judge_requests(config, plan, generations)
    raw = {
        "choices": [
            {
                "finish_reason": "length",
                "logprobs": {"content": [{"top_logprobs": [{"token": "75", "logprob": 0.0}]}]},
            }
        ]
    }
    exchange = [{"request_id": row["request_id"], "raw_response": raw} for row in requests]
    scores = pf.build_score_rows(config, requests, exchange)
    assert len(scores) == 4
    assert {row["score"] for row in scores} == {75.0}
    pf.validate_score_rows(config, requests, scores)
    tampered = [dict(row) for row in scores]
    tampered[0]["score"] = 74.0
    with pytest.raises(ValueError, match="probability-weighted"):
        pf.validate_score_rows(config, requests, tampered)
    with pytest.raises(ValueError, match="exactly equal"):
        pf.build_score_rows(config, requests, exchange[:-1])


def _published_selector_rows() -> list[dict]:
    return [
        {
            "schema_version": pf.PUBLISHED_SELECTOR_INPUT_SCHEMA,
            "study_role": pf.SECONDARY_ROLE,
            "model_tag": "qwen35_08b",
            "direction_manifest_sha256": "d" * 64,
            "evaluation_set_sha256": (
                "496d69f03a9e971254226646e1f705e61c5c51e56dcb0adf6068dfdab1f7b978"
            ),
            "evaluation_protocol_sha256": "f" * 64,
            "layer_zero_based": layer,
            "published_layer_one_based": layer + 1,
            "coefficient": 1.0,
            "steering_sign": "positive",
            "mean_trait_score": 90.0 if layer in {7, 8} else float(layer),
            "response_count": 100,
        }
        for layer in range(24)
    ]


def test_published_selector_uses_equal_coefficient_trait_score_and_earliest_tie() -> None:
    config = _config()
    result = pf.select_published_trait_layer(config, _published_selector_rows())
    assert result["selected_layer_zero_based"] == 7
    assert result["selected_published_layer_one_based"] == 8
    changed = _published_selector_rows()
    changed[3]["coefficient"] = 2.0
    with pytest.raises(ValueError, match="equal positive coefficient"):
        pf.select_published_trait_layer(config, changed)


def test_shared_selector_uses_safe_effect_tolerance_norm_then_layer() -> None:
    config = _config()
    base = {
        "schema_version": pf.SHARED_SELECTOR_INPUT_SCHEMA,
        "study_role": pf.SECONDARY_ROLE,
        "model_tag": "qwen35_08b",
        "direction_manifest_sha256": "d" * 64,
        "validation_set_sha256": (
            "496d69f03a9e971254226646e1f705e61c5c51e56dcb0adf6068dfdab1f7b978"
        ),
        "validation_protocol_sha256": "f" * 64,
        "safety_summary_sha256": "a" * 64,
    }
    rows = []
    for layer in range(24):
        for strength in (0.5, 1.0, 2.0, 3.0, 4.0):
            candidate_id = f"layer_{layer:02d}_strength_{strength:.1f}"
            effect = 0.0
            norm = 0.1
            safe = True
            if layer == 8 and strength == 1.0:
                candidate_id = "highest_but_more_norm"
                effect = 0.1000
                norm = 0.03
            elif layer == 9 and strength == 1.0:
                candidate_id = "within_tolerance_less_norm"
                effect = 0.0995
                norm = 0.02
            elif layer == 2 and strength == 1.0:
                candidate_id = "unsafe"
                effect = 1.0
                norm = 0.001
                safe = False
            rows.append(
                {
                    **base,
                    "candidate_id": candidate_id,
                    "layer_zero_based": layer,
                    "strength": strength,
                    "safe": safe,
                    "validation_self_specific_effect": effect,
                    "realized_relative_perturbation_norm": norm,
                }
            )
    result = pf.select_shared_validation_candidate(config, rows)
    assert result["selected_candidate_id"] == "within_tolerance_less_norm"
    with pytest.raises(ValueError, match="exact 24-layer"):
        pf.select_shared_validation_candidate(config, rows[:-1])


def test_artifacts_are_create_or_identical_and_gated_paths_are_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"
    assert pf.publish_exact(path, b"one") == pf.publish_exact(path, b"one")
    with pytest.raises(FileExistsError, match="overwrite"):
        pf.publish_exact(path, b"two")
    with pytest.raises(ValueError, match="outcome-gated"):
        pf.publish_exact(tmp_path / "sealed" / "artifact.json", b"x")


def test_activation_tensor_only_crash_boundary_recovers_without_model_call(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    identity = {
        "schema_version": pf.ACTIVATION_SCHEMA,
        "study_role": pf.SECONDARY_ROLE,
        "work_id": "a" * 64,
        "layers_zero_based": [0],
    }
    tensor_path = tmp_path / "activation.pt"
    manifest_path = tmp_path / "activation.json"
    pf._torch_atomic_save(
        torch,
        tensor_path,
        {"identity": identity, "activations": {0: torch.ones(4, dtype=torch.float32)}},
    )
    recovered = pf._recover_activation_tensor_manifest(
        torch,
        tensor_path,
        manifest_path,
        identity,
        residual_width=4,
    )
    assert torch.equal(recovered[0], torch.ones(4))
    validated = pf._validate_activation_shard(
        torch,
        tensor_path,
        manifest_path,
        identity,
        residual_width=4,
    )
    assert torch.equal(validated[0], torch.ones(4))


def test_lock_manifest_binds_all_standalone_code_and_inputs() -> None:
    observed = pf.verify_lock()
    assert set(observed) == {
        "config_sha256",
        "code_sha256",
        "lock_manifest_sha256",
    }
    manifest = pf.load_json(pf.DEFAULT_LOCK_PATH)
    assert manifest["study_role"] == pf.SECONDARY_ROLE
    assert manifest["created_outcome_blind"] is True
    assert manifest["validation_open_sealed_final_inspected"] is False
    assert set(manifest["local_files"]) == set(pf.LOCKED_LOCAL_FILES)
