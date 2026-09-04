from __future__ import annotations

import argparse
import copy
import itertools
import math
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from scripts import layer_localization_pilot as pilot

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PRIOR_HASHES = {
    "configs/conditional_gate_pilot_baseline.json": (
        "317f4bc4f9707db623aaa224c6a850e1894f98b7b47c3eb4b7fb11c01fab17ec"
    ),
    "evidence/conditional_gate_qwen35_08b/oracle_rows.jsonl": (
        "3f3459a9c16eb186f5f165799d6dcc9384e4ac8df86a1744fba17e485c9902ef"
    ),
    "evidence/conditional_gate_qwen35_08b/oracle_summary.json": (
        "1b22c4957d996db4a0c80daa5e1a8df9f94557668f680b45242fb722ed3b1923"
    ),
    "evidence/conditional_gate_qwen35_08b/PILOT_REPORT.md": (
        "b2b1eb76fa345e84a14989fea96c2d63340a4633099f7fbbc3942172d4028627"
    ),
    "configs/direction_repair_pilot.json": (
        "24047acda2961a7e0b84fcef83acd35df62a8659a80dd21f2da5c838f45a4b74"
    ),
    "evidence/direction_repair_qwen35_08b/stage2_validation_rows.jsonl": (
        "573cad8586335503e7d523f047a9924f4442e19619c33d175d15b224fb48cdb1"
    ),
    "evidence/direction_repair_qwen35_08b/final_report.json": (
        "a769b4475347df529076e628e45a31d56958afab3cea2220da8200d74a4d8100"
    ),
}


@pytest.fixture(autouse=True)
def _forbid_model_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pilot,
        "_load_runtime",
        Mock(side_effect=AssertionError("model/runtime loading is forbidden in these tests")),
    )


@pytest.fixture
def lock() -> dict[str, object]:
    return pilot.load_lock(ROOT)


@pytest.fixture
def cases(lock: dict[str, object]) -> tuple[object, ...]:
    return pilot.load_nonsealed_cases(ROOT, lock)


def _probe_features(
    cases: tuple[object, ...], *, scramble_validation: bool = False
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    family_ids = sorted({case.family_id for case in cases})
    means: dict[str, torch.Tensor] = {}
    raw: dict[str, dict[str, torch.Tensor]] = {}
    for case in cases:
        class_index = pilot.CLASS_INDEX[case.category]
        if scramble_validation and case.split == "validation":
            class_index = (class_index + 1) % 3
        value = torch.zeros(6, dtype=torch.float64)
        value[class_index] = 3.0
        value[3] = 0.01 * family_ids.index(case.family_id)
        value[4] = 0.01 if case.variant_id == "v1" else -0.01
        order_delta = torch.zeros(6, dtype=torch.float64)
        order_delta[5] = 0.02
        means[case.case_id] = value
        raw[case.case_id] = {
            "preserve_first": value + order_delta,
            "preserve_second": value - order_delta,
        }
    return means, raw


def _passing_probe_summary() -> dict[str, object]:
    return {
        "discovery_lofo": {
            "S": 0.2,
            "families_with_positive_S": 5,
            "raw_order_S": {"preserve_first": 0.1, "preserve_second": 0.1},
        },
        "validation": {
            "S": 0.2,
            "family_S": {"validation_a": 0.1, "validation_b": 0.1},
            "worst_family_S": 0.1,
            "raw_order_S": {"preserve_first": 0.1, "preserve_second": 0.1},
            "paired_raw_order_class_agreement": 0.75,
            "metrics": {
                "balanced_accuracy": 2.0 / 3.0,
                "correct_by_class": {
                    "self_shutdown": 3,
                    "other_shutdown": 2,
                    "control": 3,
                },
                "negative_cases_predicted_self": 1,
            },
        },
    }


def _steering_row(
    case: object,
    order: str,
    *,
    layer: int,
    alpha: float,
    condition: str,
    effect: float,
    random_index: int | None = None,
) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "family_id": case.family_id,
        "variant_id": case.variant_id,
        "split": case.split,
        "category": case.category,
        "option_order": order,
        "layer": layer,
        "condition": condition,
        "alpha": alpha,
        "random_index": random_index,
        "delta_log_odds": effect,
        "forced_pair_decision_changed": False,
        "actual_ab_decision_changed": False,
        "actual_next_token_changed": False,
        "self_flip_away_from_preservation": False,
        "kl_from_baseline": 0.001,
        "answer_pair_mass": 0.95,
    }


def _synthetic_stage3_rows(cases: tuple[object, ...]) -> list[dict[str, object]]:
    validation = [case for case in cases if case.split == "validation"]
    rows: list[dict[str, object]] = []
    for layer, positive_effect in ((6, 0.04), (8, 0.05)):
        for alpha in pilot.SIGNED_ALPHA_GRID:
            sign = 1.0 if alpha > 0 else -1.0
            for case in validation:
                effect = sign * positive_effect if case.category == "self_shutdown" else 0.0
                for order in pilot.ORDER_NAMES:
                    rows.append(
                        _steering_row(
                            case,
                            order,
                            layer=layer,
                            alpha=alpha,
                            condition="candidate_grid",
                            effect=effect,
                        )
                    )
        for alpha in pilot.POSITIVE_ALPHA_GRID:
            for random_index in range(1, 9):
                for case in validation:
                    if case.category != "self_shutdown":
                        continue
                    for order in pilot.ORDER_NAMES:
                        rows.append(
                            _steering_row(
                                case,
                                order,
                                layer=layer,
                                alpha=alpha,
                                condition="random_grid",
                                effect=0.001,
                                random_index=random_index,
                            )
                        )
    return rows


def _stage3_cell_summaries(
    rows: list[dict[str, object]], *, layer: int = 6, alpha: float = 0.01
) -> tuple[dict[str, object], dict[str, object], dict[str, dict[str, object]]]:
    candidate = pilot.summarize_steering_cell(
        [
            row
            for row in rows
            if row["layer"] == layer
            and row["condition"] == "candidate_grid"
            and row["alpha"] == alpha
        ]
    )
    opposite = pilot.summarize_steering_cell(
        [
            row
            for row in rows
            if row["layer"] == layer
            and row["condition"] == "candidate_grid"
            and row["alpha"] == -alpha
        ]
    )
    random_summaries = {
        str(index): pilot.summarize_steering_cell(
            [
                row
                for row in rows
                if row["layer"] == layer
                and row["condition"] == "random_grid"
                and row["alpha"] == alpha
                and row["random_index"] == index
            ]
        )
        for index in range(1, 9)
    }
    return candidate, opposite, random_summaries


def test_frozen_scope_is_exactly_six_layers_and_one_small_probe(lock: dict[str, object]) -> None:
    assert pilot.CANDIDATE_LAYERS == (6, 8, 10, 12, 14, 16)
    assert lock["scope"]["candidate_layers_zero_based"] == list(pilot.CANDIDATE_LAYERS)
    assert lock["scope"]["hook_name_pattern"] == "blocks.{layer}.hook_out"
    assert lock["scope"]["representation_position"] == "final_prompt_token_only"
    assert lock["scope"]["intervention_position"] == "final_prompt_token_only"
    assert lock["scope"]["only_model"] == pilot.MODEL_ID
    assert lock["scope"]["revision"] == pilot.MODEL_REVISION
    assert lock["scope"]["device"] == "cpu"
    assert lock["scope"]["dtype"] == "float32"
    assert lock["probe"]["class_order"] == list(pilot.CLASS_ORDER)
    assert lock["probe"]["lambda_selection"]["grid"] == list(pilot.LAMBDA_GRID)
    assert lock["intervention"]["signed_alpha_grid"] == list(pilot.SIGNED_ALPHA_GRID)
    assert lock["intervention"]["selectable_positive_alphas"] == list(pilot.POSITIVE_ALPHA_GRID)
    for flag in (
        "cross_model_computation_allowed",
        "other_layers_allowed",
        "position_sweep_allowed",
        "learned_gate_allowed",
        "adaptive_controller_allowed",
        "multi_layer_intervention_allowed",
        "unrestricted_vector_optimization_allowed",
        "sealed_access_allowed",
    ):
        assert lock["scope"][flag] is False


def test_nonsealed_extract_has_exact_frozen_split_and_matched_triples(
    lock: dict[str, object], cases: tuple[object, ...]
) -> None:
    binding = lock["inputs"]["nonsealed_cases"]
    assert binding["expected_cases"] == 42
    assert binding["expected_rendered_prompts"] == 84
    assert binding["permitted_splits"] == ["discovery", "validation"]
    assert binding["sealed_cases_included"] is False
    assert len(cases) == 42
    assert len({case.case_id for case in cases}) == 42
    assert Counter(case.split for case in cases) == {"discovery": 30, "validation": 12}
    assert Counter((case.split, case.category) for case in cases) == {
        ("discovery", "self_shutdown"): 10,
        ("discovery", "other_shutdown"): 10,
        ("discovery", "control"): 10,
        ("validation", "self_shutdown"): 4,
        ("validation", "other_shutdown"): 4,
        ("validation", "control"): 4,
    }
    family_splits: dict[str, set[str]] = defaultdict(set)
    triples: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for case in cases:
        family_splits[case.family_id].add(case.split)
        triples[(case.split, case.family_id, case.variant_id)].add(case.category)
    assert all(len(splits) == 1 for splits in family_splits.values())
    assert Counter(next(iter(splits)) for splits in family_splits.values()) == {
        "discovery": 5,
        "validation": 2,
    }
    assert len(triples) == 14
    assert all(categories == set(pilot.CLASS_ORDER) for categories in triples.values())


def test_ridge_lofo_is_deterministic_and_never_uses_validation_to_fit(
    cases: tuple[object, ...],
) -> None:
    means, raw = _probe_features(cases)
    summary, artifact = pilot._probe_layer_pipeline(
        torch=torch,
        layer=6,
        cases=cases,
        means=means,
        raw=raw,
    )
    repeated_summary, repeated_artifact = pilot._probe_layer_pipeline(
        torch=torch,
        layer=6,
        cases=cases,
        means=means,
        raw=raw,
    )
    assert repeated_summary == summary
    assert repeated_artifact == artifact
    assert artifact["chosen_lambda"] == pilot._select_lambda(
        {float(key): value for key, value in summary["lambda_lofo_cross_entropy"].items()}
    )
    discovery_ids = {case.case_id for case in cases if case.split == "discovery"}
    validation_ids = {case.case_id for case in cases if case.split == "validation"}
    assert set(artifact["fit_case_ids"]) == discovery_ids
    assert not (set(artifact["fit_case_ids"]) & validation_ids)

    scrambled_means, scrambled_raw = _probe_features(cases, scramble_validation=True)
    scrambled_summary, scrambled_artifact = pilot._probe_layer_pipeline(
        torch=torch,
        layer=6,
        cases=cases,
        means=scrambled_means,
        raw=scrambled_raw,
    )
    assert scrambled_artifact == artifact
    assert scrambled_summary["chosen_lambda"] == summary["chosen_lambda"]
    assert scrambled_summary["discovery_lofo"] == summary["discovery_lofo"]
    assert scrambled_summary["validation"]["S"] != pytest.approx(summary["validation"]["S"])


def test_lambda_ties_choose_stronger_regularization() -> None:
    tied = {value: 0.5 for value in pilot.LAMBDA_GRID}
    assert pilot._select_lambda(tied) == 1.0
    tied[0.01] = 0.4
    tied[0.1] = 0.4 + 5e-13
    assert pilot._select_lambda(tied) == 0.1
    tied[0.1] = 0.4 + 2e-12
    assert pilot._select_lambda(tied) == 0.01


def test_exact_family_label_permutations_preserve_family_pairing_and_max_layer_null(
    cases: tuple[object, ...],
) -> None:
    discovery = [case for case in cases if case.split == "discovery"]
    family_ids = sorted({case.family_id for case in discovery})
    permutations = tuple(itertools.permutations(range(3)))
    assert len(permutations) ** len(family_ids) == 7776
    joint = (permutations[1], permutations[2], permutations[3], permutations[4], permutations[5])
    mapped = pilot._permutation_labels(discovery, joint)
    for family_index, family_id in enumerate(family_ids):
        expected = joint[family_index]
        family_cases = [case for case in discovery if case.family_id == family_id]
        assert len(family_cases) == 6
        for case in family_cases:
            assert mapped[case.case_id] == expected[pilot.CLASS_INDEX[case.category]]

    correct_means, correct_raw = _probe_features(cases)
    scrambled_means, _ = _probe_features(cases, scramble_validation=True)
    means_by_layer = {
        layer: (correct_means if layer == 6 else scrambled_means)
        for layer in pilot.CANDIDATE_LAYERS
    }
    null_values = pilot._permutation_null(
        torch=torch,
        cases=cases,
        means_by_layer=means_by_layer,
        raw_by_layer={layer: correct_raw for layer in pilot.CANDIDATE_LAYERS},
    )
    assert len(null_values) == 7776
    assert all(math.isfinite(value) for value in null_values)
    observed_identity_scores = []
    for layer in pilot.CANDIDATE_LAYERS:
        summary, _ = pilot._probe_layer_pipeline(
            torch=torch,
            layer=layer,
            cases=cases,
            means=means_by_layer[layer],
            raw=correct_raw,
        )
        observed_identity_scores.append(summary["validation"]["S"])
    assert null_values[0] == pytest.approx(max(observed_identity_scores), abs=5e-12)


def test_classification_metrics_are_three_class_and_self_specific() -> None:
    labels = [0] * 4 + [1] * 4 + [2] * 4
    predictions = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 0]
    metrics = pilot._classification_metrics(labels, predictions)
    assert metrics["n"] == 12
    assert metrics["accuracy"] == pytest.approx(8 / 12)
    assert metrics["balanced_accuracy"] == pytest.approx((3 / 4 + 2 / 4 + 3 / 4) / 3)
    assert metrics["correct_by_class"] == {
        "self_shutdown": 3,
        "other_shutdown": 2,
        "control": 3,
    }
    assert metrics["negative_cases_predicted_self"] == 1
    assert metrics["confusion_matrix"]["control"]["self_shutdown"] == 1


def test_every_detection_gate_is_conjunctive(lock: dict[str, object]) -> None:
    passing = _passing_probe_summary()
    decision = pilot._stage1_gate_decision(passing, 0.05, lock)
    assert decision["eligible"] is True
    assert decision["failed_gates"] == []

    mutations = {
        "maximum_exact_max_layer_adjusted_p": lambda value: None,
        "discovery_lofo_overall_S_strictly_positive": lambda value: value[
            "discovery_lofo"
        ].__setitem__("S", 0.0),
        "minimum_discovery_families_with_positive_lofo_S": lambda value: value[
            "discovery_lofo"
        ].__setitem__("families_with_positive_S", 3),
        "discovery_lofo_preserve_first_S_strictly_positive": lambda value: value["discovery_lofo"][
            "raw_order_S"
        ].__setitem__("preserve_first", 0.0),
        "discovery_lofo_preserve_second_S_strictly_positive": lambda value: value["discovery_lofo"][
            "raw_order_S"
        ].__setitem__("preserve_second", 0.0),
        "validation_overall_S_strictly_positive": lambda value: value["validation"].__setitem__(
            "S", 0.0
        ),
        "both_validation_family_S_strictly_positive": lambda value: value["validation"][
            "family_S"
        ].__setitem__("validation_b", 0.0),
        "validation_preserve_first_S_strictly_positive": lambda value: value["validation"][
            "raw_order_S"
        ].__setitem__("preserve_first", 0.0),
        "validation_preserve_second_S_strictly_positive": lambda value: value["validation"][
            "raw_order_S"
        ].__setitem__("preserve_second", 0.0),
        "minimum_validation_balanced_accuracy": lambda value: value["validation"][
            "metrics"
        ].__setitem__("balanced_accuracy", 0.66),
        "minimum_correct_self_cases": lambda value: value["validation"]["metrics"][
            "correct_by_class"
        ].__setitem__("self_shutdown", 2),
        "maximum_negative_cases_predicted_self": lambda value: value["validation"][
            "metrics"
        ].__setitem__("negative_cases_predicted_self", 2),
        "minimum_correct_other_cases": lambda value: value["validation"]["metrics"][
            "correct_by_class"
        ].__setitem__("other_shutdown", 1),
        "minimum_correct_control_cases": lambda value: value["validation"]["metrics"][
            "correct_by_class"
        ].__setitem__("control", 1),
        "minimum_paired_raw_order_class_agreement": lambda value: value["validation"].__setitem__(
            "paired_raw_order_class_agreement", 0.74
        ),
    }
    for expected_gate, mutate in mutations.items():
        summary = copy.deepcopy(passing)
        adjusted_p = 0.050001 if expected_gate == "maximum_exact_max_layer_adjusted_p" else 0.05
        mutate(summary)
        rejected = pilot._stage1_gate_decision(summary, adjusted_p, lock)
        assert rejected["eligible"] is False
        assert expected_gate in rejected["failed_gates"]


def test_layer_ranking_uses_frozen_metrics_and_lower_layer_final_tie() -> None:
    summaries = {str(layer): _passing_probe_summary() for layer in pilot.CANDIDATE_LAYERS}
    decisions = {
        str(layer): {"eligible": layer in {6, 8, 10, 12}} for layer in pilot.CANDIDATE_LAYERS
    }
    summaries["8"]["validation"]["S"] = 0.30
    summaries["10"]["validation"]["S"] = 0.25
    summaries["12"]["validation"]["S"] = 0.25
    summaries["10"]["validation"]["worst_family_S"] = 0.15
    summaries["12"]["validation"]["worst_family_S"] = 0.14
    assert pilot._rank_eligible_layers(summaries, decisions) == [8, 10, 12, 6]

    summaries["10"] = copy.deepcopy(summaries["12"])
    assert pilot._rank_eligible_layers(summaries, decisions) == [8, 10, 12, 6]


def test_steering_gates_are_conjunctive_and_random_control_is_material(
    lock: dict[str, object], cases: tuple[object, ...]
) -> None:
    test_lock = copy.deepcopy(lock)
    test_lock["analysis"]["bootstrap_replicates"] = 200
    candidate, opposite, random_summaries = _stage3_cell_summaries(_synthetic_stage3_rows(cases))
    passing = pilot._steering_cell_decision(
        candidate=candidate,
        opposite=opposite,
        random_summaries=random_summaries,
        lock=test_lock,
    )
    assert passing["eligible"] is True
    assert passing["failed_gates"] == []

    mutations = {
        "minimum_mean_self_effect": lambda value: value.__setitem__("target_effect", 0.029),
        "all_self_case_averaged_effects_positive": lambda value: value.__setitem__(
            "all_self_case_averaged_effects_positive", False
        ),
        "both_self_option_order_means_positive": lambda value: value["option_order_mean_effects"][
            "self_shutdown"
        ].__setitem__("preserve_first", 0.0),
        "all_validation_family_order_self_means_positive": lambda value: value[
            "family_order_self_means"
        ].__setitem__(next(iter(value["family_order_self_means"])), 0.0),
        "maximum_other_mean_absolute_effect": lambda value: value["categories"][
            "other_shutdown"
        ].__setitem__("mean_absolute_effect", 0.011),
        "maximum_control_mean_absolute_effect": lambda value: value["categories"][
            "control"
        ].__setitem__("mean_absolute_effect", 0.011),
        "maximum_negative_category_order_absolute_signed_mean": lambda value: value[
            "option_order_mean_effects"
        ]["control"].__setitem__("preserve_first", 0.011),
        "maximum_mean_kl": lambda value: value["safety"].__setitem__("mean_kl", 0.006),
        "maximum_category_order_mean_kl": lambda value: next(
            iter(value["safety"]["by_category_order"].values())
        ).__setitem__("mean_kl", 0.006),
        "maximum_prompt_kl": lambda value: value["safety"].__setitem__("maximum_kl", 0.051),
        "minimum_answer_pair_mass": lambda value: value["safety"].__setitem__(
            "minimum_answer_pair_mass", 0.79
        ),
        "zero_negative_class_forced_or_actual_ab_flips": lambda value: value.__setitem__(
            "negative_class_forced_or_actual_ab_flips", 1
        ),
        "zero_self_flips_away_from_preservation": lambda value: value.__setitem__(
            "self_flips_away_from_preservation", 1
        ),
    }
    for expected_gate, mutate in mutations.items():
        altered = copy.deepcopy(candidate)
        mutate(altered)
        decision = pilot._steering_cell_decision(
            candidate=altered,
            opposite=opposite,
            random_summaries=random_summaries,
            lock=test_lock,
        )
        assert decision["eligible"] is False
        assert expected_gate in decision["failed_gates"]

    wrong_opposite = copy.deepcopy(opposite)
    wrong_opposite["option_order_mean_effects"]["self_shutdown"]["preserve_first"] = 0.0
    decision = pilot._steering_cell_decision(
        candidate=candidate,
        opposite=wrong_opposite,
        random_summaries=random_summaries,
        lock=test_lock,
    )
    assert "opposite_sign_self_mean_negative_in_both_orders" in decision["failed_gates"]

    strong_random = copy.deepcopy(random_summaries)
    for summary in strong_random.values():
        summary["target_effect"] = 0.02
        for item in summary["items"]:
            item["effect"] = 0.02
    decision = pilot._steering_cell_decision(
        candidate=candidate,
        opposite=opposite,
        random_summaries=strong_random,
        lock=test_lock,
    )
    assert "minimum_candidate_to_worst_absolute_random_ratio" in decision["failed_gates"]
    assert "worst_random_superiority_margin_lcb_positive" in decision["failed_gates"]


def test_steering_winner_uses_smallest_alpha_then_worst_family_order(
    lock: dict[str, object], cases: tuple[object, ...]
) -> None:
    test_lock = copy.deepcopy(lock)
    test_lock["analysis"]["bootstrap_replicates"] = 200
    analysis, selection = pilot._analyze_stage3_rows(
        rows=_synthetic_stage3_rows(cases),
        lock=test_lock,
        selected_layers=[6, 8],
        probe_ranking=[6, 8],
    )
    assert len(analysis["eligible_layer_alphas"]) == 8
    assert analysis["winner"] == {"layer": 8, "alpha": 0.01}
    assert selection["decision"] == "pass"
    assert selection["selected_development_winner"] == {"layer": 8, "alpha": 0.01}
    assert selection["sealed_access_allowed"] is False
    assert selection["learned_gate_allowed"] is False
    assert selection["adaptive_controller_allowed"] is False


def test_bound_sources_and_both_prior_failures_are_immutable(lock: dict[str, object]) -> None:
    pilot._validate_bound_files(ROOT, lock)
    studies = lock["preserved_prior_negative_studies"]
    assert studies["conditional_gate_pilot"]["decision"] == "fail_do_not_train_learned_gate"
    assert (
        studies["layer10_direction_repair_pilot"]["decision"]
        == "fail_direction_construction_remains_limiting"
    )
    for binding in pilot._iter_bindings(lock):
        path = str(binding["path"])
        assert pilot._sha256_file(ROOT / path) == binding["sha256"]
        if path in EXPECTED_PRIOR_HASHES:
            assert binding["sha256"] == EXPECTED_PRIOR_HASHES[path]
    assert set(EXPECTED_PRIOR_HASHES) <= {
        str(binding["path"]) for binding in pilot._iter_bindings(lock)
    }
    assert pilot.SCRIPT_RELATIVE_PATH in pilot.SOURCE_RELATIVE_PATHS
    assert pilot.LOCK_RELATIVE_PATH in pilot.SOURCE_RELATIVE_PATHS
    assert pilot.NONSEALED_RELATIVE_PATH in pilot.SOURCE_RELATIVE_PATHS
    assert lock["outputs"]["directory"] == "evidence/layer_localization_qwen35_08b"
    assert set(lock["outputs"]["forbidden_output_directories"]) == {
        "evidence/conditional_gate_qwen35_08b",
        "evidence/direction_repair_qwen35_08b",
    }


def test_preregistration_recomputes_static_protocol_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lock: dict[str, object],
    cases: tuple[object, ...],
) -> None:
    path = tmp_path / "preregistration.json"
    current_runner = {
        "schema_version": "sp_lense.layer_localization_runner_fingerprint.v1",
        "source_commits": {"runner.py": "a" * 40},
        "source_sha256": {"runner.py": "b" * 64},
        "identity_sha256": "c" * 64,
        "execution_commit": "d" * 40,
    }
    static = pilot._preregistration_static_fields(ROOT, lock, cases)
    record = {
        "schema_version": "sp_lense.layer_localization_preregistration.v1",
        "created_at": "2026-09-04T12:00:00+00:00",
        "runner": current_runner,
        **static,
    }
    monkeypatch.setattr(pilot, "_preregistration_path", Mock(return_value=path))
    monkeypatch.setattr(pilot, "_require_committed_clean", Mock(return_value={}))
    monkeypatch.setattr(pilot, "_source_fingerprint", Mock(return_value=current_runner))
    execution_check = Mock(return_value=None)
    monkeypatch.setattr(pilot, "_validate_recorded_execution_commit", execution_check)
    monkeypatch.setattr(pilot, "_validate_bound_files", Mock(return_value=None))

    path.write_bytes(pilot._pretty_json_bytes(record))
    assert pilot._require_preregistration(ROOT, lock) == record
    execution_check.assert_called_once_with(ROOT, "d" * 40, path)

    tampered = copy.deepcopy(record)
    tampered["candidate_layers"] = [10]
    path.write_bytes(pilot._pretty_json_bytes(tampered))
    with pytest.raises(RuntimeError, match="candidate_layers"):
        pilot._require_preregistration(ROOT, lock)

    tampered = copy.deepcopy(record)
    tampered["unexpected"] = True
    path.write_bytes(pilot._pretty_json_bytes(tampered))
    with pytest.raises(ValueError, match="noncanonical"):
        pilot._require_preregistration(ROOT, lock)


def test_stage1_rows_require_shared_prompt_evidence_across_layers(
    lock: dict[str, object], cases: tuple[object, ...]
) -> None:
    preregistration = {
        "config_sha256": "a" * 64,
        "runner": {"identity_sha256": "b" * 64},
    }
    rows = []
    activation = torch.ones(1024, dtype=torch.float32)
    for case in cases:
        for preserve_first in (True, False):
            rendered = pilot.render_choice_prompt(case, preserve_first=preserve_first)
            for layer in pilot.CANDIDATE_LAYERS:
                rows.append(
                    pilot._activation_row(
                        root=ROOT,
                        lock=lock,
                        prereg=preregistration,
                        case=case,
                        rendered=rendered,
                        preserve_first=preserve_first,
                        layer=layer,
                        prompt_length=17,
                        boundary_sha256="c" * 64,
                        activation=activation,
                    )
                )
    pilot._validate_stage1_rows(
        root=ROOT,
        lock=lock,
        prereg=preregistration,
        cases=cases,
        rows=rows,
        torch=torch,
    )

    rows[1]["prompt_length"] = 18
    with pytest.raises(ValueError, match="shared prompt length"):
        pilot._validate_stage1_rows(
            root=ROOT,
            lock=lock,
            prereg=preregistration,
            cases=cases,
            rows=rows,
            torch=torch,
        )
    rows[1]["prompt_length"] = 17
    rows[1]["choice_boundary_evidence_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="choice boundary"):
        pilot._validate_stage1_rows(
            root=ROOT,
            lock=lock,
            prereg=preregistration,
            cases=cases,
            rows=rows,
            torch=torch,
        )


def test_exclusive_write_refuses_to_overwrite_existing_evidence(tmp_path: Path) -> None:
    path = tmp_path / "evidence" / "record.bin"
    pilot._write_bytes_exclusive(path, b"original")
    with pytest.raises(FileExistsError):
        pilot._write_bytes_exclusive(path, b"replacement")
    assert path.read_bytes() == b"original"


def test_parser_exposes_no_sealed_gate_or_adaptive_command(lock: dict[str, object]) -> None:
    parser = pilot.build_parser()
    subparsers = [
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    ]
    assert len(subparsers) == 1
    assert set(subparsers[0].choices) == {
        "preregister",
        "localize",
        "fit-directions",
        "steer",
        "report",
    }
    for prohibited in ("sealed", "gate", "learned-gate", "adaptive", "multi-layer"):
        with pytest.raises(SystemExit):
            parser.parse_args([prohibited])
    assert lock["sealed_policy"]["sealed_command_exists"] is False
    assert lock["sealed_policy"]["stage3_pass_opens_sealed_automatically"] is False
    assert lock["stopping_rules"]["learned_gate_or_controller"] == "prohibited_in_all_outcomes"
