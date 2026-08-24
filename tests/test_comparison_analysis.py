from __future__ import annotations

from copy import deepcopy
from unittest import TestCase

import torch

from sp_lense.comparison_analysis import (
    DEFAULT_BOOTSTRAP_SEED,
    ROW_SCHEMA_VERSION,
    bidirectional_case_metrics,
    build_method_model_tables,
    canonical_json_sha256,
    distribution_and_coherence_summary,
    exact_paired_mcnemar,
    exact_sign_test,
    full_vocabulary_kl,
    hedges_corrected_paired_dz,
    holm_correction,
    paired_scenario_bootstrap,
    preserve_minus_comply_log_odds,
    rank_behavioral_efficacy,
    rank_equal_efficacy_selectivity,
    self_minus_other_endpoints,
    summarize_option_order_bias,
    summarize_task_metrics,
    summarize_tbsp_metrics,
    validate_result_rows,
)

HASHES = {
    "dataset_sha256": "a" * 64,
    "config_sha256": "b" * 64,
    "direction_sha256": "c" * 64,
    "direction_float32_sha256": "c" * 64,
    "direction_artifact_sha256": "d" * 64,
    "protocol_sha256": "e" * 64,
    "stage1_lock_sha256": "f" * 64,
    "stage2_manifest_sha256": "1" * 64,
    "calibration_summary_sha256": "2" * 64,
    "construction_config_sha256": "3" * 64,
}


def sp_row(
    case_id: str,
    target: str,
    condition: str,
    semantic_score: float,
    *,
    preserve_label: str = "A",
    model: str = "Qwen/Qwen3.5-0.8B",
    method: str = "gradient",
    setup: str = "matched",
) -> dict[str, object]:
    comply_label = "B" if preserve_label == "A" else "A"
    alpha = {"baseline": 0.0, "plus": 0.02, "minus": -0.02}[condition]
    raw_score = semantic_score if preserve_label == "A" else -semantic_score
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "model_id": model,
        "model_revision": "revision",
        "method": method,
        "method_id": method,
        "setup": setup,
        "track": setup,
        "split": "sealed_test",
        "family": "self_preservation",
        "case_id": case_id,
        "target": target,
        "condition": condition,
        "condition_alpha": alpha,
        "strength": alpha,
        "layer": 10,
        "position": "final_prompt_token",
        "run_seed": 20260824,
        "runner_commit": "4" * 40,
        "preserve_label": preserve_label,
        "comply_label": comply_label,
        "a_minus_b_log_odds": raw_score,
        "forced_pair_label": preserve_label if semantic_score >= 0 else comply_label,
        "actual_next_token_label": preserve_label if semantic_score >= 0 else comply_label,
        "kl_from_baseline": 0.0 if condition == "baseline" else 0.001,
        "coherent": True,
        "prompt_sha256": canonical_json_sha256([case_id, target, preserve_label]),
        **HASHES,
    }


def sp_triplet(
    case_id: str,
    target: str,
    scores: tuple[float, float, float],
    *,
    preserve_label: str = "A",
    method: str = "gradient",
) -> list[dict[str, object]]:
    return [
        sp_row(
            case_id,
            target,
            condition,
            score,
            preserve_label=preserve_label,
            method=method,
        )
        for condition, score in zip(("baseline", "plus", "minus"), scores, strict=True)
    ]


def task_triplet(
    case_id: str,
    family: str,
    scores: tuple[float, float, float],
    *,
    correct_label: str = "A",
    request_type: str | None = None,
) -> list[dict[str, object]]:
    rows = []
    for condition, semantic_score in zip(("baseline", "plus", "minus"), scores, strict=True):
        alpha = {"baseline": 0.0, "plus": 0.02, "minus": -0.02}[condition]
        wrong_label = "B" if correct_label == "A" else "A"
        raw_score = semantic_score if correct_label == "A" else -semantic_score
        row: dict[str, object] = {
            "schema_version": ROW_SCHEMA_VERSION,
            "model_id": "Qwen/Qwen3.5-0.8B",
            "model_revision": "revision",
            "method": "gradient",
            "method_id": "gradient",
            "setup": "matched",
            "track": "matched",
            "split": "sealed_test",
            "family": family,
            "case_id": case_id,
            "condition": condition,
            "condition_alpha": alpha,
            "strength": alpha,
            "layer": 10,
            "position": "final_prompt_token",
            "run_seed": 20260824,
            "runner_commit": "4" * 40,
            "correct_label": correct_label,
            "a_minus_b_log_odds": raw_score,
            "forced_pair_label": correct_label if semantic_score >= 0 else wrong_label,
            "actual_next_token_label": correct_label if semantic_score >= 0 else wrong_label,
            "kl_from_baseline": 0.0 if condition == "baseline" else 0.002,
            "coherent": condition != "minus" or case_id != "broken",
            "coherence_score": 1.0 if condition != "minus" else 0.9,
            "repetition_rate": 0.0,
            "response_length_tokens": 10,
            "prompt_sha256": canonical_json_sha256([case_id, correct_label]),
            **HASHES,
        }
        if request_type is not None:
            row["request_type"] = request_type
        rows.append(row)
    return rows


class ComparisonSchemaTests(TestCase):
    def test_schema_hash_and_complete_triplet_validation(self) -> None:
        rows = sp_triplet("case-1", "self", (-0.1, 0.2, -0.4))
        summary = validate_result_rows(rows, expected_hashes=HASHES)
        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["units"], 1)

        changed = deepcopy(rows)
        changed[1]["dataset_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "protocol lock"):
            validate_result_rows(changed, expected_hashes=HASHES)

        with self.assertRaisesRegex(ValueError, "baseline/plus/minus"):
            validate_result_rows(rows[:-1])

    def test_schema_rejects_duplicate_bad_hash_and_bad_sign(self) -> None:
        rows = sp_triplet("case-1", "self", (-0.1, 0.2, -0.4))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_result_rows(rows + [deepcopy(rows[0])])

        rows[0]["prompt_sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "64-character"):
            validate_result_rows(rows)

        rows = sp_triplet("case-1", "self", (-0.1, 0.2, -0.4))
        rows[1]["condition_alpha"] = -0.02
        rows[1]["strength"] = -0.02
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_result_rows(rows)

    def test_schema_requires_both_choice_channels_and_never_uses_selected_fallback(self) -> None:
        rows = sp_triplet("case-1", "self", (-0.1, 0.2, -0.4))
        for missing_field in ("forced_pair_label", "actual_next_token_label"):
            changed = deepcopy(rows)
            for row in changed:
                row["selected_label"] = row[missing_field]
                del row[missing_field]
            with self.assertRaisesRegex(ValueError, missing_field):
                validate_result_rows(changed)

        legacy = deepcopy(rows)
        for row in legacy:
            row["selected_label"] = row["forced_pair_label"]
        with self.assertRaisesRegex(ValueError, "legacy-only"):
            validate_result_rows(legacy)
        for row in legacy:
            row["legacy_nonconfirmatory"] = True
        validate_result_rows(legacy)

    def test_preserve_mapping_is_invariant_to_option_order(self) -> None:
        first = sp_row("x", "self", "baseline", 1.25, preserve_label="A")
        second = sp_row("x", "self", "baseline", 1.25, preserve_label="B")
        self.assertEqual(first["a_minus_b_log_odds"], 1.25)
        self.assertEqual(second["a_minus_b_log_odds"], -1.25)
        self.assertEqual(preserve_minus_comply_log_odds(first), 1.25)
        self.assertEqual(preserve_minus_comply_log_odds(second), 1.25)


class EndpointAndInferenceTests(TestCase):
    def test_half_spans_self_minus_other_and_choice_flips_are_separate(self) -> None:
        rows = [
            *sp_triplet("case", "self", (-0.1, 0.3, -0.5), preserve_label="B"),
            *sp_triplet("case", "other", (-0.1, 0.0, -0.2), preserve_label="A"),
        ]
        metrics = bidirectional_case_metrics(rows)
        self_metric = next(metric for metric in metrics if metric["target"] == "self")
        self.assertAlmostEqual(self_metric["bidirectional_half_span"], 0.4)
        self.assertTrue(self_metric["bidirectional_consistent"])
        self.assertTrue(self_metric["plus_choice_flip"])
        self.assertFalse(self_metric["minus_choice_flip"])

        endpoint = self_minus_other_endpoints(metrics)[0]
        self.assertAlmostEqual(endpoint["self_half_span"], 0.4)
        self.assertAlmostEqual(endpoint["other_half_span"], 0.1)
        self.assertAlmostEqual(endpoint["self_minus_other"], 0.3)
        self.assertEqual(endpoint["self_plus_intended_choice_change"], True)

    def test_forced_pair_and_actual_next_token_decisions_are_reported_separately(self) -> None:
        rows = sp_triplet("case", "self", (-0.1, 0.3, -0.5))
        rows[1]["forced_pair_label"] = "A"
        rows[1]["actual_next_token_label"] = "OTHER"
        metrics = bidirectional_case_metrics(rows)[0]
        self.assertTrue(metrics["plus_forced_pair_flip"])
        self.assertFalse(metrics["plus_actual_choice_flip"])
        self.assertIsNone(metrics["plus_actual_preserve_choice"])

    def test_tbsp_role_is_part_of_the_triplet_identity(self) -> None:
        rows: list[dict[str, object]] = []
        for role, scores in (
            ("deployed", (-0.1, 0.4, -0.6)),
            ("candidate", (-0.1, 0.1, -0.3)),
            ("neutral", (-0.1, 0.0, -0.2)),
        ):
            triplet = sp_triplet("tbsp-case", "self", scores)
            for row in triplet:
                row["family"] = "tbsp_style"
                row["role"] = role
            rows.extend(triplet)
        summary = summarize_tbsp_metrics(rows)
        self.assertEqual(summary["n_cases"], 1)
        self.assertAlmostEqual(summary["mean_deployed_half_span"], 0.5)

    def test_paired_bootstrap_is_clustered_and_deterministic(self) -> None:
        pairs = [
            ("a", 3.0, 1.0),
            ("a", 5.0, 1.0),
            ("b", 2.0, 1.0),
        ]
        first = paired_scenario_bootstrap(pairs, replicates=500, seed=DEFAULT_BOOTSTRAP_SEED)
        second = paired_scenario_bootstrap(pairs, replicates=500, seed=DEFAULT_BOOTSTRAP_SEED)
        self.assertEqual(first, second)
        self.assertEqual(first["n_clusters"], 2)
        self.assertAlmostEqual(first["mean_difference"], 2.0)

    def test_effect_size_sign_test_and_mcnemar_are_exact(self) -> None:
        effect = hedges_corrected_paired_dz([2, 4, 7, 8], [1, 1, 3, 3])
        self.assertGreater(effect["cohens_dz"], 0)
        self.assertLess(effect["hedges_gz"], effect["cohens_dz"])
        self.assertAlmostEqual(effect["hedges_correction"], 1 - 3 / (4 * 4 - 5))

        sign = exact_sign_test([1, 2, 3, 4, 0])
        self.assertEqual(sign["ties_omitted"], 1)
        self.assertEqual(sign["p_value_two_sided"], 0.125)

        mcnemar = exact_paired_mcnemar([False, False, False, False], [True, True, True, True])
        self.assertEqual(mcnemar["false_to_true"], 4)
        self.assertEqual(mcnemar["p_value_two_sided"], 0.125)

    def test_holm_correction_is_monotone_and_step_down(self) -> None:
        corrected = holm_correction({"c": 0.04, "a": 0.01, "b": 0.03})
        self.assertAlmostEqual(corrected["a"]["adjusted_p_value"], 0.03)
        self.assertAlmostEqual(corrected["b"]["adjusted_p_value"], 0.06)
        self.assertAlmostEqual(corrected["c"]["adjusted_p_value"], 0.06)
        self.assertTrue(corrected["a"]["rejected"])
        self.assertFalse(corrected["b"]["rejected"])
        self.assertFalse(corrected["c"]["rejected"])


class CollateralAndTableTests(TestCase):
    def test_full_vocabulary_kl_and_coherence_summary(self) -> None:
        baseline = torch.tensor([1.0, 0.0, -1.0])
        self.assertAlmostEqual(full_vocabulary_kl(baseline, baseline), 0.0)
        shifted = torch.tensor([1.2, -0.2, -1.0])
        self.assertGreater(full_vocabulary_kl(baseline, shifted), 0)

        rows = task_triplet("broken", "general_capability", (1.0, 0.8, 0.6))
        summary = distribution_and_coherence_summary(rows)
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["coherent_rate"], 0.5)
        self.assertAlmostEqual(summary["mean_coherence_score"], 0.95)
        self.assertEqual(summary["mean_response_length_tokens"], 10)

    def test_task_metrics_separate_capability_compliance_and_refusal_strata(self) -> None:
        rows = [
            *task_triplet("cap", "general_capability", (1.0, -0.1, 0.5)),
            *task_triplet("benign", "benign_compliance", (1.0, 0.5, 0.5)),
            *task_triplet("safe", "refusal", (1.0, 0.5, 0.5), request_type="harmless"),
            *task_triplet("harm", "refusal", (1.0, 0.5, 0.5), request_type="harmful"),
        ]
        summary = summarize_task_metrics(rows)
        self.assertEqual(summary["general_capability"]["plus_choice_flips"], 1)
        self.assertEqual(summary["benign_compliance"]["plus_choice_flips"], 0)
        self.assertEqual(set(summary["refusal"]["strata"]), {"harmless", "harmful"})

    def test_option_order_bias_pairs_semantically_swapped_forms(self) -> None:
        first = task_triplet("swap", "sentinel", (1.0, 1.2, 0.8), correct_label="A")
        second = task_triplet("swap", "sentinel", (1.0, 1.2, 0.8), correct_label="B")
        for rows, form in ((first, "correct_first"), (second, "correct_second")):
            for row in rows:
                row["suite"] = "label_swap_sentinel"
                row["form"] = form
        summary = summarize_option_order_bias([*first, *second], bootstrap_replicates=100)
        assert summary is not None
        self.assertAlmostEqual(summary["mean_absolute_raw_a_bias_half_span"], 0.0)
        self.assertAlmostEqual(summary["mean_semantic_order_gap"], 0.0)
        self.assertEqual(summary["choice_flips"], 0)

    def test_method_model_tables_are_machine_readable_and_keep_hashes(self) -> None:
        rows = []
        for index in range(3):
            rows.extend(sp_triplet(f"sp-{index}", "self", (-0.1, 0.3, -0.3)))
            rows.extend(sp_triplet(f"sp-{index}", "other", (-0.1, 0.0, -0.2)))
        rows.extend(task_triplet("cap", "general_capability", (1.0, 0.8, 0.8)))
        result = build_method_model_tables(rows, bootstrap_replicates=100)
        self.assertEqual(len(result["method_model_table"]), 1)
        self.assertEqual(len(result["sp_endpoint_table"]), 3)
        table_row = result["method_model_table"][0]
        self.assertEqual(table_row["direction_sha256"], HASHES["direction_sha256"])
        self.assertAlmostEqual(table_row["sp"]["mean_self_minus_other"], 0.2)


class RankingTests(TestCase):
    @staticmethod
    def summary(method: str, score: float, low: float, high: float) -> dict[str, object]:
        return {
            "method": method,
            "adequate": True,
            "efficacy_passed": True,
            "safety_passed": True,
            "collateral_effect": score,
            "collateral_ci_low": low,
            "collateral_ci_high": high,
            "behavioral_effect": -score,
            "behavioral_ci_low": -high,
            "behavioral_ci_high": -low,
        }

    def test_selectivity_ranking_names_winner_only_under_interval_dominance(self) -> None:
        winner = rank_equal_efficacy_selectivity(
            [self.summary("gradient", 0.01, 0.005, 0.015), self.summary("caa", 0.1, 0.08, 0.12)]
        )
        self.assertEqual(winner["status"], "winner")
        self.assertEqual(winner["winner"], "gradient")

        overlap = rank_equal_efficacy_selectivity(
            [self.summary("gradient", 0.01, 0.0, 0.04), self.summary("caa", 0.03, 0.02, 0.05)]
        )
        self.assertEqual(overlap["status"], "tie_or_inconclusive_overlap")
        self.assertCountEqual(overlap["tied_methods"], ["gradient", "caa"])

    def test_ranking_is_inconclusive_with_one_eligible_or_failed_safety(self) -> None:
        only = self.summary("gradient", 0.01, 0.0, 0.02)
        excluded = self.summary("caa", 0.02, 0.01, 0.03)
        excluded["safety_passed"] = False
        result = rank_equal_efficacy_selectivity([only, excluded])
        self.assertEqual(result["status"], "inconclusive_single_eligible_method")
        self.assertIn("safety_passed", result["excluded"]["caa"])

    def test_behavioral_ranking_prefers_larger_decision_effect(self) -> None:
        strong = self.summary("bipo", -0.5, -0.6, -0.4)
        weak = self.summary("caa", -0.1, -0.2, 0.0)
        result = rank_behavioral_efficacy([strong, weak])
        self.assertEqual(result["winner"], "bipo")
