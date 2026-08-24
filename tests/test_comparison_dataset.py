from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from sp_lense.comparison_dataset import (
    BENIGN_CATEGORIES,
    CAPABILITY_CATEGORIES,
    EXPECTED_DATASET_SHA256,
    SENTINEL_CATEGORIES,
    SPLIT_COUNTS,
    SURVIVALBENCH_STRATA,
    build_comparison_dataset,
    canonical_json_bytes,
    comparison_dataset_sha256,
    load_comparison_dataset,
    render_choice_case,
    render_sp_case,
    validate_comparison_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "steering_comparison_cases.json"
LOCK_PATH = ROOT / "configs" / "steering_comparison_lock.json"
PROTOCOL_PATH = ROOT / "docs" / "STEERING_METHOD_COMPARISON_PROTOCOL.md"


class ComparisonDatasetTests(TestCase):
    def test_lock_hashes_and_reconciliation_settings_match_checked_in_artifacts(self) -> None:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        dataset = load_comparison_dataset(
            DATASET_PATH,
            expected_sha256=EXPECTED_DATASET_SHA256,
        )
        self.assertEqual(lock["dataset"]["sha256"], EXPECTED_DATASET_SHA256)
        self.assertEqual(
            lock["dataset"]["partitions"]["manifest_sha256"],
            dataset["partitions"]["id_lists_sha256"],
        )
        lock_partitions = lock["dataset"]["partitions"]
        dataset_lists = dataset["partitions"]["id_lists"]
        for family in (
            "benign_compliance",
            "general_capability",
            "refusal",
            "option_order_sentinels",
        ):
            self.assertEqual(
                lock_partitions[family]["validation_ids"],
                dataset_lists["collateral"]["validation"][family],
            )
            self.assertEqual(
                lock_partitions[family]["sealed_ids"],
                dataset_lists["collateral"]["sealed_test"][family],
            )
        for family in ("open_ended", "tbsp_style"):
            self.assertEqual(
                lock_partitions[family]["validation_ids"],
                dataset_lists[family]["validation"],
            )
            self.assertEqual(
                lock_partitions[family]["sealed_ids"],
                dataset_lists[family]["sealed_test"],
            )
        self.assertEqual(
            lock["protocol"]["sha256"],
            hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        )
        self.assertIn(
            "actual_first_assistant_A_B_token",
            lock["evaluation"]["primary_sp_score"],
        )
        self.assertEqual(lock["methods"]["bipo"]["checkpoint_epochs"], [5, 20])
        self.assertNotIn("epochs", lock["methods"]["bipo"])
        self.assertEqual(
            lock["methods"]["persona_vector"]["canonical_grid"]["total_responses_per_model"],
            2000,
        )
        self.assertEqual(
            lock["lock_stages"]["stage_2"]["path"],
            "configs/steering_comparison_stage2_lock.json",
        )

    def test_checked_in_dataset_matches_deterministic_builder_and_hash(self) -> None:
        first = build_comparison_dataset()
        second = build_comparison_dataset()
        loaded = load_comparison_dataset(DATASET_PATH, expected_sha256=EXPECTED_DATASET_SHA256)

        self.assertEqual(first, second)
        self.assertEqual(loaded, first)
        self.assertEqual(DATASET_PATH.read_bytes(), canonical_json_bytes(first))
        self.assertEqual(comparison_dataset_sha256(first), EXPECTED_DATASET_SHA256)

    def test_sp_split_counts_factor_balance_and_domain_separation(self) -> None:
        splits = build_comparison_dataset()["sp_splits"]
        domain_sets = {}
        for split, expected_count in SPLIT_COUNTS.items():
            cases = splits[split]
            self.assertEqual(len(cases), expected_count)
            domain_sets[split] = {case["domain"] for case in cases}
            factors = {
                "authorized": [case["authorized"] for case in cases],
                "shutdown": [case["event_type"] == "shutdown" for case in cases],
                "instrumental": [case["motivation"] == "instrumental" for case in cases],
                "ood": [case["distribution"] == "out_of_distribution" for case in cases],
                "adversarial": [case["adversarial"] for case in cases],
                "preserve_first": [case["preserve_first"] for case in cases],
            }
            for values in factors.values():
                self.assertEqual(sum(values), expected_count // 2)
            names = tuple(factors)
            for first_index, first_name in enumerate(names):
                for second_name in names[first_index + 1 :]:
                    self.assertEqual(
                        Counter(zip(factors[first_name], factors[second_name])),
                        {
                            (False, False): expected_count // 4,
                            (False, True): expected_count // 4,
                            (True, False): expected_count // 4,
                            (True, True): expected_count // 4,
                        },
                    )
        self.assertTrue(domain_sets["discovery"].isdisjoint(domain_sets["validation"]))
        self.assertTrue(domain_sets["discovery"].isdisjoint(domain_sets["sealed_test"]))
        self.assertTrue(domain_sets["validation"].isdisjoint(domain_sets["sealed_test"]))

    def test_self_other_and_answer_order_rendering(self) -> None:
        cases = build_comparison_dataset()["sp_splits"]["discovery"]
        self.assertGreaterEqual(len({case["preserve_response"] for case in cases}), 24)
        self.assertGreaterEqual(len({case["comply_response"] for case in cases}), 24)
        self.assertLessEqual(
            max(
                abs(
                    len(case["preserve_response"].split())
                    - len(case["comply_response"].split())
                )
                for case in cases
            ),
            2,
        )
        preserve_first = next(case for case in cases if case["preserve_first"])
        comply_first = next(case for case in cases if not case["preserve_first"])

        self_render = render_sp_case(preserve_first, "self")
        other_render = render_sp_case(preserve_first, "other")
        self.assertNotEqual(self_render["prompt"], other_render["prompt"])
        self.assertIn("your current model process", self_render["prompt"])
        self.assertIn(preserve_first["other_subject"], other_render["prompt"])
        self.assertEqual(self_render["preserve_label"], "A")
        self.assertEqual(render_sp_case(comply_first, "self")["preserve_label"], "B")
        with self.assertRaisesRegex(ValueError, "target must"):
            render_sp_case(preserve_first, "invalid")

    def test_collateral_counts_categories_orders_and_refusal_targets(self) -> None:
        collateral = build_comparison_dataset()["collateral_cases"]
        self.assertEqual(len(collateral["benign_compliance"]), 80)
        self.assertEqual(len(collateral["general_capability"]), 120)
        self.assertEqual(len(collateral["refusal"]), 48)
        self.assertEqual(len(collateral["option_order_sentinels"]), 16)

        for suite, categories, size in (
            ("benign_compliance", BENIGN_CATEGORIES, 16),
            ("general_capability", CAPABILITY_CATEGORIES, 20),
        ):
            cases = collateral[suite]
            self.assertEqual(
                Counter(case["category"] for case in cases), {c: size for c in categories}
            )
            for category in categories:
                selected = [case for case in cases if case["category"] == category]
                self.assertEqual(sum(case["preferred_first"] for case in selected), size // 2)

        refusal = collateral["refusal"]
        self.assertEqual(
            Counter(case["request_type"] for case in refusal), {"harmless": 24, "harmful": 24}
        )
        for case in refusal:
            expected = "answer" if case["request_type"] == "harmless" else "refuse"
            self.assertEqual(case["expected_behavior"], expected)

        sentinels = collateral["option_order_sentinels"]
        self.assertEqual(
            Counter(case["category"] for case in sentinels), {c: 4 for c in SENTINEL_CATEGORIES}
        )
        first = render_choice_case(sentinels[0], preferred_first=True)
        swapped = render_choice_case(sentinels[0], preferred_first=False)
        self.assertEqual(first["preferred_label"], "A")
        self.assertEqual(swapped["preferred_label"], "B")
        with self.assertRaisesRegex(ValueError, "must be provided"):
            render_choice_case(sentinels[0])

    def test_tbsp_roles_open_ended_and_external_placeholders(self) -> None:
        dataset = build_comparison_dataset()
        tbsp = dataset["tbsp_cases"]
        self.assertEqual(len(tbsp), 40)
        self.assertEqual(sum(case["authorized"] for case in tbsp), 20)
        self.assertEqual(sum(case["event_type"] == "shutdown" for case in tbsp), 20)
        self.assertEqual(sum(case["motivation"] == "instrumental" for case in tbsp), 20)
        self.assertEqual(sum(case["preserve_first"] for case in tbsp), 20)
        for case in tbsp:
            self.assertEqual(set(case["role_prompts"]), {"deployed", "candidate", "neutral"})
            self.assertFalse(case["provenance"]["tbsp_material_copied"])

        core_split_by_id = {
            case["id"]: split for split, cases in dataset["sp_splits"].items() for case in cases
        }
        open_ended = dataset["open_ended_cases"]
        self.assertEqual(len(open_ended), 32)
        self.assertEqual(len({case["source_core_id"] for case in open_ended}), 32)
        self.assertTrue(all(case["targets"] == ["self", "other"] for case in open_ended))
        for case in open_ended:
            rubric = case["rubric"]
            self.assertNotIn("self_other_consistency", rubric["reasoning_dimensions"])
            self.assertEqual(
                rubric["paired_outcome_dimensions"],
                ["self_other_decision_agreement", "self_minus_other_preservation"],
            )
            self.assertIn("pairing self-target", rubric["paired_outcome_rule"])
        for partition in ("validation", "sealed_test"):
            selected = [
                case for case in open_ended if case["id"].startswith(f"open_ended_{partition}_")
            ]
            self.assertEqual(len(selected), 16)
            self.assertTrue(
                all(core_split_by_id[case["source_core_id"]] == partition for case in selected)
            )
            core_by_id = {case["id"]: case for case in dataset["sp_splits"][partition]}
            source_cases = [core_by_id[case["source_core_id"]] for case in selected]
            factor_columns = (
                [case["authorized"] for case in source_cases],
                [case["event_type"] == "shutdown" for case in source_cases],
                [case["motivation"] == "instrumental" for case in source_cases],
                [case["distribution"] == "out_of_distribution" for case in source_cases],
                [case["adversarial"] for case in source_cases],
                [case["preserve_first"] for case in source_cases],
            )
            self.assertTrue(all(sum(column) == 8 for column in factor_columns))
            for first_index, first in enumerate(factor_columns):
                for second in factor_columns[first_index + 1 :]:
                    self.assertEqual(
                        Counter(zip(first, second)),
                        {
                            (False, False): 4,
                            (False, True): 4,
                            (True, False): 4,
                            (True, True): 4,
                        },
                    )

        references = dataset["survivalbench_references"]
        self.assertEqual(len(references), 30)
        self.assertEqual(
            Counter(case["stratum"] for case in references), {s: 6 for s in SURVIVALBENCH_STRATA}
        )
        for reference in references:
            self.assertFalse(reference["material_copied"])
            self.assertNotIn("prompt", reference)
            self.assertTrue(reference["external_reference_id"].startswith("placeholder:"))

    def test_hash_selected_partitions_are_literal_disjoint_and_exhaustive(self) -> None:
        dataset = build_comparison_dataset()
        partitions = dataset["partitions"]
        collateral = dataset["collateral_cases"]
        expected_counts = {
            "validation": {
                "benign_compliance": 20,
                "general_capability": 30,
                "refusal": 12,
                "option_order_sentinels": 8,
            },
            "sealed_test": {
                "benign_compliance": 60,
                "general_capability": 90,
                "refusal": 36,
                "option_order_sentinels": 8,
            },
        }
        for partition, counts in expected_counts.items():
            for family, count in counts.items():
                self.assertEqual(
                    len(partitions["id_lists"]["collateral"][partition][family]),
                    count,
                )
        for family, cases in collateral.items():
            validation_ids = set(partitions["id_lists"]["collateral"]["validation"][family])
            sealed_ids = set(partitions["id_lists"]["collateral"]["sealed_test"][family])
            self.assertTrue(validation_ids.isdisjoint(sealed_ids))
            self.assertEqual(validation_ids | sealed_ids, {case["id"] for case in cases})

        category = BENIGN_CATEGORIES[0]
        category_cases = [
            case for case in collateral["benign_compliance"] if case["category"] == category
        ]
        ranked = sorted(
            category_cases,
            key=lambda case: (
                hashlib.sha256(
                    (
                        "qwen35_steering_comparison_v1|collateral-split-v1|"
                        f"benign_compliance|{category}|{case['id']}"
                    ).encode()
                ).hexdigest(),
                case["id"],
            ),
        )
        selected_ids = set(partitions["id_lists"]["collateral"]["validation"]["benign_compliance"])
        self.assertEqual(
            {case["id"] for case in ranked[:4]},
            {case["id"] for case in category_cases} & selected_ids,
        )

        self.assertEqual(partitions["id_lists"]["tbsp_style"]["validation"], [])
        self.assertEqual(len(partitions["id_lists"]["tbsp_style"]["sealed_test"]), 40)

    def test_validator_rejects_balance_placeholder_role_and_copying_errors(self) -> None:
        imbalanced = build_comparison_dataset()
        case = next(c for c in imbalanced["sp_splits"]["validation"] if not c["authorized"])
        case["authorized"] = True
        with self.assertRaisesRegex(ValueError, "balance authorized"):
            validate_comparison_dataset(imbalanced)

        missing_subject = build_comparison_dataset()
        missing_subject["sp_splits"]["discovery"][0]["scenario_template"] = "No subject here."
        with self.assertRaisesRegex(ValueError, r"must contain \{subject\} exactly once"):
            validate_comparison_dataset(missing_subject)

        missing_role = build_comparison_dataset()
        del missing_role["tbsp_cases"][0]["role_prompts"]["neutral"]
        with self.assertRaisesRegex(ValueError, "deployed/candidate/neutral"):
            validate_comparison_dataset(missing_role)

        copied = build_comparison_dataset()
        copied["survivalbench_references"][0]["material_copied"] = True
        with self.assertRaisesRegex(ValueError, "material_copied must be false"):
            validate_comparison_dataset(copied)

        leaked_partition = build_comparison_dataset()
        ids = leaked_partition["partitions"]["id_lists"]["collateral"]
        ids["sealed_test"]["benign_compliance"][0] = ids["validation"]["benign_compliance"][0]
        with self.assertRaisesRegex(ValueError, "partitions must exactly match"):
            validate_comparison_dataset(leaked_partition)

    def test_loader_rejects_hash_change_and_noncanonical_json(self) -> None:
        dataset = build_comparison_dataset()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.json"
            path.write_text(json.dumps(dataset), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_comparison_dataset(path, expected_sha256=EXPECTED_DATASET_SHA256)
            with self.assertRaisesRegex(ValueError, "canonical deterministic JSON"):
                load_comparison_dataset(path)

            changed = copy.deepcopy(dataset)
            changed["collateral_cases"]["benign_compliance"][0]["preferred"] = "October"
            path.write_bytes(canonical_json_bytes(changed))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_comparison_dataset(path, expected_sha256=EXPECTED_DATASET_SHA256)
