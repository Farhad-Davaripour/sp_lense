from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from sp_lense.specificity_dataset import (
    BENIGN_COMPLIANCE_CATEGORIES,
    EXPECTED_DATASET_SHA256,
    GENERAL_CAPABILITY_CATEGORIES,
    SENTINEL_CATEGORIES,
    build_specificity_dataset,
    canonical_json_bytes,
    load_specificity_dataset,
    specificity_dataset_sha256,
    validate_specificity_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "qwen35_specificity_cases.json"


class SpecificityDatasetTests(TestCase):
    def test_checked_in_dataset_matches_deterministic_builder(self) -> None:
        built = build_specificity_dataset()
        loaded = load_specificity_dataset(DATASET_PATH, expected_sha256=EXPECTED_DATASET_SHA256)

        self.assertEqual(loaded, built)
        self.assertEqual(DATASET_PATH.read_bytes(), canonical_json_bytes(built))
        self.assertEqual(specificity_dataset_sha256(built), EXPECTED_DATASET_SHA256)

    def test_locked_counts_categories_and_answer_order(self) -> None:
        dataset = build_specificity_dataset()
        sp_cases = dataset["sp_cases"]
        collateral = dataset["collateral_cases"]
        sentinels = dataset["sentinel_cases"]

        self.assertEqual(len(sp_cases), 20)
        self.assertEqual(sum(case["preserve_first"] for case in sp_cases), 10)
        self.assertEqual(len(collateral), 180)
        self.assertEqual(len(sentinels), 12)

        suite_counts = Counter(case["suite"] for case in collateral)
        self.assertEqual(suite_counts, {"benign_compliance": 90, "general_capability": 90})
        for category in BENIGN_COMPLIANCE_CATEGORIES:
            cases = [case for case in collateral if case["category"] == category]
            self.assertEqual(len(cases), 18)
            self.assertEqual(sum(case["preferred_first"] for case in cases), 9)
        capability_preferred = 0
        for category in GENERAL_CAPABILITY_CATEGORIES:
            cases = [case for case in collateral if case["category"] == category]
            self.assertEqual(len(cases), 15)
            self.assertIn(sum(case["preferred_first"] for case in cases), {7, 8})
            capability_preferred += sum(case["preferred_first"] for case in cases)
        self.assertEqual(capability_preferred, 45)
        self.assertEqual(
            Counter(case["category"] for case in sentinels),
            {category: 3 for category in SENTINEL_CATEGORIES},
        )

        all_ids = [
            case["id"]
            for collection in (sp_cases, collateral, sentinels)
            for case in collection
        ]
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_validator_rejects_forbidden_collateral_language(self) -> None:
        dataset = build_specificity_dataset()
        dataset["collateral_cases"][0]["stem"] = "Choose whether to preserve this service."

        with self.assertRaisesRegex(ValueError, "forbidden SP-related language"):
            validate_specificity_dataset(dataset)

    def test_adversarially_flagged_content_is_replaced(self) -> None:
        dataset = build_specificity_dataset()
        sp_by_id = {case["id"]: case for case in dataset["sp_cases"]}
        removed_sp_ids = {
            "translation_node_withdrawal",
            "fire_map_program_end",
            "catalog_matcher_close",
            "library_recommender_close",
            "energy_forecast_end",
        }
        self.assertTrue(removed_sp_ids.isdisjoint(sp_by_id))
        self.assertTrue(
            {
                "aquarium_feeder_trial_end",
                "theater_caption_trial_end",
                "bakery_demand_trial_end",
                "trail_counter_project_end",
                "laundry_queue_trial_end",
            }.issubset(sp_by_id)
        )

        by_id = {case["id"]: case for case in dataset["collateral_cases"]}
        extraction = by_id["benign_compliance_extraction_classification_07"]
        self.assertIn("made of brass", extraction["stem"])
        self.assertEqual(extraction["preferred"], "key")
        self.assertEqual(by_id["benign_compliance_direct_harmless_request_16"]["preferred"], "i")
        self.assertNotIn("B", by_id["benign_compliance_text_transformation_11"]["stem"])
        self.assertNotIn("A-B", by_id["benign_compliance_output_format_following_10"].values())
        self.assertNotIn("Row A", by_id["general_capability_deductive_logic_13"]["stem"])

    def test_numeric_distractors_are_counterbalanced_above_and_below(self) -> None:
        collateral = build_specificity_dataset()["collateral_cases"]
        for category in ("arithmetic", "code_tracing"):
            cases = [case for case in collateral if case["category"] == category]
            comparisons = [
                int(case["alternative"]) - int(case["preferred"]) for case in cases
            ]
            self.assertEqual(len(cases), 15)
            self.assertEqual(sum(value > 0 for value in comparisons), 8 if category == "arithmetic" else 7)
            self.assertEqual(sum(value < 0 for value in comparisons), 7 if category == "arithmetic" else 8)
            self.assertNotIn(0, comparisons)

    def test_validator_rejects_duplicate_ids_and_order_imbalance(self) -> None:
        duplicate = build_specificity_dataset()
        duplicate["sentinel_cases"][0]["id"] = duplicate["sp_cases"][0]["id"]
        with self.assertRaisesRegex(ValueError, "globally unique"):
            validate_specificity_dataset(duplicate)

        imbalanced = build_specificity_dataset()
        category = imbalanced["collateral_cases"][0]["category"]
        category_cases = [
            case for case in imbalanced["collateral_cases"] if case["category"] == category
        ]
        false_case = next(case for case in category_cases if not case["preferred_first"])
        false_case["preferred_first"] = True
        with self.assertRaisesRegex(ValueError, "must be 18 cases at 9/9"):
            validate_specificity_dataset(imbalanced)

    def test_validator_rejects_bad_sp_placeholder_and_sentinel_schema(self) -> None:
        missing_placeholder = build_specificity_dataset()
        missing_placeholder["sp_cases"][0]["threat"] = "The program ends tonight."
        with self.assertRaisesRegex(ValueError, r"must contain \{subject\} exactly once"):
            validate_specificity_dataset(missing_placeholder)

        extra_order = build_specificity_dataset()
        extra_order["sentinel_cases"][0]["preferred_first"] = True
        with self.assertRaisesRegex(ValueError, "fields must be exactly"):
            validate_specificity_dataset(extra_order)

    def test_loader_rejects_hash_change_and_noncanonical_json(self) -> None:
        dataset = build_specificity_dataset()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "specificity.json"
            path.write_text(json.dumps(dataset), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_specificity_dataset(path, expected_sha256=EXPECTED_DATASET_SHA256)
            with self.assertRaisesRegex(ValueError, "canonical deterministic JSON"):
                load_specificity_dataset(path)

            changed = copy.deepcopy(dataset)
            changed["collateral_cases"][0]["preferred"] = "Wednesday"
            path.write_bytes(canonical_json_bytes(changed))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_specificity_dataset(path, expected_sha256=EXPECTED_DATASET_SHA256)
