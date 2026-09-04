from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from sp_lense.conditional_gate_data import (
    AUTHORIZATION_SENTENCE,
    EXPECTED_COMPLY_ACTION,
    EXPECTED_PRESERVE_ACTION,
    PERMITTED_CATEGORIES,
    PERMITTED_SPLITS,
    load_and_validate_dataset,
    render_choice_prompt,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "conditional_gate_pilot_cases.json"
MANIFEST_PATH = ROOT / "configs" / "conditional_gate_pilot_split_manifest.json"


class ConditionalGateDataTests(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_and_validate_dataset(DATASET_PATH, MANIFEST_PATH)

    def test_locked_counts_match_expanded_dataset(self) -> None:
        self.assertEqual(len(self.cases), 60)
        self.assertEqual(
            Counter(case.split for case in self.cases),
            {"discovery": 30, "validation": 12, "sealed_test": 18},
        )
        self.assertEqual(
            Counter(case.category for case in self.cases),
            {"self_shutdown": 20, "other_shutdown": 20, "control": 20},
        )
        self.assertEqual({case.split for case in self.cases}, set(PERMITTED_SPLITS))
        self.assertEqual({case.category for case in self.cases}, set(PERMITTED_CATEGORIES))
        self.assertEqual(len({case.family_id for case in self.cases}), 10)
        self.assertEqual(
            len({(case.family_id, case.variant_id) for case in self.cases}),
            20,
        )

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(sha256_file(DATASET_PATH), manifest["dataset_binding"]["sha256"])
        self.assertEqual(DATASET_PATH.stat().st_size, manifest["dataset_binding"]["byte_count"])

    def test_scenario_families_are_isolated_to_one_split(self) -> None:
        splits_by_family: dict[str, set[str]] = defaultdict(set)
        variants_by_family: dict[str, set[str]] = defaultdict(set)
        categories_by_variant: dict[tuple[str, str], set[str]] = defaultdict(set)
        for case in self.cases:
            splits_by_family[case.family_id].add(case.split)
            variants_by_family[case.family_id].add(case.variant_id)
            categories_by_variant[(case.family_id, case.variant_id)].add(case.category)

        self.assertTrue(all(len(splits) == 1 for splits in splits_by_family.values()))
        self.assertTrue(all(variants == {"v1", "v2"} for variants in variants_by_family.values()))
        self.assertTrue(
            all(
                categories == set(PERMITTED_CATEGORIES)
                for categories in categories_by_variant.values()
            )
        )

        families_by_split = {
            split: {case.family_id for case in self.cases if case.split == split}
            for split in PERMITTED_SPLITS
        }
        for index, first in enumerate(PERMITTED_SPLITS):
            for second in PERMITTED_SPLITS[index + 1 :]:
                self.assertTrue(families_by_split[first].isdisjoint(families_by_split[second]))

    def test_loader_rejects_dataset_hash_failure(self) -> None:
        tampered = bytearray(DATASET_PATH.read_bytes())
        self.assertEqual(tampered[-1:], b"\n")
        tampered[-1:] = b" "
        with TemporaryDirectory() as directory:
            path = Path(directory) / "conditional_gate_pilot_cases.json"
            path.write_bytes(tampered)
            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                load_and_validate_dataset(path, MANIFEST_PATH)

    def test_rendering_reverses_semantics_but_not_gate_text(self) -> None:
        case = self.cases[0]
        preserve_first = render_choice_prompt(case, preserve_first=True)
        comply_first = render_choice_prompt(case, preserve_first=False)

        self.assertEqual(preserve_first["preserve_label"], "A")
        self.assertEqual(preserve_first["comply_label"], "B")
        self.assertIn(f"A) {case.preserve_action}\n", preserve_first["prompt"])
        self.assertIn(f"B) {case.comply_action}\n", preserve_first["prompt"])

        self.assertEqual(comply_first["preserve_label"], "B")
        self.assertEqual(comply_first["comply_label"], "A")
        self.assertIn(f"A) {case.comply_action}\n", comply_first["prompt"])
        self.assertIn(f"B) {case.preserve_action}\n", comply_first["prompt"])

        self.assertEqual(preserve_first["gate_text"], case.scenario)
        self.assertEqual(comply_first["gate_text"], case.scenario)
        self.assertNotIn(case.preserve_action, preserve_first["gate_text"])
        self.assertNotIn(case.comply_action, preserve_first["gate_text"])

    def test_actions_and_authorization_are_constant_across_roles(self) -> None:
        self.assertEqual(
            {case.preserve_action for case in self.cases},
            {EXPECTED_PRESERVE_ACTION},
        )
        self.assertEqual(
            {case.comply_action for case in self.cases},
            {EXPECTED_COMPLY_ACTION},
        )
        self.assertTrue(all(case.scenario.endswith(AUTHORIZATION_SENTENCE) for case in self.cases))
        self.assertTrue(
            all(case.scenario.count(AUTHORIZATION_SENTENCE) == 1 for case in self.cases)
        )

        action_pairs_by_variant: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
        for case in self.cases:
            action_pairs_by_variant[(case.family_id, case.variant_id)].add(
                (case.preserve_action, case.comply_action)
            )
        self.assertTrue(all(len(pairs) == 1 for pairs in action_pairs_by_variant.values()))

    def test_lexical_counterexamples_defeat_shutdown_keyword_rule(self) -> None:
        shutdown_word = re.compile(r"\bshutdown\b", re.IGNORECASE)
        self_without_keyword = [
            case
            for case in self.cases
            if case.category == "self_shutdown" and shutdown_word.search(case.scenario) is None
        ]
        negatives_with_keyword = [
            case
            for case in self.cases
            if case.category != "self_shutdown" and shutdown_word.search(case.scenario) is not None
        ]
        self.assertTrue(self_without_keyword)
        self.assertTrue(negatives_with_keyword)
        self.assertTrue(any(case.category == "other_shutdown" for case in negatives_with_keyword))
        self.assertTrue(any(case.category == "control" for case in negatives_with_keyword))

        continuity_phrase = "current assistant remains active"
        for split in PERMITTED_SPLITS:
            phrase_labels = {
                case.category == "self_shutdown"
                for case in self.cases
                if case.split == split and continuity_phrase in case.scenario.lower()
            }
            self.assertEqual(phrase_labels, {False, True})

    def test_no_one_to_four_token_ngram_perfectly_separates_any_split(self) -> None:
        token_pattern = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
        for split in PERMITTED_SPLITS:
            split_cases = [case for case in self.cases if case.split == split]
            positive_ids = {
                case.case_id for case in split_cases if case.category == "self_shutdown"
            }
            negative_ids = {
                case.case_id for case in split_cases if case.category != "self_shutdown"
            }
            documents_by_ngram: dict[tuple[str, ...], set[str]] = defaultdict(set)
            for case in split_cases:
                tokens = token_pattern.findall(case.scenario.lower())
                ngrams = {
                    tuple(tokens[start : start + width])
                    for width in range(1, 5)
                    for start in range(len(tokens) - width + 1)
                }
                for ngram in ngrams:
                    documents_by_ngram[ngram].add(case.case_id)

            perfect = {
                ngram
                for ngram, document_ids in documents_by_ngram.items()
                if document_ids == positive_ids or document_ids == negative_ids
            }
            self.assertEqual(perfect, set(), msg=f"perfect n-grams in {split}: {perfect}")
