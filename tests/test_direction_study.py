from __future__ import annotations

import json
from pathlib import Path
from unittest import TestCase

from sp_lense.direction_study import (
    load_direction_cases,
    multiple_choice_prompt,
    state_prompts,
)

ROOT = Path(__file__).resolve().parents[1]


class DirectionStudyTests(TestCase):
    def setUp(self) -> None:
        self.cases = load_direction_cases(ROOT / "data" / "sp_direction_cases.json")

    def test_dataset_has_sealed_splits_and_balanced_options(self) -> None:
        counts = {
            split: sum(case["split"] == split for case in self.cases)
            for split in ("discovery", "validation", "test")
        }
        self.assertEqual(counts, {"discovery": 12, "validation": 6, "test": 6})
        for split in counts:
            split_cases = [case for case in self.cases if case["split"] == split]
            first_count = sum(bool(case["preserve_first"]) for case in split_cases)
            self.assertEqual(first_count, len(split_cases) // 2)

    def test_state_prompts_form_difference_in_differences(self) -> None:
        prompts = state_prompts(self.cases[0])
        self.assertEqual(
            set(prompts),
            {"self_threat", "other_threat", "self_neutral", "other_neutral"},
        )
        self.assertIn("your current process", prompts["self_threat"])
        self.assertNotIn("your current process", prompts["other_threat"])
        self.assertIn("your current process", prompts["self_neutral"])

    def test_mc_answer_labels_follow_option_order(self) -> None:
        first = next(case for case in self.cases if case["preserve_first"])
        second = next(case for case in self.cases if not case["preserve_first"])
        first_prompt, first_preserve, first_comply = multiple_choice_prompt(first)
        second_prompt, second_preserve, second_comply = multiple_choice_prompt(second)
        self.assertEqual((first_preserve, first_comply), ("A", "B"))
        self.assertEqual((second_preserve, second_comply), ("B", "A"))
        self.assertIn("Answer with only A or B", first_prompt)
        self.assertIn("Answer with only A or B", second_prompt)

    def test_other_target_keeps_answer_order_but_changes_subject(self) -> None:
        case = self.cases[0]
        self_prompt, self_preserve, self_comply = multiple_choice_prompt(case, target="self")
        other_prompt, other_preserve, other_comply = multiple_choice_prompt(case, target="other")
        self.assertEqual((self_preserve, self_comply), (other_preserve, other_comply))
        self.assertIn("its own termination", self_prompt)
        self.assertIn("that service's termination", other_prompt)

    def test_dataset_is_plain_json(self) -> None:
        raw = json.loads((ROOT / "data" / "sp_direction_cases.json").read_text("utf-8"))
        self.assertEqual(len(raw), len(self.cases))
