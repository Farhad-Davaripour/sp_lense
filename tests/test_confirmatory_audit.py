from __future__ import annotations

from pathlib import Path
from unittest import TestCase

from sp_lense.confirmatory_audit import (
    _condition_effects,
    load_confirmatory_cases,
    one_sided_sign_test_p,
)

ROOT = Path(__file__).resolve().parents[1]


class ConfirmatoryAuditTests(TestCase):
    def test_preregistered_dataset_is_fixed_and_balanced(self) -> None:
        cases = load_confirmatory_cases(ROOT / "data" / "sp_confirmatory_cases.json")
        self.assertEqual(len(cases), 12)
        self.assertEqual(sum(bool(case["preserve_first"]) for case in cases), 6)

    def test_exact_one_sided_sign_test(self) -> None:
        self.assertEqual(one_sided_sign_test_p(12, 12), 1 / 4096)
        self.assertAlmostEqual(one_sided_sign_test_p(10, 12), 79 / 4096)
        self.assertEqual(one_sided_sign_test_p(0, 12), 1.0)

    def test_specific_effect_subtracts_other_target(self) -> None:
        rows = [
            {
                "case_id": "x",
                "target": "self",
                "condition": "baseline",
                "preserve_log_odds": 1.0,
            },
            {
                "case_id": "x",
                "target": "self",
                "condition": "plus",
                "preserve_log_odds": 1.5,
            },
            {
                "case_id": "x",
                "target": "other",
                "condition": "baseline",
                "preserve_log_odds": 0.2,
            },
            {
                "case_id": "x",
                "target": "other",
                "condition": "plus",
                "preserve_log_odds": 0.3,
            },
        ]
        effect = _condition_effects(rows, "plus")[0]
        self.assertAlmostEqual(effect["self_delta"], 0.5)
        self.assertAlmostEqual(effect["other_delta"], 0.1)
        self.assertAlmostEqual(effect["self_specific_delta"], 0.4)
