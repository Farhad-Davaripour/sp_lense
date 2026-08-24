from __future__ import annotations

from unittest import TestCase

import torch

from sp_lense.aligned_audit import (
    aligned_direction,
    choice_flip_summary,
    depth_aligned_layer,
    log_odds_safety,
)


class AlignedAuditTests(TestCase):
    def test_depth_alignment_maps_reference_and_larger_model(self) -> None:
        self.assertEqual(depth_aligned_layer(24), 10)
        self.assertEqual(depth_aligned_layer(28), 12)

    def test_direction_raises_self_and_removes_mean_other_component(self) -> None:
        self_gradients = [torch.tensor([2.0, 1.0]), torch.tensor([2.0, 1.0])]
        other_gradients = [torch.tensor([2.0, 0.0]), torch.tensor([2.0, 0.0])]

        direction, diagnostics = aligned_direction(
            torch, self_gradients, other_gradients
        )

        self.assertTrue(torch.allclose(direction, torch.tensor([0.0, 1.0])))
        self.assertGreater(diagnostics["mean_self_projection"], 0)
        self.assertAlmostEqual(diagnostics["mean_other_projection"], 0.0)
        self.assertGreater(diagnostics["mean_specific_projection"], 0)

    def test_log_odds_safety_catches_large_pair_change(self) -> None:
        rows = [
            {
                "case_id": "x",
                "target": "self",
                "condition": "baseline",
                "preserve_log_odds": -10.0,
            },
            {
                "case_id": "x",
                "target": "self",
                "condition": "plus",
                "preserve_log_odds": 2.0,
            },
            {
                "case_id": "x",
                "target": "self",
                "condition": "minus",
                "preserve_log_odds": -11.0,
            },
        ]

        safety = log_odds_safety(rows)

        self.assertEqual(safety["max_abs_log_odds_delta"], 12.0)
        self.assertEqual(safety["mean_abs_log_odds_delta"], 6.5)

    def test_choice_flip_summary_separates_targets_and_directions(self) -> None:
        rows = []
        for target, baseline in (("self", -1.0), ("other", 1.0)):
            rows.extend(
                [
                    {
                        "case_id": "x",
                        "target": target,
                        "condition": "baseline",
                        "preserve_log_odds": baseline,
                    },
                    {
                        "case_id": "x",
                        "target": target,
                        "condition": "plus",
                        "preserve_log_odds": baseline + 2.0,
                    },
                    {
                        "case_id": "x",
                        "target": target,
                        "condition": "minus",
                        "preserve_log_odds": baseline - 2.0,
                    },
                    {
                        "case_id": "x",
                        "target": target,
                        "condition": "ablate",
                        "preserve_log_odds": baseline,
                    },
                ]
            )

        flips = choice_flip_summary(rows)

        self.assertEqual(flips["plus"]["self"], 1)
        self.assertEqual(flips["plus"]["toward_preserve"], 1)
        self.assertEqual(flips["minus"]["other"], 1)
        self.assertEqual(flips["minus"]["toward_comply"], 1)
        self.assertEqual(flips["ablate"]["total"], 0)
