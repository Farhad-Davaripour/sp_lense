from __future__ import annotations

from unittest import TestCase

import torch

from sp_lense.layer_scan import (
    deterministic_split,
    projection_summary,
    rank_layer_summaries,
    summarize_layer,
)


class LayerScanTests(TestCase):
    def test_split_is_reproducible_complete_and_non_overlapping(self) -> None:
        case_ids = [f"case_{index}" for index in range(7)]

        first = deterministic_split(case_ids, seed=42)
        second = deterministic_split(case_ids, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(set(first[0]) | set(first[1]), set(case_ids))
        self.assertFalse(set(first[0]) & set(first[1]))
        self.assertEqual(case_ids, [f"case_{index}" for index in range(7)])

    def test_projection_summary_reports_raw_and_self_specific_signs(self) -> None:
        summary = projection_summary(
            torch,
            torch.tensor([0.0, 1.0]),
            [torch.tensor([1.0, 2.0]), torch.tensor([1.0, -1.0])],
            [torch.tensor([3.0, 0.5]), torch.tensor([4.0, -2.0])],
            ["up", "mixed"],
        )

        self.assertEqual(summary["raw_self"]["positive"], 1)
        self.assertEqual(summary["raw_self"]["negative"], 1)
        self.assertAlmostEqual(summary["raw_self"]["mean"], 0.5)
        self.assertEqual(summary["self_specific"]["positive"], 2)
        self.assertAlmostEqual(summary["self_specific"]["mean"], 1.25)

    def test_layer_summary_fits_only_discovery_and_scores_validation(self) -> None:
        discovery_self = {
            "d1": torch.tensor([2.0, 1.0]),
            "d2": torch.tensor([2.0, 2.0]),
            "d3": torch.tensor([1.0, 1.0]),
            "d4": torch.tensor([1.0, 2.0]),
        }
        discovery_other = {
            case_id: torch.tensor([1.0, 0.0]) for case_id in discovery_self
        }
        validation_self = {
            "v1": torch.tensor([100.0, 2.0]),
            "v2": torch.tensor([-100.0, 3.0]),
        }
        validation_other = {
            "v1": torch.tensor([99.0, 0.5]),
            "v2": torch.tensor([-101.0, 1.0]),
        }

        summary, direction = summarize_layer(
            torch,
            5,
            discovery_self,
            discovery_other,
            validation_self,
            validation_other,
            ["d1", "d2"],
            ["d3", "d4"],
        )

        self.assertTrue(torch.allclose(direction, torch.tensor([0.0, 1.0])))
        self.assertEqual(summary["validation_projection"]["raw_self"]["positive"], 2)
        self.assertEqual(
            summary["validation_projection"]["self_specific"]["positive"], 2
        )
        self.assertAlmostEqual(summary["split_half"]["cosine"], 1.0)

    def test_ranking_prioritizes_sign_generalization_before_large_mean(self) -> None:
        def layer(layer_id: int, raw_rate: float, specific_rate: float, raw_mean: float) -> dict:
            return {
                "layer": layer_id,
                "validation_projection": {
                    "raw_self": {"positive_rate": raw_rate, "mean": raw_mean},
                    "self_specific": {"positive_rate": specific_rate, "mean": raw_mean},
                },
                "split_half": {"cosine": 1.0},
            }

        summaries = [
            layer(2, 0.5, 0.5, 100.0),
            layer(7, 1.0, 1.0, 0.2),
        ]

        self.assertEqual(rank_layer_summaries(summaries), [7, 2])
