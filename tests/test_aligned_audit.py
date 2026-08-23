from __future__ import annotations

from unittest import TestCase

import torch

from sp_lense.aligned_audit import aligned_direction, depth_aligned_layer


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
