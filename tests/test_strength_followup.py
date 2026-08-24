from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase

import torch

from sp_lense.strength_followup import validate_axis_payload


class StrengthFollowupTests(TestCase):
    def setUp(self) -> None:
        self.config = SimpleNamespace(
            model=SimpleNamespace(id="model", revision="revision")
        )

    def test_validates_model_revision_and_shape(self) -> None:
        payload = {
            "model": "model",
            "layer": 3,
            "direction": torch.ones(4),
            "metadata": {"model": {"model_revision": "revision"}},
        }

        layer, direction = validate_axis_payload(payload, self.config, d_model=4)

        self.assertEqual(layer, 3)
        self.assertEqual(tuple(direction.shape), (4,))

    def test_rejects_wrong_model(self) -> None:
        payload = {"model": "other", "layer": 3, "direction": torch.ones(4)}

        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_axis_payload(payload, self.config, d_model=4)
