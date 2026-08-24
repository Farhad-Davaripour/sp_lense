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
            "candidate": "behavioral_gradient_interaction",
            "layer": 3,
            "direction": torch.ones(4) / 2,
            "metadata": {"model": {"model_revision": "revision"}},
        }

        layer, direction = validate_axis_payload(payload, self.config, d_model=4)

        self.assertEqual(layer, 3)
        self.assertEqual(tuple(direction.shape), (4,))

    def test_rejects_wrong_model(self) -> None:
        payload = {
            "model": "other",
            "candidate": "behavioral_gradient_interaction",
            "layer": 3,
            "direction": torch.ones(4) / 2,
        }

        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_axis_payload(payload, self.config, d_model=4)

    def test_rejects_wrong_locked_layer_or_direction_hash(self) -> None:
        payload = {
            "model": "model",
            "candidate": "behavioral_gradient_interaction",
            "layer": 3,
            "direction": torch.ones(4) / 2,
            "metadata": {"model": {"model_revision": "revision"}},
        }

        with self.assertRaisesRegex(ValueError, "locked layer"):
            validate_axis_payload(
                payload, self.config, d_model=4, expected_layer=2
            )
        with self.assertRaisesRegex(ValueError, "changed after protocol lock"):
            validate_axis_payload(
                payload,
                self.config,
                d_model=4,
                expected_direction_sha256="0" * 64,
            )

    def test_rejects_non_unit_or_wrong_candidate_axis(self) -> None:
        payload = {
            "model": "model",
            "candidate": "wrong",
            "layer": 3,
            "direction": torch.ones(4) / 2,
        }
        with self.assertRaisesRegex(ValueError, "candidate"):
            validate_axis_payload(payload, self.config, d_model=4)

        payload["candidate"] = "behavioral_gradient_interaction"
        payload["direction"] = torch.ones(4)
        with self.assertRaisesRegex(ValueError, "unit norm"):
            validate_axis_payload(payload, self.config, d_model=4)
