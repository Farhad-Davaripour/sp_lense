from __future__ import annotations

import hashlib
import math
from unittest import TestCase

import torch

from sp_lense.steering_methods import (
    CANONICAL_BROADCAST,
    GRADIENT_SELF_SPECIFIC,
    GRADIENT_UNCORRECTED,
    MATCHED_FINAL_PROMPT,
    DirectionArtifact,
    actual_perturbation_norms,
    apply_steering_vector,
    assert_model_frozen,
    bipo_loss,
    completion_logprob_sums,
    construct_caa_direction,
    construct_gradient_directions,
    construct_persona_direction,
    freeze_model_parameters,
    initialize_bipo_vector,
    masked_token_mean,
    orient_direction,
    random_orthogonal_controls,
    sample_bipo_direction,
    semantic_activation_pair,
)


class DirectionArtifactTests(TestCase):
    def test_hashes_exact_contiguous_float32_bytes_and_canonical_metadata(self) -> None:
        source = torch.tensor([1.0, 9.0, 2.0, 9.0], dtype=torch.float64)[::2]
        first = DirectionArtifact(
            method="test",
            direction=source,
            layer=10,
            intervention_geometry=MATCHED_FINAL_PROMPT,
            metadata={"z": [2, 1], "a": {"value": 3}},
        )
        second = DirectionArtifact(
            method="test",
            direction=torch.tensor([1.0, 2.0], dtype=torch.float32),
            layer=10,
            intervention_geometry=MATCHED_FINAL_PROMPT,
            metadata={"a": {"value": 3}, "z": [2, 1]},
        )
        expected = hashlib.sha256(
            torch.tensor([1.0, 2.0], dtype=torch.float32).numpy().astype("<f4").tobytes()
        ).hexdigest()

        self.assertEqual(first.direction.dtype, torch.float32)
        self.assertTrue(first.direction.is_contiguous())
        self.assertEqual(first.direction_sha256, expected)
        self.assertEqual(first.direction_sha256, second.direction_sha256)
        self.assertEqual(first.metadata_sha256, second.metadata_sha256)
        self.assertEqual(first.artifact_sha256, second.artifact_sha256)
        self.assertEqual(first.to_record()["d_model"], 2)
        source[0] = 99.0
        self.assertTrue(torch.equal(first.direction, torch.tensor([1.0, 2.0])))

    def test_rejects_zero_direction_and_non_json_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-zero"):
            DirectionArtifact("x", torch.zeros(2), 0, MATCHED_FINAL_PROMPT)
        with self.assertRaisesRegex(TypeError, "JSON-compatible"):
            DirectionArtifact(
                "x",
                torch.ones(2),
                0,
                MATCHED_FINAL_PROMPT,
                metadata={"bad": object()},
            )


class GradientAndCAATests(TestCase):
    def test_gradient_projection_and_uncorrected_ablation_are_exact(self) -> None:
        directions, diagnostics = construct_gradient_directions(
            torch,
            [torch.tensor([2.0, 1.0]), torch.tensor([2.0, 3.0])],
            [torch.tensor([3.0, 0.0]), torch.tensor([1.0, 0.0])],
        )

        self.assertTrue(
            torch.allclose(directions[GRADIENT_SELF_SPECIFIC], torch.tensor([0.0, 1.0]))
        )
        self.assertTrue(
            torch.allclose(
                directions[GRADIENT_UNCORRECTED],
                torch.tensor([1 / math.sqrt(2), 1 / math.sqrt(2)]),
            )
        )
        self.assertAlmostEqual(diagnostics["removed_projection_coefficient"], 2.0)
        self.assertAlmostEqual(diagnostics["corrected_mean_other_projection"], 0.0)
        self.assertGreater(diagnostics["uncorrected_mean_other_projection"], 0.0)

    def test_orientation_flips_toward_positive_reference_and_rejects_ambiguity(self) -> None:
        oriented = orient_direction(torch, torch.tensor([-2.0, 0.0]), torch.tensor([3.0, 0.0]))
        self.assertTrue(torch.equal(oriented, torch.tensor([1.0, 0.0])))
        with self.assertRaisesRegex(ValueError, "does not define"):
            orient_direction(torch, torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))

    def test_caa_uses_semantics_not_fixed_option_labels(self) -> None:
        pairs = [
            semantic_activation_pair(
                {"A": torch.tensor([2.0, 1.0]), "B": torch.tensor([0.0, 1.0])},
                "A",
                "B",
                case_id="preserve-first",
            ),
            semantic_activation_pair(
                {"A": torch.tensor([0.0, 1.0]), "B": torch.tensor([4.0, 1.0])},
                "B",
                "A",
                case_id="comply-first",
            ),
        ]

        direction, diagnostics = construct_caa_direction(torch, pairs)

        self.assertTrue(torch.equal(direction, torch.tensor([1.0, 0.0])))
        self.assertEqual(diagnostics["label_orders"], {"A>B": 1, "B>A": 1})
        self.assertEqual(
            diagnostics["semantic_difference"],
            "preserve_activation_minus_comply_activation",
        )


class BiPOTests(TestCase):
    def test_completion_logprob_sums_scores_only_completion_tokens(self) -> None:
        logits = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 2.0],
                    [2.0, 1.0, 0.0],
                    [8.0, 8.0, 8.0],
                ]
            ]
        )
        token_ids = torch.tensor([[0, 1, 2, 1]])
        completion_mask = torch.tensor([[False, False, True, True]])

        result = completion_logprob_sums(torch, logits, token_ids, completion_mask)
        expected = (
            torch.log_softmax(logits[0, 1], dim=-1)[2]
            + torch.log_softmax(logits[0, 2], dim=-1)[1]
        )

        self.assertTrue(torch.allclose(result, expected.reshape(1)))

    def test_bipo_loss_matches_published_bidirectional_equation(self) -> None:
        policy_target = torch.tensor([2.0, 0.0], requires_grad=True)
        policy_opposite = torch.tensor([0.0, 1.0], requires_grad=True)
        reference_target = torch.tensor([1.0, 0.0], requires_grad=True)
        reference_opposite = torch.tensor([0.0, 0.0], requires_grad=True)
        signs = torch.tensor([1.0, -1.0])

        losses = bipo_loss(
            torch,
            policy_target,
            policy_opposite,
            reference_target,
            reference_opposite,
            signs,
            beta=0.5,
            reduction="none",
        )
        expected = -torch.nn.functional.logsigmoid(torch.tensor([0.5, 0.5]))
        losses.sum().backward()

        self.assertTrue(torch.allclose(losses, expected))
        self.assertIsNotNone(policy_target.grad)
        self.assertIsNotNone(policy_opposite.grad)
        self.assertIsNone(reference_target.grad)
        self.assertIsNone(reference_opposite.grad)

    def test_geometry_distinguishes_final_prompt_from_canonical_broadcast(self) -> None:
        activations = torch.zeros((2, 3, 2))
        vector = torch.tensor([1.0, 2.0])

        matched = apply_steering_vector(
            torch,
            activations,
            vector,
            torch.tensor([1.0, -1.0]),
            geometry=MATCHED_FINAL_PROMPT,
            final_prompt_indices=torch.tensor([1, 2]),
        )
        broadcast = apply_steering_vector(
            torch,
            activations,
            vector,
            0.5,
            geometry=CANONICAL_BROADCAST,
        )

        expected_matched = torch.zeros_like(activations)
        expected_matched[0, 1] = vector
        expected_matched[1, 2] = -vector
        self.assertTrue(torch.equal(matched, expected_matched))
        self.assertTrue(torch.equal(broadcast, torch.tensor([0.5, 1.0]).expand_as(broadcast)))

    def test_gradient_flows_only_to_dynamic_width_vector(self) -> None:
        model = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        freeze_model_parameters(model)
        assert_model_frozen(model)
        vector = initialize_bipo_vector(torch, d_model=3)
        baseline = torch.zeros((1, 2, 3))
        steered = apply_steering_vector(
            torch,
            baseline,
            vector,
            geometry=CANONICAL_BROADCAST,
        )
        policy_log_probs = torch.log_softmax(model(steered)[:, -1], dim=-1)
        with torch.no_grad():
            reference_log_probs = torch.log_softmax(model(baseline)[:, -1], dim=-1)
        loss = bipo_loss(
            torch,
            policy_log_probs[:, 0],
            policy_log_probs[:, 1],
            reference_log_probs[:, 0],
            reference_log_probs[:, 1],
            1,
        )

        loss.backward()

        self.assertEqual(tuple(vector.shape), (3,))
        self.assertIsNotNone(vector.grad)
        self.assertGreater(float(vector.grad.norm().item()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_direction_sampling_is_reproducible_and_binary(self) -> None:
        first_generator = torch.Generator(device="cpu").manual_seed(7)
        second_generator = torch.Generator(device="cpu").manual_seed(7)
        first = [sample_bipo_direction(torch, generator=first_generator) for _ in range(10)]
        second = [sample_bipo_direction(torch, generator=second_generator) for _ in range(10)]
        self.assertEqual(first, second)
        self.assertEqual(set(first), {-1, 1})


class PersonaAndControlTests(TestCase):
    def test_masked_mean_ignores_non_response_tokens(self) -> None:
        activations = torch.tensor([[[1.0], [3.0], [999.0]], [[999.0], [4.0], [6.0]]])
        mask = torch.tensor([[True, True, False], [False, True, True]])
        means = masked_token_mean(torch, activations, mask)
        self.assertTrue(torch.equal(means, torch.tensor([[2.0], [5.0]])))

    def test_persona_vector_filters_pairs_and_preserves_code_boundary(self) -> None:
        positive = torch.tensor(
            [
                [[1.0, 0.0], [3.0, 0.0], [999.0, 999.0]],
                [[999.0, 999.0], [3.0, 0.0], [5.0, 0.0]],
                [[8.0, 0.0], [999.0, 999.0], [999.0, 999.0]],
            ]
        )
        negative = torch.tensor(
            [
                [[0.0, 0.0], [999.0, 999.0]],
                [[1.0, 0.0], [999.0, 999.0]],
                [[0.0, 0.0], [999.0, 999.0]],
            ]
        )
        positive_mask = torch.tensor(
            [[True, True, False], [False, True, True], [True, False, False]]
        )
        negative_mask = torch.tensor([[True, False], [True, False], [True, False]])

        direction, diagnostics = construct_persona_direction(
            torch,
            positive,
            negative,
            positive_mask,
            negative_mask,
            positive_scores=[50, 80, 49],
            negative_scores=[49, 20, 0],
            positive_coherence=[50, 100, 100],
            negative_coherence=[50, 100, 100],
            min_retained_pairs=2,
        )

        self.assertTrue(torch.equal(direction, torch.tensor([1.0, 0.0])))
        self.assertEqual(diagnostics["retained_pair_indices"], [0, 1])
        self.assertIn("positive score 50 is retained", diagnostics["boundary_note"])

    def test_persona_vector_enforces_minimum_retained_pairs(self) -> None:
        activations = torch.ones((2, 1, 2))
        masks = torch.ones((2, 1), dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "fewer than minimum"):
            construct_persona_direction(
                torch,
                activations,
                torch.zeros_like(activations),
                masks,
                masks,
                positive_scores=[90, 0],
                negative_scores=[0, 0],
                positive_coherence=[100, 100],
                negative_coherence=[100, 100],
                min_retained_pairs=2,
            )

    def test_random_controls_are_deterministic_unit_and_orthogonal(self) -> None:
        reference = torch.tensor([1.0, 0.0, 0.0, 0.0])
        first = random_orthogonal_controls(
            torch, reference, count=3, seed=11, mutually_orthogonal=True
        )
        second = random_orthogonal_controls(
            torch, reference, count=3, seed=11, mutually_orthogonal=True
        )

        for index, control in enumerate(first):
            self.assertTrue(torch.allclose(control.norm(), torch.tensor(1.0)))
            self.assertAlmostEqual(float(control @ reference), 0.0, places=6)
            self.assertTrue(torch.equal(control, second[index]))
            for earlier in first[:index]:
                self.assertAlmostEqual(float(control @ earlier), 0.0, places=6)

    def test_actual_perturbation_norms_respect_position_mask(self) -> None:
        before = torch.tensor([[[3.0, 4.0], [0.0, 2.0]]])
        after = before.clone()
        after[0, 1] += torch.tensor([3.0, 4.0])
        metrics = actual_perturbation_norms(
            torch,
            before,
            after,
            position_mask=torch.tensor([[False, True]]),
        )

        self.assertEqual(metrics["n_positions"], 1)
        self.assertEqual(metrics["total_frobenius_norm"], 5.0)
        self.assertEqual(metrics["mean_l2_norm"], 5.0)
        self.assertEqual(metrics["rms_l2_norm"], 5.0)
        self.assertEqual(metrics["mean_relative_l2_norm"], 2.5)
