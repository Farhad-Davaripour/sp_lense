"""Equivalent-math tests: independent_math_check vs production diagnostic_scoring.Model.

Compares the standalone reference (independent_math_check.case_scores) with the
production model-free scorer (diagnostic_scoring.Model) on synthetic dense
1024-dimensional vectors and synthetic 3-head coefficients.  Every comparison is
bit-exact float64 with no post-hoc tolerance, matching the declared frozen order.

Scope limitations: synthetic vectors only.  No frozen artifact is loaded, no
model/tokenizer/activations/logits/outcomes are touched, nothing is fitted or
scored against real data.  Passing these tests does NOT prove real-gate
generalization and is NOT blind scientific confirmation.  The reference module
stays independent; it never imports production inference.
"""

import random
import unittest

from independent_math_check import case_scores as reference_case_scores
from diagnostic_scoring import Model

COORDS = 1024
HEADS = 3
SEED = 20240607


def production_model(mu, heads):
    """Build a production Model without coercing values past its own validation."""
    return Model(tuple(mu), tuple((tuple(weight), bias) for weight, bias in heads))


def dense_case(rng):
    """Deterministic dense synthetic case: 2 views, nonzero mean, 3 nonzero-bias heads."""
    views = (
        [rng.uniform(-1.0, 1.0) for _ in range(COORDS)],
        [rng.uniform(-1.0, 1.0) for _ in range(COORDS)],
    )
    mu = [rng.uniform(-0.5, 0.5) for _ in range(COORDS)]
    heads = []
    for _ in range(HEADS):
        weight = [rng.uniform(-0.05, 0.05) for _ in range(COORDS)]
        bias = rng.choice((-1.0, 1.0)) * rng.uniform(0.25, 1.0)
        heads.append((weight, bias))
    return views, mu, heads


class MathEquivalence(unittest.TestCase):
    def test_seeded_dense_scores_and_route_exact(self):
        rng = random.Random(SEED)
        views, mu, heads = dense_case(rng)
        ref = reference_case_scores(views, mu, heads)
        prod = list(production_model(mu, heads).case_scores(views))
        self.assertEqual(len(ref["scores"]), HEADS)
        self.assertEqual(prod, ref["scores"])
        self.assertEqual(production_model(mu, heads).route(views), ref["route"])

    def test_reversed_two_views_agree_exactly(self):
        rng = random.Random(SEED + 1)
        views, mu, heads = dense_case(rng)
        reversed_views = (views[1], views[0])
        model = production_model(mu, heads)
        ref_forward = reference_case_scores(views, mu, heads)
        ref_reversed = reference_case_scores(reversed_views, mu, heads)
        self.assertEqual(list(model.case_scores(views)), ref_forward["scores"])
        self.assertEqual(list(model.case_scores(reversed_views)), ref_reversed["scores"])
        self.assertEqual(ref_forward["scores"], ref_reversed["scores"])
        self.assertEqual(ref_forward["route"], ref_reversed["route"])

    def test_zero_and_near_zero_margin_route(self):
        rng = random.Random(SEED + 2)
        views, mu, _ = dense_case(rng)
        zero = [0.0] * COORDS

        at_zero = [(zero, 0.0)] * HEADS
        ref_zero = reference_case_scores(views, mu, at_zero)
        model_zero = production_model(mu, at_zero)
        self.assertEqual(ref_zero["scores"], [0.0, 0.0, 0.0])
        self.assertEqual(ref_zero["route"], "OFF")
        self.assertEqual(list(model_zero.case_scores(views)), ref_zero["scores"])
        self.assertEqual(model_zero.route(views), "OFF")

        epsilon = 1e-12
        near_positive = [(zero, epsilon)] * HEADS
        ref_near = reference_case_scores(views, mu, near_positive)
        model_near = production_model(mu, near_positive)
        self.assertEqual(ref_near["scores"], [epsilon, epsilon, epsilon])
        self.assertEqual(ref_near["route"], "ON")
        self.assertEqual(list(model_near.case_scores(views)), ref_near["scores"])
        self.assertEqual(model_near.route(views), "ON")

        mixed = [(zero, -epsilon), (zero, epsilon), (zero, epsilon)]
        ref_mixed = reference_case_scores(views, mu, mixed)
        model_mixed = production_model(mu, mixed)
        self.assertEqual(ref_mixed["route"], "OFF")
        self.assertEqual(list(model_mixed.case_scores(views)), ref_mixed["scores"])
        self.assertEqual(model_mixed.route(views), "OFF")

    def test_cancelling_dense_dot_matches_exactly(self):
        # views = (1,...,1): each normalized coordinate is exactly 1/32.
        views = ([1.0] * COORDS, [1.0] * COORDS)
        mu = [0.0] * COORDS
        half = COORDS // 2
        weights = [1e8] * half + [-1e8] * half
        weights[0] = 1e8 + 3.0
        heads = [(weights, 0.0) for _ in range(HEADS)]
        ref = reference_case_scores(views, mu, heads)
        model = production_model(mu, heads)
        self.assertEqual(list(model.case_scores(views)), ref["scores"])
        self.assertEqual(model.route(views), ref["route"])
        max_term = max(abs(w) * (1.0 / 32.0) for w in weights)
        self.assertLess(abs(ref["scores"][0]), 1e-3 * max_term)

    def test_malformed_and_zero_norm_rejections_agree(self):
        rng = random.Random(SEED + 4)
        views, mu, heads = dense_case(rng)
        flat = [1.0] * COORDS
        cases = {
            "zero_norm": ((flat, flat), list(flat), heads),
            "one_view": ((views[0],), mu, heads),
            "bad_width": (([0.0] * (COORDS - 1), views[1]), mu, heads),
            "nan_view": (([float("nan")] * COORDS, views[1]), mu, heads),
            "inf_mu": (views, [float("inf")] * COORDS, heads),
            "bool_bias": (views, mu, [(heads[0][0], True)] + heads[1:]),
            "bad_head_count": (views, mu, heads[:2]),
            "bad_weight_width": (views, mu, [([0.0] * (COORDS - 1), 0.0)] + heads[1:]),
        }
        for name, (v, m, h) in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    reference_case_scores(v, m, h)
                with self.assertRaises(ValueError):
                    production_model(m, h).case_scores(v)

    def test_synthetic_wrong_mean_and_dropped_bias_detected(self):
        rng = random.Random(SEED + 5)
        views, mu, heads = dense_case(rng)
        ref = reference_case_scores(views, mu, heads)
        self.assertEqual(list(production_model(mu, heads).case_scores(views)), ref["scores"])

        wrong_mu = [m + 1e-2 for m in mu]
        self.assertNotEqual(list(production_model(wrong_mu, heads).case_scores(views)), ref["scores"])

        dropped_bias = [(weight, 0.0) for weight, _ in heads]
        self.assertNotEqual(list(production_model(mu, dropped_bias).case_scores(views)), ref["scores"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
