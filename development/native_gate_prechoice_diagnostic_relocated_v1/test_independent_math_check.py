"""Minimal known-answer tests for independent_math_check.case_scores.

Scope: 4 cases only. This is NOT a dense-equivalence or mutation review of the
production scorer; that work is deferred and unproved by these tests.
"""

import math
import unittest

from independent_math_check import case_scores

COORDS = 1024
WEIGHT = 1.0 / 1024.0


def unit_heads(biases=(0.0, 0.0, 0.0)):
    return [([WEIGHT] * COORDS, float(b)) for b in biases]


def base_views():
    return ([1.0] * COORDS, [3.0] * COORDS)


class CaseScoresKnownAnswers(unittest.TestCase):
    def test_dense_all_ones_route_on_score_one_over_32(self):
        views = base_views()
        result = case_scores(views, [1.0] * COORDS, unit_heads())
        self.assertEqual(result["route"], "ON")
        self.assertEqual(result["scores"], [1.0 / 32.0, 1.0 / 32.0, 1.0 / 32.0])

    def test_first_bias_negative_one_over_32_first_score_zero_route_off(self):
        views = base_views()
        result = case_scores(views, [1.0] * COORDS, unit_heads((-1.0 / 32.0, 0.0, 0.0)))
        self.assertEqual(result["scores"][0], 0.0)
        self.assertEqual(result["route"], "OFF")

    def test_first_bias_negative_one_over_16_first_score_negative_route_off(self):
        views = base_views()
        result = case_scores(views, [1.0] * COORDS, unit_heads((-1.0 / 16.0, 0.0, 0.0)))
        self.assertTrue(result["scores"][0] < 0.0)
        self.assertEqual(result["route"], "OFF")

    def test_rejections(self):
        good_views = base_views()
        good_mu = [1.0] * COORDS
        good_heads = unit_heads()

        cases = {
            "zero_delta": (([1.0] * COORDS, [1.0] * COORDS), [1.0] * COORDS, good_heads),
            "nan_view": (([float("nan")] * COORDS, [3.0] * COORDS), good_mu, good_heads),
            "inf_view": (([float("inf")] * COORDS, [3.0] * COORDS), good_mu, good_heads),
            "bool_view": (([True] * COORDS, [3.0] * COORDS), good_mu, good_heads),
            "bool_bias": (good_views, good_mu, [([WEIGHT] * COORDS, True)] * 3),
            "bad_view_count": (([1.0] * COORDS,), good_mu, good_heads),
            "bad_coord_count": (([1.0] * (COORDS - 1), [3.0] * COORDS), good_mu, good_heads),
            "bad_mu_length": (good_views, [1.0] * (COORDS - 1), good_heads),
            "bad_head_count": (good_views, good_mu, unit_heads()[:2]),
            "bad_weight_length": (good_views, good_mu, [([WEIGHT] * (COORDS - 1), 0.0)] * 3),
            "not_a_sequence": (42, good_mu, good_heads),
        }
        for name, args in cases.items():
            with self.subTest(case=name):
                self.assertRaises(ValueError, case_scores, *args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
