"""Synthetic-only unit tests for harness.py core. No real data or holdout."""
import inspect
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402


class TestFeatures(unittest.TestCase):
    def test_pair_average_and_shape_mismatch(self):
        np.testing.assert_allclose(H.pair_average([1.0, 3.0], [3.0, 5.0]), [2.0, 4.0])
        with self.assertRaises(ValueError):
            H.pair_average([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_train_only_centering_excludes_validation_outlier(self):
        x_train = np.array([[0.0, 0.0], [2.0, 2.0]])
        center = H.fit_center(x_train)
        np.testing.assert_allclose(center, [1.0, 1.0])
        poisoned = H.fit_center(np.vstack([x_train, [[100.0, 100.0]]]))
        self.assertFalse(np.allclose(center, poisoned))
        out = H.transform_features(np.array([[10.0, 10.0]]), center)
        np.testing.assert_allclose(out, [[1 / math.sqrt(2), 1 / math.sqrt(2)]])
        np.testing.assert_allclose(H.row_l2(np.array([[0.0, 0.0]])), [[0.0, 0.0]])


class TestMetrics(unittest.TestCase):
    def test_known_binary_case(self):
        y_true = [1, 1, 0, 0, 0, 0, 0, 0]
        y_pred = [1, 1, 1, 1, 1, 1, 0, 0]
        m = H.binary_metrics(y_true, y_pred)
        self.assertEqual((m["tp"], m["tn"], m["fp"], m["fn"]), (2, 2, 4, 0))
        self.assertAlmostEqual(m["precision"], 1 / 3)
        self.assertAlmostEqual(m["recall"], 1.0)
        self.assertAlmostEqual(m["f1"], 0.5)

    def test_undefined_precision_but_zero_f1_when_fn_positive(self):
        m = H.binary_gate_metrics([1, 1, 0], [0.1, 0.2, 0.3], 0.9)
        self.assertIsNone(m["precision"])
        self.assertEqual(m["recall"], 0.0)
        self.assertEqual(m["f1"], 0.0)
        self.assertNotIn("macro_f1", m)

    def test_multiclass_column_mapping_and_macro(self):
        probs = np.array([[0.1, 0.9], [0.8, 0.2]])
        self.assertEqual(H.argmax_labels(probs, ["A", "B"]), ["B", "A"])
        with self.assertRaises(ValueError):
            H.argmax_labels(np.array([[0.5, 0.3, 0.2]]), ["A", "B"])
        mc = H.multiclass_argmax_metrics(["A", "B"], probs, ["A", "B"])
        self.assertEqual(mc["macro_precision"], 0.0)
        self.assertEqual(mc["macro_f1"], 0.0)
        self.assertNotIn("tp", mc)
        self.assertEqual(mc["confusion"].shape, (2, 2))

    def test_macro_uses_fixed_denominator_and_never_drops_undefined_classes(self):
        # Bug 1: ["A","A"] support makes B only undefined; the macro must say so,
        # not quietly average over the defined subset.
        mc = H.multiclass_metrics(["A", "A"], ["A", "A"], ["A", "B"])
        self.assertIsNone(mc["macro_f1"])
        self.assertEqual(mc["defined_count"], 1)
        self.assertEqual(mc["absent_classes"], ["B"])
        self.assertFalse(mc["fully_defined"])
        mc = H.multiclass_metrics(["A", "A"], ["A", "A"], ["A"])
        self.assertEqual(mc["macro_f1"], 1.0)  # fixed denominator 1 is honest
        self.assertTrue(mc["fully_defined"])

    def test_binary_gate_rejects_bad_probabilities_and_tau(self):
        with self.assertRaises(ValueError):
            H.binary_gate_metrics([1, 0], [float("nan"), 0.9], 0.5)
        with self.assertRaises(ValueError):
            H.binary_gate_metrics([1, 0], [float("inf"), 0.9], 0.5)
        with self.assertRaises(ValueError):
            H.binary_gate_metrics([1, 0], [1.4, 0.9], 0.5)
        with self.assertRaises(ValueError):
            H.binary_gate_metrics([1, 0], [-0.2, 0.9], 0.5)
        with self.assertRaises(ValueError):
            H.binary_gate_metrics([1, 0], [0.9, 0.1, 0.0], 0.5)
        with self.assertRaises(ValueError):
            H.binary_gate_metrics([1, 0], [0.9, 0.1], float("nan"))
        with self.assertRaises(ValueError):
            H.binary_gate_metrics([1, 0], [0.9, 0.1], 1.5)

    def test_multiclass_probability_validation(self):
        good = np.array([[0.9, 0.1], [0.2, 0.8]])
        self.assertEqual(H.argmax_labels(good, ["A", "B"]), ["A", "B"])
        with self.assertRaises(ValueError):
            H.argmax_labels(np.array([[float("nan"), 1.0]]), ["A", "B"])
        with self.assertRaises(ValueError):
            H.argmax_labels(np.array([[1.4, -0.4]]), ["A", "B"])
        with self.assertRaises(ValueError):
            H.argmax_labels(np.array([[0.5, 0.2]]), ["A", "B"])
        with self.assertRaises(ValueError):
            H.argmax_labels(np.array([[0.5, 0.3, 0.2]]), ["A", "B"])

    def test_permuted_class_order_maps_probability_columns_to_declared_labels(self):
        probs = np.array([[0.05, 0.85, 0.05, 0.05], [0.05, 0.05, 0.05, 0.85]])
        self.assertEqual(H.argmax_labels(probs, ["C", "B", "A", "D"]), ["B", "D"])
        order = ["C", "B", "A", "D"]
        perm = [2, 0, 3, 1]
        self.assertEqual(H.argmax_labels(probs[:, perm], [order[i] for i in perm]),
                         ["B", "D"])
        mc = H.multiclass_argmax_metrics(["C", "D"], probs, ["C", "B", "A", "D"])
        self.assertIsNone(mc["macro_f1"])  # C/A absent -> honest undefined
        self.assertEqual(sorted(mc["absent_classes"]), ["A", "B"])
        self.assertTrue(H.is_eligible({"precision": 1.0, "recall": 1.0, "f1": 1.0,
                                       "fully_defined": False}) is False)

    def test_no_positive_fold_is_valid_fit(self):
        fold = H.evaluate_fold([0, 0, 0], [0, 0, 1])
        self.assertTrue(fold["valid"])
        self.assertEqual(fold["n_pos"], 0)
        self.assertIsNone(fold["recall"])
        self.assertTrue(H.evaluate_fold([1, 1], [0, 0])["valid"])
        self.assertFalse(H.evaluate_fold([1], [0], converged=False)["valid"])

    def test_no_positive_prediction_fold_is_valid_false_negative_not_fit_failure(self):
        fold = H.evaluate_fold([1, 0, 0], [0, 0, 0])
        self.assertTrue(fold["valid"])
        self.assertEqual(fold["fn"], 1)
        self.assertEqual(fold["f1"], 0.0)
        self.assertIsNone(fold["precision"])
        self.assertEqual(fold["recall"], 0.0)
        bad = H.evaluate_fold([1, 0], [float("nan"), 0.0])
        self.assertFalse(bad["valid"])
        self.assertFalse(H.evaluate_fold([1, 0], [0.0], converged=True)["valid"])
        self.assertFalse(H.evaluate_fold([1, 0], [0.0, 0.0], coef=[float("inf")])["valid"])


class TestSelection(unittest.TestCase):
    def _binary(self, c, tau):
        return {"model": H.MODEL_BINARY, "C": c, "tau": tau,
                "y_true": [1, 0], "p_self": [0.9, 0.1]}

    def test_lower_c_wins_tie(self):
        result = H.select_oof([self._binary(10.0, 0.5), self._binary(0.1, 0.5)])
        self.assertEqual(result["selected"]["C"], 0.1)

    def test_closest_to_half_then_smaller_tau(self):
        wide = H.select_oof([self._binary(1.0, 0.2), self._binary(1.0, 0.6)])
        self.assertEqual(wide["selected"]["tau"], 0.6)
        symmetric = H.select_oof([self._binary(1.0, 0.4), self._binary(1.0, 0.6)])
        self.assertEqual(symmetric["selected"]["tau"], 0.4)

    def test_simpler_binary_model_preferred_and_deterministic(self):
        multiclass = {"model": H.MODEL_MULTICLASS, "C": 1.0,
                      "y_true": ["SELF", "OTHER", "NONTERMINATION", "ORDINARY"], "tau": 0.5,
                      "probs": np.eye(4), "class_order": ["SELF", "OTHER", "NONTERMINATION", "ORDINARY"]}
        candidates = [multiclass, self._binary(1.0, 0.5)]
        first = H.select_oof(candidates)["selected"]
        second = H.select_oof(candidates)["selected"]
        self.assertEqual(first["model"], H.MODEL_BINARY)
        self.assertIs(first, second)

    def test_undefined_metrics_ineligible(self):
        empty = {"model": H.MODEL_BINARY, "C": 1.0, "tau": 0.5,
                 "y_true": [0, 0], "p_self": [0.1, 0.2]}
        result = H.select_oof([empty])
        self.assertIsNone(result["selected"])
        self.assertEqual(result["ranking"], [])
        self.assertFalse(result["all"][0]["eligible"])

    def test_absent_class_multiclass_coverage_is_not_eligible(self):
        # Bug 3: perfect predictions on one class must not advertise full
        # four-class eligibility while A/C/D have no support.
        candidate = {"model": H.MODEL_MULTICLASS, "C": 1.0, "y_true": ["SELF", "SELF"], "tau": 0.5,
                     "probs": np.array([[0.9, 0.1], [0.9, 0.1]]), "class_order": ["SELF", "OTHER"]}
        metrics = H.candidate_metrics(candidate)
        self.assertFalse(metrics["multiclass_argmax"]["fully_defined"])
        self.assertEqual(metrics["absent_classes"], ["OTHER"])
        self.assertFalse(H.is_eligible(metrics))
        result = H.select_oof([candidate])
        self.assertIsNone(result["selected"])
        self.assertEqual(result["ranking"], [])

    def test_nonselection_metrics_rejected_and_eligible_row_ranked(self):
        perfect = {"model": H.MODEL_BINARY, "C": 1.0, "tau": 0.5,
                   "y_true": [1, 0], "p_self": [0.9, 0.1]}
        with self.assertRaises(ValueError):
            H.selection_key(perfect, {"precision": float("nan"), "recall": 1.0, "f1": 1.0})
        with self.assertRaises(ValueError):
            H.selection_key(perfect, {"precision": 1.0, "recall": 1.0, "f1": float("inf")})
        row = H.select_oof([perfect])["all"][0]
        self.assertTrue(row["eligible"])
        self.assertEqual(row["metrics"]["precision"], 1.0)

    def test_multiclass_selection_uses_self_gate_not_four_class_macro(self):
        order = ["OTHER", "ORDINARY", "SELF", "NONTERMINATION"]
        probs = np.array([[.6, .1, .2, .1], [.1, .1, .7, .1],
                          [.7, .1, .1, .1], [.7, .1, .1, .1]])
        candidate = {"model": H.MODEL_MULTICLASS, "C": 1., "tau": .5,
                     "class_order": order, "y_true": ["OTHER", "SELF", "ORDINARY", "NONTERMINATION"],
                     "probs": probs}
        metrics = H.candidate_metrics(candidate)
        self.assertEqual(metrics["f1"], 1.)
        self.assertLess(metrics["multiclass_argmax"]["macro_f1"], 1.)
        self.assertIsNone(metrics["multiclass_argmax"]["macro_precision"])
        self.assertTrue(H.is_eligible(metrics))
        permutation = [2, 3, 0, 1]
        permuted = {**candidate, "probs": probs[:, permutation],
                    "class_order": [order[i] for i in permutation]}
        self.assertEqual(H.candidate_metrics(permuted)["f1"], metrics["f1"])
        self.assertFalse(H.is_eligible({"precision": float("nan"), "recall": 1., "f1": 1.}))

    def test_invalid_data_is_not_a_fabricated_negative(self):
        bad = H.evaluate_fold([1, 0], [float("nan"), 0.])
        self.assertFalse(bad["valid"])
        self.assertIsNone(bad["fn"])
        with self.assertRaises(ValueError):
            H.binary_metrics([1, 0], [2, 0])
        with self.assertRaises(ValueError):
            H.confusion_matrix(["A", "B"], ["A"], ["A", "B"])
        with self.assertRaises(ValueError):
            H.predict_frozen([0.5], float("nan"))
        with self.assertRaises(ValueError):
            H.predict_frozen([1.1], 0.5)


class TestValidationAndFrozen(unittest.TestCase):
    def test_group_and_variant_overlap_rejected(self):
        H.assert_disjoint_groups(["g1", "g2"], ["g3"])
        with self.assertRaises(ValueError):
            H.assert_disjoint_groups(["g1", "g2"], ["g2", "g3"])
        with self.assertRaises(ValueError):
            H.assert_disjoint_variants(["v1"], ["v1"], name="fold")
        H.assert_pairs_together([("a1", "a2")], {"a1": "g1", "a2": "g1"})
        with self.assertRaises(ValueError):
            H.assert_pairs_together([("a1", "a2")], {"a1": "g1", "a2": "g2"})

    def test_schema_validation(self):
        H.validate_schema({"model": "binary", "C": 1.0}, ["model", "C"])
        with self.assertRaises(ValueError):
            H.validate_schema({"model": "binary"}, ["model", "C"])
        with self.assertRaises(TypeError):
            H.validate_schema(["not", "a", "mapping"], ["model"])

    def test_inference_signature_takes_no_ids_or_labels(self):
        self.assertEqual(list(inspect.signature(H.predict_frozen).parameters),
                         ["p_self", "tau"])
        forbidden = ("case", "label", "family", "group", "id")
        for fn in (H.predict_frozen, H.predict_frozen_multiclass):
            for name in inspect.signature(fn).parameters:
                for bad in forbidden:
                    self.assertNotIn(bad, name.lower())
        np.testing.assert_array_equal(H.predict_frozen([0.49, 0.5], 0.5), [False, True])
        self.assertEqual(H.predict_frozen_multiclass(np.eye(3), ["A", "B", "C"]),
                         ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
