"""Fake-estimator driver tests; no Qwen, real data or scientific fits."""
import json
import unittest
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

import grouped_driver as D


def fixture():
    x = np.arange(80, dtype=float).reshape(40, 2)
    labels = np.tile(D.CANON, 10)
    groups = np.repeat(np.arange(10), 4)
    return x, labels, groups, groups % 5


class Factory:
    def __init__(self, mode="normal"):
        self.mode, self.calls = mode, []

    def __call__(self, model, C):
        owner, index = self, len(self.calls)
        record = {"model": model, "C": C}
        self.calls.append(record)

        class Fake:
            coef_ = np.zeros((1, 2))
            classes_ = np.array([1, 0]) if model == "binary" else np.array(
                ["OTHER", "ORDINARY", "SELF", "NONTERMINATION"])

            def fit(self, x, y):
                record.update(x=np.array(x, copy=True), y=np.array(y, copy=True))
                if index == 0 and owner.mode == "fail":
                    raise RuntimeError("intentional fake fit failure")
                if index == 0 and owner.mode == "warn":
                    warnings.warn("intentional nonconvergence", ConvergenceWarning)
                return self

            def predict_proba(self, x):
                p = 0. if owner.mode == "no_positive" else .7
                row = [p, 1-p] if model == "binary" else [(1-p)/3, (1-p)/3, p, (1-p)/3]
                out = np.tile(row, (len(x), 1))
                if index == 0 and owner.mode == "bad_probability":
                    out[0, 0] = np.nan
                return out

        return Fake()


class DriverTests(unittest.TestCase):
    def test_full_grid_mapping_and_serialization(self):
        factory = Factory()
        report = D.run_grouped_screen(*fixture(), fit_factory=factory)
        self.assertEqual(report["fit_count"], 40)
        self.assertEqual(len(factory.calls), 40)
        self.assertEqual(len(report["fold_results"]), 40)
        self.assertEqual(report["selection"]["n_candidates"], 152)
        self.assertIsNotNone(report["selection"]["selected"])
        for candidate in report["candidate_results"]:
            self.assertTrue(candidate["complete"])
            np.testing.assert_allclose(candidate["oof"]["p_self"], .7)
        json.dumps(report, allow_nan=False)

    def test_training_only_center_and_binary_targets(self):
        factory = Factory()
        x, labels, groups, folds = fixture()
        D.run_grouped_screen(x, labels, groups, folds, fit_factory=factory)
        for held in range(5):
            training = x[folds != held]
            expected = training - training.mean(axis=0)
            expected /= np.maximum(np.linalg.norm(expected, axis=1, keepdims=True), 1e-12)
            np.testing.assert_allclose(factory.calls[held]["x"], expected)
            np.testing.assert_array_equal(factory.calls[held]["y"], (labels[folds != held] == "SELF").astype(int))

    def test_failed_fold_never_pools_successful_siblings(self):
        for mode in ("fail", "warn", "bad_probability"):
            with self.subTest(mode=mode):
                report = D.run_grouped_screen(*fixture(), fit_factory=Factory(mode))
                self.assertEqual(report["fit_count"], 40)
                bad = report["candidate_results"][0]
                self.assertFalse(bad["complete"])
                self.assertEqual(bad["missing_folds"], [0])
                self.assertIsNone(bad["oof"])
                self.assertEqual(report["selection"]["n_candidates"], 133)
                self.assertEqual(sum(c["complete"] for c in report["candidate_results"]), 7)

    def test_no_positive_predictions_are_valid_false_negatives(self):
        report = D.run_grouped_screen(*fixture(), fit_factory=Factory("no_positive"))
        self.assertTrue(all(c["complete"] for c in report["candidate_results"]))
        self.assertIsNone(report["selection"]["selected"])
        for fold in report["fold_results"][:20]:
            self.assertTrue(fold["valid"])
            self.assertEqual(fold["metrics"]["fn"], 2)
            self.assertEqual(fold["metrics"]["f1"], 0.)
            self.assertIsNone(fold["metrics"]["precision"])

    def test_bad_input_rejected_before_factory(self):
        for mode in ("group", "label", "fold", "nan"):
            x, labels, groups, folds = fixture()
            if mode == "group": groups[4] = groups[0]
            if mode == "label": labels[0] = "INVALID"
            if mode == "fold": folds[:] = 0
            if mode == "nan": x[0, 0] = np.nan
            factory = Factory()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                D.run_grouped_screen(x, labels, groups, folds, fit_factory=factory)
            self.assertEqual(factory.calls, [])

    def test_binary_probability_and_duplicate_classes_rejected(self):
        class Bad:
            classes_ = np.array([1, 0])
            def predict_proba(self, x): return np.tile([1.2, -.2], (len(x), 1))
        with self.assertRaises(ValueError):
            D._fold_probabilities("binary", Bad(), np.zeros((2, 2)), 2)
        Bad.classes_ = np.array([1, 1])
        with self.assertRaises(ValueError):
            D._fold_probabilities("binary", Bad(), np.zeros((2, 2)), 2)


if __name__ == "__main__":
    unittest.main()
