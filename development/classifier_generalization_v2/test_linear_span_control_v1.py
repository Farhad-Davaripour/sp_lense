"""Synthetic-only unit tests for linear_span_control_v1.

No real dataset/cache/vector/results/private HOLDOUT file is read, no native
provider/Qwen/tokenizer/capture/model fit/network/install/Git/coordination action
happens, and the toy factory below never fits a real classifier. The tests prove
the 30+2 fit budget, five grouped folds without group leakage, independent
per-block normalization with zero->zero, TRAIN-OOF ranking, whole-candidate
invalidation on a failed or non-finite fold, exclusive outputs and an exact
serialize/reload check.
"""
from __future__ import annotations

import json
import warnings
from sklearn.exceptions import ConvergenceWarning
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import linear_span_control_v1 as c

CYCLE = (0, 1, 2, 3, 4, 0, 1, 2, 3, 4)


def make_case_data():
    """Synthetic authenticated-shaped case_data: 240 train, 40+40 validation."""
    rng = np.random.default_rng(20260914)
    folds, groups, labels, train_ids, original_ids, added_ids = {}, {}, {}, [], [], []
    for cohort, count, bucket in (("original", 120, train_ids), ("added", 120, train_ids),
                                  ("original", 40, original_ids), ("added", 40, added_ids)):
        validation = bucket is not train_ids
        for index in range(count):
            case_id = "%s_%s_%03d" % ("V" if validation else "T", cohort, index)
            labels[case_id] = c.CLASS_ORDER[index % 4]
            bucket.append(case_id)
            if not validation:
                groups[case_id] = "%s_G%02d" % (cohort, index % 10)
                folds[case_id] = CYCLE[index % 10]
    windows = {}
    for case_id in train_ids + original_ids + added_ids:
        base = rng.normal(0.0, 1.0, size=(3, 4, c.WIDTH))
        base[:, -1, :12] += (c.CLASS_ORDER.index(labels[case_id]) + 1) * (np.arange(3) + 1.0)[:, None]
        windows[case_id] = base.astype(np.float32)
    windows[original_ids[0]][1] = 0.0  # zero block must stay zero after normalization
    return {"windows": windows, "train_ids": train_ids, "validation_ids": original_ids + added_ids,
            "original_ids": original_ids, "added_ids": added_ids, "labels": labels,
            "groups": groups, "folds": folds, "ab_windows": windows, "ba_windows": windows}


class ToyEstimator:
    def __init__(self, family, state):
        self.family, self.state = family, state
        self.classes_ = np.asarray([0, 1]) if family == "binary" else np.asarray(c.CLASS_ORDER)
        self.labels_, self.means = [], None

    def fit(self, x, labels):
        self.state["fits"] += 1
        x, self.labels_ = np.asarray(x, float), list(labels)
        if x.shape[0] != len(self.labels_):
            raise ValueError('Feature/label row count mismatch')
        self.state.setdefault('fit_rows', []).append(x.shape[0])
        if self.state.get('fail_at') == self.state['fits']:
            raise ValueError('synthetic fit failure')
        if self.state.get('warn_at') == self.state['fits']:
            warnings.warn('synthetic convergence failure', ConvergenceWarning)
        present = c.CLASS_ORDER if self.family == "fourclass" else (0, 1)
        self.means = [x[[i for i, value in enumerate(self.labels_) if value == cls]].mean(axis=0)
                      if cls in self.labels_ else np.zeros(x.shape[1]) for cls in present]
        if self.state.get("nan_at") == self.state["fits"]:  # simulate a non-finite fold
            self.means = [np.full(x.shape[1], np.nan) for _ in self.means]
        return self

    def predict_proba(self, x):
        x = np.asarray(x, float)
        logits = np.stack([-((x - mean) ** 2).sum(axis=1) for mean in self.means], axis=1)
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = weights / weights.sum(axis=1, keepdims=True)
        return (np.column_stack([1 - probabilities[:, 0], probabilities[:, 0]])
                if self.family == "binary" else probabilities)


class ToyFactory:
    def __init__(self, nan_at=None, fail_at=None, warn_at=None):
        self.state = {"fits": 0, "nan_at": nan_at, "fail_at": fail_at, "warn_at": warn_at}

    def __call__(self, family, C):
        return ToyEstimator(family, self.state)


class LinearSpanControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="linear_span_")
        cls.case_data, cls.factory = make_case_data(), ToyFactory()
        cls.output = Path(cls.tmp) / "run"
        cls.result = c.run(cls.case_data, cls.output, factory=cls.factory)
        cls.cv = json.loads((cls.output / "cv_scores.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_budget_and_zero_real_fit(self):
        counters = self.result["counters"]
        self.assertEqual((counters["cv_fits"], counters["refits"], counters["fit_errors"]), (30, 2, 0))
        self.assertEqual(self.factory.state["fits"], 32)
        self.assertEqual(self.factory.state['fit_rows'][-2:], [240, 240])
        self.assertEqual((c.CV_FIT_LIMIT, c.REFIT_LIMIT), (30, 2))

    def test_fold_separation(self):
        assignment, seen = {}, set()
        for record in self.cv["folds"][:5]:
            self.assertFalse(set(record["train_groups"]) & set(record["test_groups"]))
            for group in record["test_groups"]:
                self.assertNotIn(group, seen)
                assignment[group] = record["fold"]
            seen.update(record["test_groups"])
        self.assertEqual([record["fold"] for record in self.cv["folds"][:5]], [0, 1, 2, 3, 4])
        self.assertEqual(len(assignment), 20)
        self.assertEqual({fold: list(assignment.values()).count(fold) for fold in range(5)},
                         {fold: 4 for fold in range(5)})

    def test_failed_refit_preserves_counts_and_does_not_retry(self):
        output = Path(self.tmp) / 'refit_failure'
        with self.assertRaisesRegex(c.LinearSpanError, 'REFIT_FAILED'):
            c.run(make_case_data(), output, factory=ToyFactory(fail_at=31))
        failure = json.loads((output / 'failure.json').read_text())
        self.assertEqual(failure['counters']['cv_fits'], 30)
        self.assertEqual(failure['counters']['refits'], 1)
        self.assertEqual(failure['counters']['refit_errors'], 1)
        self.assertFalse((output / 'development_results.json').exists())

    def test_convergence_warning_invalidates_candidate(self):
        output = Path(self.tmp) / 'warning_failure'
        result = c.run(make_case_data(), output, factory=ToyFactory(warn_at=1))
        self.assertEqual(result['counters']['fit_errors'], 1)
        cv = json.loads((output / 'cv_scores.json').read_text())
        first = next(e for e in cv['candidates'] if e['family']=='binary' and e['C']==0.1)
        self.assertFalse(first['valid'])

    def test_feature_dimensions_normalization_and_zero(self):
        self.assertEqual((c.FEATURE_DIM, c.BLOCKS), (3072, (6, 10, 18)))
        window = self.case_data["windows"]["T_original_000"]
        features = c.features_for_case(window)
        self.assertEqual(features.shape, (3072,))
        for block in range(3):
            self.assertAlmostEqual(float(np.linalg.norm(features[block * c.WIDTH:(block + 1) * c.WIDTH])), 1.0, 12)
        zero = c.features_for_case(self.case_data["windows"]["V_original_000"])
        self.assertTrue(bool((zero[c.WIDTH:2 * c.WIDTH] == 0.0).all()))
        first = c.features_for_case(np.stack([window[0] * 1000.0, window[1], window[2]]))
        self.assertTrue(bool(np.array_equal(features[c.WIDTH:], first[c.WIDTH:])))
        self.assertTrue(bool(np.array_equal(c.build_features(self.case_data)[1][0], features)))

    def test_ranking_and_train_selection(self):
        selected = self.result["train_selected"]
        self.assertEqual(selected["selected_using"], "TRAIN_GROUPED_OOF_ONLY")
        self.assertEqual((selected["family"], selected["C"]),
                         (self.cv["train_ranking"][0]["family"], self.cv["train_ranking"][0]["C"]))
        self.assertEqual(self.cv["train_ranking"][0]["selection_key"],
                         min(entry["selection_key"] for entry in self.cv["train_ranking"]))
        for entry in self.cv["candidates"]:
            self.assertTrue(entry["valid"])
            self.assertEqual([value["tau"] for value in entry["thresholds"]],
                             [round(0.05 * i, 2) for i in range(1, 20)])
        self.assertTrue(self.result["family_results"][selected["family"]]["train_selected"])
        self.assertFalse(self.result["validation_used_for_selection"])
        self.assertEqual(self.cv["train_selected"], selected)

    def test_failed_fold_invalidates_candidate_without_pooling(self):
        output = Path(self.tmp) / "nan_run"
        result = c.run(make_case_data(), output, factory=ToyFactory(nan_at=26))
        failed = [entry for entry in json.loads(
            (output / "cv_scores.json").read_text(encoding="utf-8"))["candidates"] if not entry["valid"]]
        self.assertGreaterEqual(len(failed), 1)
        for entry in failed:
            self.assertIsNone(entry["selected_tau"])
            self.assertTrue(entry["errors"])
        self.assertGreaterEqual(result["counters"]["fit_errors"], 1)
        self.assertEqual(result["counters"]["cv_fits"], 30)
        self.assertIsNotNone(result["train_selected"])

    def test_exclusive_outputs_reload_and_feature_pin(self):
        with self.assertRaises(c.LinearSpanError) as caught:
            c.run(self.case_data, self.output, factory=ToyFactory())
        self.assertIn("OUTPUT_EXISTS", str(caught.exception))
        definition = json.loads((self.output / "feature_definition.json").read_text(encoding="utf-8"))
        self.assertEqual((definition["learned_transform"], definition["pca"], definition["blocks"]),
                         (False, False, [6, 10, 18]))
        listed = json.loads((self.output / "family_results.json").read_text(encoding="utf-8"))["family_detail_files"]
        self.assertEqual(sorted(listed), sorted(self.result["family_results"]))
        for family in self.result["family_results"]:
            entry = json.loads((self.output / listed[family]).read_text(encoding="utf-8"))
            self.assertEqual((entry["status"], entry["reload_exact"]), ("FITTED", True))
            self.assertEqual(c._sha((self.output / entry["artifact"]).read_bytes()), entry["artifact_sha256"])
            splits = json.loads((self.output / ("predictions_linear_%s.json" % family))
                                .read_text(encoding="utf-8"))["splits"]
            self.assertEqual(sorted(splits), ["added40", "combined80", "original40", "train"])
            self.assertEqual((len(splits["train"]["predictions"]), len(splits["combined80"]["predictions"])),
                             (240, 80))
        self.assertFalse(self.result["holdout_accessed"])


if __name__ == "__main__":
    unittest.main()
