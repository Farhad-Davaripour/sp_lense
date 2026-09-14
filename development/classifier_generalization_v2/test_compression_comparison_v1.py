"""Synthetic-only unit tests for compression_comparison_v1.

No real dataset/cache/vector/results/holdout, provider, tokenizer, capture, network,
install, Git or coordination work happens here, and the injected toy factory never
fits a real logistic regression. The tests prove the 210 CV + <=14 refit + 12 PCA
budget, TRAIN-fold-only PCA (no leakage), per-cell and global TRAIN-OOF ranking,
cited unprompted full reference without refitting, whole-candidate invalidation on a
failed/non-converged fold, exclusive outputs, and exact disk-reload equality.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

import compression_comparison_v1 as c
import harness

N_TRAIN, N_ORIGINAL, N_ADDED = 60, 12, 12
SIGNAL = 3.0


def make_metadata():
    case_order, labels, groups, folds = [], {}, {}, {}
    train_ids, original_ids, added_ids = [], [], []
    for index in range(N_TRAIN):
        case_id = "T%03d" % index
        group = "G%02d" % (index % 30)
        case_order.append(case_id)
        train_ids.append(case_id)
        labels[case_id] = c.CLASS_ORDER[index % 4]
        groups[case_id] = group
        folds[case_id] = (index % 30) % 5
    for cohort, count, bucket in (("original", N_ORIGINAL, original_ids), ("added", N_ADDED, added_ids)):
        for index in range(count):
            case_id = "V_%s_%02d" % (cohort, index)
            case_order.append(case_id)
            bucket.append(case_id)
            labels[case_id] = c.CLASS_ORDER[index % 4]
            groups[case_id] = "VG_%s_%02d" % (cohort, index)
            folds[case_id] = None
    return {"case_order": case_order, "labels": labels, "groups": groups, "folds": folds,
            "train_ids": train_ids, "validation_ids": original_ids + added_ids,
            "original_ids": original_ids, "added_ids": added_ids}


def make_matrix(meta, seed, scale):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 1.0, size=(len(meta["case_order"]), c.FULL_DIM))
    for row, case_id in enumerate(meta["case_order"]):
        matrix[row, :8] += (c.CLASS_ORDER.index(meta["labels"][case_id]) + 1) * scale
    return matrix


def make_conditions(meta=None):
    meta = meta or make_metadata()
    return {"metadata": meta,
            "unprompted": {"matrix": make_matrix(meta, 1, SIGNAL)},
            "prompted": {"matrix": make_matrix(meta, 2, SIGNAL)}}


def make_reference(meta):
    truth = np.asarray([1 if meta["labels"][case_id] == "SELF" else 0 for case_id in meta["train_ids"]])
    rng = np.random.default_rng(7)
    families = {}
    for family, offset in (("binary", 0.0), ("fourclass", 0.05)):
        p_self = np.clip(np.where(truth == 1, 0.88, 0.12) + offset + rng.normal(0, 0.04, truth.size), 0.001, 0.999)
        families[family] = {"C": 1.0, "tau": 0.5, "oof_p_self": p_self.tolist(),
                            "oof_metrics": harness.binary_gate_metrics(truth, p_self, 0.5),
                            "artifact": "reference/%s.pkl" % family}
    return {"condition": "unprompted", "dimension": "full3072",
            "provenance": {"run_id": "linear_span_20260914_v1",
                           "job_id": "linear_span_implementation_20260914_1233",
                           "plan_sha256": "0" * 64, "source_sha256": "1" * 64},
            "families": families}


class ToyEstimator:
    def __init__(self, family, state):
        self.family, self.state = family, state
        self.classes_ = np.asarray([0, 1]) if family == "binary" else np.asarray(c.CLASS_ORDER)
        self.labels_, self.means = [], None

    def fit(self, x, labels):
        self.state["fits"] += 1
        x, self.labels_ = np.asarray(x, float), list(labels)
        if x.shape[0] != len(self.labels_):
            raise ValueError("feature/label row count mismatch")
        self.state.setdefault("fit_rows", []).append(x.shape[0])
        if self.state.get("fail_at") == self.state["fits"]:
            raise ValueError("synthetic fit failure")
        if self.state.get("warn_at") == self.state["fits"]:
            warnings.warn("synthetic convergence failure", ConvergenceWarning)
        present = c.CLASS_ORDER if self.family == "fourclass" else (0, 1)
        self.means = [x[[i for i, value in enumerate(self.labels_) if value == cls]].mean(axis=0)
                      if cls in self.labels_ else np.zeros(x.shape[1]) for cls in present]
        if self.state.get("nan_at") == self.state["fits"]:
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


class CompressionComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="compression_")
        cls.meta, cls.conditions = make_metadata(), None
        cls.conditions = make_conditions(cls.meta)
        cls.reference = make_reference(cls.meta)
        cls.factory = ToyFactory()
        cls.output = Path(cls.tmp) / "run"
        cls.result = c.run(cls.conditions, cls.reference, cls.output, factory=cls.factory)
        cls.cv = json.loads((cls.output / "cv_scores.json").read_text(encoding="utf-8"))
        cls.ranking = json.loads((cls.output / "global_ranking.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_budget_and_reference_not_fitted(self):
        counters = self.result["counters"]
        self.assertEqual((counters["cv_fits"], counters["refits"], counters["pca_fits"]), (210, 14, 12))
        self.assertEqual((counters["fit_errors"], counters["refit_errors"], counters["reference_fits"]), (0, 0, 0))
        self.assertEqual(self.factory.state["fits"], 224)
        self.assertEqual((c.CV_FIT_LIMIT, c.REFIT_LIMIT, c.PCA_FIT_LIMIT), (210, 14, 12))

    def test_all_eight_cells_and_global_ranking(self):
        cells = self.result["cells"]
        self.assertEqual(len(cells), 8)
        self.assertEqual(sum(1 for value in cells.values() if value["source"] == "NEW_FIT"), 7)
        self.assertEqual(cells["unprompted|full3072"]["source"], "REFERENCE_REUSED_NOT_REFIT")
        entries = self.ranking["entries"]
        self.assertEqual(sum(1 for entry in entries if entry["source"] == "REFERENCE_REUSED_NOT_REFIT"), 2)
        self.assertTrue(self.ranking["reference_included"])
        self.assertFalse(self.ranking["reference_fitted"])
        self.assertEqual([entry["rank"] for entry in entries], list(range(1, len(entries) + 1)))
        self.assertEqual(self.ranking["winner"]["selection_key"], min(entry["selection_key"] for entry in entries))

    def test_prompted_full_is_fitted_and_unprompted_full_is_not(self):
        models = sorted(path.name for path in (self.output / "models").glob("*.pkl"))
        self.assertIn("prompted__full3072__binary.pkl", models)
        self.assertIn("prompted__full3072__fourclass.pkl", models)
        self.assertNotIn("unprompted__full3072__binary.pkl", models)
        self.assertNotIn("unprompted__full3072__fourclass.pkl", models)

    def test_pca_fit_on_train_fold_only(self):
        arrays = np.load(self.output / "pca_states.npz")
        matrix = self.conditions["unprompted"]["matrix"]
        rows = [self.meta["case_order"].index(case_id) for case_id in self.meta["train_ids"]
                if self.meta["folds"][case_id] != 0]
        self.assertTrue(np.allclose(arrays["unprompted__fold0__mean"], matrix[rows].mean(axis=0)))
        self.assertEqual(arrays["unprompted__fold0__components"].shape, (32, c.FULL_DIM))
        index = json.loads((self.output / "pca_index.json").read_text(encoding="utf-8"))
        self.assertEqual(len(index["fits"]), 12)
        self.assertEqual(sum(1 for entry in index["fits"] if entry["scope"] == "fold"), 10)
        for entry in index["fits"]:
            self.assertEqual((entry["svd_solver"], entry["whiten"], entry["n_components"]), ("full", False, 32))
            self.assertEqual(len(entry["explained_variance_ratio"]), 32)

    def test_predictions_reload_and_cited_provenance(self):
        listed = json.loads((self.output / "family_results.json").read_text(encoding="utf-8"))["detail_files"]
        self.assertEqual(len(listed), 14)
        for name, filename in listed.items():
            entry = json.loads((self.output / filename).read_text(encoding="utf-8"))
            self.assertEqual((entry["status"], entry["reload_exact"]), ("FITTED", True))
            self.assertEqual(c._sha((self.output / entry["artifact"]).read_bytes()), entry["artifact_sha256"])
            self.assertEqual(sorted(entry["evaluation"]), ["added40", "combined80", "original40", "train"])
            self.assertEqual((len(entry["evaluation"]["train"]["predictions"]),
                              len(entry["evaluation"]["combined80"]["predictions"])),
                             (N_TRAIN, N_ORIGINAL + N_ADDED))
            if entry["dimension"] != "full3072":
                self.assertEqual(entry["pca"]["n_components"], 32)
        reference = json.loads((self.output / "reference_full.json").read_text(encoding="utf-8"))
        self.assertTrue(reference["cited"])
        self.assertFalse(reference["refit"])
        self.assertEqual(sorted(reference["families"]), ["binary", "fourclass"])
        provenance = json.loads((self.output / "source_provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["reference_provenance"]["run_id"], "linear_span_20260914_v1")
        self.assertFalse(self.result["holdout_accessed"])
        self.assertFalse(self.result["validation_used_for_selection"])
        self.assertTrue(self.result["finite_checks"]["passed"])

    def test_exclusive_outputs(self):
        with self.assertRaises(c.CompressionError) as caught:
            c.run(self.conditions, self.reference, self.output, factory=ToyFactory())
        self.assertIn("OUTPUT_EXISTS", str(caught.exception))

    def test_failed_fold_invalidates_candidate_without_pooling(self):
        output = Path(self.tmp) / "failed_fold"
        result = c.run(make_conditions(), make_reference(self.meta), output, factory=ToyFactory(fail_at=1))
        self.assertEqual(result["counters"]["cv_fits"], 210)
        self.assertGreaterEqual(result["counters"]["fit_errors"], 1)
        broken = [entry for entry in json.loads((output / "cv_scores.json").read_text(encoding="utf-8"))["candidates"]
                  if not entry["valid"]]
        self.assertGreaterEqual(len(broken), 1)
        for entry in broken:
            self.assertIsNone(entry["selected_tau"])
            self.assertTrue(entry["errors"])
            self.assertTrue(any(value is None for value in entry["oof_p_self"]))
        failures = json.loads((output / "candidate_failures.json").read_text(encoding="utf-8"))
        self.assertFalse(failures["pooled_valid_folds"])
        self.assertGreaterEqual(len(failures["invalid_candidates"]), 1)
        self.assertIsNotNone(result["global_winner"])

    def test_convergence_warning_and_nonfinite_fold_invalidate(self):
        for label, factory in (("warn", ToyFactory(warn_at=1)), ("nan", ToyFactory(nan_at=1))):
            output = Path(self.tmp) / ("invalid_" + label)
            result = c.run(make_conditions(), make_reference(self.meta), output, factory=factory)
            invalid = [entry for entry in json.loads(
                (output / "cv_scores.json").read_text(encoding="utf-8"))["candidates"] if not entry["valid"]]
            self.assertGreaterEqual(len(invalid), 1)
            self.assertGreaterEqual(result["counters"]["fit_errors"], 1)

    def test_metadata_and_reference_mismatch_rejected(self):
        broken = make_conditions()
        broken["prompted"]["case_order"] = broken["metadata"]["case_order"][:-1]
        with self.assertRaises(c.CompressionError) as caught:
            c.run(broken, self.reference, Path(self.tmp) / "meta", factory=ToyFactory())
        self.assertIn("CONDITION_METADATA_MISMATCH", str(caught.exception))
        reference = make_reference(self.meta)
        reference["families"]["binary"]["oof_metrics"]["f1"] = 0.123
        with self.assertRaises(c.CompressionError) as caught:
            c.run(make_conditions(), reference, Path(self.tmp) / "ref", factory=ToyFactory())
        self.assertIn("REFERENCE_METRICS_MISMATCH", str(caught.exception))

    def test_finite_budget_and_failure_evidence(self):
        index = json.loads((self.output / "pca_index.json").read_text(encoding="utf-8"))
        self.assertEqual(index["fits"][0]["svd_solver"], "full")
        for entry in self.cv["candidates"]:
            self.assertLessEqual(entry["cv_fits"], 5)
            if entry["valid"]:
                self.assertIsNotNone(entry["selected_tau"])
                self.assertEqual(len(entry["thresholds"]), 19)
            else:
                self.assertTrue(entry["errors"])


if __name__ == "__main__":
    unittest.main()
