"""Synthetic-only unit tests for the identity three-question fit pipeline.

No real dataset, cache, vector, capture, tokenizer, model, network, install, Git
or coordination action happens. Every fixture is synthetic. Sklearn is used for
the required exact PCA32 fits and for exactly ONE tiny real
``grouped_driver._default_factory`` TOY classifier fit that proves the binary
targets are integer 0/1 and the estimator API shape is correct (the earlier fake
factory only missed the class-label encoding bug). All other classifier fits use
a deterministic toy factory.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import grouped_driver as driver
import identity_fit_v1 as fit
import span_classifier_driver_v1 as span

CYCLE = (0, 1, 2, 3, 4, 0, 1, 2, 3, 4)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def pin(path, root):
    resolved = Path(path).resolve()
    return {"path": resolved.relative_to(Path(root).resolve()).as_posix(),
            "sha256": sha(resolved.read_bytes()), "bytes": resolved.stat().st_size}


# --------------------------------------------------------------------------- #
# synthetic case data (240 TRAIN / 40 original / 40 added)
# --------------------------------------------------------------------------- #
def make_case_data():
    rng = np.random.default_rng(20260914)
    folds, groups, labels, train_ids, original_ids, added_ids = {}, {}, {}, [], [], []
    for cohort, count, bucket in (("original", 120, train_ids), ("added", 120, train_ids),
                                  ("original", 40, original_ids), ("added", 40, added_ids)):
        validation = bucket is not train_ids
        for index in range(count):
            case_id = "%s_%s_%03d" % ("V" if validation else "T", cohort, index)
            labels[case_id] = fit.CLASS_ORDER[index % 4]
            groups[case_id] = "%s_G%02d" % (cohort, index % 10)
            bucket.append(case_id)
            if not validation:
                folds[case_id] = CYCLE[index % 10]
    windows = {}
    for case_id in train_ids + original_ids + added_ids:
        base = rng.normal(0.0, 0.1, size=(3, 4, fit.span.WIDTH))
        if labels[case_id] == "SELF":
            base[:, -1, :8] += 2.0
        windows[case_id] = base.astype(np.float32)
    group_fold = {}
    for case_id in train_ids:
        group_fold[groups[case_id]] = folds[case_id]
    return {"windows": windows, "ab_windows": dict(windows), "ba_windows": dict(windows),
            "case_order": train_ids + original_ids + added_ids,
            "train_ids": train_ids, "validation_ids": original_ids + added_ids,
            "original_ids": original_ids, "added_ids": added_ids, "labels": labels,
            "groups": groups, "folds": folds, "group_fold": group_fold}


def make_old_control(case_data):
    evaluation = {}
    for split, key in fit.SPLITS:
        evaluation[split] = {"predictions": [
            {"case_id": case_id, "truth": case_data["labels"][case_id],
             "p_self": 0.9 if case_data["labels"][case_id] == "SELF" else 0.1,
             "predicted_self": case_data["labels"][case_id] == "SELF"}
            for case_id in case_data[key]]}
    return {
        "run_id": fit.OLD_CONTROL_RUN_ID, "condition": fit.OLD_CONTROL_CONDITION,
        "dimension": fit.OLD_DIMENSION, "family": fit.OLD_FAMILY, "C": fit.OLD_C,
        "oof_case_ids": list(case_data["train_ids"]),
        "oof_truth": [case_data["labels"][case_id] for case_id in case_data["train_ids"]],
        "oof_p_self": [0.9 if case_data["labels"][case_id] == "SELF" else 0.1
                       for case_id in case_data["train_ids"]],
        "saved_tau": 0.35, "saved_oof_metrics": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        "saved_evaluation": evaluation, "group_fold": case_data["group_fold"],
        "artifact": "models/prompted__pca32__binary.pkl", "artifact_sha256": "0" * 64,
        "cv_scores_sha256": "0" * 64, "model_results_sha256": "0" * 64,
    }


class ToyEstimator:
    """Deterministic binary estimator that refuses non-0/1 targets."""

    def __init__(self, family, state):
        self.family, self.state = family, state
        self.classes_ = np.asarray([0, 1])
        self.pos_mean = self.neg_mean = None

    def fit(self, x, labels):
        labels = list(labels)
        assert self.family == "binary", "toy only implements binary"
        assert all(type(value) is int and value in (0, 1) for value in labels), "binary targets must be 0/1"
        self.state["fits"] += 1
        if self.state.get("fail_at") == self.state["fits"]:
            raise ValueError("synthetic fit failure")
        x = np.asarray(x, float)
        pos = x[[index for index, value in enumerate(labels) if value == 1]]
        neg = x[[index for index, value in enumerate(labels) if value == 0]]
        self.pos_mean = pos.mean(axis=0) if len(pos) else np.zeros(x.shape[1])
        self.neg_mean = neg.mean(axis=0) if len(neg) else np.zeros(x.shape[1])
        return self

    def predict_proba(self, x):
        x = np.asarray(x, float)
        logits = np.stack([-((x - self.neg_mean) ** 2).sum(axis=1),
                           -((x - self.pos_mean) ** 2).sum(axis=1)], axis=1)
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        return weights / weights.sum(axis=1, keepdims=True)


class ToyFactory:
    def __init__(self, fail_at=None):
        self.state = {"fits": 0, "fail_at": fail_at}

    def __call__(self, family, C):
        return ToyEstimator(family, self.state)


# --------------------------------------------------------------------------- #
# mini authenticated-shape fixtures for the pure verifiers
# --------------------------------------------------------------------------- #
def mini_index(records, case_order=("C1",)):
    views = len(case_order) * len(span.ORDERS)
    return {"schema": fit.IDENTITY_INDEX_SCHEMA, "condition": fit.IDENTITY_CONDITION,
            "query": fit.tokens.IDENTITY_QUERY, "query_sha256": fit.tokens.QUERY_SHA256,
            "lock_sha256": "a" * 64, "run_id": "identity_capture_20260914_v1",
            "binary_file": "windows.f32", "binary_schema": fit.IDENTITY_WINDOWS_SCHEMA,
            "blocks": list(span.BLOCKS), "orders": list(span.ORDERS), "dtype": "float32",
            "byte_order": "little", "float_bytes": 4, "width": span.WIDTH, "window_max": span.WINDOW_MAX,
            "case_order": list(case_order), "case_count": len(case_order), "view_count": views,
            "forward_count": views, "raw_bytes": 4 * len(records), "records": records}


def mini_records(prefix=1):
    records = []
    for order in span.ORDERS:
        for block in span.BLOCKS:
            records.append({"case_id": "C1", "condition": fit.IDENTITY_CONDITION,
                            "query_sha256": fit.tokens.QUERY_SHA256, "order": order, "block": block,
                            "prefix_length": prefix, "readout_index": prefix - 1,
                            "last_shared_token_id": 32, "prefix_sha256": "b" * 64,
                            "input_ids_sha256": "%s_%s" % (order, "c" * 60)})
    return records


def mini_sanity():
    rows = [{"case_id": "C1", "split": "TRAIN", "order": order, "tokens": 1, "prefix_tokens": 1,
             "last_shared_token_id": 32, "input_sha256": "%s_%s" % (order, "c" * 60)}
            for order in span.ORDERS]
    return {"status": "PASS", "condition": fit.IDENTITY_CONDITION,
            "query_sha256": fit.tokens.QUERY_SHA256, "exact_decode_reencode": True,
            "query_in_shared_prefix": True, "last_shared_token_ids": [32], "max_full_tokens": 1,
            "counts": {"cases": 1, "views": 2, "tokenizer_loads": 1, "model_loads": 0, "forwards": 0,
                       "fits": 0},
            "adapter_sha256": "d" * 64, "reference_locks": [], "fits_existing_320_cap": True, "rows": rows}


MINI_CAPS = {"forwards": 2, "tokens_per_view": 320, "raw_bytes": 1 << 20}


class IdentityFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="identity_fit_"))
        cls.case_data = make_case_data()
        cls.old_control = make_old_control(cls.case_data)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # -- threshold rule ---------------------------------------------------- #
    def test_select_tau_f1_first(self):
        truth = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        oof = [0.9, 0.8, 0.7, 0.65, 0.3, 0.2, 0.15, 0.1, 0.05, 0.0]
        result = fit.select_tau(oof, truth)
        self.assertIsNotNone(result["selected_tau"])
        best = max((entry for entry in result["thresholds"] if entry["eligible"]),
                   key=lambda entry: (entry["f1"], min(entry["precision"], entry["recall"]),
                                      -abs(entry["tau"] - 0.5), -entry["tau"]))
        self.assertEqual(result["selected_tau"], best["tau"])
        self.assertEqual(result["selected_tau"], 0.5)
        self.assertEqual(result["oof_metrics"]["f1"], 1.0)

    def test_empty_positive_predictions_p_null_f1_zero(self):
        result = fit.select_tau([0.0] * 10, [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertIsNone(result["selected_tau"])
        self.assertEqual(result["error"], "NO_ELIGIBLE_THRESHOLD")
        for entry in result["thresholds"]:
            self.assertIsNone(entry["precision"])
            self.assertEqual(entry["f1"], 0.0)
            self.assertFalse(entry["eligible"])

    # -- plan / pins ------------------------------------------------------- #
    def test_unresolved_source_pins_fail_before_any_fit(self):
        root = self.tmp / "unresolved"
        plan = {"source_files": {"development/classifier_generalization_v2/identity_fit_v1.py": None},
                "runtime_packages": {"numpy": "2.5.3"}}
        with self.assertRaises(fit.IdentityFitError) as caught:
            fit.verify_sources_and_runtime(plan, root)
        self.assertIn("UNRESOLVED_PIN", str(caught.exception))

    def test_source_pin_rejection(self):
        root = self.tmp / "sources"
        (root / "development" / "classifier_generalization_v2").mkdir(parents=True)
        target = root / "development" / "classifier_generalization_v2" / "mod.py"
        target.write_text("x = 1\n", encoding="utf-8")
        good = {target.relative_to(root).as_posix(): sha(target.read_bytes())}
        fit.verify_sources_and_runtime({"source_files": good, "runtime_packages": {}}, root)
        bad = {target.relative_to(root).as_posix(): "0" * 64}
        with self.assertRaises(fit.IdentityFitError) as caught:
            fit.verify_sources_and_runtime({"source_files": bad, "runtime_packages": {}}, root)
        self.assertIn("SOURCE_MISMATCH", str(caught.exception))
        lock = {"source_files": bad}
        with self.assertRaises(fit.IdentityFitError) as caught:
            fit.verify_source_pins(lock, root)
        self.assertIn("IDENTITY_SOURCE_MISMATCH", str(caught.exception))

    def test_delivered_plan_is_well_formed(self):
        plan_path = Path(fit.__file__).resolve().parent / "IDENTITY_FIT_PLAN_V1.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertEqual(plan["schema"], fit.PLAN_SCHEMA)
        self.assertEqual(plan["job_id"], fit.JOB_ID)
        self.assertEqual(plan["classifier_fits"], 6)
        self.assertEqual(plan["thresholds"], list(fit.TAUS))
        self.assertEqual(plan["folds"], list(fit.FOLDS))
        self.assertFalse(plan["scientific_execution_authorized"])
        # The delivered plan is the prepare-resolved artifact reviewed for release.
        self.assertEqual(plan["pin_status"], "RESOLVED")

    # -- wrong query / cache mix / pair alignment -------------------------- #
    def test_wrong_query_and_cache_mix_rejected(self):
        index = mini_index(mini_records())
        windows = b"\x00\x00\x00\x00" * len(index["records"])
        index["raw_bytes"] = len(windows)
        fit.verify_index_binding(index, MINI_CAPS, "a" * 64, "identity_capture_20260914_v1", windows)
        wrong_query = dict(index, query="wrong", query_sha256="0" * 64)
        with self.assertRaises(fit.IdentityFitError) as caught:
            fit.verify_index_binding(wrong_query, MINI_CAPS, "a" * 64, "identity_capture_20260914_v1", windows)
        self.assertIn("IDENTITY_INDEX_QUERY", str(caught.exception))
        mixed = dict(index, condition=fit.OLD_CONTROL_CONDITION)
        with self.assertRaises(fit.IdentityFitError) as caught:
            fit.verify_index_binding(mixed, MINI_CAPS, "a" * 64, "identity_capture_20260914_v1", windows)
        self.assertIn("IDENTITY_INDEX_CONDITION", str(caught.exception))
        wrong_lock = dict(index, lock_sha256="9" * 64)
        with self.assertRaises(fit.IdentityFitError) as caught:
            fit.verify_index_binding(wrong_lock, MINI_CAPS, "a" * 64, "identity_capture_20260914_v1", windows)
        self.assertIn("IDENTITY_INDEX_LOCK", str(caught.exception))

    def test_pair_prefix_misalignment_rejected(self):
        index = mini_index(mini_records())
        sanity = mini_sanity()
        caps = dict(MINI_CAPS)
        fit.verify_last_token_provenance(index, sanity, caps)
        broken = json.loads(json.dumps(index))
        broken["records"][0]["prefix_length"] = 2
        broken["records"][0]["readout_index"] = 1
        with self.assertRaises(fit.IdentityFitError) as caught:
            fit.verify_last_token_provenance(broken, sanity, caps)
        self.assertIn("IDENTITY_BLOCK_PREFIX_MISMATCH", str(caught.exception))

    # -- old control reuse ------------------------------------------------- #
    def test_old_control_matches_c10_without_pickle_load(self):
        root = self.tmp / "control"
        run_dir = root / fit.STUDY_DIR / "runs" / fit.OLD_CONTROL_RUN_ID
        run_dir.mkdir(parents=True)
        candidate = {"condition": fit.OLD_CONTROL_CONDITION, "dimension": fit.OLD_DIMENSION,
                     "family": fit.OLD_FAMILY, "C": fit.OLD_C, "valid": True,
                     "selected_tau": 0.35, "oof_metrics": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                     "thresholds": [], "oof_p_self": [0.9 if t == "SELF" else 0.1
                                                      for t in self.old_control["oof_truth"]]}
        cv = {"oof_case_ids": self.old_control["oof_case_ids"], "oof_truth": self.old_control["oof_truth"],
              "candidates": [candidate], "folds": [
                  {"condition": fit.OLD_CONTROL_CONDITION, "fold": fold,
                   "train_groups": [g for g, f in self.case_data["group_fold"].items() if f != fold],
                   "test_groups": [g for g, f in self.case_data["group_fold"].items() if f == fold]}
                  for fold in fit.FOLDS]}
        prediction = {"condition": fit.OLD_CONTROL_CONDITION, "dimension": fit.OLD_DIMENSION,
                      "family": fit.OLD_FAMILY, "C": fit.OLD_C, "status": "FITTED", "reload_exact": True,
                      "artifact": "models/prompted__pca32__binary.pkl",
                      "artifact_sha256": "0" * 64,
                      "evaluation": {split: {"predictions": self.old_control["saved_evaluation"][split]["predictions"]}
                                     for split, _ in fit.SPLITS}}
        write_json(run_dir / "cv_scores.json", cv)
        write_json(run_dir / "model_results_prompted__pca32__binary.json", prediction)
        plan = {"old_control": {"run_id": fit.OLD_CONTROL_RUN_ID, "condition": fit.OLD_CONTROL_CONDITION,
                                "dimension": fit.OLD_DIMENSION, "family": fit.OLD_FAMILY, "C": fit.OLD_C,
                                "cv_scores": pin(run_dir / "cv_scores.json", root),
                                "model_results": pin(run_dir / "model_results_prompted__pca32__binary.json", root)}}
        loaded = fit.load_old_control(plan, root)
        self.assertEqual(loaded["C"], fit.OLD_C)
        self.assertEqual(loaded["saved_tau"], 0.35)
        self.assertFalse(loaded["pickle_loaded"])
        self.assertEqual((loaded["classifier_fits"], loaded["pca_fits"], loaded["refits"]), (0, 0, 0))
        self.assertEqual(loaded["group_fold"], self.case_data["group_fold"])
        self.assertFalse((run_dir / "models" / "prompted__pca32__binary.pkl").exists())
        mis = dict(plan)
        mis["old_control"] = dict(plan["old_control"], C=1.0)
        with self.assertRaises(fit.IdentityFitError):
            fit.load_old_control(mis, root)

    def test_load_identity_case_data_reuses_span_loader(self):
        with mock.patch.object(fit.span, "load_case_data", return_value="sentinel") as called:
            auth = {"index": {"i": 1}, "windows_raw": b"w"}
            plan = {"manifests": {"blueprint": {"path": "b", "sha256": "0" * 64}}}
            self.assertEqual(fit.load_identity_case_data(auth, plan, "root"), "sentinel")
        self.assertEqual(called.call_count, 1)
        source, passed_plan, root = called.call_args.args
        self.assertEqual(source, {"index": {"i": 1}, "windows_raw": b"w"})
        self.assertEqual(passed_plan, {"manifests": plan["manifests"]})
        self.assertEqual(root, "root")

    # -- alignment --------------------------------------------------------- #
    def test_control_alignment_rejections(self):
        case = fit._validate_case_data(self.case_data)
        fit.verify_control_alignment(case, self.old_control)
        wrong_group = dict(self.old_control, group_fold={"X": 0})
        with self.assertRaises(fit.IdentityFitError) as caught:
            fit.verify_control_alignment(case, wrong_group)
        self.assertIn("CONTROL_GROUP_FOLD", str(caught.exception))
        wrong_ids = dict(self.old_control, oof_case_ids=list(reversed(self.old_control["oof_case_ids"])))
        with self.assertRaises(fit.IdentityFitError) as caught:
            fit.verify_control_alignment(case, wrong_ids)
        self.assertIn("CONTROL_OOF_IDS", str(caught.exception))
        leaky = dict(self.case_data)
        leaky["folds"] = dict(self.case_data["folds"])
        first = leaky["train_ids"][0]
        leaky["folds"][first] = (leaky["folds"][first] + 1) % 5
        with self.assertRaises(fit.IdentityFitError) as caught:
            fit._validate_case_data(leaky)
        self.assertIn("GROUP_MULTI_FOLD", str(caught.exception))

    # -- end to end -------------------------------------------------------- #
    def test_compare_end_to_end_budget_splits_and_exclusive_output(self):
        factory = ToyFactory()
        output = self.tmp / "run_ok"
        result = fit.compare(self.case_data, self.old_control, output, factory=factory)
        counters = result["counters"]
        self.assertEqual((counters["cv_fits"], counters["refits"], counters["pca_fits"]), (5, 1, 6))
        self.assertEqual((counters["old_cv_fits"], counters["old_pca_fits"], counters["old_refits"]), (0, 0, 0))
        self.assertEqual(factory.state["fits"], 6)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertFalse(result["holdout_accessed"])
        self.assertFalse(result["validation_used_for_selection"])
        for name in ("cv_scores.json", "comparison.json", "old_control.json", "identity_results.json",
                     "development_results.json", "pca_index.json", "feature_definition.json"):
            self.assertTrue((output / name).is_file(), name)
        comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
        for cell in comparison["cells"].values():
            self.assertEqual(set(cell["evaluation"]), {"train", "original40", "added40", "combined80"})
            self.assertIn("negative_class_false_positives", cell["evaluation"]["combined80"])
        identity = comparison["cells"][fit.IDENTITY_CONDITION + "|pca32|binary"]
        self.assertEqual(set(identity["evaluation"]["combined80"]["answer_order_consistency"]),
                         {"gate_agreements", "total", "max_probability_difference"})
        self.assertEqual(identity["retained_variance"]["n_components"], 32)
        self.assertGreater(identity["retained_variance"]["cumulative_explained_variance"], 0.0)
        control = comparison["cells"][fit.OLD_CONTROL_CONDITION + "|pca32|binary"]
        self.assertEqual(control["source"], "SAVED_REUSE")
        self.assertTrue(comparison["matched_control_baseline"]["pickle_loaded"] is False)
        with self.assertRaises(fit.IdentityFitError) as caught:
            fit.compare(self.case_data, self.old_control, output, factory=ToyFactory())
        self.assertIn("OUTPUT_EXISTS", str(caught.exception))

    def test_failure_preservation(self):
        output = self.tmp / "run_fail"
        factory = ToyFactory(fail_at=2)
        with self.assertRaises(Exception):
            fit.compare(self.case_data, self.old_control, output, factory=factory)
        self.assertTrue((output / "failure.json").is_file())
        failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
        self.assertEqual(failure["status"], "failed")
        self.assertFalse(failure["holdout_accessed"])

    # -- the one real sklearn TOY fit -------------------------------------- #
    def test_real_sklearn_default_factory_binary_classes(self):
        x = np.asarray([[0.0, 0.0], [0.2, 0.1], [0.1, 0.3], [0.3, 0.2],
                        [2.0, 2.0], [2.2, 2.1], [2.1, 2.3], [2.3, 2.2]])
        y = [0, 0, 0, 0, 1, 1, 1, 1]
        estimator = driver._default_factory("binary", 10.0)
        estimator.fit(x, y)
        self.assertEqual(sorted(int(value) for value in estimator.classes_), [0, 1])
        probabilities = np.asarray(estimator.predict_proba(x), float)
        self.assertEqual(probabilities.shape, (8, 2))
        self.assertTrue(bool(np.allclose(probabilities.sum(axis=1), 1.0)))
        self.assertTrue(bool(np.all(np.asarray(estimator.predict(x)) == np.asarray(y))))


if __name__ == "__main__":
    unittest.main()
