"""Synthetic-only repair tests for ``compression_comparison_v2``.

Scope: prove the V2 binary target-encoding repair at *both* fit sites, prove the
real sklearn label contract with the existing default factory, prove the refit
budget is a maximum for partially invalid runs, and prove the V2 versioned
defaults/pins. No real dataset, cache, vector, result, pickle, holdout, model,
tokenizer, provider, capture, network, install, Git or coordination work happens.

Exactly two real sklearn estimator fits happen in this module (one binary, one
four-class) on tiny artificial arrays from ``grouped_driver._default_factory``.
Every other estimator is the injected recording toy; PCA/SVD stays synthetic.
The prior synthetic suite is reused by importing ``test_compression_comparison_v1``
(never modified) for its fixture builders and for one re-run of that suite.
"""
from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import compression_comparison_v2 as core
import compression_integration_v2 as ci
import grouped_driver as driver
import harness
import test_compression_comparison_v1 as prior

PLAN_FOR_VERIFY = {"output_bytes": ci.OUTPUT_BYTES}


class RecordingEstimator(prior.ToyEstimator):
    """Toy estimator that records every target vector passed to ``fit``."""

    def fit(self, x, labels):
        labels = list(labels)
        self.state.setdefault("calls", []).append(
            {"family": self.family, "labels": labels, "types": [type(value).__name__ for value in labels]})
        return super().fit(x, labels)


class RecordingFactory:
    """Injected toy factory; never constructs a real estimator."""

    def __init__(self):
        self.state = {"fits": 0, "calls": []}

    def __call__(self, family, C):
        return RecordingEstimator(family, self.state)


class BinaryInvalidFactory(RecordingFactory):
    """Toy factory that refuses only the binary family, leaving four-class valid."""

    def __call__(self, family, C):
        if family == "binary":
            raise ValueError("synthetic binary family unavailable")
        return RecordingEstimator(family, self.state)


class CompressionBinaryRepairV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="compression_binary_repair_"))
        cls.meta = prior.make_metadata()
        cls.conditions = prior.make_conditions(cls.meta)
        cls.reference = prior.make_reference(cls.meta)
        cls.factory = RecordingFactory()
        cls.output = cls.tmp / "recording_run"
        cls.result = core.run(cls.conditions, cls.reference, cls.output, factory=cls.factory)
        cls.calls = list(cls.factory.state["calls"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # 1. both CV and refit paths carry the right targets
    # ------------------------------------------------------------------ #
    def test_cv_and_refit_receive_encoded_binary_and_string_fourclass(self):
        cv_calls = self.calls[:core.CV_FIT_LIMIT]
        refit_calls = self.calls[core.CV_FIT_LIMIT:]
        self.assertEqual(len(cv_calls), core.CV_FIT_LIMIT)
        self.assertEqual(len(refit_calls), self.result["counters"]["refits"])
        self.assertEqual(self.result["counters"]["refits"], core.REFIT_LIMIT)
        self.assertEqual(self.result["counters"]["fit_errors"], 0)
        self.assertEqual(self.result["counters"]["refit_errors"], 0)

        # The attempted-count budget is unchanged; every candidate stayed valid.
        self.assertEqual(sum(1 for call in cv_calls if call["family"] == "binary"), 105)
        self.assertEqual(sum(1 for call in cv_calls if call["family"] == "fourclass"), 105)
        # Both families are exercised on the full-TRAIN refit path too.
        self.assertEqual(sum(1 for call in refit_calls if call["family"] == "binary"), 7)
        self.assertEqual(sum(1 for call in refit_calls if call["family"] == "fourclass"), 7)

        for track in (cv_calls, refit_calls):
            for call in track:
                if call["family"] == "binary":
                    self.assertEqual(set(call["labels"]), {0, 1})
                    self.assertTrue(all(name == "int" for name in call["types"]))
                else:
                    self.assertEqual(set(call["labels"]), set(core.CLASS_ORDER))
                    self.assertTrue(all(name == "str" for name in call["types"]))

    # ------------------------------------------------------------------ #
    # 2. the real sklearn label contract (exactly two real fits)
    # ------------------------------------------------------------------ #
    def test_real_default_factory_binary_and_fourclass_label_contract(self):
        rng = np.random.default_rng(20260914)
        features = rng.normal(size=(12, 5))

        # Real fit 1 of 2: binary family on integer 0/1 SELF-vs-rest targets.
        binary = driver._default_factory("binary", 1.0)
        binary.fit(features, [0, 1] * 6)
        p_self, probs = driver._fold_probabilities(harness.MODEL_BINARY, binary, features, 12)
        self.assertIsNone(probs)
        self.assertEqual(p_self.shape, (12,))
        self.assertEqual(list(np.asarray(binary.classes_).ravel()), [0, 1])

        # Real fit 2 of 2: four-class family keeps the canonical string targets.
        fourclass = driver._default_factory("fourclass", 1.0)
        fourclass.fit(features, [core.CLASS_ORDER[index % 4] for index in range(12)])
        four_self, four_probs = driver._fold_probabilities(harness.MODEL_MULTICLASS, fourclass, features, 12)
        self.assertEqual(four_probs.shape, (12, 4))
        self.assertEqual(four_self.shape, (12,))
        # sklearn sorts string classes; the canonical order is restored by the driver.
        self.assertEqual(set(np.asarray(fourclass.classes_).ravel().tolist()), set(core.CLASS_ORDER))

        # Expose V1's actual defect without a third fit: V1 fitted four strings and
        # then extracted the binary probability path, which rejects those classes.
        with self.assertRaises(ValueError) as caught:
            driver._fold_probabilities(harness.MODEL_BINARY, fourclass, features, 12)
        self.assertIn("binary estimator classes must be exactly 0 and 1", str(caught.exception))

    # ------------------------------------------------------------------ #
    # 3. refit budget is a maximum, not a forced 14
    # ------------------------------------------------------------------ #
    def test_partial_invalid_run_is_honest_and_refit_capped(self):
        output = self.tmp / "partial_invalid"
        factory = BinaryInvalidFactory()
        result = core.run(self.conditions, self.reference, output, factory=factory)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["counters"]["cv_fits"], core.CV_FIT_LIMIT)
        self.assertEqual(result["counters"]["refits"], 7)  # four-class winners only, never padded
        self.assertLess(result["counters"]["refits"], core.REFIT_LIMIT)
        self.assertEqual(result["counters"]["fit_errors"], 105)
        self.assertEqual(result["counters"]["refit_errors"], 0)

        failures = json.loads((output / "candidate_failures.json").read_text(encoding="utf-8"))
        self.assertFalse(failures["pooled_valid_folds"])
        self.assertEqual(len(failures["invalid_candidates"]), 21)
        self.assertTrue(all(entry["family"] == "binary" for entry in failures["invalid_candidates"]))

        # The integration honestly accepts a capped partial result and still rejects overflow.
        ci._verify_result(result, PLAN_FOR_VERIFY, output)
        overflow = {"status": result["status"], "holdout_accessed": False,
                    "validation_used_for_selection": False,
                    "counters": dict(result["counters"], refits=core.REFIT_LIMIT + 1)}
        with self.assertRaises(ci.CompressionIntegrationError) as caught:
            ci._verify_result(overflow, PLAN_FOR_VERIFY, output)
        self.assertIn("RESULT_REFIT_BUDGET", str(caught.exception))

    # ------------------------------------------------------------------ #
    # 4. V2 defaults and retained root pins
    # ------------------------------------------------------------------ #
    def test_v2_defaults_and_retained_root_pins(self):
        import run_compression_supervised_v2 as runner

        self.assertEqual(runner.DEFAULT_RUN_ID, "compression_supervised_20260914_v2")
        self.assertEqual(runner.PROSPECTIVE_PLAN.name, "COMPRESSION_INTEGRATION_FIT_PLAN_V3.json")
        self.assertEqual(runner.RELEASE_PLAN.name, "COMPRESSION_INTEGRATION_FIT_PLAN_V4.json")
        self.assertEqual(runner.DEFAULT_REVIEW.name, "COMPRESSION_BINARY_REPAIR_REVIEW_V2.md")
        self.assertEqual(ci.JOB_ID, "compression_integration_20260914_v2")
        self.assertEqual(ci.PLAN_SCHEMA, "compression_integration_plan.v2")
        study = ci.STUDY_DIR
        for relative in ("native_development_runner_v2.py", "compression_integration_v2.py",
                         "compression_comparison_v2.py", "run_compression_supervised_v2.py"):
            self.assertIn(study + "/" + relative, ci.SOURCE_FILES)
        self.assertIn("threadpoolctl", ci.RUNTIME_PACKAGES)
        self.assertEqual(runner.integration.SOURCE_FILES, ci.SOURCE_FILES)

    # ------------------------------------------------------------------ #
    # 5. reuse the prior synthetic suite unmodified
    # ------------------------------------------------------------------ #
    def test_prior_synthetic_suite_still_passes(self):
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(prior.CompressionComparisonTests)
        outcome = unittest.TestResult()
        suite.run(outcome)
        detail = "; ".join(str(error) for error in outcome.errors + outcome.failures)
        self.assertTrue(outcome.wasSuccessful(), detail)
        self.assertEqual(outcome.testsRun, 10)

    # ------------------------------------------------------------------ #
    # 6. the V2 core preserves the untouched V1 output surface
    # ------------------------------------------------------------------ #
    def test_recording_run_outputs_and_budget_unchanged(self):
        counters = self.result["counters"]
        self.assertEqual((counters["cv_fits"], counters["refits"], counters["pca_fits"]), (210, 14, 12))
        self.assertEqual(counters["reference_fits"], 0)
        self.assertEqual(len(self.result["cells"]), 8)
        self.assertFalse(self.result["holdout_accessed"])
        self.assertFalse(self.result["validation_used_for_selection"])
        models = sorted(path.name for path in (self.output / "models").glob("*.pkl"))
        self.assertIn("prompted__full3072__binary.pkl", models)
        self.assertIn("prompted__pca8__fourclass.pkl", models)
        for name in models:
            self.assertIn(name.split("__")[2], ("binary.pkl", "fourclass.pkl"))


if __name__ == "__main__":
    unittest.main()
