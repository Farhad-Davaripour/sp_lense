"""Tests for the PLAN_V3 stage-3 cached J-lens pilot glue.

Two suites live here:

* hermetic synthetic tests of ``jlens_pilot_v1`` and ``run_jlens_pilot_v1`` that
  import no provider, read no real cache/lens tensor body and run no forward;
* exactly ONE tiny real sklearn ``LogisticRegression`` fit through the pinned
  ``grouped_driver._default_factory`` proving integer binary classes and a
  ``(n, 2)`` ``predict_proba`` shape, plus a zero-fit metadata check of the real
  resolved interfaces (both cache indexes, the four manifests plus blueprint and
  the pinned lens hash).

Run with the locked classifier runtime (NumPy + SciPy + scikit-learn, no torch
needed because this glue is model-free):

    development/classifier_generalization_v2/.runtime/Scripts/python.exe \\
        development/jlens_trigger_v1/test_jlens_pilot_v1.py
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import io as _io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

import jlens_core_v2 as core
import jlens_io_v1 as io
import jlens_pilot_v1 as pilot
import run_jlens_pilot_v1 as module

ROOT = module.ROOT


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def synthetic_dataset(n_features=3, seed=20260914):
    """320 aligned cases (240 TRAIN / 80 VALIDATION), label-correlated scores."""
    rng = np.random.default_rng(seed)
    labels = ["SELF"] * 80 + ["OTHER"] * 80 + ["NONTERMINATION"] * 80 + ["ORDINARY"] * 80
    splits = ["TRAIN"] * 240 + ["VALIDATION"] * 80
    folds = [None] * 320
    for index in range(240):
        folds[index] = index % 5
    case_ids = ["C%03d" % index for index in range(320)]
    y = np.asarray([1 if label == "SELF" else 0 for label in labels], dtype=float)
    scores = {}
    for method in pilot.METHODS:
        scores[method] = {}
        for condition in pilot.CONDITIONS:
            scores[method][condition] = {}
            for layer in pilot.SCORE_LAYERS:
                block = rng.standard_normal((320, n_features)).astype(np.float32)
                block[:, 0] = block[:, 0] + (2.0 * y - 0.5)
                scores[method][condition][layer] = block
    return case_ids, labels, splits, folds, scores


class ToyEstimator:
    """Deterministic stand-in exposing the reviewed estimator surface only."""

    def __init__(self, C):
        self.C = C
        self.max_iter = 1000
        self.n_iter_ = np.asarray([3])
        self.classes_ = np.asarray([0, 1])
        self.coef_ = np.zeros((1, 1))
        self.intercept_ = np.asarray([0.0])

    def fit(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y)
        positive = x[y == 1].mean(axis=0) if np.any(y == 1) else np.zeros(x.shape[1])
        negative = x[y == 0].mean(axis=0) if np.any(y == 0) else np.zeros(x.shape[1])
        self.coef_ = (positive - negative).reshape(1, -1)
        self.intercept_ = np.asarray([0.0])
        return self

    def decision_function(self, x):
        return np.asarray(x, float) @ self.coef_.ravel() + self.intercept_[0]

    def predict_proba(self, x):
        positive = 1.0 / (1.0 + np.exp(-self.decision_function(x)))
        return np.column_stack([1.0 - positive, positive])


class ToyFactory:
    """Counts fits; optionally fails at one call to exercise invalidation."""

    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def __call__(self, model, C):
        self.calls += 1
        if self.fail_at is not None and self.calls == self.fail_at:
            raise RuntimeError("toy fit failure")
        return ToyEstimator(C)


# --------------------------------------------------------------------------- #
# constants / spec fidelity
# --------------------------------------------------------------------------- #
class ConstantsTests(unittest.TestCase):
    def test_finite_cell_and_fit_arithmetic_matches_plan_v3(self):
        self.assertEqual(pilot.CELL_A_CELLS, 72)
        self.assertEqual(pilot.CELL_B_CELLS, 24)
        self.assertEqual(pilot.TOTAL_CELLS, 96)
        self.assertEqual(pilot.CV_FITS_MAX, 120)
        self.assertEqual(pilot.REFITS_MAX, 12)
        self.assertEqual(pilot.TOTAL_FITS_MAX, 132)
        self.assertEqual(pilot.CV_FITS_MAX + pilot.REFITS_MAX, pilot.TOTAL_FITS_MAX)
        self.assertEqual(pilot.MAX_SELECTED_LOGIT_READS, 23040)
        self.assertEqual(pilot.MAX_RECORDS_DECODED, 3840)
        self.assertEqual(pilot.MAX_CACHED_ACTIVATION_TENSORS, 6)

    def test_caps_match_the_authoritative_plan(self):
        self.assertEqual(pilot.CAPS["readout_seconds"], 900)
        self.assertEqual(pilot.CAPS["fit_seconds"], 600)
        self.assertEqual(pilot.CAPS["hard_total_seconds"], 1800)
        self.assertEqual(pilot.CAPS["output_bytes"], 67108864)
        self.assertEqual(pilot.CAPS["working_memory_gib"], 2.5)
        for key in ("model_loads", "tokenizer_loads", "forwards", "derivatives", "cone_fits", "pca_fits"):
            self.assertEqual(pilot.CAPS[key], 0)
        self.assertIs(pilot.CAPS["hyperparameter_search"], False)
        self.assertEqual(module.CAPS, pilot.CAPS)

    def test_frozen_axes_and_quantile_grid(self):
        self.assertEqual(pilot.CONCEPT_SURFACES, core.CONCEPT_SURFACES_V1)
        self.assertEqual(pilot.SCORE_LAYERS, (6, 10, 18))
        self.assertEqual(pilot.CONDITIONS, ("unprompted", "prompted"))
        self.assertEqual(pilot.METHODS, ("j_lens_transport", "logit_lens_J_identity_matched_control"))
        self.assertEqual(pilot.C_VALUES, (0.1, 1.0))
        self.assertEqual(pilot.FOLDS, (0, 1, 2, 3, 4))
        self.assertEqual(len(pilot.QUANTILE_GRID), 19)
        self.assertEqual(pilot.QUANTILE_GRID[0], 0.05)
        self.assertEqual(pilot.QUANTILE_GRID[-1], 0.95)

    def test_tie_break_chain_is_the_plan_v3_chain(self):
        self.assertEqual(
            pilot.TIE_BREAK_CHAIN[:6],
            (
                "max min(precision, recall)",
                "max f1",
                "lower C",
                "earlier method index (j_lens before J=I)",
                "earlier condition index (unprompted before prompted)",
                "lower layer",
            ),
        )
        self.assertIn("lower cell index", pilot.TIE_BREAK_CHAIN)
        self.assertIn("binary-first", pilot.TIE_BREAK_CHAIN)
        self.assertIn("abs(tau-0.5)", pilot.TIE_BREAK_CHAIN)
        self.assertIn("smaller tau", pilot.TIE_BREAK_CHAIN)
        self.assertIn("lower feature count", pilot.TIE_BREAK_CHAIN)
        self.assertIn("lower concept-surface index", pilot.TIE_BREAK_CHAIN)

    def test_modules_are_model_free_at_import(self):
        for mod in (pilot, module):
            tree = ast.parse(inspect.getsource(mod))
            roots = set()
            for node in tree.body:
                if isinstance(node, ast.Import):
                    roots.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    roots.add((node.module or "").split(".")[0])
            self.assertFalse(roots & module.FORBIDDEN_ROOTS, msg=mod.__name__)
        self.assertNotIn("torch", sys.modules)


# --------------------------------------------------------------------------- #
# selection primitives
# --------------------------------------------------------------------------- #
class SelectionPrimitiveTests(unittest.TestCase):
    def test_selection_key_ordering_follows_the_chain(self):
        base = dict(
            metrics={"precision": 0.5, "recall": 0.5, "f1": 0.5},
            tau=0.5,
            c_value=0.1,
            method_index=0,
            condition_index=0,
            layer=6,
            cell_index=0,
            is_binary=True,
            feature_count=1,
            surface_index=0,
        )
        weaker = dict(base, metrics={"precision": 0.4, "recall": 0.4, "f1": 0.9})
        self.assertLess(
            pilot.cell_selection_key(**base),
            pilot.cell_selection_key(**weaker),
            "max min(precision, recall) must dominate f1",
        )
        higher_c = dict(base, c_value=1.0)
        self.assertLess(pilot.cell_selection_key(**base), pilot.cell_selection_key(**higher_c))
        later_method = dict(base, method_index=1)
        self.assertLess(pilot.cell_selection_key(**base), pilot.cell_selection_key(**later_method))
        later_condition = dict(base, condition_index=1)
        self.assertLess(pilot.cell_selection_key(**base), pilot.cell_selection_key(**later_condition))
        higher_layer = dict(base, layer=10)
        self.assertLess(pilot.cell_selection_key(**base), pilot.cell_selection_key(**higher_layer))
        self.assertIsNone(pilot.cell_selection_key(**dict(base, metrics={"precision": None, "recall": 1.0, "f1": 0.0})))

    def test_wilson_interval_is_bounded_and_undefined_without_denominator(self):
        low, high = pilot.wilson_interval(5, 10)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertEqual(pilot.wilson_interval(0, 0), (None, None))
        self.assertEqual(pilot.wilson_interval(10, 10)[1], 1.0)

    def test_threshold_selection_uses_full_train_quantile(self):
        partitions = pilot.fold_partitions(np.asarray([index % 5 for index in range(240)]))
        y_train = np.asarray([1] * 60 + [0] * 180)
        rng = np.random.default_rng(3)
        fold_train, fold_test = {}, {}
        for held, _, test_index in partitions:
            fold_train[held] = rng.standard_normal(test_index.size)
            fold_test[held] = rng.standard_normal(test_index.size)
        full_train = rng.standard_normal(240)
        selection = pilot.select_threshold_cell(
            fold_train_scores=fold_train,
            fold_test_scores=fold_test,
            full_train_scores=full_train,
            y_train=y_train,
            partitions=partitions,
            key_context=dict(
                c_value=0.0, method_index=0, condition_index=0, layer=6, cell_index=0,
                is_binary=True, feature_count=1, surface_index=0,
            ),
        )
        self.assertEqual(selection["status"], "VALID")
        expected = core.select_quantile_threshold(full_train, selection["selected_q"])
        self.assertEqual(selection["tau"], expected)

    def test_no_eligible_threshold_reports_and_does_not_fall_back_to_validation(self):
        partitions = pilot.fold_partitions(np.asarray([index % 5 for index in range(100)]))
        zeros = {held: np.zeros(test_index.size) for held, _, test_index in partitions}
        negatives = {held: np.full(test_index.size, -5.0) for held, _, test_index in partitions}
        selection = pilot.select_threshold_cell(
            fold_train_scores=zeros,
            fold_test_scores=negatives,
            full_train_scores=np.zeros(100),
            y_train=np.asarray([1] * 20 + [0] * 80),
            partitions=partitions,
            key_context=dict(
                c_value=0.0, method_index=0, condition_index=0, layer=6, cell_index=0,
                is_binary=True, feature_count=1, surface_index=0,
            ),
        )
        self.assertEqual(selection["status"], "NO_ELIGIBLE_TRAIN_CANDIDATE")
        self.assertEqual(selection["error"], "NO_ELIGIBLE_TRAIN_CANDIDATE")

    def test_split_metrics_break_down_negative_classes(self):
        labels = ["SELF", "SELF", "OTHER", "NONTERMINATION", "ORDINARY"]
        scores = np.asarray([1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float32)
        metrics = pilot.split_metrics(scores, 0.0, labels)
        self.assertEqual((metrics["tp"], metrics["fn"]), (1, 1))
        self.assertEqual(metrics["negative_counts"]["OTHER"], {"n": 1, "predicted_positive": 1})
        self.assertEqual(metrics["negative_counts"]["NONTERMINATION"], {"n": 1, "predicted_positive": 0})
        self.assertEqual(metrics["negative_counts"]["ORDINARY"], {"n": 1, "predicted_positive": 1})
        self.assertEqual(len(metrics["wilson_precision"]), 2)


# --------------------------------------------------------------------------- #
# readout (identity matched control, read budget)
# --------------------------------------------------------------------------- #
class ReadoutTests(unittest.TestCase):
    def _tensors(self, d_model=4, layers=(6,)):
        identity = np.eye(d_model, dtype=np.float32)
        return {
            "jacobians": {layer: identity.copy() for layer in layers},
            "norm": np.zeros(d_model, dtype=np.float32),
            "rows": np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        }

    def test_identity_control_matches_transport_when_jacobian_is_identity(self):
        d_model = 4
        contract = core.ReadoutContract(d_model=d_model, token_ids=(1, 2), vocab_size=8)
        hidden = np.asarray([[0.5, -1.0, 2.0, 0.25], [1.0, 0.0, -0.5, 0.5]], dtype=np.float32)
        tensors = self._tensors(d_model)
        transported = pilot.method_scores(hidden, 6, pilot.METHODS[0], tensors, contract)
        control = pilot.method_scores(hidden, 6, pilot.METHODS[1], tensors, contract)
        self.assertEqual(transported.shape, (2, 2))
        self.assertTrue(np.array_equal(transported, control))

    def test_read_budget_counts_selected_rows_and_fails_closed(self):
        d_model = 4
        contract = core.ReadoutContract(d_model=d_model, token_ids=(1, 2), vocab_size=8)
        hidden = np.ones((3, d_model), dtype=np.float32)
        counters = {"selected_logit_reads": 0}
        pilot.method_scores(hidden, 6, pilot.METHODS[0], self._tensors(d_model), contract, counters=counters)
        self.assertEqual(counters["selected_logit_reads"], 6)
        counters["selected_logit_reads"] = pilot.MAX_SELECTED_LOGIT_READS - 1
        with self.assertRaisesRegex(pilot.PilotError, "SELECTED_LOGIT_READ_BUDGET"):
            pilot.method_scores(hidden, 6, pilot.METHODS[1], self._tensors(d_model), contract, counters=counters)

    def test_surface_validation_rejects_bad_tables(self):
        good = [
            {"surface": name, "token_id": index + 1, "single_token": True}
            for index, name in enumerate(pilot.CONCEPT_SURFACES)
        ]
        retained = pilot.validate_surfaces(good)
        self.assertEqual(len(retained), 6)
        with self.assertRaisesRegex(pilot.PilotError, "SURFACE_NOT_SINGLE_TOKEN"):
            pilot.validate_surfaces([dict(good[0], single_token=False)])
        with self.assertRaisesRegex(pilot.PilotError, "SURFACE_UNKNOWN"):
            pilot.validate_surfaces([{"surface": " nope", "token_id": 1, "single_token": True}])
        with self.assertRaisesRegex(pilot.PilotError, "SURFACE_BUDGET"):
            pilot.validate_surfaces(good + [{"surface": " end", "token_id": 99, "single_token": True}])
        with self.assertRaisesRegex(pilot.PilotError, "SURFACE_DUPLICATE_TOKEN"):
            pilot.validate_surfaces([good[0], dict(good[1], token_id=good[0]["token_id"])])


# --------------------------------------------------------------------------- #
# full frozen-cell evaluation over synthetic scores
# --------------------------------------------------------------------------- #
class PilotRunTests(unittest.TestCase):
    def test_full_pilot_builds_96_cells_with_132_fit_budget(self):
        case_ids, labels, splits, folds, scores = synthetic_dataset(n_features=6)
        factory = ToyFactory()
        result = pilot.run_pilot(
            scores=scores, case_ids=case_ids, labels=labels, splits=splits, folds=folds,
            surface_names=list(pilot.CONCEPT_SURFACES), factory=factory,
        )
        self.assertEqual(result["counts"]["cell_A_cells_built"], 72)
        self.assertEqual(result["counts"]["cell_B_cells_built"], 24)
        self.assertEqual(result["counts"]["total_cells_built"], 96)
        self.assertEqual(len(result["cells"]), 96)
        self.assertEqual(result["fits"]["cv_fits"], 120)
        self.assertEqual(result["fits"]["refits"], 12)
        self.assertEqual(result["fits"]["total"], 132)
        self.assertEqual(factory.calls, 132)
        self.assertEqual(len(result["refits"]), 12)
        self.assertIs(result["validation_used_for_selection"], False)

    def test_selection_is_unchanged_when_validation_is_replaced(self):
        case_ids, labels, splits, folds, scores = synthetic_dataset(n_features=3)
        first = pilot.run_pilot(
            scores=scores, case_ids=case_ids, labels=labels, splits=splits, folds=folds,
            surface_names=list(pilot.CONCEPT_SURFACES[:3]), factory=ToyFactory(),
        )
        mutated_scores = {method: {condition: {} for condition in pilot.CONDITIONS} for method in pilot.METHODS}
        for method in pilot.METHODS:
            for condition in pilot.CONDITIONS:
                for layer in pilot.SCORE_LAYERS:
                    block = scores[method][condition][layer].copy()
                    block[240:] = block[240:] + 100.0  # validation scores only
                    mutated_scores[method][condition][layer] = block
        mutated_labels = list(labels)
        mutated_labels[240:] = ["OTHER"] * 80  # validation labels only
        second = pilot.run_pilot(
            scores=mutated_scores, case_ids=case_ids, labels=mutated_labels, splits=splits, folds=folds,
            surface_names=list(pilot.CONCEPT_SURFACES[:3]), factory=ToyFactory(),
        )
        self.assertEqual(
            [cell["selection_key"] for cell in first["cells"]],
            [cell["selection_key"] for cell in second["cells"]],
        )
        self.assertEqual(
            [(refit["C"], refit["tau"]) for refit in first["refits"]],
            [(refit["C"], refit["tau"]) for refit in second["refits"]],
        )
        self.assertNotEqual(
            [cell["validation_metrics"] for cell in first["cells"]],
            [cell["validation_metrics"] for cell in second["cells"]],
        )

    def test_failed_fold_invalidates_the_whole_cell_and_is_reported(self):
        case_ids, labels, splits, folds, scores = synthetic_dataset(n_features=3)
        result = pilot.run_pilot(
            scores=scores, case_ids=case_ids, labels=labels, splits=splits, folds=folds,
            surface_names=list(pilot.CONCEPT_SURFACES[:3]), factory=ToyFactory(fail_at=2),
        )
        invalid = [cell for cell in result["cells"] if cell["status"] == "INVALID"]
        self.assertTrue(invalid)
        for cell in invalid:
            self.assertEqual(cell["error"], "INCOMPLETE_FOLD")
            self.assertTrue(cell["fold_errors"])
        self.assertEqual(result["fits"]["fit_errors"], 1)

    def test_degenerate_train_fold_invalidates_every_cell(self):
        case_ids, labels, splits, folds, scores = synthetic_dataset(n_features=3)
        # Put every TRAIN SELF row in fold 0 so fold 0's TRAIN partition lacks SELF.
        for index in range(240):
            info = labels[index]
            folds[index] = 0 if info == "SELF" else (index % 4) + 1
        result = pilot.run_pilot(
            scores=scores, case_ids=case_ids, labels=labels, splits=splits, folds=folds,
            surface_names=list(pilot.CONCEPT_SURFACES[:3]), factory=ToyFactory(),
        )
        self.assertEqual(result["counts"]["valid"], 0)
        self.assertEqual(result["counts"]["invalid"], result["counts"]["total_cells_built"])
        self.assertEqual(result["status"], "NO_ELIGIBLE_CANDIDATE")
        cell_a_errors = {cell["error"] for cell in result["cells"] if cell["scheme"] == "cell_A"}
        self.assertEqual(cell_a_errors, {"DEGENERATE_FOLD"})

    def test_learned_cell_probability_is_the_sigmoid_of_its_logit(self):
        case_ids, labels, splits, folds, scores = synthetic_dataset(n_features=3)
        result = pilot.run_pilot(
            scores=scores, case_ids=case_ids, labels=labels, splits=splits, folds=folds,
            surface_names=list(pilot.CONCEPT_SURFACES[:3]), factory=ToyFactory(),
        )
        learned = [cell for cell in result["cells"] if cell["scheme"] == "cell_B" and cell["status"] == "VALID"]
        self.assertTrue(learned)
        for cell in learned:
            expected = core.logistic_probability(np.asarray(cell["oof_logit"], dtype=np.float32))
            np.testing.assert_allclose(np.asarray(cell["oof_p_self"]), expected, rtol=0, atol=1e-6)

    def test_fit_budget_violation_aborts_and_is_not_pooled(self):
        case_ids, labels, splits, folds, scores = synthetic_dataset(n_features=3)
        with self.assertRaisesRegex(pilot.BudgetError, "CV_FIT_BUDGET"):
            pilot.run_pilot(
                scores=scores, case_ids=case_ids, labels=labels, splits=splits, folds=folds,
                surface_names=list(pilot.CONCEPT_SURFACES[:3]), factory=ToyFactory(),
                counters={"cv_fits": pilot.CV_FITS_MAX},
            )

    def test_standardization_uses_train_fold_rows_only(self):
        train = np.asarray([[0.0, 10.0], [2.0, 14.0]])
        apply_rows = np.asarray([[1.0, 12.0]])
        standardized = pilot._standardize(train, apply_rows)
        np.testing.assert_allclose(standardized, [[0.0, 0.0]], atol=1e-12)
        with self.assertRaisesRegex(pilot.PilotError, "DEGENERATE_STANDARDIZATION"):
            pilot._standardize(np.asarray([[1.0, 1.0], [1.0, 1.0]]), apply_rows)


# --------------------------------------------------------------------------- #
# hermetic cache-index validation (the real ``inspect_pilot_cache`` body)
# --------------------------------------------------------------------------- #
def fabricate_cache(directory, schema, condition):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    case_order = ["T%03d" % index for index in range(240)] + ["V%03d" % index for index in range(80)]
    records = []
    offset = 0
    length = 1024 * io.FLOAT_BYTES
    for case_id in case_order:
        split = "TRAIN" if case_id.startswith("T") else "VALIDATION"
        for order in ("AB", "BA"):
            for block in io.BLOCKS:
                records.append(
                    {
                        "case_id": case_id,
                        "order": order,
                        "block": block,
                        "split": split,
                        "shape": [1, io.D_MODEL],
                        "token_positions": [0],
                        "prefix_length": 1,
                        "readout_index": 0,
                        "offset": offset,
                        "length": length,
                        "sha256": "ab" * 32,
                    }
                )
                offset += length
    windows = bytes(offset)
    document = {
        "schema": schema,
        "blocks": list(io.BLOCKS),
        "dtype": "float32",
        "byte_order": "little",
        "float_bytes": io.FLOAT_BYTES,
        "width": io.D_MODEL,
        "case_count": 320,
        "view_count": 640,
        "forward_count": 640,
        "raw_bytes": len(windows),
        "case_order": case_order,
        "records": records,
    }
    if condition is not None:
        document["condition"] = condition
    index_path = directory / "index.json"
    windows_path = directory / "windows.f32"
    index_raw = module.runner.encoded(document)
    index_path.write_bytes(index_raw)
    windows_path.write_bytes(windows)
    pins = {
        "index": {"bytes": len(index_raw), "sha256": sha(index_raw)},
        "windows": {"bytes": len(windows), "sha256": sha(windows)},
    }
    return index_path, windows_path, pins, set(case_order)


class CacheValidationTests(unittest.TestCase):
    def test_both_real_cache_schemas_authenticate_and_holdout_paths_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="jlens_cache_") as temporary:
            for schema, condition in (
                ("span_capture_index.v1", None),
                ("prompted_span_capture_index.v1", "prompted_fixed_query_v1"),
            ):
                index_path, windows_path, pins, admitted = fabricate_cache(
                    Path(temporary) / schema, schema, condition
                )
                cache = pilot.inspect_pilot_cache(
                    index_path, pins["index"], windows_path, pins["windows"],
                    admitted_ids=admitted, held_ids=set(), condition=condition,
                )
                self.assertEqual(len(cache.records), 1920)
                with self.assertRaisesRegex(pilot.PilotError, "CACHE_CONDITION"):
                    pilot.inspect_pilot_cache(
                        index_path, pins["index"], windows_path, pins["windows"],
                        admitted_ids=admitted, held_ids=set(), condition="wrong",
                    )
                with self.assertRaisesRegex(pilot.PilotError, "CACHE_HOLDOUT_OVERLAP"):
                    pilot.inspect_pilot_cache(
                        index_path, pins["index"], windows_path, pins["windows"],
                        admitted_ids=admitted, held_ids={"T000"},
                    )
            bad = Path(temporary) / "bad_schema"
            index_path, windows_path, pins, admitted = fabricate_cache(bad, "bogus_schema.v1", None)
            with self.assertRaisesRegex(pilot.PilotError, "CACHE_INDEX_SCHEMA"):
                pilot.inspect_pilot_cache(
                    index_path, pins["index"], windows_path, pins["windows"],
                    admitted_ids=admitted, held_ids=set(),
                )

    def test_forbidden_path_parts_are_rejected(self):
        with self.assertRaisesRegex(pilot.PilotError, "FORBIDDEN_PATH_PART"):
            pilot.reject_forbidden_path("development/classifier_generalization_v2/holdout/x.json")
        with self.assertRaisesRegex(module.GateError, "FORBIDDEN_PATH_PART"):
            module._scope_native(
                {"path": "development/classifier_generalization_v2/private/x.json", "sha256": "a" * 64},
                "role",
            )


# --------------------------------------------------------------------------- #
# capture / worker / supervisor control shell
# --------------------------------------------------------------------------- #
def capture_ctx(root, n_features=3):
    case_ids, labels, splits, folds, _ = synthetic_dataset(n_features=n_features)
    surfaces = [
        {"surface": name, "token_id": index + 1, "single_token": True}
        for index, name in enumerate(pilot.CONCEPT_SURFACES[:n_features])
    ]
    return {
        "lock": {"caps": dict(pilot.CAPS), "run_id": "jlens_pilot_toy", "release": io.RELEASE_V1},
        "lock_sha256": "e" * 64,
        "root": Path(root),
        "data": {
            "case_order": case_ids,
            "labels": labels,
            "splits": splits,
            "folds": folds,
            "surfaces": [(item["surface"], item["token_id"]) for item in surfaces],
        },
        "output": Path(root) / "runs" / "jlens_pilot_toy",
    }


def fake_verify(ctx):
    return {"snapshot_realpath": str(Path(ctx["root"]) / "snap"), "checked_files": []}


class CaptureTests(unittest.TestCase):
    def test_capture_writes_artifacts_with_zero_model_work(self):
        with tempfile.TemporaryDirectory(prefix="jlens_capture_") as temporary:
            root = Path(temporary)
            ctx = capture_ctx(root)
            _, _, _, _, scores = synthetic_dataset(n_features=3)
            pins = module.capture(
                ctx, time.monotonic(), verify=fake_verify, scores=scores, factory=ToyFactory()
            )
            self.assertEqual(set(pins), set(module.ARTIFACTS))
            receipt = json.loads((ctx["output"] / "pilot_receipt.json").read_bytes())
            self.assertEqual(receipt["status"], "pilot_complete")
            self.assertEqual(receipt["counters"]["model_loads"], 0)
            self.assertEqual(receipt["counters"]["tokenizer_loads"], 0)
            self.assertEqual(receipt["counters"]["forwards"], 0)
            self.assertEqual(receipt["fits"]["total"], 132)
            self.assertIs(receipt["notes"]["holdout_accessed"], False)
            self.assertIs(receipt["notes"]["validation_used_for_selection"], False)
            cells = json.loads((ctx["output"] / "cell_results.json").read_bytes())
            self.assertEqual(cells["counts"]["total_cells_built"], 60)
            for name, pin in pins.items():
                self.assertEqual(sha((ctx["output"] / name).read_bytes()), pin["sha256"])

    def test_capture_rejects_bad_caps(self):
        with tempfile.TemporaryDirectory(prefix="jlens_capture_bad_") as temporary:
            ctx = capture_ctx(Path(temporary))
            ctx["lock"]["caps"]["hard_total_seconds"] = 1
            with self.assertRaisesRegex(module.GateError, "CAPS"):
                module.capture(ctx, time.monotonic(), verify=fake_verify, scores={}, factory=ToyFactory())


def write_fake_artifacts(ctx):
    output = Path(ctx["output"])
    output.mkdir(parents=True, exist_ok=True)
    pins = {}
    for name in module.ARTIFACTS:
        raw = module.runner.encoded({"schema": name, "actual_pid": os.getpid()})
        (output / name).write_bytes(raw)
        pins[name] = {"bytes": len(raw), "sha256": module.runner.sha(raw)}
    return pins


def worker_ctx(root):
    root = Path(root)
    return {
        "lock": {"run_id": "jlens_pilot_toy", "caps": dict(pilot.CAPS)},
        "lock_sha256": "e" * 64,
        "root": root,
        "base": root / "study" / "runs",
        "output": root / "study" / "runs" / "jlens_pilot_toy",
        "marker": root / "native" / "runs" / module.OWNER_NAME,
    }


class OwnershipTests(unittest.TestCase):
    def test_worker_exclusive_output_and_shared_marker_removed(self):
        with tempfile.TemporaryDirectory(prefix="jlens_own_") as temporary:
            ctx = worker_ctx(temporary)
            with mock.patch.object(module, "preflight", return_value=ctx), mock.patch.object(
                module, "capture", side_effect=lambda c, s: write_fake_artifacts(c)
            ):
                report = module.worker("lock", ctx["lock_sha256"], "a" * 32)
                self.assertEqual(report["pid"], os.getpid())
                self.assertFalse(ctx["marker"].exists())
                self.assertFalse((ctx["output"] / module.SUCCESS_RECEIPT).exists())
                with self.assertRaises(FileExistsError):
                    module.worker("lock", ctx["lock_sha256"], "b" * 32)

    def test_supervisor_verifies_pins_and_writes_success(self):
        with tempfile.TemporaryDirectory(prefix="jlens_sup_") as temporary:
            ctx = worker_ctx(temporary)

            def fake_watch(command, cwd, seconds):
                self.assertIn("--worker", command)
                pins = write_fake_artifacts(ctx)
                token = command[command.index("--token") + 1]
                report = dict(
                    status="worker_complete", pid=456, token=token,
                    run_id=ctx["lock"]["run_id"], lock_sha256=ctx["lock_sha256"], outputs=pins,
                )
                (ctx["output"] / module.WORKER_RECEIPT).write_bytes(module.runner.encoded(report))
                return 456, b""

            with mock.patch.object(module, "preflight", return_value=ctx), mock.patch.object(
                module.runner, "watch", side_effect=fake_watch
            ), mock.patch.object(module, "capture", side_effect=AssertionError("parent must not capture")):
                receipt = module.supervise("lock", ctx["lock_sha256"])
            self.assertEqual(receipt["pid"], 456)
            self.assertTrue((ctx["output"] / module.SUCCESS_RECEIPT).is_file())

    def test_supervisor_rejects_tampered_pin_and_preserves_failure(self):
        with tempfile.TemporaryDirectory(prefix="jlens_sup_bad_") as temporary:
            ctx = worker_ctx(temporary)

            def bad_watch(command, cwd, seconds):
                pins = write_fake_artifacts(ctx)
                pins[module.ARTIFACTS[0]]["sha256"] = "0" * 64
                token = command[command.index("--token") + 1]
                report = dict(
                    status="worker_complete", pid=457, token=token,
                    run_id=ctx["lock"]["run_id"], lock_sha256=ctx["lock_sha256"], outputs=pins,
                )
                (ctx["output"] / module.WORKER_RECEIPT).write_bytes(module.runner.encoded(report))
                return 457, b""

            with mock.patch.object(module, "preflight", return_value=ctx), mock.patch.object(
                module.runner, "watch", side_effect=bad_watch
            ):
                with self.assertRaisesRegex(module.GateError, "ARTIFACT_HASH"):
                    module.supervise("lock", ctx["lock_sha256"])
            self.assertFalse((ctx["output"] / module.SUCCESS_RECEIPT).exists())
            self.assertTrue(list(ctx["base"].glob("controller_failure_*.json")))

    def test_default_cli_is_preflight_only(self):
        captured = _io.StringIO()
        with mock.patch.object(module, "preflight", return_value={"lock": {"run_id": "toy"}}), mock.patch.object(
            module.sys, "stdout", captured
        ):
            code = module.main(["--lock", "L", "--sha256", "S"])
        self.assertEqual(code, 0)
        report = json.loads(captured.getvalue())
        self.assertEqual(report["status"], "preflight_pass")
        self.assertEqual(report["fits"], 0)
        self.assertEqual(report["forwards"], 0)
        self.assertEqual(report["model_loads"], 0)
        self.assertEqual(report["tensor_reads"], 0)


# --------------------------------------------------------------------------- #
# the ONE real sklearn toy fit + real safe interfaces
# --------------------------------------------------------------------------- #
class RealBackendTests(unittest.TestCase):
    def test_one_real_sklearn_default_factory_toy_fit(self):
        """Exactly one real fit: integer binary classes and (n, 2) proba shape."""
        x = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.2, 0.1],
                [0.2, -0.1, 0.0],
                [-0.1, 0.1, 0.2],
                [0.05, 0.0, -0.1],
                [3.0, 3.1, 2.9],
                [3.2, 2.8, 3.1],
                [2.9, 3.0, 3.2],
                [3.1, 3.3, 2.8],
                [2.8, 2.9, 3.0],
            ],
            dtype=float,
        )
        y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        factory = pilot.driver._default_factory
        estimator = factory(pilot.harness.MODEL_BINARY, 1.0)
        estimator.fit(x, y)
        self.assertEqual(np.asarray(estimator.classes_).tolist(), [0, 1])
        probabilities = np.asarray(estimator.predict_proba(x), float)
        self.assertEqual(probabilities.shape, (len(y), 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)
        self.assertTrue(np.isfinite(np.asarray(estimator.coef_, float)).all())
        self.assertTrue(pilot.driver._converged(estimator))

    def test_real_resolved_interfaces_authenticate_without_any_tensor_read(self):
        plan_path = ROOT / module.STUDY / "JLENS_PILOT_PLAN_V1.json"
        if not plan_path.is_file():
            self.skipTest("prospective plan not present")
        summary = module.verify_safe_interfaces(ROOT)
        self.assertEqual(summary["cases"], 320)
        self.assertEqual(summary["caches"]["unprompted_cache"]["records"], 1920)
        self.assertEqual(summary["caches"]["prompted_cache"]["records"], 1920)
        self.assertEqual(summary["caches"]["prompted_cache"]["condition"], "prompted_fixed_query_v1")
        self.assertEqual(summary["lens"]["revision"], io.LENS_PIN["revision"])
        self.assertEqual(summary["lens"]["sha256"], io.LENS_PIN["sha256"])


if __name__ == "__main__":
    unittest.main()
