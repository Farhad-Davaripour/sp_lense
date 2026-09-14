"""Synthetic-only tests for ``f1_cached_selection_v2``.

No real dataset/cache/result/probability/vector/holdout/pickle/model/tokenizer/lens,
no provider, capture, network, install, Git, coordination or estimator fit happens
here. Every fixture is a hand-built synthetic pool of the exact 48 candidates and 16
artifact pins, with 240 TRAIN OOF rows. The suite proves the prospective F1-first rule
beats the frozen minPR-first rule, that validation can never reach ``select``, that a
mismatched-C saved refit fails closed for both new and reused-reference cells, and that
finite/coverage/case-order defects are rejected whole.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import unittest
from pathlib import Path

import numpy as np

import f1_cached_selection_v2 as sel

X_ID = sel.candidate_id(sel.SOURCE_COMPRESSION, "unprompted", "pca8", "binary", 1.0)
Y_ID = sel.candidate_id(sel.SOURCE_COMPRESSION, "prompted", "pca32", "binary", 0.1)
REF_ID = sel.candidate_id(sel.SOURCE_REFERENCE, "unprompted", "full3072", "binary", 10.0)
X_CELL = sel.cell_key(sel.SOURCE_COMPRESSION, "unprompted", "pca8", "binary")
REF_CELL = sel.cell_key(sel.SOURCE_REFERENCE, "unprompted", "full3072", "binary")
IDLE_ID = sel.candidate_id(sel.SOURCE_COMPRESSION, "unprompted", "pca16", "fourclass", 10.0)


def decode(cid):
    source, condition, dimension, family, token = cid.split("|")
    return source, condition, dimension, family, float(token[1:])


def train_ids():
    return ["T%03d" % index for index in range(sel.OOF_ROWS)]


def train_truth():
    return ["SELF" if index < 60 else sel.CLASS_ORDER[1 + (index % 3)] for index in range(sel.OOF_ROWS)]


def make_pool():
    ids, truth = train_ids(), train_truth()
    candidates = {}
    for cid in sel.expected_candidate_ids():
        source, condition, dimension, family, C = decode(cid)
        candidates[cid] = {"source": source, "condition": condition, "dimension": dimension,
                           "family": family, "C": C, "oof_case_ids": list(ids), "oof_truth": list(truth),
                           "oof_p_self": [0.5] * sel.OOF_ROWS}
    artifacts = {}
    for cell in sel.expected_cell_keys():
        stem = cell.replace("|", "__")
        artifacts[cell] = {"C": 1.0, "model": "models/%s.pkl" % stem,
                           "model_sha256": hashlib.sha256((cell + "model").encode()).hexdigest(),
                           "predictions": "model_results_%s.json" % stem,
                           "predictions_sha256": hashlib.sha256((cell + "pred").encode()).hexdigest()}
    return {"source_pins": {"compression_cv_scores": "a" * 64, "reference_cv_scores": "b" * 64},
            "train_ids": ids, "truth": truth, "candidates": candidates, "artifacts": artifacts}


def p_x():
    """F1-first winner: P=1.0, R=0.6, F1=0.75, minPR=0.6 at tau=0.5."""
    p = np.full(sel.OOF_ROWS, 0.1)
    p[:36] = 0.9
    return p


def p_y():
    """Historical minPR-first preference: P=R=0.7, F1=0.7, minPR=0.7 at tau=0.5.

    The 18 false-positive rows share the TP rows' 0.9 so no intermediate threshold
    can separate them and inflate F1 above the intended 0.7.
    """
    p = np.full(sel.OOF_ROWS, 0.1)
    p[:42] = 0.9
    p[60:78] = 0.9
    return p


def p_perfect():
    p = np.full(sel.OOF_ROWS, 0.1)
    p[:60] = 0.9
    return p


def base_pool():
    pool = make_pool()
    pool["candidates"][X_ID]["oof_p_self"] = p_x().tolist()
    pool["candidates"][Y_ID]["oof_p_self"] = p_y().tolist()
    return pool


def split_fixture():
    """Synthetic combined80: TP16/TN55/FP5/FN4 at tau=0.5 (matching the audited cell)."""
    case_ids = ["V%03d" % index for index in range(80)]
    truth = ["SELF" if index < 20 else "OTHER" for index in range(80)]
    p = np.full(80, 0.1)
    p[:16] = 0.9
    p[20:25] = 0.9
    return case_ids, truth, p


class FixtureTests(unittest.TestCase):
    def test_canonical_bounds_and_thresholds_are_finite_and_exact(self):
        ids = sel.expected_candidate_ids()
        self.assertEqual(len(ids), sel.CANDIDATE_LIMIT)
        self.assertEqual(len(set(ids)), sel.CANDIDATE_LIMIT)
        compression = [cid for cid in ids if cid.startswith(sel.SOURCE_COMPRESSION + "|")]
        reference = [cid for cid in ids if cid.startswith(sel.SOURCE_REFERENCE + "|")]
        self.assertEqual(len(compression), sel.NEW_CANDIDATE_LIMIT)
        self.assertEqual(len(reference), sel.REFERENCE_CANDIDATE_LIMIT)
        self.assertEqual(len(sel.expected_cell_keys()), sel.CELL_LIMIT)
        self.assertEqual(sel.THRESHOLDS, tuple(round(0.05 * index, 2) for index in range(1, 20)))
        self.assertEqual(len(sel.THRESHOLDS), sel.THRESHOLD_LIMIT)
        self.assertEqual(sel.THRESHOLDS[0], 0.05)
        self.assertEqual(sel.THRESHOLDS[-1], 0.95)
        self.assertEqual(sel.BOUNDS["oof_rows"], 240)
        self.assertEqual(sel.BOUNDS["validation_rows"], 80)
        self.assertEqual(sel.BOUNDS["estimator_fits"], 0)
        self.assertEqual(sel.BOUNDS["model_loads"], 0)
        self.assertEqual(sel.BOUNDS["seconds"], 60)
        self.assertEqual(sel.BOUNDS["output_bytes"], 16 * 1024 * 1024)
        self.assertEqual(set(sel.artifact_template()), set(sel.expected_cell_keys()))

    def test_binary_family_is_the_first_tie_break(self):
        metrics = {"precision": 0.7, "recall": 0.7, "f1": 0.7}
        binary = sel.policy_key("binary", 10.0, 3072, "prompted", 0.5, metrics)
        four = sel.policy_key("fourclass", 10.0, 3072, "prompted", 0.5, metrics)
        self.assertLess(binary, four)

    def test_f1_first_key_beats_min_pr_first(self):
        high_f1 = {"precision": 1.0, "recall": 0.6, "f1": 0.75}
        high_pr = {"precision": 0.7, "recall": 0.7, "f1": 0.7}
        self.assertLess(sel.policy_key("binary", 1.0, 8, "unprompted", 0.5, high_f1),
                        sel.policy_key("binary", 1.0, 8, "unprompted", 0.5, high_pr))


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.pool = base_pool()

    def test_select_is_train_only_and_returns_frozen_identity(self):
        before = copy.deepcopy(self.pool)
        result = sel.select(self.pool)
        self.assertEqual(self.pool, before, "select must not mutate its input")
        self.assertEqual(result["status"], "SELECTED")
        self.assertEqual(result["validation_used_for_selection"], False)
        self.assertEqual(result["validation_rows_read_during_selection"], 0)
        self.assertEqual(result["estimator_fits"], 0)
        self.assertEqual(result["model_deserializations"], 0)
        self.assertEqual(result["counts"]["candidates"], sel.CANDIDATE_LIMIT)
        self.assertEqual(result["counts"]["new_candidates"], sel.NEW_CANDIDATE_LIMIT)
        self.assertEqual(result["counts"]["reference_candidates"], sel.REFERENCE_CANDIDATE_LIMIT)
        self.assertEqual(len(result["ranking"]), sel.CANDIDATE_LIMIT)
        self.assertEqual(result["winner"]["candidate_id"], X_ID)
        self.assertEqual(len(result["frozen_identity"]), 64)
        self.assertEqual(sel.select(self.pool)["frozen_identity"], result["frozen_identity"])
        self.assertLessEqual(result["result_json_bytes"], sel.OUTPUT_BYTES)
        json.dumps(result["frozen_identity_payload"], allow_nan=False)

    def test_f1_first_picks_the_lower_min_pr_candidate(self):
        winner = sel.select(self.pool)["winner"]
        self.assertEqual(winner["candidate_id"], X_ID)
        self.assertAlmostEqual(winner["oof_metrics"]["f1"], 0.75, places=12)
        self.assertAlmostEqual(winner["oof_metrics"]["precision"], 1.0, places=12)
        self.assertAlmostEqual(winner["oof_metrics"]["recall"], 0.6, places=12)
        self.assertEqual(winner["tau"], 0.5)
        # The frozen historical minPR-first rule would have preferred Y instead.
        ranking = {entry["candidate_id"]: entry for entry in sel.select(self.pool)["ranking"]}
        x, y = ranking[X_ID], ranking[Y_ID]
        self.assertGreater(x["oof_metrics"]["f1"], y["oof_metrics"]["f1"])
        self.assertLess(min(x["oof_metrics"]["precision"], x["oof_metrics"]["recall"]),
                        min(y["oof_metrics"]["precision"], y["oof_metrics"]["recall"]))

    def test_matching_c_artifact_authorizes_post_freeze_recompute(self):
        result = sel.select(self.pool)
        self.assertEqual(result["artifact_match"]["status"], "MATCHING_C_ARTIFACT")
        self.assertEqual(result["post_freeze"]["action"], "recompute_saved_predictions_at_tau")
        self.assertEqual(result["post_freeze"]["tau"], result["winner"]["tau"])
        self.assertEqual(result["post_freeze"]["splits"], list(sel.POST_FREEZE_SPLITS))

    def test_mismatched_c_new_refit_fails_closed(self):
        pool = copy.deepcopy(self.pool)
        pool["artifacts"][X_CELL]["C"] = 0.1
        result = sel.select(pool)
        self.assertEqual(result["winner"]["candidate_id"], X_ID)
        self.assertEqual(result["artifact_match"]["status"], "REQUIRES_SEPARATELY_LOCKED_REFIT")
        self.assertEqual(result["artifact_match"]["selected_C"], 1.0)
        self.assertEqual(result["artifact_match"]["artifact_C"], 0.1)
        self.assertEqual(result["post_freeze"]["action"], "requires_separately_locked_refit")
        with self.assertRaises(sel.F1SelectionError) as caught:
            sel.recompute_split_metrics(result, "combined80", *split_fixture())
        self.assertIn("REFIT_REQUIRED", str(caught.exception))

    def test_mismatched_c_reference_baseline_fails_closed(self):
        pool = copy.deepcopy(self.pool)
        pool["candidates"][REF_ID]["oof_p_self"] = p_perfect().tolist()
        result = sel.select(pool)
        self.assertEqual(result["winner"]["candidate_id"], REF_ID)
        self.assertEqual(result["artifact_match"]["status"], "REQUIRES_SEPARATELY_LOCKED_REFIT")
        pool["artifacts"][REF_CELL]["C"] = 10.0
        matched = sel.select(pool)
        self.assertEqual(matched["artifact_match"]["status"], "MATCHING_C_ARTIFACT")
        self.assertEqual(matched["post_freeze"]["action"], "recompute_saved_predictions_at_tau")

    def test_no_eligible_threshold_candidate_is_dropped(self):
        pool = copy.deepcopy(self.pool)
        pool["candidates"][IDLE_ID]["oof_p_self"] = [0.0] * sel.OOF_ROWS
        result = sel.select(pool)
        self.assertEqual(result["counts"]["eligible_candidates"], sel.CANDIDATE_LIMIT - 1)
        self.assertNotIn(IDLE_ID, [entry["candidate_id"] for entry in result["ranking"]])

    def test_frozen_identity_binds_every_oof_array_and_artifact_c(self):
        baseline = sel.select(self.pool)["frozen_identity"]
        changed_oof = copy.deepcopy(self.pool)
        changed_oof["candidates"][IDLE_ID]["oof_p_self"][0] = 0.25
        self.assertNotEqual(sel.select(changed_oof)["frozen_identity"], baseline)
        changed_c = copy.deepcopy(self.pool)
        changed_c["artifacts"][sel.cell_key(sel.SOURCE_COMPRESSION, "prompted", "pca16", "fourclass")]["C"] = 10.0
        self.assertNotEqual(sel.select(changed_c)["frozen_identity"], baseline)


class ValidationFirewallTests(unittest.TestCase):
    def setUp(self):
        self.pool = base_pool()

    def test_select_accepts_no_validation_argument(self):
        parameters = inspect.signature(sel.select).parameters
        self.assertEqual(list(parameters), ["pool"])

    def test_validation_keys_in_the_pool_are_rejected(self):
        for key in ("validation", "validation_p_self", "combined80", "original40", "added40"):
            pool = copy.deepcopy(self.pool)
            pool[key] = {"would_change_the_winner": True}
            with self.assertRaises(sel.F1SelectionError) as caught:
                sel.select(pool)
            self.assertIn("POOL_UNKNOWN_FIELD", str(caught.exception))

    def test_post_freeze_recompute_matches_the_existing_metric_helper(self):
        import harness

        frozen = sel.select(self.pool)
        case_ids, truth, p = split_fixture()
        out = sel.recompute_split_metrics(frozen, "combined80", case_ids, truth, p)
        expected = harness.binary_gate_metrics(np.asarray([label == "SELF" for label in truth]), p,
                                               frozen["winner"]["tau"])
        for key in ("tp", "tn", "fp", "fn", "precision", "recall", "f1"):
            self.assertEqual(out["metrics"][key], expected[key])
        self.assertEqual(out["metrics"]["tp"], 16)
        self.assertEqual(out["metrics"]["tn"], 55)
        self.assertEqual(out["metrics"]["fp"], 5)
        self.assertEqual(out["metrics"]["fn"], 4)
        self.assertAlmostEqual(out["metrics"]["f1"], 32.0 / 41.0, places=12)
        self.assertEqual(out["selection_unchanged"], True)
        self.assertEqual(out["frozen_identity"], frozen["frozen_identity"])

    def test_post_freeze_case_order_mismatch_is_rejected(self):
        frozen = sel.select(self.pool)
        case_ids, truth, p = split_fixture()
        with self.assertRaises(sel.F1SelectionError) as caught:
            sel.recompute_split_metrics(frozen, "combined80", case_ids, truth, p,
                                        expected_case_ids=list(reversed(case_ids)))
        self.assertIn("SPLIT_CASE_ORDER", str(caught.exception))
        with self.assertRaises(sel.F1SelectionError) as truth_caught:
            sel.recompute_split_metrics(frozen, "combined80", case_ids, truth, p,
                                        expected_truth=list(reversed(truth)))
        self.assertIn("SPLIT_TRUTH_ORDER", str(truth_caught.exception))


class RejectionTests(unittest.TestCase):
    def setUp(self):
        self.pool = base_pool()

    def assert_code(self, code, pool):
        with self.assertRaises(sel.F1SelectionError) as caught:
            sel.select(pool)
        self.assertIn(code, str(caught.exception))

    def test_candidate_case_order_mismatch(self):
        pool = copy.deepcopy(self.pool)
        pool["candidates"][Y_ID]["oof_case_ids"] = list(reversed(pool["train_ids"]))
        self.assert_code("CANDIDATE_CASE_ORDER", pool)

    def test_candidate_truth_order_mismatch(self):
        pool = copy.deepcopy(self.pool)
        pool["candidates"][Y_ID]["oof_truth"] = list(reversed(pool["truth"]))
        self.assert_code("CANDIDATE_TRUTH_ORDER", pool)

    def test_candidate_metadata_and_c_mismatch(self):
        pool = copy.deepcopy(self.pool)
        pool["candidates"][Y_ID]["C"] = 1.0
        self.assert_code("CANDIDATE_C_MISMATCH", pool)
        pool = copy.deepcopy(self.pool)
        pool["candidates"][Y_ID]["dimension"] = "pca8"
        self.assert_code("CANDIDATE_METADATA_MISMATCH", pool)

    def test_nonfinite_out_of_range_and_short_oof_rows(self):
        pool = copy.deepcopy(self.pool)
        pool["candidates"][Y_ID]["oof_p_self"][7] = float("nan")
        self.assert_code("CANDIDATE_OOF_FINITE", pool)
        pool = copy.deepcopy(self.pool)
        pool["candidates"][Y_ID]["oof_p_self"][7] = 1.5
        self.assert_code("CANDIDATE_OOF_RANGE", pool)
        pool = copy.deepcopy(self.pool)
        pool["candidates"][Y_ID]["oof_p_self"] = pool["candidates"][Y_ID]["oof_p_self"][:-1]
        self.assert_code("CANDIDATE_OOF_SHAPE", pool)

    def test_train_id_and_truth_coverage_failures(self):
        pool = copy.deepcopy(self.pool)
        pool["train_ids"][5] = pool["train_ids"][4]
        self.assert_code("TRAIN_IDS_DUPLICATE", pool)
        pool = copy.deepcopy(self.pool)
        pool["truth"] = ["SELF"] * sel.OOF_ROWS
        self.assert_code("TRUTH_COVERAGE", pool)
        pool = copy.deepcopy(self.pool)
        pool["truth"][0] = "BOGUS"
        self.assert_code("TRUTH_LABEL", pool)

    def test_candidate_artifact_set_and_pin_shape_failures(self):
        pool = copy.deepcopy(self.pool)
        del pool["candidates"][Y_ID]
        self.assert_code("CANDIDATE_SET", pool)
        pool = copy.deepcopy(self.pool)
        del pool["artifacts"][X_CELL]
        self.assert_code("ARTIFACT_SET", pool)
        pool = copy.deepcopy(self.pool)
        pool["artifacts"][X_CELL]["model_sha256"] = "not-a-sha"
        self.assert_code("ARTIFACT_MODEL_SHA", pool)
        pool = copy.deepcopy(self.pool)
        pool["artifacts"][X_CELL]["C"] = 0.3
        self.assert_code("ARTIFACT_C", pool)
        pool = copy.deepcopy(self.pool)
        pool["source_pins"] = {}
        self.assert_code("SOURCE_PINS_SCHEMA", pool)

    def test_module_stays_model_free(self):
        source = Path(sel.__file__).read_text(encoding="utf-8")
        for token in ("import pickle", "pickle.loads", "import sklearn", "from sklearn",
                      "import torch", "import transformers"):
            self.assertNotIn(token, source)



class FrozenRepairTests(unittest.TestCase):
    def test_forged_digest_rejected(self):
        frozen = sel.select(base_pool())
        frozen["frozen_identity"] = "a" * 64
        with self.assertRaisesRegex(sel.F1SelectionError, "FROZEN_IDENTITY_MISMATCH"):
            sel.recompute_split_metrics(frozen, "combined80", *split_fixture())

    def test_tampered_post_fields_rejected(self):
        for field, value in [("tau", 0.05), ("C", 10.0), ("cell", "wrong"),
                             ("predictions_sha256", "f" * 64), ("predictions", "wrong.json")]:
            frozen = sel.select(base_pool())
            frozen["post_freeze"][field] = value
            with self.subTest(field=field), self.assertRaises(sel.F1SelectionError):
                sel.recompute_split_metrics(frozen, "combined80", *split_fixture())

    def test_mismatched_refit_cannot_be_bypassed(self):
        pool = base_pool()
        winner = sel.select(pool)["winner"]
        cell = sel.cell_key(winner["source"], winner["condition"], winner["dimension"], winner["family"])
        pool["artifacts"][cell]["C"] = 10.0 if winner["C"] != 10.0 else 0.1
        frozen = sel.select(pool)
        frozen["post_freeze"] = {"action": "recompute_saved_predictions_at_tau", "tau": 0.05}
        with self.assertRaisesRegex(sel.F1SelectionError, "REFIT_REQUIRED"):
            sel.recompute_split_metrics(frozen, "combined80", *split_fixture())

    def test_no_positive_validation_is_reported(self):
        frozen = sel.select(base_pool())
        ids, truth, _ = split_fixture()
        result = sel.recompute_split_metrics(frozen, "combined80", ids, truth, np.zeros(80))
        self.assertIsNone(result["metrics"]["precision"])
        self.assertEqual(result["metrics"]["recall"], 0.0)
        self.assertEqual(result["metrics"]["f1"], 0.0)
        self.assertEqual(result["metrics"]["fn"], 20)

if __name__ == "__main__":
    unittest.main()
