"""Synthetic-only unit tests for span_classifier_driver_v1.

No real manifest, vector, result, snapshot or private HOLDOUT file is read, no
model/tokenizer/capture/network/install/Git/coordination action happens, and no
real XGBoost fit is performed. A deterministic in-memory toy factory proves the
50 CV + 10 refit budget, the 5+1 shared-transform fits, grouped-fold isolation,
invalid-fold candidate invalidation, and exact serialize/reload.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import span_classifier_driver_v1 as span

CLASS_ORDER = span.CLASS_ORDER
WIDTH = span.WIDTH
BLOCKS = span.BLOCKS
ORDERS = span.ORDERS
BASE = "development/classifier_generalization_v2"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def jb(value):
    return (json.dumps(value, indent=2) + "\n").encode()


# --------------------------------------------------------------------------- #
# deterministic picklable toy estimator (no XGBoost, no native providers)
# --------------------------------------------------------------------------- #
class ToyEstimator:
    def __init__(self, family, state):
        self.family = family
        self.state = state
        self.classes_ = np.asarray([0, 1]) if family == "binary" else np.asarray(CLASS_ORDER)

    def fit(self, x, labels):
        self.state["fits"] += 1
        if self.state.get("fail_at") == self.state["fits"]:
            raise RuntimeError("toy_injected_fold_failure")
        x = np.asarray(x, float)
        labels = list(labels)
        means = []
        for label in CLASS_ORDER:
            rows = x[[index for index, value in enumerate(labels) if value == label]]
            means.append(rows.mean(axis=0) if rows.size else np.zeros(x.shape[1]))
        self.means = np.stack(means)
        return self

    def predict_proba(self, x):
        x = np.asarray(x, float)
        logits = np.stack([-((x - self.means[index]) ** 2).sum(axis=1) for index in range(len(CLASS_ORDER))], axis=1)
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        probabilities = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-300)
        if self.family == "binary":
            return np.column_stack([1 - probabilities[:, 0], probabilities[:, 0]])
        return probabilities


class ToyFactory:
    def __init__(self, fail_at=None):
        self.state = {"fits": 0, "fail_at": fail_at}

    def __call__(self, family):
        return ToyEstimator(family, self.state)


# --------------------------------------------------------------------------- #
# synthetic fixture
# --------------------------------------------------------------------------- #
FOLD_T = [0, 1, 2, 3, 4, 0, 1]
FOLD_XT = [2, 3, 4, 0, 1, 2, 3]


def _cases():
    rows = []
    for index in range(120):
        rows.append(("T%02d_S%03d" % (index % 7 + 1, index), "T%02d" % (index % 7 + 1), "TRAIN", "original",
                     CLASS_ORDER[index % 4], FOLD_T[index % 7]))
    for index in range(120):
        rows.append(("XT%02d_S%03d" % (index % 7 + 1, index), "XT%02d" % (index % 7 + 1), "TRAIN", "added",
                     CLASS_ORDER[index % 4], FOLD_XT[index % 7]))
    for index in range(40):
        rows.append(("V%02d_S%03d" % (index % 4 + 1, index), "V%02d" % (index % 4 + 1), "VALIDATION", "original",
                     CLASS_ORDER[index % 4], None))
    for index in range(40):
        rows.append(("XV%02d_S%03d" % (index % 4 + 1, index), "XV%02d" % (index % 4 + 1), "VALIDATION", "added",
                     CLASS_ORDER[index % 4], None))
    return rows


def build_fixture(root):
    root = Path(root)
    base = root / BASE
    base.mkdir(parents=True, exist_ok=True)
    rows = _cases()
    blueprint_groups = []
    for group, fold in [("T%02d" % (i + 1), FOLD_T[i]) for i in range(7)]:
        blueprint_groups.append({"group_id": group, "split": "TRAIN", "development_fold": fold})
    for group, fold in [("XT%02d" % (i + 1), FOLD_XT[i]) for i in range(7)]:
        blueprint_groups.append({"group_id": group, "split": "TRAIN", "development_fold": fold})
    for group in ["V%02d" % (i + 1) for i in range(4)]:
        blueprint_groups.append({"group_id": group, "split": "VALIDATION", "development_fold": None})
    for group in ["XV%02d" % (i + 1) for i in range(4)]:
        blueprint_groups.append({"group_id": group, "split": "VALIDATION", "development_fold": None})
    (base / "BLUEPRINT.json").write_bytes(jb({"groups": blueprint_groups}))

    manifests = {}
    for role, (split, cohort) in span.MANIFEST_ROLES.items():
        subset = [row for row in rows if row[2] == split and row[3] == cohort]
        cases = [
            {
                "case_id": row[0],
                "group_id": row[1],
                "split": row[2],
                "class_label": row[4],
                "development_fold": row[5],
            }
            for row in subset
        ]
        (base / (role + ".json")).write_bytes(jb({"split": split, "case_count": len(cases), "cases": cases}))
        manifests[role] = role + ".json"

    rng = np.random.default_rng(20260914)
    chunks, records, case_order = [], [], []
    for index, row in enumerate(rows):
        case_id, _group, split, _cohort, label, _fold = row
        case_order.append(case_id)
        prefix_length = 1 + (index % 4)
        window_length = min(span.WINDOW_MAX, prefix_length)
        signal = np.zeros(WIDTH, dtype=np.float64)
        signal[:8] = CLASS_ORDER.index(label) + 1.0
        views = {}
        for order in ORDERS:
            noise = rng.normal(0.0, 0.05, size=(len(BLOCKS), window_length, WIDTH))
            views[order] = (signal[None, None, :] + noise).astype("<f4")
        for order in ORDERS:
            for block_index, block in enumerate(BLOCKS):
                payload = views[order][block_index].astype("<f4").tobytes()
                offset = sum(len(chunk) for chunk in chunks)
                chunks.append(payload)
                records.append(
                    {
                        "case_id": case_id,
                        "split": split,
                        "order": order,
                        "block": block,
                        "shape": [window_length, WIDTH],
                        "dtype": "float32",
                        "byte_order": "little",
                        "float_bytes": 4,
                        "token_positions": list(range(prefix_length - window_length, prefix_length)),
                        "prefix_length": prefix_length,
                        "readout_index": prefix_length - 1,
                        "final_input_index": prefix_length,
                        "prefix_sha256": "0" * 64,
                        "input_ids_sha256": "1" * 64,
                        "offset": offset,
                        "length": len(payload),
                        "sha256": sha(payload),
                    }
                )
    windows_raw = b"".join(chunks)
    (base / "windows.f32").write_bytes(windows_raw)
    index = {
        "schema": span.INDEX_SCHEMA,
        "lock_sha256": None,
        "run_id": "span_capture_test_v1",
        "blocks": list(BLOCKS),
        "orders": list(ORDERS),
        "dtype": "float32",
        "byte_order": "little",
        "float_bytes": 4,
        "width": WIDTH,
        "window_max": span.WINDOW_MAX,
        "case_order": case_order,
        "case_count": len(case_order),
        "view_count": len(records) // len(BLOCKS),
        "forward_count": len(records) // len(BLOCKS),
        "raw_bytes": len(windows_raw),
        "records": records,
    }
    lock = {
        "schema": span.LOCK_SCHEMA,
        "scientific_execution_authorized": True,
        "run_id": "span_capture_test_v1",
        "caps": {"seconds": 1800, "forwards": 640, "tokens_per_view": 320, "model_loads": 1,
                 "tokenizer_loads": 1, "fits": 0, "derivatives": 0, "raw_bytes": len(windows_raw)},
    }
    lock_raw = jb(lock)
    lock_sha = sha(lock_raw)
    (base / "lock.json").write_bytes(lock_raw)
    index_raw = jb(dict(index, lock_sha256=lock_sha))
    (base / "index.json").write_bytes(index_raw)
    receipt = {
        "status": "complete",
        "lock_sha256": lock_sha,
        "run_id": "span_capture_test_v1",
        "outputs": {
            "windows.f32": {"bytes": len(windows_raw), "sha256": sha(windows_raw)},
            "index.json": {"bytes": len(index_raw), "sha256": sha(index_raw)},
        },
    }
    receipt_raw = jb(receipt)
    (base / "receipt.json").write_bytes(receipt_raw)

    plan = {
        "schema": span.PLAN_SCHEMA,
        "job_id": span.JOB_ID,
        "run_id": "span_classifier_fit_test_v1",
        "seconds": 600,
        "cv_fit_limit": span.CV_FIT_LIMIT,
        "refit_limit": span.REFIT_LIMIT,
        "transform_fit_limit": span.TRANSFORM_FIT_LIMIT,
        "thresholds": list(span.TAUS),
        "xgboost": dict(span.EXPECTED_XGB_PARAMS),
        "representations": list(span.REPRESENTATIONS),
        "holdout_access": False,
        "source": {
            "receipt": {"path": BASE + "/receipt.json", "bytes": len(receipt_raw), "sha256": sha(receipt_raw)},
            "lock": {"path": BASE + "/lock.json", "bytes": len(lock_raw), "sha256": lock_sha},
            "index": {"path": BASE + "/index.json", "bytes": len(index_raw), "sha256": sha(index_raw)},
            "windows": {"path": BASE + "/windows.f32", "bytes": len(windows_raw), "sha256": sha(windows_raw)},
        },
        "manifests": {
            role: {"path": BASE + "/" + filename, "sha256": sha((base / filename).read_bytes())}
            for role, filename in manifests.items()
        },
    }
    plan["manifests"]["blueprint"] = {"path": BASE + "/BLUEPRINT.json", "sha256": sha((base / "BLUEPRINT.json").read_bytes())}
    plan_raw = jb(plan)
    (base / "fit_plan.json").write_bytes(plan_raw)
    return {
        "root": str(root),
        "plan": plan,
        "plan_path": str(base / "fit_plan.json"),
        "plan_sha256": sha(plan_raw),
        "lock_sha256": lock_sha,
        "windows_raw": windows_raw,
    }


def write_plan(root, plan):
    path = Path(root) / BASE / "fit_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = jb(plan)
    path.write_bytes(raw)
    return str(path), sha(raw)


def clone_fixture(source_root):
    destination = tempfile.mkdtemp(prefix="span_driver_clone_")
    shutil.copytree(source_root, destination, dirs_exist_ok=True, ignore=shutil.ignore_patterns("runs"))
    return destination


# --------------------------------------------------------------------------- #
class SpanClassifierDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="span_driver_")
        cls.fixture = build_fixture(cls.tmp)
        cls.factory = ToyFactory()
        cls.result = span.run(
            cls.fixture["plan_path"], cls.fixture["plan_sha256"], cls.fixture["lock_sha256"],
            root=cls.tmp, factory=cls.factory,
        )
        cls.output = Path(cls.tmp) / BASE / "runs" / cls.result["run_id"]
        cls.manifest = span.load_manifests(cls.fixture["plan"]["manifests"], cls.tmp)
        cls.source = span.load_source(cls.fixture["plan"], cls.fixture["lock_sha256"], cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # -- provenance and budget --------------------------------------------- #
    def test_zero_model_preflight(self):
        report = span.preflight(
            self.fixture["plan_path"], self.fixture["plan_sha256"], self.fixture["lock_sha256"], root=self.tmp
        )
        self.assertEqual(report["status"], "preflight_pass")
        self.assertEqual(report["records"], 1920)
        self.assertEqual((report["train"], report["validation"]), (240, 80))
        self.assertEqual((report["validation_original"], report["validation_added"]), (40, 40))
        self.assertEqual(report["fits"], 0)

    def test_budget_and_transform_sharing(self):
        counters = self.result["counters"]
        self.assertEqual(counters["cv_fits"], 50)
        self.assertEqual(counters["refits"], 10)
        self.assertEqual(counters["transform_fits"], 6)
        self.assertEqual(counters["fit_errors"], 0)
        self.assertEqual(counters["candidates"], 10)
        self.assertEqual(self.factory.state["fits"], 60)
        self.assertEqual(self.result["status"], "COMPLETE")
        self.assertFalse(self.result["holdout_accessed"])

    def test_fold_isolation(self):
        folds = json.loads((self.output / "cross_validation.json").read_text(encoding="utf-8"))["folds"]
        self.assertEqual([record["fold"] for record in folds], [0, 1, 2, 3, 4])
        assignment = {}
        for record in folds:
            self.assertFalse(set(record["train_groups"]) & set(record["test_groups"]))
            for group in record["test_groups"]:
                self.assertNotIn(group, assignment)
                assignment[group] = record["fold"]
        self.assertEqual(len(assignment), len(set(assignment)))

    def test_representation_partition(self):
        self.assertEqual(span.FEATURE_DIM, 8272)
        self.assertEqual([span.REPRESENTATION_WIDTHS[name] for name in span.REPRESENTATIONS], [3072, 3072, 2048, 36, 44])
        covered = []
        for name in span.REPRESENTATIONS:
            window = span.REPRESENTATION_SLICES[name]
            covered.extend(range(window.start, window.stop))
        self.assertEqual(covered, list(range(8272)))
        self.assertTrue(self.result["features"]["separate_models_per_representation"])

    def test_selection_separation_and_reload(self):
        self.assertEqual(self.result["train_selected"]["selected_using"], "TRAIN_GROUPED_OOF_ONLY")
        self.assertIsNotNone(self.result["validation_winner_exploratory"])
        self.assertEqual(
            self.result["validation_winner_exploratory"]["selected_using"],
            "EXPLORATORY_VALIDATION_ONLY_NOT_FOR_SELECTION",
        )
        self.assertFalse(self.result["validation_used_for_selection"])
        self.assertEqual(len(self.result["candidates"]), 10)
        four_class_seen = False
        for entry in self.result["candidates"]:
            self.assertEqual(entry["status"], "FITTED")
            self.assertTrue(entry["reload_exact"])
            combined = entry["evaluation"]["validation_combined"]
            self.assertIn("negative_class_false_positives", combined)
            self.assertIn("answer_order_consistency", combined)
            if entry["family"] == "fourclass":
                four_class_seen = True
                self.assertIn("four_class", combined)
                self.assertIn("macro_f1", combined["four_class"])
        self.assertTrue(four_class_seen)

    def test_exclusive_outputs(self):
        with self.assertRaises(span.SpanDriverError) as caught:
            span.run(
                self.fixture["plan_path"], self.fixture["plan_sha256"], self.fixture["lock_sha256"],
                root=self.tmp, factory=ToyFactory(),
            )
        self.assertIn("OUTPUT_EXISTS", str(caught.exception))

    def test_no_real_estimator_imported(self):
        self.assertNotIn("xgboost", sys.modules)

    # -- decoder authentication -------------------------------------------- #
    def _mutate_index(self, **changes):
        index = copy.deepcopy(self.source["index"])
        for key, value in changes.items():
            index[key] = value
        return index

    def test_decoder_rejects_bad_accounting_and_layout(self):
        cases = [
            ("INDEX_RECORDS", lambda index: index.__setitem__("records", index["records"][:-1])),
            ("INDEX_CASE_COUNT", lambda index: index.__setitem__("case_count", 319)),
            ("INDEX_CASE_UNKNOWN", lambda index: index.__setitem__("case_order", index["case_order"][:-1] + ["UNKNOWN_CASE"])),
            ("INDEX_DTYPE", lambda index: index.__setitem__("dtype", "float64")),
            ("INDEX_BYTE_ORDER", lambda index: index.__setitem__("byte_order", "big")),
            ("INDEX_WIDTH", lambda index: index.__setitem__("width", 512)),
            ("RECORD_DTYPE", lambda index: index["records"][0].__setitem__("dtype", "float64")),
            ("RECORD_WIDTH", lambda index: index["records"][0].__setitem__("shape", [index["records"][0]["shape"][0], 512])),
            ("RECORD_POSITIONS_LAST", lambda index: index["records"][6].__setitem__("token_positions", [0, 2])),
            ("RECORD_OFFSET_BOUNDS", lambda index: index["records"][0].__setitem__("offset", len(self.fixture["windows_raw"]) + 1)),
            ("RECORD_OVERLAP", lambda index: (
                index["records"][0].__setitem__("offset", index["records"][1]["offset"]),
                index["records"][0].__setitem__("length", index["records"][1]["length"]),
                index["records"][0].__setitem__("sha256", index["records"][1]["sha256"]),
            )),
            ("RECORD_DUPLICATE", lambda index: index["records"][1].__setitem__("block", index["records"][0]["block"])),
            ("RECORD_WINDOW_LENGTH", lambda index: index["records"][0].__setitem__("shape", [17, WIDTH])),
        ]
        for code, mutate in cases:
            with self.subTest(code=code):
                index = self._mutate_index()
                mutate(index)
                with self.assertRaises(span.SpanDriverError) as caught:
                    span.decode_index(index, self.fixture["windows_raw"], self.manifest)
                self.assertIn(code, str(caught.exception))

    def test_decoder_rejects_record_sha(self):
        index = self._mutate_index()
        index["records"][0]["sha256"] = "0" * 64
        with self.assertRaises(span.SpanDriverError) as caught:
            span.decode_index(index, self.fixture["windows_raw"], self.manifest)
        self.assertIn("RECORD_SHA", str(caught.exception))

    def test_decoder_rejects_manifest_unknown_case(self):
        manifest = copy.deepcopy(self.manifest)
        del manifest["cases"][self.source["index"]["case_order"][0]]
        with self.assertRaises(span.SpanDriverError) as caught:
            span.decode_index(self.source["index"], self.fixture["windows_raw"], manifest)
        self.assertIn("INDEX_CASE_UNKNOWN", str(caught.exception))

    # -- source authentication --------------------------------------------- #
    def _clone(self):
        root = clone_fixture(self.tmp)
        self.addCleanup(shutil.rmtree, root, True)
        return root

    def _pinned_bytes(self, root, plan, role, raw):
        pin = dict(plan["source"][role])
        (Path(root) / pin["path"]).write_bytes(raw)
        pin["bytes"] = len(raw)
        pin["sha256"] = sha(raw)
        plan["source"][role] = pin
        return pin

    def test_source_receipt_status_and_lock_sha(self):
        for code, change in (
            ("RECEIPT_STATUS", lambda receipt: receipt.__setitem__("status", "started")),
            ("RECEIPT_STATUS", lambda receipt: receipt.__setitem__("status", "capture_complete")),
            ("RECEIPT_LOCK_SHA", lambda receipt: receipt.__setitem__("lock_sha256", "0" * 64)),
            ("RECEIPT_ARTIFACT_SHA", lambda receipt: receipt["outputs"]["windows.f32"].__setitem__("sha256", "0" * 64)),
        ):
            with self.subTest(code=code):
                root = self._clone()
                plan = copy.deepcopy(self.fixture["plan"])
                receipt = json.loads((Path(root) / plan["source"]["receipt"]["path"]).read_bytes())
                change(receipt)
                self._pinned_bytes(root, plan, "receipt", jb(receipt))
                with self.assertRaises(span.SpanDriverError) as caught:
                    span.load_source(plan, self.fixture["lock_sha256"], root)
                self.assertIn(code, str(caught.exception))

    def test_source_lock_authorization_and_pin(self):
        root = self._clone()
        plan = copy.deepcopy(self.fixture["plan"])
        lock = json.loads((Path(root) / plan["source"]["lock"]["path"]).read_bytes())
        lock["scientific_execution_authorized"] = False
        pin = self._pinned_bytes(root, plan, "lock", jb(lock))
        receipt = json.loads((Path(root) / plan["source"]["receipt"]["path"]).read_bytes())
        receipt["lock_sha256"] = pin["sha256"]
        self._pinned_bytes(root, plan, "receipt", jb(receipt))
        with self.assertRaises(span.SpanDriverError) as caught:
            span.load_source(plan, pin["sha256"], root)
        self.assertIn("LOCK_NOT_AUTHORIZED", str(caught.exception))

        root = self._clone()
        plan = copy.deepcopy(self.fixture["plan"])
        plan["source"]["receipt"]["sha256"] = "0" * 64
        with self.assertRaises(span.SpanDriverError) as caught:
            span.load_source(plan, self.fixture["lock_sha256"], root)
        self.assertIn("PIN_SHA", str(caught.exception))

    # -- plan / holdout guards --------------------------------------------- #
    def test_plan_rejections(self):
        scratch = tempfile.mkdtemp(prefix="span_plan_scratch_")
        self.addCleanup(shutil.rmtree, scratch, True)
        mutations = [
            ("PLAN_SCHEMA", lambda plan: plan.__setitem__("schema", "other")),
            ("PLAN_SECONDS", lambda plan: plan.__setitem__("seconds", 601)),
            ("PLAN_CV_LIMIT", lambda plan: plan.__setitem__("cv_fit_limit", 51)),
            ("PLAN_TAUS", lambda plan: plan.__setitem__("thresholds", [0.5])),
            ("PLAN_XGB_PARAMS", lambda plan: plan["xgboost"].__setitem__("max_depth", 3)),
            ("PLAN_HOLDOUT_ACCESS", lambda plan: plan.__setitem__("holdout_access", True)),
        ]
        for code, mutate in mutations:
            with self.subTest(code=code):
                plan = copy.deepcopy(self.fixture["plan"])
                mutate(plan)
                path, digest = write_plan(scratch, plan)
                with self.assertRaises(span.SpanDriverError) as caught:
                    span.load_plan(path, digest)
                self.assertIn(code, str(caught.exception))

    def test_holdout_path_rejected(self):
        plan = copy.deepcopy(self.fixture["plan"])
        plan["manifests"]["original_train"] = dict(plan["manifests"]["original_train"])
        plan["manifests"]["original_train"]["path"] = BASE + "/holdout_custody/HOLDOUT_ADMITTED_INDEX_V15.json"
        with self.assertRaises(span.SpanDriverError) as caught:
            span.load_manifests(plan["manifests"], self.tmp)
        self.assertIn("PATH_FORBIDDEN", str(caught.exception))

    # -- invalid fold invalidates the whole candidate ---------------------- #
    def test_invalid_fold_invalidates_candidate(self):
        root = clone_fixture(self.tmp)
        self.addCleanup(shutil.rmtree, root, True)
        plan = copy.deepcopy(self.fixture["plan"])
        plan["run_id"] = "span_classifier_fit_test_invalid_v1"
        path, digest = write_plan(root, plan)
        factory = ToyFactory(fail_at=7)
        result = span.run(path, digest, self.fixture["lock_sha256"], root=root, factory=factory)
        invalid = [entry for entry in result["candidates"] if entry["status"] == "INVALID_FOLD"]
        self.assertTrue(invalid)
        for entry in invalid:
            self.assertFalse(entry["reload_exact"])
            self.assertIsNone(entry["evaluation"])
        self.assertEqual(result["counters"]["cv_fits"], 50)
        self.assertEqual(result["counters"]["fit_errors"], 1)
        self.assertEqual(result["counters"]["refits"], 9)
        self.assertEqual(result["counters"]["transform_fits"], 6)


if __name__ == "__main__":
    unittest.main()
