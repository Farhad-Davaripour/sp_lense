"""Synthetic-only tests for compression_integration_v1.

Everything here runs against temporary fixtures: no real dataset, cache, result,
vector, holdout, provider, tokenizer, capture, model or pickle is read, no real
estimator is fit, and there is no network, install, Git, coordination or global
config work. The injected toy factory replaces the real logistic regression, and
the synthetic spawned span/prompted captures stand in for the real providers.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import compression_comparison_v1 as core
import compression_integration_v1 as ci
import harness
import linear_span_control_v1 as linear
import native_capture_contract as contract
import prompted_input_adapter_v1 as tokens
import span_classifier_driver_v1 as span

CLASS_ORDER = core.CLASS_ORDER
BLOCKS, ORDERS, WIDTH, WINDOW = span.BLOCKS, span.ORDERS, span.WIDTH, span.WINDOW_MAX
STUDY = "development/classifier_generalization_v2"
SPAN_RUN = "span_synth_capture_v1"
SPAN_FIT_RUN = "span_synth_fit_v1"
PROMPTED_RUN = "prompted_synth_capture_v1"
BASELINE_RUN = ci.BASELINE_RUN_ID
FIXTURE_RUN_ID = "compression_synth_v1"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_bytes(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def write_json(path, value):
    return write_bytes(path, (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())


def pin(path, root):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": path.relative_to(Path(root).resolve()).as_posix(), "sha256": sha(raw), "bytes": len(raw)}


def build_cases():
    manifests = {role: [] for role in ("original_train", "added_train", "original_validation", "added_validation")}
    labels, groups, folds, splits = {}, {}, {}, {}
    train_ids, original_ids, added_ids = [], [], []
    blueprint = []

    def emit(role, split, case_id, group_id, label, fold):
        case = {"case_id": case_id, "split": split, "group_id": group_id,
                "class_label": label, "development_fold": fold}
        manifests[role].append(case)
        labels[case_id], groups[case_id], folds[case_id], splits[case_id] = label, group_id, fold, split

    for prefix, role in (("TO", "original_train"), ("TA", "added_train")):
        for group in range(5):
            group_id = "G%s%d" % (prefix, group)
            blueprint.append({"group_id": group_id, "split": "TRAIN", "development_fold": group})
            for index in range(24):
                case_id = "%s%d_%02d" % (prefix, group, index)
                emit(role, "TRAIN", case_id, group_id, CLASS_ORDER[index % 4], group)
                train_ids.append(case_id)
    for prefix, role in (("VO", "original_validation"), ("VA", "added_validation")):
        for index in range(40):
            group_id = "G%s%02d" % (prefix, index)
            case_id = "%s%02d" % (prefix, index)
            blueprint.append({"group_id": group_id, "split": "VALIDATION", "development_fold": None})
            emit(role, "VALIDATION", case_id, group_id, CLASS_ORDER[index % 4], None)
            (original_ids if prefix == "VO" else added_ids).append(case_id)
    return {"case_order": train_ids + original_ids + added_ids, "train_ids": train_ids,
            "original_ids": original_ids, "added_ids": added_ids, "labels": labels, "groups": groups,
            "folds": folds, "splits": splits, "manifests": manifests, "blueprint": blueprint}


def build_vectors(cases, seed):
    rng = np.random.default_rng(seed)
    vectors = {}
    for case_id in cases["case_order"]:
        base = np.zeros(WIDTH, dtype=np.float64)
        base[CLASS_ORDER.index(cases["labels"][case_id])] = 1.0
        for order in ORDERS:
            vectors[(case_id, order)] = (base + rng.normal(0.0, 0.05, WIDTH)).astype(np.float32).reshape(1, WIDTH)
    return vectors


def build_records(cases, vectors, condition):
    records, chunks, offset, input_hashes = [], [], 0, {}
    prefix_sha = sha(("prefix|" + condition).encode())
    for case_id in cases["case_order"]:
        for order in ORDERS:
            input_hashes[(case_id, order)] = sha(("input|%s|%s|%s" % (condition, case_id, order)).encode())
            for block in BLOCKS:
                payload = vectors[(case_id, order)].astype("<f4").tobytes()
                records.append({
                    "case_id": case_id, "condition": condition, "split": cases["splits"][case_id],
                    "order": order, "block": block, "shape": [1, WIDTH], "dtype": "float32",
                    "byte_order": "little", "float_bytes": 4, "token_positions": [0],
                    "prefix_length": 1, "readout_index": 0, "final_input_index": 2,
                    "prefix_sha256": prefix_sha, "input_ids_sha256": input_hashes[(case_id, order)],
                    "offset": offset, "length": len(payload), "sha256": sha(payload),
                })
                chunks.append(payload)
                offset += len(payload)
    return records, b"".join(chunks), input_hashes


def build_manifests(root, cases):
    pins = {}
    for role, split in (("original_train", "TRAIN"), ("added_train", "TRAIN"),
                        ("original_validation", "VALIDATION"), ("added_validation", "VALIDATION")):
        document = {"split": split, "case_count": len(cases["manifests"][role]), "cases": cases["manifests"][role]}
        path = write_json(root / STUDY / "manifests" / (role + ".json"), document)
        pins[role] = pin(path, root)
    blueprint = write_json(root / STUDY / "manifests" / "blueprint.json", {"groups": cases["blueprint"]})
    pins["blueprint"] = pin(blueprint, root)
    return pins


def build_span_capture(root, cases, manifest_pins, source_pins):
    run_dir = root / STUDY / "runs" / SPAN_RUN
    records, windows_raw, _ = build_records(cases, build_vectors(cases, 101), "unprompted")
    lock = {"schema": span.LOCK_SCHEMA, "scientific_execution_authorized": True, "run_id": SPAN_RUN,
            "caps": {"fits": 0, "derivatives": 0}}
    lock_path = write_json(run_dir / "lock.json", lock)
    lock_sha = sha(lock_path.read_bytes())
    index = {"schema": span.INDEX_SCHEMA, "condition": "unprompted", "lock_sha256": lock_sha, "run_id": SPAN_RUN,
             "blocks": list(BLOCKS), "orders": list(ORDERS), "dtype": "float32", "byte_order": "little",
             "float_bytes": 4, "width": WIDTH, "window_max": WINDOW, "case_order": cases["case_order"],
             "case_count": len(cases["case_order"]), "view_count": len(cases["case_order"]) * len(ORDERS),
             "forward_count": len(cases["case_order"]) * len(ORDERS), "raw_bytes": len(windows_raw),
             "records": records}
    index_path = write_json(run_dir / "index.json", index)
    windows_path = write_bytes(run_dir / "windows.f32", windows_raw)
    receipt = {"status": "complete", "run_id": SPAN_RUN, "lock_sha256": lock_sha,
               "outputs": {"windows.f32": pin(windows_path, root), "index.json": pin(index_path, root)}}
    receipt_path = write_json(run_dir / "supervisor_success.json", receipt)
    plan = {"schema": span.PLAN_SCHEMA, "job_id": span.JOB_ID, "run_id": SPAN_FIT_RUN, "seconds": 600,
            "cv_fit_limit": 50, "refit_limit": 10, "transform_fit_limit": 6, "thresholds": list(span.TAUS),
            "xgboost": dict(span.EXPECTED_XGB_PARAMS), "representations": list(span.REPRESENTATIONS),
            "holdout_access": False,
            "source": {"receipt": pin(receipt_path, root), "lock": pin(lock_path, root),
                       "index": pin(index_path, root), "windows": pin(windows_path, root)},
            "manifests": manifest_pins}
    plan_path = write_json(root / STUDY / "span_synth_plan.json", plan)
    return {"plan_path": plan_path, "plan_pin": pin(plan_path, root), "lock_sha": lock_sha}


def build_prompted_capture(root, cases, adapter_sha):
    run_dir = root / STUDY / "runs" / PROMPTED_RUN
    records, windows_raw, input_hashes = build_records(cases, build_vectors(cases, 202), tokens.PROMPTED_CONDITION)
    rows = [{"case_id": case_id, "split": cases["splits"][case_id], "order": order, "tokens": 3,
             "prefix_tokens": 1, "last_shared_token_id": contract.LAST_SHARED_ID,
             "input_sha256": input_hashes[(case_id, order)]}
            for case_id in cases["case_order"] for order in ORDERS]
    sanity = {"status": "PASS", "query": tokens.FIXED_QUERY, "query_sha256": tokens.QUERY_SHA256,
              "adapter_sha256": adapter_sha, "exact_decode_reencode": True, "query_in_shared_prefix": True,
              "last_shared_token_ids": [contract.LAST_SHARED_ID], "max_full_tokens": 3, "max_prefix_tokens": 1,
              "counts": {"cases": len(cases["case_order"]), "views": len(cases["case_order"]) * len(ORDERS),
                          "tokenizer_loads": 1, "model_loads": 0, "forwards": 0, "fits": 0},
              "rows": rows}
    sanity_path = write_json(run_dir / "prompt_sanity.json", sanity)
    lock = {"schema": ci.PROMPTED_LOCK_SCHEMA, "scientific_execution_authorized": True, "run_id": PROMPTED_RUN,
            "caps": {"seconds": 1800, "forwards": len(cases["case_order"]) * len(ORDERS), "tokens_per_view": 8,
                      "model_loads": 1, "tokenizer_loads": 1, "fits": 0, "derivatives": 0,
                      "raw_bytes": len(windows_raw), "output_bytes": 192 * 1024 * 1024},
            "inputs": {"prompt_sanity": pin(sanity_path, root)},
            "reference_locks": [{"run_id": "old_a"}, {"run_id": "old_b"}]}
    lock_path = write_json(run_dir / "lock.json", lock)
    lock_sha = sha(lock_path.read_bytes())
    index = {"schema": ci.PROMPTED_INDEX_SCHEMA, "condition": tokens.PROMPTED_CONDITION,
             "query": tokens.FIXED_QUERY, "query_sha256": tokens.QUERY_SHA256, "lock_sha256": lock_sha,
             "run_id": PROMPTED_RUN, "blocks": list(BLOCKS), "orders": list(ORDERS), "dtype": "float32",
             "byte_order": "little", "float_bytes": 4, "width": WIDTH, "window_max": WINDOW,
             "case_order": cases["case_order"], "case_count": len(cases["case_order"]),
             "view_count": len(cases["case_order"]) * len(ORDERS),
             "forward_count": len(cases["case_order"]) * len(ORDERS), "raw_bytes": len(windows_raw),
             "records": records}
    index_path = write_json(run_dir / "index.json", index)
    windows_path = write_bytes(run_dir / "windows.f32", windows_raw)
    receipt = {"schema": ci.PROMPTED_RECEIPT_SCHEMA, "status": "capture_complete",
               "condition": tokens.PROMPTED_CONDITION, "query_sha256": tokens.QUERY_SHA256,
               "lock_sha256": lock_sha, "run_id": PROMPTED_RUN,
               "artifacts": {"windows.f32": pin(windows_path, root), "index.json": pin(index_path, root)}}
    receipt_path = write_json(run_dir / "capture_receipt.json", receipt)
    success = {"status": "complete", "run_id": PROMPTED_RUN, "lock_sha256": lock_sha,
               "outputs": {"windows.f32": pin(windows_path, root), "index.json": pin(index_path, root),
                            "capture_receipt.json": pin(receipt_path, root)}}
    success_path = write_json(run_dir / "supervisor_success.json", success)
    return {"lock": pin(lock_path, root), "index": pin(index_path, root), "windows": pin(windows_path, root),
            "receipt": pin(receipt_path, root), "success": pin(success_path, root), "run_dir": run_dir}


def build_baseline(root, cases):
    run_dir = root / STUDY / "runs" / BASELINE_RUN
    train_ids = cases["train_ids"]
    truth = [1 if cases["labels"][case_id] == "SELF" else 0 for case_id in train_ids]
    rng = np.random.default_rng(303)
    candidates, metrics = [], {}
    for family in core.FAMILIES:
        p_self = np.clip(np.where(np.asarray(truth) == 1, 0.9, 0.1) + rng.normal(0.0, 0.03, len(train_ids)),
                         0.001, 0.999)
        metrics[family] = harness.binary_gate_metrics(truth, p_self, 0.5)
        candidates.append({"family": family, "C": 1.0, "valid": True, "selected_tau": 0.5,
                           "oof_metrics": metrics[family], "oof_p_self": p_self.tolist()})
    cv_path = write_json(run_dir / "cv_scores.json", {"oof_case_ids": train_ids,
                         "oof_truth": [cases["labels"][case_id] for case_id in train_ids],
                         "candidates": candidates, "folds": []})
    for family in core.FAMILIES:
        model_path = write_bytes(run_dir / ("estimator_linear_%s.pkl" % family),
                                 ("synthetic-%s-model" % family).encode())
        prediction_path = write_json(run_dir / ("predictions_linear_%s.json" % family), {"family": family})
        write_json(run_dir / ("family_%s.json" % family),
                   {"family": family, "status": "FITTED", "C": 1.0, "selected_tau": 0.5,
                    "oof_metrics": metrics[family], "artifact": model_path.name,
                    "artifact_sha256": sha(model_path.read_bytes()), "reload_exact": True,
                    "evaluation": {"train": {"self_gate": metrics[family]}}})
    write_json(run_dir / "development_results.json",
               {"status": "COMPLETE", "job_id": "linear_span_implementation_20260914_1233",
                "counters": {"cv_fits": 30, "refits": 2}, "holdout_accessed": False})
    write_json(run_dir / "fit_plan.json", {"schema": "linear_span_fit_plan.v1"})
    write_json(run_dir / "supervisor_success.json", {"status": "complete"})
    return cv_path


def build_root(root):
    root = Path(root)
    source_pins = {relative: sha(write_bytes(root / relative, ("# synthetic %s\n" % relative).encode()).read_bytes())
                   for relative in ci.SOURCE_FILES}
    cases = build_cases()
    manifest_pins = build_manifests(root, cases)
    span_capture = build_span_capture(root, cases, manifest_pins, source_pins)
    prompted = build_prompted_capture(
        root, cases, source_pins[STUDY + "/prompted_input_adapter_v1.py"])
    build_baseline(root, cases)
    baseline_dir = root / STUDY / "runs" / BASELINE_RUN
    plan = {
        "schema": ci.PLAN_SCHEMA, "job_id": ci.JOB_ID, "run_id": FIXTURE_RUN_ID, "seconds": ci.SECONDS,
        "output_bytes": ci.OUTPUT_BYTES, "cv_fit_limit": ci.CV_FIT_LIMIT, "refit_limit": ci.REFIT_LIMIT,
        "pca_fit_limit": ci.PCA_FIT_LIMIT, "holdout_access": False, "threads": 1,
        "C": list(core.C_VALUES), "thresholds": list(core.TAUS), "folds": list(core.FOLDS),
        "conditions": list(core.CONDITIONS), "source_files": source_pins,
        "runtime_packages": {name: importlib.metadata.version(name) for name in ci.RUNTIME_PACKAGES},
        "reference_span": {"run_id": SPAN_RUN, "plan": span_capture["plan_pin"]},
        "reference_baseline": {"run_id": BASELINE_RUN,
                               "files": {name: pin(baseline_dir / name, root) for name in ci.BASELINE_NAMES}},
        "prompted_source": {"run_id": PROMPTED_RUN, "forwards": len(cases["case_order"]) * len(ORDERS),
                            "tokens_per_view": 8, **{key: prompted[key]
                                                     for key in ("lock", "index", "windows", "receipt", "success")}},
        "scientific_execution_authorized": True,
    }
    plan_path = write_json(root / STUDY / "compression_integration_plan.json", plan)
    case_data = {"cases": cases, "span_capture": span_capture, "prompted": prompted}
    return {"plan_path": plan_path, "plan_sha": sha(plan_path.read_bytes()), "plan": plan, "case_data": case_data}


class ToyEstimator:
    def __init__(self, family, state):
        self.family, self.state = family, state
        self.classes_ = np.asarray([0, 1]) if family == "binary" else np.asarray(CLASS_ORDER)
        self.labels_, self.means = [], None

    def fit(self, x, labels):
        self.state["fits"] += 1
        x, self.labels_ = np.asarray(x, float), list(labels)
        if x.shape[0] != len(self.labels_):
            raise ValueError("feature/label row count mismatch")
        present = [0, 1] if self.family == "binary" else list(CLASS_ORDER)
        self.means = [x[[i for i, value in enumerate(self.labels_) if value == cls]].mean(axis=0)
                      if cls in self.labels_ else np.zeros(x.shape[1]) for cls in present]
        return self

    def predict_proba(self, x):
        x = np.asarray(x, float)
        logits = np.stack([-((x - mean) ** 2).sum(axis=1) for mean in self.means], axis=1)
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = weights / weights.sum(axis=1, keepdims=True)
        return (np.column_stack([1 - probabilities[:, 0], probabilities[:, 0]])
                if self.family == "binary" else probabilities)


class ToyFactory:
    def __init__(self):
        self.state = {"fits": 0}

    def __call__(self, family, C):
        return ToyEstimator(family, self.state)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="compression_integration_"))
        cls.root = cls.tmp / "pristine"
        cls.fixture = build_root(cls.root)
        cls.plan_path, cls.plan_sha = cls.fixture["plan_path"], cls.fixture["plan_sha"]
        cls.inputs = ci.build_inputs(cls.fixture["plan"], cls.root)
        cls.preflight = ci.preflight(cls.plan_path, cls.plan_sha, cls.root)
        cls.factory = ToyFactory()
        cls.result = ci.run(cls.plan_path, cls.plan_sha, cls.root, factory=cls.factory)
        cls.output = ci.output_dir(cls.fixture["plan"], cls.root)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def copied_root(self, name):
        destination = self.tmp / name
        shutil.copytree(self.root, destination)
        return destination

    def test_job_and_zero_fit_preflight(self):
        self.assertEqual(ci.JOB_ID, "compression_integration_20260914_takeover_v1")
        self.assertEqual(ci.PROMPTED_LOCK_SCHEMA, "prompted_span_development_execution.v1")
        summary = self.preflight
        self.assertEqual(summary["status"], "preflight_pass")
        self.assertEqual((summary["cases"], summary["train"], summary["validation"],
                          summary["original"], summary["added"], summary["feature_width"]),
                         (320, 240, 80, 40, 40, 3072))
        self.assertEqual((summary["fits"], summary["pca_fits"]), (0, 0))
        self.assertEqual(summary["query_sha256"], tokens.QUERY_SHA256)
        self.assertFalse(summary["holdout_accessed"])
        self.assertEqual(summary["prompted_rows"], 640)
        self.assertEqual((ci.SECONDS, ci.OUTPUT_BYTES), (600, 256 * 1024 * 1024))

    def test_per_layer_l2_pair_average_features(self):
        conditions, prompted = self.inputs["conditions"], self.inputs["prompted"]
        order = conditions["metadata"]["case_order"]
        for case_id in (prompted["train_ids"][0], prompted["original_ids"][0], prompted["added_ids"][0]):
            row = order.index(case_id)
            expected = linear.features_for_case(prompted["windows"][case_id])
            self.assertTrue(np.allclose(conditions["prompted"]["matrix"][row], expected))
            unprompted_expected = linear.features_for_case(self.inputs["unprompted"]["windows"][case_id])
            self.assertTrue(np.allclose(conditions["unprompted"]["matrix"][row], unprompted_expected))
        validation_id = prompted["original_ids"][0]
        average = 0.5 * (prompted["ab_windows"][validation_id].astype(np.float64)
                         + prompted["ba_windows"][validation_id].astype(np.float64))
        self.assertTrue(np.allclose(prompted["windows"][validation_id], average))
        index = conditions["metadata"]["validation_ids"].index(validation_id)
        self.assertTrue(np.allclose(conditions["prompted"]["ab_matrix"][index],
                                    linear.features_for_case(prompted["ab_windows"][validation_id])))
        self.assertTrue(np.allclose(conditions["prompted"]["ba_matrix"][index],
                                    linear.features_for_case(prompted["ba_windows"][validation_id])))

    def test_run_budget_reference_and_exclusive_output(self):
        counters = self.result["counters"]
        self.assertEqual((counters["cv_fits"], counters["refits"], counters["pca_fits"]), (210, 14, 12))
        self.assertEqual((counters["reference_fits"], counters["fit_errors"], counters["refit_errors"]), (0, 0, 0))
        self.assertEqual(self.factory.state["fits"], 224)
        self.assertFalse(self.result["holdout_accessed"])
        self.assertFalse(self.result["validation_used_for_selection"])
        models = sorted(path.name for path in (self.output / "models").glob("*.pkl"))
        self.assertIn("prompted__full3072__binary.pkl", models)
        self.assertNotIn("unprompted__full3072__binary.pkl", models)
        self.assertNotIn("unprompted__full3072__fourclass.pkl", models)
        pca = json.loads((self.output / "pca_index.json").read_text(encoding="utf-8"))
        self.assertEqual(len(pca["fits"]), 12)
        total = sum(path.stat().st_size for path in self.output.rglob("*") if path.is_file())
        self.assertLessEqual(total, ci.OUTPUT_BYTES)
        reference = json.loads((self.output / "reference_full.json").read_text(encoding="utf-8"))
        self.assertTrue(reference["cited"] and not reference["refit"])
        self.assertEqual(reference["provenance"]["reference_baseline"]["run_id"], BASELINE_RUN)
        self.assertEqual(self.result["reference"]["provenance"]["run_id"], BASELINE_RUN)
        with self.assertRaises(ci.CompressionIntegrationError) as caught:
            ci.run(self.plan_path, self.plan_sha, self.root, factory=ToyFactory())
        self.assertIn("OUTPUT_EXISTS", str(caught.exception))

    def test_identity_mismatch_rejected(self):
        unprompted = self.inputs["unprompted"]
        broken = copy.deepcopy(unprompted)
        first = broken["train_ids"][0]
        broken["labels"][first] = CLASS_ORDER[(CLASS_ORDER.index(broken["labels"][first]) + 1) % 4]
        with self.assertRaises(ci.CompressionIntegrationError) as caught:
            ci.build_conditions(unprompted, broken, {})
        self.assertIn("CONDITION_META_MISMATCH", str(caught.exception))
        broken = copy.deepcopy(unprompted)
        broken["train_ids"] = broken["train_ids"][:-1]
        with self.assertRaises(ci.CompressionIntegrationError) as caught:
            ci.build_conditions(unprompted, broken, {})
        self.assertIn("CONDITION_CASE_MISMATCH", str(caught.exception))

    def test_record_accounting_and_last_token_provenance(self):
        index = self.inputs["auth"]["index"]
        sanity = self.inputs["auth"]["sanity"]
        caps = self.inputs["auth"]["lock"]["caps"]
        short = copy.deepcopy(index)
        short["records"] = short["records"][:-1]
        with self.assertRaises(ci.CompressionIntegrationError) as caught:
            ci.verify_last_token_provenance(short, sanity, caps)
        self.assertIn("PROMPTED_RECORD_ACCOUNTING", str(caught.exception))
        for field, value, code in (("prefix_tokens", 2, "PROMPTED_SANITY_PREFIX"),
                                   ("input_sha256", "0" * 64, "PROMPTED_SANITY_INPUT")):
            broken = copy.deepcopy(sanity)
            broken["rows"][0][field] = value
            with self.assertRaises(ci.CompressionIntegrationError) as caught:
                ci.verify_last_token_provenance(index, broken, caps)
            self.assertIn(code, str(caught.exception))
        broken = copy.deepcopy(sanity)
        broken["last_shared_token_ids"] = [contract.LAST_SHARED_ID + 1]
        with self.assertRaises(ci.CompressionIntegrationError) as caught:
            ci.verify_last_token_provenance(index, broken, caps)
        self.assertIn("PROMPTED_SANITY_BOUNDARY", str(caught.exception))

    def test_query_fingerprint_and_pin_tampering(self):
        root = self.copied_root("tamper_query")
        index_path = root / self.fixture["plan"]["prompted_source"]["index"]["path"]
        index = json.loads(index_path.read_bytes())
        index["query_sha256"] = "0" * 64
        write_json(index_path, index)
        plan = copy.deepcopy(self.fixture["plan"])
        plan["prompted_source"]["index"] = pin(index_path, root)
        with self.assertRaises(ci.CompressionIntegrationError) as caught:
            ci.authenticate_prompted(plan, root)
        self.assertIn("PROMPTED_QUERY", str(caught.exception))
        root = self.copied_root("tamper_pin")
        windows_path = root / self.fixture["plan"]["prompted_source"]["windows"]["path"]
        windows_path.write_bytes(windows_path.read_bytes() + b"\x00")
        with self.assertRaises(ci.CompressionIntegrationError) as caught:
            ci.authenticate_prompted(self.fixture["plan"], root)
        self.assertIn("PIN_SHA", str(caught.exception))

    def test_baseline_pin_and_reference_mismatch(self):
        root = self.copied_root("tamper_baseline")
        plan = copy.deepcopy(self.fixture["plan"])
        cv_path = root / plan["reference_baseline"]["files"]["cv_scores.json"]["path"]
        cv = json.loads(cv_path.read_bytes())
        cv["oof_truth"][0] = CLASS_ORDER[(CLASS_ORDER.index(cv["oof_truth"][0]) + 1) % 4]
        write_json(cv_path, cv)
        plan["reference_baseline"]["files"]["cv_scores.json"] = pin(cv_path, root)
        with self.assertRaises(ci.CompressionIntegrationError) as caught:
            ci.build_reference(plan, root, self.inputs["unprompted"])
        self.assertIn("BASELINE_OOF_TRUTH", str(caught.exception))
        root = self.copied_root("tamper_model")
        plan = copy.deepcopy(self.fixture["plan"])
        model_path = root / plan["reference_baseline"]["files"]["estimator_linear_binary.pkl"]["path"]
        model_path.write_bytes(b"changed")
        plan["reference_baseline"]["files"]["estimator_linear_binary.pkl"] = pin(model_path, root)
        with self.assertRaises(ci.CompressionIntegrationError) as caught:
            ci.build_reference(plan, root, self.inputs["unprompted"])
        self.assertIn("BASELINE_ARTIFACT_SHA", str(caught.exception))

    def test_source_and_plan_pins(self):
        root = self.copied_root("tamper_source")
        relative = STUDY + "/harness.py"
        (root / relative).write_bytes(b"# changed\n")
        with self.assertRaises(ci.CompressionIntegrationError) as caught:
            ci.verify_sources_and_runtime(self.fixture["plan"], root)
        self.assertIn("SOURCE_MISMATCH", str(caught.exception))
        with self.assertRaises(ci.CompressionIntegrationError) as caught:
            ci.load_plan(self.plan_path, "0" * 64)
        self.assertIn("PLAN_DIGEST", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
