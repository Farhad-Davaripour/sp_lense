"""Tests for the two-phase ``run_jlens_pilot_v2`` integration.

Hermetic: no provider import, no real lens/model/cache tensor, no forward. The
real Phase-B preflight and run are exercised separately by the finite lock. Run
with the locked classifier runtime (NumPy + SciPy + scikit-learn, no torch):

    development/classifier_generalization_v2/.runtime/Scripts/python.exe \\
        development/jlens_trigger_v1/test_run_jlens_pilot_v2.py
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import io as _io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np

import jlens_core_v2 as core
import jlens_io_v1 as io
import jlens_pilot_v1 as pilot
import run_jlens_pilot_v2 as module

ROOT = module.ROOT


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def toy_dataset(n_features=3, seed=20260914):
    rng = np.random.default_rng(seed)
    labels = ["SELF"] * 80 + ["OTHER"] * 80 + ["NONTERMINATION"] * 80 + ["ORDINARY"] * 80
    splits = ["TRAIN"] * 240 + ["VALIDATION"] * 80
    folds = [index % 5 for index in range(240)] + [None] * 80
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
    def __init__(self, C):
        self.C = C
        self.n_iter_ = np.asarray([3])
        self.classes_ = np.asarray([0, 1])
        self.coef_ = np.zeros((1, 1))
        self.intercept_ = np.asarray([0.0])

    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y)
        positive = x[y == 1].mean(axis=0)
        negative = x[y == 0].mean(axis=0)
        self.coef_ = (positive - negative).reshape(1, -1)
        return self

    def decision_function(self, x):
        return np.asarray(x, float) @ self.coef_.ravel() + self.intercept_[0]

    def predict_proba(self, x):
        positive = 1.0 / (1.0 + np.exp(-self.decision_function(x)))
        return np.column_stack([1.0 - positive, positive])


class ToyFactory:
    def __call__(self, model, C):
        return ToyEstimator(C)


def write_export(directory, *, lens_sha=None, array_keys=("J6", "J10", "J18"), corrupt=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {name: np.eye(io.D_MODEL, dtype=np.float32) for name in array_keys}
    if corrupt == "shape":
        payload[array_keys[0]] = np.zeros((2, 2), dtype=np.float32)
    if corrupt == "nonfinite":
        payload[array_keys[0]] = np.full((io.D_MODEL, io.D_MODEL), np.nan, dtype=np.float32)
    buffer = _io.BytesIO()
    np.savez(buffer, **payload)
    raw = buffer.getvalue()
    npz_path = directory / "lens_jacobians.npz"
    npz_path.write_bytes(raw)
    receipt = {
        "schema": "jlens_lens_export.v1",
        "release": io.RELEASE_V1,
        "lens": {
            "repo": io.LENS_PIN["repo"],
            "revision": io.LENS_PIN["revision"],
            "filename": io.LENS_PIN["filename"],
            "bytes": io.LENS_PIN["bytes"],
            "sha256": lens_sha or io.LENS_PIN["sha256"],
        },
        "layers": list(io.BLOCKS),
        "arrays": {name: {"shape": [io.D_MODEL, io.D_MODEL], "dtype": "float32"} for name in array_keys},
        "export": {"filename": "lens_jacobians.npz", "bytes": len(raw), "sha256": sha(raw)},
        "counters": {"model_loads": 0, "tokenizer_loads": 0, "forwards": 0, "derivatives": 0, "fits": 0},
    }
    receipt_path = directory / "lens_export_receipt.json"
    receipt_raw = module.runner.encoded(receipt)
    receipt_path.write_bytes(receipt_raw)
    return npz_path, receipt_path, raw, receipt_raw


def export_entry(root, npz_path, receipt_path, raw, receipt_raw):
    study = module.STUDY.as_posix() + "/"
    return {
        "path": Path(npz_path).relative_to(root).as_posix(),
        "bytes": len(raw),
        "sha256": sha(raw),
        "receipt": {
            "path": Path(receipt_path).relative_to(root).as_posix(),
            "bytes": len(receipt_raw),
            "sha256": sha(receipt_raw),
        },
    }


class ConstantsTests(unittest.TestCase):
    def test_schema_and_inputs(self):
        self.assertEqual(module.SCHEMA, "jlens_pilot_execution.v2")
        self.assertEqual(set(module.INPUT_ROLES), {
            "lens", "lens_export", "model_snapshot_lock", "unprompted_cache",
            "prompted_cache", "manifests", "surfaces",
        })
        self.assertEqual(module.CAPS, pilot.CAPS)

    def test_v2_runner_is_model_free_at_import(self):
        tree = ast.parse(inspect.getsource(module))
        roots = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                roots.add((node.module or "").split(".")[0])
        self.assertFalse(roots & module.FORBIDDEN_ROOTS)
        self.assertNotIn("torch", sys.modules)


class LensExportJacobianTests(unittest.TestCase):
    def test_reads_three_float32_matrices(self):
        with tempfile.TemporaryDirectory(prefix="jlens_npz_") as temporary:
            npz_path, _, _, _ = write_export(Path(temporary))
            jacobians = module.lens_export_jacobians(str(npz_path))
            self.assertEqual(set(jacobians), set(io.BLOCKS))
            for matrix in jacobians.values():
                self.assertEqual(matrix.shape, (io.D_MODEL, io.D_MODEL))
                self.assertEqual(matrix.dtype, np.float32)

    def test_rejects_wrong_keys_shape_and_nonfinite(self):
        with tempfile.TemporaryDirectory(prefix="jlens_npz_bad_") as temporary:
            npz_path, _, _, _ = write_export(Path(temporary) / "keys", array_keys=("J6", "J10"))
            with self.assertRaisesRegex(module.GateError, "EXPORT_KEYS"):
                module.lens_export_jacobians(str(npz_path))
            npz_path, _, _, _ = write_export(Path(temporary) / "shape", corrupt="shape")
            with self.assertRaisesRegex(module.GateError, "JACOBIAN_SHAPE"):
                module.lens_export_jacobians(str(npz_path))
            npz_path, _, _, _ = write_export(Path(temporary) / "nan", corrupt="nonfinite")
            with self.assertRaisesRegex(module.GateError, "JACOBIAN_NONFINITE"):
                module.lens_export_jacobians(str(npz_path))


class LensExportPinTests(unittest.TestCase):
    def _root(self, temporary):
        root = Path(temporary)
        npz_path, receipt_path, raw, receipt_raw = write_export(
            root / module.STUDY / "runs" / "lens_export_test")
        entry = export_entry(root, npz_path, receipt_path, raw, receipt_raw)
        return root, npz_path, receipt_path, raw, receipt_raw, entry

    def test_accepts_bound_export(self):
        with tempfile.TemporaryDirectory(prefix="jlens_pin_") as temporary:
            root, _, _, _, _, entry = self._root(temporary)
            pin, path, receipt = module._lens_export({"lens_export": entry}, root)
            self.assertEqual(pin["sha256"], entry["sha256"])
            self.assertEqual(receipt["lens"]["sha256"], io.LENS_PIN["sha256"])

    def test_rejects_wrong_lens_binding_and_wrong_export_hash(self):
        with tempfile.TemporaryDirectory(prefix="jlens_pin_bad_") as temporary:
            root, _, _, _, _, entry = self._root(temporary)
            tampered = dict(entry, sha256="0" * 64)
            with self.assertRaises(module.GateError):
                module._lens_export({"lens_export": tampered}, root)
        with tempfile.TemporaryDirectory(prefix="jlens_pin_lens_") as temporary:
            root = Path(temporary)
            npz_path, receipt_path, raw, receipt_raw = write_export(
                root / module.STUDY / "runs" / "lens_export_test", lens_sha="1" * 64)
            entry = export_entry(root, npz_path, receipt_path, raw, receipt_raw)
            with self.assertRaisesRegex(module.GateError, "LENS_EXPORT_LENS_BINDING"):
                module._lens_export({"lens_export": entry}, root)


class SurfaceReceiptTests(unittest.TestCase):
    def test_surface_receipt_matches_parity_v4_evidence_and_frozen_order(self):
        receipt_path = ROOT / module.STUDY / "JLENS_SURFACE_TOKEN_RECEIPT_V1.json"
        receipt = json.loads(receipt_path.read_bytes())
        parity_path = ROOT / module.STUDY / "runs" / "jlens_parity_20260914_v4" / "parity_receipt.json"
        parity_raw = parity_path.read_bytes()
        self.assertEqual(sha(parity_raw), receipt["evidence"]["parity_receipt"]["sha256"])
        parity = json.loads(parity_raw)
        self.assertEqual(parity["status"], "parity_complete")
        ids = [item["token_id"] for item in receipt["surfaces"]]
        self.assertEqual(ids, parity["selected_token_ids"])
        self.assertEqual([item["surface"] for item in receipt["surfaces"]], list(core.CONCEPT_SURFACES_V1))
        self.assertTrue(all(item["single_token"] is True for item in receipt["surfaces"]))
        retained = pilot.validate_surfaces(
            [{"surface": item["surface"], "token_id": item["token_id"], "single_token": True}
             for item in receipt["surfaces"]]
        )
        self.assertEqual(len(retained), 6)

    def test_handoff_pilot_hashes_are_preserved(self):
        import subprocess
        unchanged = {
            "run_jlens_pilot_v1.py": "244b5ff838a72a6091dd7fbabf42bd5c75d82984209252cbee90de6018967e5f",
            "test_jlens_pilot_v1.py": "690bde84a9eb7b990ac6b638c50b863249bfe126fe691cbf4d7340ea8252f36d",
            "JLENS_PILOT_PLAN_V1.json": "11445088a5cb3b7824a3eff414fbb067a36bef398f7272cbcb0792b5483a9680",
        }
        for name, digest in unchanged.items():
            self.assertEqual(sha((ROOT / module.STUDY / name).read_bytes()), digest, name)
        # jlens_pilot_v1.py was extended additively for per-split reporting; its
        # original handoff bytes remain reachable at the integration base commit.
        base = "f465a793ecd607e12d09aa34e138791c38da8622"
        blob = subprocess.run(
            ["git", "-C", str(ROOT), "cat-file", "blob",
             base + ":development/jlens_trigger_v1/jlens_pilot_v1.py"],
            capture_output=True, check=True,
        ).stdout
        self.assertEqual(sha(blob), "94ce54256d1766fd1d0edb2267944654a705c64c868c0af55f873c2e2a0c3e46")


class CaptureSmokeTests(unittest.TestCase):
    def test_capture_runs_without_torch_through_injected_scores(self):
        with tempfile.TemporaryDirectory(prefix="jlens_v2_capture_") as temporary:
            root = Path(temporary)
            case_ids, labels, splits, folds, scores = toy_dataset(n_features=3)
            surfaces = [
                {"surface": name, "token_id": index + 1, "single_token": True}
                for index, name in enumerate(pilot.CONCEPT_SURFACES[:3])
            ]
            ctx = {
                "lock": {"caps": dict(pilot.CAPS), "run_id": "jlens_pilot_v2_toy", "release": io.RELEASE_V1},
                "lock_sha256": "e" * 64,
                "root": root,
                "data": {
                    "case_order": case_ids,
                    "labels": labels,
                    "splits": splits,
                    "folds": folds,
                    "surfaces": [(item["surface"], item["token_id"]) for item in surfaces],
                    "lens_export": {"pin": {"sha256": "a" * 64}},
                },
                "output": root / "runs" / "jlens_pilot_v2_toy",
            }
            pins = module.capture(
                ctx, time.monotonic(),
                verify=lambda _ctx: {"snapshot_realpath": str(root / "snap"), "checked_files": []},
                scores=scores, factory=ToyFactory(),
            )
            self.assertEqual(set(pins), set(module.ARTIFACTS))
            receipt = json.loads((ctx["output"] / "pilot_receipt.json").read_bytes())
            self.assertEqual(receipt["status"], "pilot_complete")
            self.assertEqual(receipt["fits"]["total"], 132)
            self.assertIs(receipt["notes"]["torch_imported_in_phase_b"], False)
            self.assertIs(receipt["notes"]["lens_loaded_from_pinned_phase_a_export"], True)
            self.assertEqual(receipt["lens_export_sha256"], "a" * 64)
            self.assertNotIn("torch", sys.modules)


class ValidationSplitTests(unittest.TestCase):
    def test_split_metrics_cover_original40_added40_combined80(self):
        case_ids, labels, splits, folds, scores = toy_dataset(n_features=3)
        groups = ["original40"] * 40 + ["added40"] * 40
        result = pilot.run_pilot(
            scores=scores, case_ids=case_ids, labels=labels, splits=splits, folds=folds,
            surface_names=list(pilot.CONCEPT_SURFACES[:3]), factory=ToyFactory(),
            validation_groups=groups,
        )
        counts = {"original40": 40, "added40": 40, "combined80": 80}
        cell_a = [cell for cell in result["cells"] if cell["scheme"] == "cell_A" and cell["status"] == "VALID"]
        self.assertTrue(cell_a)
        for cell in cell_a:
            split = cell["validation_split_metrics"]
            self.assertEqual(set(split), set(counts))
            for name, total in counts.items():
                metrics = split[name]
                self.assertEqual(metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"], total)
        valid_refits = [refit for refit in result["refits"] if refit["status"] == "VALID"]
        self.assertTrue(valid_refits)
        for refit in valid_refits:
            self.assertEqual(set(refit["validation_split_metrics"]), set(counts))

    def test_grouping_is_reporting_only_and_does_not_change_selection(self):
        case_ids, labels, splits, folds, scores = toy_dataset(n_features=3)
        groups = ["original40"] * 40 + ["added40"] * 40
        plain = pilot.run_pilot(
            scores=scores, case_ids=case_ids, labels=labels, splits=splits, folds=folds,
            surface_names=list(pilot.CONCEPT_SURFACES[:3]), factory=ToyFactory(),
        )
        grouped = pilot.run_pilot(
            scores=scores, case_ids=case_ids, labels=labels, splits=splits, folds=folds,
            surface_names=list(pilot.CONCEPT_SURFACES[:3]), factory=ToyFactory(),
            validation_groups=groups,
        )
        self.assertEqual([cell["selection_key"] for cell in plain["cells"]],
                         [cell["selection_key"] for cell in grouped["cells"]])
        self.assertEqual([(refit["C"], refit["tau"]) for refit in plain["refits"]],
                         [(refit["C"], refit["tau"]) for refit in grouped["refits"]])
        self.assertIsNone(plain["cells"][0]["validation_split_metrics"])
        self.assertIsNotNone(grouped["cells"][0]["validation_split_metrics"])


if __name__ == "__main__":
    unittest.main()
