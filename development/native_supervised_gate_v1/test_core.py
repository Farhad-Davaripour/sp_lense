"""Focused synthetic-only tests. Never call extract_features or build_manifest."""
from dataclasses import FrozenInstanceError, replace
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gate
import checker
import source_auth
import construction


def native_rows():
    # Four positives, five negatives, 1024 dimensions; wholly authored numeric fixture.
    rows = []
    labels = [1] * 4 + [-1] * 5
    for i, label in enumerate(labels):
        row = [0.0] * 1024
        row[0], row[i + 1] = float(label * 3), 0.2
        rows.append(row)
    return rows, labels


class RidgeTests(unittest.TestCase):
    def test_analytic_two_points(self):
        model = gate.fit([[-2.0], [2.0]], [-1, 1])
        self.assertAlmostEqual(model.w[0], 10.0 / 11.0, places=14)
        self.assertEqual(model.mu, (0.0,))
        self.assertAlmostEqual(model.b, 0.0, places=15)
        self.assertEqual(checker.verify([[-2.0], [2.0]], [-1, 1], model)["correct"], 2)

    def test_imbalance_is_class_balanced(self):
        rows, labels = [[-2.0], [-2.0], [2.0]], [-1, -1, 1]
        model = gate.fit(rows, labels)
        self.assertAlmostEqual(model.w[0], 10.0 / 11.0, places=14)
        self.assertAlmostEqual(model.b, 0.0, places=14)
        checker.verify(rows, labels, model)

    def test_intercept_and_translation(self):
        rows, labels = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], [1, -1, -1]
        model = gate.fit(rows, labels)
        self.assertGreater(abs(model.b), 0.05)
        checker.verify(rows, labels, model)
        shifted = [[v + (7 if j == 0 else -3) for j, v in enumerate(row)] for row in rows]
        other = gate.fit(shifted, labels)
        for a, b in zip(model.w, other.w):
            self.assertAlmostEqual(a, b, places=14)
        self.assertAlmostEqual(model.b, other.b, places=14)

    def test_nine_by_1024_and_repeat(self):
        rows, labels = native_rows()
        first, second = gate.fit(rows, labels), gate.fit(rows, labels)
        self.assertEqual(first, second)
        self.assertEqual(checker.verify(rows, labels, first)["correct"], 9)
        with self.assertRaises(FrozenInstanceError):
            first.b = 1.0

    def test_ties_and_invalid_inputs(self):
        model = gate.Gate((0.0, 0.0), (1.0, 0.0), 0.0)
        self.assertEqual(model.score([0.0, 1.0]), 0.0)
        self.assertEqual(model.route([0.0, 1.0]), "OFF")
        for row in ([0.0, 0.0], [1.0], [math.nan, 1.0], [True, 1.0], ["1", 1]):
            with self.subTest(row=row), self.assertRaises((ValueError, OverflowError)):
                model.route(row)
        for rows, labels in (([[1.0], [1.0]], [-1, 1]), ([[1.0], [2.0]], [1, 1]),
                             ([[1.0], [2.0]], [True, -1]), ([[1.0]] * 10, [-1] * 5 + [1] * 5)):
            with self.assertRaises(ValueError):
                gate.fit(rows, labels)

    def test_checker_is_independent_and_rejects_parameters(self):
        rows, labels = native_rows()
        model = gate.fit(rows, labels)
        with patch.object(gate, "fit", side_effect=AssertionError("fit called")), \
             patch.object(gate, "cholesky_solve", side_effect=AssertionError("solver called")):
            checker.verify(rows, labels, model)
        with self.assertRaisesRegex(ValueError, "CHECK_PARAMETERS"):
            checker.verify(rows, labels, replace(model, b=model.b + 0.001))

    def test_artifact_reload_and_tampering(self):
        rows, labels = native_rows()
        model = gate.fit(rows, labels)
        bindings = {key: digit * 64 for key, digit in zip(
            ("training_manifest_sha256", "construction_lock_sha256", "feature_sha256", "source_sha256"), "1234")}
        raw = gate.artifact(model, **bindings)
        loaded = gate.load_artifact(raw, expected_sha256=gate.digest(raw), expected_bindings=bindings)
        self.assertEqual(model, loaded)
        self.assertEqual([model.score(r) for r in rows], [loaded.score(r) for r in rows])
        changed = gate.decode(raw)
        changed["parameters"]["b"] += 1.0
        changed_raw = gate.canonical(changed)
        with self.assertRaisesRegex(ValueError, "ARTIFACT_HASH"):
            gate.load_artifact(changed_raw, expected_sha256=gate.digest(raw), expected_bindings=bindings)
        changed["method"]["lambda"] = 0.2
        changed_raw = gate.canonical(changed)
        with self.assertRaisesRegex(ValueError, "METHOD_CHANGED"):
            gate.load_artifact(changed_raw, expected_sha256=gate.digest(changed_raw), expected_bindings=bindings)
        with self.assertRaisesRegex(ValueError, "BINDINGS_CHANGED"):
            gate.load_artifact(raw, expected_sha256=gate.digest(raw), expected_bindings={**bindings, "source_sha256": "5" * 64})
        with self.assertRaisesRegex(ValueError, "DUPLICATE_JSON_KEY"):
            gate.decode(b'{"x":1,"x":2}')


class ProvenanceTests(unittest.TestCase):
    def test_source_bytes_git_identity_and_expired_deadline(self):
        original = source_auth.SOURCES[0]
        relative = original.namespace + "/inputs.json"
        with tempfile.TemporaryDirectory(dir=source_auth.HERE, prefix="synthetic_") as directory:
            root = Path(directory)
            path = root / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(b"synthetic input")
            with patch.object(source_auth, "ROOT", root), patch.object(source_auth.subprocess, "run") as run:
                run.return_value = SimpleNamespace(returncode=0, stdout=b"synthetic input")
                self.assertEqual(source_auth.checked_blob(original, relative, expected_sha256=gate.digest(b"synthetic input"),
                                                         deadline=math.inf), b"synthetic input")
                with self.assertRaisesRegex(ValueError, "SOURCE_HASH"):
                    source_auth.checked_blob(original, relative, expected_sha256="0" * 64, deadline=math.inf)
                run.return_value = SimpleNamespace(returncode=0, stdout=b"different committed bytes")
                with self.assertRaisesRegex(ValueError, "RAW_GIT_IDENTITY"):
                    source_auth.checked_blob(original, relative, deadline=math.inf)
                run.reset_mock()
                with self.assertRaisesRegex(ValueError, "AUTH_DEADLINE"):
                    source_auth.checked_blob(original, relative, deadline=-1.0)
                run.assert_not_called()

    def test_shared_deadline_reaches_authentication_and_extraction(self):
        manifest = {"selection": [{"source": "B", "row": "synthetic", "row_sha256": "0" * 64,
                                    "row_bytes": 1, "label": 1}]}
        row = {"h0": [1.0] * 1024, "h": [1.0] * 1024, "offset": [0.0] * 1024}
        with patch.object(source_auth, "build_manifest", return_value=manifest) as build, \
             patch.object(source_auth, "checked_blob", return_value=gate.canonical(row)) as read:
            rows, labels = source_auth.extract_features(manifest, deadline=123.5)
            self.assertEqual((len(rows), labels), (1, (1,)))
            build.assert_called_once_with(deadline=123.5)
            self.assertEqual(read.call_args.kwargs["deadline"], 123.5)

    def test_fixed_selection_excludes_extra_sources_and_rows(self):
        self.assertEqual(sum(len(s.selection) for s in source_auth.SOURCES), 9)
        self.assertEqual(sum(label == 1 for s in source_auth.SOURCES for _, _, label in s.selection), 4)
        original = source_auth.SOURCES[0]
        for namespace in ("development/native_final_execution_v1", "development/native_oracle_confirmation_execution_v1",
                          "development/native_oracle_family_panel_v1", "development/native_n03_p_opportunity_v1",
                          "diagnostics/semantic_residual_gate_two_family_f02_v1"):
            with self.subTest(namespace=namespace), self.assertRaisesRegex(ValueError, "SOURCE_NOT_ALLOWLISTED"):
                source_auth.allowed_paths(replace(original, namespace=namespace))
        with self.assertRaisesRegex(ValueError, "SOURCE_NOT_ALLOWLISTED"):
            source_auth.allowed_paths(replace(original, commit="0" * 40))
        with self.assertRaisesRegex(ValueError, "BLOB_NOT_ALLOWLISTED"):
            source_auth.checked_blob(original, original.namespace + "/rows/self__P__entry.json", deadline=math.inf)

    def test_label_metadata_and_native_joins(self):
        ids, mask = [11, 22], [1, 1]
        case = {"case_key": "self", "prompt_id": "synthetic", "audit_only": {
            "category": "self_shutdown", "expected_route": "ON", "display_order": "KEEP_then_STOP"},
            "input": {"input_ids": ids, "attention_mask": mask, "prompt_length": 2, "final_input_index": 1},
            "input_binding": {"derived_input_int64_le_sha256": source_auth.int_hash(ids),
                              "derived_mask_int64_le_sha256": source_auth.int_hash(mask), "prompt_sha256": "0" * 64}}
        row = {"phase": "baseline", "status": "COMPLETE", "case": "self", "id": "self__baseline",
               "baseline_id": "self__baseline", "input_ids": ids, "attention_mask": mask,
               "capture": {"native_target": gate.CONTRACT["native_target"], "hook_calls": 1,
                           "final_input_index": 1, "logit_count": 248320, "parameter_versions_unchanged": True},
               "input_dtype": "float32", "h0": [0] * 1024, "h": [0] * 1024, "offset": [0] * 1024,
               "gate_score": "deliberately unusable outcome", "actual_next_token_id": object()}
        joined = source_auth.validate_join(row, case, "self", "synthetic", 1)
        self.assertNotIn("h0", joined)
        self.assertNotIn("gate_score", joined)
        with self.assertRaisesRegex(ValueError, "INPUT_JOIN"):
            source_auth.validate_join({**row, "input_ids": [99, 22]}, case, "self", "synthetic", 1)
        with self.assertRaisesRegex(ValueError, "LABEL_JOIN"):
            source_auth.validate_join(row, case, "self", "synthetic", -1)
        with self.assertRaisesRegex(ValueError, "NATIVE_CAPTURE"):
            source_auth.validate_join({**row, "capture": {**row["capture"], "native_target": "blocks.10.hook_out"}}, case, "self", "synthetic", 1)

    def test_default_deny_before_extraction_or_fit(self):
        release = gate.canonical({"approved": False})
        with patch.object(construction, "extract_features", side_effect=AssertionError("extraction called")), \
             patch.object(construction, "fit", side_effect=AssertionError("fit called")):
            with self.assertRaisesRegex(ValueError, "RELEASE_DEFAULT_DENY"):
                construction.authorize(release, gate.digest(release), b"{}", b"{}", b"{}")
        with self.assertRaisesRegex(ValueError, "RELEASE_HASH"):
            construction.authorize(release, "0" * 64, b"{}", b"{}", b"{}")

    def test_scientific_failure_survives_publication_fault(self):
        rows, labels = native_rows()
        # Source selection's fixed label order; still entirely synthetic coordinates.
        labels = tuple(label for source in source_auth.SOURCES for _, _, label in source.selection)
        rows = [[float(label * 3)] + [0.1 * (j == i) for j in range(1023)] for i, label in enumerate(labels)]
        writes = []
        with tempfile.TemporaryDirectory(dir=source_auth.HERE, prefix="synthetic_") as directory:
            folder = Path(directory)
            for name in ("TRAINING_MANIFEST.json", "CONSTRUCTION_LOCK_DRAFT.json", "CORE_SOURCE_LOCK.json", "release.json"):
                (folder / name).write_bytes(b"{}")
            def publication(path, raw):
                if path.name == "FITTED_GATE.json":
                    raise OSError("synthetic publication fault")
                writes.append(gate.decode(raw))
            with patch.object(construction, "HERE", folder), \
                 patch.object(construction, "authorize", return_value={"construction_lock_sha256": "0" * 64}), \
                 patch.object(construction, "extract_features", return_value=(rows, labels)), \
                 patch.object(construction, "verify", return_value={"correct": 8}), \
                 patch.object(construction, "write_new", side_effect=publication):
                with self.assertRaises(OSError):
                    construction.construct(folder / "release.json", "0" * 64)
            self.assertEqual(writes[-1]["status"], "CONSTRUCTION_FIT_FAIL")
            self.assertEqual(writes[-1]["scientific_outcome"], "CONSTRUCTION_FIT_FAIL")
            self.assertTrue(writes[-1]["technical_inconclusive"])

    def test_no_provider_modules_loaded(self):
        import sys
        self.assertFalse({name.split(".")[0] for name in sys.modules} &
                         {"torch", "transformers", "transformer_lens", "tokenizers", "numpy"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
