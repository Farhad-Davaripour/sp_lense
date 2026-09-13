"""Tests for the accepted-checkpoint reuse gate and its wiring into ``load_model``.

Model-free and history-free: the real cases read only frozen metadata and the already
saved classifier coefficients, while the negative cases use synthetic in-memory metadata
fixtures under a temporary root. No Qwen/model/tokenizer/provider import occurs, no
activation or diagnostic row is read, nothing is fitted and no real scoring is run.

Mocks are confined to tests (module path/hash globals and an import/open tripwire);
production modules are never monkeypatched.
"""
import builtins, importlib, io, json, sys, tempfile, unittest, unittest.mock
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diagnostic_reader as reader
import diagnostic_scoring as scoring

SOURCE_FILES = ("gate.py", "checker.py", "source_auth.py", "construction.py")


def _write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _doc(path, value):
    return _write(path, reader.canonical(value))


def build_fixture(root, *, artifact=b"SYNTHETIC_PINNED_GATE_ARTIFACT\n", data_lock=b'{"input_data_lock":1}',
                  freeze_update=None, release_update=None, result_update=None, review_update=None,
                  terminal_update=None, final_update=None, bindings_update=None):
    """Write one fully self-consistent synthetic checkpoint tree and return its pins."""
    root = Path(root)
    fit = root / "fit"
    _write(root / "train_capture/DATA_LOCK.json", data_lock)
    _write(fit / "construction_attempt_001/PRECHOICE29_GATE.json", artifact)
    artifact_sha = reader.sha(artifact)
    sources = {name: reader.sha(("SOURCE:" + name).encode("ascii")) for name in SOURCE_FILES}
    for name in SOURCE_FILES:
        _write(fit / name, ("SOURCE:" + name).encode("ascii"))
    source_sha = reader.sha(_write(fit / "CORE_SOURCE_LOCK.json", reader.canonical(sources)))
    manifest_sha = reader.sha(_write(fit / "TRAINING_MANIFEST.json", b'{"manifest":1}'))
    contract_sha = reader.sha(_write(fit / "CONSTRUCTION_LOCK_DRAFT.json", b'{"lock":1}'))
    release_doc = {"approved": True, "operation": "one_construction_fit", "source_sha256": source_sha,
                   "training_manifest_sha256": manifest_sha, "construction_lock_sha256": contract_sha,
                   "model_permission": False, "tokenizer_permission": False, "evaluation_permission": False}
    release_doc.update(release_update or {})
    release_sha = reader.sha(_doc(fit / "RELEASE.json", release_doc))
    result_doc = {"scientific_pass": True, "status": "PRECHOICE29_TRAINING_PASS", "fits_attempted": 3,
                  "stages": [{"id": "PRECHOICE29", "training_count": 29, "training_correct": 29,
                              "heads": [{"status": "PASS"}] * 3}],
                  "artifacts": {"PRECHOICE29": artifact_sha}}
    result_doc.update(result_update or {})
    result_sha = reader.sha(_doc(fit / "construction_attempt_001/RESULT.json", result_doc))
    terminal_doc = {"process_technical_complete": True, "exit_code": 0}
    terminal_doc.update(terminal_update or {})
    terminal_sha = reader.sha(_doc(fit / "fit_owner_attempt_001/TERMINAL.json", terminal_doc))
    final_doc = {"terminal_sha256": terminal_sha, "deadline_fault": False, "storage_fault": False}
    final_doc.update(final_update or {})
    final_sha = reader.sha(_doc(fit / "fit_owner_attempt_001/FINALIZATION.json", final_doc))
    review_doc = {"schema": "prechoice_fit_independent_review.v1", "status": "PASS",
                  "artifact_sha256": artifact_sha, "result_sha256": result_sha, "source_sha256": source_sha,
                  "release_sha256": release_sha, "all_three_certificates_verified": True,
                  "all29_training_correct": True, "exact_artifact_reload": True}
    review_doc.update(review_update or {})
    review_sha = reader.sha(_doc(fit / "INDEPENDENT_ACTUAL_FIT_REVIEW.json", review_doc))
    bindings = {"input_contract_sha256": manifest_sha, "construction_lock_sha256": contract_sha,
                "source_sha256": source_sha, "retained_input_data_lock_sha256": reader.sha(data_lock),
                "development_only": True, "arm": "PRECHOICE29"}
    bindings.update(bindings_update or {})
    freeze_doc = {"schema": "prechoice29_accepted_artifact.v1", "approved": True, "arm": "PRECHOICE29",
                  "diagnostic_inputs_used": False, "diagnostic_execution_permission": False,
                  "artifact_sha256": artifact_sha, "result_sha256": result_sha, "source_sha256": source_sha,
                  "release_sha256": release_sha, "owner_terminal_sha256": terminal_sha,
                  "owner_finalization_sha256": final_sha, "independent_review_sha256": review_sha,
                  "artifact_bindings": bindings}
    freeze_doc.update(freeze_update or {})
    freeze_sha = reader.sha(_doc(fit / "FROZEN_PRECHOICE_ARTIFACT.json", freeze_doc))
    return {"fit": fit, "freeze_sha": freeze_sha, "artifact_sha": artifact_sha, "review_sha": review_sha}


class AcceptedCheckpointReuseTests(unittest.TestCase):
    def test_01_accepted_checkpoint_reuse_from_the_relocated_root(self):
        record = reader.verify_accepted_checkpoint()
        self.assertEqual(record["schema"], reader.CHECKPOINT_REUSE_SCHEMA)
        self.assertEqual(record["status"], "REUSED_ALREADY_ACCEPTED_CHECKPOINT")
        self.assertEqual(record["accepted_checkpoint_sha256"], reader.ACCEPTED_CHECKPOINT_SHA256)
        self.assertEqual(record["independent_review_sha256"], reader.INDEPENDENT_REVIEW_SHA256)
        self.assertEqual(record["freeze_sha256"], reader.FREEZE_SHA256)
        self.assertEqual(record["retained_input_data_lock_sha256"],
                         "f1e5bc36099239506ef77d7e50666388c4ccb916eeed0aa21149cdd1f6c45e27")
        self.assertIs(record["historical_training_replayed"], False)
        self.assertIs(record["independent_reproduction"], False)
        self.assertIs(record["reuse_of_accepted_checkpoint"], True)
        for permission in ("model_permission", "tokenizer_permission", "evaluation_permission",
                           "diagnostic_execution_permission"):
            self.assertIs(record[permission], False)
        self.assertEqual(reader.FIT, reader.ROOT / "development/native_gate_prechoice_readout_v1/fit")
        self.assertTrue(str(reader.ROOT).endswith("SP_lens"))
        self.assertFalse(reader.ROOT.joinpath("OneDrive").exists())

    def test_02_real_coefficients_load_without_provider_or_activation_rows(self):
        for name in ("torch", "transformers", "tokenizers", "safetensors", "numpy"):
            self.assertNotIn(name, sys.modules)
        model = scoring.load_model()
        self.assertEqual(len(model.mu), scoring.WIDTH)
        self.assertEqual(len(model.heads), len(scoring.HEADS))
        raw = json.loads((scoring.FIT / "construction_attempt_001/PRECHOICE29_GATE.json").read_bytes())
        self.assertEqual(reader.sha(reader.read_bytes(scoring.ARTIFACT, "ARTIFACT")),
                         reader.ACCEPTED_CHECKPOINT_SHA256)
        self.assertEqual(model.mu, tuple(raw["parameters"]["mu"]))
        self.assertEqual([(head[0], head[1]) for head in model.heads],
                         [(tuple(saved["w"]), float(saved["b"])) for saved in raw["parameters"]["heads"]])

    def test_03_retired_root_and_historical_entrypoints_are_never_reached(self):
        calls, imported = [], []
        real_import_module, real_open = importlib.import_module, io.open
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in ("torch", "transformers", "tokenizers", "safetensors", "numpy"):
                raise AssertionError("PROVIDER_IMPORT:" + name)
            return real_import(name, *args, **kwargs)

        def trip(name):
            def called(*_args, **_kwargs):
                calls.append(name)
                raise AssertionError("HISTORICAL_ENTRYPOINT_CALLED:" + name)
            return called

        def spy_import(name, *args, **kwargs):
            imported.append(name)
            module = real_import_module(name, *args, **kwargs)
            if name == "construction":
                for attr in ("authorize", "construct", "run"):
                    if hasattr(module, attr):
                        setattr(module, attr, trip("construction." + attr))
                source_auth = getattr(module, "source_auth", None)
                if source_auth is not None:
                    for attr in ("load_saved", "extract_features"):
                        if hasattr(source_auth, attr):
                            setattr(source_auth, attr, trip("source_auth." + attr))
            return module

        def guarded_open(file, *args, **kwargs):
            if "onedrive" in str(file).lower():
                calls.append("retired_root:" + str(file))
                raise AssertionError("RETIRED_ROOT_READ")
            if any(part in str(file).replace("\\", "/").split("/") for part in ("real_evidence", "rows", "logits")):
                raise AssertionError("ACTIVATION_OR_DIAGNOSTIC_READ")
            return real_open(file, *args, **kwargs)

        with unittest.mock.patch.object(importlib, "import_module", side_effect=spy_import), \
                unittest.mock.patch("io.open", side_effect=guarded_open), \
                unittest.mock.patch("builtins.__import__", new=guarded_import):
            record = reader.verify_accepted_checkpoint()
            self.assertEqual(record["status"], "REUSED_ALREADY_ACCEPTED_CHECKPOINT")
            self.assertNotIn("construction", imported)
            model = scoring.load_model()
        self.assertEqual(calls, [])
        self.assertIn("construction", imported)
        self.assertEqual(len(model.heads), len(scoring.HEADS))

    def test_04_loader_consumes_verified_bindings_without_reopening_freeze(self):
        record = reader.verify_accepted_checkpoint()
        original = scoring._bounded_file

        def checked_read(path, *args, **kwargs):
            self.assertNotEqual(path, scoring.FROZEN)
            return original(path, *args, **kwargs)

        with unittest.mock.patch.object(reader, "verify_accepted_checkpoint", return_value=record) as verified, \
                unittest.mock.patch.object(scoring, "_bounded_file", side_effect=checked_read):
            model = scoring.load_model()
        verified.assert_called_once_with()
        self.assertEqual(len(model.heads), 3)

    def test_05_failed_shared_verifier_prevents_decoder_import(self):
        with unittest.mock.patch.object(reader, "verify_accepted_checkpoint", side_effect=ValueError("DENIED")), \
                unittest.mock.patch.object(importlib, "import_module") as importer:
            with self.assertRaisesRegex(ValueError, "DENIED"):
                scoring.load_model()
        importer.assert_not_called()


class CheckpointFixtureTests(unittest.TestCase):
    def _run(self, code, *, build=None, mutate=None):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = build_fixture(tmp, **(build or {}))
            if mutate is not None:
                mutate(fixture["fit"])
            with unittest.mock.patch.object(reader, "FREEZE_SHA256", fixture["freeze_sha"]), \
                    unittest.mock.patch.object(reader, "ACCEPTED_CHECKPOINT_SHA256", fixture["artifact_sha"]), \
                    unittest.mock.patch.object(reader, "INDEPENDENT_REVIEW_SHA256", fixture["review_sha"]):
                if code is None:
                    return reader.verify_accepted_checkpoint(fit=fixture["fit"])
                with self.assertRaises(ValueError) as caught:
                    reader.verify_accepted_checkpoint(fit=fixture["fit"])
                self.assertEqual(str(caught.exception), code)

    def test_01_mocked_fixture_control_passes_and_reports_reuse(self):
        record = self._run(None)
        self.assertEqual(record["status"], "REUSED_ALREADY_ACCEPTED_CHECKPOINT")
        self.assertIs(record["historical_training_replayed"], False)

    def test_02_corrupted_frozen_bytes_fail_closed(self):
        cases = (
            ("freeze", "FROZEN_FILE_HASH", None,
             lambda fit: (fit / "FROZEN_PRECHOICE_ARTIFACT.json").write_bytes(b" ")),
            ("artifact", "FROZEN_ARTIFACT_HASH", None,
             lambda fit: (fit / "construction_attempt_001/PRECHOICE29_GATE.json").write_bytes(b"tampered")),
            ("review", "FROZEN_FILE_HASH", None,
             lambda fit: (fit / "INDEPENDENT_ACTUAL_FIT_REVIEW.json").write_bytes(b"tampered")),
            ("source", "EXACT_FIT_SOURCE_HASH", None, lambda fit: (fit / "gate.py").write_bytes(b"tampered")),
        )
        for name, code, build, mutate in cases:
            with self.subTest(case=name):
                self._run(code, build=build, mutate=mutate)

    def test_03_failed_acceptance_flags_and_joins_fail_closed(self):
        cases = (
            ("approved", "NEW_TRAINING_ARTIFACT_ONLY", {"freeze_update": {"approved": False}}, None),
            ("training_only", "NEW_TRAINING_ARTIFACT_ONLY",
             {"freeze_update": {"diagnostic_inputs_used": True}}, None),
            ("release_permission", "EXACT_TRAINING_ONLY_RELEASE",
             {"release_update": {"model_permission": True}}, None),
            ("review_status", "INDEPENDENT_CERTIFICATION_REQUIRED",
             {"review_update": {"status": "FAIL"}}, None),
            ("certificates", "INDEPENDENT_CERTIFICATION_REQUIRED",
             {"review_update": {"all_three_certificates_verified": False}}, None),
            ("result_artifact_join", "RESULT_ARTIFACT_JOIN",
             {"result_update": {"artifacts": {"PRECHOICE29": "0" * 64}}}, None),
            ("all29_training", "ALL29_TRAINING_CORRECT",
             {"result_update": {"stages": [{"id": "PRECHOICE29", "training_count": 29, "training_correct": 28,
                                            "heads": [{"status": "PASS"}] * 3}]}}, None),
            ("owner_terminal", "CLOSED_RETAINED_FIT_OWNER",
             {"terminal_update": {"process_technical_complete": False}}, None),
            ("artifact_bindings", "EXACT_RELEASE_DERIVED_ARTIFACT_BINDINGS",
             {"bindings_update": {"arm": "OTHER"}}, None),
            ("retained_lock", "RETAINED_INPUT_DATA_LOCK", None,
             lambda fit: (fit.parent / "train_capture/DATA_LOCK.json").write_bytes(b'{"input_data_lock":2}')),
        )
        for name, code, build, mutate in cases:
            with self.subTest(case=name):
                self._run(code, build=build, mutate=mutate)


if __name__ == "__main__":
    unittest.main(verbosity=2)
