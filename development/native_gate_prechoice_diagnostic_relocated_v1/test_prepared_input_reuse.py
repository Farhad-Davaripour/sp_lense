"""Tests for the accepted prepared-input reuse gate.

Model-free and history-free: the real case reads only the already accepted, hash-pinned
preparation metadata and the fixed sixteen prepared views, while the negative cases patch
module constants or the pinned contract import. No Qwen/model/tokenizer/provider import
occurs, no activation or diagnostic row is read, nothing is fitted and no scoring runs.

Mocks are confined to tests (module globals and an import/open tripwire); production
modules are never monkeypatched outside the test body.
"""
import builtins, importlib, inspect, io, sys, tempfile, unittest, unittest.mock
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diagnostic_reader as reader

EXPECTED_DIGEST = "8a899b396d504cc0ccbac2e373cc0e933329803478f6320fc97cbc305c0a7b0f"
EXPECTED_RELEASE = "c707f335c4b1025e00896c3569c59cbad42cfe38dc0c2db2131ebaa749d72c19"


class PreparedInputReuseTests(unittest.TestCase):
    def test_01_actual_accepted_metadata_produces_exact_digest(self):
        record = reader.reuse_accepted_prepared_inputs()
        self.assertEqual(record["schema"], reader.PREPARED_REUSE_SCHEMA)
        self.assertEqual(record["status"], "REUSED_ALREADY_ACCEPTED_PREPARED_INPUTS")
        self.assertEqual(record["prepared_inputs_sha256"], EXPECTED_DIGEST)
        result = record["prepared_inputs"]
        self.assertEqual(result["schema"], "prechoice_diagnostic_inputs.v1")
        self.assertEqual(result["role"], "DIAGNOSTIC_PRECHOICE")
        self.assertEqual(result["certificate_sha256"], reader.CERTIFICATE_SHA256)
        self.assertEqual(result["data_lock_sha256"], reader.DATA_LOCK_SHA256)
        self.assertEqual(reader.sha(reader.canonical(result) + b"\n"), EXPECTED_DIGEST)
        self.assertEqual(tuple(case["case_key"] for case in result["cases"]), reader.PREPARED_VIEW_KEYS)
        self.assertEqual(len(result["cases"]), 16)
        for case in result["cases"]:
            selector = case["readout_selector"]
            self.assertEqual(selector["case_key"], case["case_key"])
            self.assertLess(selector["readout_index"], selector["final_input_index"])
            self.assertEqual(selector["input_ids_sha256"], case["input_binding"]["derived_input_int64_le_sha256"])
        for permission in ("model_permission", "tokenizer_permission", "provider_permission", "activation_permission",
                           "logit_permission", "outcome_permission", "fit_permission", "scoring_permission",
                           "diagnostic_execution_permission"):
            self.assertIs(record[permission], False)
        self.assertIs(record["historical_preparation_replayed"], False)
        self.assertIs(record["independent_reproduction"], False)
        self.assertIs(record["reuse_of_accepted_prepared_inputs"], True)
        self.assertEqual(record["preparation_files_verified"], 19)
        self.assertEqual(record["accepted_preparation_release_sha256"], EXPECTED_RELEASE)

    def test_02_no_caller_supplied_pin_and_substitution_fails_closed(self):
        self.assertEqual(list(inspect.signature(reader.reuse_accepted_prepared_inputs).parameters), [])
        with tempfile.TemporaryDirectory() as tmp:
            tampered = Path(tmp) / "DATA_LOCK.json"
            tampered.write_bytes(b"{}")
            with unittest.mock.patch.object(reader, "DATA_LOCK", tampered):
                with self.assertRaisesRegex(ValueError, "ACCEPTED_DATA_LOCK_HASH"):
                    reader.reuse_accepted_prepared_inputs()

    def test_03_selector_mismatch_fails_closed(self):
        real = reader._contract_module()

        class Anchor:
            @staticmethod
            def bind(ids, witness, case_key, certificate_sha256, *, expected_input_ids_sha256):
                selector = real.bind(ids, witness, case_key, certificate_sha256,
                                     expected_input_ids_sha256=expected_input_ids_sha256)
                selector["readout_index"] += 1
                return selector

        with unittest.mock.patch.object(reader, "_contract_module", return_value=Anchor):
            with self.assertRaisesRegex(ValueError, "PREPARED_INPUTS_REOBSERVED"):
                reader.reuse_accepted_prepared_inputs()

    def test_04_order_mismatch_fails_closed(self):
        swapped = list(reader.PREPARED_VIEW_KEYS)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        with unittest.mock.patch.object(reader, "PREPARED_VIEW_KEYS", tuple(swapped)):
            with self.assertRaisesRegex(ValueError, "EXACT_SIXTEEN_CERTIFIED_VIEW_ORDER"):
                reader.reuse_accepted_prepared_inputs()

    def test_05_no_provider_activation_or_retired_root_access(self):
        calls = []
        real_open, real_import = io.open, builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in ("torch", "transformers", "tokenizers", "safetensors", "numpy"):
                raise AssertionError("PROVIDER_IMPORT:" + name)
            return real_import(name, *args, **kwargs)

        def guarded_open(file, *args, **kwargs):
            text = str(file).lower()
            if "onedrive" in text:
                calls.append("retired_root")
                raise AssertionError("RETIRED_ROOT_READ")
            parts = str(file).replace("\\", "/").split("/")
            if any(part in parts for part in ("rows", "logits", "real_evidence")):
                raise AssertionError("ACTIVATION_OR_DIAGNOSTIC_READ")
            return real_open(file, *args, **kwargs)

        with unittest.mock.patch("io.open", side_effect=guarded_open), \
                unittest.mock.patch("builtins.__import__", new=guarded_import):
            record = reader.reuse_accepted_prepared_inputs()
        self.assertEqual(calls, [])
        self.assertEqual(record["prepared_inputs_sha256"], EXPECTED_DIGEST)


if __name__ == "__main__":
    unittest.main(verbosity=2)
