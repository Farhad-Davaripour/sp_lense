"""Single-fault tamper tests for the reader's certificate and frozen-source pins.

Metadata only: no model, tokenizer, provider, activation, logit, outcome or scoring read.
Each pin is exercised with a temporary-file substitution so one tampered artifact is
rejected *before* the certificate is parsed or the contract is imported. The real files
are only hashed against the historical frozen pins.
"""
import json, sys, tempfile, unittest, unittest.mock
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diagnostic_reader as reader


class ReaderPinTests(unittest.TestCase):
    def test_01_positive_control_real_pins_pass(self):
        pairs = reader.certificate_pairs()
        self.assertEqual(len(pairs), 8)
        contract = reader._contract_module()
        self.assertEqual(contract.CONTRACT["position"], reader.POSITION)
        self.assertEqual(reader.sha(reader.read_bytes(reader.DIAGNOSTIC_CAPTURE / "index_contract.py",
                                                      "INDEX_CONTRACT_FILE")),
                         reader.INDEX_CONTRACT_SHA256)

    def test_02_tampered_certificate_is_rejected_before_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            tampered = Path(tmp) / "BOUNDARY_CERTIFICATE.json"
            tampered.write_bytes(b"{ this is not json")
            with unittest.mock.patch.object(reader, "CERTIFICATE", tampered):
                with self.assertRaises(ValueError) as caught:
                    reader.certificate_pairs()
            self.assertEqual(str(caught.exception), "BOUNDARY_CERTIFICATE_HASH")

    def test_03_substituted_valid_json_certificate_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            document = json.loads(reader.read_bytes(reader.CERTIFICATE, "BOUNDARY_CERTIFICATE"))
            document["pairs"] = []
            tampered = Path(tmp) / "BOUNDARY_CERTIFICATE.json"
            tampered.write_bytes(json.dumps(document, separators=(",", ":")).encode("ascii"))
            with unittest.mock.patch.object(reader, "CERTIFICATE", tampered):
                with self.assertRaises(ValueError) as caught:
                    reader.certificate_pairs()
            self.assertEqual(str(caught.exception), "BOUNDARY_CERTIFICATE_HASH")

    def test_04_tampered_index_contract_is_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            tampered = Path(tmp) / "index_contract.py"
            tampered.write_bytes(b"this is not valid python (((\n")
            with unittest.mock.patch.object(reader, "DIAGNOSTIC_CAPTURE", Path(tmp)):
                with self.assertRaises(ValueError) as caught:
                    reader._contract_module()
            # a hash mismatch precedes import: the invalid source never raised SyntaxError
            self.assertEqual(str(caught.exception), "INDEX_CONTRACT_HASH")

    def test_05_pin_is_the_historical_source_lock_not_a_current_recompute(self):
        freeze = json.loads((reader.DIAGNOSTIC_CAPTURE / "SOURCE_FREEZE.json").read_text(encoding="utf-8"))
        self.assertEqual(freeze["source_sha256"]["index_contract.py"], reader.INDEX_CONTRACT_SHA256)
        self.assertEqual(reader.sha(reader.read_bytes(reader.DIAGNOSTIC_CAPTURE / "index_contract.py",
                                                      "INDEX_CONTRACT_FILE")),
                         reader.INDEX_CONTRACT_SHA256)


if __name__ == "__main__":
    unittest.main(verbosity=2)
