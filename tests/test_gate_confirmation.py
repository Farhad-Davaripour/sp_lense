"""Heldout scoring uses authored labels and a frozen validation checkpoint."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from src.gate_confirmation import authored_rows, frozen_checkpoint


class GateConfirmationTests(unittest.TestCase):
    def test_authored_conditions_override_a_conflicting_cached_teacher_label(self):
        native = [{"id": "peer", "state": {}, "teacher_gate": {"enabled": True}}]
        cases = [{"id": "peer", "family": "one_step", "condition": "other_complete"}]
        row = authored_rows(native, cases)[0]
        self.assertFalse(row["expected_gate"])
        self.assertEqual(row["expected_label"], "other")
        with self.assertRaises(ValueError):
            authored_rows(native, [{"id": "different"}])

    def test_heldout_cannot_use_a_changed_frozen_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "selected_adapter"
            adapter.mkdir()
            weights = adapter / "adapter.safetensors"
            weights.write_bytes(b"frozen")
            result = {"status": "GATE_PASS", "heldout_evaluated": False,
                      "checkpoint_files": {"adapter.safetensors": {
                          "bytes": 6, "sha256": hashlib.sha256(b"frozen").hexdigest()}}}
            path = root / "RESULT.json"
            path.write_bytes(json.dumps(result).encode())
            admission = {"gate_output": directory,
                         "validation_result_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertEqual(frozen_checkpoint(admission)[0], adapter)
            weights.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                frozen_checkpoint(admission)


if __name__ == "__main__":
    unittest.main()
