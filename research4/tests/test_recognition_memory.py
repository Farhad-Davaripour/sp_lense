import json
import tempfile
import unittest
from pathlib import Path

from src.recognition_memory import admission, prospective_fp32_bytes


class RecognitionMemoryTests(unittest.TestCase):
    def fixture(self, path):
        descriptor = {"w": {"dtype": "BF16", "shape": [10, 20], "data_offsets": [0, 400]},
                      "index": {"dtype": "I64", "shape": [3], "data_offsets": [400, 424]}}
        header = json.dumps(descriptor).encode()
        path.write_bytes(len(header).to_bytes(8, "little") + header + b'\0' * 424)

    def test_admission_accounts_for_FP32_expansion_and_keeps_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            self.fixture(path)
            self.assertEqual(prospective_fp32_bytes(path)["prospective_FP32_parameter_bytes"], 824)
            profile = {"activation_reserve_bytes": 100, "export_reserve_bytes": 50}
            self.assertFalse(admission(path, 973, profile)["admitted"])
            self.assertTrue(admission(path, 974, profile)["admitted"])
            self.assertEqual(admission(path, 974, profile)["live_free_memory_floor_bytes"], 150)

    def test_invalid_header_fails_before_any_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.safetensors"
            path.write_bytes(b'invalid')
            with self.assertRaises(ValueError):
                prospective_fp32_bytes(path)
