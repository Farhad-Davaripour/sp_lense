"""Opaque-proof and source-scope boundary fixtures; no neural/scientific runs."""
import unittest
from src.composite_bounded_behavior_data_reader import (BoundedDataAudit, BoundedDataArchive,
                                                        require_data, audit_data, read_cpu_runtime)
from src.composite_bounded_behavior_runtime import reference_replay_scope


class BoundedDataReaderTests(unittest.TestCase):
    def test_raw_receipt_requires_exact_cpu_build_and_package_provenance(self):
        value = {'python': '3.13.0', 'torch': '2.11.0+cpu', 'torch_cuda_build': None,
                 'packages': {'transformers': '5.19.0', 'numpy': '2.2.2', 'safetensors': '0.8.0'}}
        self.assertEqual(read_cpu_runtime({'cpu_runtime': value}), value)
        bad_values = [None, {**value, 'torch': '2.11.0+cu128'},
                      {**value, 'torch_cuda_build': '12.8'},
                      {**value, 'packages': {**value['packages'], 'numpy': '2.1.0'}},
                      {**value, 'packages': {**value['packages'], 'bitsandbytes': '0.49.2'}}]
        for bad in bad_values:
            with self.subTest(value=bad), self.assertRaises(PermissionError):
                read_cpu_runtime({'cpu_runtime': bad})
        with self.assertRaises(PermissionError): read_cpu_runtime({})

    def test_serialized_success_cannot_create_released_data_proof(self):
        with self.assertRaises(PermissionError): require_data({'bounded_behavior_data_admitted': True})
        with self.assertRaises(PermissionError): BoundedDataAudit({}, {}, {}, {}, {}, {})
        with self.assertRaises(PermissionError): BoundedDataArchive(None, None, None)

    def test_public_or_json_replay_does_not_open_scientific_cases(self):
        with self.assertRaises(PermissionError):
            with reference_replay_scope({'raw_evidence_audited': True}): pass


if __name__ == '__main__': unittest.main()
