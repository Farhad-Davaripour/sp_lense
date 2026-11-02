"""Synthetic byte-envelope tests; no source/data/model allocation."""
import unittest
from zipfile import ZipInfo
from src.composite_task_skill_source_package import validate_source_envelope, source_transport_limit


class ApprovedSamplingTransportTests(unittest.TestCase):
    def test_new_cpu_and_legacy_transports_keep_128_mib(self):
        for name in ['BOUNDED_SAMPLING_AMENDMENT_PACKAGE_FILES.json',
                     'BOUNDED_BEHAVIOR_DATA_PACKAGE_FILES.json', 'TASK_TRAINING_PACKAGE_FILES.json']:
            with self.subTest(name=name):
                entry = ZipInfo('source/inputs/opaque.r4blob')
                entry.file_size = 128 * 1024**2
                validate_source_envelope([entry], name)
                entry.file_size += 1
                with self.assertRaises(ValueError):
                    validate_source_envelope([entry], name)

    def test_only_registered_fit_transport_gets_256_mib(self):
        name = 'BOUNDED_SAMPLING_FIT_PACKAGE_FILES.json'
        entry = ZipInfo('source/inputs/opaque.r4blob')
        entry.file_size = 256 * 1024**2
        validate_source_envelope([entry], name)
        entry.file_size += 1
        with self.assertRaises(ValueError):
            validate_source_envelope([entry], name)
        with self.assertRaises(PermissionError):
            source_transport_limit('BOUNDED_SAMPLING_UNREGISTERED_FILES.json')


if __name__ == '__main__':
    unittest.main()
