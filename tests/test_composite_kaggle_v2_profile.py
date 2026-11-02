from pathlib import Path
from unittest.mock import patch
import unittest
from src import composite_kaggle_v2_profile as profile
from src.composite_v2_data_contract import validate_workspace


class KaggleV2ProfileTests(unittest.TestCase):
    def test_exact_registered_private_study_identity_and_namespace(self):
        value=profile.identity('farhad89/research4-generalized-data-v2')
        self.assertEqual(validate_workspace(value),value)
        self.assertEqual(profile.data_destination(value,'a'*40),
            '/kaggle/working/research4_campaign/runs/'+'a'*40+'/data_generalized_v2/export_01')
        with self.assertRaises(PermissionError):profile.identity('other/research4-generalized-data-v2')
        with self.assertRaises(PermissionError):profile.identity('farhad89/research3-untouched')
        bad={**value,'artifact_root':'/kaggle/working/unrelated'}
        with self.assertRaises(PermissionError):validate_workspace(bad)

    def test_platform_correction_does_not_silently_allow_databricks_or_local_runtime(self):
        with patch('src.composite_runtime.require_scientific_runtime'),patch('src.composite_runtime.platform',return_value='databricks'):
            with self.assertRaisesRegex(PermissionError,'Current Research4 remainder'):profile.require_v2_runtime()
        with patch.dict('os.environ',{'RESEARCH4_EXECUTION_PLATFORM':'kaggle'},clear=True):
            with self.assertRaises(RuntimeError):profile.require_v2_runtime()


if __name__=='__main__':unittest.main()
