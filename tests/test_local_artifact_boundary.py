"""Portable checks for missing or changed exact-source handoff artifacts."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import company_models as models
from tests.support.local_artifacts import require_local_files


class LocalArtifactBoundaryTests(unittest.TestCase):
    def test_missing_artifact_is_an_explicit_local_integration_skip(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing-synthetic.py"
            with self.assertRaisesRegex(unittest.SkipTest, "Local-only exact-source integration"):
                require_local_files(missing)

    def test_present_but_changed_security_contract_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic-contract.py"
            path.write_bytes(b"# changed synthetic bytes\n")
            require_local_files(path)
            expected = hashlib.sha256(b"# original synthetic bytes\n").hexdigest()
            with patch.object(models, "SECURITY500_HANDOFF_PATH", path), \
                    patch.object(models, "SECURITY500_HANDOFF_SHA", expected), \
                    patch.object(models, "build_model", side_effect=AssertionError("No model build")) as build:
                with self.assertRaisesRegex(ValueError, "Exact Root two-route handoff source required"):
                    models.security500_handoff_contract()
            build.assert_not_called()

    def test_changed_continuity_public_metadata_fails_before_module_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            public = root / "synthetic-public.json"
            helper = root / "synthetic-helper.py"
            public.write_bytes(b'{"synthetic": true}')
            helper.write_bytes(b'raise AssertionError("must not execute")\n')
            require_local_files(public, helper)
            with patch.object(models, "CONTINUITY_TRANSFER_PUBLIC_PATH", public), \
                    patch.object(models, "CONTINUITY_TRANSFER_HANDOFF_PATH", helper), \
                    patch.object(models, "CONTINUITY_TRANSFER_PUBLIC_SHA", "0" * 64), \
                    patch.object(models, "CONTINUITY_TRANSFER_HANDOFF_SHA", "1" * 64):
                with self.assertRaisesRegex(ValueError, "public/helper source required"):
                    models.continuity_transfer_handoff_contract()


if __name__ == "__main__":
    unittest.main()
