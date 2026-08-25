from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPT_PATH = Path(__file__).with_name("validate_jspace_cache.py")
SPEC = importlib.util.spec_from_file_location("validate_jspace_cache", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class JspaceCacheValidatorTests(unittest.TestCase):
    def test_validate_cache_returns_bound_receipt(self) -> None:
        manifest = {
            "model": {"id": "model"},
            "layer": 10,
            "atoms": {"file_sha256": "a" * 64},
            "token_labels": {"file_sha256": "b" * 64},
        }
        validated = SimpleNamespace(manifest=manifest, manifest_sha256="c" * 64)
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch(
                "sp_lense.jspace_comparison.validate_jspace_atom_manifest",
                return_value=validated,
            ) as canonical_validator,
        ):
            path = Path(directory) / "atoms_manifest.json"
            result = VALIDATOR.validate_cache(path)
        canonical_validator.assert_called_once_with(path.resolve())
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["manifest_sha256"], "c" * 64)
        self.assertEqual(result["atoms_file_sha256"], "a" * 64)

    def test_main_fails_closed_for_missing_or_invalid_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing_manifest.json"
            with (
                mock.patch.object(
                    VALIDATOR,
                    "validate_cache",
                    side_effect=FileNotFoundError("missing atoms.pt"),
                ),
                mock.patch.object(
                    VALIDATOR.sys,
                    "argv",
                    ["validate_jspace_cache.py", "--manifest", str(path)],
                ),
            ):
                self.assertEqual(VALIDATOR.main(), 1)


if __name__ == "__main__":
    unittest.main()
