from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_PATH = Path(__file__).with_name("validate_jspace_record.py")
SPEC = importlib.util.spec_from_file_location("validate_jspace_record", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class JspaceRecordValidatorTests(unittest.TestCase):
    def _fixture(self, root: Path, *, layer: int = 10) -> dict[str, object]:
        direction = root / "artifacts" / "steering_comparison" / "direction.json"
        direction.parent.mkdir(parents=True)
        direction.write_text('{"direction":true}\n', encoding="utf-8")
        setup = {
            "setup_id": "s" * 64,
            "model_id": "Qwen/Qwen3.5-0.8B",
            "model_revision": "revision",
            "model_config_sha256": "a" * 64,
            "method_id": "gradient",
            "track": "matched",
            "selected_layer": layer,
            "direction_float32_sha256": "b" * 64,
            "direction_artifact_sha256": "c" * 64,
            "direction_path": "artifacts/steering_comparison/direction.json",
        }
        plan = root / "plan.json"
        plan.write_text(json.dumps({"setups": [setup]}), encoding="utf-8")
        lens = {
            "repository": "lens/repository",
            "filename": "lens.pt",
            "file_sha256": "d" * 64,
            "file_size_bytes": 123,
            "revision": "lens-revision",
            "n_prompts": 12,
            "source_layers": [10],
            "fitted_model_id": setup["model_id"],
            "fitted_model_revision": "fitted-revision",
            "transfer_status": "same_checkpoint",
        }
        lock = root / "lock.json"
        lock.write_text(
            json.dumps(
                {
                    "sources": {"j_space": {"commit": "1" * 40}},
                    "evaluation": {
                        "j_space": {"models": {setup["model_id"]: {"lens": lens}}}
                    }
                }
            ),
            encoding="utf-8",
        )
        record = {
            "schema_version": "sp_lense.jspace_record.v2",
            "model_id": setup["model_id"],
            "model_revision": setup["model_revision"],
            "model_config_sha256": setup["model_config_sha256"],
            "method": setup["method_id"],
            "setup": setup["track"],
            "layer": setup["selected_layer"],
            "direction_float32_sha256": setup["direction_float32_sha256"],
            "direction_artifact_sha256": setup["direction_artifact_sha256"],
            "direction_file_sha256": hashlib.sha256(direction.read_bytes()).hexdigest(),
            "lens_provenance": {
                "file_sha256": lens["file_sha256"],
                "revision": lens["revision"],
                "source_layers": lens["source_layers"],
            },
            "non_gating": True,
            "used_for_primary_ranking": False,
        }
        return {
            "direction": direction,
            "setup": setup,
            "plan": plan,
            "lock": lock,
            "lens": lens,
            "record": record,
        }

    def test_available_layer_binds_record_to_setup_direction_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            atoms_manifest = root / "atoms_manifest.json"
            atoms_manifest.write_text(
                json.dumps(
                    {
                        "model": {
                            "id": fixture["setup"]["model_id"],
                            "revision": fixture["setup"]["model_revision"],
                            "config_sha256": fixture["setup"]["model_config_sha256"],
                        },
                        "lens": fixture["lens"],
                        "construction": {"reference_repository_commit": "1" * 40},
                        "atoms": {"file_sha256": "e" * 64, "float32_sha256": "f" * 64}
                    }
                ),
                encoding="utf-8",
            )
            record = {
                **fixture["record"],
                "status": "not_run_resource_limited",
                "atoms_manifest_sha256": "1" * 64,
                "atoms_file_sha256": "e" * 64,
                "atoms_float32_sha256": "f" * 64,
            }
            record_path = root / "record.jsonl"
            record_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            cache_receipt = {
                "model_id": fixture["setup"]["model_id"],
                "layer": 10,
                "manifest_sha256": "1" * 64,
            }
            with mock.patch.object(VALIDATOR, "validate_cache", return_value=cache_receipt):
                receipt = VALIDATOR.validate_record(
                    repo_root=root,
                    plan_path=fixture["plan"],
                    lock_path=fixture["lock"],
                    setup_id=fixture["setup"]["setup_id"],
                    record_path=record_path,
                    atoms_manifest_path=atoms_manifest,
                )
            self.assertEqual(receipt["status"], "valid")

            record["direction_file_sha256"] = "0" * 64
            record_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "direction-file hash is stale"):
                VALIDATOR.validate_record(
                    repo_root=root,
                    plan_path=fixture["plan"],
                    lock_path=fixture["lock"],
                    setup_id=fixture["setup"]["setup_id"],
                    record_path=record_path,
                    atoms_manifest_path=atoms_manifest,
                )

            record["direction_file_sha256"] = hashlib.sha256(
                fixture["direction"].read_bytes()
            ).hexdigest()
            record_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            tampered_manifest = json.loads(atoms_manifest.read_text(encoding="utf-8"))
            tampered_manifest["lens"]["revision"] = "wrong-lens-revision"
            atoms_manifest.write_text(json.dumps(tampered_manifest), encoding="utf-8")
            with (
                mock.patch.object(
                    VALIDATOR, "validate_cache", return_value=cache_receipt
                ),
                self.assertRaisesRegex(ValueError, "cache lens provenance mismatch"),
            ):
                VALIDATOR.validate_record(
                    repo_root=root,
                    plan_path=fixture["plan"],
                    lock_path=fixture["lock"],
                    setup_id=fixture["setup"]["setup_id"],
                    record_path=record_path,
                    atoms_manifest_path=atoms_manifest,
                )

    def test_unavailable_layer_requires_exact_not_run_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root, layer=23)
            record = {
                **fixture["record"],
                "status": "not_run_lens_layer_unavailable",
                "available_source_layers": [10],
            }
            record_path = root / "record.jsonl"
            record_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            receipt = VALIDATOR.validate_record(
                repo_root=root,
                plan_path=fixture["plan"],
                lock_path=fixture["lock"],
                setup_id=fixture["setup"]["setup_id"],
                record_path=record_path,
                atoms_manifest_path=None,
            )
            self.assertEqual(receipt["jspace_status"], "not_run_lens_layer_unavailable")


if __name__ == "__main__":
    unittest.main()
