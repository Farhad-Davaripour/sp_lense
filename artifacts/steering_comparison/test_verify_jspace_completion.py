from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("verify_jspace_completion.py")
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("verify_jspace_completion", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


class JspaceCompletionVerifierTests(unittest.TestCase):
    def _fixture(self, root: Path, *, primary: bool = True) -> dict[str, Path]:
        direction = root / "artifacts" / "steering_comparison" / "direction.json"
        direction.parent.mkdir(parents=True)
        direction.write_text('{"direction":true}\n', encoding="utf-8")
        setup_id = "a" * 64
        model_id = "Qwen/Qwen3.5-0.8B"
        setup = {
            "setup_id": setup_id,
            "model_id": model_id,
            "model_tag": "qwen35_08b",
            "model_revision": "revision",
            "model_config_sha256": "b" * 64,
            "method_id": "gradient" if primary else "random_control_00",
            "track": "matched",
            "selected_layer": 23,
            "direction_float32_sha256": "c" * 64,
            "direction_artifact_sha256": "d" * 64,
            "direction_path": "artifacts/steering_comparison/direction.json",
        }
        plan = root / "artifacts" / "steering_comparison" / "sealed" / "plan.json"
        plan.parent.mkdir(parents=True)
        plan.write_text(
            json.dumps(
                {
                    "schema_version": "sp_lense.locked_open_plan.v1",
                    "split": "sealed_test",
                    "setup_count": 1,
                    "setups": [setup],
                }
            ),
            encoding="utf-8",
        )
        lens = {
            "file_sha256": "e" * 64,
            "revision": "lens-revision",
            "source_layers": [10],
        }
        lock = root / "configs" / "steering_comparison_lock.json"
        lock.parent.mkdir(parents=True)
        lock.write_text(
            json.dumps(
                {"evaluation": {"j_space": {"models": {model_id: {"lens": lens}}}}}
            ),
            encoding="utf-8",
        )
        status = root / "artifacts" / "steering_comparison" / "jspace_status.json"
        status.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "state": "complete",
                    "detail": "exact coverage complete",
                    "process_id": 123,
                    "updated_at_utc": "2026-08-25T00:00:00Z",
                }
            ),
            encoding="utf-8",
        )
        records = root / "artifacts" / "steering_comparison" / "jspace" / "records"
        atoms = root / "artifacts" / "steering_comparison" / "jspace" / "atoms"
        if primary:
            records.mkdir(parents=True)
            record = {
                "schema_version": "sp_lense.jspace_record.v2",
                "status": "not_run_lens_layer_unavailable",
                "model_id": model_id,
                "model_revision": setup["model_revision"],
                "model_config_sha256": setup["model_config_sha256"],
                "method": setup["method_id"],
                "setup": setup["track"],
                "layer": setup["selected_layer"],
                "direction_float32_sha256": setup["direction_float32_sha256"],
                "direction_artifact_sha256": setup["direction_artifact_sha256"],
                "direction_file_sha256": hashlib.sha256(direction.read_bytes()).hexdigest(),
                "lens_provenance": lens,
                "available_source_layers": [10],
                "non_gating": True,
                "used_for_primary_ranking": False,
            }
            (records / f"direction_000_{setup_id[:16]}.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
        return {
            "root": root,
            "plan": plan,
            "lock": lock,
            "status": status,
            "records": records,
            "atoms": atoms,
        }

    def _verify(self, fixture: dict[str, Path]) -> dict[str, object]:
        return VERIFIER.verify_completion(
            repo_root=fixture["root"],
            plan_path=fixture["plan"],
            lock_path=fixture["lock"],
            status_path=fixture["status"],
            records_directory=fixture["records"],
            atoms_root=fixture["atoms"],
        )

    def test_exact_unavailable_layer_record_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt = self._verify(self._fixture(Path(directory)))
        self.assertEqual(receipt["status"], "valid_complete")
        self.assertEqual(receipt["record_count"], 1)
        self.assertEqual(
            receipt["record_status_counts"], {"not_run_lens_layer_unavailable": 1}
        )
        self.assertEqual(len(receipt["artifacts"]), 1)
        self.assertEqual(receipt["artifacts"][0]["path"], receipt["record_paths"][0])

    def test_completion_receipt_is_deterministic_and_hash_binds_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            receipt = self._verify(fixture)
            output = fixture["root"] / "artifacts" / "steering_comparison" / "jspace_completion.json"
            VERIFIER.write_receipt(output, receipt)
            first = output.read_bytes()
            VERIFIER.write_receipt(output, self._verify(fixture))
            self.assertEqual(output.read_bytes(), first)
            record_path = fixture["root"] / receipt["record_paths"][0]
            self.assertEqual(
                receipt["artifacts"][0]["sha256"],
                hashlib.sha256(record_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(receipt["artifacts"][0]["size_bytes"], record_path.stat().st_size)

    def test_extra_or_missing_completion_state_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            (fixture["records"] / "stale.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact canonical record set"):
                self._verify(fixture)

            (fixture["records"] / "stale.jsonl").unlink()
            status = json.loads(fixture["status"].read_text(encoding="utf-8"))
            status["state"] = "failed"
            fixture["status"].write_text(json.dumps(status), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "explicit completed state"):
                self._verify(fixture)

    def test_no_primary_direction_is_an_explicit_derived_skip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt = self._verify(self._fixture(Path(directory), primary=False))
        self.assertTrue(receipt["explicit_no_primary_direction_skip"])
        self.assertEqual(receipt["record_count"], 0)
        self.assertEqual(receipt["artifact_paths"], [])


if __name__ == "__main__":
    unittest.main()
