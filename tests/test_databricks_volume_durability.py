import copy
import json
import pathlib
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from src.volume_durable_io import (
    SUCCESS_MANIFEST, digest, file_descriptor,
    local_atomic_bytes, publish_immutable_file, publish_tree,
    publish_failure_manifest, publish_update_receipt,
    verify_failure_manifest, verify_manifest,
)
from src.databricks_continuity_partition import (
    PARENT_SUCCESS_MANIFEST, _parent_manifest,
    _verify_durable_completion, _verify_parent_manifest)


def progress_row(update):
    value = {
        "version": "research4-parent-mlflow-journal-v4",
        "intent_sha256": "a" * 64,
        "plan_sha256": "b" * 64,
        "optimizer_update": update,
        "loss": float(update + 1),
    }
    value["progress_record_sha256"] = digest(value)
    return value


class VolumeDurabilityTests(unittest.TestCase):
    @staticmethod
    def write_progress(root, *updates):
        (pathlib.Path(root) / "MLFLOW_PROGRESS.jsonl").write_text(
            "".join(json.dumps(progress_row(update), sort_keys=True) + "\n"
                    for update in updates), encoding="utf-8")

    @staticmethod
    def build_completed_run(root, *, child_identity="v7"):
        root = pathlib.Path(root); local = root / "local-output"
        local.mkdir(parents=True)
        (local / "MLFLOW_PROGRESS.jsonl").write_bytes(b"")
        publish_tree(
            local, root / "output", status="complete",
            manifest_name=SUCCESS_MANIFEST, required_update_count=0,
            metadata={
                "identity": child_identity,
                "intent_sha256": "a" * 64,
                "plan_sha256": "b" * 64,
                "executed_optimizer_step_count": 0,
                "durable_published_update_count": 0,
            })
        (root / "records.jsonl").write_bytes(b"record\n")
        (root / "TERMINAL.json").write_text("{}\n", encoding="utf-8")
        (root / "output" / "PARENT_FINALIZATION.json").write_text(
            "{}\n", encoding="utf-8")
        _parent_manifest(
            root, identity="v7", intent_sha256="a" * 64,
            plan_sha256="b" * 64)
        return root

    def test_local_fsync_and_volume_publish_uses_sequential_copy_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "local" / "value.bin"
            volume = root / "volume" / "value.bin"
            with patch("src.volume_durable_io.os.fsync") as fsync:
                local_atomic_bytes(source, b"exact-bytes")
            fsync.assert_called_once()
            events = []

            def sequential(read_stream, write_stream):
                events.append("copy")
                while True:
                    block = read_stream.read(3)
                    if not block:
                        break
                    write_stream.write(block)

            with patch("src.volume_durable_io.os.fsync",
                       side_effect=AssertionError("Volume fsync forbidden")), \
                    patch("src.volume_durable_io.os.replace",
                          side_effect=AssertionError("Volume replace forbidden")):
                result = publish_immutable_file(
                    source, volume, copy_file=sequential)
            self.assertEqual(events, ["copy"])
            self.assertEqual(volume.read_bytes(), b"exact-bytes")
            self.assertEqual(result["sha256"], file_descriptor(source)["sha256"])

    def test_exact_existing_is_idempotent_but_conflict_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "source"; target = root / "volume"
            local_atomic_bytes(source, b"first")
            first = publish_immutable_file(source, target)
            second = publish_immutable_file(source, target)
            self.assertFalse(first["idempotent"])
            self.assertTrue(second["idempotent"])
            local_atomic_bytes(source, b"different")
            with self.assertRaises(FileExistsError):
                publish_immutable_file(source, target)
            self.assertEqual(target.read_bytes(), b"first")

    def test_parent_extra_is_required_only_after_parent_file_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            local = root / "local"; volume = root / "volume"
            local.mkdir()
            (local / "MLFLOW_PROGRESS.jsonl").write_bytes(b"")
            metadata = {
                "identity": "v7", "intent_sha256": "a" * 64,
                "plan_sha256": "b" * 64,
                "executed_optimizer_step_count": 0,
                "durable_published_update_count": 0,
            }
            publish_tree(
                local, volume, status="complete",
                manifest_name=SUCCESS_MANIFEST, required_update_count=0,
                metadata=metadata)
            verify_manifest(
                volume, manifest_name=SUCCESS_MANIFEST,
                expected_status="complete", required_update_count=0,
                allowed_extra=set(), expected_metadata=metadata)
            with self.assertRaisesRegex(
                    PermissionError, "missing or extra"):
                verify_manifest(
                    volume, manifest_name=SUCCESS_MANIFEST,
                    expected_status="complete", required_update_count=0,
                    allowed_extra={"PARENT_FINALIZATION.json"},
                    expected_metadata=metadata)
            (volume / "PARENT_FINALIZATION.json").write_text(
                "{}\n", encoding="utf-8")
            verify_manifest(
                volume, manifest_name=SUCCESS_MANIFEST,
                expected_status="complete", required_update_count=0,
                allowed_extra={"PARENT_FINALIZATION.json"},
                expected_metadata=metadata)

    def test_torn_copy_fails_and_never_creates_success_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "local" / "source"
            target = root / "volume" / "source"
            local_atomic_bytes(source, b"complete-value")

            def torn(read_stream, write_stream):
                write_stream.write(read_stream.read(4))

            with self.assertRaises(IOError):
                publish_immutable_file(source, target, copy_file=torn)
            self.assertFalse((root / "volume" / SUCCESS_MANIFEST).exists())
            failure = publish_failure_manifest(
                root / "local", root / "volume",
                metadata={"identity": "v7", "intent_sha256": "a" * 64,
                          "plan_sha256": "b" * 64},
                error_type="IOError", error="torn copy")
            self.assertEqual(
                verify_failure_manifest(root / "volume")["manifest_sha256"],
                failure["manifest_sha256"])

    def test_concurrent_publishers_never_overwrite_immutable_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory); target = root / "volume" / "value"
            first = root / "first"; second = root / "second"
            local_atomic_bytes(first, b"first-complete")
            local_atomic_bytes(second, b"second-complete")
            barrier = threading.Barrier(2)

            def publish(source):
                barrier.wait()
                return publish_immutable_file(source, target)

            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(publish, source)
                           for source in (first, second)]
            outcomes = []
            for future in futures:
                try:
                    outcomes.append(future.result())
                except FileExistsError:
                    outcomes.append("conflict")
            self.assertEqual(sum(value == "conflict" for value in outcomes), 1)
            self.assertIn(target.read_bytes(), {b"first-complete", b"second-complete"})

    def test_update_receipts_are_immutable_and_ordered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory); local = root / "local"; volume = root / "volume"
            with self.assertRaises(PermissionError):
                publish_update_receipt(local, volume, progress_row(1))
            publish_update_receipt(local, volume, progress_row(0))
            publish_update_receipt(local, volume, progress_row(1))
            self.write_progress(local, 0, 1)
            manifest = publish_tree(
                local, volume, status="complete", manifest_name=SUCCESS_MANIFEST,
                required_update_count=2, metadata={"identity": "v7"})
            self.assertEqual(
                [row["optimizer_update"] for row in manifest["update_receipts"]],
                [0, 1])
            self.assertEqual(
                verify_manifest(volume, manifest_name=SUCCESS_MANIFEST,
                                expected_status="complete",
                                required_update_count=2)["manifest_sha256"],
                manifest["manifest_sha256"])

    def test_missing_gap_extra_and_conflict_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory); local = root / "local"; volume = root / "volume"
            publish_update_receipt(local, volume, progress_row(0))
            with self.assertRaises(PermissionError):
                publish_tree(local, volume, status="complete",
                             manifest_name=SUCCESS_MANIFEST,
                             required_update_count=2)
            bad = progress_row(0); bad["loss"] = 99.0
            bad_payload = dict(bad); bad_payload.pop("progress_record_sha256")
            bad["progress_record_sha256"] = digest(bad_payload)
            with self.assertRaises(PermissionError):
                publish_update_receipt(local, volume, bad)

    def test_failure_manifest_cannot_satisfy_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory); local = root / "local"; volume = root / "volume"
            publish_update_receipt(local, volume, progress_row(0))
            self.write_progress(local, 0)
            local_atomic_bytes(
                local / "RANK_0_PARTIAL.json",
                b'{"executed_optimizer_step_count":2,"durable_published_update_count":1}\n')
            failure = publish_failure_manifest(
                local, volume,
                metadata={"identity": "v7", "intent_sha256": "a" * 64,
                          "plan_sha256": "b" * 64},
                error_type="RuntimeError", error="expected failure")
            self.assertEqual(
                [row["path"] for row in failure["embedded_local_files"]],
                ["MLFLOW_PROGRESS.jsonl", "RANK_0_PARTIAL.json"])
            with self.assertRaises(PermissionError):
                verify_failure_manifest(
                    volume, expected_metadata={
                        "identity": "wrong", "intent_sha256": "a" * 64,
                        "plan_sha256": "b" * 64})
            with self.assertRaises(FileNotFoundError):
                verify_manifest(volume, manifest_name=SUCCESS_MANIFEST,
                                expected_status="complete",
                                required_update_count=1)

    def test_manifest_rejects_tamper_and_extra_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory); local = root / "local"; volume = root / "volume"
            publish_update_receipt(local, volume, progress_row(0))
            self.write_progress(local, 0)
            local_atomic_bytes(local / "checkpoint" / "weights.bin", b"weights")
            publish_tree(local, volume, status="complete",
                         manifest_name=SUCCESS_MANIFEST,
                         required_update_count=1)
            (volume / "unexpected").write_bytes(b"extra")
            with self.assertRaises(PermissionError):
                verify_manifest(volume, manifest_name=SUCCESS_MANIFEST,
                                expected_status="complete",
                                required_update_count=1)
            (volume / "unexpected").unlink()
            (volume / "checkpoint" / "weights.bin").write_bytes(b"tamper")
            with self.assertRaises(PermissionError):
                verify_manifest(volume, manifest_name=SUCCESS_MANIFEST,
                                expected_status="complete",
                                required_update_count=1)

    def test_progress_receipt_and_completion_metadata_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory); local = root / "local"; volume = root / "volume"
            publish_update_receipt(local, volume, progress_row(0))
            changed = progress_row(0); changed["loss"] = 99.0
            payload = dict(changed); payload.pop("progress_record_sha256")
            changed["progress_record_sha256"] = digest(payload)
            (local / "MLFLOW_PROGRESS.jsonl").write_text(
                json.dumps(changed, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaises(PermissionError):
                publish_tree(local, volume, status="complete",
                             manifest_name=SUCCESS_MANIFEST,
                             required_update_count=1,
                             metadata={"identity": "v7"})

            self.write_progress(local, 0)
            manifest = publish_tree(
                local, volume, status="complete",
                manifest_name=SUCCESS_MANIFEST,
                required_update_count=1, metadata={"identity": "v7"})
            with self.assertRaises(PermissionError):
                verify_manifest(
                    volume, manifest_name=SUCCESS_MANIFEST,
                    expected_status="complete", required_update_count=1,
                    expected_metadata={"identity": "wrong"})
            self.assertEqual(manifest["metadata"], {"identity": "v7"})

    def test_local_atomic_write_rejects_volume_path(self):
        with self.assertRaises(PermissionError):
            local_atomic_bytes("/Volumes/catalog/schema/volume/value", b"x")

    def test_parent_manifest_is_last_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "output").mkdir(parents=True)
            (root / "records.jsonl").write_bytes(b"record\n")
            (root / "TERMINAL.json").write_text("{}\n", encoding="utf-8")
            (root / "output" / SUCCESS_MANIFEST).write_text(
                "{}\n", encoding="utf-8")
            (root / "output" / "PARENT_FINALIZATION.json").write_text(
                "{}\n", encoding="utf-8")
            value = _parent_manifest(
                root, identity="v7", intent_sha256="a" * 64,
                plan_sha256="b" * 64)
            self.assertTrue((root / PARENT_SUCCESS_MANIFEST).is_file())
            self.assertEqual(
                _verify_parent_manifest(
                    root, identity="v7", intent_sha256="a" * 64,
                    plan_sha256="b" * 64)["manifest_sha256"],
                value["manifest_sha256"])
            (root / "TERMINAL.json").write_text(
                '{"tampered":true}\n', encoding="utf-8")
            with self.assertRaises(PermissionError):
                _verify_parent_manifest(
                    root, identity="v7", intent_sha256="a" * 64,
                    plan_sha256="b" * 64)

    def test_finalizer_boundary_rejects_missing_or_mismatched_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.build_completed_run(directory)
            proof = _verify_durable_completion(
                root, identity="v7", intent_sha256="a" * 64,
                plan_sha256="b" * 64, optimizer_updates=0)
            self.assertEqual(proof["child"]["status"], "complete")

        with tempfile.TemporaryDirectory() as directory:
            root = self.build_completed_run(directory, child_identity="wrong")
            with self.assertRaises(PermissionError):
                _verify_durable_completion(
                    root, identity="v7", intent_sha256="a" * 64,
                    plan_sha256="b" * 64, optimizer_updates=0)

        for missing in (f"output/{SUCCESS_MANIFEST}",
                        PARENT_SUCCESS_MANIFEST):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                root = self.build_completed_run(directory)
                (root / missing).unlink()
                with self.assertRaises((FileNotFoundError, PermissionError)):
                    _verify_durable_completion(
                        root, identity="v7", intent_sha256="a" * 64,
                        plan_sha256="b" * 64, optimizer_updates=0)

        with tempfile.TemporaryDirectory() as directory:
            root = self.build_completed_run(directory)
            (root / "TERMINAL.json").write_text(
                '{"tampered":true}\n', encoding="utf-8")
            with self.assertRaises(PermissionError):
                _verify_durable_completion(
                    root, identity="v7", intent_sha256="a" * 64,
                    plan_sha256="b" * 64, optimizer_updates=0)


if __name__ == "__main__":
    unittest.main()
