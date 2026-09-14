"""Tests for snapshot_verifier.py using ONLY temporary fabricated files.

No real snapshot, model, tokenizer, weights, vectors, caches or datasets are
read or rehashed here.  Every lock and every byte hashed by these tests lives
under a ``tempfile.TemporaryDirectory`` created in ``setUp``.

Run from the repository root with the study runtime:

    development\\classifier_generalization_v2\\.runtime\\Scripts\\python.exe \
        -W error development/classifier_generalization_v2/test_snapshot_verifier.py -v
"""

import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import snapshot_verifier as sv


class _FakeClock:
    """Monotonic clock stub; repeats its final value once exhausted."""

    def __init__(self, values):
        self._values = list(values)
        self._index = 0

    def __call__(self):
        if not self._values:
            return 0.0
        value = self._values[min(self._index, len(self._values) - 1)]
        self._index += 1
        return value


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


class SnapshotVerifierTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="snapshot_verifier_")
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "hub"
        self.snap_rel = Path("models--test--tiny") / "snapshots" / sv.FIXED_REVISION
        self.snap = self.root / self.snap_rel
        self.snap.mkdir(parents=True)
        self.snap_rel_str = self.snap_rel.as_posix()
        self.bounds = {
            "max_files": 16,
            "max_file_bytes": 1024 * 1024,
            "max_total_bytes": 4 * 1024 * 1024,
            "deadline_seconds": 30.0,
        }

    def tearDown(self):
        self._tmp.cleanup()

    # -- helpers -----------------------------------------------------------
    def _write(self, name, data, directory=None):
        path = (directory or self.snap) / name
        path.write_bytes(data)
        return (name, len(data), _sha256(data))

    def _entry(self, name, data, directory=None):
        return self._write(name, data, directory)

    def _lock(self, entries, identity, *, snap_rel=None,
              revision=sv.FIXED_REVISION, schema=sv.LOCK_SCHEMA):
        return {
            "schema": schema,
            "revision": revision,
            "snapshot_relative_path": snap_rel or self.snap_rel_str,
            "files": [
                {"name": name, "bytes": size, "sha256": digest}
                for (name, size, digest) in entries
            ],
            "tokenizer_identity_files": list(identity),
        }

    def _lock_bytes(self, lock):
        return json.dumps(lock).encode("utf-8")

    def _verify(self, lock, *, expected=None, root=None, bounds=None, **kw):
        raw = self._lock_bytes(lock)
        params = dict(self.bounds)
        if bounds:
            params.update(bounds)
        params.update(kw)
        return sv.verify_snapshot(
            raw,
            expected_lock_sha256=expected if expected is not None else _sha256(raw),
            allowed_root=root if root is not None else self.root,
            **params,
        )

    def _assert_code(self, code, lock, *, expected=None, root=None,
                     bounds=None, **kw):
        with self.assertRaises(sv.SnapshotVerificationError) as ctx:
            self._verify(lock, expected=expected, root=root, bounds=bounds, **kw)
        self.assertEqual(ctx.exception.code, code)
        return ctx.exception

    # -- correct receipt ---------------------------------------------------
    def test_correct_receipt(self):
        tokenizer = self._entry("tokenizer.json", b'{"tokens":1}')
        vocab = self._entry("vocab.json", b"vocab-bytes")
        weights = self._entry("model.safetensors", b"W" * 4096)
        lock = self._lock([tokenizer, vocab, weights],
                          ["tokenizer.json", "vocab.json"])
        receipt = self._verify(lock)

        self.assertEqual(receipt["status"], "verified")
        self.assertEqual(receipt["schema"], sv.RECEIPT_SCHEMA)
        self.assertEqual(receipt["job_id"], sv.JOB_ID)
        self.assertEqual(receipt["revision"], sv.FIXED_REVISION)
        self.assertEqual(receipt["lock_sha256"], _sha256(self._lock_bytes(lock)))
        self.assertEqual([item["name"] for item in receipt["checked_files"]],
                         ["tokenizer.json", "vocab.json", "model.safetensors"])
        for item in receipt["checked_files"]:
            self.assertTrue(item["verified"])
            self.assertEqual(item["bytes"], item["bytes_read"])
            self.assertFalse(item["symlink_resolved"])
        self.assertEqual(receipt["bytes_hashed_total"],
                         sum(item["bytes"] for item in receipt["checked_files"]))

        expected_aggregate = hashlib.sha256(
            b"tokenizer.json\x00"
            + str(tokenizer[1]).encode("ascii") + b"\x00"
            + tokenizer[2].encode("ascii") + b"\x00"
            + b"vocab.json\x00"
            + str(vocab[1]).encode("ascii") + b"\x00"
            + vocab[2].encode("ascii") + b"\x00"
        ).hexdigest()
        self.assertEqual(receipt["aggregate_tokenizer_identity_sha256"],
                         expected_aggregate)
        self.assertEqual(receipt["tokenizer_identity_files"],
                         ["tokenizer.json", "vocab.json"])

        for flag in ("tokenizer_loaded", "model_loaded", "weights_tensor_parsed",
                     "model_execution_authorized",
                     "scientific_execution_authorized",
                     "recursive_search_performed", "network_access_performed"):
            self.assertIs(receipt[flag], False)
        json.dumps(receipt)  # JSON-compatible.

    # -- lock digest before file access ------------------------------------
    def test_mismatched_lock_hash_has_zero_reads(self):
        lock = self._lock([self._entry("tokenizer.json", b"abc")],
                          ["tokenizer.json"])
        raw = self._lock_bytes(lock)
        calls = []
        with self.assertRaises(sv.SnapshotVerificationError) as ctx:
            sv.verify_snapshot(
                raw,
                expected_lock_sha256="0" * 64,
                allowed_root=self.tmp / "does-not-exist",
                fs_probe=calls.append,
                **self.bounds
            )
        self.assertEqual(ctx.exception.code, "LOCK_DIGEST_MISMATCH")
        self.assertEqual(calls, [])

    def test_digest_checked_before_json_parse(self):
        raw = b"{not json at all"
        calls = []
        with self.assertRaises(sv.SnapshotVerificationError) as ctx:
            sv.verify_snapshot(
                raw,
                expected_lock_sha256="1" * 64,
                allowed_root=self.tmp / "does-not-exist",
                fs_probe=calls.append,
                **self.bounds
            )
        self.assertEqual(ctx.exception.code, "LOCK_DIGEST_MISMATCH")
        self.assertEqual(calls, [])

    def test_invalid_json_after_matching_digest(self):
        raw = b"{not json at all"
        calls = []
        with self.assertRaises(sv.SnapshotVerificationError) as ctx:
            sv.verify_snapshot(
                raw,
                expected_lock_sha256=_sha256(raw),
                allowed_root=self.tmp / "does-not-exist",
                fs_probe=calls.append,
                **self.bounds
            )
        self.assertEqual(ctx.exception.code, "LOCK_JSON_INVALID")
        self.assertEqual(calls, [])

    # -- stale / tampered --------------------------------------------------
    def test_mutable_lock_snapshotted_once(self):
        entry = self._entry("tokenizer.json", b"original")
        lock = self._lock([entry], ["tokenizer.json"])
        raw = self._lock_bytes(lock)

        class ChangingBytes(bytearray):
            calls = 0

            def __bytes__(self):
                self.calls += 1
                return raw if self.calls == 1 else b"{}"

        changing = ChangingBytes(raw)
        receipt = sv.verify_snapshot(
            changing, expected_lock_sha256=_sha256(raw),
            allowed_root=self.root, **self.bounds)
        self.assertEqual(changing.calls, 1)
        self.assertEqual(receipt["lock_sha256"], _sha256(raw))
        self.assertEqual(receipt["checked_files"][0]["name"], "tokenizer.json")

    def test_identity_selection_rejects_weights_and_config_only(self):
        for names in (["model.safetensors"], ["tokenizer_config.json"],
                      ["tokenizer.json", "model.safetensors"]):
            with self.subTest(names=names):
                calls = []
                lock = self._lock([(name, 1, "a" * 64) for name in names], names)
                self._assert_code("IDENTITY_FILES_INVALID", lock,
                                  fs_probe=calls.append)
                self.assertEqual(calls, [])

    def test_vocab_and_merges_identity_allowed(self):
        entries = [self._entry("vocab.json", b"vocab"),
                   self._entry("merges.txt", b"merges")]
        receipt = self._verify(self._lock(entries, [e[0] for e in entries]))
        self.assertEqual(receipt["status"], "verified")

    def test_windows_alias_names_rejected_before_access(self):
        for name in ("tokenizer.json:hidden", "tokenizer.json.", "tokenizer.json ",
                     "NUL", "nul.txt", "CON", "COM1.json", "LPT².txt",
                     "a?b.json", "a|b.json", "a\x1fb.json"):
            with self.subTest(name=name):
                calls = []
                lock = self._lock([(name, 1, "a" * 64)], [name])
                self._assert_code("FILE_NAME_INVALID", lock, fs_probe=calls.append)
                self.assertEqual(calls, [])

    def test_windows_alias_snapshot_segments_rejected_before_access(self):
        for segment in ("models:stream", "models.", "models ", "NUL"):
            with self.subTest(segment=segment):
                calls = []
                lock = self._lock([("tokenizer.json", 1, "a" * 64)],
                                  ["tokenizer.json"],
                                  snap_rel=segment + "/" + sv.FIXED_REVISION)
                self._assert_code("SNAPSHOT_PATH_INVALID", lock,
                                  fs_probe=calls.append)
                self.assertEqual(calls, [])

    def test_huge_deadline_rejected_before_access(self):
        calls = []
        lock = self._lock([("tokenizer.json", 1, "a" * 64)], ["tokenizer.json"])
        self._assert_code("LIMIT_INVALID", lock,
                          bounds={"deadline_seconds": 10 ** 400},
                          fs_probe=calls.append)
        self.assertEqual(calls, [])

    def test_opened_descriptor_must_match_prechecked_file(self):
        entry = self._entry("tokenizer.json", b"fixture")
        lock = self._lock([entry], ["tokenizer.json"])
        real_fstat = os.fstat

        def replaced_descriptor(fd):
            info = real_fstat(fd)
            fields = {key: getattr(info, key) for key in
                      ("st_mode", "st_size", "st_mtime_ns", "st_ino", "st_dev")}
            fields["st_ino"] += 1
            return SimpleNamespace(**fields)

        with mock.patch.object(sv.os, "fstat", replaced_descriptor):
            self._assert_code("FILE_CHANGED_DURING_HASH", lock)

    def test_descriptor_changes_during_read_rejected(self):
        entry = self._entry("tokenizer.json", b"fixture")
        lock = self._lock([entry], ["tokenizer.json"])
        real_fstat = os.fstat
        calls = []

        def changed_descriptor(fd):
            info = real_fstat(fd)
            calls.append(fd)
            fields = {key: getattr(info, key) for key in
                      ("st_mode", "st_size", "st_mtime_ns", "st_ino", "st_dev")}
            if len(calls) == 2:
                fields["st_mtime_ns"] += 1
            return SimpleNamespace(**fields)

        with mock.patch.object(sv.os, "fstat", changed_descriptor):
            self._assert_code("FILE_CHANGED_DURING_HASH", lock)
        self.assertEqual(len(calls), 2)

    def test_path_escape_between_stat_and_open_rejected(self):
        entry = self._entry("tokenizer.json", b"fixture")
        lock = self._lock([entry], ["tokenizer.json"])
        real_realpath = os.path.realpath
        calls = []

        def switched_path(path, *args, **kwargs):
            if os.path.basename(os.fspath(path)) == "tokenizer.json":
                calls.append(path)
                if len(calls) > 1:
                    return str(self.tmp / "outside.json")
            return real_realpath(path, *args, **kwargs)

        with mock.patch.object(sv.os.path, "realpath", switched_path):
            self._assert_code("SYMLINK_ESCAPE", lock)

    def test_hardlink_alias_allowed_with_documented_path_scope(self):
        payload = b"fixture-only"
        outside = self.tmp / "outside-tokenizer.json"
        outside.write_bytes(payload)
        try:
            os.link(outside, self.snap / "tokenizer.json")
        except (OSError, NotImplementedError, AttributeError):
            self.skipTest("hardlink creation unavailable on this platform")
        lock = self._lock([("tokenizer.json", len(payload), _sha256(payload))],
                          ["tokenizer.json"])
        self.assertEqual(self._verify(lock)["status"], "verified")

    def test_tampered_file_same_size(self):
        entry = self._entry("tokenizer.json", b"original-bytes")
        lock = self._lock([entry], ["tokenizer.json"])
        (self.snap / "tokenizer.json").write_bytes(b"tampered-bytes")
        self.assertEqual(len(b"tampered-bytes"), len(b"original-bytes"))
        self._assert_code("FILE_HASH_MISMATCH", lock)

    def test_stale_file_size_mismatch(self):
        self._write("tokenizer.json", b"short")
        stale = ("tokenizer.json", 999, _sha256(b"short"))
        lock = self._lock([stale], ["tokenizer.json"])
        self._assert_code("FILE_SIZE_MISMATCH", lock)

    def test_file_grew_after_lock_declaration(self):
        self._write("tokenizer.json", b"0123456789")
        declared = ("tokenizer.json", 4, _sha256(b"0123"))
        lock = self._lock([declared], ["tokenizer.json"])
        self._assert_code("FILE_SIZE_MISMATCH", lock)

    # -- names -------------------------------------------------------------
    def test_duplicate_basename_case_insensitive(self):
        calls = []
        lock = self._lock(
            [("vocab.json", 4, "a" * 64), ("Vocab.json", 4, "b" * 64)],
            ["vocab.json"])
        self._assert_code("FILE_NAME_DUPLICATE", lock, root=self.root,
                          fs_probe=calls.append)
        self.assertEqual(calls, [])

    def test_name_traversal(self):
        calls = []
        lock = self._lock([("../escape.bin", 1, "a" * 64)], ["../escape.bin"])
        self._assert_code("FILE_NAME_TRAVERSAL", lock, fs_probe=calls.append)
        self.assertEqual(calls, [])

    def test_name_absolute(self):
        calls = []
        lock = self._lock([("C:escape.bin", 1, "a" * 64)], ["C:escape.bin"])
        self._assert_code("FILE_NAME_ABSOLUTE", lock, fs_probe=calls.append)
        self.assertEqual(calls, [])

    def test_snapshot_relative_path_traversal(self):
        calls = []
        lock = self._lock([("tokenizer.json", 1, "a" * 64)],
                          ["tokenizer.json"],
                          snap_rel="../outside/" + sv.FIXED_REVISION)
        self._assert_code("SNAPSHOT_PATH_TRAVERSAL", lock, fs_probe=calls.append)
        self.assertEqual(calls, [])

    def test_snapshot_relative_path_absolute(self):
        calls = []
        lock = self._lock([("tokenizer.json", 1, "a" * 64)],
                          ["tokenizer.json"],
                          snap_rel="C:/outside/" + sv.FIXED_REVISION)
        self._assert_code("SNAPSHOT_PATH_ABSOLUTE", lock, fs_probe=calls.append)
        self.assertEqual(calls, [])

    def test_snapshot_basename_must_be_revision(self):
        calls = []
        lock = self._lock([("tokenizer.json", 1, "a" * 64)],
                          ["tokenizer.json"],
                          snap_rel="models--test--tiny/snapshots/not-the-revision")
        self._assert_code("SNAPSHOT_REVISION_DIRNAME", lock, fs_probe=calls.append)
        self.assertEqual(calls, [])

    # -- missing / types / hashes -----------------------------------------
    def test_missing_required_file(self):
        lock = self._lock([("tokenizer.json", 5, _sha256(b"12345"))],
                          ["tokenizer.json"])
        self._assert_code("FILE_MISSING", lock)

    def test_false_types_and_bool(self):
        cases = [
            ("SIZE_INVALID", {"bytes": True}),
            ("SIZE_INVALID", {"bytes": "8"}),
            ("SIZE_INVALID", {"bytes": 1.5}),
            ("SIZE_INVALID", {"bytes": -1}),
            ("HASH_INVALID", {"sha256": "not-a-hash"}),
            ("HASH_INVALID", {"sha256": 123}),
        ]
        for code, override in cases:
            entry = {"name": "tokenizer.json", "bytes": 1, "sha256": "a" * 64}
            entry.update(override)
            lock = self._lock([(entry["name"], entry["bytes"], entry["sha256"])],
                              ["tokenizer.json"])
            self._assert_code(code, lock)

    def test_malformed_top_level_shapes(self):
        good = ("tokenizer.json", 1, "a" * 64)
        self._assert_code("LOCK_SCHEMA_INVALID",
                          self._lock([good], ["tokenizer.json"], schema="other"))
        self._assert_code("REVISION_MISMATCH",
                          self._lock([good], ["tokenizer.json"], revision="deadbeef"))
        bad_files = self._lock([good], ["tokenizer.json"])
        bad_files["files"] = "nope"
        self._assert_code("FILES_INVALID", bad_files)
        bad_identity = self._lock([good], ["tokenizer.json"])
        bad_identity["tokenizer_identity_files"] = ["absent.json"]
        self._assert_code("IDENTITY_FILES_INVALID", bad_identity)
        bad_identity["tokenizer_identity_files"] = []
        self._assert_code("IDENTITY_FILES_INVALID", bad_identity)

    def test_duplicate_json_keys_rejected(self):
        raw = (b'{"schema":"snapshot_lock.v1","schema":"snapshot_lock.v1",'
               b'"revision":"' + sv.FIXED_REVISION.encode("ascii") + b'"}')
        with self.assertRaises(sv.SnapshotVerificationError) as ctx:
            sv.verify_snapshot(
                raw,
                expected_lock_sha256=_sha256(raw),
                allowed_root=self.root,
                **self.bounds
            )
        self.assertEqual(ctx.exception.code, "LOCK_DUPLICATE_KEY")

    def test_invalid_lock_inputs(self):
        with self.assertRaises(sv.SnapshotVerificationError) as ctx:
            sv.verify_snapshot(
                "not bytes",
                expected_lock_sha256="a" * 64,
                allowed_root=self.root,
                **self.bounds
            )
        self.assertEqual(ctx.exception.code, "LOCK_BYTES_INVALID")
        raw = self._lock_bytes(self._lock([("tokenizer.json", 1, "a" * 64)],
                                          ["tokenizer.json"]))
        with self.assertRaises(sv.SnapshotVerificationError) as ctx:
            sv.verify_snapshot(
                raw,
                expected_lock_sha256="xyz",
                allowed_root=self.root,
                **self.bounds
            )
        self.assertEqual(ctx.exception.code, "EXPECTED_LOCK_HASH_INVALID")

    # -- finite file / byte limits ----------------------------------------
    def test_file_count_limit_before_access(self):
        calls = []
        lock = self._lock(
            [("a.bin", 1, "a" * 64), ("b.bin", 1, "b" * 64)],
            ["a.bin"])
        self._assert_code("FILE_COUNT_EXCEEDED", lock,
                          bounds={"max_files": 1}, fs_probe=calls.append)
        self.assertEqual(calls, [])

    def test_declared_file_oversize_rejected_before_access(self):
        calls = []
        lock = self._lock([("big.bin", 4096, "a" * 64)], ["big.bin"])
        self._assert_code("FILE_OVERSIZE", lock,
                          bounds={"max_file_bytes": 16},
                          fs_probe=calls.append)
        self.assertEqual(calls, [])

    def test_declared_total_budget_rejected_before_access(self):
        calls = []
        lock = self._lock(
            [("a.bin", 10, "a" * 64), ("b.bin", 10, "b" * 64)],
            ["a.bin"])
        self._assert_code("BUDGET_EXCEEDED", lock,
                          bounds={"max_total_bytes": 15},
                          fs_probe=calls.append)
        self.assertEqual(calls, [])

    def test_runtime_actual_file_larger_than_declared(self):
        self._write("tokenizer.json", b"Z" * 64)
        declared = ("tokenizer.json", 8, _sha256(b"Z" * 64))
        lock = self._lock([declared], ["tokenizer.json"])
        self._assert_code("FILE_SIZE_MISMATCH", lock,
                          bounds={"max_file_bytes": 128})

    def test_invalid_bounds(self):
        entry = ("tokenizer.json", 1, "a" * 64)
        lock = self._lock([entry], ["tokenizer.json"])
        self._assert_code("LIMIT_INVALID", lock, bounds={"max_files": 0})
        self._assert_code("LIMIT_INVALID", lock,
                          bounds={"deadline_seconds": float("inf")})
        self._assert_code("LIMIT_INVALID", lock, bounds={"max_total_bytes": True})

    # -- deadline ----------------------------------------------------------
    def test_deadline_exceeded(self):
        self._write("tokenizer.json", b"payload")
        entry = ("tokenizer.json", 7, _sha256(b"payload"))
        lock = self._lock([entry], ["tokenizer.json"])
        clock = _FakeClock([0.0, 1000.0])
        with self.assertRaises(sv.SnapshotVerificationError) as ctx:
            self._verify(lock, clock=clock, bounds={"deadline_seconds": 1.0})
        self.assertEqual(ctx.exception.code, "DEADLINE_EXCEEDED")

    # -- symlinks ----------------------------------------------------------
    def test_symlink_escape_rejected(self):
        secret = self.tmp / "outside-secret.bin"
        secret.write_bytes(b"outside-bytes")
        link = self.snap / "tokenizer.json"
        try:
            os.symlink(secret, link)
        except (OSError, NotImplementedError, AttributeError):
            self.skipTest("symlink creation unavailable on this platform")
        entry = ("tokenizer.json", len(b"outside-bytes"), _sha256(b"outside-bytes"))
        lock = self._lock([entry], ["tokenizer.json"])
        self._assert_code("SYMLINK_ESCAPE", lock)

    def test_symlink_within_allowed_root_allowed(self):
        target_dir = self.root / "models--test--tiny" / "targets"
        target_dir.mkdir(parents=True, exist_ok=True)
        payload = b"in-root-target"
        (target_dir / "vectors.bin").write_bytes(payload)
        link = self.snap / "tokenizer.json"
        try:
            os.symlink(target_dir / "vectors.bin", link)
        except (OSError, NotImplementedError, AttributeError):
            self.skipTest("symlink creation unavailable on this platform")
        entry = ("tokenizer.json", len(payload), _sha256(payload))
        lock = self._lock([entry], ["tokenizer.json"])
        receipt = self._verify(lock)
        self.assertTrue(receipt["checked_files"][0]["symlink_resolved"])

    def _simulated_link_patches(self, basename, resolved_target):
        """Make only ``basename`` look like a symlink resolving to target.

        Used because this Windows host denies SeCreateSymbolicLinkPrivilege;
        the fabricated files still live in the temporary directory.
        """
        real_lstat = os.lstat
        real_realpath = os.path.realpath

        def fake_lstat(path, *args, **kwargs):
            if os.path.basename(os.fspath(path)) == basename:
                info = real_lstat(path)
                mode = (info.st_mode & 0o170000) | stat.S_IFLNK
                return os.stat_result(
                    (mode, info.st_ino, info.st_dev, info.st_nlink,
                     info.st_uid, info.st_gid, info.st_size,
                     info.st_atime, info.st_mtime, info.st_ctime))
            return real_lstat(path, *args, **kwargs)

        def fake_realpath(path, *args, **kwargs):
            if os.path.basename(os.fspath(path)) == basename:
                return resolved_target
            return real_realpath(path, *args, **kwargs)

        return (
            mock.patch.object(sv.os, "lstat", fake_lstat),
            mock.patch.object(sv.os.path, "realpath", fake_realpath),
        )

    def test_symlink_escape_simulated_without_os_symlink(self):
        payload = b"payload-bytes"
        self._write("tokenizer.json", payload)
        entry = ("tokenizer.json", len(payload), _sha256(payload))
        lock = self._lock([entry], ["tokenizer.json"])
        outside = os.path.join(str(self.tmp), "outside", "real.bin")
        lstat_patch, realpath_patch = self._simulated_link_patches(
            "tokenizer.json", outside)
        with lstat_patch, realpath_patch:
            with self.assertRaises(sv.SnapshotVerificationError) as ctx:
                self._verify(lock)
        self.assertEqual(ctx.exception.code, "SYMLINK_ESCAPE")

    def test_snapshot_dir_symlink_escape_simulated(self):
        self._write("tokenizer.json", b"abc")
        entry = ("tokenizer.json", 3, _sha256(b"abc"))
        lock = self._lock([entry], ["tokenizer.json"])
        outside = os.path.join(str(self.tmp), "outside", sv.FIXED_REVISION)
        lstat_patch, realpath_patch = self._simulated_link_patches(
            sv.FIXED_REVISION, outside)
        with lstat_patch, realpath_patch:
            with self.assertRaises(sv.SnapshotVerificationError) as ctx:
                self._verify(lock)
        self.assertEqual(ctx.exception.code, "SNAPSHOT_OUTSIDE_ROOT")

    def test_snapshot_parent_traversal_rejected_before_resolution(self):
        outside = self.tmp / "elsewhere" / "snapshots" / sv.FIXED_REVISION
        outside.mkdir(parents=True)
        lock = self._lock([("tokenizer.json", 1, "a" * 64)],
                          ["tokenizer.json"],
                          snap_rel=os.path.join(
                              "..", "elsewhere", "snapshots", sv.FIXED_REVISION))
        # Relative path is internally consistent, so it must resolve beneath
        # the root; the ".." segment is rejected first.
        self._assert_code("SNAPSHOT_PATH_TRAVERSAL", lock)


if __name__ == "__main__":
    unittest.main(verbosity=2)
