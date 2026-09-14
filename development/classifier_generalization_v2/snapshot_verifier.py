"""Small stdlib-only snapshot-file verifier for prospective lock JSON.

This module verifies raw bytes only.  It does NOT load a tokenizer, construct a
model, parse tensors, read a vocabulary, recurse a directory, download, repair,
or import any third-party library.  A successful call returns a JSON-compatible
receipt that states, explicitly, that no tokenizer/model was loaded and that no
model execution is authorized.

Verification order (important):

1. Validate the raw lock bytes and the externally supplied expected lock SHA256.
2. Hash the raw lock bytes with SHA-256 and compare BEFORE any filesystem access.
3. Parse and validate the lock schema, the fixed revision, unique basename file
   entries with integer sizes and 64-hex SHA256 values, and the tokenizer
   identity selection.
4. Resolve the snapshot directory beneath the caller's allowed cache root using
   ``realpath`` containment; the snapshot basename must equal the fixed revision.
5. Stat each required file, reject oversize/mismatched/irregular entries before
   reading, then stream 8 MiB chunks under the caller's finite total-byte and
   deadline caps, checking the opened descriptor against pre/post path stats
   and the actual byte count.
6. Derive one aggregate tokenizer identity digest from the verified selected
   tokenizer-file entries.

All file paths are resolved with ``realpath`` and checked beneath the allowed
root before and after reading. Descriptor stats narrow replacement races;
they do not provide an atomic filesystem sandbox or race immunity. Hardlinks
are allowed: confinement describes resolved paths, not all storage aliases.
The digest authenticates bytes only against the caller-supplied lock pin;
recognized tokenizer filenames do not authenticate file semantics or origin.

No recursive search, no downloads, no repairs, no third-party imports.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat as stat_module
import time

__all__ = [
    "SnapshotVerificationError",
    "verify_snapshot",
    "FIXED_REVISION",
    "LOCK_SCHEMA",
    "RECEIPT_SCHEMA",
    "CHUNK_BYTES",
    "JOB_ID",
]

JOB_ID = "snapshot_verifier_supervisor_repair_20260914_0535"
LOCK_SCHEMA = "snapshot_lock.v1"
RECEIPT_SCHEMA = "snapshot_verifier_receipt.v1"
FIXED_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
CHUNK_BYTES = 8 * 1024 * 1024  # 8 MiB
_HEX64 = re.compile(r"[0-9a-fA-F]{64}")
_DRIVE = re.compile(r"^[A-Za-z]:")
_TOKENIZER_NAMES = frozenset({
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
})
_WINDOWS_DEVICES = frozenset({
    "CON", "PRN", "AUX", "NUL", "CLOCK$", "CONIN$", "CONOUT$",
    *("COM" + digit for digit in "123456789¹²³"),
    *("LPT" + digit for digit in "123456789¹²³"),
})


class SnapshotVerificationError(Exception):
    """Raised for every rejected lock, path, file or limit condition."""

    def __init__(self, code, detail=None):
        self.code = code
        self.detail = detail
        message = code if detail is None else "%s: %s" % (code, detail)
        super().__init__(message)


def _fail(code, detail=None):
    raise SnapshotVerificationError(code, detail)


class _DuplicateKey(ValueError):
    pass


def _reject_duplicate_keys(pairs):
    seen = set()
    result = {}
    for key, value in pairs:
        if key in seen:
            raise _DuplicateKey(key)
        seen.add(key)
        result[key] = value
    return result


def _is_within(path, root):
    try:
        candidate = os.path.normcase(os.path.abspath(path))
        base = os.path.normcase(os.path.abspath(root))
        return os.path.commonpath([candidate, base]) == base
    except ValueError:
        return False


def _invalid_windows_component(value):
    return (value.endswith((".", " "))
            or any(ord(char) < 32 or char in '<>:"|?*' for char in value)
            or value.split(".", 1)[0].rstrip(" ").upper() in _WINDOWS_DEVICES)


def _validate_relative_path(value, label):
    if type(value) is not str or value == "":
        _fail("SNAPSHOT_PATH_INVALID", label)
    if "\x00" in value:
        _fail("SNAPSHOT_PATH_INVALID", label)
    normalized = value.replace("\\", "/")
    if normalized.startswith("/") or _DRIVE.match(normalized):
        _fail("SNAPSHOT_PATH_ABSOLUTE", label)
    segments = normalized.split("/")
    for segment in segments:
        if segment in ("", ".", ".."):
            _fail("SNAPSHOT_PATH_TRAVERSAL", label)
        if _invalid_windows_component(segment):
            _fail("SNAPSHOT_PATH_INVALID", label)
    return segments


def _validate_basename(value):
    if type(value) is not str or value == "":
        _fail("FILE_NAME_INVALID", "name")
    if "\x00" in value:
        _fail("FILE_NAME_INVALID", "name")
    if "/" in value or "\\" in value:
        _fail("FILE_NAME_TRAVERSAL", value)
    if value in (".", ".."):
        _fail("FILE_NAME_TRAVERSAL", value)
    if _DRIVE.match(value):
        _fail("FILE_NAME_ABSOLUTE", value)
    if _invalid_windows_component(value):
        _fail("FILE_NAME_INVALID", value)
    return value


def _positive_int(value, label):
    if type(value) is not int or value <= 0:
        _fail("LIMIT_INVALID", label)
    return value


class _Verifier:
    def __init__(self, clock, fs_probe, max_files, max_file_bytes,
                 max_total_bytes, deadline_seconds):
        self.clock = clock
        self.fs_probe = fs_probe
        self.max_files = max_files
        self.max_file_bytes = max_file_bytes
        self.max_total_bytes = max_total_bytes
        self.deadline_seconds = deadline_seconds
        self.deadline = None
        self.root_real = None
        self.fs_calls = 0

    def _touch(self, path):
        """Record one filesystem-touching intent (proves zero-access phases)."""
        self.fs_calls += 1
        if self.fs_probe is not None:
            self.fs_probe(os.fspath(path))

    def _check_deadline(self):
        if self.clock() > self.deadline:
            _fail("DEADLINE_EXCEEDED", "deadline_seconds=%r" % (self.deadline_seconds,))

    # -- lock schema -------------------------------------------------------
    def validate_lock(self, lock):
        if type(lock) is not dict:
            _fail("LOCK_SCHEMA_INVALID", "top level")
        if lock.get("schema") != LOCK_SCHEMA:
            _fail("LOCK_SCHEMA_INVALID", "schema")
        revision = lock.get("revision")
        if type(revision) is not str or revision != FIXED_REVISION:
            _fail("REVISION_MISMATCH", "revision")
        segments = _validate_relative_path(
            lock.get("snapshot_relative_path"), "snapshot_relative_path")
        if segments[-1] != FIXED_REVISION:
            _fail("SNAPSHOT_REVISION_DIRNAME", "snapshot basename")
        raw_files = lock.get("files")
        if type(raw_files) is not list or len(raw_files) == 0:
            _fail("FILES_INVALID", "files")
        if len(raw_files) > self.max_files:
            _fail("FILE_COUNT_EXCEEDED", "files=%d" % len(raw_files))
        seen = {}
        files = []
        declared_total = 0
        for item in raw_files:
            if type(item) is not dict:
                _fail("FILE_ENTRY_INVALID", "entry")
            name = _validate_basename(item.get("name"))
            key = name.casefold()
            if key in seen:
                _fail("FILE_NAME_DUPLICATE", name)
            seen[key] = name
            size = item.get("bytes")
            if type(size) is not int or size < 0:
                _fail("SIZE_INVALID", name)
            if size > self.max_file_bytes:
                _fail("FILE_OVERSIZE", name)
            digest = item.get("sha256")
            if type(digest) is not str or _HEX64.fullmatch(digest) is None:
                _fail("HASH_INVALID", name)
            declared_total += size
            files.append({"name": name, "bytes": size,
                          "sha256": digest.lower()})
        if declared_total > self.max_total_bytes:
            _fail("BUDGET_EXCEEDED", "declared_total=%d" % declared_total)
        raw_identity = lock.get("tokenizer_identity_files")
        if type(raw_identity) is not list or len(raw_identity) == 0:
            _fail("IDENTITY_FILES_INVALID", "tokenizer_identity_files")
        if len(raw_identity) > self.max_files:
            _fail("IDENTITY_FILES_INVALID", "too many")
        identity_keys = []
        for name in raw_identity:
            if type(name) is not str:
                _fail("IDENTITY_FILES_INVALID", "type")
            key = name.casefold()
            if key not in seen:
                _fail("IDENTITY_FILES_INVALID", name)
            if key in identity_keys:
                _fail("IDENTITY_FILES_INVALID", "duplicate")
            if key not in _TOKENIZER_NAMES:
                _fail("IDENTITY_FILES_INVALID", "not a tokenizer filename")
            identity_keys.append(key)
        if ("tokenizer.json" not in identity_keys
                and not {"vocab.json", "merges.txt"}.issubset(identity_keys)):
            _fail("IDENTITY_FILES_INVALID", "missing tokenizer data files")
        return {"segments": segments, "files": files, "seen": seen,
                "identity_keys": identity_keys,
                "declared_total": declared_total}

    # -- paths -------------------------------------------------------------
    def resolve_snapshot(self, allowed_root, segments):
        root_path = os.fspath(allowed_root)
        if not isinstance(root_path, str):
            _fail("ROOT_INVALID", "allowed_root")
        self._touch(root_path)
        if not os.path.isdir(root_path):
            _fail("ROOT_NOT_DIRECTORY")
        root_real = os.path.realpath(root_path)
        self.root_real = root_real
        candidate = os.path.join(root_path, *segments)
        self._touch(candidate)
        if not os.path.isdir(candidate):
            _fail("SNAPSHOT_NOT_DIRECTORY")
        candidate_real = os.path.realpath(candidate)
        if not _is_within(candidate_real, root_real):
            _fail("SNAPSHOT_OUTSIDE_ROOT")
        return candidate_real

    # -- files -------------------------------------------------------------
    @staticmethod
    def _stat_signature(info):
        return (info.st_mode, info.st_size, info.st_mtime_ns,
                info.st_ino, info.st_dev)

    def _assert_path_stable(self, candidate, expected):
        self._touch(candidate)
        current = os.path.realpath(candidate)
        if not _is_within(current, self.root_real):
            _fail("SYMLINK_ESCAPE", os.path.basename(candidate))
        if os.path.normcase(current) != os.path.normcase(expected):
            _fail("FILE_CHANGED_DURING_HASH", os.path.basename(candidate))

    def hash_file(self, snapshot_dir, entry, budget):
        name = entry["name"]
        candidate = os.path.join(snapshot_dir, name)
        self._touch(candidate)
        try:
            link_info = os.lstat(candidate)
        except FileNotFoundError:
            _fail("FILE_MISSING", name)
        except OSError:
            _fail("FILE_STAT_ERROR", name)
        read_path = os.path.realpath(candidate)
        if not _is_within(read_path, self.root_real):
            _fail("SYMLINK_ESCAPE", name)
        symlink_resolved = stat_module.S_ISLNK(link_info.st_mode)
        self._touch(read_path)
        try:
            pre = os.stat(read_path)
        except OSError:
            _fail("FILE_STAT_ERROR", name)
        if not stat_module.S_ISREG(pre.st_mode):
            _fail("FILE_NOT_REGULAR", name)
        declared = entry["bytes"]
        if pre.st_size != declared:
            _fail("FILE_SIZE_MISMATCH", name)
        if budget["bytes"] + declared > self.max_total_bytes:
            _fail("BUDGET_EXCEEDED", name)
        self._check_deadline()
        hasher = hashlib.sha256()
        read_total = 0
        try:
            self._assert_path_stable(candidate, read_path)
            with open(read_path, "rb") as handle:
                opened = os.fstat(handle.fileno())
                if (not stat_module.S_ISREG(opened.st_mode)
                        or self._stat_signature(opened) != self._stat_signature(pre)):
                    _fail("FILE_CHANGED_DURING_HASH", name)
                self._assert_path_stable(candidate, read_path)
                while True:
                    self._check_deadline()
                    chunk = handle.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    read_total += len(chunk)
                    if read_total > declared:
                        _fail("FILE_CHANGED_DURING_HASH", name)
                    hasher.update(chunk)
                    budget["bytes"] += len(chunk)
                    if budget["bytes"] > self.max_total_bytes:
                        _fail("BUDGET_EXCEEDED", name)
                closed = os.fstat(handle.fileno())
                if self._stat_signature(closed) != self._stat_signature(opened):
                    _fail("FILE_CHANGED_DURING_HASH", name)
                self._assert_path_stable(candidate, read_path)
        except SnapshotVerificationError:
            raise
        except OSError:
            _fail("FILE_READ_ERROR", name)
        self._check_deadline()
        self._touch(read_path)
        try:
            post = os.stat(read_path)
        except OSError:
            _fail("FILE_STAT_ERROR", name)
        if self._stat_signature(post) != self._stat_signature(pre):
            _fail("FILE_CHANGED_DURING_HASH", name)
        if read_total != declared:
            _fail("FILE_CHANGED_DURING_HASH", name)
        observed = hasher.hexdigest()
        if observed != entry["sha256"]:
            _fail("FILE_HASH_MISMATCH", name)
        return {
            "name": name,
            "bytes": declared,
            "sha256": observed,
            "bytes_read": read_total,
            "verified": True,
            "mtime_ns": pre.st_mtime_ns,
            "symlink_resolved": symlink_resolved,
        }


def _aggregate_identity(entries):
    hasher = hashlib.sha256()
    for entry in entries:
        hasher.update(entry["name"].encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(str(entry["bytes"]).encode("ascii"))
        hasher.update(b"\x00")
        hasher.update(entry["sha256"].encode("ascii"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def verify_snapshot(lock_bytes, *, expected_lock_sha256, allowed_root,
                    max_files, max_file_bytes, max_total_bytes,
                    deadline_seconds, clock=time.monotonic, fs_probe=None):
    """Verify one prospective snapshot lock against fabricated or real bytes.

    ``lock_bytes`` is the raw prospective lock JSON.  ``expected_lock_sha256``
    is the externally supplied lock digest and is checked before any filesystem
    access.  ``fs_probe`` is an optional callable invoked with each path the
    verifier is about to touch; tests use it to prove a zero-access phase.
    ``clock`` is an injectable monotonic seconds source.  Returns a
    JSON-compatible receipt; raises ``SnapshotVerificationError`` on rejection.
    """
    if not isinstance(lock_bytes, (bytes, bytearray)):
        _fail("LOCK_BYTES_INVALID", type(lock_bytes).__name__)
    if type(expected_lock_sha256) is not str or \
            _HEX64.fullmatch(expected_lock_sha256) is None:
        _fail("EXPECTED_LOCK_HASH_INVALID")
    max_files = _positive_int(max_files, "max_files")
    max_file_bytes = _positive_int(max_file_bytes, "max_file_bytes")
    max_total_bytes = _positive_int(max_total_bytes, "max_total_bytes")
    if type(deadline_seconds) not in (int, float) or \
            isinstance(deadline_seconds, bool):
        _fail("LIMIT_INVALID", "deadline_seconds")
    try:
        deadline_seconds = float(deadline_seconds)
    except (OverflowError, ValueError):
        _fail("LIMIT_INVALID", "deadline_seconds")
    if not math.isfinite(deadline_seconds) or deadline_seconds <= 0:
        _fail("LIMIT_INVALID", "deadline_seconds")

    # Step 2: lock digest before any filesystem access.
    # Snapshot once: digest, parse and receipt must bind the exact same bytes.
    raw = bytes(lock_bytes)
    lock_digest = hashlib.sha256(raw).hexdigest()
    if lock_digest != expected_lock_sha256.lower():
        _fail("LOCK_DIGEST_MISMATCH")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        _fail("LOCK_ENCODING_INVALID")
    try:
        lock = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except _DuplicateKey as exc:
        _fail("LOCK_DUPLICATE_KEY", str(exc))
    except ValueError:
        _fail("LOCK_JSON_INVALID")

    verifier = _Verifier(clock, fs_probe, max_files, max_file_bytes,
                         max_total_bytes, deadline_seconds)
    start = clock()
    verifier.deadline = start + float(deadline_seconds)
    if not math.isfinite(verifier.deadline):
        _fail("LIMIT_INVALID", "absolute deadline")
    validated = verifier.validate_lock(lock)
    verifier._check_deadline()

    snapshot_real = verifier.resolve_snapshot(allowed_root,
                                              validated["segments"])
    budget = {"bytes": 0}
    checked = []
    for entry in validated["files"]:
        checked.append(verifier.hash_file(snapshot_real, entry, budget))
    verifier._check_deadline()

    by_key = {item["name"].casefold(): item for item in checked}
    identity_entries = [by_key[key] for key in validated["identity_keys"]]
    aggregate = _aggregate_identity(identity_entries)
    elapsed_ms = int(round((clock() - start) * 1000.0))

    return {
        "schema": RECEIPT_SCHEMA,
        "job_id": JOB_ID,
        "status": "verified",
        "revision": FIXED_REVISION,
        "lock_sha256": lock_digest,
        "snapshot_relative_path": "/".join(validated["segments"]),
        "snapshot_realpath": snapshot_real,
        "allowed_root_realpath": verifier.root_real,
        "checked_files": checked,
        "aggregate_tokenizer_identity_sha256": aggregate,
        "tokenizer_identity_files": [item["name"] for item in identity_entries],
        "bytes_hashed_total": budget["bytes"],
        "limits": {
            "max_files": max_files,
            "max_file_bytes": max_file_bytes,
            "max_total_bytes": max_total_bytes,
            "deadline_seconds": float(deadline_seconds),
        },
        "elapsed_ms": elapsed_ms,
        "tokenizer_loaded": False,
        "model_loaded": False,
        "weights_tensor_parsed": False,
        "model_execution_authorized": False,
        "scientific_execution_authorized": False,
        "recursive_search_performed": False,
        "network_access_performed": False,
    }
