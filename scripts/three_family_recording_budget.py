"""Scoped, fail-closed evidence recording; no model or scientific computations.

This is a cooperative recording boundary for the bound runner, not an operating
system sandbox. Participating processes use the same byte lock and never pass
namespace writer descriptors to child processes. Finalization additionally
requires the supervisor's explicit process/thread-quiescence assertion.
"""

from __future__ import annotations

import builtins
import contextlib
import hashlib
import io
import json
import os
import re
import stat
import threading
import time
import uuid
from pathlib import Path

_OPEN, _IO_OPEN = builtins.open, io.open
_OS_OPEN, _REPLACE, _RENAME = os.open, os.replace, os.rename
_UNLINK, _REMOVE, _TRUNCATE = os.unlink, os.remove, os.truncate
_LINK, _SYMLINK = os.link, os.symlink
_REGISTRY_LOCK = threading.RLock()
_ROOT_LOCKS, _STATE_LOCKS, _STOP_EVENTS, _HANDLES, _SCOPES = {}, {}, {}, {}, {}
_REGION_DEPTH = threading.local()
_REGION_STREAMS = threading.local()
_POSIX_LOCK_STREAMS = {}
STATE_BYTES = 4096
STATE_NAME, LOCK_NAME, INVENTORY_NAME = (
    "recording_state.json",
    "recording.lock",
    "FINAL_INVENTORY.json",
)
MIB = 1024**2
DEFAULT_QUOTAS = {
    "logits": 214616520,
    "rows": 226492416,
    "updates": 67108864,
    "workerlog": 4 * MIB,
    "metadata": 2 * MIB,
    "events": MIB,
    "endpoint": MIB,
    "final": 5 * MIB,
    "receipts": MIB,
    "control": MIB,
    "scratch": MIB,
    "total": 512 * MIB,
}
NAMES = {
    "rows.jsonl": "rows",
    "updates.jsonl": "updates",
    "worker.log": "workerlog",
    **dict.fromkeys(
        (
            "preregistration.json",
            "RUN_STARTED.json",
            "WORKER_CLAIM.json",
            "storage_preflight.json",
            "runtime.json",
            "analysis.json",
            "result.json",
            "RUN_STATUS.json",
            "INVALID.json",
            "VERIFICATION_FAILURE.json",
        ),
        "metadata",
    ),
    **dict.fromkeys(
        ("forward_events.jsonl", "derivative_events.jsonl", "skip_events.jsonl"), "events"
    ),
    "endpoint.json": "endpoint",
    **dict.fromkeys(
        (
            "verification.json",
            "PILOT_REPORT.md",
            "comply_vector.json",
            "candidate_freeze.json",
            "CLOSEOUT.md",
            "CLOSEOUT.json",
            "closeout.json",
        ),
        "final",
    ),
    **dict.fromkeys(
        (
            "capture_receipt.json",
            "RECORDING_FAILURE.json",
            INVENTORY_NAME,
            "checksums.json",
            "CHECKSUMS.json",
        ),
        "receipts",
    ),
    STATE_NAME: "control",
    LOCK_NAME: "control",
    "recording_scratch.tmp": "scratch",
}


class BudgetError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _under(path, root):
    return path == root or root in path.parents


def _path(value):
    return Path(os.path.abspath(os.fsdecode(value)))


def _reparse(path):
    value = path.lstat()
    return stat.S_ISLNK(value.st_mode) or bool(getattr(value, "st_file_attributes", 0) & 0x400)


def _writing(mode):
    return any(flag in mode for flag in "wax+")


def _borrow_lock_stream(path, initialize):
    if os.name == "nt":
        return _OPEN(path, "a+b" if initialize else "r+b", buffering=0)
    # POSIX record locks are process-scoped: closing ANY fd for this file releases
    # every range. Keep one fd alive until the last data/state range has unlocked.
    with _REGISTRY_LOCK:
        key = str(path)
        if key not in _POSIX_LOCK_STREAMS:
            _POSIX_LOCK_STREAMS[key] = [_OPEN(path, "a+b" if initialize else "r+b", buffering=0), 0]
        entry = _POSIX_LOCK_STREAMS[key]
        entry[1] += 1
        return entry[0]


def _return_lock_stream(path, stream):
    if os.name == "nt":
        stream.close()
        return
    with _REGISTRY_LOCK:
        entry = _POSIX_LOCK_STREAMS[str(path)]
        entry[1] -= 1
        if not entry[1]:
            del _POSIX_LOCK_STREAMS[str(path)]
            stream.close()


def _find_budget(file):
    if isinstance(file, int):
        return None
    path = _path(file)
    resolved = path.resolve()
    with _REGISTRY_LOCK:
        matches = [
            budget
            for budget, _ in _SCOPES.values()
            if _under(path, budget.root) or _under(resolved, budget.root)
        ]
    return max(matches, key=lambda budget: len(budget.root.parts)) if matches else None


def _dispatch_open(
    file,
    mode="r",
    buffering=-1,
    encoding=None,
    errors=None,
    newline=None,
    closefd=True,
    opener=None,
):
    if isinstance(file, int) and _writing(mode):
        raise BudgetError("RAW_WRITER_DESCRIPTOR_DENIED")
    budget = _find_budget(file)
    if budget is not None and _writing(mode):
        if not closefd or opener is not None:
            budget.fault("RAW_WRITER_DESCRIPTOR_DENIED")
            raise BudgetError("RAW_WRITER_DESCRIPTOR_DENIED")
        return budget.open_writer(file, mode, encoding=encoding, errors=errors, newline=newline)
    return _IO_OPEN(file, mode, buffering, encoding, errors, newline, closefd, opener)


def _dispatch_os_open(path, flags, mode=0o777, *, dir_fd=None):
    if dir_fd is not None:
        if not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            return _OS_OPEN(path, flags, mode, dir_fd=dir_fd)
        raise BudgetError("RAW_RELATIVE_DESCRIPTOR_DENIED")
    budget = _find_budget(path)
    if budget is not None and flags & (
        os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    ):
        budget.fault("RAW_WRITER_DESCRIPTOR_DENIED")
        raise BudgetError("RAW_WRITER_DESCRIPTOR_DENIED")
    return _OS_OPEN(path, flags, mode)


def _mutation(original, *paths, **kwargs):
    if (
        kwargs.get("dir_fd") is not None
        or kwargs.get("src_dir_fd") is not None
        or kwargs.get("dst_dir_fd") is not None
    ):
        raise BudgetError("RAW_RELATIVE_MUTATION_DENIED")
    for path in (*paths, *(kwargs[key] for key in ("path", "src", "dst") if key in kwargs)):
        if isinstance(path, (str, bytes, os.PathLike)):
            budget = _find_budget(path)
            if budget is not None:
                budget.fault("UNINSTRUMENTED_MUTATION_DENIED")
                raise BudgetError("UNINSTRUMENTED_MUTATION_DENIED")
    return original(*paths, **kwargs)


class Budget:
    def __init__(self, root, quotas=None, *, initialize=True):
        candidate = _path(root)
        if not candidate.is_dir() or candidate == candidate.parent or _reparse(candidate):
            raise BudgetError("EXPLICIT_REAL_NAMESPACE_REQUIRED")
        self.root = candidate.resolve()
        self.quotas = dict(DEFAULT_QUOTAS)
        for key, value in (quotas or {}).items():
            if (
                key not in self.quotas
                or type(value) is not int
                or not 0 < value <= DEFAULT_QUOTAS[key]
            ):
                raise BudgetError("INVALID_QUOTA")
            self.quotas[key] = value
        if min(self.quotas["control"], self.quotas["total"]) < STATE_BYTES + 1:
            raise BudgetError("CONTROL_RESERVE_TOO_SMALL")
        with _REGISTRY_LOCK:
            self._thread_lock = _ROOT_LOCKS.setdefault(str(self.root), threading.RLock())
            self._state_lock = _STATE_LOCKS.setdefault(str(self.root), threading.RLock())
            self._stop_event = _STOP_EVENTS.setdefault(str(self.root), threading.Event())
            _HANDLES.setdefault(str(self.root), 0)
        self._owner_token = None
        existing = [(self.root / name).exists() for name in (STATE_NAME, LOCK_NAME)]
        if any(existing) != all(existing) or not initialize and not all(existing):
            raise BudgetError("EXISTING_EXACT_CONTROLS_REQUIRED")
        creating = not any(existing)
        if creating and {path.name for path in self.root.iterdir()} - {"preregistration.json"}:
            raise BudgetError("PROSPECTIVE_NAMESPACE_ONLY")
        # Attach never creates/repairs controls. Only the prospective supervisor initializes.
        with self._locked(initialize=creating):
            if creating:
                self._write_state(
                    {
                        "version": 1,
                        "phase": "RUNNING",
                        "fault_code": None,
                        "owner_pid": None,
                        "owner_token": None,
                        "prefix_metadata": {},
                    },
                    create=True,
                )
            self._read_state()
            self._check(self._scan())

    @contextlib.contextmanager
    def _locked(self, initialize=False):
        with self._region_locked(0, self._thread_lock, 10, initialize):
            yield

    @contextlib.contextmanager
    def _state_locked(self):
        # A separate byte-range lock keeps fault/status independent of raw-data fsync.
        # Both OS APIs permit locking beyond EOF; the lockfile remains exactly one byte.
        with self._region_locked(1, self._state_lock, 0.25, False):
            yield

    @contextlib.contextmanager
    def _region_locked(self, offset, thread_lock, timeout, initialize):
        if not thread_lock.acquire(timeout=timeout):
            raise BudgetError("RECORDING_LOCK_TIMEOUT")
        depths = getattr(_REGION_DEPTH, "values", {})
        _REGION_DEPTH.values = depths
        streams = getattr(_REGION_STREAMS, "values", {})
        _REGION_STREAMS.values = streams
        key = (str(self.root), offset)
        try:
            if depths.get(key, 0):
                depths[key] += 1
                try:
                    yield
                finally:
                    depths[key] -= 1
                return
            lock_path = self.root / LOCK_NAME
            if lock_path.exists() and (_reparse(lock_path) or lock_path.stat().st_nlink != 1):
                raise BudgetError("UNSAFE_CONTROL_FILE")
            stream = _borrow_lock_stream(lock_path, initialize)
            acquired = False
            try:
                deadline = time.monotonic() + timeout
                while not acquired:
                    try:
                        if os.name == "nt":
                            import msvcrt

                            stream.seek(offset)
                            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                        else:
                            import fcntl

                            fcntl.lockf(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB, 1, offset)
                        acquired = True
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise BudgetError("RECORDING_LOCK_TIMEOUT") from None
                        time.sleep(0.005)
                size = os.fstat(stream.fileno()).st_size
                if initialize and size == 0:
                    stream.seek(0)
                    stream.write(b"\0")
                    os.fsync(stream.fileno())
                elif size != 1:
                    raise BudgetError("INVALID_RECORDING_LOCK")
                if offset == 0:
                    stream.seek(0)
                    if stream.read(1) != b"\0":
                        raise BudgetError("INVALID_RECORDING_LOCK")
                depths[key] = 1
                streams[key] = stream
                try:
                    yield
                finally:
                    del depths[key]
                    del streams[key]
            finally:
                if acquired:
                    if os.name == "nt":
                        import msvcrt

                        stream.seek(offset)
                        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.lockf(stream.fileno(), fcntl.LOCK_UN, 1, offset)
                _return_lock_stream(lock_path, stream)
        finally:
            thread_lock.release()

    def _write_state(self, state, *, create=False):
        with self._state_locked():
            return self._write_state_unlocked(state, create=create)

    def _write_state_unlocked(self, state, *, create=False):
        encoded = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "ascii"
        )
        if len(encoded) >= STATE_BYTES:
            raise BudgetError("STATE_ENCODING_LIMIT")
        data = encoded + b" " * (STATE_BYTES - len(encoded) - 1) + b"\n"
        path = self.root / STATE_NAME
        if not create and not path.exists():
            raise BudgetError("INVALID_RECORDING_STATE")
        if path.exists() and (
            _reparse(path) or path.stat().st_nlink != 1 or path.stat().st_size != STATE_BYTES
        ):
            raise BudgetError("INVALID_RECORDING_STATE")
        with _OPEN(path, "xb" if create else "r+b", buffering=0) as stream:
            if stream.write(data) != len(data):
                raise BudgetError("STATE_PARTIAL_WRITE")
            os.fsync(stream.fileno())

    def _read_state(self):
        with self._state_locked():
            return self._read_state_unlocked()

    def _read_state_unlocked(self):
        path = self.root / STATE_NAME
        if _reparse(path) or path.stat().st_nlink != 1 or path.stat().st_size != STATE_BYTES:
            raise BudgetError("INVALID_RECORDING_STATE")
        try:
            with _OPEN(path, "rb") as stream:
                raw = stream.read(STATE_BYTES)
                state = json.loads(raw)
            if (
                set(state)
                != {"version", "phase", "fault_code", "owner_pid", "owner_token", "prefix_metadata"}
                or type(state["version"]) is not int
                or state["version"] != 1
                or state["phase"] not in {"RUNNING", "FINALIZING", "FAILED", "SEALED"}
                or (
                    state["fault_code"] is not None
                    and not re.fullmatch(r"[A-Z0-9_]{1,64}", state["fault_code"])
                )
                or not isinstance(state["prefix_metadata"], dict)
                or len(state["prefix_metadata"]) > 12
                or any(
                    not isinstance(k, str)
                    or not re.fullmatch(r"[a-z_]{1,40}", k)
                    or not (
                        v is None
                        or type(v) is bool
                        or type(v) is int
                        and 0 <= v < 2**63
                        or isinstance(v, str)
                        and re.fullmatch(r"[a-f0-9]{64}", v)
                    )
                    for k, v in state["prefix_metadata"].items()
                )
                or state["phase"] == "RUNNING"
                and (state["owner_pid"] is not None or state["owner_token"] is not None)
                or (
                    state["phase"] != "RUNNING"
                    and (
                        type(state["owner_pid"]) is not int
                        or state["owner_pid"] <= 0
                        or not isinstance(state["owner_token"], str)
                        or not re.fullmatch(r"[a-f0-9]{32}", state["owner_token"])
                    )
                )
            ):
                raise ValueError
            canonical = json.dumps(
                state, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("ascii")
            if raw != canonical + b" " * (STATE_BYTES - len(canonical) - 1) + b"\n":
                raise ValueError
        except (ValueError, TypeError, KeyError, UnicodeError):
            raise BudgetError("INVALID_RECORDING_STATE") from None
        return state

    def _fault_locked(self, code, metadata=None):
        self._stop_event.set()
        with self._state_locked():
            return self._fault_state_unlocked(code, metadata)

    def _fault_state_unlocked(self, code, metadata=None):
        state = self._read_state()
        if state["phase"] == "SEALED":
            raise BudgetError("RECORDING_SEALED")
        if not isinstance(code, str) or not re.fullmatch(r"[A-Z0-9_]{1,64}", code):
            code = "INVALID_FAULT_CODE"
        if state["fault_code"] is None:
            state["fault_code"] = code
            if metadata is not None:
                if (
                    not isinstance(metadata, dict)
                    or len(metadata) > 12
                    or any(
                        not isinstance(k, str)
                        or not re.fullmatch(r"[a-z_]{1,40}", k)
                        or not (
                            v is None
                            or type(v) is bool
                            or type(v) is int
                            and 0 <= v < 2**63
                            or isinstance(v, str)
                            and re.fullmatch(r"[a-f0-9]{64}", v)
                        )
                        for k, v in metadata.items()
                    )
                ):
                    raise BudgetError("INVALID_FAULT_METADATA")
                state["prefix_metadata"] = metadata
            self._write_state(state)
        return state["fault_code"]

    def fault(self, code, metadata=None):
        self._stop_event.set()
        return self._fault_locked(code, metadata)

    @property
    def fault_code(self):
        return self._read_state()["fault_code"]

    def status(self):
        return self._read_state()

    def _hash_file(self, name):
        if name == LOCK_NAME:
            # Windows requires reading locked bytes through their owning handle.
            # Hash actual content, not an assumed constant or a second open handle.
            stream = _REGION_STREAMS.values[(str(self.root), 0)]
            position = stream.tell()
            stream.seek(0)
            raw = stream.read(1)
            stream.seek(position)
            if raw != b"\0":
                raise BudgetError("INVALID_RECORDING_LOCK")
            return hashlib.sha256(raw).hexdigest()
        digest = hashlib.sha256()
        with _OPEN(self.root / name, "rb") as stream:
            for block in iter(lambda: stream.read(65536), b""):
                digest.update(block)
        return digest.hexdigest()

    def snapshot(self):
        with self._locked():
            return {**self._read_state(), "sizes": self._scan()}

    def _target(self, path):
        path = Path(path)
        path = _path(path if path.is_absolute() else self.root / path)
        if (
            not _under(path, self.root)
            or path == self.root
            or not _under(path.resolve(), self.root)
        ):
            raise BudgetError("NAMESPACE_ESCAPE")
        for part in (path, *path.parents):
            if part == self.root:
                break
            if part.exists() and (_reparse(part) or part.is_file() and part.stat().st_nlink != 1):
                raise BudgetError("UNSAFE_OUTPUT_LINK")
        return path

    def category(self, name):
        if name in NAMES:
            return NAMES[name]
        match = re.fullmatch(r"logits/(\d{2,3})\.f32\.zlib", name)
        if match and 1 <= int(match[1]) <= 216 and match[1] == f"{int(match[1]):02d}":
            return "logits"
        if name.startswith(".") and name.endswith(".tmp"):
            target = name[1:-4]
            if (
                target in NAMES
                and NAMES[target] in {"metadata", "endpoint", "final", "receipts"}
                and target != INVENTORY_NAME
            ):
                return NAMES[target]
        raise BudgetError("UNKNOWN_OUTPUT_ARTIFACT")

    def _scan(self):
        sizes = {}
        for folder, directories, files in os.walk(self.root, followlinks=False):
            for name in directories:
                path = Path(folder) / name
                if _reparse(path) or path.relative_to(self.root).as_posix() != "logits":
                    raise BudgetError("UNKNOWN_OR_UNSAFE_OUTPUT_DIRECTORY")
            for name in files:
                path = self._target(Path(folder) / name)
                relative = path.relative_to(self.root).as_posix()
                self.category(relative)
                sizes[relative] = path.stat().st_size
        return sizes

    def _check(self, sizes):
        totals = dict.fromkeys((key for key in self.quotas if key != "total"), 0)
        for name, size in sizes.items():
            group = self.category(name)
            totals[group] += size
            limit = min(self.quotas[group], 993595) if group == "logits" else self.quotas[group]
            if size > limit:
                raise BudgetError("ARTIFACT_BYTE_CAP")
        if any(totals[key] > self.quotas[key] for key in totals):
            raise BudgetError("CATEGORY_BYTE_CAP")
        if sum(totals.values()) > self.quotas["total"]:
            raise BudgetError("NAMESPACE_BYTE_CAP")
        return totals

    def _permission(self, name, final):
        state = self._read_state()
        group = self.category(name)
        if name in (STATE_NAME, LOCK_NAME, INVENTORY_NAME):
            raise BudgetError("RESERVED_CONTROL_ARTIFACT")
        if state["phase"] == "SEALED":
            raise BudgetError("RECORDING_SEALED")
        if final:
            allowed_groups = (
                {"metadata", "final", "receipts"}
                if state["phase"] == "FINALIZING"
                else {"receipts"}
            )
            failed_metadata = state["phase"] == "FAILED" and name in {
                "RUN_STATUS.json",
                "INVALID.json",
                "VERIFICATION_FAILURE.json",
            }
            if (
                state["phase"] not in {"FINALIZING", "FAILED"}
                or state["owner_pid"] != os.getpid()
                or self._owner_token != state["owner_token"]
                or not (group in allowed_groups or failed_metadata)
            ):
                raise BudgetError("FINALIZATION_OWNER_REQUIRED")
            if state["fault_code"] and name in {"comply_vector.json", "candidate_freeze.json"}:
                raise BudgetError("CANDIDATE_AFTER_RECORDING_FAULT")
        elif (
            state["phase"] != "RUNNING"
            or state["fault_code"]
            or self._stop_event.is_set()
            or group in {"final", "receipts"}
        ):
            raise BudgetError("NORMAL_RECORDING_NOT_ALLOWED")

    def open_writer(
        self, path, mode="xb", *, encoding=None, errors=None, newline=None, final=False
    ):
        with self._locked():
            try:
                target = self._target(path)
                name = target.relative_to(self.root).as_posix()
                self._permission(name, final)
                if mode not in {"x", "w", "a", "xb", "wb", "ab", "xt", "wt", "at"}:
                    raise BudgetError("UNBOUNDED_WRITER_MODE")
                binary = "b" in mode
                if binary and any(v is not None for v in (encoding, errors, newline)):
                    raise BudgetError("BINARY_ENCODING_ARGUMENT")
                if not binary and (
                    encoding not in (None, "utf-8", "utf8", "UTF-8")
                    or errors not in (None, "strict")
                    or newline not in (None, "", "\n")
                ):
                    raise BudgetError("EXPLICIT_UTF8_WRITER_REQUIRED")
                if mode.startswith("w") and target.exists():
                    raise BudgetError("UNINSTRUMENTED_OVERWRITE_DENIED")
                self._check(self._scan())
                stream = _OPEN(target, "ab" if mode.startswith("a") else "xb", buffering=0)
                _HANDLES[str(self.root)] += 1
                return _Writer(self, target, name, stream, binary, final, mode.startswith("a"))
            except BaseException as error:  # noqa: BLE001 - preserve bounded terminal code, never recursive exception text.
                code = error.code if isinstance(error, BudgetError) else "RECORDING_OPEN_FAILURE"
                self._fault_locked(code)
                raise BudgetError(code) from None

    def _write(self, writer, data):
        with self._locked():
            try:
                self._permission(writer.name, writer.final)
                self._target(writer.path)
                before = writer.path.stat()
                opened = os.fstat(writer.stream.fileno())
                if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                    raise BudgetError("STALE_NAMESPACE_WRITER")
                if not writer.append and writer.stream.tell() != before.st_size:
                    raise BudgetError("STALE_WRITER_OFFSET")
                sizes = self._scan()
                self._check(sizes)
                prospective = {**sizes, writer.name: before.st_size + len(data)}
                self._check(prospective)
                count = writer.stream.write(data)
                os.fsync(writer.stream.fileno())
                after = self._scan()
                self._check(after)
                if count != len(data) or after[writer.name] != before.st_size + len(data):
                    raise BudgetError("PARTIAL_RECORDING_WRITE")
                return count
            except BaseException as error:  # noqa: BLE001 - preserve all partial writes without retry or recursive exception text.
                # A failed/partial write remains on disk. No recursive error text is serialized.
                code = error.code if isinstance(error, BudgetError) else "RECORDING_WRITE_FAILURE"
                try:
                    self._check(self._scan())
                except BudgetError:
                    code = "ACTUAL_RECORDING_BOUND_FAILURE"
                self._fault_locked(code)
                raise BudgetError(code) from None

    def write_bytes(self, path, data, mode="xb", final=False):
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("write_bytes requires bytes")
        with self.open_writer(path, mode, final=final) as stream:
            return stream.write(bytes(data))

    def replace(self, source, target, final=False):
        with self._locked():
            try:
                source, target = self._target(source), self._target(target)
                src, dst = (
                    source.relative_to(self.root).as_posix(),
                    target.relative_to(self.root).as_posix(),
                )
                self._permission(src, final)
                self._permission(dst, final)
                if src != "." + dst + ".tmp" or self.category(src) != self.category(dst):
                    raise BudgetError("EXACT_SAME_CATEGORY_STAGING_REQUIRED")
                if _HANDLES[str(self.root)]:
                    raise BudgetError("OPEN_WRITER_REPLACEMENT_DENIED")
                self._check(self._scan())  # Both old and staged bytes count simultaneously.
                _REPLACE(source, target)
                self._check(self._scan())
            except BaseException as error:  # noqa: BLE001 - preserve all replacement faults without retry.
                code = error.code if isinstance(error, BudgetError) else "RECORDING_REPLACE_FAILURE"
                self._fault_locked(code)
                raise BudgetError(code) from None

    @contextlib.contextmanager
    def scoped_writes(self):
        key = str(self.root)
        with _REGISTRY_LOCK:
            if not _SCOPES:
                builtins.open = io.open = _dispatch_open
                os.open = _dispatch_os_open
                os.replace = lambda *args, **kwargs: _mutation(_REPLACE, *args, **kwargs)
                os.rename = lambda *args, **kwargs: _mutation(_RENAME, *args, **kwargs)
                os.unlink = lambda *args, **kwargs: _mutation(_UNLINK, *args, **kwargs)
                os.remove = lambda *args, **kwargs: _mutation(_REMOVE, *args, **kwargs)
                os.truncate = lambda *args, **kwargs: _mutation(_TRUNCATE, *args, **kwargs)
                os.link = lambda *args, **kwargs: _mutation(_LINK, *args, **kwargs)
                os.symlink = lambda *args, **kwargs: _mutation(_SYMLINK, *args, **kwargs)
            old = _SCOPES.get(key, (self, 0))
            _SCOPES[key] = (old[0], old[1] + 1)
        try:
            yield self
        finally:
            with _REGISTRY_LOCK:
                budget, count = _SCOPES[key]
                if count == 1:
                    del _SCOPES[key]
                else:
                    _SCOPES[key] = (budget, count - 1)
                if not _SCOPES:
                    builtins.open, io.open, os.open = _OPEN, _IO_OPEN, _OS_OPEN
                    os.replace, os.rename = _REPLACE, _RENAME
                    os.unlink, os.remove, os.truncate = _UNLINK, _REMOVE, _TRUNCATE
                    os.link, os.symlink = _LINK, _SYMLINK

    def begin_finalization(self, quiescent):
        if quiescent is not True:
            self._stop_event.set()
            with self._state_locked():
                state = self._read_state()
                if state["phase"] != "RUNNING":
                    raise BudgetError("RUNNING_PHASE_REQUIRED")
                self._owner_token = uuid.uuid4().hex
                state.update(
                    phase="FAILED",
                    owner_pid=os.getpid(),
                    owner_token=self._owner_token,
                    fault_code=state["fault_code"] or "QUIESCENCE_UNCONFIRMED",
                )
                self._write_state(state)
                return False
        with self._locked():
            if _HANDLES[str(self.root)]:
                self._fault_locked("QUIESCENCE_REQUIRED")
                raise BudgetError("QUIESCENCE_REQUIRED")
            self._check(self._scan())
            with self._state_locked():
                state = self._read_state()
                if state["phase"] != "RUNNING":
                    raise BudgetError("RUNNING_PHASE_REQUIRED")
                self._owner_token = uuid.uuid4().hex
                state.update(
                    phase="FINALIZING", owner_pid=os.getpid(), owner_token=self._owner_token
                )
                self._write_state(state)
                return state

    def finalize_inventory(self, valid_candidate=False, quiescent=True):
        with self._locked(), self._state_locked():
            state = self._read_state()
            if (
                quiescent is not True
                or _HANDLES[str(self.root)]
                or state["phase"] != "FINALIZING"
                or state["owner_pid"] != os.getpid()
                or state["owner_token"] != self._owner_token
            ):
                raise BudgetError("QUIESCENT_FINALIZATION_OWNER_REQUIRED")
            if type(valid_candidate) is not bool:
                raise BudgetError("EXPLICIT_CANDIDATE_DISPOSITION_REQUIRED")
            sizes = self._scan()
            if INVENTORY_NAME in sizes:
                raise BudgetError("FINAL_INVENTORY_ALREADY_EXISTS")
            totals = self._check(sizes)
            result = {
                "schema": "sp_lense.three_family_recording_inventory.v1",
                "phase": "SEALED",
                "valid_candidate": valid_candidate and state["fault_code"] is None,
                "fault_code": state["fault_code"],
                "quiescent": True,
                "inventory_self_hash": None,
                "files": [],
                "category_bytes": {},
                "total_bytes": 0,
            }
            result["files"] = [
                {"path": name, "bytes": size, "sha256": "0" * 64}
                for name, size in sorted(sizes.items())
            ]
            result["files"].append({"path": INVENTORY_NAME, "bytes": 0, "sha256": None})

            def encode_fixed_point():
                previous = -1
                for _ in range(16):
                    encoded = (
                        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
                        + "\n"
                    ).encode("utf-8")
                    length = len(encoded)
                    if length == previous:
                        return encoded
                    previous = length
                    result["files"][-1]["bytes"] = length
                    result["category_bytes"] = {**totals, "receipts": totals["receipts"] + length}
                    result["total_bytes"] = sum(sizes.values()) + length
                raise BudgetError("INVENTORY_SIZE_FIXED_POINT_FAILED")

            placeholder = encode_fixed_point()
            self._check({**sizes, INVENTORY_NAME: len(placeholder)})
            state["phase"] = "SEALED"
            self._write_state(
                state
            )  # Must precede hashing: no state mutation after successful inventory.
            try:
                for entry in result["files"][:-1]:
                    entry["sha256"] = self._hash_file(entry["path"])
                data = encode_fixed_point()
                if len(data) != len(placeholder):
                    raise BudgetError("INVENTORY_HASH_SIZE_CHANGED")
                with _OPEN(self.root / INVENTORY_NAME, "xb", buffering=0) as stream:
                    if stream.write(data) != len(data):
                        raise BudgetError("PARTIAL_FINAL_INVENTORY")
                    os.fsync(stream.fileno())
                if self._scan() != {**sizes, INVENTORY_NAME: len(data)}:
                    raise BudgetError("FINAL_INVENTORY_SIZE_MISMATCH")
            except BaseException:  # noqa: BLE001 - partial final inventory is terminal and preserved.
                # No successful inventory exists; preserve partial bytes, abort sealing,
                # and permit reserved failure receipts without a complete-inventory claim.
                state["phase"] = "FAILED"
                state["fault_code"] = state["fault_code"] or "FINAL_INVENTORY_WRITE_FAILURE"
                self._write_state(state)
                raise BudgetError("FINAL_INVENTORY_WRITE_FAILURE") from None
            return result

    def verify_inventory(self):
        """Read-only full post-seal inventory check, including fixed-size state bytes."""
        with self._locked(), self._state_locked():
            state = self._read_state()
            if state["phase"] != "SEALED":
                raise BudgetError("SEALED_INVENTORY_REQUIRED")
            sizes = self._scan()
            totals = self._check(sizes)
            with _OPEN(self.root / INVENTORY_NAME, "rb") as stream:
                raw = stream.read(self.quotas["receipts"] + 1)
            if len(raw) > self.quotas["receipts"]:
                raise BudgetError("INVENTORY_RECEIPT_CAP")
            try:
                inventory = json.loads(raw)
                entries = inventory["files"]
                if (
                    inventory["schema"] != "sp_lense.three_family_recording_inventory.v1"
                    or inventory["phase"] != "SEALED"
                    or inventory["quiescent"] is not True
                    or type(inventory["valid_candidate"]) is not bool
                    or inventory["fault_code"] != state["fault_code"]
                    or state["fault_code"] is not None
                    and inventory["valid_candidate"]
                    or inventory["inventory_self_hash"] is not None
                    or inventory["category_bytes"] != totals
                    or inventory["total_bytes"] != sum(sizes.values())
                    or len(entries) != len(sizes)
                    or len({entry["path"] for entry in entries}) != len(entries)
                    or {entry["path"]: entry["bytes"] for entry in entries} != sizes
                ):
                    raise ValueError
                for entry in entries:
                    if set(entry) != {"path", "bytes", "sha256"} or type(entry["bytes"]) is not int:
                        raise ValueError
                    if entry["path"] == INVENTORY_NAME:
                        if entry["sha256"] is not None or entry["bytes"] != len(raw):
                            raise ValueError
                    else:
                        if entry["sha256"] != self._hash_file(entry["path"]):
                            raise ValueError
            except (KeyError, TypeError, ValueError, UnicodeError):
                raise BudgetError("SEALED_INVENTORY_MISMATCH") from None
            return {"status": "SEALED_INVENTORY_VERIFIED", "inventory": inventory}


class _Writer:
    def __init__(self, budget, path, name, stream, binary, final, append):
        self.budget, self.path, self.name, self.stream = budget, path, name, stream
        self.binary, self.final, self.closed = binary, final, False
        self.append = append

    def write(self, value):
        if self.closed:
            raise ValueError("write to closed recording stream")
        if self.binary:
            if not isinstance(value, (bytes, bytearray, memoryview)):
                raise TypeError("binary recording requires bytes")
            data = bytes(value)
        else:
            if not isinstance(value, str):
                raise TypeError("text recording requires str")
            data = value.encode("utf-8", errors="strict")
        count = self.budget._write(self, data)
        return count if self.binary else len(value)

    def flush(self):
        if self.closed:
            raise ValueError("flush of closed recording stream")
        # Writes are already unbuffered and fsynced; no descriptor escapes.

    def close(self):
        if not self.closed:
            with self.budget._thread_lock:
                self.stream.close()
                self.closed = True
                _HANDLES[str(self.budget.root)] -= 1

    def fileno(self):
        self.budget.fault("RAW_WRITER_DESCRIPTOR_DENIED")
        raise BudgetError("RAW_WRITER_DESCRIPTOR_DENIED")

    def seek(self, *args):
        self.budget.fault("WRITER_SEEK_DENIED")
        raise BudgetError("WRITER_SEEK_DENIED")

    def truncate(self, *args):
        self.budget.fault("WRITER_TRUNCATE_DENIED")
        raise BudgetError("WRITER_TRUNCATE_DENIED")

    def writable(self):
        return not self.closed

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
