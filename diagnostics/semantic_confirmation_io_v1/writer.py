"""Serialized single-owner evidence writer; no model, tokenizer or supervisor.

Completed and partial bytes are retained. Sticky failure blocks ordinary writes;
only native closeout remains possible, subject to actual space and file checks.
External concurrent mutation is detected at reconciliation boundaries, not made
impossible by these checks. A future process-level owner must exclude it.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import threading
from pathlib import Path

from binding import binding_record, load_bound_resource


class EvidenceIOError(OSError):
    pass


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def relative_name(value):
    if not isinstance(value, str) or not value or any(c in value for c in '\x00<>:"|?*'):
        raise EvidenceIOError("invalid relative evidence path")
    parts = value.replace("\\", "/").split("/")
    reserved = {"con", "prn", "aux", "nul"} | {f"{p}{i}" for p in ("com", "lpt") for i in range(1, 10)}
    if any(p in ("", ".", "..") or p.endswith((" ", ".")) or p.split(".")[0].casefold() in reserved for p in parts):
        raise EvidenceIOError("absolute/traversal/Windows alias rejected")
    return "/".join(parts).casefold()


def linklike(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


class EvidenceWriter:
    def __init__(self, root, *, preparation_root, component_settings=None, write_function=None):
        self.contract, self.helper = load_bound_resource()
        expected = self.helper.component_settings(self.contract)
        components = component_settings or {name: copy.deepcopy(expected) for name in
                      ("recorder", "worker_supervisor", "audit_supervisor", "saved_judge")}
        self.helper.validate_component_settings(self.contract, components)
        self.settings = expected
        self.preparation_root = Path(preparation_root).resolve(strict=True)
        self.root = Path(root).resolve(strict=False)
        if not self.root.is_relative_to(self.preparation_root) or self.root == self.preparation_root:
            raise EvidenceIOError("fixture root must be inside preparation namespace")
        self.root.mkdir(parents=True, exist_ok=False)
        self.owner = (os.getpid(), threading.get_ident())
        self.lock = threading.RLock()
        self.ledger = self.helper.ResourceLedger(self.contract)
        self.records = {}
        self.failures = []
        self.sticky_failure = False
        self.sealed = False
        self.write_function = write_function or os.write

    def _owner(self):
        if self.owner != (os.getpid(), threading.get_ident()):
            raise EvidenceIOError("single owner process/thread required")

    def _target(self, name):
        key = relative_name(name)
        target = self.root.joinpath(*key.split("/"))
        if not target.resolve(strict=False).is_relative_to(self.root):
            raise EvidenceIOError("resolved evidence path escapes root")
        for part in (target, *target.parents):
            if part == self.root:
                break
            if part.exists() and linklike(part):
                raise EvidenceIOError("symlink/junction evidence path rejected")
        return key, target

    def _scan(self):
        found = {}
        for directory, folders, names in os.walk(self.root, followlinks=False):
            for name in folders:
                if linklike(Path(directory) / name):
                    raise EvidenceIOError("external linked directory")
            for name in names:
                path = Path(directory) / name
                if linklike(path) or path.stat().st_nlink > 1:
                    raise EvidenceIOError("external linked file")
                actual_name = path.relative_to(self.root).as_posix()
                key = relative_name(actual_name)
                if key != actual_name or key in found:
                    raise EvidenceIOError("noncanonical/colliding filesystem path")
                raw = path.read_bytes()
                found[key] = {"path": key, "actual_bytes": len(raw), "actual_sha256": sha(raw)}
        return found

    def _measure_preparation(self):
        total = 0
        for directory, folders, names in os.walk(self.preparation_root, followlinks=False):
            for name in folders:
                if linklike(Path(directory) / name):
                    raise EvidenceIOError("linked preparation directory")
            for name in names:
                path = Path(directory) / name
                if linklike(path):
                    raise EvidenceIOError("linked preparation file")
                total += path.stat().st_size
        return total

    def _fail(self, kind, error):
        self.sticky_failure = True
        self.failures.append({"kind": kind, "type": type(error).__name__, "message": str(error)})

    def _reconciliation(self):
        found = self._scan()
        issues = []
        files = []
        for name in sorted(set(found) | set(self.records)):
            observed, recorded = found.get(name), self.records.get(name)
            if recorded is None:
                issues.append("untracked_external:" + name)
                files.append({**observed, "category": "untracked_external", "complete": False,
                              "expected_bytes": None, "expected_sha256": None})
                continue
            item = {**recorded, **(observed or {"actual_bytes": 0, "actual_sha256": None})}
            item["present"] = observed is not None
            if observed is None:
                issues.append("missing_recorded:" + name)
            elif (observed["actual_bytes"], observed["actual_sha256"]) != (recorded["recorded_bytes"], recorded["recorded_sha256"]):
                issues.append("externally_changed:" + name)
            if not recorded["complete"]:
                issues.append("incomplete_write:" + name)
            files.append(item)
        by_category = {k: 0 for k in self.contract["storage"]["categories_bytes"]}
        by_category["untracked_external"] = 0
        for item in files:
            by_category[item["category"]] += item["actual_bytes"]
        return {"files": files, "issues": issues, "actual_bytes_by_category": by_category,
                "actual_total_bytes": sum(item["actual_bytes"] for item in found.values()),
                "reserved": self.ledger.report()}

    def reconcile(self):
        with self.lock:
            self._owner()
            result = self._reconciliation()
            if result["issues"] and not self.sticky_failure:
                self._fail("reconciliation", EvidenceIOError("; ".join(result["issues"])))
            return result

    def _write(self, category, name, data, reserve, *, append=False, logit=False):
        with self.lock:
            self._owner()
            if self.sticky_failure or self.sealed:
                raise EvidenceIOError("ordinary writes blocked after failure/seal")
            if not isinstance(data, bytes):
                raise EvidenceIOError("complete serialized bytes required")
            key = target = old = trial = None
            intended = None
            try:
                key, target = self._target(name)
                reconciliation = self._reconciliation()
                if reconciliation["issues"]:
                    raise EvidenceIOError("; ".join(reconciliation["issues"]))
                old = self.records.get(key)
                if (not append and target.exists()) or (append and target.exists() and old is None):
                    raise EvidenceIOError("exclusive creation/owned append required")
                previous = target.read_bytes() if append and target.exists() else b""
                intended = previous + data
                trial = copy.deepcopy(self.ledger)
                reserve(trial, key, len(data))
                if self._measure_preparation() + len(data) > self.contract["storage"]["preparation_bytes"]:
                    raise EvidenceIOError("32 MiB preparation cap")
                self.ledger = trial  # Full attempted reservation retained on partial failure.
                target.parent.mkdir(parents=True, exist_ok=True)
                self._target(key)  # Recheck containment after directory creation.
                flags = os.O_WRONLY | getattr(os, "O_BINARY", 0)
                flags |= os.O_APPEND if append and target.exists() else os.O_CREAT | os.O_EXCL
                descriptor = os.open(target, flags, 0o600)
                try:
                    offset = 0
                    while offset < len(data):
                        count = self.write_function(descriptor, data[offset:])
                        if type(count) is not int or not 0 < count <= len(data) - offset:
                            raise EvidenceIOError("invalid/zero OS write progress")
                        offset += count
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                actual = target.read_bytes()
                if actual != intended:
                    raise EvidenceIOError("post-write complete bytes differ")
                self.records[key] = {"path": key, "category": category, "complete": True,
                    "expected_bytes": len(intended), "expected_sha256": sha(intended),
                    "recorded_bytes": len(actual), "recorded_sha256": sha(actual), "logit": logit}
                return copy.deepcopy(self.records[key])
            except BaseException as error:
                if trial is not None and self.ledger is trial and target is not None:
                    actual = target.read_bytes() if target.exists() else b""
                    self.records[key] = {"path": key, "category": category, "complete": False,
                        "expected_bytes": len(intended), "expected_sha256": sha(intended),
                        "recorded_bytes": len(actual), "recorded_sha256": sha(actual), "logit": logit}
                self._fail("write", error)
                raise EvidenceIOError(str(error)) from error

    def write_logits(self, name, raw):
        self.helper.decode_logits(self.contract, raw)  # Exact length and finite float32.
        return self._write("logits", name, raw, lambda ledger, key, n: ledger.add_logit(key, n), logit=True)

    def write_row(self, name, raw):
        return self._write("rows", name, raw, lambda ledger, key, n: ledger.add_row(key, n))

    def append_event(self, stream, raw):
        if not isinstance(raw, bytes) or not raw.endswith(b"\n"):
            raise EvidenceIOError("complete serialized event including newline required")
        if stream.endswith(".jsonl") and raw.count(b"\n") != 1:
            raise EvidenceIOError("exactly one complete JSONL record per append")
        json.loads(raw)
        return self._write("ledgers", stream, raw, lambda ledger, key, n: ledger.add_event(key, n), append=True)

    def append_log(self, stream, raw):
        return self._write("logs", "logs/" + stream, raw,
                           lambda ledger, key, n: ledger.add_log(key.split("/")[-1], n), append=True)

    def write_source(self, name, raw):
        return self._write("sources_inputs", name, raw,
                           lambda ledger, key, n: ledger.add_bundle_file("sources_inputs", key, n))

    def closeout(self, name="closeout/index.json"):
        with self.lock:
            self._owner()
            if self.sealed:
                raise EvidenceIOError("already sealed; no overwrite")
            reconciliation = self._reconciliation()
            if reconciliation["issues"] and not self.sticky_failure:
                self._fail("reconciliation", EvidenceIOError("; ".join(reconciliation["issues"])))
            record = {"schema": "sp_lense.confirmation_io_index.v1", "binding": binding_record(),
                "settings": self.settings, "status": "INCOMPLETE" if self.sticky_failure else "COMPLETE",
                "sticky_failure": self.sticky_failure, "failures": copy.deepcopy(self.failures),
                "reconciliation": reconciliation, "index_excludes_itself": True,
                "real_run_authorized": False, "model_work": False}
            raw = (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
            key, target = self._target(name)
            trial = copy.deepcopy(self.ledger)
            trial.add_bundle_file("failure_closeout", key, len(raw))
            if reconciliation["actual_total_bytes"] + len(raw) > self.contract["storage"]["total_bytes"]:
                raise EvidenceIOError("actual files leave insufficient closeout space")
            if self._measure_preparation() + len(raw) > self.contract["storage"]["preparation_bytes"]:
                raise EvidenceIOError("preparation cap leaves insufficient closeout space")
            target.parent.mkdir(parents=True, exist_ok=True)
            self._target(key)
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
            try:
                offset = 0
                while offset < len(raw):
                    count = os.write(descriptor, raw[offset:])
                    if count <= 0:
                        raise EvidenceIOError("closeout made no progress")
                    offset += count
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if target.read_bytes() != raw:
                raise EvidenceIOError("closeout bytes differ")
            self.ledger = trial
            self.sealed = True
            return {"path": key, "bytes": len(raw), "sha256": sha(raw), "status": record["status"],
                    "actual_total_including_index": reconciliation["actual_total_bytes"] + len(raw),
                    "remaining_closeout_bytes": self.ledger.report()["remaining_closeout_bytes"]}
