"""Scoped nonnumeric writer bindings for the one three-family attempt."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

from scripts import envelope_transfer_diagnostic as runtime_io
from scripts import saved_offset_order_bridge_io as audit_io

DIAGNOSTIC_PREFIX_BYTES = 4096
DIAGNOSTIC_KEYS = {"reason", "error", "fault", "cleanup_error"}
PREREGISTRATION_CAP = 2 * 1024**2


def diagnostic_text(text):
    """Full length/digest only for an already-known concrete string, never arbitrary repr."""
    digest = hashlib.sha256()
    prefix = bytearray()
    size = 0
    for start in range(0, len(text), 4096):
        chunk = text[start : start + 4096].encode("utf-8", errors="backslashreplace")
        digest.update(chunk)
        size += len(chunk)
        prefix.extend(chunk[: max(0, DIAGNOSTIC_PREFIX_BYTES - len(prefix))])
    return {
        "prefix": bytes(prefix).decode("utf-8", errors="replace"),
        "prefix_encoding": "UTF-8 with backslashreplace; display may replace split final codepoint",
        "prefix_hex": bytes(prefix).hex(),
        "prefix_bytes": len(prefix),
        "full_utf8_bytes": size,
        "full_utf8_sha256": digest.hexdigest(),
        "full_value_known": True,
        "truncated": size > len(prefix),
    }


def bound_diagnostics(value):
    """Keep successful scientific records byte-identical; disclose long error-field prefixes."""
    if not isinstance(value, dict):
        return value
    result = dict(value)
    metadata = {}
    for key in DIAGNOSTIC_KEYS:
        item = result.get(key)
        if isinstance(item, str) and (
            len(item) > 4096 or len(item.encode("utf-8", errors="backslashreplace")) > 4096
        ):
            spec = diagnostic_text(item)
            result[key] = spec["prefix"]
            metadata[key] = spec
    if metadata:
        result["recording_diagnostic_prefixes"] = metadata
    return result


def namespace_path(root, path):
    if isinstance(path, int):
        return False
    selected = os.path.normcase(str(Path(root).absolute()))
    for candidate_path in (Path(path).absolute(), Path(path).resolve()):
        candidate = os.path.normcase(str(candidate_path))
        try:
            if os.path.commonpath([candidate, selected]) == selected:
                return True
        except ValueError:
            continue
    return False


def encoded_json(value, *, sorted_keys=False, line=False):
    value = bound_diagnostics(value)
    options = {"sort_keys": sorted_keys, "allow_nan": False}
    if not line:
        options["indent"] = 2
    return (json.dumps(value, **options) + "\n").encode("utf-8")


@contextmanager
def bind_writers(budget, *, final=False, intercept_paths=True):
    """Restore shared helper globals on exit; writes outside this namespace are unchanged."""
    saved_runtime_new, saved_append = runtime_io.write_new, runtime_io.append_row
    saved_audit_new = audit_io.write_new

    def new_runtime(path, value):
        if not namespace_path(budget.root, path):
            return saved_runtime_new(path, value)
        bounded = bound_diagnostics(value)
        result = budget.write_bytes(path, encoded_json(bounded), final=final)
        if "recording_diagnostic_prefixes" in bounded:
            budget.fault("EXCEPTION_METADATA_PREFIX")
        return result

    def new_audit(path, value):
        if not namespace_path(budget.root, path):
            return saved_audit_new(path, value)
        if Path(path).name in {"comply_vector.json", "candidate_freeze.json"}:
            value = {**value, "valid_only_with_complete_final_recording_inventory": True}
        bounded = bound_diagnostics(value)
        result = budget.write_bytes(path, encoded_json(bounded, sorted_keys=True), final=final)
        if "recording_diagnostic_prefixes" in bounded:
            budget.fault("EXCEPTION_METADATA_PREFIX")
        return result

    def append(path, value):
        if not namespace_path(budget.root, path):
            return saved_append(path, value)
        bounded = bound_diagnostics(value)
        result = budget.write_bytes(path, encoded_json(bounded, line=True), mode="ab", final=final)
        if "recording_diagnostic_prefixes" in bounded:
            budget.fault("EXCEPTION_METADATA_PREFIX")
        return result

    runtime_io.write_new, runtime_io.append_row = new_runtime, append
    audit_io.write_new = new_audit
    try:
        if intercept_paths:
            with budget.scoped_writes():
                yield
        else:
            yield
    finally:
        runtime_io.write_new, runtime_io.append_row = saved_runtime_new, saved_append
        audit_io.write_new = saved_audit_new


def write_preregistration(path, record):
    """The prospective namespace remains preregistration-only: no run-control files yet."""
    content = encoded_json(record)
    if len(content) > PREREGISTRATION_CAP:
        raise ValueError("prospective preregistration exceeds reserved metadata bytes")
    if Path(path).name != "preregistration.json":
        raise ValueError("preregistration writer cannot write runtime evidence")
    # No other process is authorized and the namespace is freshly, exclusively created.
    # This path is called before any runtime interception or worker exists.
    with Path(path).open("xb", buffering=0) as stream:
        written = stream.write(content)
        if written != len(content):
            raise OSError("partial preregistration write; no retry or launch")
        os.fsync(stream.fileno())
