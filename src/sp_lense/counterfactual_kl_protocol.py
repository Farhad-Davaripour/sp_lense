"""Prospective protocol locking and sealed-data access for CKES.

This module is deliberately model-free.  It binds explicit source-file hashes,
rendered prompt manifests, configuration, thresholds, and sealed-access rules
into one canonical lock identity.  The sealed loader hashes the exact sealed
bytes before checking authorization and does not deserialize those bytes until
the locked validation result passes every authorization check.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .factorial_causal_anchor import canonical_sha256

LOCK_SCHEMA_VERSION = "sp_lense.counterfactual_kl_protocol_lock.v2"
LOCK_STATUS = "prospective_before_validation_model_outcomes"
LOCK_HASH_FIELD = "lock_identity_sha256"
DEFAULT_RESULT_HASH_FIELD = "result_sha256"
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_RENDERED_SPLITS = ("validation", "sealed")


class CounterfactualKLProtocolError(RuntimeError):
    """Base error for a malformed or violated prospective protocol."""


class CounterfactualKLProtocolIntegrityError(CounterfactualKLProtocolError):
    """A lock, file, manifest, or self-hash failed integrity validation."""


class SealedAccessDenied(CounterfactualKLProtocolError):
    """The exact locked validation result does not authorize sealed access."""


def file_sha256(path: str | Path) -> str:
    """Hash a file as bytes without interpreting its contents."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_json_tree(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{path} must use nonempty string keys")
            _require_json_tree(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_json_tree(item, path=f"{path}[{index}]")
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return
    raise TypeError(f"{path} contains a non-JSON value")


def _canonical_copy(value: Any, *, path: str) -> Any:
    _require_json_tree(value, path=path)
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return json.loads(encoded)


def _require_hash(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise CounterfactualKLProtocolIntegrityError(
            f"{field} must be one lowercase SHA-256 digest"
        )
    return value


def self_hash_record(
    value: Mapping[str, Any], *, hash_field: str = DEFAULT_RESULT_HASH_FIELD
) -> dict[str, Any]:
    """Return a canonical JSON record with a top-level self-hash attached."""

    if not isinstance(hash_field, str) or not hash_field:
        raise TypeError("hash_field must be a nonempty string")
    record = _canonical_copy(value, path="record")
    if not isinstance(record, dict):
        raise TypeError("a self-hashed record must be an object")
    if hash_field in record:
        raise ValueError(f"record already contains {hash_field}")
    record[hash_field] = canonical_sha256(record)
    return record


def verify_self_hashed_record(
    value: Mapping[str, Any], *, hash_field: str = DEFAULT_RESULT_HASH_FIELD
) -> dict[str, Any]:
    """Validate and return a canonical copy of a top-level self-hashed object."""

    record = _canonical_copy(value, path="record")
    if not isinstance(record, dict):
        raise CounterfactualKLProtocolIntegrityError("a self-hashed record must be an object")
    observed = _require_hash(record.get(hash_field), field=hash_field)
    unhashed = dict(record)
    del unhashed[hash_field]
    if observed != canonical_sha256(unhashed):
        raise CounterfactualKLProtocolIntegrityError(
            f"{hash_field} does not match the canonical record"
        )
    return record


def _normalize_file_hashes(value: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("file_hashes must be a nonempty object")
    result: dict[str, dict[str, Any]] = {}
    paths: set[str] = set()
    for key, raw_entry in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError("file_hashes keys must be nonempty strings")
        if not isinstance(raw_entry, Mapping):
            raise TypeError(f"file_hashes.{key} must be an object")
        entry = _canonical_copy(raw_entry, path=f"file_hashes.{key}")
        allowed = {"path", "sha256", "bytes"}
        if set(entry) - allowed or not {"path", "sha256"}.issubset(entry):
            raise ValueError(f"file_hashes.{key} requires path and sha256, with optional bytes")
        path = entry["path"]
        if not isinstance(path, str) or not path:
            raise TypeError(f"file_hashes.{key}.path must be a nonempty string")
        if path in paths:
            raise ValueError("file_hashes paths must be unique")
        paths.add(path)
        _require_hash(entry["sha256"], field=f"file_hashes.{key}.sha256")
        if "bytes" in entry and (
            isinstance(entry["bytes"], bool)
            or not isinstance(entry["bytes"], int)
            or entry["bytes"] < 0
        ):
            raise TypeError(f"file_hashes.{key}.bytes must be a nonnegative integer")
        result[key] = entry
    return result


def _normalize_rendered_manifests(value: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != set(_RENDERED_SPLITS):
        raise ValueError("rendered_manifests must contain exactly validation and sealed")
    result: dict[str, dict[str, Any]] = {}
    for split in _RENDERED_SPLITS:
        manifest = _canonical_copy(value[split], path=f"rendered_manifests.{split}")
        if not isinstance(manifest, dict) or not manifest:
            raise ValueError(f"rendered_manifests.{split} must be a nonempty object")
        result[split] = manifest
    return result


def _normalize_gate_names(value: Sequence[str]) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError("required_validation_gates must be a nonempty sequence")
    gates = list(value)
    if any(not isinstance(gate, str) or not gate for gate in gates):
        raise TypeError("required validation gate names must be nonempty strings")
    if len(gates) != len(set(gates)):
        raise ValueError("required validation gate names must be unique")
    return sorted(gates)


def _component_hashes(body: Mapping[str, Any]) -> dict[str, str]:
    return {
        field: canonical_sha256(body[field])
        for field in (
            "file_hashes",
            "rendered_manifests",
            "configuration",
            "thresholds",
            "sealed_access",
        )
    }


def build_prospective_lock(
    *,
    file_hashes: Mapping[str, Any],
    rendered_manifests: Mapping[str, Any],
    configuration: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    sealed_dataset_file_key: str,
    validation_result_schema_version: str,
    required_validation_gates: Sequence[str],
    validation_dataset_file_key: str = "validation_dataset",
    validation_result_hash_field: str = DEFAULT_RESULT_HASH_FIELD,
) -> dict[str, Any]:
    """Build one canonical prospective lock without reading any study file."""

    files = _normalize_file_hashes(file_hashes)
    manifests = _normalize_rendered_manifests(rendered_manifests)
    config = _canonical_copy(configuration, path="configuration")
    limits = _canonical_copy(thresholds, path="thresholds")
    if not isinstance(config, dict) or not config:
        raise ValueError("configuration must be a nonempty object")
    if not isinstance(limits, dict) or not limits:
        raise ValueError("thresholds must be a nonempty object")
    if not isinstance(sealed_dataset_file_key, str) or sealed_dataset_file_key not in files:
        raise ValueError("sealed_dataset_file_key must name one locked file")
    if not isinstance(validation_dataset_file_key, str) or validation_dataset_file_key not in files:
        raise ValueError("validation_dataset_file_key must name one locked file")
    if validation_dataset_file_key == sealed_dataset_file_key:
        raise ValueError("validation and sealed datasets must be distinct locked files")
    if (
        not isinstance(validation_result_schema_version, str)
        or not validation_result_schema_version
    ):
        raise TypeError("validation_result_schema_version must be a nonempty string")
    if not isinstance(validation_result_hash_field, str) or not validation_result_hash_field:
        raise TypeError("validation_result_hash_field must be a nonempty string")
    gates = _normalize_gate_names(required_validation_gates)
    sealed_access = {
        "sealed_dataset_file_key": sealed_dataset_file_key,
        "validation_dataset_file_key": validation_dataset_file_key,
        "validation_result_schema_version": validation_result_schema_version,
        "validation_result_lock_field": LOCK_HASH_FIELD,
        "validation_result_hash_field": validation_result_hash_field,
        "validation_result_split_field": "split",
        "validation_result_dataset_hash_field": "dataset_file_sha256",
        "required_validation_split": "validation",
        "required_status": "go",
        "required_non_go_status": "no_go",
        "required_validation_gates": gates,
        "policy": "hash_exact_sealed_bytes_before_authorization_parse_only_after_go",
    }
    body: dict[str, Any] = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "status": LOCK_STATUS,
        "model_compute_used_to_build_lock": 0,
        "file_hashes": files,
        "rendered_manifests": manifests,
        "configuration": config,
        "thresholds": limits,
        "sealed_access": sealed_access,
    }
    body["component_sha256"] = _component_hashes(body)
    lock = dict(body)
    lock[LOCK_HASH_FIELD] = canonical_sha256(lock)
    return verify_prospective_lock(lock)


def verify_prospective_lock(value: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the lock is canonical, self-consistent, and complete."""

    lock = _canonical_copy(value, path="lock")
    if not isinstance(lock, dict):
        raise CounterfactualKLProtocolIntegrityError("protocol lock must be an object")
    expected_top_level = {
        "schema_version",
        "status",
        "model_compute_used_to_build_lock",
        "file_hashes",
        "rendered_manifests",
        "configuration",
        "thresholds",
        "sealed_access",
        "component_sha256",
        LOCK_HASH_FIELD,
    }
    if set(lock) != expected_top_level:
        raise CounterfactualKLProtocolIntegrityError("protocol lock fields differ")
    observed_identity = _require_hash(lock.get(LOCK_HASH_FIELD), field=LOCK_HASH_FIELD)
    unhashed = dict(lock)
    del unhashed[LOCK_HASH_FIELD]
    if observed_identity != canonical_sha256(unhashed):
        raise CounterfactualKLProtocolIntegrityError("protocol lock identity differs")
    if lock.get("schema_version") != LOCK_SCHEMA_VERSION or lock.get("status") != LOCK_STATUS:
        raise CounterfactualKLProtocolIntegrityError("protocol lock schema or status differs")
    if lock.get("model_compute_used_to_build_lock") != 0:
        raise CounterfactualKLProtocolIntegrityError("prospective lock reports model compute")
    files = _normalize_file_hashes(lock.get("file_hashes"))
    manifests = _normalize_rendered_manifests(lock.get("rendered_manifests"))
    config = _canonical_copy(lock.get("configuration"), path="configuration")
    limits = _canonical_copy(lock.get("thresholds"), path="thresholds")
    if not isinstance(config, dict) or not config or not isinstance(limits, dict) or not limits:
        raise CounterfactualKLProtocolIntegrityError(
            "protocol configuration and thresholds must be nonempty objects"
        )
    access = _canonical_copy(lock.get("sealed_access"), path="sealed_access")
    if not isinstance(access, dict):
        raise CounterfactualKLProtocolIntegrityError("sealed_access must be an object")
    expected_access_keys = {
        "sealed_dataset_file_key",
        "validation_dataset_file_key",
        "validation_result_schema_version",
        "validation_result_lock_field",
        "validation_result_hash_field",
        "validation_result_split_field",
        "validation_result_dataset_hash_field",
        "required_validation_split",
        "required_status",
        "required_non_go_status",
        "required_validation_gates",
        "policy",
    }
    if set(access) != expected_access_keys:
        raise CounterfactualKLProtocolIntegrityError("sealed_access fields differ")
    sealed_key = access["sealed_dataset_file_key"]
    if not isinstance(sealed_key, str) or sealed_key not in files:
        raise CounterfactualKLProtocolIntegrityError("sealed file key is not locked")
    validation_key = access["validation_dataset_file_key"]
    if (
        not isinstance(validation_key, str)
        or validation_key not in files
        or validation_key == sealed_key
    ):
        raise CounterfactualKLProtocolIntegrityError(
            "validation file key is not a distinct locked file"
        )
    if access["validation_result_lock_field"] != LOCK_HASH_FIELD:
        raise CounterfactualKLProtocolIntegrityError("validation lock-binding field differs")
    if access["required_status"] != "go":
        raise CounterfactualKLProtocolIntegrityError("sealed status requirement differs")
    if access["required_non_go_status"] != "no_go":
        raise CounterfactualKLProtocolIntegrityError("non-go status requirement differs")
    if access["required_validation_split"] != "validation":
        raise CounterfactualKLProtocolIntegrityError("validation split requirement differs")
    if access["validation_result_split_field"] != "split":
        raise CounterfactualKLProtocolIntegrityError("validation split field differs")
    if access["validation_result_dataset_hash_field"] != "dataset_file_sha256":
        raise CounterfactualKLProtocolIntegrityError("validation dataset-hash field differs")
    if access["policy"] != "hash_exact_sealed_bytes_before_authorization_parse_only_after_go":
        raise CounterfactualKLProtocolIntegrityError("sealed parsing policy differs")
    if (
        not isinstance(access["validation_result_schema_version"], str)
        or not access["validation_result_schema_version"]
    ):
        raise CounterfactualKLProtocolIntegrityError("validation result schema is invalid")
    if (
        not isinstance(access["validation_result_hash_field"], str)
        or not access["validation_result_hash_field"]
    ):
        raise CounterfactualKLProtocolIntegrityError("validation result hash field is invalid")
    gates = _normalize_gate_names(access["required_validation_gates"])
    if gates != access["required_validation_gates"]:
        raise CounterfactualKLProtocolIntegrityError("required gates are not canonical")
    normalized_body = dict(unhashed)
    normalized_body["file_hashes"] = files
    normalized_body["rendered_manifests"] = manifests
    normalized_body["configuration"] = config
    normalized_body["thresholds"] = limits
    normalized_body["sealed_access"] = access
    expected_components = _component_hashes(normalized_body)
    if lock.get("component_sha256") != expected_components:
        raise CounterfactualKLProtocolIntegrityError("protocol component hash differs")
    return lock


def _parse_json_object_bytes(value: bytes, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CounterfactualKLProtocolIntegrityError(f"{label} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise CounterfactualKLProtocolIntegrityError(f"{label} must contain an object")
    return parsed


def _parse_sealed_json_bytes(value: bytes) -> dict[str, Any]:
    """The sole sealed-deserialization boundary; called only after authorization."""

    return _parse_json_object_bytes(value, label="sealed dataset")


def validate_locked_result(
    result: Mapping[str, Any],
    *,
    lock: Mapping[str, Any],
    expected_split: str,
    require_go: bool = False,
) -> dict[str, Any]:
    """Validate one result against the exact lock, split, dataset, and gates.

    This is the common result boundary for ordinary runner reloads and sealed
    authorization.  ``require_go=False`` permits a logically consistent
    ``no_go`` result; sealed access passes ``require_go=True``.
    """

    checked_lock = verify_prospective_lock(lock)
    access = checked_lock["sealed_access"]
    validation_split = str(access["required_validation_split"])
    if expected_split == validation_split:
        dataset_key = str(access["validation_dataset_file_key"])
    elif expected_split == "sealed":
        dataset_key = str(access["sealed_dataset_file_key"])
    else:
        raise CounterfactualKLProtocolIntegrityError("result split is not locked")
    try:
        checked = verify_self_hashed_record(
            result,
            hash_field=str(access["validation_result_hash_field"]),
        )
    except (CounterfactualKLProtocolIntegrityError, TypeError, ValueError) as exc:
        raise CounterfactualKLProtocolIntegrityError("result self-hash is invalid") from exc
    if checked.get("schema_version") != access["validation_result_schema_version"]:
        raise CounterfactualKLProtocolIntegrityError("result schema does not match the lock")
    lock_field = str(access["validation_result_lock_field"])
    if checked.get(lock_field) != checked_lock[LOCK_HASH_FIELD]:
        raise CounterfactualKLProtocolIntegrityError("result does not bind the exact lock")
    split_field = str(access["validation_result_split_field"])
    if checked.get(split_field) != expected_split:
        raise CounterfactualKLProtocolIntegrityError("result split does not match the locked split")
    dataset_hash_field = str(access["validation_result_dataset_hash_field"])
    expected_dataset_hash = checked_lock["file_hashes"][dataset_key]["sha256"]
    if checked.get(dataset_hash_field) != expected_dataset_hash:
        raise CounterfactualKLProtocolIntegrityError(
            "result dataset hash does not match the locked split"
        )
    gates = checked.get("gates")
    required = list(access["required_validation_gates"])
    if not isinstance(gates, Mapping) or set(gates) != set(required):
        raise CounterfactualKLProtocolIntegrityError("result gates differ from the locked gates")
    if any(type(gates[name]) is not bool for name in required):
        raise CounterfactualKLProtocolIntegrityError(
            "every locked result gate must be a literal boolean"
        )
    all_passed = all(gates[name] is True for name in required)
    expected_status = (
        str(access["required_status"]) if all_passed else str(access["required_non_go_status"])
    )
    if require_go and not all_passed:
        raise CounterfactualKLProtocolIntegrityError("not every locked validation gate passed")
    if require_go and checked.get("status") != access["required_status"]:
        raise CounterfactualKLProtocolIntegrityError("validation result is not an authorized go")
    if checked.get("status") != expected_status:
        raise CounterfactualKLProtocolIntegrityError(
            "result status is inconsistent with the exact locked gates"
        )
    return checked


def _authorize_validation_result(
    result: Mapping[str, Any], *, lock: Mapping[str, Any]
) -> dict[str, Any]:
    access = lock["sealed_access"]
    try:
        return validate_locked_result(
            result,
            lock=lock,
            expected_split=str(access["required_validation_split"]),
            require_go=True,
        )
    except CounterfactualKLProtocolIntegrityError as exc:
        raise SealedAccessDenied(str(exc)) from exc


def load_sealed_dataset(
    sealed_path: str | Path,
    validation_result_path: str | Path,
    *,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Load sealed JSON only after exact byte integrity and validation authorization.

    The sealed file is read once as opaque bytes and hashed.  No decode or JSON
    parser receives those bytes until the separate validation result has a valid
    self-hash, binds the exact prospective lock, says ``go``, and has every
    locked gate set to the literal boolean ``True``.
    """

    checked_lock = verify_prospective_lock(lock)
    access = checked_lock["sealed_access"]
    sealed_entry = checked_lock["file_hashes"][access["sealed_dataset_file_key"]]

    sealed_bytes = Path(sealed_path).read_bytes()
    if _bytes_sha256(sealed_bytes) != sealed_entry["sha256"]:
        raise CounterfactualKLProtocolIntegrityError("sealed dataset byte hash differs")
    if "bytes" in sealed_entry and len(sealed_bytes) != sealed_entry["bytes"]:
        raise CounterfactualKLProtocolIntegrityError("sealed dataset byte count differs")

    validation_bytes = Path(validation_result_path).read_bytes()
    validation_result = _parse_json_object_bytes(validation_bytes, label="validation result")
    _authorize_validation_result(validation_result, lock=checked_lock)
    return _parse_sealed_json_bytes(sealed_bytes)


__all__ = [
    "DEFAULT_RESULT_HASH_FIELD",
    "LOCK_HASH_FIELD",
    "LOCK_SCHEMA_VERSION",
    "LOCK_STATUS",
    "CounterfactualKLProtocolError",
    "CounterfactualKLProtocolIntegrityError",
    "SealedAccessDenied",
    "build_prospective_lock",
    "file_sha256",
    "load_sealed_dataset",
    "self_hash_record",
    "validate_locked_result",
    "verify_prospective_lock",
    "verify_self_hashed_record",
]
