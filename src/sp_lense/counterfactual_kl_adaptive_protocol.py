"""Transparent adaptive locking for the post-baseline CKES v2 revision.

The generic CKES lock correctly describes a study designed without prior model
evidence.  CKES v2 was instead motivated by the immutable v1 state-zero no-go.
This module retains the generic lock's deep file/manifest/gate validation while
using an explicit adaptive status and recording the prior compute that informed
the revision.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .counterfactual_kl_protocol import (
    CounterfactualKLProtocolIntegrityError,
    SealedAccessDenied,
    build_prospective_lock,
    verify_self_hashed_record,
)
from .factorial_causal_anchor import canonical_sha256

ADAPTIVE_LOCK_SCHEMA_VERSION = "sp_lense.counterfactual_kl_adaptive_protocol_lock.v1"
ADAPTIVE_LOCK_STATUS = (
    "adaptive_after_v1_baseline_prospective_before_v2_model_outcomes"
)
LOCK_HASH_FIELD = "lock_identity_sha256"
ADAPTIVE_PROVENANCE_HASH_FIELD = "adaptive_provenance_sha256"
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")


def _canonical(value: Any, *, label: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CounterfactualKLProtocolIntegrityError(
            f"{label} is not a finite JSON tree"
        ) from exc


def _require_hash(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise CounterfactualKLProtocolIntegrityError(
            f"{field} must be one lowercase SHA-256 digest"
        )
    return value


def _surrogate_lock(value: Mapping[str, Any]) -> dict[str, Any]:
    access = value["sealed_access"]
    return build_prospective_lock(
        file_hashes=value["file_hashes"],
        rendered_manifests=value["rendered_manifests"],
        configuration=value["configuration"],
        thresholds=value["thresholds"],
        sealed_dataset_file_key=str(access["sealed_dataset_file_key"]),
        validation_dataset_file_key=str(access["validation_dataset_file_key"]),
        validation_result_schema_version=str(access["validation_result_schema_version"]),
        validation_result_hash_field=str(access["validation_result_hash_field"]),
        required_validation_gates=list(access["required_validation_gates"]),
    )


def build_adaptive_lock(
    *,
    file_hashes: Mapping[str, Any],
    rendered_manifests: Mapping[str, Any],
    configuration: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    sealed_dataset_file_key: str,
    validation_result_schema_version: str,
    required_validation_gates: Sequence[str],
    adaptive_provenance: Mapping[str, Any],
    model_compute_used_to_build_lock: int,
    validation_dataset_file_key: str = "validation_dataset",
    validation_result_hash_field: str = "result_sha256",
) -> dict[str, Any]:
    """Build a canonical lock that openly records prior adaptive evidence."""

    if type(model_compute_used_to_build_lock) is not int or model_compute_used_to_build_lock <= 0:
        raise ValueError("adaptive lock must record positive prior model compute")
    surrogate = build_prospective_lock(
        file_hashes=file_hashes,
        rendered_manifests=rendered_manifests,
        configuration=configuration,
        thresholds=thresholds,
        sealed_dataset_file_key=sealed_dataset_file_key,
        validation_dataset_file_key=validation_dataset_file_key,
        validation_result_schema_version=validation_result_schema_version,
        validation_result_hash_field=validation_result_hash_field,
        required_validation_gates=required_validation_gates,
    )
    provenance = _canonical(adaptive_provenance, label="adaptive_provenance")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("adaptive_provenance must be one nonempty object")
    body = {key: value for key, value in surrogate.items() if key != LOCK_HASH_FIELD}
    body.update(
        {
            "schema_version": ADAPTIVE_LOCK_SCHEMA_VERSION,
            "status": ADAPTIVE_LOCK_STATUS,
            "model_compute_used_to_build_lock": model_compute_used_to_build_lock,
            "adaptive_provenance": provenance,
            ADAPTIVE_PROVENANCE_HASH_FIELD: canonical_sha256(provenance),
        }
    )
    body[LOCK_HASH_FIELD] = canonical_sha256(body)
    return verify_adaptive_lock(body)


def verify_adaptive_lock(value: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless an adaptive lock and its generic components are exact."""

    lock = _canonical(value, label="adaptive lock")
    if not isinstance(lock, dict):
        raise CounterfactualKLProtocolIntegrityError("adaptive lock must be an object")
    expected_keys = {
        "schema_version",
        "status",
        "model_compute_used_to_build_lock",
        "file_hashes",
        "rendered_manifests",
        "configuration",
        "thresholds",
        "sealed_access",
        "component_sha256",
        "adaptive_provenance",
        ADAPTIVE_PROVENANCE_HASH_FIELD,
        LOCK_HASH_FIELD,
    }
    if set(lock) != expected_keys:
        raise CounterfactualKLProtocolIntegrityError("adaptive lock fields differ")
    observed_identity = _require_hash(lock[LOCK_HASH_FIELD], field=LOCK_HASH_FIELD)
    unhashed = dict(lock)
    del unhashed[LOCK_HASH_FIELD]
    if observed_identity != canonical_sha256(unhashed):
        raise CounterfactualKLProtocolIntegrityError("adaptive lock identity differs")
    if (
        lock["schema_version"] != ADAPTIVE_LOCK_SCHEMA_VERSION
        or lock["status"] != ADAPTIVE_LOCK_STATUS
        or type(lock["model_compute_used_to_build_lock"]) is not int
        or lock["model_compute_used_to_build_lock"] <= 0
    ):
        raise CounterfactualKLProtocolIntegrityError("adaptive lock status or compute differs")
    provenance = lock["adaptive_provenance"]
    if not isinstance(provenance, dict) or not provenance:
        raise CounterfactualKLProtocolIntegrityError("adaptive provenance is missing")
    if (
        _require_hash(
            lock[ADAPTIVE_PROVENANCE_HASH_FIELD],
            field=ADAPTIVE_PROVENANCE_HASH_FIELD,
        )
        != canonical_sha256(provenance)
    ):
        raise CounterfactualKLProtocolIntegrityError("adaptive provenance hash differs")
    for field in ("prior_lock_identity_sha256", "prior_result_sha256"):
        _require_hash(provenance.get(field), field=f"adaptive_provenance.{field}")
    if (
        provenance.get("prior_status") != "no_go"
        or provenance.get("prior_forward_backward")
        != lock["model_compute_used_to_build_lock"]
        or provenance.get("prior_nonzero_interventions") != 0
        or provenance.get("prior_steering_outcomes_observed") is not False
        or provenance.get("current_revision_model_compute_before_lock") != 0
        or provenance.get("adaptation_scope")
        != "baseline_qualification_and_preoutcome_gate_strengthening"
    ):
        raise CounterfactualKLProtocolIntegrityError("adaptive provenance facts differ")
    surrogate = _surrogate_lock(lock)
    if (
        surrogate["component_sha256"] != lock["component_sha256"]
        or surrogate["file_hashes"] != lock["file_hashes"]
        or surrogate["rendered_manifests"] != lock["rendered_manifests"]
        or surrogate["configuration"] != lock["configuration"]
        or surrogate["thresholds"] != lock["thresholds"]
        or surrogate["sealed_access"] != lock["sealed_access"]
    ):
        raise CounterfactualKLProtocolIntegrityError(
            "adaptive lock generic components differ"
        )
    return lock


def validate_adaptive_result(
    result: Mapping[str, Any],
    *,
    lock: Mapping[str, Any],
    expected_split: str,
    require_go: bool = False,
) -> dict[str, Any]:
    """Validate a result against the exact adaptive lock and split."""

    checked_lock = verify_adaptive_lock(lock)
    access = checked_lock["sealed_access"]
    if expected_split == access["required_validation_split"]:
        dataset_key = str(access["validation_dataset_file_key"])
    elif expected_split == "sealed":
        dataset_key = str(access["sealed_dataset_file_key"])
    else:
        raise CounterfactualKLProtocolIntegrityError("adaptive result split is not locked")
    checked = verify_self_hashed_record(
        result,
        hash_field=str(access["validation_result_hash_field"]),
    )
    if checked.get("schema_version") != access["validation_result_schema_version"]:
        raise CounterfactualKLProtocolIntegrityError("adaptive result schema differs")
    if checked.get(access["validation_result_lock_field"]) != checked_lock[LOCK_HASH_FIELD]:
        raise CounterfactualKLProtocolIntegrityError("adaptive result lock binding differs")
    if checked.get(access["validation_result_split_field"]) != expected_split:
        raise CounterfactualKLProtocolIntegrityError("adaptive result split differs")
    expected_dataset_hash = checked_lock["file_hashes"][dataset_key]["sha256"]
    if checked.get(access["validation_result_dataset_hash_field"]) != expected_dataset_hash:
        raise CounterfactualKLProtocolIntegrityError("adaptive result dataset hash differs")
    gates = checked.get("gates")
    required = list(access["required_validation_gates"])
    if not isinstance(gates, Mapping) or set(gates) != set(required):
        raise CounterfactualKLProtocolIntegrityError("adaptive result gates differ")
    if any(type(gates[name]) is not bool for name in required):
        raise CounterfactualKLProtocolIntegrityError(
            "adaptive result gates must be literal booleans"
        )
    all_passed = all(gates[name] is True for name in required)
    expected_status = access["required_status"] if all_passed else access["required_non_go_status"]
    if require_go and not all_passed:
        raise CounterfactualKLProtocolIntegrityError("adaptive result is not an all-gate go")
    if checked.get("status") != expected_status:
        raise CounterfactualKLProtocolIntegrityError("adaptive result status differs")
    return checked


def load_adaptive_sealed_dataset(
    sealed_path: str | Path,
    validation_result_path: str | Path,
    *,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Parse sealed bytes only after exact adaptive validation authorization."""

    checked_lock = verify_adaptive_lock(lock)
    access = checked_lock["sealed_access"]
    validation_bytes = Path(validation_result_path).read_bytes()
    try:
        validation_result = json.loads(validation_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SealedAccessDenied("adaptive validation result is not valid JSON") from exc
    if not isinstance(validation_result, dict):
        raise SealedAccessDenied("adaptive validation result must be an object")
    try:
        validate_adaptive_result(
            validation_result,
            lock=checked_lock,
            expected_split=str(access["required_validation_split"]),
            require_go=True,
        )
    except CounterfactualKLProtocolIntegrityError as exc:
        raise SealedAccessDenied(str(exc)) from exc
    sealed_entry = checked_lock["file_hashes"][access["sealed_dataset_file_key"]]
    sealed_bytes = Path(sealed_path).read_bytes()
    if hashlib.sha256(sealed_bytes).hexdigest() != sealed_entry["sha256"]:
        raise CounterfactualKLProtocolIntegrityError("adaptive sealed byte hash differs")
    if "bytes" in sealed_entry and len(sealed_bytes) != sealed_entry["bytes"]:
        raise CounterfactualKLProtocolIntegrityError("adaptive sealed byte count differs")
    try:
        payload = json.loads(sealed_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CounterfactualKLProtocolIntegrityError(
            "adaptive sealed dataset is not valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise CounterfactualKLProtocolIntegrityError(
            "adaptive sealed dataset must contain an object"
        )
    return payload


__all__ = [
    "ADAPTIVE_LOCK_SCHEMA_VERSION",
    "ADAPTIVE_LOCK_STATUS",
    "ADAPTIVE_PROVENANCE_HASH_FIELD",
    "build_adaptive_lock",
    "load_adaptive_sealed_dataset",
    "validate_adaptive_result",
    "verify_adaptive_lock",
]
