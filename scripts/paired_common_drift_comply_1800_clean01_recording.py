"""Clean01 recording identity and new entrypoint-coverage certificate; no model work."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import paired_common_drift_comply_1800_recording as parent

ROOT = parent.ROOT
POLICY = "configs/paired_common_drift_comply_1800_clean01_recording_policy.json"
POLICY_SHA = "c52ad43ff2b4f57b414218951071bee4110bfd036abf322d49430a3fb81f3625"
OUTPUT = "evidence/paired_common_drift_comply_three_family_1800_clean01_v1_qwen35_08b"
CERTIFICATE = "docs/paired_common_drift_comply_1800_clean01_preparation.json"
CERTIFICATE_SCHEMA = "sp_lense.paired_common_drift_comply_1800_clean01_preparation.v1"
PREPARATION_REPORT = "docs/PAIRED_COMMON_DRIFT_COMPLY_1800_CLEAN01_PREPARATION.md"
PARENT_CERTIFICATE_SHA = "fb25e4a26b51e4838cb726e9864bad4e953a38fe993dcac7f10c38b9424d7f80"
HISTORICAL_DISPOSITION = "Immutable historical source certificate with known CLI-entrypoint coverage gap; not current launch approval."
PairedBudget = parent.PairedBudget
QUOTAS, RECORD_CAPS = parent.QUOTAS, parent.RECORD_CAPS
ALLOWED, LOGITS, IMMUTABLE_HELPERS = parent.ALLOWED, parent.LOGITS, parent.IMMUTABLE_HELPERS
ENTRYPOINT_COVERAGE_KEYS = frozenset(
    {
        "actual_cli_entrypoint_tested",
        "worker_dispatch_reached",
        "pre_backend_load_sentinel_tested",
        "supervisor_exact_command_bounded_capture_tested",
        "single_worker_no_retry_tested",
        "unknown_command_failed_closed",
    }
)
NEW_SOURCE_PATHS = frozenset(
    {
        "configs/paired_common_drift_comply_three_family_1800_clean01_v1.json",
        POLICY,
        "scripts/paired_common_drift_comply_1800_clean01_plan.py",
        "scripts/paired_common_drift_comply_1800_clean01.py",
        "scripts/verify_paired_common_drift_comply_1800_clean01.py",
        "scripts/paired_common_drift_comply_1800_clean01_recording.py",
        "tests/test_paired_common_drift_comply_1800_clean01_plan.py",
        "tests/test_paired_common_drift_comply_1800_clean01_recording.py",
        "tests/test_paired_common_drift_comply_1800_clean01.py",
        "docs/PAIRED_COMMON_DRIFT_COMPLY_1800_CLEAN01_V1.md",
        PREPARATION_REPORT,
    }
)
CERTIFICATE_SOURCE_PATHS = frozenset(
    {*parent.CERTIFICATE_SOURCE_PATHS, parent.CERTIFICATE, *NEW_SOURCE_PATHS}
)


def recording_policy(root=ROOT):
    original = parent.recording_policy(root)
    raw = (Path(root) / POLICY).read_bytes()
    if hashlib.sha256(raw).hexdigest() != POLICY_SHA:
        raise ValueError("exact clean01 recording policy bytes required")
    policy = json.loads(raw)
    expected = {
        **original,
        "schema": "sp_lense.paired_common_drift_comply_1800_clean01_recording_policy.v1",
        "scope": "One fresh clean01 paired-common-drift COMPLY construction:216F/96D/8updates/1800s including loading; no retry; preparation is not run authorization",
        "output_namespace": OUTPUT,
        "provenance": {
            "parent_policy": parent.POLICY,
            "parent_policy_sha256": parent.POLICY_SHA,
            "parent_preparation_certificate": parent.CERTIFICATE,
            "parent_preparation_certificate_sha256": PARENT_CERTIFICATE_SHA,
            "parent_certificate_disposition": HISTORICAL_DISPOSITION,
            "change": "Fresh clean01 launcher-repair identity only; unchanged 1800-second deadline, scientific method, serializers and storage budgets; no prior output coordinates.",
        },
    }
    if policy != expected:
        raise ValueError("clean01 identity/provenance-only recording policy required")
    return policy


def historical_certificate(root=ROOT):
    """Authenticate provenance bytes, never assert the prior launcher was covered."""
    root = Path(root)
    raw = (root / parent.CERTIFICATE).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PARENT_CERTIFICATE_SHA:
        raise ValueError("exact historical coverage-gap certificate required")
    report = json.loads(raw)
    hashes = report.get("source_sha256")
    if not (
        isinstance(hashes, dict)
        and set(hashes) == parent.CERTIFICATE_SOURCE_PATHS
        and all(
            hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
            for path, digest in hashes.items()
        )
    ):
        raise ValueError("exact historical certificate source bytes required")
    return report


def require_certificate(root=ROOT):
    """Require newly covered CLI source; historical readiness is not inherited."""
    root = Path(root)
    recording_policy(root)
    prior = historical_certificate(root)
    report = json.loads((root / CERTIFICATE).read_bytes())
    hashes, coverage = report.get("source_sha256"), report.get("entrypoint_coverage")
    zero_fields = ("model_loads", "tokenizer_loads", "real_forwards", "real_derivatives")
    if not (
        report.get("schema") == CERTIFICATE_SCHEMA
        and report.get("status") == "MODEL_FREE_PREPARATION_CERTIFIED"
        and report.get("storage_certified") is True
        and report.get("accounting_certified") is True
        and report.get("run_authorized") is False
        and all(type(report.get(key)) is int and report[key] == 0 for key in zero_fields)
        and isinstance(coverage, dict)
        and all(coverage.get(key) is True for key in ENTRYPOINT_COVERAGE_KEYS)
        and report.get("recording_policy_sha256") == POLICY_SHA
        and isinstance(hashes, dict)
        and set(hashes) == CERTIFICATE_SOURCE_PATHS
        and hashes[parent.CERTIFICATE] == PARENT_CERTIFICATE_SHA
        and all(hashes[path] == digest for path, digest in prior["source_sha256"].items())
        and all(
            hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
            for path, digest in hashes.items()
        )
    ):
        raise ValueError("positive exact-source clean01 entrypoint-covered preparation required")
    return report
