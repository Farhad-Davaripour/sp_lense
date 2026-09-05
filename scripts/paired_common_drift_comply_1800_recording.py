"""1800-second identity/certificate wrapper; immutable paired recording mechanics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import paired_common_drift_comply_recording as parent

ROOT = parent.ROOT
POLICY = "configs/paired_common_drift_comply_1800_recording_policy.json"
POLICY_SHA = "319a1d6cd760febd92d014e662c62df90c4da982e26f28688100aefbdba7c084"
OUTPUT = "evidence/paired_common_drift_comply_three_family_1800_v1_qwen35_08b"
CERTIFICATE = "docs/paired_common_drift_comply_1800_preparation.json"
CERTIFICATE_SCHEMA = "sp_lense.paired_common_drift_comply_1800_preparation.v1"
PREPARATION_REPORT = "docs/PAIRED_COMMON_DRIFT_COMPLY_1800_PREPARATION.md"
PARENT_CERTIFICATE_SHA = "57569686bee52c56d9cbb74ff764c2200814613c83d1119a45a38c2343075253"
PairedBudget = parent.PairedBudget
QUOTAS = parent.QUOTAS
RECORD_CAPS = parent.RECORD_CAPS
ALLOWED = parent.ALLOWED
LOGITS = parent.LOGITS
IMMUTABLE_HELPERS = parent.IMMUTABLE_HELPERS
NEW_SOURCE_PATHS = frozenset(
    {
        "configs/paired_common_drift_comply_three_family_1800_v1.json",
        POLICY,
        "scripts/paired_common_drift_comply_1800_plan.py",
        "scripts/paired_common_drift_comply_1800.py",
        "scripts/verify_paired_common_drift_comply_1800.py",
        "scripts/paired_common_drift_comply_1800_recording.py",
        "scripts/paired_common_drift_comply_1800_binding.py",
        "tests/test_paired_common_drift_comply_1800_plan.py",
        "tests/test_paired_common_drift_comply_1800_recording.py",
        "tests/test_paired_common_drift_comply_1800.py",
        "docs/PAIRED_COMMON_DRIFT_COMPLY_1800_V1.md",
        PREPARATION_REPORT,
    }
)
CERTIFICATE_SOURCE_PATHS = frozenset(
    {*parent.CERTIFICATE_SOURCE_PATHS, parent.CERTIFICATE, *NEW_SOURCE_PATHS}
)


def recording_policy(root=ROOT):
    """Allow only successor identity/provenance and the 1800-second deadline."""
    original = parent.recording_policy(root)
    raw = (Path(root) / POLICY).read_bytes()
    if hashlib.sha256(raw).hexdigest() != POLICY_SHA:
        raise ValueError("exact 1800 paired recording policy bytes required")
    policy = json.loads(raw)
    expected = {
        **original,
        "schema": "sp_lense.paired_common_drift_comply_1800_recording_policy.v1",
        "scope": original["scope"].replace("1200s", "1800s"),
        "timeout_including_loading_seconds": 1800,
        "limitations": [text.replace("1200s", "1800s") for text in original["limitations"]],
        "output_namespace": OUTPUT,
        "provenance": {
            "parent_policy": parent.POLICY,
            "parent_policy_sha256": parent.POLICY_SHA,
            "change": "Fresh-zero successor identity and 1800-second worker deadline only; no reuse of partial vectors or expanded scientific/storage budgets.",
        },
    }
    if policy != expected:
        raise ValueError("1800 identity/deadline-only recording policy required")
    return policy


def require_certificate(root=ROOT):
    """Both exact model-free source certificates are required; neither authorizes a run."""
    root = Path(root)
    recording_policy(root)
    if (
        hashlib.sha256((root / parent.CERTIFICATE).read_bytes()).hexdigest()
        != PARENT_CERTIFICATE_SHA
    ):
        raise ValueError("exact immutable parent preparation certificate required")
    prior = parent.require_certificate(root)
    report = json.loads((root / CERTIFICATE).read_bytes())
    hashes = report.get("source_sha256")
    zero_fields = ("model_loads", "tokenizer_loads", "real_forwards", "real_derivatives")
    if not (
        report.get("schema") == CERTIFICATE_SCHEMA
        and report.get("status") == "MODEL_FREE_PREPARATION_CERTIFIED"
        and report.get("storage_certified") is True
        and report.get("accounting_certified") is True
        and report.get("run_authorized") is False
        and all(type(report.get(key)) is int and report[key] == 0 for key in zero_fields)
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
        raise ValueError(
            "positive exact-source 1800 paired model-free preparation certificate required"
        )
    return report
