"""Clean01 recording/certificate-only fake checks; no entrypoint or model execution."""

from __future__ import annotations

import hashlib
import json

import pytest

from scripts import paired_common_drift_comply_1800_clean01_recording as recording


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def fake_certificate(root):
    for name in recording.CERTIFICATE_SOURCE_PATHS:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = (
            ("synthetic source: " + name + "\n").encode()
            if name in recording.NEW_SOURCE_PATHS and name != recording.POLICY
            else (recording.ROOT / name).read_bytes()
        )
        target.write_bytes(data)
    report = {
        "schema": recording.CERTIFICATE_SCHEMA,
        "status": "MODEL_FREE_PREPARATION_CERTIFIED",
        "storage_certified": True,
        "accounting_certified": True,
        "run_authorized": False,
        "model_loads": 0,
        "tokenizer_loads": 0,
        "real_forwards": 0,
        "real_derivatives": 0,
        "entrypoint_coverage": dict.fromkeys(recording.ENTRYPOINT_COVERAGE_KEYS, True),
        "recording_policy_sha256": recording.POLICY_SHA,
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in recording.CERTIFICATE_SOURCE_PATHS
        },
    }
    (root / recording.CERTIFICATE).write_bytes(encoded(report))
    return report


def test_policy_identity_only_and_exact_budget_alias():
    old, new = recording.parent.recording_policy(), recording.recording_policy()
    allowed = {"schema", "scope", "output_namespace", "provenance"}
    assert set(new) == set(old)
    assert all(new[key] == value for key, value in old.items() if key not in allowed)
    assert new["timeout_including_loading_seconds"] == 1800
    assert recording.PairedBudget is recording.parent.PairedBudget
    assert recording.RECORD_CAPS is recording.parent.RECORD_CAPS
    assert new["combined_category_bound_bytes"] == 524995016 < new["quotas"]["total"] == 536870912
    assert new["auxiliary_ceiling_bytes"] == 16777216
    assert new["minimum_free_bytes"] == 1073741824
    assert (
        "known CLI-entrypoint coverage gap" in new["provenance"]["parent_certificate_disposition"]
    )


def test_new_positive_certificate_does_not_inherit_prior_readiness(tmp_path, monkeypatch):
    report = fake_certificate(tmp_path)

    def forbid(*args, **kwargs):
        raise AssertionError("historical launcher readiness must not be inherited")

    monkeypatch.setattr(recording.parent, "require_certificate", forbid)
    assert recording.require_certificate(tmp_path) == report
    assert len(report["source_sha256"]) == 44
    assert recording.CERTIFICATE not in report["source_sha256"]
    assert recording.PREPARATION_REPORT in report["source_sha256"]


@pytest.mark.parametrize("key", sorted(recording.ENTRYPOINT_COVERAGE_KEYS))
@pytest.mark.parametrize("value", [None, False, 1])
def test_each_entrypoint_coverage_attestation_requires_literal_true(tmp_path, key, value):
    report = fake_certificate(tmp_path)
    if value is None:
        del report["entrypoint_coverage"][key]
    else:
        report["entrypoint_coverage"][key] = value
    (tmp_path / recording.CERTIFICATE).write_bytes(encoded(report))
    with pytest.raises(ValueError, match="entrypoint-covered"):
        recording.require_certificate(tmp_path)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("model_loads", 1),
        ("tokenizer_loads", 1),
        ("real_forwards", 1),
        ("real_derivatives", 1),
        ("model_loads", False),
        ("run_authorized", True),
        ("storage_certified", False),
        ("accounting_certified", False),
        ("status", "PREPARATION_INCOMPLETE"),
    ],
)
def test_no_nonzero_real_work_or_incomplete_promotion(tmp_path, key, value):
    report = fake_certificate(tmp_path)
    report[key] = value
    (tmp_path / recording.CERTIFICATE).write_bytes(encoded(report))
    with pytest.raises(ValueError, match="entrypoint-covered"):
        recording.require_certificate(tmp_path)


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "new_source", "old_source", "old_certificate", "new_policy"]
)
def test_exact_frozen_source_and_policy_bytes_required(tmp_path, mutation):
    report = fake_certificate(tmp_path)
    name = "scripts/paired_common_drift_comply_1800_clean01.py"
    if mutation == "missing":
        del report["source_sha256"][name]
    elif mutation == "extra":
        report["source_sha256"]["unexpected.json"] = "0" * 64
    else:
        path = {
            "new_source": name,
            "old_source": "scripts/paired_common_drift_comply_1800.py",
            "old_certificate": recording.parent.CERTIFICATE,
            "new_policy": recording.POLICY,
        }[mutation]
        (tmp_path / path).write_bytes(b"changed source")
    (tmp_path / recording.CERTIFICATE).write_bytes(encoded(report))
    with pytest.raises(ValueError):
        recording.require_certificate(tmp_path)


def test_missing_new_certificate_never_created(tmp_path):
    fake_certificate(tmp_path)
    path = tmp_path / recording.CERTIFICATE
    path.unlink()
    with pytest.raises(FileNotFoundError):
        recording.require_certificate(tmp_path)
    assert not path.exists()
