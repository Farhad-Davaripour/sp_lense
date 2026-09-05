"""Changed wrapper-only fake tests; no model, historical result audit or subprocess."""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json

import pytest

from scripts import paired_common_drift_comply_1800_recording as recording
from scripts.three_family_recording_budget import BudgetError


def encoded(value):
    return (json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode()


def fake_certificate(root):
    """Copy immutable preparation source bytes only; no run artifacts or results."""
    parent = recording.parent
    for name in recording.CERTIFICATE_SOURCE_PATHS:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name in recording.NEW_SOURCE_PATHS and name != recording.POLICY:
            data = ("synthetic source: " + name + "\n").encode()
        else:
            data = (recording.ROOT / name).read_bytes()
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
        "recording_policy_sha256": recording.POLICY_SHA,
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in recording.CERTIFICATE_SOURCE_PATHS
        },
    }
    assert report["source_sha256"][parent.CERTIFICATE] == recording.PARENT_CERTIFICATE_SHA
    (root / recording.CERTIFICATE).write_bytes(encoded(report))
    return report


def test_only_identity_deadline_and_provenance_policy_change():
    old = recording.parent.recording_policy()
    new = recording.recording_policy()
    changed = {"schema", "scope", "timeout_including_loading_seconds", "limitations"}
    added = {"output_namespace", "provenance"}
    assert set(new) == set(old) | added
    assert all(new[key] == value for key, value in old.items() if key not in changed)
    assert new["timeout_including_loading_seconds"] == 1800
    assert old["timeout_including_loading_seconds"] == 1200
    assert new["output_namespace"] == recording.OUTPUT
    assert recording.PairedBudget is recording.parent.PairedBudget
    assert recording.RECORD_CAPS is recording.parent.RECORD_CAPS
    assert new["combined_category_bound_bytes"] == 524995016 < new["quotas"]["total"] == 536870912
    assert new["auxiliary_ceiling_bytes"] == 16777216
    assert new["minimum_free_bytes"] == 1073741824
    assert (new["maximum_forwards"], new["maximum_derivatives"], new["maximum_updates"]) == (
        216,
        96,
        8,
    )


def test_wrapper_imports_no_model_or_runtime(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "transformers", "transformer_lens", "tokenizers"}
        assert name not in {
            "scripts.paired_common_drift_comply_1800",
            "scripts.paired_common_drift_comply_1800_plan",
            "scripts.verify_paired_common_drift_comply_1800",
            "scripts.paired_common_drift_comply_optimizer",
        }
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location(
        "_recording_1800_import_probe", recording.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PairedBudget is recording.parent.PairedBudget


def test_positive_certificate_exact_old_and_new_sources(tmp_path):
    report = fake_certificate(tmp_path)
    assert recording.require_certificate(tmp_path) == report
    assert len(report["source_sha256"]) == 32
    assert recording.PREPARATION_REPORT in report["source_sha256"]
    assert recording.CERTIFICATE not in report["source_sha256"]


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "dirty_new", "dirty_old", "dirty_parent_certificate"]
)
def test_certificate_rejects_changed_source_or_parent(tmp_path, mutation):
    report = fake_certificate(tmp_path)
    path = "scripts/paired_common_drift_comply_1800.py"
    if mutation == "missing":
        del report["source_sha256"][path]
    elif mutation == "extra":
        report["source_sha256"]["unexpected.json"] = "0" * 64
    else:
        path = {
            "dirty_new": path,
            "dirty_old": "scripts/paired_common_drift_comply_optimizer.py",
            "dirty_parent_certificate": recording.parent.CERTIFICATE,
        }[mutation]
        (tmp_path / path).write_bytes(b"changed bytes")
    (tmp_path / recording.CERTIFICATE).write_bytes(encoded(report))
    with pytest.raises(ValueError):
        recording.require_certificate(tmp_path)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("status", "PREPARATION_INCOMPLETE"),
        ("storage_certified", False),
        ("accounting_certified", False),
        ("run_authorized", True),
        ("model_loads", 1),
        ("tokenizer_loads", 1),
        ("real_forwards", 1),
        ("real_derivatives", 1),
        ("model_loads", False),
    ],
)
def test_certificate_never_promotes_incomplete_or_model_work(tmp_path, key, value):
    report = fake_certificate(tmp_path)
    report[key] = value
    (tmp_path / recording.CERTIFICATE).write_bytes(encoded(report))
    with pytest.raises(ValueError, match="positive exact-source 1800"):
        recording.require_certificate(tmp_path)


def test_new_policy_or_missing_certificate_never_repaired(tmp_path):
    fake_certificate(tmp_path)
    certificate = tmp_path / recording.CERTIFICATE
    certificate.unlink()
    with pytest.raises(FileNotFoundError):
        recording.require_certificate(tmp_path)
    assert not certificate.exists()
    policy = tmp_path / recording.POLICY
    policy.write_bytes(policy.read_bytes() + b" ")
    with pytest.raises(ValueError, match="exact 1800 paired recording policy bytes"):
        recording.recording_policy(tmp_path)


def test_unchanged_namespace_bound_and_fake_failure_seal(tmp_path):
    root = tmp_path / "fresh_1800"
    root.mkdir()
    budget = recording.PairedBudget(root, quotas={"rows": 32})
    row = encoded({"row": "complete"})
    budget.write_bytes("rows.jsonl", row, mode="ab")
    with pytest.raises(BudgetError, match="ARTIFACT_BYTE_CAP"):
        budget.write_bytes("rows.jsonl", row, mode="ab")
    assert (root / "rows.jsonl").read_bytes() == row
    budget.begin_finalization(quiescent=True)
    budget.write_bytes("capture_receipt.json", encoded({"status": "INCONCLUSIVE"}), final=True)
    with pytest.raises(BudgetError, match="CANDIDATE_AFTER_RECORDING_FAULT"):
        budget.write_bytes("comply_vector.json", b"{}", final=True)
    inventory = budget.finalize_inventory(valid_candidate=True, quiescent=True)
    assert inventory["valid_candidate"] is False
    assert (
        recording.PairedBudget(root, initialize=False).verify_inventory()["inventory"] == inventory
    )
