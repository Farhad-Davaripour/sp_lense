"""Reuse applicable focused synthetic cases; add only clean01 identity/gate checks."""

from __future__ import annotations

import copy
import importlib.util
import json
import marshal
from types import SimpleNamespace

import pytest

from scripts import iterative_guarded_preserve as old_job
from scripts import iterative_guarded_preserve_clean01 as entry
from scripts import iterative_guarded_preserve_clean01_plan as protocol
from scripts import iterative_guarded_preserve_plan as old_plan
from scripts import verify_iterative_guarded_preserve as old_audit
from scripts import verify_iterative_guarded_preserve_clean01 as audit_entry

# Independent test-module globals route the existing focused tests to clean01.
# No existing test module/source or old runner/audit globals are mutated.
_spec = importlib.util.spec_from_file_location(
    "_clean01_reused_focused_tests", protocol.ROOT / "tests/test_iterative_guarded_preserve.py"
)
reused = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reused)
reused.job, reused.audit, reused.protocol = entry.engine, audit_entry.engine, protocol
for _name, _value in vars(reused).items():
    if _name.startswith("test_") and _name != "test_plan_original_only_goals_and_fresh_zero":
        globals()[_name] = _value


def test_plan_original_only_goals_and_fresh_zero():
    original, clean = old_plan.build_plan(), protocol.build_plan()
    for k, v in original.items():
        if k not in ("config", "output_namespace", "input_sha256"):
            assert clean[k] == v, k
    assert clean["initial_shared_w"] == [0.0] * 1024
    assert (
        clean["frozen_goals_sha256"]
        == "2d7a242d78ed0b63e0a2def082665b4d5c89c9d7f5fcbc2ee090135655eb46be"
    )
    cfg = {k: v for k, v in clean["config"].items() if k != "clean_successor"}
    cfg["output_namespace"] = old_plan.OUTPUT
    assert cfg == original["config"]
    assert clean["predecessor"]["predecessor_scientific_observations"] is False
    assert clean["output_namespace"] != original["output_namespace"]
    assert all(clean["input_sha256"][p] == h for p, h in original["input_sha256"].items())


def test_isolated_namespace_and_identical_engine_bytecode():
    assert entry.engine is not old_job and audit_entry.engine is not old_audit
    assert old_job.protocol is old_plan and old_audit.protocol is old_plan
    assert entry.engine.protocol is audit_entry.engine.protocol is protocol
    assert entry.engine.OUTPUT == audit_entry.engine.OUTPUT == protocol.ROOT / protocol.OUTPUT
    assert old_job.OUTPUT == old_audit.OUTPUT == protocol.ROOT / old_plan.OUTPUT
    for name in (
        "increment",
        "drive",
        "accepts",
        "guarded_accepts",
        "stopping",
        "summarize",
        "evaluate",
        "worker",
        "run",
        "freeze",
        "require_freeze",
        "source_identity",
        "supervise",
    ):
        assert marshal.dumps(getattr(entry.engine, name).__code__) == marshal.dumps(
            getattr(old_job, name).__code__
        ), name
    for name in (
        "verify_data",
        "verify_update",
        "replay",
        "verify",
        "freeze_verified_candidate",
        "accepts",
        "guarded_accepts",
        "summary",
        "report",
    ):
        assert marshal.dumps(getattr(audit_entry.engine, name).__code__) == marshal.dumps(
            getattr(old_audit, name).__code__
        ), name
    assert entry.engine.optimizer is old_job.optimizer and entry.engine.project is old_job.project
    assert entry.engine.score_float32_logits is old_job.score_float32_logits
    assert protocol.storage_preflight is old_plan.storage_preflight


@pytest.mark.parametrize("field", ["step_cap", "total_cap", "aim", "acceptance", "timeout_seconds"])
def test_scientific_config_change_rejected(monkeypatch, field):
    original = protocol.read

    def changed(path):
        value = original(path)
        if path == protocol.ROOT / protocol.CONFIG:
            value = copy.deepcopy(value)
            value[field] += 0.01
        return value

    monkeypatch.setattr(protocol, "read", changed)
    with pytest.raises(ValueError, match="unchanged scientific config"):
        protocol.build_plan()


def fake_prelaunch(tmp_path, monkeypatch, fail_index=None):
    lock = {
        "source_commit": "f" * 40,
        "environment": {"synthetic": True},
        "source_sha256": {},
        "plan": "synthetic",
    }
    protocol.io.write_new(tmp_path / "preregistration.json", lock)
    monkeypatch.setattr(entry, "OUTPUT", tmp_path)
    monkeypatch.setattr(entry.engine, "require_freeze", lambda: lock)
    monkeypatch.setattr(entry.engine.base, "environment", lambda: lock["environment"])
    monkeypatch.setattr(entry, "lock_commit", lambda _: "e" * 40)
    calls = []

    def git(args, **kwargs):
        calls.append(args)
        index = len(calls) - 1
        assert kwargs == {"capture_output": True, "text": True, "check": False}
        return SimpleNamespace(
            returncode=128 if index == fail_index else 0,
            stdout=["tree\n", "a" * 40 + "\n", "", ""][index],
            stderr="synthetic denied" if index == fail_index else "",
        )

    monkeypatch.setattr(entry.subprocess, "run", git)
    return calls


def test_one_shot_prelaunch_success_before_any_worker(tmp_path, monkeypatch):
    calls = fake_prelaunch(tmp_path, monkeypatch)
    pre = entry.prelaunch()
    assert pre["status"] == "passed" and len(calls) == 4
    assert protocol.check_prelaunch(tmp_path) == pre
    assert not (tmp_path / "WORKER_CLAIM.json").exists()
    assert not (tmp_path / "RUN_STARTED.json").exists()
    with pytest.raises(ValueError, match="no resume"):
        entry.prelaunch()
    assert len(calls) == 4


def test_failed_prelaunch_stops_without_read_retry_or_worker(tmp_path, monkeypatch):
    calls = fake_prelaunch(tmp_path, monkeypatch, fail_index=2)
    pre = entry.prelaunch()
    assert pre["status"] == "INCONCLUSIVE" and len(calls) == 3
    assert pre["git_checks"][-1]["returncode"] == 128
    assert pre["git_checks"][-1]["stderr"] == "synthetic denied"
    with pytest.raises(ValueError, match="prelaunch failed"):
        entry.run()
    assert not (tmp_path / "WORKER_CLAIM.json").exists()
    assert not (tmp_path / "RUN_STARTED.json").exists()
    monkeypatch.setattr(audit_entry, "OUTPUT", tmp_path)
    monkeypatch.setattr(protocol, "build_plan", lambda: "synthetic")
    result = audit_entry.verify()
    assert result["status"] == "INCONCLUSIVE" and result["model_calls"] == 0
    assert result["scientific_observations"] is False
    assert len(calls) == 3


def test_source_identity_failure_retained_without_worker(tmp_path, monkeypatch):
    calls = fake_prelaunch(tmp_path, monkeypatch)

    def failed():
        raise ValueError("synthetic full source cleanliness failure")

    monkeypatch.setattr(entry.engine, "require_freeze", failed)
    pre = entry.prelaunch()
    assert pre["status"] == "INCONCLUSIVE" and not pre["source_identity_passed"]
    assert len(calls) == 4 and "source cleanliness failure" in pre["fault"]
    with pytest.raises(ValueError, match="prelaunch failed"):
        entry.run()


@pytest.mark.parametrize(
    "artifact", ["RUN_STARTED.json", "WORKER_CLAIM.json", "endpoint.json", "rows.jsonl"]
)
def test_no_resume_after_any_partial_runtime_state(tmp_path, monkeypatch, artifact):
    fake_prelaunch(tmp_path, monkeypatch)
    entry.prelaunch()
    protocol.io.write_new(tmp_path / artifact, {"synthetic": True})
    with pytest.raises(ValueError, match="no resume"):
        entry.run()


def test_prelaunch_record_hash_tamper_blocks_worker(tmp_path, monkeypatch):
    fake_prelaunch(tmp_path, monkeypatch)
    entry.prelaunch()
    claim = protocol.read(tmp_path / "PRELAUNCH_CLAIM.json")
    claim["pid"] += 1
    (tmp_path / "PRELAUNCH_CLAIM.json").write_text(json.dumps(claim))
    with pytest.raises(ValueError, match="bound one-shot"):
        entry.run()
