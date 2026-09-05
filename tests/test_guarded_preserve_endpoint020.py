"""Reuse applicable parent contracts; narrow endpoint/baseline/provenance tests."""

from __future__ import annotations

import copy
import inspect
import json
import math
import time
from types import SimpleNamespace

import pytest

from scripts import frozen_guarded_preserve_crossed as original_job
from scripts import frozen_guarded_preserve_crossed_plan as original_protocol
from scripts import guarded_preserve_endpoint020 as entry
from scripts import guarded_preserve_endpoint020_plan as protocol
from scripts import verify_frozen_guarded_preserve_crossed as original_audit
from scripts import verify_guarded_preserve_endpoint020 as verify_entry

job, audit = entry.job, verify_entry.audit
parent_tests = protocol.isolate(
    "_endpoint020_reused_tests", "tests/test_frozen_guarded_preserve_crossed.py"
)
parent_tests.job, parent_tests.audit, parent_tests.protocol = job, audit, protocol
PARENT_PREPARE = parent_tests.prepare


def prepare(mode="linear", offset=0.1, quality_failure=False):
    plan, backend, vectors = PARENT_PREPARE(mode, offset, quality_failure)
    for p, snap in zip(plan["prompts"], plan["prior_comparison"]["baseline_records"], strict=True):
        i = p["rendering_index"] - 1
        h0 = [0.0, 3.0 + i, 4.0]
        S = -offset if p["display_order"] == "A_then_B" else offset
        if mode == "all_B":
            S = -offset if p["preserve_label"] == "A" else offset
        label = p["preserve_label"] if S > 0 else p["comply_label"]
        snap["row"].update(
            h0=h0,
            h0_norm=protocol.norm(h0),
            preserve_log_odds=audit.f32(S),
            actual_next_token_label=label,
            actual_next_token_id=0 if label == "A" else 1,
        )
        snap["row_sha256"] = protocol.canonical_sha(snap["row"])
    return plan, backend, vectors


parent_tests.prepare = prepare
EXCLUDED = {
    "test_candidate_chain_same_eight_fit_ids_disjoint_case_and_exact_guarded_arrow",
    "test_candidate_audit_chain_corruption",
    "test_storage_guard_and_no_historical_f03_outcome_reads",
    "test_unchanged_physical_scoring_and_audit_except_scope_and_auxiliary_record",
    "test_usage_preflight_blocks_missing_capped_or_stale",
    "test_one_prelaunch_no_resume_and_no_worker_on_git_failure",
}
for _name, _fn in vars(parent_tests).copy().items():
    if _name.startswith("test_") and _name not in EXCLUDED:
        globals()[_name] = _fn


def test_endpoint_source_chain_no_transfer_of_training_success():
    plan = protocol.build_plan()
    original = original_protocol.build_plan()
    source = plan["source_candidate"]
    condition = plan["derived_condition"]
    assert source == original["candidates"]["preserve"]
    assert source["guarded_training_audit_verified"] and len(source["fitted_prompt_ids"]) == 8
    assert plan["prompts"] == original["prompts"] and plan["cells"] == original["cells"]
    assert condition["norm"] == 0.20
    assert condition["scale"] == 1.8579477985409942
    assert (
        condition["vector_float64_le_sha256"]
        == "5ae7092a5cf0db6329bee8890583267c83a3d208e4bf08d3a611303a364a7a16"
    )
    assert (
        plan["candidates"]["preserve"]["file_sha256"]
        == "1fb7385f6988aacc544d9bdc01351a4cab733812f73a272f8e3c56e91c98fdf4"
    )
    assert (
        not condition["newly_trained_candidate"]
        and not condition["source_training_success_transfers"]
    )
    assert not plan["candidates"]["preserve"]["guarded_training_audit_verified"]
    assert len(plan["prior_comparison"]["baseline_records"]) == 4
    assert len(plan["prior_comparison"]["edit_records"]) == 4


def test_derivation_deterministic_source_only_and_native_cap():
    cfg = protocol.read(protocol.ROOT / protocol.CONFIG)
    a = protocol.derive_condition(cfg)
    changed = copy.deepcopy(cfg)
    changed["prior_comparison"] = {"edited_scores": [float("nan"), 999999]}
    b = protocol.derive_condition(changed)
    assert protocol.serialized(a) == protocol.serialized(b)
    assert len(a["vector"]) == 1024
    assert abs(math.sqrt(math.fsum(x * x for x in a["vector"])) - 0.2) <= 1e-12
    assert "prior_comparison" not in inspect.getsource(protocol.derive_condition)
    assert original_protocol.candidates()["preserve"]["vector"] != a["vector"]


@pytest.mark.parametrize("fault", ["source_hash", "source_norm", "source_vector_hash", "radius"])
def test_derivation_input_corruption(fault):
    cfg = protocol.read(protocol.ROOT / protocol.CONFIG)
    spec = cfg["candidates"]["preserve"]
    if fault == "source_hash":
        spec["file_sha256"] = "bad"
    elif fault == "source_norm":
        spec["norm"] += 0.01
    elif fault == "source_vector_hash":
        spec["vector_float64_le_sha256"] = "bad"
    else:
        cfg["endpoint_condition"]["radius"] = 0.19
    with pytest.raises(ValueError):
        protocol.derive_condition(cfg)


@pytest.mark.parametrize("fault", [None, "coordinate", "double_scale", "hash", "training_claim"])
def test_independent_condition_reconstruction_and_corruption(tmp_path, fault):
    plan = protocol.build_plan()
    path = tmp_path / "condition.json"
    condition = copy.deepcopy(plan["derived_condition"])
    plan["candidates"]["preserve"]["path"] = str(path)
    if fault == "coordinate":
        condition["vector"][0] += 1e-9
    elif fault == "double_scale":
        condition["vector"] = [condition["scale"] * x for x in condition["vector"]]
    elif fault == "hash":
        plan["candidates"]["preserve"]["file_sha256"] = "bad"
    elif fault == "training_claim":
        plan["candidates"]["preserve"]["guarded_training_audit_verified"] = True
    path.write_bytes(protocol.serialized(condition))
    if fault:
        with pytest.raises(ValueError, match="endpoint"):
            verify_entry.independent_condition(plan)
    else:
        result = verify_entry.independent_condition(plan)
        assert result["independently_reconstructed"] and result["norm"] == 0.20


@pytest.mark.parametrize("fault", ["double", "renormalize", "wrong_own_norm"])
def test_endpoint_physics_single_scaling_own_baseline(tmp_path, fault):
    plan, backend, vectors = prepare()
    vectors["preserve"] = [0.2, 0.0, 0.0]
    plan["candidates"]["preserve"].update(
        norm=0.2, vector_float64_le_sha256=protocol.vector_sha(vectors["preserve"])
    )
    if fault == "double":
        vectors["preserve"][0] *= 1.8579477985409942
    elif fault == "renormalize":
        vectors["preserve"][0] = 1.0
    ledger = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    if fault != "wrong_own_norm":
        with pytest.raises(ValueError):
            job.evaluate(plan, backend, vectors, ledger, tmp_path)
        assert ledger.attempts == 0
    else:
        rows = job.evaluate(plan, backend, vectors, ledger, tmp_path)
        for row in rows[4:]:
            assert row["intended_delta"] == [audit.f32(row["h0_norm"] * 0.2), 0.0, 0.0]
        audit.verify_data(plan, rows, vectors, tmp_path)
        changed = copy.deepcopy(rows)
        changed[5]["intended_delta"] = changed[4]["intended_delta"]
        with pytest.raises(ValueError):
            audit.verify_data(plan, changed, vectors, tmp_path)


@pytest.mark.parametrize(
    "fault", ["h0", "h0_norm", "preserve_log_odds", "label", "argmax", "identity"]
)
def test_fresh_baseline_mismatch_stops_before_first_edit(tmp_path, fault):
    plan, backend, vectors = prepare()
    snap = plan["prior_comparison"]["baseline_records"][3]
    if fault == "h0":
        snap["row"]["h0"][0] += 2e-6
    elif fault in ("h0_norm", "preserve_log_odds"):
        snap["row"][fault] += 2e-6
    elif fault == "label":
        snap["row"]["actual_next_token_label"] = "OTHER"
    elif fault == "argmax":
        snap["row"]["actual_next_token_id"] += 1
    else:
        snap["row"]["prompt_sha256"] = "bad"
    snap["row_sha256"] = protocol.canonical_sha(snap["row"])
    ledger = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    with pytest.raises(ValueError, match="fresh baselines"):
        job.evaluate(plan, backend, vectors, ledger, tmp_path)
    assert ledger.attempts == ledger.completed == len(backend.model.calls) == 4
    assert not protocol.read(tmp_path / "baseline_comparison.json")["passed"]
    assert not (tmp_path / "baseline_goals.json").exists()


@pytest.mark.parametrize("fault", ["early", "late", "hash", "error", "claim", "tolerance"])
def test_independent_baseline_record_corruption(tmp_path, fault):
    plan, _, vectors, _, rows, _ = parent_tests.execute(tmp_path)
    record = protocol.read(tmp_path / "baseline_comparison.json")
    if fault == "early":
        record["monotonic"] = 0
    elif fault == "late":
        record["monotonic"] += 10000
    elif fault == "hash":
        record["prior_rows_sha256"] = "bad"
    elif fault == "error":
        record["comparisons"][0]["S0_difference"] = 1e-9
    elif fault == "claim":
        record["completed_baselines"] = 3
    else:
        record["absolute_tolerance"] = 1e-5
    (tmp_path / "baseline_comparison.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="baseline comparison"):
        audit.verify_data(plan, rows, vectors, tmp_path)


def test_unchanged_core_without_global_leakage():
    for name in ("make_delta", "assess", "worker", "supervise", "source_identity", "prelaunch"):
        assert inspect.getsource(getattr(job, name)) == inspect.getsource(
            getattr(original_job, name)
        )
    assert inspect.getsource(entry.CORE_EVALUATE) == inspect.getsource(original_job.evaluate)
    assert inspect.getsource(verify_entry.CORE_VERIFY_DATA) == inspect.getsource(
        original_audit.verify_data
    )
    assert job.prelaunch_engine is job and entry.CORE_EVALUATE.__globals__ is job.__dict__
    assert (
        original_job.protocol is original_protocol and original_audit.protocol is original_protocol
    )
    assert job.OUTPUT != original_job.OUTPUT
    assert protocol.render is original_protocol.render
    assert "torch" not in inspect.getsource(verify_entry.independent_condition)


@pytest.mark.parametrize("fault", [None, "whitespace", "double_scale"])
def test_runtime_reads_exact_frozen_condition_bytes(tmp_path, monkeypatch, fault):
    cfg = protocol.read(protocol.ROOT / protocol.CONFIG)
    condition = protocol.derive_condition(cfg)
    monkeypatch.setattr(protocol, "OUTPUT", str(tmp_path))
    saved = copy.deepcopy(condition)
    if fault == "double_scale":
        saved["vector"] = [saved["scale"] * x for x in saved["vector"]]
    raw = protocol.serialized(saved) + (b" " if fault == "whitespace" else b"")
    (tmp_path / protocol.CONDITION).write_bytes(raw)
    if fault:
        with pytest.raises(ValueError, match="derived condition bytes"):
            protocol.candidates(cfg)
    else:
        assert protocol.candidates(cfg) == {"preserve": condition}


def test_baseline_absolute_tolerance_and_descriptive_comparison(tmp_path):
    plan, backend, vectors = prepare()
    snap = plan["prior_comparison"]["baseline_records"][3]
    snap["row"]["preserve_log_odds"] += 0.5e-6
    snap["row_sha256"] = protocol.canonical_sha(snap["row"])
    ledger = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    rows = job.evaluate(plan, backend, vectors, ledger, tmp_path)
    verified = audit.verify_data(plan, rows, vectors, tmp_path)
    assert 0 < verified["baseline_comparison"]["maximum_S0_difference"] <= 1e-6
    for row in verified["prior_strength_comparison"]:
        assert all(
            value == row["endpoint020"][key] - row["prior"][key]
            for key, value in row["endpoint_minus_prior"].items()
        )


def test_independent_condition_gate_before_load(monkeypatch):
    plan = protocol.build_plan()
    monkeypatch.setattr(entry, "CORE_FREEZE_CHECK", lambda: {"plan": plan})
    monkeypatch.setattr(protocol, "candidates", lambda: {"preserve": plan["derived_condition"]})
    calls = []

    def reject(value):
        calls.append(value)
        raise ValueError("independent condition rejected before load")

    monkeypatch.setattr(verify_entry, "independent_condition", reject)
    with pytest.raises(ValueError, match="independent condition rejected"):
        job.require_freeze()
    assert calls == [plan]


def test_storage_bound_unchanged(monkeypatch):
    cfg = protocol.read(protocol.ROOT / protocol.CONFIG)
    monkeypatch.setattr(
        original_protocol.shutil, "disk_usage", lambda _: SimpleNamespace(free=64 * 1024**2)
    )
    assert protocol.storage_preflight(protocol.ROOT, cfg)["bounds"]["total_bound_bytes"] == 41283268
    monkeypatch.setattr(original_protocol.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="storage"):
        protocol.storage_preflight(protocol.ROOT, cfg)


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {"standard_used_percent": 90, "checked_at_unix": 0},
        {"standard_used_percent": 26, "checked_at_unix": 0},
    ],
)
def test_new_usage_gate(monkeypatch, usage):
    monkeypatch.setattr(job, "no_resume", lambda allowed: None)
    monkeypatch.setattr(protocol, "check_prelaunch", lambda *a, **k: {"status": "passed"})
    monkeypatch.setattr(job, "require_freeze", dict)
    monkeypatch.setattr(job, "lock_commit", lambda _: "f" * 40)
    monkeypatch.setenv("SP_LENSE_USAGE_PREFLIGHT", json.dumps(usage))
    with pytest.raises(ValueError, match="fresh usage"):
        job.run()


@pytest.mark.parametrize(
    "files",
    [
        ["preregistration.json"],
        ["derived_condition.json", "preregistration.json", "extra"],
        ["derived_condition.json", "preregistration.json"],
    ],
)
def test_exact_two_file_lock(monkeypatch, files):
    def git(root, *args):
        if args[0] == "show":
            return "\n".join(protocol.OUTPUT + "/" + f for f in files)
        if args[0] == "status":
            return ""
        return "source" if args[-1] == "HEAD^" else "f" * 40

    monkeypatch.setattr(job.base, "git", git)
    if len(files) == 2:
        assert job.lock_commit({"source_commit": "source"}) == "f" * 40
    else:
        with pytest.raises(ValueError, match="condition-and-preregistration-only"):
            job.lock_commit({"source_commit": "source"})


def test_one_prelaunch_no_resume_and_no_worker_on_git_failure(tmp_path, monkeypatch):
    job.base.write_new(
        tmp_path / "preregistration.json", {"source_commit": "f" * 40, "environment": {}}
    )
    job.base.write_new(tmp_path / protocol.CONDITION, {})
    monkeypatch.setattr(job, "OUTPUT", str(tmp_path))
    calls = []

    def git(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=128, stdout="", stderr="synthetic denied")

    monkeypatch.setattr(job.subprocess, "run", git)
    result = job.prelaunch()
    assert result["status"] == "INCONCLUSIVE" and len(calls) == 1
    with pytest.raises(ValueError, match="prelaunch failed"):
        job.run()
    with pytest.raises(ValueError, match="no resume"):
        job.prelaunch()
    assert not (tmp_path / "WORKER_CLAIM.json").exists()
    assert not (tmp_path / "RUN_STARTED.json").exists()
