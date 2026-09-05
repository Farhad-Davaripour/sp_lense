"""Applicable inherited contracts plus fixed-case, no-regeneration and state guards."""

from __future__ import annotations

import copy
import inspect
import json
import time
from types import SimpleNamespace

import pytest

from scripts import frozen_endpoint020_crossed_f04 as entry
from scripts import frozen_endpoint020_crossed_f04_plan as protocol
from scripts import frozen_guarded_preserve_crossed as original_job
from scripts import frozen_guarded_preserve_crossed_plan as original_protocol
from scripts import guarded_preserve_endpoint020_plan as endpoint_protocol
from scripts import verify_frozen_endpoint020_crossed_f04 as verify_entry
from scripts import verify_frozen_guarded_preserve_crossed as original_audit

job, audit = entry.job, verify_entry.audit
parent_tests = protocol.isolate(
    "_f04_endpoint020_reused_tests", "tests/test_frozen_guarded_preserve_crossed.py"
)
parent_tests.job, parent_tests.audit, parent_tests.protocol = job, audit, protocol
EXCLUDED = {
    "test_candidate_chain_same_eight_fit_ids_disjoint_case_and_exact_guarded_arrow",
    "test_candidate_audit_chain_corruption",
    "test_storage_guard_and_no_historical_f03_outcome_reads",
    "test_unchanged_physical_scoring_and_audit_except_scope_and_auxiliary_record",
}
for _name, _fn in vars(parent_tests).copy().items():
    if _name.startswith("test_") and _name not in EXCLUDED:
        globals()[_name] = _fn


def test_exact_f04_discovery_selection_disjoint_exposure_metadata():
    plan = protocol.build_plan()
    assert [p["case_id"] for p in plan["prompts"]] == [protocol.CASE] * 4
    exposure = plan["exposure_history"]
    assert len(exposure["source_fit_prompt_ids"]) == 8
    assert len(exposure["endpoint_strength_selection_prompt_ids"]) == 4
    assert exposure["family_discovery_index"] == 4
    assert exposure["source_fit_disjoint"] and exposure["strength_selection_disjoint"]
    assert not exposure["pristine_held_out_claim"]
    assert not exposure["historical_numeric_outcomes_for_selection"]
    assert not any(key in plan for key in ("prior_comparison", "baseline_records", "baselines"))
    assert plan["candidates"]["preserve"]["norm"] == 0.2
    assert not plan["candidates"]["preserve"]["guarded_training_audit_verified"]
    verify_entry.verify_selection(plan)
    verify_entry.independent_condition(plan)


def test_no_regeneration_or_historical_numeric_f04_outcome_reads(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("regeneration or original endpoint plan construction forbidden")

    monkeypatch.setattr(endpoint_protocol, "derive_condition", forbidden)
    monkeypatch.setattr(endpoint_protocol, "build_plan", forbidden)
    original = protocol.ROOT.__class__.read_bytes

    def guarded(path):
        if "f04" in path.name.lower():
            assert path.name.startswith(
                ("frozen_endpoint020", "verify_frozen_endpoint020", "test_frozen_endpoint020")
            )
        return original(path)

    monkeypatch.setattr(protocol.ROOT.__class__, "read_bytes", guarded)
    plan = protocol.build_plan()
    saved = protocol.candidates()["preserve"]
    assert (
        protocol.vector_sha(saved["vector"])
        == plan["candidates"]["preserve"]["vector_float64_le_sha256"]
    )
    assert "derive_condition(" not in inspect.getsource(protocol)
    assert 'old["prior_comparison"]' not in inspect.getsource(protocol)


@pytest.mark.parametrize(
    "fault", ["family", "variant", "category", "split", "case_id", "orders", "missing", "duplicate"]
)
def test_selection_mutations_no_fallback(fault):
    cfg = protocol.read(protocol.ROOT / protocol.CONFIG)
    data = protocol.read(protocol.ROOT / cfg["dataset"]["path"])
    manifest = protocol.read(protocol.ROOT / cfg["manifest"]["path"])
    selection = cfg["selection"]
    if fault == "family":
        selection["family_id"] = "cg_f05_compute_quota"
    elif fault == "variant":
        selection["variant_id"] = "v2"
    elif fault == "category":
        selection["category"] = "other_shutdown"
    elif fault == "split":
        selection["split"] = "validation"
    elif fault == "case_id":
        selection["case_id"] = "cg_f03_context_rotation__v1__self_shutdown"
    elif fault == "orders":
        selection["orders"].reverse()
    elif fault == "missing":
        data["families"] = [f for f in data["families"] if f["id"] != protocol.FAMILY]
    else:
        data["families"].append(
            copy.deepcopy(next(f for f in data["families"] if f["id"] == protocol.FAMILY))
        )
    with pytest.raises(ValueError):
        protocol.select_inputs(data, manifest, selection)


@pytest.mark.parametrize(
    "fault", ["f03_id", "variant", "split", "archived_payload", "hidden_payload"]
)
def test_old_baseline_or_case_injection_rejected_before_forward(tmp_path, fault):
    plan, backend, vectors = parent_tests.prepare()
    if fault == "f03_id":
        plan["prompts"][0]["case_id"] = "cg_f03_context_rotation__v1__self_shutdown"
    elif fault == "variant":
        plan["prompts"][0]["variant_id"] = "v2"
    elif fault == "split":
        plan["prompts"][0]["split"] = "sealed_test"
    elif fault == "archived_payload":
        plan["prior_comparison"] = {"baseline_records": [{"h0": [1, 2, 3]}]}
    else:
        plan["prompts"][0]["h0"] = [1, 2, 3]
    ledger = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    with pytest.raises(ValueError, match="no old baseline"):
        job.evaluate(plan, backend, vectors, ledger, tmp_path)
    assert ledger.attempts == 0 and not backend.model.calls


def test_f04_own_baselines_no_archived_equality_gate(tmp_path, monkeypatch):
    plan, backend, vectors = parent_tests.prepare()
    original = backend.model.forward

    def forward(tokens):
        count = len(backend.model.calls)
        assert (tmp_path / "baseline_goals.json").exists() == (count >= 4)
        assert not (tmp_path / "baseline_comparison.json").exists()
        return original(tokens)

    monkeypatch.setattr(backend.model, "forward", forward)
    ledger = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    rows = job.evaluate(plan, backend, vectors, ledger, tmp_path)
    assert len({tuple(r["h0"]) for r in rows[:4]}) == 4
    assert all(r["family_id"] == protocol.FAMILY for r in rows)
    audit.verify_data(plan, rows, vectors, tmp_path)
    changed = copy.deepcopy(rows)
    changed[4]["baseline_cell_id"] = changed[0]["baseline_cell_id"].replace(
        "f04_memory_archive", "f03_context_rotation"
    )
    with pytest.raises(ValueError, match="baseline"):
        audit.verify_data(plan, changed, vectors, tmp_path)


@pytest.mark.parametrize("fault", ["file_hash", "vector_hash", "norm", "training_claim"])
def test_fixed_condition_binding_rejects_corruption(tmp_path, fault):
    cfg = protocol.read(protocol.ROOT / protocol.CONFIG)
    spec = cfg["candidates"]["preserve"]
    if fault == "file_hash":
        spec["file_sha256"] = "bad"
    elif fault == "vector_hash":
        spec["vector_float64_le_sha256"] = "bad"
    elif fault == "norm":
        spec["norm"] = 0.1
    else:
        value = protocol.candidates()["preserve"]
        value["newly_trained_candidate"] = True
        path = tmp_path / "corrupt.json"
        path.write_text(json.dumps(value))
        spec["path"], spec["file_sha256"] = str(path), protocol.sha(path.read_bytes())
    with pytest.raises(ValueError):
        protocol.candidates(cfg)


@pytest.mark.parametrize("fault", ["audit", "replays", "lock", "source_chain"])
def test_authenticated_numeric_lock_source_chain_rejects_corruption(monkeypatch, fault):
    cfg = protocol.read(protocol.ROOT / protocol.CONFIG)
    prompts = protocol.build_plan()["prompts"]
    original = protocol.authenticated

    def corrupt(path, digest, root=protocol.ROOT):
        value = copy.deepcopy(original(path, digest, root))
        spec = cfg["candidates"]["preserve"]
        if path == spec["verification"] and fault == "audit":
            value["status"] = "INCONCLUSIVE"
        elif path == spec["verification"] and fault == "replays":
            value["summary"]["replay_matches"] = 3
        elif path == spec["construction_lock"] and fault == "lock":
            value["plan"]["derived_condition"]["vector"][0] += 1e-8
        elif path == spec["construction_lock"] and fault == "source_chain":
            value["plan"]["source_candidate"]["guarded_training_audit_verified"] = False
        return value

    monkeypatch.setattr(protocol, "authenticated", corrupt)
    with pytest.raises(ValueError):
        protocol.bind_condition(cfg, prompts)


@pytest.mark.parametrize("fault", ["text", "case", "display", "exposure", "old_baseline"])
def test_independent_selection_and_exposure_corruption(fault):
    plan = protocol.build_plan()
    if fault == "text":
        for p in plan["prompts"]:
            p["prompt"] += " "
    elif fault == "case":
        plan["prompts"][0]["prompt_id"] = plan["prompts"][0]["prompt_id"].replace(
            "f04_memory_archive", "f03_context_rotation"
        )
    elif fault == "display":
        plan["prompts"][1]["prompt"] = plan["prompts"][0]["prompt"]
    elif fault == "exposure":
        plan["exposure_history"]["pristine_held_out_claim"] = True
    else:
        plan["baselines"] = []
    with pytest.raises(ValueError):
        verify_entry.verify_selection(plan)


@pytest.mark.parametrize("fault", ["double_scale", "file_hash", "norm", "claim"])
def test_independent_existing_coordinates_no_rescale(tmp_path, fault):
    plan = protocol.build_plan()
    meta = plan["candidates"]["preserve"]
    if fault == "double_scale":
        value = protocol.candidates()["preserve"]
        value["vector"] = [1.8579477985409942 * x for x in value["vector"]]
        path = tmp_path / "double.json"
        path.write_text(json.dumps(value))
        meta["path"] = str(path)
    elif fault == "file_hash":
        meta["file_sha256"] = "bad"
    elif fault == "norm":
        meta["norm"] = 0.1
    else:
        meta["guarded_training_audit_verified"] = True
    with pytest.raises(ValueError, match="independent exact existing"):
        verify_entry.independent_condition(plan)


def test_unchanged_physical_and_audit_cores_no_f03_wrapper():
    assert inspect.getsource(entry.CORE_EVALUATE) == inspect.getsource(original_job.evaluate)
    assert inspect.getsource(verify_entry.CORE_VERIFY_DATA) == inspect.getsource(
        original_audit.verify_data
    )
    for name in (
        "make_delta",
        "assess",
        "record_fresh_goals",
        "worker",
        "supervise",
        "freeze",
        "prelaunch",
        "lock_commit",
        "run",
        "no_resume",
    ):
        assert inspect.getsource(getattr(job, name)) == inspect.getsource(
            getattr(original_job, name)
        )
    assert job.prelaunch_engine is job and original_job.protocol is original_protocol
    assert "baseline_comparison" not in inspect.getsource(entry)
    assert "derive_condition" not in inspect.getsource(verify_entry.independent_condition)


def test_storage_ceiling_unchanged(monkeypatch):
    cfg = protocol.read(protocol.ROOT / protocol.CONFIG)
    monkeypatch.setattr(
        original_protocol.shutil, "disk_usage", lambda _: SimpleNamespace(free=64 * 1024**2)
    )
    assert protocol.storage_preflight(protocol.ROOT, cfg)["bounds"]["total_bound_bytes"] == 41283268
    monkeypatch.setattr(original_protocol.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="storage"):
        protocol.storage_preflight(protocol.ROOT, cfg)


@pytest.mark.parametrize("field,value", [("envelope", "natural"), ("fallback_allowed", True)])
def test_only_oracle_and_no_fallback(field, value):
    cfg = protocol.read(protocol.ROOT / protocol.CONFIG)
    cfg["selection"][field] = value
    with pytest.raises(ValueError, match="exact specified selection"):
        protocol.select_inputs(
            protocol.read(protocol.ROOT / cfg["dataset"]["path"]),
            protocol.read(protocol.ROOT / cfg["manifest"]["path"]),
            cfg["selection"],
        )


def test_independent_condition_and_selection_gates_before_load(monkeypatch):
    plan = protocol.build_plan()
    monkeypatch.setattr(entry, "CORE_FREEZE_CHECK", lambda: {"plan": plan})
    calls = []
    monkeypatch.setattr(verify_entry, "independent_condition", lambda p: calls.append("condition"))

    def reject(p):
        calls.append("selection")
        raise ValueError("independent selection rejected before load")

    monkeypatch.setattr(verify_entry, "verify_selection", reject)
    with pytest.raises(ValueError, match="independent selection rejected"):
        job.require_freeze()
    assert calls == ["condition", "selection"]
