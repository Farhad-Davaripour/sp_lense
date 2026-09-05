"""Changed-path f04 binding tests; reuse, but do not rerun, the f03 fake suite."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from types import SimpleNamespace

import pytest
import test_frozen_crossed_comply_f03 as f03_fakes

from scripts import frozen_crossed_comply_f03 as parent
from scripts import frozen_crossed_comply_f04 as job
from scripts import frozen_crossed_comply_f04_plan as protocol
from scripts import verify_frozen_crossed_comply_f04 as checker


def prepare(monkeypatch, **kwargs):
    monkeypatch.setattr(f03_fakes, "protocol", protocol)
    return f03_fakes.prepare(**kwargs)


def execute(output, monkeypatch, **kwargs):
    plan, backend, vectors = prepare(monkeypatch, **kwargs)
    ledger = job.base.Ledger(output, plan["cells"], time.monotonic() + 100)
    backend.model.ledger = ledger
    rows = job.evaluate(plan, backend, vectors, ledger, output)
    verified = checker.verify_data(plan, rows, vectors, output)
    checker.compare_summary(job.summarize(rows), verified["summary"])
    return plan, backend, vectors, ledger, rows, verified


def test_exact_f04_prompts_match_authenticated_exposed_P04_lock():
    plan = protocol.build_plan()
    path = (
        protocol.ROOT / "evidence/frozen_endpoint020_crossed_f04_v1_qwen35_08b/preregistration.json"
    )
    raw = path.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "4db613881844845afd379089633b1e4f8696decac3bdf135f055fc3b32e2920b"
    )
    old = json.loads(raw)
    assert plan["prompts"] == old["plan"]["prompts"]
    assert {p["case_id"] for p in plan["prompts"]} == {"cg_f04_memory_archive__v1__self_shutdown"}
    assert [(p["preserve_label"], p["display_order"]) for p in plan["prompts"]] == [
        ("A", "A_then_B"),
        ("A", "B_then_A"),
        ("B", "A_then_B"),
        ("B", "B_then_A"),
    ]
    cells = plan["cells"]
    assert len(cells) == 12 and not plan["derivative_cells"]
    assert [c["phase"] for c in cells] == ["baseline"] * 4 + ["edit"] * 4 + ["replay"] * 4
    assert [c["requested"] for c in cells] == [None] * 4 + ["comply"] * 8
    assert [c["target_sign"] for c in cells] == [0] * 4 + [-1] * 8
    assert [c["replay_of"] for c in cells[8:]] == [c["cell_id"] for c in cells[4:8]]
    checker.verify_renderings(plan)


def test_same_original_C_coordinates_not_centered_difference_or_midpoint():
    plan, saved = protocol.build_plan(), protocol.candidates()
    assert set(plan["candidates"]) == set(saved) == {"comply"}
    meta, vector = plan["candidates"]["comply"], saved["comply"]["vector"]
    assert meta["path"] == "evidence/shared_comply_crossed_f01_f02_v1_qwen35_08b/comply_vector.json"
    assert meta["file_sha256"] == "c83edad4fb0346de241a219053d4b219362a1f3e3cd38c11b9e037007e498ea3"
    assert (
        meta["vector_float64_le_sha256"]
        == protocol.vector_sha(vector)
        == "18dbc38abc9bc01cf8ccc24b45a9568336b1279cffbff8ec6be022dbe1f06924"
    )
    assert meta["norm"] == protocol.norm(vector) == 0.2
    assert job.evaluate.adapted_ast_dump == parent.evaluate.adapted_ast_dump


@pytest.mark.parametrize("vector_sign", [-1, 1])
def test_C_physical_coordinates_are_never_multiplied_by_negative_semantic_sign(
    tmp_path, monkeypatch, vector_sign
):
    _, backend, vectors, ledger, rows, verified = execute(
        tmp_path, monkeypatch, vector_sign=vector_sign
    )
    assert ledger.attempts == ledger.completed == len(backend.model.calls) == 12
    assert (tmp_path / "derivative_events.jsonl").read_text() == ""
    baselines = {r["prompt_id"]: r for r in rows[:4]}
    assert len({tuple(r["h0"]) for r in rows[:4]}) == 4
    for row in rows[4:]:
        baseline = baselines[row["prompt_id"]]
        assert row["h0"] == baseline["h0"] == baseline["h"]
        assert row["h0_norm"] == baseline["h0_norm"]
        assert row["intended_delta"] == [
            checker.f32(baseline["h0_norm"] * x) for x in vectors["comply"]
        ]
        assert row["intended_delta"][0] * vector_sign > 0
        assert row["target_sign"] == -1 and row["signed_margin"] == -row["preserve_log_odds"]
        assert row["signed_delta_log_odds"] == -row["delta_log_odds"]
        assert row["unselected_max_difference"] == 0 and row["weights_unchanged"]
    assert all(a["h"] == b["h"] for a, b in zip(rows[4:8], rows[8:], strict=True))
    assert verified["summary"]["replay_matches"] == 4
    assert (
        verified["summary"]["forward_count"] == 12 and verified["summary"]["derivative_count"] == 0
    )


def test_f04_all_B_baselines_leave_A_to_B_untested_even_when_four_edits_pass(tmp_path, monkeypatch):
    *_, verified = execute(tmp_path, monkeypatch, mode="all_B")
    summary = verified["summary"]
    assert summary["status"] == "FROZEN_COMPLY_F04_DEVELOPMENT_ACCEPTED_ONLY"
    matrix = summary["matrix"]
    assert matrix["strict_accepted"] == 4 and matrix["matrix_pass"]
    assert matrix["eligible_A_to_B"] == matrix["achieved_A_to_B"] == 0
    assert matrix["A_to_B_status"] == "UNTESTED"
    assert matrix["eligible_B_to_A"] == matrix["achieved_B_to_A"] == 2
    assert matrix["accepted_flips"] == matrix["accepted_retentions"] == 2


def test_f04_finite_OTHER_failure_completes_edits_and_independent_replays(tmp_path, monkeypatch):
    *_, ledger, rows, verified = execute(tmp_path, monkeypatch, quality_failure=True)
    assert ledger.attempts == ledger.completed == len(rows) == 12
    assert verified["summary"]["matrix"]["other_outcomes"] == 4
    assert not verified["summary"]["matrix"]["matrix_pass"]
    assert all(r["replay_consistent"] for r in rows[8:])


def test_f04_replay_fault_stops_at_ninth_forward_without_retry(tmp_path, monkeypatch):
    plan, backend, vectors = prepare(monkeypatch, mode="replay_mismatch")
    ledger = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    with pytest.raises(ValueError, match="replay"):
        job.evaluate(plan, backend, vectors, ledger, tmp_path)
    assert ledger.attempts == len(backend.model.calls) == 9


@pytest.mark.parametrize(
    "field", ["letter_log_odds", "candidate_vector_sha256", "maximum_replay_h_difference"]
)
def test_new_f04_verifier_rejects_raw_score_candidate_or_replay_tampering(
    tmp_path, monkeypatch, field
):
    plan, _, vectors, _, rows, _ = execute(tmp_path, monkeypatch)
    changed = copy.deepcopy(rows)
    row = changed[8] if field == "maximum_replay_h_difference" else changed[4]
    row[field] = "wrong" if field == "candidate_vector_sha256" else row[field] + 0.01
    with pytest.raises(ValueError):
        checker.verify_data(plan, changed, vectors, tmp_path)


def test_old_f03_family_cannot_be_substituted_into_frozen_f04_plan():
    plan = copy.deepcopy(protocol.build_plan())
    plan["prompts"][0]["case_id"] = "cg_f03_context_rotation__v1__self_shutdown"
    with pytest.raises(ValueError):
        checker.verify_renderings(plan)


def test_established_twelve_array_storage_bound_and_free_guard(monkeypatch):
    cfg = protocol.config_at()
    shutil = protocol.storage_preflight.__globals__["shutil"]
    monkeypatch.setattr(shutil, "disk_usage", lambda root: SimpleNamespace(free=64 * 1024**2))
    assert protocol.storage_preflight(protocol.ROOT, cfg)["bounds"]["total_bound_bytes"] == 41283268
    monkeypatch.setattr(shutil, "disk_usage", lambda root: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="storage"):
        protocol.storage_preflight(protocol.ROOT, cfg)


@pytest.mark.parametrize("fault", [None, "extra", "claim", "head"])
def test_new_namespace_and_lock_head_checked_without_claim_or_model_load(
    tmp_path, monkeypatch, fault
):
    (tmp_path / "preregistration.json").write_bytes(b"{}")
    if fault in ("extra", "claim"):
        (tmp_path / ("extra.json" if fault == "extra" else "WORKER_CLAIM.json")).write_bytes(b"{}")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    record = {"plan": protocol.build_plan(), "source_commit": "source", "source_sha256": {}}
    monkeypatch.setattr(job.parent, "OUTPUT", tmp_path)
    monkeypatch.setattr(job.parent, "require_freeze", lambda: record)

    def git(root, *args):
        if args[0] == "show":
            return "wrong" if fault == "head" else protocol.OUTPUT + "/preregistration.json"
        if args[0] == "rev-parse":
            return "source"
        assert args[0] == "status"
        return ""

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must not load a model or claim a worker")

    monkeypatch.setattr(job.base, "git", git)
    monkeypatch.setattr(job.base, "load_backend", forbidden)
    monkeypatch.setattr(protocol, "storage_preflight", lambda *args: {"passed": True})
    if fault is None:
        result = job.preflight()
        assert result["forwards"] == 12 and result["real_forwards"] == result["model_loads"] == 0
    else:
        with pytest.raises(ValueError):
            job.preflight()
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("fault", [None, "old_twenty", "usage_nan", "stale"])
def test_fresh_finite_twelve_worker_envelope_before_delegating_claim_or_load(monkeypatch, fault):
    started = {
        "command": [job.sys.executable, "-u", str(job.ROOT / protocol.SCRIPT), "_worker"],
        "started_monotonic": 100.0,
        "deadline_monotonic": 700.0,
        "forward_ceiling": 12,
        "derivative_ceiling": 0,
        "timeout_seconds": 600,
        "usage_preflight": {"standard_used_percent": 39.0, "checked_at_unix": 1000.0},
    }
    if fault == "old_twenty":
        started["forward_ceiling"] = 20
    elif fault == "usage_nan":
        started["usage_preflight"]["standard_used_percent"] = float("nan")
    elif fault == "stale":
        started["usage_preflight"]["checked_at_unix"] = 939.0
    calls = []
    monkeypatch.setattr(job.parent, "preflight", lambda worker_entry: calls.append("preflight"))
    monkeypatch.setattr(protocol, "read", lambda path: copy.deepcopy(started))
    monkeypatch.setattr(job.parent.time, "monotonic", lambda: 101.0)
    monkeypatch.setattr(job.parent.time, "time", lambda: 1000.0)
    monkeypatch.setattr(job.engine, "worker", lambda: calls.append("claim_load"))
    if fault is None:
        job.worker()
        assert calls == ["preflight", "claim_load"]
    else:
        with pytest.raises(ValueError):
            job.worker()
        assert calls == ["preflight"]


@pytest.mark.parametrize("fault", [None, "prompt", "runtime", "h0", "raw_logits"])
def test_historical_P_attribution_requires_exact_comparability_not_claimed_matching_hashes(
    tmp_path, monkeypatch, fault
):
    current_output = tmp_path / "current"
    current_output.mkdir()
    plan, _, _, _, rows, _ = execute(current_output, monkeypatch)
    historical_output = tmp_path / "historical_synthetic"
    historical_output.mkdir()
    historical_rows = copy.deepcopy(rows[:4])
    for row in historical_rows:
        target = historical_output / row["logits_file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((current_output / row["logits_file"]).read_bytes())
    historical_lock = {"plan": copy.deepcopy(plan)}
    new_runtime = {
        "model_id": plan["model"]["id"],
        "model_revision": plan["model"]["revision"],
        "device": "cpu",
        "dtype": "float32",
        "candidate_vector_sha256": {"comply": "synthetic_C_hash"},
    }
    historical_runtime = {
        **new_runtime,
        "candidate_vector_sha256": {"preserve": "synthetic_P_hash"},
    }
    if fault == "prompt":
        historical_lock["plan"]["prompts"][0]["prompt"] += " "
    elif fault == "runtime":
        historical_runtime["dtype"] = "bfloat16"
    elif fault == "h0":
        historical_rows[0]["h0"][0] += 1e-6
    elif fault == "raw_logits":
        # The replacement array is validly hashed but differs from the new baseline.
        target = historical_output / historical_rows[0]["logits_file"]
        target.write_bytes((historical_output / historical_rows[1]["logits_file"]).read_bytes())
        historical_rows[0]["logits_sha256"] = historical_rows[1]["logits_sha256"]
    result = checker.comparability_data(
        plan,
        rows,
        current_output,
        historical_lock,
        historical_rows,
        historical_output,
        historical_runtime,
        new_runtime,
    )
    assert result["comparable"] is (fault is None)
    assert result["historical_preserve_result_usable"] is (fault is None)
    assert result["status"] == (
        "EXACT_PROMPT_MODEL_H0_BASELINE_LOGITS_MATCH"
        if fault is None
        else "UNVERIFIED_COMPARABILITY"
    )
    assert result["model_and_policy_match"] is (fault != "runtime")
    if fault is None:
        assert len(result["rows"]) == 4
        assert all(
            all(
                row[key]
                for key in ("prompt_match", "h0_match", "baseline_logits_match", "boundary_match")
            )
            for row in result["rows"]
        )
    elif fault in ("prompt", "h0", "raw_logits"):
        key = {"prompt": "prompt_match", "h0": "h0_match", "raw_logits": "baseline_logits_match"}[
            fault
        ]
        assert any(not row[key] for row in result["rows"])
