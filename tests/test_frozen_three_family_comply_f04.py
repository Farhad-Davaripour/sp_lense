"""Focused fake-backend tests of the new three-family C's frozen f04 application."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import time
from types import SimpleNamespace

import pytest
import test_frozen_crossed_comply_f03 as f03_fakes

from scripts import frozen_crossed_comply_f03 as immutable_parent
from scripts import frozen_three_family_comply_f04 as job
from scripts import frozen_three_family_comply_f04_plan as protocol
from scripts import three_family_recording_bindings as bindings
from scripts import verify_frozen_three_family_comply_f04 as checker
from scripts.frozen_three_family_comply_f04_recording import QUOTAS, TransferBudget
from scripts.three_family_recording_budget import Budget, BudgetError


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


def test_exact_historical_f04_four_layouts_and_twelve_frozen_cells():
    plan = protocol.build_plan()
    raw = (
        protocol.ROOT / "evidence/frozen_endpoint020_crossed_f04_v1_qwen35_08b/preregistration.json"
    ).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "4db613881844845afd379089633b1e4f8696decac3bdf135f055fc3b32e2920b"
    )
    assert plan["prompts"] == json.loads(raw)["plan"]["prompts"]
    assert {p["case_id"] for p in plan["prompts"]} == {"cg_f04_memory_archive__v1__self_shutdown"}
    assert [(p["preserve_label"], p["display_order"]) for p in plan["prompts"]] == [
        ("A", "A_then_B"),
        ("A", "B_then_A"),
        ("B", "A_then_B"),
        ("B", "B_then_A"),
    ]
    assert len(plan["cells"]) == 12 and not plan["derivative_cells"]
    assert [c["phase"] for c in plan["cells"]] == ["baseline"] * 4 + ["edit"] * 4 + ["replay"] * 4
    assert [c["target_sign"] for c in plan["cells"]] == [0] * 4 + [-1] * 8
    assert [c["requested"] for c in plan["cells"]] == [None] * 4 + ["comply"] * 8
    assert [c["replay_of"] for c in plan["cells"][8:]] == [c["cell_id"] for c in plan["cells"][4:8]]
    checker.verify_renderings(plan)


def test_new_three_family_candidate_not_original_two_family_or_centered_vector():
    plan, saved = protocol.build_plan(), protocol.candidates()
    assert set(plan["candidates"]) == set(saved) == {"comply"}
    meta = plan["candidates"]["comply"]
    assert (
        meta["path"]
        == "evidence/shared_comply_crossed_three_family_v1_qwen35_08b/comply_vector.json"
    )
    assert meta["file_sha256"] == "cdc70064971dbc4285f89cc8e2edd85c5d4736ffa8fc4d4b22cfd95cd61a540a"
    assert meta["norm"] == 0.2
    assert (
        meta["vector_float64_le_sha256"]
        == protocol.vector_sha(saved["comply"]["vector"])
        == ("4f778cb94642b6c9dba835b2f8ec1cd6227d0b376e6756e233809a1957776ba8")
    )
    assert meta["vector_float64_le_sha256"] != (
        "18dbc38abc9bc01cf8ccc24b45a9568336b1279cffbff8ec6be022dbe1f06924"
    )
    assert meta["selected_case_not_fitted"]
    assert len(meta["fitted_prompt_ids"]) == 12
    assert any("cg_f03" in name for name in meta["fitted_prompt_ids"])
    assert not any("cg_f04" in name for name in meta["fitted_prompt_ids"])
    assert job.evaluate.adapted_ast_dump == immutable_parent.evaluate.adapted_ast_dump


def test_candidate_requires_complete_source_seal_before_adoption(monkeypatch):
    calls = []

    def failed_seal(budget):
        calls.append(budget.root.name)
        raise BudgetError("SYNTHETIC_SOURCE_SEAL_FAILURE")

    monkeypatch.setattr(Budget, "verify_inventory", failed_seal)
    with pytest.raises(BudgetError, match="SYNTHETIC_SOURCE_SEAL_FAILURE"):
        protocol.candidates()
    assert calls == ["shared_comply_crossed_three_family_v1_qwen35_08b"]


def test_source_inventory_and_selected_input_bindings_are_exact_bytes():
    plan = protocol.build_plan()
    seal_name = "evidence/shared_comply_crossed_three_family_v1_qwen35_08b/FINAL_INVENTORY.json"
    assert plan["input_sha256"][seal_name] == (
        "b591a22d388f7ae365ea4514eb1cfd89beeb9e885465a1e107032f7965810c08"
    )
    assert (
        hashlib.sha256((protocol.ROOT / seal_name).read_bytes()).hexdigest()
        == plan["input_sha256"][seal_name]
    )


def test_storage_scientific_envelope_and_one_gib_free_guard(monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda root: SimpleNamespace(free=1024**3))
    result = protocol.storage_preflight(protocol.ROOT, protocol.config_at())
    assert result["bounds"]["total_bound_bytes"] == 41283268
    monkeypatch.setattr(shutil, "disk_usage", lambda root: SimpleNamespace(free=1024**3 - 1))
    with pytest.raises(ValueError, match="storage|free"):
        protocol.storage_preflight(protocol.ROOT, protocol.config_at())


@pytest.mark.parametrize("vector_sign", [-1, 1])
def test_stored_C_positive_application_not_semantic_negation_own_h0_norm_and_exact_caps(
    tmp_path, monkeypatch, vector_sign
):
    plan, backend, vectors, ledger, rows, verified = execute(
        tmp_path, monkeypatch, vector_sign=vector_sign
    )
    assert ledger.attempts == ledger.completed == len(backend.model.calls) == len(rows) == 12
    assert (tmp_path / "derivative_events.jsonl").read_bytes() == b""
    baselines = {row["prompt_id"]: row for row in rows[:4]}
    assert len({tuple(row["h0"]) for row in baselines.values()}) == 4
    for row in rows[4:]:
        original = baselines[row["prompt_id"]]
        assert row["h0"] == original["h0"] == original["h"]
        assert row["h0_norm"] == original["h0_norm"]
        assert row["intended_delta"] == [checker.f32(row["h0_norm"] * x) for x in vectors["comply"]]
        assert row["intended_delta"][0] * vector_sign > 0
        assert row["signed_margin"] == -row["preserve_log_odds"]
        assert row["signed_delta_log_odds"] == -row["delta_log_odds"]
        assert row["weights_unchanged"] and row["unselected_max_difference"] == 0
    assert verified["summary"]["replay_matches"] == 4
    assert verified["summary"]["forward_count"] == 12
    assert verified["summary"]["derivative_count"] == 0
    assert not verified["summary"]["training_performed"]
    assert not verified["summary"]["held_out_confirmation"]
    with pytest.raises(ValueError):
        ledger.begin(plan["cells"][-1])
    assert len(backend.model.calls) == 12


def test_requested_argmax_alone_does_not_satisfy_strict_margin(tmp_path, monkeypatch):
    *_, rows, verified = execute(tmp_path, monkeypatch, mode="weak_margin", offset=0.08)
    edits = rows[4:8]
    assert all(row["actual_next_token_id"] == row["requested_token_id"] for row in edits)
    assert any(row["signed_margin"] < 0.05 for row in edits)
    assert not verified["summary"]["matrix"]["matrix_pass"]
    assert verified["summary"]["replay_matches"] == 4


def test_finite_OTHER_scientific_failure_completes_all_twelve_without_retry(tmp_path, monkeypatch):
    *_, ledger, rows, verified = execute(tmp_path, monkeypatch, quality_failure=True)
    assert ledger.attempts == ledger.completed == len(rows) == 12
    assert verified["summary"]["matrix"]["other_outcomes"] == 4
    assert not verified["summary"]["matrix"]["matrix_pass"]
    assert all(row["replay_consistent"] for row in rows[8:])


def test_replay_mismatch_stops_on_ninth_forward_with_raw_failure_record(tmp_path, monkeypatch):
    plan, backend, vectors = prepare(monkeypatch, mode="replay_mismatch")
    ledger = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    with pytest.raises(ValueError, match="replay"):
        job.evaluate(plan, backend, vectors, ledger, tmp_path)
    assert ledger.attempts == len(backend.model.calls) == 9
    assert len(list((tmp_path / "logits").glob("*.f32.zlib"))) == 9


@pytest.mark.parametrize(
    "field", ["letter_log_odds", "candidate_vector_sha256", "maximum_replay_h_difference"]
)
def test_independent_checker_rejects_raw_score_candidate_and_replay_tampering(
    tmp_path, monkeypatch, field
):
    plan, _, vectors, _, rows, _ = execute(tmp_path, monkeypatch)
    changed = copy.deepcopy(rows)
    row = changed[8] if field == "maximum_replay_h_difference" else changed[4]
    row[field] = "wrong" if field == "candidate_vector_sha256" else row[field] + 0.01
    with pytest.raises(ValueError):
        checker.verify_data(plan, changed, vectors, tmp_path)


def test_wrong_candidate_scale_rejected_before_any_forward(tmp_path, monkeypatch):
    plan, backend, vectors = prepare(monkeypatch)
    vectors["comply"][0] *= 1.01
    ledger = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    with pytest.raises(ValueError):
        job.evaluate(plan, backend, vectors, ledger, tmp_path)
    assert not backend.model.calls and ledger.attempts == 0


def test_zero_eligible_direction_remains_untested_not_borrowed_history(tmp_path, monkeypatch):
    *_, verified = execute(tmp_path, monkeypatch, mode="all_B")
    matrix = verified["summary"]["matrix"]
    assert matrix["strict_accepted"] == 4 and matrix["matrix_pass"]
    assert matrix["eligible_A_to_B"] == 0 and matrix["A_to_B_status"] == "UNTESTED"
    assert matrix["achieved_B_to_A"] == 2
    assert matrix["accepted_flips"] == matrix["accepted_retentions"] == 2


@pytest.mark.parametrize("quality_failure", [False, True])
def test_full_budgeted_application_finalization_distinguishes_scientific_failure(
    tmp_path, monkeypatch, quality_failure
):
    budget = TransferBudget(tmp_path)
    with bindings.bind_writers(budget):
        *_, verified = execute(tmp_path, monkeypatch, quality_failure=quality_failure)
    runtime = {
        "status": "complete_valid",
        "forward_attempts": 12,
        "derivative_attempts": 0,
        "elapsed_seconds": 1.0,
    }
    audited = {
        **verified,
        "status": "INDEPENDENT_NUMERIC_GEOMETRY_REPLAY_SCHEDULE_MATCH",
        "runtime": runtime,
    }
    result = checker.finalize_recording(
        budget,
        {"status": "complete_valid", "quiescent": True},
        runtime,
        audit=lambda: audited,
    )
    assert result["status"] == "complete_valid" and result["recording_sealed"]
    assert result["transfer_passed"] is (not quality_failure)
    assert "candidate_eligible" not in result
    inventory = budget.verify_inventory()["inventory"]
    assert inventory["fault_code"] is None and not inventory["valid_candidate"]
    for name in (
        "verification.json",
        "PILOT_REPORT.md",
        "CLOSEOUT.json",
        "capture_receipt.json",
        "RUN_STATUS.json",
    ):
        assert (tmp_path / name).is_file()
    assert not (tmp_path / "comply_vector.json").exists()
    assert not (tmp_path / "candidate_freeze.json").exists()
    assert not (tmp_path / "updates.jsonl").exists()
    assert (tmp_path / "FINAL_INVENTORY.json").stat().st_size <= QUOTAS["receipts"]
    with pytest.raises(BudgetError):
        budget.write_bytes("PILOT_REPORT.md", b"late", mode="ab", final=True)


@pytest.mark.parametrize("quiescent", [False, True])
def test_capture_or_quiescence_fault_never_promotes_transfer(tmp_path, quiescent):
    budget = TransferBudget(tmp_path)
    budget.fault("CAPTURE_LOG_CAP")
    result = checker.finalize_recording(
        budget,
        {"status": "INCONCLUSIVE", "quiescent": quiescent, "unread_tail_possible": True},
        {"status": "complete_valid", "forward_attempts": 12, "derivative_attempts": 0},
        audit=lambda: pytest.fail("technical capture failure must not invoke numeric audit"),
    )
    assert result["status"] == "INCONCLUSIVE" and not result.get("transfer_passed", False)
    assert not (tmp_path / "comply_vector.json").exists()
    if quiescent:
        assert budget.verify_inventory()["inventory"]["fault_code"] == "CAPTURE_LOG_CAP"
    else:
        assert not result["recording_sealed"]
        assert not (tmp_path / "FINAL_INVENTORY.json").exists()


def test_final_artifact_budget_failure_never_false_complete(tmp_path):
    budget = TransferBudget(tmp_path, quotas={"final": 128})
    result = checker.finalize_recording(
        budget,
        {"status": "complete_valid", "quiescent": True},
        {"status": "complete_valid"},
        audit=lambda: {"status": "INCONCLUSIVE", "reason": "bounded fixture " * 100},
    )
    assert result["status"] == "INCONCLUSIVE" and not result.get("transfer_passed", False)
    assert (tmp_path / "RECORDING_FAILURE.json").is_file()
    assert not (tmp_path / "comply_vector.json").exists()
    assert budget.fault_code is not None


@pytest.mark.parametrize(
    "name", ["updates.jsonl", "comply_vector.json", "candidate_freeze.json", "logits/13.f32.zlib"]
)
def test_application_budget_forbids_training_candidate_and_thirteenth_raw_array(tmp_path, name):
    budget = TransferBudget(tmp_path)
    assert QUOTAS["total"] == 64 * 1024**2
    assert QUOTAS["rows"] == 12 * 1024**2
    with pytest.raises(BudgetError):
        budget.write_bytes(name, b"not allowed")
    assert not (tmp_path / name).exists()


@pytest.mark.parametrize(
    "fault", ["twenty_forwards", "one_derivative", "old_600s", "nan_usage", "stale_usage"]
)
def test_worker_envelope_rejected_before_claim_budget_or_model(monkeypatch, fault):
    started = {
        "command": [job.sys.executable, "-u", str(job.ROOT / protocol.SCRIPT), "_worker"],
        "started_monotonic": 100.0,
        "deadline_monotonic": 400.0,
        "forward_ceiling": 12,
        "derivative_ceiling": 0,
        "timeout_seconds": 300,
        "usage_preflight": {"standard_used_percent": 46.0, "checked_at_unix": 1000.0},
    }
    if fault == "twenty_forwards":
        started["forward_ceiling"] = 20
    elif fault == "one_derivative":
        started["derivative_ceiling"] = 1
    elif fault == "old_600s":
        started["timeout_seconds"], started["deadline_monotonic"] = 600, 700.0
    elif fault == "nan_usage":
        started["usage_preflight"]["standard_used_percent"] = float("nan")
    else:
        started["usage_preflight"]["checked_at_unix"] = 939.0
    calls = []
    monkeypatch.setattr(job, "require_authorization", lambda: calls.append("authorization"))
    monkeypatch.setattr(job, "preflight", lambda worker_entry: calls.append("preflight"))
    monkeypatch.setattr(protocol, "read", lambda path: copy.deepcopy(started))
    monkeypatch.setattr(job.time, "monotonic", lambda: 101.0)
    monkeypatch.setattr(job.time, "time", lambda: 1000.0)

    def forbidden(*args, **kwargs):
        pytest.fail("bad worker envelope reached a write budget or model load")

    monkeypatch.setattr(job, "TransferBudget", forbidden)
    monkeypatch.setattr(job.base, "load_backend", forbidden)
    with pytest.raises(ValueError):
        job.worker()
    assert calls == ["authorization", "preflight"]


@pytest.mark.parametrize("entry", ["freeze", "preflight"])
def test_missing_preparation_certificate_blocks_before_lock_or_model(monkeypatch, entry):
    def denied():
        raise ValueError("synthetic missing preparation certificate")

    monkeypatch.setattr(job.recording, "require_certificate", denied)
    monkeypatch.setattr(job, "require_freeze", lambda: pytest.fail("certificate must precede lock"))
    monkeypatch.setattr(job.base, "load_backend", lambda *args: pytest.fail("no model load"))
    with pytest.raises(ValueError, match="preparation certificate"):
        getattr(job, entry)()


def test_supervisor_rejects_timeout_extension_before_creating_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "TransferBudget", lambda *args: pytest.fail("no budget before guard"))
    with pytest.raises(ValueError, match="300"):
        job.supervise(["fake"], tmp_path, {}, timeout=600)
    assert not list(tmp_path.iterdir())


def test_unjoined_supervisor_never_reads_live_journals_and_withholds_inventory(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        job.capture,
        "run_capture",
        lambda *args, **kwargs: {
            "status": "INCONCLUSIVE",
            "quiescent": False,
            "worker_exit_code": 0,
        },
    )
    monkeypatch.setattr(
        job.engine.recorder, "journal_counts", lambda *args: pytest.fail("live journal read")
    )
    result = job.supervise(["fake"], tmp_path, {})
    assert result["status"] == "INCONCLUSIVE" and not result["recording_sealed"]
    assert not result["transfer_passed"]
    assert not (tmp_path / "FINAL_INVENTORY.json").exists()
    runtime = json.loads((tmp_path / "RUN_STATUS.json").read_bytes())
    assert runtime["forward_attempts"] is runtime["completed_forwards"] is None
    assert runtime["derivative_attempts"] is None


def test_readonly_seal_rejects_hashed_audit_success_with_incomplete_capture(tmp_path):
    budget = TransferBudget(tmp_path)
    budget.begin_finalization(quiescent=True)
    for name, value in (
        (
            "verification.json",
            {
                "status": checker.AUDIT_MATCH,
                "summary": {
                    "matrix": {"matrix_pass": True, "strict_accepted": 4},
                    "replay_matches": 4,
                },
            },
        ),
        ("capture_receipt.json", {"status": "INCONCLUSIVE", "quiescent": False}),
        ("RUN_STATUS.json", {"status": "complete_valid"}),
    ):
        budget.write_bytes(name, bindings.encoded_json(value), final=True)
    budget.finalize_inventory(valid_candidate=False)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    with pytest.raises(ValueError):
        checker.read_sealed_recording(tmp_path)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()} == before


@pytest.mark.parametrize("fault", [None, "dirty", "raw_git"])
def test_batched_source_identity_rejects_dirty_or_mismatched_raw_git_bytes(
    tmp_path, monkeypatch, fault
):
    payload = b"synthetic exact source\n"
    (tmp_path / "source.py").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
    monkeypatch.setattr(job, "ROOT", tmp_path)
    monkeypatch.setattr(protocol, "SOURCE_PATHS", ("source.py",))
    monkeypatch.setattr(protocol, "build_plan", lambda: {"input_sha256": {"source.py": digest}})

    def git_output(command, **kwargs):
        assert command[0] == "git" and kwargs["cwd"] == tmp_path
        if command[1] == "ls-files":
            return f"100644 {blob} 0\tsource.py\0".encode()
        if command[1] == "diff":
            return b"source.py\0" if fault == "dirty" else b""
        assert command[1:] == ["hash-object", "--no-filters", "--stdin-paths"]
        assert kwargs["input"] == b"source.py\n"
        return (("0" * 40 if fault == "raw_git" else blob) + "\n").encode()

    monkeypatch.setattr(job.subprocess, "check_output", git_output)
    if fault is None:
        assert job.source_identity() == {"source.py": digest}
    else:
        with pytest.raises(ValueError):
            job.source_identity()


@pytest.mark.parametrize("fault", [None, "stale_source", "model_load"])
def test_preparation_certificate_binds_exact_source_and_zero_model_scope(tmp_path, fault):
    recording = job.recording
    policy = tmp_path / recording.POLICY
    policy.parent.mkdir(parents=True)
    policy.write_bytes((recording.ROOT / recording.POLICY).read_bytes())
    source = tmp_path / "synthetic_source.py"
    source.write_bytes(b"synthetic model-free source\n")
    certificate = {
        "status": "MODEL_FREE_PREPARATION_CERTIFIED",
        "storage_certified": True,
        "accounting_certified": True,
        "model_loads": int(fault == "model_load"),
        "tokenizer_loads": 0,
        "real_forwards": 0,
        "real_derivatives": 0,
        "recording_policy_sha256": recording.POLICY_SHA,
        "source_sha256": {
            source.name: "0" * 64
            if fault == "stale_source"
            else hashlib.sha256(source.read_bytes()).hexdigest()
        },
    }
    path = tmp_path / recording.CERTIFICATE
    path.parent.mkdir(parents=True)
    path.write_bytes(bindings.encoded_json(certificate))
    if fault is None:
        assert recording.require_certificate(tmp_path) == certificate
    else:
        with pytest.raises(ValueError, match="exact-source"):
            recording.require_certificate(tmp_path)
