"""Focused integration tests: fake tensors only, no model loading or solver benchmark."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import time

import pytest
import test_shared_comply_crossed_three_family as fake
import torch

from scripts import shared_comply_crossed_three_family as job
from scripts import three_family_recording_bindings as bindings
from scripts import verify_shared_comply_crossed_three_family as checker
from scripts.three_family_recording_budget import Budget, BudgetError


@pytest.fixture(scope="module")
def recorded_fake(tmp_path_factory):
    output = tmp_path_factory.mktemp("recorded_three_family_fake")
    plan, backend = fake.prepare()
    budget = Budget(output)
    with bindings.bind_writers(budget):
        forwards = job.ForwardLedger(output, plan["cells"], time.monotonic() + 120)
        derivatives = job.Derivatives(
            torch, output, plan["derivative_cells"], time.monotonic() + 120
        )
        rows, summary = job.evaluate(plan, backend, forwards, derivatives, output)
    verified = checker.verify_data(plan, rows, output)
    verified["audited_endpoint_sha256"] = hashlib.sha256(
        (output / "endpoint.json").read_bytes()
    ).hexdigest()
    verified["audited_result_sha256"] = hashlib.sha256(
        (output / "result.json").read_bytes()
    ).hexdigest()
    checker.compare_summary(summary, verified["summary"])
    return output, budget, rows, verified, forwards, derivatives


def test_all_scientific_write_routes_are_budgeted_without_changing_fake_numbers(recorded_fake):
    output, budget, rows, verified, forwards, derivatives = recorded_fake
    assert forwards.attempts == 48 and derivatives.attempts == 12
    assert verified["summary"]["final_accepted"] == 12
    assert len(rows) == 48
    assert len(list((output / "logits").glob("*.f32.zlib"))) == 48
    assert budget.fault_code is None
    assert (output / "rows.jsonl").stat().st_size > 0
    assert (output / "updates.jsonl").stat().st_size > 0
    assert job.engine.base.write_new is bindings.runtime_io.write_new


def test_json_helper_serialization_and_outside_namespace_are_preserved(tmp_path):
    namespace = tmp_path / "attempt"
    namespace.mkdir()
    budget = Budget(namespace)
    value = {"z": [1.0, -0.2], "a": "UTF8: é", "reason": "accepted"}
    runtime = bindings.runtime_io.write_new
    audit = bindings.audit_io.write_new
    runtime(tmp_path / "outside_runtime.json", value)
    audit(tmp_path / "outside_audit.json", value)
    with bindings.bind_writers(budget):
        bindings.runtime_io.write_new(namespace / "runtime.json", value)
        bindings.audit_io.write_new(namespace / "analysis.json", value)
        bindings.runtime_io.write_new(tmp_path / "outside_again.json", value)
    assert (namespace / "runtime.json").read_bytes() == (
        tmp_path / "outside_runtime.json"
    ).read_bytes()
    assert (namespace / "analysis.json").read_bytes() == (
        tmp_path / "outside_audit.json"
    ).read_bytes()
    assert (tmp_path / "outside_again.json").read_bytes() == (
        tmp_path / "outside_runtime.json"
    ).read_bytes()
    assert bindings.runtime_io.write_new is runtime and bindings.audit_io.write_new is audit


def test_long_known_exception_text_is_explicitly_prefixed_with_known_length_and_digest(tmp_path):
    budget = Budget(tmp_path)
    message = "error: " + "終" * 10000
    with bindings.bind_writers(budget):
        bindings.runtime_io.write_new(
            tmp_path / "INVALID.json", {"status": "INCONCLUSIVE", "reason": message}
        )
    record = json.loads((tmp_path / "INVALID.json").read_bytes())
    spec = record["recording_diagnostic_prefixes"]["reason"]
    raw = message.encode()
    assert spec["truncated"] and spec["full_value_known"]
    assert spec["prefix_bytes"] <= 4096
    assert bytes.fromhex(spec["prefix_hex"]) == raw[:4096]
    assert spec["full_utf8_bytes"] == len(raw)
    assert spec["full_utf8_sha256"] == hashlib.sha256(raw).hexdigest()
    assert record["status"] == "INCONCLUSIVE"


def test_complete_finalization_includes_candidate_audit_report_and_own_size(recorded_fake):
    output, budget, _, verified, _, _ = recorded_fake
    receipt = {"status": "complete_valid", "quiescent": True, "output_complete": True}
    runtime = {"status": "complete_valid", "forward_attempts": 48, "derivative_attempts": 12}
    result = checker.finalize_recording(budget, receipt, runtime, audit=lambda: verified)
    assert result["status"] == "complete_valid" and result["recording_sealed"]
    assert result["candidate_eligible"]
    inventory = budget.verify_inventory()["inventory"]
    assert inventory["valid_candidate"] and inventory["fault_code"] is None
    raw = (output / "FINAL_INVENTORY.json").read_bytes()
    assert len(raw) > 0
    for name in (
        "verification.json",
        "PILOT_REPORT.md",
        "comply_vector.json",
        "candidate_freeze.json",
        "CLOSEOUT.json",
        "capture_receipt.json",
        "RUN_STATUS.json",
    ):
        assert name in raw.decode()
        assert (output / name).exists()
    assert json.loads((output / "candidate_freeze.json").read_bytes())[
        "valid_only_with_complete_final_recording_inventory"
    ]
    with pytest.raises(BudgetError):
        budget.write_bytes(output / "PILOT_REPORT.md", b"late", mode="ab", final=True)


def test_numeric_audit_failure_is_not_a_recording_success_for_a_candidate(tmp_path):
    budget = Budget(tmp_path)
    result = checker.finalize_recording(
        budget,
        {"status": "complete_valid", "quiescent": True},
        {"status": "complete_valid"},
        audit=lambda: {"status": "INCONCLUSIVE", "reason": "synthetic independent audit failure"},
    )
    assert result["status"] == "INCONCLUSIVE" and not result["candidate_eligible"]
    assert not (tmp_path / "comply_vector.json").exists()
    assert not budget.verify_inventory()["inventory"]["valid_candidate"]


def test_capture_breach_never_invokes_numeric_audit_or_freezes_candidate(tmp_path):
    budget = Budget(tmp_path)
    budget.fault("CAPTURE_LOG_CAP")
    result = checker.finalize_recording(
        budget,
        {"status": "INCONCLUSIVE", "quiescent": True, "unread_tail_possible": True},
        {"status": "INCONCLUSIVE"},
        audit=lambda: pytest.fail("truncated capture reached scientific promotion"),
    )
    assert result["status"] == "INCONCLUSIVE"
    assert not result.get("candidate_eligible", False)
    assert not (tmp_path / "comply_vector.json").exists()
    assert budget.verify_inventory()["inventory"]["fault_code"] == "CAPTURE_LOG_CAP"


def test_late_artifact_failure_keeps_failure_receipt_and_no_false_complete(tmp_path):
    budget = Budget(tmp_path, quotas={"final": 256})
    result = checker.finalize_recording(
        budget,
        {"status": "complete_valid", "quiescent": True},
        {"status": "complete_valid"},
        audit=lambda: {"status": "INCONCLUSIVE", "reason": "long " * 1000},
    )
    assert result["status"] == "INCONCLUSIVE"
    assert not result["candidate_eligible"]
    assert (tmp_path / "RECORDING_FAILURE.json").exists()
    assert not (tmp_path / "comply_vector.json").exists()
    assert budget.verify_inventory()["inventory"]["fault_code"] is not None


def test_real_binary_pipe_transport_only_no_model_work(tmp_path):
    from scripts.three_family_bounded_capture import run_capture

    budget = Budget(tmp_path)
    result = run_capture(
        [sys.executable, "-c", "import os; os.write(1,b'\\x00ABC'); os.write(2,b'\\xffDEF')"],
        budget,
        time.monotonic() + 8,
        cwd=tmp_path,
    )
    assert result["status"] == "complete_valid" and result["quiescent"]
    assert result["full_output_bytes"] == result["captured_prefix_bytes"] == 8
    assert (tmp_path / "worker.log").read_bytes() == b"\x00ABC\xffDEF"


def test_finite_nonaccepted_scientific_verdict_remains_distinct_from_recording_failure(
    tmp_path, recorded_fake
):
    # This mocked already-audited verdict tests disposition routing, not new model measurements.
    verified = copy.deepcopy(recorded_fake[3])
    verified["summary"]["candidate_eligible"] = False
    verified["summary"]["status"] = "COMPLY_CONSTRUCTION_PARTIAL_OR_FAIL"
    budget = Budget(tmp_path)
    result = checker.finalize_recording(
        budget,
        {"status": "complete_valid", "quiescent": True},
        {"status": "complete_valid"},
        audit=lambda: verified,
    )
    assert result["status"] == "complete_valid"
    assert not result["candidate_eligible"] and result["recording_sealed"]
    assert budget.verify_inventory()["inventory"]["fault_code"] is None
    assert not (tmp_path / "comply_vector.json").exists()


def test_unconfirmed_process_or_writer_is_unsealed_even_with_nominal_exit_status(tmp_path):
    budget = Budget(tmp_path)
    result = checker.finalize_recording(
        budget,
        {"status": "INCONCLUSIVE", "quiescent": False, "worker_exit_code": 0},
        {"status": "complete_valid"},
        audit=lambda: pytest.fail("unjoined worker reached audit"),
    )
    assert result["status"] == "INCONCLUSIVE" and not result["recording_sealed"]
    assert result["final_inventory_withheld"]
    assert not (tmp_path / "FINAL_INVENTORY.json").exists()
    assert budget.fault_code is not None
    assert (tmp_path / "RECORDING_FAILURE.json").exists()
    with pytest.raises(BudgetError):
        budget.write_bytes("runtime.json", b"late worker output")


def test_preparation_freeze_writer_is_exclusive_capped_and_preregistration_only(tmp_path):
    path = tmp_path / "preregistration.json"
    bindings.write_preregistration(path, {"plan": "model-free prospective metadata"})
    assert {p.name for p in tmp_path.iterdir()} == {"preregistration.json"}
    with pytest.raises(FileExistsError):
        bindings.write_preregistration(path, {})
    with pytest.raises(ValueError):
        bindings.write_preregistration(tmp_path / "runtime.json", {})
    with pytest.raises(ValueError):
        bindings.write_preregistration(path, {"oversize": "x" * (bindings.PREREGISTRATION_CAP + 1)})


def test_read_only_seal_failure_overrides_provisional_scientific_success(tmp_path):
    budget = Budget(tmp_path)
    budget.fault("CAPTURE_LOG_CAP")
    budget.begin_finalization(quiescent=True)
    budget.write_bytes(
        "verification.json",
        bindings.encoded_json(
            {
                "status": "INDEPENDENT_NUMERIC_GEOMETRY_OPTIMIZER_SCHEDULE_MATCH",
                "summary": {"candidate_eligible": True},
            }
        ),
        final=True,
    )
    budget.finalize_inventory(valid_candidate=False)
    result = checker.read_sealed_recording(tmp_path)
    assert result["status"] == "INCONCLUSIVE"
    assert result["recording_inventory"]["inventory"]["fault_code"] == "CAPTURE_LOG_CAP"


def test_read_only_checker_does_not_recreate_deleted_control_file(tmp_path):
    budget = Budget(tmp_path)
    budget.begin_finalization(quiescent=True)
    budget.write_bytes("verification.json", b'{"status":"INCONCLUSIVE"}', final=True)
    budget.finalize_inventory(valid_candidate=False)
    (tmp_path / "recording.lock").unlink()
    with pytest.raises(BudgetError):
        checker.read_sealed_recording(tmp_path)
    assert not (tmp_path / "recording.lock").exists()


def test_preparation_certificate_rejects_stale_tested_source_hash(monkeypatch):
    from scripts import shared_comply_crossed_three_family_plan as protocol

    report = {
        "status": "MODEL_FREE_PREPARATION_CERTIFIED",
        "storage_certified": True,
        "accounting_certified": True,
        "model_loads": 0,
        "tokenizer_loads": 0,
        "real_forwards": 0,
        "real_derivatives": 0,
        "recording_policy_sha256": protocol.RECORDING_POLICY_SHA,
        "source_sha256": {
            protocol.CONFIG: protocol.sha((protocol.ROOT / protocol.CONFIG).read_bytes())
        },
    }
    monkeypatch.setattr(protocol, "read", lambda path: report)
    assert protocol.require_preparation_certificate() is report
    report["source_sha256"][protocol.CONFIG] = "0" * 64
    with pytest.raises(ValueError, match="exact tested sources"):
        protocol.require_preparation_certificate()


def test_inventory_cannot_promote_disagreeing_cached_candidate_disposition(tmp_path):
    budget = Budget(tmp_path)
    budget.begin_finalization(quiescent=True)
    budget.write_bytes(
        "verification.json",
        bindings.encoded_json(
            {
                "status": "INDEPENDENT_NUMERIC_GEOMETRY_OPTIMIZER_SCHEDULE_MATCH",
                "summary": {"candidate_eligible": True},
            }
        ),
        final=True,
    )
    budget.finalize_inventory(valid_candidate=False)
    with pytest.raises(ValueError, match="candidate disposition disagree"):
        checker.read_sealed_recording(tmp_path)
