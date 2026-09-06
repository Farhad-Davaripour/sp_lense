"""New-only compact admission and actual native/full-vocabulary finalization tests."""

from __future__ import annotations

import copy
import json
import math
import struct
import sys
import time
from pathlib import Path

import pytest
from certified_descent_comply_fake_recording import build_full_recording

from scripts import certified_descent_comply_recording as recording
from scripts import three_family_recording_bindings as bindings
from scripts import verify_certified_descent_comply as checker
from scripts.three_family_recording_budget import BudgetError


def minimal_result(output, *, eligible=False):
    """Small admission/error fixture only; main coverage below uses full verify()."""
    rows = [
        {
            "cell_id": f"prompt_{i}__{condition}",
            "condition": condition,
            "rows_jsonl_line": stage * 12 + i + 1,
            "recorded_gradient_present": False,
        }
        for stage, condition in enumerate(("baseline", "final"))
        for i in range(12)
    ]
    finals = [{"cell_id": row["cell_id"], "accepted": eligible} for row in rows[12:]]
    vector = [0.1] + [0.0] * 1023
    endpoint = {
        "w": vector,
        "vector_float64_le_sha256": recording.sha(struct.pack("<1024d", *vector)),
    }
    result = {**endpoint, "final_cell_ids": [row["cell_id"] for row in finals]}
    budget = recording.PairedBudget(output)
    for name, value in (("endpoint.json", endpoint), ("result.json", result)):
        budget.write_bytes(output / name, recording.encode(value))
    verified = {
        "status": checker.AUDIT_MATCH,
        "summary": {
            "candidate_eligible": eligible,
            "final_accepted": 12 if eligible else 0,
            "final_cells": finals,
            "forward_count": 24,
            "derivative_count": 0,
            "attempted_updates": 0,
            "paired_response_trajectory": [],
        },
        "row_checks": rows,
        "optimizer_checks": [],
        "audited_endpoint_sha256": recording.sha((output / "endpoint.json").read_bytes()),
        "audited_result_sha256": recording.sha((output / "result.json").read_bytes()),
    }
    return budget, verified


CAPTURE = {"status": "complete_valid", "quiescent": True}
RUNTIME = {"status": "complete_valid"}


def finish(budget, verified, *, report=None, audit=None, capture=None):
    return recording.finalize_recording(
        budget,
        CAPTURE if capture is None else capture,
        RUNTIME,
        audit=(lambda: verified) if audit is None else audit,
        prepare_candidate_records=lambda value, digest: checker.prepare_candidate_records(
            value, digest, budget.root
        ),
        render_report=(lambda value: "unit fixture only\n") if report is None else report,
    )


@pytest.mark.parametrize("accepted", [True, False], ids=["interior_candidate", "finite_negative"])
def test_full_native_full_vocabulary_actual_finalizer_and_reader(tmp_path, monkeypatch, accepted):
    """No fabricated audit callback: full independent verify() executes exactly once."""
    started = time.perf_counter()
    measured = {"budget_write_calls": 0, "budget_write_seconds": 0.0}
    original_write = recording.PairedBudget.write_bytes

    def measured_write(self, path, data, mode="xb", final=False):
        before_write = time.perf_counter()
        try:
            return original_write(self, path, data, mode=mode, final=final)
        finally:
            measured["budget_write_calls"] += 1
            measured["budget_write_seconds"] += time.perf_counter() - before_write

    # Observation only: every payload/path/mode and every real budgeted write
    # is delegated unchanged. No cached admission, filesystem check or audit.
    monkeypatch.setattr(recording.PairedBudget, "write_bytes", measured_write)
    budget, runtime, captured, rows, result = build_full_recording(tmp_path, accepted=accepted)
    constructed = time.perf_counter()
    print(
        json.dumps(
            {
                "fixture_phase": "recording_complete_audit_not_started",
                "accepted_scripted": accepted,
                "construction_seconds": constructed - started,
                **measured,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for module in (
        checker,
        checker.inherited,
        checker.inherited.inherited,
        checker.legacy,
        checker.engine,
    ):
        monkeypatch.setattr(module, "OUTPUT", tmp_path)
    before = set(sys.modules)
    closed = checker.finalize_recording(budget, captured, runtime)
    finalized = time.perf_counter()
    print(
        json.dumps(
            {
                "fixture_phase": "actual_finalizer_returned",
                "accepted_scripted": accepted,
                "finalizer_seconds": finalized - constructed,
                "status": closed["status"],
                **measured,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    assert closed["status"] == "complete_valid", closed
    assert closed["candidate_eligible"] is accepted
    durable = checker.read_sealed_recording(tmp_path)
    assert durable["status"] == checker.AUDIT_MATCH
    assert len(durable["row_checks"]) == len(rows) == 216
    assert len(durable["optimizer_checks"]) == 8
    assert len(durable["summary"]["paired_response_trajectory"]) == 10
    assert durable["summary"]["forward_count"] == 216
    assert durable["summary"]["derivative_count"] == 96
    assert durable["summary"]["candidate_eligible"] is accepted
    assert 0 < result["net"] < 0.20
    assert all(row["logit_count"] == 248320 and len(row["h0"]) == 1024 for row in rows)
    assert not any(
        name.split(".")[0] in {"torch", "transformers", "transformer_lens"}
        for name in set(sys.modules) - before
    )
    sizes = {entry["path"]: entry for entry in durable["recording_inventory"]["inventory"]["files"]}
    # Actual sealed category includes every file, controls and inventory itself.
    inventory = durable["recording_inventory"]["inventory"]
    assert inventory["fault_code"] is None and inventory["quiescent"] is True
    assert inventory["valid_candidate"] is accepted
    assert inventory["total_bytes"] <= recording.QUOTAS["total"]
    assert (
        sum(
            v["bytes"]
            for name, v in sizes.items()
            if not name.startswith("logits/") and name not in {"rows.jsonl", "updates.jsonl"}
        )
        <= 16 * 1024**2
    )
    assert (
        sum(v["bytes"] for name, v in sizes.items() if name in recording.FINAL_CAPS) <= 5 * 1024**2
    )
    for name, cap in recording.FINAL_CAPS.items():
        if (tmp_path / name).exists():
            assert (tmp_path / name).stat().st_size <= cap
    assert (tmp_path / "verification.json").stat().st_size <= 1015808
    assert (
        max(
            len(line.encode()) + 1 for line in (tmp_path / "updates.jsonl").read_text().splitlines()
        )
        <= 8 * 1024**2
    )
    assert (
        max(len(line.encode()) + 1 for line in (tmp_path / "rows.jsonl").read_text().splitlines())
        <= 1024**2
    )
    # The exact complete serializer is also exercised with worst-size finite
    # binary64 native components. These encodings are never written as evidence
    # or passed off as scientifically valid rows/updates.
    largest = max(rows, key=lambda value: len(bindings.encoded_json(value, line=True)))
    extreme = -float.fromhex("0x1.fffffffffffffp+1023")
    inflated = copy.deepcopy(largest)
    vectors = [
        key for key, value in inflated.items() if isinstance(value, list) and len(value) == 1024
    ]
    assert len(vectors) == 7
    for key in vectors:
        inflated[key] = [extreme] * 1024
    assert len(bindings.encoded_json(inflated, line=True)) < 350000 < 1024**2
    updates = [json.loads(line) for line in (tmp_path / "updates.jsonl").read_text().splitlines()]
    for index, update in enumerate(updates, 1):
        assert update["status"] == "ready" and update["step_admitted"] is True
        assert len(update["history_w_sha256"]) == index
        solve = update["solver"]
        assert solve["gradient_count"] == solve["proposal_count"] == 1
        assert 1 <= solve["trial_count"] <= 53
        assert solve["chosen_j"] == solve["trials"][-1]["j"]
        assert solve["trials"][-1]["certificate"]["status"] == "ADMITTED_DESCENT"
        assert solve["trials"][-1]["certificate"]["geometry_valid"] is True
        assert update["w_after_sha256"] == solve["trials"][-1]["certificate"]["w_next_sha256"]
        assert update["point_repaired"] is False and update["near_optimality_gate"] is False
    inflated_update = max(updates, key=lambda value: len(bindings.encoded_json(value, line=True)))
    inflated_update = copy.deepcopy(inflated_update)
    vectors = [
        key
        for key, value in inflated_update.items()
        if isinstance(value, list) and len(value) == 1024
    ]
    assert set(vectors) == {"w_before", "w_after"}
    for key in vectors:
        inflated_update[key] = [extreme] * 1024
    assert len(bindings.encoded_json(inflated_update, line=True)) < 400000 < 8 * 1024**2
    print(
        json.dumps(
            {
                "full_native_recording_case": "interior_candidate"
                if accepted
                else "finite_negative",
                "output": str(tmp_path),
                "rows": len(rows),
                "updates": len(updates),
                "forwards": durable["summary"]["forward_count"],
                "derivatives": durable["summary"]["derivative_count"],
                "candidate_eligible": accepted,
                "construction_seconds": constructed - started,
                "finalizer_seconds": finalized - constructed,
                "reader_and_assertions_seconds": time.perf_counter() - finalized,
                **measured,
                "native_net": result["net"],
                "namespace_bytes": inventory["total_bytes"],
                "final_artifact_bytes": {
                    name: sizes[name]["bytes"] for name in recording.FINAL_CAPS if name in sizes
                },
                "verification_sha256": recording.sha((tmp_path / "verification.json").read_bytes()),
                "inventory_sha256": recording.sha((tmp_path / "FINAL_INVENTORY.json").read_bytes()),
                "inflated_row_bytes": len(bindings.encoded_json(inflated, line=True)),
                "inflated_update_bytes": len(bindings.encoded_json(inflated_update, line=True)),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def test_policy_exact_ceilings_and_helpers():
    policy = recording.recording_policy()
    assert policy["combined_category_bound_bytes"] == 524995016 < 512 * 1024**2
    assert policy["auxiliary_ceiling_bytes"] == 16 * 1024**2
    assert policy["minimum_free_bytes"] == 1024**3
    assert 262144 + 216 * 2048 + 8 * 32768 + 10 * 4096 + 8192 == 1015808
    assert sum(recording.FINAL_CAPS.values()) < 5 * 1024**2


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(unreviewed=[0.0] * 1024),
        lambda value: value["row_checks"][0].update(gradient=[0.0] * 1024),
        lambda value: value["row_checks"][0].update(signed_margin=math.nan),
        lambda value: value["row_checks"][0].update(cell_id="x" * 3000),
        lambda value: value["row_checks"][0].update(logits_sha256="x" * 64),
        lambda value: value["summary"].update(forward_count=216),
        lambda value: value["summary"].update(hidden=[0.0] * 1024),
        lambda value: value["summary"].update(candidate_eligible=True, final_accepted=12),
    ],
)
def test_invalid_or_oversized_compact_payload_faults_before_open(tmp_path, mutation):
    budget, verified = minimal_result(tmp_path)
    mutation(verified)
    closed = finish(budget, verified)
    assert closed["status"] == "INCONCLUSIVE" and not closed["candidate_eligible"]
    assert not (tmp_path / "verification.json").exists()
    assert not (tmp_path / "comply_vector.json").exists()
    assert recording.read_sealed_recording(tmp_path)["status"] == "INCONCLUSIVE"


def test_report_overflow_precedes_any_scientific_final_write(tmp_path):
    budget, verified = minimal_result(tmp_path, eligible=True)
    closed = finish(budget, verified, report=lambda value: "x" * (256 * 1024))
    assert closed["status"] == "INCONCLUSIVE"
    assert not (tmp_path / "verification.json").exists()
    assert not (tmp_path / "comply_vector.json").exists()


def test_zero_byte_verification_fault_wins_before_parse(tmp_path, monkeypatch):
    budget, verified = minimal_result(tmp_path, eligible=True)
    original = budget.write_bytes

    def fail_after_open(path, data, mode="xb", final=False):
        if Path(path).name == "verification.json":
            with budget.open_writer(path, "xb", final=True):
                pass
            budget.fault("ARTIFACT_BYTE_CAP")
            raise BudgetError("ARTIFACT_BYTE_CAP")
        return original(path, data, mode=mode, final=final)

    monkeypatch.setattr(budget, "write_bytes", fail_after_open)
    closed = finish(budget, verified)
    assert closed["status"] == "INCONCLUSIVE"
    assert (tmp_path / "verification.json").stat().st_size == 0
    assert not (tmp_path / "comply_vector.json").exists()
    read = recording.read_sealed_recording(tmp_path)
    assert read["status"] == "INCONCLUSIVE"
    assert read["recording_inventory"]["inventory"]["fault_code"] == "ARTIFACT_BYTE_CAP"


def test_sticky_fault_prevents_audit_and_candidate(tmp_path):
    budget, verified = minimal_result(tmp_path)
    budget.fault("FIRST_FAILURE")
    closed = finish(budget, verified, audit=lambda: pytest.fail("faulted attempt audited"))
    assert closed["status"] == "INCONCLUSIVE"
    assert closed["final_inventory"]["fault_code"] == "FIRST_FAILURE"
    assert not (tmp_path / "verification.json").exists()


def test_fault_arriving_during_audit_prevents_scientific_writes(tmp_path):
    budget, verified = minimal_result(tmp_path, eligible=True)

    def fault_after_audit():
        budget.fault("CAPTURE_DEADLINE")
        return verified

    closed = finish(budget, verified, audit=fault_after_audit)
    assert closed["status"] == "INCONCLUSIVE"
    assert not (tmp_path / "verification.json").exists()
    assert not (tmp_path / "comply_vector.json").exists()
    durable = recording.read_sealed_recording(tmp_path)
    assert durable["status"] == "INCONCLUSIVE"
    assert durable["recording_inventory"]["inventory"]["fault_code"] == "CAPTURE_DEADLINE"


def test_unquiescent_does_not_audit_or_seal(tmp_path):
    budget, verified = minimal_result(tmp_path)
    closed = finish(
        budget,
        verified,
        capture={"status": "INCONCLUSIVE", "quiescent": False},
        audit=lambda: pytest.fail("live mutable evidence audited"),
    )
    assert closed["status"] == "INCONCLUSIVE" and closed["final_inventory_withheld"]
    assert not (tmp_path / "FINAL_INVENTORY.json").exists()
    assert not (tmp_path / "verification.json").exists()


@pytest.mark.parametrize("name", ["verification.json", "endpoint.json", "candidate_freeze.json"])
def test_sealed_raw_tampering_is_rejected(tmp_path, name):
    budget, verified = minimal_result(tmp_path, eligible=True)
    assert finish(budget, verified)["status"] == "complete_valid"
    # Test-only hostile external writer, intentionally outside cooperative scope.
    with (tmp_path / name).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises((BudgetError, ValueError)):
        recording.read_sealed_recording(tmp_path)


def test_whole_record_admission_and_final_reserve(tmp_path):
    budget = recording.PairedBudget(tmp_path)
    with pytest.raises(BudgetError):
        budget.write_bytes(tmp_path / "rows.jsonl", b" " * (1024**2 + 1))
    assert not (tmp_path / "rows.jsonl").exists()
    assert budget.fault_code == "PAIRED_RECORD_BYTE_CAP"


def test_update_record_cap_precedes_open(tmp_path):
    budget = recording.PairedBudget(tmp_path)
    with pytest.raises(BudgetError):
        budget.write_bytes(tmp_path / "updates.jsonl", b" " * (8 * 1024**2 + 1))
    assert not (tmp_path / "updates.jsonl").exists()


def test_fixed_hash_containers_and_no_native_arrays(tmp_path):
    _budget, _verified = minimal_result(tmp_path)
    keys = {"A_row_sha256": ["0" * 64] * 12, "D_exact_row_sha256": ["1" * 64] * 6}
    recording._tree(keys)
    for key in keys:
        bad = copy.deepcopy(keys)
        bad[key][0] = "not_a_digest"
        with pytest.raises(ValueError):
            recording._tree(bad)


def test_certificate_requires_new_full_pipeline_attestations(tmp_path, monkeypatch):
    # This existing tiny test isolates inherited attestation validation. The new
    # evidence gate is exercised without this stub in the preparation test file.
    monkeypatch.setattr(recording, "_require_preparation_evidence", lambda certificate, root: None)
    monkeypatch.setattr(recording.frozen, "recording_policy", lambda root: {})
    monkeypatch.setattr(recording.frozen, "certificate_source_paths", lambda: frozenset())
    monkeypatch.setattr(recording.frozen, "IMMUTABLE_HELPERS", {})
    certificate = {
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
        "source_sha256": {},
        "entrypoint_coverage": dict.fromkeys(recording.ENTRYPOINT_COVERAGE_KEYS, True),
        "finalization_coverage": dict.fromkeys(recording.FINALIZATION_COVERAGE_KEYS, True),
    }
    path = tmp_path / recording.CERTIFICATE
    path.parent.mkdir()
    path.write_bytes(recording.encode(certificate))
    assert recording.require_certificate(tmp_path)["run_authorized"] is False
    for key in recording.FINALIZATION_COVERAGE_KEYS:
        bad = copy.deepcopy(certificate)
        bad["finalization_coverage"][key] = False
        path.write_bytes(recording.encode(bad))
        with pytest.raises(ValueError):
            recording.require_certificate(tmp_path)
