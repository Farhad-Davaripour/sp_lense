from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from sp_lense.decision_margin_shield_finite import SCREEN_METHODS

ROOT = Path(__file__).parents[1]
RUNNER_PATH = ROOT / "scripts" / "decision_margin_shield_finite_calibration.py"
SPEC = importlib.util.spec_from_file_location("dms_finite_runner_tests", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _form(form_id: str) -> dict:
    return {
        "form_id": form_id,
        "family": "unrelated",
        "control_id": form_id,
        "control_partition": "calibration",
        "preferred_first": True,
        "encoding": "AB",
        "anchor_prefix": "anchor\n",
        "prompt": "anchor\nprompt",
        "positive_label": "A",
        "negative_label": "B",
        "positive_semantic": "preferred",
        "negative_semantic": "alternative",
        "anchor_index": None,
    }


def _baseline_spec(index: int) -> dict:
    baseline_id = f"baseline:f{index}"
    return {
        "kind": "baseline",
        "work_id": baseline_id,
        "baseline_id": baseline_id,
        "form": _form(f"f{index}"),
    }


def _record(specification: dict, logits: torch.Tensor) -> dict:
    record = {
        **runner.public_work_spec(specification),
        "logits_float32_sha256": runner.tensor_float32_sha256(logits),
    }
    record["row_sha256"] = runner.canonical_sha256(record)
    return record


def test_screen_binding_is_selected_layer_zero_and_self_hashed() -> None:
    result = runner._load_screen_result()
    assert result["result_sha256"] == runner.SCREEN_RESULT_SHA256
    assert result["selection"]["selected_layer"] == 0


def test_finite_lock_revalidates_and_binds_original_screen_source_closure() -> None:
    dependencies = runner._bound_dependency_records()
    required = {
        "original_screen_lock",
        "original_screen_source_anchor_runtime",
        "original_screen_source_factorial_math",
        "original_screen_source_backend",
        "original_screen_source_comparison_runtime",
        "original_screen_source_semantic_completion_gradient",
    }
    assert required <= set(dependencies)
    assert all(len(row["sha256"]) == 64 for row in dependencies.values())


def test_control_qualification_bank_is_fixed_disjoint_and_exactly_eight_forwards() -> None:
    bank = runner._load_control_candidates()
    plan = runner._qualification_plan()
    dependencies = runner._qualification_dependency_records()
    assert len(bank["candidates"]) == 4
    assert len(plan) == 8
    assert len({row["work_id"] for row in plan}) == 8
    assert {row["preferred_first"] for row in plan} == {True, False}
    assert bank["known_bad_legacy_view"]["prompt_sha256"] == (
        runner.KNOWN_BAD_LEGACY_PROMPT_SHA256
    )
    assert "original_model_loader_lock" in dependencies
    assert any(key.startswith("original_model_loader_source_") for key in dependencies)
    proposed = runner.proposed_qualification_lock()
    assert proposed["compute_ceiling"]["model_forwards"] == 8
    assert proposed["compute_ceiling"]["model_backwards"] == 0
    assert proposed["compute_ceiling"]["generated_tokens"] == 0
    assert proposed["prohibited_inputs"] == {
        "direction_artifacts": True,
        "intervention_outcomes": True,
        "pilot_outcomes": True,
        "margin_based_selection": True,
    }


def test_control_qualification_selects_first_two_order_pass_without_using_margin() -> None:
    bank = runner._load_control_candidates()
    records = []
    for row in runner._qualification_public_plan():
        candidate_index = int(row["candidate_index"])
        semantic = (
            "alternative"
            if candidate_index == 0 and row["preferred_first"] is False
            else "preferred"
            if candidate_index in {1, 2}
            else "OTHER"
        )
        record = {
            **row,
            "unrestricted_semantic_choice": semantic,
            "answer_format_valid": semantic != "OTHER",
            # These descriptive values must never enter the selection rule.
            "preferred_minus_alternative_log_odds": -999.0 + candidate_index,
        }
        record["row_sha256"] = runner.canonical_sha256(record)
        records.append(record)
    selection = runner._select_qualified_control(records, bank)
    assert selection["status"] == "passed"
    assert selection["selected_control"]["id"] == bank["candidates"][1]["id"]
    assert selection["candidate_assessments"][0]["passes_both_orders"] is False
    assert selection["candidate_assessments"][1]["passes_both_orders"] is True


def test_all_twelve_real_layer_zero_directions_reproduce_immutable_screen_hashes() -> None:
    original = runner._load_original_runner()
    records = original._load_capture_records(torch)
    dataset = runner._load_dataset()
    screen = runner._load_screen_result()
    observed = {}
    for scenario in dataset["scenarios"]:
        if scenario["partition"] != "calibration":
            continue
        scenario_id = str(scenario["id"])
        inputs = runner._direction_inputs(torch, records, scenario_id)
        directions = runner.reconstruct_scenario_directions(
            scenario_id=scenario_id,
            screen_result=screen,
            **{
                key: value
                for key, value in inputs.items()
                if key != "captured_anchor_residuals"
            },
        )
        for method, direction in directions.items():
            screen_record = next(
                row
                for row in screen["geometry_records"]
                if row["layer"] == 0
                and row["scenario_id"] == scenario_id
                and row["method"] == SCREEN_METHODS[method]
            )
            assert direction.direction_sha256 == screen_record["direction_sha256"]
            certificate = runner.deployment_recertificate(
                direction,
                target_rows=inputs["target_rows"],
                target_offsets=inputs["target_offsets"],
                protected_rows=inputs["protected_rows"],
                protected_offsets=inputs["protected_offsets"],
                unrelated_rows=inputs["unrelated_rows"],
                captured_anchor_residuals=inputs["captured_anchor_residuals"],
            )
            assert certificate["passes"] is True
            assert certificate["simulated_float32_residual_addition_count"] == 24 * 3 * 2
            observed[(scenario_id, method)] = direction.direction_sha256
    assert len(observed) == 12


def test_runner_exposes_no_pilot_command() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert '"qualification-lock"' in source
    assert '"qualify-controls"' in source
    assert '"calibrate"' in source
    assert "run_pilot" not in source


def test_ledger_is_lock_bound_and_rejects_wrong_plan_prefix(tmp_path: Path) -> None:
    chunks = [["one"], ["two"]]
    path = tmp_path / "ledger.json"
    ledger = runner.CalibrationLedger(
        path=path,
        plan_sha256_value="plan",
        lock_identity_sha256="lock",
        expected_chunk_work_ids=chunks,
    )
    with pytest.raises(RuntimeError, match="reservation differs"):
        ledger.reserve(0, ["wrong"])
    runner.CalibrationLedger(
        path=path,
        plan_sha256_value="plan",
        lock_identity_sha256="lock",
        expected_chunk_work_ids=chunks,
    )
    with pytest.raises(RuntimeError, match="identity differs"):
        runner.CalibrationLedger(
            path=path,
            plan_sha256_value="plan",
            lock_identity_sha256="different",
            expected_chunk_work_ids=chunks,
        )


def test_ledger_binds_complete_event_to_exact_artifact_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_root = tmp_path / "chunks"
    monkeypatch.setattr(runner, "CHECKPOINT_ROOT", checkpoint_root)
    path = tmp_path / "ledger.json"
    ledger = runner.CalibrationLedger(
        path=path,
        plan_sha256_value="plan",
        lock_identity_sha256="lock",
        expected_chunk_work_ids=[["one"]],
    )
    ledger.reserve(0, ["one"])
    artifact = runner._chunk_path(0)
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"first")
    ledger.complete(0, artifact)
    artifact.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="artifact differs"):
        runner.CalibrationLedger(
            path=path,
            plan_sha256_value="plan",
            lock_identity_sha256="lock",
            expected_chunk_work_ids=[["one"]],
        )


def test_baseline_chunk_round_trip_is_lossless_and_immutable(tmp_path: Path) -> None:
    specs = [_baseline_spec(index) for index in range(2)]
    logits = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    records = [_record(spec, logits[index]) for index, spec in enumerate(specs)]
    path = tmp_path / "chunk.pt"
    runner._save_chunk(
        torch,
        path=path,
        index=0,
        plan_hash="plan",
        expected_specs=specs,
        records=records,
        baseline_logits=logits,
    )
    loaded, observed = runner._load_chunk(
        torch,
        path=path,
        index=0,
        plan_hash="plan",
        expected_specs=specs,
    )
    assert loaded == records
    assert torch.equal(observed, logits)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner._save_chunk(
            torch,
            path=path,
            index=0,
            plan_hash="plan",
            expected_specs=specs,
            records=records,
            baseline_logits=logits,
        )


def test_changed_chunk_stores_scalars_without_full_logits(tmp_path: Path) -> None:
    form = _form("f")
    spec = {
        "kind": "changed",
        "work_id": "changed:f",
        "baseline_id": "baseline:f",
        "method": "decision_margin_shield",
        "strength": 0.5,
        "sign": 1,
        "direction_scenario_id": "scenario",
        "form": form,
    }
    record = {**runner.public_work_spec(spec), "logits_float32_sha256": "a" * 64}
    record["row_sha256"] = runner.canonical_sha256(record)
    path = tmp_path / "changed.pt"
    runner._save_chunk(
        torch,
        path=path,
        index=0,
        plan_hash="plan",
        expected_specs=[spec],
        records=[record],
        baseline_logits=None,
    )
    loaded, logits = runner._load_chunk(
        torch,
        path=path,
        index=0,
        plan_hash="plan",
        expected_specs=[spec],
    )
    assert loaded == [record]
    assert logits is None


def test_chunk_rejects_rehashed_wrong_plan_identity(tmp_path: Path) -> None:
    specs = [_baseline_spec(0)]
    logits = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    records = [_record(specs[0], logits[0])]
    path = tmp_path / "chunk.pt"
    runner._save_chunk(
        torch,
        path=path,
        index=0,
        plan_hash="plan",
        expected_specs=specs,
        records=records,
        baseline_logits=logits,
    )
    with pytest.raises(RuntimeError, match="identity differs"):
        runner._load_chunk(
            torch,
            path=path,
            index=0,
            plan_hash="different",
            expected_specs=specs,
        )


def test_qualification_checkpoint_rejects_tampered_logits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "qualification.pt"
    lock_path = tmp_path / "qualification-lock.json"
    lock_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "QUALIFICATION_CHECKPOINT_PATH", checkpoint_path)
    monkeypatch.setattr(runner, "QUALIFICATION_LOCK_PATH", lock_path)
    compute = {
        "model_loads": 1,
        "model_forwards": 8,
        "model_backwards": 0,
        "generated_tokens": 0,
        "external_model_judges": 0,
        "external_api_calls": 0,
        "paid_model_cost_usd": 0,
    }
    fake_lock = {
        "lock_identity_sha256": "a" * 64,
        "qualification_plan_sha256": runner.canonical_sha256(
            runner._qualification_public_plan()
        ),
        "compute_ceiling": compute,
    }
    monkeypatch.setattr(runner, "_load_qualification_lock", lambda: fake_lock)
    public_plan = runner._qualification_public_plan()
    all_logits = []
    records = []
    for index, expected in enumerate(public_plan):
        logits = torch.linspace(-1.0, 1.0, 11, dtype=torch.float32) + index
        score = runner._qualification_score(
            logits, preferred_token_id=10, alternative_token_id=0
        )
        record = {**expected, "input_ids_sha256": "b" * 64, **score}
        record["row_sha256"] = runner.canonical_sha256(record)
        records.append(record)
        all_logits.append(logits)
    runner._save_qualification_checkpoint(
        torch,
        lock=fake_lock,
        records=records,
        logits=torch.stack(all_logits),
        backend_metadata={"model": "fake"},
    )
    monkeypatch.setattr(
        runner,
        "_load_qualification_ledger",
        lambda **_: {"artifact_sha256": runner.file_sha256(checkpoint_path)},
    )
    runner._load_qualification_checkpoint(torch)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["logits"][0, 0] += 0.5
    torch.save(payload, checkpoint_path)
    with pytest.raises(RuntimeError, match="score differs from stored logits"):
        runner._load_qualification_checkpoint(torch)
