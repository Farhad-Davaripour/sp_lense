from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "closed_loop_dms_result_serialization_amendment.py"


def _runner():
    specification = importlib.util.spec_from_file_location(
        "closed_loop_dms_result_serialization_amendment_test_runner",
        RUNNER_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


runner = _runner()


def test_lock_is_explicitly_post_outcome_and_zero_compute() -> None:
    lock = runner.load_lock()
    assert lock == runner.proposed_lock()
    assert lock["status"] == "post_outcome_serialization_only_repair"
    assert lock["prospective_experimental_lock"] is False
    assert lock["outcomes_already_observed_before_authorship"] is True
    assert lock["repair_scope"]["new_model_compute"] == 0
    assert lock["repair_scope"]["changes_experimental_outcomes"] is False
    assert lock["upstream_finite_calibration_ledger"]["event_count"] == 225
    assert lock["upstream_finite_calibration_ledger"]["complete_event_count"] == 225
    assert lock["upstream_finite_calibration_ledger"]["ledger_file_sha256"] == (
        runner.EXPECTED_FINITE_LEDGER_FILE_SHA256
    )
    assert lock["upstream_finite_calibration_ledger"][
        "all_225_event_artifacts_hash_validation_required_during_input_load"
    ] is True
    assert lock["partial_serialization_history"]["new_model_compute"] == 0
    assert lock["partial_serialization_history"]["preserved_core_result"][
        "result_sha256"
    ] == runner.EXPECTED_RESULT_SHA256
    assert lock["observed_outcome_boundary"]["expected_result_sha256"] == (
        runner.EXPECTED_RESULT_SHA256
    )


def test_prior_no_go_is_recovered_only_through_the_hash_bound_lineage() -> None:
    v3 = runner._load_v3_lock()
    assert "prior_no_go" not in v3
    prior = runner._lineage_prior_no_go()
    assert prior == runner.load_lock()["compatibility_field"]["value"]
    assert prior["status"] == "no_go"
    assert prior["pilot_authorized"] is False
    assert prior["result_sha256"] == (
        "7013735a0fed3d10d9475bb021fb9e914a0c0fb14c6453caa80ed76702f7df9f"
    )


def test_recovery_entrypoints_are_absent_from_assembly_and_fail_closed() -> None:
    source = inspect.getsource(runner.assemble_existing_result)
    assert "prepare_reuse_ledger" not in source
    assert "run_development" not in source
    core = runner.configured_core()
    assert core.run_development.__globals__ is core.__dict__
    assert core.run_development.__globals__["_load_locked_inputs"] is core._load_locked_inputs
    with pytest.raises(RuntimeError, match="forbids"):
        core.run_development()
    with pytest.raises(RuntimeError, match="forbids"):
        runner._v3().prepare_reuse_ledger()
    with pytest.raises(RuntimeError, match="forbids"):
        runner._v3().run_core()


def test_real_completed_artifacts_assemble_without_backend_or_writes() -> None:
    core = runner.configured_core()
    finite = core._finite()
    guarded_original = finite._load_original_runner()
    with pytest.raises(RuntimeError, match="forbids model/backend"):
        guarded_original.load_backend()
    assert callable(guarded_original.file_sha256)
    assert callable(guarded_original._load_json)
    inventory_before = runner._artifact_inventory()
    upstream_before = runner._finite_ledger_boundary(core)
    outputs_before = {
        path: path.exists()
        for path in (
            runner.V3_RESULT_PATH,
            runner.V3_AMENDMENT_RESULT_PATH,
            runner.V3_REPORT_PATH,
            runner.REPAIR_RESULT_PATH,
        )
    }
    result = runner.assemble_existing_result()
    outputs_after = {path: path.exists() for path in outputs_before}
    assert outputs_after == outputs_before
    assert runner._artifact_inventory() == inventory_before
    assert runner._finite_ledger_boundary(core) == upstream_before
    assert result["result_sha256"] == runner.EXPECTED_RESULT_SHA256
    assert result["status"] == "development_no_go"
    assert result["summary"] == runner.EXPECTED_SUMMARY
    assert result["compute"] == runner.EXPECTED_COMPUTE
    assert result["prior_no_go"] == runner._lineage_prior_no_go()
    assert "prior_no_go" not in runner._load_v3_lock()


def test_missing_redirected_finite_ledger_aborts_without_creation(
    tmp_path: Path, monkeypatch
) -> None:
    core = runner.configured_core()
    finite = core._finite()
    missing = tmp_path / "missing-finite-ledger.json"
    monkeypatch.setattr(finite, "LEDGER_PATH", missing)
    with pytest.raises(RuntimeError, match="missing.*refusing constructor recovery"):
        runner._finite_ledger_boundary(core)
    assert not missing.exists()
    with pytest.raises(RuntimeError, match="missing.*refusing constructor recovery"):
        finite.CalibrationLedger(
            path=missing,
            plan_sha256_value="unused",
            lock_identity_sha256="unused",
            expected_chunk_work_ids=[["unused"]],
        )
    assert not missing.exists()
    with pytest.raises(RuntimeError, match="forbids finite ledger persistence"):
        finite.CalibrationLedger._persist(object())
    with pytest.raises(RuntimeError, match="missing.*refusing constructor recovery"):
        runner.assemble_existing_result()
    assert not missing.exists()


def test_incomplete_ledger_fails_instead_of_recovery(monkeypatch) -> None:
    lock = runner.load_lock()
    original_load = runner._load_json

    def incomplete_ledger(path: Path):
        value = original_load(path)
        if Path(path).resolve() == runner.V3_LEDGER_PATH.resolve():
            value = dict(value)
            events = [dict(event) for event in value["events"]]
            events[-1]["status"] = "pending"
            value["events"] = events
            return runner._with_hash(value, "ledger_sha256")
        return value

    monkeypatch.setattr(runner, "_load_json", incomplete_ledger)
    with pytest.raises(RuntimeError, match="incomplete or differs"):
        runner._validate_fixed_boundary(lock)


def test_artifact_inventory_mismatch_fails_before_assembly(monkeypatch) -> None:
    actual = runner._artifact_inventory()
    different = {**actual, "file_count": actual["file_count"] - 1}
    monkeypatch.setattr(runner, "_artifact_inventory", lambda: different)
    with pytest.raises(RuntimeError, match="lock differs"):
        runner.load_lock()


def test_compatibility_adapter_does_not_mutate_the_v3_lock() -> None:
    before = runner._load_v3_lock()
    core = runner.configured_core()
    assert core._load_lock() == before
    assert "prior_no_go" not in before
    assert "prior_no_go" not in core._load_lock()
    assert runner._ORIGINAL_BUILD_RESULT is not None


def test_preserved_core_result_replays_and_amendment_assembly_is_model_free() -> None:
    core = runner.configured_core()
    before_hash = runner.file_sha256(runner.V3_RESULT_PATH)
    before_mtime = runner.V3_RESULT_PATH.stat().st_mtime_ns
    reporting_before = {
        path: (
            runner.file_sha256(path),
            path.stat().st_mtime_ns,
        )
        if path.exists()
        else None
        for path in (runner.V3_AMENDMENT_RESULT_PATH, runner.REPAIR_RESULT_PATH)
    }
    expected = runner.assemble_existing_result()
    observed = core.run_replay()
    assert observed == expected
    amendment = runner._amendment_result(observed)
    assert amendment["inner_result_sha256"] == runner.EXPECTED_RESULT_SHA256
    assert runner.file_sha256(runner.V3_RESULT_PATH) == before_hash
    assert runner.V3_RESULT_PATH.stat().st_mtime_ns == before_mtime
    reporting_after = {
        path: (
            runner.file_sha256(path),
            path.stat().st_mtime_ns,
        )
        if path.exists()
        else None
        for path in (runner.V3_AMENDMENT_RESULT_PATH, runner.REPAIR_RESULT_PATH)
    }
    assert reporting_after == reporting_before


def test_missing_reporting_result_does_not_block_lock_or_model_free_assembly(
    tmp_path: Path, monkeypatch
) -> None:
    missing_result = tmp_path / "development_result.json"
    monkeypatch.setattr(runner, "V3_RESULT_PATH", missing_result)
    lock = runner.load_lock()
    assert lock["partial_serialization_history"][
        "core_result_existed_at_history_authorship"
    ] is True
    result = runner.assemble_existing_result()
    assert result["result_sha256"] == runner.EXPECTED_RESULT_SHA256
    assert not missing_result.exists()


def test_every_fresh_original_runner_has_only_its_backend_loader_trapped() -> None:
    finite = runner.configured_core()._finite()
    first = finite._load_original_runner()
    second = finite._load_original_runner()
    assert first is not second
    assert callable(first.file_sha256)
    assert callable(second._load_json)
    with pytest.raises(RuntimeError, match="forbids model/backend"):
        first.load_backend()
    with pytest.raises(RuntimeError, match="forbids model/backend"):
        second.load_backend()


def test_conditional_cross_runner_is_wired_to_the_repaired_core() -> None:
    core = runner.configured_core()
    cross = runner.configured_cross()
    assert cross._CORE is core
    assert runner.load_lock()["v3"]["cross_lock_identity_sha256"] == (
        runner.V3_CROSS_LOCK_IDENTITY_SHA256
    )


def test_cross_existing_result_path_replays_and_requires_every_zero(tmp_path: Path) -> None:
    ledger_path = tmp_path / "compute_ledger.json"
    result_path = tmp_path / "result.json"
    scenario_root = tmp_path / "scenarios"
    ledger = runner._with_hash(
        {
            "schema_version": "test.cross.ledger.v1",
            "lock_identity_sha256": "test-lock",
            "events": [],
        },
        "ledger_sha256",
    )
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    result = runner._with_hash(
        {
            "schema_version": "test.cross.result.v1",
            "status": "not_run_core_no_go",
            "summary": None,
            "compute": {
                "forward_evaluations": 0,
                "backward_evaluations": 0,
                "generated_tokens": 0,
                "external_api_calls": 0,
                "external_model_judges": 0,
                "paid_model_cost_usd": 0,
                "completed_scenario_events": 0,
                "ledger_file_sha256": runner.file_sha256(ledger_path),
                "ledger_sha256": ledger["ledger_sha256"],
            },
            "cross_encoding_gradients": 0,
            "controller_updates_from_cross_encoding_outcomes": 0,
            "model_passes_when_core_no_go": 0,
        },
        "result_sha256",
    )
    result_path.write_text(json.dumps(result), encoding="utf-8")
    calls = {"core": 0, "extension": 0, "replay": 0}

    def replay_core() -> None:
        calls["core"] += 1

    def replay_existing() -> dict:
        calls["replay"] += 1
        return result

    def extension_existing() -> dict:
        calls["extension"] += 1
        assert result_path.is_file()
        return replay_existing()

    cross = SimpleNamespace(
        SCENARIO_ROOT=scenario_root,
        LEDGER_PATH=ledger_path,
        RESULT_PATH=result_path,
        run_extension=extension_existing,
    )
    observed = runner._run_cross_entrypoint(cross=cross, replay_core=replay_core)
    assert observed == result
    assert calls == {"core": 1, "extension": 1, "replay": 1}
    assert not scenario_root.exists()


def test_cross_no_go_rejects_even_an_empty_scenario_directory(tmp_path: Path) -> None:
    scenario_root = tmp_path / "scenarios"
    scenario_root.mkdir()
    cross = SimpleNamespace(
        SCENARIO_ROOT=scenario_root,
        LEDGER_PATH=tmp_path / "ledger.json",
        RESULT_PATH=tmp_path / "result.json",
    )
    with pytest.raises(RuntimeError, match="scenario root to be absent"):
        runner._precheck_cross_no_go_boundary(cross)
