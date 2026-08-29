from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORTER_PATH = ROOT / "scripts" / "counterfactual_tangent_shield_v1_no_go_report.py"


def _load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


reporter = _load_module("cts_v1_no_go_report_tests", REPORTER_PATH)


def _runner_with_forbidden_backend(monkeypatch: pytest.MonkeyPatch):
    runner = reporter._load_runner()

    def forbidden_backend():
        pytest.fail("the no-go reporter must never load a model backend")

    monkeypatch.setattr(runner, "load_backend", forbidden_backend)
    return runner


def test_model_free_report_validates_and_preserves_the_no_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner_with_forbidden_backend(monkeypatch)
    result_path = tmp_path / "construction_no_go_result.json"
    report_path = tmp_path / "CONSTRUCTION_NO_GO.md"
    result = reporter.run_no_go_report(
        runner=runner,
        result_path=result_path,
        report_path=report_path,
    )

    assert result["status"] == "construction_no_go"
    assert result["decision"] == {
        "construction_gate_passes": False,
        "eligible_direction_count": 0,
        "calibration_changed_row_count": 0,
        "pilot_authorized": False,
        "reason": (
            "All 88 frozen construction records were ineligible under the locked "
            "target, nuisance, and L2 constraints, so no intervention candidate "
            "was eligible for finite calibration."
        ),
    }
    assert result["validation"]["direction_bank"]["direction_record_count"] == 88
    assert result["validation"]["direction_bank"]["eligible_direction_count"] == 0
    assert result["validation"]["calibration_freeze"]["baseline_count"] == 72
    assert result["validation"]["calibration_freeze"]["changed_count"] == 0
    assert result["validation"]["completed_calibration_ledger"][
        "completed_chunk_count"
    ] == 3
    assert result["validation"]["baseline_checkpoints"]["row_count"] == 72
    boundary = result["validation"]["capture_to_finite_boundary_binding"]
    assert boundary["count"] == 64
    assert boundary["exact_match_count"] == 64
    assert boundary["maximum_absolute_difference"] == 0.0
    collateral = result["validation"]["collateral_baseline_correctness"]
    assert collateral["view_count"] == 8
    assert collateral["correct_view_count"] == 7
    assert collateral["invalid_view_count"] == 0
    edge = result["validation"]["known_runner_error_provenance"]
    assert edge["error_type"] == "RuntimeError"
    assert edge["error"] == reporter.EXPECTED_RUNNER_ERROR
    assert edge["triggering_changed_row_count"] == 0
    assert edge["provenance_kind"].startswith("model_free_deterministic_reproduction")
    assert result["compute"]["reporter_model_forwards"] == 0
    assert result["compute"]["reporter_model_backwards"] == 0

    persisted = json.loads(result_path.read_text(encoding="utf-8"))
    runner._verify_hash(persisted, "result_sha256")
    markdown = report_path.read_text(encoding="utf-8")
    assert runner.file_sha256(result_path) in markdown
    assert result["result_sha256"] in markdown
    assert "No intervention effect or decision change was measured" in markdown


def test_completed_prefix_validation_fails_closed_on_pending_event(
    tmp_path: Path,
) -> None:
    runner = reporter._load_runner()
    ledger = runner._load_json(runner.CALIBRATION_LEDGER_PATH)
    ledger["events"][-1]["status"] = "pending"
    event = dict(ledger["events"][-1])
    event.pop("event_sha256")
    ledger["events"][-1] = runner._with_hash(event, "event_sha256")
    ledger = runner._with_hash(
        {key: value for key, value in ledger.items() if key != "ledger_sha256"},
        "ledger_sha256",
    )
    ledger_path = tmp_path / "pending_ledger.json"
    ledger_path.write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="prefix is not fully complete"):
        reporter._validate_completed_ledger_prefix(runner, ledger_path=ledger_path)


def test_immutable_outputs_are_idempotent_but_never_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner_with_forbidden_backend(monkeypatch)
    result_path = tmp_path / "construction_no_go_result.json"
    report_path = tmp_path / "CONSTRUCTION_NO_GO.md"
    first = reporter.run_no_go_report(
        runner=runner,
        result_path=result_path,
        report_path=report_path,
    )
    original_result = result_path.read_bytes()
    original_report = report_path.read_bytes()
    second = reporter.run_no_go_report(
        runner=runner,
        result_path=result_path,
        report_path=report_path,
    )
    assert second == first
    assert result_path.read_bytes() == original_result
    assert report_path.read_bytes() == original_report

    result_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differing immutable no-go result"):
        reporter.run_no_go_report(
            runner=runner,
            result_path=result_path,
            report_path=report_path,
        )


def test_direction_validation_rejects_any_eligible_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = reporter._load_runner()
    manifest = runner._load_json(runner.DIRECTION_MANIFEST_PATH)
    manifest["eligible_direction_count"] = 1
    monkeypatch.setattr(runner, "_validate_direction_manifest", lambda: manifest)
    monkeypatch.setattr(runner, "_load_directions", dict)

    with pytest.raises(RuntimeError, match="zero-eligible no-go"):
        reporter._validate_direction_no_go(runner)
