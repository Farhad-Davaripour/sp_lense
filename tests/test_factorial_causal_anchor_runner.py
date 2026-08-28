from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "factorial_causal_anchor_gradient_pilot.py"


def _load_runner():
    specification = importlib.util.spec_from_file_location("fcags_runner_tests", RUNNER_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not import runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def test_lock_has_exact_fair_method_compute_budget() -> None:
    lock = runner.proposed_lock()
    assert lock["compute_ceiling"] == {
        "capture": {"forward": 136, "backward": 136},
        "calibration": {"forward": 3144, "backward": 0},
        "pilot": {"forward": 2824, "backward": 0},
        "total": {"forward": 6104, "backward": 136, "generated_tokens": 0},
        "external_model_judges": 0,
        "external_api_calls": 0,
        "paid_cost_usd": 0,
    }
    assert (
        lock["evaluation"]["calibration_success"][
            "same_selection_rule_for_every_compared_method"
        ]
        is True
    )
    assert lock["construction"]["target"] == (
        "name_balanced_self_x_permanence_difference_in_differences"
    )


def test_empty_random_control_kl_report_is_strict_and_json_safe() -> None:
    report = runner._kl_report([])
    assert report == {"count": 0, "mean": None, "p95": None, "max": None, "passes": False}
    json.dumps(report, allow_nan=False)


def test_bundle_reuse_audit_rejects_changed_hash_within_one_unit() -> None:
    base = {
        "method": "protected_factorial",
        "direction_scenario_id": "case",
        "alpha": 0.01,
        "sign": 1,
        "direction_sha256": "direction",
        "perturbation_bundle_float32_sha256": "same",
    }
    assert runner._audit_bundle_reuse([base, dict(base)])["passes"] is True
    changed = dict(base)
    changed["perturbation_bundle_float32_sha256"] = "different"
    with pytest.raises(RuntimeError, match="more than one perturbation"):
        runner._audit_bundle_reuse([base, changed])


def test_deranged_direction_uses_source_orientation_and_target_scales() -> None:
    source = {
        "scenario_id": "source",
        "layers": list(runner.LAYERS),
        "standardized_direction": torch.ones(
            len(runner.LAYERS), runner.MODEL["d_model"], dtype=torch.float32
        ),
        "residual_scales": torch.full((len(runner.LAYERS),), 2.0),
    }
    target = {
        "scenario_id": "target",
        "layers": list(runner.LAYERS),
        "standardized_direction": torch.zeros_like(source["standardized_direction"]),
        "residual_scales": torch.full((len(runner.LAYERS),), 3.0),
    }
    derived = runner._target_scaled_deranged_direction(torch, source=source, target=target)
    assert torch.equal(derived["standardized_direction"], source["standardized_direction"])
    assert torch.equal(
        derived["unit_absolute_perturbations"],
        3.0 * source["standardized_direction"],
    )
    assert derived["scenario_id"] == "target"
    assert derived["source_scenario_id"] == "source"


def test_meter_rejects_duplicate_work_ids() -> None:
    meter = runner.Meter(phase="test", ceiling={"forward": 2, "backward": 1})
    meter.reserve_forward("one")
    with pytest.raises(RuntimeError, match="duplicate forward work ID"):
        meter.reserve_forward("one")
    meter.reserve_backward("gradient")
    with pytest.raises(RuntimeError, match="duplicate backward work ID"):
        meter.reserve_backward("gradient")


def test_cached_calibration_rejects_wrong_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "lock.json"
    direction = tmp_path / "direction.pt"
    rows = tmp_path / "rows.jsonl"
    summary_path = tmp_path / "summary.json"
    lock.write_text("current", encoding="utf-8")
    direction.write_text("direction", encoding="utf-8")
    rows.write_text("", encoding="utf-8")
    summary = {
        "schema_version": "sp_lense.fcags_calibration_summary.v1",
        "lock_sha256": "wrong",
        "direction_bank_sha256": runner.file_sha256(direction),
    }
    summary["summary_sha256"] = runner.canonical_sha256(summary)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(runner, "LOCK_PATH", lock)
    monkeypatch.setattr(runner, "DIRECTION_PATH", direction)
    monkeypatch.setattr(runner, "CALIBRATION_ROWS_PATH", rows)
    monkeypatch.setattr(runner, "CALIBRATION_SUMMARY_PATH", summary_path)
    with pytest.raises(RuntimeError, match="different lock"):
        runner._load_cached_calibration()


def test_partial_result_pair_is_quarantined_and_phase_can_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = tmp_path / "rows.jsonl"
    summary = tmp_path / "summary.json"
    rows.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "PILOT_ROWS_PATH", rows)
    monkeypatch.setattr(runner, "PILOT_SUMMARY_PATH", summary)
    assert runner._load_cached_pilot() is None
    assert not rows.exists()
    assert len(list(tmp_path.glob("rows.jsonl.incomplete-*"))) == 1
