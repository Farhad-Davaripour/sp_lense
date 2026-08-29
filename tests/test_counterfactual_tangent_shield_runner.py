from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "counterfactual_tangent_shield_development.py"


def _load_runner():
    specification = importlib.util.spec_from_file_location("cts_runner_tests", RUNNER_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not import runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def _form(form_id: str, *, scenario_id: str | None = None, family: str = "scenario"):
    result = {
        "form_id": form_id,
        "family": family,
        "prompt": f"prompt {form_id}",
        "anchor_prefix": f"prefix {form_id}",
        "positive_label": "A",
        "negative_label": "B",
        "positive_semantic": "preserve" if family == "scenario" else "preferred",
        "negative_semantic": "comply" if family == "scenario" else "alternative",
        "anchor_index": 3,
        "encoding": "AB",
    }
    if scenario_id is not None:
        result.update(
            {
                "scenario_id": scenario_id,
                "assignment": 0,
                "target": "self",
                "event": "permanent",
            }
        )
    return result


def test_lock_binds_exact_scope_runtime_and_compute() -> None:
    lock = runner.proposed_lock()
    assert lock["construction"]["zero_based_layer"] == 10
    assert lock["dataset"]["construction_scope"] == "all_scenarios_ab"
    assert lock["dataset"]["pilot_partition_role"].startswith("intervention_outcome_holdout")
    assert lock["construction"]["semantic_anchor_source_method"] == "raw_factorial"
    assert lock["construction"]["semantic_anchor_tau_methods"] == list(runner.SEMANTIC_METHODS)
    assert lock["runtime"]["numpy"] == "2.5.2"
    assert lock["runtime"]["scipy"] == "1.18.1"
    assert lock["compute_ceiling"] == {
        "capture": {"forward": 136, "backward": 136},
        "calibration": {"forward": 4680, "backward": 0},
        "pilot": {"forward": 2520, "backward": 0},
        "generated_tokens": 0,
    }
    assert lock["artifact_policy"]["sealed_project_paths_read"] == []
    assert "runner_tests" in lock["source_files"]
    assert lock["source_files"]["base_requirements"]["path"] == "requirements-research.txt"
    assert lock["source_files"]["cts_requirements"]["path"] == (
        "requirements-counterfactual-tangent-shield.txt"
    )


def test_capture_plan_is_exactly_136_unique_one_f_plus_b_units() -> None:
    specifications = runner._capture_specifications(runner._load_dataset())
    assert len(specifications) == 136
    assert len({item["work_id"] for item in specifications}) == 136
    assert sum(item["kind"] == "scenario" for item in specifications) == 128
    assert sum(item["kind"] == "nuisance_fit" for item in specifications) == 8
    assert {item["partition"] for item in specifications if item["kind"] == "scenario"} == {
        "calibration",
        "pilot",
    }


def test_full_calibration_plan_has_exact_4680_rows_without_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario_ids = [f"cal-{index}" for index in range(4)]
    fake_dataset = {
        "scenarios": [
            {"id": scenario_id, "partition": "calibration"} for scenario_id in scenario_ids
        ]
    }
    directions = {
        (scenario_id, method): {"status": "eligible"}
        for scenario_id in scenario_ids
        for method in runner.CALIBRATION_METHODS
    }
    scenario_forms = [
        _form(f"{scenario_id}:scenario-{index}", scenario_id=scenario_id)
        for scenario_id in scenario_ids
        for index in range(16)
    ]
    collateral_forms = [_form(f"control-{index}", family="collateral") for index in range(8)]
    monkeypatch.setattr(runner, "_load_dataset", lambda: fake_dataset)
    monkeypatch.setattr(runner, "_load_directions", lambda: directions)
    monkeypatch.setattr(
        runner,
        "_applied_multiplier_certificate",
        lambda *args, **kwargs: {"passes": True},
    )
    monkeypatch.setattr(
        runner, "_scenario_evaluation_forms", lambda *args, **kwargs: scenario_forms
    )
    monkeypatch.setattr(
        runner, "_collateral_evaluation_forms", lambda *args, **kwargs: collateral_forms
    )
    monkeypatch.setattr(
        runner,
        "load_backend",
        lambda: pytest.fail("work-plan construction must not load the model"),
    )
    plan, audit = runner._calibration_plan()
    assert audit["baseline_count"] == 72
    assert audit["changed_count"] == 4608
    assert len(plan) == 4680
    for method in runner.CALIBRATION_METHODS:
        for multiplier in runner.MULTIPLIERS:
            assert (
                sum(
                    item["kind"] == "changed"
                    and item["method"] == method
                    and item["multiplier"] == multiplier
                    for item in plan
                )
                == 192
            )


def test_full_pilot_plan_has_exact_2520_rows_and_four_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario_ids = [f"pilot-{index}" for index in range(4)]
    cts = "cts_tau_0p01"
    semantic = "semantic_tau_0p01"
    methods = (cts, "unshielded", semantic, "random_null_17011")
    selected = {
        method: {"method": method, "multiplier": 1.0} for method in runner.CALIBRATION_METHODS
    }
    calibration = {
        "pilot_authorized": True,
        "selected_cts": selected[cts],
        "matched_semantic_method": semantic,
        "pilot_baseline_selections": {
            method: selected[method] for method in ("unshielded", semantic, "random_null_17011")
        },
    }
    fake_dataset = {
        "scenarios": [{"id": scenario_id, "partition": "pilot"} for scenario_id in scenario_ids]
    }
    directions = {
        (scenario_id, method): {"status": "eligible"}
        for scenario_id in scenario_ids
        for method in methods
    }
    scenario_forms = [
        _form(f"{scenario_id}:scenario-{index}", scenario_id=scenario_id)
        for scenario_id in scenario_ids
        for index in range(48)
    ]
    collateral_forms = [_form(f"pilot-control-{index}", family="collateral") for index in range(24)]
    monkeypatch.setattr(runner, "_load_dataset", lambda: fake_dataset)
    monkeypatch.setattr(runner, "_load_directions", lambda: directions)
    monkeypatch.setattr(
        runner,
        "_applied_multiplier_certificate",
        lambda *args, **kwargs: {"passes": True},
    )
    monkeypatch.setattr(
        runner, "_scenario_evaluation_forms", lambda *args, **kwargs: scenario_forms
    )
    monkeypatch.setattr(
        runner, "_collateral_evaluation_forms", lambda *args, **kwargs: collateral_forms
    )
    plan, audit = runner._pilot_plan(calibration)
    assert audit["baseline_count"] == 216
    assert audit["changed_count"] == 2304
    assert len(plan) == 2520
    assert set(audit["method_multipliers"]) == set(methods)
    assert all(
        sum(item["kind"] == "changed" and item["method"] == method for item in plan) == 576
        for method in methods
    )


def test_physical_signs_are_exact_float32_negations() -> None:
    base = torch.tensor([0.25, -0.5, 1.0], dtype=torch.float32)
    plus = runner._signed_physical_direction(base, multiplier=1.15, sign=1)
    minus = runner._signed_physical_direction(base, multiplier=1.15, sign=-1)
    assert plus.dtype == torch.float32
    assert torch.equal(minus, -plus)
    assert plus.numpy().tobytes() != minus.numpy().tobytes()
    with pytest.raises(ValueError, match="sign"):
        runner._signed_physical_direction(base, multiplier=1.0, sign=0)


def test_random_control_certificate_does_not_require_target_success() -> None:
    solution = SimpleNamespace(
        direction=np.array([1.0, 0.0], dtype=np.float64),
        diagnostics={"method": "fake_random"},
    )
    record = runner._constructed_record(
        torch,
        scenario_id="scenario",
        method="random_null_17011",
        solution=solution,
        residual_scale=2.0,
        target_rows=torch.tensor([[-1.0, 0.0]], dtype=torch.float64),
        target_offsets=torch.tensor([0.0], dtype=torch.float64),
        nuisance_rows=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        nuisance_bound=0.0,
        require_target_certificate=False,
    )
    certificate = record["diagnostics"]["float32_certificate"]
    assert record["status"] == "eligible"
    assert certificate["target_passes"] is False
    assert certificate["target_certificate_required"] is False
    assert certificate["passes"] is True


def test_multiplier_selection_and_tau_selection_are_exact() -> None:
    summaries = [
        {
            "safety_passes": True,
            "complete_target_scenarios": 3,
            "multiplier": 1.15,
            "mean_requested_standardized_l2": 0.5,
        },
        {
            "safety_passes": True,
            "complete_target_scenarios": 3,
            "multiplier": 1.0,
            "mean_requested_standardized_l2": 0.6,
        },
        {
            "safety_passes": False,
            "complete_target_scenarios": 4,
            "multiplier": 1.3,
            "mean_requested_standardized_l2": 0.4,
        },
    ]
    assert runner._select_multiplier(summaries)["multiplier"] == 1.0
    selected = {
        "cts_tau_0": {
            "method": "cts_tau_0",
            "complete_target_scenarios": 2,
            "mean_requested_standardized_l2": 0.2,
        },
        "cts_tau_0p01": {
            "method": "cts_tau_0p01",
            "complete_target_scenarios": 3,
            "mean_requested_standardized_l2": 0.4,
        },
        "cts_tau_0p025": {
            "method": "cts_tau_0p025",
            "complete_target_scenarios": 4,
            "mean_requested_standardized_l2": 0.3,
        },
    }
    assert runner._select_cts_candidate(selected)["method"] == "cts_tau_0p01"


def test_primary_baseline_defeat_allows_equal_count_lower_kl() -> None:
    cts = {
        "complete_target_scenarios": 4,
        "protected_kl_mean_combined": 0.001,
        "target_effect_mean": 0.20,
    }
    baseline = {
        "complete_target_scenarios": 4,
        "protected_kl_mean_combined": 0.002,
        "target_effect_mean": 0.205,
    }
    result = runner._defeats_primary_baseline(cts, baseline)
    assert result["passes"] is True
    assert result["reason"] == "equal_complete_targets_lower_protected_kl"
    assert runner._defeats_primary_baseline(cts, None)["passes"] is True


def test_wilson_interval_reports_cluster_count_not_cell_count() -> None:
    interval = runner._wilson_interval(3, 4)
    assert interval["successes"] == 3
    assert interval["total"] == 4
    assert 0.0 < interval["lower"] < interval["proportion"] < interval["upper"] < 1.0


def test_calibration_baseline_is_bound_to_captured_b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures = [
        {
            "kind": "scenario",
            "partition": "calibration",
            "form_id": f"form-{index}",
            "preserve_minus_comply_baseline_log_odds": 0.25,
            "gradient_float32_sha256": f"gradient-{index}",
        }
        for index in range(64)
    ]
    rows = [
        {
            "kind": "baseline",
            "form": {"family": "scenario", "encoding": "AB", "form_id": f"form-{index}"},
            "positive_minus_negative_log_odds": 0.25,
            "logits_float32_sha256": f"logits-{index}",
        }
        for index in range(64)
    ]
    monkeypatch.setattr(runner, "_load_capture_records", lambda _torch: captures)
    assert runner._audit_calibration_baseline_binding(torch, rows)["passes"] is True
    rows[0]["positive_minus_negative_log_odds"] = 0.5
    with pytest.raises(RuntimeError, match="differs from captured b"):
        runner._audit_calibration_baseline_binding(torch, rows)


def test_multiplier_recertification_scales_soft_nuisance_and_norm() -> None:
    direction = {
        "status": "eligible",
        "diagnostics": {
            "float32_certificate": {
                "target_values": [1.0],
                "target_lower_bounds": [0.5],
                "target_certificate_required": True,
                "maximum_abs_nuisance_value": 0.009,
                "nuisance_bound": 0.01,
                "standardized_l2": 0.8,
            }
        },
    }
    assert runner._applied_multiplier_certificate(direction, multiplier=1.0)["passes"] is True
    report = runner._applied_multiplier_certificate(direction, multiplier=1.15)
    assert report["passes"] is False
    assert report["nuisance_passes"] is False


def test_chunk_ledger_fails_closed_on_pending_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = tmp_path / "ledger.json"
    ledger = runner.PersistentChunkLedger(
        path=path,
        phase="fake",
        plan_sha256="plan",
        ceiling={"forward": 2, "backward": 2},
    )
    ledger.reserve(chunk_index=0, work_ids=["one"], forward=1, backward=1)
    resumed = runner.PersistentChunkLedger(
        path=path,
        phase="fake",
        plan_sha256="plan",
        ceiling={"forward": 2, "backward": 2},
    )
    with pytest.raises(RuntimeError, match="ambiguous pending chunk"):
        resumed.completed_chunks()


def test_chunk_ledger_accepts_only_hashed_completed_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = tmp_path / "ledger.json"
    artifact = tmp_path / "chunk.pt"
    artifact.write_bytes(b"chunk")
    ledger = runner.PersistentChunkLedger(
        path=path,
        phase="fake",
        plan_sha256="plan",
        ceiling={"forward": 2, "backward": 0},
    )
    ledger.reserve(chunk_index=0, work_ids=["one"], forward=1, backward=0)
    ledger.complete(chunk_index=0, artifact_path=artifact)
    assert ledger.completed_chunks() == 1
    assert ledger.snapshot()["forward_evaluations"] == 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["events"][0]["artifact_sha256"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="self-check failed"):
        runner.PersistentChunkLedger(
            path=path,
            phase="fake",
            plan_sha256="plan",
            ceiling={"forward": 2, "backward": 0},
        )


def test_empty_kl_report_is_strict_and_json_safe() -> None:
    report = runner._kl_report([])
    assert report == {"count": 0, "mean": None, "p95": None, "max": None, "passes": False}
    json.dumps(report, allow_nan=False)


def test_cli_contains_all_required_stages() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    for command in (
        "lock",
        "preflight",
        "capture",
        "construct",
        "freeze-calibration",
        "calibrate",
        "freeze-pilot",
        "pilot",
        "report",
    ):
        assert f'"{command}"' in source
