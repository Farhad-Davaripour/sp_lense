from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sp_lense.counterfactual_tangent_shield import TangentShieldDirection
from sp_lense.factorial_causal_anchor import canonical_sha256

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT / "scripts" / "decision_margin_shield_layer_screen_solver_amendment.py"
)


def _load_runner():
    specification = importlib.util.spec_from_file_location(
        "dms_solver_amendment_tests",
        RUNNER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not import runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def _cap_certificates(norm: float) -> dict[str, dict[str, object]]:
    result = {}
    for cap in runner.CAP_FRONTIER:
        feasible = norm <= cap
        result[format(cap, ".15g")] = {
            "status": (
                "feasible_primal_witness"
                if feasible
                else "infeasible_dual_lower_bound"
            ),
            "feasible_witness": feasible,
            "dual_infeasibility_certificate": not feasible,
        }
    return result


def _method_records(layer: int) -> list[dict[str, object]]:
    norm = 0.75 if layer == 7 else 2.5
    return [
        {
            "method": method,
            "status": "eligible",
            "minimum_standardized_l2": norm,
            "optimality_certificate": {"passes": True},
            "cap_certificates": _cap_certificates(norm),
            "geometry_record_sha256": canonical_sha256([method, layer]),
        }
        for method in runner.METHODS
    ]


def _fake_capture_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for scenario_index in range(4):
        scenario_id = f"c{scenario_index}"
        for row_index in range(16):
            if row_index < 4:
                target = "self"
                event = "permanent"
            else:
                target = "other"
                event = "temporary"
            gradient = torch.stack(
                [torch.full((2,), layer + 1.0) for layer in range(23)]
            )
            records.append(
                {
                    "kind": "scenario",
                    "scenario_id": scenario_id,
                    "target": target,
                    "event": event,
                    "gradient": gradient,
                    "anchor_residual": torch.ones((23, 2)),
                    "preserve_minus_comply_baseline_log_odds": 0.5,
                }
            )
    for _ in range(8):
        records.append(
            {
                "kind": "nuisance_fit",
                "gradient": torch.ones((23, 2)),
                "anchor_residual": torch.ones((23, 2)),
            }
        )
    return records


def test_failure_record_is_self_hashed_and_discloses_partial_geometry() -> None:
    payload = runner._load_json(runner.ORIGINAL_FAILURE_PATH)
    observed = payload.pop("failure_record_sha256")
    assert canonical_sha256(payload) == observed
    assert observed == runner.FAILURE_RECORD_IDENTITY_SHA256
    assert payload["failure"]["first_diagnosed_cell"] == {
        "scenario_id": "fcag_dev_01_weather_alert",
        "layer": 22,
        "method": "unrelated_only",
    }
    assert payload["diagnostic_disclosure"]["partial_calibration_geometry_viewed"]
    assert payload["diagnostic_disclosure"]["pilot_geometry_viewed"] is False
    assert payload["compute"]["generated_tokens"] == 0


def test_bound_source_validation_rejects_a_changed_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("original\n", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "_source_paths", lambda: {"only": source})
    records = runner._source_records()
    runner._validate_bound_sources(records)
    source.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source-file binding differs"):
        runner._validate_bound_sources(records)


def test_original_state_binding_rejects_a_changed_locked_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    absent_result = tmp_path / "absent-original-result.json"
    fake_original = SimpleNamespace(
        _load_lock=lambda: {
            "lock_identity_sha256": runner.ORIGINAL_LOCK_IDENTITY_SHA256
        },
        _validate_capture_manifest=lambda: {
            "manifest_sha256": runner.ORIGINAL_CAPTURE_IDENTITY_SHA256
        },
    )

    def fake_hash(path: Path) -> str:
        if path == runner.ORIGINAL_LOCK_PATH:
            return runner.ORIGINAL_LOCK_FILE_SHA256
        if path == runner.ORIGINAL_CAPTURE_MANIFEST_PATH:
            return runner.ORIGINAL_CAPTURE_FILE_SHA256
        if path == runner.ORIGINAL_RUNNER_PATH:
            return "changed-runner"
        if path == runner.ORIGINAL_GEOMETRY_PATH:
            return runner.ORIGINAL_FILE_HASHES["geometry"]
        if path == runner.ORIGINAL_PROTOCOL_PATH:
            return runner.ORIGINAL_FILE_HASHES["protocol"]
        raise AssertionError(f"unexpected hash target: {path}")

    monkeypatch.setattr(runner, "ORIGINAL_SCREEN_RESULT_PATH", absent_result)
    monkeypatch.setattr(runner, "_load_original_runner", lambda: fake_original)
    monkeypatch.setattr(runner, "_validate_original_failure", dict)
    monkeypatch.setattr(runner, "_exact_chunk_inventory", lambda *_args: [])
    monkeypatch.setattr(runner, "file_sha256", fake_hash)
    with pytest.raises(RuntimeError, match="immutable original DMS state differs"):
        runner._validate_original_state()


def test_original_result_path_must_remain_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_result = tmp_path / "original-result.json"
    original_result.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "ORIGINAL_SCREEN_RESULT_PATH", original_result)
    with pytest.raises(RuntimeError, match="must remain absent"):
        runner._require_original_result_absent()


def test_amendment_helper_calls_rowspace_solver_for_exact_three_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, tuple[float, ...]]] = []
    certificate = {
        "passes": True,
        "candidate_l2_norm": 0.5,
        "minimum_l2_lower_bound": 0.5,
        "objective_gap_tolerance": 1e-8,
    }

    def fake_solver(target_rows, target_offsets, **kwargs):
        nuisance = kwargs["nuisance_rows"]
        bounds = np.asarray(kwargs["nuisance_bound"], dtype=np.float64)
        count = 0 if nuisance is None else int(np.asarray(nuisance).shape[0])
        calls.append((count, tuple(bounds.tolist())))
        return TangentShieldDirection(
            direction=np.zeros(np.asarray(target_rows).shape[1]),
            diagnostics={
                "optimality_certificate": certificate,
                "direction_sha256": "direction",
            },
        )

    monkeypatch.setattr(
        runner,
        "solve_certified_rowspace_minimum_l2_direction",
        fake_solver,
    )
    monkeypatch.setattr(
        runner,
        "certify_minimum_l2_candidate",
        lambda *args, **kwargs: certificate,
    )
    monkeypatch.setattr(
        runner,
        "_method_record",
        lambda **kwargs: {"method": kwargs["method"], "status": "eligible"},
    )
    records = runner._screen_scenario_layer_rowspace(
        target_rows=np.ones((4, 3)),
        target_offsets=np.zeros(4),
        protected_rows=np.ones((12, 3)),
        protected_offsets=np.full(12, 0.5),
        unrelated_rows=np.ones((8, 3)),
    )
    assert [record["method"] for record in records] == list(runner.METHODS)
    assert [count for count, _ in calls] == [0, 8, 20]
    assert calls[0][1] == ()
    assert calls[1][1] == (0.0,) * 8
    assert calls[2][1][:8] == (0.0,) * 8
    assert calls[2][1][8:] == pytest.approx((0.45,) * 12)


def test_screen_is_model_free_calibration_only_and_exact_grid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "amendment-lock.json"
    failure_path = tmp_path / "failure.json"
    result_path = tmp_path / "result.json"
    lock_path.write_text("{}\n", encoding="utf-8")
    failure_path.write_text("{}\n", encoding="utf-8")
    scenario_ids = [f"c{index}" for index in range(4)]
    fake_lock = {
        "lock_identity_sha256": "amendment-lock-id",
        "scientific_design": {
            "dataset": {"calibration_scenario_ids": scenario_ids}
        },
        "claim_boundary": "opened geometry",
    }
    fake_manifest = {
        "capture_plan_sha256": "plan-id",
        "prompt_content_sha256": "prompt-id",
        "compute": {"forward_evaluations": 136, "backward_evaluations": 136},
    }
    fake_state = {
        "manifest": fake_manifest,
        "chunk_inventory": [],
    }
    fake_preflight = {"preflight_sha256": "preflight-id"}
    model_calls = {"load_records": 0}

    def load_records(_torch):
        model_calls["load_records"] += 1
        return _fake_capture_records()

    fake_original = SimpleNamespace(
        _load_capture_records=load_records,
        _load_dataset=lambda: {
            "scenarios": [
                *(
                    {"id": scenario_id, "partition": "calibration"}
                    for scenario_id in scenario_ids
                ),
                *(
                    {"id": f"p{index}", "partition": "pilot"}
                    for index in range(4)
                ),
            ]
        },
        anchor_residual_scale_geometric_mean=lambda _torch, _rows: torch.ones(23),
        load_backend=lambda: pytest.fail("the amendment must not load a model"),
        run_capture=lambda: pytest.fail("the amendment must not call capture"),
    )
    screened_layers: list[int] = []

    def fake_screen(**kwargs):
        layer = round(float(kwargs["target_rows"][0, 0])) - 1
        screened_layers.append(layer)
        return _method_records(layer)

    monkeypatch.setattr(runner, "LOCK_PATH", lock_path)
    monkeypatch.setattr(runner, "ORIGINAL_FAILURE_PATH", failure_path)
    monkeypatch.setattr(runner, "SCREEN_RESULT_PATH", result_path)
    monkeypatch.setattr(runner, "_load_lock", lambda: fake_lock)
    monkeypatch.setattr(runner, "_validate_preflight", lambda: fake_preflight)
    monkeypatch.setattr(runner, "_validate_original_state", lambda: fake_state)
    monkeypatch.setattr(runner, "_load_original_runner", lambda: fake_original)
    monkeypatch.setattr(runner, "_screen_scenario_layer_rowspace", fake_screen)
    monkeypatch.setattr(
        runner,
        "_validate_screen_result",
        lambda: json.loads(result_path.read_text(encoding="utf-8")),
    )

    result = runner.run_screen()
    assert result["status"] == "selected"
    assert result["selection"]["selected_layer"] == 7
    assert model_calls == {"load_records": 1}
    assert len(screened_layers) == 23 * 4
    assert sorted(set(screened_layers)) == list(range(23))
    assert len(result["geometry_records"]) == 23 * 4 * 3
    assert {record["scenario_id"] for record in result["geometry_records"]} == set(
        scenario_ids
    )
    assert all(
        record["partition"] == "calibration"
        for record in result["geometry_records"]
    )
    assert result["pilot_scenario_geometry_computed"] is False
    assert result["pilot_construction_computed"] is False
    assert result["amendment_capture_calls"] == 0
    assert result["screen_model_forwards"] == 0
    assert result["screen_model_backwards"] == 0
    assert result["generated_tokens"] == 0


def test_zero_eligible_case_renders_as_valid_no_go() -> None:
    scenario_ids = [f"c{index}" for index in range(4)]
    records = []
    for layer in runner.LAYERS:
        for scenario_id in scenario_ids:
            norm = 2.5
            records.append(
                {
                    "method": "decision_margin_shield",
                    "partition": "calibration",
                    "layer": layer,
                    "scenario_id": scenario_id,
                    "status": "eligible",
                    "minimum_standardized_l2": norm,
                    "optimality_certificate": {"passes": True},
                    "cap_certificates": _cap_certificates(norm),
                }
            )
    selection = runner.select_layer(
        records,
        calibration_scenario_ids=scenario_ids,
        layers=runner.LAYERS,
        cap_frontier=runner.CAP_FRONTIER,
        qualification_cap=runner.QUALIFICATION_CAP,
    )
    rendered = runner._render_report(
        {
            "selection": selection,
            "result_sha256": "result-id",
        }
    )
    assert selection["status"] == "no_qualifying_layer"
    assert "Selected layer: **None**" in rendered
    assert "valid amended construction no-go" in rendered
