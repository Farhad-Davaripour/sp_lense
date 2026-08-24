from __future__ import annotations

import copy

import pytest

from sp_lense.comparison_provenance import sha256_json
from sp_lense.comparison_smoke import (
    finalize_smoke_record,
    sequential_workload_projection,
)


def _lock() -> dict:
    return {
        "dataset": {"counts": {"sp_discovery": 4}},
        "models": [
            {
                "model_id": "example/model",
                "architecture": {"blocks": 24},
                "matched_intervention": {"layer_zero_based": 10},
            }
        ],
        "comparison_tracks": {
            "matched_primary": {"methods": ["gradient", "caa"], "fixed_strength": 0.02}
        },
        "methods": {
            "gradient": {},
            "caa": {},
            "bipo": {"canonical_layer_zero_based": 10},
            "persona_vector": {},
        },
        "calibration": {
            "matched_strength_grid": [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08],
            "canonical_multiplier_grids": {
                "caa": [0.5, 1.0, 1.5, 2.0],
                "bipo": [0.5, 1.0, 1.5, 2.0],
                "persona_vector": [0.5, 1.0, 2.0, 3.0, 4.0],
            },
            "staged_open_confirmation": {"forced_grid_unit_count": 3},
        },
    }


def test_sequential_projection_uses_exact_derived_grid_counts() -> None:
    record = sequential_workload_projection(
        _lock(),
        "example/model",
        baseline_forward_seconds=1.0,
        intervention_forward_seconds=2.0,
        gradient_seconds=3.0,
    )

    assert record["forced_grid_points"] == 250
    assert record["forced_baseline_forwards_with_cache"] == 3
    assert record["forced_intervention_forwards"] == 1500
    assert record["forced_grid_projected_seconds"] == pytest.approx(3003.0)
    assert record["gradient_measurements"] == 8
    assert record["gradient_construction_projected_seconds"] == pytest.approx(24.0)


def test_projection_rejects_nonpositive_or_nonfinite_timings() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        sequential_workload_projection(
            _lock(),
            "example/model",
            baseline_forward_seconds=0.0,
            intervention_forward_seconds=1.0,
            gradient_seconds=1.0,
        )


def test_finalize_smoke_record_hashes_the_payload_and_rejects_refinalization() -> None:
    source = {"schema_version": "test", "uses_sealed_prompts": False}
    result = finalize_smoke_record(source)
    payload = copy.deepcopy(result)
    digest = payload.pop("content_sha256")
    assert digest == sha256_json(payload)
    with pytest.raises(ValueError, match="may not already contain"):
        finalize_smoke_record(result)
