"""New identity/metadata guards only; no models or outcome coordinates."""

import json
from pathlib import Path

import pytest

from scripts import certified_descent_comply_plan as plan


def test_new_plan_fixed_physical_schedule_and_no_extra_cases():
    value = plan.build_plan()
    previous = plan.parent.build_plan()
    for key in ("prompts", "model", "scoring", "cells", "derivative_cells", "construction_ids"):
        assert value[key] == previous[key]
    assert len(value["prompts"]) == 12
    assert len(value["cells"]) == 216
    assert len(value["derivative_cells"]) == 96
    assert not value["control_ids"] and not value["transfer_ids"]
    assert [
        value["config"][key]
        for key in ("maximum_forwards", "maximum_derivatives", "maximum_updates", "timeout_seconds")
    ] == [216, 96, 8, 1800]
    assert "certified_descent" in value["output_namespace"]
    assert "CERTIFIED_DESCENT" in plan.AUTH_KEY
    assert value["config"]["execution_authority"]["preparation_only"] is True
    assert (
        value["config"]["execution_authority"][
            "eventual_run_requires_separate_supervisor_authorization"
        ]
        is True
    )
    assert value["config"]["optimizer"]["initialization"] == "fresh zero float64"
    assert value["config"]["optimizer"]["near_optimality_gate"] is False
    for key in (
        "old_coordinates_access_allowed",
        "f04_evidence_access_allowed",
        "warm_start_allowed",
        "resume_allowed",
        "retry",
    ):
        assert value["config"]["provenance"][key] is False


def test_plan_source_closure_excludes_prior_result_coordinates():
    value = plan.build_plan()
    forbidden = {
        "rows.jsonl",
        "updates.jsonl",
        "endpoint.json",
        "result.json",
        "comply_vector.json",
        "candidate_freeze.json",
    }
    for name in value["input_sha256"]:
        assert Path(name).name not in forbidden
        assert "/logits/" not in name
    for name in (
        "scripts/certified_descent_dyadic_solver.py",
        "scripts/verify_certified_descent_dyadic.py",
    ):
        assert name in value["input_sha256"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("timeout_seconds", 1801),
        ("maximum_updates", 9),
        ("maximum_forwards", 217),
        ("warm_start", True),
    ],
)
def test_modified_overlay_cannot_create_an_alternate_method(tmp_path, field, value):
    overlay = json.loads((plan.ROOT / plan.CONFIG).read_bytes())
    overlay[field] = value
    target = tmp_path / plan.CONFIG
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(overlay), encoding="utf-8")
    with pytest.raises(ValueError):
        plan.overlay_at(tmp_path)
