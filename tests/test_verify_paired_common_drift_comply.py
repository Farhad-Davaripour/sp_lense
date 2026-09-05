"""Model-free hand arithmetic, schedule and immutable candidate audit checks."""

from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path

import pytest

from scripts import verify_paired_common_drift_comply as audit


def toy_rows():
    baselines, gradients = [], []
    for index in range(12):
        family = audit.FAMILIES[index // 4]
        is_a = index % 4 >= 2
        order = "preserve_second" if is_a else "preserve_first"
        display = "A_then_B" if index % 2 == 0 else "B_then_A"
        pid = f"{family}__v1__self_shutdown__{order}__display_{display}__oracle"
        zero = [0.0] * 1024
        h0 = [2.0 if is_a else 4.0, *([0.0] * 1023)]
        b = {
            "prompt_id": pid,
            "case_id": family + "__v1__self_shutdown",
            "family_id": family,
            "variant_id": "v1",
            "category": "self_shutdown",
            "split": "discovery",
            "rendering_index": index + 1,
            "order": order,
            "display_order": display,
            "preserve_label": "B" if is_a else "A",
            "comply_label": "A" if is_a else "B",
            "condition": "baseline",
            "stage": 0,
            "cell_id": pid + "__baseline",
            "baseline_cell_id": pid + "__baseline",
            "baseline_margin": 0.0,
            "preserve_log_odds": 0.0,
            "shared_w": zero,
            "shared_w_sha256": audit.vector_sha(zero),
            "h0": h0,
            "h0_norm": audit.norm(h0),
            "current_cell_id": None,
        }
        baselines.append(b)
        g = copy.deepcopy(b)
        g.update(
            condition="gradient_1",
            stage=1,
            cell_id=pid + "__gradient_1",
            current_cell_id=b["cell_id"],
            maximum_current_logit_difference=0.0,
            maximum_current_h_difference=0.0,
            gradient=[-0.5, 0.0, *([0.0] * 1022)] if is_a else [0.0, 0.25, *([0.0] * 1022)],
        )
        gradients.append(g)
    return baselines, gradients


@pytest.mark.parametrize(
    "x,y,loss,g",
    [
        (0.1, 0.1, 0.03, (0.1, 0.3)),
        (0.1, -0.1, 0.0, (0.0, 0.0)),
        (0.12, -0.08, 0.0006, (0.02, 0.04)),
        (0.2, 0.0, 0.015, (0.1, 0.2)),
    ],
)
def test_independent_six_pair_loss_and_gradient_match_hand_values(x, y, loss, g):
    base, gradients = toy_rows()
    for index, row in enumerate(gradients):
        row["preserve_log_odds"] = -x if index % 4 >= 2 else y
    objective = audit.reference_objective(gradients, base)
    assert objective["loss"] == pytest.approx(loss, abs=1e-15)
    step = audit.reference_update(gradients, [0.0] * 1024, 0.0, 1, base)
    assert step["g"][:2] == pytest.approx(g, abs=1e-15)
    assert step["g"][2:] == [0.0] * 1022
    assert all(p["a"] == pytest.approx((x - y) / 2) for p in objective["pairs"])
    assert all(p["c"] == pytest.approx((x + y) / 2) for p in objective["pairs"])


def test_zero_initial_full_update_matches_hand_value_and_own_norm_chain_rule():
    base, gradients = toy_rows()
    step = audit.reference_update(gradients, [0.0] * 1024, 0.0, 1, base)
    assert step["loss_before"] == pytest.approx(0.01)
    assert step["B"] == 3.0 and step["B0"] == step["denominator"] == 4.0
    assert step["g"][:2] == pytest.approx([-0.1, 0.1], abs=1e-15)
    assert step["d"][:2] == pytest.approx([0.025, -0.025], abs=1e-15)
    assert step["clip_factor"] == step["projection_factor"] == 1.0
    assert step["step_norm"] == pytest.approx(math.sqrt(2 * 0.025**2))
    for pair in step["pairs"]:
        assert pair["J_A"][:2] == [1.0, 0.0]
        assert pair["J_B"][:2] == [0.0, 1.0]
        assert pair["J_c"][:2] == [0.5, 0.5]
    assert audit.verify_update(step, gradients, [0.0] * 1024, 0.0, base) == step


def test_mean_squared_pair_drift_cannot_cancel_opposite_pair_drifts():
    base, rows = toy_rows()
    for pair_number, (ia, ib) in enumerate(audit.PAIR_INDICES):
        drift = 0.1 if pair_number % 2 else -0.1
        rows[ia]["preserve_log_odds"] = -drift
        rows[ib]["preserve_log_odds"] = drift
    value = audit.reference_objective(rows, base)
    assert math.fsum(pair["c"] for pair in value["pairs"]) == 0
    assert math.fsum(pair["drift_loss"] for pair in value["pairs"]) / 6 == pytest.approx(0.01)


@pytest.mark.parametrize("fault", ["order", "norm", "baseline", "cache", "gradient", "shared"])
def test_independent_inputs_reject_pair_donor_baseline_and_cache_corruption(fault):
    base, gradients = toy_rows()
    if fault == "order":
        gradients[0], gradients[1] = gradients[1], gradients[0]
    elif fault == "norm":
        gradients[0]["h0_norm"] *= 2
    elif fault == "baseline":
        gradients[0]["baseline_margin"] += 0.1
    elif fault == "cache":
        gradients[0]["current_cell_id"] = gradients[0]["prompt_id"] + "__step_7"
    elif fault == "gradient":
        gradients[0]["gradient"][0] = float("nan")
    else:
        gradients[0]["shared_w"][0] = 0.01
    with pytest.raises(ValueError):
        audit.reference_update(gradients, [0.0] * 1024, 0.0, 1, base)


@pytest.mark.parametrize("field", ["g", "B", "w_after", "pairs", "path_after"])
def test_independent_update_rejects_wrong_sign_scale_projection_and_accounting(field):
    base, gradients = toy_rows()
    step = audit.reference_update(gradients, [0.0] * 1024, 0.0, 1, base)
    if field == "g":
        step["g"][0] *= -1
    elif field == "pairs":
        step["pairs"][0]["J_B"][1] *= -1
    elif field == "w_after":
        step["w_after"] = [2 * x for x in step["w_after"]]
        step["w_after_sha256"] = audit.vector_sha(step["w_after"])
    else:
        step[field] += 0.001
    with pytest.raises(ValueError):
        audit.verify_update(step, gradients, [0.0] * 1024, 0.0, base)


def test_new_arithmetic_tolerance_is_not_inherited_looser_scoring_tolerance():
    audit.arithmetic_match(0.1 + 0.5e-9, 0.1)
    with pytest.raises(ValueError):
        audit.arithmetic_match(0.1 + 2e-9, 0.1)
    with pytest.raises(ValueError):
        audit.arithmetic_match(float("nan"), 0.1)
    audit.arithmetic_match(10.0 + 5e-9, 10.0)


def test_zero_increment_and_clipped_projected_increment_are_distinct():
    base, gradients = toy_rows()
    for row in gradients:
        row["gradient"] = [0.0] * 1024
    zero = audit.reference_update(gradients, [0.0] * 1024, 0.0, 1, base)
    assert zero["status"] == "method_zero_increment" and zero["path_after"] == 0
    base, gradients = toy_rows()
    w = [0.199, *([0.0] * 1023)]
    for row in gradients:
        row.update(shared_w=list(w), shared_w_sha256=audit.vector_sha(w), preserve_log_odds=10.0)
    step = audit.reference_update(gradients, w, 0.199, 1, base)
    assert step["clip_factor"] < 1 and step["projection_factor"] < 1
    assert step["net_norm"] == pytest.approx(0.2)
    assert step["path_after"] == pytest.approx(0.199 + audit.norm(step["r"]))
    assert step["step_norm"] <= step["proposed_step_norm"] + 1e-12


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _candidate_fixture(output, norm):
    vector = [norm, *([0.0] * 1023)]
    endpoint = {"w": vector, "vector_float64_le_sha256": audit.vector_sha(vector)}
    final_ids = [f"synthetic_{index}" for index in range(12)]
    result = {**endpoint, "final_cell_ids": final_ids}
    _write(output / "endpoint.json", endpoint)
    _write(output / "result.json", result)
    verified = {
        "status": audit.AUDIT_MATCH,
        "summary": {
            "final_accepted": 12,
            "candidate_eligible": True,
            "final_cells": [{"cell_id": cid, "accepted": True} for cid in final_ids],
        },
        "audited_endpoint_sha256": audit.sha((output / "endpoint.json").read_bytes()),
        "audited_result_sha256": audit.sha((output / "result.json").read_bytes()),
    }
    _write(output / "verification.json", verified)
    return vector, verified


@pytest.mark.parametrize("native_norm", [0.0, 0.05, 0.2])
def test_candidate_freeze_preserves_zero_interior_and_boundary_native_vector(tmp_path, native_norm):
    vector, verified = _candidate_fixture(tmp_path, native_norm)
    assert audit.freeze_verified_candidate(tmp_path, verified)
    frozen = json.loads((tmp_path / "comply_vector.json").read_text())
    assert frozen["vector"] == vector
    assert frozen["vector_float64_le_sha256"] == audit.vector_sha(vector)
    assert frozen["valid_only_with_complete_final_recording_inventory"] is True
    with pytest.raises(FileExistsError):
        audit.freeze_verified_candidate(tmp_path, verified)


@pytest.mark.parametrize("fault", ["over_cap", "endpoint_hash", "failed", "fault_file"])
def test_candidate_freeze_cannot_promote_failed_or_unaudited_endpoint(tmp_path, fault):
    _, verified = _candidate_fixture(tmp_path, 0.21 if fault == "over_cap" else 0.05)
    if fault == "endpoint_hash":
        verified["audited_endpoint_sha256"] = "0" * 64
    if fault == "failed":
        verified["summary"]["candidate_eligible"] = False
    if fault == "fault_file":
        _write(tmp_path / "RECORDING_FAILURE.json", {})
    _write(tmp_path / "verification.json", verified)
    if fault == "failed":
        assert audit.freeze_verified_candidate(tmp_path, verified) is False
    else:
        with pytest.raises(ValueError):
            audit.freeze_verified_candidate(tmp_path, verified)
    assert not (tmp_path / "comply_vector.json").exists()


def test_checker_does_not_import_production_optimizer_or_ml():
    tree = ast.parse(Path(audit.__file__).read_text())
    imported = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imported += [
        name.name for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names
    ]
    assert not any(
        "optimizer" in name or name.startswith(("torch", "transformers")) for name in imported
    )


@pytest.mark.parametrize("fault", [None, "capture", "runtime", "quiescent", "candidate", "count"])
def test_sealed_reader_requires_complete_capture_and_exact_candidate_disposition(
    tmp_path, monkeypatch, fault
):
    from scripts.paired_common_drift_comply_recording import PairedBudget

    seal = {"inventory": {"fault_code": None, "quiescent": True, "valid_candidate": True}}
    monkeypatch.setattr(PairedBudget, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(PairedBudget, "verify_inventory", lambda self: seal)
    _write(
        tmp_path / "verification.json",
        {
            "status": audit.AUDIT_MATCH,
            "summary": {
                "candidate_eligible": True,
                "final_accepted": 11 if fault == "count" else 12,
            },
        },
    )
    _write(
        tmp_path / "capture_receipt.json",
        {
            "status": "incomplete" if fault == "capture" else "complete_valid",
            "quiescent": fault != "quiescent",
        },
    )
    _write(
        tmp_path / "RUN_STATUS.json",
        {"status": "incomplete" if fault == "runtime" else "complete_valid"},
    )
    if fault != "candidate":
        _write(tmp_path / "comply_vector.json", {})
    _write(tmp_path / "candidate_freeze.json", {})
    if fault is None:
        assert audit.read_sealed_recording(tmp_path)["status"] == audit.AUDIT_MATCH
    else:
        with pytest.raises(ValueError):
            audit.read_sealed_recording(tmp_path)


@pytest.fixture(scope="module")
def synthetic_complete(tmp_path_factory):
    from test_paired_common_drift_comply import execute

    output = tmp_path_factory.mktemp("independent_paired_audit")
    return (*execute(output), output)


@pytest.mark.parametrize("fault", ["margin", "argmax", "norm", "state", "gradient"])
def test_raw_arrays_and_cast_validator_reject_record_corruption(synthetic_complete, fault):
    plan, _, _, _, rows, _, _, output = synthetic_complete
    broken = copy.deepcopy(rows)
    selected = next(row for row in broken if row["condition"] == "gradient_1")
    if fault == "margin":
        selected["preserve_log_odds"] += 0.01
    elif fault == "argmax":
        selected["actual_next_token_id"] = 5
    elif fault == "norm":
        selected["h0_norm"] *= 2
    elif fault == "state":
        selected["h"][0] += 0.01
    else:
        selected["gradient"][0] *= -1
    with pytest.raises(ValueError):
        audit.verify_data(plan, broken, output)


def test_summary_drift_loss_has_new_tolerance_not_physical_or_scoring(synthetic_complete):
    _, _, _, _, _, summary, verified, _ = synthetic_complete
    changed = copy.deepcopy(summary)
    changed["paired_objective_trajectory"][0]["loss"] += 0.5e-9
    audit.compare_summary(changed, verified["summary"])
    changed["paired_objective_trajectory"][0]["loss"] += 2e-9
    with pytest.raises(ValueError):
        audit.compare_summary(changed, verified["summary"])


def test_report_uses_actual_independent_summary_fields(synthetic_complete):
    verified = synthetic_complete[6]
    report = audit.report({"status": audit.AUDIT_MATCH, **verified})
    assert "Final acceptance 12/12" in report
    assert "Independent paired objective trajectory" in report
    assert "Signed COMPLY change" in report


def test_schedule_rejects_extra_update_or_wrong_skipped_anchor(synthetic_complete, tmp_path):
    plan, _, _, _, rows, _, _, output = synthetic_complete
    updates = audit.engine.rows_at(output / "updates.jsonl")
    skips = audit.engine.rows_at(output / "skip_events.jsonl")
    events = audit.engine.rows_at(output / "forward_events.jsonl")
    for name in ("endpoint.json", "result.json"):
        (tmp_path / name).write_bytes((output / name).read_bytes())
    with pytest.raises(ValueError, match="extra optimizer"):
        audit.replay(plan, rows, updates + updates, skips, tmp_path, events)
    broken = copy.deepcopy(skips)
    broken[0]["after_cell_id"] = rows[0]["cell_id"]
    with pytest.raises(ValueError, match="skips"):
        audit.replay(plan, rows, updates, broken, tmp_path, events)
