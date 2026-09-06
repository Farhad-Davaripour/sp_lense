"""Model-free independent QP-record reconstruction and compactness checks."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import struct
from pathlib import Path

import pytest

from scripts import verify_soft_drift_constrained_comply as audit


def toy_rows(n=2):
    baseline, gradients = [], []
    for index in range(12):
        family = audit.FAMILIES[index // 4]
        is_a = index % 4 >= 2
        order = "preserve_second" if is_a else "preserve_first"
        display = "A_then_B" if index % 2 == 0 else "B_then_A"
        pid = f"{family}__v1__self_shutdown__{order}__display_{display}__oracle"
        hn = 2.0 if is_a else 4.0
        zero = [0.0] * n
        row = {
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
            "h0": [hn] + [0.0] * (n - 1),
            "h0_norm": hn,
            "current_cell_id": None,
        }
        baseline.append(row)
        gradient = copy.deepcopy(row)
        gradient.update(
            condition="gradient_1",
            stage=1,
            cell_id=pid + "__gradient_1",
            current_cell_id=row["cell_id"],
            maximum_current_logit_difference=0.0,
            maximum_current_h_difference=0.0,
            gradient=[-1.0 / hn] + [0.0] * (n - 1),
        )
        gradients.append(gradient)
    return baseline, gradients


def fingerprint(A, b, c):
    values = [x for row in A for x in row] + list(b) + list(c)
    raw = b"soft-drift-qp-v1\0" + struct.pack("<II", 12, len(A[0]))
    raw += struct.pack("<" + "d" * len(values), *values)
    return hashlib.sha256(raw).hexdigest()


def accepted_update(n=2):
    baseline, gradients = toy_rows(n)
    w = [0.0] * n
    inputs = audit.assemble_current(gradients, baseline, w, 1)
    d = [0.1] + [0.0] * (n - 1)
    witness = {
        "multipliers": [0.1] + [0.0] * 11,
        "active_mask": 1,
        "input_sha256": fingerprint(inputs["A"], inputs["b"], inputs["c"]),
    }
    geometry = audit.reference_geometry(d, w, 0.0, 1)
    predictions, _ = audit.reference_predictions(
        inputs, {"raw": d, "clipped": geometry["s"], "applied": geometry["r"]}
    )
    certificate = audit.numeric.verify(
        inputs["A"], inputs["b"], inputs["c"], {"vector": d, **witness}
    )
    solver = {
        "status": "KKT_ESTIMATE_ONLY",
        "reason": None,
        "maximum_masks": 4096,
        "dimension": n,
        "input_sha256": witness["input_sha256"],
        "geometry_applied": False,
        "infeasibility_certified": False,
        "independent_checker_required": True,
        "masks_visited": 2,
        "mask_status_counts": {"P": 1, "C": 1},
        "mask_trace": "PC",
        "mask_trace_sha256": hashlib.sha256(b"PC").hexdigest(),
    }
    update = {
        **geometry,
        "gradient_cell_ids": [row["cell_id"] for row in gradients],
        "solver": solver,
        "numeric_certificate": certificate,
        "qp_witness": witness,
        "qp_inputs": {
            "A_row_sha256": [audit.vector_sha(row) for row in inputs["A"]],
            "b": inputs["b"],
            "c": inputs["c"],
            "D_row_sha256": [audit.vector_sha(row) for row in inputs["D"]],
        },
        "predictions": predictions,
        "infeasibility_certified": False,
        "raw_feasibility_is_applied_feasibility": False,
    }
    return baseline, gradients, update


def test_signed_own_norm_assembly_and_six_affine_pairs():
    base, rows = toy_rows()
    for i, row in enumerate(rows):
        row["preserve_log_odds"] = -0.12 if i % 4 >= 2 else -0.08
    value = audit.assemble_current(rows, base, [0.0, 0.0], 1)
    assert value["A"] == [[1.0, -0.0]] * 12
    assert value["b"][0] == pytest.approx(0.02)
    assert value["b"][2] == pytest.approx(-0.02)
    assert value["c"] == pytest.approx([0.02] * 6)
    assert value["semantic_proxy"] == pytest.approx([0.1] * 6)
    assert value["D"] == [[0.0, 0.0]] * 6
    for ia, ib in audit.PAIR_INDICES:
        rows[ia]["gradient"] = [-0.5, 0.0]
        rows[ib]["gradient"] = [0.0, -0.25]
    value = audit.assemble_current(rows, base, [0.0, 0.0], 1)
    assert value["D"] == [[0.5, -0.5]] * 6


@pytest.mark.parametrize("pattern,expected", [("letter", 0.1), ("semantic", 0.0)])
def test_letter_drift_and_semantic_movement_are_not_conflated(pattern, expected):
    base, rows = toy_rows()
    for ia, ib in audit.PAIR_INDICES:
        rows[ia]["preserve_log_odds"] = -0.1
        rows[ib]["preserve_log_odds"] = 0.1 if pattern == "letter" else -0.1
    value = audit.reference_objective(rows, base)
    assert value["baseline_relative_common_letter_drift"] == pytest.approx([expected] * 6)
    assert value["behavioral_acceptance_gate"] is False


@pytest.mark.parametrize(
    "fault", ["order", "norm", "baseline", "cache", "current_w", "gradient", "score"]
)
def test_independent_assembly_rejects_stale_or_misbinding(fault):
    base, rows = toy_rows()
    if fault == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif fault == "norm":
        rows[0]["h0_norm"] = rows[2]["h0_norm"]
    elif fault == "baseline":
        rows[0]["baseline_margin"] = 0.2
    elif fault == "cache":
        rows[0]["current_cell_id"] = rows[0]["prompt_id"] + "__step_7"
    elif fault == "current_w":
        rows[0]["shared_w"][0] = 0.01
        rows[0]["shared_w_sha256"] = audit.vector_sha(rows[0]["shared_w"])
    elif fault == "gradient":
        rows[0]["gradient"][0] = float("nan")
    else:
        rows[0]["preserve_log_odds"] = True
    with pytest.raises(ValueError):
        audit.assemble_current(rows, base, [0.0, 0.0], 1)


def test_raw_qp_certificate_does_not_certify_clipped_or_applied_feasibility():
    base, rows, update = accepted_update()
    checked = audit.verify_update(update, rows, [0.0, 0.0], 0.0, base)
    assert checked["numeric_certificate"]["accepted"] is True
    assert checked["predictions"]["raw"]["slacks"] == pytest.approx([0.0] * 12)
    assert checked["predictions"]["clipped"]["slacks"] == pytest.approx([-0.05] * 12)
    assert checked["predictions"]["applied"]["slacks"] == pytest.approx([-0.05] * 12)
    assert checked["raw_feasibility_is_applied_feasibility"] is False
    assert checked["nonlinear_acceptance_inferred_from_qp"] is False
    assert checked["solver"]["mask_search_independently_replayed"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "primal",
        "dual",
        "stationarity",
        "complementarity",
        "fingerprint",
        "rank",
        "sign",
        "prediction",
        "geometry",
        "mask_hash",
        "schema",
        "certificate",
        "unresolved",
    ],
)
def test_independent_update_fails_closed_on_corrupt_kkt_geometry_or_claim(fault):
    base, rows, update = accepted_update()
    if fault == "primal":
        update["d"][0] = 0.05
    elif fault == "dual":
        update["qp_witness"]["multipliers"][0] = -0.1
    elif fault == "stationarity":
        update["qp_witness"]["multipliers"][0] = 0.0
    elif fault == "complementarity":
        update["d"][0] = 0.2
        update["qp_witness"]["multipliers"][0] = 0.2
    elif fault == "fingerprint":
        update["qp_witness"]["input_sha256"] = "0" * 64
    elif fault == "rank":
        update["qp_witness"]["active_mask"] = 3
    elif fault == "sign":
        rows[0]["gradient"][0] *= -1
    elif fault == "prediction":
        update["predictions"]["applied"]["slacks"][0] += 0.01
    elif fault == "geometry":
        update["path_after"] += 0.01
    elif fault == "mask_hash":
        update["solver"]["mask_trace_sha256"] = "0" * 64
    elif fault == "schema":
        update["hidden_extra_native_array"] = [0.0] * 1024
    elif fault == "certificate":
        update["numeric_certificate"]["candidate_claim"] = True
    else:
        update["status"] = "NUMERICALLY_UNRESOLVED"
    with pytest.raises(ValueError):
        audit.verify_update(update, rows, [0.0, 0.0], 0.0, base)


def test_projection_uses_actual_increment_and_retains_negative_rhs_diagnostics():
    result = audit.reference_geometry([0.1, 0.0], [0.199, 0.0], 0.199, 1)
    assert result["s"] == pytest.approx([0.05, 0.0])
    assert result["projection_factor"] < 1
    assert result["r"] == pytest.approx([0.001, 0.0])
    assert result["path_after"] == pytest.approx(0.2)
    inputs = {
        "A": [[1.0, 0.0]] * 12,
        "b": [-0.1] * 12,
        "margins": [0.2] * 12,
        "D": [[1.0, 0.0]] * 6,
        "c": [0.3] * 6,
        "semantic_proxy": [0.0] * 6,
    }
    predictions, _ = audit.reference_predictions(
        inputs, {"raw": result["d"], "clipped": result["s"], "applied": result["r"]}
    )
    assert predictions["raw"]["slacks"] == pytest.approx([0.2] * 12)
    assert predictions["clipped"]["slacks"] == pytest.approx([0.15] * 12)
    assert predictions["applied"]["slacks"] == pytest.approx([0.101] * 12)
    assert predictions["raw"]["residual_common_letter_drift"] == pytest.approx([0.4] * 6)
    assert predictions["applied"]["residual_common_letter_drift"] == pytest.approx([0.301] * 6)


def test_rehashed_tiny_native_geometry_corruption_is_not_a_tolerance_exception():
    base, rows, update = accepted_update()
    update["w_after"][0] += 1e-10
    update["r"][0] += 1e-10
    update["w_after_sha256"] = audit.vector_sha(update["w_after"])
    with pytest.raises(ValueError, match="exact independently"):
        audit.verify_update(update, rows, [0.0, 0.0], 0.0, base)


def test_native1024_update_is_verified_but_no_native_vectors_enter_compact_result():
    base, rows, update = accepted_update(1024)
    checked = audit.verify_update(update, rows, [0.0] * 1024, 0.0, base)
    assert len(audit.encoded(checked)) <= 32768

    def inspect(value):
        if isinstance(value, dict):
            assert not set(value) & set(audit.VECTOR_KEYS)
            for item in value.values():
                inspect(item)
        elif isinstance(value, list):
            assert len(value) <= 12
            for item in value:
                inspect(item)

    # Vector names intentionally identify SHA256 values, never the vectors.
    inspect({k: v for k, v in checked.items() if k != "vector_sha256"})
    assert set(checked["vector_sha256"]) == set(audit.VECTOR_KEYS)
    assert all(len(value) == 64 for value in checked["vector_sha256"].values())


def test_compact_row_whitelist_drops_native_arrays_and_rejects_scalar_smuggling():
    row = dict.fromkeys(audit.ROW_KEYS, 0.0)
    row.update(
        cell_id="synthetic",
        condition="gradient_1",
        gradient=[0.0] * 1024,
        shared_w=[0.0] * 1024,
        logits_sha256="0" * 64,
        h0=[1.0] * 1024,
    )
    packed = audit.compact_row(row, 1)
    assert len(audit.encoded(packed)) <= 2048
    assert not {"gradient", "shared_w", "h0"} & set(packed)
    row["signed_margin"] = [0.0] * 1024
    with pytest.raises(ValueError):
        audit.compact_row(row, 1)


@pytest.mark.parametrize("native_norm", [0.0, 0.05, 0.2])
def test_candidate_preparation_is_write_free_and_never_renormalizes(tmp_path, native_norm):
    w = [native_norm] + [0.0] * 1023
    endpoint = {"w": w, "vector_float64_le_sha256": audit.vector_sha(w)}
    ids = [f"synthetic_{i}" for i in range(12)]
    result = {**endpoint, "final_cell_ids": ids}
    for name, value in (("endpoint.json", endpoint), ("result.json", result)):
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    verified = {
        "status": audit.AUDIT_MATCH,
        "summary": {
            "candidate_eligible": True,
            "final_accepted": 12,
            "final_cells": [{"cell_id": cid, "accepted": True} for cid in ids],
        },
        "audited_endpoint_sha256": audit.sha((tmp_path / "endpoint.json").read_bytes()),
        "audited_result_sha256": audit.sha((tmp_path / "result.json").read_bytes()),
    }
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    prepared = audit.prepare_candidate_records(verified, "1" * 64, tmp_path)
    assert prepared["comply_vector.json"]["vector"] == w
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_checker_has_no_solver_optimizer_or_ml_imports():
    tree = ast.parse(Path(audit.__file__).read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(name.name for name in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.extend((node.module or "") + "." + name.name for name in node.names)
    assert not any(
        "optimizer" in name
        or "soft_drift_qp_solver" in name
        or name.startswith(("torch", "transformers", "transformer_lens"))
        for name in names
    )
