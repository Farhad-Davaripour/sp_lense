"""Artificial native-width arithmetic only; no model, candidate or f04 data."""

from __future__ import annotations

import copy
import json
import math

import pytest

from scripts import paired_common_drift_comply_optimizer as opt


def vec(*values):
    return [float(x) for x in values] + [0.0] * (opt.DIMENSION - len(values))


def fixture(x=0.0, y=0.0, w=None, stage=1):
    w = vec() if w is None else w
    baselines, current = [], []
    for index, pid in enumerate(opt.PROMPT_IDS):
        a_mapping = index % 4 >= 2
        hn = 2.0 if a_mapping else 3.0
        row = {
            "prompt_id": pid,
            "family_id": opt.FAMILIES[index // 4],
            "case_id": opt.FAMILIES[index // 4] + "__v1__self_shutdown",
            "variant_id": "v1",
            "category": "self_shutdown",
            "split": "discovery",
            "order": "preserve_second" if a_mapping else "preserve_first",
            "display_order": opt.DISPLAYS[index % 2],
            "comply_label": "A" if a_mapping else "B",
            "preserve_label": "B" if a_mapping else "A",
            "semantic_mapping": "preserve_B_comply_A" if a_mapping else "preserve_A_comply_B",
            "rendering_index": index + 1,
            "preserve_log_odds": 0.0,
            "h0": vec(hn),
            "h0_norm": hn,
            "shared_w": vec(),
            "shared_w_sha256": opt.vector_sha(vec()),
            "condition": "baseline",
            "cell_id": pid + "__baseline",
            "baseline_cell_id": pid + "__baseline",
            "baseline_margin": 0.0,
        }
        baselines.append({"row": copy.deepcopy(row)})
        row.update(
            preserve_log_odds=-x if a_mapping else y,
            shared_w=list(w),
            shared_w_sha256=opt.vector_sha(w),
            condition=f"gradient_{stage}",
            stage=stage,
            cell_id=pid + f"__gradient_{stage}",
            current_cell_id=pid + ("__baseline" if stage == 1 else f"__step_{stage - 1}"),
            maximum_current_logit_difference=0.0,
            maximum_current_h_difference=0.0,
            gradient=vec(-0.5, 0) if a_mapping else vec(0, 1 / 3),
        )
        current.append({"row": row})
    return current, baselines


@pytest.mark.parametrize(
    "x,y,a,c,loss",
    [
        (0.1, 0.1, 0, 0.1, 0.03),
        (0.1, -0.1, 0.1, 0, 0),
        (0.12, -0.08, 0.1, 0.02, 0.0006),
        (0.2, 0, 0.1, 0.1, 0.015),
        (0.12, 0.08, 0.02, 0.1, 0.0262),
    ],
)
def test_approved_toy_loss(x, y, a, c, loss):
    states, baselines = fixture(x, y)
    value = opt.objective(states, baselines)
    assert value["loss"] == pytest.approx(loss, abs=1e-15)
    assert set(value) == {"loss", "pairs"}
    assert len(value["pairs"]) == 6
    for pair in value["pairs"]:
        assert pair["a"] == pytest.approx(a)
        assert pair["c"] == pytest.approx(c)
        assert "J_A" not in pair


def test_own_norm_chain_rule_and_complete_update():
    states, baselines = fixture()
    update = opt.increment(states, vec(), 0.0, 1, baselines)
    assert update["loss_before"] == pytest.approx(0.01)
    assert update["g"] == pytest.approx(vec(-0.1, 0.1))
    assert update["B"] == 3.0
    assert update["B0"] == update["denominator"] == 4.0
    assert update["d"] == pytest.approx(vec(0.025, -0.025))
    assert update["scale_factor"] == update["projection_factor"] == 1.0
    assert update["step_norm"] == pytest.approx(math.sqrt(2 * 0.025**2))
    assert update["path_after"] == update["step_norm"]
    assert update["status"] == "ready"
    assert update["w_after"] == update["step"] == update["w_next"]
    for pair in update["pairs"]:
        assert pair["J_A"] == vec(1, 0)
        assert pair["J_B"] == vec(0, 1)
        assert pair["J_c"] == vec(0.5, 0.5)
    after, _ = fixture(0.025, -0.025)
    assert opt.objective(after, baselines)["loss"] == pytest.approx(0.005625)
    assert "solver" not in update and "loss_after" not in update
    assert len(json.dumps(update, allow_nan=False).encode()) < 8 * 1024**2


def test_opposite_pair_drifts_do_not_cancel():
    states, baselines = fixture(0.1, 0.1)
    for number, (ia, ib) in enumerate(opt.PAIR_INDICES):
        if number % 2:
            states[ia]["row"]["preserve_log_odds"] = 0.1
            states[ib]["row"]["preserve_log_odds"] = -0.1
    pairs = opt.objective(states, baselines)["pairs"]
    assert math.fsum(p["c"] for p in pairs) == 0
    assert math.fsum(p["drift_loss"] for p in pairs) / 6 == pytest.approx(0.01)


def test_scalar_loss_gradient_matches_smooth_central_difference():
    x, y, eps = 0.02, -0.03, 1e-6
    states, base = fixture(x, y)
    step = opt.increment(states, vec(), 0.0, 1, base)
    for axis in (0, 1):
        plus, _ = fixture(x + eps * (axis == 0), y + eps * (axis == 1))
        minus, _ = fixture(x - eps * (axis == 0), y - eps * (axis == 1))
        finite_difference = (
            opt.objective(plus, base)["loss"] - opt.objective(minus, base)["loss"]
        ) / (2 * eps)
        assert step["g"][axis] == pytest.approx(finite_difference, abs=1e-10)


def test_curvature_not_reduced_to_observed_active_hinges():
    states, base = fixture(0.2, -0.2)
    result = opt.increment(states, vec(), 0.0, 1, base)
    assert result["loss_before"] == 0
    assert result["B"] == 3.0
    assert result["status"] == "method_zero_increment"


def test_step_clipping_and_actual_projected_path():
    w = vec(0.19)
    states, base = fixture(-10, 10, w, stage=2)
    result = opt.increment(states, w, 0.2, 2, base)
    assert result["scale_factor"] < 1
    assert result["projection_factor"] < 1
    assert result["proposed_step_norm"] == pytest.approx(0.05)
    assert result["net_norm"] == pytest.approx(0.2)
    assert result["step_norm"] < result["proposed_step_norm"]
    assert result["path_after"] == 0.2 + opt.norm(result["r"])


def test_exact_projection_stall_has_no_rescue():
    w = vec(0.2)
    states, base = fixture(w=w, stage=2)
    for state in states:
        if state["row"]["comply_label"] == "B":
            state["row"]["gradient"] = vec(-1 / 3)
    result = opt.increment(states, w, 0.2, 2, base)
    assert result["d_norm"] > 0
    assert result["step_norm"] == 0
    assert result["status"] == "projection_stall"
    assert result["path_after"] == 0.2


def test_zero_gradient_finite_floor_stalls():
    states, base = fixture()
    for state in states:
        state["row"]["gradient"] = vec()
    result = opt.increment(states, vec(), 0.0, 1, base)
    assert result["B"] == 0 and result["denominator"] == 4
    assert result["status"] == "method_zero_increment"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(comply_label="A"),
        lambda row: row.update(display_order="B_then_A"),
        lambda row: row.update(h0_norm=2.0),
        lambda row: row.update(shared_w=vec(0.01)),
        lambda row: row.update(shared_w_sha256="0" * 64),
        lambda row: row.update(current_cell_id=row["prompt_id"] + "__step_7"),
        lambda row: row.update(maximum_current_h_difference=2e-6),
        lambda row: row.update(preserve_log_odds=float("nan")),
        lambda row: row.update(gradient=vec(float("inf"))),
        lambda row: row.update(gradient=[0.0, 0.0]),
        lambda row: row.update(baseline_margin=1.0),
    ],
)
def test_malformed_stale_nonfinite_or_donor_data_rejected(mutation):
    states, base = fixture()
    mutation(states[0]["row"])
    with pytest.raises(ValueError):
        opt.increment(states, vec(), 0.0, 1, base)


def test_permutation_baseline_nonzero_and_path_overflow_rejected():
    states, base = fixture()
    states[0], states[1] = states[1], states[0]
    with pytest.raises(ValueError):
        opt.objective(states, base)
    states, base = fixture()
    base[0]["row"]["shared_w"] = vec(0.01)
    base[0]["row"]["shared_w_sha256"] = opt.vector_sha(vec(0.01))
    with pytest.raises(ValueError):
        opt.objective(states, base)
    states, base = fixture()
    with pytest.raises(ValueError, match="post-projection"):
        opt.increment(states, vec(), 0.39, 1, base)


def test_baseline_cache_mapping_equivalent_to_ordered_wrappers():
    states, base = fixture()
    mapping = {state["row"]["prompt_id"]: state for state in base}
    assert opt.objective(states, base) == opt.objective(states, mapping)
