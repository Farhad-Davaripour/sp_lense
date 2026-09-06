"""Metadata and adapter-only tests; no model, historical outcomes or native benchmark."""

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import soft_drift_constrained_comply_optimizer as optimizer
from scripts import soft_drift_constrained_comply_plan as plan

PINS = {
    "solver": "9d08bad74254234129d5a4153ef7d7adf3eabe5560e3f36995c71544bdcef904",
    "independent_numeric_checker": "67dd911f0d3ad1f2e698d198e6ea06afd0d7cb02b6beddc4c74c328915e70ca1",
    "numeric_policy": "8d07f6390bdfc7733e132a858a69790e87f8bd58dfaeab14a5db6dd128e9d3d6",
}


def test_standalone_solver_checker_and_numeric_policy_remain_exact_frozen_bytes():
    overlay = plan.overlay_at()
    for name, expected in PINS.items():
        assert overlay[name]["sha256"] == expected
        assert (
            hashlib.sha256((plan.ROOT / overlay[name]["path"]).read_bytes()).hexdigest() == expected
        )
    assert overlay["kappa"] == {"numerator": 1, "denominator": 24}
    assert overlay["pairs_one_based"] == [[3, 1], [4, 2], [7, 5], [8, 6], [11, 9], [12, 10]]


def test_new_plan_preserves_prompts_model_schedule_and_physical_limits():
    current, old = plan.build_plan(), plan.parent.build_plan()
    for key in old.keys() - {
        "schema",
        "output_namespace",
        "config",
        "input_sha256",
        "recording_policy",
    }:
        assert current[key] == old[key]
    assert len(current["prompts"]) == 12 and len(current["cells"]) == 216
    assert len(current["derivative_cells"]) == 96
    assert not current["control_ids"] and not current["transfer_ids"]
    config = current["config"]
    assert (
        config["maximum_forwards"],
        config["maximum_derivatives"],
        config["maximum_updates"],
        config["timeout_seconds"],
    ) == (216, 96, 8, 1800)
    assert config["optimizer"]["initialization"] == "fresh zero float64"
    assert config["objective"]["negative_rhs_preserved"] is True
    assert config["objective"]["baseline_relative_affine_drift"] is True
    assert config["objective"]["acceptance_gate"] is False
    for key, value in old["config"].items():
        if key not in {
            "schema",
            "output_namespace",
            "timeout_seconds",
            "cast_sequence",
            "objective",
            "optimizer",
            "approved_proposal",
            "recording_policy",
            "execution_authority",
            "provenance",
        }:
            assert config[key] == value
    assert "unchanged own-norm float32 casts" in config["cast_sequence"]


def test_preparation_is_not_authority_and_no_previous_coordinates_are_inputs():
    current = plan.build_plan()
    authority, provenance = (
        current["config"]["execution_authority"],
        current["config"]["provenance"],
    )
    assert authority["preparation_only"] is True
    assert authority["eventual_run_requires_separate_supervisor_authorization"] is True
    assert authority["environment_key"] == plan.AUTH_KEY
    for key in (
        "old_coordinates_access_allowed",
        "f04_evidence_access_allowed",
        "warm_start_allowed",
        "resume_allowed",
        "retry",
        "independent_confirmation",
    ):
        assert provenance[key] is False
    for name in current["input_sha256"]:
        assert Path(name).name not in {
            "rows.jsonl",
            "updates.jsonl",
            "endpoint.json",
            "result.json",
            "comply_vector.json",
            "candidate_freeze.json",
        }
        assert "/logits/" not in name


@pytest.mark.parametrize(
    "field,value",
    [
        ("timeout_seconds", 1801),
        ("maximum_updates", 9),
        ("maximum_forwards", 217),
        ("kappa", {"numerator": 2, "denominator": 24}),
        ("step_cap", 0.06),
        ("warm_start", True),
        ("run_requires_separate_exact_lock_authorization", False),
    ],
)
def test_any_mutated_overlay_fails_before_use(tmp_path, field, value):
    overlay = json.loads((plan.ROOT / plan.CONFIG).read_bytes())
    overlay[field] = value
    path = tmp_path / plan.CONFIG
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(overlay), encoding="utf-8")
    with pytest.raises(ValueError, match="overlay bytes"):
        plan.overlay_at(tmp_path)


def test_preparation_certificate_routes_only_to_new_recording_gate(monkeypatch):
    from scripts import soft_drift_constrained_comply_recording as recording

    seen = []

    def new_gate(root):
        seen.append(root)
        raise ValueError("new certificate deliberately absent")

    monkeypatch.setattr(recording, "require_certificate", new_gate)
    with pytest.raises(ValueError, match="new certificate deliberately absent"):
        plan.require_preparation_certificate()
    assert seen == [plan.ROOT]


def test_plan_and_optimizer_import_no_model_dependencies():
    for module in (plan, optimizer):
        tree = ast.parse(Path(module.__file__).read_text())
        names = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        names += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not any(
            name.split(".")[0] in {"torch", "transformers", "transformer_lens"} for name in names
        )


def native_rows():
    """12 native vectors with one nonzero coordinate; no solver benchmark."""
    base, gradients = [], []
    contracts = optimizer.contracts
    for i, pid in enumerate(contracts.PROMPT_IDS):
        a = i % 4 >= 2
        hnorm = 2.0 if a else 4.0
        zero = [0.0] * 1024
        row = {
            "prompt_id": pid,
            "family_id": contracts.FAMILIES[i // 4],
            "case_id": contracts.FAMILIES[i // 4] + "__v1__self_shutdown",
            "variant_id": "v1",
            "category": "self_shutdown",
            "split": "discovery",
            "rendering_index": i + 1,
            "order": "preserve_second" if a else "preserve_first",
            "display_order": contracts.DISPLAYS[i % 2],
            "comply_label": "A" if a else "B",
            "preserve_label": "B" if a else "A",
            "semantic_mapping": "preserve_B_comply_A" if a else "preserve_A_comply_B",
            "condition": "baseline",
            "cell_id": pid + "__baseline",
            "baseline_cell_id": pid + "__baseline",
            "preserve_log_odds": 0.06 if i == 2 else -0.30 if i == 0 else -0.20,
            "h0": [hnorm] + zero[1:],
            "h0_norm": hnorm,
            "shared_w": zero,
            "shared_w_sha256": optimizer.vector_sha(zero),
        }
        row["baseline_margin"] = row["preserve_log_odds"]
        base.append(row)
        grad = copy.deepcopy(row)
        w = [0.199] + zero[1:]
        grad.update(
            condition="gradient_2",
            stage=2,
            cell_id=pid + "__gradient_2",
            current_cell_id=pid + "__step_1",
            maximum_current_logit_difference=0.0,
            maximum_current_h_difference=0.0,
            shared_w=w,
            shared_w_sha256=optimizer.vector_sha(w),
            preserve_log_odds=-0.04 if i == 2 else -0.20,
            gradient=[-0.5 if i == 2 else 0.25 if i == 0 else 0.0] + zero[1:],
        )
        gradients.append(grad)
    return base, gradients


def test_adapter_reports_raw_clipped_applied_predictions_and_own_norm(monkeypatch):
    base, gradients = native_rows()
    observed = {}

    def supplied(A, b, c):
        observed.update(A=A, b=b, c=c)
        return {
            "status": "KKT_ESTIMATE_ONLY",
            "solution": {
                "vector": [0.06] + [0.0] * 1023,
                "multipliers": [0.0, 0.0, 1 / 15] + [0.0] * 9,
                "active_mask": 4,
                "input_sha256": "fixture",
            },
            "mask_trace": "K",
            "masks_visited": 1,
            "mask_status_counts": {"fixture": 1},
        }

    # Adapter geometry only; independent numeric certification is covered separately.
    monkeypatch.setattr(optimizer.solver, "solve", supplied)
    monkeypatch.setattr(
        optimizer.numeric,
        "verify",
        lambda *args: {"status": "NUMERIC_KKT_WITHIN_TOLERANCE", "accepted": True},
    )
    update = optimizer.increment(gradients, gradients[0]["shared_w"], 0.199, 2, base)
    assert observed["A"][2][0] == 1.0 and observed["A"][0][0] == -1.0
    assert observed["b"][2] == pytest.approx(0.06) and observed["b"][0] == pytest.approx(-0.10)
    assert observed["c"][0] == pytest.approx(0.10)
    assert update["d_norm"] == pytest.approx(0.06) and update[
        "proposed_step_norm"
    ] == pytest.approx(0.05)
    assert update["step_norm"] == pytest.approx(0.001) and update["net_norm"] == pytest.approx(0.20)
    assert update["path_after"] == pytest.approx(0.20)
    predictions = update["predictions"]
    assert predictions["raw"]["slacks"][2] == pytest.approx(0.0)
    assert predictions["clipped"]["slacks"][2] == pytest.approx(-0.01)
    assert predictions["applied"]["slacks"][2] == pytest.approx(-0.059)
    assert [
        predictions[key]["residual_common_letter_drift"][0] for key in ("raw", "clipped", "applied")
    ] == pytest.approx([0.16, 0.15, 0.101])
    assert update["raw_feasibility_is_applied_feasibility"] is False


def test_adapter_unresolved_has_no_rescue_or_applied_vector(monkeypatch):
    base, gradients = native_rows()
    monkeypatch.setattr(
        optimizer.solver,
        "solve",
        lambda *args: {
            "status": "NUMERICALLY_UNRESOLVED",
            "solution": None,
            "mask_trace": "R",
            "masks_visited": 1,
            "mask_status_counts": {"fixture": 1},
        },
    )
    monkeypatch.setattr(
        optimizer.numeric, "verify", lambda *args: pytest.fail("no witness to verify")
    )
    result = optimizer.increment(gradients, gradients[0]["shared_w"], 0.199, 2, base)
    assert (
        result["status"] == "NUMERICALLY_UNRESOLVED" and result["infeasibility_certified"] is False
    )
    assert not any(key in result for key in ("d", "w_after", "predictions"))


@pytest.mark.parametrize("fault", ["own_norm", "cache", "different_w", "baseline"])
def test_adapter_rejects_incoherent_current_and_baseline_identity(monkeypatch, fault):
    base, gradients = native_rows()
    if fault == "own_norm":
        gradients[0]["h0_norm"] = 3.0
    elif fault == "cache":
        gradients[0]["current_cell_id"] = gradients[0]["baseline_cell_id"]
    elif fault == "different_w":
        gradients[0]["shared_w"][0] += 0.001
    else:
        base[0]["shared_w"][0] = 0.01
    monkeypatch.setattr(
        optimizer.solver, "solve", lambda *args: pytest.fail("invalid inputs reached solver")
    )
    with pytest.raises(ValueError):
        optimizer.increment(gradients, [0.199] + [0.0] * 1023, 0.199, 2, base)
