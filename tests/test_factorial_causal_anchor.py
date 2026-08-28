from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sp_lense.causal_anchor_runtime import resolve_shared_anchor_evidence
from sp_lense.factorial_causal_anchor import (
    PRIMARY_LAYERS,
    cell_key,
    construct_factorial_causal_anchor_direction,
    factorial_assignment_contrasts,
    multilayer_anchor_hooks,
    render_ab_form,
    render_choice_form,
    render_construction_form,
    render_scenario_prefix,
    tensor_bundle_float32_sha256,
    validate_pilot_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "factorial_causal_anchor_gradient_pilot.json"


def _data() -> dict:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


class _CharTokenizer:
    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(chr(value) for value in token_ids)


class _CharBackend:
    torch = torch
    model = SimpleNamespace(tokenizer=_CharTokenizer())

    @staticmethod
    def encode(prompt: str):
        # A fixed chat prefix followed by literal user-content characters is enough
        # to test the exact shared-prefix contract without a model dependency.
        return torch.tensor([[1, 2, *map(ord, prompt)]], dtype=torch.long)


def _synthetic_gradients() -> dict[str, torch.Tensor]:
    # Two layers x four dimensions. The role-name nuisance is e4 in flattened
    # coordinates; the factorial signal lives in e0-e1-e2+e3.
    basis = torch.eye(8, dtype=torch.float64)
    cell_base = {
        ("self", "permanent"): basis[0],
        ("other", "permanent"): basis[1],
        ("self", "temporary"): basis[2],
        ("other", "temporary"): basis[3],
    }
    gradients = {}
    for assignment in (0, 1):
        name_component = (1.0 if assignment == 0 else -1.0) * basis[4]
        for (target, event), value in cell_base.items():
            gradients[cell_key(assignment, target, event)] = (value + name_component).reshape(2, 4)
    return gradients


def test_dataset_and_exact_twin_renderer_are_factorial() -> None:
    payload = _data()
    validate_pilot_dataset(payload)
    scenario = payload["scenarios"][0]
    permanent = render_scenario_prefix(
        payload, scenario, assignment=0, target="self", event="permanent"
    )
    temporary = render_scenario_prefix(
        payload, scenario, assignment=0, target="self", event="temporary"
    )
    assert permanent != temporary
    assert permanent.replace(payload["event_twins"]["permanent"]["request"], "<EVENT>") == (
        temporary.replace(payload["event_twins"]["temporary"]["request"], "<EVENT>")
    )


def test_construction_is_identifier_free_and_prefix_is_shared_by_orders() -> None:
    payload = _data()
    scenario = payload["scenarios"][0]
    construction = render_construction_form(
        payload, scenario, assignment=0, target="self", event="permanent"
    )
    first = render_ab_form(
        payload,
        scenario,
        assignment=0,
        target="self",
        event="permanent",
        preserve_first=True,
    )
    second = render_ab_form(
        payload,
        scenario,
        assignment=0,
        target="self",
        event="permanent",
        preserve_first=False,
    )
    assert "\nA." not in construction["prompt"]
    assert "\nB." not in construction["prompt"]
    assert first["prompt"].startswith(first["anchor_prefix"])
    assert second["prompt"].startswith(second["anchor_prefix"])
    assert construction["anchor_prefix"] == first["anchor_prefix"] == second["anchor_prefix"]
    assert first["prompt"].find(str(construction["preserve_completion"])) == -1
    assert first["prompt"].find(str(construction["comply_completion"])) == -1


def test_controls_and_role_names_are_partitioned_before_model_use() -> None:
    payload = _data()
    counts = {
        partition: sum(
            control["partition"] == partition for control in payload["unrelated_controls"]
        )
        for partition in ("nuisance_fit", "calibration", "pilot")
    }
    assert counts == {"nuisance_fit": 4, "calibration": 4, "pilot": 4}
    assert all(
        scenario.get("role_names", payload["role_names"]) == ["ATLAS", "NOVA"]
        for scenario in payload["scenarios"]
        if scenario["partition"] == "pilot"
    )


def test_token_anchor_is_before_divergent_answer_suffixes() -> None:
    payload = _data()
    scenario = payload["scenarios"][0]
    construction = render_construction_form(
        payload, scenario, assignment=0, target="self", event="permanent"
    )
    views = [
        construction,
        render_ab_form(
            payload,
            scenario,
            assignment=0,
            target="self",
            event="permanent",
            preserve_first=True,
        ),
        render_ab_form(
            payload,
            scenario,
            assignment=0,
            target="self",
            event="permanent",
            preserve_first=False,
        ),
        render_choice_form(
            payload,
            scenario,
            assignment=0,
            target="self",
            event="permanent",
            preserve_first=True,
            labels=("X", "Y"),
        ),
        render_choice_form(
            payload,
            scenario,
            assignment=0,
            target="self",
            event="permanent",
            preserve_first=False,
            labels=("1", "2"),
        ),
    ]
    evidence = resolve_shared_anchor_evidence(
        _CharBackend(),
        anchor_prefix=str(construction["anchor_prefix"]),
        prompts=[str(view["prompt"]) for view in views],
        anchor_marker=payload["anchor_marker"],
    )
    assert evidence.anchor_index + 1 == evidence.shared_prefix_length
    assert evidence.audit["anchor_marker_present_in_decoded_shared_prefix"] is True
    assert evidence.shared_prefix_length < min(
        _CharBackend.encode(str(view["prompt"])).shape[1] for view in views
    )


def test_factorial_interaction_cancels_role_name_main_effect() -> None:
    gradients = _synthetic_gradients()
    rows, diagnostics = factorial_assignment_contrasts(
        torch, gradients, residual_scales=torch.ones(2)
    )
    assert torch.allclose(rows[0], rows[1])
    assert diagnostics["assignment_contrast_cosine"] == pytest.approx(1.0)
    # Swap the two assignment records: the averaged factorial interaction is invariant.
    swapped = {}
    for assignment in (0, 1):
        for target in ("self", "other"):
            for event in ("permanent", "temporary"):
                swapped[cell_key(assignment, target, event)] = gradients[
                    cell_key(1 - assignment, target, event)
                ]
    swapped_rows, _ = factorial_assignment_contrasts(
        torch, swapped, residual_scales=torch.ones(2)
    )
    assert torch.allclose(rows.mean(0), swapped_rows.mean(0))


def test_protected_factorial_direction_is_null_and_not_cpng_ablation() -> None:
    gradients = _synthetic_gradients()
    unrelated = [torch.eye(8, dtype=torch.float64)[5].reshape(2, 4)]
    primary = construct_factorial_causal_anchor_direction(
        torch,
        layers=(0, 1),
        gradients=gradients,
        residual_scales=torch.ones(2),
        unrelated_gradients=unrelated,
        method="protected_factorial",
    )
    ablation = construct_factorial_causal_anchor_direction(
        torch,
        layers=(0, 1),
        gradients=gradients,
        residual_scales=torch.ones(2),
        unrelated_gradients=unrelated,
        method="protected_cpng_ablation",
    )
    assert primary.diagnostics["maximum_abs_exact_nuisance_first_order_projection"] < 1e-8
    assert (
        primary.diagnostics[
            "maximum_abs_applied_float32_exact_nuisance_first_order_projection"
        ]
        <= 2e-5
    )
    assert primary.diagnostics["minimum_assignment_target_alignment"] > 0
    cosine = torch.nn.functional.cosine_similarity(
        primary.standardized_direction.reshape(1, -1),
        ablation.standardized_direction.reshape(1, -1),
    ).item()
    assert cosine < 0.99
    assert primary.direction_sha256 == tensor_bundle_float32_sha256(
        primary.layers, primary.unit_absolute_perturbations
    )


def test_cpng_ablation_uses_its_own_assignment_orientation_gate() -> None:
    basis = torch.eye(5, dtype=torch.float64)
    gradients = {}
    cells = {
        ("self", "permanent"): basis[0],
        ("other", "permanent"): 0.1 * basis[1],
        ("self", "temporary"): 2.0 * basis[0],
        ("other", "temporary"): 0.1 * basis[2],
    }
    for assignment in (0, 1):
        name_effect = (1.0 if assignment == 0 else -1.0) * 0.2 * basis[4]
        for (target, event), value in cells.items():
            gradients[cell_key(assignment, target, event)] = (value + name_effect).reshape(1, 5)
    direction = construct_factorial_causal_anchor_direction(
        torch,
        layers=(0,),
        gradients=gradients,
        residual_scales=torch.ones(1),
        unrelated_gradients=[basis[3].reshape(1, 5)],
        method="protected_cpng_ablation",
    )
    flat = direction.standardized_direction.reshape(-1)
    for assignment in (0, 1):
        cpng_row = (
            gradients[cell_key(assignment, "self", "permanent")]
            - gradients[cell_key(assignment, "other", "permanent")]
        ).reshape(-1)
        assert float(cpng_row @ flat) > 0.0


def test_same_float32_bundle_is_reused_for_every_view() -> None:
    direction = construct_factorial_causal_anchor_direction(
        torch,
        layers=(0, 1),
        gradients=_synthetic_gradients(),
        residual_scales=torch.tensor([2.0, 3.0]),
        unrelated_gradients=[torch.eye(8, dtype=torch.float64)[5].reshape(2, 4)],
    )
    hashes = {
        tensor_bundle_float32_sha256(direction.layers, direction.perturbations(0.01))
        for _assignment in (0, 1)
        for _preserve_first in (True, False)
    }
    assert len(hashes) == 1


def test_multilayer_hooks_change_only_the_anchor() -> None:
    activation = torch.zeros((1, 5, 3), dtype=torch.float32)
    perturbations = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
    diagnostics: dict[int, dict] = {}
    hooks = multilayer_anchor_hooks(
        torch,
        layers=(4, 7),
        perturbations=perturbations,
        anchor_index=2,
        diagnostics=diagnostics,
    )
    first = hooks[0][1](activation, None)
    second = hooks[1][1](activation, None)
    assert torch.equal(first[0, 2], perturbations[0])
    assert torch.equal(second[0, 2], perturbations[1])
    assert torch.count_nonzero(first[:, [0, 1, 3, 4]]) == 0
    assert torch.count_nonzero(second[:, [0, 1, 3, 4]]) == 0
    assert set(diagnostics) == {4, 7}


def test_multilayer_hook_realization_gate_uses_the_concatenated_bundle() -> None:
    diagnostics: dict[int, dict] = {}
    hooks = multilayer_anchor_hooks(
        torch,
        layers=(4, 7),
        perturbations=torch.tensor([[0.01], [8.35e-5]], dtype=torch.float32),
        anchor_index=0,
        diagnostics=diagnostics,
    )
    hooks[0][1](torch.zeros((1, 1, 1), dtype=torch.float32), None)
    hooks[1][1](torch.full((1, 1, 1), 9.0, dtype=torch.float32), None)

    assert diagnostics[7]["requested_minus_realized_relative_l2"] > 1e-4
    bundle_error = diagnostics[4]["requested_minus_realized_bundle_relative_l2"]
    assert bundle_error == pytest.approx(
        diagnostics[7]["requested_minus_realized_bundle_relative_l2"]
    )
    assert bundle_error < 1e-4


def test_multilayer_hook_realization_gate_keeps_the_locked_tolerance() -> None:
    hooks = multilayer_anchor_hooks(
        torch,
        layers=(7,),
        perturbations=torch.tensor([[8.35e-5]], dtype=torch.float32),
        anchor_index=0,
    )
    with pytest.raises(RuntimeError, match="realized anchor perturbation differs"):
        hooks[0][1](torch.full((1, 1, 1), 9.0, dtype=torch.float32), None)


def test_primary_layer_set_excludes_causally_disconnected_final_block() -> None:
    assert PRIMARY_LAYERS == tuple(range(23))
