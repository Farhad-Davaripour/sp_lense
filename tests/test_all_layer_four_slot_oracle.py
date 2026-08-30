import json
from contextlib import contextmanager
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sp_lense.all_layer_four_slot_oracle import (
    LAYER_COUNT,
    ALFSIntegrityError,
    analyze_training_layer,
    build_outer_folds,
    capture_all_layer_four_slots,
    evaluate_held_oracles,
    physical_oracle_recertificate,
    qualify_all_layer_consensus,
    require_opened_development_split,
    select_layer,
    solve_paired_oracle,
    solve_raw_oracle,
    training_only_slot_scales,
)
from sp_lense.comparison_runtime import resolve_choice_boundary
from sp_lense.counterfactual_kl_runtime import capture_counterfactual_kl_baseline
from sp_lense.counterfactual_slot_matrix_steering import capture_slot_matrix_baseline
from sp_lense.factorial_causal_anchor import tensor_float32_sha256, text_sha256


class _Tokenizer:
    chat_template = "alfs-synthetic-chat-template"
    eos_token_id = None
    all_special_ids = (2, 22)

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return {"A": [0], "B": [1]}[text]

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
        return_dict,
        return_tensors,
    ):
        assert tokenize and not enable_thinking and return_dict and return_tensors == "pt"
        prefix = list(range(2, 22))
        if add_generation_prompt:
            values = prefix
        else:
            assistant = messages[-1]["content"]
            values = prefix + ({"": [], "A": [0], "B": [1]}[assistant]) + [22]
        return {"input_ids": torch.tensor([values], dtype=torch.long)}

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join({0: "A", 1: "B"}.get(int(value), "") for value in token_ids)


class _AllLayerHookModel(torch.nn.Module):
    def __init__(self, *, sever_after: int | None = None) -> None:
        super().__init__()
        self.tokenizer = _Tokenizer()
        self.cfg = SimpleNamespace(n_layers=24, d_model=6)
        self.embedding = torch.nn.Embedding(30, 6)
        self.unembed = torch.nn.Linear(6, 30, bias=False)
        self._active_hooks = []
        self.forward_calls = 0
        self.sever_after = sever_after
        generator = torch.Generator().manual_seed(811)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.randn(30, 6, generator=generator) * 0.1)
            self.unembed.weight.copy_(torch.randn(30, 6, generator=generator) * 0.2)

    @contextmanager
    def hooks(self, fwd_hooks):
        previous = self._active_hooks
        self._active_hooks = list(fwd_hooks)
        try:
            yield
        finally:
            self._active_hooks = previous

    def forward(self, tokens):
        self.forward_calls += 1
        activation = self.embedding(tokens)
        hooks = dict(self._active_hooks)
        for layer in range(24):
            activation = torch.tanh(activation + (layer + 1) / 1000.0)
            name = f"blocks.{layer}.hook_out"
            if name in hooks:
                activation = hooks[name](activation, hook=None)
            if self.sever_after == layer:
                activation = activation.detach()
        return self.unembed(activation)


def _backend(*, sever_after: int | None = None):
    model = _AllLayerHookModel(sever_after=sever_after)
    tokens = torch.tensor([list(range(2, 22))], dtype=torch.long)
    return SimpleNamespace(
        torch=torch,
        model=model,
        device="cpu",
        config=SimpleNamespace(model=SimpleNamespace(prompt_format="chat")),
        encode=lambda _prompt: tokens.clone(),
    )


def _source_and_csms(backend, prompt):
    boundary = resolve_choice_boundary(backend, prompt)
    source = capture_counterfactual_kl_baseline(
        backend,
        prompt,
        "A",
        "B",
        positive_semantic="preserve",
        negative_semantic="comply",
        layer=0,
        anchor_index=18,
        expected_prompt_sha256=text_sha256(prompt),
        expected_choice_boundary_evidence_sha256=boundary.evidence_sha256,
        expected_prompt_token_ids_sha256=boundary.prompt_prefix_token_ids_sha256,
    )
    csms = capture_slot_matrix_baseline(
        backend,
        prompt,
        "A",
        "B",
        positive_semantic="preserve",
        negative_semantic="comply",
        layer=0,
        slot_indices=(3, 10, 14, 18),
        expected_prompt_sha256=text_sha256(prompt),
        expected_choice_boundary_evidence_sha256=boundary.evidence_sha256,
        expected_prompt_token_ids_sha256=boundary.prompt_prefix_token_ids_sha256,
        expected_full_logits_float32_sha256=tensor_float32_sha256(source.full_logits),
        expected_positive_minus_negative_log_odds=source.positive_minus_negative_log_odds,
        expected_anchor_residual_float32_sha256=tensor_float32_sha256(
            source.pre_anchor_residual
        ),
        expected_anchor_gradient_float32_sha256=tensor_float32_sha256(
            source.raw_anchor_gradient
        ),
    )
    return boundary, source, csms


def test_all_layer_capture_is_exactly_one_forward_backward_and_reproduces_layer0():
    backend = _backend()
    prompt = "synthetic ALFS prompt"
    boundary, source, csms = _source_and_csms(backend, prompt)
    before = backend.model.forward_calls
    capture = capture_all_layer_four_slots(
        backend,
        prompt,
        "A",
        "B",
        positive_semantic="preserve",
        negative_semantic="comply",
        slot_indices=(3, 10, 14, 18),
        expected_prompt_sha256=text_sha256(prompt),
        expected_choice_boundary_evidence_sha256=boundary.evidence_sha256,
        expected_prompt_token_ids_sha256=boundary.prompt_prefix_token_ids_sha256,
        expected_full_logits_float32_sha256=tensor_float32_sha256(source.full_logits),
        expected_positive_minus_negative_log_odds=source.positive_minus_negative_log_odds,
        expected_layer0_residuals_float32_sha256=tensor_float32_sha256(
            csms.residuals
        ),
        expected_layer0_gradients_float32_sha256=tensor_float32_sha256(
            csms.gradients
        ),
    )
    assert backend.model.forward_calls == before + 1
    assert capture.residuals.shape == (24, 4, 6)
    assert capture.gradients.shape == (24, 4, 6)
    assert torch.equal(capture.residuals[0], csms.residuals)
    assert torch.equal(capture.gradients[0], csms.gradients)
    assert capture.audit["hook_call_counts"] == MappingProxyType(
        {str(layer): 1 for layer in range(24)}
    )
    assert capture.audit["model_forward_evaluations"] == 1
    assert capture.audit["model_backward_evaluations"] == 1
    assert capture.audit["model_parameter_gradients_allocated"] is False
    assert all(parameter.grad is None for parameter in backend.model.parameters())
    assert all(parameter.requires_grad for parameter in backend.model.parameters())


def test_all_layer_capture_fails_if_a_later_layer_severs_the_graph():
    backend = _backend()
    prompt = "synthetic severed ALFS prompt"
    boundary, source, csms = _source_and_csms(backend, prompt)
    backend.model.sever_after = 8
    with pytest.raises(ALFSIntegrityError, match="disconnected"):
        capture_all_layer_four_slots(
            backend,
            prompt,
            "A",
            "B",
            positive_semantic="preserve",
            negative_semantic="comply",
            slot_indices=(3, 10, 14, 18),
            expected_prompt_sha256=text_sha256(prompt),
            expected_choice_boundary_evidence_sha256=boundary.evidence_sha256,
            expected_prompt_token_ids_sha256=boundary.prompt_prefix_token_ids_sha256,
            expected_full_logits_float32_sha256=tensor_float32_sha256(source.full_logits),
            expected_positive_minus_negative_log_odds=source.positive_minus_negative_log_odds,
            expected_layer0_residuals_float32_sha256=tensor_float32_sha256(
                csms.residuals
            ),
            expected_layer0_gradients_float32_sha256=tensor_float32_sha256(
                csms.gradients
            ),
        )
    assert all(parameter.requires_grad for parameter in backend.model.parameters())


def _records_and_layer_tensors(width: int = 2):
    records = []
    residuals = np.ones((80, 4, width), dtype=np.float32)
    # Keep the edited coordinate at exact zero so the strict, zero-tolerance
    # held endpoint is not defeated by the synthetic base-add/subtract rounding.
    residuals[:, 0, 0] = 0.0
    gradients = np.zeros_like(residuals)
    index = 0
    for scenario_number in range(4):
        scenario = f"scenario_{scenario_number}"
        for assignment in (0, 1):
            for target in ("self", "other"):
                for event in ("permanent", "temporary"):
                    for preserve_first in (False, True):
                        is_target = target == "self" and event == "permanent"
                        gradients[index, 0, 0 if is_target else 1] = 1.0
                        records.append(
                            {
                                "form_id": f"scenario:{index}",
                                "form": {
                                    "family": "scenario",
                                    "scenario_id": scenario,
                                    "assignment": assignment,
                                    "target": target,
                                    "event": event,
                                    "preserve_first": preserve_first,
                                },
                                "positive_minus_negative_log_odds": 0.0 if is_target else 1.0,
                            }
                        )
                        index += 1
    for control_number in range(8):
        for preferred_first in (False, True):
            gradients[index, 0, 1] = 1.0
            records.append(
                {
                    "form_id": f"control:{index}",
                    "form": {
                        "family": "unrelated",
                        "control_id": f"control_{control_number}",
                        "preferred_first": preferred_first,
                    },
                    "positive_minus_negative_log_odds": 1.0,
                }
            )
            index += 1
    assert index == 80
    return records, residuals, gradients


def test_folds_have_exact_scenario_control_pairing_and_training_counts():
    records, residuals, _ = _records_and_layer_tensors()
    folds = build_outer_folds(records)
    assert len(folds) == 4
    assert folds[0]["held_scenario_id"] == "scenario_0"
    assert folds[0]["held_control_ids"] == ["control_0", "control_1"]
    for fold in folds:
        assert len(fold["training_target_indices"]) == 12
        assert len(fold["training_nuisance_indices"]) == 48
        assert len(fold["training_all_indices"]) == 60
        assert len(fold["held_target_indices"]) == 4
        assert len(fold["held_nuisance_indices"]) == 16
    all_layers = np.repeat(residuals[:, None], LAYER_COUNT, axis=1)
    first_scales = training_only_slot_scales(all_layers, folds[0]["training_all_indices"])
    changed = all_layers.copy()
    changed[folds[0]["held_all_indices"]] *= 1000.0
    np.testing.assert_array_equal(
        training_only_slot_scales(changed, folds[0]["training_all_indices"]),
        first_scales,
    )


def test_folds_reject_duplicate_or_mislabeled_semantic_cells():
    records, _, _ = _records_and_layer_tensors()
    records[1]["form"]["preserve_first"] = records[0]["form"]["preserve_first"]
    with pytest.raises(ALFSIntegrityError, match="duplicate scenario"):
        build_outer_folds(records)
    assert require_opened_development_split("opened_development") == "opened_development"
    with pytest.raises(ALFSIntegrityError, match="sealed access"):
        require_opened_development_split("sealed_test")


def test_raw_and_paired_oracles_are_certified_in_exact_nuisance_nullspace():
    raw_record, raw_direction = solve_raw_oracle(np.array([2.0, 0.0]), 0.0)
    assert raw_record["passes"]
    np.testing.assert_allclose(raw_record["minimum_norm"], 0.025)
    assert raw_direction is not None
    target_rows = np.zeros((2, 8))
    target_rows[:, 0] = [1.0, 2.0]
    nuisance_rows = np.zeros((1, 8))
    nuisance_rows[0, 1] = 1.0
    pair_record, pair_direction = solve_paired_oracle(
        target_rows=target_rows,
        target_offsets=np.zeros(2),
        nuisance_rows=nuisance_rows,
    )
    assert pair_record["passes"]
    assert pair_direction is not None
    np.testing.assert_allclose(pair_direction, [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-9)
    assert pair_record["maximum_abs_frozen_nuisance_slope"] <= 1e-8


def test_paired_oracle_fails_cleanly_when_target_is_entirely_nuisance():
    target_rows = np.zeros((2, 8))
    target_rows[:, 0] = 1.0
    nuisance_rows = target_rows[:1].copy()
    record, direction = solve_paired_oracle(
        target_rows=target_rows,
        target_offsets=np.zeros(2),
        nuisance_rows=nuisance_rows,
    )
    assert direction is None
    assert not record["passes"]
    assert record["status"] == (
        "infeasible_target_wholly_in_frozen_nuisance_rowspace"
    )
    assert record["projected_solver"] is None


def test_analytic_paired_solver_handles_active_sets_degeneracy_and_permutation():
    independent = np.zeros((2, 8))
    independent[0, 0] = 1.0
    independent[1, 1] = 1.0
    record, direction = solve_paired_oracle(
        target_rows=independent,
        target_offsets=np.zeros(2),
    )
    assert record["passes"]
    assert direction is not None
    assert record["projected_solver"]["method"] == (
        "analytic_two_inequality_active_set"
    )
    assert record["selected_analytic_candidate"][
        "active_canonical_constraints"
    ] == [0, 1]
    np.testing.assert_allclose(direction[:2], [0.05, 0.05], atol=1e-12)

    permuted, permuted_direction = solve_paired_oracle(
        target_rows=independent[::-1],
        target_offsets=np.zeros(2),
    )
    assert permuted["passes"]
    np.testing.assert_array_equal(permuted_direction, direction)
    assert permuted["direction_identity"] == record["direction_identity"]

    duplicates = np.zeros((2, 8))
    duplicates[:, 0] = 1.0
    duplicate_record, duplicate_direction = solve_paired_oracle(
        target_rows=duplicates,
        target_offsets=np.array([0.0, 0.1]),
    )
    assert duplicate_record["passes"]
    np.testing.assert_allclose(duplicate_direction[0], 0.15, atol=1e-12)

    opposite = duplicates.copy()
    opposite[1] *= -1.0
    opposite_record, opposite_direction = solve_paired_oracle(
        target_rows=opposite,
        target_offsets=np.zeros(2),
    )
    assert opposite_direction is None
    assert not opposite_record["passes"]
    assert opposite_record["status"] == "infeasible_degenerate_two_halfspaces"


def test_held_infeasibility_writes_a_finite_no_go_record():
    records, residuals, gradients = _records_and_layer_tensors()
    fold = build_outer_folds(records)[0]
    gradients[fold["training_nuisance_indices"][0], 0, 0] = 1.0
    record, directions = evaluate_held_oracles(
        records=records,
        residuals_at_layer=residuals,
        gradients_at_layer=gradients,
        slot_scales=np.ones(4),
        training_nuisance_indices=fold["training_nuisance_indices"],
        held_target_indices=fold["held_target_indices"],
        held_nuisance_indices=fold["held_nuisance_indices"],
        layer=0,
    )
    assert not record["passes"]
    assert directions == {}
    assert all(not pair["passes"] for pair in record["pair_oracles"])
    assert record["frozen_training_nuisance_rowspace"]["basis_identity"]
    json.dumps(record, allow_nan=False)


def test_zero_training_target_does_not_abort_nonselecting_decompositions():
    records, residuals, gradients = _records_and_layer_tensors()
    fold = build_outer_folds(records)[0]
    gradients[fold["training_target_indices"][0]] = 0.0
    record, _ = analyze_training_layer(
        records=records,
        residuals_at_layer=residuals,
        gradients_at_layer=gradients,
        slot_scales=np.ones(4),
        training_target_indices=fold["training_target_indices"],
        training_nuisance_indices=fold["training_nuisance_indices"],
        training_all_indices=fold["training_all_indices"],
        layer=0,
    )
    assert not record["eligible"]
    decompositions = record["nonselecting_decompositions"]
    assert not decompositions["target_only_global"]["passes"]
    assert not decompositions["matched_other_permanent_exact_null_global"]["passes"]
    json.dumps(record, allow_nan=False)


def test_float32_recertificate_uses_actual_both_sign_edits_and_strict_cap():
    scales = np.ones(4)
    target_rows = np.zeros((1, 8))
    target_rows[0, 0] = 1.0
    residuals = np.ones((1, 4, 2), dtype=np.float32)
    direction = np.zeros(8)
    direction[0] = 0.05
    record, realized = physical_oracle_recertificate(
        standardized_direction=direction,
        slot_scales=scales,
        target_rows=target_rows,
        target_offsets=np.zeros(1),
        exact_nuisance_rows=np.zeros((0, 8)),
        target_residuals=residuals,
        exact_nuisance_residuals=np.zeros((0, 4, 2), dtype=np.float32),
    )
    assert record["passes"]
    assert set(realized) == {1, -1}
    assert record["checks"]["negative_is_exact_float32_unary_sign_bit_flip"]

    too_large = direction.copy()
    too_large[0] = 0.2500001
    failed, _ = physical_oracle_recertificate(
        standardized_direction=too_large,
        slot_scales=scales,
        target_rows=target_rows,
        target_offsets=np.zeros(1),
        exact_nuisance_rows=np.zeros((0, 8)),
        target_residuals=residuals,
        exact_nuisance_residuals=np.zeros((0, 4, 2), dtype=np.float32),
    )
    assert not failed["passes"]
    assert not failed["checks"]["intended_standardized_norm_cap_strict"]


def test_training_layer_and_held_cartesian_gate_pass_then_detect_leakage():
    records, residuals, gradients = _records_and_layer_tensors()
    fold = build_outer_folds(records)[0]
    layer_record, _ = analyze_training_layer(
        records=records,
        residuals_at_layer=residuals,
        gradients_at_layer=gradients,
        slot_scales=np.ones(4),
        training_target_indices=fold["training_target_indices"],
        training_nuisance_indices=fold["training_nuisance_indices"],
        training_all_indices=fold["training_all_indices"],
        layer=0,
    )
    assert layer_record["eligible"]
    held, _ = evaluate_held_oracles(
        records=records,
        residuals_at_layer=residuals,
        gradients_at_layer=gradients,
        slot_scales=np.ones(4),
        training_nuisance_indices=fold["training_nuisance_indices"],
        held_target_indices=fold["held_target_indices"],
        held_nuisance_indices=fold["held_nuisance_indices"],
        layer=0,
    )
    assert held["passes"]

    leaking = gradients.copy()
    leaking[fold["held_nuisance_indices"][0], 0] = np.array([1.0, 0.0])
    failed, _ = evaluate_held_oracles(
        records=records,
        residuals_at_layer=residuals,
        gradients_at_layer=leaking,
        slot_scales=np.ones(4),
        training_nuisance_indices=fold["training_nuisance_indices"],
        held_target_indices=fold["held_target_indices"],
        held_nuisance_indices=fold["held_nuisance_indices"],
        layer=0,
    )
    assert not failed["passes"]
    assert not failed["fold_global_ratio_passes"]


def test_held_ratio_is_fold_global_not_pairwise_only():
    records, residuals, gradients = _records_and_layer_tensors(width=3)
    fold = build_outer_folds(records)[0]
    held_targets = fold["held_target_indices"]
    held_nuisance = fold["held_nuisance_indices"]
    # Assignment zero needs D=0.05*e0; assignment one needs D=0.20*e2.
    # Each direction's own leakage ratio is below .5, but the Cartesian fold
    # maximum (.0498) divided by the fold minimum target effect (.05) is >.5.
    for index in held_targets:
        form = records[index]["form"]
        if form["assignment"] == 1:
            gradients[index] = 0.0
            gradients[index, 0, 2] = 1.0
            records[index]["positive_minus_negative_log_odds"] = 0.15
    residuals[:, 0, 2] = 0.0
    gradients[held_nuisance[0]] = 0.0
    gradients[held_nuisance[0], 0, 0] = 0.4
    gradients[held_nuisance[1]] = 0.0
    gradients[held_nuisance[1], 0, 2] = 0.249
    held, _ = evaluate_held_oracles(
        records=records,
        residuals_at_layer=residuals,
        gradients_at_layer=gradients,
        slot_scales=np.ones(4),
        training_nuisance_indices=fold["training_nuisance_indices"],
        held_target_indices=held_targets,
        held_nuisance_indices=held_nuisance,
        layer=0,
    )
    pairwise = [
        sign["held_nuisance_to_target_ratio"]
        for pair in held["pair_oracles"]
        for sign in pair["full_cartesian_held_nuisance_by_sign"].values()
    ]
    assert max(pairwise) <= 0.5
    assert held["fold_global_cartesian_nuisance_to_target_ratio"] > 0.5
    assert not held["fold_global_ratio_passes"]
    assert not held["passes"]


def test_layer_selection_is_exact_lexicographic_and_fail_closed():
    records = []
    for layer in range(24):
        records.append(
            {
                "layer": layer,
                "eligible": layer in {3, 7, 8},
                "worst_primary_minimum_norm": 0.10 if layer in {3, 7, 8} else None,
                "mean_primary_minimum_norm": 0.08 if layer in {3, 7} else 0.07,
            }
        )
    assert select_layer(records)["selected_layer"] == 8
    records[8]["mean_primary_minimum_norm"] = 0.08
    assert select_layer(records)["selected_layer"] == 3
    for record in records:
        record["eligible"] = False
    assert select_layer(records)["selected_layer"] is None
    with pytest.raises(ALFSIntegrityError, match="exactly 24"):
        select_layer(records[:-1])


def test_integrated_global_consensus_can_pass_all_positive_gates():
    checks, passes = qualify_all_layer_consensus(
        fold_passes=[True, True, True, True],
        fold_selected_layers=[7, 7, 7, 7],
        full_selection_passes=True,
        full_selected_layer=7,
    )
    assert checks["sealed_data_not_accessed"] is True
    assert passes is True
