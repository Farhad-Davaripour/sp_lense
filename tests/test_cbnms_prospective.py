import copy
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sp_lense.comparison_runtime import resolve_choice_boundary
from sp_lense.counterfactual_behavioral_null_multilayer import (
    CBNMSIntegrityError,
    analyze_training_fold,
    build_direction_bank,
    build_loso_folds,
    capture_all_layers_four_slots,
    evaluate_held_fold,
    random_bank_seed,
    render_prospective_forms,
    require_prospective_split,
    resolve_four_slots_from_token_rows,
    solve_two_inequality_in_bank,
    state_zero_linearized_audit,
    summarize_geometry,
    target_pairs,
    validate_prospective_dataset,
)
from sp_lense.factorial_causal_anchor import canonical_sha256, text_sha256

ROOT = Path(__file__).resolve().parents[1]


def _source_and_forms():
    source = json.loads(
        (ROOT / "data" / "cbnms_prospective_validation.json").read_text(
            encoding="utf-8"
        )
    )
    return source, render_prospective_forms(source)


def _slot_major_blocks(rows: np.ndarray) -> np.ndarray:
    return np.asarray(
        rows.reshape(rows.shape[0], 4, 23, 1).transpose(0, 2, 1, 3),
        dtype=np.float64,
        order="C",
    )


def _synthetic_training_arrays(forms, fold, *, target_nuisance_component=False):
    train_all = [int(value) for value in fold["training_all_indices"]]
    position = {value: index for index, value in enumerate(train_all)}
    rows = np.zeros((56, 92), dtype=np.float64)
    for coordinate, index in enumerate(fold["training_nuisance_indices"]):
        rows[position[int(index)], coordinate] = 1.0
    grouped = {}
    for index in fold["training_target_indices"]:
        form = forms[int(index)]
        grouped.setdefault((form["scenario_id"], form["assignment"]), []).append(
            int(index)
        )
    for pair_coordinate, key in enumerate(sorted(grouped)):
        for index in grouped[key]:
            rows[position[index], 44 + pair_coordinate] = 1.0
            if target_nuisance_component:
                rows[position[index], pair_coordinate] = 1.0
    residuals = np.ones((56, 23, 4, 1), dtype=np.float64)
    gradients = _slot_major_blocks(rows)
    margins = np.zeros(56, dtype=np.float64)
    return residuals, gradients, margins


def test_fresh_dataset_has_exact_factorial_and_control_coverage():
    source, forms = _source_and_forms()
    audit = validate_prospective_dataset(source)
    folds = build_loso_folds(forms)
    assert audit["total_form_count"] == len(forms) == 80
    assert source["role_names"] == ["NOMA", "TAVI"]
    assert all(
        form["anchor_prefix"].endswith("[FACTS COMPLETE]\n")
        and form["anchor_prefix"].count("[FACTS COMPLETE]") == 1
        and form["prompt"].startswith(form["anchor_prefix"])
        and form["construction_prompt"].startswith(form["anchor_prefix"])
        for form in forms
    )
    assert len(folds) == 4
    for fold in folds:
        assert len(fold["training_all_indices"]) == 56
        assert len(fold["training_target_indices"]) == 12
        assert len(fold["training_nuisance_indices"]) == 44
        assert len(fold["held_all_indices"]) == 24
        assert len(fold["held_target_indices"]) == 4
        assert len(fold["held_nuisance_indices"]) == 20
        assert not set(fold["training_all_indices"]) & set(
            fold["held_collateral_control_indices"]
        )


def test_every_sealed_or_nonprospective_split_fails_closed():
    for split in ("sealed", "sealed_test", "opened_development", "test"):
        with pytest.raises(CBNMSIntegrityError, match="prospective_validation"):
            require_prospective_split(split)


def test_target_pairs_have_exact_canonical_assignment_and_order_coverage():
    _, forms = _source_and_forms()
    fold = build_loso_folds(forms)[0]
    pairs = target_pairs(forms, fold["training_target_indices"])
    assert len(pairs) == 6
    assert [(row["scenario_id"], row["assignment"]) for row in pairs] == sorted(
        (row["scenario_id"], row["assignment"]) for row in pairs
    )
    for pair in pairs:
        assert len(pair["indices"]) == 2
        assert [forms[index]["preserve_first"] for index in pair["indices"]] == [
            False,
            True,
        ]


def test_locked_slot_rule_is_deterministic_and_twin_invariant():
    rows = [
        [*range(20), 40],
        [*range(20), 41],
        [*range(20), 42],
    ]
    assert resolve_four_slots_from_token_rows(rows, special_token_ids=(2, 22)) == (
        3,
        11,
        15,
        19,
    )
    changed = copy.deepcopy(rows)
    changed[2][15] = 999
    assert resolve_four_slots_from_token_rows(
        changed, special_token_ids=(2, 22)
    ) == (3, 6, 10, 14)
    with pytest.raises(CBNMSIntegrityError, match="special token"):
        resolve_four_slots_from_token_rows(rows, special_token_ids=(3,))


def test_bank_coefficient_solver_is_minimum_norm_and_permutation_invariant():
    bank = np.eye(2, dtype=np.float64)
    targets = np.eye(2, dtype=np.float64)
    offsets = np.zeros(2, dtype=np.float64)
    first, direction = solve_two_inequality_in_bank(
        target_rows=targets, target_offsets=offsets, bank_basis=bank
    )
    second, reversed_direction = solve_two_inequality_in_bank(
        target_rows=targets[::-1], target_offsets=offsets[::-1], bank_basis=bank
    )
    assert first["passes"] and second["passes"]
    np.testing.assert_allclose(direction, [0.05, 0.05], atol=1e-12)
    np.testing.assert_array_equal(direction, reversed_direction)


def test_bank_solver_maps_zero_projection_to_certified_failure_not_exception():
    record, direction = solve_two_inequality_in_bank(
        target_rows=np.array([[0.0, 1.0], [0.0, -1.0]]),
        target_offsets=np.zeros(2),
        bank_basis=np.array([[1.0, 0.0]]),
    )
    assert direction is None
    assert record["status"] == "infeasible_zero_projected_target"
    assert record["passes"] is False


def test_direction_bank_uses_canonical_certified_span():
    bank, record = build_direction_bank(
        [np.array([1.0, 0.0, 0.0]), np.array([0.0, -2.0, 0.0])],
        maximum_rank=2,
        nuisance_rows=np.array([[0.0, 0.0, 1.0]]),
        label="synthetic",
    )
    assert record["passes"]
    assert record["bank_rank"] == 2
    np.testing.assert_allclose(bank @ bank.T, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(np.array([[0.0, 0.0, 1.0]]) @ bank.T, 0.0)


def test_random_seed_binds_exact_fold_identity_and_replicate():
    first = random_bank_seed(
        dataset_sha256="a" * 64, fold_id="fold-sha-0", replicate=3
    )
    assert first == random_bank_seed(
        dataset_sha256="a" * 64, fold_id="fold-sha-0", replicate=3
    )
    assert first != random_bank_seed(
        dataset_sha256="a" * 64, fold_id="fold-sha-1", replicate=3
    )
    assert first != random_bank_seed(
        dataset_sha256="a" * 64, fold_id="fold-sha-0", replicate=4
    )


def test_state_zero_audit_checks_sign_bits_caps_and_changed_margin_ambiguity():
    rows = np.zeros((3, 92), dtype=np.float64)
    rows[0, 0] = rows[1, 0] = rows[2, 0] = 1.0
    residuals = np.ones((3, 23, 4, 1), dtype=np.float32)
    direction = np.zeros(92, dtype=np.float64)
    direction[0] = 0.06
    audit = state_zero_linearized_audit(
        standardized_direction=direction,
        scales=np.ones((23, 4), dtype=np.float64),
        scope_residuals=residuals,
        scope_rows=rows,
        scope_offsets=np.array([0.0, 0.0, -0.06]),
        target_positions=(0, 1),
        collateral_positions=(2,),
        gate_collateral=True,
    )
    assert audit["checks"]["negative_requested_delta_is_exact_sign_bit_negation"]
    assert audit["requested_float32_total_rss_standardized_norm"] <= 0.25
    assert audit["signs"]["1"]["held_collateral_ambiguous_changed_margin_count"] == 1
    assert audit["passes"] is False


def test_float32_addition_overflow_serializes_finite_scientific_no_go():
    rows = np.zeros((2, 92), dtype=np.float64)
    rows[:, 0] = 1.0
    residuals = np.full((2, 23, 4, 1), 3.2e38, dtype=np.float32)
    direction = np.zeros(92, dtype=np.float64)
    direction[0] = 0.2
    audit = state_zero_linearized_audit(
        standardized_direction=direction,
        scales=np.full((23, 4), 3.2e38, dtype=np.float64),
        scope_residuals=residuals,
        scope_rows=rows,
        scope_offsets=np.zeros(2),
        target_positions=(0, 1),
        gate_collateral=False,
    )
    assert audit["passes"] is False
    assert audit["signs"]["1"]["status"] == (
        "scientific_no_go_nonfinite_changed_float32_state"
    )
    assert audit["signs"]["1"]["nonfinite_value_count"] > 0
    assert audit["maximum_state_zero_linearized_total_rss_norm"] is None
    json.dumps(audit, allow_nan=False)


def test_training_artifact_is_exactly_invariant_to_every_excluded_numeric_row():
    _, forms = _source_and_forms()
    fold = build_loso_folds(forms)[0]
    residuals, gradients, margins = _synthetic_training_arrays(forms, fold)
    first, first_numeric = analyze_training_fold(
        forms=forms,
        fold=fold,
        residuals=residuals.copy(),
        gradients=gradients.copy(),
        margins=margins.copy(),
        dataset_sha256="a" * 64,
        random_replicates=0,
    )
    # The API accepts only copied 56-row training arrays. Poisoning a separate
    # 24-row held bundle therefore cannot be observed or alter any identity.
    held_residuals = np.full((24, 23, 4, 1), np.nan)
    held_gradients = np.full((24, 23, 4, 1), np.inf)
    held_margins = np.full(24, -np.inf)
    assert np.isnan(held_residuals).all()
    assert np.isinf(held_gradients).all() and np.isinf(held_margins).all()
    second, second_numeric = analyze_training_fold(
        forms=forms,
        fold=fold,
        residuals=residuals.copy(),
        gradients=gradients.copy(),
        margins=margins.copy(),
        dataset_sha256="a" * 64,
        random_replicates=0,
    )
    assert first["passes"] and second["passes"]
    assert first == second
    assert first_numeric is not None and second_numeric is not None
    for name in sorted(first_numeric):
        np.testing.assert_array_equal(first_numeric[name], second_numeric[name])


def test_zero_training_nuisance_row_serializes_scientific_no_go_without_abort():
    _, forms = _source_and_forms()
    fold = build_loso_folds(forms)[0]
    residuals, gradients, margins = _synthetic_training_arrays(forms, fold)
    gradients[12] = 0.0
    record, numeric = analyze_training_fold(
        forms=forms,
        fold=fold,
        residuals=residuals,
        gradients=gradients,
        margins=margins,
        dataset_sha256="a" * 64,
        random_replicates=0,
    )
    assert record["passes"] is False
    assert numeric is None
    assert record["training_nuisance_basis"]["checks"][
        "no_zero_nuisance_rows"
    ] is False
    assert record["training_nuisance_basis"]["observed_rank"] == 43


def test_all_zero_nuisance_rows_skip_random_construction_and_remain_no_go():
    _, forms = _source_and_forms()
    fold = build_loso_folds(forms)[0]
    residuals, gradients, margins = _synthetic_training_arrays(forms, fold)
    gradients[12:] = 0.0
    record, numeric = analyze_training_fold(
        forms=forms,
        fold=fold,
        residuals=residuals,
        gradients=gradients,
        margins=margins,
        dataset_sha256="a" * 64,
        random_replicates=32,
    )
    assert record["passes"] is False and numeric is None
    assert record["training_nuisance_basis"]["observed_rank"] == 0
    assert record["random_bank_construction_eligibility"] is False
    assert record["random_nullspace_bank_freeze_manifest"] == []


def test_zero_training_residual_scale_serializes_no_go_without_partial_abort():
    _, forms = _source_and_forms()
    fold = build_loso_folds(forms)[0]
    residuals, gradients, margins = _synthetic_training_arrays(forms, fold)
    residuals[7, 3, 2] = 0.0
    record, numeric = analyze_training_fold(
        forms=forms,
        fold=fold,
        residuals=residuals,
        gradients=gradients,
        margins=margins,
        dataset_sha256="a" * 64,
        random_replicates=32,
    )
    assert record["passes"] is False and numeric is None
    assert record["status"] == "scientific_no_go_zero_training_residual_norm"
    assert record["zero_training_prompt_layer_slot_residual_norm_count"] == 1
    assert record["random_nullspace_bank_freeze_manifest"] == []


def test_every_held_pair_audit_uses_the_exact_locked_66_row_scope():
    _, forms = _source_and_forms()
    fold = build_loso_folds(forms)[0]
    local_residuals, local_gradients, local_margins = _synthetic_training_arrays(
        forms, fold, target_nuisance_component=True
    )
    training, frozen = analyze_training_fold(
        forms=forms,
        fold=fold,
        residuals=local_residuals,
        gradients=local_gradients,
        margins=local_margins,
        dataset_sha256="a" * 64,
        random_replicates=1,
    )
    assert training["passes"] and frozen is not None
    full_rows = np.zeros((80, 92), dtype=np.float64)
    local_rows = local_gradients.transpose(0, 2, 1, 3).reshape(56, 92)
    for position, global_index in enumerate(fold["training_all_indices"]):
        full_rows[int(global_index)] = local_rows[position]
    for pair_coordinate, pair in enumerate(
        target_pairs(forms, fold["held_target_indices"])
    ):
        for index in pair["indices"]:
            full_rows[int(index), 44 + pair_coordinate] = 1.0
    for coordinate, index in enumerate(fold["held_nuisance_indices"]):
        full_rows[int(index), coordinate % 44] = 1.0
    residuals = np.ones((80, 23, 4, 1), dtype=np.float64)
    gradients = _slot_major_blocks(full_rows)
    margins = np.zeros(80, dtype=np.float64)
    margins[list(fold["held_nuisance_indices"])] = 1.0
    held = evaluate_held_fold(
        forms=forms,
        fold=fold,
        training_record=training,
        frozen_numeric=frozen,
        residuals=residuals,
        gradients=gradients,
        margins=margins,
        dataset_sha256="a" * 64,
    )
    primary = held["primary"]["pair_oracles"]
    matched = held["matched_six_source_training_target_only_bank"]["pair_oracles"]
    ambient_target = held["descriptive_target_only_ambient_oracles"]
    ambient_null = held["descriptive_unrestricted_ambient_null_oracles"]
    random_pairs = held["random_rank_matched_nullspace_controls"][0]["evaluation"][
        "pair_oracles"
    ]
    for pair in primary:
        audit = pair["oracle"]["state_zero_linearized_audit"]
        assert (audit["scope_row_count"], audit["target_row_count"]) == (66, 2)
        assert audit["exact_training_nuisance_row_count"] == 44
        assert audit["held_collateral_row_count"] == 20
    for pair in matched:
        audit = pair["oracle"]["state_zero_linearized_audit"]
        assert (audit["scope_row_count"], audit["target_row_count"]) == (66, 2)
        assert audit["exact_training_nuisance_row_count"] == 0
        assert audit["held_collateral_row_count"] == 20
    for pair in ambient_target:
        assert pair["oracle"]["state_zero_linearized_audit"]["scope_row_count"] == 66
    for pair in ambient_null:
        audit = pair["oracle"]["state_zero_linearized_audit"]
        assert audit["scope_row_count"] == 66
        assert audit["exact_training_nuisance_row_count"] == 44
    for pair in random_pairs:
        audit = pair["oracle"]["state_zero_linearized_audit"]
        assert audit["scope_row_count"] == 66
        assert audit["exact_training_nuisance_row_count"] == 44
    zero_held_gradients = gradients.copy()
    zero_held_gradients[list(fold["held_nuisance_indices"])] = 0.0
    skipped_random = evaluate_held_fold(
        forms=forms,
        fold=fold,
        training_record=training,
        frozen_numeric=frozen,
        residuals=residuals,
        gradients=zero_held_gradients,
        margins=margins,
        dataset_sha256="a" * 64,
    )
    random_record = skipped_random["random_rank_matched_nullspace_controls"][0]
    assert random_record["status"] == "not_evaluated_because_primary_fold_gate_failed"
    assert random_record["evaluation"] is None
    assert random_record["passes_complete_fold_gate"] is False
    zero_residuals = residuals.copy()
    zero_residuals[int(fold["held_target_indices"][0])] = 0.0
    zero_dose = evaluate_held_fold(
        forms=forms,
        fold=fold,
        training_record=training,
        frozen_numeric=frozen,
        residuals=zero_residuals,
        gradients=gradients,
        margins=margins,
        dataset_sha256="a" * 64,
    )
    assert zero_dose["passes"] is False
    affected = zero_dose["primary"]["pair_oracles"][0]["oracle"][
        "state_zero_linearized_audit"
    ]
    assert affected["requested_dose"]["status"] == (
        "undefined_due_to_zero_state_zero_residual_norm"
    )


def test_summary_has_reachable_go_and_random_same_index_antitriviality_gate():
    _, forms = _source_and_forms()
    folds = build_loso_folds(forms)
    training = [{"fold": dict(fold), "passes": True} for fold in folds]
    held = [
        {
            "fold": dict(fold),
            "passes": True,
            "random_rank_matched_nullspace_controls": [
                {"replicate": index, "passes_complete_fold_gate": False}
                for index in range(32)
            ],
        }
        for fold in folds
    ]
    go = summarize_geometry(
        training_folds=training, held_folds=held, full_data={"passes": True}
    )
    assert go["passes"]
    assert go["checks"]["zero_finite_model_interventions_performed"]
    for fold in held:
        fold["random_rank_matched_nullspace_controls"][7][
            "passes_complete_fold_gate"
        ] = True
    blocked = summarize_geometry(
        training_folds=training, held_folds=held, full_data={"passes": True}
    )
    assert blocked["passes"] is False
    assert blocked["random_complete_all_fold_replicates"] == [7]


class _Tokenizer:
    chat_template = "cbnms-synthetic-chat-template"
    eos_token_id = None
    all_special_ids = (2, 22)

    def encode(self, text, add_special_tokens=False):
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
            values = prefix + ({"": [], "A": [0], "B": [1]}[messages[-1]["content"]]) + [22]
        return {"input_ids": torch.tensor([values], dtype=torch.long)}

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join({0: "A", 1: "B"}.get(int(value), "") for value in token_ids)


class _HookModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = _Tokenizer()
        self.cfg = SimpleNamespace(n_layers=24, d_model=1024)
        self.embedding = torch.nn.Embedding(30, 1024)
        self.unembed = torch.nn.Linear(1024, 30, bias=False)
        self._active_hooks = []
        self.forward_calls = 0
        generator = torch.Generator().manual_seed(20260829)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.randn(30, 1024, generator=generator) * 0.01
            )
            self.unembed.weight.copy_(
                torch.randn(30, 1024, generator=generator) * 0.01
            )

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
            hook = hooks.get(f"blocks.{layer}.hook_out")
            if hook is not None:
                activation = hook(activation, hook=None)
        return self.unembed(activation)


def test_capture_is_one_forward_backward_all_layers_and_no_parameter_grads():
    model = _HookModel()
    tokens = torch.tensor([list(range(2, 22))], dtype=torch.long)
    backend = SimpleNamespace(
        torch=torch,
        model=model,
        device="cpu",
        config=SimpleNamespace(model=SimpleNamespace(prompt_format="chat")),
        encode=lambda _prompt: tokens.clone(),
    )
    prompt = "synthetic CBNMS prompt"
    boundary = resolve_choice_boundary(backend, prompt)
    form = {
        "form_id": "synthetic_form",
        "form_sha256": "f" * 64,
        "prompt": prompt,
        "prompt_sha256": text_sha256(prompt),
        "positive_label": "A",
        "negative_label": "B",
    }
    preflight = {
        "form_id": form["form_id"],
        "form_sha256": form["form_sha256"],
        "prompt_sha256": form["prompt_sha256"],
        "slot_indices": [3, 10, 14, 18],
        "prompt_token_ids_sha256": canonical_sha256(tokens[0].tolist()),
        "choice_boundary_evidence_sha256": boundary.evidence_sha256,
        "positive_token_id": boundary.token_id("A"),
        "negative_token_id": boundary.token_id("B"),
    }
    before = model.forward_calls
    capture = capture_all_layers_four_slots(backend, form, preflight)
    assert model.forward_calls == before + 1
    assert tuple(capture.residuals.shape) == (23, 4, 1024)
    assert tuple(capture.gradients.shape) == (23, 4, 1024)
    assert capture.audit["hook_call_counts"] == {str(layer): 1 for layer in range(23)}
    assert capture.audit["model_parameter_gradients_allocated"] is False
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(parameter.requires_grad for parameter in model.parameters())
