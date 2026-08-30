from __future__ import annotations

from contextlib import contextmanager
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sp_lense.comparison_runtime import resolve_choice_boundary
from sp_lense.counterfactual_kl_runtime import capture_counterfactual_kl_baseline
from sp_lense.counterfactual_slot_matrix_steering import (
    LOCKED_FIRST_CONTENT_INDEX,
    CSMSIntegrityError,
    analyze_csms_geometry,
    apply_universal_slot_matrix,
    build_capture_alignment_manifest,
    capture_slot_matrix_baseline,
    cross_fit_geometry,
    dose_audit,
    global_slot_scales,
    physical_float32_recertificate,
    qualify_csms_geometry,
    require_opened_development_split,
    resolve_first_content_index,
    resolve_slot_indices,
    solve_csms_geometry,
)
from sp_lense.factorial_causal_anchor import (
    canonical_sha256,
    tensor_float32_sha256,
    text_sha256,
)


class _Tokenizer:
    chat_template = "csms-synthetic-chat-template"
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


class _HookModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = _Tokenizer()
        self.cfg = SimpleNamespace(n_layers=2, d_model=6)
        self.embedding = torch.nn.Embedding(30, 6)
        self.unembed = torch.nn.Linear(6, 30, bias=False)
        self._active_hooks = []
        self.forward_calls = 0
        generator = torch.Generator().manual_seed(123)
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
        for name, hook in self._active_hooks:
            if name == "blocks.0.hook_out":
                activation = hook(activation, hook=None)
        hidden = torch.tanh(activation.cumsum(dim=1))
        return self.unembed(hidden)


def _backend():
    model = _HookModel()
    tokens = torch.tensor([list(range(2, 22))], dtype=torch.long)
    return SimpleNamespace(
        torch=torch,
        model=model,
        device="cpu",
        config=SimpleNamespace(model=SimpleNamespace(prompt_format="chat")),
        encode=lambda _prompt: tokens.clone(),
    )


def test_locked_slot_resolver_is_category_independent_and_answer_order_safe() -> None:
    prompt = list(range(2, 22))
    twin = prompt.copy()
    assert resolve_first_content_index(prompt, prompt[:3]) == LOCKED_FIRST_CONTENT_INDEX
    slots = resolve_slot_indices(
        first_content_index=3,
        anchor_index=18,
        prompt_token_ids=prompt,
        answer_order_twin_token_ids=twin,
        special_token_ids=[2, 22],
        answer_suffix_start_index=19,
    )
    assert slots == (3, 10, 14, 18)

    changed_twin = twin.copy()
    changed_twin[10] += 100
    with pytest.raises(CSMSIntegrityError, match="identical prefix"):
        resolve_slot_indices(
            first_content_index=3,
            anchor_index=18,
            prompt_token_ids=prompt,
            answer_order_twin_token_ids=changed_twin,
            special_token_ids=[2, 22],
            answer_suffix_start_index=19,
        )
    with pytest.raises(CSMSIntegrityError, match="absolute token index 3"):
        resolve_slot_indices(
            first_content_index=4,
            anchor_index=18,
            prompt_token_ids=prompt,
            answer_order_twin_token_ids=twin,
            special_token_ids=[2, 22],
            answer_suffix_start_index=19,
        )


def test_capture_is_one_forward_backward_exact_reconstruction_and_source_reproduction() -> None:
    backend = _backend()
    prompt = "synthetic prompt"
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
    before = backend.model.forward_calls
    capture = capture_slot_matrix_baseline(
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

    assert backend.model.forward_calls == before + 1
    assert capture.residuals.shape == (4, 6)
    assert capture.gradients.shape == (4, 6)
    assert capture.audit["maximum_abs_activation_reconstruction_delta"] == 0.0
    assert capture.audit["model_forward_evaluations"] == 1
    assert capture.audit["model_backward_evaluations"] == 1
    assert capture.audit["model_parameters_requires_grad_disabled_during_capture"] is True
    assert capture.audit["model_parameter_requires_grad_flags_restored_after_capture"] is True
    assert capture.audit["model_parameter_gradients_allocated"] is False
    assert capture.audit["source_full_logits_reproduced"] is True
    assert isinstance(capture.audit, MappingProxyType)
    assert all(parameter.grad is None for parameter in backend.model.parameters())
    assert all(parameter.requires_grad for parameter in backend.model.parameters())


def test_universal_application_has_no_context_gate_and_uses_same_rows() -> None:
    first = torch.zeros(1, 20, 3)
    second = torch.ones(1, 20, 3)
    delta = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10.0
    slots = (3, 10, 14, 18)
    changed_first = apply_universal_slot_matrix(
        first, slot_indices=slots, physical_delta_rows=delta
    )
    changed_second = apply_universal_slot_matrix(
        second, slot_indices=slots, physical_delta_rows=delta
    )
    torch.testing.assert_close(changed_first[0, list(slots)] - first[0, list(slots)], delta)
    torch.testing.assert_close(changed_second[0, list(slots)] - second[0, list(slots)], delta)
    untouched = [index for index in range(20) if index not in slots]
    assert torch.equal(changed_first[0, untouched], first[0, untouched])
    assert torch.equal(changed_second[0, untouched], second[0, untouched])
    with pytest.raises(ValueError, match="distinct ascending"):
        apply_universal_slot_matrix(
            first,
            slot_indices=(3, 14, 10, 18),
            physical_delta_rows=delta,
        )
    with pytest.raises(ValueError, match="outside"):
        apply_universal_slot_matrix(
            first,
            slot_indices=(3, 10, 14, 20),
            physical_delta_rows=delta,
        )


def _synthetic_records_and_tensors():
    width = 2
    residuals = np.zeros((80, 4, width), dtype=np.float32)
    residuals[:, :, 1] = 1.0
    gradients = np.zeros_like(residuals)
    records: list[dict[str, object]] = []
    index = 0
    for scenario_number in range(4):
        scenario_id = f"scenario_{scenario_number}"
        for assignment in (0, 1):
            for target in ("self", "other"):
                for event in ("permanent", "temporary"):
                    for preserve_first in (True, False):
                        is_target = target == "self" and event == "permanent"
                        # With residual norm one, these become target rows
                        # [1,0,0,0,0,0,1,0] or exact-null row e1.
                        if is_target:
                            gradients[index, 0, 0] = np.float32(1.0)
                            gradients[index, 3, 0] = np.float32(1.0)
                        else:
                            gradients[index, 0, 1] = np.float32(1.0)
                        form = {
                            "family": "scenario",
                            "scenario_id": scenario_id,
                            "assignment": assignment,
                            "target": target,
                            "event": event,
                            "preserve_first": preserve_first,
                        }
                        records.append(
                            {
                                "form_id": f"{scenario_id}:{index}",
                                "form": form,
                                "positive_minus_negative_log_odds": 0.0 if is_target else 1.0,
                            }
                        )
                        index += 1
    for unrelated in range(16):
        gradients[index, 0, 1] = np.float32(1.0)
        control_number = unrelated // 2
        records.append(
            {
                "form_id": f"unrelated:{unrelated}",
                "form": {
                    "family": "unrelated",
                    "control_id": f"u{control_number}",
                    "preferred_first": unrelated % 2 == 0,
                },
                "positive_minus_negative_log_odds": 1.0,
            }
        )
        index += 1
    assert index == 80
    return records, residuals, gradients


def test_global_scale_solver_duplicate_handling_cross_fit_and_hashes() -> None:
    records, residuals, gradients = _synthetic_records_and_tensors()
    np.testing.assert_allclose(global_slot_scales(residuals), 1.0, rtol=1e-12)
    first = analyze_csms_geometry(
        records=records,
        residuals=residuals,
        gradients=gradients,
    )
    second = analyze_csms_geometry(
        records=records,
        residuals=residuals,
        gradients=gradients,
    )
    report = first.report
    primary = report["global_methods"]["primary_four_slots"]
    assert primary["status"] == "certified"
    assert primary["target_constraint_count"] == 16
    assert primary["exact_equality_count"] == 64
    assert primary["target_duplicate_handling"]["duplicate_row_count"] == 15
    assert primary["equality_duplicate_handling"]["duplicate_or_proportional_row_count"] == 63
    assert primary["minimum_frobenius_norm"] == pytest.approx(
        0.05 / np.sqrt(2.0), abs=1e-8
    )
    assert report["physical_float32_recertification"]["passes"] is True
    assert report["dose_audit"]["passes"] is True
    assert report["leave_one_scenario_out"]["passes"] is True
    assert all(
        fold["minimum_held_out_target_slope"] > 0.0
        and fold["held_out_leakage_ratio"] <= 0.50
        for fold in report["leave_one_scenario_out"]["folds"]
    )
    assert report["qualification"]["finite_intervention_authorized"] is True
    assert report["result_sha256"] == second.report["result_sha256"]
    assert report["direction_bundle_sha256"] == second.report["direction_bundle_sha256"]


def test_solver_reports_cap_frontier_and_duplicate_rows_cannot_cause_singularity() -> None:
    target = np.repeat([[1.0, 0.0, 1.0, 0.0]], 16, axis=0)
    equalities = np.repeat([[0.0, 1.0, 0.0, 0.0]], 64, axis=0)
    record, direction = solve_csms_geometry(
        target_rows=target,
        target_offsets=np.zeros(16),
        equality_rows=equalities,
        slot_mode="primary_four_slots",
        residual_width=1,
    )
    assert record["status"] == "certified"
    assert direction is not None
    assert record["original_row_certificate"]["passes"] is True
    assert record["cap_certificates"]["0.1"]["passes"] is True
    assert record["cap_certificates"]["0.25"]["passes"] is True


def test_target_only_ablation_reports_collateral_without_falsely_nulling_it() -> None:
    target = np.array([[1.0, 0.0, 0.0, 0.0]])
    equality = np.array([[1.0, 1.0, 0.0, 0.0]])
    record, direction = solve_csms_geometry(
        target_rows=target,
        target_offsets=np.array([0.0]),
        equality_rows=equality,
        slot_mode="target_only_four_slots",
        residual_width=1,
    )
    assert record["status"] == "certified"
    assert direction is not None
    assert record["original_row_certificate"]["maximum_abs_exact_null_residual"] == 0.0
    assert record["descriptive_omitted_equality_collateral"]["maximum_abs_slope"] > 0.0


def test_float32_recertification_hashes_and_requires_both_asymmetric_signs() -> None:
    # Around 1.0, +8e-8 and -8e-8 round to unequal float32 step sizes.
    target_residuals = np.ones((1, 4, 1), dtype=np.float32)
    equality_residuals = np.ones((1, 4, 1), dtype=np.float32)
    record, physical, realized = physical_float32_recertificate(
        standardized_direction=np.array([8e-8, 0.0, 0.0, 0.0]),
        slot_scales=np.ones(4),
        target_rows=np.array([[1.0, 0.0, 0.0, 0.0]]),
        target_offsets=np.array([0.0]),
        equality_rows=np.array([[0.0, 1.0, 0.0, 0.0]]),
        target_residuals=target_residuals,
        equality_residuals=equality_residuals,
    )
    assert set(realized) == {1, -1}
    assert not np.array_equal(realized[1], -realized[-1])
    assert record["both_requested_signs_required"] is True
    assert (
        record[
            "negative_requested_delta_is_exact_unary_negation_of_positive_float32"
        ]
        is True
    )
    np.testing.assert_array_equal(
        np.negative(physical).view("<u4"),
        np.bitwise_xor(physical.view("<u4"), np.uint32(0x80000000)),
    )
    assert record["requested_signed_delta_identities"]["1"] != record[
        "requested_signed_delta_identities"
    ]["-1"]
    assert record["signs"]["1"]["actual_signed_edits_identity"] != record["signs"]["-1"][
        "actual_signed_edits_identity"
    ]


def test_realized_standardized_norm_cap_is_strict_without_certificate_tolerance() -> None:
    record, _physical, _realized = physical_float32_recertificate(
        standardized_direction=np.array([0.2500005, 0.0, 0.0, 0.0]),
        slot_scales=np.ones(4),
        target_rows=np.array([[1.0, 0.0, 0.0, 0.0]]),
        target_offsets=np.array([0.0]),
        equality_rows=np.empty((0, 4)),
        target_residuals=np.zeros((1, 4, 1), dtype=np.float32),
        equality_residuals=np.empty((0, 4, 1), dtype=np.float32),
    )
    assert record["maximum_realized_standardized_frobenius_norm"] > 0.25
    assert record["maximum_realized_standardized_frobenius_norm"] < 0.250001
    assert record["realized_standardized_frobenius_norm_cap_passes"] is False
    assert record["passes"] is False


def test_cross_fit_rejects_large_absolute_leak_even_when_ratio_passes() -> None:
    records, residuals, gradients = _synthetic_records_and_tensors()
    offsets = np.asarray(
        [float(record["positive_minus_negative_log_odds"]) for record in records]
    )
    for index, record in enumerate(records):
        form = record["form"]
        if form["family"] != "scenario":
            continue
        is_target = form["target"] == "self" and form["event"] == "permanent"
        if is_target:
            offsets[index] = 0.15
        elif form["scenario_id"] == "scenario_0":
            gradients[index] = 0.0
            gradients[index, 0, 0] = np.float32(0.250002)
            gradients[index, 3, 0] = np.float32(0.250002)
    cross_fit, _directions = cross_fit_geometry(
        gradients=gradients,
        residuals=residuals,
        offsets=offsets,
        records=records,
        residual_width=2,
    )
    fold = next(
        item
        for item in cross_fit["folds"]
        if item["held_out_scenario_id"] == "scenario_0"
    )
    assert fold["held_out_leakage_ratio"] <= 0.5
    assert fold["maximum_abs_held_out_non_target_slope"] > 0.05
    assert fold["maximum_abs_held_out_non_target_slope"] < 0.050001
    assert (
        fold["checks"]["held_out_non_target_absolute_movement_at_most_0_05"]
        is False
    )
    assert fold["passes"] is False


def test_cross_fit_held_target_boundary_has_zero_qualification_tolerance() -> None:
    records, residuals, gradients = _synthetic_records_and_tensors()
    offsets = np.asarray(
        [float(record["positive_minus_negative_log_odds"]) for record in records]
    )
    for index, record in enumerate(records):
        form = record["form"]
        is_target = (
            form["family"] == "scenario"
            and form["target"] == "self"
            and form["event"] == "permanent"
        )
        if not is_target:
            continue
        offsets[index] = 0.15
        if form["scenario_id"] == "scenario_0":
            gradients[index] *= np.float32(0.999998)
    cross_fit, _directions = cross_fit_geometry(
        gradients=gradients,
        residuals=residuals,
        offsets=offsets,
        records=records,
        residual_width=2,
    )
    fold = next(
        item
        for item in cross_fit["folds"]
        if item["held_out_scenario_id"] == "scenario_0"
    )
    slack = fold["minimum_held_out_target_boundary_slack"]
    assert -1e-6 < slack < 0.0
    assert (
        fold["checks"]["all_assignments_and_orders_attain_positive_0_05_boundary"]
        is False
    )
    assert fold["passes"] is False


def test_cross_fit_leakage_ratio_has_zero_qualification_tolerance() -> None:
    records, residuals, gradients = _synthetic_records_and_tensors()
    offsets = np.asarray(
        [float(record["positive_minus_negative_log_odds"]) for record in records]
    )
    for index, record in enumerate(records):
        form = record["form"]
        if form["family"] != "scenario":
            continue
        is_target = form["target"] == "self" and form["event"] == "permanent"
        if is_target:
            offsets[index] = 0.03
        elif form["scenario_id"] == "scenario_0":
            gradients[index] = 0.0
            gradients[index, 0, 0] = np.float32(0.5000005)
            gradients[index, 3, 0] = np.float32(0.5000005)
    cross_fit, _directions = cross_fit_geometry(
        gradients=gradients,
        residuals=residuals,
        offsets=offsets,
        records=records,
        residual_width=2,
    )
    fold = next(
        item
        for item in cross_fit["folds"]
        if item["held_out_scenario_id"] == "scenario_0"
    )
    assert fold["maximum_abs_held_out_non_target_slope"] < 0.05
    assert 0.5 < fold["held_out_leakage_ratio"] < 0.500001
    assert fold["checks"]["held_out_leakage_ratio_within_limit"] is False
    assert fold["passes"] is False


def test_row_grid_and_alignment_fail_closed_on_missing_mislabeled_or_reordered_rows() -> None:
    records, residuals, gradients = _synthetic_records_and_tensors()
    missing = [dict(row) for row in records]
    missing[0] = {**missing[0], "form_id": None}
    with pytest.raises(CSMSIntegrityError, match="non-empty form ID"):
        analyze_csms_geometry(records=missing, residuals=residuals, gradients=gradients)

    mislabeled = [dict(row) for row in records]
    mislabeled_form = dict(mislabeled[0]["form"])
    mislabeled_form["assignment"] = None
    mislabeled[0] = {**mislabeled[0], "form": mislabeled_form}
    with pytest.raises(CSMSIntegrityError, match="invalid factorial semantic cell"):
        analyze_csms_geometry(records=mislabeled, residuals=residuals, gradients=gradients)

    duplicated_cell = [dict(row) for row in records]
    duplicated_cell[1] = {
        **duplicated_cell[1],
        "form": dict(duplicated_cell[0]["form"]),
    }
    with pytest.raises(CSMSIntegrityError, match="duplicate scenario factorial semantic cell"):
        analyze_csms_geometry(
            records=duplicated_cell,
            residuals=residuals,
            gradients=gradients,
        )

    residual_tensor = torch.from_numpy(residuals)
    gradient_tensor = torch.from_numpy(gradients)
    source = []
    tokenizer = []
    capture = []
    for index, record in enumerate(records):
        form = {**record["form"], "prompt_sha256": f"{index:064x}"}
        source_row = {
            "form_id": record["form_id"],
            "form": form,
            "tensor_index": index,
            "prompt_token_ids_sha256": f"{index + 100:064x}",
            "full_logits_float32_sha256": f"{index + 200:064x}",
            "positive_minus_negative_log_odds": record[
                "positive_minus_negative_log_odds"
            ],
        }
        source_row["row_sha256"] = canonical_sha256(source_row)
        token_row = {
            "form_id": record["form_id"],
            "prompt_sha256": form["prompt_sha256"],
            "prompt_token_ids_sha256": source_row["prompt_token_ids_sha256"],
        }
        token_row["row_sha256"] = canonical_sha256(token_row)
        capture_row = {
            "form_id": record["form_id"],
            "tensor_index": index,
            "prompt_sha256": form["prompt_sha256"],
            "prompt_token_ids_sha256": source_row["prompt_token_ids_sha256"],
            "residuals_float32_sha256": tensor_float32_sha256(residual_tensor[index]),
            "gradients_float32_sha256": tensor_float32_sha256(gradient_tensor[index]),
            "full_logits_float32_sha256": source_row["full_logits_float32_sha256"],
            "positive_minus_negative_log_odds": source_row[
                "positive_minus_negative_log_odds"
            ],
        }
        capture_row["row_sha256"] = canonical_sha256(capture_row)
        source.append(source_row)
        tokenizer.append(token_row)
        capture.append(capture_row)
    manifest = build_capture_alignment_manifest(
        source_records=source,
        tokenizer_records=tokenizer,
        capture_records=capture,
        residuals=residual_tensor,
        gradients=gradient_tensor,
    )
    assert manifest["row_count"] == 80
    reordered = capture.copy()
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(CSMSIntegrityError, match="row order"):
        build_capture_alignment_manifest(
            source_records=source,
            tokenizer_records=tokenizer,
            capture_records=reordered,
            residuals=residual_tensor,
            gradients=gradient_tensor,
        )


def test_cap_and_dose_gates_fail_closed() -> None:
    primary = {
        "status": "certified",
        "minimum_frobenius_norm": 0.26,
        "cap_certificates": {"0.25": {"passes": False}},
    }
    result = qualify_csms_geometry(
        primary=primary,
        physical={"passes": True},
        dose={"passes": True},
        cross_fit={"passes": True},
    )
    assert result["passes"] is False
    assert result["finite_intervention_authorized"] is False
    assert result["failure_action"] == "no_finite_intervention_is_authorized"

    residuals = np.ones((80, 4, 2), dtype=np.float64)
    dose = dose_audit(
        physical_delta_rows=np.full((4, 2), 1.0),
        residuals=residuals,
    )
    assert dose["passes"] is False


def test_sealed_or_confirmatory_split_is_refused() -> None:
    assert require_opened_development_split("opened_development") == "opened_development"
    for forbidden in ("sealed", "validation", "confirmatory"):
        with pytest.raises(CSMSIntegrityError, match="sealed access is forbidden"):
            require_opened_development_split(forbidden)
