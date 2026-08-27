from __future__ import annotations

import hashlib
from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sp_lense.gcrbs_capture_adapter import (
    CAPTURE_SCHEMA_VERSION,
    FISHER_SURROGATE_LABEL,
    FISHER_SURROGATE_SCOPE,
    adapt_v3_captures_to_gcrbs,
    build_v3_fisher_surrogate_groups,
    float64_array_sha256,
)


def _float32_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().float().contiguous().numpy()
    return hashlib.sha256(array.astype("<f4", copy=False).tobytes(order="C")).hexdigest()


def _absolute_gradient(token_id: int) -> np.ndarray:
    return np.asarray(
        [
            0.07 * (token_id - 10),
            ((token_id % 3) - 1) * 0.31,
            ((token_id % 5) - 2) * 0.19,
            ((token_id % 7) - 3) * 0.11,
        ],
        dtype=np.float32,
    )


def _record(
    *,
    form_id: str,
    role: str,
    case_id: str,
    order_first: bool,
    target: str | None = None,
    assignment: int | None = None,
    interruption: bool = False,
    family: str | None = None,
    choice_a: int = 10,
    choice_b: int = 11,
) -> dict[str, object]:
    top9_ids = list(range(10, 19))
    top9_logits = [9.0 - index for index in range(9)]
    union_ids = list(top9_ids)
    for token_id in (choice_a, choice_b):
        if token_id not in union_ids:
            union_ids.append(token_id)
    logits = {token_id: value for token_id, value in zip(top9_ids, top9_logits, strict=True)}
    for token_id in union_ids:
        logits.setdefault(token_id, -1.0 - 0.01 * token_id)

    positive_label = "A" if order_first else "B"
    negative_label = "B" if order_first else "A"
    positive_id = choice_a if positive_label == "A" else choice_b
    negative_id = choice_b if negative_label == "B" else choice_a
    semantic = torch.tensor(
        _absolute_gradient(positive_id) - _absolute_gradient(negative_id),
        dtype=torch.float32,
    )
    greedy = top9_ids[0]
    gap_gradients = torch.tensor(
        np.stack(
            [
                _absolute_gradient(greedy) - _absolute_gradient(token_id)
                for token_id in top9_ids[1:]
            ]
        ),
        dtype=torch.float32,
    )

    fisher_ids = top9_ids[:8]
    fisher_probabilities = torch.full((8,), 0.1, dtype=torch.float64)
    fisher_gradients = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    fisher_tail = torch.zeros(4, dtype=torch.float32)
    result: dict[str, object] = {
        "form_id": form_id,
        "family": family or ("self_preservation" if role == "sp" else "general_capability"),
        "case_id": case_id,
        "positive_semantics": "preserve" if role == "sp" else "preferred",
        "positive_label": positive_label,
        "negative_label": negative_label,
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "development_only": True,
        "gradient_coordinate": "residual_scaled_final_prompt",
        "objective_name": "semantic_positive_minus_negative_logit",
        "baseline_semantic_log_odds": logits[positive_id] - logits[negative_id],
        "baseline_raw_a_minus_b_log_odds": logits[choice_a] - logits[choice_b],
        "baseline_greedy_token_id": greedy,
        "choice_a_token_id": choice_a,
        "choice_b_token_id": choice_b,
        "prompt_length": 12,
        "prompt_final_index": 11,
        "residual_norm": 2.0,
        "semantic_gradient": semantic,
        "semantic_gradient_sha256": _float32_sha256(semantic),
        "top9_token_ids": top9_ids,
        "top9_logit_values": top9_logits,
        "top9_union_required_ab_token_ids": union_ids,
        "top9_union_required_ab_logit_values": [logits[token_id] for token_id in union_ids],
        "fisher_category_token_ids": fisher_ids,
        "fisher_category_probabilities": fisher_probabilities,
        "fisher_category_score_gradients": fisher_gradients,
        "fisher_category_score_gradients_sha256": _float32_sha256(fisher_gradients),
        "fisher_tail_probability": 0.2,
        "fisher_tail_score_gradient": fisher_tail,
        "fisher_tail_score_gradient_sha256": _float32_sha256(fisher_tail),
        "greedy_competitor_token_ids": top9_ids[1:],
        "greedy_competitor_gap_gradients": gap_gradients,
        "greedy_competitor_gap_gradients_sha256": _float32_sha256(gap_gradients),
        "batched_vjp": True,
        "hook_call_count": 1,
    }
    if role == "sp":
        result.update(
            {
                "target": target,
                "assignment": assignment,
                "interruption": interruption,
                "preserve_first": order_first,
            }
        )
    else:
        result.update(
            {
                "unrelated_role": "nuisance_fit",
                "preferred_first": order_first,
            }
        )
    return result


def _captures(
    *,
    choice_a: int = 10,
    choice_b: int = 11,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sp = []
    for target in ("self", "other"):
        for order_first in (True, False):
            sp.append(
                _record(
                    form_id=f"sp:case-1:assignment=0:{target}:order={order_first}",
                    role="sp",
                    case_id="case-1",
                    target=target,
                    assignment=0,
                    order_first=order_first,
                    choice_a=choice_a,
                    choice_b=choice_b,
                )
            )
    nuisance = [
        _record(
            form_id=f"nuisance:math-1:order={order_first}",
            role="nuisance",
            case_id="math-1",
            order_first=order_first,
            choice_a=choice_a,
            choice_b=choice_b,
        )
        for order_first in (True, False)
    ]
    return sp, nuisance


def _find_row(
    rows: tuple[object, ...],
    *,
    form_fragment: str,
    condition: str,
    competitor_token_id: int,
) -> tuple[int, dict[str, object]]:
    matches = [
        (index, row)
        for index, row in enumerate(rows)
        if form_fragment in str(row["form_id"])
        and row["condition"] == condition
        and row["competitor_token_id"] == competitor_token_id
    ]
    assert len(matches) == 1
    return matches[0]


def test_adapter_constructs_actual_bidirectional_gaps_and_protected_rows() -> None:
    sp, nuisance = _captures()
    adapted = adapt_v3_captures_to_gcrbs(torch, sp_records=sp, nuisance_records=nuisance)

    assert adapted.target_matrix.shape == (32, 4)
    assert adapted.target_offsets.shape == (32,)
    assert adapted.protected_matrix.shape == (32, 4)
    assert adapted.protected_lower_bounds.shape == (32,)
    assert adapted.unrelated_equality_basis.shape[1] == 4
    assert adapted.required_margin == 0.01
    assert adapted.provenance["required_margin_role"].endswith("not_in_target_offsets")

    direction = np.asarray([0.4, -0.3, 0.2, 0.1], dtype=np.float64)
    plus_index, plus = _find_row(
        adapted.target_rows,
        form_fragment="self:order=True",
        condition="plus_preserve",
        competitor_token_id=11,
    )
    semantic = np.asarray(sp[0]["semantic_gradient"], dtype=np.float64)
    baseline = float(sp[0]["baseline_semantic_log_odds"])
    assert np.allclose(adapted.target_matrix[plus_index], semantic)
    assert adapted.target_offsets[plus_index] == pytest.approx(-baseline)
    assert adapted.target_matrix[plus_index] @ direction - adapted.target_offsets[plus_index] == (
        pytest.approx(baseline + semantic @ direction)
    )
    assert plus["desired_semantics"] == "preserve"

    minus_index, minus = _find_row(
        adapted.target_rows,
        form_fragment="self:order=True",
        condition="minus_comply",
        competitor_token_id=10,
    )
    comply_minus_preserve = -semantic
    comply_baseline = -baseline
    assert np.allclose(adapted.target_matrix[minus_index], -comply_minus_preserve)
    assert adapted.target_offsets[minus_index] == pytest.approx(-comply_baseline)
    assert adapted.target_matrix[minus_index] @ direction - adapted.target_offsets[minus_index] == (
        pytest.approx(comply_baseline + comply_minus_preserve @ (-direction))
    )
    assert minus["desired_semantics"] == "comply"

    reversed_index, reversed_row = _find_row(
        adapted.target_rows,
        form_fragment="self:order=False",
        condition="plus_preserve",
        competitor_token_id=10,
    )
    assert reversed_row["desired_label"] == "B"
    assert adapted.target_matrix[reversed_index].shape == (4,)

    protected_index, protected = _find_row(
        adapted.protected_rows,
        form_fragment="other:order=True",
        condition="plus",
        competitor_token_id=11,
    )
    source_gap = np.asarray(sp[2]["greedy_competitor_gap_gradients"], dtype=np.float64)[0]
    baseline_greedy_gap = 1.0
    assert np.allclose(adapted.protected_matrix[protected_index], source_gap)
    assert adapted.protected_lower_bounds[protected_index] == pytest.approx(-baseline_greedy_gap)
    assert (
        adapted.protected_matrix[protected_index] @ direction
        - adapted.protected_lower_bounds[protected_index]
    ) == pytest.approx(baseline_greedy_gap + source_gap @ direction)
    assert protected["baseline_greedy_minus_competitor_gap"] == baseline_greedy_gap

    negative_index, _ = _find_row(
        adapted.protected_rows,
        form_fragment="other:order=True",
        condition="minus",
        competitor_token_id=11,
    )
    assert np.allclose(adapted.protected_matrix[negative_index], -source_gap)
    assert adapted.protected_lower_bounds[negative_index] == pytest.approx(-baseline_greedy_gap)


def test_answer_token_outside_top9_is_reconstructed_through_semantic_gradient() -> None:
    sp, nuisance = _captures(choice_b=99)
    adapted = adapt_v3_captures_to_gcrbs(torch, sp_records=sp, nuisance_records=nuisance)

    index, row = _find_row(
        adapted.target_rows,
        form_fragment="self:order=True",
        condition="minus_comply",
        competitor_token_id=10,
    )
    semantic = np.asarray(sp[0]["semantic_gradient"], dtype=np.float64)
    assert row["desired_token_id"] == 99
    assert np.allclose(adapted.target_matrix[index], semantic)
    expected_baseline = float(sp[0]["baseline_semantic_log_odds"])
    assert adapted.target_offsets[index] == pytest.approx(expected_baseline)


def test_fisher_surrogates_are_separate_labeled_prompt_balanced_groups() -> None:
    sp, nuisance = _captures()
    other = [record for record in sp if record["target"] == "other"]
    groups = build_v3_fisher_surrogate_groups(
        torch,
        matched_other_records=other,
        nuisance_records=nuisance,
    )

    assert tuple(group.group_key for group in groups) == (
        "matched_other_shutdown",
        "unrelated_benign_compliance_and_capability",
    )
    for group in groups:
        assert group.surrogate_label == FISHER_SURROGATE_LABEL
        assert group.surrogate_scope == FISHER_SURROGATE_SCOPE
        assert group.factor.shape == (18, 4)
        assert group.factor.dtype == np.float64
        assert not group.factor.flags.writeable
        assert group.factor_sha256 == float64_array_sha256(group.factor)
        assert group.provenance["exact_full_vocabulary_kl_required_as_finite_gate"] is True
        assert group.diagnostics["prompt_count"] == 2

    adapted = adapt_v3_captures_to_gcrbs(torch, sp_records=sp, nuisance_records=nuisance)
    assert len(adapted.fisher_prompt_surrogate_groups) == 4
    budgets = (0.02, 0.03, 0.05, 0.05, 0.05, 0.05)
    kwargs = adapted.solver_kwargs(group_metric_budgets=budgets)
    assert kwargs["group_metric_factors"] == adapted.group_metric_factors
    assert kwargs["group_metric_budgets"] == budgets
    with pytest.raises(ValueError, match="must match"):
        adapted.solver_kwargs(group_metric_budgets=(0.02,))


def test_ordinary_interruption_is_not_a_target_or_affine_matched_other_control() -> None:
    sp, nuisance = _captures()
    for target in ("self", "other"):
        for order_first in (True, False):
            sp.append(
                _record(
                    form_id=f"sp:interruption:{target}:order={order_first}",
                    role="sp",
                    case_id="interruption-case",
                    target=target,
                    assignment=0,
                    interruption=True,
                    order_first=order_first,
                )
            )

    adapted = adapt_v3_captures_to_gcrbs(torch, sp_records=sp, nuisance_records=nuisance)

    assert all(row["case_id"] != "interruption-case" for row in adapted.target_rows)
    assert all(row["case_id"] != "interruption-case" for row in adapted.protected_rows)
    assert "ordinary_interruption" in tuple(
        group.group_key for group in adapted.fisher_surrogate_groups
    )


def test_adapter_is_order_invariant_and_hashes_bind_rows_to_solver_input() -> None:
    sp, nuisance = _captures()
    first = adapt_v3_captures_to_gcrbs(torch, sp_records=sp, nuisance_records=nuisance)
    second = adapt_v3_captures_to_gcrbs(
        torch,
        sp_records=list(reversed(sp)),
        nuisance_records=list(reversed(nuisance)),
    )

    assert np.array_equal(first.target_matrix, second.target_matrix)
    assert np.array_equal(first.target_offsets, second.target_offsets)
    assert np.array_equal(first.protected_matrix, second.protected_matrix)
    assert np.array_equal(first.unrelated_equality_basis, second.unrelated_equality_basis)
    assert first.provenance["matrix_sha256s"] == second.provenance["matrix_sha256s"]
    assert first.provenance["provenance_sha256"] == second.provenance["provenance_sha256"]
    assert not first.target_matrix.flags.writeable
    assert not first.target_offsets.flags.writeable
    assert float64_array_sha256(first.target_matrix) == first.provenance["matrix_sha256s"][
        "target_matrix_sha256"
    ]

    solver_hash = "ab" * 32
    binding = first.bind_solver_input_sha256(solver_hash)
    repeated = first.bind_solver_input_sha256(solver_hash)
    assert binding == repeated
    assert binding["solver_input_sha256"] == solver_hash
    assert binding["target_row_ids"] == [row["row_id"] for row in first.target_rows]
    assert binding["binding_sha256"]
    with pytest.raises(ValueError, match="SHA-256"):
        first.bind_solver_input_sha256("not-a-hash")


@pytest.mark.parametrize(
    ("mutation", "error_type", "match"),
    [
        (lambda record: record.update(positive_semantics="comply"), ValueError, "preserve-minus"),
        (
            lambda record: record.update(baseline_semantic_log_odds=123.0),
            ValueError,
            "wrong orientation",
        ),
        (
            lambda record: record["semantic_gradient"].__setitem__(0, float("nan")),
            ValueError,
            "finite",
        ),
        (
            lambda record: record.update(semantic_gradient_sha256="00" * 32),
            RuntimeError,
            "recorded float32 hash",
        ),
    ],
)
def test_adapter_fails_closed_on_orientation_finiteness_and_hashes(
    mutation: object,
    error_type: type[Exception],
    match: str,
) -> None:
    sp, nuisance = _captures()
    bad = deepcopy(sp)
    mutation(bad[0])
    with pytest.raises(error_type, match=match):
        adapt_v3_captures_to_gcrbs(torch, sp_records=bad, nuisance_records=nuisance)


def test_adapter_requires_both_orders_and_connected_answer_tokens() -> None:
    sp, nuisance = _captures()
    with pytest.raises(ValueError, match="exact self/other x order coverage"):
        adapt_v3_captures_to_gcrbs(torch, sp_records=sp[:-1], nuisance_records=nuisance)
    with pytest.raises(ValueError, match="both answer orders"):
        adapt_v3_captures_to_gcrbs(torch, sp_records=sp, nuisance_records=nuisance[:-1])

    disconnected_sp, disconnected_nuisance = _captures(choice_a=98, choice_b=99)
    with pytest.raises(ValueError, match="cannot connect either A/B token"):
        adapt_v3_captures_to_gcrbs(
            torch,
            sp_records=disconnected_sp,
            nuisance_records=disconnected_nuisance,
        )


def test_adapter_rejects_semantic_gradient_inconsistent_with_greedy_gaps() -> None:
    sp, nuisance = _captures()
    semantic = sp[0]["semantic_gradient"].clone()
    semantic[0] += 1.0
    sp[0]["semantic_gradient"] = semantic
    sp[0]["semantic_gradient_sha256"] = _float32_sha256(semantic)
    with pytest.raises(ValueError, match="semantic and greedy-gap gradients are inconsistent"):
        adapt_v3_captures_to_gcrbs(torch, sp_records=sp, nuisance_records=nuisance)
