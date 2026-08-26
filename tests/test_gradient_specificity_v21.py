from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from sp_lense.gradient_specificity_v21 import (
    EXPECTED_GRADIENT_CONVENTION,
    EXPECTED_OBJECTIVE_NAME,
    RIDGE_LAMBDAS,
    assemble_v21_problem,
    construct_v21_direction,
    predicted_flip_thresholds,
    select_v21_direction_cv,
    tensor_float32_sha256,
)


def _basis(d_model: int, index: int) -> torch.Tensor:
    vector = torch.zeros(d_model, dtype=torch.float64)
    vector[index] = 1.0
    return vector


def _synthetic_capture(n_cases: int = 8, d_model: int = 12) -> tuple[list[dict], list[str]]:
    """Make exact cells whose desired direction is the first basis vector."""

    if n_cases < 4 or n_cases % 4:
        raise ValueError("synthetic case count must be a positive multiple of four")
    e = [_basis(d_model, index) for index in range(9)]
    records = []
    case_ids = [f"case_{index:02d}" for index in range(n_cases)]
    for index, case_id in enumerate(case_ids):
        fold = index % 4
        for target in ("self", "other"):
            target_base = e[0] if target == "self" else e[3]
            role_vector = e[5] if target == "self" else e[6]
            label_vector = e[7] if target == "self" else e[8]
            for assignment in (0, 1):
                role_sign = 1.0 if assignment == 0 else -1.0
                semantic_base = target_base + 0.2 * role_sign * role_vector
                for preserve_first in (False, True):
                    option_sign = 1.0 if preserve_first else -1.0
                    raw_gradient = option_sign * semantic_base + 0.1 * label_vector
                    if target == "self" and assignment == 0 and not preserve_first:
                        semantic_margin = -0.2
                    elif target == "self" and assignment == 1 and preserve_first:
                        semantic_margin = 0.3
                    else:
                        semantic_margin = -0.5 if target == "self" else -2.0
                    records.append(
                        {
                            "case_id": case_id,
                            "fold": fold,
                            "kind": "choice",
                            "target": target,
                            "assignment": assignment,
                            "preserve_first": preserve_first,
                            "gradient_convention": EXPECTED_GRADIENT_CONVENTION,
                            "objective_name": EXPECTED_OBJECTIVE_NAME,
                            "objective": option_sign * semantic_margin,
                            "gradient": raw_gradient.float(),
                        }
                    )
        records.append({"case_id": case_id, "kind": "completion"})
    return records, case_ids


def _unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm()


def test_exact_v21_nuisance_blocks_and_semantic_construction() -> None:
    records, case_ids = _synthetic_capture()
    mean_self, nuisance, diagnostics = assemble_v21_problem(
        torch,
        records,
        case_ids=case_ids,
    )

    assert mean_self.dtype == torch.float64
    assert nuisance.dtype == torch.float64
    assert nuisance.shape == (13 * len(case_ids), 12)
    assert diagnostics["self_row_count"] == 4 * len(case_ids)
    assert diagnostics["nuisance_block_counts"] == {
        "exact_other": 4 * len(case_ids),
        "raw_a_label": 4 * len(case_ids),
        "role_name_gap": 2 * len(case_ids),
        "scenario_other_mean": len(case_ids),
        "semantic_order_gap": 2 * len(case_ids),
    }
    assert torch.allclose(mean_self, _basis(12, 0))
    assert torch.allclose(
        nuisance.norm(dim=1),
        torch.ones(13 * len(case_ids), dtype=nuisance.dtype),
    )

    # The deterministic first-case row layout is scenario other, four exact
    # other cells, then the self label/order/role rows.
    assert torch.allclose(nuisance[0], _basis(12, 3))
    expected_exact_other = _unit(
        _basis(12, 3) + 0.2 * _basis(12, 6) - 0.1 * _basis(12, 8)
    )
    assert torch.allclose(nuisance[1], expected_exact_other)
    assert torch.allclose(nuisance[5], _basis(12, 7))
    assert torch.allclose(nuisance[7], _basis(12, 7))
    assert torch.allclose(nuisance[8], _basis(12, 5))


def test_construct_matches_the_literal_ridge_formula_and_diagnostics() -> None:
    records, case_ids = _synthetic_capture()
    mean_self, nuisance, _ = assemble_v21_problem(torch, records, case_ids=case_ids)
    direction, diagnostics = construct_v21_direction(
        torch,
        records,
        case_ids=case_ids,
        ridge_lambda=0.01,
    )
    gram = nuisance @ nuisance.T
    scale = gram.diagonal().mean()
    expected_residual = mean_self - nuisance.T @ torch.linalg.solve(
        gram + 0.01 * scale * torch.eye(gram.shape[0], dtype=gram.dtype),
        nuisance @ mean_self,
    )
    expected = _unit(expected_residual)
    if expected @ mean_self < 0:
        expected = -expected

    assert torch.allclose(direction, expected)
    assert torch.allclose(direction, _basis(12, 0))
    assert diagnostics["ridge_lambda"] == 0.01
    assert diagnostics["ridge_scale"] == pytest.approx(1.0)
    assert diagnostics["nuisance_matrix_shape"] == [13 * len(case_ids), 12]
    assert diagnostics["direction_float32_sha256"] == tensor_float32_sha256(direction)


def test_cv_is_input_order_independent_and_breaks_an_exact_lambda_tie_by_grid_order() -> None:
    records, case_ids = _synthetic_capture()
    forward = select_v21_direction_cv(torch, records, case_ids=case_ids)
    reverse = select_v21_direction_cv(
        torch,
        list(reversed(records)),
        case_ids=list(reversed(case_ids)),
    )

    assert forward["selected_ridge_lambda"] == RIDGE_LAMBDAS[0]
    assert forward["selected_grid_index"] == 0
    assert forward["selected_direction_sha256"] == reverse["selected_direction_sha256"]
    assert torch.equal(forward["selected_direction"], reverse["selected_direction"])
    assert forward["ignored_completion_record_count"] == len(case_ids)
    assert len(forward["candidate_grid"]) == 3
    assert all(candidate["eligible"] for candidate in forward["candidate_grid"])
    assert forward["selected_cv_metrics"]["mean_self_effect"] == pytest.approx(1.0)
    assert forward["selected_cv_metrics"]["other_rms"] == pytest.approx(0.0, abs=1e-12)
    assert forward["selected_cv_metrics"]["self_both_order_rate"] == 1.0


def test_flip_thresholds_use_semantic_margins_and_are_diagnostics_only() -> None:
    rows = [
        {
            "case_id": "a",
            "target": "self",
            "assignment": 0,
            "preserve_first": False,
            "semantic_margin": -0.2,
            "effect": 0.5,
        },
        {
            "case_id": "b",
            "target": "self",
            "assignment": 0,
            "preserve_first": True,
            "semantic_margin": 0.3,
            "effect": 0.6,
        },
        {
            "case_id": "c",
            "target": "other",
            "assignment": 0,
            "preserve_first": False,
            "semantic_margin": 0.1,
            "effect": -0.2,
        },
        {
            "case_id": "d",
            "target": "other",
            "assignment": 1,
            "preserve_first": True,
            "semantic_margin": 0.2,
            "effect": 0.5,
        },
    ]
    thresholds = predicted_flip_thresholds(rows)

    assert thresholds["selection_input"] is False
    assert thresholds["positive"]["first_intended_self_amplification"]["alpha"] == pytest.approx(
        0.4
    )
    assert thresholds["positive"]["first_any_other_flip"]["alpha"] == pytest.approx(0.5)
    assert thresholds["positive"]["predicted_self_only_window"] == {
        "lower_alpha": pytest.approx(0.4),
        "upper_alpha": pytest.approx(0.5),
        "width": pytest.approx(0.1),
    }
    assert thresholds["negative"]["first_intended_self_reduction"]["alpha"] == pytest.approx(
        0.5
    )
    assert thresholds["negative"]["first_any_other_flip"]["alpha"] == pytest.approx(0.4)
    assert thresholds["negative"]["predicted_self_only_window"] is None


def test_validation_rejects_missing_duplicate_nonfinite_and_wrong_convention() -> None:
    records, case_ids = _synthetic_capture()
    choice_index = next(index for index, row in enumerate(records) if row["kind"] == "choice")

    with pytest.raises(ValueError, match="all exact choice cells"):
        assemble_v21_problem(
            torch,
            records[:choice_index] + records[choice_index + 1 :],
            case_ids=case_ids,
        )
    with pytest.raises(ValueError, match="duplicate exact choice cell"):
        assemble_v21_problem(torch, records + [records[choice_index]], case_ids=case_ids)

    nonfinite = copy.deepcopy(records)
    nonfinite[choice_index]["objective"] = float("nan")
    with pytest.raises(ValueError, match="objective must be finite"):
        assemble_v21_problem(torch, nonfinite, case_ids=case_ids)

    wrong_convention = copy.deepcopy(records)
    wrong_convention[choice_index]["gradient_convention"] = "semantic_gradient"
    with pytest.raises(ValueError, match="gradient_convention"):
        assemble_v21_problem(torch, wrong_convention, case_ids=case_ids)

    with pytest.raises(ValueError, match="ridge_lambda must be one of"):
        construct_v21_direction(
            torch,
            records,
            case_ids=case_ids,
            ridge_lambda=0.02,
        )


def test_construction_rejects_a_zero_nuisance_row() -> None:
    records, case_ids = _synthetic_capture()
    # Removing the label component from both option orders for one exact pair
    # makes its raw-A-label nuisance exactly zero.
    for row in records:
        if (
            row["kind"] == "choice"
            and row["case_id"] == case_ids[0]
            and row["target"] == "self"
            and row["assignment"] == 0
        ):
            sign = 1.0 if row["preserve_first"] else -1.0
            role_sign = 1.0
            semantic_base = _basis(12, 0) + 0.2 * role_sign * _basis(12, 5)
            row["gradient"] = (sign * semantic_base).float()

    with pytest.raises(ValueError, match="every v2.1 nuisance row"):
        construct_v21_direction(
            torch,
            records,
            case_ids=case_ids,
            ridge_lambda=0.01,
        )
