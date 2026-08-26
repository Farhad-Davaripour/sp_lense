from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sp_lense.gradient_specificity_adaptive import (
    RAW_GRADIENT_CONVENTION,
    construct_adaptive_direction,
    construct_adaptive_direction_bank,
    evaluate_adaptive_direction,
    lookup_adaptive_direction,
    semantic_ab_gradient,
    tensor_float32_sha256,
)


def _row(
    case_id: str,
    assignment: int,
    target: str,
    preserve_first: bool,
    gradient: torch.Tensor,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "assignment": assignment,
        "target": target,
        "kind": "choice",
        "preserve_first": preserve_first,
        "gradient_convention": RAW_GRADIENT_CONVENTION,
        "gradient": gradient,
    }


def _quartet(case_id: str, assignment: int) -> list[dict[str, object]]:
    basis = torch.eye(8, dtype=torch.float32)
    signal = (2.0 + 0.1 * assignment) * basis[0]
    self_gap = basis[3]
    self_a = signal + self_gap
    self_b = signal - self_gap
    other_a = basis[1] + basis[2]
    other_b = 2.0 * basis[1] - basis[2]
    return [
        _row(case_id, assignment, "self", True, self_a),
        _row(case_id, assignment, "self", False, -self_b),
        _row(case_id, assignment, "other", True, other_a),
        _row(case_id, assignment, "other", False, -other_b),
    ]


def test_semantic_ab_gradient_applies_exactly_one_order_sign() -> None:
    raw = torch.tensor([1.0, -2.0])
    assert torch.equal(
        semantic_ab_gradient(torch, raw, preserve_first=True),
        raw.double(),
    )
    assert torch.equal(
        semantic_ab_gradient(torch, raw, preserve_first=False),
        -raw.double(),
    )
    with pytest.raises(TypeError, match="bool"):
        semantic_ab_gradient(torch, raw, preserve_first=1)  # type: ignore[arg-type]


def test_exact_svd_direction_nulls_other_and_label_but_moves_both_self_orders() -> None:
    records = _quartet("case_a", 0)
    direction, diagnostics = construct_adaptive_direction(
        torch,
        list(reversed(records)),
        case_id="case_a",
        assignment=0,
    )
    assert direction.dtype == torch.float32
    assert direction.device.type == "cpu"
    assert direction.is_contiguous()
    assert float(torch.linalg.vector_norm(direction)) == pytest.approx(1.0, abs=1e-7)
    assert torch.allclose(direction, torch.eye(8)[0], atol=1e-6, rtol=0.0)

    result = evaluate_adaptive_direction(
        torch,
        direction,
        records,
        case_id="case_a",
        assignment=0,
    )
    assert result["same_direction_applied_to_all_four_cells"] is True
    assert result["both_self_orders_positive"] is True
    assert result["self_semantic_projections"] == pytest.approx(
        {"preserve_A": 2.0, "preserve_B": 2.0},
        abs=1e-6,
    )
    assert result["maximum_abs_matched_other_projection"] < 1e-7
    assert result["maximum_abs_nuisance_projection"] < 1e-7
    assert diagnostics["nuisance_rows"] == 5
    assert diagnostics["nuisance_rank"] == 3
    assert diagnostics["float32_projection_summary"]["both_self_orders_positive"] is True
    assert diagnostics["direction_float32_sha256"] == tensor_float32_sha256(direction)
    assert len(diagnostics["diagnostics_sha256"]) == 64


def test_bank_is_order_independent_and_target_cannot_select_a_different_direction() -> None:
    records = [
        *_quartet("case_b", 1),
        *_quartet("case_a", 0),
        *_quartet("case_b", 0),
        *_quartet("case_a", 1),
    ]
    first = construct_adaptive_direction_bank(
        torch,
        records,
        case_ids=["case_b", "case_a"],
    )
    second = construct_adaptive_direction_bank(
        torch,
        list(reversed(records)),
        case_ids=["case_a", "case_b"],
    )
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert len(first["entries"]) == 4

    self_direction = lookup_adaptive_direction(
        first,
        case_id="case_a",
        assignment=1,
        target="self",
    )
    other_direction = lookup_adaptive_direction(
        first,
        case_id="case_a",
        assignment=1,
        target="other",
    )
    assert self_direction is other_direction
    assert self_direction.data_ptr() == other_direction.data_ptr()
    with pytest.raises(ValueError, match="target"):
        lookup_adaptive_direction(
            first,
            case_id="case_a",
            assignment=1,
            target="unmatched",
        )


def test_constructor_fails_closed_on_incomplete_duplicate_or_malformed_cells() -> None:
    records = _quartet("case_a", 0)
    with pytest.raises(ValueError, match="four-cell coverage"):
        construct_adaptive_direction(
            torch,
            records[:-1],
            case_id="case_a",
            assignment=0,
        )
    with pytest.raises(ValueError, match="duplicate"):
        construct_adaptive_direction(
            torch,
            [*records, records[0]],
            case_id="case_a",
            assignment=0,
        )

    wrong_convention = [dict(row) for row in records]
    wrong_convention[0]["gradient_convention"] = "semantic_preserve_minus_comply"
    with pytest.raises(ValueError, match="gradient_convention"):
        construct_adaptive_direction(
            torch,
            wrong_convention,
            case_id="case_a",
            assignment=0,
        )

    nonfinite = [dict(row) for row in records]
    nonfinite[0]["gradient"] = torch.tensor(
        [float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    with pytest.raises(ValueError, match="finite"):
        construct_adaptive_direction(
            torch,
            nonfinite,
            case_id="case_a",
            assignment=0,
        )

    all_records = [*_quartet("case_a", 0), *_quartet("case_a", 1)]
    with pytest.raises(ValueError, match="case-assignment coverage"):
        construct_adaptive_direction_bank(
            torch,
            all_records[:4],
            case_ids=["case_a"],
        )


def test_constructor_rejects_signal_fully_contained_in_nuisance_span() -> None:
    basis = torch.eye(5)
    records = [
        _row("degenerate", 0, "self", True, basis[0]),
        _row("degenerate", 0, "self", False, -basis[0]),
        _row("degenerate", 0, "other", True, basis[0]),
        _row("degenerate", 0, "other", False, -basis[1]),
    ]
    with pytest.raises(RuntimeError, match="outside the nuisance span"):
        construct_adaptive_direction(
            torch,
            records,
            case_id="degenerate",
            assignment=0,
        )
