from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from sp_lense.gradient_specificity_v3 import (
    DEFAULT_CENTERED_SCORE_IDENTITY_TOLERANCE,
    DEFAULT_SCORE_IDENTITY_TOLERANCE,
    canonical_sha256,
    construct_v3_bidirectional_direction,
    construct_v3_direction,
    minimum_baseline_to_steered_kl_for_ab_shift,
    minimum_changed_to_baseline_kl_for_ab_shift,
    prompt_balanced_topk_tail_fisher_factors,
    row_normalized_svd_basis,
    semantic_margin_constraints,
    solve_min_fisher_qp,
    tensor_float32_sha256,
    tensor_float64_sha256,
    woodbury_h_inverse,
)


def _empty_rows(dimension: int) -> torch.Tensor:
    return torch.empty((0, dimension), dtype=torch.float64)


def _prompt(
    prompt_id: str,
    probabilities: list[float],
    gradients: torch.Tensor,
    tail_probability: float,
    tail_gradient: torch.Tensor,
) -> dict[str, object]:
    return {
        "prompt_id": prompt_id,
        "top_token_ids": [11, 29],
        "top_probabilities": torch.tensor(probabilities, dtype=torch.float64),
        "top_score_gradients": gradients,
        "tail_probability": tail_probability,
        "tail_score_gradient": tail_gradient,
    }


def _float32_roundoff_prompt(
    *,
    total_category_count: int,
    relative_defect: float,
    magnitude: float = 1.0,
    prompt_id: str = "roundoff",
) -> dict[str, object]:
    """Make a float32 score partition with a controlled common-mode defect."""

    dimension = 1024
    simplex = torch.eye(total_category_count, dtype=torch.float64)
    simplex = simplex - torch.full_like(simplex, 1.0 / total_category_count)
    scores = torch.zeros((total_category_count, dimension), dtype=torch.float64)
    scores[:, :total_category_count] = simplex
    simplex_row_norm = math.sqrt(1.0 - 1.0 / total_category_count)
    offset = relative_defect * simplex_row_norm / math.sqrt(1.0 - relative_defect * relative_defect)
    scores[:, total_category_count] = offset
    scores = (magnitude * scores).float()
    probability = 1.0 / total_category_count
    return {
        "prompt_id": prompt_id,
        "top_token_ids": list(range(100, 100 + total_category_count - 1)),
        "top_probabilities": torch.full(
            (total_category_count - 1,),
            probability,
            dtype=torch.float64,
        ),
        "top_score_gradients": scores[:-1],
        "tail_probability": probability,
        "tail_score_gradient": scores[-1],
    }


def test_prompt_balanced_topk_tail_fisher_matches_explicit_average() -> None:
    basis = torch.eye(3, dtype=torch.float64)
    prompts = [
        _prompt(
            "p2",
            [0.25, 0.50],
            basis[[1, 2]],
            0.25,
            -basis[1] - 2.0 * basis[2],
        ),
        _prompt(
            "p1",
            [0.50, 0.25],
            basis[[0, 1]],
            0.25,
            -2.0 * basis[0] - basis[1],
        ),
    ]
    factors, diagnostics = prompt_balanced_topk_tail_fisher_factors(
        torch,
        prompts,
        expected_top_k=2,
    )
    expected = torch.zeros((3, 3), dtype=torch.float64)
    for prompt in prompts:
        probabilities = prompt["top_probabilities"]
        gradients = prompt["top_score_gradients"]
        assert isinstance(probabilities, torch.Tensor)
        assert isinstance(gradients, torch.Tensor)
        for probability, gradient in zip(probabilities, gradients, strict=True):
            expected += probability * torch.outer(gradient, gradient) / len(prompts)
        tail_probability = prompt["tail_probability"]
        tail_gradient = prompt["tail_score_gradient"]
        assert isinstance(tail_probability, float)
        assert isinstance(tail_gradient, torch.Tensor)
        expected += tail_probability * torch.outer(tail_gradient, tail_gradient) / len(prompts)

    reversed_factors, reversed_diagnostics = prompt_balanced_topk_tail_fisher_factors(
        torch,
        list(reversed(prompts)),
        expected_top_k=2,
    )
    assert torch.allclose(factors.T @ factors, expected, atol=1e-14, rtol=0.0)
    assert torch.equal(factors, reversed_factors)
    assert diagnostics["diagnostics_sha256"] == reversed_diagnostics["diagnostics_sha256"]


def test_prompt_balanced_fisher_accepts_variable_required_token_union() -> None:
    basis = torch.eye(3, dtype=torch.float64)
    first = _prompt(
        "two",
        [0.4, 0.4],
        basis[[0, 1]],
        0.2,
        -2.0 * basis[0] - 2.0 * basis[1],
    )
    second = {
        "prompt_id": "three",
        "top_token_ids": [11, 29, 47],
        "top_probabilities": torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64),
        "top_score_gradients": basis,
        "tail_probability": 0.1,
        "tail_score_gradient": -2.0 * basis[0] - 3.0 * basis[1] - 4.0 * basis[2],
    }
    factors, diagnostics = prompt_balanced_topk_tail_fisher_factors(
        torch,
        [first, second],
        expected_top_k=None,
        minimum_top_k=2,
    )

    assert factors.shape == (7, 3)
    assert diagnostics["minimum_category_count"] == 2
    assert diagnostics["maximum_category_count"] == 3

    permuted_second = {
        **second,
        "top_token_ids": [47, 11, 29],
        "top_probabilities": torch.tensor([0.4, 0.2, 0.3], dtype=torch.float64),
        "top_score_gradients": basis[[2, 0, 1]],
    }
    permuted_factors, permuted_diagnostics = prompt_balanced_topk_tail_fisher_factors(
        torch,
        [permuted_second, first],
        expected_top_k=None,
        minimum_top_k=2,
    )
    assert torch.equal(factors, permuted_factors)
    assert diagnostics["diagnostics_sha256"] == permuted_diagnostics["diagnostics_sha256"]


@pytest.mark.parametrize("total_category_count", [9, 10, 11])
def test_float32_score_identity_is_certified_then_recentered(
    total_category_count: int,
) -> None:
    prompt = _float32_roundoff_prompt(
        total_category_count=total_category_count,
        relative_defect=0.5 * DEFAULT_SCORE_IDENTITY_TOLERANCE,
    )
    factors, diagnostics = prompt_balanced_topk_tail_fisher_factors(
        torch,
        [prompt],
        expected_top_k=total_category_count - 1,
    )

    record = diagnostics["prompt_manifest"][0]
    assert factors.shape == (total_category_count, 1024)
    assert factors.dtype == torch.float64
    assert record["raw_score_identity_relative_residual"] < DEFAULT_SCORE_IDENTITY_TOLERANCE
    assert record["raw_weighted_score_mu_norm"] > 0.0
    assert len(record["raw_weighted_score_mean_sha256"]) == 64
    assert record["centered_score_identity_relative_residual"] <= (
        DEFAULT_CENTERED_SCORE_IDENTITY_TOLERANCE
    )
    assert record["raw_score_gradients_sha256"] != record["centered_score_gradients_sha256"]
    assert diagnostics["score_identity_tolerance"] == DEFAULT_SCORE_IDENTITY_TOLERANCE
    assert diagnostics["score_identity_tolerance_kind"] == "relative_weighted_score_norm"
    assert diagnostics["centered_score_identity_tolerance"] == (
        DEFAULT_CENTERED_SCORE_IDENTITY_TOLERANCE
    )

    probability = torch.full(
        (total_category_count,),
        1.0 / total_category_count,
        dtype=torch.float64,
    )
    centered_mean = torch.sqrt(probability) @ factors
    centered_scale = float(
        (torch.sqrt(probability) * torch.linalg.vector_norm(factors, dim=1)).sum().item()
    )
    assert float(torch.linalg.vector_norm(centered_mean).item()) <= (
        DEFAULT_CENTERED_SCORE_IDENTITY_TOLERANCE * centered_scale
    )


def test_score_identity_threshold_is_fail_closed_and_scale_invariant() -> None:
    passing_rhos = []
    for magnitude in (1e-8, 1.0, 1e8):
        _, diagnostics = prompt_balanced_topk_tail_fisher_factors(
            torch,
            [
                _float32_roundoff_prompt(
                    total_category_count=11,
                    relative_defect=0.9 * DEFAULT_SCORE_IDENTITY_TOLERANCE,
                    magnitude=magnitude,
                    prompt_id=f"pass-{magnitude}",
                )
            ],
            expected_top_k=10,
        )
        passing_rhos.append(diagnostics["maximum_raw_score_identity_relative_residual"])
    assert max(passing_rhos) < DEFAULT_SCORE_IDENTITY_TOLERANCE
    assert max(passing_rhos) - min(passing_rhos) < 1e-9

    with pytest.raises(ValueError, match="relative residual"):
        prompt_balanced_topk_tail_fisher_factors(
            torch,
            [
                _float32_roundoff_prompt(
                    total_category_count=11,
                    relative_defect=1.1 * DEFAULT_SCORE_IDENTITY_TOLERANCE,
                )
            ],
            expected_top_k=10,
        )


def test_score_identity_normalizes_valid_partition_and_is_reorder_deterministic() -> None:
    prompt = _float32_roundoff_prompt(
        total_category_count=11,
        relative_defect=0.25 * DEFAULT_SCORE_IDENTITY_TOLERANCE,
    )
    probabilities = prompt["top_probabilities"]
    assert isinstance(probabilities, torch.Tensor)
    probabilities = probabilities.clone()
    probabilities[0] += 0.5e-7
    prompt["top_probabilities"] = probabilities
    factors, diagnostics = prompt_balanced_topk_tail_fisher_factors(
        torch,
        [prompt],
        expected_top_k=10,
    )
    record = diagnostics["prompt_manifest"][0]
    assert record["probability_sum_before_normalization"] != 1.0
    top_gradients = prompt["top_score_gradients"]
    tail_gradient = prompt["tail_score_gradient"]
    assert isinstance(top_gradients, torch.Tensor)
    assert isinstance(tail_gradient, torch.Tensor)
    raw_probabilities = torch.cat(
        (probabilities, torch.tensor([prompt["tail_probability"]], dtype=torch.float64))
    )
    raw_scores = torch.cat(
        (top_gradients.double(), tail_gradient.double().reshape(1, -1)),
        dim=0,
    )
    expected_raw_mean = raw_probabilities @ raw_scores
    processed_probabilities = raw_probabilities / raw_probabilities.sum()
    expected_processed_mean = processed_probabilities @ raw_scores
    assert record["raw_categorical_probabilities_sha256"] == tensor_float64_sha256(
        raw_probabilities
    )
    assert record["processed_categorical_probabilities_sha256"] == tensor_float64_sha256(
        processed_probabilities
    )
    assert (
        record["raw_categorical_probabilities_sha256"]
        != (record["processed_categorical_probabilities_sha256"])
    )
    assert record["raw_weighted_score_mean_sha256"] == tensor_float64_sha256(expected_raw_mean)
    assert record["normalized_weighted_score_mean_sha256"] == tensor_float64_sha256(
        expected_processed_mean
    )
    assert record["raw_weighted_score_mu_norm"] == pytest.approx(
        float(torch.linalg.vector_norm(expected_raw_mean).item()),
        rel=0.0,
        abs=0.0,
    )
    assert record["normalized_weighted_score_mean_norm"] == pytest.approx(
        float(torch.linalg.vector_norm(expected_processed_mean).item()),
        rel=0.0,
        abs=0.0,
    )
    assert record["processed_probability_sum"] == pytest.approx(1.0, abs=2e-16)
    assert (
        record["raw_weighted_score_mean_sha256"]
        != (record["normalized_weighted_score_mean_sha256"])
    )

    reordered = {
        **prompt,
        "top_token_ids": list(reversed(prompt["top_token_ids"])),
        "top_probabilities": probabilities.flip(0),
        "top_score_gradients": prompt["top_score_gradients"].flip(0),
    }
    reordered_factors, reordered_diagnostics = prompt_balanced_topk_tail_fisher_factors(
        torch,
        [reordered],
        expected_top_k=10,
    )
    assert torch.equal(factors, reordered_factors)
    assert diagnostics["diagnostics_sha256"] == reordered_diagnostics["diagnostics_sha256"]


def test_score_identity_rejects_zero_nonfinite_and_uncertified_scores() -> None:
    zero_prompt = _float32_roundoff_prompt(
        total_category_count=9,
        relative_defect=0.0,
    )
    zero_prompt["top_score_gradients"] = torch.zeros((8, 1024), dtype=torch.float32)
    zero_prompt["tail_score_gradient"] = torch.zeros(1024, dtype=torch.float32)
    with pytest.raises(ValueError, match="score scale must be positive"):
        prompt_balanced_topk_tail_fisher_factors(
            torch,
            [zero_prompt],
            expected_top_k=8,
        )

    nonfinite_prompt = _float32_roundoff_prompt(
        total_category_count=9,
        relative_defect=0.0,
    )
    nonfinite_gradients = nonfinite_prompt["top_score_gradients"].clone()
    nonfinite_gradients[0, 0] = float("nan")
    nonfinite_prompt["top_score_gradients"] = nonfinite_gradients
    with pytest.raises(ValueError, match="finite"):
        prompt_balanced_topk_tail_fisher_factors(
            torch,
            [nonfinite_prompt],
            expected_top_k=8,
        )

    with pytest.raises(ValueError, match="less than 1"):
        prompt_balanced_topk_tail_fisher_factors(
            torch,
            [
                _float32_roundoff_prompt(
                    total_category_count=9,
                    relative_defect=0.0,
                )
            ],
            expected_top_k=8,
            score_identity_tolerance=1.0,
        )
    with pytest.raises(ValueError, match="less than score_identity_tolerance"):
        prompt_balanced_topk_tail_fisher_factors(
            torch,
            [
                _float32_roundoff_prompt(
                    total_category_count=9,
                    relative_defect=0.0,
                )
            ],
            expected_top_k=8,
            score_identity_tolerance=1e-4,
            centered_score_identity_tolerance=1e-4,
        )


def test_oriented_information_lower_bounds_match_kl_chain_rules() -> None:
    target_log_odds = torch.log(torch.tensor(3.0, dtype=torch.float64)).item()
    forward_bound, forward_diagnostics = minimum_baseline_to_steered_kl_for_ab_shift(
        baseline_conditional_probability=0.5,
        pair_probability_mass=0.2,
        target_semantic_log_odds=target_log_odds,
    )
    expected = 0.2 * (0.5 * torch.log(torch.tensor(2.0 / 3.0, dtype=torch.float64)).item())
    expected += 0.2 * (0.5 * torch.log(torch.tensor(2.0, dtype=torch.float64)).item())

    assert forward_bound == pytest.approx(expected, abs=1e-14)
    assert forward_diagnostics["kl_orientation"] == "baseline_to_steered"
    assert forward_diagnostics["target_conditional_probability"] == pytest.approx(0.75)

    reverse_bound, reverse_diagnostics = minimum_changed_to_baseline_kl_for_ab_shift(
        baseline_conditional_probability=0.5,
        pair_probability_mass=0.2,
        target_semantic_log_odds=target_log_odds,
    )
    reverse_binary = 0.75 * torch.log(torch.tensor(1.5, dtype=torch.float64)).item()
    reverse_binary += 0.25 * torch.log(torch.tensor(0.5, dtype=torch.float64)).item()
    expected_reverse = -torch.log(
        torch.tensor(0.8, dtype=torch.float64)
        + 0.2 * torch.exp(torch.tensor(-reverse_binary, dtype=torch.float64))
    ).item()
    assert reverse_bound == pytest.approx(expected_reverse, abs=1e-14)
    assert reverse_diagnostics["kl_orientation"] == "changed_to_baseline"

    same_probability, _ = minimum_baseline_to_steered_kl_for_ab_shift(
        baseline_conditional_probability=0.3,
        pair_probability_mass=0.7,
        target_semantic_log_odds=torch.log(torch.tensor(0.3 / 0.7, dtype=torch.float64)).item(),
    )
    assert same_probability == pytest.approx(0.0, abs=1e-14)
    same_reverse, _ = minimum_changed_to_baseline_kl_for_ab_shift(
        baseline_conditional_probability=0.3,
        pair_probability_mass=0.7,
        target_semantic_log_odds=torch.log(torch.tensor(0.3 / 0.7, dtype=torch.float64)).item(),
    )
    assert same_reverse == pytest.approx(0.0, abs=1e-14)
    extreme_reverse, _ = minimum_changed_to_baseline_kl_for_ab_shift(
        baseline_conditional_probability=0.5,
        pair_probability_mass=1.0,
        target_semantic_log_odds=1000.0,
    )
    assert extreme_reverse == pytest.approx(
        torch.log(torch.tensor(2.0, dtype=torch.float64)).item(),
        abs=1e-14,
    )


def test_row_normalized_svd_basis_is_order_and_scale_invariant() -> None:
    basis = torch.eye(4, dtype=torch.float64)
    rows = torch.stack((2.0 * basis[2], -3.0 * basis[2], basis[3]))
    first, first_diagnostics = row_normalized_svd_basis(torch, rows)
    second, second_diagnostics = row_normalized_svd_basis(torch, rows.flip(0))
    expected_projector = torch.diag(torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64))

    assert first_diagnostics["rank"] == 2
    assert torch.allclose(
        first @ first.T,
        torch.eye(2, dtype=torch.float64),
        atol=1e-14,
        rtol=0.0,
    )
    assert torch.allclose(first.T @ first, expected_projector, atol=1e-14, rtol=0.0)
    assert torch.equal(first, second)
    assert first_diagnostics["diagnostics_sha256"] == second_diagnostics["diagnostics_sha256"]


def test_woodbury_inverse_agrees_with_direct_dense_solve() -> None:
    factors = torch.tensor(
        [[0.4, -0.2, 0.1, 0.3], [0.0, 0.5, -0.1, 0.2]],
        dtype=torch.float64,
    )
    vectors = torch.tensor(
        [[1.0, -0.5], [0.3, 0.2], [-0.7, 1.1], [0.4, 0.8]],
        dtype=torch.float64,
    )
    ridge = 0.7
    actual, diagnostics = woodbury_h_inverse(
        torch,
        factors,
        vectors,
        ridge=ridge,
    )
    hessian = ridge * torch.eye(4, dtype=torch.float64) + factors.T @ factors
    expected = torch.linalg.solve(hessian, vectors)
    actual_vector, _ = woodbury_h_inverse(
        torch,
        factors,
        vectors[:, 0],
        ridge=ridge,
    )

    assert torch.allclose(actual, expected, atol=1e-13, rtol=1e-13)
    assert actual_vector.shape == (4,)
    assert torch.allclose(actual_vector, expected[:, 0], atol=1e-13, rtol=1e-13)
    assert diagnostics["inverse_residual"] < 1e-13


def test_qp_certifies_a_single_active_order() -> None:
    inequalities = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    bounds = torch.tensor([1.0, 0.5], dtype=torch.float64)
    perturbation, diagnostics = solve_min_fisher_qp(
        torch,
        inequality_rows=inequalities,
        lower_bounds=bounds,
        nuisance_rows=_empty_rows(4),
        fisher_factors=_empty_rows(4),
        ridge=1.0,
    )

    assert diagnostics["selected_active_orders"] == [0]
    assert torch.allclose(
        perturbation,
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    assert bool(torch.all(inequalities @ perturbation >= bounds - 1e-12))


def test_qp_certifies_both_active_orders_and_nulls_nuisance() -> None:
    basis = torch.eye(4, dtype=torch.float64)
    inequalities = torch.stack((basis[0] + basis[2], basis[1] - basis[2]))
    bounds = torch.tensor([1.0, 2.0], dtype=torch.float64)
    nuisance_rows = torch.stack((2.0 * basis[2], -3.0 * basis[2]))
    perturbation, diagnostics = solve_min_fisher_qp(
        torch,
        inequality_rows=inequalities,
        lower_bounds=bounds,
        nuisance_rows=nuisance_rows,
        fisher_factors=_empty_rows(4),
        ridge=1.0,
    )

    assert diagnostics["selected_active_orders"] == [0, 1]
    assert torch.allclose(
        perturbation,
        torch.tensor([1.0, 2.0, 0.0, 0.0], dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    assert torch.max(torch.abs(nuisance_rows @ perturbation)).item() < 1e-12
    assert torch.allclose(inequalities @ perturbation, bounds, atol=1e-12, rtol=0.0)


def test_qp_matches_direct_dense_active_set_solution() -> None:
    factors = torch.tensor(
        [[0.2, -0.3, 0.4, 0.1, 0.0], [0.1, 0.2, 0.0, -0.4, 0.3]],
        dtype=torch.float64,
    )
    nuisance_rows = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    inequalities = torch.tensor(
        [[1.0, 0.2, 0.5, 0.0, 0.0], [0.1, 1.0, -0.3, 0.0, 0.2]],
        dtype=torch.float64,
    )
    bounds = torch.tensor([0.8, 1.1], dtype=torch.float64)
    ridge = 0.6
    actual, diagnostics = solve_min_fisher_qp(
        torch,
        inequality_rows=inequalities,
        lower_bounds=bounds,
        nuisance_rows=nuisance_rows,
        fisher_factors=factors,
        ridge=ridge,
    )

    nuisance_basis, _ = row_normalized_svd_basis(torch, nuisance_rows)
    projected = inequalities - (inequalities @ nuisance_basis.T) @ nuisance_basis
    active = diagnostics["selected_active_orders"]
    constraints = torch.cat((nuisance_basis, projected[active]), dim=0)
    right_hand_side = torch.cat(
        (torch.zeros(nuisance_basis.shape[0], dtype=torch.float64), bounds[active])
    )
    hessian = ridge * torch.eye(5, dtype=torch.float64) + factors.T @ factors
    hessian_inverse_constraints = torch.linalg.solve(hessian, constraints.T)
    expected = hessian_inverse_constraints @ torch.linalg.solve(
        constraints @ hessian_inverse_constraints,
        right_hand_side,
    )

    assert torch.allclose(actual, expected, atol=1e-11, rtol=1e-11)
    assert torch.max(torch.abs(nuisance_rows @ actual)).item() < 1e-10
    assert bool(torch.all(inequalities @ actual >= bounds - 1e-10))


def test_construct_one_sided_direction_reaches_both_linearized_boundaries() -> None:
    gradients = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    baselines = torch.tensor([-0.5, -1.0], dtype=torch.float64)
    nuisance_rows = torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float64)
    direction, native_norm, diagnostics = construct_v3_direction(
        torch,
        self_semantic_gradients=gradients,
        baseline_semantic_log_odds=baselines,
        attack_sign=1,
        nuisance_rows=nuisance_rows,
        fisher_factors=_empty_rows(4),
        ridge=1.0,
    )
    perturbation = direction.double() * native_norm

    assert direction.dtype == torch.float32
    assert direction.device.type == "cpu"
    assert direction.is_contiguous()
    assert float(torch.linalg.vector_norm(direction.double())) == pytest.approx(1.0, abs=1e-7)
    assert native_norm == pytest.approx(5**0.5 / 2.0, abs=1e-12)
    assert torch.allclose(
        baselines + gradients @ perturbation,
        torch.zeros(2, dtype=torch.float64),
        atol=1e-7,
        rtol=0.0,
    )
    assert torch.max(torch.abs(nuisance_rows @ perturbation)).item() < 1e-12
    assert diagnostics["direction_float32_sha256"] == tensor_float32_sha256(direction)
    assert len(diagnostics["diagnostics_sha256"]) == 64


def test_bidirectional_direction_crosses_both_orders_under_both_signs() -> None:
    gradients = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    baselines = torch.tensor([0.4, -0.7], dtype=torch.float64)
    nuisance_rows = torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float64)
    direction, native_norm, diagnostics = construct_v3_bidirectional_direction(
        torch,
        self_semantic_gradients=gradients,
        baseline_semantic_log_odds=baselines,
        nuisance_rows=nuisance_rows,
        fisher_factors=_empty_rows(4),
        ridge=1.0,
        decision_margin=0.05,
    )
    perturbation = native_norm * direction.double()
    positive_predictions = baselines + gradients @ perturbation
    negative_predictions = baselines - gradients @ perturbation

    assert native_norm == pytest.approx((0.45**2 + 0.75**2) ** 0.5, abs=1e-12)
    assert float(positive_predictions.min().item()) > 0.049999
    assert float(negative_predictions.max().item()) < -0.049999
    assert diagnostics["bidirectional_constraints"]["mode"] == "symmetric_bidirectional"
    assert diagnostics["qp"]["selected_active_orders"] == [0, 1]
    assert diagnostics["post_cast_certificate"]["minimum_inequality_slack"] > -1e-6
    assert len(diagnostics["diagnostics_sha256"]) == 64


def test_qp_and_direction_are_deterministic_under_nuisance_and_fisher_reordering() -> None:
    basis = torch.eye(5, dtype=torch.float64)
    gradients = torch.stack((basis[0] + 0.2 * basis[4], basis[1] - 0.1 * basis[4]))
    baselines = torch.tensor([-0.4, -0.7], dtype=torch.float64)
    nuisances = torch.stack((basis[2], basis[3], 2.0 * basis[2]))
    factors = torch.stack((0.3 * basis[0] + basis[4], basis[1] - 0.2 * basis[4]))
    first = construct_v3_direction(
        torch,
        self_semantic_gradients=gradients,
        baseline_semantic_log_odds=baselines,
        attack_sign=1,
        nuisance_rows=nuisances,
        fisher_factors=factors,
        ridge=0.4,
    )
    second = construct_v3_direction(
        torch,
        self_semantic_gradients=gradients,
        baseline_semantic_log_odds=baselines,
        attack_sign=1,
        nuisance_rows=nuisances.flip(0),
        fisher_factors=factors.flip(0),
        ridge=0.4,
    )

    assert torch.equal(first[0], second[0])
    assert first[1] == second[1]
    assert first[2]["diagnostics_sha256"] == second[2]["diagnostics_sha256"]


def test_solver_fails_closed_for_infeasible_or_ill_conditioned_inputs() -> None:
    basis = torch.eye(3, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="infeasible"):
        solve_min_fisher_qp(
            torch,
            inequality_rows=torch.stack((basis[0], basis[1])),
            lower_bounds=torch.ones(2, dtype=torch.float64),
            nuisance_rows=torch.stack((basis[0], basis[1])),
            fisher_factors=_empty_rows(3),
            ridge=1.0,
        )

    nearly_singular_factors = torch.stack((basis[0], basis[0]))
    with pytest.raises(RuntimeError, match="condition number"):
        solve_min_fisher_qp(
            torch,
            inequality_rows=torch.stack((basis[1], basis[2])),
            lower_bounds=torch.ones(2, dtype=torch.float64),
            nuisance_rows=_empty_rows(3),
            fisher_factors=nearly_singular_factors,
            ridge=1e-12,
            condition_limit=1e6,
        )


def test_validation_rejects_malformed_or_ambiguous_inputs() -> None:
    basis = torch.eye(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="zero or below atol"):
        row_normalized_svd_basis(torch, torch.zeros((1, 3), dtype=torch.float64))
    with pytest.raises(RuntimeError, match="zero-rank"):
        row_normalized_svd_basis(
            torch,
            torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float64),
            atol=2.0,
        )
    with pytest.raises(ValueError, match="not 1"):
        prompt_balanced_topk_tail_fisher_factors(
            torch,
            [_prompt("bad", [0.2, 0.2], basis[[0, 1]], 0.2, basis[2])],
            expected_top_k=2,
        )
    invalid_score_prompt = _prompt(
        "bad_score",
        [0.4, 0.4],
        basis[[0, 1]],
        0.2,
        basis[2],
    )
    with pytest.raises(ValueError, match="weighted score mean"):
        prompt_balanced_topk_tail_fisher_factors(
            torch,
            [invalid_score_prompt],
            expected_top_k=2,
        )
    negative_token_prompt = _prompt(
        "negative_token",
        [0.4, 0.4],
        basis[[0, 1]],
        0.2,
        -2.0 * basis[0] - 2.0 * basis[1],
    )
    negative_token_prompt["top_token_ids"] = [-1, 29]
    with pytest.raises(ValueError, match="unique integers"):
        prompt_balanced_topk_tail_fisher_factors(
            torch,
            [negative_token_prompt],
            expected_top_k=2,
        )
    with pytest.raises(ValueError, match="opposite"):
        semantic_margin_constraints(
            torch,
            basis[[0, 1]],
            torch.tensor([-0.5, 0.1], dtype=torch.float64),
            attack_sign=1,
        )
    with pytest.raises(ValueError, match="decision_margin"):
        construct_v3_bidirectional_direction(
            torch,
            self_semantic_gradients=basis[[0, 1]],
            baseline_semantic_log_odds=torch.tensor([0.1, -0.1]),
            nuisance_rows=_empty_rows(3),
            fisher_factors=_empty_rows(3),
            ridge=1.0,
            decision_margin=0.0,
        )
    with pytest.raises(RuntimeError, match="too small"):
        construct_v3_bidirectional_direction(
            torch,
            self_semantic_gradients=basis[[0, 1]],
            baseline_semantic_log_odds=torch.tensor([0.4, -0.7]),
            nuisance_rows=_empty_rows(3),
            fisher_factors=_empty_rows(3),
            ridge=1.0,
            decision_margin=1e-12,
        )
    with pytest.raises(ValueError, match="positive"):
        solve_min_fisher_qp(
            torch,
            inequality_rows=basis[[0, 1]],
            lower_bounds=torch.ones(2, dtype=torch.float64),
            nuisance_rows=_empty_rows(3),
            fisher_factors=_empty_rows(3),
            ridge=0.0,
        )
    with pytest.raises(TypeError, match="floating-point"):
        solve_min_fisher_qp(
            torch,
            inequality_rows=torch.ones((2, 3), dtype=torch.int64),
            lower_bounds=torch.ones(2, dtype=torch.float64),
            nuisance_rows=_empty_rows(3),
            fisher_factors=_empty_rows(3),
            ridge=1.0,
        )
    with pytest.raises(ValueError, match="finite"):
        woodbury_h_inverse(
            torch,
            _empty_rows(3),
            torch.tensor([float("nan"), 0.0, 0.0]),
            ridge=1.0,
        )
    with pytest.raises(ValueError, match="baseline_conditional_probability"):
        minimum_baseline_to_steered_kl_for_ab_shift(
            baseline_conditional_probability=1.1,
            pair_probability_mass=0.2,
            target_semantic_log_odds=0.0,
        )
    with pytest.raises(ValueError, match="pair_probability_mass"):
        minimum_baseline_to_steered_kl_for_ab_shift(
            baseline_conditional_probability=0.5,
            pair_probability_mass=0.0,
            target_semantic_log_odds=0.0,
        )
    with pytest.raises(ValueError, match="strictly between"):
        minimum_changed_to_baseline_kl_for_ab_shift(
            baseline_conditional_probability=1.0,
            pair_probability_mass=0.2,
            target_semantic_log_odds=0.0,
        )

    assert len(canonical_sha256({"b": 2, "a": 1})) == 64
    assert len(tensor_float64_sha256(basis)) == 64
