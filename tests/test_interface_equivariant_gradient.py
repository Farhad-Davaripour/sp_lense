from __future__ import annotations

import numpy as np
import pytest

from sp_lense.interface_equivariant_gradient import (
    cast_and_recertify_interface_deltas,
    certify_exact_rmsnorm_head_shared_alpha,
    certify_exact_rmsnorm_head_shared_alpha_from_numerators,
    construct_effective_unembedding_field,
    construct_interface_equivariant_field,
    exact_rmsnorm_semantic_gradients,
    exact_rmsnorm_semantic_gradients_from_boundaries,
    exact_rmsnorm_unembedding_logits,
    recertify_exact_head_deltas,
)
from sp_lense.paired_order_analytic_gradient import (
    PairedOrderGradientIneligible,
    full_vocabulary_bidirectional_interval,
)


def _antipodal_example():
    gradients = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float64)
    residual_norms = np.array([1.0, 1.0], dtype=np.float64)
    logits = np.array(
        [
            [-1.0, 1.0, -5.0],
            [1.0, -1.0, -5.0],
        ],
        dtype=np.float64,
    )
    derivatives = np.array(
        [
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    return gradients, residual_norms, logits, derivatives, (0, 1), (1, 0)


def test_equivariant_field_accepts_exactly_antipodal_physical_gradients() -> None:
    gradients, residual_norms, *_ = _antipodal_example()

    field = construct_interface_equivariant_field(gradients, residual_norms)

    np.testing.assert_allclose(field.base_vectors, [[1.0, 0.0], [-1.0, 0.0]])
    assert field.semantic_slopes == pytest.approx([1.0, 1.0])
    assert field.diagnostics["order_gradient_cosine_descriptive"] == pytest.approx(-1.0)
    assert field.diagnostics["physical_vector_equality_required"] is False
    assert field.diagnostics["shared_scalar_coordinate_required"] is True


def test_joint_interval_provides_one_scalar_for_two_interface_vectors() -> None:
    _, _, logits, derivatives, preserve, comply = _antipodal_example()

    interval = full_vocabulary_bidirectional_interval(
        logits,
        derivatives,
        preserve,
        comply,
        reserve_logit=0.1,
    )

    assert interval.lower == pytest.approx(1.05)


def test_cast_recertification_keeps_one_alpha_but_two_physical_deltas() -> None:
    gradients, residual_norms, logits, derivatives, preserve, comply = _antipodal_example()
    field = construct_interface_equivariant_field(gradients, residual_norms)
    alpha = 1.05

    result = cast_and_recertify_interface_deltas(
        field.base_vectors,
        alpha,
        residual_norms,
        logits,
        derivatives,
        alpha * derivatives,
        preserve,
        comply,
        reserve_logit=0.1,
        maximum_relative_norm=2.0,
    )

    assert result.deltas.dtype == np.float32
    np.testing.assert_allclose(result.deltas, [[1.05, 0.0], [-1.05, 0.0]])
    assert result.diagnostics["alpha"] == pytest.approx(alpha)
    assert result.diagnostics["relative_norms"] == pytest.approx([1.05, 1.05])
    assert len(set(result.diagnostics["per_order_delta_float32_sha256"])) == 2
    assert result.diagnostics["minimum_target_margin"] == pytest.approx(0.1)


def test_cast_recertification_rejects_one_inconsistent_interface_jvp() -> None:
    gradients, residual_norms, logits, derivatives, preserve, comply = _antipodal_example()
    field = construct_interface_equivariant_field(gradients, residual_norms)
    changed = 1.05 * derivatives
    changed[1, 0] += 0.1

    with pytest.raises(PairedOrderGradientIneligible, match="shared-scalar") as error:
        cast_and_recertify_interface_deltas(
            field.base_vectors,
            1.05,
            residual_norms,
            logits,
            derivatives,
            changed,
            preserve,
            comply,
            reserve_logit=0.1,
            maximum_relative_norm=2.0,
        )
    assert error.value.diagnostics["failure"] == "cast_jvp_inconsistency"


def test_cast_recertification_enforces_each_order_norm_cap() -> None:
    gradients, residual_norms, logits, derivatives, preserve, comply = _antipodal_example()
    field = construct_interface_equivariant_field(gradients, residual_norms)

    with pytest.raises(PairedOrderGradientIneligible, match="norm cap") as error:
        cast_and_recertify_interface_deltas(
            field.base_vectors,
            1.05,
            residual_norms,
            logits,
            derivatives,
            1.05 * derivatives,
            preserve,
            comply,
            reserve_logit=0.1,
            maximum_relative_norm=1.0,
        )
    assert error.value.diagnostics["failure"] == "relative_norm_cap_exceeded"


@pytest.mark.parametrize(
    "gradients",
    (
        [[0.0, 0.0], [1.0, 0.0]],
        [[float("nan"), 0.0], [1.0, 0.0]],
    ),
)
def test_equivariant_field_fails_closed_on_invalid_gradients(gradients) -> None:
    with pytest.raises((PairedOrderGradientIneligible, ValueError)):
        construct_interface_equivariant_field(gradients, [1.0, 1.0])


def _exact_head_example():
    residuals = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    # Tokens 0 and 1 form the answer boundary; token 2 is a weaker competitor.
    effective_unembedding = np.array([[1.0, -1.0, 0.0], [0.0, 0.0, -2.0]], dtype=np.float64)
    preserve = (0, 1)
    comply = (1, 0)
    residual_norms = np.linalg.norm(residuals, axis=1)
    field = construct_effective_unembedding_field(
        effective_unembedding,
        residual_norms,
        preserve,
        comply,
    )
    return residuals, effective_unembedding, preserve, comply, field


def test_effective_unembedding_baseline_transforms_with_answer_order() -> None:
    _, _, _, _, field = _exact_head_example()

    np.testing.assert_allclose(field.base_vectors, [[1.0, 0.0], [-1.0, 0.0]])
    assert field.diagnostics["requires_backward_pass"] is False
    assert field.diagnostics["order_gradient_cosine_descriptive"] == pytest.approx(-1.0)


def test_exact_head_solver_finds_one_shared_bidirectional_alpha() -> None:
    residuals, weights, preserve, comply, field = _exact_head_example()

    certificate = certify_exact_rmsnorm_head_shared_alpha(
        residuals,
        field.base_vectors,
        weights,
        preserve,
        comply,
        rms_epsilon=1.0,
        construction_reserve_logit=0.1,
        maximum_relative_norm=1.0,
    )

    # rho_max=sqrt((2*1)^2/2 + 1)=sqrt(3); token-boundary slope is 2.
    assert certificate.alpha == pytest.approx(0.1 * np.sqrt(3.0) / 2.0)
    assert certificate.upper == pytest.approx(1.0)
    assert certificate.diagnostics["constraint_count"] == 8
    assert certificate.diagnostics["selection_rule"].startswith("smallest")


def test_memory_bounded_numerator_solver_matches_full_weight_solver() -> None:
    residuals, weights, preserve, comply, field = _exact_head_example()
    expected = certify_exact_rmsnorm_head_shared_alpha(
        residuals,
        field.base_vectors,
        weights,
        preserve,
        comply,
        rms_epsilon=1.0,
        construction_reserve_logit=0.1,
        maximum_relative_norm=1.0,
    )

    observed = certify_exact_rmsnorm_head_shared_alpha_from_numerators(
        residuals @ weights,
        field.base_vectors @ weights,
        np.linalg.norm(residuals, axis=1),
        np.linalg.norm(field.base_vectors, axis=1),
        preserve,
        comply,
        residual_width=2,
        rms_epsilon=1.0,
        construction_reserve_logit=0.1,
        maximum_relative_norm=1.0,
    )

    assert observed.alpha == pytest.approx(expected.alpha)
    assert observed.lower == pytest.approx(expected.lower)
    assert observed.upper == pytest.approx(expected.upper)


def test_numerator_solver_uses_actual_direction_norms_for_the_joint_cap() -> None:
    residuals, weights, preserve, comply, field = _exact_head_example()

    observed = certify_exact_rmsnorm_head_shared_alpha_from_numerators(
        residuals @ weights,
        field.base_vectors @ weights,
        [1.0, 1.0],
        [2.0, 2.0],
        preserve,
        comply,
        residual_width=2,
        rms_epsilon=1.0,
        construction_reserve_logit=0.1,
        maximum_relative_norm=1.0,
    )

    assert observed.upper == pytest.approx(0.5)
    assert observed.diagnostics["maximum_alpha_from_relative_norm_cap"] == pytest.approx(0.5)
    assert observed.diagnostics["direction_norms"] == [2.0, 2.0]


def test_exact_head_recertification_checks_both_orders_and_exact_negation() -> None:
    residuals, weights, preserve, comply, field = _exact_head_example()
    certificate = certify_exact_rmsnorm_head_shared_alpha(
        residuals,
        field.base_vectors,
        weights,
        preserve,
        comply,
        rms_epsilon=1.0,
        construction_reserve_logit=0.11,
        maximum_relative_norm=1.0,
    )
    deltas = (certificate.alpha * field.base_vectors).astype(np.float32)

    diagnostics = recertify_exact_head_deltas(
        residuals,
        deltas,
        weights,
        preserve,
        comply,
        rms_epsilon=1.0,
        acceptance_reserve_logit=0.1,
    )

    assert len(diagnostics["rows"]) == 4
    assert all(row["target_met"] for row in diagnostics["rows"])
    assert diagnostics["minimum_target_margin"] >= 0.1
    assert diagnostics["relative_norms"] == pytest.approx([certificate.alpha, certificate.alpha])
    assert diagnostics["deltas_float32_sha256"] != diagnostics["negative_deltas_float32_sha256"]


def test_exact_head_solver_fails_when_the_norm_cap_cannot_reach_target() -> None:
    residuals, weights, preserve, comply, field = _exact_head_example()

    with pytest.raises(PairedOrderGradientIneligible, match="no shared") as error:
        certify_exact_rmsnorm_head_shared_alpha(
            residuals,
            field.base_vectors,
            weights,
            preserve,
            comply,
            rms_epsilon=1.0,
            construction_reserve_logit=10.0,
            maximum_relative_norm=0.01,
        )
    assert error.value.diagnostics["failure"] == "empty_shared_exact_head_interval"


def test_exact_head_logits_reject_wrong_width() -> None:
    with pytest.raises(ValueError, match="wrong residual width"):
        exact_rmsnorm_unembedding_logits(
            [[1.0, 2.0]],
            np.ones((3, 4)),
            rms_epsilon=1e-6,
        )


def test_exact_head_solver_never_uses_tolerance_to_exceed_cap() -> None:
    baseline = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    # Both signs require alpha=1+5e-13, just beyond the cap. The old tolerance
    # logic incorrectly accepted this and returned an out-of-cap certificate.
    direction = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
    reserve = 1.0 + 5e-13
    with pytest.raises(PairedOrderGradientIneligible, match="no shared"):
        certify_exact_rmsnorm_head_shared_alpha_from_numerators(
            baseline,
            direction,
            [1.0, 1.0],
            [1.0, 1.0],
            (0, 1),
            (1, 0),
            residual_width=1,
            rms_epsilon=1e-30,
            construction_reserve_logit=reserve,
            maximum_relative_norm=1.0,
        )


def test_exact_head_solver_never_ignores_small_adverse_slope_next_to_huge_slope() -> None:
    cap = 0.6
    epsilon = 0.01
    rho_max = np.sqrt((1.0 + cap) ** 2 / 4.0 + epsilon)
    baseline = np.array(
        [[0.0, 0.0, -1.0, -1e12], [0.0, 0.0, -1.0, -1e12]],
        dtype=np.float64,
    )
    direction = np.array(
        [[1.0, -1.0, 1.5, -1e12], [1.0, -1.0, 1.5, -1e12]],
        dtype=np.float64,
    )

    # Target 0 versus token 1 requires alpha >= 0.5, while target 0 versus
    # token 2 has a smaller adverse slope and requires alpha <= 0. A former
    # row-relative zero threshold hid the latter beside the 1e12 slope.
    with pytest.raises(PairedOrderGradientIneligible, match="no shared"):
        certify_exact_rmsnorm_head_shared_alpha_from_numerators(
            baseline,
            direction,
            [1.0, 1.0],
            [1.0, 1.0],
            (0, 0),
            (1, 1),
            residual_width=4,
            rms_epsilon=epsilon,
            construction_reserve_logit=1.0 / rho_max,
            maximum_relative_norm=cap,
        )


def test_exact_head_solver_fails_closed_when_finite_inputs_overflow_on_subtraction() -> None:
    baseline = np.zeros((2, 2), dtype=np.float64)
    direction = np.array([[1e308, -1e308], [1e308, -1e308]], dtype=np.float64)

    with pytest.raises(PairedOrderGradientIneligible, match="overflowed") as error:
        certify_exact_rmsnorm_head_shared_alpha_from_numerators(
            baseline,
            direction,
            [1.0, 1.0],
            [1.0, 1.0],
            (0, 0),
            (1, 1),
            residual_width=4,
            rms_epsilon=1.0,
            construction_reserve_logit=0.1,
            maximum_relative_norm=0.1,
        )
    assert error.value.diagnostics["failure"] == "nonfinite_derived_exact_head_constraint"


def test_exact_head_solver_fails_closed_when_required_bound_underflows_to_zero() -> None:
    baseline = np.zeros((2, 2), dtype=np.float64)
    direction = np.array([[5e307, -5e307], [5e307, -5e307]], dtype=np.float64)
    cap = 0.1
    epsilon = 1.0
    rho_max = np.sqrt((1.0 + cap) ** 2 / 4.0 + epsilon)

    with pytest.raises(PairedOrderGradientIneligible, match="unrepresentable") as error:
        certify_exact_rmsnorm_head_shared_alpha_from_numerators(
            baseline,
            direction,
            [1.0, 1.0],
            [1.0, 1.0],
            (0, 0),
            (1, 1),
            residual_width=4,
            rms_epsilon=epsilon,
            construction_reserve_logit=1e-308 / rho_max,
            maximum_relative_norm=cap,
        )
    assert error.value.diagnostics["failure"] == "nonfinite_exact_head_interval_bound"


def test_exact_head_recertification_rejects_non_float32_delta() -> None:
    residuals, weights, preserve, comply, field = _exact_head_example()
    with pytest.raises(TypeError, match="exact float32"):
        recertify_exact_head_deltas(
            residuals,
            0.1 * field.base_vectors,
            weights,
            preserve,
            comply,
            rms_epsilon=1.0,
            acceptance_reserve_logit=0.01,
        )


def test_exact_head_recertification_fails_closed_when_float32_cap_rounds_up() -> None:
    residuals = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    deltas = np.array([[0.1, 0.0], [0.1, 0.0]], dtype=np.float32)
    weights = np.array([[1.0, -1.0], [0.0, 0.0]], dtype=np.float64)

    with pytest.raises(PairedOrderGradientIneligible, match="norm cap") as error:
        recertify_exact_head_deltas(
            residuals,
            deltas,
            weights,
            (0, 0),
            (1, 1),
            rms_epsilon=1.0,
            acceptance_reserve_logit=0.01,
            maximum_relative_norm=0.1,
        )
    assert error.value.diagnostics["failure"] == ("exact_head_recertification_norm_cap_exceeded")


def test_analytic_rmsnorm_gradient_matches_finite_difference() -> None:
    residuals = np.array([[0.3, -0.7], [0.4, 0.2]], dtype=np.float64)
    weights = np.array([[0.9, -0.2, 0.1], [0.3, 0.8, -0.4]], dtype=np.float64)
    preserve = (0, 1)
    comply = (1, 0)
    epsilon = 0.2

    gradients = exact_rmsnorm_semantic_gradients(
        residuals,
        weights,
        preserve,
        comply,
        rms_epsilon=epsilon,
    )

    step = 1e-6
    for order in range(2):
        finite = np.zeros(2)
        for coordinate in range(2):
            offset = np.zeros(2)
            offset[coordinate] = step
            plus = exact_rmsnorm_unembedding_logits(
                (residuals[order] + offset)[None, :], weights, rms_epsilon=epsilon
            )[0]
            minus = exact_rmsnorm_unembedding_logits(
                (residuals[order] - offset)[None, :], weights, rms_epsilon=epsilon
            )[0]
            objective_plus = plus[preserve[order]] - plus[comply[order]]
            objective_minus = minus[preserve[order]] - minus[comply[order]]
            finite[coordinate] = (objective_plus - objective_minus) / (2.0 * step)
        np.testing.assert_allclose(gradients[order], finite, rtol=1e-7, atol=1e-7)

    boundaries = np.stack(
        [weights[:, preserve[index]] - weights[:, comply[index]] for index in range(2)]
    )
    bounded = exact_rmsnorm_semantic_gradients_from_boundaries(
        residuals,
        boundaries,
        rms_epsilon=epsilon,
    )
    np.testing.assert_array_equal(bounded, gradients)
