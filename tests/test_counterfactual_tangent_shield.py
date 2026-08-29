from __future__ import annotations

import numpy as np
import pytest

from sp_lense.counterfactual_tangent_shield import (
    TangentShieldInfeasibleError,
    build_projected_semantic_anchor_baseline,
    build_seeded_random_null_control,
    solve_minimum_l2_direction,
)


def test_minimum_l2_uses_absolute_offsets_and_vector_margin() -> None:
    result = solve_minimum_l2_direction(
        np.eye(2),
        np.array([-1.0, 2.0]),
        margin=np.array([0.5, 0.25]),
        l2_cap=3.0,
    )

    np.testing.assert_allclose(result.direction, [1.5, 2.25], atol=1e-10, rtol=0.0)
    certificate = result.diagnostics["certificate"]
    assert certificate["passes"] is True
    assert certificate["checks"] == {
        "finite": True,
        "target": True,
        "exact_nuisance": True,
        "soft_nuisance": True,
        "l2_cap": True,
    }
    assert result.direction.flags.writeable is False


def test_exact_nuisance_equality_is_eliminated_in_null_coordinates() -> None:
    result = solve_minimum_l2_direction(
        [[1.0, 0.0], [0.0, 1.0]],
        [1.0, 2.0],
        nuisance_rows=[[1.0, -1.0]],
        nuisance_bound=0.0,
        l2_cap=4.0,
    )

    np.testing.assert_allclose(result.direction, [2.0, 2.0], atol=2e-8, rtol=0.0)
    assert result.diagnostics["svd"]["rank"] == 1
    assert result.diagnostics["reduced_dimension"] == 1
    assert (
        result.diagnostics["certificate"]["maximum_abs_exact_nuisance_residual"]
        <= 2e-8
    )


def test_scalar_soft_nuisance_bound_can_be_active_at_the_optimum() -> None:
    result = solve_minimum_l2_direction(
        [[1.0, 1.0]],
        [1.0],
        nuisance_rows=[[0.0, 1.0]],
        nuisance_bound=0.1,
    )

    np.testing.assert_allclose(result.direction, [0.9, 0.1], atol=2e-8, rtol=0.0)
    certificate = result.diagnostics["certificate"]
    assert certificate["minimum_soft_nuisance_slack"] == pytest.approx(0.0, abs=2e-8)
    assert certificate["checks"]["soft_nuisance"] is True


def test_mixed_exact_and_soft_nuisance_bounds_are_certified() -> None:
    result = solve_minimum_l2_direction(
        [[1.0, 1.0, 0.0]],
        [1.0],
        nuisance_rows=[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        nuisance_bound=[0.0, 0.1],
    )

    np.testing.assert_allclose(result.direction, [0.9, 0.1, 0.0], atol=2e-8, rtol=0.0)
    input_record = result.diagnostics["input_record"]
    assert input_record["exact_nuisance_row_count"] == 1
    assert input_record["soft_nuisance_row_count"] == 1
    assert result.diagnostics["certificate"]["passes"] is True


def test_minimum_norm_above_cap_is_provably_infeasible() -> None:
    with pytest.raises(TangentShieldInfeasibleError, match="exceeds l2_cap"):
        solve_minimum_l2_direction([[1.0]], [1.0], l2_cap=0.5)


def test_full_rank_exact_nuisance_can_make_targets_infeasible() -> None:
    with pytest.raises(TangentShieldInfeasibleError, match="leaves no direction"):
        solve_minimum_l2_direction(
            [[1.0, 0.0]],
            [1.0],
            nuisance_rows=np.eye(2),
            nuisance_bound=0.0,
        )


def test_minimum_l2_output_and_hashes_are_deterministic() -> None:
    kwargs = {
        "margin": 0.05,
        "nuisance_rows": [[1.0, -1.0, 0.0]],
        "nuisance_bound": 0.0,
        "l2_cap": 2.0,
    }
    first = solve_minimum_l2_direction(
        [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
        [0.5, -0.25],
        **kwargs,
    )
    second = solve_minimum_l2_direction(
        [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
        [0.5, -0.25],
        **kwargs,
    )

    assert np.array_equal(first.direction, second.direction)
    assert first.diagnostics["direction_sha256"] == second.diagnostics["direction_sha256"]
    assert first.diagnostics["diagnostics_sha256"] == second.diagnostics["diagnostics_sha256"]


def test_projected_anchor_is_minimally_scaled_without_sign_flip() -> None:
    result = build_projected_semantic_anchor_baseline(
        [1.0, 1.0],
        [[2.0, 0.0]],
        [1.0],
        nuisance_rows=[[0.0, 1.0]],
        nuisance_bound=0.0,
        l2_cap=1.0,
    )

    np.testing.assert_allclose(result.direction, [0.5, 0.0], atol=1e-12, rtol=0.0)
    assert result.diagnostics["selected_scale"] == pytest.approx(0.5)
    assert result.diagnostics["orientation_rule"] == "supplied_anchor_sign_no_posthoc_flip"
    assert result.diagnostics["certificate"]["passes"] is True


def test_projected_anchor_fails_when_orientation_or_soft_bound_is_incompatible() -> None:
    with pytest.raises(TangentShieldInfeasibleError, match="nonpositive slope"):
        build_projected_semantic_anchor_baseline([-1.0, 0.0], [[1.0, 0.0]], [1.0])

    with pytest.raises(TangentShieldInfeasibleError, match="nuisance/norm limits"):
        build_projected_semantic_anchor_baseline(
            [1.0, 1.0],
            [[1.0, 0.0]],
            [1.0],
            nuisance_rows=[[0.0, 1.0]],
            nuisance_bound=0.5,
        )


def test_seeded_random_control_is_repeatable_null_and_norm_matched() -> None:
    nuisance = np.array([[1.0, 1.0, 0.0]])
    first = build_seeded_random_null_control(3, 2.0, seed=17011, nuisance_rows=nuisance)
    second = build_seeded_random_null_control(3, 2.0, seed=17011, nuisance_rows=nuisance)
    different = build_seeded_random_null_control(3, 2.0, seed=17027, nuisance_rows=nuisance)

    assert np.array_equal(first.direction, second.direction)
    assert first.diagnostics["diagnostics_sha256"] == second.diagnostics["diagnostics_sha256"]
    assert not np.array_equal(first.direction, different.direction)
    assert np.linalg.norm(first.direction) == pytest.approx(2.0, abs=2e-8)
    np.testing.assert_allclose(nuisance @ first.direction, [0.0], atol=2e-8, rtol=0.0)
    assert all(first.diagnostics["certificate_checks"].values())


def test_positive_random_control_is_impossible_in_zero_dimensional_null() -> None:
    with pytest.raises(TangentShieldInfeasibleError, match="zero-dimensional"):
        build_seeded_random_null_control(2, 1.0, seed=1, nuisance_rows=np.eye(2))


@pytest.mark.parametrize(
    ("call", "error_type", "message"),
    [
        (
            lambda: solve_minimum_l2_direction([[1.0, np.nan]], [1.0]),
            ValueError,
            "finite",
        ),
        (
            lambda: solve_minimum_l2_direction([[1.0 + 1.0j]], [1.0]),
            TypeError,
            "real numbers",
        ),
        (
            lambda: solve_minimum_l2_direction([[1.0]], [1.0], margin=-0.1),
            ValueError,
            "nonnegative",
        ),
        (
            lambda: solve_minimum_l2_direction(
                [[1.0, 0.0]],
                [1.0],
                nuisance_rows=[[1.0]],
            ),
            ValueError,
            "width",
        ),
        (
            lambda: solve_minimum_l2_direction([[1.0]], [1.0], nuisance_bound=True),
            TypeError,
            "real scalar",
        ),
        (
            lambda: build_seeded_random_null_control(True, 1.0, seed=1),
            TypeError,
            "integer",
        ),
        (
            lambda: build_seeded_random_null_control(2, 1.0, seed=-1),
            ValueError,
            "nonnegative",
        ),
    ],
)
def test_invalid_inputs_fail_closed(call: object, error_type: type[Exception], message: str) -> None:
    with pytest.raises(error_type, match=message):
        call()  # type: ignore[operator]
