from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import sp_lense.global_counterfactual_robust_boundary as gcrbs
from sp_lense.global_counterfactual_robust_boundary import (
    GCRBSInfeasibleError,
    GCRBSSolverError,
    solve_global_counterfactual_robust_boundary,
)


def test_feasible_max_min_optimum_has_certified_canonical_orientation() -> None:
    solution = solve_global_counterfactual_robust_boundary(
        target_matrix=[[1.0, 0.0], [0.0, 1.0]],
        target_offsets=[0.0, 0.0],
        l2_cap=1.0,
    )

    expected = np.full(2, 1.0 / np.sqrt(2.0))
    assert solution.direction.dtype == np.float64
    assert np.allclose(solution.direction, expected, atol=3e-8, rtol=0.0)
    assert solution.gamma == pytest.approx(1.0 / np.sqrt(2.0), abs=3e-8)
    assert solution.diagnostics["passes_strict_primal_certificate"] is True
    assert solution.diagnostics["positive_common_margin"] is True
    assert solution.diagnostics["orientation_rule"] == (
        "target_rows_define_positive_orientation_no_posthoc_sign_flip"
    )
    assert solution.diagnostics["recomputed_objective_gamma"] == solution.gamma
    assert min(solution.diagnostics["target_constraint_residuals_at_reported_gamma"]) == (
        pytest.approx(0.0, abs=1e-12)
    )
    with pytest.raises(ValueError):
        solution.direction[0] = 0.0


def test_incompatible_answer_order_constraints_report_no_positive_common_margin() -> None:
    # One order asks for d >= gamma + .25 and the other asks for
    # -d >= gamma + .25.  Their best shared margin is therefore -.25 at d=0.
    solution = solve_global_counterfactual_robust_boundary(
        target_matrix=[[1.0], [-1.0]],
        target_offsets=[0.25, 0.25],
        l2_cap=1.0,
    )

    assert np.array_equal(solution.direction, np.zeros(1, dtype=np.float64))
    assert solution.gamma == pytest.approx(-0.25, abs=3e-8)
    assert solution.diagnostics["positive_common_margin"] is False
    assert solution.diagnostics["passes_strict_primal_certificate"] is True
    assert solution.diagnostics["certified_l2_relaxation_dual_upper_bound"] == pytest.approx(
        solution.gamma, abs=1e-12
    )
    dual = solution.diagnostics["l2_relaxation_dual_upper_bound_certificate"]
    assert dual["selected"]["lambda"] == pytest.approx([0.5, 0.5])
    assert dual["selected"]["passes_simplex_certificate"] is True
    assert dual["never_claims_tightness"] is True


def test_unrelated_equality_basis_is_enforced_in_original_coordinates() -> None:
    solution = solve_global_counterfactual_robust_boundary(
        target_matrix=[[0.0, 1.0]],
        target_offsets=[0.0],
        unrelated_equality_basis=[[1.0, 0.0], [2.0, 0.0]],
        l2_cap=1.0,
    )

    assert solution.direction[0] == pytest.approx(0.0, abs=1e-14)
    assert solution.direction[1] == pytest.approx(1.0, abs=3e-8)
    assert solution.gamma == pytest.approx(1.0, abs=3e-8)
    assert solution.diagnostics["unrelated_equality_rank"] == 1
    assert solution.diagnostics["maximum_abs_unrelated_equality_residual"] < 1e-12
    assert solution.diagnostics["certificate_checks"]["unrelated_equality"] is True


def test_protected_lower_bound_changes_the_max_min_solution() -> None:
    solution = solve_global_counterfactual_robust_boundary(
        target_matrix=[[0.0, 1.0]],
        target_offsets=[0.0],
        protected_matrix=[[1.0, 0.0]],
        protected_lower_bounds=[0.5],
        l2_cap=1.0,
    )

    assert solution.direction[0] == pytest.approx(0.5, abs=3e-8)
    assert solution.direction[1] == pytest.approx(np.sqrt(0.75), abs=3e-8)
    assert solution.gamma == pytest.approx(np.sqrt(0.75), abs=3e-8)
    assert solution.diagnostics["minimum_protected_constraint_residual"] >= -2e-8
    assert solution.diagnostics["feasibility_restoration"]["needed"] is True


def test_infeasible_protected_constraints_fail_closed() -> None:
    with pytest.raises(GCRBSInfeasibleError, match="infeasible"):
        solve_global_counterfactual_robust_boundary(
            target_matrix=[[1.0]],
            target_offsets=[0.0],
            protected_matrix=[[1.0]],
            protected_lower_bounds=[2.0],
            l2_cap=1.0,
        )


def test_global_metric_and_group_factor_budgets_are_recomputed() -> None:
    solution = solve_global_counterfactual_robust_boundary(
        target_matrix=[[1.0, 0.0]],
        target_offsets=[0.0],
        l2_cap=10.0,
        metric_matrix=[[1.0, 0.0], [0.0, 1.0]],
        metric_budget=2.0,
        group_metric_factors=[[[2.0, 0.0]], [[0.0, 1.0]]],
        group_metric_budgets=[0.5, 3.0],
    )

    assert solution.direction[0] == pytest.approx(0.5, abs=3e-8)
    assert solution.direction[1] == pytest.approx(0.0, abs=1e-12)
    assert solution.gamma == pytest.approx(0.5, abs=3e-8)
    assert solution.diagnostics["metric_quadratic_value"] == pytest.approx(0.125)
    assert solution.diagnostics["metric_budget_residual"] == pytest.approx(1.875)
    assert solution.diagnostics["group_quadratic_values"][0] == pytest.approx(0.5)
    assert solution.diagnostics["group_budget_residuals"][0] >= -2e-8
    assert solution.diagnostics["certificate_checks"]["metric_budget"] is True
    assert solution.diagnostics["certificate_checks"]["group_budgets"] is True


def test_zero_dimensional_null_space_is_certified_without_optimizer() -> None:
    solution = solve_global_counterfactual_robust_boundary(
        target_matrix=[[1.0]],
        target_offsets=[0.3],
        unrelated_equality_basis=[[1.0]],
        protected_matrix=[[-1.0]],
        protected_lower_bounds=[0.0],
        l2_cap=0.0,
    )

    assert np.array_equal(solution.direction, np.zeros(1, dtype=np.float64))
    assert solution.gamma == pytest.approx(-0.3)
    assert solution.diagnostics["reduced_dimension"] == 0
    assert solution.diagnostics["certified_main_attempt_count"] == 0


def test_identical_inputs_are_bitwise_deterministic_with_identical_hashes() -> None:
    kwargs = {
        "target_matrix": [[1.0, 0.5], [0.25, 1.0]],
        "target_offsets": [0.1, -0.2],
        "unrelated_equality_basis": [[1.0, -1.0]],
        "protected_matrix": [[1.0, 1.0]],
        "protected_lower_bounds": [0.1],
        "l2_cap": 0.8,
        "metric_matrix": [[2.0, 0.1], [0.1, 1.0]],
        "metric_budget": 0.7,
    }

    first = solve_global_counterfactual_robust_boundary(**kwargs)
    second = solve_global_counterfactual_robust_boundary(**kwargs)

    assert np.array_equal(first.direction, second.direction)
    assert first.gamma == second.gamma
    assert first.diagnostics["input_sha256"] == second.diagnostics["input_sha256"]
    assert first.diagnostics["direction_sha256"] == second.diagnostics["direction_sha256"]
    assert first.diagnostics["diagnostics_sha256"] == second.diagnostics["diagnostics_sha256"]
    assert len(first.diagnostics["diagnostics_sha256"]) == 64


def test_random_feasible_primals_never_exceed_certified_dual_upper_bounds() -> None:
    generator = np.random.default_rng(20260827)
    for _ in range(8):
        target = generator.normal(size=(5, 3))
        offsets = generator.normal(scale=0.2, size=5)
        solution = solve_global_counterfactual_robust_boundary(
            target_matrix=target,
            target_offsets=offsets,
            unrelated_equality_basis=[[1.0, -1.0, 0.5]],
            l2_cap=0.7,
        )

        upper = solution.diagnostics["certified_l2_relaxation_dual_upper_bound"]
        assert solution.gamma <= upper + 1e-12
        assert solution.diagnostics["dual_upper_bound_minus_primal_gamma"] >= -1e-12
        assert solution.diagnostics["certificate_checks"][
            "primal_below_l2_relaxation_dual_upper_bound"
        ]


def test_dual_optimizer_failure_retains_the_uniform_valid_upper_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def never_converges(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            success=False,
            status=9,
            nit=2_000,
            x=np.array([0.5, 0.5], dtype=np.float64),
        )

    monkeypatch.setattr(gcrbs, "_DUAL_MINIMIZE", never_converges)
    solution = solve_global_counterfactual_robust_boundary(
        target_matrix=[[1.0], [-1.0]],
        target_offsets=[0.25, 0.25],
        l2_cap=1.0,
    )

    dual = solution.diagnostics["l2_relaxation_dual_upper_bound_certificate"]
    assert dual["optimizer_success_count"] == 0
    assert dual["uniform_fallback"]["passes_simplex_certificate"] is True
    assert dual["uniform_fallback"]["certified_upper_bound"] == pytest.approx(
        solution.gamma, abs=1e-12
    )
    assert solution.gamma <= dual["certified_upper_bound"]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"target_matrix": [[1.0, np.nan]]}, "finite"),
        ({"target_offsets": [0.0, 1.0]}, "length"),
        ({"unrelated_equality_basis": [[1.0, 0.0, 0.0]]}, "width"),
        ({"protected_matrix": [[1.0, 0.0]]}, "supplied together"),
        (
            {"metric_matrix": [[1.0, 2.0], [0.0, 1.0]], "metric_budget": 1.0},
            "symmetric",
        ),
        (
            {"metric_matrix": [[1.0, 0.0], [0.0, -0.1]], "metric_budget": 1.0},
            "positive semidefinite",
        ),
        ({"group_metric_factors": [[[1.0, 0.0]]]}, "equal length"),
        ({"l2_cap": -1.0}, "nonnegative"),
    ],
)
def test_malformed_inputs_are_rejected_before_optimization(
    overrides: dict[str, object], match: str
) -> None:
    kwargs: dict[str, object] = {
        "target_matrix": [[1.0, 0.0]],
        "target_offsets": [0.0],
        "l2_cap": 1.0,
    }
    kwargs.update(overrides)

    with pytest.raises((TypeError, ValueError), match=match):
        solve_global_counterfactual_robust_boundary(**kwargs)


def test_solver_nonconvergence_never_returns_an_uncertified_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def never_converges(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            success=False,
            status=9,
            nit=2_000,
            x=np.array([0.0, 0.0], dtype=np.float64),
        )

    monkeypatch.setattr(gcrbs, "_MINIMIZE", never_converges)

    with pytest.raises(GCRBSSolverError, match="no deterministic max-min attempt"):
        solve_global_counterfactual_robust_boundary(
            target_matrix=[[1.0]],
            target_offsets=[0.0],
            l2_cap=1.0,
        )


def test_success_status_with_primal_breach_is_still_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def lies_about_success(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            success=True,
            status=0,
            nit=1,
            # d=10 violates the declared L2 cap even though status says success.
            x=np.array([10.0, 10.0], dtype=np.float64),
        )

    monkeypatch.setattr(gcrbs, "_MINIMIZE", lies_about_success)

    with pytest.raises(GCRBSSolverError, match="no deterministic max-min attempt"):
        solve_global_counterfactual_robust_boundary(
            target_matrix=[[1.0]],
            target_offsets=[0.0],
            l2_cap=1.0,
        )
