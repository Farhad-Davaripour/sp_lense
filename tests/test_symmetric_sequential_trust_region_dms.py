from __future__ import annotations

import numpy as np
import pytest

import sp_lense.symmetric_sequential_trust_region_dms as sequential_module
from sp_lense.symmetric_sequential_trust_region_dms import (
    FLOAT32_RAW_CONSTRAINT_TOLERANCE,
    SCHEMA_VERSION,
    SymmetricSequentialDMSCertificateError,
    SymmetricSequentialDMSInfeasibleError,
    SymmetricSequentialTrustRegionUpdate,
    revalidate_symmetric_sequential_trust_region_update,
    solve_symmetric_sequential_trust_region_update,
)


def _analytic_problem() -> dict[str, object]:
    return {
        "current_direction": np.array([0.2, -0.1, 0.05]),
        "target_plus_margins": np.array([0.2]),
        "target_plus_gradients": np.array([[1.0, 0.0, 0.0]]),
        "target_minus_margins": np.array([-0.1]),
        "target_minus_gradients": np.array([[0.0, 1.0, 0.0]]),
        "protected_plus_margins": np.array([0.3, -0.4]),
        "protected_plus_gradients": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "protected_minus_margins": np.array([0.3, -0.4]),
        "protected_minus_gradients": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "protected_baseline_signs": np.array([1.0, -1.0]),
        "unrelated_baseline_margins": np.array([0.0]),
        "unrelated_plus_margins": np.array([0.2]),
        "unrelated_plus_gradients": np.array([[0.0, 0.0, 1.0]]),
        "unrelated_minus_margins": np.array([-0.2]),
        "unrelated_minus_gradients": np.array([[0.0, 0.0, 1.0]]),
        "optimization_target_margin": 0.6,
        "protected_margin": 0.0,
        "physical_residual_scale": 1.0,
        "progress_fraction": 0.5,
        "trust_radius": 0.4,
    }


def test_known_update_uses_one_vector_for_both_exact_deployment_signs() -> None:
    result = solve_symmetric_sequential_trust_region_update(**_analytic_problem())

    assert isinstance(result, SymmetricSequentialTrustRegionUpdate)
    np.testing.assert_allclose(result.ideal_update, [0.2, 0.25, -0.1], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(result.update, [0.2, 0.25, -0.1], atol=3e-8, rtol=0.0)
    np.testing.assert_allclose(result.updated_direction, [0.4, 0.15, -0.05], atol=3e-8, rtol=0.0)
    np.testing.assert_array_equal(result.positive_deployed_direction, result.updated_direction)
    np.testing.assert_array_equal(result.negative_deployed_direction, -result.updated_direction)
    assert all(
        not value.flags.writeable
        for value in (
            result.update,
            result.updated_direction,
            result.positive_deployed_direction,
            result.negative_deployed_direction,
            result.positive_physical_float32,
            result.negative_physical_float32,
        )
    )
    with pytest.raises(TypeError):
        result.diagnostics["new"] = "not mutable"
    with pytest.raises(TypeError):
        result.diagnostics["realized_deployment_certificate"]["checks"]["finite"] = False

    ideal_certificate = result.diagnostics["ideal_solver_certificate"]
    realized_certificate = result.diagnostics["realized_deployment_certificate"]
    assert ideal_certificate["passes"] is True
    assert realized_certificate["passes"] is True
    assert all(ideal_certificate["checks"].values())
    assert all(realized_certificate["checks"].values())
    assert result.diagnostics["schema_version"] == SCHEMA_VERSION
    assert result.diagnostics["next_state_is_realized_direction"] is True
    assert revalidate_symmetric_sequential_trust_region_update(result)["passes"] is True


def test_target_plus_and_minus_chain_rule_inequalities_are_derived_correctly() -> None:
    result = solve_symmetric_sequential_trust_region_update(
        np.array([0.3]),
        target_plus_margins=np.array([-0.2]),
        target_plus_gradients=np.array([[1.0]]),
        target_minus_margins=np.array([0.2]),
        target_minus_gradients=np.array([[1.0]]),
        optimization_target_margin=0.2,
        physical_residual_scale=1.0,
        progress_fraction=0.5,
        trust_radius=0.3,
    )

    # Plus: m+new = -0.2 + u.  Minus: m-new = 0.2 - u, hence
    # its comply-oriented margin is -0.2 + u.  Both require u >= 0.2.
    np.testing.assert_allclose(result.ideal_update, [0.2], atol=1e-12, rtol=0.0)
    certificate = result.diagnostics["ideal_solver_certificate"]
    np.testing.assert_allclose(certificate["target_oriented_current_margins"], [-0.2, -0.2])
    np.testing.assert_allclose(certificate["target_required_progress"], [0.2, 0.2])
    np.testing.assert_allclose(certificate["target_oriented_next_margins"], [0.0, 0.0], atol=1e-12)
    assert (
        result.diagnostics["constraint_record"]["target_minus_oriented_formula"]
        == "-m_minus_new=-m_minus+g_minus@u"
    )


def test_protected_rows_preserve_the_baseline_semantic_decision_side() -> None:
    result = solve_symmetric_sequential_trust_region_update(
        np.array([0.0]),
        target_plus_margins=np.array([0.0]),
        target_plus_gradients=np.array([[1.0]]),
        target_minus_margins=np.array([0.0]),
        target_minus_gradients=np.array([[1.0]]),
        protected_plus_margins=np.array([-0.2]),
        protected_plus_gradients=np.array([[1.0]]),
        protected_minus_margins=np.array([-0.2]),
        protected_minus_gradients=np.array([[0.0]]),
        protected_baseline_signs=np.array([-1.0]),
        optimization_target_margin=0.4,
        protected_margin=0.0,
        physical_residual_scale=1.0,
        progress_fraction=0.5,
        trust_radius=0.3,
    )

    # The target needs u >= 0.2.  The negative-semantic protected plus row
    # requires -(-0.2 + u) >= 0, or u <= 0.2.
    np.testing.assert_allclose(result.ideal_update, [0.2], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        result.diagnostics["realized_deployment_certificate"]["protected_oriented_next_margins"],
        [0.0, 0.2],
        atol=FLOAT32_RAW_CONSTRAINT_TOLERANCE,
    )


def test_unrelated_equalities_move_each_sign_the_fixed_fraction_to_baseline() -> None:
    result = solve_symmetric_sequential_trust_region_update(**_analytic_problem())
    certificate = result.diagnostics["realized_deployment_certificate"]

    np.testing.assert_allclose(certificate["unrelated_plus_next_margins"], [0.1])
    np.testing.assert_allclose(certificate["unrelated_minus_next_margins"], [-0.1])
    np.testing.assert_allclose(certificate["unrelated_plus_desired_margins"], [0.1])
    np.testing.assert_allclose(certificate["unrelated_minus_desired_margins"], [-0.1])
    assert certificate["checks"]["unrelated_path_return_within_float32_tolerance"] is True


def test_complete_updated_direction_retains_the_baseline_unrelated_exact_null() -> None:
    result = solve_symmetric_sequential_trust_region_update(
        np.array([0.2, 0.0]),
        target_plus_margins=np.array([0.0]),
        target_plus_gradients=np.array([[0.0, 1.0]]),
        target_minus_margins=np.array([0.0]),
        target_minus_gradients=np.array([[0.0, 1.0]]),
        baseline_unrelated_gradients=np.array([[1.0, 0.0]]),
        optimization_target_margin=0.4,
        physical_residual_scale=1.0,
        progress_fraction=0.5,
        trust_radius=0.4,
    )

    # Target progress needs u_y=+0.2.  Exact retention of G0(D+u)=0 needs
    # u_x=-0.2, rather than merely G0*u=0.
    np.testing.assert_allclose(result.ideal_update, [-0.2, 0.2], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(result.updated_direction, [0.0, 0.2], atol=3e-8)
    certificate = result.diagnostics["realized_deployment_certificate"]
    np.testing.assert_allclose(
        certificate["baseline_unrelated_updated_projections"], [0.0], atol=1e-12
    )
    assert certificate["checks"]["baseline_unrelated_null_within_float32_tolerance"] is True
    assert (
        result.diagnostics["constraint_record"]["baseline_unrelated_null_equality"] == "G0@u=-G0@D"
    )


def test_raw_label_sign_changes_and_row_order_do_not_change_the_update() -> None:
    original = _analytic_problem()
    first = solve_symmetric_sequential_trust_region_update(**original)

    recoded = dict(original)
    # Reversing the raw A/B log-odds convention flips margins, gradients, and
    # the sign naming the baseline semantic side.  The scientific constraint is
    # unchanged.  The unrelated equalities are likewise invariant to recoding.
    recoded["protected_plus_margins"] = -original["protected_plus_margins"]
    recoded["protected_plus_gradients"] = -original["protected_plus_gradients"]
    recoded["protected_minus_margins"] = -original["protected_minus_margins"]
    recoded["protected_minus_gradients"] = -original["protected_minus_gradients"]
    recoded["protected_baseline_signs"] = -original["protected_baseline_signs"]
    recoded["unrelated_baseline_margins"] = -original["unrelated_baseline_margins"]
    recoded["unrelated_plus_margins"] = -original["unrelated_plus_margins"]
    recoded["unrelated_plus_gradients"] = -original["unrelated_plus_gradients"]
    recoded["unrelated_minus_margins"] = -original["unrelated_minus_margins"]
    recoded["unrelated_minus_gradients"] = -original["unrelated_minus_gradients"]
    second = solve_symmetric_sequential_trust_region_update(**recoded)

    np.testing.assert_array_equal(first.update, second.update)
    assert (
        first.diagnostics["constraint_record"]["constraint_record_sha256"]
        == second.diagnostics["constraint_record"]["constraint_record_sha256"]
    )

    order_problem = {
        "current_direction": np.zeros(3),
        "target_plus_margins": np.array([0.0, 0.1]),
        "target_plus_gradients": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "target_minus_margins": np.array([0.0, -0.1]),
        "target_minus_gradients": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "optimization_target_margin": 0.4,
        "physical_residual_scale": 1.0,
        "progress_fraction": 0.5,
        "trust_radius": 0.5,
    }
    forward = solve_symmetric_sequential_trust_region_update(**order_problem)
    permutation = np.array([1, 0])
    reversed_problem = dict(order_problem)
    for field in (
        "target_plus_margins",
        "target_plus_gradients",
        "target_minus_margins",
        "target_minus_gradients",
    ):
        reversed_problem[field] = order_problem[field][permutation]
    reversed_result = solve_symmetric_sequential_trust_region_update(**reversed_problem)
    np.testing.assert_array_equal(forward.update, reversed_result.update)
    assert (
        forward.diagnostics["constraint_record"]["constraint_record_sha256"]
        == reversed_result.diagnostics["constraint_record"]["constraint_record_sha256"]
    )


def test_already_satisfied_targets_cannot_regress_under_an_unrelated_equality() -> None:
    with pytest.raises(
        SymmetricSequentialDMSInfeasibleError,
        match="infeasible|conflicts",
    ):
        solve_symmetric_sequential_trust_region_update(
            np.array([0.0]),
            target_plus_margins=np.array([1.0]),
            target_plus_gradients=np.array([[1.0]]),
            target_minus_margins=np.array([-1.0]),
            target_minus_gradients=np.array([[1.0]]),
            unrelated_baseline_margins=np.array([0.0]),
            unrelated_plus_margins=np.array([0.2]),
            unrelated_plus_gradients=np.array([[1.0]]),
            unrelated_minus_margins=np.array([-0.2]),
            unrelated_minus_gradients=np.array([[1.0]]),
            optimization_target_margin=0.5,
            physical_residual_scale=1.0,
            progress_fraction=0.5,
            trust_radius=1.0,
        )


def test_certified_minimum_outside_the_fixed_trust_radius_fails_closed() -> None:
    problem = _analytic_problem()
    problem["trust_radius"] = 0.3
    with pytest.raises(
        SymmetricSequentialDMSInfeasibleError, match="exceeds the fixed trust radius"
    ):
        solve_symmetric_sequential_trust_region_update(**problem)


def test_inconsistent_unrelated_affine_equalities_fail_closed() -> None:
    with pytest.raises(SymmetricSequentialDMSInfeasibleError, match="mutually inconsistent"):
        solve_symmetric_sequential_trust_region_update(
            np.array([0.0]),
            target_plus_margins=np.array([1.0]),
            target_plus_gradients=np.array([[0.0]]),
            target_minus_margins=np.array([-1.0]),
            target_minus_gradients=np.array([[0.0]]),
            unrelated_baseline_margins=np.array([1.0]),
            unrelated_plus_margins=np.array([0.0]),
            unrelated_plus_gradients=np.array([[1.0]]),
            unrelated_minus_margins=np.array([0.0]),
            unrelated_minus_gradients=np.array([[1.0]]),
            optimization_target_margin=0.5,
            physical_residual_scale=1.0,
            progress_fraction=1.0,
            trust_radius=2.0,
        )


def test_baseline_null_and_path_return_conflict_fails_closed() -> None:
    with pytest.raises(SymmetricSequentialDMSInfeasibleError, match="mutually inconsistent"):
        solve_symmetric_sequential_trust_region_update(
            np.array([0.2]),
            target_plus_margins=np.array([1.0]),
            target_plus_gradients=np.array([[0.0]]),
            target_minus_margins=np.array([-1.0]),
            target_minus_gradients=np.array([[0.0]]),
            unrelated_baseline_margins=np.array([0.2]),
            unrelated_plus_margins=np.array([0.0]),
            unrelated_plus_gradients=np.array([[1.0]]),
            unrelated_minus_margins=np.array([0.0]),
            unrelated_minus_gradients=np.array([[-1.0]]),
            baseline_unrelated_gradients=np.array([[1.0]]),
            optimization_target_margin=0.5,
            physical_residual_scale=1.0,
            progress_fraction=1.0,
            trust_radius=1.0,
        )


def test_target_and_protected_decision_side_conflict_is_infeasible() -> None:
    with pytest.raises(SymmetricSequentialDMSInfeasibleError, match="infeasible"):
        solve_symmetric_sequential_trust_region_update(
            np.array([0.0]),
            target_plus_margins=np.array([0.0]),
            target_plus_gradients=np.array([[1.0]]),
            target_minus_margins=np.array([0.0]),
            target_minus_gradients=np.array([[1.0]]),
            protected_plus_margins=np.array([0.0]),
            protected_plus_gradients=np.array([[-1.0]]),
            protected_minus_margins=np.array([1.0]),
            protected_minus_gradients=np.array([[0.0]]),
            protected_baseline_signs=np.array([1.0]),
            optimization_target_margin=0.4,
            protected_margin=0.0,
            physical_residual_scale=1.0,
            progress_fraction=0.5,
            trust_radius=1.0,
        )


def test_zero_unrelated_gradient_cannot_claim_nonzero_return() -> None:
    with pytest.raises(
        SymmetricSequentialDMSInfeasibleError,
        match="zero unrelated-gradient equality",
    ):
        solve_symmetric_sequential_trust_region_update(
            np.array([0.0]),
            target_plus_margins=np.array([1.0]),
            target_plus_gradients=np.array([[0.0]]),
            target_minus_margins=np.array([-1.0]),
            target_minus_gradients=np.array([[0.0]]),
            unrelated_baseline_margins=np.array([1.0]),
            unrelated_plus_margins=np.array([0.0]),
            unrelated_plus_gradients=np.array([[0.0]]),
            unrelated_minus_margins=np.array([1.0]),
            unrelated_minus_gradients=np.array([[0.0]]),
            optimization_target_margin=0.5,
            physical_residual_scale=1.0,
            progress_fraction=0.5,
            trust_radius=1.0,
        )


def test_candidate_certificate_failure_is_never_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_certificate = sequential_module._certify_update

    def corrupt_certificate(**kwargs):
        certificate = real_certificate(**kwargs)
        certificate = dict(certificate)
        certificate["checks"] = dict(certificate["checks"])
        certificate["checks"]["kkt_stationarity"] = False
        certificate["minimum_norm_checks_pass"] = False
        certificate["passes"] = False
        return certificate

    monkeypatch.setattr(sequential_module, "_certify_update", corrupt_certificate)
    with pytest.raises(SymmetricSequentialDMSCertificateError, match="kkt_stationarity"):
        solve_symmetric_sequential_trust_region_update(**_analytic_problem())


def test_results_and_hashes_are_deterministic_within_the_pinned_runtime() -> None:
    first = solve_symmetric_sequential_trust_region_update(**_analytic_problem())
    second = solve_symmetric_sequential_trust_region_update(**_analytic_problem())

    np.testing.assert_array_equal(first.update, second.update)
    np.testing.assert_array_equal(first.updated_direction, second.updated_direction)
    assert first.diagnostics["input_sha256"] == second.diagnostics["input_sha256"]
    assert first.diagnostics["update_sha256"] == second.diagnostics["update_sha256"]
    assert (
        first.diagnostics["realized_deployment_certificate"]["certificate_sha256"]
        == second.diagnostics["realized_deployment_certificate"]["certificate_sha256"]
    )
    assert first.diagnostics["diagnostics_sha256"] == second.diagnostics["diagnostics_sha256"]
    assert first.diagnostics["determinism_scope"] == "deterministic_within_pinned_runtime"


def test_required_margins_and_physical_scale_cannot_be_omitted() -> None:
    missing_target = _analytic_problem()
    missing_target.pop("optimization_target_margin")
    with pytest.raises(TypeError, match="optimization_target_margin"):
        solve_symmetric_sequential_trust_region_update(**missing_target)

    missing_scale = _analytic_problem()
    missing_scale.pop("physical_residual_scale")
    with pytest.raises(TypeError, match="physical_residual_scale"):
        solve_symmetric_sequential_trust_region_update(**missing_scale)

    missing_protected = _analytic_problem()
    missing_protected.pop("protected_margin")
    with pytest.raises(TypeError, match="protected_margin is required"):
        solve_symmetric_sequential_trust_region_update(**missing_protected)


def test_float32_deployment_is_authoritative_and_round_trips_for_next_state() -> None:
    problem = _analytic_problem()
    problem["physical_residual_scale"] = 7.25
    result = solve_symmetric_sequential_trust_region_update(**problem)

    expected_positive = np.asarray(
        7.25 * result.ideal_updated_direction, dtype=np.float32, order="C"
    )
    expected_negative = np.negative(expected_positive)
    assert result.positive_physical_float32.tobytes() == expected_positive.tobytes()
    assert result.negative_physical_float32.tobytes() == expected_negative.tobytes()
    np.testing.assert_array_equal(
        result.realized_direction,
        result.positive_physical_float32.astype(np.float64) / 7.25,
    )
    np.testing.assert_array_equal(
        result.realized_update,
        result.realized_direction - result.current_direction,
    )
    next_step_physical = np.asarray(7.25 * result.realized_direction, dtype=np.float32, order="C")
    assert next_step_physical.tobytes() == result.positive_physical_float32.tobytes()
    assert result.diagnostics["realized_deployment_certificate"]["passes"] is True
    assert revalidate_symmetric_sequential_trust_region_update(result)["passes"] is True


def test_negative_physical_vector_toggles_the_signed_zero_bit() -> None:
    result = solve_symmetric_sequential_trust_region_update(
        np.array([0.2, 0.0]),
        target_plus_margins=np.array([0.0]),
        target_plus_gradients=np.array([[0.0, 1.0]]),
        target_minus_margins=np.array([0.0]),
        target_minus_gradients=np.array([[0.0, 1.0]]),
        baseline_unrelated_gradients=np.array([[1.0, 0.0]]),
        optimization_target_margin=0.4,
        physical_residual_scale=3.0,
        progress_fraction=0.5,
        trust_radius=0.4,
    )

    assert result.positive_physical_float32[0] == 0.0
    assert result.negative_physical_float32[0] == 0.0
    assert not np.signbit(result.positive_physical_float32[0])
    assert np.signbit(result.negative_physical_float32[0])
    assert (
        result.negative_physical_float32.tobytes()
        == np.negative(result.positive_physical_float32).tobytes()
    )
    assert (
        result.diagnostics["realized_deployment_certificate"]["checks"][
            "negative_physical_is_bytewise_unary_negation"
        ]
        is True
    )


def test_cast_induced_raw_constraint_failure_is_rejected() -> None:
    with pytest.raises(
        SymmetricSequentialDMSCertificateError,
        match="realized float32 deployment failed.*target_fractional_progress",
    ):
        solve_symmetric_sequential_trust_region_update(
            np.array([0.99900005]),
            target_plus_margins=np.array([-3.95]),
            target_plus_gradients=np.array([[1000.0]]),
            target_minus_margins=np.array([3.95]),
            target_minus_gradients=np.array([[1000.0]]),
            optimization_target_margin=np.array([0.05]),
            physical_residual_scale=1024.0,
            progress_fraction=0.25,
            trust_radius=0.25,
        )


def test_integrity_revalidation_detects_post_return_array_mutation() -> None:
    result = solve_symmetric_sequential_trust_region_update(**_analytic_problem())
    result.positive_physical_float32.setflags(write=True)
    result.positive_physical_float32[0] = np.nextafter(
        result.positive_physical_float32[0], np.float32(np.inf)
    )
    with pytest.raises(
        SymmetricSequentialDMSCertificateError,
        match="failed integrity revalidation",
    ):
        revalidate_symmetric_sequential_trust_region_update(result)


def test_integrity_revalidation_can_bind_an_external_diagnostics_hash() -> None:
    result = solve_symmetric_sequential_trust_region_update(**_analytic_problem())
    expected = result.diagnostics["diagnostics_sha256"]
    assert (
        revalidate_symmetric_sequential_trust_region_update(
            result, expected_diagnostics_sha256=expected
        )["passes"]
        is True
    )
    with pytest.raises(
        SymmetricSequentialDMSCertificateError,
        match="expected_diagnostics_hash",
    ):
        revalidate_symmetric_sequential_trust_region_update(
            result, expected_diagnostics_sha256="0" * 64
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("progress_fraction", 0.0, "positive"),
        ("progress_fraction", 1.1, r"\(0, 1\]"),
        ("trust_radius", 0.0, "positive"),
        ("physical_residual_scale", 0.0, "positive"),
        ("protected_baseline_signs", np.array([0.0, -1.0]), r"only -1 or \+1"),
    ],
)
def test_invalid_fixed_parameters_and_semantic_signs_are_rejected(
    field: str, value: object, message: str
) -> None:
    problem = _analytic_problem()
    problem[field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        solve_symmetric_sequential_trust_region_update(**problem)


def test_partial_optional_families_and_shape_mismatches_are_rejected() -> None:
    with pytest.raises(ValueError, match="all four protected branch arrays"):
        solve_symmetric_sequential_trust_region_update(
            np.zeros(2),
            target_plus_margins=np.array([0.0]),
            target_plus_gradients=np.array([[1.0, 0.0]]),
            target_minus_margins=np.array([0.0]),
            target_minus_gradients=np.array([[1.0, 0.0]]),
            protected_plus_margins=np.array([0.1]),
            optimization_target_margin=0.4,
            physical_residual_scale=1.0,
        )
    with pytest.raises(ValueError, match=r"shape \[rows, 2\]"):
        solve_symmetric_sequential_trust_region_update(
            np.zeros(2),
            target_plus_margins=np.array([0.0]),
            target_plus_gradients=np.array([[1.0, 0.0, 0.0]]),
            target_minus_margins=np.array([0.0]),
            target_minus_gradients=np.array([[1.0, 0.0]]),
            optimization_target_margin=0.4,
            physical_residual_scale=1.0,
        )
