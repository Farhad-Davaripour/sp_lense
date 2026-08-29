from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from sp_lense.counterfactual_kl_extragradient_surgery import (
    construct_common_ascent_lookahead,
    revalidate_counterfactual_kl_extragradient_update,
    solve_counterfactual_kl_extragradient_update,
)
from sp_lense.symmetric_sequential_trust_region_dms import (
    SymmetricSequentialDMSInfeasibleError,
    solve_symmetric_sequential_trust_region_update,
)


def test_common_ascent_is_unit_positive_and_exactly_nuisance_nulled() -> None:
    target = np.asarray(
        [
            [1.0, 0.2, 0.0, 0.4],
            [1.0, -0.1, 0.0, -0.2],
            [0.8, 0.3, 0.0, 0.1],
            [1.1, -0.2, 0.0, -0.1],
        ]
    )
    nuisance = np.asarray([[0.0, 0.0, 0.0, 1.0]])
    result = construct_common_ascent_lookahead(
        np.zeros(4),
        oriented_target_gradients=target,
        baseline_unrelated_gradients=nuisance,
    )
    assert result.diagnostics["passes"] is True
    assert np.linalg.norm(result.direction) == pytest.approx(1.0)
    assert np.min(target @ result.direction) > 0.0
    assert np.max(np.abs(nuisance @ result.direction)) <= 2e-5
    assert np.linalg.norm(result.lookahead_direction) == pytest.approx(1.0 / 32.0)


def test_conflicting_target_cone_fails_closed() -> None:
    with pytest.raises(SymmetricSequentialDMSInfeasibleError, match="collapses"):
        construct_common_ascent_lookahead(
            np.zeros(3),
            oriented_target_gradients=np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            baseline_unrelated_gradients=np.asarray([[0.0, 0.0, 1.0]]),
        )


def _nominal_problem() -> dict[str, object]:
    return {
        "current_direction": np.zeros(4),
        "target_plus_margins": np.asarray([-1.0, -1.0]),
        "target_plus_gradients": np.asarray(
            [[1.0, 0.2, 0.0, 0.0], [1.0, -0.2, 0.0, 0.0]]
        ),
        "target_minus_margins": np.asarray([-1.0, -1.0]),
        "target_minus_gradients": np.asarray(
            [[1.0, 0.2, 0.0, 0.0], [1.0, -0.2, 0.0, 0.0]]
        ),
        "optimization_target_margin": 0.1,
        "physical_residual_scale": 1.0,
        "protected_plus_margins": np.asarray([1.0]),
        "protected_plus_gradients": np.asarray([[0.0, 0.0, 1.0, 0.0]]),
        "protected_minus_margins": np.asarray([1.0]),
        "protected_minus_gradients": np.asarray([[0.0, 0.0, 1.0, 0.0]]),
        "protected_baseline_signs": np.asarray([1.0]),
        "protected_margin": np.asarray([0.1]),
        "unrelated_baseline_margins": np.asarray([0.5]),
        "unrelated_plus_margins": np.asarray([0.5]),
        "unrelated_plus_gradients": np.asarray([[0.0, 0.0, 0.0, 1.0]]),
        "unrelated_minus_margins": np.asarray([0.5]),
        "unrelated_minus_gradients": np.asarray([[0.0, 0.0, 0.0, 1.0]]),
        "baseline_unrelated_gradients": np.asarray([[0.0, 0.0, 0.0, 1.0]]),
        "progress_fraction": 0.25,
        "trust_radius": 0.5,
    }


def test_centered_qp_preserves_nominal_constraints_and_kl_tangents() -> None:
    problem = _nominal_problem()
    nominal = solve_symmetric_sequential_trust_region_update(**problem)
    common = construct_common_ascent_lookahead(
        problem["current_direction"],
        oriented_target_gradients=np.vstack(
            (problem["target_plus_gradients"], problem["target_minus_gradients"])
        ),
        baseline_unrelated_gradients=problem["baseline_unrelated_gradients"],
    )
    result = solve_counterfactual_kl_extragradient_update(
        nominal,
        common_ascent_lookahead=common,
        lookahead_kl_values=np.asarray([0.006, 0.006]),
        lookahead_kl_shared_gradients=np.asarray(
            [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        ),
        **{key: value for key, value in problem.items() if key != "current_direction"},
    )
    replay = revalidate_counterfactual_kl_extragradient_update(result)
    certificate = result.diagnostics["realized_deployment_certificate"]
    assert replay["passes"] is True
    assert certificate["passes"] is True
    assert certificate["maximum_tangent_kl"] <= 0.02 + 2e-5
    assert certificate["mean_tangent_kl"] <= 0.005 + 2e-5
    assert min(certificate["target_realized_progress"]) >= min(
        certificate["target_required_progress"]
    ) - 2e-5
    assert result.negative_physical_float32.tobytes() == (
        -result.positive_physical_float32
    ).tobytes()
    assert all(
        not value.flags.writeable
        for value in (
            result.current_direction,
            result.nominal_update,
            result.ideal_update,
            result.ideal_updated_direction,
            result.realized_update,
            result.realized_direction,
            result.positive_deployed_direction,
            result.negative_deployed_direction,
            result.positive_physical_float32,
            result.negative_physical_float32,
        )
    )
    json.dumps(result.as_record(), sort_keys=True)
    json.dumps(common.as_record(), sort_keys=True)


def test_impossible_kl_tangent_fails_closed() -> None:
    problem = _nominal_problem()
    nominal = solve_symmetric_sequential_trust_region_update(**problem)
    common = construct_common_ascent_lookahead(
        problem["current_direction"],
        oriented_target_gradients=np.vstack(
            (problem["target_plus_gradients"], problem["target_minus_gradients"])
        ),
        baseline_unrelated_gradients=problem["baseline_unrelated_gradients"],
    )
    with pytest.raises(SymmetricSequentialDMSInfeasibleError):
        solve_counterfactual_kl_extragradient_update(
            nominal,
            common_ascent_lookahead=common,
            lookahead_kl_values=np.asarray([1.0]),
            lookahead_kl_shared_gradients=np.zeros((1, 4)),
            **{key: value for key, value in problem.items() if key != "current_direction"},
        )


def test_nominal_input_mismatch_and_candidate_tampering_fail_closed() -> None:
    problem = _nominal_problem()
    nominal = solve_symmetric_sequential_trust_region_update(**problem)
    common = construct_common_ascent_lookahead(
        problem["current_direction"],
        oriented_target_gradients=np.vstack(
            (problem["target_plus_gradients"], problem["target_minus_gradients"])
        ),
        baseline_unrelated_gradients=problem["baseline_unrelated_gradients"],
    )
    call = {
        "common_ascent_lookahead": common,
        "lookahead_kl_values": np.asarray([0.006, 0.006]),
        "lookahead_kl_shared_gradients": np.asarray(
            [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        ),
        **{key: value for key, value in problem.items() if key != "current_direction"},
    }
    with pytest.raises(
        Exception, match="different inputs"
    ):
        solve_counterfactual_kl_extragradient_update(
            nominal,
            **{**call, "progress_fraction": 0.125},
        )
    result = solve_counterfactual_kl_extragradient_update(nominal, **call)
    arbitrary = np.asarray(result.positive_physical_float32).copy()
    arbitrary[0] += np.float32(0.25)
    arbitrary.flags.writeable = False
    tampered = replace(result, positive_physical_float32=arbitrary)
    assert revalidate_counterfactual_kl_extragradient_update(tampered)["passes"] is False
