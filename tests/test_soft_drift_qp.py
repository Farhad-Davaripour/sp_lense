"""Small supplied-witness tests only; no native benchmark or model dependencies."""

import copy
import hashlib
import json
import math
import struct
import sys
from fractions import Fraction

import pytest

from scripts import soft_drift_qp_solver as solver
from scripts import verify_soft_drift_qp as checker

PAIRS = ((2, 0), (3, 1), (6, 4), (7, 5), (10, 8), (11, 9))


def problem(pairs=(), dimension=2):
    A, b, c = [[0.0] * dimension for _ in range(12)], [-1.0] * 12, [0.0] * 6
    for p, (aa, ab, ba, bb, drift) in enumerate(pairs):
        ia, ib = PAIRS[p]
        A[ia], A[ib], b[ia], b[ib], c[p] = list(aa), list(ab), ba, bb, drift
    return A, b, c


def signature(A, b, c):
    raw = b"soft-drift-qp-v1\0" + struct.pack("<II", 12, len(A[0]))
    raw += b"".join(struct.pack("<d", float(x)) for row in A for x in row)
    raw += b"".join(struct.pack("<d", float(x)) for x in [*b, *c])
    return hashlib.sha256(raw).hexdigest()


def witness(A, b, c, vector, active=()):
    mu = [0.0] * 12
    for index, value in active:
        mu[index] = value
    return {
        "vector": vector,
        "multipliers": mu,
        "active_mask": sum(1 << index for index, _ in active),
        "input_sha256": signature(A, b, c),
    }


def accepted(A, b, c, candidate):
    result = checker.verify(A, b, c, candidate)
    assert result["status"] == "NUMERIC_KKT_WITHIN_TOLERANCE"
    assert result["accepted"] is True
    return result


def rejected(A, b, c, candidate):
    result = checker.verify(A, b, c, candidate)
    assert result["status"] == "NUMERICALLY_UNRESOLVED"
    assert result["accepted"] is False


def test_negative_rhs_supplied_optimum_and_soft_drift_increase():
    A, b, c = problem([((1.0, 0.0), (-1.0, 0.0), 0.06, -0.10, 0.1)])
    hand = witness(A, b, c, [0.06, 0.0], [(2, 1 / 15)])
    accepted(A, b, c, hand)
    result = solver.solve(A, b, c)
    assert result["status"] == "KKT_ESTIMATE_ONLY"
    assert result["solution"]["vector"] == pytest.approx([0.06, 0.0], abs=1e-12)
    accepted(A, b, c, result["solution"])
    assert c[0] + result["solution"]["vector"][0] == pytest.approx(0.16)
    assert b[0] == -0.10
    assert 0.05 < b[2]  # Raw-QP feasibility does not certify the clipped step.


def affine():
    return problem(
        [((1.0, 0.0), (-1.0, 0.0), -0.05, -0.05, 1.0), ((0.0, 4.0), (0.0, 4.0), 0.10, 0.10, 0.0)]
    )


def test_affine_term_duplicate_rows_and_known_native_optimum():
    A, b, c = affine()
    accepted(A, b, c, witness(A, b, c, [-1 / 25, 1 / 40], [(1, 1 / 160)]))
    result = solver.solve(A, b, c)
    assert result["status"] == "KKT_ESTIMATE_ONLY"
    assert result["solution"]["vector"] == pytest.approx([-1 / 25, 1 / 40], abs=1e-12)
    accepted(A, b, c, result["solution"])


def test_redundant_dual_support_is_conservatively_unresolved():
    A, b, c = affine()
    hand = witness(A, b, c, [-1 / 25, 1 / 40], [(1, 1 / 320), (3, 1 / 320)])
    result = checker.verify(A, b, c, hand)
    assert result["accepted"] is False
    assert result["reason"] == "ACTIVE_RANK_NOT_INTERIOR"


@pytest.mark.parametrize("other_gradient,common,semantic", [(0.5, 0.02, 0.0), (-0.5, 0.0, 0.02)])
def test_pure_letter_and_semantic_signs(other_gradient, common, semantic):
    norms, gradients = [1.0] * 12, [[0.0, 0.0] for _ in range(12)]
    norms[2], norms[0] = 2.0, 4.0
    gradients[2], gradients[0] = [-1.0, 0.0], [other_gradient, 0.0]
    value = solver.assemble(norms, gradients, [0.2] * 12, [0.2] * 12)
    da, db = 0.01 * value["A"][2][0], 0.01 * value["A"][0][0]
    assert (da - db) / 2 == pytest.approx(common)
    assert (da + db) / 2 == pytest.approx(semantic)


def test_all_negative_rhs_empty_mask_means_nonzero_d0_not_zero():
    A, b, c = problem([((1.0, 0.0), (-1.0, 0.0), -1.0, -1.0, 0.1)])
    accepted(A, b, c, witness(A, b, c, [-0.004, 0.0]))
    rejected(A, b, c, witness(A, b, c, [0.0, 0.0]))
    result = solver.solve(A, b, c)
    assert result["solution"]["active_mask"] == 0
    assert result["solution"]["vector"] == pytest.approx([-0.004, 0.0], abs=1e-12)
    accepted(A, b, c, result["solution"])


def test_one_dimensional_empty_solution_is_supported():
    A, b, c = problem([((1.0,), (-1.0,), -1.0, -1.0, 0.1)], dimension=1)
    result = solver.solve(A, b, c)
    assert result["status"] == "KKT_ESTIMATE_ONLY"
    assert result["solution"]["vector"] == pytest.approx([-0.004], abs=1e-12)
    accepted(A, b, c, result["solution"])


def test_zero_penalty_is_algebraic_limiting_identity_not_solver_switch():
    # Exact supplied isotropic witness: no coefficient sweep or production flag.
    F = Fraction
    d, mu = (F(0), F(1, 40)), F(1, 160)
    H0, q0 = ((F(1), F(0)), (F(0), F(1))), (F(0), F(0))
    assert tuple(sum(H0[i][j] * d[j] for j in range(2)) + q0[i] for i in range(2)) == (0, 4 * mu)
    assert 4 * d[1] == F(1, 10)
    assert -F(1, 20) <= d[0] <= F(1, 20)


def test_assemble_keeps_own_norms_negative_rhs_and_baseline_affine_drift():
    norms, gradients = [1.0] * 12, [[0.0, 0.0] for _ in range(12)]
    margins, baseline = [0.2] * 12, [0.2] * 12
    norms[2], norms[0] = 2.0, 4.0
    gradients[2], gradients[0] = [-1.0, 0.0], [1.0, 0.0]
    margins[2], margins[0], baseline[2], baseline[0] = 0.04, 0.20, -0.06, 0.30
    value = solver.assemble(norms, gradients, margins, baseline)
    assert value["A"][2] == [2.0, 0.0] and value["A"][0] == [-4.0, 0.0]
    assert value["b"][2] == pytest.approx(0.06) and value["b"][0] == pytest.approx(-0.10)
    assert value["c"] == pytest.approx([0.10, 0.0, 0.0, 0.0, 0.0, 0.0])
    # A +.01 shared displacement: letter movements .02,.04, not own-norm averaging.
    ma, mb = 0.01 * value["A"][2][0], 0.01 * value["A"][0][0]
    assert (ma - mb) / 2 == pytest.approx(0.03)
    assert (ma + mb) / 2 == pytest.approx(-0.01)


@pytest.mark.parametrize("fault", ["zero_norm", "negative_norm", "nan_norm", "shape"])
def test_assemble_rejects_invalid_original_norm_or_shape(fault):
    norms, gradients = [1.0] * 12, [[0.0, 0.0] for _ in range(12)]
    if fault == "shape":
        gradients[0].pop()
    else:
        norms[0] = {"zero_norm": 0.0, "negative_norm": -1.0, "nan_norm": math.nan}[fault]
    with pytest.raises((ValueError, TypeError)):
        solver.assemble(norms, gradients, [0.2] * 12, [0.2] * 12)


def test_nearly_opposed_rows_are_numerically_unresolved_not_infeasibility():
    A, b, c = problem([((1.0, 0.0), (-1.0, 1e-8), 0.1, 0.1, 0.0)])
    result = solver.solve(A, b, c)
    assert result["status"] == "NUMERICALLY_UNRESOLVED" and result["solution"] is None
    assert result["masks_visited"] == 4096
    rejected(A, b, c, witness(A, b, c, [0.1, 2e7], [(2, 2e15), (0, 2e15)]))


def test_exact_contradiction_remains_unresolved_without_farkas_claim():
    A, b, c = problem([((1.0, 0.0), (-1.0, 0.0), 0.1, 0.1, 0.0)])
    result = solver.solve(A, b, c)
    assert result["status"] == "NUMERICALLY_UNRESOLVED" and result["solution"] is None


def test_repeat_determinism_and_compact_mask_trace():
    A, b, c = affine()
    first = solver.solve(A, b, c)
    assert first == solver.solve(copy.deepcopy(A), b[:], c[:])
    trace = first["mask_trace"]
    assert len(trace) == first["masks_visited"] <= 4096
    assert sum(first["mask_status_counts"].values()) == first["masks_visited"]
    encoded = json.dumps(trace)
    assert len(encoded.encode()) <= 524288
    assert all(key not in encoded for key in ('"vector"', '"multipliers"', '"inverse"', '"gram"'))


def test_mask_arithmetic_failure_is_counted_and_fail_closed(monkeypatch):
    def broken(*args, **kwargs):
        raise ArithmeticError("synthetic direct-residual failure")

    monkeypatch.setattr(solver, "_residuals", broken)
    A, b, c = problem()
    result = solver.solve(A, b, c)
    assert result["status"] == "NUMERICALLY_UNRESOLVED"
    assert result["solution"] is None
    assert result["masks_visited"] == 1 and result["mask_trace"] == "X"


@pytest.mark.parametrize(
    "fault", ["primal", "dual", "stationarity", "complementarity", "fingerprint", "mask"]
)
def test_checker_rejects_corrupt_supplied_witness(fault):
    A, b, c = problem([((1.0, 0.0), (-1.0, 0.0), 0.06, -0.10, 0.1)])
    hand = witness(A, b, c, [0.06, 0.0], [(2, 1 / 15)])
    if fault == "primal":
        hand["vector"][0] = 0.05
    elif fault == "dual":
        hand["multipliers"][2] = -1e-30
    elif fault == "stationarity":
        hand["vector"][1] = 0.001
    elif fault == "complementarity":
        hand["vector"][0] = 0.08
        hand["multipliers"][2] = 0.08 + (0.1 + 0.08) / 24  # Stationary, feasible, noncomplementary.
    elif fault == "fingerprint":
        hand["input_sha256"] = "0" * 64
    else:
        hand["active_mask"] = 0
    rejected(A, b, c, hand)


def test_checker_verifies_hand_candidate_without_solver_call(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("checker delegated to solver")

    monkeypatch.setattr(solver, "solve", forbidden)
    A, b, c = affine()
    accepted(A, b, c, witness(A, b, c, [-0.04, 0.025], [(1, 1 / 160)]))


def test_fixed_residual_policy_does_not_accept_marginal_tau_band():
    A, b, c = problem()
    u = 2.0**-53
    gamma = (2 + 64) * u / (1 - (2 + 64) * u)
    tau = 128 * gamma
    # Zero A gives H=I,q=0, so the only nonzero residual is stationarity.
    accepted(A, b, c, witness(A, b, c, [tau / 16, 0.0]))
    rejected(A, b, c, witness(A, b, c, [tau / 2, 0.0]))


def test_dual_sign_is_exact_even_below_all_residual_tolerances():
    A, b, c = problem()
    hand = witness(A, b, c, [0.0, 0.0])
    hand["multipliers"][0] = -1e-30
    rejected(A, b, c, hand)


@pytest.mark.parametrize(
    "fault", ["nan", "infinity", "eleven_rows", "bad_dimension", "bad_b", "bad_c"]
)
def test_malformed_inputs_cannot_produce_certificate(fault):
    A, b, c = affine()
    hand = witness(A, b, c, [-0.04, 0.025], [(1, 1 / 160)])
    if fault == "nan":
        A[0][0] = math.nan
    elif fault == "infinity":
        c[0] = math.inf
    elif fault == "eleven_rows":
        A.pop()
    elif fault == "bad_dimension":
        A[0].pop()
    elif fault == "bad_b":
        b.pop()
    else:
        c.pop()
    try:
        result = solver.solve(A, b, c)
    except (ValueError, TypeError):
        pass
    else:
        assert result["status"] == "NUMERICALLY_UNRESOLVED" and result["solution"] is None
    rejected(A, b, c, hand)


@pytest.mark.parametrize(
    "fault", ["extra_key", "nan_vector", "inf_dual", "bool_mask", "too_large_mask"]
)
def test_malformed_solution_is_fail_closed(fault):
    A, b, c = affine()
    hand = witness(A, b, c, [-0.04, 0.025], [(1, 1 / 160)])
    if fault == "extra_key":
        hand["metrics"] = {"passed": True}
    elif fault == "nan_vector":
        hand["vector"][0] = math.nan
    elif fault == "inf_dual":
        hand["multipliers"][1] = math.inf
    elif fault == "bool_mask":
        hand["active_mask"] = True
    else:
        hand["active_mask"] = 4096
    rejected(A, b, c, hand)


def test_standard_library_runtime_has_no_ml_modules():
    assert not ({"torch", "transformers", "transformer_lens", "tokenizers"} & set(sys.modules))
