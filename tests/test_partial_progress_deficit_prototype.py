"""Preregistered tiny generated-PG and independent admission checks; native smoke excluded."""

import copy
import hashlib
import json
import struct
import time
from pathlib import Path

import pytest

from scripts import partial_progress_deficit_solver as solver
from scripts import verify_partial_progress_deficit as checker

FIXTURES = (
    Path(__file__).resolve().parents[1] / "configs/partial_progress_deficit_synthetic_fixtures.json"
)
SPEC = json.loads(FIXTURES.read_bytes())
CASES = {case["id"]: case for case in SPEC["cases"]}


def vector_sha(vector):
    return hashlib.sha256(struct.pack("<" + "d" * len(vector), *vector)).hexdigest()


def fingerprint(problem):
    return hashlib.sha256(
        json.dumps(problem, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def problem_for(name):
    case = CASES[name]
    A, b = [[0.0, 0.0] for _ in range(12)], [-1.0] * 12
    for index in (0, 2):
        A[index], b[index] = case[f"A_{index}"][:], case[f"b_{index}"]
    w = case["w"][:]
    return {
        "A": A,
        "b": b,
        "c": SPEC["defaults"]["c"][:],
        "w": w,
        "w_sha256": vector_sha(w),
        "history": copy.deepcopy(case["history"]),
        "path_upper": case["path_upper"],
    }


def context(problem):
    return checker.prepare(problem, fingerprint(problem), time.monotonic() + 10.0)


@pytest.fixture(scope="module")
def generated():
    # Exactly once per declared tiny case. Native fixture is deliberately absent.
    results = {}
    for name in CASES:
        result = solver.solve(problem_for(name), fingerprint(problem_for(name)))
        results[name] = result
        print(
            "PARTIAL_PROGRESS_TINY_RESULT="
            + json.dumps(
                {
                    "fixture": name,
                    **{
                        key: result.get(key)
                        for key in (
                            "status",
                            "terminal_reason",
                            "reason",
                            "iterations",
                            "certificate",
                            "witness",
                            "projection_case_counts",
                            "problem_sha256",
                            "step_admitted",
                            "elapsed_seconds",
                            "certificates_checked",
                            "exception_detail",
                        )
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            flush=True,
        )
    return results


@pytest.mark.parametrize("name", list(CASES))
def test_generated_projected_gradient_respects_preregistered_outcomes(name, generated):
    result = generated[name]
    assert result["status"] in CASES[name]["allowed_statuses"]
    assert 0 <= result["iterations"] <= 200
    assert result["problem_sha256"] == fingerprint(problem_for(name))
    if result["status"] == "NUMERICALLY_UNRESOLVED":
        assert result["step_admitted"] is False and result["w_next"] is None
    else:
        witness = result["witness"]
        cert = checker.check(
            context(problem_for(name)), witness["w_next"], witness["u"], witness["v"]
        )
        assert cert["status"] == result["status"]
        assert result["step_admitted"] is (result["status"] == "ADMITTED_PROGRESS")
        if result["step_admitted"]:
            assert result["w_next"] == witness["w_next"]


def test_two_active_ball_projection_is_actually_exercised(generated):
    assert generated["both_balls_active"]["projection_case_counts"].get("both_active", 0) > 0


def test_exact_zero_cases_do_not_force_movement(generated):
    for name in ("negative_rhs_zero", "opposing_zero"):
        assert generated[name]["status"] == "EXACT_ZERO_OPTIMUM"
        assert generated[name]["w_next"] is None


def test_binary64_literal_point20_is_rejected_without_repair():
    case = CASES["literal_net_boundary"]
    result = checker.check(
        context(problem_for("literal_net_boundary")),
        case["literal_candidate"],
        case["literal_u"],
        case["literal_v"],
    )
    assert result["status"] == "NUMERICALLY_UNRESOLVED"
    assert result["reason"] == "serialized_geometry_violation"


def test_rounding_erased_gain_is_not_admitted():
    problem = problem_for("rounding_erased_gain")
    assert problem["w"][0] + 1e-18 == problem["w"][0]
    result = checker.check(context(problem), problem["w"][:], [0.0, 0.0], [0.0, 0.0])
    assert result["status"] == "EPSILON_OPTIMAL_ZERO"


def test_independent_checker_accepts_hand_interior_without_solver(monkeypatch):
    monkeypatch.setattr(
        solver, "solve", lambda *args: pytest.fail("checker called production solver")
    )
    result = checker.check(context(problem_for("interior")), [1 / 70, 0.0], [0.0, 0.0], [0.0, 0.0])
    assert result["status"] == "ADMITTED_PROGRESS"


@pytest.mark.parametrize("fault", ["step", "normal_u", "normal_v", "nan_candidate", "nan_normal"])
def test_independent_checker_rejects_altered_candidate_or_normal(fault):
    candidate, u, v = [1 / 70, 0.0], [0.0, 0.0], [0.0, 0.0]
    if fault == "step":
        candidate[0] = 0.051
    elif fault == "normal_u":
        u[0] = 1.0
    elif fault == "normal_v":
        v[1] = 1.0
    elif fault == "nan_candidate":
        candidate[0] = float("nan")
    else:
        u[0] = float("nan")
    if fault in ("nan_candidate", "nan_normal"):
        with pytest.raises((ValueError, TypeError)):
            checker.check(context(problem_for("interior")), candidate, u, v)
    else:
        result = checker.check(context(problem_for("interior")), candidate, u, v)
        if fault == "step":
            assert result["status"] == "NUMERICALLY_UNRESOLVED"
            assert result["reason"] == "serialized_geometry_violation"
        else:
            assert result["status"] == "NOT_CERTIFIED"
            assert result["reason"] == "gap_or_gain_not_certified"


@pytest.mark.parametrize(
    "fault",
    [
        "fingerprint",
        "w_hash",
        "last_history",
        "first_history",
        "history_step",
        "path_low",
        "path_high",
        "path_negative",
    ],
)
def test_authenticated_current_history_and_path_tampering_is_rejected(fault):
    problem = problem_for("literal_net_boundary")
    if fault == "fingerprint":
        expected = "0" * 64
    else:
        if fault == "w_hash":
            problem["w_sha256"] = "0" * 64
        elif fault == "last_history":
            problem["history"][-1][0] = 0.18
        elif fault == "first_history":
            problem["history"][0][0] = 0.001
        elif fault == "history_step":
            problem["history"] = [[0.0, 0.0], problem["w"][:]]
        elif fault == "path_low":
            problem["path_upper"] = "1/10"
        elif fault == "path_high":
            problem["path_upper"] = "41/100"
        else:
            problem["path_upper"] = "-1/100"
        expected = fingerprint(problem)
    with pytest.raises((ValueError, TypeError)):
        checker.prepare(problem, expected, time.monotonic() + 10.0)


def test_expired_deadline_does_not_issue_admission():
    problem = problem_for("interior")
    with pytest.raises((ValueError, TimeoutError)):
        checker.prepare(problem, fingerprint(problem), time.monotonic() - 1.0)


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_deadline_is_rejected(deadline):
    problem = problem_for("interior")
    with pytest.raises(ValueError, match="finite_deadline_required"):
        checker.prepare(problem, fingerprint(problem), deadline)


def test_exponent_path_is_rejected_before_fraction_parsing(monkeypatch):
    problem = problem_for("interior")
    problem["path_upper"] = "1e-1000000000"
    original_fraction = checker.F

    def checked_fraction(value=0, *args, **kwargs):
        assert value != problem["path_upper"], "unbounded exponent reached rational parser"
        return original_fraction(value, *args, **kwargs)

    checked_fraction.from_float = original_fraction.from_float
    monkeypatch.setattr(checker, "F", checked_fraction)
    with pytest.raises(ValueError, match="bounded_rational_path_upper_required"):
        checker.prepare(problem, fingerprint(problem), time.monotonic() + 10.0)


def test_solver_float_copies_cannot_mutate_sealed_admission_inputs():
    problem = problem_for("interior")
    ctx = context(problem)
    candidate, zero = [1 / 70, 0.0], [0.0, 0.0]
    before = checker.check(ctx, candidate, zero, zero)
    assert before["status"] == "ADMITTED_PROGRESS"
    ctx["A"][2][0], ctx["b"][2], ctx["w"][0] = 1e6, 1e6, 0.2
    ctx["c"][0], ctx["D"][0][0], ctx["M"], ctx["rho"] = 1e6, 1e6, 1.0, 1.0
    problem["A"][2][0], problem["b"][2], problem["w"][0] = -1e6, -1e6, -0.2
    after = checker.check(ctx, candidate, zero, zero)
    assert after == before


def test_native_recipe_is_dense_fixed_and_excluded_from_pytest_solve():
    # Metadata-only inspection: do not materialize, solve, certify or benchmark it here.
    native = SPEC["native_smoke"]
    assert native["dimension"] == 1024 and native["rows"] == 12
    assert native["executions"] == 1 and native["hard_seconds"] == 10
    assert native["pytest_execution_forbidden"] is True
    assert (
        hashlib.sha256(native["canonical_problem_json"].encode()).hexdigest()
        == native["problem_sha256"]
    )
    assert "dense_native_once" not in CASES
