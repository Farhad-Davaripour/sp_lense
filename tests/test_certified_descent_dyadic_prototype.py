"""Frozen tiny synthetic rays only; the native smoke is never parsed or solved here."""

import copy
import hashlib
import json
import struct
import time
from fractions import Fraction as F
from pathlib import Path

import pytest

from scripts import certified_descent_dyadic_solver as solver
from scripts import verify_certified_descent_dyadic as checker

SPEC = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "configs/certified_descent_dyadic_synthetic_fixtures.json"
    ).read_bytes()
)
CASES = {case["id"]: case for case in SPEC["cases"]}


def fingerprint(problem):
    return hashlib.sha256(
        json.dumps(problem, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def vector_sha(vector):
    return hashlib.sha256(struct.pack("<" + "d" * len(vector), *vector)).hexdigest()


def problem_for(name):
    case = CASES[name]
    A, b = [[0, 0] for _ in range(12)], [-1] * 12
    for key, row in case["rows"].items():
        A[int(key)], b[int(key)] = row["A"][:], row["b"]
    w = case["w"][:]
    return {
        "A": A,
        "b": b,
        "c": case.get("c", SPEC["defaults"]["c"])[:],
        "w": w,
        "w_sha256": vector_sha(w),
        "history": copy.deepcopy(case["history"]),
        "path_upper": case["path_upper"],
    }


def context(problem):
    return checker.prepare(problem, fingerprint(problem), time.monotonic() + 10)


def components(problem, endpoint):
    # Independent rational test evaluator; never imports production evaluator math.
    r = [F(x) - F(w) for x, w in zip(endpoint, problem["w"])]
    deficits = [
        max(F(0), F(b) - sum(F(a) * v for a, v in zip(row, r)))
        for row, b in zip(problem["A"], problem["b"])
    ]
    drift = [
        F(c) + sum((F(a) - F(b)) * v / 2 for a, b, v in zip(problem["A"][ia], problem["A"][ib], r))
        for c, (ia, ib) in zip(problem["c"], SPEC["pairs_zero_based"])
    ]
    values = {
        "proximal": sum(x * x for x in r) / 2,
        "deficit": sum(x * x for x in deficits) / 24,
        "drift": sum(x * x for x in drift) / 48,
    }
    values["total"] = sum(values.values())
    return values


def objective(problem, endpoint):
    return components(problem, endpoint)["total"]


def zero_gradient(problem):
    return [
        -sum(F(row[k]) * max(F(0), F(b)) for row, b in zip(problem["A"], problem["b"])) / 12
        + sum(
            (F(problem["A"][ia][k]) - F(problem["A"][ib][k])) * F(c) / 2
            for c, (ia, ib) in zip(problem["c"], SPEC["pairs_zero_based"])
        )
        / 24
        for k in range(len(problem["w"]))
    ]


@pytest.fixture(scope="module")
def generated():
    results = {}
    for name in CASES:
        problem = problem_for(name)
        result = solver.solve(problem, fingerprint(problem))
        results[name] = result
        print(
            "CERTIFIED_DESCENT_TINY_RESULT="
            + json.dumps(
                {"fixture": name, **result}, sort_keys=True, separators=(",", ":"), allow_nan=False
            ),
            flush=True,
        )
    return results


@pytest.mark.parametrize("name", list(CASES))
def test_generated_frozen_outcomes(name, generated):
    result = generated[name]
    assert result["status"] == CASES[name]["expected_status"]
    assert result["gradient_count"] <= 1
    assert result["proposal_count"] <= 1
    assert result["trial_count"] == len(result["trials"]) <= 53
    assert [trial["j"] for trial in result["trials"]] == list(range(result["trial_count"]))
    assert all(trial["lambda"] == 2.0 ** -trial["j"] for trial in result["trials"])
    if result["status"] == "ADMITTED_DESCENT":
        assert result["chosen_j"] == result["trials"][-1]["j"]
        assert all(t["certificate"]["status"] != "ADMITTED_DESCENT" for t in result["trials"][:-1])
        p = problem_for(name)
        assert objective(p, result["w_next"]) < objective(p, p["w"])
        r = [F(x) - F(w) for x, w in zip(result["w_next"], p["w"])]
        s = -sum(g * x for g, x in zip(zero_gradient(p), r))
        J0 = objective(p, p["w"])
        assert s > 0
        assert J0 - objective(p, result["w_next"]) > s / 4 + F(1, 2**30) * max(F(1), J0)
        assert checker.check(context(p), result["w_next"])["status"] == "ADMITTED_DESCENT"
    else:
        assert result["w_next"] is None
        assert result["chosen_j"] is None


def test_exhausted_ray_is_not_stationarity(generated):
    result = generated["ray_below_floor"]
    assert result["trial_count"] == 53
    assert result["status"] == "NO_CERTIFIED_STEP"
    assert any(zero_gradient(problem_for("ray_below_floor")))
    assert generated["erased_gain"]["status"] == "NO_CERTIFIED_STEP"
    assert generated["erased_gain"]["trial_count"] <= 53


@pytest.mark.parametrize("name", ["positive_deficits_zero", "negative_deficits_zero"])
def test_exact_stationarity_is_not_zero_loss(name, generated):
    p = problem_for(name)
    assert not any(zero_gradient(p))
    assert generated[name]["gradient_count"] == 0
    assert generated[name]["proposal_count"] == 0
    assert generated[name]["trial_count"] == 0
    if name == "positive_deficits_zero":
        assert objective(p, p["w"]) > 0


def test_supplied_boundary_rounding_and_half_step():
    p = problem_for("both_active_boundary")
    full = CASES["both_active_boundary"]["supplied_full_endpoint"]
    half = CASES["both_active_boundary"]["supplied_half_endpoint"]
    assert sum(F(x) ** 2 for x in full) > F(1, 25)
    assert checker.check(context(p), full)["status"] != "ADMITTED_DESCENT"
    assert checker.check(context(p), half)["status"] == "ADMITTED_DESCENT"


def test_checker_independent_of_generator(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Checker called the proposal generator")

    monkeypatch.setattr(solver, "solve", forbidden)
    monkeypatch.setattr(solver.generation, "gradient", forbidden)
    monkeypatch.setattr(solver.generation, "project_intersection", forbidden)
    p = problem_for("affine_interior")
    endpoint = [p["w"][0] + 1 / 290, 0]
    assert checker.check(context(p), endpoint)["status"] == "ADMITTED_DESCENT"


@pytest.mark.parametrize("fault", ["hash", "w_hash", "history", "path"])
def test_input_authentication_rejects_tampering(fault):
    p = problem_for("affine_interior")
    expected = fingerprint(p)
    if fault == "hash":
        p["b"][2] += 0.01
    elif fault == "w_hash":
        p["w_sha256"] = "0" * 64
    elif fault == "history":
        p["history"][-1][0] += 0.01
    else:
        p["path_upper"] = "0"
    if fault != "hash":
        expected = fingerprint(p)
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        checker.prepare(p, expected, time.monotonic() + 10)


@pytest.mark.parametrize("endpoint", [[0.3, 0], [0.1, 0], [0.09, 0]])
def test_forged_norm_zero_or_uphill_point_cannot_admit(endpoint):
    p = problem_for("affine_interior")
    assert checker.check(context(p), endpoint)["status"] != "ADMITTED_DESCENT"


def test_nonfinite_endpoint_rejected():
    with pytest.raises(ValueError):
        checker.check(context(problem_for("affine_interior")), [float("nan"), 0])


def test_actual_generated_halving(generated):
    result = generated["net_axis_halving"]
    assert result["chosen_j"] == CASES["net_axis_halving"]["expected_chosen_j"]
    assert result["trials"][0]["certificate"]["geometry_valid"] is False
    assert result["trials"][1]["certificate"]["status"] == "ADMITTED_DESCENT"
    assert result["proposal_count"] == 1


def test_large_M_descriptive_bound_is_not_gate(generated):
    result = generated["large_M_admitted"]
    assert result["status"] == "ADMITTED_DESCENT"
    assert F(result["descriptive_gap"]["descriptive_gap_upper"]) > F(
        result["selected_certificate"]["eta"]
    )


@pytest.mark.parametrize("name", list(CASES))
def test_every_actual_trial_exact_geometry_cost_and_identity(name, generated):
    p = problem_for(name)
    result = generated[name]
    for trial in result["trials"]:
        endpoint = [float(w) + trial["lambda"] * v for w, v in zip(p["w"], result["proposal"]["p"])]
        actual = [F(x) - F(w) for x, w in zip(endpoint, p["w"])]
        cert = trial["certificate"]
        geo = cert["geometry"]
        assert cert["w_next_sha256"] == vector_sha(endpoint)
        assert [F(v) for v in cert["actual_displacement"]] == actual
        step2 = sum(x * x for x in actual)
        net2 = sum(F(x) ** 2 for x in endpoint)
        path = F(p["path_upper"])
        rho = min(F(1, 20), F(2, 5) - path)
        assert F(geo["actual_step_squared"]) == step2
        assert F(geo["net_squared"]) == net2
        assert F(geo["rho"]) == rho
        assert F(geo["path_before_upper"]) == path
        norm_bound = F(geo["path_after_upper"]) - path
        assert norm_bound >= 0 and norm_bound**2 >= step2
        assert norm_bound == 0 or (norm_bound - F(1, 2**80)) ** 2 < step2
        assert cert["geometry_valid"] == (
            step2 <= rho * rho and net2 <= F(1, 25) and path + norm_bound <= F(2, 5)
        )
        assert {key: F(value) for key, value in cert["components_before"].items()} == components(
            p, p["w"]
        )
        assert {key: F(value) for key, value in cert["components_after"].items()} == components(
            p, endpoint
        )
        gain = objective(p, p["w"]) - objective(p, endpoint)
        s = -sum(g * r for g, r in zip(zero_gradient(p), actual))
        assert F(cert["gain"]) == gain and F(cert["s"]) == s
        assert F(cert["sufficient_decrease_excess"]) == gain - s / 4 - F(cert["eta"])


def test_sealed_candidate_math_ignores_mutable_copies_and_old_objective(monkeypatch):
    p = problem_for("affine_interior")
    ctx = context(p)
    endpoint = [0.1 + 1 / 290, 0]
    reference = checker.check(ctx, endpoint)

    def forbidden(*args, **kwargs):
        raise AssertionError("candidate checker used old numerical oracle")

    monkeypatch.setattr(solver.generation, "gradient", forbidden)
    monkeypatch.setattr(checker.auth, "_objective", forbidden)
    for key in ("A", "b", "c", "w"):
        ctx[key] = None
    assert checker.check(ctx, endpoint) == reference


@pytest.mark.parametrize("phase", ["prepare", "diagnostic", "serialization"])
def test_deadline_cannot_issue_endpoint(monkeypatch, phase):
    clock = [0.0]
    monkeypatch.setattr(solver.time, "monotonic", lambda: clock[0])
    if phase == "prepare":
        original = checker.prepare

        def late(*args, **kwargs):
            value = original(*args, **kwargs)
            clock[0] = 11.0
            return value

        monkeypatch.setattr(checker, "prepare", late)
    elif phase == "diagnostic":
        original = checker.diagnostic

        def late(*args, **kwargs):
            value = original(*args, **kwargs)
            clock[0] = 11.0
            return value

        monkeypatch.setattr(checker, "diagnostic", late)
    else:
        original = solver.encoded

        def late(value):
            result = original(value)
            if value.get("status") == "ADMITTED_DESCENT":
                clock[0] = 11.0
            return result

        monkeypatch.setattr(solver, "encoded", late)
    p = problem_for("affine_interior")
    result = solver.solve(p, fingerprint(p))
    assert result["status"] == "NO_CERTIFIED_STEP"
    assert result["w_next"] is None and result["step_admitted"] is False
    assert result["terminal_reason"] == "COMBINED_TEN_SECOND_DEADLINE"
    assert result["gradient_count"] == (0 if phase == "prepare" else 1)


def test_incomplete_full_result_serialization_cannot_issue(monkeypatch):
    original = solver.encoded
    calls = []

    def fail_full(value):
        calls.append(value["status"])
        if value["status"] == "ADMITTED_DESCENT":
            raise ValueError("test-only interrupted JSON encoding")
        return original(value)

    monkeypatch.setattr(solver, "encoded", fail_full)
    p = problem_for("affine_interior")
    result = solver.solve(p, fingerprint(p))
    assert calls == ["ADMITTED_DESCENT", "NO_CERTIFIED_STEP"]
    assert result["status"] == "NO_CERTIFIED_STEP" and result["w_next"] is None
    assert result["failure_receipt_only"] is True
    assert result["terminal_reason"].startswith("RESULT_SERIALIZATION_FAILED:")


def test_total_encoder_failure_is_unserialized_no_step(monkeypatch):
    def fail(value):
        raise ValueError("test-only complete encoder outage")

    monkeypatch.setattr(solver, "encoded", fail)
    p = problem_for("affine_interior")
    result = solver.solve(p, fingerprint(p))
    assert result["status"] == "NO_CERTIFIED_STEP" and result["w_next"] is None
    assert result["serialization_checked"] is False


@pytest.mark.parametrize("fail", [False, True])
def test_observer_cannot_authorize_or_mutate_result(fail):
    def observer(snapshot):
        snapshot.clear()
        snapshot["status"] = "ADMITTED_DESCENT"
        if fail:
            raise ValueError("test-only observer outage")
        return {"w_next": [123, 123]}

    p = problem_for("affine_interior")
    result = solver.solve(p, fingerprint(p), observer=observer)
    if fail:
        assert result["status"] == "NO_CERTIFIED_STEP" and result["w_next"] is None
    else:
        assert result["status"] == "ADMITTED_DESCENT"
        assert result["w_next"] != [123, 123]
        assert checker.check(context(p), result["w_next"])["status"] == "ADMITTED_DESCENT"


def test_interrupted_trial_preserves_attempt_without_step(monkeypatch):
    def interrupted(*args, **kwargs):
        raise TimeoutError("test-only interrupted check")

    monkeypatch.setattr(checker, "check", interrupted)
    p = problem_for("affine_interior")
    result = solver.solve(p, fingerprint(p))
    assert result["status"] == "NO_CERTIFIED_STEP" and result["w_next"] is None
    assert result["trial_count"] == len(result["trials"]) == 1
    assert result["trials"][0]["j"] == 0
    assert result["trials"][0].get("certificate") is None


def test_native_metadata_hash_only():
    native = SPEC["native_smoke"]
    assert native["executions"] == 1 and native["hard_seconds"] == 10
    assert native["pytest_execution_forbidden"] is True
    assert (
        hashlib.sha256(native["canonical_problem_json"].encode()).hexdigest()
        == native["problem_sha256"]
    )
    # Intentionally no json.loads(native["canonical_problem_json"]) or solver call.
