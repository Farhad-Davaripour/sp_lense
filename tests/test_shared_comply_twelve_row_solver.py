"""Focused synthetic bound-extension, enumeration, and independent KKT checks."""

from __future__ import annotations

import ast
import copy
import inspect
import json
import math
import subprocess
import textwrap
from decimal import localcontext

import pytest

from scripts import benchmark_shared_comply_twelve as benchmark
from scripts import shared_comply_twelve_row_solver as solver
from scripts import shared_preserve_eight_row_solver as original
from scripts import verify_shared_direction_feasibility as independent

CONFIG = {
    "rank_relative_pivot_floor": 1e-12,
    "primal_absolute_tolerance": 1e-9,
    "kkt_absolute_tolerance": 1e-8,
}


def orthogonal():
    return [[float(i == j) for j in range(12)] for i in range(12)]


def certify(A, b, vector, multipliers):
    with localcontext() as context:
        context.prec = 80
        return independent.certificate(
            [[independent.Interval(x) for x in row] for row in A],
            [independent.Interval(x) for x in b],
            vector,
            multipliers,
            {
                **CONFIG,
                "interval_safety_absolute": "1e-40",
                "interval_safety_relative": "1e-40",
                "dual_denominator_floor": "1e-24",
                "radius": 0.2,
                "radius_comparison_guard": "1e-12",
            },
        )


@pytest.fixture(scope="module")
def positive():
    A, b = orthogonal(), [0.125] * 12
    return A, b, solver.solve(A, b, CONFIG)


@pytest.fixture(scope="module")
def redundant():
    A, b = [[1.0]] * 12, [0.125] * 12
    return A, b, solver.solve(A, b, CONFIG)


def test_only_upper_bound_changes_compiled_function():
    source = textwrap.dedent(inspect.getsource(original.solve))
    expected = ast.parse(source)
    constants = [
        node for node in ast.walk(expected) if isinstance(node, ast.Constant) and node.value == 8
    ]
    assert len(constants) == 1
    constants[0].value = 12
    namespace = dict(vars(original))
    exec(compile(expected, "expected", "exec"), namespace)  # noqa: S102 - trusted immutable definition; test exact compiled equivalence.
    expected_code = namespace["solve"].__code__
    assert solver.solve.__code__.co_code == expected_code.co_code
    assert solver.solve.__code__.co_consts == expected_code.co_consts
    for name in ("linear_solve", "metrics", "kkt_valid", "dot", "norm"):
        assert solver.solve.__globals__[name] is getattr(original, name)


@pytest.mark.parametrize(
    "change",
    [
        lambda source: source.replace("<= 8", "<= 12"),
        lambda source: source.replace("<= 8", "< 8"),
        lambda source: source + "\nextra = 8\n",
        lambda source: source.replace("def solve(", "def alternative("),
    ],
)
def test_bound_adaptation_rejects_unexpected_shape(change):
    with pytest.raises(Exception, match="single original eight-row bound"):
        solver.extend_bound(change(inspect.getsource(original.solve)))


def test_historical_parent_remains_eight_rows():
    with pytest.raises(Exception, match="finite small native system"):
        original.solve(orthogonal(), [0.125] * 12, CONFIG)
    assert "<= 8" in inspect.getsource(original.solve)


def test_full4096_masks_and_all_bits_are_retained(positive):
    _, _, result = positive
    assert result["status"] == "KKT_ESTIMATE_ONLY"
    assert [entry["mask"] for entry in result["active_sets"]] == list(range(4096))
    for entry in result["active_sets"]:
        assert entry["active"] == [i for i in range(12) if entry["mask"] & (1 << i)]
        assert len(entry["pivots"]) <= len(entry["active"])
        assert "vector" not in entry and "multipliers" not in entry
    assert result["solution"]["active_mask"] == 4095


def test_exact_positive_solution_passes_independent_certificate(positive):
    A, b, result = positive
    selected = result["solution"]
    assert selected["vector"] == selected["multipliers"] == [0.125] * 12
    cert = certify(A, b, selected["vector"], selected["multipliers"])
    assert cert["kkt_verified"] and cert["lambda_nonnegative"]
    independent.compare(selected["metrics"], cert["metrics"], 1e-9)
    assert cert["denominator_is_exact_infeasibility_claim"] is False


def test_negative_rhs_is_not_absolute_valued_or_clipped():
    A, b = orthogonal(), [0.125] + [-0.25] * 11
    result = solver.solve(A, b, CONFIG)
    assert result["solution"]["active_mask"] == 1
    assert result["solution"]["vector"] == [0.125] + [0.0] * 11
    assert certify(A, b, result["solution"]["vector"], result["solution"]["multipliers"])[
        "kkt_verified"
    ]


def test_all_negative_rhs_selects_mask_zero():
    A, b = orthogonal(), [-0.125] * 12
    result = solver.solve(A, b, CONFIG)
    assert result["solution"]["active_mask"] == 0
    assert result["solution"]["vector"] == result["solution"]["multipliers"] == [0.0] * 12
    assert certify(A, b, result["solution"]["vector"], result["solution"]["multipliers"])[
        "kkt_verified"
    ]


def test_c_direction_sign_is_preserved_by_minimum_norm_constraints():
    result = solver.solve([[-1.0]] * 12, [0.125] * 12, CONFIG)
    assert result["solution"]["vector"] == [-0.125]
    assert result["solution"]["multipliers"] == [0.125] + [0.0] * 11


def test_redundancy_rank_skip_and_equal_objective_tie_remain_deterministic(redundant):
    A, b, result = redundant
    assert result["solution"]["active_mask"] == 1
    assert result["solution"]["vector"] == [0.125]
    singletons = [entry for entry in result["active_sets"] if len(entry["active"]) == 1]
    assert len(singletons) == 12
    assert all(entry["status"] == "kkt_valid" for entry in singletons)
    assert len({entry["metrics"]["primal_objective"] for entry in singletons}) == 1
    assert all(
        entry["status"] == "rank_deficient_or_near_dependent_skipped"
        for entry in result["active_sets"]
        if len(entry["active"]) > 1
    )
    assert certify(A, b, result["solution"]["vector"], result["solution"]["multipliers"])[
        "kkt_verified"
    ]


def test_infeasible_system_remains_unresolved_without_certificate():
    result = solver.solve([[1.0]] * 6 + [[-1.0]] * 6, [0.125] * 12, CONFIG)
    assert result["status"] == "NUMERICALLY_UNRESOLVED"
    assert result["solution"] is None
    assert len(result["active_sets"]) == 4096
    assert set(result) == {"status", "solution", "active_sets"}


def test_near_dependent_pivot_uses_original_floor():
    A = [[1.0, 0.0], [1.0, 1e-8]] + [[1.0, 0.0]] * 10
    result = solver.solve(A, [0.125] * 12, CONFIG)
    assert result["active_sets"][3]["status"] == "rank_deficient_or_near_dependent_skipped"
    assert result["solution"]["active_mask"] == 1


@pytest.mark.parametrize("rhs, expected_mask", [(0.5e-9, 0), (2e-9, 1)])
def test_marginal_primal_tolerance_is_unchanged(rhs, expected_mask):
    result = solver.solve([[1.0]] * 12, [rhs] * 12, CONFIG)
    assert result["solution"]["active_mask"] == expected_mask


@pytest.mark.parametrize(
    "A,b",
    [
        ([], []),
        ([[1.0]] * 13, [0.125] * 13),
        ([[1.0]] * 12, [0.125] * 11),
        ([[]] * 12, [0.125] * 12),
        ([[1.0], [1.0, 2.0]], [0.125, 0.125]),
        ([[math.nan]] * 12, [0.125] * 12),
        ([[1.0]] * 12, [math.inf] * 12),
    ],
)
def test_finite_small_native_guard(A, b):
    with pytest.raises(Exception, match="finite small native system"):
        solver.solve(A, b, CONFIG)


@pytest.mark.parametrize(
    "fault", ["vector", "negative_dual", "stationarity", "rhs", "constraint_sign"]
)
def test_independent_certificate_catches_corrupted_selected_solution(positive, fault):
    A, b, result = copy.deepcopy(positive)
    vector, multipliers = result["solution"]["vector"], result["solution"]["multipliers"]
    if fault == "vector":
        vector[0] -= 0.01
    elif fault == "negative_dual":
        multipliers[0] = -0.125
    elif fault == "stationarity":
        multipliers[0] += 0.01
    elif fault == "rhs":
        b[0] += 0.01
    else:
        A[0][0] = -A[0][0]
    assert not certify(A, b, vector, multipliers)["kkt_verified"]


def test_independent_metric_comparison_catches_fabrication(positive):
    A, b, result = copy.deepcopy(positive)
    solution = result["solution"]
    cert = certify(A, b, solution["vector"], solution["multipliers"])
    solution["metrics"]["primal_objective"] += 0.01
    with pytest.raises(ValueError):
        independent.compare(solution["metrics"], cert["metrics"], 1e-9)


def test_journal_bound_includes_all4096_full_schema_entries(positive):
    bound = benchmark.journal_bound()
    assert bound["maximum_float_fields_per_mask"] == 32
    assert bound["maximum_4096_mask_journal_bytes"] < 8 * 1024 * 1024
    assert (
        bound["maximum_eight_update_mask_journals_bytes"]
        == 8 * bound["maximum_4096_mask_journal_bytes"]
    )
    assert bound["vectors_per_mask"] == 0
    assert (
        len(json.dumps(positive[2]["active_sets"], sort_keys=True).encode())
        <= bound["maximum_4096_mask_journal_bytes"]
    )
    largest = max(
        len(repr(x))
        for x in [
            float.fromhex("0x1.fffffffffffffp+1023"),
            -float.fromhex("0x1.fffffffffffffp+1023"),
            -5e-324,
        ]
    )
    assert largest <= bound["finite_binary64_bytes_reserved"]


def test_benchmark_has_only_four_fixed_native_cases_and_no_model_calls():
    declaration = benchmark.DECLARATION
    assert declaration["cases"] == [
        "positive_orthogonal",
        "negative_rhs_zero_optimum",
        "rank_deficient_redundant",
        "opposing_infeasible",
    ]
    assert declaration["per_case_wall_limit_seconds"] == 20
    assert declaration["whole_benchmark_wall_limit_seconds"] == 90
    assert declaration["maximum_case_attempts"] == 1
    assert all(
        declaration[key] == 0
        for key in ["model_loads", "tokenizer_loads", "forwards", "derivatives"]
    )
    for name in declaration["cases"]:
        A, b, config = benchmark.problem(name)
        assert len(A) == len(b) == 12 and all(len(row) == 1024 for row in A)
        assert config == CONFIG
    with pytest.raises(ValueError, match="fixed benchmark"):
        benchmark.problem("fallback")


def test_benchmark_timeouts_are_external_preserved_and_not_retried(monkeypatch):
    calls = []

    def timed_out(command, **kwargs):
        calls.append((command, kwargs))
        raise subprocess.TimeoutExpired(
            command, kwargs["timeout"], output=b"partial", stderr=b"diagnostic"
        )

    monkeypatch.setattr(benchmark.subprocess, "run", timed_out)
    result = benchmark.benchmark()
    assert result["status"] == "AFFORDABILITY_UNVERIFIED"
    assert len(calls) == 4
    assert [row["case"] for row in result["cases"]] == benchmark.DECLARATION["cases"]
    assert all(
        row["status"] == "TIMEOUT"
        and row["partial_stdout"] == "partial"
        and row["retry_allowed"] is False
        for row in result["cases"]
    )
    assert all(
        0 < kwargs["timeout"] <= 20 and kwargs["capture_output"] and kwargs["check"] is False
        for _, kwargs in calls
    )


def test_benchmark_whole_deadline_stops_without_later_attempts(monkeypatch):
    ticks = iter([0.0, 91.0, 92.0, 93.0, 94.0, 95.0])
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        benchmark.subprocess, "run", lambda *args, **kwargs: pytest.fail("deadline exceeded")
    )
    result = benchmark.benchmark()
    assert result["status"] == "AFFORDABILITY_UNVERIFIED"
    assert all(row["status"] == "NOT_ATTEMPTED_WHOLE_LIMIT" for row in result["cases"])
