"""Focused tests of changed formulations, reuse, separate certificates and split."""

from __future__ import annotations

import copy
import subprocess
import sys
from decimal import Decimal, localcontext

import pytest

from scripts import shared_direction_feasibility_io as old_io
from scripts import shared_direction_linear_feasibility as old_solver
from scripts import two_outcome_feasibility_io as io
from scripts import two_outcome_linear_feasibility as solver
from scripts import verify_shared_direction_feasibility as old_audit
from scripts import verify_two_outcome_feasibility as audit


@pytest.fixture
def config():
    return io.read(io.ROOT / io.CONFIG)


def row(S=0.0, g=None, dataset="f01_v1", order="preserve_first"):
    return {
        "dataset": dataset,
        "order": order,
        "prompt_id": dataset + order,
        "initial_gradient_cell_id": dataset + order + "__gradient_1",
        "S": S,
        "h0": [1.0, 0.0],
        "g": [1.0, 0.0] if g is None else g,
    }


def test_exact_reuse_and_unchanged_numeric_policy(config):
    previous = io.read(io.ROOT / old_io.CONFIG)
    for key, value in previous.items():
        if key not in ("schema", "output_namespace", "provenance_source_sha256"):
            assert config[key] == value
    assert solver.solve is old_solver.solve
    assert audit.certificate is old_audit.certificate
    assert audit.compare is old_audit.compare
    assert io.initial_rows is old_io.initial_rows


def test_negative_rhs_is_not_abs_clamped_or_dropped(config):
    records = [row(-0.4), row(0.4), row(-0.2), row(0.2)]
    Ap, bp = solver.formulation(records, "preserve", config["margin"])
    Ac, bc = solver.formulation(records, "comply", config["margin"])
    assert len(Ap) == len(Ac) == len(bp) == len(bc) == 4
    assert bp == [0.45, -0.35000000000000003, 0.25, -0.15000000000000002]
    assert bc == [-0.35000000000000003, 0.45, -0.15000000000000002, 0.25]
    assert Ap == [[1.0, 0.0]] * 4
    assert Ac == [[-1.0, -0.0]] * 4


def test_negative_rhs_can_be_an_active_constraint(config):
    A, b = [[1.0, 0.0], [-1.0, 1.0]], [1.0, -0.2]
    solution = solver.solve(A, b, config)["solution"]
    assert solution["active"] == [0, 1]
    assert solution["vector"] == pytest.approx([1.0, 0.8])
    assert min(solution["multipliers"]) > 0
    with localcontext() as ctx:
        ctx.prec = 80
        cert = audit.certificate(
            [[old_audit.Interval(x) for x in a] for a in A],
            [old_audit.Interval(str(x)) for x in b],
            solution["vector"],
            solution["multipliers"],
            config,
        )
    assert cert["kkt_verified"]
    assert cert["decision"] == "OUTSIDE_RADIUS_LINEAR_SURROGATE_CERTIFIED"


def test_comply_vector_is_applied_without_an_extra_negative(config):
    records = [row()]
    A, b = solver.formulation(records, "comply", config["margin"])
    solution = solver.solve(A, b, config)["solution"]
    assert solution["vector"] == [-0.05, 0.0]
    observed = solver.table(records, solution["vector"], "comply", config["margin"])
    with localcontext() as ctx:
        ctx.prec = 80
        expected = audit.table(records, solution["vector"], "comply", config)
        audit.compare(observed, expected, 1e-9)
    assert observed[0]["predicted_signed_margin"] == 0.05
    wrong = solver.table(records, [0.05, 0.0], "comply", config["margin"])
    assert wrong[0]["predicted_signed_margin"] == -0.05


def test_retention_constraint_changes_feasibility(config):
    records = [row(-0.4, [1.0, 0.0]), row(0.4, [-1.0, 0.0])]
    A, b = solver.formulation(records, "preserve", config["margin"])
    assert b[1] < 0
    assert solver.table(records, None, "preserve", config["margin"])[1]["baseline_already_correct"]
    assert solver.solve(A[:1], b[:1], config)["solution"] is not None
    assert solver.solve(A, b, config)["solution"] is None  # Must not skip already-correct row.


@pytest.mark.parametrize(
    "other", ["OUTSIDE_RADIUS_LINEAR_SURROGATE_CERTIFIED", "NUMERICALLY_UNRESOLVED"]
)
def test_one_within_outcome_is_not_both(other):
    certs = {
        "preserve": {"decision": "WITHIN_RADIUS_LINEAR_SURROGATE_VERIFIED"},
        "comply": {"decision": other},
    }
    result = audit.joint_status(certs)
    assert not result["both_outcomes_within_radius_verified"]
    assert not result["causal_pass"]


def test_separate_caps_not_joint_sum(config):
    with localcontext() as ctx:
        ctx.prec = 80
        cert = audit.certificate(
            [
                [old_audit.Interval(1), old_audit.Interval(0)],
                [old_audit.Interval(0), old_audit.Interval(1)],
            ],
            [old_audit.Interval("0.1")] * 2,
            [0.1, 0.1],
            [0.1, 0.1],
            config,
        )
    assert cert["decision"] == "WITHIN_RADIUS_LINEAR_SURROGATE_VERIFIED"
    assert 2 * cert["metrics"]["norm"] > 0.20
    result = audit.joint_status({"preserve": cert, "comply": copy.deepcopy(cert)})
    assert result["both_outcomes_within_radius_verified"]
    assert not result["joint_sum_of_norms_cap_used"]
    assert not result["causal_pass"]


def test_each_outside_outcome_needs_its_own_dual_certificate(config):
    with localcontext() as ctx:
        ctx.prec = 80
        valid = audit.certificate(
            [[old_audit.Interval(-1)]], [old_audit.Interval(1)], [-1.0], [1.0], config
        )
        bad = audit.certificate(
            [[old_audit.Interval(1)]], [old_audit.Interval("0.1")], [10.0], [0.0], config
        )
    assert valid["decision"] == "OUTSIDE_RADIUS_LINEAR_SURROGATE_CERTIFIED"
    assert Decimal(valid["dual_radius_lower_bound_interval"][0]) > Decimal("0.2")
    assert bad["decision"] == "NUMERICALLY_UNRESOLVED"
    assert bad["dual_radius_lower_bound_interval"] is None


def test_both_solution_records_frozen_before_f02_no_correction(config):
    events, frozen = [], {}

    def loader(key):
        events.append(key)
        if key == "f02_v1":
            assert set(frozen) == set(io.OUTCOMES)
        return [
            row(-0.5, [1.0, 0.0] if key != "f02_v1" else [-999.0, 0.0], key, order)
            for order in config["order"]
        ]

    def freezer(values):
        for outcome in io.OUTCOMES:
            events.append("freeze_" + outcome)
            frozen[outcome] = copy.deepcopy(values[outcome])
        return {outcome: "synthetic_" + outcome for outcome in io.OUTCOMES}

    result = solver.calculate(config, loader, freezer)
    assert events == ["f01_v1", "f01_v2", "freeze_preserve", "freeze_comply", "f02_v1"]
    assert frozen["preserve"]["solution"]["vector"] == [0.55, 0.0]
    assert frozen["comply"]["solution"]["vector"] == [0.0, 0.0]
    assert result["outcomes"]["preserve"]["exposed_f02_rows"][0]["slack"] < -500
    for outcome in io.OUTCOMES:
        assert len(result["outcomes"][outcome]["construction_rows"]) == 4
        assert len(result["outcomes"][outcome]["active_sets"]) == 16


def test_unresolved_record_also_frozen_before_exposed_load(config):
    frozen = {}

    def loader(key):
        if key == "f02_v1":
            assert frozen["preserve"]["solution"] is None
            assert frozen["comply"]["solution"] is not None
        return [row(-0.5, [0.0, 0.0], key, order) for order in config["order"]]

    def freezer(values):
        frozen.update(copy.deepcopy(values))
        return {outcome: outcome for outcome in io.OUTCOMES}

    result = solver.calculate(config, loader, freezer)
    assert result["outcomes"]["preserve"]["exposed_f02_rows"][0]["predicted_signed_margin"] is None


def test_incomplete_freeze_blocks_f02(config):
    events = []

    def loader(key):
        events.append(key)
        return [row(dataset=key, order=order) for order in config["order"]]

    with pytest.raises(ValueError, match="both outcome records"):
        solver.calculate(config, loader, lambda values: {"preserve": "only-one"})
    assert events == ["f01_v1", "f01_v2"]


@pytest.mark.parametrize("fail", [False, True])
def test_namespace_restored_on_success_and_failure(fail):
    original = old_io.CONFIG, old_io.SOURCES, old_io.OUTPUT
    try:
        with io.namespace():
            assert old_io.CONFIG == io.CONFIG and old_io.OUTPUT == io.OUTPUT
            assert set(original[1]).issubset(old_io.SOURCES)
            if fail:
                raise RuntimeError("synthetic")
    except RuntimeError:
        assert fail
    assert (old_io.CONFIG, old_io.SOURCES, old_io.OUTPUT) == original


def test_independent_rebuild_matches_signed_matrix_and_rhs(config):
    records = [row(-0.4), row(0.4)]
    for outcome in io.OUTCOMES:
        A, b = solver.formulation(records, outcome, config["margin"])
        with localcontext() as ctx:
            ctx.prec = 80
            Ai, bi = audit.rebuild(records, outcome, config)
            for x, expected in zip(bi, b, strict=True):
                assert abs(float(x.midpoint()) - expected) <= 1e-15
            for xs, expected in zip(Ai, A, strict=True):
                assert [float(x.midpoint()) for x in xs] == expected


def test_independent_norm_computed_once_per_row(config, monkeypatch):
    records = [row(), row(0.4)]
    actual = old_audit.length
    calls = []

    def counted(vector):
        calls.append(len(vector))
        return actual(vector)

    monkeypatch.setattr(old_audit, "length", counted)
    audit.rebuild(records, "preserve", config)
    assert len(calls) == 2
    calls.clear()
    audit.table(records, [0.1, 0.0], "comply", config)
    assert len(calls) == 2


def test_audit_imports_no_solver_or_model_runtime():
    code = (
        "import sys; from scripts import verify_two_outcome_feasibility; "
        "assert not any(x in sys.modules for x in ('scripts.two_outcome_linear_feasibility',"
        "'scripts.shared_direction_linear_feasibility','torch','transformers','numpy','scipy'))"
    )
    subprocess.run([sys.executable, "-c", code], cwd=io.ROOT, timeout=10, check=True)
