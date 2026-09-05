"""Focused synthetic geometry, certification, initial-state and split guards."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from decimal import Decimal, localcontext

import pytest

from scripts import shared_direction_feasibility_io as io
from scripts import shared_direction_linear_feasibility as solver
from scripts import verify_shared_direction_feasibility as audit


@pytest.fixture
def config():
    return io.read(io.ROOT / io.CONFIG)


def assess(A, b, solution, config):
    with localcontext() as ctx:
        ctx.prec = 80
        return audit.certificate(
            [[audit.Interval(x) for x in row] for row in A],
            [audit.Interval(str(x)) for x in b],
            solution["vector"],
            solution["multipliers"],
            config,
        )


def test_orthogonal_minimum_and_valid_dual_radius_certificate(config):
    A, b = [[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0]
    result = solver.solve(A, b, config)
    assert len(result["active_sets"]) == 4
    solution = result["solution"]
    assert solution["vector"] == [1.0, 1.0]
    assert solution["multipliers"] == [1.0, 1.0]
    cert = assess(A, b, solution, config)
    assert cert["kkt_verified"]
    assert cert["decision"] == "OUTSIDE_RADIUS_LINEAR_SURROGATE_CERTIFIED"
    bound = cert["dual_radius_lower_bound_interval"]
    with localcontext() as ctx:
        ctx.prec = 100
        expected = Decimal(2).sqrt()
    assert Decimal(bound[0]) < expected < Decimal(bound[1])


def test_strict_verified_within_radius_rounded_inward_constraints(config):
    A, b = [[1.0, 0.0], [0.0, 1.0]], [0.1, 0.1]
    solution = solver.solve(A, b, config)["solution"]
    cert = assess(A, b, solution, config)
    assert cert["decision"] == "WITHIN_RADIUS_LINEAR_SURROGATE_VERIFIED"
    assert all(Decimal(x[0]) > 0 for x in cert["primal_residual_intervals"])


def test_parallel_singular_full_set_uses_independent_subset(config):
    result = solver.solve([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], [1.0, 2.0, 3.0], config)
    assert result["solution"]["vector"] == [1.0, 0.0]
    assert len(result["solution"]["active"]) == 1
    assert result["active_sets"][-1]["status"] == "rank_deficient_or_near_dependent_skipped"


def test_negative_independent_multiplier_rejects_only_that_set(config):
    result = solver.solve([[1.0, 0.0], [1.0, 1.0]], [1.0, 0.1], config)
    assert result["solution"]["vector"] == [1.0, 0.0]
    assert result["active_sets"][-1]["status"] == "negative_or_nonfinite_multiplier_independent_set"


@pytest.mark.parametrize(
    "A,b",
    [
        ([[1.0, 0.0], [-1.0, 0.0]], [1.0, 1.0]),
        ([[0.0, 0.0]], [1.0]),
        ([[1.0, 0.0], [-1.0, 1e-8]], [1.0, 1.0]),
    ],
)
def test_opposed_zero_and_near_dependent_are_unresolved_not_infeasible(A, b, config):
    result = solver.solve(A, b, config)
    assert result["solution"] is None
    assert result["status"] == "NUMERICALLY_UNRESOLVED"


def test_near_dependent_redundancy_can_still_have_independent_kkt(config):
    A, b = [[1.0, 0.0], [1.0, 1e-8]], [1.0, 1.0]
    result = solver.solve(A, b, config)
    assert result["solution"]["vector"] == [1.0, 0.0]
    assert result["active_sets"][-1]["status"] == "rank_deficient_or_near_dependent_skipped"
    assert assess(A, b, result["solution"], config)["kkt_verified"]


@pytest.mark.parametrize("S", [-2.0, -0.05, 0.0, 0.05, 2.0])
def test_exact_both_sign_reduction(S, config):
    records = [{"h0": [3.0, 4.0], "g": [2.0, -1.0], "S": S}]
    A, b = solver.system(records, config["margin"])
    assert A == [[10.0, -5.0]]
    assert b == [0.05 + abs(S)]
    for w in ([0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]):
        effect = solver.dot(A[0], w)
        both = all(t * S + effect >= config["margin"] for t in (-1, 1))
        assert both == (effect >= b[0])


def test_false_overcap_candidate_is_not_a_certificate(config):
    A, b = [[1.0, 0.0], [0.0, 1.0]], [0.1, 0.1]
    fake = {"vector": [10.0, 10.0], "multipliers": [0.0, 0.0]}
    cert = assess(A, b, fake, config)
    assert cert["metrics"]["norm"] > 0.20
    assert not cert["kkt_verified"]
    assert cert["dual_radius_lower_bound_interval"] is None
    assert cert["decision"] == "NUMERICALLY_UNRESOLVED"


def test_negative_multiplier_cannot_certify_radius(config):
    fake = {"vector": [1.0], "multipliers": [-1.0]}
    cert = assess([[-1.0]], [1.0], fake, config)
    assert not cert["lambda_nonnegative"]
    assert cert["dual_radius_lower_bound_interval"] is None
    assert cert["decision"] == "NUMERICALLY_UNRESOLVED"


def test_nearly_zero_denominator_not_exact_infeasibility(config):
    fake = {"vector": [1.0], "multipliers": [1.0]}
    cert = assess([[1e-30]], [1.0], fake, config)
    assert cert["dual_radius_lower_bound_interval"] is None
    assert not cert["denominator_is_exact_infeasibility_claim"]
    assert cert["decision"] == "NUMERICALLY_UNRESOLVED"


def test_radius_guard_keeps_boundary_unresolved(config):
    solution = solver.solve([[1.0]], [0.2000000000005], config)["solution"]
    assert (
        assess([[1.0]], [0.2000000000005], solution, config)["decision"] == "NUMERICALLY_UNRESOLVED"
    )


def synthetic_records(key, config):
    return [
        {
            "dataset": key,
            "order": order,
            "prompt_id": key + order,
            "baseline_cell_id": key + order + "baseline",
            "initial_gradient_cell_id": key + order + "gradient_1",
            "S": -0.5 if j == 0 else 0.5,
            "h0": [1.0, 0.0],
            "g": [1.0, 0.0] if j == 0 else [0.0, 1.0],
        }
        for j, order in enumerate(config["order"])
    ]


def test_split_freeze_precedes_f02_and_no_exposed_correction(config):
    events, frozen = [], []

    def loader(key):
        events.append(key)
        if key == "f02_v1":
            assert len(frozen) == 1
        rows = synthetic_records(key, config)
        if key == "f02_v1":
            for row in rows:
                row["g"] = [-1000.0, -1000.0]
        return rows

    def freezer(value):
        events.append("freeze")
        frozen.append(copy.deepcopy(value))
        return "synthetic-sha"

    result = solver.calculate(config, loader, freezer)
    assert events == ["f01_v1", "f01_v2", "freeze", "f02_v1"]
    assert frozen[0]["solution"]["vector"] == [0.55, 0.55]
    assert result["solution_sha256"] == "synthetic-sha"
    assert all(row["residual"] < 0 for row in result["exposed_f02_rows"])
    wrong = copy.deepcopy(config)
    wrong["construction"] = ["f01_v1", "f02_v1"]
    with pytest.raises(ValueError, match="fixed construction"):
        solver.calculate(wrong, loader, freezer)


def test_unresolved_solution_freezes_without_loading_f02(config):
    events = []

    def loader(key):
        events.append(key)
        rows = synthetic_records(key, config)
        for row in rows:
            row["g"] = [0.0, 0.0]
        return rows

    def freezer(value):
        assert value["solution"] is None
        events.append("freeze")
        return "unresolved-sha"

    result = solver.calculate(config, loader, freezer)
    assert events == ["f01_v1", "f01_v2", "freeze"]
    assert result["exposed_f02_rows"] == []


def test_directed_interval_encloses_independent_120_digit_value(config):
    with localcontext() as ctx:
        ctx.prec = 120
        exact = (Decimal(2).sqrt() * Decimal(3) + Decimal("0.1")) / Decimal(7)
    got = ((audit.Interval(2).sqrt() * 3 + audit.Interval("0.1")) / 7).padded(config)
    assert got.lo < exact < got.hi
    assert got.hi - got.lo < Decimal("1e-38")
    assert audit.Interval(-2, 3).square().strings() == ["0", "9"]


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {"standard_used_percent": 90, "checked_at_unix": 0},
        {"standard_used_percent": 21, "checked_at_unix": 0},
    ],
)
def test_missing_capped_or_stale_usage_blocks(bad, monkeypatch):
    monkeypatch.setenv("SP_LENSE_USAGE_PREFLIGHT", json.dumps(bad))
    with pytest.raises(ValueError, match="fresh standard usage"):
        io.usage()


def fake_bundle(tmp_path, config, mutation=None):
    cfg = copy.deepcopy(config)
    spec = cfg["inputs"]["f01_v1"]
    folder = tmp_path / spec["namespace"]
    folder.mkdir(parents=True)
    prompts, rows = [], []
    for index, order in enumerate(cfg["order"]):
        pid = "fake_" + order
        prompt = {
            "prompt_id": pid,
            "prompt_sha256": "fake",
            "order": order,
            "category": "self_shutdown",
            "preserve_label": "A" if index == 0 else "B",
            "comply_label": "B" if index == 0 else "A",
        }
        prompts.append(prompt)
        base = {
            **prompt,
            "cell_id": pid + "__baseline",
            "condition": "baseline",
            "family_id": spec["family_id"],
            "variant_id": "v1",
            "split": "discovery",
            "integrity_passed": True,
            "h": [1.0] + [0.0] * 1023,
            "h0": [1.0] + [0.0] * 1023,
            "preserve_log_odds": 0.25,
            "boundary_sha256": "boundary",
            "prompt_length": 12,
            "choice_a_token_id": 32,
            "choice_b_token_id": 33,
        }
        grad = {
            **base,
            "cell_id": pid + "__gradient_1",
            "condition": "gradient_1",
            "step": 1,
            "current_cell_id": base["cell_id"],
            "baseline_cell_id": base["cell_id"],
            "cumulative_offset": [0.0] * 1024,
            "net_norm": 0.0,
            "maximum_current_logit_difference": 0.0,
            "baseline_margin": 0.25,
            "gradient": [2.0] + [0.0] * 1023,
            "target_sign": -1 if index else 1,
        }
        if mutation:
            mutation(grad)
        later = {**grad, "cell_id": pid + "__gradient_2", "gradient": [-999.0] + [0.0] * 1023}
        rows.extend([base, grad, later])
    status = {"status": "complete_valid"}
    data = {
        "rows.jsonl": rows,
        "verification.json": {
            "classification": "PASS",
            "status": status,
            "absolute_tolerance": 2e-5,
            "relative_tolerance": 0,
        },
        "RUN_STATUS.json": status,
        "runtime.json": {
            "model_id": "Qwen/Qwen3.5-0.8B",
            "model_revision": "2fc06364715b967f1860aea9cf38778875588b17",
            "device": "cpu",
            "dtype": "float32",
            "d_model": 1024,
        },
        "preregistration.json": {
            "source_sha256": dict(config["provenance_source_sha256"]),
            "plan": {
                "prompts": prompts,
                "intervention": {
                    "hook": "blocks.10.hook_out",
                    "position": "final encoded prompt token",
                },
            },
        },
    }
    for name, value in data.items():
        raw = (
            "\n".join(json.dumps(r) for r in value)
            if name.endswith(".jsonl")
            else json.dumps(value)
        ).encode()
        (folder / name).write_bytes(raw)
        spec["sha256"][name] = io.sha(raw)
    return cfg


def test_initial_gradient_only_and_semantic_not_target_signed(tmp_path, config):
    cfg = fake_bundle(tmp_path, config)
    rows = io.initial_rows("f01_v1", cfg, tmp_path)
    assert len(rows) == 2
    assert all(r["g"][0] == 2.0 for r in rows)
    assert all(r["initial_gradient_cell_id"].endswith("gradient_1") for r in rows)
    file = tmp_path / cfg["inputs"]["f01_v1"]["namespace"] / "rows.jsonl"
    file.write_bytes(file.read_bytes() + b" ")
    with pytest.raises(ValueError, match="input changed"):
        io.initial_rows("f01_v1", cfg, tmp_path)


def test_historical_capture_source_must_match(tmp_path, config):
    cfg = fake_bundle(tmp_path, config)
    cfg["provenance_source_sha256"]["src/sp_lense/comparison_runtime.py"] = "wrong-source"
    with pytest.raises(ValueError, match="capture/engine source identity"):
        io.initial_rows("f01_v1", cfg, tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("step", 2),
        ("current_cell_id", "later"),
        ("h", [2.0] + [0.0] * 1023),
        ("baseline_margin", 0.4),
    ],
)
def test_wrong_gradient_state_is_rejected(tmp_path, config, field, value):
    cfg = fake_bundle(tmp_path, config, lambda grad: grad.update({field: value}))
    with pytest.raises(ValueError, match="original baseline"):
        io.initial_rows("f01_v1", cfg, tmp_path)


def test_runtime_imports_are_stdlib_only_and_audit_does_not_import_solver():
    code = (
        "import sys; from scripts import verify_shared_direction_feasibility; "
        "assert 'scripts.shared_direction_linear_feasibility' not in sys.modules; "
        "from scripts import shared_direction_linear_feasibility; "
        "assert not any(x in sys.modules for x in ('torch','transformers','numpy','scipy'))"
    )
    subprocess.run([sys.executable, "-c", code], cwd=io.ROOT, timeout=10, check=True)
