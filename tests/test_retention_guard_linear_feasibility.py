"""Focused synthetic math/schedule and authenticated initial-input tests; no model imports."""

from __future__ import annotations

import copy
import inspect
import json
import subprocess
import sys
import time
from decimal import localcontext

import pytest

from scripts import retention_guard_feasibility_io as io
from scripts import retention_guard_linear_feasibility as job
from scripts import shared_preserve_eight_row_solver as solver
from scripts import verify_retention_guard_feasibility as audit
from scripts import verify_shared_direction_feasibility as dec


@pytest.fixture(scope="module")
def config():
    return io.read(io.ROOT / io.CONFIG)


@pytest.fixture(scope="module")
def archived(config):
    blobs, lock, _ = io.authenticate(config)
    prefix = [json.loads(line) for line in blobs["rows.jsonl"].splitlines()[:16]]
    return prefix, lock["plan"]


def toy():
    return [
        {
            "S0": 0.5 if i % 2 == 0 else -0.5,
            "h0": [1.0] + [0.0] * 7,
            "g": [float(j == i) * (1 if i % 2 == 0 else -1) for j in range(8)],
            "prompt_id": f"toy{i}",
        }
        for i in range(8)
    ]


def solved_item(name="originalP"):
    rows = toy()
    item = job.system(rows, name)
    item["solver"] = solver.solve(item["A"], item["b"], item["tolerances"])
    return rows, item


@pytest.mark.parametrize("name", io.ORDER)
def test_rhs_sign_and_scale(name):
    records = toy()
    inputs = job.system(records, name)
    independent, _, _ = audit.rebuild(records, name)
    assert independent == inputs
    t = 1 if name.endswith("P") else -1
    assert inputs["A"] == [[t * x for x in r["g"]] for r in records]
    original = [0.10 - t * r["S0"] for r in records]
    assert inputs["b"] == (
        [max(x, 0) for x in original] if name.startswith("guarded") else original
    )
    if name.startswith("original"):
        assert min(inputs["b"]) < 0
    assert inputs["scale"] == 1 and inputs["tolerances"]["primal_absolute_tolerance"] == 1e-9


def test_four_calls_and_256_masks_only(config):
    seen, events = [], []

    def solve(A, b, policy):
        seen.append((A, b, policy))
        return solver.solve(A, b, policy)

    result = job.calculate(
        toy(), config, lambda e, n: events.append((e, n)), time.monotonic() + 10, solve
    )
    assert len(seen) == result["qp_solves"] == 4
    assert events == [(e, n) for n in io.ORDER for e in ("attempt", "complete")]
    assert all(
        [x["mask"] for x in r["solver"]["active_sets"]] == list(range(256))
        for r in result["objectives"]
    )


def test_unresolved_does_not_rescue(config):
    calls = []

    def solve(A, b, policy):
        calls.append(1)
        return {
            "solution": None,
            "status": "NUMERICALLY_UNRESOLVED",
            "active_sets": [{"mask": i} for i in range(256)],
        }

    result = job.calculate(toy(), config, lambda *x: None, time.monotonic() + 10, solve)
    assert len(calls) == 4 and all(x["solver"]["solution"] is None for x in result["objectives"])


def test_deadline_before_first_solve(config):
    def forbidden(*args):
        pytest.fail("solve after deadline")

    with pytest.raises(ValueError, match="deadline"):
        job.calculate(toy(), config, lambda *x: None, time.monotonic() - 1, forbidden)


def test_native_inputs_original_only(config, archived):
    prefix, plan = archived
    rows = io.select_rows(prefix, plan, config)
    assert len(rows) == 8
    assert [r["prompt_id"] for r in rows] == config["selected_prompt_ids"]
    assert all(r["initial_gradient_cell_id"].endswith("__gradient_1") for r in rows)
    assert all(r["h0_norm"] == io.norm(r["h0"]) for r in rows)
    assert all(r["gradient_semantics"] == "grad_h(z_preserve-z_comply)" for r in rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("condition", "gradient_2"),
        ("stage", 2),
        ("target_sign", -1),
        ("requested", "comply"),
        ("weights_unchanged", False),
        ("current_cell_id", "later"),
        ("preserve_label", "B"),
        ("maximum_current_logit_difference", 1e-7),
        ("h0_norm", 42),
        ("preserve_log_odds", 12),
        ("derivative_attempts", 2),
    ],
)
def test_corrupt_initial_lineage(field, value, config, archived):
    prefix, plan = copy.deepcopy(archived)
    prefix[8][field] = value
    with pytest.raises(ValueError):
        io.select_rows(prefix, plan, config)


@pytest.mark.parametrize("field", ["shared_w", "intended_delta", "actual_delta", "h0"])
def test_nonzero_initial_vector_rejected(field, config, archived):
    prefix, plan = copy.deepcopy(archived)
    prefix[8][field][0] += 0.01
    with pytest.raises(ValueError):
        io.select_rows(prefix, plan, config)


def test_missing_reordered_or_later_rows(config, archived):
    prefix, plan = copy.deepcopy(archived)
    with pytest.raises(ValueError):
        io.select_rows(prefix[:15], plan, config)
    prefix[8], prefix[9] = prefix[9], prefix[8]
    with pytest.raises(ValueError):
        io.select_rows(prefix, plan, config)


def test_corrupt_raw_input_hash(config, tmp_path):
    ns = tmp_path / config["input_namespace"]
    ns.mkdir(parents=True)
    (ns / "rows.jsonl").write_text('{"gradient": [-1]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="input hash"):
        io.authenticate(config, tmp_path)


def test_source_chain_corruption(config):
    altered = copy.deepcopy(config)
    altered["provenance_source_sha256"]["src/sp_lense/comparison_runtime.py"] = "0" * 64
    with pytest.raises(ValueError, match="source chain"):
        io.authenticate(altered)


@pytest.mark.parametrize("name", io.ORDER)
def test_independent_eight_row_kkt_and_certificate(name, config):
    rows, item = solved_item(name)
    with localcontext() as ctx:
        ctx.prec = 80
        verified = audit.verify_objective(item, rows, config)
    assert verified["numeric_status"] == "INDEPENDENT_KKT_POLICY_MATCH"
    assert verified["masks"] == 256
    assert len(verified["multipliers"]) == 8
    assert verified["conservative_status"] == "OUTSIDE_RADIUS_LINEAR_SURROGATE_CERTIFIED"


@pytest.mark.parametrize("kind", ["sign", "rhs", "mask", "multiplier", "metric", "log"])
def test_analysis_corruption_rejected(kind, config):
    rows, item = solved_item()
    if kind == "sign":
        item["A"][0][0] *= -1
    elif kind == "rhs":
        item["b"][0] = 0
    elif kind == "mask":
        item["solver"]["solution"]["active_mask"] ^= 1
    elif kind == "multiplier":
        item["solver"]["solution"]["multipliers"][0] = -0.1
    elif kind == "metric":
        item["solver"]["solution"]["metrics"]["norm"] += 1
    else:
        item["solver"]["active_sets"].reverse()
    with pytest.raises(ValueError):
        audit.verify_objective(item, rows, config)


@pytest.mark.parametrize(
    "case,expected",
    [
        ("within", "WITHIN_RADIUS_LINEAR_SURROGATE_VERIFIED"),
        ("over", "OUTSIDE_RADIUS_LINEAR_SURROGATE_CERTIFIED"),
        ("ambiguous", "NUMERICALLY_UNRESOLVED"),
        ("boundary", "NUMERICALLY_UNRESOLVED"),
    ],
)
def test_analytic_numeric_vs_rigorous(case, expected, config):
    rhs, w, lam = {
        "within": (-1.0, 0.0, 0.0),
        "over": (0.3, 0.3, 0.3),
        "ambiguous": (0.1, 0.1, 0.1),
        "boundary": (0.2, 0.2, 0.2),
    }[case]
    policy = {**config, "primal_absolute_tolerance": 1e-9, "kkt_absolute_tolerance": 1e-8}
    cert = dec.certificate([[dec.Interval(1)]], [dec.Interval(rhs)], [w], [lam], policy)
    assert cert["kkt_verified"] and cert["decision"] == expected
    checked = audit.certificate_policy(cert, config)
    if case == "ambiguous":
        assert w < 0.2 and checked["negative_primal_interval_slack"]
    if case == "over":
        assert checked["dual_radius_bound_permitted"]


def test_corrupt_certificate_rejected(config):
    policy = {**config, "primal_absolute_tolerance": 1e-9, "kkt_absolute_tolerance": 1e-8}
    cert = dec.certificate([[dec.Interval(1)]], [dec.Interval(0.1)], [0.1], [0.1], policy)
    cert["decision"] = "WITHIN_RADIUS_LINEAR_SURROGATE_VERIFIED"
    with pytest.raises(ValueError, match="certificate"):
        audit.certificate_policy(cert, config)


def test_authentication_uses_no_later_numeric_rows(config, monkeypatch):
    original = io.authenticate

    def prefix_only(*args):
        blobs, lock, paths = original(*args)
        blobs["rows.jsonl"] = (
            b"\n".join(blobs["rows.jsonl"].splitlines()[:16]) + b"\nINVALID_LATER_STATE"
        )
        return blobs, lock, paths

    monkeypatch.setattr(io, "authenticate", prefix_only)
    records, _ = io.load_inputs(config)
    assert len(records) == 8


def test_no_model_imports_and_no_solver_in_audit():
    source = inspect.getsource(audit)
    assert "solver.solve(" not in source and "import shared_preserve_eight_row_solver" not in source
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from scripts import retention_guard_linear_feasibility as r; "
                "r.io.forbid_models(); import torch"
            ),
        ],
        cwd=io.ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert result.returncode != 0 and "model/tokenizer import prohibited" in result.stderr


@pytest.mark.parametrize(
    "usage",
    [
        {"standard_used_percent": 90, "checked_at_unix": time.time()},
        {"standard_used_percent": 26, "checked_at_unix": time.time() - 120},
        None,
    ],
)
def test_bad_usage_blocks(usage, monkeypatch):
    monkeypatch.setenv("SP_LENSE_USAGE_PREFLIGHT", json.dumps(usage))
    with pytest.raises(ValueError, match="usage"):
        io.usage()


def test_external180_timeout_no_retry(tmp_path, monkeypatch):
    calls = []

    class Process:
        def __init__(self, *a, **k):
            calls.append(1)
            self.dead = False

        def wait(self, timeout):
            if not self.dead:
                raise subprocess.TimeoutExpired("toy", timeout)
            return -1

        def poll(self):
            return -1 if self.dead else None

        def kill(self):
            self.dead = True

    monkeypatch.setattr(job.subprocess, "Popen", Process)
    status = job.supervise(["synthetic"], tmp_path, {"standard_used_percent": 26})
    assert len(calls) == 1 and status["status"] == "INCONCLUSIVE"
    assert status["maximum_seconds"] == 180 and status["qp_attempts"] == 0
    assert status["retries_allowed"] is False


def test_fixed_config_no_scope_expansion(config):
    for key, value in (
        ("maximum_solves", 5),
        ("radius", 0.21),
        ("predictor_aim", 0.05),
        ("maximum_seconds", 181),
        ("selected_prompt_ids", []),
    ):
        changed = copy.deepcopy(config)
        changed[key] = value
        with pytest.raises(ValueError):
            io.fixed_config(changed)


def test_independent_schedule_corruption():
    expected = [(e, n) for n in io.ORDER for e in ("attempt", "complete")]
    expected.append(("attempt", "independent_audit"))
    events = [{"event": e, "name": n, "monotonic": i + 1} for i, (e, n) in enumerate(expected)]
    started = {
        "started_monotonic": 0,
        "deadline_monotonic": 180,
        "maximum_seconds": 180,
        "qp_ceiling": 4,
        "audit_ceiling": 1,
        "model_calls": 0,
        "derivatives": 0,
    }
    audit.verify_events(events, started)
    for corrupted in (events[:-1], events + [events[-1]], list(reversed(events))):
        with pytest.raises(ValueError):
            audit.verify_events(corrupted, started)


def test_report_keeps_guard_optional_and_old_passes(config):
    results = []
    for name in io.ORDER:
        rows, item = solved_item(name)
        results.append(audit.verify_objective(item, rows, config))
    report = audit.report({"status": "test", "objectives": results, "cost_comparison": []})
    assert "OPTIONAL" in report and "Old passes remain passes" in report
    assert "does not prove it CAUSED" in report and "AB-only" in report
    assert "REPORT+STOP" in report
    tables = [line for line in report.splitlines() if line.startswith("|")]
    widths = None
    for line in report.splitlines():
        if not line.startswith("|"):
            widths = None
        else:
            size = len(line.split("|"))
            assert widths in (None, size)
            widths = size
    assert len(tables) > 32
