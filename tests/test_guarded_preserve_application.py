"""Focused fixed-application semantics, archive identity, geometry, and raw audit tests."""

from __future__ import annotations

import ast
import copy
import inspect
import textwrap
import time
from types import SimpleNamespace

import pytest
import torch
from test_local_controllability_positive_control import Model, setup

from scripts import crossed_pair_probe as parent
from scripts import guarded_preserve_application as job
from scripts import guarded_preserve_application_plan as protocol
from scripts import verify_crossed_pair_probe as parent_audit
from scripts import verify_guarded_preserve_application as audit


def prepare(mode="linear", quality_failure=False, bad_forecast=False):
    plan, backend = setup(protocol.build_plan())
    plan["model"]["d_model"] = 3
    vectors = {"preserve": [0.17275760436993742, 0.0, 0.0]}
    plan["candidates"]["preserve"].update(
        norm=protocol.norm(vectors["preserve"]),
        vector_float64_le_sha256=protocol.vector_sha(vectors["preserve"]),
    )
    for i, p in enumerate(plan["prompts"]):
        h0 = [0.0, 3.0 + i, 4.0]
        S0 = -0.125 if p["preserve_label"] == "A" else 0.5
        plan["archived_baselines"][p["prompt_id"]].update(
            archived_h0=h0,
            archived_h0_norm=protocol.norm(h0),
            archived_S0=S0,
            archived_argmax_id=1,
            archived_label="B",
            archived_retention=p["preserve_label"] == "B",
            guarded_goal=max(0.10, S0),
            frozen_linear_prediction=100 if bad_forecast else max(0.10, S0) + 0.1,
        )

    class SemanticModel(Model):
        def forward(self, tokens):
            assert not torch.is_grad_enabled() and not self.weight.requires_grad
            self.calls.append(int(tokens[0, 1]))
            p = self.by_token[int(tokens[0, 1])]
            index = plan["prompts"].index(p)
            h = torch.tensor([0.0, 3.0 + index, 4.0]).repeat(1, tokens.shape[-1], 1)
            for name, function in self.active_hooks:
                assert name == "blocks.10.hook_out"
                h = function(h, hook=SimpleNamespace(name=name))
            x = h[..., 0]
            baseline = -0.125 if p["preserve_label"] == "A" else 0.5
            speed = (
                0.03
                if mode == "weak"
                else -0.05
                if mode == "retention_weakening" and p["preserve_label"] == "B"
                else 1.0
            )
            S = baseline + speed * x
            if mode == "goal_fail":
                S = torch.where(
                    x != 0,
                    torch.full_like(x, 0.075 if p["preserve_label"] == "A" else 0.5),
                    torch.full_like(x, baseline),
                )
            a = S if p["preserve_label"] == "A" else -S
            other = (
                torch.where(x != 0, torch.full_like(a, 10), torch.full_like(a, -20))
                if quality_failure
                else torch.full_like(a, -20)
            )
            out = torch.stack(
                [a, torch.zeros_like(a), other, *[torch.full_like(a, -20) for _ in range(3)]],
                dim=-1,
            )
            if mode == "nonfinite":
                out[..., 0] = float("nan")
            if mode == "derivative":
                torch.autograd.grad(None, None)
            if mode == "weight":
                self.weight.add_(0.01)
            if mode == "baseline_mismatch" and len(self.calls) == 1:
                out[..., 0] += 0.01
            if mode == "replay_mismatch" and len(self.calls) == 17:
                out[..., 0] += 0.01
            if hasattr(self, "ledger"):
                assert self.ledger.attempts == len(self.calls) and self.ledger.pending
            return out

    model = SemanticModel(plan["prompts"])
    model.by_token = {model.tokenizer.ids[p["prompt"]]: p for p in plan["prompts"]}
    backend.model = model
    return plan, backend, vectors


def execute(output, **kwargs):
    plan, backend, vectors = prepare(**kwargs)
    ledger = job.base.Ledger(output, plan["cells"], time.monotonic() + 100)
    backend.model.ledger = ledger
    rows = job.evaluate(plan, backend, vectors, ledger, output)
    verified = audit.verify_data(plan, rows, vectors, output)
    audit.compare_summary(job.summarize(rows), verified["summary"])
    return plan, backend, vectors, ledger, rows, verified


def test_exact24_order_own_original_states_and_three_axes(tmp_path):
    plan, backend, vectors, ledger, rows, verified = execute(tmp_path)
    assert ledger.attempts == ledger.completed == len(backend.model.calls) == 24
    assert [r["phase"] for r in rows] == ["baseline"] * 8 + ["edit"] * 8 + ["replay"] * 8
    assert len({tuple(r["h0"]) for r in rows[:8]}) == 8
    assert [r["requested_label"] for r in rows[8:16]] == ["A", "B"] * 4
    for r in rows:
        f = plan["archived_baselines"][r["prompt_id"]]
        assert r["h0"] == f["archived_h0"]
        assert r["archived_baseline_matches"] and r["weights_unchanged"]
        if r["phase"] != "baseline":
            assert r["intended_delta"] == [audit.f32(r["h0_norm"] * x) for x in vectors["preserve"]]
            assert r["actual_norm"] <= r["h0_norm"] * 0.17275760436993742 + 1e-6
        assert r["unselected_max_difference"] == 0
        assert r["delta_from_archived_S0"] == r["preserve_log_odds"] - f["archived_S0"]
    s = verified["summary"]
    assert s["original_outcome_accepted"] == s["guarded_goal_met_count"] == s["replay_matches"] == 8
    assert s["retention_nonweakening_count"] == s["archived_retention_total"] == 4
    assert s["hypothesis_realized"] and s["construction_only"]
    assert s["accepted_flips"] == s["actual_B_to_A"] == 4
    assert s["actual_A_to_B"] == 0 and not s["gate_allowed"]
    assert all(a["h"] == b["h"] for a, b in zip(rows[8:16], rows[16:], strict=True))
    assert backend.model.active_hooks == [] and backend.model.weight.requires_grad
    assert (tmp_path / "derivative_events.jsonl").read_text() == ""
    with pytest.raises(ValueError):
        ledger.begin(plan["cells"][-1])


def test_optional_retention_failure_does_not_redefine_outcome(tmp_path):
    *_, verified = execute(tmp_path, mode="retention_weakening")
    s = verified["summary"]
    assert s["original_outcome_accepted"] == 8
    assert s["retention_nonweakening_count"] == 0 and s["archived_retention_total"] == 4
    assert s["guarded_goal_met_count"] == 4 and not s["hypothesis_realized"]


def test_guarded_aim_failure_but_old_outcome_and_retention_pass(tmp_path):
    *_, verified = execute(tmp_path, mode="goal_fail")
    s = verified["summary"]
    assert s["original_outcome_accepted"] == 8
    assert s["retention_nonweakening_count"] == 4
    assert s["guarded_goal_met_count"] == 4 and not s["hypothesis_realized"]


def test_forecast_error_is_not_gate(tmp_path):
    *_, verified = execute(tmp_path, bad_forecast=True)
    s = verified["summary"]
    assert s["hypothesis_realized"] and not s["forecast_error_is_acceptance_gate"]
    assert all(r["forecast_error"] < -90 for r in s["cells"] if r["phase"] == "edit")


@pytest.mark.parametrize("kwargs", [{"mode": "weak"}, {"quality_failure": True}])
def test_finite_scientific_failures_complete24(tmp_path, kwargs):
    *_, ledger, rows, verified = execute(tmp_path, **kwargs)
    assert ledger.completed == len(rows) == 24
    assert verified["summary"]["original_outcome_accepted"] < 8


@pytest.mark.parametrize(
    "mode,attempts",
    [
        ("nonfinite", 1),
        ("derivative", 1),
        ("weight", 1),
        ("baseline_mismatch", 1),
        ("replay_mismatch", 17),
    ],
)
def test_technical_fault_no_rescue(tmp_path, mode, attempts):
    plan, backend, vectors = prepare(mode)
    ledger = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    with pytest.raises((ValueError, RuntimeError)):
        job.evaluate(plan, backend, vectors, ledger, tmp_path)
    assert ledger.attempts == len(backend.model.calls) == attempts
    assert backend.model.active_hooks == []


@pytest.mark.parametrize(
    "which",
    ["archived_S0", "archived_h0_norm", "archived_argmax_id", "archived_label", "archived_h0"],
)
def test_archive_drift_stops_before_any_application(tmp_path, which):
    plan, backend, vectors = prepare()
    f = plan["archived_baselines"][plan["prompts"][0]["prompt_id"]]
    if which == "archived_h0":
        f[which][0] = 0.01
    elif which == "archived_label":
        f[which] = "A"
    else:
        f[which] += 0.01
    ledger = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    with pytest.raises(ValueError, match="archived baseline"):
        job.evaluate(plan, backend, vectors, ledger, tmp_path)
    assert ledger.attempts == 1


@pytest.mark.parametrize("value,expected", [(1e-6, True), (1.0001e-6, False)])
def test_archived_baseline_absolute_tolerance_no_relative(value, expected):
    row = {
        "phase": "baseline",
        "baseline_margin": 0.0,
        "h0_norm": 1.0,
        "h0": [1.0, 0.0],
        "baseline_argmax_id": 1,
        "baseline_label": "B",
        "preserve_log_odds": 0.0,
        "quality_valid": True,
    }
    frozen = {
        "archived_S0": value,
        "archived_h0_norm": 1.0,
        "archived_h0": [1.0, 0.0],
        "archived_argmax_id": 1,
        "archived_label": "B",
        "archived_retention": False,
        "guarded_goal": 0.1,
        "frozen_linear_prediction": 0.2,
    }
    assert job.application_fields(row, frozen)["archived_baseline_matches"] is expected
    assert audit.application_fields(row, frozen) == job.application_fields(row, frozen)


@pytest.mark.parametrize(
    "which,threshold", [("original", 0.05 - 1e-6), ("goal", 0.10 - 1e-6), ("retention", 0.5 - 1e-6)]
)
def test_three_criterion_boundaries(which, threshold):
    row = {
        "phase": "edit",
        "target_sign": 1,
        "actual_next_token_id": 0,
        "requested_token_id": 0,
        "baseline_argmax_id": 0,
        "baseline_label": "A",
        "baseline_margin": 0.5,
        "h0_norm": 1.0,
        "h0": [1.0],
        "preserve_log_odds": threshold,
        "signed_margin": threshold,
        "answer_pair_mass": 0.8,
        "kl_from_baseline": -1e-6,
        "quality_valid": True,
    }
    f = {
        "archived_S0": 0.5,
        "archived_h0_norm": 1.0,
        "archived_h0": [1.0],
        "archived_argmax_id": 0,
        "archived_label": "A",
        "archived_retention": True,
        "guarded_goal": 0.10,
        "frozen_linear_prediction": 999.0,
    }
    key = {
        "original": "requested_accepted",
        "goal": "guarded_goal_met",
        "retention": "retention_nonweakening",
    }[which]

    def measured():
        return {**job.assess(row), **job.application_fields(row, f)}

    assert measured()[key]
    row["preserve_log_odds"] -= 1e-10
    row["signed_margin"] -= 1e-10
    assert not measured()[key]


@pytest.mark.parametrize(
    "field",
    [
        "guarded_goal",
        "frozen_linear_prediction",
        "archived_S0",
        "forecast_error",
        "delta_from_archived_S0",
        "guarded_goal_met",
        "retention_nonweakening",
        "letter_log_odds",
        "delta_letter_log_odds",
        "candidate_vector_sha256",
        "intended_delta",
        "replay_consistent",
        "maximum_archived_h0_difference",
    ],
)
def test_independent_corruption_rejection(tmp_path, field):
    plan, _, vectors, _, rows, _ = execute(tmp_path)
    changed = copy.deepcopy(rows)
    index = 16 if field == "replay_consistent" else 9 if field == "retention_nonweakening" else 8
    value = changed[index][field]
    if isinstance(value, bool):
        changed[index][field] = not value
    elif isinstance(value, list):
        changed[index][field][0] += 0.01
    elif isinstance(value, str):
        changed[index][field] = "corrupt"
    else:
        changed[index][field] += 0.01
    with pytest.raises(ValueError):
        audit.verify_data(plan, changed, vectors, tmp_path)


def ast_source(function):
    return ast.dump(
        ast.parse(textwrap.dedent(inspect.getsource(function))), include_attributes=False
    )


def test_physical_engine_unchanged_except_archive_check_and_count():
    source = (
        textwrap.dedent(inspect.getsource(parent.evaluate))
        .replace("/20", "/24")
        .replace("== 20", "== 24")
        .replace('"20/0 accounting"', '"24/0 accounting"')
    )
    source = source.replace(
        "                faults = []",
        '                row.update(application_fields(row, plan["archived_baselines"][p["prompt_id"]]))\n'
        "                faults = []\n"
        '                if not row["archived_baseline_matches"]:\n'
        '                    faults.append("fresh baseline differs from frozen archived baseline")',
    )
    assert ast_source(job.evaluate) == ast.dump(ast.parse(source), include_attributes=False)
    assert ast_source(job.make_delta) == ast_source(parent.make_delta)
    assert ast_source(job.assess) == ast_source(parent.assess)


def test_independent_raw_audit_parent_arithmetic_unchanged():
    source = textwrap.dedent(inspect.getsource(parent_audit.verify_data))
    source = source.replace("    verify_renderings(plan)\n", "").replace("20", "24")
    source = source.replace(
        'plan["cells"][:4]] == ["baseline"] * 4', 'plan["cells"][:8]] == ["baseline"] * 8'
    )
    source = source.replace(
        '{"preserve", "comply"}, "two candidate bindings"',
        '{"preserve"}, "one guarded-P proposal binding"',
    )
    source = source.replace(
        "        independent = outcome(current)",
        "        independent = outcome(current)\n"
        '        independent.update(application_fields({**current, **independent}, plan["archived_baselines"][row["prompt_id"]]))\n'
        '        require(independent["archived_baseline_matches"], "independent frozen archived baseline mismatch")',
    )
    assert ast_source(audit.verify_data) == ast.dump(ast.parse(source), include_attributes=False)


def test_report_keeps_three_axes_and_forecast_descriptive(tmp_path):
    *_, verified = execute(tmp_path)
    text = audit.report({"status": "verified", "runtime": {"elapsed_seconds": 1.0}, **verified})
    assert "Original outcome acceptance: 8/8" in text
    assert "non-weakening: 4/4" in text and "Guarded goals: 8/8" in text
    assert "no posthoc forecast-error gate" in text and "SAME eight fitting prompts" in text
    width = None
    for line in text.splitlines():
        if not line.startswith("|"):
            width = None
        else:
            n = len(line.split("|"))
            assert width in (None, n)
            width = n


def test_bad_proposal_is_rejected_before_backend_load(tmp_path, monkeypatch):
    plan, _, vectors = prepare()
    monkeypatch.setattr(job, "OUTPUT", str(tmp_path))
    monkeypatch.setattr(job, "require_freeze", lambda: {"plan": plan})
    job.base.write_new(tmp_path / "RUN_STARTED.json", {"deadline_monotonic": time.monotonic() + 30})
    wrong = list(vectors["preserve"])
    wrong[0] *= 2
    monkeypatch.setattr(protocol, "candidates", lambda: {"preserve": {"vector": wrong}})
    monkeypatch.setattr(job.base, "load_backend", lambda *args: pytest.fail("backend loaded"))
    with pytest.raises(ValueError, match="serialized vector"):
        job.worker()
    assert (tmp_path / "INVALID.json").exists()
    assert not (tmp_path / "runtime.json").exists()


def test_external600_timeout_is_terminal(tmp_path, monkeypatch):
    import subprocess

    calls = []

    class Process:
        pid = 123

        def __init__(self, *args, **kwargs):
            calls.append(1)
            self.dead = False

        def wait(self, timeout):
            if not self.dead:
                raise subprocess.TimeoutExpired("synthetic", timeout)
            return -1

        def poll(self):
            return -1 if self.dead else None

        def kill(self):
            self.dead = True

    monkeypatch.setattr(job.subprocess, "Popen", Process)
    status = job.supervise(["synthetic"], tmp_path, {"standard_used_percent": 27})
    assert len(calls) == 1 and status["status"] == "INCONCLUSIVE"
    assert status["retries_allowed"] is False
    assert status["forward_attempts"] == 0
