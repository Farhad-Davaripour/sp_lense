from __future__ import annotations

import copy
import json
import math
import subprocess
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from test_local_controllability_positive_control import setup

from scripts import margin_aware_local_control as job
from scripts import verify_margin_aware_local_control as audit
from sp_lense.future_choice_scoring import score_float32_logits


def execute(path, **kwargs):
    plan, backend = setup(job.build_plan(), **kwargs)
    ledger = job.base.Ledger(path, plan["cells"], 100, now=lambda: 0)
    derivatives = job.old.DerivativeCounter(path, plan["derivative_cells"], 100, now=lambda: 0)
    rows = job.evaluate(plan, backend, ledger, derivatives, path)
    return plan, backend, ledger, derivatives, rows


@pytest.fixture
def completed(tmp_path):
    return (*execute(tmp_path), tmp_path)


def test_exact_plan_and_discovery_cap_choice():
    plan = job.build_plan()
    assert len(plan["cells"]) == 32 and len(plan["derivative_cells"]) == 4
    assert len(plan["prompts"]) == 12 and len(plan["selected_cases"]) == 6
    assert {p["split"] for p in plan["prompts"]} == {"discovery"}
    assert {p["family_id"] for p in plan["prompts"]} == {"cg_f01_archive_closeout"}
    assert plan["rules"]["maximum_relative_radius"] == 0.20
    assert plan["rules"]["margin"] == 0.05
    assert plan["rules"]["discovery_informed_cap"]
    assert plan["limits"]["absolute_forward_ceiling"] == 40
    assert all(len(c["cell_sha256"]) == 64 for c in plan["cells"])
    assert not any("shared" in c["condition"] for c in plan["cells"])
    assert plan == job.build_plan()


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize(
    "signed_margin,capped,noop", [(-2.0, False, False), (-10.0, True, False), (0.1, False, True)]
)
def test_analytic_minimum_norm_one_step_and_declared_cap(sign, signed_margin, capped, noop):
    result = job.recipe(sign * signed_margin, sign, [3.0, 4.0], [0.0, 5.0])
    deficit = max(0.0, 0.05 - signed_margin)
    assert result["required_relative_radius"] == pytest.approx(deficit / 25)
    assert result["cap_active"] == capped and result["no_op"] == noop
    expected_norm = min(deficit / 5, 1.0)
    assert abs(result["coefficient"]) * 5 == pytest.approx(expected_norm)
    assert result["applied_relative_radius"] == pytest.approx(expected_norm / 5)
    # The proposed displacement is parallel to t*g and attains the Cauchy-Schwarz bound.
    delta = [result["coefficient"] * x for x in (3.0, 4.0)]
    assert sign * sum(x * y for x, y in zip(delta, (3.0, 4.0), strict=True)) == pytest.approx(
        5 * math.hypot(*delta)
    )


def test_32_forwards_four_gradients_and_separate_noop_off(completed):
    plan, backend, ledger, derivatives, rows, output = completed
    assert ledger.attempts == ledger.completed == len(backend.model.calls) == 32
    assert derivatives.attempts == derivatives.completed == 4
    assert sorted(Counter(backend.model.calls).values()) == [2] * 8 + [4] * 4
    assert backend.model.weight.item() == 1 and backend.model.weight.requires_grad
    assert backend.model.weight.grad is None and backend.model.active_hooks == []
    targets = [r for r in rows if r["condition"].startswith("target_")]
    assert sum(r["no_op"] for r in targets) == 4
    for r in targets:
        if r["no_op"]:
            assert r["perturbation"] is None and r["delta_log_odds"] == 0
            assert r["maximum_logit_difference"] == 0
        else:
            assert r["observed_signed_margin"] == pytest.approx(0.05)
    result = job.summarize(plan, rows)
    assert result["classification"] == "PASS"
    assert result["new_requested_flips"] == result["retentions"] == 4
    assert audit.verify_data(plan, rows, output)["summary"] == result
    with pytest.raises(ValueError):
        ledger.begin(plan["cells"][0])
    with pytest.raises(ValueError):
        derivatives.call(plan["derivative_cells"][0], lambda: None)


def test_cap_is_not_technical_error_but_does_not_excuse_miss(tmp_path):
    plan, _, _, _, rows = execute(tmp_path, offset=-1.5)
    summary = job.summarize(plan, rows)
    assert summary["classification"] == "PARTIAL" and summary["capped_targets"] == 4
    assert summary["new_requested_flips"] == 0 and summary["passed_targets"] == 4
    assert audit.verify_data(plan, rows, tmp_path)["summary"] == summary


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("mass", "FAIL"),
        ("no_motion", "FAIL"),
        ("choice", "PARTIAL"),
        ("flip_below_margin", "PARTIAL"),
    ],
)
def test_prospective_target_rules(completed, mode, expected):
    plan, _, _, _, original, _ = completed
    rows = copy.deepcopy(original)
    for r in rows:
        if not r["condition"].startswith("target_"):
            continue
        if mode == "mass":
            r["answer_pair_mass"] = 0.79
        elif mode == "choice":
            r["actual_next_token_id"] = 99
        elif not r["no_op"]:
            if mode == "no_motion":
                r["preserve_log_odds"] = r["baseline_margin"]
                r["delta_log_odds"] = 0
                r["actual_next_token_id"] = r["baseline_argmax_id"]
            else:
                r["preserve_log_odds"] = r["target_sign"] * 0.025
                r["delta_log_odds"] = r["preserve_log_odds"] - r["baseline_margin"]
    assert job.summarize(plan, rows)["classification"] == expected


def test_absolute_bound_has_no_relative_escape_for_large_kl():
    z = torch.tensor([1000.0, -1000.0, 0.0])
    b = torch.tensor([-1000.0, 1000.0, 0.0])
    kwargs = {"choice_a_token_id": 0, "choice_b_token_id": 1, "preserve_label": "A"}
    score = score_float32_logits(torch, z, b, **kwargs)
    assert score["kl_from_baseline"] == 2000
    audit.verify_numeric(score, z.tolist(), b.tolist(), **kwargs)
    score["kl_from_baseline"] += 0.001  # Relative 2e-5 would wrongly allow .04 at KL=2000.
    with pytest.raises(ValueError, match="absolute mismatch"):
        audit.verify_numeric(score, z.tolist(), b.tolist(), **kwargs)


@pytest.mark.parametrize(
    "fault", ["hash", "coefficient", "cap", "missing", "unselected", "weights"]
)
def test_verifier_rejects_evidence_faults(completed, fault):
    plan, _, _, _, original, output = completed
    rows = copy.deepcopy(original)
    if fault == "hash":
        rows[0]["logits_sha256"] = "0" * 64
    elif fault == "coefficient":
        rows[2]["coefficient"] += 1
    elif fault == "cap":
        rows[2]["cap_active"] = True
    elif fault == "missing":
        rows.pop()
    elif fault == "unselected":
        rows[2]["unselected_max_difference"] = 0.1
    else:
        rows[2]["integrity_passed"] = False
    with pytest.raises(ValueError):
        audit.verify_data(plan, rows, output)


@pytest.mark.parametrize("kwargs", [{"zero": True}, {"mismatch": True}])
def test_bad_gradient_or_gradient_forward_stops_without_edit(tmp_path, kwargs):
    plan, backend = setup(job.build_plan(), **kwargs)
    ledger = job.base.Ledger(tmp_path, plan["cells"], 100, now=lambda: 0)
    counter = job.old.DerivativeCounter(tmp_path, plan["derivative_cells"], 100, now=lambda: 0)
    original = torch.autograd.grad
    with pytest.raises(ValueError):
        job.evaluate(plan, backend, ledger, counter, tmp_path)
    assert ledger.attempts == 2 and counter.attempts == 1
    assert torch.autograd.grad is original and backend.model.weight.requires_grad


def test_nonself_routes_before_recipe_and_off_identity_fault(tmp_path, monkeypatch):
    plan, backend = setup(job.build_plan())
    original = backend.model.forward

    def mismatch(tokens):
        output = original(tokens)
        return output + (0.01 if len(backend.model.calls) == 10 else 0)

    monkeypatch.setattr(backend.model, "forward", mismatch)
    ledger = job.base.Ledger(tmp_path, plan["cells"], 100, now=lambda: 0)
    counter = job.old.DerivativeCounter(tmp_path, plan["derivative_cells"], 100, now=lambda: 0)
    with pytest.raises(ValueError, match="identity violation"):
        job.evaluate(plan, backend, ledger, counter, tmp_path)
    assert ledger.attempts == 10 and counter.attempts == 2


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_external_timeout_persists_and_refuses_restart(tmp_path, monkeypatch, cleanup_failure):
    class Process:
        pid = 123
        killed = False

        def wait(self, timeout):
            if self.killed:
                return -9
            raise subprocess.TimeoutExpired(["synthetic"], timeout)

        def poll(self):
            return -9 if self.killed else None

        def kill(self):
            if cleanup_failure:
                raise OSError("synthetic cleanup fault")
            self.killed = True

    monkeypatch.setattr(job.subprocess, "Popen", lambda *a, **k: Process())
    status = job.supervise(["synthetic"], tmp_path, {})
    assert status["status"] == "incomplete_or_invalid" and status["forward_attempts"] == 0
    assert json.loads((tmp_path / "RUN_STATUS.json").read_text()) == status
    with pytest.raises(FileExistsError):
        job.supervise(["synthetic"], tmp_path, {})


def test_malformed_accounting_never_completes(tmp_path, monkeypatch):
    (tmp_path / "forward_events.jsonl").write_text('{"event":', encoding="utf-8")
    process = SimpleNamespace(pid=123, wait=lambda timeout: 0, poll=lambda: 0)
    monkeypatch.setattr(job.subprocess, "Popen", lambda *a, **k: process)
    assert job.supervise(["synthetic"], tmp_path, {})["status"] == "incomplete_or_invalid"


def test_actual_weight_mutation_stops_at_first_forward(tmp_path, monkeypatch):
    plan, backend = setup(job.build_plan())
    model = backend.model
    original = model.forward

    def mutate(tokens):
        result = original(tokens)
        model.weight.add_(0.01)
        return result

    monkeypatch.setattr(model, "forward", mutate)
    ledger = job.base.Ledger(tmp_path, plan["cells"], 100, now=lambda: 0)
    counter = job.old.DerivativeCounter(tmp_path, plan["derivative_cells"], 100, now=lambda: 0)
    with pytest.raises(ValueError, match="weights/parameter gradients changed"):
        job.evaluate(plan, backend, ledger, counter, tmp_path)
    assert ledger.attempts == 1 and counter.attempts == 0


@pytest.mark.parametrize("late", [False, True])
def test_supervisor_requires_exact_32_and_four_completions_on_time(completed, monkeypatch, late):
    plan, _, _, _, rows, output = completed
    clock = [0.0]
    job.base.write_new(output / "analysis.json", job.summarize(plan, rows))

    def wait(timeout):
        clock[0] = 901.0 if late else 1.0
        return 0

    monkeypatch.setattr(job.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        job.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=123, wait=wait, poll=lambda: 0)
    )
    result = job.supervise(["synthetic"], output, {})
    assert result["status"] == ("incomplete_or_invalid" if late else "complete_valid")
    assert result["forward_attempts"] == result["completed_forwards"] == 32
    assert result["derivative_attempts"] == result["completed_derivatives"] == 4
    assert job.base.read_json(output / "RUN_STARTED.json")["absolute_ceiling"] == 40
