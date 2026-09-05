from __future__ import annotations

import copy
import math
import subprocess
from types import SimpleNamespace

import pytest
import torch
from test_local_controllability_positive_control import Model, setup

from scripts import refreshed_gradient_control as job
from scripts import verify_refreshed_gradient_control as audit


class Toy(Model):
    def __init__(self, prompts, kind="linear", offset=-0.05, quality_failure=False, mismatch=False):
        super().__init__(prompts, offset=offset, mismatch=mismatch)
        self.kind, self.quality_failure = kind, quality_failure

    def forward(self, tokens):
        assert not self.weight.requires_grad
        self.calls.append(int(tokens[0, 1]))
        h = torch.tensor([0.0, 3.0, 4.0]).repeat(1, tokens.shape[-1], 1)
        for name, function in self.active_hooks:
            assert name == "blocks.10.hook_out"
            h = function(h, hook=SimpleNamespace(name=name))
        x = h[..., 0]
        a = (
            torch.log1p(x)
            if self.kind == "nonlinear"
            else 0.3 * torch.tanh(x)
            if self.kind == "saturating"
            else x * 0
            if self.kind == "zero"
            else x
        ) + self.offset
        if self.mismatch and torch.is_grad_enabled():
            a = a + 0.01
        other = (
            torch.where(x > 0, torch.full_like(x, 10.0), torch.full_like(x, -20.0))
            if self.quality_failure
            else torch.full_like(x, -20.0)
        )
        return torch.stack(
            [a, torch.zeros_like(a), other, *[torch.full_like(a, -20.0) for _ in range(3)]], dim=-1
        )


def prepare(**kwargs):
    plan, backend = setup(job.build_plan())
    model = Toy(plan["prompts"], **kwargs)
    backend.model = model
    backend.encode = lambda prompt: model.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )["input_ids"]
    return plan, backend


def execute(output, **kwargs):
    plan, backend = prepare(**kwargs)
    forward = job.ForwardLedger(output, plan["cells"], 100, now=lambda: 0)
    derivative = job.DerivativeLedger(output, plan["derivative_cells"], 100, now=lambda: 0)
    rows, requests = job.evaluate(plan, backend, forward, derivative, output)
    checked = audit.verify_data(plan, rows, requests, forward.skips, output)
    assert checked["summary"] == job.summarize(rows, requests)
    return plan, backend, forward, derivative, rows, requests, checked


def test_frozen_v1_conditional_ceiling_and_order():
    plan = job.build_plan()
    assert len(plan["prompts"]) == 6 and len(plan["selected_cases"]) == 3
    assert {p["variant_id"] for p in plan["prompts"]} == {"v1"}
    assert {p["family_id"] for p in plan["prompts"]} == {"cg_f01_archive_closeout"}
    assert len(plan["cells"]) == 30 and len(plan["derivative_cells"]) == 8
    assert [c["condition"] for c in plan["cells"][:2]] == ["baseline", "baseline"]
    assert sum(c["optional"] for c in plan["cells"]) == 12
    assert plan["rules"]["linear_aim_margin"] == 0.10
    assert plan["rules"]["acceptance_margin"] == 0.05
    assert plan == job.build_plan()


def test_early_success_skips_without_padding_and_reference_isolated(tmp_path):
    plan, model, f, d, _rows, requests, checked = execute(tmp_path)
    assert f.attempts == f.completed == len(model.model.calls) == 18
    assert d.attempts == d.completed == 2 and len(f.skips) == 12
    assert all(r["stop_reason"] == "accepted" and r["updates"] == 1 for r in requests)
    assert checked["summary"]["classification"] == "PASS"
    assert checked["summary"]["opposed_flips"] == checked["summary"]["retentions"] == 2
    assert checked["summary"]["nonself_identities"] == 4
    assert all(r["step_relative_norm"] < 0.05 for r in checked["trajectories"])
    assert (
        model.model.active_hooks == []
        and model.model.weight.requires_grad
        and model.model.weight.grad is None
    )
    with pytest.raises(ValueError):
        f.begin(plan["cells"][0])


def test_nonlinear_current_gradients_refresh_and_beat_stale_reference(tmp_path):
    _, backend, f, d, rows, requests, checked = execute(tmp_path, kind="nonlinear", offset=-0.6)
    assert f.attempts == 30 and d.attempts == 8 and not f.skips
    assert checked["summary"]["classification"] == "PASS"
    assert checked["summary"]["reference_passes"] == 0
    assert all(r["updates"] == 4 for r in requests)
    gradients = [r for r in rows if r["condition"].startswith("gradient_")]
    for r in gradients:
        semantic_sign = 1 if r["preserve_label"] == "A" else -1
        assert r["gradient"][0] == pytest.approx(semantic_sign / (1 + r["h"][0]), rel=1e-6)
        assert r["maximum_current_logit_difference"] == 0
    assert abs(gradients[0]["gradient"][0]) == 1
    assert abs(gradients[1]["gradient"][0]) == pytest.approx(0.8)
    assert abs(gradients[3]["gradient"][0]) < 0.6
    for t in checked["trajectories"]:
        assert t["step_relative_norm"] <= 0.05 + 1e-6
        assert t["net_relative_norm"] <= t["path_relative_norm"] + 1e-6 <= 0.20 + 2e-6
    assert backend.model.weight.item() == 1
    with pytest.raises(ValueError):
        d.call(gradients[-1], lambda: None)


@pytest.mark.parametrize("sign", [-1, 1])
def test_step_budget_uses_original_norm_and_triangle_inequality(sign):
    h0, g = [0.0, 3.0, 4.0], [3.0, 4.0, 0.0]
    step = job.step_recipe(sign * -10.0, sign, g, h0)
    assert step["requested_step_norm"] == 0.25
    assert step["step_limited"]
    s = [x * step["coefficient"] for x in g]
    assert math.hypot(*s) == pytest.approx(0.25)
    assert 4 * math.hypot(*s) <= 0.20 * math.hypot(*h0) + 1e-12


def test_four_update_shortfall_is_partial_not_invalid(tmp_path):
    _, _, f, d, _, requests, checked = execute(tmp_path, kind="saturating", offset=-0.6)
    assert f.attempts == 30 and d.attempts == 8
    assert checked["summary"]["classification"] == "PARTIAL"
    assert checked["summary"]["opposed_flips"] == 0
    assert all(r["stop_reason"] == "max_updates" for r in requests)


def test_finite_mass_failure_stops_only_that_request_and_is_scientific_fail(tmp_path):
    _, _, f, d, _, requests, checked = execute(tmp_path, quality_failure=True)
    assert f.attempts == 18 and d.attempts == 2 and len(f.skips) == 12
    assert checked["summary"]["classification"] == "FAIL"
    assert checked["summary"]["nonself_identities"] == 4
    assert all(r["stop_reason"] == "quality_failure" for r in requests)


@pytest.mark.parametrize(
    "kwargs,expected_forwards,expected_derivatives,error",
    [
        ({"offset": -0.01}, 1, 0, job.EligibilityError),
        ({"kind": "zero"}, 4, 1, ValueError),
        ({"mismatch": True}, 4, 1, ValueError),
    ],
)
def test_eligibility_and_gradient_invalid_paths_do_not_expand(
    tmp_path, kwargs, expected_forwards, expected_derivatives, error
):
    plan, backend = prepare(**kwargs)
    f = job.ForwardLedger(tmp_path, plan["cells"], 100, now=lambda: 0)
    d = job.DerivativeLedger(tmp_path, plan["derivative_cells"], 100, now=lambda: 0)
    original = torch.autograd.grad
    with pytest.raises(error):
        job.evaluate(plan, backend, f, d, tmp_path)
    assert f.attempts == expected_forwards and d.attempts == expected_derivatives
    assert torch.autograd.grad is original and backend.model.weight.requires_grad


@pytest.mark.parametrize(
    "fault", ["skip", "gradient_state", "reference", "offset", "path", "stop", "identity"]
)
def test_independent_audit_rejects_branch_state_and_geometry_corruption(tmp_path, fault):
    plan, _, f, _, original, original_requests, _ = execute(tmp_path)
    rows, requests, skips = (
        copy.deepcopy(original),
        copy.deepcopy(original_requests),
        copy.deepcopy(f.skips),
    )
    grad = next(r for r in rows if r["condition"].startswith("gradient_"))
    step = next(r for r in rows if r["condition"].startswith("step_"))
    ref = next(r for r in rows if r["condition"] == "reference")
    if fault == "skip":
        skips[0]["reason"] = "quality_failure"
    elif fault == "gradient_state":
        grad["current_cell_id"] = ref["cell_id"]
    elif fault == "reference":
        ref["reference_recipe"]["coefficient"] += 1
    elif fault == "offset":
        step["previous_offset"][0] = 1
    elif fault == "path":
        step["path_norm"] += 1
    elif fault == "stop":
        requests[0]["stop_reason"] = "max_updates"
    else:
        rows[-1]["integrity_passed"] = False
    with pytest.raises((ValueError, KeyError)):
        audit.verify_data(plan, rows, requests, skips, tmp_path)


def test_mandatory_cells_cannot_skip_and_derivatives_cannot_repeat(tmp_path):
    plan = job.build_plan()
    f = job.ForwardLedger(tmp_path, plan["cells"], 100, now=lambda: 0)
    with pytest.raises(ValueError):
        f.skip(plan["cells"][0], "accepted", "fake")
    d = job.DerivativeLedger(tmp_path, plan["derivative_cells"], 100, now=lambda: 0)
    d.call(plan["derivative_cells"][0], lambda: 1)
    with pytest.raises(ValueError):
        d.call(plan["derivative_cells"][0], lambda: 1)
    assert d.attempts == 1


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_timeout_and_no_restart_persist(tmp_path, monkeypatch, cleanup_failure):
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
    result = job.supervise(["synthetic"], tmp_path, {})
    assert result["status"] == "incomplete_or_invalid" and result["forward_attempts"] == 0
    with pytest.raises(FileExistsError):
        job.supervise(["synthetic"], tmp_path, {})


@pytest.mark.parametrize("late", [False, True])
def test_supervisor_accepts_declared_early_completion_only_on_time(tmp_path, monkeypatch, late):
    _, _, _, _, rows, requests, _ = execute(tmp_path)
    job.base.write_new(tmp_path / "analysis.json", job.summarize(rows, requests))
    clock = [0.0]

    def wait(timeout):
        clock[0] = 901.0 if late else 1.0
        return 0

    monkeypatch.setattr(job.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        job.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=123, wait=wait, poll=lambda: 0)
    )
    result = job.supervise(["synthetic"], tmp_path, {})
    assert result["status"] == ("incomplete_or_invalid" if late else "complete_valid")
    assert (
        result["forward_attempts"] == 18
        and result["derivative_attempts"] == 2
        and result["skipped_forwards"] == 12
    )


def test_absolute_numeric_contract_no_relative_escape():
    with pytest.raises(ValueError, match="absolute mismatch"):
        audit.close(2000.001, 2000.0, "KL", 2e-5)


@pytest.mark.parametrize("call_index", [3, 12])
def test_retention_and_nonself_are_independent_identity_forwards(tmp_path, monkeypatch, call_index):
    plan, backend = prepare()
    model = backend.model
    original = model.forward

    def change(tokens):
        logits = original(tokens)
        return logits + (0.01 if len(model.calls) == call_index else 0.0)

    monkeypatch.setattr(model, "forward", change)
    f = job.ForwardLedger(tmp_path, plan["cells"], 100, now=lambda: 0)
    d = job.DerivativeLedger(tmp_path, plan["derivative_cells"], 100, now=lambda: 0)
    with pytest.raises(ValueError, match="identity mismatch"):
        job.evaluate(plan, backend, f, d, tmp_path)
    assert f.attempts == call_index


def test_weight_change_is_immediately_invalid(tmp_path, monkeypatch):
    plan, backend = prepare()
    model = backend.model
    original = model.forward

    def mutate(tokens):
        logits = original(tokens)
        model.weight.add_(0.01)
        return logits

    monkeypatch.setattr(model, "forward", mutate)
    f = job.ForwardLedger(tmp_path, plan["cells"], 100, now=lambda: 0)
    d = job.DerivativeLedger(tmp_path, plan["derivative_cells"], 100, now=lambda: 0)
    with pytest.raises(ValueError, match="weights/parameter gradients changed"):
        job.evaluate(plan, backend, f, d, tmp_path)
    assert f.attempts == 1 and d.attempts == 0


def test_malformed_journal_cannot_be_complete(tmp_path, monkeypatch):
    (tmp_path / "skip_events.jsonl").write_text('{"cell":', encoding="utf-8")
    monkeypatch.setattr(
        job.subprocess,
        "Popen",
        lambda *a, **k: SimpleNamespace(pid=123, wait=lambda timeout: 0, poll=lambda: 0),
    )
    result = job.supervise(["synthetic"], tmp_path, {})
    assert result["status"] == "incomplete_or_invalid"


def test_deadline_prevents_both_kinds_of_call(tmp_path):
    plan = job.build_plan()
    f = job.ForwardLedger(tmp_path, plan["cells"], 1, now=lambda: 2)
    d = job.DerivativeLedger(tmp_path, plan["derivative_cells"], 1, now=lambda: 2)
    with pytest.raises(ValueError, match="deadline"):
        f.begin(plan["cells"][0])
    with pytest.raises(ValueError, match="deadline"):
        d.call(plan["derivative_cells"][0], lambda: 1)
    assert f.attempts == d.attempts == 0
