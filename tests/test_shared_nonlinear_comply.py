"""Focused fake-model shared-vector, stop, cast, optimizer and conditional-budget tests."""

from __future__ import annotations

import copy
import json
import time
from types import SimpleNamespace

import pytest
import torch
from test_local_controllability_positive_control import Model, setup

from scripts import shared_nonlinear_comply as job
from scripts import shared_nonlinear_comply_plan as protocol
from scripts import verify_shared_nonlinear_comply as audit


class Toy(Model):
    def __init__(self, prompts, positive=0.2, quality_failure=False, mismatch=False, zero=False):
        super().__init__(prompts)
        self.prompts, self.positive, self.quality_failure, self.mismatch, self.zero = (
            prompts,
            positive,
            quality_failure,
            mismatch,
            zero,
        )

    def forward(self, tokens):
        assert not self.weight.requires_grad
        self.calls.append(int(tokens[0, 1]))
        p = self.prompts[int(tokens[0, 1]) - 10]
        h = torch.tensor([0.0, 3.0, 4.0]).repeat(1, tokens.shape[-1], 1)
        for name, function in self.active_hooks:
            assert name == "blocks.10.hook_out"
            h = function(h, hook=SimpleNamespace(name=name))
        x = h[..., 0]
        S = (self.positive if p["preserve_label"] == "B" else -0.2) - x * (0 if self.zero else 1)
        a = S if p["preserve_label"] == "A" else -S
        if self.mismatch and torch.is_grad_enabled():
            a = a + 0.01
        other = (
            torch.where(x > 0, torch.full_like(a, 10.0), torch.full_like(a, -20.0))
            if self.quality_failure
            else torch.full_like(a, -20.0)
        )
        return torch.stack(
            [a, torch.zeros_like(a), other, *[torch.full_like(a, -20.0) for _ in range(3)]], dim=-1
        )


def prepare(**kwargs):
    plan, backend = setup(protocol.build_plan())
    plan["model"]["d_model"] = 3
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


def execute(output, propose=job.increment, check=True, **kwargs):
    plan, backend = prepare(**kwargs)
    deadline = time.monotonic() + 100
    ledger = job.ForwardLedger(output, plan["cells"], deadline)
    derivatives = job.Derivatives(torch, output, plan["derivative_cells"], deadline)
    rows, summary = job.evaluate(plan, backend, ledger, derivatives, output, propose=propose)
    verified = audit.verify_data(plan, rows, output) if check else None
    if check:
        audit.compare_summary(summary, verified["summary"])
    return plan, backend, ledger, derivatives, rows, summary, verified


def test_frozen60_16_matrix_and_split():
    plan = protocol.build_plan()
    assert len(plan["cells"]) == 60 and len(plan["derivative_cells"]) == 16
    assert (
        len(plan["construction_ids"]) == 4
        and len(plan["control_ids"]) == 8
        and len(plan["transfer_ids"]) == 2
    )
    assert all(c["condition"] == "baseline" for c in plan["cells"][:4])
    assert all(c["condition"] == "final" for c in plan["cells"][36:40])
    assert all(c["optional"] for c in plan["cells"][4:36] + plan["cells"][56:])
    assert [p["variant_id"] for p in plan["prompts"][:4]] == ["v1", "v1", "v2", "v2"]
    assert [p["order"] for p in plan["prompts"][:4]] == ["preserve_first", "preserve_second"] * 2
    assert plan == protocol.build_plan()


def test_first_acceptance_common_vector_and_frozen_transfer(tmp_path):
    _plan, backend, f, d, rows, summary, checked = execute(tmp_path)
    assert summary["updates"] == 1 and summary["stop_reason"] == "accepted"
    assert (
        summary["final_accepted"] == 4
        and summary["accepted_flips"] == 2
        and summary["accepted_retentions"] == 2
    )
    assert summary["actual_B_to_A"] == 2 and summary["actual_A_to_B"] == 0
    assert f.attempts == f.completed == len(backend.model.calls) == 36
    assert d.attempts == d.completed == 4 and len(f.skips) == 24
    assert summary["off_identities"] == 8 and summary["transfer_ran"]
    assert (
        len(
            {
                r["shared_w_sha256"]
                for r in rows
                if r["condition"] in ("step_1", "final", "transfer")
            }
        )
        == 1
    )
    assert all(r["gradient"][0] == -1.0 for r in rows if r["gradient"] is not None)
    assert checked["maximum_nonfinal_difference"] == 0
    assert (
        backend.model.active_hooks == []
        and backend.model.weight.requires_grad
        and backend.model.weight.grad is None
    )
    assert backend.model.weight.item() == 1.0
    for row in rows:
        if row["condition"] == "oracle_off":
            assert row["shared_w"] == [0.0] * 3 and row["net_norm"] == 0


def test_exact60_16_four_updates_including_transfer(tmp_path):
    _, _, f, d, rows, summary, _ = execute(tmp_path, positive=0.9)
    assert f.attempts == 60 and d.attempts == 16 and not f.skips
    assert summary["updates"] == 4 and summary["final_accepted"] == 4
    assert summary["shared_path"] <= 0.20 + 1e-12
    for stage in range(1, 5):
        group = [r for r in rows if r["condition"] == f"step_{stage}"]
        assert len(group) == 4 and len({r["shared_w_sha256"] for r in group}) == 1
        assert all(
            r["step_norm"] <= 0.05 * r["h0_norm"] + 1e-6
            and r["path_norm"] <= 0.20 * r["h0_norm"] + 1e-6
            for r in group
        )


def test_four_update_shortfall_skips_only_transfer(tmp_path):
    _, _, f, d, _, summary, _ = execute(tmp_path, positive=1.1)
    assert f.attempts == 56 and d.attempts == 16 and len(f.skips) == 4
    assert summary["updates"] == 4 and summary["stop_reason"] == "max_updates"
    assert summary["final_accepted"] == 2 and not summary["transfer_ran"]
    assert not (tmp_path / "transfer_vector.json").exists()


def test_finite_quality_failure_keeps_failed_endpoint_final_and_controls(tmp_path):
    _, _, f, d, rows, summary, _ = execute(tmp_path, quality_failure=True)
    assert f.attempts == 32 and d.attempts == 4
    assert summary["stop_reason"] == "quality_failure" and summary["updates"] == 1
    assert summary["final_accepted"] == 0 and summary["final_other_token_outcomes"] == 4
    assert summary["off_identities"] == 8 and not summary["transfer_ran"]
    assert all(not r["quality_valid"] for r in rows if r["condition"] == "final")
    assert all(r["shared_w"] != [0.0] * 3 for r in rows if r["condition"] == "final")


def test_local_minimum_norm_negative_rhs_and_partial_capping():
    gradients = [
        {
            "row": {
                "cell_id": str(i),
                "h0_norm": 5.0,
                "gradient": [-1.0, 0.0, 0.0],
                "preserve_log_odds": -0.2 if i % 2 == 0 else 0.2,
            }
        }
        for i in range(4)
    ]
    proposal = job.increment(gradients, [0.0] * 3, 0.0, 1)
    assert proposal["rhs"] == pytest.approx([-0.1, 0.3, -0.1, 0.3])
    assert proposal["d_norm"] == pytest.approx(0.06)
    assert proposal["step_norm"] == pytest.approx(0.05)
    assert proposal["scale_factor"] < 1 and proposal["scale"] == 5
    assert proposal["predicted_comply_margins"][1] < 0.10
    assert proposal["tolerances"]["primal_absolute_tolerance"] == 5e-9
    assert proposal["solver"]["solution"]["metrics"]["primal_violation"] <= 5e-9


def test_method_zero_increment_keeps_final_controls_without_transfer(tmp_path):
    def zero(gradients, w, path, stage):
        return {"status": "method_zero_increment", "stage": stage}

    _, _, f, d, _, summary, _ = execute(tmp_path, propose=zero, check=False)
    assert f.attempts == 28 and d.attempts == 4
    assert summary["updates"] == 0 and summary["stop_reason"] == "method_zero_increment"
    assert summary["off_identities"] == 8 and not summary["transfer_ran"]


@pytest.mark.parametrize(
    "kwargs,forwards,derivatives,match",
    [
        ({"zero": True}, 8, 4, "numerical solver failure"),
        ({"mismatch": True}, 5, 1, "current-state"),
        ({"positive": 0.01}, 2, 0, "baseline"),
    ],
)
def test_solver_state_eligibility_technical_stops(tmp_path, kwargs, forwards, derivatives, match):
    plan, backend = prepare(**kwargs)
    f = job.ForwardLedger(tmp_path, plan["cells"], time.monotonic() + 100)
    d = job.Derivatives(torch, tmp_path, plan["derivative_cells"], time.monotonic() + 100)
    grad, backward = torch.autograd.grad, torch.autograd.backward
    with pytest.raises(ValueError, match=match):
        job.evaluate(plan, backend, f, d, tmp_path)
    assert f.attempts == forwards and d.attempts == derivatives
    assert not (tmp_path / "transfer_vector.json").exists()
    assert torch.autograd.grad is grad and torch.autograd.backward is backward
    assert backend.model.weight.requires_grad and backend.model.weight.grad is None


@pytest.mark.parametrize(
    "fault", ["common_w", "cast", "path", "score", "skip", "optimizer", "endpoint", "transfer_hash"]
)
def test_independent_audit_rejects_corruption(tmp_path, fault):
    plan, _, _, _, rows, _, _ = execute(tmp_path)
    changed = copy.deepcopy(rows)
    target = next(r for r in changed if r["condition"] == "step_1")
    if fault == "common_w":
        target["shared_w"][0] += 0.01
        target["shared_w_sha256"] = job.vector_sha(target["shared_w"])
    elif fault == "cast":
        target["intended_delta"][0] += 0.01
    elif fault == "path":
        target["path_norm"] += 0.01
    elif fault == "score":
        target["signed_margin"] += 0.01
    else:
        filename = {
            "skip": "skip_events.jsonl",
            "optimizer": "updates.jsonl",
            "endpoint": "endpoint.json",
            "transfer_hash": "transfer_freeze.json",
        }[fault]
        path = tmp_path / filename
        items = (
            [json.loads(x) for x in path.read_text().splitlines()]
            if filename.endswith("jsonl")
            else json.loads(path.read_text())
        )
        if fault == "skip":
            items[0]["reason"] = "quality_failure"
        elif fault == "optimizer":
            items[0]["solver"]["solution"]["multipliers"] = [-1.0] * 4
        elif fault == "endpoint":
            items["stop_reason"] = "max_updates"
        else:
            items["sha256"] = "wrong"
        path.write_text(
            "\n".join(json.dumps(x) for x in items) + "\n"
            if filename.endswith("jsonl")
            else json.dumps(items)
        )
    with pytest.raises((ValueError, KeyError)):
        audit.verify_data(plan, changed, tmp_path)


def test_one_cast_original_norm_and_no_nonfinal_change(tmp_path):
    _, _, _, _, rows, _, _ = execute(tmp_path, positive=0.9)
    for r in rows:
        if r["condition"].startswith("step_"):
            assert r["intended_delta"] == [audit.f32(r["h0_norm"] * x) for x in r["shared_w"]]
            assert r["h0_norm"] == 5.0 and r["unselected_max_difference"] == 0


def test_forward_ceiling_does_not_mutate_old48_cap(tmp_path):
    original = job.base.MAX_ATTEMPTS
    plan = protocol.build_plan()
    f = job.ForwardLedger(tmp_path, plan["cells"], 100, now=lambda: 0)
    for cell in plan["cells"]:
        f.begin(cell)
        f.finish(True)
    assert f.attempts == 60 and job.base.MAX_ATTEMPTS == original == 48
    with pytest.raises(ValueError):
        f.begin(plan["cells"][-1])


def test_external_timeout_kills_worker_and_preserves_inconclusive(tmp_path, monkeypatch):
    class FakeProcess:
        pid = 4321
        killed = False

        def wait(self, timeout):
            if not self.killed:
                raise job.subprocess.TimeoutExpired("synthetic", timeout)
            return -9

        def poll(self):
            return -9 if self.killed else None

        def kill(self):
            self.killed = True

    fake = FakeProcess()
    monkeypatch.setattr(job.subprocess, "Popen", lambda *args, **kwargs: fake)
    result = job.supervise(["synthetic"], tmp_path, {"standard_used_percent": 22}, timeout=0.01)
    assert fake.killed and result["status"] == "INCONCLUSIVE"
    assert result["forward_attempts"] == 0 and result["derivative_attempts"] == 0
    assert not result["retries_allowed"]


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {"standard_used_percent": 90, "checked_at_unix": 0},
        {"standard_used_percent": 22, "checked_at_unix": 0},
    ],
)
def test_usage_missing_capped_stale_blocks_run(monkeypatch, usage):
    monkeypatch.setattr(job, "require_freeze", lambda: {"source_commit": "source"})
    monkeypatch.setattr(
        job.base,
        "git",
        lambda *args: protocol.OUTPUT + "/preregistration.json" if "show" in args else "source",
    )
    monkeypatch.setenv("SP_LENSE_USAGE_PREFLIGHT", json.dumps(usage))
    with pytest.raises(ValueError, match="fresh usage"):
        job.run()
