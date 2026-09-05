"""Focused fake-model shared-vector, stop, cast, optimizer and conditional-budget tests."""

from __future__ import annotations

import copy
import json
import time
from types import SimpleNamespace

import pytest
import torch
from test_local_controllability_positive_control import Model, setup

from scripts import projected_shared_comply as job
from scripts import projected_shared_comply_plan as protocol
from scripts import verify_projected_shared_comply as audit


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


def test_frozen92_32_matrix_and_split():
    plan = protocol.build_plan()
    assert len(plan["cells"]) == 92 and len(plan["derivative_cells"]) == 32
    assert (
        len(plan["construction_ids"]) == 4
        and len(plan["control_ids"]) == 8
        and len(plan["transfer_ids"]) == 2
    )
    assert all(c["condition"] == "baseline" for c in plan["cells"][:4])
    assert all(c["condition"] == "final" for c in plan["cells"][68:72])
    assert all(c["optional"] for c in plan["cells"][4:68] + plan["cells"][88:])
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
    assert d.attempts == d.completed == 4 and len(f.skips) == 56
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


def test_four_updates_then_skips_remaining_and_runs_transfer(tmp_path):
    _, _, f, d, rows, summary, _ = execute(tmp_path, positive=0.9)
    assert f.attempts == 60 and d.attempts == 16 and len(f.skips) == 32
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


def test_projection_stall_counts_attempt_and_skips_remaining(tmp_path):
    _, _, f, d, _, summary, _ = execute(tmp_path, positive=1.1)
    assert f.attempts == 60 and d.attempts == 20 and len(f.skips) == 32
    assert summary["updates"] == 4 and summary["attempted_updates"] == 5
    assert summary["stop_reason"] == "projection_stall"
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
    assert f.attempts == 92 and job.base.MAX_ATTEMPTS == original == 48
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


@pytest.mark.parametrize(
    "w,d,expected",
    [
        ([0.0, 0.0], [0.0, 0.0], "method_zero_increment"),
        ([0.0, 0.0], [0.03, 0.04], "ready"),
        ([0.20, 0.0], [0.10, 0.0], "projection_stall"),
        ([0.20, 0.0], [0.0, 0.10], "ready"),
        ([0.20, 0.0], [-0.10, 0.0], "ready"),
        ([-0.12, 0.16], [0.06, -0.08], "ready"),
    ],
)
def test_projection_nonexpansive_interior_boundary_and_zero(w, d, expected):
    p = job.project(w, d)
    assert p["status"] == expected
    assert p["step_norm"] <= p["proposed_step_norm"] + 1e-12
    assert p["step_norm"] <= 0.05 + 1e-12 and p["net_norm"] <= 0.20 + 1e-12
    assert p["r"] == [a - b for a, b in zip(p["w_next"], w, strict=True)]
    if p["unprojected_net_norm"] <= 0.20:
        assert p["w_next"] == p["u"] and p["projection_distance"] == 0
    else:
        assert p["net_norm"] == pytest.approx(0.20) and p["projection_factor"] < 1


def test_proposed_actual_predictions_signed_loss_and_independent_kkt():
    rows = [
        {
            "cell_id": str(i),
            "h0": [0.0, 0.0, 1.0],
            "h0_norm": 1.0,
            "gradient": g,
            "preserve_log_odds": S,
        }
        for i, (g, S) in enumerate(
            [
                ([0.0, -1.0, 0.0], 0.0),
                ([1.0, 0.0, 0.0], -0.2),
                ([0.0, -1.0, 0.0], 0.0),
                ([1.0, 0.0, 0.0], -0.2),
            ]
        )
    ]
    p = job.increment([{"row": r} for r in rows], [0.20, 0.0, 0.0], 0.20, 5)
    assert p["projection_factor"] < 1
    assert p["proposed_constraint_changes"][0] > p["actual_constraint_changes"][0]
    assert p["signed_projection_loss"][0] > 0
    assert p["signed_projection_loss"][1] < 0
    assert p["path_after"] == 0.20 + job.norm(p["r"])
    assert p["path_after"] < p["proposed_path_after"] <= 0.40
    checked = audit.verify_update(p, rows, [0.20, 0.0, 0.0], 0.20)
    assert checked["optimizer_kkt_verified"]
    for field in (
        "r",
        "s",
        "u",
        "w_next",
        "signed_projection_loss",
        "proposed_constraint_changes",
        "actual_constraint_residuals",
    ):
        changed = copy.deepcopy(p)
        changed[field][0] += 0.001
        with pytest.raises(ValueError):
            audit.verify_update(changed, rows, [0.20, 0.0, 0.0], 0.20)


@pytest.mark.parametrize("last_accepted", [True, False])
def test_full_eight_round_driver_schedule_common_vector_and_conditional92(tmp_path, last_accepted):
    # Pure scheduling harness, not a model or optimizer-success claim.
    plan = protocol.build_plan()
    plan["model"]["d_model"] = 2
    ledger = job.ForwardLedger(tmp_path, plan["cells"], time.monotonic() + 100)
    seen, proposals, endpoints, freezes = [], [], [], []

    def call(cell, w, current):
        ledger.begin(cell)
        ledger.finish(True)
        seen.append((cell, list(w)))
        accepted = last_accepted and cell["condition"] in ("step_8", "final", "transfer")
        row = {
            **cell,
            "answer_pair_mass": 1.0,
            "kl_from_baseline": 0.0,
            "actual_next_token_id": 1 if accepted else 0,
            "requested_token_id": 1,
            "preserve_log_odds": -0.1 if accepted else 0.1,
        }
        return {"row": row}

    def propose(gradients, w, path, stage):
        projected = job.project(w, [0.02, 0.0])
        return {**projected, "path_after": path + projected["step_norm"]}

    result = job.drive(
        plan,
        call,
        ledger.skip,
        propose,
        proposals.append,
        endpoints.append,
        lambda *x: freezes.append((len(seen), x)),
    )
    assert ledger.attempts == ledger.completed == len(seen) == (92 if last_accepted else 88)
    assert len(ledger.skips) == (0 if last_accepted else 4)
    assert sum(c["condition"].startswith("gradient_") for c, _ in seen) == 32
    assert result["attempted_updates"] == result["updates"] == len(proposals) == 8
    assert result["stop_reason"] == ("accepted" if last_accepted else "max_updates")
    assert result["transfer_ran"] == last_accepted
    if last_accepted:
        assert freezes[0][0] == 88
    else:
        assert not freezes
    for stage in range(1, 9):
        assert len({tuple(w) for c, w in seen if c["condition"] == f"step_{stage}"}) == 1
    assert result["path"] <= 0.40 and result["net"] <= 0.20


def test_actual_path_can_exceed_point20_but_not_point40():
    w, path = [0.0, 0.0], 0.0
    for i in range(8):
        p = job.project(w, [0.05 if i % 2 == 0 else -0.05, 0.0])
        path += p["step_norm"]
        w = p["w_next"]
        assert job.norm(w) <= 0.20 and path <= 0.40 + 1e-12
    assert path == pytest.approx(0.40) and job.norm(w) == 0
    assert path > 0.20


def test_independent_final_disagreement_aborts_before_transfer(tmp_path, monkeypatch):
    plan, backend = prepare()
    original = backend.model.forward

    def corrupt_final(tokens):
        logits = original(tokens)
        # 4 baselines + 4 gradients + 4 accepted scores; first independent final.
        if len(backend.model.calls) == 13:
            logits = logits.clone()
            logits[..., 0] += 0.01
        return logits

    monkeypatch.setattr(backend.model, "forward", corrupt_final)
    f = job.ForwardLedger(tmp_path, plan["cells"], time.monotonic() + 100)
    d = job.Derivatives(torch, tmp_path, plan["derivative_cells"], time.monotonic() + 100)
    with pytest.raises(ValueError, match="current-state/independent identity"):
        job.evaluate(plan, backend, f, d, tmp_path)
    assert f.attempts == 13 and d.attempts == 4
    assert (tmp_path / "endpoint.json").exists()
    assert not (tmp_path / "transfer_vector.json").exists()
    assert not (tmp_path / "result.json").exists()


def test_reused_method_structural_changes_only():
    import ast
    import inspect
    import textwrap

    from scripts import shared_nonlinear_comply as old
    from scripts import verify_shared_nonlinear_comply as old_audit

    def tree(function):
        return ast.parse(textwrap.dedent(inspect.getsource(function)))

    # Only the physical path limit and progress strings change in Session.call.
    original = tree(old.Session.call)

    class Allowed(ast.NodeTransformer):
        def visit_Compare(self, node):
            self.generic_visit(node)
            if (
                isinstance(node.left, ast.Subscript)
                and isinstance(node.left.slice, ast.Constant)
                and node.left.slice.value == "path_norm"
            ):
                for x in ast.walk(node):
                    if isinstance(x, ast.Constant) and x.value == 0.20:
                        x.value = 0.40
            return node

        def visit_Constant(self, node):
            if isinstance(node.value, str):
                node.value = node.value.replace("/<=60", "/<=92").replace("/<=16", "/<=32")
            return node

    assert ast.dump(Allowed().visit(original)) == ast.dump(tree(job.Session.call))

    for before, after, replacements in (
        (
            old.ForwardLedger.begin,
            job.ForwardLedger.begin,
            {60: 92, "forward60/order": "forward92/order"},
        ),
        (
            old.Derivatives.call,
            job.Derivatives.call,
            {16: 32, "derivative16/order/deadline": "derivative32/order/deadline"},
        ),
        (
            old_audit.verify_data,
            audit.verify_data,
            {60: 92, 16: 32, "60/16 ceiling": "92/32 ceiling"},
        ),
    ):

        class Limited(ast.NodeTransformer):
            def __init__(self, replacements):
                self.replacements = replacements

            def visit_Constant(self, node):
                if node.value in self.replacements:
                    node.value = self.replacements[node.value]
                return node

            def visit_Compare(self, node):
                self.generic_visit(node)
                if isinstance(node.left, ast.Name) and node.left.id == "expected_path":
                    for x in ast.walk(node):
                        if isinstance(x, ast.Constant) and x.value == 0.20:
                            x.value = 0.40
                return node

        assert ast.dump(Limited(replacements).visit(tree(before))) == ast.dump(tree(after))
