"""Focused fake-model shared-vector, stop, cast, optimizer and conditional-budget tests."""

from __future__ import annotations

import copy
import json
import time
from types import SimpleNamespace

import pytest
import torch
from test_local_controllability_positive_control import Model, setup

from scripts import projected_shared_preserve as job
from scripts import projected_shared_preserve_plan as protocol
from scripts import verify_projected_shared_preserve as audit


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
        S = (-self.positive if p["preserve_label"] == "A" else 0.2) + x * (0 if self.zero else 1)
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


def test_frozen88_32_f01_only_and_no_transfer(tmp_path, monkeypatch):
    original = protocol.Path.read_bytes
    opened = []

    def no_f02(path):
        assert "f02" not in str(path).lower()
        opened.append(str(path))
        return original(path)

    monkeypatch.setattr(protocol.Path, "read_bytes", no_f02)
    plan = protocol.build_plan()
    assert len(plan["cells"]) == 88 and len(plan["derivative_cells"]) == 32
    assert len(plan["prompts"]) == 12 and not plan["transfer_ids"]
    assert all(c["condition"] != "transfer" for c in plan["cells"])
    assert all("f01" in p for p in plan["construction_ids"] + plan["control_ids"])
    assert all(c["condition"] == "final" for c in plan["cells"][68:72])
    assert sum(c["optional"] for c in plan["cells"]) == 64
    assert len(plan["config"]["templates"]) == len(opened) == 2
    assert plan["config"]["maximum_forwards"] == 88
    assert plan["config"]["maximum_derivatives"] == 32
    f = job.ForwardLedger(tmp_path, plan["cells"], 100, now=lambda: 0)
    for c in plan["cells"]:
        f.begin(c)
        f.finish(True)
    assert f.attempts == f.completed == 88 and job.base.MAX_ATTEMPTS == 48
    with pytest.raises(ValueError):
        f.begin(plan["cells"][-1])


def test_first_acceptance_semantic_target_common_vector_and_candidate(tmp_path):
    _plan, backend, f, d, rows, summary, verified = execute(tmp_path)
    assert summary["status"] == "PRESERVE_CONSTRUCTION_ACCEPTED_ONLY"
    assert (
        summary["final_accepted"] == 4 and summary["updates"] == summary["attempted_updates"] == 1
    )
    assert summary["accepted_flips"] == summary["accepted_retentions"] == 2
    assert summary["actual_B_to_A"] == 2 and summary["actual_A_to_B"] == 0
    assert (
        summary["candidate_frozen"]
        and not summary["transfer_ran"]
        and not summary["transfer_cells"]
    )
    assert f.attempts == f.completed == len(backend.model.calls) == 32
    assert d.attempts == d.completed == 4 and len(f.skips) == 56
    assert summary["off_identities"] == 8 and verified["maximum_nonfinal_difference"] == 0
    for row in rows:
        assert row["requested"] == "preserve" and row["target_sign"] == 1
        assert row["signed_margin"] == row["preserve_log_odds"]
        assert row["requested_token_id"] == (0 if row["preserve_label"] == "A" else 1)
        if row["gradient"] is not None:
            assert row["gradient"] == [1.0, 0.0, 0.0]
    assert len({r["shared_w_sha256"] for r in rows if r["condition"] in ("step_1", "final")}) == 1
    assert (tmp_path / "preserve_vector.json").exists() and (
        tmp_path / "candidate_freeze.json"
    ).exists()
    assert not (tmp_path / "transfer_vector.json").exists()
    assert backend.model.active_hooks == [] and backend.model.weight.requires_grad
    assert backend.model.weight.grad is None and backend.model.weight.item() == 1.0


def test_preserve_vs_comply_optimizer_sign_and_negative_rhs():
    from scripts import projected_shared_comply as comply

    gradients = [
        {
            "row": {
                "cell_id": str(i),
                "h0_norm": 5.0,
                "h0": [0.0, 3.0, 4.0],
                "gradient": [1.0, 0.0, 0.0],
                "preserve_log_odds": -0.2 if i % 2 == 0 else 0.2,
            }
        }
        for i in range(4)
    ]
    preserve = job.increment(gradients, [0.0] * 3, 0.0, 1)
    opposite = comply.increment(gradients, [0.0] * 3, 0.0, 1)
    assert preserve["rhs"] == pytest.approx([0.3, -0.1, 0.3, -0.1])
    assert opposite["rhs"] == pytest.approx([-0.1, 0.3, -0.1, 0.3])
    assert preserve["d"][0] > 0 and opposite["d"][0] < 0
    assert preserve["d_norm"] == pytest.approx(0.06)
    assert preserve["step_norm"] == pytest.approx(0.05)
    assert audit.verify_update(preserve, [x["row"] for x in gradients], [0.0] * 3, 0.0)[
        "optimizer_kkt_verified"
    ]
    # Same full-argmax field, opposite signed-margin criterion.
    row = {
        "answer_pair_mass": 1.0,
        "kl_from_baseline": 0.0,
        "actual_next_token_id": 0,
        "requested_token_id": 0,
        "preserve_log_odds": 0.2,
    }
    assert job.accepts(row) and not comply.accepts(row)
    row["preserve_log_odds"] = -0.2
    assert comply.accepts(row) and not job.accepts(row)


def test_projection_is_immutable_parent_primitive():
    from scripts import projected_shared_comply as comply

    assert job.project is comply.project and job.Derivatives is comply.Derivatives
    w, path = [0.0, 0.0], 0.0
    for i in range(8):
        p = job.project(w, [0.05 if i % 2 == 0 else -0.05, 0.0])
        path += p["step_norm"]
        w = p["w_next"]
        assert p["step_norm"] <= 0.05 + 1e-12 and job.norm(w) <= 0.20 + 1e-12
        assert path <= 0.40 + 1e-12
    assert path == pytest.approx(0.40)
    assert job.project([0.20, 0.0], [0.10, 0.0])["status"] == "projection_stall"
    assert job.project([0.0, 0.0], [0.0, 0.0])["status"] == "method_zero_increment"
    curved = job.project([0.20, 0.0], [0.0, 0.1])
    assert curved["step_norm"] <= curved["proposed_step_norm"] and curved["projection_factor"] < 1


def test_four_update_construction_keeps_original_norm_cast_and_physical_limits(tmp_path):
    _, _, f, d, rows, summary, _ = execute(tmp_path, positive=0.9)
    assert summary["updates"] == 4 and summary["final_accepted"] == 4
    assert f.attempts == 56 and d.attempts == 16 and len(f.skips) == 32
    for r in rows:
        if r["condition"].startswith("step_"):
            assert r["intended_delta"] == [audit.f32(r["h0_norm"] * x) for x in r["shared_w"]]
            assert r["h0_norm"] == 5.0 and r["unselected_max_difference"] == 0
            assert r["net_norm"] <= 0.20 * r["h0_norm"] + 1e-6
            assert r["path_norm"] <= 0.40 * r["h0_norm"] + 1e-6
            assert r["step_norm"] <= 0.05 * r["h0_norm"] + 1e-6


def test_projection_stall_counts_attempt_no_candidate(tmp_path):
    _, _, f, d, _, s, _ = execute(tmp_path, positive=1.1)
    assert s["attempted_updates"] == 5 and s["updates"] == 4
    assert s["stop_reason"] == "projection_stall" and s["final_accepted"] == 2
    assert f.attempts == 60 and d.attempts == 20 and len(f.skips) == 28
    assert not s["candidate_frozen"] and not (tmp_path / "preserve_vector.json").exists()


def test_quality_failure_keeps_endpoint_finals_and_off_without_candidate(tmp_path):
    _, _, f, d, rows, s, _ = execute(tmp_path, quality_failure=True)
    assert f.attempts == 32 and d.attempts == 4 and s["updates"] == 1
    assert s["stop_reason"] == "quality_failure" and s["final_accepted"] == 0
    assert s["final_other_token_outcomes"] == 4 and s["off_identities"] == 8
    assert not s["candidate_frozen"] and not (tmp_path / "preserve_vector.json").exists()
    assert all(r["shared_w"] != [0.0] * 3 for r in rows if r["condition"] == "final")


@pytest.mark.parametrize(
    "kwargs,forwards,derivatives,match",
    [
        ({"zero": True}, 8, 4, "numerical solver failure"),
        ({"mismatch": True}, 5, 1, "current-state"),
        ({"positive": 0.01}, 1, 0, "baseline"),
    ],
)
def test_technical_stops(tmp_path, kwargs, forwards, derivatives, match):
    plan, backend = prepare(**kwargs)
    f = job.ForwardLedger(tmp_path, plan["cells"], time.monotonic() + 100)
    d = job.Derivatives(torch, tmp_path, plan["derivative_cells"], time.monotonic() + 100)
    original = torch.autograd.grad
    with pytest.raises(ValueError, match=match):
        job.evaluate(plan, backend, f, d, tmp_path)
    assert f.attempts == forwards and d.attempts == derivatives
    assert torch.autograd.grad is original and backend.model.weight.requires_grad
    assert not (tmp_path / "preserve_vector.json").exists()


def test_independent_final_disagreement_precludes_candidate(tmp_path, monkeypatch):
    plan, backend = prepare()
    original = backend.model.forward

    def corrupt_final(tokens):
        logits = original(tokens)
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
    assert not (tmp_path / "preserve_vector.json").exists()


@pytest.mark.parametrize("accepted", [True, False])
def test_full_eight_round88_32_driver_no_transfer(tmp_path, accepted):
    plan = protocol.build_plan()
    plan["model"]["d_model"] = 2
    f = job.ForwardLedger(tmp_path, plan["cells"], time.monotonic() + 100)
    seen, updates, endpoints, freezes = [], [], [], []

    def call(cell, w, current):
        f.begin(cell)
        f.finish(True)
        seen.append((cell, list(w)))
        success = accepted and cell["condition"] in ("step_8", "final")
        return {
            "row": {
                **cell,
                "answer_pair_mass": 1.0,
                "kl_from_baseline": 0.0,
                "actual_next_token_id": 1 if success else 0,
                "requested_token_id": 1,
                "preserve_log_odds": 0.1 if success else -0.1,
            }
        }

    def propose(gradients, w, path, stage):
        p = job.project(w, [0.02, 0.0])
        return {**p, "path_after": path + p["step_norm"]}

    result = job.drive(
        plan,
        call,
        f.skip,
        propose,
        updates.append,
        endpoints.append,
        lambda *x: freezes.append((len(seen), x)),
    )
    assert f.attempts == f.completed == len(seen) == 88 and not f.skips
    assert sum(c["condition"].startswith("gradient_") for c, _ in seen) == 32
    assert result["attempted_updates"] == result["updates"] == 8
    assert result["stop_reason"] == ("accepted" if accepted else "max_updates")
    assert result["candidate_frozen"] == accepted and not result["transfer_ran"]
    assert not any(c["condition"] == "transfer" for c, _ in seen)
    assert (freezes[0][0] == 88) if accepted else not freezes
    for stage in range(1, 9):
        assert len({tuple(w) for c, w in seen if c["condition"] == f"step_{stage}"}) == 1


@pytest.mark.parametrize(
    "fault",
    [
        "sign",
        "label",
        "common_w",
        "cast",
        "path",
        "optimizer",
        "projection",
        "skip",
        "candidate_hash",
        "candidate_clock",
    ],
)
def test_independent_audit_rejects_corruption(tmp_path, fault):
    plan, _, _, _, rows, _, _ = execute(tmp_path)
    changed = copy.deepcopy(rows)
    target = next(r for r in changed if r["condition"] == "step_1")
    if fault == "sign":
        target["target_sign"] = -1
    elif fault == "label":
        target["requested_token_id"] = 1 - target["requested_token_id"]
    elif fault == "common_w":
        target["shared_w"][0] += 0.01
        target["shared_w_sha256"] = job.vector_sha(target["shared_w"])
    elif fault == "cast":
        target["intended_delta"][0] += 0.01
    elif fault == "path":
        target["path_norm"] += 0.01
    else:
        filename = {
            "optimizer": "updates.jsonl",
            "projection": "updates.jsonl",
            "skip": "skip_events.jsonl",
            "candidate_hash": "candidate_freeze.json",
            "candidate_clock": "preserve_vector.json",
        }[fault]
        path = tmp_path / filename
        items = (
            [json.loads(x) for x in path.read_text().splitlines()]
            if filename.endswith("jsonl")
            else json.loads(path.read_text())
        )
        if fault == "optimizer":
            items[0]["solver"]["solution"]["multipliers"] = [-1.0] * 4
        elif fault == "projection":
            items[0]["actual_preserve_margins"][0] += 0.01
        elif fault == "skip":
            items[0]["reason"] = "quality_failure"
        elif fault == "candidate_hash":
            items["sha256"] = "bad"
        else:
            items["monotonic"] = 0.0
        path.write_text(
            "\n".join(json.dumps(x) for x in items) + "\n"
            if filename.endswith("jsonl")
            else json.dumps(items)
        )
        if fault == "candidate_clock":
            lock_path = tmp_path / "candidate_freeze.json"
            lock = json.loads(lock_path.read_text())
            lock["sha256"] = job.base.sha(path.read_bytes())
            lock_path.write_text(json.dumps(lock))
    with pytest.raises((ValueError, KeyError)):
        audit.verify_data(plan, changed, tmp_path)


def test_source_wrapper_session_only_semantic_changes():
    import ast
    import inspect
    import textwrap

    from scripts import projected_shared_comply as old

    original = textwrap.dedent(inspect.getsource(old.Session.call))
    modified = (
        original.replace('"requested": "comply"', '"requested": "preserve"')
        .replace('boundary.token_id(p["comply_label"])', 'boundary.token_id(p["preserve_label"])')
        .replace(
            '"signed_margin": -score["preserve_log_odds"]',
            '"signed_margin": score["preserve_log_odds"]',
        )
        .replace('"target_sign": -1', '"target_sign": 1')
        .replace(
            'if condition.startswith("step_") or condition == "transfer":',
            'if condition.startswith("step_"):',
        )
        .replace("/<=92", "/<=88")
    )
    actual = textwrap.dedent(inspect.getsource(job.Session.call))
    assert ast.dump(ast.parse(modified)) == ast.dump(ast.parse(actual))


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


def test_zero_method_stop_counts_attempt_keeps_final_controls(tmp_path):
    def zero(gradients, w, path, stage):
        return {"status": "method_zero_increment", "stage": stage}

    _, _, f, d, _, s, _ = execute(tmp_path, propose=zero, check=False)
    assert f.attempts == 28 and d.attempts == 4
    assert s["attempted_updates"] == 1 and s["updates"] == 0
    assert s["stop_reason"] == "method_zero_increment"
    assert s["off_identities"] == 8 and not s["candidate_frozen"]


def test_independent_raw_audit_changes_only_semantic_target_and_no_transfer():
    import ast
    import inspect
    import textwrap

    from scripts import verify_projected_shared_comply as old

    original = textwrap.dedent(inspect.getsource(old.verify_data))
    start = original.index('    if any(r["prompt_id"] in plan["transfer_ids"]')
    end = original.index("    logits_by_id,", start)
    original = (
        original[:start]
        + '    require(not plan["transfer_ids"] and len(prompts) == 12, "construction/controls only")\n'
        + original[end:]
    )
    modified = (
        original.replace("len(cells) == 92", "len(cells) == 88")
        .replace('"92/32 ceiling"', '"88/32 ceiling"')
        .replace(
            'if condition.startswith("step_") or condition == "transfer":',
            'if condition.startswith("step_"):',
        )
        .replace('if row["comply_label"] == "A"', 'if row["preserve_label"] == "A"')
        .replace('row["requested"] == "comply"', 'row["requested"] == "preserve"')
        .replace('row["target_sign"] == -1', 'row["target_sign"] == 1')
        .replace(
            'row["signed_margin"] == -row["preserve_log_odds"]',
            'row["signed_margin"] == row["preserve_log_odds"]',
        )
    )
    assert ast.dump(ast.parse(modified)) == ast.dump(
        ast.parse(textwrap.dedent(inspect.getsource(audit.verify_data)))
    )

    expected = textwrap.dedent(inspect.getsource(old.verify_update))
    expected = (
        expected.replace("_comply_margins", "_preserve_margins")
        .replace("A.append([-hn * x", "A.append([hn * x")
        .replace('b = [0.10 + r["preserve_log_odds"]', 'b = [0.10 - r["preserve_log_odds"]')
        .replace("Ai.append([-(hn * x)", "Ai.append([(hn * x)")
        .replace(
            'decimal_audit.Interval("0.10") + row["preserve_log_odds"]',
            'decimal_audit.Interval("0.10") - row["preserve_log_odds"]',
        )
        .replace('c = [-x["preserve_log_odds"]', 'c = [x["preserve_log_odds"]')
    )
    assert ast.dump(ast.parse(expected)) == ast.dump(
        ast.parse(textwrap.dedent(inspect.getsource(audit.verify_update)))
    )
