from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from collections import Counter
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from scripts import local_controllability_positive_control as job
from scripts import verify_local_controllability as verify


class Tokenizer:
    chat_template = "synthetic-only"
    eos_token_id = None

    def __init__(self, prompts):
        self.ids = {p["prompt"]: i + 10 for i, p in enumerate(prompts)}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
        return_dict,
        return_tensors,
    ):
        assert tokenize and not enable_thinking and return_dict and return_tensors == "pt"
        ids = [2, self.ids[messages[0]["content"]], 3]
        if not add_generation_prompt:
            ids += {"": [], "A": [0], "B": [1]}[messages[-1]["content"]] + [4, 5]
        return {"input_ids": torch.tensor([ids])}

    def decode(self, ids, **kwargs):
        return "".join({0: "A", 1: "B"}.get(int(i), "") for i in ids)


class Model(torch.nn.Module):
    def __init__(self, prompts, *, mismatch=False, zero=False, offset=-0.05):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.tokenizer = Tokenizer(prompts)
        self.active_hooks = []
        self.calls = []
        self.mismatch, self.zero, self.offset = mismatch, zero, offset

    @contextmanager
    def hooks(self, *, fwd_hooks):
        previous = self.active_hooks
        self.active_hooks = [*previous, *fwd_hooks]
        try:
            yield
        finally:
            self.active_hooks = previous

    def forward(self, tokens):
        assert not self.weight.requires_grad
        self.calls.append(int(tokens[0, 1]))
        h = torch.tensor([0.0, 3.0, 4.0]).repeat(1, tokens.shape[-1], 1)
        for name, function in self.active_hooks:
            assert name == "blocks.10.hook_out"
            h = function(h, hook=SimpleNamespace(name=name))
        a = h[..., 0] * (0 if self.zero else 1) + self.offset
        if self.mismatch and torch.is_grad_enabled():
            a = a + 0.01
        return torch.stack(
            [a, torch.zeros_like(a), *[torch.full_like(a, -20.0) for _ in range(4)]], dim=-1
        )


def setup(plan, **kwargs):
    plan = copy.deepcopy(plan)
    plan["scoring"].update(choice_a_token_id=0, choice_b_token_id=1)
    model = Model(plan["prompts"], **kwargs)
    backend = SimpleNamespace(
        torch=torch,
        model=model,
        device="cpu",
        dtype_name="float32",
        config=SimpleNamespace(model=SimpleNamespace(prompt_format="chat")),
    )
    backend.encode = lambda prompt: model.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )["input_ids"]
    return plan, backend


def evaluate(output, **kwargs):
    plan, backend = setup(job.build_plan(), **kwargs)
    ledger = job.base.Ledger(output, plan["cells"], 100, now=lambda: 0)
    derivatives = job.DerivativeCounter(output, plan["derivative_cells"], 100, now=lambda: 0)
    rows = job.evaluate(plan, backend, torch.tensor([1.0, 0.0, 0.0]), ledger, derivatives, output)
    return plan, backend, ledger, derivatives, rows


@pytest.fixture
def completed(tmp_path):
    return (*evaluate(tmp_path), tmp_path)


def test_plan_fixed_discovery_family_and_budget():
    plan = job.build_plan()
    assert len(plan["selected_cases"]) == 6
    assert len(plan["prompts"]) == 12
    assert len(plan["cells"]) == 40
    assert len(plan["derivative_cells"]) == 4
    assert {p["split"] for p in plan["prompts"]} == {"discovery"}
    assert {p["envelope"] for p in plan["prompts"]} == {"oracle"}
    assert all("cg_f01_archive_closeout" in p["case_id"] for p in plan["prompts"])
    for p in plan["prompts"]:
        assert hashlib.sha256(p["prompt"].encode()).hexdigest() == p["prompt_sha256"]
        conditions = [c["condition"] for c in plan["cells"] if c["prompt_id"] == p["prompt_id"]]
        assert tuple(conditions) == (
            job.SELF_CONDITIONS if p["category"] == "self_shutdown" else ("baseline", "oracle_off")
        )
    assert plan["limits"]["spare_attempts_authorize_extra_calls"] is False
    assert plan["rules"]["target_maximum_kl"] is None


def test_exact_counts_gradients_signs_geometry_and_independent_off(completed):
    _plan, backend, ledger, derivatives, rows, output = completed
    assert ledger.attempts == ledger.completed == len(backend.model.calls) == 40
    assert derivatives.attempts == derivatives.completed == 4
    assert backend.model.active_hooks == []
    assert backend.model.weight.item() == 1 and backend.model.weight.requires_grad
    assert backend.model.weight.grad is None
    assert sorted(Counter(backend.model.calls).values()) == [2] * 8 + [6] * 4
    for r in rows:
        assert r["integrity_passed"] and r["unselected_max_difference"] == 0
        if r["condition"] == "gradient":
            expected = [1.0, 0.0, 0.0] if r["preserve_label"] == "A" else [-1.0, 0.0, 0.0]
            assert r["gradient"] == expected
            assert r["maximum_logit_difference"] == 0
        if r["condition"].startswith("local_"):
            assert r["target_sign"] * r["delta_log_odds"] == pytest.approx(0.1)
            assert r["relative_perturbation_norm"] == pytest.approx(0.02)
            assert r["predicted_signed_gain"] == pytest.approx(0.1)
        if r["condition"] == "oracle_off":
            assert r["maximum_logit_difference"] == r["kl_from_baseline"] == 0
    assert len(list((output / "logits").iterdir())) == 40


def test_local_pass_does_not_require_shared_pass_and_verifier_independent(completed):
    plan, _, _, _, rows, output = completed
    summary = job.summarize(plan, rows)
    checked = verify.verify_data(plan, rows, output)
    assert summary["classification"] == checked["classification"] == "PASS"
    assert summary["local"]["new_requested_flips"] == 4
    assert summary["local"]["successful_retentions"] == 4
    assert summary["shared"]["classification"] == "PARTIAL"
    assert checked["shared"]["passed_cells"] == summary["shared"]["passed_cells"] == 4
    assert summary["learned_gate_allowed"] is False


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("no_flip", "PARTIAL"),
        ("no_movement", "FAIL"),
        ("mass", "FAIL"),
        ("choice", "PARTIAL"),
        ("kl", "FAIL"),
    ],
)
def test_negative_outcomes_not_implementation_faults(completed, mode, expected):
    plan, _, _, _, original, _ = completed
    rows = copy.deepcopy(original)
    for row in rows:
        if not row["condition"].startswith("local_"):
            continue
        if mode == "no_flip":
            row["actual_next_token_id"] = row["baseline_argmax_id"]
        elif mode == "no_movement":
            row["delta_log_odds"] = 0
            row["actual_next_token_id"] = row["baseline_argmax_id"]
        elif mode == "mass":
            row["answer_pair_mass"] = 0.79
        elif mode == "choice":
            row["actual_next_token_id"] = 99
        else:
            row["kl_from_baseline"] = -2e-6
    assert job.summarize(plan, rows)["classification"] == expected


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"zero": True}, "invalid gradient"),
        ({"mismatch": True}, "ordinary/gradient logits mismatch"),
    ],
)
def test_invalid_gradient_or_baseline_stops_at_second_forward(tmp_path, kwargs, error):
    plan, backend = setup(job.build_plan(), **kwargs)
    ledger = job.base.Ledger(tmp_path, plan["cells"], 100, now=lambda: 0)
    derivatives = job.DerivativeCounter(tmp_path, plan["derivative_cells"], 100, now=lambda: 0)
    original_grad = torch.autograd.grad
    with pytest.raises(ValueError, match=error):
        job.evaluate(plan, backend, torch.tensor([1.0, 0.0, 0.0]), ledger, derivatives, tmp_path)
    assert ledger.attempts == 2 and derivatives.attempts == 1
    assert torch.autograd.grad is original_grad
    assert backend.model.weight.requires_grad
    assert not job.base.read_rows(tmp_path / "rows.jsonl")[-1]["integrity_passed"]


def test_derivative_cap_deadline_failure_and_retry(tmp_path):
    cells = job.build_plan()["derivative_cells"]
    count = job.DerivativeCounter(tmp_path, cells, 100, now=lambda: 0)
    for cell in cells:
        assert count.call(cell, lambda: 3) == 3
    with pytest.raises(ValueError, match="budget"):
        count.call(cells[0], lambda: None)
    with pytest.raises(ValueError, match="no retry"):
        job.DerivativeCounter(tmp_path, cells, 100)
    bad = tmp_path / "failed"
    bad.mkdir()
    count = job.DerivativeCounter(bad, cells, 100, now=lambda: 0)
    with pytest.raises(ZeroDivisionError):
        count.call(cells[0], lambda: 1 / 0)
    assert count.attempts == 1 and count.completed == 0
    with pytest.raises(ValueError):
        count.call(cells[1], lambda: None)
    late = tmp_path / "late"
    late.mkdir()
    count = job.DerivativeCounter(late, cells, 1, now=lambda: 2)
    with pytest.raises(ValueError, match="deadline"):
        count.call(cells[0], lambda: None)
    assert count.attempts == 0


@pytest.mark.parametrize("corruption", ["hash", "margin", "sign", "gradient", "missing"])
def test_independent_verifier_rejects_corrupt_evidence(completed, corruption):
    plan, _, _, _, original, output = completed
    rows = copy.deepcopy(original)
    if corruption == "hash":
        rows[0]["logits_sha256"] = "0" * 64
    elif corruption == "margin":
        rows[2]["preserve_log_odds"] += 1
    elif corruption == "sign":
        rows[2]["target_sign"] *= -1
    elif corruption == "gradient":
        rows[1]["gradient"][0] *= -1
    else:
        rows.pop()
    with pytest.raises(ValueError):
        verify.verify_data(plan, rows, output)


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_timeout_persists_inconclusive_and_refuses_second_launch(
    tmp_path, monkeypatch, cleanup_failure
):
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
                raise OSError("synthetic cleanup failure")
            self.killed = True

    process = Process()
    monkeypatch.setattr(job.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(job.time, "monotonic", lambda: 0)
    status = job.supervise(["synthetic"], tmp_path, {"standard_used_percent": 17})
    assert status["status"] == "incomplete_or_invalid"
    assert status["forward_attempts"] == status["derivative_attempts"] == 0
    assert status["retries_allowed"] is False
    assert json.loads((tmp_path / "RUN_STATUS.json").read_text()) == status
    with pytest.raises(FileExistsError):
        job.supervise(["synthetic"], tmp_path, {})


def test_forward_budget_does_not_use_eight_unused_slots(completed):
    plan, _, ledger, _, _, _ = completed
    with pytest.raises(ValueError):
        ledger.begin(plan["cells"][0])
    assert ledger.attempts == 40


def test_nonfinite_derivative_preserved_without_active_edit(tmp_path, monkeypatch):
    plan, backend = setup(job.build_plan())
    ledger = job.base.Ledger(tmp_path, plan["cells"], 100, now=lambda: 0)
    counter = job.DerivativeCounter(tmp_path, plan["derivative_cells"], 100, now=lambda: 0)
    monkeypatch.setattr(
        torch.autograd,
        "grad",
        lambda objective, activation, **kw: (torch.full_like(activation, float("nan")),),
    )
    with pytest.raises(ValueError, match="nonfinite gradient"):
        job.evaluate(plan, backend, torch.tensor([1.0, 0.0, 0.0]), ledger, counter, tmp_path)
    assert ledger.attempts == 2 and counter.attempts == 1
    assert len(list((tmp_path / "logits").iterdir())) == 2


def test_off_identity_fault_stops_at_independent_replay(tmp_path, monkeypatch):
    plan, backend = setup(job.build_plan())
    original = backend.model.forward

    def change_off(tokens):
        logits = original(tokens)
        return logits + (0.01 if len(backend.model.calls) == 14 else 0)

    monkeypatch.setattr(backend.model, "forward", change_off)
    ledger = job.base.Ledger(tmp_path, plan["cells"], 100, now=lambda: 0)
    counter = job.DerivativeCounter(tmp_path, plan["derivative_cells"], 100, now=lambda: 0)
    with pytest.raises(ValueError, match="oracle-off identity violation"):
        job.evaluate(plan, backend, torch.tensor([1.0, 0.0, 0.0]), ledger, counter, tmp_path)
    assert ledger.attempts == 14 and counter.attempts == 2


def test_scientific_summary_refuses_integrity_fault(completed):
    plan, _, _, _, original, _ = completed
    rows = copy.deepcopy(original)
    rows[-1]["integrity_passed"] = False
    with pytest.raises(ValueError, match="integrity invalid"):
        job.summarize(plan, rows)


def test_journal_malformed_marks_incomplete(tmp_path, monkeypatch):
    (tmp_path / "forward_events.jsonl").write_text('{"event":', encoding="utf-8")
    process = SimpleNamespace(pid=123, wait=lambda timeout: 0, poll=lambda: 0)
    monkeypatch.setattr(job.subprocess, "Popen", lambda *args, **kwargs: process)
    status = job.supervise(["synthetic"], tmp_path, {})
    assert status["status"] == "incomplete_or_invalid"
    job.base.write_new(tmp_path / "preregistration.json", {})
    assert verify.verify(tmp_path)["classification"] == "INCONCLUSIVE"
