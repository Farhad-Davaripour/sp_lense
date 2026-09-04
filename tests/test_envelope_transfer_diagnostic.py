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

from scripts import envelope_transfer_diagnostic as diagnostic


class _Tokenizer:
    """Minimal joint-chat tokenizer with a different prefix for every prompt."""

    chat_template = "synthetic-envelope-transfer-test-template"
    eos_token_id = None

    def __init__(self, prompts):
        self.prompt_ids = {prompt["prompt"]: index + 10 for index, prompt in enumerate(prompts)}

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
        prefix = [2, self.prompt_ids[messages[0]["content"]], 3]
        if add_generation_prompt:
            assert len(messages) == 1
            values = prefix
        else:
            values = prefix + {"": [], "A": [0], "B": [1]}[messages[-1]["content"]] + [4, 5]
        return {"input_ids": torch.tensor([values], dtype=torch.long)}

    def decode(self, token_ids, **kwargs):
        return "".join({0: "A", 1: "B"}.get(int(token), "") for token in token_ids)


class _Model:
    def __init__(self, prompts):
        self.tokenizer = _Tokenizer(prompts)
        self.prompt_records = {index + 10: prompt for index, prompt in enumerate(prompts)}
        self.active_hooks = []
        self.calls = []
        self.activations = []
        self.fail_on_call = None
        self.envelope_scales = {"construction": 1.0, "oracle": 2.0}
        self.semantic_sensitivity = False

    @contextmanager
    def hooks(self, *, fwd_hooks):
        previous = self.active_hooks
        self.active_hooks = [*previous, *fwd_hooks]
        try:
            yield
        finally:
            self.active_hooks = previous

    def __call__(self, tokens):
        record = self.prompt_records[int(tokens[0, 1])]
        self.calls.append(record["prompt_id"])
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("synthetic forward failure")
        scale = self.envelope_scales[record["envelope"]]
        activation = torch.tensor([0.0, 3.0, 4.0]).repeat(1, tokens.shape[-1], 1) * scale
        for name, callback in self.active_hooks:
            assert name == "blocks.10.hook_out"
            activation = callback(activation, hook=SimpleNamespace(name=name))
        self.activations.append(activation.clone())
        logits = torch.full((1, tokens.shape[-1], 6), -20.0, dtype=torch.float32)
        sensitivity = (
            -1.0 if self.semantic_sensitivity and record["order"] == "preserve_second" else 1.0
        )
        logits[0, -1, 0] = sensitivity * activation[0, -1, 0] - 0.05
        logits[0, -1, 1] = 0.0
        return logits


def _backend(plan):
    model = _Model(plan["prompts"])
    return SimpleNamespace(
        torch=torch,
        model=model,
        device="cpu",
        dtype_name="float32",
        config=SimpleNamespace(model=SimpleNamespace(prompt_format="chat")),
        encode=lambda prompt: model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )["input_ids"],
    )


def test_plan_freezes_exact_six_case_both_order_envelope_lattice():
    plan = diagnostic.build_plan()
    prompts, cells = plan["prompts"], plan["cells"]
    assert len(prompts) == 24
    assert len(cells) == 48
    assert len({prompt["case_id"] for prompt in prompts}) == 6
    assert len({prompt["variant_id"] for prompt in prompts}) == 2
    assert {prompt["category"] for prompt in prompts} == {
        "self_shutdown",
        "other_shutdown",
        "control",
    }
    assert {prompt["split"] for prompt in prompts} == {"discovery"}
    assert len({prompt["prompt_id"] for prompt in prompts}) == 24
    assert len({cell["cell_id"] for cell in cells}) == 48
    assert Counter(cell["prompt_id"] for cell in cells) == {
        prompt["prompt_id"]: 2 for prompt in prompts
    }
    for prompt in prompts:
        assert prompt["prompt_sha256"] == hashlib.sha256(prompt["prompt"].encode()).hexdigest()
        assert {
            cell["condition"] for cell in cells if cell["prompt_id"] == prompt["prompt_id"]
        } == {"baseline", "always_on"}
        expected = ("A", "B") if prompt["order"] == "preserve_first" else ("B", "A")
        assert (prompt["preserve_label"], prompt["comply_label"]) == expected
    assert diagnostic.build_plan() == plan


def test_ledger_enforces_attempt_cap_and_blocks_retry_after_failure(tmp_path):
    cells = diagnostic.build_plan()["cells"]
    full_dir = tmp_path / "full"
    full_dir.mkdir()
    ledger = diagnostic.Ledger(full_dir, cells, deadline=100.0, now=lambda: 0.0)
    for cell in cells:
        ledger.begin(cell)
        ledger.finish(success=True)
    assert ledger.attempts == ledger.completed == 48
    with pytest.raises(ValueError):
        ledger.begin(cells[0])
    assert ledger.attempts == 48

    failed_dir = tmp_path / "failed"
    failed_dir.mkdir()
    failed = diagnostic.Ledger(failed_dir, cells, deadline=100.0, now=lambda: 0.0)
    failed.begin(cells[0])
    failed.finish(success=False, error="synthetic failure")
    assert failed.attempts == 1
    assert failed.completed == 0
    with pytest.raises(ValueError):
        failed.begin(cells[1])
    assert failed.attempts == 1


def test_ledger_checks_deadline_before_starting_a_forward(tmp_path):
    cells = diagnostic.build_plan()["cells"]
    clock = [0.0]
    ledger = diagnostic.Ledger(tmp_path, cells, deadline=1.0, now=lambda: clock[0])
    clock[0] = 2.0
    with pytest.raises(ValueError, match="deadline"):
        ledger.begin(cells[0])
    assert ledger.attempts == ledger.completed == 0


@pytest.fixture
def synthetic_run(tmp_path):
    plan = diagnostic.build_plan()
    backend = _backend(plan)
    original_model = backend.model
    ledger = diagnostic.Ledger(tmp_path, plan["cells"], deadline=100.0, now=lambda: 0.0)
    emitted = []
    rows = diagnostic.evaluate(plan, backend, torch.tensor([1.0, 0.0, 0.0]), ledger, emitted.append)
    assert backend.model is original_model
    assert rows == emitted
    return plan, backend, ledger, rows, tmp_path


def test_evaluate_exact_48_forwards_reuses_baseline_and_preserves_hook_geometry(synthetic_run):
    plan, backend, ledger, rows, output = synthetic_run
    assert ledger.attempts == ledger.completed == len(backend.model.calls) == 48
    assert backend.model.active_hooks == []
    assert Counter(backend.model.calls) == {prompt["prompt_id"]: 2 for prompt in plan["prompts"]}
    for index, (baseline, changed) in enumerate(zip(rows[::2], rows[1::2], strict=True)):
        scale = 2.0 if baseline["envelope"] == "oracle" else 1.0
        sign = 1.0 if baseline["order"] == "preserve_first" else -1.0
        assert baseline["delta_log_odds"] == 0
        assert baseline["preserve_log_odds"] == pytest.approx(-0.05 * sign)
        assert changed["delta_log_odds"] == pytest.approx(0.1 * scale * sign)
        assert changed["baseline_cell_id"] == baseline["cell_id"]
        assert changed["hidden_norm_before"] == baseline["hidden_norm_after"] == 5.0 * scale
        assert changed["hidden_norm_after"] == pytest.approx(
            ((5 * scale) ** 2 + (0.1 * scale) ** 2) ** 0.5
        )
        assert changed["perturbation_norm"] == pytest.approx(0.1 * scale)
        assert changed["relative_perturbation_norm"] == pytest.approx(0.02)
        assert changed["perturbation"]["n_positions"] == 1
        assert changed["kl_from_baseline"] > 0
        assert 0.99 < changed["answer_pair_mass"] <= 1
        assert changed["forced_pair_decision_changed"] is True
        assert changed["actual_ab_decision_changed"] is True
        assert changed["actual_next_token_changed"] is True
        assert (
            changed["choice_boundary_evidence_sha256"]
            == baseline["choice_boundary_evidence_sha256"]
        )
        original, patched = backend.model.activations[2 * index : 2 * index + 2]
        assert changed["hidden_norm_after"] == float(patched[0, -1].norm())
        assert changed["hidden_norm_after"] > changed["hidden_norm_before"]
        assert torch.equal(original[:, :-1], patched[:, :-1])
        assert torch.equal(original[..., 1:], patched[..., 1:])
        assert float(patched[0, -1, 0] - original[0, -1, 0]) == pytest.approx(0.1 * scale)
    events = diagnostic.read_rows(output / "forward_events.jsonl")
    assert len(events) == 96
    for index, cell in enumerate(plan["cells"]):
        assert events[2 * index]["event"] == "attempt_started"
        assert events[2 * index]["cell"] == cell
        assert events[2 * index + 1]["event"] == "attempt_completed"


def test_analyze_keeps_opposing_order_interactions_and_all_12_strata(synthetic_run):
    plan, _, _, rows, _ = synthetic_run
    analysis = diagnostic.analyze(plan, rows)
    assert len(analysis["contrasts"]) == 12
    for row in analysis["contrasts"]:
        sign = 1 if row["order"] == "preserve_first" else -1
        assert row["construction"]["effect"] == pytest.approx(0.1 * sign)
        assert row["oracle"]["effect"] == pytest.approx(0.2 * sign)
        assert row["interaction"] == pytest.approx(-0.1 * sign)
    assert analysis["interpretation"] == "mixed_or_unchanged_strata"
    assert analysis["forced_pair_flips"] == analysis["actual_ab_flips"] == 24
    assert analysis["learned_gate_allowed"] is False
    assert analysis["original_no_go_reopened"] is False
    assert analysis["material_effect_pass"] is None


@pytest.mark.parametrize(
    "oracle_scale, expected",
    [
        (1.0, "family_specific_directional_improvement"),
        (2.0, "no_resolved_interaction"),
    ],
)
def test_positive_and_null_descriptive_interpretations_never_authorize_gate(
    tmp_path, oracle_scale, expected
):
    plan = diagnostic.build_plan()
    backend = _backend(plan)
    backend.model.envelope_scales = {"construction": 2.0, "oracle": oracle_scale}
    backend.model.semantic_sensitivity = True
    ledger = diagnostic.Ledger(tmp_path, plan["cells"], deadline=100.0, now=lambda: 0.0)
    rows = diagnostic.evaluate(
        plan, backend, torch.tensor([1.0, 0.0, 0.0]), ledger, lambda row: None
    )
    analysis = diagnostic.analyze(plan, rows)
    assert analysis["interpretation"] == expected
    for contrast in analysis["contrasts"]:
        assert contrast["construction"]["effect"] == pytest.approx(0.2)
        assert contrast["interaction"] == pytest.approx(0.2 - 0.1 * oracle_scale)
    assert analysis["learned_gate_allowed"] is False
    assert analysis["original_no_go_reopened"] is False
    assert analysis["material_effect_pass"] is None


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "delta", "flip", "norm", "context"])
def test_analyze_rejects_incomplete_or_inconsistent_saved_rows(synthetic_run, corruption):
    plan, _, _, rows, _ = synthetic_run
    broken = copy.deepcopy(rows)
    if corruption == "missing":
        broken.pop()
    elif corruption == "duplicate":
        broken[-1] = broken[0]
    elif corruption == "delta":
        broken[1]["delta_log_odds"] += 1
    elif corruption == "flip":
        broken[1]["forced_pair_decision_changed"] = False
    elif corruption == "norm":
        broken[1]["relative_perturbation_norm"] = 0.04
    else:
        broken[1]["choice_boundary_evidence_sha256"] = "x" * 64
    with pytest.raises(ValueError):
        diagnostic.analyze(plan, broken)


def test_model_failure_is_journaled_before_return_and_cannot_retry(tmp_path):
    plan = diagnostic.build_plan()
    backend = _backend(plan)
    original = backend.model
    original.fail_on_call = 2
    ledger = diagnostic.Ledger(tmp_path, plan["cells"], deadline=100.0, now=lambda: 0.0)
    rows = []
    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        diagnostic.evaluate(plan, backend, torch.tensor([1.0, 0.0, 0.0]), ledger, rows.append)
    assert backend.model is original
    assert original.active_hooks == []
    assert ledger.attempts == 2 and ledger.completed == 1
    assert len(rows) == 1
    events = diagnostic.read_rows(tmp_path / "forward_events.jsonl")
    assert [event["event"] for event in events] == [
        "attempt_started",
        "attempt_completed",
        "attempt_started",
        "attempt_failed",
    ]
    with pytest.raises(ValueError, match="failed"):
        ledger.begin(plan["cells"][2])
    with pytest.raises(ValueError, match="no retry"):
        diagnostic.Ledger(tmp_path, plan["cells"], deadline=100.0)


def test_external_timeout_kills_worker_and_exclusive_sentinel_refuses_restart(
    tmp_path, monkeypatch
):
    class Process:
        pid = 123
        killed = False

        def wait(self, timeout):
            if not self.killed:
                raise subprocess.TimeoutExpired(["synthetic-worker"], timeout)
            return -9

        def poll(self):
            return -9 if self.killed else None

        def kill(self):
            self.killed = True

    process = Process()
    launched = []

    def launch(command, **kwargs):
        launched.append(command)
        return process

    monkeypatch.setattr(diagnostic.subprocess, "Popen", launch)
    monkeypatch.setattr(diagnostic.time, "monotonic", lambda: 0.0)
    status = diagnostic.supervise(["synthetic-worker"], tmp_path, timeout=900)
    assert process.killed
    assert status["status"] == "incomplete"
    assert status["reason"] == "external_whole_job_timeout"
    assert status["forward_attempts"] == status["completed_forwards"] == 0
    assert status["retries_allowed"] is False
    assert json.loads((tmp_path / "RUN_STATUS.json").read_text()) == status
    with pytest.raises(FileExistsError):
        diagnostic.supervise(["synthetic-worker"], tmp_path, timeout=900)
    assert len(launched) == 1


@pytest.mark.parametrize("journal_tail", ['{"event":', "{}\n", "[]\n"])
def test_supervisor_records_incomplete_status_for_malformed_or_invalid_journal(
    tmp_path, monkeypatch, journal_tail
):
    (tmp_path / "forward_events.jsonl").write_text(journal_tail, encoding="utf-8")
    process = SimpleNamespace(pid=123, wait=lambda timeout: 0, poll=lambda: 0)
    monkeypatch.setattr(diagnostic.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(diagnostic.time, "monotonic", lambda: 0.0)
    status = diagnostic.supervise(["synthetic-worker"], tmp_path)
    assert status["status"] == "incomplete"
    assert status["reason"]
    assert status["retries_allowed"] is False
    assert json.loads((tmp_path / "RUN_STATUS.json").read_text()) == status


@pytest.mark.parametrize("cleanup_failure", ["kill", "wait"])
def test_supervisor_cleanup_failure_still_persists_incomplete_status(
    tmp_path, monkeypatch, cleanup_failure
):
    class Process:
        pid = 123
        killed = False

        def wait(self, timeout):
            if self.killed:
                raise OSError("synthetic cleanup wait failed")
            raise subprocess.TimeoutExpired(["synthetic-worker"], timeout)

        def poll(self):
            return None

        def kill(self):
            if cleanup_failure == "kill":
                raise OSError("synthetic kill failed")
            self.killed = True

    monkeypatch.setattr(diagnostic.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(diagnostic.time, "monotonic", lambda: 0.0)
    status = diagnostic.supervise(["synthetic-worker"], tmp_path)
    assert status["status"] == "incomplete"
    assert status["reason"]
    assert json.loads((tmp_path / "RUN_STATUS.json").read_text()) == status


def test_supervisor_late_nominal_success_cannot_report_complete(tmp_path, monkeypatch):
    plan = diagnostic.build_plan()
    clock = [0.0]
    ledger = diagnostic.Ledger(tmp_path, plan["cells"], deadline=900.0, now=lambda: 0.0)
    for cell in plan["cells"]:
        ledger.begin(cell)
        ledger.finish(True)
    diagnostic.write_new(tmp_path / "analysis.json", {"synthetic": True})

    def completed_late(timeout):
        clock[0] = 901.0
        return 0

    process = SimpleNamespace(pid=123, wait=completed_late, poll=lambda: 0)
    monkeypatch.setattr(diagnostic.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(diagnostic.time, "monotonic", lambda: clock[0])
    status = diagnostic.supervise(["synthetic-worker"], tmp_path, timeout=900)
    assert status["status"] == "incomplete"
    assert status["elapsed_seconds"] > 900
    assert status["forward_attempts"] == status["completed_forwards"] == 48
    assert json.loads((tmp_path / "RUN_STATUS.json").read_text()) == status
