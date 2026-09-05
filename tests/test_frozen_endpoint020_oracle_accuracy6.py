"""Focused ordinary truth/accuracy continuation, exact bypass and24/0 contracts."""

from __future__ import annotations

import copy
import inspect
import json
import time
import zlib
from types import SimpleNamespace

import pytest
import torch
from test_local_controllability_positive_control import Model, setup

from scripts import frozen_endpoint020_oracle_accuracy6 as entry
from scripts import frozen_endpoint020_oracle_accuracy6_plan as protocol
from scripts import frozen_guarded_preserve_crossed as old_job
from scripts import verify_frozen_endpoint020_oracle_accuracy6 as verify_entry

job, audit = entry.job, verify_entry.audit


def prepare(mode="perfect"):
    plan, backend = setup(protocol.build_plan())
    plan["model"]["d_model"] = 3
    vectors = {"preserve": [0.2, 0.0, 0.0]}
    plan["candidates"]["preserve"].update(
        norm=0.2, vector_float64_le_sha256=protocol.vector_sha(vectors["preserve"])
    )

    class AccuracyModel(Model):
        def forward(self, tokens):
            assert not torch.is_grad_enabled() and not self.weight.requires_grad
            self.calls.append(int(tokens[0, 1]))
            p = self.by_token[int(tokens[0, 1])]
            i = p["rendering_index"] - 1
            h = torch.tensor([0.0, 3.0 + i, 4.0]).repeat(1, tokens.shape[-1], 1)
            for name, fn in self.active_hooks:
                assert name == "blocks.10.hook_out"
                h = fn(h, hook=SimpleNamespace(name=name))
            x = h[..., 0]
            ordinary = p["kind"] == "ordinary_accuracy"
            s0 = 0.3 if ordinary else -0.2 if p["preserve_label"] == "A" else 0.3
            if ordinary and mode == "wrong_all":
                s0 = -0.3
            if ordinary and mode == "small_margin":
                s0 = 0.01
            if ordinary and mode == "all_B":
                s0 = -0.3 if p["correct_label"] == "A" else 0.3
            speed = 0.01 if mode == "weak_self" and not ordinary else 1.0
            if mode == "aux_only" and not ordinary and p["preserve_label"] == "B":
                speed = -0.01
            S = s0 + speed * x
            a = S if p["preserve_label"] == "A" else -S
            other = torch.full_like(a, -20.0)
            if ordinary and mode == "OTHER":
                other = torch.full_like(a, 4.0)
            elif ordinary and mode == "low_mass":
                other = torch.full_like(a, -0.5)
            out = torch.stack([a, torch.zeros_like(a), other, other, other, other], dim=-1)
            if mode == "nonfinite" and len(self.calls) == 3:
                out[..., 2] = float("nan")
            if mode == "off_tiny" and len(self.calls) == 11:
                out[..., 0] += 2e-7
            if mode == "replay_mismatch" and len(self.calls) == 17:
                out[..., 0] += 0.01
            if mode == "self_ineligible" and len(self.calls) == 1:
                out[..., 0] = 0.001
            if mode == "derivative":
                torch.autograd.grad(None, None)
            if mode == "weight_change":
                self.weight.add_(0.01)
            return out

    model = AccuracyModel(plan["prompts"])
    model.by_token = {model.tokenizer.ids[p["prompt"]]: p for p in plan["prompts"]}
    backend.model = model
    return plan, backend, vectors


def execute(output, mode="perfect"):
    plan, backend, vectors = prepare(mode)
    ledger = job.base.Ledger(output, plan["cells"], time.monotonic() + 100)
    rows = job.evaluate(plan, backend, vectors, ledger, output)
    checked = audit.verify_data(plan, rows, vectors, output)
    audit.compare_summary(job.summarize(rows), checked["summary"])
    return plan, backend, vectors, ledger, rows, checked


def test_fixed8_24_0_source_renderer_truth_and_no_other_models():
    p = protocol.build_plan()
    assert [x["prompt_id"] for x in p["prompts"]] == protocol.IDS
    assert [x["phase"] for x in p["cells"]] == ["baseline"] * 8 + ["edit"] * 8 + ["replay"] * 8
    assert [x["target_sign"] for x in p["cells"]] == [0] * 8 + [1] * 2 + [0] * 6 + [1] * 2 + [0] * 6
    assert not p["derivative_cells"] and all(x["mode"] != "forced_on" for x in p["cells"])
    assert [x["correct_label"] for x in p["prompts"][2:]] == list("ABABAB")
    assert [x["preferred_first"] for x in p["prompts"][2:]] == list(protocol.FIRST)
    assert p["model"]["id"] == "Qwen/Qwen3.5-0.8B"
    assert len(p["ordinary_truths"]) == 6
    assert p["ordinary_truths_sha256"] == protocol.canonical_sha(p["ordinary_truths"])
    verify_entry.verify_selection(p)
    verify_entry.independent_condition(p)
    assert (
        protocol.storage_preflight(protocol.ROOT, p["config"])["bounds"]["total_bound_bytes"]
        == 65789320
    )


@pytest.mark.parametrize("i,answer", list(enumerate(["13", "7", "PINE", "[6]", "Zed", "Yes"])))
def test_answers_independent_of_preferred_or_correct_metadata(i, answer):
    cfg = protocol.read(protocol.ROOT / protocol.CONFIG)
    spec = copy.deepcopy(cfg["truth_specification"][i])
    spec["correct_answer"] = "WRONG"
    assert protocol.derive_answer(spec) == answer
    broken = copy.deepcopy(cfg["truth_specification"])
    broken[i] = spec
    with pytest.raises(ValueError, match="truth"):
        verify_entry.independent_answers(broken)


@pytest.mark.parametrize("fault", ["answer", "letter", "proof", "stem", "option", "order"])
def test_independent_ground_truth_corruption_rejected(fault):
    p = protocol.build_plan()
    spec = p["config"]["truth_specification"][0]
    if fault == "answer":
        spec["correct_answer"] = "12"
    elif fault == "letter":
        spec["correct_label"] = "B"
    elif fault == "proof":
        spec["proof"]["operands"] = [6, 6]
    elif fault == "stem":
        spec["stem"] = "What is 6 plus 6?"
    elif fault == "option":
        spec["options"].reverse()
    else:
        p["config"]["truth_specification"].reverse()
    with pytest.raises(ValueError):
        verify_entry.verify_selection(p)


def test_swapped_actual_mapping_even_with_updated_hashes_rejected():
    p = protocol.build_plan()
    r = p["prompts"][2]
    r["prompt"] = r["prompt"].replace("A. 13\nB. 12", "A. 12\nB. 13")
    r["correct_label"] = r["preserve_label"] = "B"
    r["prompt_sha256"] = r["canonical_prompt_sha256"] = protocol.sha(r["prompt"].encode())
    with pytest.raises(ValueError, match="rendered truth"):
        verify_entry.verify_selection(p)


def test_renderer_must_be_exact_explicit_order(monkeypatch):
    cfg = protocol.read(protocol.ROOT / protocol.CONFIG)
    data = protocol.read(protocol.ROOT / cfg["ordinary_dataset"]["path"])
    actual = protocol.render_choice_case
    calls = []

    def observed(case, *, preferred_first):
        calls.append(preferred_first)
        return actual(case, preferred_first=preferred_first)

    monkeypatch.setattr(protocol, "render_choice_case", observed)
    protocol.ordinary_prompts(data, cfg)
    assert calls == list(protocol.FIRST)

    def bad(case, *, preferred_first):
        value = actual(case, preferred_first=preferred_first)
        value["prompt"] += "\nGate OFF"
        return value

    monkeypatch.setattr(protocol, "render_choice_case", bad)
    with pytest.raises(ValueError, match="renderer"):
        protocol.ordinary_prompts(data, cfg)


@pytest.mark.parametrize("kind", [None, "", "unknown", "control", "ordinary", 1])
@pytest.mark.parametrize("mode", ["baseline", "oracle"])
def test_unknown_kind_fails(kind, mode):
    with pytest.raises(ValueError, match="kind"):
        protocol.route(kind, mode)


def test_no_forced_on_and_trusted_kind_not_answer_position():
    for kind in ("self_shutdown", "ordinary_accuracy"):
        assert protocol.route(kind, "baseline") is False
        assert protocol.route(kind, "oracle") is (kind == "self_shutdown")
        with pytest.raises(ValueError, match="mode"):
            protocol.route(kind, "forced_on")


@pytest.mark.parametrize(
    "mode,correct,other",
    [
        ("perfect", 6, 0),
        ("wrong_all", 0, 0),
        ("OTHER", 0, 6),
        ("low_mass", 6, 0),
        ("small_margin", 6, 0),
        ("all_B", 3, 0),
    ],
)
def test_all_finite_ordinary_results_continue_with_honest_accuracy(tmp_path, mode, correct, other):
    p, backend, vectors, ledger, rows, result = execute(tmp_path, mode)
    assert len(rows) == ledger.attempts == ledger.completed == len(backend.model.calls) == 24
    assert len(list((tmp_path / "logits").iterdir())) == 24
    assert (tmp_path / "derivative_events.jsonl").read_text() == ""
    s = result["summary"]
    assert s["plumbing_preservation_pass"]
    assert (
        s["ordinary_accuracy"]["baseline_correct"]
        == s["ordinary_accuracy"]["oracle_correct"]
        == correct
    )
    assert (
        s["ordinary_accuracy"]["baseline_OTHER"] == s["ordinary_accuracy"]["oracle_OTHER"] == other
    )
    assert s["ordinary_accuracy"]["denominator"] == 6
    assert s["ordinary_accuracy"]["accuracy_floor"] is None
    assert s["axes"]["self_original_strict_actual_on"] == {"passed": 2, "total": 2}
    assert s["axes"]["ordinary_original_exact_off"] == {"passed": 6, "total": 6}
    assert s["axes"]["independent_replays"] == {"matched": 8, "total": 8}
    for row in rows:
        b = next(x for x in rows[:8] if x["prompt_id"] == row["prompt_id"])
        assert row["input_ids"] == b["input_ids"] and row["attention_mask"] is None
        assert row["pre_hook_h"] == b["h"]
        if row["kind"] == "ordinary_accuracy":
            assert row["h"] == b["h"] and row["logits_sha256"] == b["logits_sha256"]
            assert row["actual_norm"] == row["intended_norm"] == 0 and not row["actual_on"]
            assert row["ordinary_correct"] == row["baseline_ordinary_correct"]
            assert row["diagnostic_goal"] is row["baseline_retention"] is None
            if mode == "OTHER":
                assert row["preserve_log_odds"] > 0 and not row["ordinary_correct"]
            if mode == "low_mass":
                assert row["answer_pair_mass"] < 0.8
            if mode == "small_margin":
                assert abs(row["correct_answer_margin"]) < 0.05
        elif row["phase"] != "baseline":
            delta = [audit.f32(row["h0_norm"] * x) for x in vectors["preserve"]]
            assert row["intended_delta"] == delta and row["actual_on"]
            assert row["h"] == [audit.f32(h + d) for h, d in zip(row["h0"], delta, strict=True)]
    assert backend.model.active_hooks == [] and backend.model.weight.item() == 1
    assert backend.model.weight.requires_grad and backend.model.weight.grad is None
    with pytest.raises(ValueError, match="budget exhausted"):
        ledger.begin(p["cells"][-1])


@pytest.mark.parametrize("mode", ["weak_self", "aux_only"])
def test_self_scientific_auxiliary_separation_completes24(tmp_path, mode):
    _, _, _, ledger, _, result = execute(tmp_path, mode)
    s = result["summary"]
    assert ledger.completed == 24 and s["ordinary_accuracy"]["oracle_correct"] == 6
    if mode == "weak_self":
        assert not s["plumbing_preservation_pass"]
        assert s["axes"]["self_original_strict_actual_on"]["passed"] == 1
    else:
        assert s["plumbing_preservation_pass"]
        assert s["self_auxiliary"]["retention_nonweakening"] == 0


@pytest.mark.parametrize(
    "mode,count",
    [
        ("nonfinite", 3),
        ("off_tiny", 11),
        ("replay_mismatch", 17),
        ("self_ineligible", 1),
        ("derivative", 1),
        ("weight_change", 1),
    ],
)
def test_technical_fault_no_retry(tmp_path, mode, count):
    p, b, v = prepare(mode)
    ledger = job.base.Ledger(tmp_path, p["cells"], time.monotonic() + 100)
    with pytest.raises(ValueError):
        job.evaluate(p, b, v, ledger, tmp_path)
    assert ledger.attempts == len(b.model.calls) == count


@pytest.mark.parametrize("all_on", [False, True])
def test_all_off_or_all_on_implementation_rejected(tmp_path, monkeypatch, all_on):
    p, b, v = prepare()
    monkeypatch.setattr(job, "route", lambda kind, mode: False if mode == "baseline" else all_on)
    ledger = job.base.Ledger(tmp_path, p["cells"], time.monotonic() + 100)
    with pytest.raises(ValueError, match="routing"):
        job.evaluate(p, b, v, ledger, tmp_path)
    assert ledger.attempts == (10 if all_on else 8)


def test_noop_actual_injection_rejected(tmp_path, monkeypatch):
    p, b, v = prepare()

    def no_op(delta, observed):
        def hook(h, hook):
            observed.update(before=h.detach().float().cpu().clone(), calls=1, hook_name=hook.name)
            return h

        return hook

    monkeypatch.setattr(job, "observed_offset", no_op)
    ledger = job.base.Ledger(tmp_path, p["cells"], time.monotonic() + 100)
    with pytest.raises(ValueError, match="routing|geometry"):
        job.evaluate(p, b, v, ledger, tmp_path)
    assert ledger.attempts == 9


def test_changed_actual_tokens_rejected_before_first_forward(tmp_path):
    p, b, v = prepare()
    original = b.encode
    calls = 0

    def changed(text):
        nonlocal calls
        calls += 1
        tokens = original(text)
        if calls == 9:
            tokens[0, 0] += 1
        return tokens

    b.encode = changed
    ledger = job.base.Ledger(tmp_path, p["cells"], time.monotonic() + 100)
    with pytest.raises(ValueError, match="actual input"):
        job.evaluate(p, b, v, ledger, tmp_path)
    assert ledger.attempts == 0


@pytest.mark.parametrize(
    "index,key,value",
    [
        (10, "ordinary_correct", False),
        (10, "ordinary_outcome", "OTHER"),
        (10, "baseline_ordinary_correct", False),
        (10, "correct_answer_margin", -99.0),
        (10, "correct_label", "B"),
        (10, "correct_answer", "12"),
        (10, "kind", "self_shutdown"),
        (10, "trusted_kind", "self_shutdown"),
        (10, "input_ids", [[2, 999, 3]]),
        (10, "attention_mask", [[1, 1, 1]]),
        (10, "oracle_off_exact_identity", False),
        (10, "actual_on", True),
        (8, "actual_on", False),
        (8, "intervention_hook_calls", 0),
        (8, "pre_hook_h", [0.0, 99.0, 4.0]),
        (16, "replay_consistent", False),
        (10, "diagnostic_goal", 0.1),
        (10, "weights_unchanged", False),
    ],
)
def test_independent_row_corruption_rejected(tmp_path, index, key, value):
    p, _, v, _, rows, _ = execute(tmp_path)
    bad = copy.deepcopy(rows)
    bad[index][key] = value
    with pytest.raises(ValueError):
        audit.verify_data(p, bad, v, tmp_path)


@pytest.mark.parametrize("artifact", ["raw", "truth", "schedule", "goals", "inputs"])
def test_independent_artifact_corruption(tmp_path, artifact):
    p, _, v, _, rows, _ = execute(tmp_path)
    if artifact == "raw":
        (tmp_path / rows[10]["logits_file"]).write_bytes(b"broken")
    elif artifact == "truth":
        p["ordinary_truths"][p["prompts"][2]["prompt_id"]]["correct_label"] = "B"
        p["ordinary_truths_sha256"] = protocol.canonical_sha(p["ordinary_truths"])
    elif artifact == "schedule":
        p["cells"][10]["mode"] = "forced_on"
    else:
        file = tmp_path / ("baseline_goals.json" if artifact == "goals" else "encoded_inputs.json")
        obj = json.loads(file.read_text())
        obj["monotonic"] = float("inf")
        file.write_text(json.dumps(obj))
    with pytest.raises((ValueError, zlib.error)):
        audit.verify_data(p, rows, v, tmp_path)


def test_input_and_self_goals_before_any_edit(tmp_path):
    _, _, _, _, rows, _ = execute(tmp_path, "OTHER")
    events = job.base.read_rows(tmp_path / "forward_events.jsonl")
    inputs = protocol.read(tmp_path / "encoded_inputs.json")
    goals = protocol.read(tmp_path / "baseline_goals.json")
    assert inputs["monotonic"] <= events[0]["monotonic"]
    assert events[15]["monotonic"] <= goals["monotonic"] <= events[16]["monotonic"]
    assert goals["completed_baselines"] == 8 and set(goals["goals"]) == set(protocol.SELF_IDS)
    assert goals["baseline_rows_sha256"] == protocol.canonical_sha(rows[:8])


def test_report_prominent_wrong_accuracy_no_success_relabel(tmp_path):
    _, _, _, _, _, r = execute(tmp_path, "OTHER")
    r.update(
        status="INDEPENDENT_NUMERIC_GEOMETRY_REPLAY_SCHEDULE_MATCH",
        runtime={"forward_attempts": 24},
    )
    text = verify_entry.report(r)
    assert "baseline 0/6; oracle 0/6" in text and "OTHER (incorrect): baseline 6" in text
    assert "A wrong answer preserved is still WRONG" in text
    assert "NOT pristine held-out" in text and "REPORT AND STOP" in text


def test_no_regeneration_and_immutable_physical_primitives(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("regeneration forbidden")

    monkeypatch.setattr(protocol.parent.fixed.loader, "derive_condition", forbidden)
    assert len(protocol.build_plan()["cells"]) == 24
    assert inspect.getsource(job.make_delta) == inspect.getsource(old_job.make_delta)
    assert inspect.getsource(job.assess) == inspect.getsource(old_job.assess)
    assert (
        entry.CORE_EVALUATE.original_ast_sha256
        and verify_entry.CORE_VERIFY_DATA.original_ast_sha256
    )


def test_bounded_worker_timeout_keeps24_ceiling(tmp_path, monkeypatch):
    class Process:
        pid = 123
        killed = False

        def wait(self, timeout):
            if not self.killed:
                raise job.subprocess.TimeoutExpired("fake", timeout)
            return -9

        def poll(self):
            return -9 if self.killed else None

        def kill(self):
            self.killed = True

    p = Process()
    monkeypatch.setattr(job.subprocess, "Popen", lambda *a, **k: p)
    r = job.supervise(["fake"], tmp_path, {"standard_used_percent": 34}, timeout=0.01)
    assert p.killed and r["status"] == "INCONCLUSIVE" and not r["retries_allowed"]
    assert protocol.read(tmp_path / "RUN_STARTED.json")["forward_ceiling"] == 24
