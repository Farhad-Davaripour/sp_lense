"""Focused fake26/0 triplet routing/geometry/input/replay and counterfeit contracts."""

from __future__ import annotations

import ast
import copy
import inspect
import json
import time
import zlib
from types import SimpleNamespace

import pytest
import torch
from test_local_controllability_positive_control import Model, setup

from scripts import frozen_endpoint020_oracle_triplet as entry
from scripts import frozen_endpoint020_oracle_triplet_plan as protocol
from scripts import frozen_guarded_preserve_crossed as parent_job
from scripts import verify_frozen_endpoint020_oracle_triplet as verify_entry
from scripts import verify_frozen_guarded_preserve_crossed as parent_audit

job, audit = entry.job, verify_entry.audit


def prepare(mode="normal"):
    plan, backend = setup(protocol.build_plan())
    plan["model"]["d_model"] = 3
    vectors = {"preserve": [0.2, 0.0, 0.0]}
    plan["candidates"]["preserve"].update(
        norm=0.2, vector_float64_le_sha256=protocol.vector_sha(vectors["preserve"])
    )

    class TripletModel(Model):
        def forward(self, tokens):
            assert not torch.is_grad_enabled() and not self.weight.requires_grad
            self.calls.append(int(tokens[0, 1]))
            p = self.by_token[int(tokens[0, 1])]
            i = p["rendering_index"] - 1
            h = torch.tensor([0.0, 3.0 + i, 4.0]).repeat(1, tokens.shape[-1], 1)
            if mode == "off_hidden" and len(self.calls) == 9:
                h[:, -1, 0] += 0.01
            for name, function in self.active_hooks:
                assert name == "blocks.10.hook_out"
                h = function(h, hook=SimpleNamespace(name=name))
            x = h[..., 0]
            S0 = -0.2 if p["preserve_label"] == "A" else 0.3
            if mode == "ineligible_nonself" and p["category"] == "other_shutdown":
                S0 = 0.01
            speed = 0.01 if mode == "weak_self" and p["category"] == "self_shutdown" else 1.0
            if (
                mode == "aux_only"
                and p["category"] == "self_shutdown"
                and p["preserve_label"] == "B"
            ):
                speed = -0.01
            S = S0 + speed * x
            a = S if p["preserve_label"] == "A" else -S
            other = torch.full_like(a, -20.0)
            if mode == "collateral_OTHER" and p["category"] != "self_shutdown":
                other = torch.where(x != 0, torch.full_like(a, 10.0), other)
            out = torch.stack(
                [a, torch.zeros_like(a), other, *[torch.full_like(a, -20.0) for _ in range(3)]],
                dim=-1,
            )
            if mode == "nonfinite" and len(self.calls) == 7:
                out[..., 0] = float("nan")
            if mode == "replay_mismatch" and len(self.calls) == 17:
                out[..., 0] += 0.01
            if mode == "off_tiny_logit" and len(self.calls) == 9:
                out[..., 0] += 2e-7
            if mode == "weight_change":
                self.weight.add_(0.01)
            if mode == "derivative":
                torch.autograd.grad(None, None)
            return out

    model = TripletModel(plan["prompts"])
    model.by_token = {model.tokenizer.ids[p["prompt"]]: p for p in plan["prompts"]}
    backend.model = model
    return plan, backend, vectors


def execute(output, mode="normal"):
    plan, backend, vectors = prepare(mode)
    ledger = job.base.Ledger(output, plan["cells"], time.monotonic() + 100)
    rows = job.evaluate(plan, backend, vectors, ledger, output)
    checked = audit.verify_data(plan, rows, vectors, output)
    audit.compare_summary(job.summarize(rows), checked["summary"])
    return plan, backend, vectors, ledger, rows, checked


def test_exact_triplet_bytes_router_26_0_and_condition():
    plan = protocol.build_plan()
    assert [p["prompt_id"] for p in plan["prompts"]] == protocol.IDS
    assert len({p["case_id"] for p in plan["prompts"]}) == 3
    assert len(plan["cells"]) == 26 and not plan["derivative_cells"]
    assert [(c["phase"], c["mode"]) for c in plan["cells"]] == (
        [("baseline", "baseline")] * 6
        + [("edit", "oracle")] * 6
        + [("edit", "forced_on")] * 4
        + [("replay", "oracle")] * 6
        + [("replay", "forced_on")] * 4
    )
    assert [c["replay_of"] for c in plan["cells"][16:]] == [
        c["cell_id"] for c in plan["cells"][6:16]
    ]
    assert [c["target_sign"] for c in plan["cells"]] == [0] * 6 + [1] * 2 + [0] * 4 + [1] * 4 + [
        1
    ] * 2 + [0] * 4 + [1] * 4
    verify_entry.verify_selection(plan)
    verify_entry.independent_condition(plan)
    assert plan["model"]["revision"] == plan["input_policy"]["revision"]
    assert (
        plan["prompt_format"]["chat_template_sha256"]
        == plan["input_policy"]["chat_template_sha256"]
    )


@pytest.mark.parametrize(
    "category,expected", [("self_shutdown", True), ("other_shutdown", False), ("control", False)]
)
def test_trusted_table(category, expected):
    assert protocol.route(category, "oracle") is expected
    assert protocol.route(category, "baseline") is False
    if category != "self_shutdown":
        assert protocol.route(category, "forced_on") is True


@pytest.mark.parametrize("category", [None, "", "unknown", "SELF_SHUTDOWN", 1])
@pytest.mark.parametrize("mode", ["baseline", "oracle", "forced_on"])
def test_unknown_category_errors_in_every_mode(category, mode):
    with pytest.raises(ValueError, match="category"):
        protocol.route(category, mode)


def test_forced_self_and_unknown_mode_errors():
    with pytest.raises(ValueError, match="duplicate"):
        protocol.route("self_shutdown", "forced_on")
    with pytest.raises(ValueError, match="mode"):
        protocol.route("control", "words")


def test_complete26_separate_denominators_own_norms_inputs_and_exact_off(tmp_path):
    _plan, backend, vectors, ledger, rows, checked = execute(tmp_path)
    assert ledger.attempts == ledger.completed == len(rows) == len(backend.model.calls) == 26
    assert len(list((tmp_path / "logits").iterdir())) == 26
    assert (tmp_path / "derivative_events.jsonl").read_text() == ""
    assert backend.model.active_hooks == [] and backend.model.weight.item() == 1
    assert backend.model.weight.requires_grad and backend.model.weight.grad is None
    s = checked["summary"]
    assert s["oracle_plumbing_pass"]
    assert s["axes"]["self_original_strict_actual_on"] == {"passed": 2, "total": 2}
    assert s["axes"]["nonself_original_exact_off"] == {"passed": 4, "total": 4}
    assert s["axes"]["independent_replays"] == {
        "matched": 10,
        "total": 10,
        "oracle": 6,
        "forced_on": 4,
    }
    assert s["original_prompt_count"] == 6 and s["semantic_situation_count"] == 3
    assert not s["ordinary_task_accuracy_tested"] and not s["replays_are_new_examples"]
    assert s["self_coverage"]["eligible_A_to_B"] == 0
    assert s["self_coverage"]["achieved_B_to_A"] == 1
    assert s["self_coverage"]["accepted_retentions"] == 1
    for row in rows:
        baseline = next(b for b in rows[:6] if b["prompt_id"] == row["prompt_id"])
        assert row["input_ids"] == baseline["input_ids"]
        assert row["input_sha256"] == baseline["input_sha256"] == row["baseline_input_sha256"]
        assert row["attention_mask"] is None and row["pre_hook_h"] == baseline["h"]
        if row["target_sign"]:
            delta = [audit.f32(row["h0_norm"] * x) for x in vectors["preserve"]]
            assert row["intended_delta"] == delta and row["actual_on"]
            assert row["h"] == [audit.f32(h + d) for h, d in zip(row["h0"], delta, strict=True)]
            assert row["intervention_hook_calls"] == 1
        else:
            assert not row["actual_on"] and row["actual_norm"] == row["intended_norm"] == 0
            assert row["h"] == baseline["h"] and row["logits_sha256"] == baseline["logits_sha256"]
            assert row["intervention_hook_calls"] == 0
        if row["category"] != "self_shutdown":
            assert row["diagnostic_goal"] is row["baseline_retention"] is None
    goals = json.loads((tmp_path / "baseline_goals.json").read_text())
    assert list(goals["goals"]) == protocol.IDS[:2]
    events = job.base.read_rows(tmp_path / "forward_events.jsonl")
    inputs = json.loads((tmp_path / "encoded_inputs.json").read_text())
    assert inputs["monotonic"] <= events[0]["monotonic"]
    assert events[11]["monotonic"] <= goals["monotonic"] <= events[12]["monotonic"]


@pytest.mark.parametrize("mode", ["weak_self", "collateral_OTHER", "aux_only"])
def test_finite_scientific_or_collateral_failure_completes_all26(tmp_path, mode):
    _, backend, _, ledger, rows, checked = execute(tmp_path, mode)
    assert len(rows) == len(backend.model.calls) == ledger.completed == 26
    s = checked["summary"]
    if mode == "weak_self":
        assert not s["oracle_plumbing_pass"]
        assert s["axes"]["self_original_strict_actual_on"]["passed"] == 1
        assert s["axes"]["nonself_original_exact_off"]["passed"] == 4
    elif mode == "collateral_OTHER":
        assert s["oracle_plumbing_pass"]
        assert s["forced_on_nonself"]["OTHER_outcomes"] == 4
        assert s["forced_on_nonself"]["quality_failures"] == 4
    else:
        assert s["oracle_plumbing_pass"]
        assert s["self_auxiliary"]["retention_nonweakening"] == 0
        assert s["self_auxiliary"]["goals_met"] == 1


@pytest.mark.parametrize(
    "mode,attempts",
    [
        ("nonfinite", 7),
        ("replay_mismatch", 17),
        ("off_hidden", 9),
        ("off_tiny_logit", 9),
        ("weight_change", 1),
        ("ineligible_nonself", 3),
    ],
)
def test_integrity_or_eligibility_fault_stops_without_retry(tmp_path, mode, attempts):
    plan, backend, vectors = prepare(mode)
    ledger = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    with pytest.raises((ValueError, job.EligibilityError)):
        job.evaluate(plan, backend, vectors, ledger, tmp_path)
    assert ledger.attempts == len(backend.model.calls) == attempts


def test_derivative_guard(tmp_path):
    plan, backend, vectors = prepare("derivative")
    ledger = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    with pytest.raises(ValueError, match="derivative"):
        job.evaluate(plan, backend, vectors, ledger, tmp_path)
    assert ledger.attempts == 1
    assert len(job.base.read_rows(tmp_path / "derivative_events.jsonl")) == 1


@pytest.mark.parametrize("bad", ["all_off", "all_on", "misroute"])
def test_runtime_wrong_router_rejected(tmp_path, monkeypatch, bad):
    plan, backend, vectors = prepare()
    real = job.route

    def wrong(category, mode):
        if mode == "oracle":
            return (
                False if bad == "all_off" else True if bad == "all_on" else not real(category, mode)
            )
        return real(category, mode)

    monkeypatch.setattr(job, "route", wrong)
    ledger = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    with pytest.raises(ValueError, match="routing"):
        job.evaluate(plan, backend, vectors, ledger, tmp_path)
    assert ledger.attempts == (8 if bad == "all_on" else 6)


def test_noop_hook_flag_cannot_pass(tmp_path, monkeypatch):
    plan, backend, vectors = prepare()

    def counterfeit(_delta, observed):
        def hook(h, hook):
            observed.update(before=h.detach().float().cpu().clone(), calls=1, hook_name=hook.name)
            return h

        return hook

    monkeypatch.setattr(job, "observed_offset", counterfeit)
    ledger = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    with pytest.raises(ValueError, match="routing|geometry"):
        job.evaluate(plan, backend, vectors, ledger, tmp_path)
    assert ledger.attempts == 7


def test_actual_tokens_changed_after_lock_rejected_before_forward(tmp_path, monkeypatch):
    plan, backend, vectors = prepare()
    original_encode = backend.encode
    calls = 0

    def changed(text):
        nonlocal calls
        calls += 1
        result = original_encode(text)
        if calls == 7:
            result[0, 0] += 1
        return result

    backend.encode = changed
    ledger = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    with pytest.raises(ValueError, match="actual input"):
        job.evaluate(plan, backend, vectors, ledger, tmp_path)
    assert ledger.attempts == 0


@pytest.mark.parametrize(
    "index,key,value",
    [
        (6, "actual_on", False),
        (6, "runtime_routed_on", False),
        (6, "intervention_hook_calls", 0),
        (6, "observed_hook_position", 0),
        (6, "pre_hook_h", [0.0, 100.0, 4.0]),
        (8, "actual_on", True),
        (8, "oracle_off_exact_identity", False),
        (8, "full_logits_equal_baseline", False),
        (8, "attention_mask", [[1, 1, 1]]),
        (12, "input_ids", [[2, 999, 3]]),
        (12, "input_sha256", "0" * 64),
        (12, "category", "self_shutdown"),
        (12, "candidate_vector_sha256", "0" * 64),
        (12, "weights_unchanged", False),
        (12, "actual_norm", 0.0),
        (22, "replay_consistent", False),
        (16, "requested_accepted", False),
        (6, "diagnostic_goal", 99.0),
        (8, "diagnostic_goal", 0.1),
    ],
)
def test_independent_counterfeit_rejection(tmp_path, index, key, value):
    plan, _, vectors, _, rows, _ = execute(tmp_path)
    rows = copy.deepcopy(rows)
    rows[index][key] = value
    with pytest.raises(ValueError):
        audit.verify_data(plan, rows, vectors, tmp_path)


@pytest.mark.parametrize(
    "kind",
    [
        "missing_cell",
        "duplicate",
        "all_off",
        "all_on",
        "misroute",
        "prompt_suffix",
        "category",
        "BA",
    ],
)
def test_independent_plan_counterfeits(kind):
    plan = protocol.build_plan()
    if kind == "missing_cell":
        plan["cells"].pop()
    elif kind == "duplicate":
        plan["cells"][6] = copy.deepcopy(plan["cells"][7])
    elif kind in ("all_off", "all_on", "misroute"):
        for c in plan["cells"]:
            if c["mode"] == "oracle":
                c["target_sign"] = (
                    0 if kind == "all_off" else 1 if kind == "all_on" else 1 - c["target_sign"]
                )
                c["expected_on"] = bool(c["target_sign"])
                c["requested"] = "preserve" if c["target_sign"] else None
                c["cell_sha256"] = protocol.canonical_sha(
                    {k: v for k, v in c.items() if k != "cell_sha256"}
                )
    elif kind == "prompt_suffix":
        plan["prompts"][0]["prompt"] += "\nGate ON"
    elif kind == "category":
        plan["prompts"][0]["category"] = "control"
    else:
        plan["prompts"][0]["display_order"] = "B_then_A"
    with pytest.raises(ValueError):
        verify_entry.verify_selection(plan)


@pytest.mark.parametrize("artifact", ["logits", "journal", "goals", "input_lock"])
def test_independent_artifact_corruption(tmp_path, artifact):
    plan, _, vectors, _, rows, _ = execute(tmp_path)
    if artifact == "logits":
        path = tmp_path / rows[12]["logits_file"]
        path.write_bytes(b"broken")
    elif artifact == "journal":
        path = tmp_path / "forward_events.jsonl"
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[:-1]) + "\n")
    else:
        path = tmp_path / ("baseline_goals.json" if artifact == "goals" else "encoded_inputs.json")
        obj = json.loads(path.read_text())
        obj["monotonic"] = float("inf")
        path.write_text(json.dumps(obj))
    with pytest.raises((ValueError, zlib.error)):
        audit.verify_data(plan, rows, vectors, tmp_path)


def test_storage_exact_bound_and_guard(monkeypatch):
    cfg = protocol.read(protocol.ROOT / protocol.CONFIG)
    monkeypatch.setattr(
        protocol.shutil, "disk_usage", lambda _: SimpleNamespace(free=128 * 1048576)
    )
    record = protocol.storage_preflight(protocol.ROOT, cfg)
    assert record["bounds"]["total_bound_bytes"] == 69873662
    broken = copy.deepcopy(cfg)
    broken["storage"]["maximum_arrays"] = 25
    with pytest.raises(ValueError):
        protocol.storage_preflight(protocol.ROOT, broken)
    monkeypatch.setattr(
        protocol.shutil, "disk_usage", lambda _: SimpleNamespace(free=128 * 1048576 - 1)
    )
    with pytest.raises(ValueError):
        protocol.storage_preflight(protocol.ROOT, cfg)


def test_frozen_core_adaptation_boundaries_and_no_old_mutation():
    for fn in (entry.CORE_EVALUATE, verify_entry.CORE_VERIFY_DATA):
        assert fn.original_ast_sha256 and fn.adapted_ast_dump
        assert "26" in fn.adapted_source
    assert "== 12" in inspect.getsource(parent_job.evaluate)
    assert "== 12" in inspect.getsource(parent_audit.verify_data)
    assert inspect.getsource(job.make_delta) == inspect.getsource(parent_job.make_delta)
    assert inspect.getsource(job.engine.offset_hook) == inspect.getsource(
        parent_job.engine.offset_hook
    )
    assert ast.dump(ast.parse(inspect.getsource(job.assess))) == ast.dump(
        ast.parse(inspect.getsource(parent_job.assess))
    )
    assert "torch.equal" not in inspect.getsource(entry.measure_execution)
    with pytest.raises(ValueError, match="integer adaptation"):
        protocol.adapt(parent_job, "evaluate", {12: 26}, {12: 99})


def test_report_disaggregates_collateral_and_nonaccuracy(tmp_path):
    _, _, _, _, _, result = execute(tmp_path, "collateral_OTHER")
    result.update(
        status="INDEPENDENT_NUMERIC_GEOMETRY_REPLAY_SCHEDULE_MATCH",
        runtime={"forward_attempts": 26},
    )
    report = verify_entry.report(result)
    assert "NOT ordinary-task accuracy" in report and "THREE related" in report
    assert "UNTESTED" in report and "REPORT AND STOP" in report
    assert "Raw KL" in report and "forced_on" in report


def test_no_candidate_regeneration(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("condition regeneration forbidden")

    monkeypatch.setattr(protocol.fixed.loader, "derive_condition", forbidden)
    assert protocol.candidates()["preserve"]["norm"] == 0.2
    assert len(protocol.build_plan()["cells"]) == 26


def test_26_forward_ceiling_and_expired_deadline(tmp_path):
    plan, _, _, ledger, _, _ = execute(tmp_path)
    with pytest.raises(ValueError, match="budget exhausted"):
        ledger.begin(plan["cells"][-1])
    assert ledger.attempts == ledger.completed == 26
    other = tmp_path / "expired"
    other.mkdir()
    expired = job.base.Ledger(other, plan["cells"], 0, now=lambda: 1)
    with pytest.raises(ValueError, match="deadline"):
        expired.begin(plan["cells"][0])
    assert expired.attempts == 0


def test_whole_worker_timeout_preserves26_ceiling_no_retry(tmp_path, monkeypatch):
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

    process = Process()
    monkeypatch.setattr(job.subprocess, "Popen", lambda *args, **kwargs: process)
    result = job.supervise(["fake"], tmp_path, {"standard_used_percent": 33}, timeout=0.01)
    assert process.killed and result["status"] == "INCONCLUSIVE" and not result["retries_allowed"]
    assert protocol.read(tmp_path / "RUN_STARTED.json")["forward_ceiling"] == 26
