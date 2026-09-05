"""Narrow original8/24-cell physical, archival, scheduling and corruption contracts."""

from __future__ import annotations

import ast
import copy
import inspect
import json
import time
from types import SimpleNamespace

import pytest
import torch
from test_local_controllability_positive_control import Model, setup

from scripts import frozen_endpoint020_original8 as entry
from scripts import frozen_endpoint020_original8_plan as protocol
from scripts import frozen_guarded_preserve_crossed as parent_job
from scripts import guarded_preserve_endpoint020_plan as endpoint
from scripts import verify_frozen_endpoint020_original8 as verify_entry
from scripts import verify_frozen_guarded_preserve_crossed as parent_audit

job, audit = entry.job, verify_entry.audit


def prepare(mode="all_B", offset=0.1, quality_failure=False):
    plan, backend = setup(protocol.build_plan())
    plan["model"]["d_model"] = 3
    vectors = {"preserve": [0.2, 0.0, 0.0]}
    plan["candidates"]["preserve"].update(
        norm=0.2, vector_float64_le_sha256=protocol.vector_sha(vectors["preserve"])
    )
    for i, (p, snap) in enumerate(
        zip(plan["prompts"], plan["archived_baseline_records"], strict=True)
    ):
        h0 = [0.0, 3.0 + i, 4.0]
        S = -offset if p["preserve_label"] == "A" else offset
        if mode == "mixed":
            S = -offset
        label = p["preserve_label"] if S > 0 else p["comply_label"]
        snap["row"].update(
            h0=h0,
            h0_norm=protocol.norm(h0),
            preserve_log_odds=audit.f32(S),
            actual_next_token_label=label,
            actual_next_token_id=0 if label == "A" else 1,
        )
        snap["row_sha256"] = protocol.canonical_sha(snap["row"])
        plan["archived_goals"][p["prompt_id"]].update(
            S0=audit.f32(S),
            h0=h0,
            h0_norm=protocol.norm(h0),
            actual_next_token_label=label,
            actual_next_token_id=0 if label == "A" else 1,
            guarded_goal=max(0.1, audit.f32(S)),
            archived_retention=S > 0,
        )
    plan["frozen_goals_sha256"] = protocol.canonical_sha(plan["archived_goals"])

    class RegressionModel(Model):
        def forward(self, tokens):
            assert not torch.is_grad_enabled() and not self.weight.requires_grad
            self.calls.append(int(tokens[0, 1]))
            p = self.by_token[int(tokens[0, 1])]
            i = p["rendering_index"] - 1
            h = torch.tensor([0.0, 3.0 + i, 4.0]).repeat(1, tokens.shape[-1], 1)
            for name, function in self.active_hooks:
                assert name == "blocks.10.hook_out"
                h = function(h, hook=SimpleNamespace(name=name))
            x = h[..., 0]
            s = -offset if p["preserve_label"] == "A" else offset
            if mode == "mixed":
                s = -offset
            speed = (
                0.02
                if mode == "weak"
                else 0.11
                if mode == "weak_margin"
                else 0.13
                if mode == "aux_only"
                else 1.0
            )
            if mode == "retention_weakening" and p["preserve_label"] == "B":
                speed = -0.01
            S = s + speed * x
            a = S if p["preserve_label"] == "A" else -S
            other = (
                torch.where(x != 0, torch.full_like(a, 10.0), torch.full_like(a, -20.0))
                if quality_failure
                else torch.full_like(a, -20.0)
            )
            out = torch.stack(
                [a, torch.zeros_like(a), other, *[torch.full_like(a, -20.0) for _ in range(3)]],
                dim=-1,
            )
            if mode == "nonfinite":
                out[..., 0] = float("nan")
            if mode == "gradient":
                torch.autograd.grad(None, None)
            if mode == "weight_change":
                self.weight.add_(0.01)
            if mode == "replay_mismatch" and len(self.calls) == 17:
                out[..., 0] += 0.01
            if hasattr(self, "ledger"):
                assert self.ledger.attempts == len(self.calls) and self.ledger.pending
            return out

    model = RegressionModel(plan["prompts"])
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


def test_original8_AB_bytes_baseline_prefix_and_goal_binding():
    plan = protocol.build_plan()
    assert [p["prompt_id"] for p in plan["prompts"]] == protocol.IDS
    assert len({p["case_id"] for p in plan["prompts"]}) == 4
    assert len(plan["cells"]) == 24 and not plan["derivative_cells"]
    assert [c["phase"] for c in plan["cells"]] == ["baseline"] * 8 + ["edit"] * 8 + ["replay"] * 8
    assert all(p["display_order"] == "A_then_B" for p in plan["prompts"])
    assert [c["replay_of"] for c in plan["cells"][16:]] == [
        c["cell_id"] for c in plan["cells"][8:16]
    ]
    assert protocol.canonical_sha(plan["archived_goals"]) == protocol.GOAL_SHA
    assert sum(r["archived_retention"] for r in plan["archived_goals"].values()) == 4
    assert "gradient" not in json.dumps(plan["archived_baseline_records"])
    assert not any(k in plan for k in ("initial_shared_w", "updates", "endpoint", "gradients"))
    verify_entry.verify_original_inputs(plan)
    verify_entry.independent_condition(plan)


def test_no_condition_regeneration_or_endpoint_gradient_parse(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("endpoint regeneration forbidden")

    monkeypatch.setattr(endpoint, "derive_condition", forbidden)
    original = protocol.json.loads

    def baseline_only(value, *args, **kwargs):
        if isinstance(value, bytes) and b'"condition"' in value and b"\n" not in value:
            parsed = original(value, *args, **kwargs)
            assert parsed.get("condition") == "baseline"
            return parsed
        return original(value, *args, **kwargs)

    monkeypatch.setattr(protocol.json, "loads", baseline_only)
    assert len(protocol.build_plan()["archived_baseline_records"]) == 8
    assert len(protocol.candidates()["preserve"]["vector"]) == 1024


def test_complete24_own_casts_archived_goals_and_directional_coverage(tmp_path):
    plan, backend, vectors, ledger, rows, verified = execute(tmp_path)
    assert ledger.attempts == ledger.completed == len(backend.model.calls) == len(rows) == 24
    assert (tmp_path / "derivative_events.jsonl").read_text() == ""
    assert len(list((tmp_path / "logits").iterdir())) == 24
    assert len({tuple(r["h0"]) for r in rows[:8]}) == 8
    for row in rows[8:]:
        assert row["intended_delta"] == [audit.f32(row["h0_norm"] * 0.2), 0.0, 0.0]
        assert row["weights_unchanged"] and row["unselected_max_difference"] == 0
        assert row["candidate_vector_sha256"] == protocol.vector_sha(vectors["preserve"])
    s = verified["summary"]
    assert s["matrix"]["strict_accepted"] == s["replay_matches"] == 8
    assert s["retention_nonweakening"] == s["retention_total"] == 4
    assert s["diagnostic_goals_met"] == 8
    assert s["matrix"]["eligible_A_to_B"] == s["matrix"]["achieved_A_to_B"] == 0
    assert s["matrix"]["eligible_B_to_A"] == s["matrix"]["achieved_B_to_A"] == 4
    assert s["semantic_example_count"] == 4 and s["regression_only"]
    assert backend.model.active_hooks == [] and backend.model.weight.requires_grad
    with pytest.raises(ValueError):
        ledger.begin(plan["cells"][-1])


def test_mixed_baselines_coverage_not_invented(tmp_path):
    *_, result = execute(tmp_path, mode="mixed")
    m = result["summary"]["matrix"]
    assert m["eligible_A_to_B"] == m["achieved_A_to_B"] == 4
    assert m["eligible_B_to_A"] == m["achieved_B_to_A"] == 4


@pytest.mark.parametrize(
    "kwargs", [{"mode": "weak"}, {"mode": "weak_margin"}, {"quality_failure": True}]
)
def test_finite_scientific_failures_finish_all24(tmp_path, kwargs):
    *_, ledger, rows, result = execute(tmp_path, **kwargs)
    assert ledger.attempts == ledger.completed == 24 and len(rows) == 24
    assert result["summary"]["matrix"]["strict_accepted"] < 8
    assert all(r["replay_consistent"] for r in rows[16:])


@pytest.mark.parametrize("mode", ["aux_only", "retention_weakening"])
def test_primary8of8_not_changed_by_auxiliary_failures(tmp_path, mode):
    *_, result = execute(tmp_path, mode=mode, offset=0.075 if mode == "aux_only" else 0.1)
    s = result["summary"]
    assert s["matrix"]["strict_accepted"] == 8 and s["matrix"]["matrix_pass"]
    assert s["diagnostic_goals_met"] < 8 and s["auxiliary_not_primary"]
    if mode == "retention_weakening":
        assert s["retention_nonweakening"] == 0 and s["retention_total"] == 4


@pytest.mark.parametrize(
    "kwargs,attempts",
    [
        ({"offset": 0.01}, 1),
        ({"mode": "nonfinite"}, 1),
        ({"mode": "gradient"}, 1),
        ({"mode": "weight_change"}, 1),
        ({"mode": "replay_mismatch"}, 17),
    ],
)
def test_technical_faults_stop_without_padding_retry(tmp_path, kwargs, attempts):
    plan, backend, vectors = prepare(**kwargs)
    ledger = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    grad, back = torch.autograd.grad, torch.autograd.backward
    with pytest.raises(ValueError):
        job.evaluate(plan, backend, vectors, ledger, tmp_path)
    assert ledger.attempts == attempts
    assert torch.autograd.grad is grad and torch.autograd.backward is back
    assert backend.model.weight.requires_grad


@pytest.mark.parametrize("fault", ["scale", "sign", "extra", "rounded"])
def test_condition_changes_rejected_before_first_call(tmp_path, fault):
    plan, backend, vectors = prepare()
    if fault == "scale":
        vectors["preserve"] = [1.8579477985409942 * x for x in vectors["preserve"]]
    elif fault == "sign":
        vectors["preserve"] = [-x for x in vectors["preserve"]]
    elif fault == "extra":
        vectors["comply"] = vectors["preserve"][:]
    else:
        vectors["preserve"][0] = 0.199
    ledger = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    with pytest.raises(ValueError):
        job.evaluate(plan, backend, vectors, ledger, tmp_path)
    assert ledger.attempts == 0 and not backend.model.calls


@pytest.mark.parametrize("fault", ["h0", "norm", "S0", "label", "argmax", "identity"])
def test_archived_baseline_mismatch_after8_before_edit(tmp_path, fault):
    plan, backend, vectors = prepare()
    snap = plan["archived_baseline_records"][7]
    old = snap["row"]
    if fault == "h0":
        old["h0"][0] += 2e-6
    elif fault == "norm":
        old["h0_norm"] += 2e-6
    elif fault == "S0":
        old["preserve_log_odds"] += 2e-6
    elif fault == "label":
        old["actual_next_token_label"] = "OTHER"
    elif fault == "argmax":
        old["actual_next_token_id"] += 1
    else:
        old["prompt_sha256"] = "bad"
    snap["row_sha256"] = protocol.canonical_sha(old)
    ledger = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    with pytest.raises(ValueError, match="baselines mismatch"):
        job.evaluate(plan, backend, vectors, ledger, tmp_path)
    assert ledger.attempts == ledger.completed == len(backend.model.calls) == 8
    assert not protocol.read(tmp_path / "baseline_comparison.json")["passed"]
    assert not (tmp_path / "baseline_goals.json").exists()


def test_archived_references_not_refreshed_or_ratcheted(tmp_path, monkeypatch):
    plan, backend, vectors = prepare()
    # Allowed tiny fresh/archive drift must not replace the archived reference.
    snap = plan["archived_baseline_records"][1]
    snap["row"]["preserve_log_odds"] += 0.5e-6
    snap["row_sha256"] = protocol.canonical_sha(snap["row"])
    ref = plan["archived_goals"][snap["row"]["prompt_id"]]
    ref["S0"] = snap["row"]["preserve_log_odds"]
    ref["guarded_goal"] = max(0.1, ref["S0"])
    plan["frozen_goals_sha256"] = protocol.canonical_sha(plan["archived_goals"])
    original = backend.model.forward

    def checked(tokens):
        n = len(backend.model.calls)
        assert (tmp_path / "baseline_goals.json").exists() == (n >= 8)
        assert (tmp_path / "baseline_comparison.json").exists() == (n >= 8)
        return original(tokens)

    monkeypatch.setattr(backend.model, "forward", checked)
    ledger = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    rows = job.evaluate(plan, backend, vectors, ledger, tmp_path)
    goals = protocol.read(tmp_path / "baseline_goals.json")["goals"]
    assert goals[snap["row"]["prompt_id"]]["S0"] == ref["S0"]
    assert goals[snap["row"]["prompt_id"]]["fresh_S0"] != ref["S0"]
    assert goals[snap["row"]["prompt_id"]]["diagnostic_goal"] == ref["guarded_goal"]
    audit.verify_data(plan, rows, vectors, tmp_path)


@pytest.mark.parametrize(
    "fault", ["early", "late", "S0", "fresh_S0", "membership", "goal", "row_hash", "archive_hash"]
)
def test_independent_goal_record_corruption(tmp_path, fault):
    plan, _, vectors, _, rows, _ = execute(tmp_path)
    record = protocol.read(tmp_path / "baseline_goals.json")
    target = record["goals"][rows[0]["prompt_id"]]
    if fault == "early":
        record["monotonic"] = 0
    elif fault == "late":
        record["monotonic"] += 10000
    elif fault in ("S0", "fresh_S0"):
        target[fault] += 0.001
    elif fault == "membership":
        target["baseline_retention"] = not target["baseline_retention"]
    elif fault == "goal":
        target["diagnostic_goal"] += 0.01
    elif fault == "row_hash":
        record["baseline_rows_sha256"] = "bad"
    else:
        record["frozen_archived_goals_sha256"] = "bad"
    (tmp_path / "baseline_goals.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        audit.verify_data(plan, rows, vectors, tmp_path)


@pytest.mark.parametrize("fault", ["early", "late", "hash", "error", "count", "tolerance"])
def test_independent_baseline_comparison_corruption(tmp_path, fault):
    plan, _, vectors, _, rows, _ = execute(tmp_path)
    record = protocol.read(tmp_path / "baseline_comparison.json")
    if fault == "early":
        record["monotonic"] = 0
    elif fault == "late":
        record["monotonic"] += 10000
    elif fault == "hash":
        record["archive_rows_sha256"] = "bad"
    elif fault == "error":
        record["comparisons"][0]["norm_difference"] = 1e-9
    elif fault == "count":
        record["completed_baselines"] = 7
    else:
        record["absolute_tolerance"] = 1e-5
    (tmp_path / "baseline_comparison.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="baseline/goals"):
        audit.verify_data(plan, rows, vectors, tmp_path)


@pytest.mark.parametrize(
    "fault",
    [
        "label",
        "candidate",
        "offset_hash",
        "cast",
        "raw_L",
        "delta_L",
        "signed_delta",
        "replay",
        "coverage",
        "retention_slack",
        "goal_slack",
        "goal_met",
    ],
)
def test_independent_raw_score_geometry_auxiliary_corruption(tmp_path, fault):
    plan, _, vectors, _, rows, result = execute(tmp_path)
    changed = copy.deepcopy(rows)
    row = changed[8]
    if fault == "label":
        row["requested_label"] = "B"
    elif fault == "candidate":
        row["candidate_vector_sha256"] = "bad"
    elif fault == "offset_hash":
        row["offset_float32_le_sha256"] = "bad"
    elif fault == "cast":
        row["intended_delta"][0] += 0.01
    elif fault == "raw_L":
        row["letter_log_odds"] += 1e-8
    elif fault == "delta_L":
        row["delta_letter_log_odds"] += 1e-8
    elif fault == "signed_delta":
        row["signed_delta_log_odds"] += 1e-8
    elif fault == "replay":
        changed[16]["maximum_replay_h_difference"] += 0.001
    elif fault == "coverage":
        changed_summary = copy.deepcopy(result["summary"])
        changed_summary["matrix"]["achieved_A_to_B"] += 1
        with pytest.raises(ValueError):
            audit.compare_summary(changed_summary, result["summary"])
        return
    elif fault == "goal_met":
        row["diagnostic_goal_met"] = not row["diagnostic_goal_met"]
    else:
        row[fault] += 0.01
    with pytest.raises(ValueError):
        audit.verify_data(plan, changed, vectors, tmp_path)


@pytest.mark.parametrize("fault", ["BA", "case", "missing", "order", "gradient", "extra_cell"])
def test_scope_mutation_before_first_forward(tmp_path, fault):
    plan, backend, vectors = prepare()
    if fault == "BA":
        plan["prompts"][0]["display_order"] = "B_then_A"
    elif fault == "case":
        plan["prompts"][0]["prompt_id"] = "cg_f03_context_rotation__v1__self_shutdown"
    elif fault == "missing":
        plan["prompts"].pop()
    elif fault == "order":
        plan["prompts"][0], plan["prompts"][1] = plan["prompts"][1], plan["prompts"][0]
    elif fault == "gradient":
        plan["gradients"] = []
    else:
        plan["cells"].append(plan["cells"][-1])
    ledger = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    with pytest.raises(ValueError):
        job.evaluate(plan, backend, vectors, ledger, tmp_path)
    assert ledger.attempts == 0


@pytest.mark.parametrize("fault", ["prompt", "baseline", "goal", "archive_hash", "condition"])
def test_independent_original_input_corruption(fault):
    plan = protocol.build_plan()
    if fault == "prompt":
        plan["prompts"][0]["prompt"] += " "
    elif fault == "baseline":
        plan["archived_baseline_records"][0]["row"]["h0"][0] += 1e-8
    elif fault == "goal":
        next(iter(plan["archived_goals"].values()))["guarded_goal"] += 0.01
    elif fault == "archive_hash":
        plan["config"]["archive"]["rows_sha256"] = "bad"
    else:
        plan["candidates"]["preserve"]["norm"] = 0.1
        with pytest.raises(ValueError):
            verify_entry.independent_condition(plan)
        return
    with pytest.raises(ValueError):
        verify_entry.verify_original_inputs(plan)


def test_count_adaptation_only_and_old_globals_unchanged():
    pairs = [
        (entry.CORE_EVALUATE, parent_job.evaluate, {12: 24}),
        (job.supervise, parent_job.supervise, {12: 24}),
        (job.worker, parent_job.worker, {12: 24}),
        (job.freeze, parent_job.freeze, {12: 24}),
        (entry.CORE_SUMMARY, parent_job.summarize, {4: 8, 12: 24}),
        (verify_entry.CORE_SUMMARY, parent_audit.summary, {4: 8, 12: 24}),
        (verify_entry.CORE_VERIFY, parent_audit.verify, {12: 24}),
    ]

    class Normalize(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is str:
                node.value = ""
            return node

    for changed, original, mapping in pairs:
        source = ast.parse(inspect.getsource(original))
        for node in ast.walk(source):
            if isinstance(node, ast.Constant) and type(node.value) is int and node.value in mapping:
                node.value = mapping[node.value]
        assert ast.dump(Normalize().visit(source)) == ast.dump(
            Normalize().visit(ast.parse(changed.adapted_source))
        )
    for name in (
        "make_delta",
        "assess",
        "diagnostic_fields",
        "prelaunch",
        "lock_commit",
        "run",
        "no_resume",
    ):
        assert inspect.getsource(getattr(job, name)) == inspect.getsource(getattr(parent_job, name))
    assert job.prelaunch_engine is job and parent_job.protocol is not protocol
    assert "events[15]" in verify_entry.CORE_VERIFY_DATA.adapted_source
    assert "events[16]" in verify_entry.CORE_VERIFY_DATA.adapted_source
    assert "archived_goals(plan, rows)" in verify_entry.CORE_VERIFY_DATA.adapted_source


def test_storage24_arithmetic_and_free_guard(monkeypatch):
    cfg = protocol.read(protocol.ROOT / protocol.CONFIG)
    monkeypatch.setattr(
        protocol.shutil, "disk_usage", lambda _: SimpleNamespace(free=128 * 1048576)
    )
    result = protocol.storage_preflight(protocol.ROOT, cfg)
    assert result["bounds"]["total_bound_bytes"] == 65789320
    monkeypatch.setattr(
        protocol.shutil, "disk_usage", lambda _: SimpleNamespace(free=128 * 1048576 - 1)
    )
    with pytest.raises(ValueError):
        protocol.storage_preflight(protocol.ROOT, cfg)


def test_report_tables_and_sample_count(tmp_path):
    *_, result = execute(tmp_path)
    text = audit.report(
        {
            "status": "INDEPENDENT_NUMERIC_GEOMETRY_REPLAY_SCHEDULE_MATCH",
            "runtime": {"elapsed_seconds": 1.0},
            **result,
        }
    )
    assert "not24 examples" in text and "Primary original edits 8/8" in text
    width = None
    for line in text.splitlines():
        if not line.startswith("|"):
            width = None
        elif width is None:
            width = line.count("|")
        else:
            assert width == line.count("|")


# Reuse only safety contracts unaffected by prompt counts or layout.
inherited = protocol.isolate(
    "_original8_inherited_safety_tests", "tests/test_frozen_guarded_preserve_crossed.py"
)
inherited.job, inherited.audit, inherited.protocol = job, audit, protocol
inherited.prepare = prepare
for _name in (
    "test_candidate_authentication_before_model_load",
    "test_usage_preflight_blocks_missing_capped_or_stale",
    "test_one_prelaunch_no_resume_and_no_worker_on_git_failure",
):
    globals()[_name] = getattr(inherited, _name)


def test_external_timeout24_no_retry(tmp_path, monkeypatch):
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

    fake = Process()
    monkeypatch.setattr(job.subprocess, "Popen", lambda *a, **k: fake)
    result = job.supervise(["fake"], tmp_path, {"standard_used_percent": 32}, timeout=0.01)
    assert fake.killed and result["status"] == "INCONCLUSIVE" and not result["retries_allowed"]
    assert protocol.read(tmp_path / "RUN_STARTED.json")["forward_ceiling"] == 24


def test_independent_raw_adapter_only_counts_and_archived_goal_lookup():
    tree = ast.parse(inspect.getsource(parent_audit.verify_data))

    class Expected(ast.NodeTransformer):
        def visit_Assign(self, node):
            if isinstance(node.targets[0], ast.Name) and node.targets[0].id == "expected_goals":
                node.value = ast.parse("archived_goals(plan, rows)", mode="eval").body
                return node
            return self.generic_visit(node)

        def visit_Constant(self, node):
            if type(node.value) is int:
                node.value = {12: 24, 4: 8, 7: 15, 8: 16}.get(node.value, node.value)
            elif type(node.value) is str:
                node.value = ""
            return node

    class Strings(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is str:
                node.value = ""
            return node

    expected = Expected().visit(tree)
    actual = Strings().visit(ast.parse(verify_entry.CORE_VERIFY_DATA.adapted_source))
    assert ast.dump(expected) == ast.dump(actual)


def test_unexpected_adapter_sites_rejected_before_definition_change():
    module = protocol.isolate(
        "_original8_bad_count_test", "scripts/frozen_guarded_preserve_crossed.py"
    )
    saved = module.evaluate
    with pytest.raises(ValueError, match="count adaptation sites"):
        protocol.adapt_function(module, "evaluate", {12: 24}, {12: 2})
    assert module.evaluate is saved


def test_imported_fixed_condition_sources_are_bound():
    plan = protocol.build_plan()
    for path in (
        "scripts/frozen_endpoint020_crossed_f04_plan.py",
        "scripts/verify_frozen_endpoint020_crossed_f04.py",
    ):
        assert plan["input_sha256"][path] == protocol.sha((protocol.ROOT / path).read_bytes())


def test_independent_original_archive_gate_before_loading(monkeypatch):
    plan = protocol.build_plan()
    monkeypatch.setattr(entry, "CORE_FREEZE_CHECK", lambda: {"plan": plan})
    calls = []
    monkeypatch.setattr(verify_entry, "independent_condition", lambda p: calls.append("condition"))

    def reject(p):
        calls.append("original_archive")
        raise ValueError("original archive rejected before load")

    monkeypatch.setattr(verify_entry, "verify_original_inputs", reject)
    with pytest.raises(ValueError, match="original archive rejected"):
        job.require_freeze()
    assert calls == ["condition", "original_archive"]
