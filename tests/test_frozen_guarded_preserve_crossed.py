"""Focused crossed-rendering, frozen-candidate, physical and coverage contracts."""

from __future__ import annotations

import ast
import copy
import inspect
import json
import textwrap
import time
from types import SimpleNamespace

import pytest
import torch
from test_local_controllability_positive_control import Model, setup

from scripts import crossed_pair_probe as parent
from scripts import frozen_guarded_preserve_crossed as job
from scripts import frozen_guarded_preserve_crossed_plan as protocol
from scripts import verify_crossed_pair_probe as parent_audit
from scripts import verify_frozen_guarded_preserve_crossed as audit


def prepare(mode="linear", offset=0.1, quality_failure=False):
    plan, backend = setup(protocol.build_plan())
    plan["model"]["d_model"] = 3
    vectors = {
        "preserve": [0.10764565083962835, 0.0, 0.0],
    }
    for target, vector in vectors.items():
        plan["candidates"][target]["norm"] = protocol.norm(vector)
        plan["candidates"][target]["vector_float64_le_sha256"] = protocol.vector_sha(vector)

    class SemanticModel(Model):
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
            baseline = -offset if p["display_order"] == "A_then_B" else offset
            if mode == "all_B":
                baseline = -offset if p["preserve_label"] == "A" else offset
            speed = 0.03 if mode == "weak" else 0.25 if mode == "weak_margin" else 1.0
            S = baseline + speed * x
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
            if mode == "replay_mismatch" and len(self.calls) == 9:
                out[..., 0] += 0.01
            if hasattr(self, "ledger"):
                assert self.ledger.attempts == len(self.calls) and self.ledger.pending
            return out

    model = SemanticModel(plan["prompts"])
    model.by_token = {model.tokenizer.ids[p["prompt"]]: p for p in plan["prompts"]}
    backend.model = model
    return plan, backend, vectors


def execute(output, **kwargs):
    plan, backend, vectors = prepare(**kwargs)
    f = job.base.Ledger(output, plan["cells"], time.monotonic() + 100)
    backend.model.ledger = f
    rows = job.evaluate(plan, backend, vectors, f, output)
    verified = audit.verify_data(plan, rows, vectors, output)
    audit.compare_summary(job.summarize(rows), verified["summary"])
    return plan, backend, vectors, f, rows, verified


def test_exact_truth_table_canonical_bytes_and_BA_only_permutation():
    plan = protocol.build_plan()
    prompts = plan["prompts"]
    cfg = plan["config"]
    canonical = protocol.canonical.select_inputs(
        protocol.read(protocol.ROOT / cfg["dataset"]["path"]),
        protocol.read(protocol.ROOT / cfg["manifest"]["path"]),
        cfg["selection"],
    )
    assert [(p["preserve_label"], p["comply_label"], p["display_order"]) for p in prompts] == [
        ("A", "B", "A_then_B"),
        ("A", "B", "B_then_A"),
        ("B", "A", "A_then_B"),
        ("B", "A", "B_then_A"),
    ]
    assert prompts[0]["prompt"] == canonical[0]["prompt"]
    assert prompts[2]["prompt"] == canonical[1]["prompt"]
    for start in (0, 2):
        ab, ba = prompts[start : start + 2]
        lines = ab["prompt"].splitlines(keepends=True)
        ai = next(i for i, line in enumerate(lines) if line.startswith("A) "))
        bi = next(i for i, line in enumerate(lines) if line.startswith("B) "))
        expected = lines[:]
        expected[ai], expected[bi] = lines[bi], lines[ai]
        assert ba["prompt"] == "".join(expected)
        assert ab["option_line_by_letter"] == ba["option_line_by_letter"]
        assert ab["semantic_to_letter"] == ba["semantic_to_letter"]
        assert ab["display_position_to_letter"] == {"first": "A", "second": "B"}
        assert ba["display_position_to_letter"] == {"first": "B", "second": "A"}
        assert ab["canonical_prompt_sha256"] == ba["canonical_prompt_sha256"]
    audit.verify_renderings(plan)
    assert len(plan["prompts"]) == 4 and len(plan["cells"]) == 12 and not plan["derivative_cells"]
    assert [c["phase"] for c in plan["cells"]] == ["baseline"] * 4 + ["edit"] * 4 + ["replay"] * 4
    assert [c["requested"] for c in plan["cells"][4:]] == ["preserve"] * 8
    assert [c["replay_of"] for c in plan["cells"][8:]] == [c["cell_id"] for c in plan["cells"][4:8]]
    assert len({p["prompt_id"] for p in prompts}) == 4


def test_candidate_chain_same_eight_fit_ids_disjoint_case_and_exact_guarded_arrow():
    plan = protocol.build_plan()
    saved = protocol.candidates()
    fitted = None
    for target, meta in plan["candidates"].items():
        assert meta["audit_before_freeze_verified"] and meta["selected_case_not_fitted"]
        assert len(meta["fitted_prompt_ids"]) == 8
        assert not any("cg_f03" in x for x in meta["fitted_prompt_ids"])
        assert protocol.vector_sha(saved[target]["vector"]) == meta["vector_float64_le_sha256"]
        assert protocol.norm(saved[target]["vector"]) == meta["norm"]
        assert fitted is None or fitted == meta["fitted_prompt_ids"]
        fitted = meta["fitted_prompt_ids"]
    assert set(plan["candidates"]) == {"preserve"}
    assert plan["candidates"]["preserve"]["norm"] == 0.10764565083962835
    assert plan["candidates"]["preserve"]["guarded_training_audit_verified"]


def test_own_rendering_states_labels_not_display_positions_and_full_coverage(tmp_path):
    plan, backend, vectors, f, rows, verified = execute(tmp_path)
    assert f.attempts == f.completed == len(backend.model.calls) == 12
    assert (tmp_path / "derivative_events.jsonl").read_text() == ""
    assert [r["requested_label"] for r in rows[4:8]] == ["A", "A", "B", "B"]
    assert len({tuple(r["h0"]) for r in rows[:4]}) == 4
    baselines = {r["prompt_id"]: r for r in rows[:4]}
    for r in rows:
        b = baselines[r["prompt_id"]]
        assert r["h0"] == b["h0"] == b["h"]
        assert r["h0_norm"] == b["h0_norm"]
        assert r["unselected_max_difference"] == 0 and r["weights_unchanged"]
        assert r["letter_log_odds"] == (
            r["preserve_log_odds"] if r["preserve_label"] == "A" else -r["preserve_log_odds"]
        )
        assert r["delta_letter_log_odds"] == r["letter_log_odds"] - b["letter_log_odds"]
        assert r["signed_delta_log_odds"] == r["target_sign"] * r["delta_log_odds"]
        if r["requested"]:
            assert r["intended_delta"] == [
                audit.f32(b["h0_norm"] * x) for x in vectors[r["requested"]]
            ]
    assert all(a["h"] == b["h"] for a, b in zip(rows[4:8], rows[8:], strict=True))
    s = verified["summary"]
    assert s["matrix"]["strict_accepted"] == 4 and s["matrix"]["matrix_pass"]
    assert s["baseline_availability"] == {"A": 2, "B": 2, "OTHER": 0}
    assert s["matrix"]["eligible_A_to_B"] == s["matrix"]["achieved_A_to_B"] == 1
    assert s["matrix"]["eligible_B_to_A"] == s["matrix"]["achieved_B_to_A"] == 1
    assert s["matrix"]["accepted_flips"] == s["matrix"]["accepted_retentions"] == 2
    for c in s["per_vector"].values():
        assert c["eligible_A_to_B"] == c["achieved_A_to_B"] == 1
        assert c["eligible_B_to_A"] == c["achieved_B_to_A"] == 1
    assert (
        len(s["coverage_by_vector_mapping_display"]) == 4 and len(s["descriptive_contrasts"]) == 4
    )
    assert s["replay_matches"] == 4 and not s["replays_are_new_examples"]
    assert (
        backend.model.active_hooks == []
        and backend.model.weight.requires_grad
        and backend.model.weight.grad is None
    )
    with pytest.raises(ValueError):
        f.begin(plan["cells"][-1])


def test_missing_directional_coverage_does_not_invalidate_individual_passes(tmp_path):
    *_, verified = execute(tmp_path, mode="all_B")
    s = verified["summary"]
    assert s["matrix"]["matrix_pass"] and s["matrix"]["strict_accepted"] == 4
    assert s["baseline_availability"] == {"A": 0, "B": 4, "OTHER": 0}
    assert s["matrix"]["eligible_A_to_B"] == s["matrix"]["achieved_A_to_B"] == 0
    assert (
        not s["matrix"]["eligible_both_directions"] and not s["matrix"]["achieved_both_directions"]
    )


@pytest.mark.parametrize(
    "kwargs", [{"quality_failure": True}, {"mode": "weak"}, {"mode": "weak_margin"}]
)
def test_finite_scientific_failures_complete_all12_no_gate_changes(tmp_path, kwargs):
    *_, f, rows, verified = execute(tmp_path, **kwargs)
    assert f.attempts == f.completed == 12 and len(rows) == 12
    assert verified["summary"]["matrix"]["strict_accepted"] < 4
    assert not verified["summary"]["matrix"]["matrix_pass"]
    assert all(r["replay_consistent"] for r in rows[8:])


@pytest.mark.parametrize(
    "kwargs,attempts",
    [
        ({"offset": 0.01}, 1),
        ({"mode": "nonfinite"}, 1),
        ({"mode": "gradient"}, 1),
        ({"mode": "weight_change"}, 1),
        ({"mode": "replay_mismatch"}, 9),
    ],
)
def test_technical_faults_abort_without_padding_or_retry(tmp_path, kwargs, attempts):
    plan, backend, vectors = prepare(**kwargs)
    f = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    grad, back = torch.autograd.grad, torch.autograd.backward
    with pytest.raises(ValueError):
        job.evaluate(plan, backend, vectors, f, tmp_path)
    assert f.attempts == attempts
    assert torch.autograd.grad is grad and torch.autograd.backward is back
    assert backend.model.weight.requires_grad


@pytest.mark.parametrize("fault", ["scale", "sign", "extra", "rounded"])
def test_vector_coordinate_changes_rejected_before_any_forward(tmp_path, fault):
    plan, backend, vectors = prepare()
    if fault == "scale":
        vectors["preserve"] = [x * 0.05 for x in vectors["preserve"]]
    elif fault == "sign":
        vectors["preserve"] = [-x for x in vectors["preserve"]]
    elif fault == "extra":
        vectors["comply"] = list(vectors["preserve"])
    else:
        vectors["preserve"][0] = 0.1076
    f = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    with pytest.raises(ValueError):
        job.evaluate(plan, backend, vectors, f, tmp_path)
    assert f.attempts == 0 and not backend.model.calls


@pytest.mark.parametrize("fault", ["semantic", "display", "action", "suffix", "order"])
def test_independent_rendering_corruption_rejection(fault):
    plan = copy.deepcopy(protocol.build_plan())
    p = plan["prompts"][1]
    if fault == "semantic":
        p["semantic_to_letter"]["preserve"] = "B"
    elif fault == "display":
        p["display_position_to_letter"] = {"first": "A", "second": "B"}
    elif fault == "action":
        p["option_line_by_letter"]["A"] = "A) different\n"
    elif fault == "suffix":
        p["prompt"] += " "
    else:
        plan["prompts"][1], plan["prompts"][2] = plan["prompts"][2], plan["prompts"][1]
    with pytest.raises(ValueError):
        audit.verify_renderings(plan)


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
    ],
)
def test_independent_raw_and_coverage_corruption_rejection(tmp_path, fault):
    plan, _, vectors, _, rows, verified = execute(tmp_path)
    changed = copy.deepcopy(rows)
    r = changed[4]
    if fault == "label":
        r["requested_label"] = "B"
    elif fault == "candidate":
        r["candidate_vector_sha256"] = "wrong"
    elif fault == "offset_hash":
        r["offset_float32_le_sha256"] = "wrong"
    elif fault == "cast":
        r["intended_delta"][0] += 0.01
    elif fault == "raw_L":
        r["letter_log_odds"] += 1e-8
    elif fault == "delta_L":
        r["delta_letter_log_odds"] += 1e-8
    elif fault == "signed_delta":
        r["signed_delta_log_odds"] += 1e-8
    elif fault == "replay":
        changed[8]["maximum_replay_h_difference"] += 0.001
    else:
        summary = copy.deepcopy(verified["summary"])
        summary["matrix"]["achieved_A_to_B"] += 1
        with pytest.raises(ValueError):
            audit.compare_summary(summary, verified["summary"])
        return
    with pytest.raises(ValueError):
        audit.verify_data(plan, changed, vectors, tmp_path)


@pytest.mark.parametrize("target", ["preserve"])
@pytest.mark.parametrize("fault", ["fitted_ids", "freeze", "audit", "guarded_audit"])
def test_candidate_audit_chain_corruption(monkeypatch, target, fault):
    cfg = protocol.read(protocol.ROOT / protocol.CONFIG)
    prompts = protocol.build_plan()["prompts"]
    spec = cfg["candidates"][target]
    original = protocol.authenticated

    def changed(path, digest, root=protocol.ROOT):
        value = copy.deepcopy(original(path, digest, root))
        if path == spec["construction_lock"] and fault == "fitted_ids":
            value["plan"]["construction_ids"][0] = prompts[0]["canonical_prompt_id"]
        elif path == spec["candidate_freeze"] and fault == "freeze":
            value["after_independent_audit"] = False
        elif path == spec["verification"] and fault == "audit":
            value["status"] = "INCONCLUSIVE"
        elif path == spec["verification"] and fault == "guarded_audit":
            value["summary"]["final_guarded_goals"] = 7
        return value

    monkeypatch.setattr(protocol, "authenticated", changed)
    with pytest.raises(ValueError):
        protocol.bind_candidates(cfg, prompts)


def test_storage_guard_and_no_historical_f03_outcome_reads(monkeypatch):
    cfg = protocol.read(protocol.ROOT / protocol.CONFIG)
    monkeypatch.setattr(
        protocol.shutil, "disk_usage", lambda root: SimpleNamespace(free=64 * 1024**2)
    )
    assert protocol.storage_preflight(protocol.ROOT, cfg)["bounds"]["total_bound_bytes"] == 41283268
    monkeypatch.setattr(protocol.shutil, "disk_usage", lambda root: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="storage"):
        protocol.storage_preflight(protocol.ROOT, cfg)
    original = protocol.Path.read_bytes

    def no_old_f03(path):
        assert "evidence/frozen_preserve_f03_v1" not in str(path).replace("\\", "/")
        return original(path)

    monkeypatch.setattr(protocol.Path, "read_bytes", no_old_f03)
    assert len(protocol.build_plan()["cells"]) == 12


def source(fn):
    return textwrap.dedent(inspect.getsource(fn))


def same(a, b):
    assert ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))


def test_unchanged_physical_scoring_and_audit_except_scope_and_auxiliary_record():
    same(source(parent.make_delta), source(job.make_delta))
    same(source(parent.assess), source(job.assess))
    assert job.DerivativeGuard is parent.DerivativeGuard and job.recorder is parent.recorder
    assert job.score_float32_logits is parent.score_float32_logits

    class StripAuxiliary(ast.NodeTransformer):
        def visit_Assign(self, node):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if names & {"goals", "goals_record", "expected_goals", "auxiliary"}:
                return None
            if names == {"events"}:
                return ast.Expr(value=node.value)
            return self.generic_visit(node)

        def visit_If(self, node):
            if "record_fresh_goals(" in ast.unparse(node.test) + "".join(
                ast.unparse(x) for x in node.body
            ):
                return None
            return self.generic_visit(node)

        def visit_Expr(self, node):
            text = ast.unparse(node)
            if "row.update(diagnostic_fields(" in text or (
                "require(" in text
                and (
                    "durable fresh goals after all baselines" in text
                    or "exact auxiliary goals/retention/slack" in text
                )
            ):
                return None
            return self.generic_visit(node)

        def visit_Dict(self, node):
            pairs = [
                (k, v)
                for k, v in zip(node.keys, node.values, strict=True)
                if not (k is None and isinstance(v, ast.Name) and v.id == "auxiliary")
            ]
            node.keys, node.values = [p[0] for p in pairs], [p[1] for p in pairs]
            return self.generic_visit(node)

    expected = (
        source(parent.evaluate)
        .replace("/20 forwards", "/12 forwards")
        .replace("== 20", "== 12")
        .replace("20/0 accounting", "12/0 accounting")
    )
    assert ast.dump(ast.parse(expected)) == ast.dump(
        StripAuxiliary().visit(ast.parse(source(job.evaluate)))
    )
    expected = (
        source(parent_audit.verify_data)
        .replace("== 20", "== 12")
        .replace("exact20-cell", "exact12-cell")
        .replace("two candidate bindings", "one guarded-P candidate binding")
        .replace('{"preserve", "comply"}', '{"preserve"}')
    )
    assert ast.dump(ast.parse(expected)) == ast.dump(
        StripAuxiliary().visit(ast.parse(source(audit.verify_data)))
    )
    same(source(parent_audit.verify_renderings), source(audit.verify_renderings))
    assert protocol.render is protocol.crossed.render
    assert source(job.worker).index("storage_preflight") < source(job.worker).index("load_backend")


def test_report_tables_and_contrast_arithmetic(tmp_path):
    *_, verified = execute(tmp_path)
    result = {
        "status": "INDEPENDENT_NUMERIC_GEOMETRY_REPLAY_SCHEDULE_MATCH",
        "runtime": {"elapsed_seconds": 1.0},
        **verified,
    }
    report = audit.report(result)
    width = None
    for line in report.splitlines():
        if not line.startswith("|"):
            width = None
            continue
        count = line.count("|")
        if width is None:
            width = count
        assert count == width
    indexed = {r["cell_id"]: r for r in verified["summary"]["cells"]}
    for c in verified["summary"]["descriptive_contrasts"]:
        left, right = indexed[c["left_cell_id"]], indexed[c["right_cell_id"]]
        assert all(value == right[k] - left[k] for k, value in c["right_minus_left"].items())


def test_candidate_authentication_before_model_load(tmp_path, monkeypatch):
    cfg = copy.deepcopy(protocol.read(protocol.ROOT / protocol.CONFIG))
    bad = tmp_path / "changed_candidate.json"
    bad.write_text("{}")
    cfg["candidates"]["preserve"]["path"] = str(bad)
    with pytest.raises(ValueError, match="authenticated"):
        protocol.candidates(cfg)
    output = tmp_path / "job"
    output.mkdir()
    job.base.write_new(output / "RUN_STARTED.json", {"deadline_monotonic": time.monotonic() + 100})
    plan, _, _ = prepare()
    monkeypatch.setattr(job, "no_resume", lambda allowed: None)
    monkeypatch.setattr(protocol, "check_prelaunch", lambda *args, **kwargs: {"status": "passed"})
    monkeypatch.setattr(job, "ROOT", tmp_path)
    monkeypatch.setattr(job, "OUTPUT", "job")
    monkeypatch.setattr(job, "require_freeze", lambda: {"plan": plan})
    monkeypatch.setattr(protocol, "candidates", lambda: protocol.authenticated(str(bad), "wrong"))

    def forbidden(*args):
        raise AssertionError("model load before candidate authentication")

    monkeypatch.setattr(job.base, "load_backend", forbidden)
    with pytest.raises(ValueError, match="authenticated"):
        job.worker()
    assert (output / "INVALID.json").exists()


def test_whole_job_timeout_no_retry(tmp_path, monkeypatch):
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
    result = job.supervise(["fake"], tmp_path, {"standard_used_percent": 26}, timeout=0.01)
    assert fake.killed and result["status"] == "INCONCLUSIVE" and not result["retries_allowed"]
    assert protocol.read(tmp_path / "RUN_STARTED.json")["forward_ceiling"] == 12


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {"standard_used_percent": 90, "checked_at_unix": 0},
        {"standard_used_percent": 26, "checked_at_unix": 0},
    ],
)
def test_usage_preflight_blocks_missing_capped_or_stale(monkeypatch, usage):
    monkeypatch.setattr(job, "no_resume", lambda allowed: None)
    monkeypatch.setattr(protocol, "check_prelaunch", lambda *args, **kwargs: {"status": "passed"})
    monkeypatch.setattr(job, "require_freeze", lambda: {"source_commit": "source"})

    def git(*args):
        if "show" in args:
            return protocol.OUTPUT + "/preregistration.json"
        if "status" in args:
            return ""
        return "source"

    monkeypatch.setattr(job.base, "git", git)
    monkeypatch.setenv("SP_LENSE_USAGE_PREFLIGHT", json.dumps(usage))
    with pytest.raises(ValueError, match="fresh usage"):
        job.run()


def test_primary_four_passes_even_when_auxiliary_goals_fail(tmp_path):
    *_, verified = execute(tmp_path, mode="weak_margin", offset=0.075)
    summary = verified["summary"]
    assert summary["matrix"]["strict_accepted"] == 4 and summary["matrix"]["matrix_pass"]
    assert summary["diagnostic_goals_met"] < 4 and summary["auxiliary_not_primary"]
    assert summary["replay_matches"] == 4


def test_goal_lock_written_after_all_baselines_before_first_edit(tmp_path, monkeypatch):
    plan, backend, vectors = prepare()
    original = backend.model.forward

    def checked(tokens):
        count = len(backend.model.calls)
        assert (tmp_path / "baseline_goals.json").exists() == (count >= 4)
        return original(tokens)

    monkeypatch.setattr(backend.model, "forward", checked)
    ledger = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    rows = job.evaluate(plan, backend, vectors, ledger, tmp_path)
    goals = protocol.read(tmp_path / "baseline_goals.json")
    assert goals["completed_baselines"] == 4 and goals["derivative_attempts"] == 0
    assert goals["baseline_rows_sha256"] == protocol.canonical_sha(rows[:4])
    for row in rows[:4]:
        value = goals["goals"][row["prompt_id"]]
        assert value["diagnostic_goal"] == max(0.10, row["preserve_log_odds"])
        assert value["baseline_retention"] == (
            row["actual_next_token_label"] == row["preserve_label"]
        )
        assert row["diagnostic_goal"] is None
    audit.verify_data(plan, rows, vectors, tmp_path)


@pytest.mark.parametrize("fault", ["too_early", "too_late", "S0", "membership", "row_hash"])
def test_independent_goal_lock_corruption_rejected(tmp_path, fault):
    plan, _, vectors, _, rows, _ = execute(tmp_path)
    record = protocol.read(tmp_path / "baseline_goals.json")
    if fault == "too_early":
        record["monotonic"] = 0
    elif fault == "too_late":
        record["monotonic"] += 10000
    elif fault == "S0":
        record["goals"][rows[0]["prompt_id"]]["S0"] += 0.01
    elif fault == "membership":
        record["goals"][rows[0]["prompt_id"]]["baseline_retention"] ^= True
    else:
        record["baseline_rows_sha256"] = "wrong"
    (tmp_path / "baseline_goals.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="durable fresh goals"):
        audit.verify_data(plan, rows, vectors, tmp_path)


@pytest.mark.parametrize(
    "field", ["retention_slack", "goal_slack", "diagnostic_goal_met", "retention_with_quality"]
)
def test_independent_auxiliary_fields_corruption_rejected(tmp_path, field):
    plan, _, vectors, _, rows, _ = execute(tmp_path)
    r = rows[5]
    r[field] = not r[field] if type(r[field]) is bool else r[field] + 0.01
    with pytest.raises(ValueError, match="auxiliary"):
        audit.verify_data(plan, rows, vectors, tmp_path)


def test_one_prelaunch_no_resume_and_no_worker_on_git_failure(tmp_path, monkeypatch):
    job.base.write_new(
        tmp_path / "preregistration.json", {"source_commit": "f" * 40, "environment": {}}
    )
    monkeypatch.setattr(job, "OUTPUT", str(tmp_path))
    calls = []

    def git(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=128, stdout="", stderr="synthetic denied")

    monkeypatch.setattr(job.subprocess, "run", git)
    result = job.prelaunch()
    assert result["status"] == "INCONCLUSIVE" and len(calls) == 1
    assert result["git_checks"][0]["stderr"] == "synthetic denied"
    with pytest.raises(ValueError, match="prelaunch failed"):
        job.run()
    with pytest.raises(ValueError, match="no resume"):
        job.prelaunch()
    assert not (tmp_path / "WORKER_CLAIM.json").exists()
    assert not (tmp_path / "RUN_STARTED.json").exists()
