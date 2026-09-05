"""Focused eight-row solver, fake-model execution and independent corruption tests."""

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
from test_local_controllability_positive_control import setup
from test_projected_shared_preserve import Toy

from scripts import iterative_guarded_preserve as job
from scripts import iterative_guarded_preserve_plan as protocol
from scripts import shared_preserve_eight_row_solver as solver
from scripts import shared_preserve_two_family as parent
from scripts import verify_iterative_guarded_preserve as audit
from scripts import verify_shared_preserve_two_family as parent_audit

TOL = {
    "rank_relative_pivot_floor": 1e-12,
    "primal_absolute_tolerance": 1e-9,
    "kkt_absolute_tolerance": 1e-8,
}


def prepare(**kwargs):
    plan, backend = setup(protocol.build_plan())
    plan["model"]["d_model"] = 3
    kwargs.setdefault("positive", 0.1)
    for p in plan["prompts"]:
        s0 = audit.f32(-kwargs["positive"] if p["preserve_label"] == "A" else 0.2)
        retention = p["preserve_label"] == "B"
        p.update(archived_S0=s0, guarded_goal=max(0.10, s0), archived_retention=retention)
        plan["archived_baselines"][p["prompt_id"]] = {
            "S0": s0,
            "h0": [0.0, 3.0, 4.0],
            "h0_norm": 5.0,
            "actual_next_token_id": 1,
            "actual_next_token_label": "B",
            "guarded_goal": max(0.10, s0),
            "archived_retention": retention,
        }
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
    f = job.ForwardLedger(output, plan["cells"], deadline)
    d = job.Derivatives(torch, output, plan["derivative_cells"], deadline)
    rows, summary = job.evaluate(plan, backend, f, d, output, propose=propose)
    verified = audit.verify_data(plan, rows, output) if check else None
    if check:
        audit.compare_summary(summary, verified["summary"])
    return plan, backend, f, d, rows, summary, verified


def test_exact_selection_and_schedule(tmp_path):
    plan = protocol.build_plan()
    assert len(plan["prompts"]) == 8 and len(plan["cells"]) == 144
    assert len(plan["derivative_cells"]) == 64 and sum(c["optional"] for c in plan["cells"]) == 128
    assert not plan["control_ids"] and not plan["transfer_ids"]
    assert all(c["condition"] == "final" for c in plan["cells"][136:])
    assert [p["family_id"] for p in plan["prompts"]] == plan["config"]["training_families"][
        :1
    ] * 4 + plan["config"]["training_families"][1:] * 4
    assert "reserved_unrun_prompt_ids" not in plan
    assert not any("cg_f03" in p["prompt_id"] for p in plan["prompts"])
    assert job.base.MAX_ATTEMPTS == 48
    f = job.ForwardLedger(tmp_path, plan["cells"], 100, now=lambda: 0)
    for cell in plan["cells"]:
        f.begin(cell)
        f.finish(True)
    assert f.attempts == f.completed == 144
    with pytest.raises(ValueError):
        f.begin(plan["cells"][-1])


def test_plan_original_only_goals_and_fresh_zero():
    plan = protocol.build_plan()
    assert plan["initial_shared_w"] == [0.0] * 1024
    assert [p["prompt_id"] for p in plan["prompts"]] == plan["config"]["selected_prompt_ids"]
    assert sum(p["archived_retention"] for p in plan["prompts"]) == 4
    for p in plan["prompts"]:
        archived = plan["archived_baselines"][p["prompt_id"]]
        assert p["guarded_goal"] == archived["guarded_goal"] == max(0.10, archived["S0"])
        assert p["archived_S0"] == archived["S0"]
        assert "g" not in archived and "gradient" not in archived
    src = source(protocol.build_plan)
    assert 'originals["rows.jsonl"].splitlines()[:16]' in src
    assert 'original["prompts"]' in src
    assert "select_inputs" not in src
    assert (
        "endpoint.json" not in src and "analysis.json" not in src and "preserve_vector" not in src
    )
    assert not any("guarded_preserve_application" in p for p in plan["input_sha256"])


def test_archive_hash_mismatch_blocks_plan(monkeypatch):
    original = protocol.Path.read_bytes

    def corrupt(path):
        value = original(path)
        if path.as_posix().endswith(
            "retention_guard_linear_feasibility_qwen35_08b/preregistration.json"
        ):
            return value + b" "
        return value

    monkeypatch.setattr(protocol.Path, "read_bytes", corrupt)
    with pytest.raises(ValueError, match="archive hash"):
        protocol.build_plan()


def test_storage_arithmetic_and_preload_failure(monkeypatch):
    cfg = protocol.read(protocol.ROOT / protocol.CONFIG)
    monkeypatch.setattr(
        protocol.shutil, "disk_usage", lambda root: SimpleNamespace(free=512 * 1024**2)
    )
    result = protocol.storage_preflight(protocol.ROOT, cfg)
    assert result["bounds"]["total_bound_bytes"] == 377958704
    monkeypatch.setattr(
        protocol.shutil, "disk_usage", lambda root: SimpleNamespace(free=512 * 1024**2 - 1)
    )
    with pytest.raises(ValueError, match="storage cannot fit"):
        protocol.storage_preflight(protocol.ROOT, cfg)


def test_genuine_eight_row_known_solution_256_masks_and_negative_rhs():
    A = [[float(i == j) for j in range(8)] for i in range(8)]
    b = [0.125 * (i + 1) for i in range(8)]
    result = solver.solve(A, b, TOL)
    assert result["solution"]["vector"] == b and result["solution"]["active_mask"] == 255
    assert [x["mask"] for x in result["active_sets"]] == list(range(256))
    b[3] = -2.0
    result = solver.solve(A, b, TOL)
    assert result["solution"]["vector"][3] == 0.0
    assert result["solution"]["vector"][:3] == b[:3]
    assert b[3] == -2.0


def test_eight_rank_deficient_strict_ties_and_contradiction():
    A, b = [[1.0] for _ in range(8)], [0.2] * 8
    result = solver.solve(A, b, TOL)
    assert result["solution"]["active_mask"] == 1
    assert result["active_sets"][3]["status"] == "rank_deficient_or_near_dependent_skipped"
    assert sum(x["status"] == "kkt_valid" for x in result["active_sets"]) == 8
    A[1] = [-1.0]
    assert solver.solve(A, b, TOL)["status"] == "NUMERICALLY_UNRESOLVED"
    with pytest.raises(ValueError):
        solver.solve([[1.0]] * 9, [1.0] * 9, TOL)


def test_first_acceptance_common_semantics_and_no_worker_candidate(tmp_path):
    _, backend, f, d, rows, s, verified = execute(tmp_path)
    assert s["final_accepted"] == 8 and s["updates"] == s["attempted_updates"] == 1
    assert s["accepted_flips"] == s["accepted_retentions"] == 4
    assert s["actual_B_to_A"] == 4 and s["actual_A_to_B"] == 0
    assert s["candidate_eligible"] and not s["off_identities"] and not s["transfer_ran"]
    assert f.attempts == f.completed == len(backend.model.calls) == 32
    assert d.attempts == d.completed == 8 and len(f.skips) == 112
    assert verified["maximum_nonfinal_difference"] == 0
    assert len({r["shared_w_sha256"] for r in rows if r["condition"] in ("step_1", "final")}) == 1
    for r in rows:
        assert r["target_sign"] == 1 and r["requested"] == "preserve"
        assert r["requested_token_id"] == (0 if r["preserve_label"] == "A" else 1)
        if r["gradient"] is not None:
            assert r["gradient"] == [1.0, 0.0, 0.0]
    assert not (tmp_path / "preserve_vector.json").exists()
    assert not (tmp_path / "candidate_freeze.json").exists()
    assert backend.model.active_hooks == [] and backend.model.weight.requires_grad
    assert backend.model.weight.grad is None and backend.model.weight.item() == 1.0


def test_four_update_original_norm_cast_and_physical_bounds(tmp_path):
    _, _, f, d, rows, s, _ = execute(tmp_path, positive=0.9)
    assert s["updates"] == 4 and s["final_accepted"] == 8
    assert f.attempts == 80 and d.attempts == 32 and len(f.skips) == 64
    for r in rows:
        if r["condition"].startswith("step_"):
            assert r["intended_delta"] == [audit.f32(r["h0_norm"] * x) for x in r["shared_w"]]
            assert r["h0_norm"] == 5.0 and r["unselected_max_difference"] == 0
            assert r["net_norm"] <= 0.20 * r["h0_norm"] + 1e-6
            assert r["path_norm"] <= 0.40 * r["h0_norm"] + 1e-6
            assert r["step_norm"] <= 0.05 * r["h0_norm"] + 1e-6


def test_stall_counts_attempt_keeps_last_endpoint(tmp_path):
    _, _, f, d, _, s, _ = execute(tmp_path, positive=1.1)
    assert s["attempted_updates"] == 5 and s["updates"] == 4
    assert s["stop_reason"] == "projection_stall" and s["final_accepted"] == 4
    assert f.attempts == 88 and d.attempts == 40 and len(f.skips) == 56
    assert not s["candidate_eligible"]


def test_quality_failure_completes_eight_group_then_finals(tmp_path):
    _, _, f, d, rows, s, _ = execute(tmp_path, quality_failure=True)
    assert f.attempts == 32 and d.attempts == 8 and s["updates"] == 1
    assert s["stop_reason"] == "quality_failure" and s["final_accepted"] == 0
    assert s["final_other_token_outcomes"] == 8 and s["off_identities"] == 0
    assert all(r["shared_w"] != [0.0] * 3 for r in rows if r["condition"] == "final")


@pytest.mark.parametrize(
    "kwargs,forwards,derivatives,match",
    [
        ({"zero": True}, 16, 8, "numerical solver failure"),
        ({"mismatch": True}, 9, 1, "current-state"),
        ({"positive": 0.01}, 1, 0, "baseline"),
    ],
)
def test_technical_stops_no_padding(tmp_path, kwargs, forwards, derivatives, match):
    plan, backend = prepare(**kwargs)
    f = job.ForwardLedger(tmp_path, plan["cells"], time.monotonic() + 100)
    d = job.Derivatives(torch, tmp_path, plan["derivative_cells"], time.monotonic() + 100)
    original = torch.autograd.grad
    with pytest.raises(ValueError, match=match):
        job.evaluate(plan, backend, f, d, tmp_path)
    assert f.attempts == forwards and d.attempts == derivatives
    assert torch.autograd.grad is original and backend.model.weight.requires_grad
    assert not (tmp_path / "preserve_vector.json").exists()


def test_final_disagreement_precludes_candidate(tmp_path, monkeypatch):
    plan, backend = prepare()
    original = backend.model.forward

    def corrupt(tokens):
        logits = original(tokens)
        if len(backend.model.calls) == 25:
            logits = logits.clone()
            logits[..., 0] += 0.01
        return logits

    monkeypatch.setattr(backend.model, "forward", corrupt)
    f = job.ForwardLedger(tmp_path, plan["cells"], time.monotonic() + 100)
    d = job.Derivatives(torch, tmp_path, plan["derivative_cells"], time.monotonic() + 100)
    with pytest.raises(ValueError, match="current-state/independent identity"):
        job.evaluate(plan, backend, f, d, tmp_path)
    assert f.attempts == 25 and d.attempts == 8
    assert (tmp_path / "endpoint.json").exists() and not (
        tmp_path / "preserve_vector.json"
    ).exists()


@pytest.mark.parametrize(
    "mode", ["full_success", "max_updates", "baseline_success", "gradient_quality", "zero"]
)
def test_conditional_driver_exact_groups_and_stops(tmp_path, mode):
    plan = protocol.build_plan()
    plan["model"]["d_model"] = 2
    f = job.ForwardLedger(tmp_path, plan["cells"], time.monotonic() + 100)
    seen, updates, endpoints = [], [], []

    def call(cell, w, current):
        f.begin(cell)
        f.finish(True)
        seen.append((cell, list(w)))
        success = mode == "baseline_success" or (
            mode == "full_success" and cell["condition"] in ("step_8", "final")
        )
        quality_fail = mode == "gradient_quality" and cell["condition"] == "gradient_1"
        return {
            "row": {
                **cell,
                "answer_pair_mass": 0.1 if quality_fail else 1.0,
                "kl_from_baseline": 0.0,
                "actual_next_token_id": 1 if success else 0,
                "requested_token_id": 1,
                "preserve_log_odds": 0.1 if success else -0.1,
                "guarded_goal": 0.1,
            }
        }

    def propose(gradients, w, path, stage):
        if mode == "zero":
            return {"status": "method_zero_increment", "stage": stage}
        p = job.project(w, [0.02, 0.0])
        return {**p, "path_after": path + p["step_norm"]}

    result = job.drive(plan, call, f.skip, propose, updates.append, endpoints.append)
    expected = {"baseline_success": 16, "gradient_quality": 24, "zero": 24}.get(mode, 144)
    assert f.attempts == len(seen) == expected and f.cursor == 144
    assert len(f.skips) == 144 - expected
    assert not result["transfer_ran"] and not any(c["condition"] == "oracle_off" for c, _ in seen)
    assert result["candidate_eligible"] == (mode in ("baseline_success", "full_success"))
    if mode == "gradient_quality":
        assert not updates and result["updates"] == 0 and result["attempted_updates"] == 1
    if expected == 144:
        assert sum(c["condition"].startswith("gradient_") for c, _ in seen) == 64
    for stage in range(1, 9):
        assert len({tuple(w) for c, w in seen if c["condition"] == f"step_{stage}"}) <= 1


def test_negative_rhs_and_corrupt_independent_kkt():
    gradients = [
        {
            "row": {
                "cell_id": str(i),
                "h0_norm": 5.0,
                "h0": [0.0, 3.0, 4.0],
                "gradient": [1.0, 0.0, 0.0],
                "preserve_log_odds": -0.2 if i % 2 == 0 else 0.2,
                "guarded_goal": 0.1,
            }
        }
        for i in range(8)
    ]
    proposal = job.increment(gradients, [0.0] * 3, 0.0, 1)
    assert proposal["rhs"] == pytest.approx([0.3, -0.1] * 4)
    assert proposal["d"][0] > 0 and proposal["step_norm"] == pytest.approx(0.05)
    assert audit.verify_update(proposal, [x["row"] for x in gradients], [0.0] * 3, 0.0)[
        "optimizer_kkt_verified"
    ]
    proposal["solver"]["solution"]["multipliers"] = [-1.0] * 8
    with pytest.raises(ValueError):
        audit.verify_update(proposal, [x["row"] for x in gradients], [0.0] * 3, 0.0)


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
        "premature_candidate",
        "raw_L",
        "delta_L",
        "signed_deltaS",
        "archived_S0",
        "guarded_goal",
        "goal_residual",
        "retention_nonweakening",
        "guarded_accepted",
    ],
)
def test_raw_audit_rejects_corruption(tmp_path, fault):
    plan, _, _, _, rows, _, _ = execute(tmp_path)
    changed = copy.deepcopy(rows)
    target = next(r for r in changed if r["condition"] == "step_1")
    if fault in (
        "raw_L",
        "delta_L",
        "signed_deltaS",
        "archived_S0",
        "guarded_goal",
        "goal_residual",
    ):
        target[fault] += 0.01
    elif fault in ("retention_nonweakening", "guarded_accepted"):
        target[fault] = not target[fault]
    elif fault == "sign":
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
    elif fault == "premature_candidate":
        (tmp_path / "preserve_vector.json").write_text("{}")
    else:
        path = tmp_path / ("skip_events.jsonl" if fault == "skip" else "updates.jsonl")
        items = [json.loads(x) for x in path.read_text().splitlines()]
        if fault == "optimizer":
            items[0]["solver"]["solution"]["multipliers"] = [-1.0] * 8
        elif fault == "projection":
            items[0]["actual_preserve_margins"][0] += 0.01
        else:
            items[0]["reason"] = "quality_failure"
        path.write_text("\n".join(json.dumps(x) for x in items) + "\n")
    with pytest.raises((ValueError, KeyError)):
        audit.verify_data(plan, changed, tmp_path)


def bound_verified(output, result):
    result["audited_endpoint_sha256"] = protocol.sha((output / "endpoint.json").read_bytes())
    result["audited_result_sha256"] = protocol.sha((output / "result.json").read_bytes())
    protocol.io.write_new(output / "verification.json", result)


def test_candidate_requires_durable_audit_then_binds_exact_vector(tmp_path):
    *_, verified = execute(tmp_path)
    with pytest.raises(FileNotFoundError):
        audit.freeze_verified_candidate(tmp_path, verified)
    bound_verified(tmp_path, verified)
    assert audit.freeze_verified_candidate(tmp_path, verified)
    candidate = protocol.read(tmp_path / "preserve_vector.json")
    lock = protocol.read(tmp_path / "candidate_freeze.json")
    assert candidate["vector"] == protocol.read(tmp_path / "endpoint.json")["w"]
    assert candidate["verification_sha256"] == protocol.sha(
        (tmp_path / "verification.json").read_bytes()
    )
    assert lock["sha256"] == protocol.sha((tmp_path / "preserve_vector.json").read_bytes())
    assert lock["after_independent_audit"] is True


@pytest.mark.parametrize(
    "fault",
    ["failed_final", "failed_guard_only", "audit_failed", "endpoint_changed", "technical_fault"],
)
def test_no_candidate_on_failed_or_changed_audit(tmp_path, fault):
    *_, verified = execute(tmp_path)
    if fault == "failed_final":
        verified["summary"]["final_accepted"] = 7
        verified["summary"]["candidate_eligible"] = False
    elif fault == "failed_guard_only":
        verified["summary"]["final_guarded_goals"] = 7
    elif fault == "audit_failed":
        verified["status"] = "INCONCLUSIVE"
    bound_verified(tmp_path, verified)
    if fault == "endpoint_changed":
        endpoint = protocol.read(tmp_path / "endpoint.json")
        endpoint["w"][0] += 0.01
        (tmp_path / "endpoint.json").write_text(json.dumps(endpoint))
    elif fault == "technical_fault":
        (tmp_path / "INVALID.json").write_text("{}")
    if fault in ("technical_fault", "endpoint_changed"):
        with pytest.raises(ValueError):
            audit.freeze_verified_candidate(tmp_path, verified)
    else:
        assert not audit.freeze_verified_candidate(tmp_path, verified)
    assert not (tmp_path / "preserve_vector.json").exists()


def source(fn):
    return textwrap.dedent(inspect.getsource(fn))


def same(a, b):
    assert ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))


def test_exact_solver_execution_projection_and_scoring_parity():
    assert job.optimizer is parent.optimizer
    assert job.project is parent.project
    assert job.quality is parent.quality
    assert job.score_float32_logits is parent.score_float32_logits
    same(source(parent.accepts), source(job.accepts))
    same(
        source(parent.increment).replace(
            '0.10 - r["preserve_log_odds"]', 'r["guarded_goal"] - r["preserve_log_odds"]'
        ),
        source(job.increment),
    )
    same(
        source(parent.drive).replace(
            'all(accepts(s["row"]) for s in final)', 'all(guarded_accepts(s["row"]) for s in final)'
        ),
        source(job.drive),
    )
    for name in ("begin", "skip"):
        same(source(getattr(parent.ForwardLedger, name)), source(getattr(job.ForwardLedger, name)))
    same(source(parent.Derivatives.call), source(job.Derivatives.call))
    same(
        source(parent_audit.verify_update)
        .replace('0.10 - r["preserve_log_odds"]', 'r["guarded_goal"] - r["preserve_log_odds"]')
        .replace('decimal_audit.Interval("0.10")', 'decimal_audit.Interval(row["guarded_goal"])'),
        source(audit.verify_update),
    )
    same(
        source(parent_audit.replay).replace(
            "all(accepts(r) for r in final)", "all(guarded_accepts(r) for r in final)"
        ),
        source(audit.replay),
    )


def test_physical_hook_scoring_and_recording_ast_unchanged():
    old_tree, new_tree = [ast.parse(source(f)) for f in (parent.Session.call, job.Session.call)]

    def assignments(tree):
        return {
            ast.unparse(node.targets[0]): ast.dump(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and len(node.targets) == 1
        }

    old_assign, new_assign = assignments(old_tree), assignments(new_tree)
    # Every original computation/assignment, including the original row dictionary,
    # casting, model context, hidden displacements, norms and boundary, is unchanged.
    for key, value in old_assign.items():
        assert new_assign[key] == value, key

    def calls(tree, names):
        return [
            ast.dump(n)
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and ast.unparse(n.func) in names
        ]

    names = {
        "capture_final_prompt_gradient",
        "next_token_logits",
        "score_float32_logits",
        "base.append_row",
        "self.model.hooks",
        "old.offset_hook",
    }
    assert calls(old_tree, names) == calls(new_tree, names)
    for phrase in (
        "physical net/path bound",
        "physical step/cast bound",
        "offset/component/nonfinal state",
    ):

        def arm(tree, phrase=phrase):
            return next(
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.If)
                and any(isinstance(k, ast.Constant) and k.value == phrase for k in ast.walk(n))
                and any(
                    isinstance(k, ast.Constant) and k.value == phrase
                    for b in n.body
                    for k in ast.walk(b)
                )
            )

        assert ast.dump(arm(old_tree)) == ast.dump(arm(new_tree))
    new_call = source(job.Session.call)
    assert '"kl_from_baseline",' in new_call and '"raw_L",' in new_call
    assert '"delta_L",' in new_call and '"signed_deltaS",' in new_call


def test_pinned_runtime_identity_and_preload_storage_order():
    model = protocol.build_plan()["model"]
    assert model["id"] == "Qwen/Qwen3.5-0.8B"
    assert model["revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
    constants = [
        node.value
        for node in ast.walk(ast.parse(source(audit.verify)))
        if isinstance(node, ast.Constant)
    ]
    assert model["revision"] in constants and model["id"] in constants
    worker = source(job.worker)
    assert worker.index("protocol.storage_preflight") < worker.index("base.load_backend")
    assert worker.index('"storage_preflight.json"') < worker.index("base.load_backend")


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


def test_original_eight_does_not_stop_before_combined_eight(tmp_path):
    _, _, f, d, rows, s, _ = execute(tmp_path, positive=0.17)
    first = [r for r in rows if r["condition"] == "step_1"]
    second = [r for r in rows if r["condition"] == "step_2"]
    assert len(first) == len(second) == 8
    assert all(job.accepts(r) for r in first) and not all(job.guarded_accepts(r) for r in first)
    assert job.stopping([{"row": r} for r in first]) is None
    assert audit.stop_for(first) is None
    assert all(job.guarded_accepts(r) for r in second)
    assert s["stop_reason"] == "accepted" and s["updates"] == 2
    assert s["final_accepted"] == s["final_guarded_goals"] == s["final_combined_accepted"] == 8
    assert f.attempts == 48 and d.attempts == 16 and len(f.skips) == 96
    assert all(r["guarded_goal"] == max(0.1, r["archived_S0"]) for r in rows)


def test_frozen_not_ratcheted_rhs_includes_overshoot_and_no_order_selector():
    gradients = [
        {
            "row": {
                "cell_id": str(i),
                "h0": [0.0, 3.0, 4.0],
                "h0_norm": 5.0,
                "gradient": [1.0, 0.0, 0.0],
                "guarded_goal": 0.2 if i % 2 else 0.1,
                "preserve_log_odds": 0.7 if i % 2 else -0.2,
            }
        }
        for i in range(8)
    ]
    p = job.increment(gradients, [0.0, 0.0, 0.0], 0.0, 2)
    assert p["rhs"] == pytest.approx([0.3, -0.5] * 4)
    assert all(p["rhs"][i] < 0 for i in (1, 3, 5, 7))
    assert audit.verify_update(p, [g["row"] for g in gradients], [0.0] * 3, 0.0)[
        "optimizer_kkt_verified"
    ]
    for i in (1, 3, 5, 7):
        gradients[i]["row"]["preserve_log_odds"] += 1.0
    q = job.increment(gradients, [0.0, 0.0, 0.0], 0.0, 3)
    assert q["rhs"] == pytest.approx([0.3, -1.5] * 4)
    # Overshot constraints can relax; goals never become current or running-best S.
    assert all(g["row"]["guarded_goal"] == (0.2 if i % 2 else 0.1) for i, g in enumerate(gradients))


@pytest.mark.parametrize(
    "fault", ["h0", "h0_norm", "S0", "actual_next_token_id", "actual_next_token_label"]
)
def test_archive_mismatch_before_any_gradient(tmp_path, fault):
    plan, backend = prepare()
    archive = plan["archived_baselines"][plan["construction_ids"][0]]
    if fault == "h0":
        archive[fault][0] += 0.001
    elif fault == "actual_next_token_label":
        archive[fault] = "A"
    else:
        archive[fault] += 1 if fault == "actual_next_token_id" else 0.001
    f = job.ForwardLedger(tmp_path, plan["cells"], time.monotonic() + 100)
    d = job.Derivatives(torch, tmp_path, plan["derivative_cells"], time.monotonic() + 100)
    with pytest.raises(ValueError, match="archived baseline identity"):
        job.evaluate(plan, backend, f, d, tmp_path)
    assert f.attempts == 1 and d.attempts == 0
    assert not (tmp_path / "endpoint.json").exists()


def test_terminal_only_after_earlier_original_success_and_later_failure(tmp_path):
    plan = protocol.build_plan()
    plan["model"]["d_model"] = 2
    f = job.ForwardLedger(tmp_path, plan["cells"], time.monotonic() + 100)
    endpoints, updates, seen = [], [], []

    def call(cell, w, current):
        f.begin(cell)
        f.finish(True)
        seen.append((cell, list(w)))
        # Round1 original pass only; round8 failure. Final must retain round8.
        S = 0.06 if cell["condition"] == "step_1" else -0.1
        return {
            "row": {
                **cell,
                "answer_pair_mass": 1.0,
                "kl_from_baseline": 0.0,
                "actual_next_token_id": 1 if S > 0 else 0,
                "requested_token_id": 1,
                "preserve_log_odds": S,
                "guarded_goal": 0.1,
            }
        }

    def propose(gradients, w, path, stage):
        p = job.project(w, [0.02, 0.0])
        return {**p, "path_after": path + p["step_norm"]}

    result = job.drive(plan, call, f.skip, propose, updates.append, endpoints.append)
    assert result["stop_reason"] == "max_updates" and not result["candidate_eligible"]
    assert result["updates"] == 8 and f.attempts == 144
    assert result["w"] == updates[-1]["w_after"] != updates[0]["w_after"]
    assert all(pid.endswith("__step_8") for pid in result["endpoint_cell_ids"])
    assert all(w == result["w"] for c, w in seen if c["condition"] == "final")


def test_common_w_jacobian_uses_original_norm_and_semantic_sign():
    gradients = [
        {
            "row": {
                "cell_id": str(i),
                "h0": [0.0, 3.0, 4.0],
                "h0_norm": 5.0,
                "gradient": [(-1.0 if i % 2 else 1.0), 1.0, 0.0],
                "guarded_goal": 0.1,
                "preserve_log_odds": -0.1,
            }
        }
        for i in range(8)
    ]
    p = job.increment(gradients, [0.0, 0.0, 0.0], 0.0, 1)
    assert p["d"] == pytest.approx([0.0, 0.04, 0.0], abs=1e-12)
    assert audit.verify_update(p, [g["row"] for g in gradients], [0.0] * 3, 0.0)[
        "optimizer_kkt_verified"
    ]


@pytest.mark.parametrize("fault", ["kl_from_baseline", "actual_next_token_label"])
def test_final_score_identity_extension(tmp_path, monkeypatch, fault):
    plan, backend = prepare()
    original = job.score_float32_logits
    count = 0

    def corrupt(*args, **kwargs):
        nonlocal count
        score = original(*args, **kwargs)
        count += 1
        if count == 25:
            score[fault] = score[fault] + 0.01 if fault == "kl_from_baseline" else "OTHER"
        return score

    monkeypatch.setattr(job, "score_float32_logits", corrupt)
    f = job.ForwardLedger(tmp_path, plan["cells"], time.monotonic() + 100)
    d = job.Derivatives(torch, tmp_path, plan["derivative_cells"], time.monotonic() + 100)
    with pytest.raises(ValueError, match="current-state/independent identity"):
        job.evaluate(plan, backend, f, d, tmp_path)
    assert f.attempts == 25 and d.attempts == 8
    assert not (tmp_path / "preserve_vector.json").exists()


def test_saved_stage_predictions_are_actual_projected_increment(tmp_path):
    *_, verified = execute(tmp_path, positive=0.17)
    stages = verified["construction_stages"]
    assert len(stages) == 32
    assert sum(r["condition"] == "baseline" for r in stages) == 8
    for r in stages:
        if r["condition"] == "baseline":
            assert r["predicted_S_from_actual_increment"] is None
        else:
            stage = 2 if r["condition"] == "final" else r["stage"]
            a = next(x for x in verified["optimizer_checks"] if x["stage"] == stage)
            index = next(
                i
                for i, p in enumerate(protocol.build_plan()["construction_ids"])
                if p == r["prompt_id"]
            )
            pred = a["actual_preserve_margins"][index]
            assert r["predicted_S_from_actual_increment"] == pred
            assert r["actual_minus_predicted_S"] == r["signed_margin"] - pred
