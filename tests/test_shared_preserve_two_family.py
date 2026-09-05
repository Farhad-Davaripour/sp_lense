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

from scripts import projected_shared_preserve as parent
from scripts import shared_direction_linear_feasibility as parent_solver
from scripts import shared_preserve_eight_row_solver as solver
from scripts import shared_preserve_two_family as job
from scripts import shared_preserve_two_family_plan as protocol
from scripts import verify_projected_shared_preserve as parent_audit
from scripts import verify_shared_preserve_two_family as audit

TOL = {
    "rank_relative_pivot_floor": 1e-12,
    "primal_absolute_tolerance": 1e-9,
    "kkt_absolute_tolerance": 1e-8,
}


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
    assert all("cg_f03" in p for p in plan["reserved_unrun_prompt_ids"])
    assert not any("cg_f03" in p["prompt_id"] for p in plan["prompts"])
    assert job.base.MAX_ATTEMPTS == 48
    f = job.ForwardLedger(tmp_path, plan["cells"], 100, now=lambda: 0)
    for cell in plan["cells"]:
        f.begin(cell)
        f.finish(True)
    assert f.attempts == f.completed == 144
    with pytest.raises(ValueError):
        f.begin(plan["cells"][-1])


def test_selection_does_not_expand_reserved_or_nontraining_text():
    class Forbidden(dict):
        def __getitem__(self, key):
            if key != "id":
                raise AssertionError("out-of-scope text accessed")
            return super().__getitem__(key)

    cfg = protocol.read(protocol.ROOT / protocol.CONFIG)
    data = protocol.read(protocol.ROOT / cfg["dataset"]["path"])
    manifest = protocol.read(protocol.ROOT / cfg["manifest"]["path"])
    for i, family in enumerate(data["families"]):
        if family["id"] not in cfg["training_families"]:
            data["families"][i] = Forbidden(id=family["id"])
    prompts, reserved = protocol.select_inputs(data, manifest, cfg)
    assert len(prompts) == 8 and len(reserved) == 2
    changed = copy.deepcopy(manifest)
    changed["splits"]["discovery"]["family_ids"][:2] = list(reversed(cfg["training_families"]))
    with pytest.raises(ValueError, match="manifest order"):
        protocol.select_inputs(data, changed, cfg)


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


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("mode", ["identity", "negative", "duplicate", "contradictory"])
def test_exact_old_solver_parity(n, mode):
    A = [[float(i == j) for j in range(4)] for i in range(n)]
    b = [0.2 * (i + 1) for i in range(n)]
    if mode == "negative":
        b = [-x for x in b]
    elif mode == "duplicate":
        A = [[1.0, 0.0, 0.0, 0.0] for _ in range(n)]
    elif mode == "contradictory" and n > 1:
        A[0], A[1] = [1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]
    assert solver.solve(A, b, TOL) == parent_solver.solve(A, b, TOL)


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
    ],
)
def test_raw_audit_rejects_corruption(tmp_path, fault):
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
    "fault", ["failed_final", "audit_failed", "endpoint_changed", "technical_fault"]
)
def test_no_candidate_on_failed_or_changed_audit(tmp_path, fault):
    *_, verified = execute(tmp_path)
    if fault == "failed_final":
        verified["summary"]["final_accepted"] = 7
        verified["summary"]["candidate_eligible"] = False
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


def test_exact_solver_and_execution_structural_invariance():
    same(source(parent_solver.solve).replace("len(b) <= 4", "len(b) <= 8"), source(solver.solve))
    for name in ("dot", "norm", "linear_solve", "metrics", "kkt_valid"):
        assert getattr(solver, name) is getattr(parent_solver, name)
    assert job.project is parent.project
    same(source(parent.increment), source(job.increment))
    same(
        source(parent.Session.call).replace("/<=88", "/<=144").replace("/<=32", "/<=64"),
        source(job.Session.call),
    )
    same(source(parent.Derivatives.call).replace("32", "64"), source(job.Derivatives.call))
    same(source(parent.ForwardLedger.begin).replace("88", "144"), source(job.ForwardLedger.begin))
    same(
        source(parent_audit.verify_update)
        .replace("range(16)", "range(256)")
        .replace("16 optimizer subsets", "256 optimizer subsets")
        .replace("range(4)", "range(8)"),
        source(audit.verify_update),
    )


def test_independent_raw_audit_only_count_scope_and_family_reporting_changes():
    expected = (
        source(parent_audit.verify_data)
        .replace("88", "144")
        .replace("== 32", "== 64")
        .replace("144/32", "144/64")
        .replace(
            'require(not plan["transfer_ids"] and len(prompts) == 12, "construction/controls only")',
            'require(not plan["transfer_ids"] and not plan["control_ids"] and len(prompts) == 8, "eight training only")',
        )
        .replace(
            '                "variant_id",',
            '                "family_id",\n                "variant_id",',
        )
    )
    same(expected, source(audit.verify_data))


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
