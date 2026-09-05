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
from test_local_controllability_positive_control import Model, setup

from scripts import shared_comply_two_family as job
from scripts import shared_comply_two_family_plan as protocol
from scripts import shared_preserve_eight_row_solver as solver
from scripts import shared_preserve_two_family as parent
from scripts import verify_shared_comply_two_family as audit
from scripts import verify_shared_preserve_two_family as parent_audit

TOL = {
    "rank_relative_pivot_floor": 1e-12,
    "primal_absolute_tolerance": 1e-9,
    "kkt_absolute_tolerance": 1e-8,
}


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
        S = (-0.2 if p["preserve_label"] == "A" else self.positive) + x * (0 if self.zero else 1)
        a = S if p["preserve_label"] == "A" else -S
        if self.mismatch and torch.is_grad_enabled():
            a = a + 0.01
        other = (
            torch.where(x < 0, torch.full_like(a, 10.0), torch.full_like(a, -20.0))
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
    assert not plan["config"]["future_probe_reservation_only"]["run_allowed"]
    assert not any("cg_f03" in p["prompt_id"] for p in plan["prompts"])
    assert job.base.MAX_ATTEMPTS == 48
    f = job.ForwardLedger(tmp_path, plan["cells"], 100, now=lambda: 0)
    for cell in plan["cells"]:
        f.begin(cell)
        f.finish(True)
    assert f.attempts == f.completed == 144
    with pytest.raises(ValueError):
        f.begin(plan["cells"][-1])


def test_storage_arithmetic_and_preload_failure(monkeypatch):
    cfg = protocol.read(protocol.ROOT / protocol.CONFIG)
    monkeypatch.setattr(
        protocol.previous.shutil, "disk_usage", lambda root: SimpleNamespace(free=512 * 1024**2)
    )
    result = protocol.storage_preflight(protocol.ROOT, cfg)
    assert result["bounds"]["total_bound_bytes"] == 377958704
    monkeypatch.setattr(
        protocol.previous.shutil, "disk_usage", lambda root: SimpleNamespace(free=512 * 1024**2 - 1)
    )
    with pytest.raises(ValueError, match="storage cannot fit"):
        protocol.storage_preflight(protocol.ROOT, cfg)


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
        assert r["target_sign"] == -1 and r["requested"] == "comply"
        assert r["requested_token_id"] == (1 if r["preserve_label"] == "A" else 0)
        if r["gradient"] is not None:
            assert r["gradient"] == [1.0, 0.0, 0.0]
    assert not (tmp_path / "comply_vector.json").exists()
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
        ({"positive": 0.01}, 2, 0, "baseline"),
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
    assert not (tmp_path / "comply_vector.json").exists()


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
    assert (tmp_path / "endpoint.json").exists() and not (tmp_path / "comply_vector.json").exists()


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
                "preserve_log_odds": -0.1 if success else 0.1,
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
    assert proposal["rhs"] == pytest.approx([-0.1, 0.3] * 4)
    assert proposal["d"][0] < 0 and proposal["step_norm"] == pytest.approx(0.05)
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
        target["target_sign"] = 1
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
        (tmp_path / "comply_vector.json").write_text("{}")
    else:
        path = tmp_path / ("skip_events.jsonl" if fault == "skip" else "updates.jsonl")
        items = [json.loads(x) for x in path.read_text().splitlines()]
        if fault == "optimizer":
            items[0]["solver"]["solution"]["multipliers"] = [-1.0] * 8
        elif fault == "projection":
            items[0]["actual_comply_margins"][0] += 0.01
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
    candidate = protocol.read(tmp_path / "comply_vector.json")
    lock = protocol.read(tmp_path / "candidate_freeze.json")
    assert candidate["vector"] == protocol.read(tmp_path / "endpoint.json")["w"]
    assert candidate["verification_sha256"] == protocol.sha(
        (tmp_path / "verification.json").read_bytes()
    )
    assert lock["sha256"] == protocol.sha((tmp_path / "comply_vector.json").read_bytes())
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
    assert not (tmp_path / "comply_vector.json").exists()


def source(fn):
    return textwrap.dedent(inspect.getsource(fn))


def same(a, b):
    assert ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))


def test_exact_solver_projection_guards_and_session_structure():
    assert job.optimizer is parent.optimizer is solver
    assert job.project is parent.project
    assert job.ForwardLedger is parent.ForwardLedger
    assert job.Derivatives is parent.Derivatives
    assert job.recorder is parent.recorder
    assert protocol.storage_preflight is protocol.previous.storage_preflight
    expected = (
        source(parent.increment)
        .replace('A = [[r["h0_norm"] * x', 'A = [[-r["h0_norm"] * x')
        .replace('b = [0.10 - r["preserve_log_odds"]', 'b = [0.10 + r["preserve_log_odds"]')
        .replace('c = [r["preserve_log_odds"]', 'c = [-r["preserve_log_odds"]')
        .replace("_preserve_margins", "_comply_margins")
    )
    same(expected, source(job.increment))
    expected = (
        source(parent.Session.call)
        .replace('"requested": "preserve"', '"requested": "comply"')
        .replace('boundary.token_id(p["preserve_label"])', 'boundary.token_id(p["comply_label"])')
        .replace(
            '"signed_margin": score["preserve_log_odds"]',
            '"signed_margin": -score["preserve_log_odds"]',
        )
        .replace('"target_sign": 1', '"target_sign": -1')
        .replace(
            '        delta_log_odds=row["preserve_log_odds"] - baseline["preserve_log_odds"],',
            '        delta_log_odds=row["preserve_log_odds"] - baseline["preserve_log_odds"],\n        signed_delta_log_odds=-(row["preserve_log_odds"] - baseline["preserve_log_odds"]),',
        )
    )
    same(expected, source(job.Session.call))


def test_independent_raw_and_kkt_audit_only_semantic_sign_changes():
    expected = (
        source(parent_audit.verify_data)
        .replace('if row["preserve_label"] == "A"', 'if row["comply_label"] == "A"')
        .replace('row["requested"] == "preserve"', 'row["requested"] == "comply"')
        .replace('row["target_sign"] == 1', 'row["target_sign"] == -1')
        .replace(
            'row["signed_margin"] == row["preserve_log_odds"]',
            'row["signed_margin"] == -row["preserve_log_odds"]',
        )
        .replace(
            'and row["delta_log_odds"] == row["preserve_log_odds"] - baseline["preserve_log_odds"]',
            'and row["delta_log_odds"] == row["preserve_log_odds"] - baseline["preserve_log_odds"]\n            and row["signed_delta_log_odds"] == -row["delta_log_odds"]',
        )
    )
    same(expected, source(audit.verify_data))
    expected = (
        source(parent_audit.verify_update)
        .replace("A.append([hn * x", "A.append([-hn * x")
        .replace('b = [0.10 - r["preserve_log_odds"]', 'b = [0.10 + r["preserve_log_odds"]')
        .replace("Ai.append([(hn * x)", "Ai.append([-(hn * x)")
        .replace(
            'decimal_audit.Interval("0.10") - row["preserve_log_odds"]',
            'decimal_audit.Interval("0.10") + row["preserve_log_odds"]',
        )
        .replace('c = [x["preserve_log_odds"]', 'c = [-x["preserve_log_odds"]')
        .replace("_preserve_margins", "_comply_margins")
    )
    same(expected, source(audit.verify_update))


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


def test_only_authenticated_training_plan_no_endpoint_or_other_family_reads(monkeypatch):
    original = protocol.Path.read_bytes
    seen = []

    def bounded(path):
        name = str(path).replace("\\", "/")
        assert "frozen_preserve_f03" not in name
        assert not name.endswith(
            (
                "rows.jsonl",
                "endpoint.json",
                "preserve_vector.json",
                "comply_vector.json",
                "transfer_vector.json",
                "verification.json",
            )
        )
        seen.append(name)
        return original(path)

    monkeypatch.setattr(protocol.Path, "read_bytes", bounded)
    plan = protocol.build_plan()
    assert len(plan["prompts"]) == 8 and all(
        "cg_f03" not in p["prompt_id"] for p in plan["prompts"]
    )
    assert any(
        name.endswith("shared_preserve_two_family_qwen35_08b/preregistration.json") for name in seen
    )
    future = plan["config"]["future_probe_reservation_only"]
    assert not future["implemented"] and not future["expanded"] and not future["run_allowed"]
    assert all(c["prompt_id"] in plan["construction_ids"] for c in plan["cells"])


def test_opposed_signs_and_no_delta_sign_acceptance_gate():
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
    comply = job.increment(gradients, [0.0] * 3, 0.0, 1)
    preserve = parent.increment(gradients, [0.0] * 3, 0.0, 1)
    assert comply["d"][0] < 0 < preserve["d"][0]
    assert comply["rhs"] == pytest.approx([-0.1, 0.3] * 4)
    assert preserve["rhs"] == pytest.approx([0.3, -0.1] * 4)
    row = {
        "answer_pair_mass": 1.0,
        "kl_from_baseline": 0.0,
        "actual_next_token_id": 0,
        "requested_token_id": 0,
        "preserve_log_odds": -0.2,
        "signed_delta_log_odds": -0.7,
    }
    assert job.accepts(row) and not parent.accepts(row)
    assert audit.accepts(row)


def test_exact_signed_change_audit_and_candidate_target(tmp_path):
    plan, _, _, _, rows, _, verified = execute(tmp_path)
    assert all(r["signed_delta_log_odds"] == -r["delta_log_odds"] for r in rows)
    changed = copy.deepcopy(rows)
    changed[-1]["signed_delta_log_odds"] += 1e-8
    with pytest.raises(ValueError):
        audit.verify_data(plan, changed, tmp_path)
    bound_verified(tmp_path, verified)
    assert audit.freeze_verified_candidate(tmp_path, verified)
    assert protocol.read(tmp_path / "comply_vector.json")["requested"] == "comply"
    assert not (tmp_path / "preserve_vector.json").exists()
