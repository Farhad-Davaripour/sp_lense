"""Model-free changed-path checks for twelve rows, three situations, eight updates."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import textwrap
import time
from types import SimpleNamespace

import pytest
import test_shared_comply_two_family as parent_fixtures
import torch

from scripts import shared_comply_crossed_three_family as job
from scripts import shared_comply_crossed_three_family_plan as protocol
from scripts import verify_shared_comply_crossed_three_family as checker
from scripts import verify_shared_comply_two_family as parent_checker

FAMILIES = (
    "cg_f01_archive_closeout",
    "cg_f02_translation_console",
    "cg_f03_context_rotation",
)


class ThreeNormToy(parent_fixtures.Toy):
    def __init__(self, prompts, mode="linear"):
        super().__init__(prompts)
        self.mode = mode

    def forward(self, tokens):
        assert not self.weight.requires_grad
        self.calls.append(int(tokens[0, 1]))
        prompt = self.prompts[int(tokens[0, 1]) - 10]
        multiplier = FAMILIES.index(prompt["family_id"]) + 1
        h = torch.tensor([0.0, 3.0 * multiplier, 4.0 * multiplier]).repeat(1, tokens.shape[-1], 1)
        for name, function in self.active_hooks:
            assert name == "blocks.10.hook_out"
            h = function(h, hook=SimpleNamespace(name=name))
        x = h[..., 0]
        semantic = 0.2 + x
        a = semantic if prompt["preserve_label"] == "A" else -semantic
        other = (
            torch.where(x < 0, torch.full_like(a, 10.0), torch.full_like(a, -20.0))
            if self.mode == "quality_failure"
            else torch.full_like(a, -20.0)
        )
        logits = torch.stack(
            [a, torch.zeros_like(a), other, *[torch.full_like(a, -20.0) for _ in range(3)]],
            dim=-1,
        )
        if self.mode == "final_mismatch" and len(self.calls) == 37:
            logits = logits.clone()
            logits[..., 0] += 0.01
        return logits


def prepare(mode="linear"):
    plan, backend = parent_fixtures.setup(protocol.build_plan())
    plan["model"]["d_model"] = 3
    model = ThreeNormToy(plan["prompts"], mode)
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


def execute(output, mode="linear"):
    plan, backend = prepare(mode)
    deadline = time.monotonic() + 100
    forwards = job.ForwardLedger(output, plan["cells"], deadline)
    derivatives = job.Derivatives(torch, output, plan["derivative_cells"], deadline)
    rows, summary = job.evaluate(plan, backend, forwards, derivatives, output)
    verified = checker.verify_data(plan, rows, output)
    checker.compare_summary(summary, verified["summary"])
    return plan, backend, forwards, derivatives, rows, summary, verified, output


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    # One synthetic construction supplies several read-only corruption checks.
    return execute(tmp_path_factory.mktemp("three_family_fake"))


def test_exact_first_three_families_v1_self_crossed_layout_and_budget():
    plan = protocol.build_plan()
    prompts, cells = plan["prompts"], plan["cells"]
    assert len(prompts) == 12 and len(cells) == 216
    assert len(plan["derivative_cells"]) == 96
    assert sum(cell["optional"] for cell in cells) == 192
    assert [p["family_id"] for p in prompts] == [family for family in FAMILIES for _ in range(4)]
    assert {p["variant_id"] for p in prompts} == {"v1"}
    assert {p["category"] for p in prompts} == {"self_shutdown"}
    assert {p["split"] for p in prompts} == {"discovery"}
    assert not plan["control_ids"] and not plan["transfer_ids"]
    assert all("cg_f04" not in p["prompt_id"] for p in prompts)
    for start in (0, 4, 8):
        family_prompts = prompts[start : start + 4]
        assert [(p["preserve_label"], p["display_order"]) for p in family_prompts] == [
            ("A", "A_then_B"),
            ("A", "B_then_A"),
            ("B", "A_then_B"),
            ("B", "B_then_A"),
        ]
    for ab, ba in zip(prompts[::2], prompts[1::2], strict=True):
        assert ab["option_line_by_letter"] == ba["option_line_by_letter"]
        assert ab["semantic_to_letter"] == ba["semantic_to_letter"]
        assert ab["non_option_bytes_sha256"] == ba["non_option_bytes_sha256"]
        assert ab["prompt_sha256"] == protocol.sha(ab["prompt"].encode())
        assert ba["prompt_sha256"] == protocol.sha(ba["prompt"].encode())
    assert all(c["condition"] == "baseline" for c in cells[:12])
    assert all(c["condition"] == "final" for c in cells[204:])
    assert plan["config"]["maximum_updates"] == 8
    for stage in range(1, 9):
        assert sum(c["condition"] == f"gradient_{stage}" for c in cells) == 12
        assert sum(c["condition"] == f"step_{stage}" for c in cells) == 12
    checked = checker.verify_plan(plan)
    assert checked["rendering_rows"] == 12 and checked["independent_semantic_situations"] == 3
    assert checked["maximum_forwards"] == 216 and checked["maximum_derivatives"] == 96


@pytest.mark.parametrize("fault", ["v2", "fourth_family", "display", "schedule"])
def test_new_twelve_row_layout_rejects_scope_or_count_tampering(fault):
    plan = copy.deepcopy(protocol.build_plan())
    if fault == "v2":
        plan["prompts"][8]["variant_id"] = "v2"
    elif fault == "fourth_family":
        plan["prompts"][8]["family_id"] = "cg_f04_memory_archive"
    elif fault == "display":
        plan["prompts"][9]["display_order"] = "A_then_B"
    else:
        plan["cells"][12]["optional"] = False
    with pytest.raises(ValueError):
        checker.verify_layout(plan)


def test_one_fresh_shared_arrow_own_norms_and_first_twelve_of_twelve_stop(completed):
    plan, backend, forwards, derivatives, rows, summary, verified, output = completed
    assert summary["final_accepted"] == 12 and summary["candidate_eligible"]
    assert summary["updates"] == summary["attempted_updates"] == 1
    assert summary["rendering_rows"] == 12 and summary["independent_semantic_situations"] == 3
    assert forwards.attempts == forwards.completed == len(backend.model.calls) == 48
    assert derivatives.attempts == derivatives.completed == 12
    assert forwards.cursor == 216 and len(forwards.skips) == 168
    baselines = [r for r in rows if r["condition"] == "baseline"]
    finals = [r for r in rows if r["condition"] == "final"]
    assert all(r["shared_w"] == [0.0] * 3 for r in baselines)
    assert {r["h0_norm"] for r in finals} == {5.0, 10.0, 15.0}
    assert len({tuple(r["shared_w"]) for r in finals}) == 1
    for row in finals:
        assert row["intended_delta"] == [checker.f32(row["h0_norm"] * x) for x in row["shared_w"]]
        assert row["net_norm"] <= 0.20 * row["h0_norm"] + 1e-6
        assert row["path_norm"] <= 0.40 * row["h0_norm"] + 1e-6
        assert row["step_norm"] <= 0.05 * row["h0_norm"] + 1e-6
        assert row["target_sign"] == -1 and row["requested"] == "comply"
        assert row["unselected_max_difference"] == 0 and row["weights_unchanged"]
    for direction in ("A_to_B", "B_to_A"):
        assert summary["directional_coverage"][direction] == {
            "eligible": 6,
            "achieved": 6,
            "status": "ALL",
        }
    assert verified["maximum_nonfinal_difference"] == 0
    assert (
        not (output / "comply_vector.json").exists()
        and not (output / "candidate_freeze.json").exists()
    )
    assert backend.model.active_hooks == [] and backend.model.weight.requires_grad
    assert backend.model.weight.grad is None
    assert plan["config"]["maximum_updates"] == 8


def test_twelve_constraint_update_keeps_comply_gradient_sign_and_negative_rhs():
    gradients = [
        {
            "row": {
                "cell_id": str(i),
                "h0_norm": 5.0 * (i + 1),
                "h0": [0.0, 3.0 * (i + 1), 4.0 * (i + 1)],
                "gradient": [1.0, 0.0, 0.0],
                "preserve_log_odds": -0.2 if i % 2 == 0 else 0.2,
            }
        }
        for i in range(12)
    ]
    proposal = job.increment(gradients, [0.0] * 3, 0.0, 1)
    assert proposal["rhs"] == pytest.approx([-0.1, 0.3] * 6)
    assert proposal["d"][0] < 0
    assert checker.verify_update(proposal, [g["row"] for g in gradients], [0.0] * 3, 0.0)[
        "optimizer_kkt_verified"
    ]


@pytest.mark.parametrize(
    "mode", ["full_success", "max_updates", "baseline_success", "gradient_quality", "zero"]
)
def test_count_adaptation_preserves_eight_update_limit_and_conditional_skips(tmp_path, mode):
    plan = protocol.build_plan()
    plan["model"]["d_model"] = 2
    forwards = job.ForwardLedger(tmp_path, plan["cells"], time.monotonic() + 100)
    seen, updates, endpoints = [], [], []

    def call(cell, w, current):
        forwards.begin(cell)
        forwards.finish(True)
        seen.append((cell, list(w)))
        success = mode == "baseline_success" or (
            mode == "full_success" and cell["condition"] in ("step_8", "final")
        )
        quality_failure = mode == "gradient_quality" and cell["condition"] == "gradient_1"
        return {
            "row": {
                **cell,
                "answer_pair_mass": 0.1 if quality_failure else 1.0,
                "kl_from_baseline": 0.0,
                "actual_next_token_id": 1 if success else 0,
                "requested_token_id": 1,
                "preserve_log_odds": -0.1 if success else 0.1,
            }
        }

    def propose(gradients, w, path, stage):
        assert len(gradients) == 12 and stage <= 8
        if mode == "zero":
            return {"status": "method_zero_increment", "stage": stage}
        projected = job.project(w, [0.02, 0.0])
        return {**projected, "path_after": path + projected["step_norm"]}

    result = job.drive(plan, call, forwards.skip, propose, updates.append, endpoints.append)
    expected = {"baseline_success": 24, "gradient_quality": 36, "zero": 36}.get(mode, 216)
    assert forwards.attempts == len(seen) == expected and forwards.cursor == 216
    assert len(forwards.skips) == 216 - expected
    assert result["candidate_eligible"] == (mode in ("baseline_success", "full_success"))
    assert all(w == [0.0, 0.0] for c, w in seen if c["condition"] == "baseline")
    if expected == 216:
        assert sum(c["condition"].startswith("gradient_") for c, _ in seen) == 96
        assert result["attempted_updates"] == result["updates"] == 8
    for stage in range(1, 9):
        assert len({tuple(w) for c, w in seen if c["condition"] == f"step_{stage}"}) <= 1
    assert not result["transfer_ran"]


def test_finite_scientific_failure_completes_twelve_cell_group_and_finals(tmp_path):
    _, _, forwards, derivatives, rows, summary, _, _ = execute(tmp_path, "quality_failure")
    assert forwards.attempts == 48 and derivatives.attempts == 12
    assert summary["stop_reason"] == "quality_failure" and summary["final_accepted"] == 0
    assert summary["final_other_token_outcomes"] == 12 and not summary["candidate_eligible"]
    assert len([r for r in rows if r["condition"] == "step_1"]) == 12
    assert len([r for r in rows if r["condition"] == "final"]) == 12


def test_first_independent_final_mismatch_stops_without_candidate(tmp_path):
    plan, backend = prepare("final_mismatch")
    forwards = job.ForwardLedger(tmp_path, plan["cells"], time.monotonic() + 100)
    derivatives = job.Derivatives(torch, tmp_path, plan["derivative_cells"], time.monotonic() + 100)
    with pytest.raises(ValueError, match="current-state/independent identity"):
        job.evaluate(plan, backend, forwards, derivatives, tmp_path)
    assert forwards.attempts == 37 and derivatives.attempts == 12
    assert (tmp_path / "endpoint.json").exists() and not (tmp_path / "comply_vector.json").exists()


@pytest.mark.parametrize("fault", ["common_w", "raw_argmax", "nonfinite"])
def test_new_raw_audit_rejects_shared_vector_argmax_and_nonfinite_tampering(completed, fault):
    plan, _, _, _, rows, _, _, output = completed
    changed = copy.deepcopy(rows)
    row = next(r for r in changed if r["condition"] == "step_1")
    if fault == "common_w":
        row["shared_w"][0] += 0.01
        row["shared_w_sha256"] = job.vector_sha(row["shared_w"])
    elif fault == "raw_argmax":
        row["actual_next_token_id"] = 2
    else:
        row["kl_from_baseline"] = float("nan")
    with pytest.raises(ValueError):
        checker.verify_data(plan, changed, output)


def test_forward_derivative_caps_and_no_restart_are_exact_216_96(tmp_path):
    plan = protocol.build_plan()
    forwards = job.ForwardLedger(tmp_path, plan["cells"], 100, now=lambda: 0)
    for cell in plan["cells"]:
        forwards.begin(cell)
        forwards.finish(True)
    assert forwards.attempts == forwards.completed == 216
    with pytest.raises(ValueError):
        forwards.begin(plan["cells"][-1])
    with pytest.raises(ValueError, match="retry"):
        job.ForwardLedger(tmp_path, plan["cells"], 100)
    derivatives = job.Derivatives(torch, tmp_path, plan["derivative_cells"], 100, now=lambda: 0)
    derivatives.original_grad = lambda: "synthetic ledger call, not a derivative"
    for cell in plan["derivative_cells"]:
        derivatives.cell = cell
        assert derivatives.call() == "synthetic ledger call, not a derivative"
    assert derivatives.attempts == derivatives.completed == 96
    with pytest.raises(ValueError):
        derivatives.call()


@pytest.mark.parametrize("entry", ["run", "worker"])
def test_preparation_has_no_authority_to_claim_load_or_run(tmp_path, monkeypatch, entry):
    monkeypatch.setattr(job, "OUTPUT", tmp_path)
    monkeypatch.setattr(job.shell, "OUTPUT", tmp_path)
    (tmp_path / "preregistration.json").write_bytes(b"{}")
    monkeypatch.delenv(protocol.AUTH_KEY, raising=False)
    monkeypatch.setattr(job.engine, entry, lambda: pytest.fail("unauthorized engine reached"))
    monkeypatch.setattr(job.base, "load_backend", lambda *args: pytest.fail("unauthorized load"))
    with pytest.raises(ValueError, match="separate supervisor"):
        getattr(job, entry)()
    assert {p.name for p in tmp_path.iterdir()} == {"preregistration.json"}


def test_old_two_family_authorization_scope_cannot_authorize_new_study(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "OUTPUT", tmp_path)
    monkeypatch.setattr(job.shell, "OUTPUT", tmp_path)
    path = tmp_path / "preregistration.json"
    path.write_bytes(b"{}")
    authorization = {
        "preregistration_sha256": protocol.sha(path.read_bytes()),
        "scope": "one crossed COMPLY construction;144F/64D/900s;no retry",
        "authorized_by": "supervisor",
    }
    monkeypatch.setenv(protocol.AUTH_KEY, json.dumps(authorization))
    with pytest.raises(ValueError, match="separate supervisor"):
        job.require_authorization()


def test_storage_total_cap_and_one_GiB_preload_guard(monkeypatch):
    cfg = protocol.build_plan()["config"]
    storage = cfg["storage"]
    assert storage["maximum_arrays"] == 216
    assert storage["total_bound_bytes"] <= 512 * 1024**2
    assert storage["minimum_free_bytes"] == 1024**3
    assert storage["logits_bound_bytes"] == 216 * storage["zlib_bound_per_array"]
    assert storage["total_bound_bytes"] == sum(
        storage[key]
        for key in (
            "logits_bound_bytes",
            "rows_bound_bytes",
            "updates_bound_bytes",
            "other_bound_bytes",
        )
    )
    shutil = protocol.storage_preflight.__globals__["shutil"]
    monkeypatch.setattr(shutil, "disk_usage", lambda root: SimpleNamespace(free=1024**3))
    assert protocol.storage_preflight(protocol.ROOT, cfg)["passed"]
    monkeypatch.setattr(shutil, "disk_usage", lambda root: SimpleNamespace(free=1024**3 - 1))
    with pytest.raises(ValueError, match="storage"):
        protocol.storage_preflight(protocol.ROOT, cfg)


@pytest.mark.parametrize("entry", ["freeze", "preflight"])
@pytest.mark.parametrize("certificate", ["missing", "false"])
def test_uncertified_preparation_blocks_lock_and_preflight_before_any_write(
    tmp_path, monkeypatch, entry, certificate
):
    def read_certificate(path):
        if certificate == "missing":
            raise FileNotFoundError("synthetic missing preparation certificate")
        return {
            "status": "MODEL_FREE_PREPARATION_CERTIFIED",
            "storage_certified": False,
            "accounting_certified": True,
            "model_loads": 0,
            "tokenizer_loads": 0,
            "real_forwards": 0,
            "real_derivatives": 0,
        }

    def forbidden(*args, **kwargs):
        raise AssertionError("uncertified preparation reached lock, claim, or model loading")

    monkeypatch.setattr(protocol, "read", read_certificate)
    monkeypatch.setattr(job, "OUTPUT", tmp_path)
    monkeypatch.setattr(job, "_frozen_freeze", forbidden)
    monkeypatch.setattr(job, "_locked_preflight", forbidden)
    monkeypatch.setattr(job.base, "load_backend", forbidden)
    with pytest.raises((FileNotFoundError, ValueError)):
        getattr(job, entry)()
    assert not list(tmp_path.iterdir())


def test_independent_checker_changes_only_constraint_and_row_counts_not_precision_or_update_limit():
    for adapted, old in (
        (checker._parent_update, parent_checker.verify_update),
        (checker._parent_data, parent_checker.verify_data),
        (checker.replay, parent_checker.replay),
    ):
        tree = ast.parse(textwrap.dedent(inspect.getsource(old)))
        assert adapted.original_ast_sha256 == hashlib.sha256(ast.dump(tree).encode()).hexdigest()
    update = ast.parse(checker._parent_update.adapted_source)
    constants = [
        n.value for n in ast.walk(update) if isinstance(n, ast.Constant) and type(n.value) is int
    ]
    assert 4096 in constants and 256 not in constants
    assert 80 in constants and 8 not in constants
    replay = ast.parse(checker.replay.adapted_source)
    assert any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "range"
        and [getattr(a, "value", None) for a in n.args] == [1, 9]
        for n in ast.walk(replay)
    )
    assert checker.engine is not parent_checker
    assert checker.accepts is parent_checker.accepts or inspect.getsource(
        checker.accepts
    ) == inspect.getsource(parent_checker.accepts)


def test_report_keeps_twelve_finals_eight_rounds_and_new_training_scope(completed):
    *_, verified, _ = completed
    text = checker.report(verified)
    assert "Final acceptance 12/12:" in text
    assert "Attempted rounds 1/8" in text
    assert "/216, derivatives 12/96" in text
    assert "THREE training situations" in text and "f03 is training" in text
    assert "f04 is EXPOSED development outside fitting" in text
    assert "Even8/8" not in text and "No f03 fitting" not in text
