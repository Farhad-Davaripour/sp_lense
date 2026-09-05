"""Changed-path fake tensor integration only; never a model or Hub download."""

from __future__ import annotations

import copy
import json
import struct
import time
from types import SimpleNamespace

import pytest
import torch
from test_local_controllability_positive_control import Model, setup

from scripts import paired_common_drift_comply as job
from scripts import paired_common_drift_comply_plan as protocol
from scripts import verify_paired_common_drift_comply as checker


class PairedToy(Model):
    def __init__(self, prompts, mode="linear"):
        super().__init__(prompts)
        self.prompts, self.mode = prompts, mode

    def forward(self, tokens):
        assert not self.weight.requires_grad
        self.calls.append(int(tokens[0, 1]))
        p = self.prompts[int(tokens[0, 1]) - 10]
        k = (p["rendering_index"] - 1) // 4 + 1 if self.mode == "unequal" else 1
        h = torch.tensor([0.0, 3.0 * k, 4.0 * k] + [0.0] * 1021).repeat(1, tokens.shape[-1], 1)
        for name, function in self.active_hooks:
            assert name == "blocks.10.hook_out"
            h = function(h, hook=SimpleNamespace(name=name))
        x = h[..., 0]
        offset = -0.2 if self.mode == "baseline_success" else 0.2
        semantic = offset + x * (0 if self.mode == "zero" else 1)
        a = semantic if p["preserve_label"] == "A" else -semantic
        other = torch.full_like(a, -20.0)
        if self.mode == "quality_failure":
            other = torch.where(x < 0, torch.full_like(a, 10.0), other)
        logits = torch.stack(
            [a, torch.zeros_like(a), other, *[torch.full_like(a, -20.0) for _ in range(3)]], dim=-1
        )
        if self.mode == "gradient_mismatch" and torch.is_grad_enabled():
            logits = logits.clone()
            logits[..., 0] += 0.01
        if self.mode == "replay_mismatch" and len(self.calls) == 37:
            logits = logits.clone()
            logits[..., 0] += 0.01
        return logits


def execute(output, mode="linear", check=True):
    plan, backend = setup(protocol.build_plan())
    plan["model"]["d_model"] = 1024
    model = PairedToy(plan["prompts"], mode)
    backend.model = model
    backend.encode = lambda prompt: model.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )["input_ids"]
    deadline = time.monotonic() + 60
    forwards = job.ForwardLedger(output, plan["cells"], deadline)
    derivatives = job.Derivatives(torch, output, plan["derivative_cells"], deadline)
    rows, summary = job.evaluate(plan, backend, forwards, derivatives, output)
    verified = checker.verify_data(plan, rows, output) if check else None
    if check:
        checker.compare_summary(summary, verified["summary"])
    return plan, backend, forwards, derivatives, rows, summary, verified


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    output = tmp_path_factory.mktemp("paired_fake_complete")
    return (*execute(output), output)


def test_exact_plan_and_no_real_imports():
    plan = protocol.build_plan()
    checked = checker.verify_plan(plan)
    assert len(plan["cells"]) == 216 and len(plan["derivative_cells"]) == 96
    assert checked["maximum_forwards"] == 216
    assert all("cg_f04" not in p["prompt_id"] for p in plan["prompts"])
    assert not plan["control_ids"] and not plan["transfer_ids"]
    assert plan["config"]["step_cap"] == 0.05 and plan["config"]["total_cap"] == 0.2


def test_first_acceptance_native_interior_vector_and_current_cache(completed):
    _plan, backend, f, d, rows, summary, verified, output = completed
    assert f.attempts == f.completed == len(backend.model.calls) == 48
    assert d.attempts == d.completed == 12 and len(f.skips) == 168
    assert summary["updates"] == summary["attempted_updates"] == 1
    assert summary["final_accepted"] == 12 and summary["candidate_eligible"]
    assert 0 < summary["shared_net"] < 0.2
    assert [x["condition"] for x in summary["paired_objective_trajectory"]] == [
        "baseline",
        "step_1",
        "final",
    ]
    assert (
        summary["paired_objective_trajectory"] == verified["summary"]["paired_objective_trajectory"]
    )
    assert not (output / "comply_vector.json").exists()
    for row in rows:
        assert row["weights_unchanged"] and row["unselected_max_difference"] == 0
        assert len(row["shared_w"]) == 1024
        if row["condition"] == "baseline":
            assert row["shared_w"] == [0.0] * 1024
        if row["condition"].startswith("gradient_") or row["condition"] == "final":
            assert row["maximum_current_logit_difference"] == 0
    assert backend.model.active_hooks == [] and backend.model.weight.grad is None


@pytest.mark.parametrize(
    "mode,forwards,derivatives,reason",
    [
        ("baseline_success", 24, 0, "accepted"),
        ("zero", 36, 12, "method_zero_increment"),
        ("quality_failure", 48, 12, "quality_failure"),
    ],
)
def test_finite_early_and_failed_endpoints_have_twelve_replays(
    tmp_path, mode, forwards, derivatives, reason
):
    _, _, f, d, rows, summary, _ = execute(tmp_path, mode)
    assert f.attempts == forwards and d.attempts == derivatives
    assert f.cursor == 216 and len(f.skips) + len(rows) == 216
    assert summary["stop_reason"] == reason
    assert len([r for r in rows if r["condition"] == "final"]) == 12
    assert summary["candidate_eligible"] == (mode == "baseline_success")


def test_unequal_own_norms_stay_baseline_anchored(tmp_path):
    _, _, f, d, rows, summary, _ = execute(tmp_path, "unequal")
    assert f.attempts <= 216 and d.attempts <= 96
    assert {r["h0_norm"] for r in rows} == {5.0, 10.0, 15.0}
    for r in rows:
        assert r["intended_delta"] == [
            struct.unpack("<f", struct.pack("<f", r["h0_norm"] * x))[0] for x in r["shared_w"]
        ]
    assert summary["shared_net"] <= 0.2 + 1e-12
    assert summary["shared_path"] <= 0.4 + 1e-12


@pytest.mark.parametrize("mode", ["gradient_mismatch", "replay_mismatch"])
def test_inconsistent_current_or_replay_fails_closed(tmp_path, mode):
    with pytest.raises(ValueError, match="identity"):
        execute(tmp_path, mode, check=False)
    assert not (tmp_path / "comply_vector.json").exists()


@pytest.mark.parametrize("field", ["shared_w", "gradient", "baseline_margin", "display_order"])
def test_saved_raw_rows_tampering_is_rejected(completed, field):
    plan, _, _, _, rows, _, _, output = completed
    changed = copy.deepcopy(rows)
    row = next(r for r in changed if r["condition"] == "gradient_1")
    if field in ("shared_w", "gradient"):
        row[field][0] += 0.01
    elif field == "baseline_margin":
        row[field] += 0.01
    else:
        row[field] = "B_then_A"
    with pytest.raises((ValueError, AssertionError)):
        checker.verify_data(plan, changed, output)


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"standard_used_percent": 90, "checked_at_unix": time.time()},
        {"standard_used_percent": 49, "checked_at_unix": 0},
        {"standard_used_percent": float("nan"), "checked_at_unix": time.time()},
    ],
)
def test_usage_missing_stale_or_at_ceiling_fails(value):
    with pytest.raises(ValueError):
        job.checked_usage(value)


def test_worker_cannot_load_without_separate_authority(monkeypatch):
    monkeypatch.delenv(protocol.AUTH_KEY, raising=False)
    monkeypatch.setattr(job.base, "load_backend", lambda _: pytest.fail("real loader reached"))
    with pytest.raises((ValueError, FileNotFoundError)):
        job.worker()


def test_ledger_count_deadline_and_failed_attempt_guards(tmp_path):
    plan = protocol.build_plan()
    f = job.ForwardLedger(tmp_path, plan["cells"], 100, now=lambda: 0)
    for cell in plan["cells"]:
        f.begin(cell)
        f.finish(True)
    assert f.completed == 216
    with pytest.raises(ValueError):
        f.begin(plan["cells"][-1])
    other = tmp_path / "expired"
    other.mkdir()
    expired = job.ForwardLedger(other, plan["cells"], 0, now=lambda: 0)
    with pytest.raises(ValueError, match="deadline"):
        expired.begin(plan["cells"][0])


def test_unquiescent_capture_never_reads_mutable_journals(tmp_path, monkeypatch):
    monkeypatch.setattr(
        job, "PairedBudget", lambda output: SimpleNamespace(write_bytes=lambda *a, **k: None)
    )
    monkeypatch.setattr(
        job.capture, "run_capture", lambda *a, **k: {"status": "INCONCLUSIVE", "quiescent": False}
    )
    monkeypatch.setattr(
        job.engine.recorder, "journal_counts", lambda *a: pytest.fail("unquiescent journal read")
    )
    monkeypatch.setattr(checker, "finalize_recording", lambda b, receipt, result: result)
    value = job.supervise(
        ["fake"], tmp_path, {"standard_used_percent": 49, "checked_at_unix": time.time()}
    )
    assert value["status"] == "INCONCLUSIVE" and value["forward_attempts"] is None


def test_timeout_cap_not_adjustable(tmp_path):
    with pytest.raises(ValueError, match="fixed1200"):
        job.supervise(["fake"], tmp_path, {}, timeout=1201)


def test_maximum_schedule_has_no_required_skip_file(tmp_path, monkeypatch):
    (tmp_path / "analysis.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        job, "PairedBudget", lambda output: SimpleNamespace(write_bytes=lambda *a, **k: None)
    )
    monkeypatch.setattr(
        job.capture, "run_capture", lambda *a, **k: {"status": "complete_valid", "quiescent": True}
    )
    monkeypatch.setattr(
        job.engine.recorder,
        "journal_counts",
        lambda path: (216, 216, False) if path.name == "forward_events.jsonl" else (96, 96, False),
    )
    monkeypatch.setattr(job.base, "read_rows", lambda *a: pytest.fail("absent skip journal read"))
    monkeypatch.setattr(checker, "finalize_recording", lambda b, receipt, result: result)
    value = job.supervise(
        ["fake"], tmp_path, {"standard_used_percent": 49, "checked_at_unix": time.time()}
    )
    assert value["status"] == "complete_valid" and value["skipped_cells"] == 0


def test_config_and_source_lock_reject_tampering(tmp_path, monkeypatch):
    config = protocol.read(protocol.ROOT / protocol.CONFIG)
    config["objective"]["lambda"] = 2
    target = tmp_path / protocol.CONFIG
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="config bytes"):
        protocol.config_at(tmp_path)
    monkeypatch.setattr(protocol, "build_plan", lambda: {"input_sha256": {"scoped.py": "unused"}})
    monkeypatch.setattr(protocol, "SOURCE_PATHS", ())

    def fake_git(command, **kwargs):
        if command[1] == "ls-files":
            return b"100644 abc 0\tscoped.py\0"
        return b"scoped.py\0"

    monkeypatch.setattr(job.subprocess, "check_output", fake_git)
    with pytest.raises(ValueError, match="clean tracked"):
        job.source_identity()


def test_changed_runtime_writer_and_independent_finalizer_seal_together(tmp_path):
    budget = job.PairedBudget(tmp_path)
    with job.bindings.bind_writers(budget):
        plan, _, _, _, rows, summary, _ = execute(tmp_path, check=False)
        job.base.write_new(tmp_path / "analysis.json", summary)
    verified = checker.verify_data(plan, rows, tmp_path)
    verified.update(
        audited_endpoint_sha256=protocol.sha((tmp_path / "endpoint.json").read_bytes()),
        audited_result_sha256=protocol.sha((tmp_path / "result.json").read_bytes()),
    )
    receipt = {"status": "complete_valid", "quiescent": True}
    closed = checker.finalize_recording(
        budget, receipt, {"status": "complete_valid"}, audit=lambda: verified
    )
    assert closed["status"] == "complete_valid" and closed["candidate_eligible"]
    sealed = checker.read_sealed_recording(tmp_path)
    assert sealed["recording_inventory"]["inventory"]["valid_candidate"]
    assert (
        sealed["summary"]["paired_objective_trajectory"]
        == verified["summary"]["paired_objective_trajectory"]
    )
    vector = protocol.read(tmp_path / "comply_vector.json")["vector"]
    assert 0 < job.norm(vector) < 0.2
    assert vector == protocol.read(tmp_path / "endpoint.json")["w"]
