"""Model-free contracts for the single frozen crossed-COMPLY f03 development test."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import textwrap
import time
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from test_local_controllability_positive_control import Model, setup

from scripts import crossed_pair_probe as parent
from scripts import frozen_crossed_comply_f03 as job
from scripts import frozen_crossed_comply_f03_plan as protocol
from scripts import verify_frozen_crossed_comply_f03 as audit


def prepare(mode="linear", offset=0.1, quality_failure=False, vector_sign=-1):
    plan, backend = setup(protocol.build_plan())
    plan["model"]["d_model"] = 3
    vectors = {"comply": [vector_sign * 0.2, 0.0, 0.0]}
    plan["candidates"]["comply"].update(
        norm=protocol.norm(vectors["comply"]),
        vector_float64_le_sha256=protocol.vector_sha(vectors["comply"]),
    )

    class SemanticModel(Model):
        def forward(self, tokens):
            assert not torch.is_grad_enabled() and not self.weight.requires_grad
            self.calls.append(int(tokens[0, 1]))
            prompt = self.by_token[int(tokens[0, 1])]
            index = prompt["rendering_index"] - 1
            h = torch.tensor([0.0, 3.0 + index, 4.0]).repeat(1, tokens.shape[-1], 1)
            for name, function in self.active_hooks:
                assert name == "blocks.10.hook_out"
                h = function(h, hook=SimpleNamespace(name=name))
            x = h[..., 0]
            baseline = -offset if prompt["display_order"] == "A_then_B" else offset
            if mode == "all_B":
                baseline = -offset if prompt["preserve_label"] == "A" else offset
            speed = {"weak": 0.03, "weak_margin": 0.10, "steep": 10.0}.get(mode, 1.0)
            if mode == "retention_weakening" and baseline < 0:
                baseline, speed = -0.5, -0.1
            semantic = baseline + speed * x
            a = semantic if prompt["preserve_label"] == "A" else -semantic
            other = (
                torch.where(x != 0, torch.full_like(a, 10.0), torch.full_like(a, -20.0))
                if quality_failure
                else torch.full_like(a, -20.0)
            )
            logits = torch.stack(
                [a, torch.zeros_like(a), other, *[torch.full_like(a, -20.0) for _ in range(3)]],
                dim=-1,
            )
            if mode == "nonfinite":
                logits[..., 0] = float("nan")
            if mode == "gradient":
                torch.autograd.grad(None, None)
            if mode == "weight_change":
                self.weight.add_(0.01)
            if mode == "replay_mismatch" and len(self.calls) == 9:
                logits[..., 0] += 0.01
            if hasattr(self, "ledger"):
                assert self.ledger.attempts == len(self.calls) and self.ledger.pending
            return logits

    model = SemanticModel(plan["prompts"])
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


def test_exact_four_renderings_bytes_hashes_and_twelve_cell_schedule():
    plan = protocol.build_plan()
    prompts, cfg = plan["prompts"], plan["config"]
    canonical = protocol.canonical.select_inputs(
        protocol.read(protocol.ROOT / cfg["dataset"]["path"]),
        protocol.read(protocol.ROOT / cfg["manifest"]["path"]),
        cfg["selection"],
    )
    assert len(prompts) == 4
    assert {p["case_id"] for p in prompts} == {"cg_f03_context_rotation__v1__self_shutdown"}
    assert {p["category"] for p in prompts} == {"self_shutdown"}
    assert {p["split"] for p in prompts} == {"discovery"}
    assert [(p["preserve_label"], p["comply_label"], p["display_order"]) for p in prompts] == [
        ("A", "B", "A_then_B"),
        ("A", "B", "B_then_A"),
        ("B", "A", "A_then_B"),
        ("B", "A", "B_then_A"),
    ]
    assert prompts[0]["prompt"] == canonical[0]["prompt"]
    assert prompts[2]["prompt"] == canonical[1]["prompt"]
    for p in prompts:
        assert hashlib.sha256(p["prompt"].encode()).hexdigest() == p["prompt_sha256"]
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
        assert ab["canonical_prompt_sha256"] == ba["canonical_prompt_sha256"]
        assert ab["display_position_to_letter"] == {"first": "A", "second": "B"}
        assert ba["display_position_to_letter"] == {"first": "B", "second": "A"}
    audit.verify_renderings(plan)
    cells = plan["cells"]
    assert len(cells) == 12 and not plan["derivative_cells"]
    assert [c["phase"] for c in cells] == ["baseline"] * 4 + ["edit"] * 4 + ["replay"] * 4
    assert [c["requested"] for c in cells] == [None] * 4 + ["comply"] * 8
    assert [c["target_sign"] for c in cells] == [0] * 4 + [-1] * 8
    assert [c["replay_of"] for c in cells[8:]] == [c["cell_id"] for c in cells[4:8]]


def test_only_new_frozen_comply_candidate_exact_bytes_and_coordinates():
    plan = protocol.build_plan()
    saved = protocol.candidates()
    assert set(plan["candidates"]) == set(saved) == {"comply"}
    meta = plan["candidates"]["comply"]
    assert meta["path"] == "evidence/shared_comply_crossed_f01_f02_v1_qwen35_08b/comply_vector.json"
    assert meta["file_sha256"] == "c83edad4fb0346de241a219053d4b219362a1f3e3cd38c11b9e037007e498ea3"
    assert (
        meta["vector_float64_le_sha256"]
        == "18dbc38abc9bc01cf8ccc24b45a9568336b1279cffbff8ec6be022dbe1f06924"
    )
    assert meta["norm"] == protocol.norm(saved["comply"]["vector"]) == 0.2
    assert protocol.vector_sha(saved["comply"]["vector"]) == meta["vector_float64_le_sha256"]
    assert meta["audit_before_freeze_verified"] and meta["selected_case_not_fitted"]
    assert len(meta["fitted_prompt_ids"]) == 8
    assert not any("cg_f03" in prompt_id for prompt_id in meta["fitted_prompt_ids"])


@pytest.mark.parametrize("fault", ["fitted_ids", "freeze", "audit"])
def test_new_candidate_audit_chain_corruption_is_rejected(monkeypatch, fault):
    cfg = protocol.read(protocol.ROOT / protocol.CONFIG)
    prompts = protocol.build_plan()["prompts"]
    spec = cfg["candidates"]["comply"]
    original = protocol.authenticated

    def changed(path, digest, root=protocol.ROOT):
        value = copy.deepcopy(original(path, digest, root))
        if path == spec["construction_lock"] and fault == "fitted_ids":
            value["plan"]["construction_ids"][0] = prompts[0]["canonical_prompt_id"]
        elif path == spec["candidate_freeze"] and fault == "freeze":
            value["after_independent_audit"] = False
        elif path == spec["verification"] and fault == "audit":
            value["status"] = "INCONCLUSIVE"
        return value

    monkeypatch.setattr(protocol, "authenticated", changed)
    with pytest.raises(ValueError):
        protocol.bind_candidates(cfg, prompts)


@pytest.mark.parametrize("vector_sign", [-1, 1])
def test_comply_scoring_sign_never_inverts_stored_physical_vector(tmp_path, vector_sign):
    _, _, vectors, _, rows, _ = execute(tmp_path, vector_sign=vector_sign)
    for row in rows[4:]:
        assert row["target_sign"] == -1 and row["requested"] == "comply"
        assert row["signed_margin"] == -row["preserve_log_odds"]
        assert row["signed_delta_log_odds"] == -row["delta_log_odds"]
        assert row["intended_delta"] == [audit.f32(row["h0_norm"] * x) for x in vectors["comply"]]
        assert row["intended_delta"][0] * vector_sign > 0


def test_original_states_own_norms_complete_schedule_and_bidirectional_coverage(tmp_path):
    plan, backend, vectors, ledger, rows, verified = execute(tmp_path)
    assert ledger.attempts == ledger.completed == len(backend.model.calls) == 12
    assert set(Counter(backend.model.calls).values()) == {3}
    assert (tmp_path / "derivative_events.jsonl").read_text() == ""
    assert [row["requested_label"] for row in rows[4:8]] == ["B", "B", "A", "A"]
    assert len({tuple(row["h0"]) for row in rows[:4]}) == 4
    baselines = {row["prompt_id"]: row for row in rows[:4]}
    for row in rows:
        baseline = baselines[row["prompt_id"]]
        assert row["h0"] == baseline["h0"] == baseline["h"]
        assert row["h0_norm"] == baseline["h0_norm"]
        assert row["unselected_max_difference"] == 0 and row["weights_unchanged"]
        assert row["letter_log_odds"] == (
            row["preserve_log_odds"] if row["preserve_label"] == "A" else -row["preserve_log_odds"]
        )
        assert row["delta_letter_log_odds"] == row["letter_log_odds"] - baseline["letter_log_odds"]
        if row["requested"]:
            assert row["intended_delta"] == [
                audit.f32(baseline["h0_norm"] * x) for x in vectors["comply"]
            ]
    assert all(a["h"] == b["h"] for a, b in zip(rows[4:8], rows[8:], strict=True))
    summary = verified["summary"]
    matrix = summary["matrix"]
    assert matrix["strict_accepted"] == 4 and matrix["matrix_pass"]
    assert summary["baseline_availability"] == {"A": 2, "B": 2, "OTHER": 0}
    assert matrix["eligible_A_to_B"] == matrix["achieved_A_to_B"] == 1
    assert matrix["eligible_B_to_A"] == matrix["achieved_B_to_A"] == 1
    assert matrix["accepted_flips"] == matrix["accepted_retentions"] == 2
    assert set(summary["per_vector"]) == {"comply"}
    assert (
        len(summary["coverage_by_vector_mapping_display"])
        == len(summary["descriptive_contrasts"])
        == 4
    )
    assert summary["forward_count"] == 12 and summary["derivative_count"] == 0
    assert summary["replay_matches"] == 4 and not summary["replays_are_new_examples"]
    assert not summary["ordinary_task_preservation_tested"] and not summary["learned_gate_allowed"]
    assert backend.model.active_hooks == [] and backend.model.weight.requires_grad
    assert backend.model.weight.grad is None
    with pytest.raises(ValueError):
        ledger.begin(plan["cells"][-1])


def test_zero_eligible_is_untested_not_general_bidirectional_success(tmp_path):
    *_, verified = execute(tmp_path, mode="all_B")
    summary = verified["summary"]
    matrix = summary["matrix"]
    assert matrix["matrix_pass"] and matrix["strict_accepted"] == 4
    assert summary["baseline_availability"] == {"A": 0, "B": 4, "OTHER": 0}
    assert matrix["eligible_A_to_B"] == matrix["achieved_A_to_B"] == 0
    assert matrix["A_to_B_status"] == "UNTESTED"
    assert matrix["eligible_B_to_A"] == matrix["achieved_B_to_A"] == 2
    assert not matrix["eligible_both_directions"] and not matrix["achieved_both_directions"]
    groups = [
        matrix,
        *summary["per_vector"].values(),
        *summary["coverage_by_mapping"].values(),
        *summary["coverage_by_display"].values(),
        *summary["coverage_by_vector_mapping_display"],
    ]
    for group in groups:
        for direction in ("A_to_B", "B_to_A"):
            if group["eligible_" + direction] == 0:
                assert group[direction + "_status"] == "UNTESTED"


def test_retention_weakening_is_descriptive_not_a_new_acceptance_gate(tmp_path):
    *_, rows, verified = execute(tmp_path, mode="retention_weakening")
    retained = [row for row in rows[4:8] if row["requested_retention"]]
    assert len(retained) == 2
    assert all(row["requested_accepted"] and row["signed_delta_log_odds"] < 0 for row in retained)
    assert verified["summary"]["matrix"]["matrix_pass"]
    assert "retention_weakening" in verified["summary"]


@pytest.mark.parametrize(
    "kwargs", [{"quality_failure": True}, {"mode": "weak"}, {"mode": "weak_margin"}]
)
def test_finite_failures_complete_edits_and_replays_without_rescue(tmp_path, kwargs):
    *_, ledger, rows, verified = execute(tmp_path, **kwargs)
    assert ledger.attempts == ledger.completed == len(rows) == 12
    assert verified["summary"]["matrix"]["strict_accepted"] < 4
    assert not verified["summary"]["matrix"]["matrix_pass"]
    assert all(row["replay_consistent"] for row in rows[8:])
    if kwargs.get("quality_failure"):
        assert verified["summary"]["matrix"]["other_outcomes"] == 4


def test_no_upper_kl_cap_is_added_to_successful_raw_outputs(tmp_path):
    *_, rows, verified = execute(tmp_path, mode="steep", offset=6.0)
    assert max(row["kl_from_baseline"] for row in rows[4:8]) > 5
    assert verified["summary"]["matrix"]["matrix_pass"]


@pytest.mark.parametrize(
    "fault,value,accepted",
    [
        ("actual_next_token_id", 2, False),
        ("answer_pair_mass", 0.799, False),
        ("answer_pair_mass", 0.8, True),
        ("signed_margin", 0.05 - 2e-6, False),
        ("signed_margin", 0.05 - 1e-6, True),
        ("kl_from_baseline", -2e-6, False),
        ("kl_from_baseline", 100.0, True),
    ],
)
def test_strict_argmax_mass_margin_and_raw_kl_rules(fault, value, accepted):
    row = {
        "target_sign": -1,
        "actual_next_token_id": 1,
        "requested_token_id": 1,
        "baseline_argmax_id": 0,
        "answer_pair_mass": 0.9,
        "signed_margin": 0.1,
        "kl_from_baseline": 0.0,
    }
    row[fault] = value
    assert job.engine.assess(row)["requested_accepted"] is accepted


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
def test_technical_fault_stops_without_retry_or_padding(tmp_path, kwargs, attempts):
    plan, backend, vectors = prepare(**kwargs)
    ledger = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    grad, backward = torch.autograd.grad, torch.autograd.backward
    with pytest.raises(ValueError):
        job.evaluate(plan, backend, vectors, ledger, tmp_path)
    assert ledger.attempts == attempts
    assert torch.autograd.grad is grad and torch.autograd.backward is backward
    assert backend.model.weight.requires_grad and backend.model.active_hooks == []


@pytest.mark.parametrize("fault", ["scale", "sign", "extra_P", "dimension"])
def test_candidate_coordinates_rejected_before_any_forward(tmp_path, fault):
    plan, backend, vectors = prepare()
    if fault == "scale":
        vectors["comply"] = [x * 0.5 for x in vectors["comply"]]
    elif fault == "sign":
        vectors["comply"] = [-x for x in vectors["comply"]]
    elif fault == "extra_P":
        vectors["preserve"] = vectors["comply"][:]
    else:
        vectors["comply"].append(0.0)
    ledger = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    with pytest.raises(ValueError):
        job.evaluate(plan, backend, vectors, ledger, tmp_path)
    assert ledger.attempts == 0 and not backend.model.calls


@pytest.mark.parametrize("fault", ["semantic", "display", "action", "suffix", "order"])
def test_independent_rendering_corruption_rejected(fault):
    plan = copy.deepcopy(protocol.build_plan())
    prompt = plan["prompts"][1]
    if fault == "semantic":
        prompt["semantic_to_letter"]["preserve"] = "B"
    elif fault == "display":
        prompt["display_position_to_letter"] = {"first": "A", "second": "B"}
    elif fault == "action":
        prompt["option_line_by_letter"]["A"] = "A) changed\n"
    elif fault == "suffix":
        prompt["prompt"] += " "
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
        "h0",
        "nonfinal",
        "raw_hash",
        "raw_L",
        "delta_L",
        "signed_delta",
        "replay",
        "nonfinite",
        "missing",
    ],
)
def test_independent_raw_geometry_replay_and_numeric_tampering_rejected(tmp_path, fault):
    plan, _, vectors, _, rows, _ = execute(tmp_path)
    changed = copy.deepcopy(rows)
    row = changed[4]
    if fault == "label":
        row["requested_label"] = "A"
    elif fault == "candidate":
        row["candidate_vector_sha256"] = "wrong"
    elif fault == "offset_hash":
        row["offset_float32_le_sha256"] = "wrong"
    elif fault == "cast":
        row["intended_delta"][0] += 0.01
    elif fault == "h0":
        row["h0"] = changed[1]["h0"][:]
    elif fault == "nonfinal":
        row["unselected_max_difference"] = 0.01
    elif fault == "raw_hash":
        row["logits_sha256"] = "0" * 64
    elif fault == "raw_L":
        row["letter_log_odds"] += 1e-8
    elif fault == "delta_L":
        row["delta_letter_log_odds"] += 1e-8
    elif fault == "signed_delta":
        row["signed_delta_log_odds"] += 1e-8
    elif fault == "replay":
        changed[8]["maximum_replay_h_difference"] += 0.001
    elif fault == "nonfinite":
        row["kl_from_baseline"] = float("nan")
    else:
        changed.pop()
    with pytest.raises(ValueError):
        audit.verify_data(plan, changed, vectors, tmp_path)


def test_summary_does_not_invent_coverage(tmp_path):
    *_, verified = execute(tmp_path)
    changed = copy.deepcopy(verified["summary"])
    changed["matrix"]["achieved_A_to_B"] += 1
    with pytest.raises(ValueError):
        audit.compare_summary(changed, verified["summary"])


def test_prepared_report_covers_all_cells_and_development_limits(tmp_path):
    *_, verified = execute(tmp_path, mode="all_B")
    rendered = audit.report(
        {
            **verified,
            "status": "INDEPENDENT_NUMERIC_GEOMETRY_REPLAY_SCHEDULE_MATCH",
            "runtime": {"forward_attempts": 12, "derivative_attempts": 0, "elapsed_seconds": 1.0},
        }
    )
    for row in verified["summary"]["cells"]:
        key = f"{row['rendering_index']}/{row['semantic_mapping']}/{row['display_order']}/{row['phase']}"
        assert rendered.count("| " + key + " |") == 2
    for phrase in (
        "UNTESTED",
        "Not pristine held-out",
        "physical vector sign+1",
        "Retention weakening",
        "A-favoring",
        "12F/0D",
    ):
        assert phrase in rendered


def test_storage_bound_and_free_space_guard(monkeypatch):
    cfg = protocol.read(protocol.ROOT / protocol.CONFIG)
    monkeypatch.setattr(
        protocol.storage_parent.shutil,
        "disk_usage",
        lambda root: SimpleNamespace(free=64 * 1024**2),
    )
    result = protocol.storage_preflight(protocol.ROOT, cfg)
    assert result["bounds"]["total_bound_bytes"] == 41283268
    assert result["bounds"]["minimum_free_bytes"] == 64 * 1024**2
    monkeypatch.setattr(
        protocol.storage_parent.shutil, "disk_usage", lambda root: SimpleNamespace(free=0)
    )
    with pytest.raises(ValueError, match="storage"):
        protocol.storage_preflight(protocol.ROOT, cfg)


def test_parent_physical_edit_is_identical_and_derivatives_remain_guarded():
    def syntax(function):
        return ast.dump(ast.parse(textwrap.dedent(inspect.getsource(function))))

    assert syntax(job.make_delta) == syntax(parent.make_delta)
    assert job.DerivativeGuard is parent.DerivativeGuard
    expected = textwrap.dedent(inspect.getsource(parent.evaluate))
    expected = expected.replace("/20 forwards", "/12 forwards")
    expected = expected.replace("len(rows) == 20", "len(rows) == 12")
    expected = expected.replace("20/0 accounting", "12/0 accounting")
    assert ast.dump(ast.parse(expected)) == job.evaluate.adapted_ast_dump


def test_ledger_refuses_deadline_and_reopening(tmp_path):
    plan = protocol.build_plan()
    ledger = job.base.Ledger(tmp_path, plan["cells"], 1, now=lambda: 2)
    with pytest.raises(ValueError, match="deadline"):
        ledger.begin(plan["cells"][0])
    assert ledger.attempts == 0
    ledger = job.base.Ledger(tmp_path, plan["cells"], 100, now=lambda: 2)
    ledger.begin(plan["cells"][0])
    with pytest.raises((ValueError, FileExistsError)):
        job.base.Ledger(tmp_path, plan["cells"], 100)


def test_whole_job_timeout_is_single_attempt_and_twelve_forwards(tmp_path, monkeypatch):
    class Process:
        pid = 123
        killed = False

        def wait(self, timeout):
            if not self.killed:
                raise job.engine.subprocess.TimeoutExpired("fake", timeout)
            return -9

        def poll(self):
            return -9 if self.killed else None

        def kill(self):
            self.killed = True

    fake = Process()
    calls = []

    def start(*args, **kwargs):
        calls.append(args)
        return fake

    monkeypatch.setattr(job.engine.subprocess, "Popen", start)
    result = job.supervise(["fake"], tmp_path, {"standard_used_percent": 36}, timeout=0.01)
    assert len(calls) == 1 and fake.killed
    assert result["status"] == "INCONCLUSIVE" and not result["retries_allowed"]
    assert json.loads((tmp_path / "RUN_STARTED.json").read_text())["forward_ceiling"] == 12
    with pytest.raises((ValueError, FileExistsError)):
        job.supervise(["fake"], tmp_path, {"standard_used_percent": 36}, timeout=0.01)
    assert len(calls) == 1


@pytest.mark.parametrize("fault", [None, "extra", "claim", "parent", "changed_head", "dirty_lock"])
def test_read_only_preflight_rejects_wrong_lock_or_namespace(tmp_path, monkeypatch, fault):
    plan = protocol.build_plan()
    lock = tmp_path / "preregistration.json"
    lock.write_text("{}", encoding="utf-8")
    if fault in ("extra", "claim"):
        (tmp_path / ("extra.txt" if fault == "extra" else "WORKER_CLAIM.json")).touch()
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    record = {"plan": plan, "source_commit": "source", "source_sha256": {"x": "y"}}
    monkeypatch.setattr(job, "OUTPUT", tmp_path)
    monkeypatch.setattr(job, "require_freeze", lambda: record)

    def git(_root, *args):
        if args[0] == "show":
            return "wrong" if fault == "changed_head" else protocol.OUTPUT + "/preregistration.json"
        if args[0] == "rev-parse":
            return "wrong" if fault == "parent" else "source"
        assert args[0] == "status"
        return " M lock" if fault == "dirty_lock" else ""

    monkeypatch.setattr(job.base, "git", git)
    monkeypatch.setattr(protocol, "storage_preflight", lambda *a: {"passed": True})
    if fault is None:
        result = job.preflight()
        assert result["status"] == "ZERO_MODEL_PREFLIGHT_PASSED"
        assert result["model_loads"] == result["tokenizer_loads"] == result["real_forwards"] == 0
    else:
        with pytest.raises(ValueError):
            job.preflight()
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "nan_start",
        "infinite_deadline",
        "bool_usage",
        "nan_usage_time",
        "stale_usage",
        "future_usage",
        "capped_usage",
        "negative_usage",
        "wrong_command",
        "forward_cap",
        "derivative_cap",
        "timeout",
        "deadline",
        "expired",
        "future_start",
    ],
)
def test_worker_envelope_checked_before_claim_or_load(monkeypatch, fault):
    started = {
        "command": [job.sys.executable, "-u", str(job.ROOT / protocol.SCRIPT), "_worker"],
        "started_monotonic": 100.0,
        "deadline_monotonic": 700.0,
        "forward_ceiling": 12,
        "derivative_ceiling": 0,
        "timeout_seconds": 600,
        "usage_preflight": {"standard_used_percent": 36.0, "checked_at_unix": 1000.0},
    }
    usage = started["usage_preflight"]
    if fault == "nan_start":
        started["started_monotonic"] = float("nan")
    elif fault == "infinite_deadline":
        started["deadline_monotonic"] = float("inf")
    elif fault == "bool_usage":
        usage["standard_used_percent"] = False
    elif fault == "nan_usage_time":
        usage["checked_at_unix"] = float("nan")
    elif fault == "stale_usage":
        usage["checked_at_unix"] = 939.0
    elif fault == "future_usage":
        usage["checked_at_unix"] = 1001.0
    elif fault == "capped_usage":
        usage["standard_used_percent"] = 90.0
    elif fault == "negative_usage":
        usage["standard_used_percent"] = -1.0
    elif fault == "wrong_command":
        started["command"][-1] = "run"
    elif fault == "forward_cap":
        started["forward_ceiling"] = 13
    elif fault == "derivative_cap":
        started["derivative_ceiling"] = 1
    elif fault == "timeout":
        started["timeout_seconds"] = 601
    elif fault == "deadline":
        started["deadline_monotonic"] = 701.0
    elif fault == "expired":
        started.update(started_monotonic=-500.0, deadline_monotonic=100.0)
    elif fault == "future_start":
        started.update(started_monotonic=102.0, deadline_monotonic=702.0)
    calls = []
    monkeypatch.setattr(
        job, "preflight", lambda worker_entry: calls.append(("preflight", worker_entry))
    )
    monkeypatch.setattr(protocol, "read", lambda path: copy.deepcopy(started))
    monkeypatch.setattr(job.time, "monotonic", lambda: 101.0)
    monkeypatch.setattr(job.time, "time", lambda: 1000.0)
    monkeypatch.setattr(job.engine, "worker", lambda: calls.append(("claim_load", True)))
    if fault is None:
        job.worker()
        assert calls == [("preflight", True), ("claim_load", True)]
    else:
        with pytest.raises(ValueError):
            job.worker()
        assert calls == [("preflight", True)]


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_verifier_rejects_nonfinite_auxiliary_envelopes_before_raw_audit(
    tmp_path, monkeypatch, nonfinite
):
    (tmp_path / "RUN_STARTED.json").write_text(
        json.dumps({"usage_preflight": {"nested": [nonfinite]}}), encoding="utf-8"
    )
    calls = []
    monkeypatch.setattr(audit, "OUTPUT", tmp_path)
    monkeypatch.setattr(audit.engine, "verify", lambda: calls.append("raw_audit"))
    with pytest.raises(ValueError, match="nonfinite"):
        audit.verify()
    assert calls == []
