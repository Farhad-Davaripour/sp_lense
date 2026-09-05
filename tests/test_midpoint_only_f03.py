"""Synthetic midpoint-only diagnostics: null semantic target is not physical OFF."""

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
from scripts import midpoint_only_f03 as probe
from scripts import midpoint_only_f03_plan as protocol
from scripts import verify_crossed_pair_probe as parent_checker
from scripts import verify_midpoint_only_f03 as checker

AMPLITUDE = 0.11950278487420843
NULL_FIELDS = (
    "requested",
    "target_sign",
    "requested_label",
    "requested_token_id",
    "signed_margin",
    "signed_delta_log_odds",
)
FORBIDDEN_ACCEPTANCE_FIELDS = (
    "requested_accepted",
    "requested_argmax",
    "new_requested_flip",
    "requested_retention",
)


def prepare(mode="linear", offset=0.1):
    plan, backend = setup(protocol.build_plan())
    plan["model"]["d_model"] = 3
    plan["config"]["rendered_prompts"] = copy.deepcopy(plan["prompts"])
    vector = [AMPLITUDE, 0.0, 0.0]
    vectors = {"midpoint": vector}
    plan["candidates"]["midpoint"].update(
        norm=protocol.norm(vector),
        vector_float64_le_sha256=protocol.vector_sha(vector),
        stored_vector_float64_le_sha256=protocol.vector_sha(vector),
    )

    class MidpointModel(Model):
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
            speed = -1.0 if mode == "negative_movement" else 0.0 if mode == "no_movement" else 1.0
            semantic = baseline + speed * x
            a = semantic if prompt["preserve_label"] == "A" else -semantic
            other = torch.full_like(a, -20.0)
            if mode in ("OTHER", "low_mass"):
                edited_other = 10.0 if mode == "OTHER" else -1.0
                other = torch.where(x != 0, torch.full_like(a, edited_other), other)
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

    model = MidpointModel(plan["prompts"])
    model.by_token = {model.tokenizer.ids[p["prompt"]]: p for p in plan["prompts"]}
    backend.model = model
    return plan, backend, vectors


def compare_summary(actual, expected, field=""):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            compare_summary(actual[key], value, key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for x, y in zip(actual, expected, strict=True):
            compare_summary(x, y, field)
    elif type(expected) is float and field in ("answer_pair_mass", "kl_from_baseline"):
        assert actual == pytest.approx(expected, abs=2e-5, rel=0)
    else:
        assert type(actual) is type(expected) and actual == expected


def execute(output, **kwargs):
    plan, backend, vectors = prepare(**kwargs)
    ledger = probe.base.Ledger(output, plan["cells"], time.monotonic() + 100)
    backend.model.ledger = ledger
    rows = probe.evaluate(plan, backend, vectors, ledger, output)
    verified = checker.verify_data(plan, rows, vectors, output)
    compare_summary(probe.summarize(rows), verified["summary"])
    return plan, backend, vectors, ledger, rows, verified


def test_exact_four_crossed_prompts_and_null_target_physical_on_schedule():
    plan = protocol.build_plan()
    prompts, cells = plan["prompts"], plan["cells"]
    assert len(prompts) == 4 and len(cells) == 12 and not plan["derivative_cells"]
    assert {p["case_id"] for p in prompts} == {"cg_f03_context_rotation__v1__self_shutdown"}
    assert [(p["preserve_label"], p["comply_label"], p["display_order"]) for p in prompts] == [
        ("A", "B", "A_then_B"),
        ("A", "B", "B_then_A"),
        ("B", "A", "A_then_B"),
        ("B", "A", "B_then_A"),
    ]
    for p in prompts:
        assert hashlib.sha256(p["prompt"].encode()).hexdigest() == p["prompt_sha256"]
    for index in (0, 2):
        ab, ba = prompts[index : index + 2]
        lines = ab["prompt"].splitlines(keepends=True)
        ai = next(i for i, line in enumerate(lines) if line.startswith("A) "))
        bi = next(i for i, line in enumerate(lines) if line.startswith("B) "))
        expected = lines[:]
        expected[ai], expected[bi] = lines[bi], lines[ai]
        assert ba["prompt"] == "".join(expected)
        assert ab["option_line_by_letter"] == ba["option_line_by_letter"]
        assert ab["semantic_to_letter"] == ba["semantic_to_letter"]
    assert [c["phase"] for c in cells] == ["baseline"] * 4 + ["edit"] * 4 + ["replay"] * 4
    assert all(c["requested"] is None and c["target_sign"] is None for c in cells)
    assert [c["intervention"] for c in cells] == [None] * 4 + ["midpoint"] * 8
    assert [c["intervention_on"] for c in cells] == [False] * 4 + [True] * 8
    assert [c["physical_sign"] for c in cells] == [0] * 4 + [1] * 8
    assert [c["replay_of"] for c in cells[8:]] == [c["cell_id"] for c in cells[4:8]]
    checker.verify_renderings(plan)


def test_exact_midpoint_file_hash_norm_and_no_other_candidate():
    plan, saved = protocol.build_plan(), protocol.candidates()
    assert set(plan["candidates"]) == set(saved) == {"midpoint"}
    meta, vector = plan["candidates"]["midpoint"], saved["midpoint"]["vector"]
    assert meta["path"] == "evidence/centered_pc_geometry_v1_qwen35_08b/midpoint.json"
    assert meta["file_sha256"] == "94d858ef364fd907337d112aa5672b47802289667f39d1516eaad720a0119c5b"
    assert (
        meta["vector_float64_le_sha256"]
        == protocol.vector_sha(vector)
        == "c6cf338185ff98262afba2eb137cf64bc009e17d5d29bad542ec11464308f26d"
    )
    assert meta["norm"] == protocol.norm(vector) == AMPLITUDE
    assert meta["norm"] != 0.2


def test_null_targets_do_not_turn_edits_off_and_own_original_state_is_used(tmp_path):
    plan, backend, vectors, ledger, rows, verified = execute(tmp_path)
    assert ledger.attempts == ledger.completed == len(backend.model.calls) == 12
    assert set(Counter(backend.model.calls).values()) == {3}
    assert (tmp_path / "derivative_events.jsonl").read_text() == ""
    baselines = {r["prompt_id"]: r for r in rows[:4]}
    assert len({tuple(r["h0"]) for r in rows[:4]}) == 4
    for row in rows:
        baseline = baselines[row["prompt_id"]]
        assert row["h0"] == baseline["h0"] == baseline["h"]
        assert row["h0_norm"] == baseline["h0_norm"]
        assert row["unselected_max_difference"] == 0 and row["weights_unchanged"]
        assert all(row[field] is None for field in NULL_FIELDS)
        assert all(field not in row for field in FORBIDDEN_ACCEPTANCE_FIELDS)
        on = row["phase"] != "baseline"
        assert row["intervention_on"] is on and row["physical_intervention_on"] is on
        assert row["physical_nonzero"] is on
        assert row["physical_sign"] == int(on)
        expected = (
            [checker.f32(baseline["h0_norm"] * x) for x in vectors["midpoint"]] if on else [0.0] * 3
        )
        assert row["intended_delta"] == expected
        if on:
            assert row["actual_norm"] > 0 and row["intended_delta"][0] > 0
            assert row["frozen_vector_norm"] == AMPLITUDE
    assert all(a["h"] == b["h"] for a, b in zip(rows[4:8], rows[8:], strict=True))
    summary = verified["summary"]
    assert summary["status"] == "MIDPOINT_ONLY_F03_DIAGNOSTIC_COMPLETED"
    assert summary["baseline_count"] == summary["edit_count"] == summary["replay_matches"] == 4
    assert summary["forward_count"] == 12 and summary["derivative_count"] == 0
    assert "matrix" not in summary and "per_vector" not in summary
    for row in summary["cells"]:
        assert all(field not in row for field in FORBIDDEN_ACCEPTANCE_FIELDS)
    assert backend.model.active_hooks == [] and backend.model.weight.requires_grad
    assert backend.model.weight.grad is None
    with pytest.raises(ValueError):
        ledger.begin(plan["cells"][-1])


@pytest.mark.parametrize("mode", ["OTHER", "low_mass", "negative_movement", "no_movement"])
def test_finite_descriptive_outcomes_complete_all_twelve_without_acceptance_gate(tmp_path, mode):
    *_, ledger, rows, verified = execute(tmp_path, mode=mode)
    assert ledger.attempts == ledger.completed == len(rows) == 12
    assert verified["summary"]["status"] == "MIDPOINT_ONLY_F03_DIAGNOSTIC_COMPLETED"
    assert all(r["replay_consistent"] for r in rows[8:])
    assert all(r["physical_nonzero"] for r in rows[4:])
    if mode == "OTHER":
        assert all(
            r["actual_next_token_label"] == "OTHER" and not r["quality_valid"] for r in rows[4:8]
        )
    elif mode == "low_mass":
        assert any(
            r["answer_pair_mass"] < 0.8 and r["actual_next_token_label"] != "OTHER"
            for r in rows[4:8]
        )
    elif mode == "negative_movement":
        assert all(r["delta_log_odds"] < 0 for r in rows[4:8])
    else:
        assert all(r["delta_log_odds"] == 0 and not r["actual_argmax_changed"] for r in rows[4:8])


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
def test_technical_faults_stop_without_padding_or_retry(tmp_path, kwargs, attempts):
    plan, backend, vectors = prepare(**kwargs)
    ledger = probe.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    grad, backward = torch.autograd.grad, torch.autograd.backward
    with pytest.raises(ValueError):
        probe.evaluate(plan, backend, vectors, ledger, tmp_path)
    assert ledger.attempts == attempts
    assert torch.autograd.grad is grad and torch.autograd.backward is backward
    assert backend.model.weight.requires_grad and backend.model.active_hooks == []


@pytest.mark.parametrize(
    "fault",
    ["all_off", "semantic_target", "negative_sign", "scale", "wrong_candidate", "dimension"],
)
def test_unapproved_noop_target_sign_scale_and_candidate_rejected_before_calls(tmp_path, fault):
    plan, backend, vectors = prepare()
    if fault == "all_off":
        for cell in plan["cells"]:
            cell.update(intervention=None, intervention_on=False, physical_sign=0)
            cell["cell_sha256"] = protocol.canonical_sha(
                {key: value for key, value in cell.items() if key != "cell_sha256"}
            )
        assert all(
            cell["cell_sha256"]
            == protocol.canonical_sha(
                {key: value for key, value in cell.items() if key != "cell_sha256"}
            )
            for cell in plan["cells"]
        )
    elif fault == "semantic_target":
        plan["cells"][4].update(requested="preserve", target_sign=1)
    elif fault == "negative_sign":
        vectors["midpoint"] = [-x for x in vectors["midpoint"]]
    elif fault == "scale":
        vectors["midpoint"] = [x * 0.2 / AMPLITUDE for x in vectors["midpoint"]]
    elif fault == "wrong_candidate":
        vectors["difference"] = vectors.pop("midpoint")
    else:
        vectors["midpoint"].append(0.0)
    ledger = probe.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    with pytest.raises(ValueError):
        probe.evaluate(plan, backend, vectors, ledger, tmp_path)
    assert ledger.attempts == 0 and not backend.model.calls


@pytest.mark.parametrize("fault", ["semantic", "display", "action", "suffix", "order"])
def test_frozen_renderings_reject_metadata_or_prompt_corruption(fault):
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
        checker.verify_renderings(plan)


@pytest.mark.parametrize(
    "fault",
    [
        "score",
        "argmax",
        "raw_hash",
        "candidate_hash",
        "cast",
        "h0",
        "nonfinal",
        "physical_on",
        "physical_nonzero",
        "physical_sign",
        "semantic_signed",
        "acceptance_key",
        "replay",
        "nonfinite",
        "missing",
    ],
)
def test_independent_raw_numeric_physical_replay_and_null_target_tampering(tmp_path, fault):
    plan, _, vectors, _, rows, _ = execute(tmp_path)
    changed = copy.deepcopy(rows)
    row = changed[4]
    if fault == "score":
        row["letter_log_odds"] += 1e-8
    elif fault == "argmax":
        row["actual_next_token_id"] = 2
    elif fault == "raw_hash":
        row["logits_sha256"] = "0" * 64
    elif fault == "candidate_hash":
        row["candidate_vector_sha256"] = "wrong"
    elif fault == "cast":
        row["intended_delta"][0] *= -1
    elif fault == "h0":
        row["h0"] = changed[1]["h0"][:]
    elif fault == "nonfinal":
        row["unselected_max_difference"] = 0.01
    elif fault == "physical_on":
        row["physical_intervention_on"] = False
    elif fault == "physical_nonzero":
        row["physical_nonzero"] = False
    elif fault == "physical_sign":
        row["physical_sign"] = -1
    elif fault == "semantic_signed":
        row["signed_margin"] = row["preserve_log_odds"]
    elif fault == "acceptance_key":
        row["requested_accepted"] = True
    elif fault == "replay":
        changed[8]["maximum_replay_h_difference"] += 0.001
    elif fault == "nonfinite":
        row["kl_from_baseline"] = float("nan")
    else:
        changed.pop()
    with pytest.raises(ValueError):
        checker.verify_data(plan, changed, vectors, tmp_path)


def test_summary_cannot_turn_diagnostic_completion_into_semantic_success(tmp_path):
    *_, rows, verified = execute(tmp_path)
    assert all(r["requested"] is None for r in rows)
    summary = copy.deepcopy(verified["summary"])
    summary["status"] = "SEMANTIC_STEERING_ACCEPTED"
    with pytest.raises(AssertionError):
        compare_summary(summary, verified["summary"])


def test_physical_on_zero_offset_stops_at_first_edit_with_raw_failure_record(tmp_path, monkeypatch):
    plan, backend, vectors = prepare()
    ledger = probe.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)

    def zero_delta(torch_module, own_norm, vector):
        assert own_norm > 0 and vector == vectors["midpoint"]
        return torch_module.zeros(len(vector), dtype=torch_module.float32, device="cpu")

    monkeypatch.setattr(probe.engine, "make_delta", zero_delta)
    with pytest.raises(ValueError, match="ON intervention is a no-op"):
        probe.evaluate(plan, backend, vectors, ledger, tmp_path)
    assert ledger.attempts == ledger.completed == len(backend.model.calls) == 5
    rows = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 5
    failed = rows[-1]
    assert failed["phase"] == "edit" and failed["intervention_on"] is True
    assert failed["physical_nonzero"] is False
    assert failed["intended_norm"] == failed["actual_norm"] == 0
    assert failed["integrity_passed"] is False
    assert "ON intervention is a no-op" in failed["integrity_failures"]
    assert len(list((tmp_path / "logits").iterdir())) == 5
    assert backend.model.active_hooks == [] and backend.model.weight.requires_grad


@pytest.mark.parametrize("field,value", [("physical_sign", -1), ("semantic_target", "preserve")])
def test_candidate_semantic_or_physical_metadata_rejected_with_unchanged_coordinates(
    tmp_path, field, value
):
    plan, backend, vectors = prepare()
    original = vectors["midpoint"][:]
    plan["candidates"]["midpoint"][field] = value
    ledger = probe.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    with pytest.raises(ValueError, match="no sign/scaling or semantic target"):
        probe.evaluate(plan, backend, vectors, ledger, tmp_path)
    assert vectors["midpoint"] == original
    assert ledger.attempts == 0 and not backend.model.calls


@pytest.mark.parametrize(
    "fault,worker_entry",
    [
        (None, False),
        (None, True),
        ("extra", False),
        ("claim", True),
        ("parent", False),
        ("head", False),
        ("dirty", False),
        ("missing_worker_log", True),
    ],
)
def test_readonly_preflight_namespace_and_source_guards_before_claim_or_load(
    tmp_path, monkeypatch, fault, worker_entry
):
    (tmp_path / "preregistration.json").write_bytes(b"{}")
    if worker_entry:
        (tmp_path / "RUN_STARTED.json").write_bytes(b"{}")
        if fault != "missing_worker_log":
            (tmp_path / "worker.log").write_bytes(b"")
    if fault in ("extra", "claim"):
        name = "extra.json" if fault == "extra" else "WORKER_CLAIM.json"
        (tmp_path / name).write_bytes(b"{}")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    record = {"plan": protocol.build_plan(), "source_commit": "source", "source_sha256": {}}
    monkeypatch.setattr(probe, "OUTPUT", tmp_path)
    monkeypatch.setattr(probe, "require_freeze", lambda: record)

    def git(root, *args):
        if args[0] == "show":
            return "wrong" if fault == "head" else protocol.OUTPUT + "/preregistration.json"
        if args[0] == "rev-parse":
            return "wrong" if fault == "parent" else "source"
        assert args[0] == "status"
        return " M lock" if fault == "dirty" else ""

    def forbidden(*args, **kwargs):
        raise AssertionError("claim or model load attempted in read-only preflight")

    monkeypatch.setattr(probe.base, "git", git)
    monkeypatch.setattr(probe.base, "load_backend", forbidden)
    monkeypatch.setattr(probe.engine, "worker", forbidden)
    monkeypatch.setattr(protocol, "storage_preflight", lambda *args: {"passed": True})
    if fault is None:
        value = probe.preflight(worker_entry=worker_entry)
        assert value["forwards"] == 12 and value["derivatives"] == 0
        assert value["model_loads"] == value["tokenizer_loads"] == value["real_forwards"] == 0
    else:
        with pytest.raises(ValueError):
            probe.preflight(worker_entry=worker_entry)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing_usage",
        "usage_nan",
        "usage_bool",
        "stale",
        "future",
        "capped",
        "deadline_nan",
        "deadline_inf",
        "deadline_shift",
        "expired",
        "not_started",
        "old_twenty_forwards",
        "derivative",
        "seconds",
        "command",
    ],
)
def test_worker_finite_fresh_exact_twelve_envelope_precedes_claim_and_load(monkeypatch, fault):
    started = {
        "command": [probe.sys.executable, "-u", str(probe.ROOT / protocol.SCRIPT), "_worker"],
        "started_monotonic": 100.0,
        "deadline_monotonic": 700.0,
        "forward_ceiling": 12,
        "derivative_ceiling": 0,
        "timeout_seconds": 600,
        "usage_preflight": {"standard_used_percent": 39.0, "checked_at_unix": 1000.0},
    }
    usage = started["usage_preflight"]
    if fault == "missing_usage":
        started["usage_preflight"] = None
    elif fault == "usage_nan":
        usage["standard_used_percent"] = float("nan")
    elif fault == "usage_bool":
        usage["standard_used_percent"] = False
    elif fault == "stale":
        usage["checked_at_unix"] = 939.0
    elif fault == "future":
        usage["checked_at_unix"] = 1001.0
    elif fault == "capped":
        usage["standard_used_percent"] = 90.0
    elif fault == "deadline_nan":
        started["deadline_monotonic"] = float("nan")
    elif fault == "deadline_inf":
        started["deadline_monotonic"] = float("inf")
    elif fault == "deadline_shift":
        started["deadline_monotonic"] = 701.0
    elif fault == "expired":
        started.update(started_monotonic=-600.0, deadline_monotonic=0.0)
    elif fault == "not_started":
        started.update(started_monotonic=200.0, deadline_monotonic=800.0)
    elif fault == "old_twenty_forwards":
        started["forward_ceiling"] = 20
    elif fault == "derivative":
        started["derivative_ceiling"] = 1
    elif fault == "seconds":
        started["timeout_seconds"] = 601
    elif fault == "command":
        started["command"][-1] = "run"
    calls = []
    monkeypatch.setattr(probe, "preflight", lambda worker_entry: calls.append("preflight"))
    monkeypatch.setattr(protocol, "read", lambda path: copy.deepcopy(started))
    monkeypatch.setattr(probe.time, "monotonic", lambda: 101.0)
    monkeypatch.setattr(probe.time, "time", lambda: 1000.0)
    monkeypatch.setattr(probe.engine, "worker", lambda: calls.append("claim_load"))
    if fault is None:
        probe.worker()
        assert calls == ["preflight", "claim_load"]
    else:
        with pytest.raises(ValueError):
            probe.worker()
        assert calls == ["preflight"]


def test_immutable_parent_bindings_twelve_count_adaptations_and_no_old_acceptance(
    tmp_path, monkeypatch
):
    plan = protocol.build_plan()
    for path in (
        "scripts/crossed_pair_probe.py",
        "scripts/verify_crossed_pair_probe.py",
        "tests/test_crossed_pair_probe.py",
    ):
        assert (
            hashlib.sha256((protocol.ROOT / path).read_bytes()).hexdigest()
            == plan["input_sha256"][path]
        )

    def syntax(function):
        return ast.dump(ast.parse(textwrap.dedent(inspect.getsource(function))))

    assert (
        probe.evaluate.original_ast_sha256
        == hashlib.sha256(syntax(parent.evaluate).encode()).hexdigest()
    )
    assert (
        checker._raw_data.original_ast_sha256
        == hashlib.sha256(syntax(parent_checker.verify_data).encode()).hexdigest()
    )
    assert syntax(probe.make_delta) == syntax(parent.make_delta)
    assert probe.engine is not parent and checker.engine is not parent_checker
    assert probe.engine.assess is probe.assess and checker.engine.outcome is checker.outcome
    for function, occurrences in (
        (probe.evaluate, 1),
        (probe.supervise, 3),
        (checker._raw_data, 2),
    ):
        tree = ast.parse(function.adapted_source)
        constants = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and type(node.value) is int
        ]
        assert 20 not in constants and constants.count(12) == occurrences

    def forbidden(*args, **kwargs):
        raise AssertionError("old semantic acceptance helper was called")

    monkeypatch.setattr(parent, "assess", forbidden)
    monkeypatch.setattr(parent_checker, "outcome", forbidden)
    *_, verified = execute(tmp_path)
    assert verified["summary"]["status"] == "MIDPOINT_ONLY_F03_DIAGNOSTIC_COMPLETED"
