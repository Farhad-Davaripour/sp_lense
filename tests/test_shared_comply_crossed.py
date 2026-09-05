"""Focused model-free crossed-path tests; fake tensors are not scientific evidence."""

from __future__ import annotations

import ast
import copy
import inspect
import json
import time
from types import SimpleNamespace

import pytest
import torch

from scripts import shared_comply_crossed as job
from scripts import shared_comply_crossed_plan as protocol
from scripts import shared_comply_two_family as parent
from scripts import verify_shared_comply_crossed as audit
from scripts import verify_shared_comply_two_family as parent_audit

fixtures = protocol.isolate(
    "tests._crossed_comply_fixtures", "tests/test_shared_comply_two_family.py"
)
fixtures.job, fixtures.protocol, fixtures.audit = job.engine, protocol, audit

# Focused inherited safety cases execute the NEW plan, summary and independent audit.
test_first_acceptance = fixtures.test_first_acceptance_common_semantics_and_no_worker_candidate
test_four_updates = fixtures.test_four_update_original_norm_cast_and_physical_bounds
test_stall = fixtures.test_stall_counts_attempt_keeps_last_endpoint
test_quality_group = fixtures.test_quality_failure_completes_eight_group_then_finals
test_final_disagreement = fixtures.test_final_disagreement_precludes_candidate
test_full_schedule_stops = fixtures.test_conditional_driver_exact_groups_and_stops
test_negative_rhs_kkt = fixtures.test_negative_rhs_and_corrupt_independent_kkt
test_corrupt_raw_audit = fixtures.test_raw_audit_rejects_corruption
test_durable_candidate = fixtures.test_candidate_requires_durable_audit_then_binds_exact_vector
test_no_candidate = fixtures.test_no_candidate_on_failed_or_changed_audit
test_timeout_no_retry = fixtures.test_external_timeout_kills_worker_and_preserves_inconclusive


def test_factorial_text_and_exact_schedule():
    plan = protocol.build_plan()
    assert audit.verify_plan(plan) == {
        "rendering_rows": 8,
        "independent_semantic_situations": 2,
        "maximum_forwards": 144,
        "maximum_derivatives": 64,
        "optional_cells": 128,
    }
    assert [p["family_id"] for p in plan["prompts"]] == [protocol.FAMILIES[0]] * 4 + [
        protocol.FAMILIES[1]
    ] * 4
    assert {p["variant_id"] for p in plan["prompts"]} == {"v1"}
    for ab, ba in zip(plan["prompts"][::2], plan["prompts"][1::2], strict=True):
        assert ab["option_line_by_letter"] == ba["option_line_by_letter"]
        assert ab["semantic_to_letter"] == ba["semantic_to_letter"]
        assert ab["non_option_bytes_sha256"] == ba["non_option_bytes_sha256"]
        assert ab["display_order"] == "A_then_B" and ba["display_order"] == "B_then_A"
        assert ab["prompt_sha256"] == ab["canonical_prompt_sha256"]
        assert ab["prompt_sha256"] != ba["prompt_sha256"]


@pytest.mark.parametrize(
    "fault", ["v2", "family", "label", "display", "text", "reorder", "schedule", "derivative"]
)
def test_independent_layout_rejects_tampering(fault):
    plan = copy.deepcopy(protocol.build_plan())
    p = plan["prompts"][1]
    if fault == "v2":
        p["variant_id"] = "v2"
    elif fault == "family":
        p["family_id"] = "cg_f03_context_rotation"
    elif fault == "label":
        p["comply_label"] = p["preserve_label"]
    elif fault == "display":
        p["display_order"] = "A_then_B"
    elif fault == "text":
        p["prompt"] += " "
        p["prompt_sha256"] = protocol.sha(p["prompt"].encode())
    elif fault == "reorder":
        plan["prompts"].reverse()
    elif fault == "schedule":
        plan["cells"][8]["optional"] = False
    else:
        plan["derivative_cells"].pop()
    with pytest.raises(ValueError):
        audit.verify_layout(plan)


@pytest.mark.parametrize(
    "key,value",
    [
        ("aim", 0.2),
        ("maximum_forwards", 145),
        ("maximum_derivatives", 65),
        ("maximum_updates", 9),
        ("path_cap", 0.5),
        ("target", "preserve"),
        ("variants", ["v1", "v2"]),
        ("controls_allowed", True),
    ],
)
def test_no_recipe_drift(monkeypatch, key, value):
    original = protocol.read
    cfg = copy.deepcopy(original(protocol.ROOT / protocol.CONFIG))
    cfg[key] = value
    monkeypatch.setattr(
        protocol,
        "read",
        lambda path: cfg if path == protocol.ROOT / protocol.CONFIG else original(path),
    )
    with pytest.raises(ValueError):
        protocol.build_plan()


def test_identical_numerical_core_and_isolated_globals():
    assert job.engine is not parent and job.engine.__dict__ is not parent.__dict__
    assert job.engine.protocol is protocol and parent.protocol is not protocol
    assert audit.engine is not parent_audit and parent_audit.protocol is not protocol
    for name in ("increment", "project", "drive", "Session", "evaluate"):
        assert inspect.getsource(getattr(job.engine, name)) == inspect.getsource(
            getattr(parent, name)
        )
    for name in ("replay",):
        assert inspect.getsource(getattr(audit.engine, name)) == inspect.getsource(
            getattr(parent_audit, name)
        )
    assert inspect.getsource(audit._parent_data) == inspect.getsource(parent_audit.verify_data)
    assert inspect.getsource(audit._parent_update) == inspect.getsource(parent_audit.verify_update)
    assert job.engine.optimizer is parent.optimizer
    assert job.engine.ForwardLedger is parent.ForwardLedger
    assert job.engine.Derivatives is parent.Derivatives
    assert protocol.storage_preflight is protocol.parent.storage_preflight
    tree = ast.parse(inspect.getsource(audit))
    assert not any(
        isinstance(n, ast.ImportFrom)
        and n.module
        in (
            "torch",
            "transformers",
            "scripts.shared_comply_crossed",
            "scripts.shared_comply_two_family",
        )
        for n in ast.walk(tree)
    )


class OwnNormToy(fixtures.Toy):
    def __init__(self, prompts, s0=0.2, fault=None):
        super().__init__(prompts)
        self.s0, self.fault = s0, fault

    def forward(self, tokens):
        assert not self.weight.requires_grad
        self.calls.append(int(tokens[0, 1]))
        p = self.prompts[int(tokens[0, 1]) - 10]
        multiplier = 1 if p["family_id"] == protocol.FAMILIES[0] else 2
        h = torch.tensor([0.0, 3.0 * multiplier, 4.0 * multiplier]).repeat(1, tokens.shape[-1], 1)
        if self.fault == "hidden":
            h[..., 1] = float("nan")
        for name, function in self.active_hooks:
            assert name == "blocks.10.hook_out"
            h = function(h, hook=SimpleNamespace(name=name))
        x = h[..., 0]
        S = self.s0 + (x.sqrt() if self.fault == "gradient" else x)
        a = S if p["preserve_label"] == "A" else -S
        other = float("inf") if self.fault == "logits" else -20.0
        return torch.stack(
            [a, torch.zeros_like(a), *[torch.full_like(a, other) for _ in range(4)]], dim=-1
        )


def own_prepare(s0=0.2, fault=None):
    plan, backend = fixtures.prepare()
    backend.model = OwnNormToy(plan["prompts"], s0, fault)
    backend.encode = lambda prompt: backend.model.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )["input_ids"]
    return plan, backend


def own_execute(output, s0=0.2):
    plan, backend = own_prepare(s0)
    deadline = time.monotonic() + 100
    f = job.ForwardLedger(output, plan["cells"], deadline)
    d = job.Derivatives(torch, output, plan["derivative_cells"], deadline)
    rows, summary = job.evaluate(plan, backend, f, d, output)
    verified = audit.verify_data(plan, rows, output)
    audit.compare_summary(summary, verified["summary"])
    return plan, rows, summary, verified, f, d


@pytest.mark.parametrize("s0,counts,eligible", [(0.2, (32, 8), 4), (-0.2, (16, 0), 0)])
def test_both_flip_directions_and_zero_untested(tmp_path, s0, counts, eligible):
    _, _, result, _, f, d = own_execute(tmp_path, s0)
    assert (f.attempts, d.attempts) == counts
    assert result["final_accepted"] == 8
    for direction in ("A_to_B", "B_to_A"):
        assert result["directional_coverage"][direction] == {
            "eligible": eligible,
            "achieved": eligible,
            "status": "ALL" if eligible else "UNTESTED",
        }
    assert result["independent_semantic_situations"] == 2 and result["rendering_rows"] == 8


def test_own_original_norms_common_w_and_negative_rhs(tmp_path):
    _, rows, _, _, _, _ = own_execute(tmp_path)
    finals = [r for r in rows if r["condition"] == "final"]
    assert {r["h0_norm"] for r in finals} == {5.0, 10.0}
    assert len({tuple(r["shared_w"]) for r in finals}) == 1
    for r in finals:
        assert r["intended_delta"] == [audit.f32(r["h0_norm"] * x) for x in r["shared_w"]]
        assert job.norm(r["h"]) != r["h0_norm"]
    gradients = [
        {
            "row": {
                "cell_id": str(i),
                "h0": [0.0, 3.0 * (i + 1), 4.0 * (i + 1)],
                "h0_norm": 5.0 * (i + 1),
                "gradient": [1.0, 0.0, 0.0],
                "preserve_log_odds": -0.2 if i % 2 == 0 else 0.2,
            }
        }
        for i in range(8)
    ]
    update = job.increment(gradients, [0.0] * 3, 0.0, 1)
    assert update["rhs"] == pytest.approx([-0.1, 0.3] * 4)
    assert update["d"][0] < 0
    assert audit.verify_update(update, [g["row"] for g in gradients], [0.0] * 3, 0.0)[
        "optimizer_kkt_verified"
    ]


@pytest.mark.parametrize(
    "fault,forwards,derivatives", [("logits", 1, 0), ("hidden", 1, 0), ("gradient", 9, 1)]
)
def test_nonfinite_real_path_technical_no_retry(tmp_path, fault, forwards, derivatives):
    plan, backend = own_prepare(fault=fault)
    f = job.ForwardLedger(tmp_path, plan["cells"], time.monotonic() + 100)
    d = job.Derivatives(torch, tmp_path, plan["derivative_cells"], time.monotonic() + 100)
    with pytest.raises(ValueError, match="finite"):
        job.evaluate(plan, backend, f, d, tmp_path)
    assert f.attempts == forwards and d.attempts == derivatives
    assert not (tmp_path / "comply_vector.json").exists()
    assert backend.model.active_hooks == [] and backend.model.weight.requires_grad
    assert len(list((tmp_path / "logits").iterdir())) == forwards


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_independent_audit_rejects_nonfinite(tmp_path, value):
    plan, rows, _, _, _, _ = own_execute(tmp_path)
    rows[-1]["kl_from_baseline"] = value
    with pytest.raises(ValueError, match="nonfinite"):
        audit.verify_data(plan, rows, tmp_path)


def test_ledger_caps_deadline_and_no_restart(tmp_path):
    plan = protocol.build_plan()
    f = job.ForwardLedger(tmp_path, plan["cells"], 100, now=lambda: 0)
    for cell in plan["cells"]:
        f.begin(cell)
        f.finish(True)
    assert f.attempts == f.completed == 144
    with pytest.raises(ValueError):
        f.begin(plan["cells"][-1])
    with pytest.raises(ValueError, match="retry"):
        job.ForwardLedger(tmp_path, plan["cells"], 100)
    d = job.Derivatives(torch, tmp_path, plan["derivative_cells"], 100, now=lambda: 0)
    d.original_grad = lambda: "fake"
    for cell in plan["derivative_cells"]:
        d.cell = cell
        assert d.call() == "fake"
    assert d.attempts == d.completed == 64
    with pytest.raises(ValueError):
        d.call()


def test_storage_parent_arithmetic_and_shortfall(monkeypatch):
    cfg = protocol.build_plan()["config"]
    monkeypatch.setattr(
        protocol.parent.previous.shutil, "disk_usage", lambda root: SimpleNamespace(free=536870912)
    )
    assert (
        protocol.storage_preflight(protocol.ROOT, cfg)["bounds"]["total_bound_bytes"] == 377958704
    )
    monkeypatch.setattr(
        protocol.parent.previous.shutil, "disk_usage", lambda root: SimpleNamespace(free=536870911)
    )
    with pytest.raises(ValueError, match="storage cannot fit"):
        protocol.storage_preflight(protocol.ROOT, cfg)


def authorization_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "OUTPUT", tmp_path)
    path = tmp_path / "preregistration.json"
    path.write_text("{}")
    return {
        "preregistration_sha256": protocol.sha(path.read_bytes()),
        "scope": protocol.AUTH_SCOPE,
        "authorized_by": "supervisor",
    }


@pytest.mark.parametrize("entry", ["run", "worker"])
def test_missing_authority_never_calls_engine(tmp_path, monkeypatch, entry):
    authorization_fixture(tmp_path, monkeypatch)
    monkeypatch.delenv(protocol.AUTH_KEY, raising=False)
    monkeypatch.setattr(job.engine, entry, lambda: pytest.fail("unauthorized engine reached"))
    with pytest.raises(ValueError, match="separate supervisor"):
        getattr(job, entry)()
    assert {p.name for p in tmp_path.iterdir()} == {"preregistration.json"}


@pytest.mark.parametrize("fault", ["hash", "scope", "issuer", "extra"])
def test_authorization_exact_lock_and_scope(tmp_path, monkeypatch, fault):
    value = authorization_fixture(tmp_path, monkeypatch)
    if fault == "hash":
        value["preregistration_sha256"] = "0" * 64
    elif fault == "scope":
        value["scope"] = "unbounded"
    elif fault == "issuer":
        value["authorized_by"] = "runner"
    else:
        value["retry"] = True
    monkeypatch.setenv(protocol.AUTH_KEY, json.dumps(value))
    with pytest.raises(ValueError, match="separate supervisor"):
        job.require_authorization()


@pytest.mark.parametrize(
    "name", ["WORKER_CLAIM.json", "rows.jsonl", "logits", "verification.json", "RUN_STARTED.json"]
)
def test_stale_namespace_blocks_preflight(tmp_path, monkeypatch, name):
    authorization_fixture(tmp_path, monkeypatch)
    (tmp_path / name).write_text("stale")
    with pytest.raises(ValueError, match="untouched"):
        job.untouched_namespace()


def test_zero_model_preflight_and_commit_contract(tmp_path, monkeypatch):
    authorization_fixture(tmp_path, monkeypatch)
    record = {"plan": protocol.build_plan(), "source_commit": "source", "source_sha256": {}}
    monkeypatch.setattr(job, "require_freeze", lambda: record)
    monkeypatch.setattr(job.base, "load_backend", lambda *args: pytest.fail("MODEL LOAD"))
    path = protocol.OUTPUT + "/preregistration.json"
    monkeypatch.setattr(
        job.base,
        "git",
        lambda *args: path if "show" in args else "" if "status" in args else "source",
    )
    result = job.preflight()
    assert (
        result["real_forwards"]
        == result["real_derivatives"]
        == result["model_loads"]
        == result["tokenizer_loads"]
        == 0
    )
    assert not result["run_authorized"] and not result["scientific_success"]
    assert {p.name for p in tmp_path.iterdir()} == {"preregistration.json"}
    monkeypatch.setattr(job.base, "git", lambda *args: "later-report.md")
    with pytest.raises(ValueError, match="preregistration-only"):
        job.preflight()


def test_protected_condition_never_parsed_in_plan(monkeypatch):
    original = protocol.Path.read_bytes

    def guarded(path):
        if path.as_posix().endswith(protocol.PROTECTED):
            pytest.fail("plan must not consume protected condition")
        return original(path)

    monkeypatch.setattr(protocol.Path, "read_bytes", guarded)
    assert len(protocol.build_plan()["prompts"]) == 8


def test_wrong_direction_retention_is_not_a_new_gate():
    row = {
        "answer_pair_mass": 1.0,
        "kl_from_baseline": 100.0,
        "actual_next_token_id": 0,
        "requested_token_id": 0,
        "preserve_log_odds": -0.2,
        "signed_delta_log_odds": -0.7,
    }
    assert job.accepts(row) and audit.accepts(row)


@pytest.mark.parametrize(
    "kwargs,forwards,derivatives,match",
    [
        ({"zero": True}, 16, 8, "numerical solver failure"),
        ({"mismatch": True}, 9, 1, "current-state"),
        ({"positive": 0.01}, 3, 0, "baseline"),
    ],
)
def test_crossed_technical_stops_no_padding(tmp_path, kwargs, forwards, derivatives, match):
    plan, backend = fixtures.prepare(**kwargs)
    f = job.ForwardLedger(tmp_path, plan["cells"], time.monotonic() + 100)
    d = job.Derivatives(torch, tmp_path, plan["derivative_cells"], time.monotonic() + 100)
    original = torch.autograd.grad
    with pytest.raises(ValueError, match=match):
        job.evaluate(plan, backend, f, d, tmp_path)
    assert (f.attempts, d.attempts) == (forwards, derivatives)
    assert torch.autograd.grad is original and backend.model.weight.requires_grad
    assert not (tmp_path / "comply_vector.json").exists()


test_usage_guard = fixtures.test_usage_missing_capped_stale_blocks_run


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_unused_optimizer_diagnostics_must_be_finite(value):
    gradients = [
        {
            "row": {
                "cell_id": str(i),
                "h0_norm": 5.0,
                "h0": [0.0, 3.0, 4.0],
                "gradient": [1.0, 0.0, 0.0],
                "preserve_log_odds": 0.2,
            }
        }
        for i in range(8)
    ]
    update = job.increment(gradients, [0.0] * 3, 0.0, 1)
    update["solver"]["active_sets"][0]["pivots"] = [value]
    with pytest.raises(ValueError, match="nonfinite"):
        audit.verify_update(update, [g["row"] for g in gradients], [0.0] * 3, 0.0)


@pytest.mark.parametrize("name", ["updates.jsonl", "forward_events.jsonl", "skip_events.jsonl"])
def test_nonfinite_auxiliary_journals_rejected(tmp_path, name):
    plan, rows, _, _, _, _ = own_execute(tmp_path)
    file = tmp_path / name
    values = [json.loads(x) for x in file.read_text().splitlines()]
    values[0]["unused_diagnostic"] = float("nan")
    file.write_text("\n".join(json.dumps(x) for x in values) + "\n")
    with pytest.raises(ValueError, match="nonfinite"):
        audit.verify_data(plan, rows, tmp_path)


@pytest.mark.parametrize("name", ["runtime.json", "RUN_STARTED.json", "storage_preflight.json"])
def test_nonfinite_envelopes_rejected_before_complete_audit(tmp_path, monkeypatch, name):
    plan = protocol.build_plan()
    (tmp_path / "preregistration.json").write_text(json.dumps({"plan": plan}))
    (tmp_path / name).write_text('{"numeric":NaN}')
    monkeypatch.setattr(audit, "OUTPUT", tmp_path)
    monkeypatch.setattr(audit.engine, "verify", lambda: pytest.fail("nonfinite envelope accepted"))
    with pytest.raises(ValueError, match="nonfinite"):
        audit.verify()


def test_every_numeric_report_row_has_mapping_and_display(tmp_path):
    _, _, _, verified, _, _ = own_execute(tmp_path)
    original = copy.deepcopy(verified)
    text = audit.report(verified)
    assert original == verified
    for row in verified["summary"]["final_cells"]:
        label = "/".join(
            row[k]
            for k in ("family_id", "variant_id", "order", "semantic_mapping", "display_order")
        )
        assert "| " + label + " |" in text
    for row in verified["construction_stages"]:
        label = (
            str(row["stage"])
            + "/"
            + "/".join(
                row[k]
                for k in ("family_id", "variant_id", "order", "semantic_mapping", "display_order")
            )
        )
        assert "| " + label + " |" in text
    assert "v2/first" not in text and "EXPOSED development" in text


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "head",
        "source",
        "stale",
        "usage_cap",
        "usage_nan",
        "deadline",
        "deadline_nan",
        "timeout",
        "forwards",
        "derivatives",
        "command",
        "retry",
    ],
)
def test_direct_worker_preclaim_checks(tmp_path, monkeypatch, fault):
    auth = authorization_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv(protocol.AUTH_KEY, json.dumps(auth))
    record = {"plan": protocol.build_plan(), "source_commit": "source", "source_sha256": {}}

    def frozen():
        if fault == "source":
            raise ValueError("frozen source changed")
        return record

    monkeypatch.setattr(job, "require_freeze", frozen)
    path = protocol.OUTPUT + "/preregistration.json"
    monkeypatch.setattr(
        job.base,
        "git",
        lambda *args: (
            "later.md"
            if fault == "head"
            else path
            if "show" in args
            else ""
            if "status" in args
            else "source"
        ),
    )
    started = {
        "command": [job.sys.executable, "-u", str(job.ROOT / protocol.SCRIPT), "_worker"],
        "started_monotonic": time.monotonic(),
        "timeout_seconds": 900,
        "forward_ceiling": 144,
        "derivative_ceiling": 64,
        "usage_preflight": {"standard_used_percent": 35, "checked_at_unix": time.time()},
    }
    started["deadline_monotonic"] = started["started_monotonic"] + 900
    if fault == "stale":
        started["usage_preflight"]["checked_at_unix"] -= 61
    elif fault == "usage_cap":
        started["usage_preflight"]["standard_used_percent"] = 90
    elif fault == "usage_nan":
        started["usage_preflight"]["standard_used_percent"] = float("nan")
    elif fault == "deadline":
        started["started_monotonic"] -= 901
        started["deadline_monotonic"] -= 901
    elif fault == "deadline_nan":
        started["deadline_monotonic"] = float("nan")
    elif fault == "timeout":
        started["timeout_seconds"] = 901
    elif fault == "forwards":
        started["forward_ceiling"] = 145
    elif fault == "derivatives":
        started["derivative_ceiling"] = 65
    elif fault == "command":
        started["command"] = ["different_worker"]
    elif fault == "retry":
        (tmp_path / "WORKER_CLAIM.json").write_text("{}")
    (tmp_path / "RUN_STARTED.json").write_text(json.dumps(started))
    (tmp_path / "worker.log").write_text("")
    seen = []
    monkeypatch.setattr(job.engine, "worker", lambda: seen.append("fake delegate only"))
    monkeypatch.setattr(job.base, "load_backend", lambda *args: pytest.fail("MODEL LOAD"))
    if fault is None:
        job.worker()
        assert seen == ["fake delegate only"]
    else:
        with pytest.raises(ValueError):
            job.worker()
        assert not seen
    if fault != "retry":
        assert not (tmp_path / "WORKER_CLAIM.json").exists()
