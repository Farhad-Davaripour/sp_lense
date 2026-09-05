from __future__ import annotations

import copy
import json

import pytest
from test_local_controllability_positive_control import setup
from test_refreshed_gradient_control import Toy

from scripts import refreshed_gradient_control_f02_inputs as inputs
from scripts import refreshed_gradient_control_f02_v1 as job
from scripts import verify_refreshed_gradient_control_f02_v1 as audit


def fail_old_builder(*args, **kwargs):
    raise AssertionError("old first-family builder must not run")


def test_exact_selection_without_old_builders_or_old_family_assertion(monkeypatch):
    for module in (job.v1, job.v1.previous, job.v1.old, job.base):
        monkeypatch.setattr(module, "build_plan", fail_old_builder)
    monkeypatch.setattr(job.base, "FAMILY", "forbidden_inherited_family")
    monkeypatch.setattr(job.base, "load_backend", fail_old_builder)
    plan = job.build_plan()
    assert "cg_f01" not in json.dumps(plan)
    assert {p["family_id"] for p in plan["prompts"]} == {inputs.FAMILY}
    assert {p["variant_id"] for p in plan["prompts"]} == {"v1"}
    assert {p["split"] for p in plan["prompts"]} == {"discovery"}
    assert len(plan["prompts"]) == 6 and len(plan["selected_cases"]) == 3
    assert {(p["category"], p["order"]) for p in plan["prompts"]} == {
        (c, o) for c in inputs.CATEGORIES for o in inputs.ORDERS
    }
    assert job.build_plan() == plan


def test_only_identity_and_descriptive_fields_differ_from_frozen_recipe():
    first = inputs.read(inputs.ROOT / inputs.TEMPLATE)["plan"]
    second = job.build_plan()
    allowed = {
        "schema",
        "output_namespace",
        "prompts",
        "selected_cases",
        "cells",
        "derivative_cells",
        "selection_rule",
        "rules",
        "replication_provenance",
    }
    assert {k: v for k, v in first.items() if k not in allowed} == {
        k: v for k, v in second.items() if k not in allowed
    }
    descriptive = {"population", "next_pass"}
    assert {k: v for k, v in first["rules"].items() if k not in descriptive} == {
        k: v for k, v in second["rules"].items() if k not in descriptive
    }
    original_ids = {p["prompt_id"]: (p["category"], p["order"]) for p in first["prompts"]}
    selected_ids = {p["prompt_id"]: (p["category"], p["order"]) for p in second["prompts"]}
    for a, b in zip(first["cells"], second["cells"], strict=True):
        assert original_ids[a["prompt_id"]] == selected_ids[b["prompt_id"]]
        assert [a[k] for k in ("condition", "step", "optional")] == [
            b[k] for k in ("condition", "step", "optional")
        ]
        assert b["cell_id"] == f"{b['prompt_id']}__{b['condition']}"
        unhashed = {k: v for k, v in b.items() if k != "cell_sha256"}
        assert b["cell_sha256"] == inputs.digest(
            json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
        )
    assert len(second["cells"]) == 30 and len(second["derivative_cells"]) == 8
    assert sum(c["optional"] for c in second["cells"]) == 12
    for plan in (first, second):
        cases = {c["id"]: c for c in plan["selected_cases"]}
        normalized = []
        for p in plan["prompts"]:
            case = cases[p["case_id"]]
            prompt = p["prompt"]
            for field in ("scenario", "preserve_action", "comply_action"):
                prompt = prompt.replace(case[field], f"<{field}>")
            normalized.append(prompt)
            assert p["prompt_sha256"] == inputs.digest(p["prompt"].encode())
        if plan is first:
            original_envelopes = normalized
        else:
            assert normalized == original_envelopes


def synthetic_inputs():
    plan = job.build_plan()
    cases = {c["category"]: c for c in plan["selected_cases"]}
    data = {
        "families": [
            {"id": inputs.FAMILY, "split": "discovery", "variants": [{"id": "v1", "cases": cases}]}
        ]
    }
    manifest = {
        "splits": {
            "discovery": {
                "family_ids": ["aa_preceding_id", inputs.FAMILY, "zz_later_id"],
                "expanded_case_ids": [c["id"] for c in cases.values()],
            }
        }
    }
    return data, manifest


def test_selection_does_not_visit_other_case_text_or_other_split_data():
    class Unreadable(dict):
        def __getitem__(self, key):
            raise AssertionError("unselected text/results must not be read")

    data, manifest = synthetic_inputs()
    data["families"] += [{"id": "unselected", "variants": Unreadable()}]
    data["families"][0]["variants"] += [{"id": "v2", "cases": Unreadable()}]
    manifest["splits"].update(validation=Unreadable(), sealed=Unreadable())
    prompts, selected = inputs.select_inputs(data, manifest)
    assert len(prompts) == 6 and len(selected) == 3


@pytest.mark.parametrize("fault", ["order", "variant", "split", "case"])
def test_no_alternative_family_variant_or_wrong_split(fault):
    data, manifest = synthetic_inputs()
    if fault == "order":
        manifest["splits"]["discovery"]["family_ids"].remove("aa_preceding_id")
    elif fault == "variant":
        data["families"][0]["variants"][0]["id"] = "v2"
    elif fault == "split":
        data["families"][0]["split"] = "validation"
    else:
        data["families"][0]["variants"][0]["cases"]["self_shutdown"]["id"] = "old_family_id"
    with pytest.raises(ValueError):
        inputs.select_inputs(data, manifest)


def test_build_plan_reads_only_authenticated_input_paths(monkeypatch):
    allowed_text = {inputs.ROOT / p for p in (inputs.TEMPLATE, inputs.DATASET, inputs.MANIFEST)}
    original = inputs.read
    calls = []

    def checked(path):
        assert path in allowed_text
        calls.append(path)
        return original(path)

    monkeypatch.setattr(inputs, "read", checked)
    inputs.build_plan()
    assert set(calls) == allowed_text


def execute(output, *, same_semantic=False, **kwargs):
    plan, backend = setup(job.build_plan())

    class SemanticToy(Toy):
        def forward(self, tokens):
            logits = super().forward(tokens)
            prompt = self.prompts_by_token[int(tokens[0, 1])]
            if same_semantic and prompt["order"] == "preserve_second":
                logits = logits[..., [1, 0, 2, 3, 4, 5]]
            return logits

    model = SemanticToy(plan["prompts"], **kwargs)
    model.prompts_by_token = {model.tokenizer.ids[p["prompt"]]: p for p in plan["prompts"]}
    backend.model = model
    f = job.v1.ForwardLedger(output, plan["cells"], 100, now=lambda: 0)
    d = job.v1.DerivativeLedger(output, plan["derivative_cells"], 100, now=lambda: 0)
    rows, requests = job.v1.evaluate(plan, backend, f, d, output)
    result = audit.v1.verify_data(plan, rows, requests, f.skips, output)
    assert result["summary"] == job.v1.summarize(rows, requests)
    result.update(
        classification=result["summary"]["classification"],
        status={
            "forward_attempts": f.attempts,
            "derivative_attempts": d.attempts,
            "skipped_forwards": len(f.skips),
            "elapsed_seconds": 1,
        },
    )
    result["per_order"] = audit.per_order(result, rows)
    return plan, backend, f, d, rows, requests, result


@pytest.mark.parametrize(
    "kwargs,classification,forwards,derivatives,skips",
    [
        ({}, "PASS", 18, 2, 12),
        ({"kind": "nonlinear", "offset": -0.6}, "PASS", 30, 8, 0),
        ({"kind": "saturating", "offset": -0.6}, "PARTIAL", 30, 8, 0),
        ({"quality_failure": True}, "FAIL", 18, 2, 12),
    ],
)
def test_same_engine_audits_second_family_and_same_stop_rules(
    tmp_path, kwargs, classification, forwards, derivatives, skips
):
    plan, backend, f, d, rows, _, result = execute(tmp_path, **kwargs)
    assert f.attempts == f.completed == len(backend.model.calls) == forwards
    assert d.attempts == d.completed == derivatives and len(f.skips) == skips
    assert forwards + skips == 30 and forwards == 14 + 2 * derivatives
    assert result["classification"] == classification
    assert result["summary"]["retentions"] == 2 and result["summary"]["nonself_identities"] == 4
    assert all(r["family_id"] == inputs.FAMILY and r["variant_id"] == "v1" for r in rows)
    assert backend.model.weight.item() == 1 and backend.model.weight.grad is None
    assert backend.model.active_hooks == [] and backend.model.weight.requires_grad
    assert result["absolute_tolerance"] == 2e-5 and result["relative_tolerance"] == 0
    assert {r["prompt_id"] for r in rows} == {p["prompt_id"] for p in plan["prompts"]}


@pytest.mark.parametrize(
    "offset,requested,retained", [(-0.05, "preserve", "comply"), (0.05, "comply", "preserve")]
)
def test_actual_same_direction_requests_not_assumed_opposite(tmp_path, offset, requested, retained):
    _, _, _, _, _, _, result = execute(tmp_path, same_semantic=True, offset=offset)
    assert result["classification"] == "PASS"
    assert [r["opposed_request"] for r in result["per_order"]] == [requested, requested]
    assert [r["retention_request"] for r in result["per_order"]] == [retained, retained]
    assert all(r["opposed_flip"] and r["opposed_accepted"] for r in result["per_order"])
    report = audit.report(result)
    assert "second discovery family" in report and "first discovery family" not in report
    assert "Actual requests by answer order" in report and "model-free bridge" in report
    assert "cg_f01" not in report and "fixed-recipe v2 replication" not in report


def test_audit_rejects_wrong_family_rows_and_no_relative_escape(tmp_path):
    plan, _, f, _, original, requests, _ = execute(tmp_path)
    rows = copy.deepcopy(original)
    rows[0]["family_id"] = "cg_f01_archive_closeout"
    with pytest.raises(ValueError, match="identity"):
        audit.v1.verify_data(plan, rows, requests, f.skips, tmp_path)
    with pytest.raises(ValueError, match="absolute mismatch"):
        audit.v1.close(2000.001, 2000.0, "KL", 2e-5)


def test_audit_wrapper_uses_only_exact_f02_plan(tmp_path, monkeypatch):
    for module in (job.v1, job.v1.previous, job.v1.old, job.base):
        monkeypatch.setattr(module, "build_plan", fail_old_builder)
    result = {"classification": "INCONCLUSIVE", "fault": "synthetic incomplete"}
    called = []
    monkeypatch.setattr(audit.v1, "verify", lambda output: called.append(output) or result)
    plan = inputs.build_plan()
    audit.v1.write_new(tmp_path / "preregistration.json", {"plan": plan})
    assert audit.verify(tmp_path) == result and called == [tmp_path]
    changed = copy.deepcopy(plan)
    changed["prompts"][0]["family_id"] = "old_family"
    monkeypatch.setattr(audit.v1, "read", lambda path: {"plan": changed})
    with pytest.raises(ValueError, match="f02/v1 frozen plan/input identity"):
        audit.verify(tmp_path)
    assert called == [tmp_path]


def test_eligibility_failure_stops_attempt_no_substitution(tmp_path):
    plan, backend = setup(job.build_plan(), offset=-0.01)
    f = job.v1.ForwardLedger(tmp_path, plan["cells"], 100, now=lambda: 0)
    d = job.v1.DerivativeLedger(tmp_path, plan["derivative_cells"], 100, now=lambda: 0)
    with pytest.raises(job.v1.EligibilityError):
        job.v1.evaluate(plan, backend, f, d, tmp_path)
    assert f.attempts == 1 and d.attempts == 0


def test_fresh_usage_and_namespace_guard(monkeypatch):
    monkeypatch.setattr(job, "require_freeze", lambda: {"source_commit": "source"})
    path = f"{job.OUTPUT}/preregistration.json"
    monkeypatch.setattr(
        job.base,
        "git",
        lambda root, *args: (
            path if args[0] == "show" else "source" if args[0] == "rev-parse" else ""
        ),
    )
    monkeypatch.setattr(job.time, "time", lambda: 1000)
    calls = []
    monkeypatch.setattr(job.v1, "supervise", lambda *args: calls.append(args) or {"ok": True})
    for value in (
        None,
        {},
        {"standard_used_percent": 90, "checked_at_unix": 1000},
        {"standard_used_percent": 19, "checked_at_unix": 939},
    ):
        monkeypatch.setenv("SP_LENSE_USAGE_PREFLIGHT", json.dumps(value))
        with pytest.raises(ValueError, match="fresh usage"):
            job.run()
    assert calls == []
    usage = {"standard_used_percent": 19, "checked_at_unix": 1000}
    monkeypatch.setenv("SP_LENSE_USAGE_PREFLIGHT", json.dumps(usage))
    assert job.run() == {"ok": True}
    assert calls == [
        (
            [job.sys.executable, "-u", str(job.ROOT / job.SCRIPT), "_worker"],
            job.ROOT / job.OUTPUT,
            usage,
        )
    ]
