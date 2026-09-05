from __future__ import annotations

import copy
import json

import pytest
from test_local_controllability_positive_control import setup
from test_refreshed_gradient_control import Toy

from scripts import refreshed_gradient_control_v2 as job
from scripts import verify_refreshed_gradient_control_v2 as audit


def prepare(**kwargs):
    plan, backend = setup(job.build_plan())
    backend.model = Toy(plan["prompts"], **kwargs)
    return plan, backend


def execute(output, **kwargs):
    plan, backend = prepare(**kwargs)
    forward = job.v1.ForwardLedger(output, plan["cells"], 100, now=lambda: 0)
    derivative = job.v1.DerivativeLedger(output, plan["derivative_cells"], 100, now=lambda: 0)
    rows, requests = job.v1.evaluate(plan, backend, forward, derivative, output)
    checked = audit.v1.verify_data(plan, rows, requests, forward.skips, output)
    assert checked["summary"] == job.v1.summarize(rows, requests)
    assert all(r["variant_id"] == "v2" for r in rows)
    return plan, backend, forward, derivative, rows, requests, checked


def test_only_variant_namespace_and_descriptive_fields_change():
    first, second = job.v1.build_plan(), job.build_plan()
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
    descriptive_rules = {"population", "next_pass"}
    assert {k: v for k, v in first["rules"].items() if k not in descriptive_rules} == {
        k: v for k, v in second["rules"].items() if k not in descriptive_rules
    }
    candidates = job.v1.previous.build_plan()
    assert second["prompts"] == [p for p in candidates["prompts"] if p["variant_id"] == "v2"]
    assert second["selected_cases"] == [
        c for c in candidates["selected_cases"] if c["variant_id"] == "v2"
    ]
    for a, b in zip(first["cells"], second["cells"], strict=True):
        assert b["prompt_id"] == a["prompt_id"].replace("__v1__", "__v2__")
        assert b["cell_id"] == a["cell_id"].replace("__v1__", "__v2__")
        assert [a[k] for k in ("condition", "step", "optional")] == [
            b[k] for k in ("condition", "step", "optional")
        ]
        unhashed = {k: v for k, v in b.items() if k != "cell_sha256"}
        assert b["cell_sha256"] == job.base.sha(
            json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
        )
    assert len(second["prompts"]) == 6 and len(second["cells"]) == 30
    assert len(second["derivative_cells"]) == 8
    assert job.build_plan() == second and job.v1.build_plan() == first


@pytest.mark.parametrize(
    "kwargs,classification,forwards,derivatives,skips",
    [
        ({}, "PASS", 18, 2, 12),
        ({"kind": "nonlinear", "offset": -0.6}, "PASS", 30, 8, 0),
        ({"kind": "saturating", "offset": -0.6}, "PARTIAL", 30, 8, 0),
        ({"quality_failure": True}, "FAIL", 18, 2, 12),
    ],
)
def test_v2_same_conditional_engine_and_independent_audit(
    tmp_path, kwargs, classification, forwards, derivatives, skips
):
    _, backend, f, d, _, _, result = execute(tmp_path, **kwargs)
    assert f.attempts == f.completed == len(backend.model.calls) == forwards
    assert d.attempts == d.completed == derivatives
    assert len(f.skips) == skips and forwards + skips == 30
    assert forwards == 14 + 2 * derivatives
    assert result["summary"]["classification"] == classification
    assert result["summary"]["nonself_identities"] == 4
    assert backend.model.active_hooks == [] and backend.model.weight.grad is None
    assert backend.model.weight.item() == 1 and backend.model.weight.requires_grad
    assert result["absolute_tolerance"] == 2e-5 and result["relative_tolerance"] == 0


def test_v2_ineligible_stops_without_gradient_or_retry(tmp_path):
    plan, backend = prepare(offset=-0.01)
    f = job.v1.ForwardLedger(tmp_path, plan["cells"], 100, now=lambda: 0)
    d = job.v1.DerivativeLedger(tmp_path, plan["derivative_cells"], 100, now=lambda: 0)
    with pytest.raises(job.v1.EligibilityError):
        job.v1.evaluate(plan, backend, f, d, tmp_path)
    assert f.attempts == 1 and d.attempts == 0


def test_v2_audit_rejects_v1_row_and_absolute_tolerance_escape(tmp_path):
    plan, _, f, _, original, requests, _ = execute(tmp_path)
    rows = copy.deepcopy(original)
    rows[0]["variant_id"] = "v1"
    with pytest.raises(ValueError, match="identity"):
        audit.v1.verify_data(plan, rows, requests, f.skips, tmp_path)
    with pytest.raises(ValueError, match="absolute mismatch"):
        audit.v1.close(2000.001, 2000.0, "KL", 2e-5)


def test_v2_report_labels_and_no_gate_claims(tmp_path):
    _, _, f, d, _, _, result = execute(tmp_path)
    result.update(
        classification=result["summary"]["classification"],
        status={
            "forward_attempts": f.attempts,
            "derivative_attempts": d.attempts,
            "skipped_forwards": len(f.skips),
            "elapsed_seconds": 1,
        },
    )
    report = audit.report(result)
    assert "# Refreshed-gradient v2 replication" in report
    assert "stop for supervisor review" in report
    assert "exposed within-family replication" in report
    assert "exposed v1" not in report and "| ||g|| |" not in report


def test_v2_usage_preflight_and_supervisor_namespace(monkeypatch):
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
    for usage in (
        None,
        {},
        {"standard_used_percent": 90, "checked_at_unix": 1000},
        {"standard_used_percent": 19, "checked_at_unix": 939},
    ):
        monkeypatch.setenv("SP_LENSE_USAGE_PREFLIGHT", json.dumps(usage))
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
