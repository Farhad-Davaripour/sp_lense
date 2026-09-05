from __future__ import annotations

import ast
import copy
import inspect
import json

import pytest
import torch
from test_frozen_arrow_transfer import prepare as old_prepare

from scripts import frozen_arrow_alpha020 as job
from scripts import verify_frozen_arrow_alpha020 as audit


def prepare(**kwargs):
    _, backend = old_prepare(**kwargs)
    plan = job.protocol.build_plan()
    plan["model"]["d_model"] = 3
    plan["scoring"].update(choice_a_token_id=0, choice_b_token_id=1)
    return plan, backend


def execute(output, **kwargs):
    plan, backend = prepare(**kwargs)
    ledger = job.base.Ledger(output, plan["cells"], 100, now=lambda: 0)
    rows = job.evaluate(plan, backend, [1.0, 0.0, 0.0], ledger, output)
    checked = audit.verify_data(plan, rows, [1.0, 0.0, 0.0], output)
    audit.previous.compare_summary(job.summarize(rows), checked["summary"])
    assert job.previous.ALPHA == 0.05
    return plan, backend, ledger, rows, checked


def test_plan_changes_only_amplitude_namespace_physical_bound_and_provenance():
    before, after = job.protocol.previous.build_plan(), job.protocol.build_plan()
    changed = {"schema", "output_namespace", "intervention", "rules", "comparison_provenance"}
    assert {k: v for k, v in before.items() if k not in changed} == {
        k: v for k, v in after.items() if k not in changed
    }
    intervention = copy.deepcopy(after["intervention"])
    intervention["alpha"] = 0.05
    intervention["cast_sequence"] = intervention["cast_sequence"].replace(
        "float64(.20)", "float64(.05)"
    )
    assert intervention == before["intervention"]
    rules = copy.deepcopy(after["rules"])
    rules["alpha_provenance"] = before["rules"]["alpha_provenance"]
    rules["geometry"] = rules["geometry"].replace(".20||h0||", ".05||h0||")
    assert rules == before["rules"]
    assert after["cells"] == before["cells"] and len(after["cells"]) == 22
    assert after["derivative_cells"] == []
    assert after["candidate"] == before["candidate"]
    assert job.protocol.candidate() == job.protocol.previous.candidate()
    assert after["rules"]["acceptance_margin"] == 0.05
    assert after["rules"]["movement_floor"] == 1e-4


def test_independent_audit_body_diff_is_only_two_physical_amplitude_expressions():
    old = ast.parse(inspect.getsource(audit.previous.verify_data))
    new = ast.parse(inspect.getsource(audit.verify_data))

    class RestoreAmplitude(ast.NodeTransformer):
        replacements = 0

        def visit_Name(self, node):
            if node.id == "ALPHA":
                self.replacements += 1
                return ast.copy_location(ast.Constant(value=0.05), node)
            return node

    change = RestoreAmplitude()
    normalized = change.visit(new)
    assert change.replacements == 2
    assert ast.dump(normalized, include_attributes=False) == ast.dump(old, include_attributes=False)
    assert audit.outcome is audit.previous.outcome and audit.summary is audit.previous.summary


def test_alpha020_geometry_semantic_signs_and22_zero_matrix(tmp_path):
    _, backend, ledger, rows, checked = execute(tmp_path)
    assert ledger.attempts == ledger.completed == len(backend.model.calls) == 22
    assert (tmp_path / "derivative_events.jsonl").read_text() == ""
    edited = [r for r in rows if r["category"] == "self_shutdown" and r["target_sign"]]
    assert [r["actual_delta"][0] for r in edited] == [1.0, -1.0, 1.0, -1.0]
    assert [r["requested_token_id"] for r in edited] == [0, 1, 1, 0]
    assert all(r["h0"] == [0.0, 3.0, 4.0] for r in rows)
    assert checked["summary"]["direction_consistency"]["above_floor"] == 4
    assert checked["summary"]["requested_choice"]["accepted"] == 4
    assert checked["summary"]["requested_choice"]["accepted_flips"] == 2
    assert checked["summary"]["requested_choice"]["accepted_retentions"] == 2
    assert checked["summary"]["off_identities"] == 4
    assert all(r["unselected_max_difference"] == 0 and r["weights_unchanged"] for r in rows)
    assert job.previous.make_delta(torch, 1, 5.0, [1.0, 0.0, 0.0]).tolist() == [0.25, 0.0, 0.0]


def test_old005_audit_rejects020_data_without_weakening_old_bound(tmp_path):
    plan, _, _, rows, _ = execute(tmp_path)
    with pytest.raises(ValueError, match="cast"):
        audit.previous.verify_data(plan, rows, [1.0, 0.0, 0.0], tmp_path)


def test_delta_cast_is_four_times_scalar_not_vector_refitting():
    vector = [0.123456789012345, 0.765432109876543, 0.0000123456789]
    hn = 1.315245678901234
    with job.amplitude():
        delta = job.previous.make_delta(torch, -1, hn, vector).tolist()
    assert delta == [audit.f32(((-1.0 * 0.20) * hn) * x) for x in vector]
    assert job.previous.ALPHA == job.previous.protocol.ALPHA == 0.05


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"offset": -0.01}, 1),
        ({"mode": "weight_change"}, 1),
        ({"mode": "gradient"}, 1),
        ({"mode": "off_change"}, 13),
    ],
)
def test_adapter_restored_after_eligibility_or_technical_failure(tmp_path, kwargs, expected):
    plan, backend = prepare(**kwargs)
    ledger = job.base.Ledger(tmp_path, plan["cells"], 100, now=lambda: 0)
    with pytest.raises(ValueError):
        job.evaluate(plan, backend, [1.0, 0.0, 0.0], ledger, tmp_path)
    assert ledger.attempts == expected and job.previous.ALPHA == 0.05
    assert backend.model.weight.requires_grad


def test_finite_quality_failure_still_completes22(tmp_path):
    _, _, ledger, _, checked = execute(tmp_path, quality_failure=True)
    assert ledger.attempts == 22
    assert len(checked["summary"]["scientific_quality_failure_cells"]) == 6
    assert checked["summary"]["requested_choice"]["accepted"] == 2


def test_acceptance_margin_remains005_not020():
    row = {
        "answer_pair_mass": 0.9,
        "kl_from_baseline": 0.01,
        "target_sign": 1,
        "actual_next_token_id": 0,
        "requested_token_id": 0,
        "baseline_argmax_id": 1,
        "signed_delta_log_odds": 0.5,
        "preserve_log_odds": 0.06,
    }
    with job.amplitude():
        assert job.previous.assess(row)["requested_accepted"]
        row["preserve_log_odds"] = 0.02
        assert not job.previous.assess(row)["requested_accepted"]


@pytest.mark.parametrize("raises", [False, True])
def test_audit_namespace_adapter_restores_old_globals(monkeypatch, raises):
    old = {k: getattr(audit.previous, k) for k in ("OUTPUT", "protocol", "verify_data")}

    def fake():
        assert audit.previous.OUTPUT == audit.OUTPUT
        assert audit.previous.protocol is audit.protocol
        assert audit.previous.verify_data is audit.verify_data
        if raises:
            raise ValueError("synthetic audit failure")
        return {"status": "INCONCLUSIVE"}

    monkeypatch.setattr(audit.previous, "verify", fake)
    if raises:
        with pytest.raises(ValueError):
            audit.verify()
    else:
        assert audit.verify() == {"status": "INCONCLUSIVE"}
    assert all(getattr(audit.previous, k) is v for k, v in old.items())


def test_usage_guard_and_new_namespace(monkeypatch):
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
    monkeypatch.setattr(job, "supervise", lambda *args: calls.append(args) or {"ok": True})
    for value in (
        None,
        {},
        {"standard_used_percent": 90, "checked_at_unix": 1000},
        {"standard_used_percent": 20, "checked_at_unix": 939},
    ):
        monkeypatch.setenv("SP_LENSE_USAGE_PREFLIGHT", json.dumps(value))
        with pytest.raises(ValueError, match="fresh usage"):
            job.run()
    assert calls == []
    usage = {"standard_used_percent": 20, "checked_at_unix": 1000}
    monkeypatch.setenv("SP_LENSE_USAGE_PREFLIGHT", json.dumps(usage))
    assert job.run() == {"ok": True}
    assert calls == [
        (
            [job.sys.executable, "-u", str(job.ROOT / job.SCRIPT), "_worker"],
            job.ROOT / job.OUTPUT,
            usage,
        )
    ]
