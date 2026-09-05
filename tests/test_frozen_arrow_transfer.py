from __future__ import annotations

import copy
import json
import struct
import subprocess

import pytest
import torch
from test_local_controllability_positive_control import setup
from test_refreshed_gradient_control import Toy

from scripts import frozen_arrow_transfer as job
from scripts import verify_frozen_arrow_transfer as audit


def prepare(mode="linear", offset=-0.05, quality_failure=False):
    plan, backend = setup(job.protocol.build_plan())
    plan["model"]["d_model"] = 3

    class SemanticModel(Toy):
        def forward(self, tokens):
            assert not torch.is_grad_enabled()
            out = super().forward(tokens)
            prompt = self.by_token[int(tokens[0, 1])]
            if mode == "wrong_minus":
                x = out[..., 0] - self.offset
                out[..., 0] = x.square() + x.pow(3) + self.offset
            if prompt["order"] == "preserve_second":
                out = out[..., [1, 0, 2, 3, 4, 5]]
            if mode == "bad_other_baseline" and prompt["category"] != "self_shutdown":
                out[..., 2] = 10
            if mode == "nonfinite":
                out[..., 0] = float("nan")
            if mode == "gradient":
                torch.autograd.grad(None, None)
            if mode == "weight_change":
                self.weight.add_(0.01)
            if mode == "off_change" and len(self.calls) == 13:
                out += 0.01
            return out

    model = SemanticModel(plan["prompts"], offset=offset, quality_failure=quality_failure)
    model.by_token = {model.tokenizer.ids[p["prompt"]]: p for p in plan["prompts"]}
    backend.model = model
    return plan, backend


def execute(output, **kwargs):
    plan, backend = prepare(**kwargs)
    ledger = job.base.Ledger(output, plan["cells"], 100, now=lambda: 0)
    v = [1.0, 0.0, 0.0]
    rows = job.evaluate(plan, backend, v, ledger, output)
    checked = audit.verify_data(plan, rows, v, output)
    audit.compare_summary(job.summarize(rows), checked["summary"])
    return plan, backend, ledger, rows, checked


def test_exact_plan22_zero_derivatives_and_fixed_inputs():
    plan = job.protocol.build_plan()
    assert len(plan["prompts"]) == 6 and len(plan["cells"]) == 22 and plan["derivative_cells"] == []
    assert {p["family_id"] for p in plan["prompts"]} == {"cg_f02_translation_console"}
    assert {p["variant_id"] for p in plan["prompts"]} == {"v1"}
    assert [c["condition"] for c in plan["cells"][:6]] == ["baseline"] * 6
    assert [c["condition"] for c in plan["cells"]].count("oracle_off") == 4
    assert plan["intervention"]["alpha"] == 0.05 and plan["rules"]["movement_floor"] == 1e-4
    assert all(
        c["target_sign"] == {"baseline": 0, "plus": 1, "minus": -1, "oracle_off": 0}[c["condition"]]
        for c in plan["cells"]
    )
    saved = job.protocol.candidate()
    assert job.protocol.sha(struct.pack("<1024d", *saved["vector"])) == job.protocol.VECTOR_SHA256
    assert plan == job.protocol.build_plan()


def test_semantic_signs_both_label_flip_directions_and_independent_edits(tmp_path):
    plan, backend, ledger, rows, result = execute(tmp_path)
    assert ledger.attempts == ledger.completed == len(backend.model.calls) == 22
    assert (tmp_path / "derivative_events.jsonl").read_text() == ""
    self_rows = [r for r in rows if r["category"] == "self_shutdown" and r["target_sign"]]
    assert [r["requested"] for r in self_rows] == ["preserve", "comply", "preserve", "comply"]
    assert [r["requested_token_id"] for r in self_rows] == [0, 1, 1, 0]
    assert [r["actual_delta"][0] for r in self_rows] == [0.25, -0.25, 0.25, -0.25]
    assert all(r["h0"] == [0.0, 3.0, 4.0] for r in rows)
    assert all(r["unselected_max_difference"] == 0 and r["weights_unchanged"] for r in rows)
    s = result["summary"]
    assert s["direction_consistency"]["above_floor"] == 4
    assert s["requested_choice"]["accepted"] == 4
    assert s["requested_choice"]["accepted_flips"] == 2
    assert s["requested_choice"]["accepted_retentions"] == 2
    assert s["requested_choice"]["actual_A_to_B"] == s["requested_choice"]["actual_B_to_A"] == 1
    assert s["off_identities"] == 4 and len(s["nonself_always_on_cells"]) == 8
    assert (
        backend.model.active_hooks == []
        and backend.model.weight.requires_grad
        and backend.model.weight.grad is None
    )
    with pytest.raises(ValueError):
        ledger.begin(plan["cells"][0])


@pytest.mark.parametrize("sign", [-1, 1])
def test_fixed_scalar_and_cast_sequence_no_vector_renormalization(sign):
    vector = [0.123456789012345, 0.765432109876543, 0.0000123456789]
    hn = 1.315245678901234
    expected = [
        struct.unpack("<f", struct.pack("<f", ((float(sign) * 0.05) * hn) * x))[0] for x in vector
    ]
    assert job.make_delta(torch, sign, hn, vector).tolist() == expected


def test_wrong_sign_not_concealed_by_plus_minus_contrast(tmp_path):
    _, _, _, _, result = execute(tmp_path, mode="wrong_minus")
    cells = result["summary"]["self_cells"]
    assert all(c["delta_log_odds"] > 0 for c in cells)
    assert cells[0]["delta_log_odds"] > cells[1]["delta_log_odds"]
    assert result["summary"]["direction_consistency"]["above_floor"] == 2
    assert all(not c["movement_above_floor"] for c in cells if c["condition"] == "minus")


def test_finite_quality_failure_continues_all_cells(tmp_path):
    _, _, ledger, _, result = execute(tmp_path, quality_failure=True)
    s = result["summary"]
    assert ledger.attempts == ledger.completed == 22
    assert len(s["scientific_quality_failure_cells"]) == 6
    assert s["direction_consistency"]["above_floor"] == 4
    assert s["direction_consistency"]["quality_valid_above_floor"] == 2
    assert s["requested_choice"]["accepted"] == 2 and s["off_identities"] == 4


@pytest.mark.parametrize(
    "kwargs,attempts,error",
    [
        ({"offset": -0.01}, 1, job.EligibilityError),
        ({"mode": "bad_other_baseline"}, 3, job.EligibilityError),
        ({"mode": "nonfinite"}, 1, ValueError),
        ({"mode": "gradient"}, 1, ValueError),
        ({"mode": "weight_change"}, 1, ValueError),
        ({"mode": "off_change"}, 13, ValueError),
    ],
)
def test_eligibility_and_technical_fault_stop_without_substitution(
    tmp_path, kwargs, attempts, error
):
    plan, backend = prepare(**kwargs)
    ledger = job.base.Ledger(tmp_path, plan["cells"], 100, now=lambda: 0)
    original_grad, original_backward = torch.autograd.grad, torch.autograd.backward
    with pytest.raises(error):
        job.evaluate(plan, backend, [1.0, 0.0, 0.0], ledger, tmp_path)
    assert ledger.attempts == attempts
    assert torch.autograd.grad is original_grad and torch.autograd.backward is original_backward
    assert backend.model.weight.requires_grad
    if kwargs.get("mode") == "gradient":
        assert len(job.base.read_rows(tmp_path / "derivative_events.jsonl")) == 1


def test_norm_bound_rejects_incorrect_vector_instead_of_renormalizing(tmp_path):
    plan, backend = prepare()
    ledger = job.base.Ledger(tmp_path, plan["cells"], 100, now=lambda: 0)
    with pytest.raises(ValueError, match="geometry"):
        job.evaluate(plan, backend, [2.0, 0.0, 0.0], ledger, tmp_path)
    assert ledger.attempts == 7


@pytest.mark.parametrize("fault", ["sign", "delta", "effect", "outcome", "state"])
def test_independent_audit_rejects_sign_cast_effect_or_outcome_corruption(tmp_path, fault):
    plan, _, _, original, _ = execute(tmp_path)
    rows = copy.deepcopy(original)
    r = rows[6]
    if fault == "sign":
        r["requested"] = "comply"
    elif fault == "delta":
        r["intended_delta"][0] += 1e-7
    elif fault == "effect":
        r["delta_log_odds"] += 1e-8
    elif fault == "outcome":
        r["requested_accepted"] = False
    else:
        r["h0"][0] = 0.1
    with pytest.raises(ValueError):
        audit.verify_data(plan, rows, [1.0, 0.0, 0.0], tmp_path)


def test_numeric_probability_contract_has_no_relative_escape():
    with pytest.raises(ValueError, match="absolute mismatch"):
        audit.close(2000.001, 2000.0, "mass", 2e-5)


def test_summary_preserves_exact_effects_without_probability_tolerance_escape():
    original = {"delta_log_odds": 0.123, "preserve_log_odds": 0.5, "answer_pair_mass": 0.9}
    rounded = {**original, "answer_pair_mass": 0.9 + 1e-15}
    audit.compare_summary(rounded, original)
    for field in ("delta_log_odds", "preserve_log_odds"):
        with pytest.raises(ValueError, match="exact"):
            audit.compare_summary({**original, field: original[field] + 1e-8}, original)


def test_usage_guard_and_namespace(monkeypatch):
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


def test_external_timeout_cannot_restart(tmp_path, monkeypatch):
    class Process:
        pid, killed = 123, False

        def wait(self, timeout):
            if self.killed:
                return -9
            raise subprocess.TimeoutExpired(["synthetic"], timeout)

        def poll(self):
            return -9 if self.killed else None

        def kill(self):
            self.killed = True

    monkeypatch.setattr(job.subprocess, "Popen", lambda *a, **k: Process())
    result = job.supervise(["synthetic"], tmp_path, {})
    assert result["status"] == "INCONCLUSIVE" and result["forward_attempts"] == 0
    with pytest.raises(FileExistsError):
        job.supervise(["synthetic"], tmp_path, {})
