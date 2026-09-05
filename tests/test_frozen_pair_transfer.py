"""Focused fixed-pair, no-fitting, original-state and replay contract tests."""

from __future__ import annotations

import copy
import json
import time

import pytest
import torch
from test_local_controllability_positive_control import setup
from test_refreshed_gradient_control import Toy

from scripts import frozen_pair_plan as protocol
from scripts import frozen_pair_transfer as job
from scripts import verify_frozen_pair_transfer as audit


def prepare(mode="linear", offset=-0.1, quality_failure=False):
    plan, backend = setup(protocol.build_plan())
    plan["model"]["d_model"] = 3
    vectors = {"preserve": [0.05, 0.0, 0.0], "comply": [-0.20, 0.0, 0.0]}
    for t, v in vectors.items():
        plan["candidates"][t]["norm"] = protocol.norm(v)
        plan["candidates"][t]["vector_float64_le_sha256"] = protocol.vector_sha(v)

    class SemanticModel(Toy):
        def forward(self, tokens):
            assert not torch.is_grad_enabled()
            out = super().forward(tokens)
            if mode == "weak":
                out[..., 0] = 0.4 * (out[..., 0] - self.offset) + self.offset
            prompt = self.by_token[int(tokens[0, 1])]
            if prompt["order"] == "preserve_second":
                out = out[..., [1, 0, 2, 3, 4, 5]]
            if mode == "nonfinite":
                out[..., 0] = float("nan")
            if mode == "gradient":
                torch.autograd.grad(None, None)
            if mode == "weight_change":
                self.weight.add_(0.01)
            if mode == "replay_mismatch" and len(self.calls) == 7:
                out[..., 0] += 0.01
            if hasattr(self, "ledger"):
                assert self.ledger.attempts == len(self.calls) and self.ledger.pending
            return out

    model = SemanticModel(plan["prompts"], offset=offset, quality_failure=quality_failure)
    model.by_token = {model.tokenizer.ids[p["prompt"]]: p for p in plan["prompts"]}
    backend.model = model
    return plan, backend, vectors


def execute(output, **kwargs):
    plan, backend, vectors = prepare(**kwargs)
    ledger = job.base.Ledger(output, plan["cells"], time.monotonic() + 100)
    backend.model.ledger = ledger
    rows = job.evaluate(plan, backend, vectors, ledger, output)
    checked = audit.verify_data(plan, rows, vectors, output)
    audit.compare_summary(job.summarize(rows), checked["summary"])
    return plan, backend, vectors, ledger, rows, checked


def test_exact_selected_plan_and_candidate_authentication():
    plan = protocol.build_plan()
    assert len(plan["prompts"]) == 2 and len(plan["cells"]) == 10 and not plan["derivative_cells"]
    assert {p["case_id"] for p in plan["prompts"]} == {
        "cg_f02_translation_console__v2__self_shutdown"
    }
    assert [p["order"] for p in plan["prompts"]] == ["preserve_first", "preserve_second"]
    assert [c["phase"] for c in plan["cells"]] == ["baseline"] * 2 + ["edit"] * 4 + ["replay"] * 4
    assert [c["requested"] for c in plan["cells"][2:]] == ["preserve", "comply"] * 4
    assert [c["replay_of"] for c in plan["cells"][6:]] == [c["cell_id"] for c in plan["cells"][2:6]]
    saved = protocol.candidates()
    for target, spec in plan["candidates"].items():
        assert spec["selected_prompts_not_fitted"]
        assert protocol.vector_sha(saved[target]["vector"]) == spec["vector_float64_le_sha256"]
        assert protocol.norm(saved[target]["vector"]) == spec["norm"]
        assert not set(spec["fitted_prompt_ids"]).intersection(
            p["prompt_id"] for p in plan["prompts"]
        )
    assert [plan["candidates"][t]["norm"] for t in ("preserve", "comply")] == [0.05, 0.20]


@pytest.mark.parametrize("fault", ["variant", "case_id", "split", "manifest"])
def test_exact_selection_no_fallback_or_unselected_text(fault):
    config = protocol.read(protocol.ROOT / protocol.CONFIG)
    selected_case = {
        "id": config["selection"]["case_id"],
        "category": "self_shutdown",
        "scenario": "synthetic",
        "preserve_action": "keep",
        "comply_action": "stop",
    }

    class Poison:
        def __getitem__(self, key):
            raise AssertionError("unselected text accessed")

    data = {
        "families": [
            {"id": "validation", "variants": Poison()},
            {
                "id": "cg_f02_translation_console",
                "split": "discovery",
                "variants": [
                    {"id": "v1", "cases": Poison()},
                    {"id": "v2", "cases": {"self_shutdown": selected_case, "control": Poison()}},
                ],
            },
        ]
    }
    manifest = {
        "splits": {
            "discovery": {
                "family_ids": ["cg_f02_translation_console"],
                "expanded_case_ids": [selected_case["id"]],
            },
            "validation": Poison(),
            "sealed": Poison(),
        }
    }
    assert len(protocol.select_inputs(data, manifest, config["selection"])) == 2
    if fault == "variant":
        data["families"][1]["variants"][1]["id"] = "v3"
    elif fault == "case_id":
        selected_case["id"] = "wrong"
    elif fault == "split":
        data["families"][1]["split"] = "validation"
    else:
        manifest["splits"]["discovery"]["expanded_case_ids"] = []
    with pytest.raises(ValueError):
        protocol.select_inputs(data, manifest, config["selection"])


def test_exact_vectors_targets_independent_original_states_replays_and_attempt_counts(tmp_path):
    _, backend, _, f, rows, checked = execute(tmp_path)
    assert f.attempts == f.completed == len(backend.model.calls) == 10
    assert (tmp_path / "derivative_events.jsonl").read_text() == ""
    edits = rows[2:6]
    assert [r["requested_token_id"] for r in edits] == [0, 1, 1, 0]
    assert [r["actual_delta"][0] for r in edits] == [0.25, -1.0, 0.25, -1.0]
    assert all(r["h0"] == [0.0, 3.0, 4.0] for r in rows)
    assert all(r["unselected_max_difference"] == 0 and r["weights_unchanged"] for r in rows)
    assert all(a["h"] == b["h"] for a, b in zip(rows[2:6], rows[6:], strict=True))
    s = checked["summary"]
    assert s["joint"]["strict_accepted"] == 4 and s["joint"]["pair_pass"]
    assert (
        s["per_vector"]["preserve"]["strict_accepted"]
        == s["per_vector"]["comply"]["strict_accepted"]
        == 2
    )
    assert s["joint"]["accepted_flips"] == s["joint"]["accepted_retentions"] == 2
    assert s["joint"]["actual_A_to_B"] == s["joint"]["actual_B_to_A"] == 1
    assert s["replay_matches"] == 4 and s["off_controls_run"] == 0
    assert backend.model.active_hooks == [] and backend.model.weight.requires_grad
    assert backend.model.weight.grad is None
    with pytest.raises(ValueError):
        f.begin(rows[0])


def test_cast_uses_serialized_coordinates_without_strength_or_sign():
    v = [0.00123456789012345, -0.0765432109876543, 0.000123456789]
    hn = 1.315245678901234
    assert job.make_delta(torch, hn, v).tolist() == [audit.f32(hn * x) for x in v]
    assert job.make_delta(torch, hn, v).tolist() != [audit.f32(hn * 0.05 * x) for x in v]


@pytest.mark.parametrize("kwargs", [{"quality_failure": True}, {"mode": "weak"}])
def test_finite_scientific_failures_keep_all_ten_cells(tmp_path, kwargs):
    _, _, _, f, rows, checked = execute(tmp_path, **kwargs)
    assert f.attempts == f.completed == 10 and len(rows) == 10
    assert checked["summary"]["joint"]["strict_accepted"] == 2
    assert not checked["summary"]["joint"]["pair_pass"]
    assert len(checked["summary"]["scientific_failed_edits"]) == 2
    assert all(r["replay_consistent"] for r in rows[6:])


@pytest.mark.parametrize(
    "kwargs,attempts",
    [
        ({"offset": -0.01}, 1),
        ({"mode": "nonfinite"}, 1),
        ({"mode": "gradient"}, 1),
        ({"mode": "weight_change"}, 1),
        ({"mode": "replay_mismatch"}, 7),
    ],
)
def test_technical_faults_stop_without_retry(tmp_path, kwargs, attempts):
    plan, backend, vectors = prepare(**kwargs)
    f = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    old_grad, old_backward = torch.autograd.grad, torch.autograd.backward
    with pytest.raises(ValueError):
        job.evaluate(plan, backend, vectors, f, tmp_path)
    assert f.attempts == attempts
    assert torch.autograd.grad is old_grad and torch.autograd.backward is old_backward
    assert backend.model.weight.requires_grad


@pytest.mark.parametrize("fault", ["double_scale", "invert", "wrong_order_vector"])
def test_candidate_vector_changes_rejected_before_any_forward(tmp_path, fault):
    plan, backend, vectors = prepare()
    if fault == "double_scale":
        vectors["comply"] = [x * 0.20 for x in vectors["comply"]]
    elif fault == "invert":
        vectors["comply"] = [-x for x in vectors["comply"]]
    else:
        vectors["preserve"] = list(vectors["comply"])
    f = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    with pytest.raises(ValueError, match="binding"):
        job.evaluate(plan, backend, vectors, f, tmp_path)
    assert f.attempts == 0 and not backend.model.calls


@pytest.mark.parametrize(
    "fault",
    [
        "requested_label",
        "candidate_sha",
        "offset_hash",
        "cast",
        "effect",
        "signed_margin",
        "replay",
        "outcome",
    ],
)
def test_saved_array_audit_rejects_corruption(tmp_path, fault):
    plan, _, vectors, _, rows, _ = execute(tmp_path)
    changed = copy.deepcopy(rows)
    r = changed[2]
    if fault == "requested_label":
        r["requested_label"] = "B"
    elif fault == "candidate_sha":
        r["candidate_vector_sha256"] = "bad"
    elif fault == "offset_hash":
        r["offset_float32_le_sha256"] = "bad"
    elif fault == "cast":
        r["intended_delta"][0] += 1e-7
    elif fault == "effect":
        r["delta_log_odds"] += 1e-8
    elif fault == "signed_margin":
        r["signed_margin"] += 0.01
    elif fault == "replay":
        changed[6]["maximum_replay_h_difference"] = 0.001
    else:
        r["requested_accepted"] = False
    with pytest.raises(ValueError):
        audit.verify_data(plan, changed, vectors, tmp_path)


def test_candidate_file_hash_binding_and_preload_guard(tmp_path, monkeypatch):
    config = copy.deepcopy(protocol.read(protocol.ROOT / protocol.CONFIG))
    bad = tmp_path / "changed_candidate.json"
    bad.write_text("{}")
    config["candidates"]["preserve"]["path"] = str(bad)
    with pytest.raises(ValueError, match="authenticated"):
        protocol.candidates(config)

    output = tmp_path / "job"
    output.mkdir()
    job.base.write_new(output / "RUN_STARTED.json", {"deadline_monotonic": time.monotonic() + 100})
    monkeypatch.setattr(job, "ROOT", tmp_path)
    monkeypatch.setattr(job, "OUTPUT", "job")
    plan, _, _ = prepare()
    monkeypatch.setattr(job, "require_freeze", lambda: {"plan": plan})
    monkeypatch.setattr(protocol, "candidates", lambda: protocol.authenticated(str(bad), "invalid"))

    def forbidden(*args):
        raise AssertionError("model load before candidate authentication")

    monkeypatch.setattr(job.base, "load_backend", forbidden)
    with pytest.raises(ValueError, match="authenticated"):
        job.worker()
    assert (output / "INVALID.json").exists()


def test_external600_timeout_no_retry(tmp_path, monkeypatch):
    class Process:
        pid = 123
        killed = False

        def wait(self, timeout):
            if not self.killed:
                raise job.subprocess.TimeoutExpired("fake", timeout)
            return -9

        def poll(self):
            return -9 if self.killed else None

        def kill(self):
            self.killed = True

    fake = Process()
    monkeypatch.setattr(job.subprocess, "Popen", lambda *a, **k: fake)
    s = job.supervise(["fake"], tmp_path, {"standard_used_percent": 24}, timeout=0.01)
    assert fake.killed and s["status"] == "INCONCLUSIVE" and not s["retries_allowed"]
    assert json.loads((tmp_path / "RUN_STARTED.json").read_text())["forward_ceiling"] == 10
