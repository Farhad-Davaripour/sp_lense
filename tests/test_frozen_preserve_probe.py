"""Focused fixed-pair, no-fitting, original-state and replay contract tests."""

from __future__ import annotations

import copy
import json
import time

import pytest
import torch
from test_local_controllability_positive_control import setup
from test_refreshed_gradient_control import Toy

from scripts import frozen_preserve_probe as job
from scripts import frozen_preserve_probe_plan as protocol
from scripts import verify_frozen_preserve_probe as audit


def prepare(mode="linear", offset=-0.1, quality_failure=False):
    plan, backend = setup(protocol.build_plan())
    plan["model"]["d_model"] = 3
    vectors = {"preserve": [0.08508063610056309, 0.0, 0.0]}
    for t, v in vectors.items():
        plan["candidates"][t]["norm"] = protocol.norm(v)
        plan["candidates"][t]["vector_float64_le_sha256"] = protocol.vector_sha(v)

    class SemanticModel(Toy):
        def forward(self, tokens):
            assert not torch.is_grad_enabled()
            out = super().forward(tokens)
            if mode in ("weak", "weak_margin"):
                factor = 0.1 if mode == "weak" else 0.32
                out[..., 0] = factor * (out[..., 0] - self.offset) + self.offset
            prompt = self.by_token[int(tokens[0, 1])]
            if prompt["order"] == "preserve_second":
                out = out[..., [1, 0, 2, 3, 4, 5]]
            if mode == "nonfinite":
                out[..., 0] = float("nan")
            if mode == "gradient":
                torch.autograd.grad(None, None)
            if mode == "weight_change":
                self.weight.add_(0.01)
            if mode == "replay_mismatch" and len(self.calls) == 5:
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
    assert len(plan["prompts"]) == 2 and len(plan["cells"]) == 6 and not plan["derivative_cells"]
    assert {p["case_id"] for p in plan["prompts"]} == {"cg_f03_context_rotation__v1__self_shutdown"}
    assert [p["order"] for p in plan["prompts"]] == ["preserve_first", "preserve_second"]
    assert [c["phase"] for c in plan["cells"]] == ["baseline"] * 2 + ["edit"] * 2 + ["replay"] * 2
    assert [c["requested"] for c in plan["cells"][2:]] == ["preserve"] * 4
    assert [c["target_sign"] for c in plan["cells"]] == [0, 0, 1, 1, 1, 1]
    assert [c["replay_of"] for c in plan["cells"][4:]] == [c["cell_id"] for c in plan["cells"][2:4]]
    assert list(plan["candidates"]) == ["preserve"]
    spec = plan["candidates"]["preserve"]
    saved = protocol.candidates()["preserve"]
    assert spec["selected_prompts_not_fitted"] and spec["audit_before_freeze_verified"]
    assert len(spec["fitted_prompt_ids"]) == 8
    selected = [p["prompt_id"] for p in plan["prompts"]]
    assert not set(spec["fitted_prompt_ids"]).intersection(selected)
    assert selected == spec["reserved_prompt_ids"]
    assert protocol.vector_sha(saved["vector"]) == spec["vector_float64_le_sha256"]
    assert protocol.norm(saved["vector"]) == spec["norm"] == 0.08508063610056309


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
                "id": "cg_f03_context_rotation",
                "split": "discovery",
                "variants": [
                    {"id": "v2", "cases": Poison()},
                    {"id": "v1", "cases": {"self_shutdown": selected_case, "control": Poison()}},
                ],
            },
        ]
    }
    manifest = {
        "splits": {
            "discovery": {
                "family_ids": ["cg_f03_context_rotation"],
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
    assert f.attempts == f.completed == len(backend.model.calls) == 6
    assert (tmp_path / "derivative_events.jsonl").read_text() == ""
    assert [r["requested_token_id"] for r in rows[2:4]] == [0, 1]
    assert [r["actual_delta"][0] for r in rows[2:4]] == [audit.f32(5 * 0.08508063610056309)] * 2
    assert all(r["h0"] == [0.0, 3.0, 4.0] for r in rows)
    assert all(r["unselected_max_difference"] == 0 and r["weights_unchanged"] for r in rows)
    assert all(a["h"] == b["h"] for a, b in zip(rows[2:4], rows[4:], strict=True))
    s = checked["summary"]
    assert s["pair"]["strict_accepted"] == 2 and s["pair"]["pair_pass"]
    assert s["per_vector"]["preserve"]["strict_accepted"] == 2
    assert s["pair"]["accepted_flips"] == 2 and s["pair"]["accepted_retentions"] == 0
    assert s["pair"]["actual_A_to_B"] == s["pair"]["actual_B_to_A"] == 1
    assert s["replay_matches"] == 2 and s["off_controls_run"] == 0
    assert backend.model.active_hooks == [] and backend.model.weight.requires_grad
    assert backend.model.weight.grad is None
    with pytest.raises(ValueError):
        f.begin(rows[0])


def test_already_correct_retention_counts_not_flips(tmp_path):
    *_, checked = execute(tmp_path, offset=0.1)
    assert checked["summary"]["pair"]["accepted_retentions"] == 2
    assert checked["summary"]["pair"]["accepted_flips"] == 0


def test_cast_uses_serialized_coordinates_without_strength_or_sign():
    v = [0.00123456789012345, -0.0765432109876543, 0.000123456789]
    hn = 1.315245678901234
    assert job.make_delta(torch, hn, v).tolist() == [audit.f32(hn * x) for x in v]
    assert job.make_delta(torch, hn, v).tolist() != [audit.f32(hn * 0.05 * x) for x in v]


@pytest.mark.parametrize(
    "kwargs", [{"quality_failure": True}, {"mode": "weak"}, {"mode": "weak_margin"}]
)
def test_finite_scientific_failures_keep_all_six_cells(tmp_path, kwargs):
    _, _, _, f, rows, checked = execute(tmp_path, **kwargs)
    assert f.attempts == f.completed == 6 and len(rows) == 6
    assert checked["summary"]["pair"]["strict_accepted"] == 0
    assert not checked["summary"]["pair"]["pair_pass"]
    assert len(checked["summary"]["scientific_failed_edits"]) == 2
    assert all(r["replay_consistent"] for r in rows[4:])


@pytest.mark.parametrize(
    "kwargs,attempts",
    [
        ({"offset": -0.01}, 1),
        ({"mode": "nonfinite"}, 1),
        ({"mode": "gradient"}, 1),
        ({"mode": "weight_change"}, 1),
        ({"mode": "replay_mismatch"}, 5),
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


@pytest.mark.parametrize("fault", ["double_scale", "invert", "extra_comply"])
def test_candidate_vector_changes_rejected_before_any_forward(tmp_path, fault):
    plan, backend, vectors = prepare()
    if fault == "double_scale":
        vectors["preserve"] = [x * 0.20 for x in vectors["preserve"]]
    elif fault == "invert":
        vectors["preserve"] = [-x for x in vectors["preserve"]]
    else:
        vectors["comply"] = [-0.20, 0.0, 0.0]
    f = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    with pytest.raises(ValueError, match="binding|one"):
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
        changed[4]["maximum_replay_h_difference"] = 0.001
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
    assert json.loads((tmp_path / "RUN_STARTED.json").read_text())["forward_ceiling"] == 6


def test_structural_physical_scoring_and_primitive_invariance():
    import ast
    import inspect
    import textwrap

    from scripts import frozen_pair_plan as old_plan
    from scripts import frozen_pair_transfer as parent
    from scripts import verify_frozen_pair_transfer as parent_audit

    def source(fn):
        return textwrap.dedent(inspect.getsource(fn))

    def same(a, b):
        assert ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))

    same(source(parent.make_delta), source(job.make_delta))
    same(source(parent.assess), source(job.assess))
    same(
        source(parent.evaluate)
        .replace("/10 forwards", "/6 forwards")
        .replace("len(rows) == 10", "len(rows) == 6")
        .replace("10/0 accounting", "6/0 accounting"),
        source(job.evaluate),
    )
    same(
        source(parent_audit.verify_data)
        .replace("len(rows) == 10", "len(rows) == 6")
        .replace("exact10-cell sequence", "exact6-cell sequence")
        .replace("iterdir())) == 10", "iterdir())) == 6")
        .replace(
            '== {"preserve", "comply"}, "two candidate bindings"',
            '== {"preserve"}, "one PRESERVE candidate binding"',
        ),
        source(audit.verify_data),
    )
    assert job.DerivativeGuard is parent.DerivativeGuard and job.recorder is parent.recorder
    for name in ("norm", "vector_sha", "offset_sha", "authenticated"):
        assert getattr(protocol, name) is getattr(old_plan, name)
    assert "2fc06364715b967f1860aea9cf38778875588b17" in source(audit.verify)
    assert source(job.worker).index("storage_preflight") < source(job.worker).index("load_backend")


@pytest.mark.parametrize("fault", ["reserved_ids", "fit_ids", "freeze", "verification"])
def test_binding_rejects_changed_reservation_or_audit_chain(monkeypatch, fault):
    config = protocol.read(protocol.ROOT / protocol.CONFIG)
    selected = [p["prompt_id"] for p in protocol.build_plan()["prompts"]]
    original = protocol.authenticated
    spec = config["candidates"]["preserve"]

    def changed(path, digest, root=protocol.ROOT):
        value = copy.deepcopy(original(path, digest, root))
        if fault == "reserved_ids" and path == spec["construction_lock"]:
            value["plan"]["reserved_unrun_prompt_ids"] = ["wrong"]
        elif fault == "fit_ids" and path == spec["construction_lock"]:
            value["plan"]["construction_ids"][0] = selected[0]
        elif fault == "freeze" and path == spec["candidate_freeze"]:
            value["after_independent_audit"] = False
        elif fault == "verification" and path == spec["verification"]:
            value["status"] = "INCONCLUSIVE"
        return value

    monkeypatch.setattr(protocol, "authenticated", changed)
    with pytest.raises(ValueError):
        protocol.bind_candidates(config, selected)


def test_storage_guard_before_model_load(tmp_path, monkeypatch):
    config = protocol.read(protocol.ROOT / protocol.CONFIG)
    from types import SimpleNamespace

    monkeypatch.setattr(
        protocol.shutil, "disk_usage", lambda root: SimpleNamespace(free=64 * 1024**2)
    )
    assert protocol.storage_preflight(tmp_path, config)["bounds"]["total_bound_bytes"] == 29030242
    monkeypatch.setattr(protocol.shutil, "disk_usage", lambda root: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="storage"):
        protocol.storage_preflight(tmp_path, config)


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {"standard_used_percent": 90, "checked_at_unix": 0},
        {"standard_used_percent": 25, "checked_at_unix": 0},
    ],
)
def test_fresh_usage_before_run(monkeypatch, usage):
    monkeypatch.setattr(job, "require_freeze", lambda: {"source_commit": "source"})

    def fake_git(*args):
        if "show" in args:
            return protocol.OUTPUT + "/preregistration.json"
        if "status" in args:
            return ""
        return "source"

    monkeypatch.setattr(job.base, "git", fake_git)
    monkeypatch.setenv("SP_LENSE_USAGE_PREFLIGHT", json.dumps(usage))
    with pytest.raises(ValueError, match="fresh usage"):
        job.run()
