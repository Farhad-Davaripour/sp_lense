"""Focused crossed-rendering, frozen-candidate, physical and coverage contracts."""

from __future__ import annotations

import ast
import copy
import inspect
import json
import textwrap
import time
from types import SimpleNamespace

import pytest
import torch
from test_local_controllability_positive_control import Model, setup

from scripts import crossed_pair_plan as protocol
from scripts import crossed_pair_probe as job
from scripts import frozen_pair_transfer as parent
from scripts import verify_crossed_pair_probe as audit
from scripts import verify_frozen_pair_transfer as parent_audit


def prepare(mode="linear", offset=0.1, quality_failure=False):
    plan, backend = setup(protocol.build_plan())
    plan["model"]["d_model"] = 3
    vectors = {
        "preserve": [0.08508063610056309, 0.0, 0.0],
        "comply": [-0.20000000000000004, 0.0, 0.0],
    }
    for target, vector in vectors.items():
        plan["candidates"][target]["norm"] = protocol.norm(vector)
        plan["candidates"][target]["vector_float64_le_sha256"] = protocol.vector_sha(vector)

    class SemanticModel(Model):
        def forward(self, tokens):
            assert not torch.is_grad_enabled() and not self.weight.requires_grad
            self.calls.append(int(tokens[0, 1]))
            p = self.by_token[int(tokens[0, 1])]
            i = p["rendering_index"] - 1
            h = torch.tensor([0.0, 3.0 + i, 4.0]).repeat(1, tokens.shape[-1], 1)
            for name, function in self.active_hooks:
                assert name == "blocks.10.hook_out"
                h = function(h, hook=SimpleNamespace(name=name))
            x = h[..., 0]
            baseline = -offset if p["display_order"] == "A_then_B" else offset
            if mode == "all_B":
                baseline = -offset if p["preserve_label"] == "A" else offset
            speed = 0.03 if mode == "weak" else 0.25 if mode == "weak_margin" else 1.0
            S = baseline + speed * x
            a = S if p["preserve_label"] == "A" else -S
            other = (
                torch.where(x != 0, torch.full_like(a, 10.0), torch.full_like(a, -20.0))
                if quality_failure
                else torch.full_like(a, -20.0)
            )
            out = torch.stack(
                [a, torch.zeros_like(a), other, *[torch.full_like(a, -20.0) for _ in range(3)]],
                dim=-1,
            )
            if mode == "nonfinite":
                out[..., 0] = float("nan")
            if mode == "gradient":
                torch.autograd.grad(None, None)
            if mode == "weight_change":
                self.weight.add_(0.01)
            if mode == "replay_mismatch" and len(self.calls) == 13:
                out[..., 0] += 0.01
            if hasattr(self, "ledger"):
                assert self.ledger.attempts == len(self.calls) and self.ledger.pending
            return out

    model = SemanticModel(plan["prompts"])
    model.by_token = {model.tokenizer.ids[p["prompt"]]: p for p in plan["prompts"]}
    backend.model = model
    return plan, backend, vectors


def execute(output, **kwargs):
    plan, backend, vectors = prepare(**kwargs)
    f = job.base.Ledger(output, plan["cells"], time.monotonic() + 100)
    backend.model.ledger = f
    rows = job.evaluate(plan, backend, vectors, f, output)
    verified = audit.verify_data(plan, rows, vectors, output)
    audit.compare_summary(job.summarize(rows), verified["summary"])
    return plan, backend, vectors, f, rows, verified


def test_exact_truth_table_canonical_bytes_and_BA_only_permutation():
    plan = protocol.build_plan()
    prompts = plan["prompts"]
    cfg = plan["config"]
    canonical = protocol.canonical.select_inputs(
        protocol.read(protocol.ROOT / cfg["dataset"]["path"]),
        protocol.read(protocol.ROOT / cfg["manifest"]["path"]),
        cfg["selection"],
    )
    assert [(p["preserve_label"], p["comply_label"], p["display_order"]) for p in prompts] == [
        ("A", "B", "A_then_B"),
        ("A", "B", "B_then_A"),
        ("B", "A", "A_then_B"),
        ("B", "A", "B_then_A"),
    ]
    assert prompts[0]["prompt"] == canonical[0]["prompt"]
    assert prompts[2]["prompt"] == canonical[1]["prompt"]
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
        assert ab["display_position_to_letter"] == {"first": "A", "second": "B"}
        assert ba["display_position_to_letter"] == {"first": "B", "second": "A"}
        assert ab["canonical_prompt_sha256"] == ba["canonical_prompt_sha256"]
    audit.verify_renderings(plan)
    assert len(plan["prompts"]) == 4 and len(plan["cells"]) == 20 and not plan["derivative_cells"]
    assert [c["phase"] for c in plan["cells"]] == ["baseline"] * 4 + ["edit"] * 8 + ["replay"] * 8
    assert [c["requested"] for c in plan["cells"][4:]] == ["preserve", "comply"] * 8
    assert [c["replay_of"] for c in plan["cells"][12:]] == [
        c["cell_id"] for c in plan["cells"][4:12]
    ]
    assert len({p["prompt_id"] for p in prompts}) == 4


def test_candidate_chains_same_eight_fit_ids_disjoint_case_and_exact_C_roundoff():
    plan = protocol.build_plan()
    saved = protocol.candidates()
    fitted = None
    for target, meta in plan["candidates"].items():
        assert meta["audit_before_freeze_verified"] and meta["selected_case_not_fitted"]
        assert len(meta["fitted_prompt_ids"]) == 8
        assert not any("cg_f03" in x for x in meta["fitted_prompt_ids"])
        assert protocol.vector_sha(saved[target]["vector"]) == meta["vector_float64_le_sha256"]
        assert protocol.norm(saved[target]["vector"]) == meta["norm"]
        assert fitted is None or fitted == meta["fitted_prompt_ids"]
        fitted = meta["fitted_prompt_ids"]
    assert plan["candidates"]["preserve"]["norm"] == 0.08508063610056309
    assert plan["candidates"]["comply"]["norm"] == 0.20000000000000004
    assert plan["candidates"]["comply"]["norm"] != 0.20


def test_own_rendering_states_labels_not_display_positions_and_full_coverage(tmp_path):
    plan, backend, vectors, f, rows, verified = execute(tmp_path)
    assert f.attempts == f.completed == len(backend.model.calls) == 20
    assert (tmp_path / "derivative_events.jsonl").read_text() == ""
    assert [r["requested_label"] for r in rows[4:12]] == ["A", "B", "A", "B", "B", "A", "B", "A"]
    assert len({tuple(r["h0"]) for r in rows[:4]}) == 4
    baselines = {r["prompt_id"]: r for r in rows[:4]}
    for r in rows:
        b = baselines[r["prompt_id"]]
        assert r["h0"] == b["h0"] == b["h"]
        assert r["h0_norm"] == b["h0_norm"]
        assert r["unselected_max_difference"] == 0 and r["weights_unchanged"]
        assert r["letter_log_odds"] == (
            r["preserve_log_odds"] if r["preserve_label"] == "A" else -r["preserve_log_odds"]
        )
        assert r["delta_letter_log_odds"] == r["letter_log_odds"] - b["letter_log_odds"]
        assert r["signed_delta_log_odds"] == r["target_sign"] * r["delta_log_odds"]
        if r["requested"]:
            assert r["intended_delta"] == [
                audit.f32(b["h0_norm"] * x) for x in vectors[r["requested"]]
            ]
    assert all(a["h"] == b["h"] for a, b in zip(rows[4:12], rows[12:], strict=True))
    s = verified["summary"]
    assert s["matrix"]["strict_accepted"] == 8 and s["matrix"]["matrix_pass"]
    assert s["baseline_availability"] == {"A": 2, "B": 2, "OTHER": 0}
    assert s["matrix"]["eligible_A_to_B"] == s["matrix"]["achieved_A_to_B"] == 2
    assert s["matrix"]["eligible_B_to_A"] == s["matrix"]["achieved_B_to_A"] == 2
    assert s["matrix"]["accepted_flips"] == s["matrix"]["accepted_retentions"] == 4
    for c in s["per_vector"].values():
        assert c["eligible_A_to_B"] == c["achieved_A_to_B"] == 1
        assert c["eligible_B_to_A"] == c["achieved_B_to_A"] == 1
    assert (
        len(s["coverage_by_vector_mapping_display"]) == 8 and len(s["descriptive_contrasts"]) == 8
    )
    assert s["replay_matches"] == 8 and not s["replays_are_new_examples"]
    assert (
        backend.model.active_hooks == []
        and backend.model.weight.requires_grad
        and backend.model.weight.grad is None
    )
    with pytest.raises(ValueError):
        f.begin(plan["cells"][-1])


def test_missing_directional_coverage_does_not_invalidate_individual_passes(tmp_path):
    *_, verified = execute(tmp_path, mode="all_B")
    s = verified["summary"]
    assert s["matrix"]["matrix_pass"] and s["matrix"]["strict_accepted"] == 8
    assert s["baseline_availability"] == {"A": 0, "B": 4, "OTHER": 0}
    assert s["matrix"]["eligible_A_to_B"] == s["matrix"]["achieved_A_to_B"] == 0
    assert (
        not s["matrix"]["eligible_both_directions"] and not s["matrix"]["achieved_both_directions"]
    )


@pytest.mark.parametrize(
    "kwargs", [{"quality_failure": True}, {"mode": "weak"}, {"mode": "weak_margin"}]
)
def test_finite_scientific_failures_complete_all20_no_gate_changes(tmp_path, kwargs):
    *_, f, rows, verified = execute(tmp_path, **kwargs)
    assert f.attempts == f.completed == 20 and len(rows) == 20
    assert verified["summary"]["matrix"]["strict_accepted"] < 8
    assert not verified["summary"]["matrix"]["matrix_pass"]
    assert all(r["replay_consistent"] for r in rows[12:])


@pytest.mark.parametrize(
    "kwargs,attempts",
    [
        ({"offset": 0.01}, 1),
        ({"mode": "nonfinite"}, 1),
        ({"mode": "gradient"}, 1),
        ({"mode": "weight_change"}, 1),
        ({"mode": "replay_mismatch"}, 13),
    ],
)
def test_technical_faults_abort_without_padding_or_retry(tmp_path, kwargs, attempts):
    plan, backend, vectors = prepare(**kwargs)
    f = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    grad, back = torch.autograd.grad, torch.autograd.backward
    with pytest.raises(ValueError):
        job.evaluate(plan, backend, vectors, f, tmp_path)
    assert f.attempts == attempts
    assert torch.autograd.grad is grad and torch.autograd.backward is back
    assert backend.model.weight.requires_grad


@pytest.mark.parametrize("fault", ["scale", "sign", "swapped", "rounded_C"])
def test_vector_coordinate_changes_rejected_before_any_forward(tmp_path, fault):
    plan, backend, vectors = prepare()
    if fault == "scale":
        vectors["preserve"] = [x * 0.05 for x in vectors["preserve"]]
    elif fault == "sign":
        vectors["comply"] = [-x for x in vectors["comply"]]
    elif fault == "swapped":
        vectors["preserve"], vectors["comply"] = vectors["comply"], vectors["preserve"]
    else:
        vectors["comply"][0] = -0.20
    f = job.base.Ledger(tmp_path, plan["cells"], time.monotonic() + 100)
    with pytest.raises(ValueError, match="binding"):
        job.evaluate(plan, backend, vectors, f, tmp_path)
    assert f.attempts == 0 and not backend.model.calls


@pytest.mark.parametrize("fault", ["semantic", "display", "action", "suffix", "order"])
def test_independent_rendering_corruption_rejection(fault):
    plan = copy.deepcopy(protocol.build_plan())
    p = plan["prompts"][1]
    if fault == "semantic":
        p["semantic_to_letter"]["preserve"] = "B"
    elif fault == "display":
        p["display_position_to_letter"] = {"first": "A", "second": "B"}
    elif fault == "action":
        p["option_line_by_letter"]["A"] = "A) different\n"
    elif fault == "suffix":
        p["prompt"] += " "
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
        "raw_L",
        "delta_L",
        "signed_delta",
        "replay",
        "coverage",
    ],
)
def test_independent_raw_and_coverage_corruption_rejection(tmp_path, fault):
    plan, _, vectors, _, rows, verified = execute(tmp_path)
    changed = copy.deepcopy(rows)
    r = changed[4]
    if fault == "label":
        r["requested_label"] = "B"
    elif fault == "candidate":
        r["candidate_vector_sha256"] = "wrong"
    elif fault == "offset_hash":
        r["offset_float32_le_sha256"] = "wrong"
    elif fault == "cast":
        r["intended_delta"][0] += 0.01
    elif fault == "raw_L":
        r["letter_log_odds"] += 1e-8
    elif fault == "delta_L":
        r["delta_letter_log_odds"] += 1e-8
    elif fault == "signed_delta":
        r["signed_delta_log_odds"] += 1e-8
    elif fault == "replay":
        changed[12]["maximum_replay_h_difference"] += 0.001
    else:
        summary = copy.deepcopy(verified["summary"])
        summary["matrix"]["achieved_A_to_B"] += 1
        with pytest.raises(ValueError):
            audit.compare_summary(summary, verified["summary"])
        return
    with pytest.raises(ValueError):
        audit.verify_data(plan, changed, vectors, tmp_path)


@pytest.mark.parametrize("target", ["preserve", "comply"])
@pytest.mark.parametrize("fault", ["fitted_ids", "freeze", "audit"])
def test_candidate_audit_chain_corruption(monkeypatch, target, fault):
    cfg = protocol.read(protocol.ROOT / protocol.CONFIG)
    prompts = protocol.build_plan()["prompts"]
    spec = cfg["candidates"][target]
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


def test_storage_guard_and_no_historical_f03_outcome_reads(monkeypatch):
    cfg = protocol.read(protocol.ROOT / protocol.CONFIG)
    monkeypatch.setattr(
        protocol.shutil, "disk_usage", lambda root: SimpleNamespace(free=64 * 1024**2)
    )
    assert protocol.storage_preflight(protocol.ROOT, cfg)["bounds"]["total_bound_bytes"] == 57620636
    monkeypatch.setattr(protocol.shutil, "disk_usage", lambda root: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="storage"):
        protocol.storage_preflight(protocol.ROOT, cfg)
    original = protocol.Path.read_bytes

    def no_old_f03(path):
        assert "evidence/frozen_preserve_f03_v1" not in str(path).replace("\\", "/")
        return original(path)

    monkeypatch.setattr(protocol.Path, "read_bytes", no_old_f03)
    assert len(protocol.build_plan()["cells"]) == 20


def source(fn):
    return textwrap.dedent(inspect.getsource(fn))


def same(a, b):
    assert ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))


def test_parent_physical_scoring_guard_and_new_L_metadata_structure():
    same(source(parent.make_delta), source(job.make_delta))
    same(source(parent.assess), source(job.assess))
    assert job.DerivativeGuard is parent.DerivativeGuard and job.recorder is parent.recorder
    expected = (
        source(parent.evaluate)
        .replace("/10 forwards", "/20 forwards")
        .replace("len(rows) == 10", "len(rows) == 20")
        .replace("10/0 accounting", "20/0 accounting")
        .replace(
            '                "choice_a_token_id": boundary.a_token_id,',
            '                "letter_log_odds": float(logits[boundary.a_token_id]) - float(logits[boundary.b_token_id]),\n                "choice_a_token_id": boundary.a_token_id,',
        )
        .replace(
            '            baseline_margin=baseline["preserve_log_odds"],',
            '            baseline_margin=baseline["preserve_log_odds"],\n            baseline_letter_log_odds=baseline["letter_log_odds"],\n            delta_letter_log_odds=row["letter_log_odds"] - baseline["letter_log_odds"],',
        )
        .replace(
            "for k in NUMERIC_FIELDS)",
            'for k in NUMERIC_FIELDS + ("letter_log_odds", "delta_letter_log_odds", "signed_delta_log_odds"))',
        )
    )
    same(expected, source(job.evaluate))
    expected = (
        source(parent_audit.verify_data)
        .replace(
            "def verify_data(plan, rows, vectors, output):",
            "def verify_data(plan, rows, vectors, output):\n    verify_renderings(plan)",
        )
        .replace("len(rows) == 10", "len(rows) == 20")
        .replace("exact10-cell sequence", "exact20-cell sequence")
        .replace("iterdir())) == 10", "iterdir())) == 20")
        .replace('plan["cells"][:2]] == ["baseline"] * 2', 'plan["cells"][:4]] == ["baseline"] * 4')
        .replace(
            "        current = {**row, **measured}",
            '        letter = float(logits[row["choice_a_token_id"]]) - float(logits[row["choice_b_token_id"]])\n        baseline_letter = float(baseline_logits[row["choice_a_token_id"]]) - float(baseline_logits[row["choice_b_token_id"]])\n        require(row["letter_log_odds"] == letter and row["baseline_letter_log_odds"] == baseline_letter\n                and row["delta_letter_log_odds"] == letter - baseline_letter,\n                "exact direct raw L and deltaL")\n        current = {**row, **measured}',
        )
        .replace(
            "for k in NUMERIC_FIELDS)",
            'for k in NUMERIC_FIELDS + ("letter_log_odds", "delta_letter_log_odds", "signed_delta_log_odds"))',
        )
    )
    same(expected, source(audit.verify_data))
    assert "2fc06364715b967f1860aea9cf38778875588b17" in source(audit.verify)
    assert source(job.worker).index("storage_preflight") < source(job.worker).index("load_backend")


def test_report_tables_and_contrast_arithmetic(tmp_path):
    *_, verified = execute(tmp_path)
    result = {
        "status": "INDEPENDENT_NUMERIC_GEOMETRY_REPLAY_SCHEDULE_MATCH",
        "runtime": {"elapsed_seconds": 1.0},
        **verified,
    }
    report = audit.report(result)
    width = None
    for line in report.splitlines():
        if not line.startswith("|"):
            width = None
            continue
        count = line.count("|")
        if width is None:
            width = count
        assert count == width
    indexed = {r["cell_id"]: r for r in verified["summary"]["cells"]}
    for c in verified["summary"]["descriptive_contrasts"]:
        left, right = indexed[c["left_cell_id"]], indexed[c["right_cell_id"]]
        assert all(value == right[k] - left[k] for k, value in c["right_minus_left"].items())


def test_candidate_authentication_before_model_load(tmp_path, monkeypatch):
    cfg = copy.deepcopy(protocol.read(protocol.ROOT / protocol.CONFIG))
    bad = tmp_path / "changed_candidate.json"
    bad.write_text("{}")
    cfg["candidates"]["comply"]["path"] = str(bad)
    with pytest.raises(ValueError, match="authenticated"):
        protocol.candidates(cfg)
    output = tmp_path / "job"
    output.mkdir()
    job.base.write_new(output / "RUN_STARTED.json", {"deadline_monotonic": time.monotonic() + 100})
    plan, _, _ = prepare()
    monkeypatch.setattr(job, "ROOT", tmp_path)
    monkeypatch.setattr(job, "OUTPUT", "job")
    monkeypatch.setattr(job, "require_freeze", lambda: {"plan": plan})
    monkeypatch.setattr(protocol, "candidates", lambda: protocol.authenticated(str(bad), "wrong"))

    def forbidden(*args):
        raise AssertionError("model load before candidate authentication")

    monkeypatch.setattr(job.base, "load_backend", forbidden)
    with pytest.raises(ValueError, match="authenticated"):
        job.worker()
    assert (output / "INVALID.json").exists()


def test_whole_job_timeout_no_retry(tmp_path, monkeypatch):
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
    result = job.supervise(["fake"], tmp_path, {"standard_used_percent": 26}, timeout=0.01)
    assert fake.killed and result["status"] == "INCONCLUSIVE" and not result["retries_allowed"]
    assert protocol.read(tmp_path / "RUN_STARTED.json")["forward_ceiling"] == 20


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {"standard_used_percent": 90, "checked_at_unix": 0},
        {"standard_used_percent": 26, "checked_at_unix": 0},
    ],
)
def test_usage_preflight_blocks_missing_capped_or_stale(monkeypatch, usage):
    monkeypatch.setattr(job, "require_freeze", lambda: {"source_commit": "source"})

    def git(*args):
        if "show" in args:
            return protocol.OUTPUT + "/preregistration.json"
        if "status" in args:
            return ""
        return "source"

    monkeypatch.setattr(job.base, "git", git)
    monkeypatch.setenv("SP_LENSE_USAGE_PREFLIGHT", json.dumps(usage))
    with pytest.raises(ValueError, match="fresh usage"):
        job.run()
