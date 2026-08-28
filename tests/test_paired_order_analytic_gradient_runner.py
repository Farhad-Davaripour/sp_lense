from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _runner() -> object:
    path = ROOT / "scripts" / "paired_order_analytic_gradient_development.py"
    spec = importlib.util.spec_from_file_location("sp_lense_poag_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_locked_opened_plan_has_exact_pair_and_order_coverage_without_model_call() -> None:
    runner = _runner()
    config = runner._load_config()
    _, groups = runner._active_jobs()

    assert config["intervention"]["reserve_candidates"] == [0.01, 0.03, 0.05, 0.1]
    assert len(groups) == 16
    assert sum(len(jobs) for _, jobs in groups) == 32
    assert all([bool(job["preserve_first"]) for job in jobs] == [False, True] for _, jobs in groups)
    assert len({job["prompt_sha256"] for _, jobs in groups for job in jobs}) == 32


def test_final_head_jvp_uses_only_declared_residual_and_vector() -> None:
    runner = _runner()
    weight = torch.tensor([[1.0, -1.0, 0.5], [2.0, 0.0, -0.5]], dtype=torch.float32)
    backend = SimpleNamespace(
        torch=torch,
        device=torch.device("cpu"),
        model=SimpleNamespace(
            ln_final=lambda value: 2.0 * value,
            unembed=lambda value: value @ weight,
        ),
    )
    residual = torch.tensor([3.0, 4.0])
    vector = torch.tensor([0.25, -0.5])

    primal, tangent = runner._head_jvp(backend, residual, vector)

    assert primal == pytest.approx((2.0 * residual) @ weight)
    assert tangent == pytest.approx((2.0 * vector) @ weight)


def test_capture_hook_accepts_transformer_lens_keyword_abi() -> None:
    runner = _runner()

    class HookContext:
        def __init__(self, model, callback):
            self.model = model
            self.callback = callback

        def __enter__(self):
            self.model.callback = self.callback

        def __exit__(self, exc_type, exc_value, traceback):
            self.model.callback = None

    class FakeModel:
        callback = None

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none is True

        def hooks(self, *, fwd_hooks):
            assert fwd_hooks[0][0] == "blocks.23.hook_out"
            return HookContext(self, fwd_hooks[0][1])

        def __call__(self, tokens):
            assert tuple(tokens.shape) == (1, 1)
            activation = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32)
            hooked = self.callback(activation, hook=SimpleNamespace(name="hook_out"))
            hidden = hooked[0, 0]
            logits = torch.stack([hidden[0] + 2.0 * hidden[1], -hidden[0], hidden[1], -hidden[1]])
            return logits.view(1, 1, -1)

    boundary = SimpleNamespace(
        prompt_length=1,
        token_id=lambda label: {"A": 0, "B": 1}[label],
        evidence_sha256="a" * 64,
        prompt_prefix_token_ids_sha256="b" * 64,
    )
    backend = SimpleNamespace(
        torch=torch,
        model=FakeModel(),
        encode=lambda prompt: torch.tensor([[1]], dtype=torch.long),
    )
    adaptive = SimpleNamespace(
        resolve_choice_boundary=lambda backend, prompt: boundary,
        tensor_float32_sha256=runner._raw_tensor_hash,
    )
    job = {
        "prompt": "test",
        "unit_id": "unit",
        "preserve_first": True,
        "prompt_sha256": "c" * 64,
        "positive_label": "A",
        "negative_label": "B",
    }

    row = runner._capture_order(backend, adaptive, job, layer=23)

    assert row["baseline_semantic_choice"] == "positive"
    assert row["semantic_gradient"] == pytest.approx(torch.tensor([2.0, 2.0]))
    assert row["semantic_gradient_norm"] > 0.0


def test_public_capture_manifest_excludes_large_tensors() -> None:
    runner = _runner()
    public = runner._public_order_record(
        {
            "unit_id": "u",
            "residual": torch.ones(2),
            "semantic_gradient": torch.ones(2),
            "baseline_logits": torch.ones(3),
            "base_logit_jvp": torch.ones(3),
            "baseline_logits_sha256": "a" * 64,
        }
    )

    assert public == {"unit_id": "u", "baseline_logits_sha256": "a" * 64}


def test_public_construction_manifest_never_serializes_delta_bytes() -> None:
    runner = _runner()
    public = runner._construction_public(
        {"eligible": True, "delta": torch.ones(2), "delta_sha256": "b" * 64}
    )

    assert public == {"eligible": True, "delta_sha256": "b" * 64}


def test_aggregate_recomputes_all_four_cell_gate_and_same_delta() -> None:
    runner = _runner()
    delta_hash = "d" * 64
    constructions = {
        "candidates": [
            {
                "reserve_logit": 0.01,
                "eligible_pair_count": 1,
                "pair_count": 1,
                "pairs": [
                    {
                        "case_id": "case",
                        "assignment": 0,
                        "delta_sha256": delta_hash,
                        "alpha": 0.02,
                    }
                ],
            }
        ]
    }
    capture = {
        "pairs": [
            {
                "case_id": "case",
                "assignment": 0,
                "orders": [
                    {
                        "preserve_first": False,
                        "prompt_sha256": "a" * 64,
                        "baseline_semantic_choice": "positive",
                        "baseline_argmax_token_id": 33,
                    },
                    {
                        "preserve_first": True,
                        "prompt_sha256": "b" * 64,
                        "baseline_semantic_choice": "positive",
                        "baseline_argmax_token_id": 32,
                    },
                ],
            }
        ]
    }
    cells = {}
    for order in (0, 1):
        for name, semantic, changed in (
            ("plus", "positive", False),
            ("minus", "negative", True),
        ):
            work_id = f"0:case:0:{order}:{name}"
            cells[work_id] = {
                "work_id": work_id,
                "target_met": True,
                "decision_changed": changed,
                "semantic_choice": semantic,
                "unsigned_delta_sha256": delta_hash,
            }

    result = runner._aggregate_evaluation(constructions, capture, cells)

    assert result[0]["passes"] is True
    assert result[0]["successful_pair_count"] == 1
    cells["0:case:0:1:minus"]["unsigned_delta_sha256"] = "x" * 64
    assert runner._aggregate_evaluation(constructions, capture, cells)[0]["passes"] is False


def test_structured_router_audit_assigns_exact_zero_to_all_locked_off_gate_rows() -> None:
    runner = _runner()
    adaptive, _ = runner._active_jobs()

    audit = runner._routing_audit(adaptive, torch)

    assert audit["decision_order_count"] == 128
    assert audit["active_decision_order_count"] == 32
    assert audit["zero_routed_decision_order_count"] == 96
    assert audit["zero_routed_collateral_count"] == 16
    assert audit["nonzero_off_gate_count"] == 0
    assert audit["router_scope"] == ("locked_structured_dataset_renderer_not_open_domain_free_text")


def test_protocol_guard_requires_every_bound_path_to_be_tracked_and_clean(monkeypatch) -> None:
    runner = _runner()
    calls = []

    def clean_git(*args: str) -> str:
        calls.append(args)
        return ""

    monkeypatch.setattr(runner, "_git", clean_git)
    runner._require_committed_clean([runner.CONFIG_PATH], stage="test protocol")
    assert calls[0][:2] == ("ls-files", "--error-unmatch")
    assert calls[-1][:2] == ("status", "--porcelain")

    monkeypatch.setattr(
        runner,
        "_git",
        lambda *args: (
            " M configs/paired_order_analytic_gradient_development_lock.json"
            if args[:2] == ("status", "--porcelain")
            else ""
        ),
    )
    with pytest.raises(RuntimeError, match="committed and clean"):
        runner._require_committed_clean([runner.CONFIG_PATH], stage="test protocol")


def test_attempt_ledger_fails_closed_until_returned_call_is_committed(
    tmp_path: Path,
) -> None:
    runner = _runner()
    path = tmp_path / "attempts.json"
    ledger = runner._new_attempt_ledger("capture")

    runner._reserve_attempt(
        path,
        ledger,
        work_id="capture:case:0:0:forward_backward",
        operation="full_forward_and_backward",
    )
    loaded = runner._load_attempt_ledger(path, phase="capture")
    with pytest.raises(RuntimeError, match="ambiguously interrupted"):
        runner._require_unambiguous_attempts(loaded, phase="capture")

    runner._mark_attempt_returned(path, ledger, "capture:case:0:0:forward_backward")
    with pytest.raises(RuntimeError, match="ambiguously interrupted"):
        runner._require_unambiguous_attempts(ledger, phase="capture")

    runner._commit_attempts(path, ledger, ["capture:case:0:0:forward_backward"])
    runner._validate_attempt_coverage(
        runner._load_attempt_ledger(path, phase="capture"),
        {"capture:case:0:0:forward_backward": "full_forward_and_backward"},
        phase="capture",
    )


def _valid_evaluation_cell(runner):
    delta = torch.tensor([3.0, 4.0], dtype=torch.float32)
    captured_order = {
        "preserve_token_id": 10,
        "comply_token_id": 11,
        "preserve_first": True,
        "prompt_sha256": "a" * 64,
        "baseline_semantic_choice": "negative",
        "baseline_argmax_token_id": 11,
        "choice_boundary_evidence_sha256": "b" * 64,
        "residual_norm": 10.0,
    }
    work = {
        "work_id": "0:case:0:0:plus",
        "candidate_index": 0,
        "reserve_logit": 0.01,
        "construction": {"delta": delta, "delta_sha256": runner._raw_tensor_hash(delta)},
        "key": ("case", 0),
        "captured_order": captured_order,
        "job": {"positive_label": "A", "negative_label": "B"},
        "order_index": 0,
        "name": "plus",
        "sign": 1.0,
        "wanted": "positive",
    }
    cell = {
        "work_id": work["work_id"],
        "ordinal": 0,
        "candidate_index": 0,
        "reserve_logit": 0.01,
        "case_id": "case",
        "assignment": 0,
        "order_index": 0,
        "preserve_first": True,
        "prompt_sha256": "a" * 64,
        "condition": "plus",
        "sign": 1.0,
        "wanted_semantic_choice": "positive",
        "baseline_semantic_choice": "negative",
        "baseline_argmax_token_id": 11,
        "unsigned_delta_sha256": runner._raw_tensor_hash(delta),
        "signed_delta_sha256": runner._raw_tensor_hash(delta),
        "intended_delta_l2_norm": 5.0,
        "realized_delta_l2_norm": 5.0,
        "realized_norm_absolute_error": 0.0,
        "realized_norm_allowance": 6e-5,
        "predicted_label": "A",
        "exact_argmax_token_id": 10,
        "preserve_minus_comply_log_odds": 0.0,
        "preserve_pair_probability": 0.5,
        "pair_choice": "A",
        "answer_pair_mass": 0.8,
        "full_vocabulary_kl_changed_to_baseline": 0.1,
        "perturbation": {
            "n_positions": 1,
            "total_frobenius_norm": 5.0,
            "mean_l2_norm": 5.0,
            "rms_l2_norm": 5.0,
            "max_l2_norm": 5.0,
            "mean_relative_l2_norm": 0.5,
            "max_relative_l2_norm": 0.5,
            "zero_reference_positions": 0,
        },
        "choice_boundary_evidence_sha256": "b" * 64,
        "choice_a_token_id": 10,
        "choice_b_token_id": 11,
        "semantic_choice": "positive",
        "target_met": True,
        "decision_changed": True,
    }
    runner._seal_evaluation_cell(cell, None)
    adaptive = SimpleNamespace(tensor_float32_sha256=runner._raw_tensor_hash)
    return cell, work, adaptive


def test_cell_semantics_are_derived_from_exact_argmax_even_after_coherent_reseal() -> None:
    runner = _runner()
    cell, work, adaptive = _valid_evaluation_cell(runner)
    runner._validate_evaluation_cell(cell, work, adaptive, ordinal=0, previous_sha256=None)

    tampered = copy.deepcopy(cell)
    tampered.update(
        {
            "predicted_label": "B",
            "semantic_choice": "negative",
            "target_met": False,
            "decision_changed": False,
        }
    )
    runner._seal_evaluation_cell(tampered, None)
    with pytest.raises(RuntimeError, match="predicted_label"):
        runner._validate_evaluation_cell(tampered, work, adaptive, ordinal=0, previous_sha256=None)


def test_cell_hash_rejects_metric_corruption() -> None:
    runner = _runner()
    cell, work, adaptive = _valid_evaluation_cell(runner)
    cell["full_vocabulary_kl_changed_to_baseline"] = 0.2

    with pytest.raises(RuntimeError, match="embedded hash"):
        runner._validate_evaluation_cell(cell, work, adaptive, ordinal=0, previous_sha256=None)


def test_cell_exact_argmax_must_agree_with_log_odds_after_coherent_reseal() -> None:
    runner = _runner()
    cell, work, adaptive = _valid_evaluation_cell(runner)
    cell.update(
        {
            "preserve_minus_comply_log_odds": -2.0,
            "preserve_pair_probability": 0.11920292202211755,
            "pair_choice": "B",
        }
    )
    runner._seal_evaluation_cell(cell, None)

    with pytest.raises(RuntimeError, match="exact argmax disagrees"):
        runner._validate_evaluation_cell(cell, work, adaptive, ordinal=0, previous_sha256=None)


def test_attempt_coverage_requires_the_exact_locked_execution_order() -> None:
    runner = _runner()
    ledger = runner._new_attempt_ledger("capture")
    ledger["attempts"] = [
        {"work_id": "second", "operation": "op", "status": "committed_to_artifact"},
        {"work_id": "first", "operation": "op", "status": "committed_to_artifact"},
    ]

    with pytest.raises(RuntimeError, match="differs from artifact coverage"):
        runner._validate_attempt_coverage(ledger, {"first": "op", "second": "op"}, phase="capture")
