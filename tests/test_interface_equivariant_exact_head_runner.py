from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _runner() -> object:
    path = ROOT / "scripts" / "interface_equivariant_exact_head_development.py"
    spec = importlib.util.spec_from_file_location("sp_lense_ieeh_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_method_excludes_all_tensor_bytes() -> None:
    runner = _runner()
    public = runner._public_method(
        {
            "method_id": "gradient_ray",
            "base_vectors": torch.ones((2, 3)),
            "delta": torch.ones((2, 3)),
            "delta_sha256": "a" * 64,
        }
    )

    assert public == {"method_id": "gradient_ray", "delta_sha256": "a" * 64}


def test_architecture_guard_requires_zero_vocabulary_constant_bias() -> None:
    runner = _runner()

    class Qwen3_5RMSNorm:
        def __init__(self) -> None:
            self.eps = 1e-6
            self.weight = torch.tensor([0.0, -0.25], dtype=torch.float32)

    class RMSNormalizationBridge:
        def __init__(self) -> None:
            self.original_component = Qwen3_5RMSNorm()
            self.use_native_layernorm_autograd = True
            self.uses_rms_norm = True

    model = SimpleNamespace(
        cfg=SimpleNamespace(n_layers=24, normalization_type="RMS"),
        compatibility_mode=False,
        _weights_processed=False,
        ln_final=RMSNormalizationBridge(),
        unembed=SimpleNamespace(
            W_U=torch.ones((2, 3)),
            b_U=torch.zeros(3),
        ),
    )
    backend = SimpleNamespace(
        torch=torch,
        model=model,
        device="cpu",
        dtype_name="float32",
    )
    config = {
        "model": {"residual_width": 2, "rms_epsilon": 1e-6},
        "intervention": {"residual_layer_zero_based": 23},
        "architecture_guards": {
            "expected_final_norm_bridge_class": (
                f"{RMSNormalizationBridge.__module__}.{RMSNormalizationBridge.__name__}"
            ),
            "expected_final_norm_wrapped_class": (
                f"{Qwen3_5RMSNorm.__module__}.{Qwen3_5RMSNorm.__name__}"
            ),
            "expected_normalization_type": "RMS",
            "rms_scale_parameterization": "one_plus_raw_weight_float32",
        },
    }

    observed = runner._architecture(backend, config)

    assert torch.equal(observed["gamma"], torch.tensor([1.0, 0.75]))
    assert observed["public"]["rms_scale_parameterization"] == ("one_plus_raw_weight_float32")
    assert (
        observed["public"]["raw_rms_weight_float32_sha256"]
        != observed["public"]["gamma_float32_sha256"]
    )
    assert observed["public"]["unembedding_bias_exactly_zero"] is True
    model.unembed.b_U = torch.tensor([0.0, 0.0, 1.0])
    with pytest.raises(RuntimeError, match="vocabulary-constant"):
        runner._architecture(backend, config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("use_native_layernorm_autograd", False, "native Hugging Face"),
        ("uses_rms_norm", False, "RMS semantics"),
    ],
)
def test_architecture_guard_rejects_non_native_or_non_rms_bridge(
    field: str, value: bool, message: str
) -> None:
    runner = _runner()

    class Qwen3_5RMSNorm:
        eps = 1e-6
        weight = torch.zeros(2, dtype=torch.float32)

    class RMSNormalizationBridge:
        original_component = Qwen3_5RMSNorm()
        use_native_layernorm_autograd = True
        uses_rms_norm = True

    bridge = RMSNormalizationBridge()
    setattr(bridge, field, value)
    backend = SimpleNamespace(
        torch=torch,
        device="cpu",
        dtype_name="float32",
        model=SimpleNamespace(
            cfg=SimpleNamespace(n_layers=24, normalization_type="RMS"),
            compatibility_mode=False,
            _weights_processed=False,
            ln_final=bridge,
            unembed=SimpleNamespace(W_U=torch.ones((2, 3)), b_U=torch.zeros(3)),
        ),
    )
    config = {
        "model": {"residual_width": 2, "rms_epsilon": 1e-6},
        "intervention": {"residual_layer_zero_based": 23},
        "architecture_guards": {
            "expected_final_norm_bridge_class": runner._qualified_class_name(bridge),
            "expected_final_norm_wrapped_class": runner._qualified_class_name(
                bridge.original_component
            ),
            "expected_normalization_type": "RMS",
            "rms_scale_parameterization": "one_plus_raw_weight_float32",
        },
    }

    with pytest.raises(RuntimeError, match=message):
        runner._architecture(backend, config)


@pytest.mark.parametrize(
    ("target", "value", "message"),
    [
        ("compatibility_mode", True, "compatibility mode"),
        ("_weights_processed", True, "weights were processed"),
        ("device", "cuda", "CPU float32"),
        ("dtype_name", "bfloat16", "CPU float32"),
    ],
)
def test_architecture_guard_rejects_processed_or_wrong_precision(
    target: str, value: object, message: str
) -> None:
    runner = _runner()

    class Qwen3_5RMSNorm:
        eps = 1e-6
        weight = torch.zeros(2, dtype=torch.float32)

    class RMSNormalizationBridge:
        original_component = Qwen3_5RMSNorm()
        use_native_layernorm_autograd = True
        uses_rms_norm = True

    bridge = RMSNormalizationBridge()
    model = SimpleNamespace(
        cfg=SimpleNamespace(n_layers=24, normalization_type="RMS"),
        compatibility_mode=False,
        _weights_processed=False,
        ln_final=bridge,
        unembed=SimpleNamespace(W_U=torch.ones((2, 3)), b_U=torch.zeros(3)),
    )
    backend = SimpleNamespace(
        torch=torch,
        device="cpu",
        dtype_name="float32",
        model=model,
    )
    if target in {"compatibility_mode", "_weights_processed"}:
        setattr(model, target, value)
    else:
        setattr(backend, target, value)
    config = {
        "model": {"residual_width": 2, "rms_epsilon": 1e-6},
        "intervention": {"residual_layer_zero_based": 23},
        "architecture_guards": {
            "expected_final_norm_bridge_class": runner._qualified_class_name(bridge),
            "expected_final_norm_wrapped_class": runner._qualified_class_name(
                bridge.original_component
            ),
            "expected_normalization_type": "RMS",
            "rms_scale_parameterization": "one_plus_raw_weight_float32",
        },
    }

    with pytest.raises(RuntimeError, match=message):
        runner._architecture(backend, config)


def test_qwen_one_plus_weight_head_matches_exact_numerator_formula() -> None:
    residuals = torch.tensor([[2.0, -1.0], [-0.5, 3.0]], dtype=torch.float32)
    raw_weight = torch.tensor([0.0, -0.25], dtype=torch.float32)
    gamma = raw_weight.add(1.0)
    weights = torch.tensor([[1.0, -2.0, 0.5], [0.25, 1.5, -1.0]], dtype=torch.float32)
    epsilon = 1e-6

    inverse_rms = torch.rsqrt(residuals.square().mean(dim=1, keepdim=True) + epsilon)
    native_order = ((residuals * inverse_rms) * gamma) @ weights
    numerator_order = ((residuals * gamma) @ weights) * inverse_rms

    torch.testing.assert_close(native_order, numerator_order, rtol=1e-6, atol=1e-6)


def test_actual_head_certificate_checks_both_orders_and_exact_signs() -> None:
    runner = _runner()
    weights = torch.tensor([[1.0, -1.0, 0.0], [0.0, 0.0, -2.0]])
    backend = SimpleNamespace(
        torch=torch,
        device=torch.device("cpu"),
        model=SimpleNamespace(
            ln_final=lambda value: value,
            unembed=lambda value: value @ weights,
        ),
    )
    residuals = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)
    delta = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32)

    certificate = runner._actual_head_certificate(
        backend,
        residuals,
        delta,
        (0, 1),
        (1, 0),
        acceptance_reserve=0.1,
    )

    assert len(certificate["rows"]) == 4
    assert all(row["target_met"] for row in certificate["rows"])
    assert len(set(certificate["signed_cell_delta_sha256"])) == 4

    orders = [
        {
            "baseline_logits": torch.zeros(3),
            "preserve_token_id": 0,
            "comply_token_id": 1,
        },
        {
            "baseline_logits": torch.zeros(3),
            "preserve_token_id": 1,
            "comply_token_id": 0,
        },
    ]
    runner._validate_actual_head_certificate(
        torch,
        certificate,
        delta,
        orders,
        acceptance_reserve=0.1,
        require_all_targets=True,
    )
    tampered = copy.deepcopy(certificate)
    tampered["rows"][0]["target_margin"] += 1.0
    with pytest.raises(RuntimeError, match="raw-logit-derived"):
        runner._validate_actual_head_certificate(
            torch,
            tampered,
            delta,
            orders,
            acceptance_reserve=0.1,
            require_all_targets=True,
        )


def _valid_cell(runner, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(runner, "LOGITS_ROOT", tmp_path)
    monkeypatch.setattr(runner, "_relative", lambda path: f"results/{path.name}")
    delta = torch.tensor([0.3, 0.4], dtype=torch.float32)
    logits = torch.full((12,), -10.0, dtype=torch.float32)
    logits[10] = 2.0
    logits[11] = 1.0
    baseline = logits.clone()
    baseline[10] = 1.0
    baseline[11] = 2.0
    changed_log_probs = torch.log_softmax(logits, dim=-1)
    baseline_log_probs = torch.log_softmax(baseline, dim=-1)
    pair_mass = float((changed_log_probs.exp()[10] + changed_log_probs.exp()[11]).item())
    kl = float((changed_log_probs.exp() * (changed_log_probs - baseline_log_probs)).sum().item())
    work = {
        "work_id": "gradient_ray:case:0:0:plus",
        "method_id": "gradient_ray",
        "case_id": "case",
        "assignment": 0,
        "order_index": 0,
        "condition": "plus",
        "sign": 1.0,
        "wanted": "positive",
        "job": {"positive_label": "A", "negative_label": "B"},
        "order": {
            "prompt_sha256": "a" * 64,
            "preserve_token_id": 10,
            "comply_token_id": 11,
            "baseline_semantic_choice": "negative",
            "baseline_logits": baseline,
            "residual_norm": 10.0,
            "residual_sha256": "e" * 64,
            "choice_boundary_evidence_sha256": "c" * 64,
        },
        "delta": delta,
        "captured_residual": torch.tensor([10.0, 0.0], dtype=torch.float32),
        "unsigned_order_delta_sha256": runner._raw_tensor_hash(delta),
        "minimum_target_margin": 0.01,
        "maximum_relative_norm": 0.1,
    }
    artifact = runner._logits_artifact_path(0, work["work_id"])
    artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(logits, artifact)
    cell = {
        "work_id": work["work_id"],
        "ordinal": 0,
        "method_id": "gradient_ray",
        "case_id": "case",
        "assignment": 0,
        "order_index": 0,
        "condition": "plus",
        "sign": 1.0,
        "wanted_semantic_choice": "positive",
        "prompt_sha256": "a" * 64,
        "unsigned_order_delta_sha256": runner._raw_tensor_hash(delta),
        "signed_delta_sha256": runner._raw_tensor_hash(delta),
        "logits_artifact_path": runner._relative(artifact),
        "logits_artifact_sha256": runner._sha256(artifact),
        "logits_float32_sha256": runner._raw_tensor_hash(logits),
        "exact_argmax_token_id": 10,
        "predicted_label": "A",
        "preserve_minus_comply_log_odds": 1.0,
        "preserve_pair_probability": 0.7310585786300049,
        "pair_choice": "A",
        "answer_pair_mass": pair_mass,
        "full_vocabulary_kl_changed_to_baseline": kl,
        "perturbation": {
            "n_positions": 1,
            "total_frobenius_norm": 0.5,
            "mean_l2_norm": 0.5,
            "rms_l2_norm": 0.5,
            "max_l2_norm": 0.5,
            "mean_relative_l2_norm": 0.05,
            "max_relative_l2_norm": 0.05,
            "zero_reference_positions": 0,
            "live_reference_l2_norm": 10.0,
            "live_reference_float32_sha256": "e" * 64,
        },
        "choice_boundary_evidence_sha256": "c" * 64,
        "target_margin": 1.0,
        "semantic_choice": "positive",
        "target_met": True,
        "decision_changed": True,
        "intended_delta_l2_norm": 0.500000011920929,
        "realized_delta_l2_norm": 0.5,
        "realized_relative_norm": 0.05,
    }
    runner._seal_cell(cell, None)
    return cell, work, logits


def test_evaluation_cell_rederives_semantics_margin_and_signed_delta(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _runner()
    cell, work, _logits = _valid_cell(runner, tmp_path, monkeypatch)
    runner._validate_cell(cell, work, ordinal=0, previous=None)

    tampered = copy.deepcopy(cell)
    tampered["target_margin"] = 0.0
    runner._seal_cell(tampered, None)
    with pytest.raises(RuntimeError, match="target margin"):
        runner._validate_cell(tampered, work, ordinal=0, previous=None)

    tampered = copy.deepcopy(cell)
    tampered["signed_delta_sha256"] = "d" * 64
    runner._seal_cell(tampered, None)
    with pytest.raises(RuntimeError, match="signed_delta_sha256"):
        runner._validate_cell(tampered, work, ordinal=0, previous=None)


def test_evaluation_cell_hash_rejects_unresealed_metric_change(tmp_path: Path, monkeypatch) -> None:
    runner = _runner()
    cell, work, _logits = _valid_cell(runner, tmp_path, monkeypatch)
    cell["full_vocabulary_kl_changed_to_baseline"] = 0.2

    with pytest.raises(RuntimeError, match="embedded hash"):
        runner._validate_cell(cell, work, ordinal=0, previous=None)


def test_bound_input_guard_requires_tracked_clean_files(monkeypatch) -> None:
    runner = _runner()
    calls = []

    def clean_git(*arguments: str) -> str:
        calls.append(arguments)
        return ""

    monkeypatch.setattr(runner, "_git", clean_git)
    runner._require_committed_clean([runner.CONFIG_PATH])
    assert calls[0][:2] == ("ls-files", "--error-unmatch")
    assert calls[-1][:2] == ("status", "--porcelain")

    monkeypatch.setattr(
        runner,
        "_git",
        lambda *arguments: " M locked" if arguments[:2] == ("status", "--porcelain") else "",
    )
    with pytest.raises(RuntimeError, match="committed and clean"):
        runner._require_committed_clean([runner.CONFIG_PATH])


def test_checkpoint_rejects_fractional_compute_counts() -> None:
    runner = _runner()
    checkpoint = {
        "schema_version": runner.CHECKPOINT_SCHEMA,
        "status": "in_progress",
        "references": {},
        "resident_head_revalidation": {"status": "not_started"},
        "pending_reservation": None,
        "cells": [],
        "compute": {
            "reserved_resident_head_revalidation_evaluations": 0,
            "completed_resident_head_revalidation_evaluations": 0,
            "reserved_intervention_forward_passes": 0.9,
            "completed_intervention_forward_passes": 0.9,
        },
    }
    checkpoint["cells_sha256"] = runner._canonical_sha256(checkpoint["cells"])
    checkpoint["checkpoint_sha256"] = runner._canonical_sha256(checkpoint)

    with pytest.raises(TypeError, match="non-negative integers"):
        runner._validate_checkpoint(checkpoint, {}, {})


def test_checkpoint_rejects_coherently_sealed_extra_compute_field() -> None:
    runner = _runner()
    checkpoint = {
        "schema_version": runner.CHECKPOINT_SCHEMA,
        "status": "in_progress",
        "references": {},
        "resident_head_revalidation": {"status": "not_started"},
        "pending_reservation": None,
        "cells": [],
        "compute": {
            "reserved_resident_head_revalidation_evaluations": 0,
            "completed_resident_head_revalidation_evaluations": 0,
            "reserved_intervention_forward_passes": 0,
            "completed_intervention_forward_passes": 0,
            "forged_extra_count": 0,
        },
    }
    checkpoint["cells_sha256"] = runner._canonical_sha256(checkpoint["cells"])
    checkpoint["checkpoint_sha256"] = runner._canonical_sha256(checkpoint)

    with pytest.raises(RuntimeError, match="extra or missing"):
        runner._validate_checkpoint(checkpoint, {}, {})


def test_paired_comparison_reports_signed_relative_norm_difference() -> None:
    runner = _runner()
    pair = {
        "case_id": "case",
        "assignment": 0,
        "methods": {
            "gradient_ray": {"alpha": 0.02, "relative_norms": [0.02, 0.02]},
            "effective_unembedding_ray": {"alpha": 0.03, "relative_norms": [0.03, 0.03]},
        },
        "gradient_unembedding_comparison": [
            {"cosine": 0.9},
            {"cosine": 0.8},
        ],
    }
    cells = {}
    for method_id, kl in (("gradient_ray", 0.004), ("effective_unembedding_ray", 0.006)):
        for order_index in range(2):
            for condition in ("plus", "minus"):
                key = f"{method_id}:case:0:{order_index}:{condition}"
                cells[key] = {"full_vocabulary_kl_changed_to_baseline": kl}

    result = runner._paired_gradient_unembedding_comparison({"pairs": [pair]}, cells)

    assert result["mean_gradient_minus_unembedding_relative_norm"] == pytest.approx(-0.01)
    assert result["mean_gradient_minus_unembedding_alpha"] == pytest.approx(-0.01)
    assert result["mean_gradient_minus_unembedding_kl"] == pytest.approx(-0.002)


def test_live_head_binding_rejects_fabricated_baseline_numerators() -> None:
    runner = _runner()
    weights = torch.tensor([[1.0, -1.0], [0.5, -0.5]], dtype=torch.float32)
    gamma = torch.ones(2, dtype=torch.float32)
    residuals = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    capture = {
        "pairs": [
            {
                "orders": [
                    {"residual": residuals[0]},
                    {"residual": residuals[1]},
                ]
            }
        ]
    }
    bank = {
        "pairs": [
            {
                "baseline_numerators": torch.full((2, 2), 7.0),
                "orders": [
                    {
                        "preserve_token_id": 0,
                        "comply_token_id": 1,
                        "actual_baseline_logits": torch.zeros(2),
                    },
                    {
                        "preserve_token_id": 1,
                        "comply_token_id": 0,
                        "actual_baseline_logits": torch.zeros(2),
                    },
                ],
                "methods": {},
            }
        ],
        "summary": {"conceptual_exact_head_evaluations": 2},
    }
    backend = SimpleNamespace(torch=torch)
    architecture = {"gamma": gamma, "weights": weights}
    config = {"model": {"rms_epsilon": 1e-6}}

    with pytest.raises(RuntimeError, match="baseline numerators differ"):
        runner._validate_construction_against_live_head(
            backend,
            bank,
            capture,
            architecture,
            config,
        )


def test_float32_realized_norm_can_exceed_the_nominal_delta_cap() -> None:
    runner = _runner()
    residual = torch.tensor([-0.0006082134204916656, 0.000287254253635183], dtype=torch.float32)
    delta = torch.tensor([-3.0173376217135228e-05, -6.0116162785561755e-05], dtype=torch.float32)
    residuals = torch.stack([residual, residual])
    deltas = torch.stack([delta, delta])

    nominal = float(delta.double().norm().item() / residual.double().norm().item())
    realized = runner._float32_realized_relative_norms(torch, residuals, deltas)

    assert nominal < 0.1
    assert max(realized) > 0.1


def test_construction_realized_norm_uses_the_exact_live_hook_precision() -> None:
    runner = _runner()
    residual = torch.tensor([-0.28280147910118103, -1.3571797609329224])
    delta = torch.tensor([0.03787217661738396, -0.13335978984832764])
    residuals = torch.stack([residual, residual])
    deltas = torch.stack([delta, delta])

    nominal = float(delta.double().norm().item() / residual.double().norm().item())
    realized_delta = (residual + delta) - residual
    double_only = float(realized_delta.double().norm().item() / residual.double().norm().item())
    live_convention = runner._float32_realized_relative_norms(torch, residuals, deltas)

    assert nominal < 0.1
    assert double_only < 0.1
    assert max(live_convention) > 0.1


def test_construct_never_overwrites_an_existing_attempt(tmp_path: Path, monkeypatch) -> None:
    runner = _runner()
    attempt = tmp_path / "construction_attempt_ledger.json"
    attempt.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "CONSTRUCTION_ATTEMPT_PATH", attempt)
    monkeypatch.setattr(runner, "CONSTRUCTION_PATH", tmp_path / "bank.pt")
    monkeypatch.setattr(runner, "CONSTRUCTION_MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(runner, "FREEZE_PATH", tmp_path / "freeze.json")
    monkeypatch.setattr(runner, "CHECKPOINT_PATH", tmp_path / "checkpoint.json")
    monkeypatch.setattr(runner, "LOGITS_ROOT", tmp_path / "logits")
    monkeypatch.setattr(runner, "RESULT_PATH", tmp_path / "result.json")
    monkeypatch.setattr(runner, "REPORT_PATH", tmp_path / "report.md")
    monkeypatch.setattr(runner, "_load_config", lambda: {"locked_inputs": {}})
    monkeypatch.setattr(runner, "_require_committed_clean", lambda paths: None)
    monkeypatch.setattr(runner, "_relative", lambda path: path.name)

    with pytest.raises(RuntimeError, match="single-attempt"):
        runner.run_construct()


def test_freeze_never_overwrites_an_existing_freeze(tmp_path: Path, monkeypatch) -> None:
    runner = _runner()
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "FREEZE_PATH", freeze)
    monkeypatch.setattr(runner, "CHECKPOINT_PATH", tmp_path / "checkpoint.json")
    monkeypatch.setattr(runner, "LOGITS_ROOT", tmp_path / "logits")
    monkeypatch.setattr(runner, "RESULT_PATH", tmp_path / "result.json")
    monkeypatch.setattr(runner, "REPORT_PATH", tmp_path / "report.md")
    monkeypatch.setattr(runner, "_load_config", dict)
    monkeypatch.setattr(runner, "_load_construction", lambda torch: {})
    monkeypatch.setattr(runner, "_validate_manifest", lambda bank: {})
    monkeypatch.setattr(runner, "_validate_construction_attempt", lambda bank: {})
    monkeypatch.setattr(runner, "_relative", lambda path: path.name)

    with pytest.raises(RuntimeError, match="single-attempt"):
        runner.run_freeze()


def test_evaluation_never_overwrites_an_existing_logits_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _runner()
    artifact = tmp_path / "000_deadbeef.pt"
    artifact.write_bytes(b"existing")
    monkeypatch.setattr(runner, "_relative", lambda path: path.name)

    with pytest.raises(RuntimeError, match="refuses to replace"):
        runner._require_new_logits_artifact(artifact)

    artifact.unlink()
    artifact.with_suffix(".pt.tmp").write_bytes(b"ambiguous")
    with pytest.raises(RuntimeError, match="refuses to replace"):
        runner._require_new_logits_artifact(artifact)


def test_fresh_resident_head_control_flow_reserves_validates_and_completes(
    monkeypatch,
) -> None:
    runner = _runner()
    backend = object()
    adaptive = SimpleNamespace(load_lock=dict, load_backend=lambda lock: backend)
    checkpoint = {
        "schema_version": runner.CHECKPOINT_SCHEMA,
        "status": "in_progress",
        "references": {},
        "resident_head_revalidation": {"status": "not_started"},
        "pending_reservation": None,
        "cells": [],
        "compute": {
            "reserved_resident_head_revalidation_evaluations": 0,
            "completed_resident_head_revalidation_evaluations": 0,
            "reserved_intervention_forward_passes": 0,
            "completed_intervention_forward_passes": 0,
        },
    }

    def seal_checkpoint(value):
        value["cells_sha256"] = runner._canonical_sha256(value["cells"])
        value.pop("checkpoint_sha256", None)
        value["checkpoint_sha256"] = runner._canonical_sha256(value)

    seal_checkpoint(checkpoint)
    monkeypatch.setattr(runner, "_write_checkpoint", seal_checkpoint)
    monkeypatch.setattr(runner, "_architecture", lambda loaded, config: {"public": {"h": 1}})
    monkeypatch.setattr(
        runner,
        "_validate_construction_against_live_head",
        lambda loaded, bank, capture, architecture, config: 2,
    )

    observed = runner._prepare_resident_head_for_evaluation(
        adaptive,
        checkpoint,
        {},
        {},
        {"architecture": {"h": 1}},
        {},
        {},
        head_evaluations=2,
    )

    assert observed is backend
    assert checkpoint["resident_head_revalidation"] == {"status": "complete"}
    assert checkpoint["pending_reservation"] is None
    assert checkpoint["compute"]["reserved_resident_head_revalidation_evaluations"] == 2
    assert checkpoint["compute"]["completed_resident_head_revalidation_evaluations"] == 2
