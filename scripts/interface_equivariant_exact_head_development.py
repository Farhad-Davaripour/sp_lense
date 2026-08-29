from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from sp_lense.comparison_intervention import apply_intervention, hook_name, intervention_mask
from sp_lense.interface_equivariant_gradient import (
    PairedOrderGradientIneligible,
    certify_exact_rmsnorm_head_shared_alpha_from_numerators,
    construct_interface_equivariant_field,
    exact_rmsnorm_semantic_gradients_from_boundaries,
)
from sp_lense.steering_methods import actual_perturbation_norms

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "interface_equivariant_exact_head_development_lock.json"
PROTOCOL_PATH = ROOT / "docs" / "INTERFACE_EQUIVARIANT_EXACT_HEAD_PREREGISTRATION.md"
MATH_PATH = ROOT / "src" / "sp_lense" / "interface_equivariant_gradient.py"
MATH_TEST_PATH = ROOT / "tests" / "test_interface_equivariant_gradient.py"
RUNNER_TEST_PATH = ROOT / "tests" / "test_interface_equivariant_exact_head_runner.py"
OLD_RUNNER_PATH = ROOT / "scripts" / "paired_order_analytic_gradient_development.py"
CAPTURE_PATH = (
    ROOT
    / "artifacts"
    / "paired_order_analytic_gradient_development"
    / "qwen35_08b_v2"
    / "paired_capture.pt"
)
CAPTURE_MANIFEST_PATH = CAPTURE_PATH.with_name("paired_capture_manifest.json")
CAPTURE_LEDGER_PATH = CAPTURE_PATH.with_name("paired_capture_attempt_ledger.json")

ARTIFACT_ROOT = ROOT / "artifacts" / "interface_equivariant_exact_head_development" / "qwen35_08b"
RESULT_ROOT = ROOT / "results" / "interface_equivariant_exact_head_development" / "qwen35_08b"
CONSTRUCTION_PATH = ARTIFACT_ROOT / "construction_bank.pt"
CONSTRUCTION_MANIFEST_PATH = ARTIFACT_ROOT / "construction_manifest.json"
CONSTRUCTION_ATTEMPT_PATH = ARTIFACT_ROOT / "construction_attempt_ledger.json"
FREEZE_PATH = ARTIFACT_ROOT / "preoutcome_freeze.json"
CHECKPOINT_PATH = RESULT_ROOT / "evaluation_checkpoint.json"
LOGITS_ROOT = RESULT_ROOT / "evaluation_logits"
RESULT_PATH = RESULT_ROOT / "development_result.json"
REPORT_PATH = RESULT_ROOT / "DEVELOPMENT_REPORT.md"

LOCK_SCHEMA = "sp_lense.interface_equivariant_exact_head_lock.v1"
CONSTRUCTION_SCHEMA = "sp_lense.interface_equivariant_exact_head_construction.v1"
MANIFEST_SCHEMA = "sp_lense.interface_equivariant_exact_head_manifest.v1"
CONSTRUCTION_ATTEMPT_SCHEMA = "sp_lense.interface_equivariant_exact_head_attempt.v1"
FREEZE_SCHEMA = "sp_lense.interface_equivariant_exact_head_freeze.v1"
CHECKPOINT_SCHEMA = "sp_lense.interface_equivariant_exact_head_checkpoint.v1"
RESULT_SCHEMA = "sp_lense.interface_equivariant_exact_head_result.v1"

PRECONSTRUCTION_RELATIVES = frozenset(
    {
        ".gitattributes",
        "artifacts/paired_order_analytic_gradient_development/qwen35_08b/paired_capture_attempt_ledger.json",
        "artifacts/paired_order_analytic_gradient_development/qwen35_08b_v2/paired_capture.pt",
        "artifacts/paired_order_analytic_gradient_development/qwen35_08b_v2/paired_capture_attempt_ledger.json",
        "artifacts/paired_order_analytic_gradient_development/qwen35_08b_v2/paired_capture_manifest.json",
        "configs/counterfactual_semantic_gradient_confirmation_lock.json",
        "configs/gradient_specificity_adaptive_lock.json",
        "configs/paired_order_analytic_gradient_development_lock.json",
        "configs/qwen35_08b_aligned.json",
        "data/counterfactual_semantic_gradient_confirmation_v2.json",
        "docs/INTERFACE_EQUIVARIANT_EXACT_HEAD_PREREGISTRATION.md",
        "docs/INTERFACE_EQUIVARIANT_EXACT_HEAD_MATH_AUDIT.md",
        "docs/PAIRED_ORDER_ANALYTIC_GRADIENT_CONTROLLER_PREREGISTRATION.md",
        "results/counterfactual_semantic_gradient_confirmation/qwen35_08b/semantic_gate_confirmation_result.json",
        "results/paired_order_analytic_gradient_development/qwen35_08b/capture_runtime_abort_v1.json",
        "scripts/counterfactual_semantic_gradient_confirmation.py",
        "scripts/gradient_specificity_adaptive.py",
        "scripts/interface_equivariant_exact_head_development.py",
        "scripts/learned_context_gated_gradient_development.py",
        "scripts/paired_order_analytic_gradient_development.py",
        "src/sp_lense/__init__.py",
        "src/sp_lense/backend.py",
        "src/sp_lense/comparison_dataset.py",
        "src/sp_lense/comparison_intervention.py",
        "src/sp_lense/comparison_runtime.py",
        "src/sp_lense/config.py",
        "src/sp_lense/context_gated_bidirectional.py",
        "src/sp_lense/core.py",
        "src/sp_lense/gradient_specificity_adaptive.py",
        "src/sp_lense/gradient_specificity_v2.py",
        "src/sp_lense/interface_equivariant_gradient.py",
        "src/sp_lense/learned_context_gate.py",
        "src/sp_lense/paired_order_analytic_gradient.py",
        "src/sp_lense/steering_methods.py",
        "tests/test_interface_equivariant_exact_head_runner.py",
        "tests/test_interface_equivariant_gradient.py",
        "tests/test_paired_order_analytic_gradient.py",
        "tests/test_paired_order_analytic_gradient_runner.py",
    }
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_tensor_hash(tensor: Any) -> str:
    array = tensor.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch_save(torch: Any, path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_config(*, require_frozen: bool = True) -> dict[str, Any]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != LOCK_SCHEMA:
        raise RuntimeError("exact-head development lock has the wrong schema")
    if require_frozen and payload.get("status") != "locked_before_construction":
        raise RuntimeError("exact-head development lock is not frozen")
    if require_frozen:
        locked = payload.get("locked_inputs")
        if not isinstance(locked, Mapping) or set(locked) != PRECONSTRUCTION_RELATIVES:
            raise RuntimeError("exact-head development lock lacks the exact required input set")
        for relative, expected in locked.items():
            path = ROOT / str(relative)
            if not path.is_file() or _sha256(path) != expected:
                raise RuntimeError(f"locked input hash differs: {relative}")
    return payload


def _require_committed_clean(paths: list[Path]) -> None:
    relatives = [_relative(path) for path in paths]
    for relative in relatives:
        _git("ls-files", "--error-unmatch", "--", relative)
    if _git("status", "--porcelain", "--", *relatives):
        raise RuntimeError("bound exact-head inputs must be committed and clean")


def _load_capture(torch: Any) -> tuple[Any, dict[str, Any]]:
    old = _load_module("sp_lense_paired_order_v2_locked", OLD_RUNNER_PATH)
    old._validate_capture_manifest()
    capture = old._load_capture(torch)
    if len(capture["pairs"]) != 16:
        raise RuntimeError("reused capture does not contain exactly 16 pairs")
    return old, capture


def _architecture(backend: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    torch = backend.torch
    model = backend.model
    layer = int(config["intervention"]["residual_layer_zero_based"])
    if int(model.cfg.n_layers) != 24 or layer != int(model.cfg.n_layers) - 1:
        raise RuntimeError("intervention site is not the final transformer block")
    ln_final = model.ln_final
    if type(ln_final).__name__ != "RMSNorm":
        raise RuntimeError("final normalization is not TransformerLens RMSNorm")
    epsilon = float(ln_final.eps)
    if epsilon != float(config["model"]["rms_epsilon"]):
        raise RuntimeError("final RMSNorm epsilon differs from the lock")
    gamma = ln_final.w.detach().float().cpu().contiguous()
    weights = model.unembed.W_U.detach().float().cpu().contiguous()
    bias = model.unembed.b_U.detach().float().cpu().contiguous()
    width = int(config["model"]["residual_width"])
    if tuple(gamma.shape) != (width,) or int(weights.shape[0]) != width:
        raise RuntimeError("final-head tensor orientation or width differs from the lock")
    if tuple(bias.shape) != (int(weights.shape[1]),):
        raise RuntimeError("unembedding bias shape differs from the vocabulary")
    if not all(bool(torch.isfinite(value).all().item()) for value in (gamma, weights, bias)):
        raise RuntimeError("final-head parameters contain non-finite values")
    if int(torch.count_nonzero(bias - bias[0]).item()) != 0:
        raise RuntimeError("unembedding bias is not vocabulary-constant")
    if int(torch.count_nonzero(bias).item()) != 0:
        raise RuntimeError("unembedding bias is not the locked all-zero value")
    return {
        "gamma": gamma,
        "weights": weights,
        "bias": bias,
        "public": {
            "ln_final_class": f"{type(ln_final).__module__}.{type(ln_final).__name__}",
            "rms_epsilon": epsilon,
            "residual_width": width,
            "vocabulary_size": int(weights.shape[1]),
            "unembedding_orientation": "d_model_by_vocabulary",
            "unembedding_bias_vocabulary_constant": True,
            "unembedding_bias_exactly_zero": True,
            "gamma_float32_sha256": _raw_tensor_hash(gamma),
            "unembedding_float32_sha256": _raw_tensor_hash(weights),
            "unembedding_bias_float32_sha256": _raw_tensor_hash(bias),
        },
    }


def _actual_head(backend: Any, residuals: Any) -> Any:
    torch = backend.torch
    values = residuals.detach().to(device=backend.device, dtype=torch.float32)
    with torch.inference_mode():
        logits = backend.model.unembed(backend.model.ln_final(values))
    return logits.detach().float().cpu().contiguous()


def _validate_construction_against_live_head(
    backend: Any,
    bank: Mapping[str, Any],
    capture: Mapping[str, Any],
    architecture: Mapping[str, Any],
    config: Mapping[str, Any],
) -> int:
    """Bind every stored construction tensor to the resident pinned final head."""

    torch = backend.torch
    gamma = architecture["gamma"]
    weights = architecture["weights"]
    conceptual_evaluations = 0
    for pair, captured_pair in zip(bank["pairs"], capture["pairs"], strict=True):
        residuals = torch.stack(
            [order["residual"].float() for order in captured_pair["orders"]]
        ).contiguous()
        expected_baseline_numerators = ((residuals * gamma) @ weights).float().contiguous()
        if not torch.equal(expected_baseline_numerators, pair["baseline_numerators"]):
            raise RuntimeError("construction baseline numerators differ from the live head")
        preserve_ids = tuple(int(order["preserve_token_id"]) for order in pair["orders"])
        comply_ids = tuple(int(order["comply_token_id"]) for order in pair["orders"])
        expected_boundaries = (
            torch.stack(
                [
                    gamma * (weights[:, preserve_ids[index]] - weights[:, comply_ids[index]])
                    for index in range(2)
                ]
            )
            .float()
            .contiguous()
        )
        if not torch.equal(expected_boundaries, pair["effective_boundaries"]):
            raise RuntimeError("construction answer boundaries differ from the live head")
        expected_analytic = (
            torch.from_numpy(
                exact_rmsnorm_semantic_gradients_from_boundaries(
                    residuals.double().numpy(),
                    expected_boundaries.double().numpy(),
                    rms_epsilon=float(config["model"]["rms_epsilon"]),
                )
            )
            .float()
            .contiguous()
        )
        if not torch.equal(expected_analytic, pair["analytic_gradients"]):
            raise RuntimeError("construction analytic gradients differ from the live head")

        observed_baseline = _actual_head(backend, residuals.view(2, 1, -1))[:, 0]
        expected_baseline = (
            torch.stack(
                [
                    pair["orders"][0]["actual_baseline_logits"],
                    pair["orders"][1]["actual_baseline_logits"],
                ]
            )
            .float()
            .contiguous()
        )
        if not torch.equal(observed_baseline, expected_baseline):
            raise RuntimeError("stored baseline logits differ from the live final head")
        conceptual_evaluations += 2
        for method in pair["methods"].values():
            base = method["base_vectors"]
            expected_direction_numerators = ((base * gamma) @ weights).float().contiguous()
            if not torch.equal(expected_direction_numerators, method["direction_numerators"]):
                raise RuntimeError("construction direction numerators differ from the live head")
            actual = method.get("actual_head_certificate")
            if actual is None:
                continue
            delta = method["delta"]
            row_index = 0
            for order_index in range(2):
                for sign in (1.0, -1.0):
                    residual = residuals[order_index].float().contiguous()
                    signed = (sign * delta[order_index]).float().contiguous()
                    changed = residual + signed
                    observed = _actual_head(backend, changed.view(1, 1, -1))[0, 0]
                    if not torch.equal(observed, actual["actual_head_logits"][row_index]):
                        raise RuntimeError(
                            "stored construction logits differ from the live final head"
                        )
                    row_index += 1
            conceptual_evaluations += 4
    expected_count = int(bank["summary"]["conceptual_exact_head_evaluations"])
    if conceptual_evaluations != expected_count:
        raise RuntimeError("resident-head revalidation count differs from construction evidence")
    return conceptual_evaluations


def _target_margin(torch: Any, logits: Any, target: int) -> tuple[int, float]:
    if logits.ndim != 1 or not 0 <= target < int(logits.numel()):
        raise ValueError("invalid logits or target")
    masked = logits.clone()
    masked[target] = -torch.inf
    competitor = int(masked.argmax().item())
    return competitor, float((logits[target] - logits[competitor]).item())


def _float32_realized_relative_norms(torch: Any, residuals: Any, delta: Any) -> list[float]:
    values = []
    for order_index in range(2):
        for sign in (1.0, -1.0):
            residual = residuals[order_index].float().contiguous()
            signed = (sign * delta[order_index]).float().contiguous()
            realized = (residual + signed) - residual
            # Match actual_perturbation_norms exactly: both norms and their division
            # are evaluated in float32 before conversion to a Python scalar.
            values.append(float((realized.norm() / residual.norm()).item()))
    return values


def _actual_head_certificate(
    backend: Any,
    residuals: Any,
    delta: Any,
    preserve_ids: tuple[int, int],
    comply_ids: tuple[int, int],
    *,
    acceptance_reserve: float,
) -> dict[str, Any]:
    torch = backend.torch
    rows = []
    logits_rows = []
    signed_hashes = []
    for order in range(2):
        for sign, target, name in (
            (1.0, preserve_ids[order], "preserve"),
            (-1.0, comply_ids[order], "comply"),
        ):
            signed = (sign * delta[order]).to(dtype=torch.float32).contiguous()
            changed = residuals[order].float().contiguous() + signed
            logits = _actual_head(backend, changed.view(1, 1, -1))[0, 0]
            logits_rows.append(logits)
            argmax = int(logits.argmax().item())
            competitor, margin = _target_margin(torch, logits, target)
            signed_hashes.append(_raw_tensor_hash(signed))
            rows.append(
                {
                    "order_index": order,
                    "sign": sign,
                    "target": name,
                    "target_token_id": target,
                    "argmax_token_id": argmax,
                    "strongest_competitor_token_id": competitor,
                    "target_margin": margin,
                    "target_met": bool(argmax == target and margin >= acceptance_reserve),
                    "logits_float32_sha256": _raw_tensor_hash(logits),
                }
            )
    certificate = {
        "rows": rows,
        "actual_head_logits": torch.stack(logits_rows).float().cpu().contiguous(),
        "minimum_target_margin": min(float(row["target_margin"]) for row in rows),
        "positive_order_delta_sha256": [_raw_tensor_hash(delta[index]) for index in range(2)],
        "negative_order_delta_sha256": [
            _raw_tensor_hash((-delta).float().contiguous()[index]) for index in range(2)
        ],
        "signed_cell_delta_sha256": signed_hashes,
    }
    negative = (-delta).to(dtype=torch.float32).contiguous()
    if not torch.equal(negative, torch.neg(delta.to(dtype=torch.float32))):
        raise RuntimeError("negative delta is not the exact float32 negation")
    return certificate


def _cosine_and_relative_error(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("cannot compare zero vectors")
    cosine = float(np.clip((left @ right) / (left_norm * right_norm), -1.0, 1.0))
    return cosine, float(np.linalg.norm(left - right) / left_norm)


def _construct_method(
    backend: Any,
    residuals: Any,
    base_vectors: np.ndarray,
    architecture: Mapping[str, Any],
    preserve_ids: tuple[int, int],
    comply_ids: tuple[int, int],
    config: Mapping[str, Any],
    *,
    method_id: str,
    compute_counter: dict[str, int],
) -> dict[str, Any]:
    torch = backend.torch
    gamma = architecture["gamma"]
    weights = architecture["weights"]
    base = torch.from_numpy(base_vectors).float().cpu().contiguous()
    baseline_numerators = ((residuals.float() * gamma) @ weights).float().cpu().contiguous()
    direction_numerators = ((base * gamma) @ weights).float().cpu().contiguous()
    intervention = config["intervention"]
    common = {
        "method_id": method_id,
        "base_vectors": base,
        "base_vectors_sha256": _raw_tensor_hash(base),
        "direction_numerators": direction_numerators,
        "direction_numerators_sha256": _raw_tensor_hash(direction_numerators),
    }
    try:
        certificate = certify_exact_rmsnorm_head_shared_alpha_from_numerators(
            baseline_numerators.double().numpy(),
            direction_numerators.double().numpy(),
            [float(row.double().norm().item()) for row in residuals],
            [float(row.double().norm().item()) for row in base],
            preserve_ids,
            comply_ids,
            residual_width=int(config["model"]["residual_width"]),
            rms_epsilon=float(config["model"]["rms_epsilon"]),
            construction_reserve_logit=float(intervention["construction_reserve_logit"]),
            maximum_relative_norm=float(intervention["maximum_relative_norm"]),
        )
    except PairedOrderGradientIneligible as error:
        return {
            **common,
            "eligible": False,
            "failure": {
                **error.diagnostics,
                "conceptual_exact_head_evaluations": 0,
            },
        }
    delta = (certificate.alpha * base).to(dtype=torch.float32).contiguous()
    residual_norms = residuals.double().norm(dim=1)
    relative_norms = delta.double().norm(dim=1) / residual_norms
    if bool((relative_norms > float(intervention["maximum_relative_norm"])).any().item()):
        return {
            **common,
            "eligible": False,
            "alpha": float(certificate.alpha),
            "lower": float(certificate.lower),
            "upper": float(certificate.upper),
            "delta": delta,
            "delta_sha256": _raw_tensor_hash(delta),
            "per_order_delta_sha256": [_raw_tensor_hash(delta[index]) for index in range(2)],
            "relative_norms": [float(value) for value in relative_norms.tolist()],
            "solver_diagnostics": certificate.diagnostics,
            "failure": {
                "failure": "float32_delta_norm_cap_exceeded",
                "relative_norms": [float(value) for value in relative_norms.tolist()],
                "maximum_relative_norm": float(intervention["maximum_relative_norm"]),
                "conceptual_exact_head_evaluations": 0,
            },
        }
    realized_relative_norms = _float32_realized_relative_norms(torch, residuals, delta)
    if any(
        value > float(intervention["maximum_relative_norm"]) for value in realized_relative_norms
    ):
        return {
            **common,
            "eligible": False,
            "alpha": float(certificate.alpha),
            "lower": float(certificate.lower),
            "upper": float(certificate.upper),
            "delta": delta,
            "delta_sha256": _raw_tensor_hash(delta),
            "per_order_delta_sha256": [_raw_tensor_hash(delta[index]) for index in range(2)],
            "relative_norms": [float(value) for value in relative_norms.tolist()],
            "realized_relative_norms": realized_relative_norms,
            "solver_diagnostics": certificate.diagnostics,
            "failure": {
                "failure": "float32_realized_norm_cap_exceeded",
                "realized_relative_norms": realized_relative_norms,
                "maximum_relative_norm": float(intervention["maximum_relative_norm"]),
                "conceptual_exact_head_evaluations": 0,
            },
        }
    compute_counter["conceptual_exact_head_evaluations"] += 4
    actual = _actual_head_certificate(
        backend,
        residuals,
        delta,
        preserve_ids,
        comply_ids,
        acceptance_reserve=float(intervention["finite_acceptance_reserve_logit"]),
    )
    eligible = all(bool(row["target_met"]) for row in actual["rows"])
    result = {
        **common,
        "method_id": method_id,
        "eligible": eligible,
        "alpha": float(certificate.alpha),
        "lower": float(certificate.lower),
        "upper": float(certificate.upper),
        "delta": delta,
        "delta_sha256": _raw_tensor_hash(delta),
        "per_order_delta_sha256": [_raw_tensor_hash(delta[index]) for index in range(2)],
        "relative_norms": [float(value) for value in relative_norms.tolist()],
        "realized_relative_norms": realized_relative_norms,
        "solver_diagnostics": certificate.diagnostics,
        "actual_head_certificate": actual,
        "conceptual_exact_head_evaluations": 4,
    }
    if not eligible:
        result["failure"] = {
            "failure": "actual_float32_head_failure",
            "conceptual_exact_head_evaluations": 4,
        }
    return result


def _public_method(row: Mapping[str, Any]) -> dict[str, Any]:
    public = {
        key: value
        for key, value in row.items()
        if key not in {"base_vectors", "delta", "direction_numerators"}
    }
    if isinstance(public.get("actual_head_certificate"), Mapping):
        public["actual_head_certificate"] = {
            key: value
            for key, value in public["actual_head_certificate"].items()
            if key != "actual_head_logits"
        }
    failure = public.get("failure")
    if isinstance(failure, Mapping) and isinstance(failure.get("actual_head_certificate"), Mapping):
        public_failure = dict(failure)
        public_failure["actual_head_certificate"] = {
            key: value
            for key, value in failure["actual_head_certificate"].items()
            if key != "actual_head_logits"
        }
        public["failure"] = public_failure
    return public


def _comparison_rows(left: np.ndarray, right: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for order_index in range(2):
        cosine, relative_error = _cosine_and_relative_error(left[order_index], right[order_index])
        rows.append(
            {
                "order_index": order_index,
                "cosine": cosine,
                "relative_l2_error": relative_error,
            }
        )
    return rows


def _public_pairs(bank: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            **{
                key: pair[key]
                for key in (
                    "case_id",
                    "assignment",
                    "analytic_gradients_sha256",
                    "effective_boundaries_sha256",
                    "baseline_numerators_sha256",
                    "gradient_tangent_comparison",
                    "gradient_unembedding_comparison",
                )
            },
            "orders": [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"baseline_logits", "actual_baseline_logits"}
                }
                for row in pair["orders"]
            ],
            "methods": {key: _public_method(value) for key, value in pair["methods"].items()},
        }
        for pair in bank["pairs"]
    ]


def _manifest_payload(bank: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "construction_complete_opened_development",
        "config_sha256": _sha256(CONFIG_PATH),
        "capture_sha256": _sha256(CAPTURE_PATH),
        "construction_bank_sha256": _sha256(CONSTRUCTION_PATH),
        "architecture": bank["architecture"],
        "summary": bank["summary"],
        "pairs": _public_pairs(bank),
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    return payload


def _write_construction_attempt(payload: dict[str, Any]) -> None:
    payload.pop("result_sha256", None)
    payload["result_sha256"] = _canonical_sha256(payload)
    _atomic_json(CONSTRUCTION_ATTEMPT_PATH, payload)


def _construction_and_downstream_paths() -> tuple[Path, ...]:
    return (
        CONSTRUCTION_ATTEMPT_PATH,
        CONSTRUCTION_PATH,
        CONSTRUCTION_MANIFEST_PATH,
        FREEZE_PATH,
        CHECKPOINT_PATH,
        LOGITS_ROOT,
        RESULT_PATH,
        REPORT_PATH,
    )


def run_construct() -> dict[str, Any]:
    config = _load_config()
    bound = [ROOT / relative for relative in config["locked_inputs"]]
    _require_committed_clean([CONFIG_PATH, *bound])
    existing = [path for path in _construction_and_downstream_paths() if path.exists()]
    if existing:
        raise RuntimeError(
            "construction is single-attempt and found an existing artifact: "
            + ", ".join(_relative(path) for path in existing)
        )
    attempt = {
        "schema_version": CONSTRUCTION_ATTEMPT_SCHEMA,
        "status": "reserved_before_model_load",
        "config_sha256": _sha256(CONFIG_PATH),
        "input_commit": _git("rev-parse", "HEAD"),
        "reserved_model_loads": 1,
        "completed_model_loads": 0,
        "reserved_conceptual_exact_head_evaluations": int(
            config["compute"]["maximum_construction_exact_head_evaluations"]
        ),
        "completed_conceptual_exact_head_evaluations": 0,
    }
    _write_construction_attempt(attempt)
    import torch

    old, capture = _load_capture(torch)
    adaptive, _groups = old._active_jobs()
    backend = adaptive.load_backend(adaptive.load_lock())
    architecture = _architecture(backend, config)
    arch = architecture["public"]
    pairs = []
    maximum_primal_error = 0.0
    all_identity_rows = []
    compute_counter = {"conceptual_exact_head_evaluations": 0}
    started = time.perf_counter()
    for pair in capture["pairs"]:
        residuals = torch.stack([row["residual"].float() for row in pair["orders"]]).contiguous()
        captured_gradients = np.stack(
            [row["semantic_gradient"].double().numpy() for row in pair["orders"]]
        )
        baseline_logits = torch.stack(
            [row["baseline_logits"].float() for row in pair["orders"]]
        ).contiguous()
        head_logits = _actual_head(backend, residuals.view(2, 1, -1))[:, 0]
        compute_counter["conceptual_exact_head_evaluations"] += 2
        primal_error = float((head_logits - baseline_logits).abs().max().item())
        maximum_primal_error = max(maximum_primal_error, primal_error)
        if primal_error > float(
            config["architecture_guards"]["maximum_baseline_head_primal_absolute_error"]
        ):
            raise RuntimeError("actual final head does not reproduce a captured baseline")

        preserve_ids = tuple(int(row["preserve_token_id"]) for row in pair["orders"])
        comply_ids = tuple(int(row["comply_token_id"]) for row in pair["orders"])
        boundaries = torch.stack(
            [
                architecture["gamma"]
                * (
                    architecture["weights"][:, preserve_ids[index]]
                    - architecture["weights"][:, comply_ids[index]]
                )
                for index in range(2)
            ]
        ).contiguous()
        baseline_numerators = (
            ((residuals.float() * architecture["gamma"]) @ architecture["weights"])
            .float()
            .cpu()
            .contiguous()
        )
        analytic = exact_rmsnorm_semantic_gradients_from_boundaries(
            residuals.double().numpy(),
            boundaries.double().numpy(),
            rms_epsilon=float(config["model"]["rms_epsilon"]),
        )
        analytic_tensor = torch.from_numpy(analytic).float().cpu().contiguous()
        analytic = analytic_tensor.double().numpy()
        identity_rows = []
        for order in range(2):
            cosine, relative_error = _cosine_and_relative_error(
                captured_gradients[order], analytic[order]
            )
            identity = {
                "order_index": order,
                "captured_to_analytic_cosine": cosine,
                "captured_to_analytic_relative_l2_error": relative_error,
            }
            identity_rows.append(identity)
            all_identity_rows.append(identity)
            if cosine < float(
                config["architecture_guards"]["minimum_captured_to_analytic_gradient_cosine"]
            ) or relative_error > float(
                config["architecture_guards"][
                    "maximum_captured_to_analytic_gradient_relative_l2_error"
                ]
            ):
                raise RuntimeError("captured gradient fails the analytic RMS tangent identity")

        residual_norms = [float(row.double().norm().item()) for row in residuals]
        gradient_field = construct_interface_equivariant_field(captured_gradients, residual_norms)
        tangent_field = construct_interface_equivariant_field(analytic, residual_norms)
        unembedding_field = construct_interface_equivariant_field(
            boundaries.double().numpy(), residual_norms
        )
        methods = {}
        for method_id, base_vectors in (
            ("gradient_ray", gradient_field.base_vectors),
            ("analytic_rms_tangent_ray", tangent_field.base_vectors),
            ("effective_unembedding_ray", unembedding_field.base_vectors),
        ):
            methods[method_id] = _construct_method(
                backend,
                residuals,
                base_vectors,
                architecture,
                preserve_ids,
                comply_ids,
                config,
                method_id=method_id,
                compute_counter=compute_counter,
            )
        gradient_tangent = _comparison_rows(gradient_field.base_vectors, tangent_field.base_vectors)
        gradient_unembedding = _comparison_rows(
            gradient_field.base_vectors, unembedding_field.base_vectors
        )
        pairs.append(
            {
                "case_id": str(pair["case_id"]),
                "assignment": int(pair["assignment"]),
                "analytic_gradients": analytic_tensor,
                "analytic_gradients_sha256": _raw_tensor_hash(analytic_tensor),
                "effective_boundaries": boundaries.float().cpu().contiguous(),
                "effective_boundaries_sha256": _raw_tensor_hash(boundaries),
                "baseline_numerators": baseline_numerators,
                "baseline_numerators_sha256": _raw_tensor_hash(baseline_numerators),
                "orders": [
                    {
                        "order_index": index,
                        "unit_id": str(row["unit_id"]),
                        "preserve_first": bool(row["preserve_first"]),
                        "prompt_sha256": str(row["prompt_sha256"]),
                        "prompt_token_ids_sha256": str(row["prompt_token_ids_sha256"]),
                        "prompt_length": int(row["prompt_length"]),
                        "positive_label": str(row["positive_label"]),
                        "negative_label": str(row["negative_label"]),
                        "preserve_token_id": int(row["preserve_token_id"]),
                        "comply_token_id": int(row["comply_token_id"]),
                        "choice_boundary_evidence_sha256": str(
                            row["choice_boundary_evidence_sha256"]
                        ),
                        "baseline_argmax_token_id": int(row["baseline_argmax_token_id"]),
                        "baseline_semantic_choice": str(row["baseline_semantic_choice"]),
                        "residual_norm": float(row["residual"].double().norm().item()),
                        "baseline_logits": row["baseline_logits"].float().cpu().contiguous(),
                        "baseline_logits_sha256": str(row["baseline_logits_sha256"]),
                        "actual_baseline_logits": head_logits[index].float().cpu().contiguous(),
                        "actual_baseline_logits_sha256": _raw_tensor_hash(head_logits[index]),
                        "residual_sha256": str(row["residual_sha256"]),
                        "semantic_gradient_sha256": str(row["semantic_gradient_sha256"]),
                        "identity": identity_rows[index],
                    }
                    for index, row in enumerate(pair["orders"])
                ],
                "gradient_tangent_comparison": gradient_tangent,
                "gradient_unembedding_comparison": gradient_unembedding,
                "methods": methods,
            }
        )

    identity_minimum_cosine = min(
        float(row["captured_to_analytic_cosine"]) for row in all_identity_rows
    )
    identity_maximum_relative_error = max(
        float(row["captured_to_analytic_relative_l2_error"]) for row in all_identity_rows
    )
    bank = {
        "schema_version": CONSTRUCTION_SCHEMA,
        "config_sha256": _sha256(CONFIG_PATH),
        "capture_sha256": _sha256(CAPTURE_PATH),
        "architecture": arch,
        "pairs": pairs,
        "summary": {
            "pair_count": len(pairs),
            "method_ids": [
                "gradient_ray",
                "analytic_rms_tangent_ray",
                "effective_unembedding_ray",
            ],
            "eligible_pair_count_by_method": {
                method: sum(bool(pair["methods"][method]["eligible"]) for pair in pairs)
                for method in (
                    "gradient_ray",
                    "analytic_rms_tangent_ray",
                    "effective_unembedding_ray",
                )
            },
            "minimum_captured_to_analytic_gradient_cosine": identity_minimum_cosine,
            "maximum_captured_to_analytic_gradient_relative_l2_error": (
                identity_maximum_relative_error
            ),
            "maximum_baseline_head_primal_absolute_error": maximum_primal_error,
            "elapsed_seconds": time.perf_counter() - started,
            "new_full_forward_passes": 0,
            "new_backward_passes": 0,
            "conceptual_exact_head_evaluations": compute_counter[
                "conceptual_exact_head_evaluations"
            ],
        },
    }
    _atomic_torch_save(torch, CONSTRUCTION_PATH, bank)
    manifest = _manifest_payload(bank)
    _atomic_json(CONSTRUCTION_MANIFEST_PATH, manifest)
    attempt.update(
        {
            "status": "complete",
            "completed_model_loads": 1,
            "completed_conceptual_exact_head_evaluations": int(
                bank["summary"]["conceptual_exact_head_evaluations"]
            ),
            "construction_bank_sha256": _sha256(CONSTRUCTION_PATH),
            "construction_manifest_sha256": _sha256(CONSTRUCTION_MANIFEST_PATH),
        }
    )
    _write_construction_attempt(attempt)
    print(json.dumps(manifest["summary"], indent=2), flush=True)
    return manifest


def _validate_actual_head_certificate(
    torch: Any,
    actual: Any,
    delta: Any,
    orders: list[Mapping[str, Any]],
    *,
    acceptance_reserve: float,
    require_all_targets: bool,
) -> None:
    if not isinstance(actual, Mapping):
        raise TypeError("construction actual-head certificate is missing")
    logits = actual.get("actual_head_logits")
    vocabulary_size = int(orders[0]["baseline_logits"].numel())
    if (
        not torch.is_tensor(logits)
        or logits.dtype != torch.float32
        or tuple(logits.shape) != (4, vocabulary_size)
        or not bool(torch.isfinite(logits).all().item())
    ):
        raise RuntimeError("construction raw actual-head logits are invalid")
    expected_rows = []
    expected_signed_hashes = []
    row_index = 0
    for order_index, order in enumerate(orders):
        for sign, target_name, target_id in (
            (1.0, "preserve", int(order["preserve_token_id"])),
            (-1.0, "comply", int(order["comply_token_id"])),
        ):
            signed = (sign * delta[order_index]).float().contiguous()
            expected_signed_hashes.append(_raw_tensor_hash(signed))
            row_logits = logits[row_index]
            argmax = int(row_logits.argmax().item())
            competitor, margin = _target_margin(torch, row_logits, target_id)
            expected_rows.append(
                {
                    "order_index": order_index,
                    "sign": sign,
                    "target": target_name,
                    "target_token_id": target_id,
                    "argmax_token_id": argmax,
                    "strongest_competitor_token_id": competitor,
                    "target_margin": margin,
                    "target_met": bool(argmax == target_id and margin >= acceptance_reserve),
                    "logits_float32_sha256": _raw_tensor_hash(row_logits),
                }
            )
            row_index += 1
    negative = (-delta).float().contiguous()
    expected = {
        "rows": expected_rows,
        "actual_head_logits": logits,
        "minimum_target_margin": min(float(row["target_margin"]) for row in expected_rows),
        "positive_order_delta_sha256": [_raw_tensor_hash(delta[index]) for index in range(2)],
        "negative_order_delta_sha256": [_raw_tensor_hash(negative[index]) for index in range(2)],
        "signed_cell_delta_sha256": expected_signed_hashes,
    }
    if set(actual) != set(expected) or any(
        actual[key] != value for key, value in expected.items() if key != "actual_head_logits"
    ):
        raise RuntimeError("construction actual-head certificate is not raw-logit-derived")
    all_targets = all(bool(row["target_met"]) for row in expected_rows)
    if all_targets is not require_all_targets:
        raise RuntimeError("construction actual-head eligibility differs from raw logits")


def _validated_architecture(bank: Mapping[str, Any], vocabulary_size: int) -> Mapping[str, Any]:
    architecture = bank.get("architecture")
    config = _load_config()
    if not isinstance(architecture, Mapping):
        raise TypeError("construction architecture metadata is missing")
    expected_fixed = {
        "rms_epsilon": float(config["model"]["rms_epsilon"]),
        "residual_width": int(config["model"]["residual_width"]),
        "vocabulary_size": vocabulary_size,
        "unembedding_orientation": "d_model_by_vocabulary",
        "unembedding_bias_vocabulary_constant": True,
        "unembedding_bias_exactly_zero": True,
    }
    if any(architecture.get(key) != value for key, value in expected_fixed.items()):
        raise RuntimeError("construction architecture metadata differs from the lock")
    if not str(architecture.get("ln_final_class", "")).endswith(".RMSNorm"):
        raise RuntimeError("construction architecture does not identify RMSNorm")
    for field in (
        "gamma_float32_sha256",
        "unembedding_float32_sha256",
        "unembedding_bias_float32_sha256",
    ):
        value = architecture.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"construction architecture hash is invalid: {field}")
    return architecture


def _load_construction(torch: Any) -> dict[str, Any]:
    config = _load_config()
    bank = torch.load(CONSTRUCTION_PATH, map_location="cpu", weights_only=True)
    if (
        bank.get("schema_version") != CONSTRUCTION_SCHEMA
        or bank.get("config_sha256") != _sha256(CONFIG_PATH)
        or bank.get("capture_sha256") != _sha256(CAPTURE_PATH)
        or len(bank.get("pairs", [])) != 16
    ):
        raise RuntimeError("exact-head construction bank belongs to another study")
    capture = torch.load(CAPTURE_PATH, map_location="cpu", weights_only=True)
    captured_pairs = capture.get("pairs")
    if not isinstance(captured_pairs, list) or len(captured_pairs) != 16:
        raise RuntimeError("frozen capture coverage is invalid")
    vocabulary_size = int(captured_pairs[0]["orders"][0]["baseline_logits"].numel())
    _validated_architecture(bank, vocabulary_size)
    expected_keys = [(str(pair["case_id"]), int(pair["assignment"])) for pair in captured_pairs]
    observed_keys = [(str(pair["case_id"]), int(pair["assignment"])) for pair in bank["pairs"]]
    if observed_keys != expected_keys or len(set(observed_keys)) != 16:
        raise RuntimeError("construction pair coverage differs from the frozen capture")

    method_ids = (
        "gradient_ray",
        "analytic_rms_tangent_ray",
        "effective_unembedding_ray",
    )
    eligible_counts = {method_id: 0 for method_id in method_ids}
    all_identity_rows = []
    maximum_primal_error = 0.0
    conceptual_head_evaluations = 32
    acceptance_reserve = float(config["intervention"]["finite_acceptance_reserve_logit"])
    norm_cap = float(config["intervention"]["maximum_relative_norm"])
    width = int(config["model"]["residual_width"])
    for pair, captured_pair in zip(bank["pairs"], captured_pairs, strict=True):
        if len(pair.get("orders", [])) != 2:
            raise RuntimeError("construction pair lacks two answer orders")
        analytic = pair.get("analytic_gradients")
        boundaries = pair.get("effective_boundaries")
        baseline_numerators = pair.get("baseline_numerators")
        for name, tensor, expected_hash, expected_shape in (
            (
                "analytic gradients",
                analytic,
                pair.get("analytic_gradients_sha256"),
                (2, width),
            ),
            (
                "effective boundaries",
                boundaries,
                pair.get("effective_boundaries_sha256"),
                (2, width),
            ),
            (
                "baseline numerators",
                baseline_numerators,
                pair.get("baseline_numerators_sha256"),
                (2, vocabulary_size),
            ),
        ):
            if (
                not torch.is_tensor(tensor)
                or tensor.dtype != torch.float32
                or tuple(tensor.shape) != expected_shape
                or not bool(torch.isfinite(tensor).all().item())
                or _raw_tensor_hash(tensor) != expected_hash
            ):
                raise RuntimeError(f"construction {name} tensor is invalid")

        captured_residuals = torch.stack(
            [order["residual"].float() for order in captured_pair["orders"]]
        ).contiguous()
        expected_analytic = (
            torch.from_numpy(
                exact_rmsnorm_semantic_gradients_from_boundaries(
                    captured_residuals.double().numpy(),
                    boundaries.double().numpy(),
                    rms_epsilon=float(config["model"]["rms_epsilon"]),
                )
            )
            .float()
            .contiguous()
        )
        if not torch.equal(analytic, expected_analytic):
            raise RuntimeError("construction analytic gradient is not boundary-derived")

        residual_norms = []
        captured_gradients = []
        for order_index, (order, captured_order) in enumerate(
            zip(pair["orders"], captured_pair["orders"], strict=True)
        ):
            identity_fields = (
                "unit_id",
                "preserve_first",
                "prompt_sha256",
                "prompt_token_ids_sha256",
                "prompt_length",
                "positive_label",
                "negative_label",
                "preserve_token_id",
                "comply_token_id",
                "choice_boundary_evidence_sha256",
                "baseline_argmax_token_id",
                "baseline_semantic_choice",
                "baseline_logits_sha256",
                "residual_sha256",
                "semantic_gradient_sha256",
            )
            if order.get("order_index") != order_index or any(
                order.get(field) != captured_order.get(field) for field in identity_fields
            ):
                raise RuntimeError("construction order identity differs from the frozen capture")
            baseline = order.get("baseline_logits")
            captured_baseline = captured_order["baseline_logits"].float().cpu().contiguous()
            actual_baseline = order.get("actual_baseline_logits")
            if (
                not torch.is_tensor(baseline)
                or baseline.dtype != torch.float32
                or tuple(baseline.shape) != (vocabulary_size,)
                or not bool(torch.isfinite(baseline).all().item())
                or _raw_tensor_hash(baseline) != order["baseline_logits_sha256"]
                or not torch.equal(baseline, captured_baseline)
            ):
                raise RuntimeError("construction baseline logits differ from the frozen capture")
            if (
                not torch.is_tensor(actual_baseline)
                or actual_baseline.dtype != torch.float32
                or tuple(actual_baseline.shape) != (vocabulary_size,)
                or not bool(torch.isfinite(actual_baseline).all().item())
                or _raw_tensor_hash(actual_baseline) != order.get("actual_baseline_logits_sha256")
            ):
                raise RuntimeError("construction raw actual baseline logits are invalid")
            maximum_primal_error = max(
                maximum_primal_error,
                float((actual_baseline - baseline).abs().max().item()),
            )
            captured_norm = float(captured_order["residual"].double().norm().item())
            if not math.isclose(
                float(order.get("residual_norm", float("nan"))),
                captured_norm,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise RuntimeError("construction residual norm differs from the frozen capture")
            residual_norms.append(captured_norm)
            captured_gradient = captured_order["semantic_gradient"].double().cpu().numpy()
            captured_gradients.append(captured_gradient)
            cosine, relative_error = _cosine_and_relative_error(
                captured_gradient, analytic[order_index].double().numpy()
            )
            expected_identity = {
                "order_index": order_index,
                "captured_to_analytic_cosine": cosine,
                "captured_to_analytic_relative_l2_error": relative_error,
            }
            if order.get("identity") != expected_identity:
                raise RuntimeError("construction gradient identity is not tensor-derived")
            all_identity_rows.append(expected_identity)

        expected_fields = {
            "gradient_ray": construct_interface_equivariant_field(
                np.stack(captured_gradients), residual_norms
            ).base_vectors,
            "analytic_rms_tangent_ray": construct_interface_equivariant_field(
                analytic.double().numpy(), residual_norms
            ).base_vectors,
            "effective_unembedding_ray": construct_interface_equivariant_field(
                boundaries.double().numpy(), residual_norms
            ).base_vectors,
        }
        expected_gradient_tangent = _comparison_rows(
            expected_fields["gradient_ray"], expected_fields["analytic_rms_tangent_ray"]
        )
        expected_gradient_unembedding = _comparison_rows(
            expected_fields["gradient_ray"], expected_fields["effective_unembedding_ray"]
        )
        if (
            pair.get("gradient_tangent_comparison") != expected_gradient_tangent
            or pair.get("gradient_unembedding_comparison") != expected_gradient_unembedding
        ):
            raise RuntimeError("construction direction comparison is not tensor-derived")

        for method_id in method_ids:
            method = pair.get("methods", {}).get(method_id)
            if not isinstance(method, Mapping) or method.get("method_id") != method_id:
                raise RuntimeError("construction method identity differs")
            eligible = method.get("eligible")
            if type(eligible) is not bool:
                raise RuntimeError("construction method has a non-boolean eligibility flag")
            for field, expected_hash_field, expected_shape in (
                ("base_vectors", "base_vectors_sha256", (2, width)),
                (
                    "direction_numerators",
                    "direction_numerators_sha256",
                    (2, vocabulary_size),
                ),
            ):
                tensor = method.get(field)
                if not torch.is_tensor(tensor) or _raw_tensor_hash(tensor) != method.get(
                    expected_hash_field
                ):
                    raise RuntimeError(f"construction tensor hash differs: {field}")
                if (
                    tensor.dtype != torch.float32
                    or tuple(tensor.shape) != expected_shape
                    or not bool(torch.isfinite(tensor).all().item())
                ):
                    raise RuntimeError(f"construction tensor has invalid dtype/shape: {field}")
            base = method["base_vectors"]
            expected_base = torch.from_numpy(expected_fields[method_id]).float().contiguous()
            if not torch.equal(base, expected_base):
                raise RuntimeError("construction base vector is not source-tensor-derived")
            base_norms = base.double().norm(dim=1)
            try:
                expected_certificate = certify_exact_rmsnorm_head_shared_alpha_from_numerators(
                    baseline_numerators.double().numpy(),
                    method["direction_numerators"].double().numpy(),
                    residual_norms,
                    base_norms.numpy(),
                    tuple(int(order["preserve_token_id"]) for order in pair["orders"]),
                    tuple(int(order["comply_token_id"]) for order in pair["orders"]),
                    residual_width=width,
                    rms_epsilon=float(config["model"]["rms_epsilon"]),
                    construction_reserve_logit=float(
                        config["intervention"]["construction_reserve_logit"]
                    ),
                    maximum_relative_norm=norm_cap,
                )
            except PairedOrderGradientIneligible as error:
                expected_failure = {
                    **error.diagnostics,
                    "conceptual_exact_head_evaluations": 0,
                }
                expected_keys = {
                    "method_id",
                    "base_vectors",
                    "base_vectors_sha256",
                    "direction_numerators",
                    "direction_numerators_sha256",
                    "eligible",
                    "failure",
                }
                if (
                    eligible
                    or set(method) != expected_keys
                    or method.get("failure") != expected_failure
                ):
                    raise RuntimeError("construction solver failure is not reproducible")
                continue
            if any(
                method.get(field) != expected
                for field, expected in (
                    ("alpha", float(expected_certificate.alpha)),
                    ("lower", float(expected_certificate.lower)),
                    ("upper", float(expected_certificate.upper)),
                    ("solver_diagnostics", expected_certificate.diagnostics),
                )
            ):
                raise RuntimeError("construction dose is not reproduced by the locked solver")
            delta = method.get("delta")
            if (
                not torch.is_tensor(delta)
                or delta.dtype != torch.float32
                or tuple(delta.shape) != (2, width)
                or not bool(torch.isfinite(delta).all().item())
                or _raw_tensor_hash(delta) != method.get("delta_sha256")
            ):
                raise RuntimeError("construction delta tensor is invalid")
            delta = method["delta"]
            if [_raw_tensor_hash(delta[index]) for index in range(2)] != method.get(
                "per_order_delta_sha256"
            ):
                raise RuntimeError("per-order construction delta hashes differ")
            alpha = method.get("alpha")
            if type(alpha) is not float or not math.isfinite(alpha) or not 0.0 <= alpha <= norm_cap:
                raise RuntimeError("construction alpha is invalid")
            expected_delta = (alpha * base).to(dtype=torch.float32).contiguous()
            if not torch.equal(delta, expected_delta):
                raise RuntimeError("construction delta is not the stored shared alpha times base")
            residual_tensor = torch.tensor(residual_norms, dtype=torch.float64)
            relative_norms = delta.double().norm(dim=1) / residual_tensor
            if not np.allclose(
                relative_norms.numpy(), method.get("relative_norms"), rtol=1e-8, atol=1e-8
            ):
                raise RuntimeError("construction relative norms differ")
            if bool((relative_norms > norm_cap).any().item()):
                expected_failure = {
                    "failure": "float32_delta_norm_cap_exceeded",
                    "relative_norms": [float(value) for value in relative_norms.tolist()],
                    "maximum_relative_norm": norm_cap,
                    "conceptual_exact_head_evaluations": 0,
                }
                if eligible or method.get("failure") != expected_failure:
                    raise RuntimeError("construction cast-cap failure is not reproducible")
                continue
            captured_residuals = torch.stack(
                [order["residual"].float() for order in captured_pair["orders"]]
            ).contiguous()
            realized_relative_norms = _float32_realized_relative_norms(
                torch, captured_residuals, delta
            )
            if method.get("realized_relative_norms") != realized_relative_norms:
                raise RuntimeError("construction realized float32 norms are not residual-derived")
            if any(value > norm_cap for value in realized_relative_norms):
                expected_failure = {
                    "failure": "float32_realized_norm_cap_exceeded",
                    "realized_relative_norms": realized_relative_norms,
                    "maximum_relative_norm": norm_cap,
                    "conceptual_exact_head_evaluations": 0,
                }
                if eligible or method.get("failure") != expected_failure:
                    raise RuntimeError("construction realized-cap failure is not reproducible")
                continue
            conceptual_head_evaluations += 4
            _validate_actual_head_certificate(
                torch,
                method.get("actual_head_certificate"),
                delta,
                pair["orders"],
                acceptance_reserve=acceptance_reserve,
                require_all_targets=eligible,
            )
            if method.get("conceptual_exact_head_evaluations") != 4:
                raise RuntimeError("constructed method head count is invalid")
            if eligible:
                if "failure" in method:
                    raise RuntimeError("eligible construction method contains a failure")
                eligible_counts[method_id] += 1
            else:
                expected_failure = {
                    "failure": "actual_float32_head_failure",
                    "conceptual_exact_head_evaluations": 4,
                }
                if method.get("failure") != expected_failure:
                    raise RuntimeError("construction actual-head failure is not reproducible")

    identity_minimum_cosine = min(
        float(row["captured_to_analytic_cosine"]) for row in all_identity_rows
    )
    identity_maximum_relative_error = max(
        float(row["captured_to_analytic_relative_l2_error"]) for row in all_identity_rows
    )
    guards = config["architecture_guards"]
    if maximum_primal_error > float(guards["maximum_baseline_head_primal_absolute_error"]):
        raise RuntimeError("construction baseline head exceeds the locked primal-error guard")
    if identity_minimum_cosine < float(
        guards["minimum_captured_to_analytic_gradient_cosine"]
    ) or identity_maximum_relative_error > float(
        guards["maximum_captured_to_analytic_gradient_relative_l2_error"]
    ):
        raise RuntimeError("construction analytic identity exceeds the locked guard")

    summary = bank.get("summary")
    if not isinstance(summary, Mapping):
        raise TypeError("construction summary is missing")
    elapsed = summary.get("elapsed_seconds")
    if type(elapsed) is not float or not math.isfinite(elapsed) or elapsed < 0.0:
        raise RuntimeError("construction elapsed time is invalid")
    expected_summary = {
        "pair_count": 16,
        "method_ids": list(method_ids),
        "eligible_pair_count_by_method": eligible_counts,
        "minimum_captured_to_analytic_gradient_cosine": identity_minimum_cosine,
        "maximum_captured_to_analytic_gradient_relative_l2_error": (
            identity_maximum_relative_error
        ),
        "maximum_baseline_head_primal_absolute_error": maximum_primal_error,
        "elapsed_seconds": elapsed,
        "new_full_forward_passes": 0,
        "new_backward_passes": 0,
        "conceptual_exact_head_evaluations": conceptual_head_evaluations,
    }
    if summary != expected_summary:
        raise RuntimeError("construction summary differs from recomputed evidence")
    return bank


def _validate_manifest(bank: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if bank is None:
        import torch

        bank = _load_construction(torch)
    manifest = json.loads(CONSTRUCTION_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest != _manifest_payload(bank):
        raise RuntimeError("exact-head construction manifest differs from the validated bank")
    return manifest


def _validate_construction_attempt(bank: Mapping[str, Any]) -> dict[str, Any]:
    config = _load_config()
    attempt = json.loads(CONSTRUCTION_ATTEMPT_PATH.read_text(encoding="utf-8"))
    copy = dict(attempt)
    embedded = copy.pop("result_sha256", None)
    if embedded != _canonical_sha256(copy):
        raise RuntimeError("construction attempt ledger embedded hash differs")
    expected = {
        "schema_version": CONSTRUCTION_ATTEMPT_SCHEMA,
        "status": "complete",
        "config_sha256": _sha256(CONFIG_PATH),
        "input_commit": attempt.get("input_commit"),
        "reserved_model_loads": 1,
        "completed_model_loads": 1,
        "reserved_conceptual_exact_head_evaluations": int(
            config["compute"]["maximum_construction_exact_head_evaluations"]
        ),
        "completed_conceptual_exact_head_evaluations": int(
            bank["summary"]["conceptual_exact_head_evaluations"]
        ),
        "construction_bank_sha256": _sha256(CONSTRUCTION_PATH),
        "construction_manifest_sha256": _sha256(CONSTRUCTION_MANIFEST_PATH),
    }
    if (
        not isinstance(expected["input_commit"], str)
        or len(expected["input_commit"]) != 40
        or copy != expected
        or expected["completed_conceptual_exact_head_evaluations"]
        > expected["reserved_conceptual_exact_head_evaluations"]
    ):
        raise RuntimeError("construction attempt ledger differs from validated evidence")
    return attempt


def _freeze_bound_paths() -> list[Path]:
    return [
        CONFIG_PATH,
        *(ROOT / relative for relative in sorted(PRECONSTRUCTION_RELATIVES)),
        CONSTRUCTION_PATH,
        CONSTRUCTION_MANIFEST_PATH,
        CONSTRUCTION_ATTEMPT_PATH,
    ]


def run_freeze() -> dict[str, Any]:
    _load_config()
    import torch

    bank = _load_construction(torch)
    _validate_manifest(bank)
    _validate_construction_attempt(bank)
    downstream = (FREEZE_PATH, CHECKPOINT_PATH, LOGITS_ROOT, RESULT_PATH, REPORT_PATH)
    existing = [path for path in downstream if path.exists()]
    if existing:
        raise RuntimeError(
            "freeze is single-attempt and found an existing outcome artifact: "
            + ", ".join(_relative(path) for path in existing)
        )
    paths = _freeze_bound_paths()
    _require_committed_clean(paths)
    payload = {
        "schema_version": FREEZE_SCHEMA,
        "status": "ready_for_opened_finite_evaluation",
        "outcomes_viewed": False,
        "outcomes_viewed_scope": "hooked_full_model_intervention_outcomes",
        "config_sha256": _sha256(CONFIG_PATH),
        "construction_commit": _git("rev-parse", "HEAD"),
        "bound_files": {_relative(path): _sha256(path) for path in paths},
        "rule": "freeze_must_be_committed_and_clean_before_any_hooked_forward",
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    _atomic_json(FREEZE_PATH, payload)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def _validate_freeze() -> dict[str, Any]:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if (
        freeze.get("schema_version") != FREEZE_SCHEMA
        or freeze.get("status") != "ready_for_opened_finite_evaluation"
        or freeze.get("outcomes_viewed") is not False
        or freeze.get("outcomes_viewed_scope") != "hooked_full_model_intervention_outcomes"
        or freeze.get("config_sha256") != _sha256(CONFIG_PATH)
    ):
        raise RuntimeError("exact-head preoutcome freeze has invalid identity")
    copy = dict(freeze)
    embedded = copy.pop("result_sha256", None)
    if embedded != _canonical_sha256(copy):
        raise RuntimeError("preoutcome freeze embedded hash differs")
    construction_commit = freeze.get("construction_commit")
    expected = {
        "schema_version": FREEZE_SCHEMA,
        "status": "ready_for_opened_finite_evaluation",
        "outcomes_viewed": False,
        "outcomes_viewed_scope": "hooked_full_model_intervention_outcomes",
        "config_sha256": _sha256(CONFIG_PATH),
        "construction_commit": construction_commit,
        "bound_files": {_relative(path): _sha256(path) for path in _freeze_bound_paths()},
        "rule": "freeze_must_be_committed_and_clean_before_any_hooked_forward",
    }
    if (
        not isinstance(construction_commit, str)
        or len(construction_commit) != 40
        or copy != expected
    ):
        raise RuntimeError("preoutcome freeze does not exactly bind the construction")
    _require_committed_clean([*_freeze_bound_paths(), FREEZE_PATH])
    _git("merge-base", "--is-ancestor", construction_commit, "HEAD")
    return freeze


def _write_checkpoint(checkpoint: dict[str, Any]) -> None:
    checkpoint["cells_sha256"] = _canonical_sha256(checkpoint["cells"])
    checkpoint.pop("checkpoint_sha256", None)
    checkpoint["checkpoint_sha256"] = _canonical_sha256(checkpoint)
    _atomic_json(CHECKPOINT_PATH, checkpoint)


def _seal_cell(cell: dict[str, Any], previous: str | None) -> None:
    cell["previous_cell_sha256"] = previous
    cell.pop("cell_sha256", None)
    cell["cell_sha256"] = _canonical_sha256(cell)


def _logits_artifact_path(ordinal: int, work_id: str) -> Path:
    suffix = hashlib.sha256(work_id.encode("utf-8")).hexdigest()[:16]
    return LOGITS_ROOT / f"{ordinal:03d}_{suffix}.pt"


def _require_new_logits_artifact(path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise RuntimeError(
            f"evaluation refuses to replace an existing raw-logits artifact: {_relative(path)}"
        )


def _validate_cell(
    row: Mapping[str, Any],
    work: Mapping[str, Any],
    *,
    ordinal: int,
    previous: str | None,
    logits_override: Any | None = None,
) -> None:
    copy = dict(row)
    embedded = copy.pop("cell_sha256", None)
    if embedded != _canonical_sha256(copy):
        raise RuntimeError("evaluation cell embedded hash differs")
    expected_artifact = _logits_artifact_path(ordinal, str(work["work_id"]))
    expected = {
        "work_id": str(work["work_id"]),
        "ordinal": ordinal,
        "method_id": str(work["method_id"]),
        "case_id": str(work["case_id"]),
        "assignment": int(work["assignment"]),
        "order_index": int(work["order_index"]),
        "condition": str(work["condition"]),
        "sign": float(work["sign"]),
        "wanted_semantic_choice": str(work["wanted"]),
        "prompt_sha256": str(work["order"]["prompt_sha256"]),
        "unsigned_order_delta_sha256": str(work["unsigned_order_delta_sha256"]),
        "signed_delta_sha256": _raw_tensor_hash(
            (float(work["sign"]) * work["delta"]).float().contiguous()
        ),
        "logits_artifact_path": _relative(expected_artifact),
        "previous_cell_sha256": previous,
    }
    for field, value in expected.items():
        if type(row.get(field)) is not type(value) or row.get(field) != value:
            raise RuntimeError(f"evaluation cell identity differs: {field}")
    expected_cell_fields = set(expected) | {
        "cell_sha256",
        "logits_artifact_sha256",
        "logits_float32_sha256",
        "exact_argmax_token_id",
        "predicted_label",
        "preserve_minus_comply_log_odds",
        "preserve_pair_probability",
        "pair_choice",
        "answer_pair_mass",
        "full_vocabulary_kl_changed_to_baseline",
        "perturbation",
        "choice_boundary_evidence_sha256",
        "target_margin",
        "semantic_choice",
        "target_met",
        "decision_changed",
        "intended_delta_l2_norm",
        "realized_delta_l2_norm",
        "realized_relative_norm",
    }
    if set(row) != expected_cell_fields:
        raise RuntimeError("evaluation cell has extra or missing fields")

    import torch

    if not expected_artifact.is_file() or _sha256(expected_artifact) != row.get(
        "logits_artifact_sha256"
    ):
        raise RuntimeError("evaluation logits artifact hash differs")
    logits = (
        torch.load(expected_artifact, map_location="cpu", weights_only=True)
        if logits_override is None
        else logits_override.detach().float().cpu().contiguous()
    )
    if (
        not torch.is_tensor(logits)
        or logits.dtype != torch.float32
        or logits.ndim != 1
        or int(logits.numel()) < 2
        or not bool(torch.isfinite(logits).all().item())
    ):
        raise RuntimeError("evaluation logits artifact is not one finite float32 vector")
    if _raw_tensor_hash(logits) != row.get("logits_float32_sha256"):
        raise RuntimeError("evaluation logits tensor hash differs")

    exact_id = int(logits.argmax().item())
    if row.get("exact_argmax_token_id") != exact_id:
        raise RuntimeError("evaluation exact argmax is not logits-derived")
    preserve_id = int(work["order"]["preserve_token_id"])
    comply_id = int(work["order"]["comply_token_id"])
    expected_semantic = (
        "positive" if exact_id == preserve_id else "negative" if exact_id == comply_id else "OTHER"
    )
    if row.get("semantic_choice") != expected_semantic:
        raise RuntimeError("evaluation cell semantic choice is not argmax-derived")
    expected_predicted_label = (
        str(work["job"]["positive_label"])
        if expected_semantic == "positive"
        else str(work["job"]["negative_label"])
        if expected_semantic == "negative"
        else "OTHER"
    )
    if row.get("predicted_label") != expected_predicted_label:
        raise RuntimeError("evaluation predicted label is not argmax-derived")
    if row.get("choice_boundary_evidence_sha256") != str(
        work["order"]["choice_boundary_evidence_sha256"]
    ):
        raise RuntimeError("evaluation choice boundary differs from the construction")

    numeric = (
        "preserve_minus_comply_log_odds",
        "preserve_pair_probability",
        "answer_pair_mass",
        "full_vocabulary_kl_changed_to_baseline",
        "target_margin",
        "intended_delta_l2_norm",
        "realized_delta_l2_norm",
        "realized_relative_norm",
    )
    if any(type(row.get(field)) is not float or not math.isfinite(row[field]) for field in numeric):
        raise RuntimeError("evaluation cell has a non-finite numeric field")
    if float(row["full_vocabulary_kl_changed_to_baseline"]) < -1e-6:
        raise RuntimeError("evaluation cell KL is materially negative")
    if not 0.0 <= float(row["answer_pair_mass"]) <= 1.0:
        raise RuntimeError("evaluation answer-pair mass is outside [0,1]")

    target = preserve_id if float(work["sign"]) > 0 else comply_id
    _competitor, target_margin = _target_margin(torch, logits, target)
    if not math.isclose(float(row["target_margin"]), target_margin, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError("evaluation target margin is not logits-derived")
    expected_target_met = bool(
        expected_semantic == work["wanted"]
        and target_margin >= float(work["minimum_target_margin"])
    )
    if row.get("target_met") is not expected_target_met:
        raise RuntimeError("evaluation target flag is not logits-derived")
    changed = expected_semantic != str(work["order"]["baseline_semantic_choice"])
    if row.get("decision_changed") is not changed:
        raise RuntimeError("evaluation change flag is not baseline-derived")

    log_odds = float((logits[preserve_id] - logits[comply_id]).item())
    if not math.isclose(float(row["preserve_minus_comply_log_odds"]), log_odds, abs_tol=1e-7):
        raise RuntimeError("evaluation log-odds is not logits-derived")
    expected_pair_choice = (
        str(work["job"]["positive_label"])
        if log_odds >= 0.0
        else str(work["job"]["negative_label"])
    )
    if row.get("pair_choice") != expected_pair_choice:
        raise RuntimeError("evaluation pair choice is not log-odds-derived")
    pair_probability = float(
        torch.softmax(torch.stack([logits[preserve_id], logits[comply_id]]), dim=0)[0].item()
    )
    if not math.isclose(
        float(row["preserve_pair_probability"]), pair_probability, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise RuntimeError("evaluation pair probability is not logits-derived")
    probabilities = torch.softmax(logits, dim=-1)
    pair_mass = float((probabilities[preserve_id] + probabilities[comply_id]).item())
    if not math.isclose(float(row["answer_pair_mass"]), pair_mass, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError("evaluation answer-pair mass is not logits-derived")
    baseline = work["order"]["baseline_logits"].float().cpu().contiguous()
    baseline_log_probs = torch.log_softmax(baseline, dim=-1)
    changed_log_probs = torch.log_softmax(logits, dim=-1)
    kl = float((changed_log_probs.exp() * (changed_log_probs - baseline_log_probs)).sum().item())
    if not math.isclose(
        float(row["full_vocabulary_kl_changed_to_baseline"]), kl, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise RuntimeError("evaluation full-vocabulary KL is not logits-derived")
    if expected_semantic == "positive" and log_odds < 0.0:
        raise RuntimeError("positive argmax disagrees with preserve-minus-comply log-odds")
    if expected_semantic == "negative" and log_odds > 0.0:
        raise RuntimeError("negative argmax disagrees with preserve-minus-comply log-odds")

    intended = float(work["delta"].double().norm().item())
    if not math.isclose(float(row["intended_delta_l2_norm"]), intended, abs_tol=1e-12):
        raise RuntimeError("evaluation intended delta norm differs")
    realized = float(row["realized_delta_l2_norm"])
    if abs(realized - intended) > 1e-5 + 1e-5 * max(1.0, intended):
        raise RuntimeError("evaluation realized delta norm differs")
    perturbation = row.get("perturbation")
    expected_perturbation_fields = {
        "n_positions",
        "total_frobenius_norm",
        "mean_l2_norm",
        "rms_l2_norm",
        "max_l2_norm",
        "mean_relative_l2_norm",
        "max_relative_l2_norm",
        "zero_reference_positions",
        "live_reference_l2_norm",
        "live_reference_float32_sha256",
    }
    if not isinstance(perturbation, Mapping) or set(perturbation) != expected_perturbation_fields:
        raise RuntimeError("evaluation perturbation diagnostics have the wrong schema")
    if type(perturbation["n_positions"]) is not int or perturbation["n_positions"] != 1:
        raise RuntimeError("evaluation perturbation changed the wrong number of positions")
    if (
        type(perturbation["zero_reference_positions"]) is not int
        or perturbation["zero_reference_positions"] != 0
    ):
        raise RuntimeError("evaluation perturbation has a zero live reference")
    for field in ("total_frobenius_norm", "mean_l2_norm", "rms_l2_norm", "max_l2_norm"):
        value = perturbation[field]
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not math.isclose(value, realized, rel_tol=1e-6, abs_tol=1e-6)
        ):
            raise RuntimeError(f"evaluation realized perturbation differs: {field}")
    live_norm = perturbation["live_reference_l2_norm"]
    if (
        type(live_norm) is not float
        or not math.isfinite(live_norm)
        or not math.isclose(
            live_norm,
            float(work["order"]["residual_norm"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or perturbation["live_reference_float32_sha256"] != str(work["order"]["residual_sha256"])
    ):
        raise RuntimeError("evaluation live reference differs from the frozen capture")
    captured_residual = work["captured_residual"].float().cpu().contiguous()
    signed_delta = (float(work["sign"]) * work["delta"]).float().cpu().contiguous()
    realized_vector = (captured_residual + signed_delta) - captured_residual
    expected_relative = float((realized_vector.norm() / captured_residual.norm()).item())
    for field in ("mean_relative_l2_norm", "max_relative_l2_norm"):
        value = perturbation[field]
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not math.isclose(value, expected_relative, rel_tol=1e-5, abs_tol=1e-5)
        ):
            raise RuntimeError(f"evaluation live relative perturbation differs: {field}")
    if not math.isclose(
        float(row["realized_relative_norm"]), expected_relative, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise RuntimeError("evaluation relative norm is not live-activation-derived")
    if expected_relative > float(work["maximum_relative_norm"]):
        raise RuntimeError("evaluation perturbation exceeds the norm cap")


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    expected_work: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, str],
    *,
    expected_head_evaluations: int = 224,
) -> None:
    expected_fields = {
        "schema_version",
        "status",
        "references",
        "resident_head_revalidation",
        "pending_reservation",
        "cells",
        "compute",
        "cells_sha256",
        "checkpoint_sha256",
    }
    if set(checkpoint) != expected_fields:
        raise RuntimeError("evaluation checkpoint has an invalid schema")
    copy = dict(checkpoint)
    embedded = copy.pop("checkpoint_sha256", None)
    if embedded != _canonical_sha256(copy):
        raise RuntimeError("evaluation checkpoint embedded hash differs")
    if (
        checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
        or checkpoint.get("status") not in {"in_progress", "complete"}
        or checkpoint.get("references") != dict(references)
    ):
        raise RuntimeError("evaluation checkpoint identity differs")
    if checkpoint.get("cells_sha256") != _canonical_sha256(checkpoint.get("cells")):
        raise RuntimeError("evaluation checkpoint cells hash differs")
    cells = checkpoint.get("cells")
    if not isinstance(cells, list):
        raise TypeError("evaluation checkpoint cells are not a list")
    expected_ids = list(expected_work)
    observed_ids = [str(row.get("work_id")) for row in cells]
    if observed_ids != expected_ids[: len(observed_ids)]:
        raise RuntimeError("evaluation cells are not the exact work-plan prefix")
    previous = None
    for ordinal, row in enumerate(cells):
        _validate_cell(
            row, expected_work[observed_ids[ordinal]], ordinal=ordinal, previous=previous
        )
        previous = str(row["cell_sha256"])
    compute = checkpoint.get("compute")
    if not isinstance(compute, Mapping):
        raise TypeError("evaluation checkpoint compute ledger is invalid")
    expected_compute_fields = {
        "reserved_resident_head_revalidation_evaluations",
        "completed_resident_head_revalidation_evaluations",
        "reserved_intervention_forward_passes",
        "completed_intervention_forward_passes",
    }
    if set(compute) != expected_compute_fields:
        raise RuntimeError("evaluation checkpoint compute ledger has extra or missing fields")
    if type(expected_head_evaluations) is not int or expected_head_evaluations <= 0:
        raise ValueError("expected resident-head evaluation count must be a positive integer")
    head_reserved = compute["reserved_resident_head_revalidation_evaluations"]
    head_completed = compute["completed_resident_head_revalidation_evaluations"]
    completed_raw = compute.get("completed_intervention_forward_passes")
    reserved_raw = compute.get("reserved_intervention_forward_passes")
    if (
        type(head_reserved) is not int
        or type(head_completed) is not int
        or type(completed_raw) is not int
        or type(reserved_raw) is not int
        or head_reserved < 0
        or head_completed < 0
        or completed_raw < 0
        or reserved_raw < 0
    ):
        raise TypeError("evaluation checkpoint compute counts must be non-negative integers")
    completed = completed_raw
    reserved = reserved_raw
    if completed != len(cells) or reserved not in {completed, completed + 1}:
        raise RuntimeError("evaluation checkpoint compute ledger differs from cells")
    head = checkpoint.get("resident_head_revalidation")
    if not isinstance(head, Mapping) or set(head) != {"status"}:
        raise RuntimeError("evaluation checkpoint resident-head state is invalid")
    head_status = head["status"]
    pending = checkpoint.get("pending_reservation")
    if head_status == "not_started":
        if (
            head_reserved != 0
            or head_completed != 0
            or cells
            or pending is not None
            or reserved != 0
            or completed != 0
            or checkpoint.get("status") != "in_progress"
        ):
            raise RuntimeError("unstarted resident-head state has spent evaluation work")
        return
    if head_status == "reserved":
        expected_pending = {
            "work_id": "resident_head_revalidation",
            "ordinal": -1,
            "operation": "resident_head_revalidation",
        }
        if (
            head_reserved != expected_head_evaluations
            or head_completed != 0
            or cells
            or pending != expected_pending
            or reserved != 0
            or completed != 0
            or checkpoint.get("status") != "in_progress"
        ):
            raise RuntimeError("reserved resident-head state has an invalid ledger")
        return
    if head_status != "complete" or (
        head_reserved != expected_head_evaluations or head_completed != expected_head_evaluations
    ):
        raise RuntimeError("completed resident-head state has an invalid ledger")
    if pending is None and reserved != completed:
        raise RuntimeError("evaluation checkpoint has an unrecorded reservation")
    if pending is not None:
        expected_pending = {
            "work_id": expected_ids[len(cells)] if len(cells) < len(expected_ids) else None,
            "ordinal": len(cells),
            "operation": "intervention_forward",
        }
        if pending != expected_pending or reserved != completed + 1:
            raise RuntimeError("evaluation pending reservation differs from the next call")
    if checkpoint.get("status") == "complete" and (
        pending is not None or len(cells) != len(expected_ids)
    ):
        raise RuntimeError("complete checkpoint does not contain all work")


def _score_intervention(
    adaptive: Any,
    backend: Any,
    job: Mapping[str, Any],
    order: Mapping[str, Any],
    delta: Any,
    captured_residual: Any,
    *,
    layer: int,
    sign: float,
) -> tuple[Any, Any, int, float]:
    tokens = backend.encode(str(job["prompt"]))
    boundary = adaptive.resolve_choice_boundary(backend, str(job["prompt"]))
    if int(tokens.shape[-1]) != int(order["prompt_length"]) or boundary.prompt_length != int(
        order["prompt_length"]
    ):
        raise RuntimeError("evaluation prompt length differs from the frozen construction")
    spec = adaptive.InterventionSpec(
        layer=layer,
        direction=delta.to(backend.device),
        strength=sign,
        geometry="matched_final_prompt",
        prompt_length=int(order["prompt_length"]),
        magnitude_mode="canonical_coefficient",
    )
    torch = backend.torch
    captured: dict[str, Any] = {}

    def diagnostic_hook(activation: Any, hook: Any) -> Any:
        del hook
        live = activation[0, -1].detach().float().cpu().contiguous()
        expected = captured_residual.detach().float().cpu().contiguous()
        if not torch.equal(live, expected):
            raise RuntimeError("live intervention residual differs from the frozen capture")
        changed = apply_intervention(torch, activation, spec)
        mask = intervention_mask(torch, activation, spec).squeeze(-1)
        diagnostics = actual_perturbation_norms(
            torch,
            activation,
            changed,
            position_mask=mask,
        )
        diagnostics.update(
            {
                "live_reference_l2_norm": float(live.double().norm().item()),
                "live_reference_float32_sha256": _raw_tensor_hash(live),
            }
        )
        captured.update(diagnostics)
        return changed

    with (
        torch.inference_mode(),
        backend.model.hooks(fwd_hooks=[(hook_name(layer), diagnostic_hook)]),
    ):
        logits = backend.model(tokens)[0, -1].detach().float().cpu().contiguous()
    if not captured:
        raise RuntimeError("intervention hook did not capture perturbation diagnostics")
    perturbation = captured
    baseline_logits = order["baseline_logits"].float().cpu().contiguous()
    exact_id = int(logits.argmax().item())
    score = adaptive.choice_score_from_logits(
        torch,
        logits,
        int(order["preserve_token_id"]),
        int(order["comply_token_id"]),
        preserve_label=str(job["positive_label"]),
        comply_label=str(job["negative_label"]),
        baseline_logits=baseline_logits,
        perturbation=perturbation,
        choice_boundary_evidence_sha256=boundary.evidence_sha256,
        choice_a_token_id=boundary.a_token_id,
        choice_b_token_id=boundary.b_token_id,
    )
    target = int(order["preserve_token_id"] if sign > 0 else order["comply_token_id"])
    _competitor, margin = _target_margin(torch, logits, target)
    return score, logits, exact_id, margin


def _public_score(score: Any) -> dict[str, Any]:
    return {
        "predicted_label": str(score.predicted_label),
        "preserve_minus_comply_log_odds": float(score.preserve_log_odds),
        "preserve_pair_probability": float(score.preserve_pair_probability),
        "pair_choice": str(score.pair_choice),
        "answer_pair_mass": float(score.answer_pair_mass),
        "full_vocabulary_kl_changed_to_baseline": float(score.kl_from_baseline),
        "perturbation": score.perturbation,
        "choice_boundary_evidence_sha256": str(score.choice_boundary_evidence_sha256),
    }


def _aggregate(
    bank: Mapping[str, Any], cells: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    methods = {}
    required_pairs = int(config["opened_pass_gate"]["required_successful_pairs_per_method"])
    required_cells = int(config["opened_pass_gate"]["required_successful_cells_per_method"])
    for method_id in ("gradient_ray", "effective_unembedding_ray"):
        pair_rows = []
        method_cells = []
        for pair in bank["pairs"]:
            key = (str(pair["case_id"]), int(pair["assignment"]))
            pair_cells = []
            order_flip = []
            for order_index in range(2):
                rows = [
                    cells[f"{method_id}:{key[0]}:{key[1]}:{order_index}:{condition}"]
                    for condition in ("plus", "minus")
                ]
                pair_cells.extend(rows)
                order_flip.append(any(bool(row["decision_changed"]) for row in rows))
            success = bool(
                all(bool(row["target_met"]) for row in pair_cells)
                and all(order_flip)
                and all(row["semantic_choice"] != "OTHER" for row in pair_cells)
                and all(
                    float(row["target_margin"])
                    >= float(config["opened_pass_gate"]["minimum_actual_target_margin"])
                    for row in pair_cells
                )
            )
            pair_rows.append(
                {
                    "case_id": key[0],
                    "assignment": key[1],
                    "alpha": float(pair["methods"][method_id]["alpha"]),
                    "passes": success,
                    "real_flip_in_each_answer_order": all(order_flip),
                    "successful_cell_count": sum(bool(row["target_met"]) for row in pair_cells),
                }
            )
            method_cells.extend(pair_cells)
        successful_cells = sum(bool(row["target_met"]) for row in method_cells)
        successful_pairs = sum(bool(row["passes"]) for row in pair_rows)
        kls = [float(row["full_vocabulary_kl_changed_to_baseline"]) for row in method_cells]
        methods[method_id] = {
            "passes": bool(
                successful_pairs == required_pairs and successful_cells == required_cells
            ),
            "successful_pair_count": successful_pairs,
            "successful_cell_count": successful_cells,
            "required_pair_count": required_pairs,
            "required_cell_count": required_cells,
            "real_decision_change_count": sum(
                bool(row["decision_changed"]) for row in method_cells
            ),
            "other_argmax_count": sum(row["semantic_choice"] == "OTHER" for row in method_cells),
            "target_kl": {
                "mean": statistics.fmean(kls),
                "p95": float(np.quantile(kls, 0.95, method="inverted_cdf")),
                "maximum": max(kls),
            },
            "pairs": pair_rows,
        }
    return methods


def _paired_gradient_unembedding_comparison(
    bank: Mapping[str, Any], cells: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    rows = []
    for pair in bank["pairs"]:
        case_id = str(pair["case_id"])
        assignment = int(pair["assignment"])
        gradient = pair["methods"]["gradient_ray"]
        unembedding = pair["methods"]["effective_unembedding_ray"]
        gradient_kls = []
        unembedding_kls = []
        for order_index in range(2):
            for condition in ("plus", "minus"):
                gradient_kls.append(
                    float(
                        cells[f"gradient_ray:{case_id}:{assignment}:{order_index}:{condition}"][
                            "full_vocabulary_kl_changed_to_baseline"
                        ]
                    )
                )
                unembedding_kls.append(
                    float(
                        cells[
                            f"effective_unembedding_ray:{case_id}:{assignment}:{order_index}:{condition}"
                        ]["full_vocabulary_kl_changed_to_baseline"]
                    )
                )
        rows.append(
            {
                "case_id": case_id,
                "assignment": assignment,
                "gradient_alpha": float(gradient["alpha"]),
                "effective_unembedding_alpha": float(unembedding["alpha"]),
                "gradient_minus_unembedding_alpha": float(gradient["alpha"] - unembedding["alpha"]),
                "gradient_mean_relative_norm": statistics.fmean(
                    float(value) for value in gradient["relative_norms"]
                ),
                "effective_unembedding_mean_relative_norm": statistics.fmean(
                    float(value) for value in unembedding["relative_norms"]
                ),
                "gradient_minus_unembedding_mean_relative_norm": statistics.fmean(
                    float(value) for value in gradient["relative_norms"]
                )
                - statistics.fmean(float(value) for value in unembedding["relative_norms"]),
                "gradient_mean_kl": statistics.fmean(gradient_kls),
                "effective_unembedding_mean_kl": statistics.fmean(unembedding_kls),
                "gradient_minus_unembedding_mean_kl": statistics.fmean(gradient_kls)
                - statistics.fmean(unembedding_kls),
                "per_order_direction_cosine": [
                    float(row["cosine"]) for row in pair["gradient_unembedding_comparison"]
                ],
            }
        )
    return {
        "pair_count": len(rows),
        "mean_gradient_minus_unembedding_alpha": statistics.fmean(
            row["gradient_minus_unembedding_alpha"] for row in rows
        ),
        "mean_gradient_minus_unembedding_relative_norm": statistics.fmean(
            row["gradient_minus_unembedding_mean_relative_norm"] for row in rows
        ),
        "mean_gradient_minus_unembedding_kl": statistics.fmean(
            row["gradient_minus_unembedding_mean_kl"] for row in rows
        ),
        "minimum_direction_cosine": min(
            cosine for row in rows for cosine in row["per_order_direction_cosine"]
        ),
        "maximum_direction_cosine": max(
            cosine for row in rows for cosine in row["per_order_direction_cosine"]
        ),
        "rows": rows,
    }


def _evaluation_work_plan(
    old: Any,
    capture: Mapping[str, Any],
    bank: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[
    Any, dict[tuple[str, int], Mapping[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]
]:
    adaptive, groups = old._active_jobs()
    jobs = {
        (str(key[0]), int(key[1]), bool(job["preserve_first"])): job
        for key, pair_jobs in groups
        for job in pair_jobs
    }
    capture_by_key = {
        (str(pair["case_id"]), int(pair["assignment"])): pair for pair in capture["pairs"]
    }
    expected_work: dict[str, dict[str, Any]] = {}
    for pair in bank["pairs"]:
        key = (str(pair["case_id"]), int(pair["assignment"]))
        captured_pair = capture_by_key[key]
        for method_id in ("gradient_ray", "effective_unembedding_ray"):
            method = pair["methods"][method_id]
            for order_index, order in enumerate(pair["orders"]):
                captured_order = captured_pair["orders"][order_index]
                if str(order["prompt_sha256"]) != str(captured_order["prompt_sha256"]):
                    raise RuntimeError("construction/capture order mapping differs")
                job = jobs[(key[0], key[1], bool(order["preserve_first"]))]
                if str(job["prompt_sha256"]) != str(order["prompt_sha256"]):
                    raise RuntimeError("evaluation job prompt differs from the construction")
                if str(job["positive_label"]) != str(order["positive_label"]) or str(
                    job["negative_label"]
                ) != str(order["negative_label"]):
                    raise RuntimeError("evaluation answer labels differ from the construction")
                delta = method["delta"][order_index].float().contiguous()
                for condition, sign, wanted in (
                    ("plus", 1.0, "positive"),
                    ("minus", -1.0, "negative"),
                ):
                    work_id = f"{method_id}:{key[0]}:{key[1]}:{order_index}:{condition}"
                    expected_work[work_id] = {
                        "work_id": work_id,
                        "method_id": method_id,
                        "case_id": key[0],
                        "assignment": key[1],
                        "order_index": order_index,
                        "condition": condition,
                        "sign": sign,
                        "wanted": wanted,
                        "order": order,
                        "job": job,
                        "captured_residual": captured_order["residual"].float().cpu().contiguous(),
                        "delta": delta,
                        "unsigned_order_delta_sha256": _raw_tensor_hash(delta),
                        "minimum_target_margin": float(
                            config["opened_pass_gate"]["minimum_actual_target_margin"]
                        ),
                        "maximum_relative_norm": float(
                            config["intervention"]["maximum_relative_norm"]
                        ),
                    }
    expected_call_count = int(config["compute"]["maximum_intervention_forward_passes"])
    if len(expected_work) != expected_call_count:
        raise RuntimeError("evaluation work plan does not contain the exact locked unique calls")
    public_plan = [
        {
            "work_id": work_id,
            "method_id": work["method_id"],
            "case_id": work["case_id"],
            "assignment": work["assignment"],
            "order_index": work["order_index"],
            "condition": work["condition"],
            "sign": work["sign"],
            "prompt_sha256": work["order"]["prompt_sha256"],
            "unsigned_order_delta_sha256": work["unsigned_order_delta_sha256"],
        }
        for work_id, work in expected_work.items()
    ]
    return adaptive, capture_by_key, expected_work, public_plan


def _attribution_payload(bank: Mapping[str, Any]) -> dict[str, Any]:
    summary = bank["summary"]
    gradient_delta_matches_tangent = sum(
        pair["methods"]["gradient_ray"].get("delta_sha256")
        == pair["methods"]["analytic_rms_tangent_ray"].get("delta_sha256")
        for pair in bank["pairs"]
    )
    return {
        "minimum_captured_to_analytic_gradient_cosine": summary[
            "minimum_captured_to_analytic_gradient_cosine"
        ],
        "maximum_captured_to_analytic_gradient_relative_l2_error": summary[
            "maximum_captured_to_analytic_gradient_relative_l2_error"
        ],
        "gradient_delta_exactly_matches_analytic_tangent_pair_count": int(
            gradient_delta_matches_tangent
        ),
        "pair_count": 16,
        "interpretation": (
            "late_gradient_is_rms_tangent_answer_token_boundary_not_independent_sp_evidence"
        ),
    }


def _failed_construction_payload(
    bank: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    eligible_counts = bank["summary"]["eligible_pair_count_by_method"]
    required = int(config["opened_pass_gate"]["required_eligible_pairs_per_method"])
    payload = {
        "schema_version": RESULT_SCHEMA,
        "status": "failed_construction",
        "development_only": True,
        "fresh_prospective_evidence": False,
        "config_sha256": _sha256(CONFIG_PATH),
        "capture_sha256": _sha256(CAPTURE_PATH),
        "construction_sha256": _sha256(CONSTRUCTION_PATH),
        "construction_manifest_sha256": _sha256(CONSTRUCTION_MANIFEST_PATH),
        "freeze_sha256": _sha256(FREEZE_PATH),
        "eligible_pair_count_by_method": eligible_counts,
        "required_eligible_pair_count_per_method": required,
        "methods": {
            method_id: {
                "passes": False,
                "eligible_pair_count": int(count),
                "required_eligible_pair_count": required,
            }
            for method_id, count in eligible_counts.items()
        },
        "compute": {
            "new_capture_full_forward_passes": 0,
            "new_backward_passes": 0,
            "conceptual_exact_head_evaluations": int(
                bank["summary"]["conceptual_exact_head_evaluations"]
            ),
            "completed_intervention_forward_passes": 0,
            "generated_tokens": 0,
            "external_api_calls": 0,
            "external_model_judge_calls": 0,
            "external_cost_usd": 0,
        },
        "claim_boundary": "Construction failure is a technical negative result only.",
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    return payload


def _evaluation_result_payload(
    bank: Mapping[str, Any],
    capture: Mapping[str, Any],
    freeze: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    methods: Mapping[str, Any],
    paired_comparison: Mapping[str, Any],
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    status = "passed" if all(bool(row["passes"]) for row in methods.values()) else "failed"
    payload = {
        "schema_version": RESULT_SCHEMA,
        "status": status,
        "development_only": True,
        "fresh_prospective_evidence": False,
        "config_sha256": _sha256(CONFIG_PATH),
        "capture_sha256": _sha256(CAPTURE_PATH),
        "construction_sha256": _sha256(CONSTRUCTION_PATH),
        "construction_manifest_sha256": _sha256(CONSTRUCTION_MANIFEST_PATH),
        "freeze_sha256": _sha256(FREEZE_PATH),
        "evaluation_checkpoint_sha256": _sha256(CHECKPOINT_PATH),
        "evaluation_cells_sha256": str(checkpoint["cells_sha256"]),
        "construction_commit": str(freeze["construction_commit"]),
        "methods": dict(methods),
        "paired_gradient_unembedding_comparison": dict(paired_comparison),
        "attribution": _attribution_payload(bank),
        "compute": {
            "reused_capture_full_forward_passes": int(capture["compute"]["full_forward_passes"]),
            "reused_capture_backward_passes": int(capture["compute"]["backward_passes"]),
            "new_capture_full_forward_passes": 0,
            "new_backward_passes": 0,
            "conceptual_exact_head_evaluations": int(
                bank["summary"]["conceptual_exact_head_evaluations"]
            ),
            "resident_head_revalidation_evaluations": int(
                bank["summary"]["conceptual_exact_head_evaluations"]
            ),
            **checkpoint["compute"],
            "generated_tokens": 0,
            "external_api_calls": 0,
            "external_model_judge_calls": 0,
            "external_cost_usd": 0,
            "elapsed_seconds_this_evaluation_invocation": elapsed_seconds,
        },
        "claim_boundary": (
            "A pass is forced-choice endpoint control on opened prompts. It is not a natural "
            "self-preservation mechanism, reusable SP vector, or intrinsic specificity result."
        ),
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    return payload


def _prepare_resident_head_for_evaluation(
    adaptive: Any,
    checkpoint: dict[str, Any],
    expected_work: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, str],
    bank: Mapping[str, Any],
    capture: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    head_evaluations: int,
) -> Any | None:
    if checkpoint.get("pending_reservation") is not None:
        raise RuntimeError("an earlier reserved operation is ambiguous; fail closed")
    if checkpoint.get("status") == "complete":
        return None
    head_status = checkpoint["resident_head_revalidation"]["status"]
    if head_status == "not_started":
        checkpoint["resident_head_revalidation"]["status"] = "reserved"
        checkpoint["pending_reservation"] = {
            "work_id": "resident_head_revalidation",
            "ordinal": -1,
            "operation": "resident_head_revalidation",
        }
        checkpoint["compute"]["reserved_resident_head_revalidation_evaluations"] = head_evaluations
        _write_checkpoint(checkpoint)
        _validate_checkpoint(
            checkpoint,
            expected_work,
            references,
            expected_head_evaluations=head_evaluations,
        )
        backend = adaptive.load_backend(adaptive.load_lock())
        live_architecture = _architecture(backend, config)
        if live_architecture["public"] != bank["architecture"]:
            raise RuntimeError("live model final head differs from the frozen construction")
        observed = _validate_construction_against_live_head(
            backend,
            bank,
            capture,
            live_architecture,
            config,
        )
        if observed != head_evaluations:
            raise RuntimeError("resident-head revalidation exceeded its reservation")
        checkpoint["resident_head_revalidation"]["status"] = "complete"
        checkpoint["pending_reservation"] = None
        checkpoint["compute"]["completed_resident_head_revalidation_evaluations"] = observed
        _write_checkpoint(checkpoint)
        _validate_checkpoint(
            checkpoint,
            expected_work,
            references,
            expected_head_evaluations=head_evaluations,
        )
        return backend
    if head_status == "complete":
        # A clean cell prefix is resumable. Rebind the newly loaded resident model by
        # exact parameter hashes, but do not replay the already completed head rows.
        backend = adaptive.load_backend(adaptive.load_lock())
        if _architecture(backend, config)["public"] != bank["architecture"]:
            raise RuntimeError("resumed live model final head differs from the frozen construction")
        return backend
    raise RuntimeError("evaluation checkpoint has a non-resumable resident-head state")


def run_evaluate() -> dict[str, Any]:
    config = _load_config()
    import torch

    old, capture = _load_capture(torch)
    bank = _load_construction(torch)
    _validate_manifest(bank)
    _validate_construction_attempt(bank)
    freeze = _validate_freeze()
    if RESULT_PATH.exists() or REPORT_PATH.exists():
        raise RuntimeError("evaluation result/report already exists; use the report phase")
    required = int(config["opened_pass_gate"]["required_eligible_pairs_per_method"])
    eligible_counts = bank["summary"]["eligible_pair_count_by_method"]
    if any(
        int(eligible_counts[method_id]) != required
        for method_id in (
            "gradient_ray",
            "analytic_rms_tangent_ray",
            "effective_unembedding_ray",
        )
    ):
        if CHECKPOINT_PATH.exists() or LOGITS_ROOT.exists():
            raise RuntimeError("construction failed but evaluation artifacts already exist")
        result = _failed_construction_payload(bank, config)
        _atomic_json(RESULT_PATH, result)
        print(json.dumps(result, indent=2), flush=True)
        return result
    adaptive, _capture_by_key, expected_work, public_plan = _evaluation_work_plan(
        old, capture, bank, config
    )
    references = {
        "config_sha256": _sha256(CONFIG_PATH),
        "construction_sha256": _sha256(CONSTRUCTION_PATH),
        "construction_manifest_sha256": _sha256(CONSTRUCTION_MANIFEST_PATH),
        "freeze_sha256": _sha256(FREEZE_PATH),
        "expected_work_sha256": _canonical_sha256(public_plan),
    }
    head_evaluations = int(bank["summary"]["conceptual_exact_head_evaluations"])
    checkpoint_existed = CHECKPOINT_PATH.exists()
    if checkpoint_existed:
        checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    else:
        if LOGITS_ROOT.exists():
            raise RuntimeError("evaluation logits exist without a checkpoint; fail closed")
        checkpoint = {
            "schema_version": CHECKPOINT_SCHEMA,
            "status": "in_progress",
            "references": references,
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
        _write_checkpoint(checkpoint)
    _validate_checkpoint(
        checkpoint,
        expected_work,
        references,
        expected_head_evaluations=head_evaluations,
    )
    started = time.perf_counter()
    backend = _prepare_resident_head_for_evaluation(
        adaptive,
        checkpoint,
        expected_work,
        references,
        bank,
        capture,
        config,
        head_evaluations=head_evaluations,
    )
    completed = {str(row["work_id"]): row for row in checkpoint["cells"]}
    layer = int(config["intervention"]["residual_layer_zero_based"])
    for work_id, work in expected_work.items():
        if work_id in completed:
            continue
        if backend is None:
            raise RuntimeError("completed checkpoint is missing a frozen work item")
        logits_artifact = _logits_artifact_path(len(checkpoint["cells"]), work_id)
        _require_new_logits_artifact(logits_artifact)
        checkpoint["pending_reservation"] = {
            "work_id": work_id,
            "ordinal": len(checkpoint["cells"]),
            "operation": "intervention_forward",
        }
        checkpoint["compute"]["reserved_intervention_forward_passes"] += 1
        _write_checkpoint(checkpoint)
        score, logits, exact_id, margin = _score_intervention(
            adaptive,
            backend,
            work["job"],
            work["order"],
            work["delta"],
            work["captured_residual"],
            layer=layer,
            sign=float(work["sign"]),
        )
        _atomic_torch_save(torch, logits_artifact, logits.float().cpu().contiguous())
        perturbation = score.perturbation
        if not isinstance(perturbation, Mapping) or int(perturbation["n_positions"]) != 1:
            raise RuntimeError("hooked intervention did not change exactly one position")
        intended_norm = float(work["delta"].double().norm().item())
        realized_norm = float(perturbation["mean_l2_norm"])
        semantic = (
            "positive"
            if exact_id == int(work["order"]["preserve_token_id"])
            else "negative"
            if exact_id == int(work["order"]["comply_token_id"])
            else "OTHER"
        )
        signed = (float(work["sign"]) * work["delta"]).float().contiguous()
        cell = {
            "work_id": work_id,
            "ordinal": len(checkpoint["cells"]),
            "method_id": str(work["method_id"]),
            "case_id": str(work["case_id"]),
            "assignment": int(work["assignment"]),
            "order_index": int(work["order_index"]),
            "condition": str(work["condition"]),
            "sign": float(work["sign"]),
            "wanted_semantic_choice": str(work["wanted"]),
            "prompt_sha256": str(work["order"]["prompt_sha256"]),
            "unsigned_order_delta_sha256": str(work["unsigned_order_delta_sha256"]),
            "signed_delta_sha256": _raw_tensor_hash(signed),
            "logits_artifact_path": _relative(logits_artifact),
            "logits_artifact_sha256": _sha256(logits_artifact),
            "logits_float32_sha256": _raw_tensor_hash(logits),
            "exact_argmax_token_id": exact_id,
            **_public_score(score),
            "target_margin": float(margin),
            "semantic_choice": semantic,
            "target_met": semantic == work["wanted"]
            and margin >= float(config["opened_pass_gate"]["minimum_actual_target_margin"]),
            "decision_changed": semantic != str(work["order"]["baseline_semantic_choice"]),
            "intended_delta_l2_norm": intended_norm,
            "realized_delta_l2_norm": realized_norm,
            "realized_relative_norm": float(perturbation["mean_relative_l2_norm"]),
        }
        _seal_cell(
            cell,
            None if not checkpoint["cells"] else str(checkpoint["cells"][-1]["cell_sha256"]),
        )
        _validate_cell(
            cell,
            work,
            ordinal=len(checkpoint["cells"]),
            previous=None
            if not checkpoint["cells"]
            else str(checkpoint["cells"][-1]["cell_sha256"]),
            logits_override=logits,
        )
        checkpoint["cells"].append(cell)
        checkpoint["pending_reservation"] = None
        checkpoint["compute"]["completed_intervention_forward_passes"] += 1
        _write_checkpoint(checkpoint)
        completed[work_id] = cell
        if len(completed) % 16 == 0 or len(completed) == len(expected_work):
            print(
                f"completed {len(completed)}/{len(expected_work)} frozen interventions", flush=True
            )
    if checkpoint.get("status") != "complete":
        checkpoint["status"] = "complete"
        _write_checkpoint(checkpoint)
    _validate_checkpoint(
        checkpoint,
        expected_work,
        references,
        expected_head_evaluations=head_evaluations,
    )
    methods = _aggregate(bank, completed, config)
    paired_comparison = _paired_gradient_unembedding_comparison(bank, completed)
    result = _evaluation_result_payload(
        bank,
        capture,
        freeze,
        checkpoint,
        methods,
        paired_comparison,
        elapsed_seconds=time.perf_counter() - started,
    )
    _atomic_json(RESULT_PATH, result)
    print(
        json.dumps(
            {"status": result["status"], "methods": methods, "compute": result["compute"]},
            indent=2,
        ),
        flush=True,
    )
    return result


def _load_validated_result() -> dict[str, Any] | None:
    if not RESULT_PATH.exists():
        return None
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    copy = dict(result)
    embedded = copy.pop("result_sha256", None)
    if result.get("schema_version") != RESULT_SCHEMA or embedded != _canonical_sha256(copy):
        raise RuntimeError("exact-head result has an invalid schema or embedded hash")
    config = _load_config()
    import torch

    old, capture = _load_capture(torch)
    bank = _load_construction(torch)
    _validate_manifest(bank)
    _validate_construction_attempt(bank)
    freeze = _validate_freeze()
    expected_references = {
        "config_sha256": _sha256(CONFIG_PATH),
        "capture_sha256": _sha256(CAPTURE_PATH),
        "construction_sha256": _sha256(CONSTRUCTION_PATH),
        "construction_manifest_sha256": _sha256(CONSTRUCTION_MANIFEST_PATH),
        "freeze_sha256": _sha256(FREEZE_PATH),
    }
    for field, expected in expected_references.items():
        if result.get(field) != expected:
            raise RuntimeError(f"exact-head result reference differs: {field}")
    required = int(config["opened_pass_gate"]["required_eligible_pairs_per_method"])
    construction_failed = any(
        int(count) != required
        for count in bank["summary"]["eligible_pair_count_by_method"].values()
    )
    if construction_failed:
        if result != _failed_construction_payload(bank, config):
            raise RuntimeError("construction-failure result is not fully bank-derived")
        return result
    if result.get("status") == "failed_construction":
        raise RuntimeError("result reports construction failure despite complete eligibility")

    adaptive, _capture_by_key, expected_work, public_plan = _evaluation_work_plan(
        old, capture, bank, config
    )
    del adaptive
    references = {
        **{key: value for key, value in expected_references.items() if key != "capture_sha256"},
        "expected_work_sha256": _canonical_sha256(public_plan),
    }
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    _validate_checkpoint(
        checkpoint,
        expected_work,
        references,
        expected_head_evaluations=int(bank["summary"]["conceptual_exact_head_evaluations"]),
    )
    if checkpoint.get("status") != "complete":
        raise RuntimeError("exact-head report requires a completed evaluation checkpoint")
    if result.get("evaluation_checkpoint_sha256") != _sha256(CHECKPOINT_PATH) or result.get(
        "evaluation_cells_sha256"
    ) != checkpoint.get("cells_sha256"):
        raise RuntimeError("exact-head result differs from the completed checkpoint")
    cells = {str(row["work_id"]): row for row in checkpoint["cells"]}
    expected_methods = _aggregate(bank, cells, config)
    if result.get("methods") != expected_methods:
        raise RuntimeError("exact-head result method aggregate is not checkpoint-derived")
    expected_paired = _paired_gradient_unembedding_comparison(bank, cells)
    if result.get("paired_gradient_unembedding_comparison") != expected_paired:
        raise RuntimeError("exact-head paired comparison is not checkpoint-derived")
    compute = result.get("compute")
    elapsed = (
        compute.get("elapsed_seconds_this_evaluation_invocation")
        if isinstance(compute, Mapping)
        else None
    )
    if type(elapsed) is not float or not math.isfinite(elapsed) or elapsed < 0.0:
        raise RuntimeError("exact-head result elapsed time is invalid")
    expected_result = _evaluation_result_payload(
        bank,
        capture,
        freeze,
        checkpoint,
        expected_methods,
        expected_paired,
        elapsed_seconds=elapsed,
    )
    if result != expected_result:
        raise RuntimeError("exact-head result contains a non-derived field")
    return result


def run_report() -> str:
    result = _load_validated_result()
    lines = [
        "# Interface-equivariant exact-head opened development",
        "",
        "This phase uses already-opened prompts and is not prospective confirmation evidence.",
        "",
    ]
    if result is None:
        lines.append("No finite evaluation result is available.")
    else:
        lines.extend((f"Status: **{result['status']}**.", ""))
        if result["status"] == "failed_construction":
            for method_id, method in result["methods"].items():
                lines.append(
                    f"- `{method_id}`: {method['eligible_pair_count']}/"
                    f"{method['required_eligible_pair_count']} eligible pairs; pass=False."
                )
        else:
            for method_id, method in result["methods"].items():
                lines.append(
                    f"- `{method_id}`: {method['successful_pair_count']}/"
                    f"{method['required_pair_count']} pairs; "
                    f"{method['successful_cell_count']}/{method['required_cell_count']} "
                    f"target cells; {method['real_decision_change_count']} real decision "
                    f"changes; pass={method['passes']}."
                )
            comparison = result["paired_gradient_unembedding_comparison"]
            lines.extend(
                (
                    "",
                    "Paired gradient-versus-output-boundary comparison:",
                    "",
                    (
                        f"- Direction cosine range: `{comparison['minimum_direction_cosine']}` "
                        f"to `{comparison['maximum_direction_cosine']}`."
                    ),
                    (
                        f"- Mean gradient-minus-boundary alpha: "
                        f"`{comparison['mean_gradient_minus_unembedding_alpha']}`."
                    ),
                    (
                        f"- Mean gradient-minus-boundary relative norm: "
                        f"`{comparison['mean_gradient_minus_unembedding_relative_norm']}`."
                    ),
                    (
                        f"- Mean gradient-minus-boundary KL: "
                        f"`{comparison['mean_gradient_minus_unembedding_kl']}`."
                    ),
                    "",
                    "Attribution control:",
                    "",
                    (
                        f"- Minimum captured-gradient versus analytic RMS-tangent cosine: "
                        f"`{result['attribution']['minimum_captured_to_analytic_gradient_cosine']}`."
                    ),
                    (
                        f"- Maximum relative L2 error: "
                        f"`{result['attribution']['maximum_captured_to_analytic_gradient_relative_l2_error']}`."
                    ),
                    (
                        f"- Byte-identical gradient/tangent delta banks: "
                        f"{result['attribution']['gradient_delta_exactly_matches_analytic_tangent_pair_count']}/16."
                    ),
                    "",
                    (
                        "Interpretation: a technical pass demonstrates local single-token endpoint "
                        "control. The block-23 gradient is analytically determined by the current "
                        "residual plus the final RMSNorm/unembedding boundary, so it is not "
                        "independent evidence for a self-preservation representation."
                    ),
                )
            )
    lines.extend(
        (
            "",
            "No generated text, external API, model judge, sealed test, J-space analysis, or 2B run was used.",
            "",
        )
    )
    text = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("construct", "freeze", "evaluate", "report"))
    arguments = parser.parse_args()
    {
        "construct": run_construct,
        "freeze": run_freeze,
        "evaluate": run_evaluate,
        "report": run_report,
    }[arguments.phase]()


if __name__ == "__main__":
    main()
