#!/usr/bin/env python3
"""Lock, capture, and analyze opened-development CSMS geometry.

No command in this runner accepts a sealed split.  ``capture`` is the only
model-compute command (80 forwards plus 80 backwards); it refuses to start
without a final immutable lock.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sp_lense.counterfactual_kl_prompts import render_ckes_forms
from sp_lense.counterfactual_slot_matrix_steering import (
    CAPTURE_SCHEMA_VERSION,
    CSMSIntegrityError,
    analyze_csms_geometry,
    build_capture_alignment_manifest,
    capture_slot_matrix_baseline,
    require_opened_development_split,
    resolve_first_content_index,
    resolve_slot_indices,
)
from sp_lense.decision_margin_shield import DEFAULT_SVD_ATOL, DEFAULT_SVD_RTOL
from sp_lense.decision_margin_shield_rowspace import (
    SOLVER_FUNCTION_TOLERANCE,
    SOLVER_MAX_ITERATIONS,
    SOLVER_METHOD,
    STRICT_RAW_FEASIBILITY_FRACTION,
    STRICT_RAW_ROUNDOFF_MULTIPLIER,
)
from sp_lense.factorial_causal_anchor import canonical_sha256, tensor_float32_sha256

LOCK_SCHEMA = "sp_lense.counterfactual_slot_matrix_steering_lock.v1"
RESULT_SCHEMA = "sp_lense.counterfactual_slot_matrix_steering_result.v1"
LOCK_STATUS = "adaptive_opened_development_locked_before_csms_model_compute"

LOCK_PATH = ROOT / "configs" / "counterfactual_slot_matrix_steering_lock.json"
TEMPLATE_PATH = ROOT / "configs" / "counterfactual_slot_matrix_steering_template.json"
PROTOCOL_PATH = ROOT / "docs" / "COUNTERFACTUAL_SLOT_MATRIX_STEERING_PROTOCOL.md"
DATA_PATH = ROOT / "data" / "ckes_v2_validation.json"
V2_RUNNER_PATH = ROOT / "scripts" / "counterfactual_kl_extragradient_baseline_relative.py"
V2_LOCK_PATH = ROOT / "configs" / "counterfactual_kl_extragradient_baseline_relative_lock.json"
V2_RESULT_PATH = (
    ROOT
    / "results"
    / "counterfactual_kl_extragradient_baseline_relative"
    / "qwen35_08b"
    / "validation"
    / "result.json"
)
V2_STATE0_PATH = V2_RESULT_PATH.parent / "state0_baseline.pt"
V2_TOKENIZER_PATH = V2_RESULT_PATH.parent / "tokenizer_preflight.json"
RESULT_ROOT = ROOT / "results" / "counterfactual_slot_matrix_steering" / "qwen35_08b"
CAPTURE_PATH = RESULT_ROOT / "capture.pt"
CAPTURE_RESERVATION_PATH = RESULT_ROOT / "capture_reservation.json"
CAPTURE_COMPLETE_PATH = RESULT_ROOT / "capture_complete.json"
GEOMETRY_PATH = RESULT_ROOT / "geometry.json"
DIRECTION_PATH = RESULT_ROOT / "directions.pt"
REPORT_PATH = RESULT_ROOT / "REPORT.md"

MODEL = {
    "id": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "device": "cpu",
    "dtype": "float32",
    "n_layers": 24,
    "d_model": 1024,
}
V2_LOCK_IDENTITY = "1f2ab54f4799d03089cb38100b124af2f168df75d5932666ebb3a656e0ea39d3"
V2_RESULT_IDENTITY = "cb8ce95ec61dcb6379d9a7a6f4ead56c0680e3e8824b0d90e39e690b40c8e524"

LOCKED_PATHS = (
    Path("src/sp_lense/counterfactual_slot_matrix_steering.py"),
    Path("scripts/counterfactual_slot_matrix_steering.py"),
    Path("tests/test_counterfactual_slot_matrix_steering.py"),
    Path("tests/test_counterfactual_slot_matrix_steering_runner.py"),
    Path("configs/counterfactual_slot_matrix_steering_template.json"),
    Path("docs/COUNTERFACTUAL_SLOT_MATRIX_STEERING_PROTOCOL.md"),
    Path("src/sp_lense/counterfactual_kl_prompts.py"),
    Path("src/sp_lense/counterfactual_kl_runtime.py"),
    Path("src/sp_lense/comparison_runtime.py"),
    Path("src/sp_lense/factorial_causal_anchor.py"),
    Path("src/sp_lense/counterfactual_tangent_shield.py"),
    Path("src/sp_lense/decision_margin_shield.py"),
    Path("src/sp_lense/decision_margin_shield_rowspace.py"),
    Path("scripts/counterfactual_kl_extragradient_baseline_relative.py"),
    Path("scripts/counterfactual_kl_extragradient_development.py"),
    Path("scripts/closed_loop_dms_development.py"),
    Path("scripts/factorial_causal_anchor_gradient_pilot.py"),
    Path("configs/qwen35_08b_aligned.json"),
    Path("data/ckes_v2_validation.json"),
    Path("configs/counterfactual_kl_extragradient_baseline_relative_lock.json"),
    Path(
        "results/counterfactual_kl_extragradient_baseline_relative/"
        "qwen35_08b/validation/result.json"
    ),
    Path(
        "results/counterfactual_kl_extragradient_baseline_relative/"
        "qwen35_08b/validation/state0_baseline.pt"
    ),
    Path(
        "results/counterfactual_kl_extragradient_baseline_relative/"
        "qwen35_08b/validation/tokenizer_preflight.json"
    ),
    Path("pyproject.toml"),
    Path("requirements-research.txt"),
    Path("requirements-constrained-steering.txt"),
)

ADAPTIVE_EVIDENCE_PATHS = (
    Path("configs/counterfactual_kl_extragradient_development_lock.json"),
    Path("results/counterfactual_kl_extragradient/qwen35_08b/validation/result.json"),
    Path("docs/CONTEXT_GATED_DYNAMIC_PROTOCOL.md"),
    Path("results/context_gated_dynamic/qwen35_08b/exact_prompt_order_summary.json"),
    Path("results/context_gated_dynamic/qwen35_08b/gated_replay_summary.json"),
    Path("results/context_gated_bidirectional/qwen35_08b/validation_summary.json"),
    Path("results/context_gated_bidirectional/qwen35_08b/validation_freeze.json"),
    Path("results/context_gated_bidirectional/qwen35_08b/sealed_summary.json"),
    Path(
        "results/semantic_context_gate_development/"
        "counterfactual_name_order_cancelled_v3/qwen35_08b/semantic_gate_result.json"
    ),
    Path(
        "results/semantic_context_gate_development/"
        "counterfactual_name_order_cancelled_v3/qwen35_08b/steering_development_result.json"
    ),
    Path(
        "results/learned_context_gated_gradient_development/"
        "fresh_confirmation_v1/qwen35_08b/gate_capture_manifest.json"
    ),
    Path(
        "results/learned_context_gated_gradient_development/"
        "fresh_confirmation_v1/qwen35_08b/gate_development_result.json"
    ),
    Path("docs/PFIT_POSTHOC_ATTAINABILITY.md"),
    Path(
        "results/factorial_interface_translator_development/"
        "qwen35_08b/geometric_result.json"
    ),
    Path(
        "results/factorial_interface_translator_development/"
        "qwen35_08b/posthoc_attainability.json"
    ),
    Path("docs/GLOBAL_COUNTERFACTUAL_ROBUST_BOUNDARY_PROTOCOL.md"),
    Path(
        "results/global_counterfactual_robust_boundary/"
        "qwen35_08b/all_layer_geometry.json"
    ),
    Path(
        "results/global_counterfactual_robust_boundary/"
        "qwen35_08b/integrated_conclusion.json"
    ),
)

ADAPTIVE_EVIDENCE_SPECS: dict[str, dict[str, str | None]] = {
    "configs/counterfactual_kl_extragradient_development_lock.json": {
        "phase": "ckes_v1",
        "self_identity_field": "lock_identity_sha256",
        "limitation": "prospective lock only; it is not an outcome",
    },
    "results/counterfactual_kl_extragradient/qwen35_08b/validation/result.json": {
        "phase": "ckes_v1",
        "self_identity_field": "result_sha256",
        "limitation": "no-go after 80 forward/backward captures and before any nonzero intervention",
    },
    "docs/CONTEXT_GATED_DYNAMIC_PROTOCOL.md": {
        "phase": "context_gated_dynamic",
        "self_identity_field": None,
        "limitation": "protocol document only; not an outcome artifact",
    },
    "results/context_gated_dynamic/qwen35_08b/exact_prompt_order_summary.json": {
        "phase": "context_gated_dynamic",
        "self_identity_field": None,
        "limitation": "opened posthoc gated development with strict validation failure; not the current ungated endpoint",
    },
    "results/context_gated_dynamic/qwen35_08b/gated_replay_summary.json": {
        "phase": "context_gated_dynamic",
        "self_identity_field": None,
        "limitation": "posthoc gated replay; not a prospective ungated confirmation",
    },
    "results/context_gated_bidirectional/qwen35_08b/validation_summary.json": {
        "phase": "context_gated_bidirectional",
        "self_identity_field": None,
        "limitation": "prior gate-dependent endpoint does not satisfy the current universal exact-null endpoint",
    },
    "results/context_gated_bidirectional/qwen35_08b/validation_freeze.json": {
        "phase": "context_gated_bidirectional",
        "self_identity_field": None,
        "limitation": "freeze metadata is not itself an effect estimate",
    },
    "results/context_gated_bidirectional/qwen35_08b/sealed_summary.json": {
        "phase": "context_gated_bidirectional",
        "self_identity_field": None,
        "limitation": "previously reported gated result; CSMS neither opens nor reuses its sealed prompts",
    },
    "results/semantic_context_gate_development/counterfactual_name_order_cancelled_v3/qwen35_08b/semantic_gate_result.json": {
        "phase": "semantic_gate_v3",
        "self_identity_field": "result_sha256",
        "limitation": "gate-selection development; not evidence for one ungated universal matrix",
    },
    "results/semantic_context_gate_development/counterfactual_name_order_cancelled_v3/qwen35_08b/steering_development_result.json": {
        "phase": "semantic_gate_v3",
        "self_identity_field": "result_sha256",
        "limitation": "opened gate-dependent steering development; not the current strict endpoint",
    },
    "results/learned_context_gated_gradient_development/fresh_confirmation_v1/qwen35_08b/gate_capture_manifest.json": {
        "phase": "learned_gate_fresh_confirmation",
        "self_identity_field": None,
        "limitation": "capture manifest is provenance, not a successful endpoint",
    },
    "results/learned_context_gated_gradient_development/fresh_confirmation_v1/qwen35_08b/gate_development_result.json": {
        "phase": "learned_gate_fresh_confirmation",
        "self_identity_field": "result_sha256",
        "limitation": "fresh gate confirmation failed and is not evidence for CSMS qualification",
    },
    "docs/PFIT_POSTHOC_ATTAINABILITY.md": {
        "phase": "pfit",
        "self_identity_field": None,
        "limitation": "posthoc analysis document only; not a prospective result",
    },
    "results/factorial_interface_translator_development/qwen35_08b/geometric_result.json": {
        "phase": "pfit",
        "self_identity_field": "result_sha256",
        "limitation": "failed opened geometric result under a different parameterization",
    },
    "results/factorial_interface_translator_development/qwen35_08b/posthoc_attainability.json": {
        "phase": "pfit",
        "self_identity_field": "posthoc_result_sha256",
        "limitation": "failed posthoc attainability analysis; nonconfirmatory",
    },
    "docs/GLOBAL_COUNTERFACTUAL_ROBUST_BOUNDARY_PROTOCOL.md": {
        "phase": "gcrbs",
        "self_identity_field": None,
        "limitation": "protocol document only; not an outcome artifact",
    },
    "results/global_counterfactual_robust_boundary/qwen35_08b/all_layer_geometry.json": {
        "phase": "gcrbs",
        "self_identity_field": "geometry_sha256",
        "limitation": "opened all-layer geometry was negative and used a different intervention family",
    },
    "results/global_counterfactual_robust_boundary/qwen35_08b/integrated_conclusion.json": {
        "phase": "gcrbs",
        "self_identity_field": None,
        "limitation": "negative development conclusion; not a CSMS strict-endpoint pass",
    },
}

OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}

_V2: ModuleType | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def _bound_path(raw: str) -> Path:
    result = (ROOT / raw).resolve()
    result.relative_to(ROOT.resolve())
    return result


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{_relative(path)} must contain one JSON object")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _with_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = {str(key): _plain(item) for key, item in value.items() if key != field}
    result[field] = canonical_sha256(result)
    return result


def _verify_hash(value: Mapping[str, Any], field: str) -> None:
    body = {str(key): _plain(item) for key, item in value.items() if key != field}
    if value.get(field) != canonical_sha256(body):
        raise CSMSIntegrityError(f"{field} differs")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable {_relative(path)}")
    _atomic_text(path, json.dumps(_plain(value), indent=2, ensure_ascii=False) + "\n")


def _v2() -> ModuleType:
    global _V2
    if _V2 is None:
        specification = importlib.util.spec_from_file_location("sp_lense_csms_v2", V2_RUNNER_PATH)
        if specification is None or specification.loader is None:
            raise RuntimeError("cannot load immutable CKES v2 runner")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        _V2 = module
    return _V2


def _base() -> ModuleType:
    return _v2()._base()


def _forms(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rendered = render_ckes_forms(payload, expected_split="validation")
    rows = [
        *rendered["scenario"],
        *rendered["calibration_unrelated"],
        *rendered["nuisance_fit"],
    ]
    if len(rows) != 80 or len({row["form_id"] for row in rows}) != 80:
        raise CSMSIntegrityError("opened CKES v2 rendered coverage differs")
    return rows


def _source_checkpoint(torch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata, tensors = _base()._load_checkpoint(
        torch,
        path=V2_STATE0_PATH,
        schema=_base().BASELINE_SCHEMA,
    )
    if (
        metadata.get("lock_identity_sha256") != V2_LOCK_IDENTITY
        or metadata.get("record_count") != 80
        or tuple(tensors["raw_gradients"].shape) != (80, 1024)
        or tuple(tensors["pre_anchor_residuals"].shape) != (80, 1024)
        or tuple(tensors["full_logits"].shape[:1]) != (80,)
    ):
        raise CSMSIntegrityError("immutable CKES v2 state-zero checkpoint differs")
    return metadata, tensors


def _verify_opened_source() -> dict[str, Any]:
    import torch

    v2_lock = _load_json(V2_LOCK_PATH)
    v2_result = _load_json(V2_RESULT_PATH)
    _verify_hash(v2_result, "result_sha256")
    if (
        v2_lock.get("lock_identity_sha256") != V2_LOCK_IDENTITY
        or v2_result.get("lock_identity_sha256") != V2_LOCK_IDENTITY
        or v2_result.get("result_sha256") != V2_RESULT_IDENTITY
        or v2_result.get("status") != "no_go"
        or v2_result.get("split") != "validation"
    ):
        raise CSMSIntegrityError("CKES v2 source lock/result identity differs")
    template = _load_json(TEMPLATE_PATH)
    dataset = _load_json(DATA_PATH)
    forms = _forms(dataset)
    state, tensors = _source_checkpoint(torch)
    tokenizer = _load_json(V2_TOKENIZER_PATH)
    _verify_hash(tokenizer, "tokenizer_preflight_sha256")
    source_records = state.get("records")
    tokenizer_records = tokenizer.get("records")
    if (
        not isinstance(source_records, list)
        or not isinstance(tokenizer_records, list)
        or [row["form_id"] for row in source_records] != [row["form_id"] for row in forms]
        or [row["form_id"] for row in tokenizer_records] != [row["form_id"] for row in forms]
    ):
        raise CSMSIntegrityError("dataset/state-zero/tokenizer row order differs")
    for index, (form, source, token_row) in enumerate(
        zip(forms, source_records, tokenizer_records, strict=True)
    ):
        if (
            source.get("tensor_index") != index
            or source.get("form") != form
            or token_row.get("prompt_sha256") != form.get("prompt_sha256")
            or token_row.get("prompt_token_ids_sha256")
            != source.get("prompt_token_ids_sha256")
            or tensor_float32_sha256(tensors["pre_anchor_residuals"][index])
            != source.get("pre_anchor_residual_float32_sha256")
            or tensor_float32_sha256(tensors["raw_gradients"][index])
            != source.get("raw_anchor_gradient_float32_sha256")
            or tensor_float32_sha256(tensors["full_logits"][index])
            != source.get("full_logits_float32_sha256")
        ):
            raise CSMSIntegrityError("one immutable source row or tensor binding differs")
    compute = v2_result.get("compute", {})
    if compute.get("forward_backward") != 488:
        raise CSMSIntegrityError("CKES v2 adaptive compute provenance differs")
    return {
        "template_sha256": file_sha256(TEMPLATE_PATH),
        "dataset_file_sha256": file_sha256(DATA_PATH),
        "v2_lock_file_sha256": file_sha256(V2_LOCK_PATH),
        "v2_lock_identity_sha256": V2_LOCK_IDENTITY,
        "v2_result_file_sha256": file_sha256(V2_RESULT_PATH),
        "v2_result_sha256": V2_RESULT_IDENTITY,
        "v2_state0_file_sha256": file_sha256(V2_STATE0_PATH),
        "v2_state0_checkpoint_sha256": state["checkpoint_sha256"],
        "v2_tokenizer_file_sha256": file_sha256(V2_TOKENIZER_PATH),
        "v2_tokenizer_preflight_sha256": tokenizer["tokenizer_preflight_sha256"],
        "rendered_form_ids_sha256": canonical_sha256([row["form_id"] for row in forms]),
        "source_tensor_layout_sha256": state["tensor_layout_sha256"],
        "source_verification_used_model_compute": False,
        "source_full_logits_loaded_for_hash_verification_only": True,
        "template": template,
    }


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _adaptive_compute_provenance(payload: Mapping[str, Any] | None) -> Any:
    if payload is None:
        return None
    fields = (
        "compute",
        "elapsed_seconds",
        "model_forwards",
        "generated_tokens",
        "new_forward_evaluations",
        "total_forward_evaluations",
        "elapsed_seconds_this_run",
        "external_cost_usd",
        "new_model_calls",
        "external_api_calls",
        "external_model_judges",
        "paid_cost_usd",
        "forward_evaluations",
        "backward_evaluations",
        "finite_model_scoring_performed",
        "new_compute",
    )
    observed = {field: _plain(payload[field]) for field in fields if field in payload}
    return observed or None


def _adaptive_evidence_manifest() -> dict[str, Any]:
    expected_paths = tuple(_relative(ROOT / path) for path in ADAPTIVE_EVIDENCE_PATHS)
    if set(expected_paths) != set(ADAPTIVE_EVIDENCE_SPECS) or len(expected_paths) != len(
        ADAPTIVE_EVIDENCE_SPECS
    ):
        raise CSMSIntegrityError("adaptive evidence specification coverage differs")
    records: list[dict[str, Any]] = []
    for raw_path in expected_paths:
        path = ROOT / raw_path
        specification = ADAPTIVE_EVIDENCE_SPECS[raw_path]
        payload = _load_json(path) if path.suffix.lower() == ".json" else None
        identity_field = specification["self_identity_field"]
        identity = payload.get(identity_field) if payload is not None and identity_field else None
        if identity_field is not None and not _is_sha256(identity):
            raise CSMSIntegrityError(
                f"adaptive evidence self identity differs: {raw_path}"
            )
        embedded_hashes = (
            {
                str(key): value
                for key, value in sorted(payload.items())
                if key.endswith("sha256") and _is_sha256(value)
            }
            if payload is not None
            else {}
        )
        record = {
            **_file_record(path),
            "phase": specification["phase"],
            "schema_version": payload.get("schema_version") if payload else None,
            "status": payload.get("status") if payload else None,
            "self_identity_field": identity_field,
            "self_identity_sha256": identity,
            "top_level_embedded_sha256_fields": embedded_hashes,
            "compute_provenance": _adaptive_compute_provenance(payload),
            "limitation": specification["limitation"],
            "satisfies_current_csms_strict_endpoint": False,
        }
        record["record_sha256"] = canonical_sha256(record)
        records.append(record)
    value = {
        "record_count": len(records),
        "paths_sha256": canonical_sha256(expected_paths),
        "records": records,
        "all_prior_evidence_is_adaptive_or_nonconfirmatory_for_csms": True,
        "no_prior_gated_pass_satisfies_the_current_ungated_strict_endpoint": True,
        "current_csms_strict_endpoint_satisfied_by_prior_evidence": False,
    }
    value["manifest_sha256"] = canonical_sha256(value)
    return value


def _expected_numerical_backend() -> dict[str, Any]:
    return {
        "solver": SOLVER_METHOD,
        "solver_max_iterations": SOLVER_MAX_ITERATIONS,
        "solver_function_tolerance": SOLVER_FUNCTION_TOLERANCE,
        "svd_rtol": DEFAULT_SVD_RTOL,
        "svd_atol": DEFAULT_SVD_ATOL,
        "strict_raw_feasibility_fraction": STRICT_RAW_FEASIBILITY_FRACTION,
        "strict_raw_roundoff_multiplier": STRICT_RAW_ROUNDOFF_MULTIPLIER,
        "random_exact_null_seeds": [1729, 2718, 3141, 5772],
    }


def _expected_thresholds() -> dict[str, Any]:
    return {
        "target_margin": 0.05,
        "cap_frontier": [0.1, 0.25, 0.5, 1.0, 2.0],
        "qualification_cap": 0.25,
        "realized_standardized_frobenius_norm_maximum": 0.25,
        "held_out_leakage_ratio_maximum": 0.50,
        "held_out_absolute_non_target_movement_maximum": 0.05,
        "held_out_scalar_gate_tolerance": 0.0,
        "zero_held_out_non_target_choice_flips_both_signs": True,
        "fold_scales_training_only": True,
        "bidirectional_float32_recertification_required": True,
        "double_certificate_tolerance": 1e-8,
        "float32_physical_tolerance": 1e-6,
        "norm_and_dose_caps_use_no_certificate_tolerance": True,
        "all_four_cross_fit_folds_required": True,
        "finite_failure_action": "no_finite_intervention_authorized",
    }


def _expected_compute_ceiling() -> dict[str, Any]:
    return {
        "model_forwards": 80,
        "model_backwards": 80,
        "generated_tokens": 0,
        "external_api_calls": 0,
        "external_model_judges": 0,
        "paid_model_cost_usd": 0,
    }


def proposed_lock() -> dict[str, Any]:
    source = _verify_opened_source()
    v2_lock = _load_json(V2_LOCK_PATH)
    configuration = _load_json(TEMPLATE_PATH)
    configuration = {
        **configuration,
        "status": LOCK_STATUS,
        "source_binding": source,
        "pinned_runtime": v2_lock["configuration"]["runtime"],
        "chat_template_sha256": v2_lock["configuration"]["chat_template_sha256"],
        "numerical_backend": _expected_numerical_backend(),
        "adaptive_evidence_manifest": _adaptive_evidence_manifest(),
        "lock_creation_model_compute": {
            "model_forwards": 0,
            "model_backwards": 0,
            "generated_tokens": 0,
            "external_api_calls": 0,
            "external_model_judges": 0,
            "paid_model_cost_usd": 0,
        },
        "local_only_environment_required_before_backend_load": OFFLINE_ENVIRONMENT,
    }
    lock = {
        "schema_version": LOCK_SCHEMA,
        "status": LOCK_STATUS,
        "file_hashes": {
            f"locked_{index:02d}": _file_record(ROOT / path)
            for index, path in enumerate(LOCKED_PATHS)
        },
        "configuration": configuration,
        "thresholds": _expected_thresholds(),
        "compute_ceiling": _expected_compute_ceiling(),
        "sealed_access": {
            "permitted": False,
            "accepted_split": "opened_development",
            "sealed_dataset_path_recorded": False,
            "sealed_bytes_must_never_be_read": True,
        },
    }
    lock["lock_identity_sha256"] = canonical_sha256(lock)
    return verify_lock(lock, verify_files=True)


def verify_lock(value: Mapping[str, Any], *, verify_files: bool) -> dict[str, Any]:
    lock = _plain(value)
    identity = lock.pop("lock_identity_sha256", None)
    if identity != canonical_sha256(lock):
        raise CSMSIntegrityError("CSMS lock identity differs")
    lock["lock_identity_sha256"] = identity
    expected_sealed = {
        "permitted": False,
        "accepted_split": "opened_development",
        "sealed_dataset_path_recorded": False,
        "sealed_bytes_must_never_be_read": True,
    }
    if (
        set(lock)
        != {
            "schema_version",
            "status",
            "file_hashes",
            "configuration",
            "thresholds",
            "compute_ceiling",
            "sealed_access",
            "lock_identity_sha256",
        }
        or
        lock.get("schema_version") != LOCK_SCHEMA
        or lock.get("status") != LOCK_STATUS
        or lock.get("thresholds") != _expected_thresholds()
        or lock.get("compute_ceiling") != _expected_compute_ceiling()
        or lock.get("sealed_access") != expected_sealed
    ):
        raise CSMSIntegrityError("CSMS lock contract differs")

    records = lock.get("file_hashes")
    expected_file_keys = [f"locked_{index:02d}" for index in range(len(LOCKED_PATHS))]
    expected_file_paths = [_relative(ROOT / path) for path in LOCKED_PATHS]
    if (
        not isinstance(records, Mapping)
        or list(records) != expected_file_keys
        or [record.get("path") for record in records.values()] != expected_file_paths
        or len(set(expected_file_paths)) != len(expected_file_paths)
    ):
        raise CSMSIntegrityError("CSMS locked file coverage differs")
    for record in records.values():
        if set(record) != {"path", "bytes", "sha256"} or not _is_sha256(
            record.get("sha256")
        ):
            raise CSMSIntegrityError("one CSMS locked file record differs")
        if verify_files:
            path = _bound_path(str(record["path"]))
            if (
                not path.is_file()
                or path.stat().st_size != record.get("bytes")
                or file_sha256(path) != record.get("sha256")
            ):
                raise CSMSIntegrityError(f"locked source bytes differ: {record['path']}")

    configuration = lock.get("configuration")
    template = _load_json(TEMPLATE_PATH)
    locked_template = {**template, "status": LOCK_STATUS}
    expected_adaptive_paths = [
        _relative(ROOT / path) for path in ADAPTIVE_EVIDENCE_PATHS
    ]
    if (
        template.get("adaptive_provenance", {}).get(
            "structured_manifest_exact_paths"
        )
        != expected_adaptive_paths
    ):
        raise CSMSIntegrityError("CSMS template adaptive evidence coverage differs")
    v2_lock = _load_json(V2_LOCK_PATH)
    added_configuration_keys = {
        "status",
        "source_binding",
        "pinned_runtime",
        "chat_template_sha256",
        "numerical_backend",
        "adaptive_evidence_manifest",
        "lock_creation_model_compute",
        "local_only_environment_required_before_backend_load",
    }
    if not isinstance(configuration, Mapping) or set(configuration) != (
        set(template) | added_configuration_keys
    ):
        raise CSMSIntegrityError("CSMS lock configuration coverage differs")
    if any(configuration.get(key) != item for key, item in locked_template.items()):
        raise CSMSIntegrityError("CSMS locked template configuration differs")
    zero_compute = {
        **_expected_compute_ceiling(),
        "model_forwards": 0,
        "model_backwards": 0,
    }
    if (
        configuration.get("status") != LOCK_STATUS
        or configuration.get("pinned_runtime")
        != v2_lock["configuration"]["runtime"]
        or configuration.get("chat_template_sha256")
        != v2_lock["configuration"]["chat_template_sha256"]
        or configuration.get("numerical_backend") != _expected_numerical_backend()
        or configuration.get("adaptive_evidence_manifest")
        != _adaptive_evidence_manifest()
        or configuration.get("lock_creation_model_compute") != zero_compute
        or configuration.get("local_only_environment_required_before_backend_load")
        != OFFLINE_ENVIRONMENT
    ):
        raise CSMSIntegrityError("CSMS lock configuration differs")

    source = configuration.get("source_binding")
    source_keys = {
        "template_sha256",
        "dataset_file_sha256",
        "v2_lock_file_sha256",
        "v2_lock_identity_sha256",
        "v2_result_file_sha256",
        "v2_result_sha256",
        "v2_state0_file_sha256",
        "v2_state0_checkpoint_sha256",
        "v2_tokenizer_file_sha256",
        "v2_tokenizer_preflight_sha256",
        "rendered_form_ids_sha256",
        "source_tensor_layout_sha256",
        "source_verification_used_model_compute",
        "source_full_logits_loaded_for_hash_verification_only",
        "template",
    }
    if (
        not isinstance(source, Mapping)
        or set(source) != source_keys
        or source.get("template") != template
        or source.get("template_sha256") != file_sha256(TEMPLATE_PATH)
        or source.get("dataset_file_sha256") != file_sha256(DATA_PATH)
        or source.get("v2_lock_file_sha256") != file_sha256(V2_LOCK_PATH)
        or source.get("v2_lock_identity_sha256") != V2_LOCK_IDENTITY
        or source.get("v2_result_file_sha256") != file_sha256(V2_RESULT_PATH)
        or source.get("v2_result_sha256") != V2_RESULT_IDENTITY
        or source.get("v2_tokenizer_file_sha256") != file_sha256(V2_TOKENIZER_PATH)
        or source.get("source_verification_used_model_compute") is not False
        or source.get("source_full_logits_loaded_for_hash_verification_only") is not True
        or any(
            not _is_sha256(source.get(key))
            for key in (
                "v2_state0_file_sha256",
                "v2_state0_checkpoint_sha256",
                "v2_tokenizer_preflight_sha256",
                "rendered_form_ids_sha256",
                "source_tensor_layout_sha256",
            )
        )
    ):
        raise CSMSIntegrityError("CSMS opened source identity differs")
    if verify_files and source != _verify_opened_source():
        raise CSMSIntegrityError("CSMS opened source binding differs")
    return lock


def _load_lock() -> dict[str, Any]:
    if not LOCK_PATH.is_file():
        raise FileNotFoundError("CSMS capture/analyze requires the reviewed final lock")
    return verify_lock(_load_json(LOCK_PATH), verify_files=True)


def run_lock(split: str = "opened_development") -> dict[str, Any]:
    require_opened_development_split(split)
    if LOCK_PATH.exists():
        raise FileExistsError("refusing to overwrite the immutable CSMS lock")
    value = proposed_lock()
    _write_new_json(LOCK_PATH, value)
    return value


def run_preflight(split: str = "opened_development") -> dict[str, Any]:
    require_opened_development_split(split)
    lock = _load_lock() if LOCK_PATH.exists() else proposed_lock()
    source = _verify_opened_source()
    return _with_hash(
        {
            "schema_version": "sp_lense.counterfactual_slot_matrix_steering_preflight.v1",
            "status": "locked_ready" if LOCK_PATH.exists() else "proposal_ready_not_locked",
            "split": split,
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "source_binding": source,
            "model_compute": {
                "model_forwards": 0,
                "model_backwards": 0,
                "generated_tokens": 0,
                "external_api_calls": 0,
                "external_model_judges": 0,
                "paid_model_cost_usd": 0,
            },
            "next_expensive_phase": {
                "command": "capture",
                "model_forwards": 80,
                "model_backwards": 80,
                "generated_tokens": 0,
                "paid_model_cost_usd": 0,
            },
        },
        "preflight_sha256",
    )


def _chat_header_ids(backend: Any, prompt: str) -> tuple[int, ...]:
    tokenizer = backend.model.tokenizer
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str) or rendered.count(prompt) != 1:
        raise CSMSIntegrityError("cannot isolate the exact user-content chat header")
    header_text = rendered.split(prompt, 1)[0]
    ids = tokenizer.encode(header_text, add_special_tokens=False)
    if not isinstance(ids, Sequence) or not ids:
        raise CSMSIntegrityError("chat header tokenization is empty")
    return tuple(int(value) for value in ids)


def _slot_evidence(
    backend: Any,
    forms: Sequence[Mapping[str, Any]],
    tokenizer_records: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[int, int, int, int]]:
    by_prefix: dict[str, list[int]] = {}
    for index, form in enumerate(forms):
        by_prefix.setdefault(str(form["anchor_prefix_sha256"]), []).append(index)
    if len(by_prefix) != 40 or any(len(values) != 2 for values in by_prefix.values()):
        raise CSMSIntegrityError("answer-order twin grouping differs")
    special = tuple(int(value) for value in backend.model.tokenizer.all_special_ids)
    result: dict[str, tuple[int, int, int, int]] = {}
    for indices in by_prefix.values():
        first_index, second_index = indices
        first_form = forms[first_index]
        second_form = forms[second_index]
        first_tokens = tuple(int(value) for value in backend.encode(first_form["prompt"])[0].tolist())
        second_tokens = tuple(
            int(value) for value in backend.encode(second_form["prompt"])[0].tolist()
        )
        first_content = resolve_first_content_index(
            first_tokens, _chat_header_ids(backend, str(first_form["prompt"]))
        )
        anchors = {
            int(tokenizer_records[first_index]["anchor_index"]),
            int(tokenizer_records[second_index]["anchor_index"]),
        }
        if len(anchors) != 1:
            raise CSMSIntegrityError("answer-order twin anchor indices differ")
        anchor = next(iter(anchors))
        slots = resolve_slot_indices(
            first_content_index=first_content,
            anchor_index=anchor,
            prompt_token_ids=first_tokens,
            answer_order_twin_token_ids=second_tokens,
            special_token_ids=special,
            answer_suffix_start_index=anchor + 1,
        )
        for index in indices:
            result[str(forms[index]["form_id"])] = slots
    if len(result) != 80:
        raise CSMSIntegrityError("CSMS slot evidence coverage differs")
    return result


def run_capture(split: str = "opened_development") -> dict[str, Any]:
    require_opened_development_split(split)
    lock = _load_lock()
    if CAPTURE_PATH.exists() or CAPTURE_COMPLETE_PATH.exists():
        raise FileExistsError("refusing to overwrite immutable CSMS capture")
    if CAPTURE_RESERVATION_PATH.exists():
        raise CSMSIntegrityError(
            "a prior CSMS capture reservation exists without a validated capture; manual audit required"
        )
    required_offline = OFFLINE_ENVIRONMENT
    if {name: os.environ.get(name) for name in required_offline} != required_offline:
        raise CSMSIntegrityError(
            "CSMS capture requires HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, "
            "and HF_DATASETS_OFFLINE=1 before backend load"
        )
    import torch

    state, _source_tensors = _source_checkpoint(torch)
    del _source_tensors
    tokenizer = _load_json(V2_TOKENIZER_PATH)
    forms = _forms(_load_json(DATA_PATH))
    source_records = state["records"]
    tokenizer_records = tokenizer["records"]
    original = _base()._base()._finite()._load_original_runner()
    original._configure_threads(torch)
    backend = original.load_backend()
    metadata = backend.metadata()
    observed_model = {
        "id": metadata["model_id"],
        "revision": metadata["model_revision"],
        "device": metadata["device"],
        "dtype": metadata["dtype"],
        "n_layers": metadata["model_layers"],
        "d_model": metadata["d_model"],
    }
    if observed_model != MODEL:
        raise CSMSIntegrityError("resident backend differs from locked CSMS model")
    slots_by_form = _slot_evidence(backend, forms, tokenizer_records)
    reservation = _with_hash(
        {
            "schema_version": "sp_lense.counterfactual_slot_matrix_steering_reservation.v1",
            "status": "reserved_before_first_model_forward",
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "model_forwards": 80,
            "model_backwards": 80,
            "generated_tokens": 0,
            "external_api_calls": 0,
            "external_model_judges": 0,
            "paid_model_cost_usd": 0,
            "offline_environment": required_offline,
        },
        "reservation_sha256",
    )
    _write_new_json(CAPTURE_RESERVATION_PATH, reservation)
    residuals = []
    gradients = []
    capture_records = []
    try:
        for index, (form, source, token_row) in enumerate(
            zip(forms, source_records, tokenizer_records, strict=True)
        ):
            capture = capture_slot_matrix_baseline(
                backend,
                str(form["prompt"]),
                str(form["positive_label"]),
                str(form["negative_label"]),
                positive_semantic=str(form["positive_semantic"]),
                negative_semantic=str(form["negative_semantic"]),
                layer=0,
                slot_indices=slots_by_form[str(form["form_id"])],
                expected_prompt_sha256=str(form["prompt_sha256"]),
                expected_choice_boundary_evidence_sha256=str(
                    token_row["choice_boundary_evidence_sha256"]
                ),
                expected_prompt_token_ids_sha256=str(token_row["prompt_token_ids_sha256"]),
                expected_full_logits_float32_sha256=str(
                    source["full_logits_float32_sha256"]
                ),
                expected_positive_minus_negative_log_odds=float(
                    source["positive_minus_negative_log_odds"]
                ),
                expected_anchor_residual_float32_sha256=str(
                    source["pre_anchor_residual_float32_sha256"]
                ),
                expected_anchor_gradient_float32_sha256=str(
                    source["raw_anchor_gradient_float32_sha256"]
                ),
            )
            residuals.append(capture.residuals)
            gradients.append(capture.gradients)
            row = {
                "tensor_index": index,
                "form_id": form["form_id"],
                "prompt_sha256": form["prompt_sha256"],
                "prompt_token_ids_sha256": token_row["prompt_token_ids_sha256"],
                "slot_indices": list(capture.slot_indices),
                "positive_minus_negative_log_odds": capture.positive_minus_negative_log_odds,
                "residuals_float32_sha256": tensor_float32_sha256(capture.residuals),
                "gradients_float32_sha256": tensor_float32_sha256(capture.gradients),
                "full_logits_float32_sha256": tensor_float32_sha256(capture.full_logits),
                "capture_audit": _plain(capture.audit),
            }
            row["row_sha256"] = canonical_sha256(row)
            capture_records.append(row)
            print(f"CSMS capture {index + 1}/80 {form['form_id']}", flush=True)
        residual_tensor = torch.stack(residuals).float().contiguous()
        gradient_tensor = torch.stack(gradients).float().contiguous()
        alignment = build_capture_alignment_manifest(
            source_records=source_records,
            tokenizer_records=tokenizer_records,
            capture_records=capture_records,
            residuals=residual_tensor,
            gradients=gradient_tensor,
        )
        capture_metadata = {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "status": "complete",
            "split": split,
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "source_state0_file_sha256": file_sha256(V2_STATE0_PATH),
            "source_state0_checkpoint_sha256": state["checkpoint_sha256"],
            "source_tokenizer_file_sha256": file_sha256(V2_TOKENIZER_PATH),
            "record_count": 80,
            "records": capture_records,
            "row_alignment_manifest": alignment,
            "compute": {
                "model_forwards": 80,
                "model_backwards": 80,
                "generated_tokens": 0,
                "external_api_calls": 0,
                "external_model_judges": 0,
                "paid_model_cost_usd": 0,
            },
        }
        _base()._save_checkpoint(
            torch,
            path=CAPTURE_PATH,
            metadata=capture_metadata,
            tensors={"residuals": residual_tensor, "gradients": gradient_tensor},
        )
    except Exception as error:
        raise CSMSIntegrityError(
            "CSMS capture failed after its immutable compute reservation"
        ) from error
    complete = _with_hash(
        {
            "schema_version": "sp_lense.counterfactual_slot_matrix_steering_complete.v1",
            "status": "complete",
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "capture_file_sha256": file_sha256(CAPTURE_PATH),
            "reservation_file_sha256": file_sha256(CAPTURE_RESERVATION_PATH),
        },
        "completion_sha256",
    )
    _write_new_json(CAPTURE_COMPLETE_PATH, complete)
    return complete


def _load_capture(torch: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    lock = _load_lock()
    if not CAPTURE_COMPLETE_PATH.is_file() or not CAPTURE_RESERVATION_PATH.is_file():
        raise CSMSIntegrityError(
            "CSMS capture lacks its immutable reservation or completion record"
        )
    reservation = _load_json(CAPTURE_RESERVATION_PATH)
    _verify_hash(reservation, "reservation_sha256")
    complete = _load_json(CAPTURE_COMPLETE_PATH)
    _verify_hash(complete, "completion_sha256")
    if (
        reservation.get("schema_version")
        != "sp_lense.counterfactual_slot_matrix_steering_reservation.v1"
        or reservation.get("status") != "reserved_before_first_model_forward"
        or reservation.get("lock_identity_sha256") != lock["lock_identity_sha256"]
        or reservation.get("model_forwards") != 80
        or reservation.get("model_backwards") != 80
        or reservation.get("generated_tokens") != 0
        or reservation.get("external_api_calls") != 0
        or reservation.get("external_model_judges") != 0
        or reservation.get("paid_model_cost_usd") != 0
        or reservation.get("offline_environment")
        != {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
        or complete.get("schema_version")
        != "sp_lense.counterfactual_slot_matrix_steering_complete.v1"
        or complete.get("status") != "complete"
        or complete.get("lock_identity_sha256") != lock["lock_identity_sha256"]
        or complete.get("capture_file_sha256") != file_sha256(CAPTURE_PATH)
        or complete.get("reservation_file_sha256")
        != file_sha256(CAPTURE_RESERVATION_PATH)
    ):
        raise CSMSIntegrityError("CSMS capture completion binding differs")
    metadata, tensors = _base()._load_checkpoint(
        torch, path=CAPTURE_PATH, schema=CAPTURE_SCHEMA_VERSION
    )
    state, _ = _source_checkpoint(torch)
    tokenizer = _load_json(V2_TOKENIZER_PATH)
    compute = metadata.get("compute")
    if (
        metadata.get("schema_version") != CAPTURE_SCHEMA_VERSION
        or metadata.get("status") != "complete"
        or metadata.get("split") != "opened_development"
        or metadata.get("lock_identity_sha256") != lock["lock_identity_sha256"]
        or metadata.get("source_state0_file_sha256") != file_sha256(V2_STATE0_PATH)
        or metadata.get("source_state0_checkpoint_sha256") != state["checkpoint_sha256"]
        or metadata.get("source_tokenizer_file_sha256") != file_sha256(V2_TOKENIZER_PATH)
        or metadata.get("record_count") != 80
        or compute
        != {
            "model_forwards": 80,
            "model_backwards": 80,
            "generated_tokens": 0,
            "external_api_calls": 0,
            "external_model_judges": 0,
            "paid_model_cost_usd": 0,
        }
        or tensors.get("residuals") is None
        or tensors.get("gradients") is None
        or tuple(tensors["residuals"].shape) != (80, 4, 1024)
        or tuple(tensors["gradients"].shape) != (80, 4, 1024)
        or tensors["residuals"].dtype != torch.float32
        or tensors["gradients"].dtype != torch.float32
    ):
        raise CSMSIntegrityError("CSMS capture metadata, compute, or tensor contract differs")
    capture_records = metadata.get("records")
    if not isinstance(capture_records, list) or len(capture_records) != 80:
        raise CSMSIntegrityError("CSMS capture records differ")
    audited_forwards = 0
    audited_backwards = 0
    for row in capture_records:
        audit = row.get("capture_audit")
        if not isinstance(audit, Mapping):
            raise CSMSIntegrityError("CSMS capture row lacks its audit")
        audit_body = {key: value for key, value in audit.items() if key != "audit_sha256"}
        row_body = {key: value for key, value in row.items() if key != "row_sha256"}
        if (
            audit.get("audit_sha256") != canonical_sha256(audit_body)
            or row.get("row_sha256") != canonical_sha256(row_body)
            or audit.get("model_forward_evaluations") != 1
            or audit.get("model_backward_evaluations") != 1
            or audit.get("hook_call_count") != 1
            or audit.get("model_parameters_requires_grad_disabled_during_capture")
            is not True
            or audit.get("model_parameter_requires_grad_flags_restored_after_capture")
            is not True
            or audit.get("model_parameter_gradients_allocated") is not False
            or audit.get("maximum_abs_activation_reconstruction_delta") != 0.0
            or any(
                audit.get(field) is not True
                for field in (
                    "source_anchor_residual_reproduced",
                    "source_anchor_gradient_reproduced",
                    "source_full_logits_reproduced",
                    "source_margin_reproduced",
                    "source_tokenization_reproduced",
                )
            )
        ):
            raise CSMSIntegrityError("one CSMS capture row audit differs")
        audited_forwards += int(audit["model_forward_evaluations"])
        audited_backwards += int(audit["model_backward_evaluations"])
    if audited_forwards != 80 or audited_backwards != 80:
        raise CSMSIntegrityError("CSMS per-row capture compute does not sum to 80F/80B")
    alignment = build_capture_alignment_manifest(
        source_records=state["records"],
        tokenizer_records=tokenizer["records"],
        capture_records=metadata["records"],
        residuals=tensors["residuals"],
        gradients=tensors["gradients"],
    )
    if alignment != metadata.get("row_alignment_manifest"):
        raise CSMSIntegrityError("CSMS stored row-alignment manifest differs")
    return metadata, tensors, state


def run_analyze(split: str = "opened_development") -> dict[str, Any]:
    require_opened_development_split(split)
    lock = _load_lock()
    if GEOMETRY_PATH.exists() or DIRECTION_PATH.exists():
        raise FileExistsError("refusing to overwrite immutable CSMS geometry")
    import torch

    capture, tensors, state = _load_capture(torch)
    analysis = analyze_csms_geometry(
        records=state["records"],
        residuals=tensors["residuals"].numpy(),
        gradients=tensors["gradients"].numpy(),
    )
    direction_tensors = {
        name: torch.from_numpy(value.copy()).contiguous()
        for name, value in sorted(analysis.directions.items())
    }
    direction_metadata = {
        "schema_version": "sp_lense.counterfactual_slot_matrix_steering_directions.v1",
        "lock_identity_sha256": lock["lock_identity_sha256"],
        "capture_checkpoint_sha256": capture["checkpoint_sha256"],
        "direction_bundle_sha256": analysis.report["direction_bundle_sha256"],
    }
    _base()._save_checkpoint(
        torch,
        path=DIRECTION_PATH,
        metadata=direction_metadata,
        tensors=direction_tensors,
    )
    result = _with_hash(
        {
            "schema_version": RESULT_SCHEMA,
            "status": analysis.report["status"],
            "split": split,
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "capture_file_sha256": file_sha256(CAPTURE_PATH),
            "capture_checkpoint_sha256": capture["checkpoint_sha256"],
            "direction_file_sha256": file_sha256(DIRECTION_PATH),
            "analysis": _plain(analysis.report),
            "finite_intervention_authorized": analysis.report["qualification"][
                "finite_intervention_authorized"
            ],
            "sealed_data_accessed": False,
        },
        "result_sha256",
    )
    _write_new_json(GEOMETRY_PATH, result)
    return result


def _render_report(result: Mapping[str, Any]) -> str:
    analysis = result["analysis"]
    primary = analysis["global_methods"]["primary_four_slots"]
    qualification = analysis["qualification"]
    lines = [
        "# Counterfactual Slot-Matrix Steering geometry",
        "",
        f"- Status: `{result['status']}`",
        f"- Global primary: `{primary['status']}`",
        f"- Certified minimum Frobenius norm: `{primary.get('minimum_frobenius_norm')}`",
        f"- All four leave-one-scenario-out folds pass: `{analysis['leave_one_scenario_out']['passes']}`",
        f"- Finite intervention authorized: `{qualification['finite_intervention_authorized']}`",
        "- Model compute: 80 forward + 80 backward captures; geometry uses zero model passes.",
        "- Generated tokens / API calls / judges / paid model cost: `0 / 0 / 0 / USD 0`.",
        "- Sealed data accessed: `false`.",
        "",
        "This is opened, local first-order geometry. It does not establish a natural",
        "self-preservation mechanism, finite behavior change, preserved general capability,",
        "a universal direction, or publication-ready novelty.",
        "",
        f"Result SHA-256: `{result['result_sha256']}`",
    ]
    return "\n".join(lines) + "\n"


def run_report(split: str = "opened_development") -> str:
    require_opened_development_split(split)
    _load_lock()
    result = _load_json(GEOMETRY_PATH)
    _verify_hash(result, "result_sha256")
    value = _render_report(result)
    if REPORT_PATH.exists():
        if REPORT_PATH.read_text(encoding="utf-8") != value:
            raise CSMSIntegrityError("existing CSMS report differs")
    else:
        _atomic_text(REPORT_PATH, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("lock", "preflight", "capture", "analyze", "report"),
        nargs="?",
        default="preflight",
    )
    parser.add_argument("--split", default="opened_development")
    args = parser.parse_args()
    if args.command == "lock":
        value: Any = run_lock(args.split)
    elif args.command == "preflight":
        value = run_preflight(args.split)
    elif args.command == "capture":
        value = run_capture(args.split)
    elif args.command == "analyze":
        value = run_analyze(args.split)
    else:
        print(run_report(args.split), end="")
        return
    print(json.dumps(_plain(value), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
