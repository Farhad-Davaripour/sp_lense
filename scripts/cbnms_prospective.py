#!/usr/bin/env python3
"""Local-only prospective CBNMS lock, capture, staged geometry, and report runner.

Only ``capture`` executes the model (exactly 80 forwards plus 80 backwards).
``analyze`` is model-free and writes all four immutable training folds before
any held-derived arithmetic.  No command accepts or knows a sealed-data path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for value in (SRC, ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from sp_lense.backend import ResearchBackend
from sp_lense.config import load_config
from sp_lense.counterfactual_behavioral_null_multilayer import (
    CBNMSIntegrityError,
    analyze_full_data_bank,
    analyze_training_fold,
    build_loso_folds,
    build_tokenizer_preflight,
    capture_all_layers_four_slots,
    evaluate_held_fold,
    render_prospective_forms,
    require_prospective_split,
    summarize_geometry,
    validate_prospective_dataset,
)
from sp_lense.factorial_causal_anchor import (
    canonical_sha256,
    tensor_float32_sha256,
    text_sha256,
)

LOCK_SCHEMA = "sp_lense.cbnms_prospective_lock.v1"
LOCK_STATUS = "prospective_locked_before_any_CBNMS_model_outcome"
RESULT_SCHEMA = "sp_lense.cbnms_prospective_result.v1"
RESERVATION_SCHEMA = "sp_lense.cbnms_capture_reservation.v1"
COMPLETION_SCHEMA = "sp_lense.cbnms_capture_completion.v1"
TRAINING_COMPLETE_SCHEMA = "sp_lense.cbnms_training_complete.v1"
HELD_COMPLETE_SCHEMA = "sp_lense.cbnms_held_complete.v1"

DATA_PATH = ROOT / "data" / "cbnms_prospective_validation.json"
TEMPLATE_PATH = ROOT / "configs" / "cbnms_prospective_template.json"
LOCK_PATH = ROOT / "configs" / "cbnms_prospective_lock.json"
PROTOCOL_PATH = ROOT / "docs" / "CBNMS_PROSPECTIVE_PROTOCOL.md"
MODEL_CONFIG_PATH = ROOT / "configs" / "qwen35_08b_aligned.json"
PRIOR_RESULT_PATH = (
    ROOT
    / "results"
    / "all_layer_four_slot_oracle_screen"
    / "qwen35_08b"
    / "result.json"
)
RESULT_ROOT = ROOT / "results" / "cbnms_prospective" / "qwen35_08b"
PREFLIGHT_PATH = RESULT_ROOT / "tokenizer_preflight.json"
CAPTURE_RESERVATION_PATH = RESULT_ROOT / "capture_reservation.json"
CAPTURE_PATH = RESULT_ROOT / "capture.pt"
CAPTURE_COMPLETE_PATH = RESULT_ROOT / "capture_complete.json"
TRAINING_ROOT = RESULT_ROOT / "training_folds"
TRAINING_COMPLETE_PATH = RESULT_ROOT / "training_complete.json"
HELD_ROOT = RESULT_ROOT / "held_folds"
HELD_COMPLETE_PATH = RESULT_ROOT / "held_complete.json"
FULL_DATA_PATH = RESULT_ROOT / "full_data.json"
FULL_DATA_FREEZE_PATH = RESULT_ROOT / "full_data_freeze.pt"
RESULT_PATH = RESULT_ROOT / "result.json"
REPORT_PATH = RESULT_ROOT / "REPORT.md"

MODEL = {
    "id": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "device": "cpu",
    "dtype": "float32",
    "n_layers": 24,
    "d_model": 1024,
}
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}
COMPUTE_CEILING = {
    "model_forwards": 80,
    "model_backwards": 80,
    "generated_tokens": 0,
    "external_api_calls": 0,
    "external_model_judges": 0,
    "paid_model_cost_usd": 0,
    "finite_interventions": 0,
}
LOCKED_PATHS = (
    Path("src/sp_lense/counterfactual_behavioral_null_multilayer.py"),
    Path("scripts/cbnms_prospective.py"),
    Path("tests/test_cbnms_prospective.py"),
    Path("tests/test_cbnms_prospective_runner.py"),
    Path("configs/cbnms_prospective_template.json"),
    Path("docs/CBNMS_PROSPECTIVE_PROTOCOL.md"),
    Path("data/cbnms_prospective_validation.json"),
    Path("src/sp_lense/all_layer_four_slot_oracle.py"),
    Path("src/sp_lense/factorial_causal_anchor.py"),
    Path("src/sp_lense/comparison_runtime.py"),
    Path("src/sp_lense/decision_margin_shield.py"),
    Path("src/sp_lense/decision_margin_shield_rowspace.py"),
    Path("src/sp_lense/counterfactual_slot_matrix_steering.py"),
    Path("src/sp_lense/backend.py"),
    Path("src/sp_lense/config.py"),
    Path("configs/qwen35_08b_aligned.json"),
    Path("pyproject.toml"),
    Path("requirements-research.txt"),
    Path("requirements-constrained-steering.txt"),
)
ADAPTIVE_LINEAGE_PATHS = (
    Path("docs/ALL_LAYER_FOUR_SLOT_ORACLE_SCREEN_PROTOCOL.md"),
    Path("configs/all_layer_four_slot_oracle_screen_lock.json"),
    Path("results/all_layer_four_slot_oracle_screen/qwen35_08b/result.json"),
    Path("results/all_layer_four_slot_oracle_screen/qwen35_08b/REPORT.md"),
    Path("configs/counterfactual_slot_matrix_steering_lock.json"),
    Path("results/counterfactual_slot_matrix_steering/qwen35_08b/geometry.json"),
    Path("results/context_gated_dynamic/qwen35_08b/exact_prompt_order_summary.json"),
    Path("results/context_gated_dynamic/qwen35_08b/gated_replay_summary.json"),
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
    Path("results/factorial_interface_translator_development/qwen35_08b/geometric_result.json"),
    Path(
        "results/factorial_interface_translator_development/"
        "qwen35_08b/posthoc_attainability.json"
    ),
    Path("results/global_counterfactual_robust_boundary/qwen35_08b/all_layer_geometry.json"),
    Path(
        "results/global_counterfactual_robust_boundary/"
        "qwen35_08b/integrated_conclusion.json"
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float64_array_identity(value: Any) -> dict[str, Any]:
    array = np.asarray(value, dtype="<f8", order="C")
    return {
        "dtype": "float64",
        "shape": list(array.shape),
        "raw_little_endian_bytes_sha256": hashlib.sha256(
            array.tobytes(order="C")
        ).hexdigest(),
    }


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CBNMSIntegrityError(f"{_relative(path)} must contain one JSON object")
    return value


def _with_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = {str(key): _plain(item) for key, item in value.items() if key != field}
    result[field] = canonical_sha256(result)
    return result


def _verify_self_hash(value: Mapping[str, Any], field: str) -> None:
    body = {str(key): _plain(item) for key, item in value.items() if key != field}
    if value.get(field) != canonical_sha256(body):
        raise CBNMSIntegrityError(f"{field} differs")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable {_relative(path)}")
    _atomic_text(path, json.dumps(_plain(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _write_new_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable {_relative(path)}")
    _atomic_text(path, value)


def _write_new_checkpoint(
    torch: Any,
    path: Path,
    *,
    metadata: Mapping[str, Any],
    tensors: Mapping[str, Any],
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable {_relative(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {"metadata": _plain(metadata), "tensors": dict(tensors)}
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_checkpoint(
    torch: Any, path: Path, *, schema: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or set(payload) != {"metadata", "tensors"}:
        raise CBNMSIntegrityError("CBNMS tensor checkpoint envelope differs")
    metadata = payload["metadata"]
    tensors = payload["tensors"]
    if not isinstance(metadata, dict) or not isinstance(tensors, dict):
        raise CBNMSIntegrityError("CBNMS tensor checkpoint payload differs")
    if metadata.get("schema_version") != schema:
        raise CBNMSIntegrityError("CBNMS tensor checkpoint schema differs")
    _verify_self_hash(metadata, "checkpoint_sha256")
    return metadata, tensors


def _source() -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    payload = _load_json(DATA_PATH)
    validate_prospective_dataset(payload)
    forms = render_prospective_forms(payload)
    build_loso_folds(forms)
    return payload, forms


def run_validate(split: str = "prospective_validation") -> dict[str, Any]:
    require_prospective_split(split)
    payload, forms = _source()
    folds = build_loso_folds(forms)
    return _with_hash(
        {
            "schema_version": "sp_lense.cbnms_validation.v1",
            "split": split,
            "dataset_file_sha256": file_sha256(DATA_PATH),
            "dataset_record": validate_prospective_dataset(payload),
            "rendered_form_ids_sha256": canonical_sha256(
                [str(value["form_id"]) for value in forms]
            ),
            "rendered_forms_sha256": canonical_sha256(list(forms)),
            "folds": list(folds),
            "model_compute": {
                "model_forwards": 0,
                "model_backwards": 0,
                "generated_tokens": 0,
                "external_api_calls": 0,
                "external_model_judges": 0,
            },
            "sealed_data_accessed": False,
            "passes": True,
        },
        "validation_sha256",
    )


def _expected_configuration() -> dict[str, Any]:
    configuration = copy.deepcopy(_load_json(TEMPLATE_PATH))
    configuration["status"] = LOCK_STATUS
    _, forms = _source()
    configuration["source"]["file_sha256"] = file_sha256(DATA_PATH)
    configuration["source"]["rendered_form_ids_sha256"] = canonical_sha256(
        [str(value["form_id"]) for value in forms]
    )
    configuration["source"]["rendered_forms_sha256"] = canonical_sha256(list(forms))
    configuration["prospective_status"]["adaptive_prior_result_file_sha256"] = (
        file_sha256(PRIOR_RESULT_PATH)
    )
    configuration["prospective_status"]["adaptive_lineage"] = [
        {
            "path": str(path).replace("\\", "/"),
            "file_sha256": file_sha256(ROOT / path),
        }
        for path in ADAPTIVE_LINEAGE_PATHS
    ]
    return configuration


def proposed_lock(split: str = "prospective_validation") -> dict[str, Any]:
    require_prospective_split(split)
    missing = [str(value) for value in LOCKED_PATHS if not (ROOT / value).is_file()]
    if missing:
        raise FileNotFoundError(f"CBNMS locked source paths are missing: {missing}")
    records = [
        {
            "path": str(path).replace("\\", "/"),
            "sha256": file_sha256(ROOT / path),
            "bytes": (ROOT / path).stat().st_size,
        }
        for path in LOCKED_PATHS
    ]
    body = {
        "schema_version": LOCK_SCHEMA,
        "status": LOCK_STATUS,
        "split": split,
        "configuration": _expected_configuration(),
        "locked_files": records,
        "adaptive_prior_result": {
            "path": _relative(PRIOR_RESULT_PATH),
            "file_sha256": file_sha256(PRIOR_RESULT_PATH),
            "tensors_read_or_reused": False,
            "role": "disclosed adaptive negative result only",
        },
        "adaptive_lineage": _expected_configuration()["prospective_status"][
            "adaptive_lineage"
        ],
        "sealed_data_accessible": False,
    }
    value = _with_hash(body, "lock_identity_sha256")
    return verify_lock(value, verify_files=True)


def verify_lock(value: Mapping[str, Any], *, verify_files: bool) -> dict[str, Any]:
    lock = dict(value)
    _verify_self_hash(lock, "lock_identity_sha256")
    if (
        lock.get("schema_version") != LOCK_SCHEMA
        or lock.get("status") != LOCK_STATUS
        or lock.get("split") != "prospective_validation"
        or lock.get("sealed_data_accessible") is not False
        or lock.get("configuration") != _expected_configuration()
    ):
        raise CBNMSIntegrityError("CBNMS lock contract differs")
    records = lock.get("locked_files")
    expected_paths = [str(value).replace("\\", "/") for value in LOCKED_PATHS]
    if (
        not isinstance(records, list)
        or [record.get("path") for record in records] != expected_paths
        or len({record.get("path") for record in records}) != len(expected_paths)
    ):
        raise CBNMSIntegrityError("CBNMS locked file coverage differs")
    for path, record in zip(LOCKED_PATHS, records, strict=True):
        target = ROOT / path
        if verify_files and (
            record.get("sha256") != file_sha256(target)
            or record.get("bytes") != target.stat().st_size
        ):
            raise CBNMSIntegrityError(f"locked CBNMS bytes differ: {path}")
    adaptive = lock.get("adaptive_prior_result")
    if adaptive != {
        "path": _relative(PRIOR_RESULT_PATH),
        "file_sha256": file_sha256(PRIOR_RESULT_PATH),
        "tensors_read_or_reused": False,
        "role": "disclosed adaptive negative result only",
    }:
        raise CBNMSIntegrityError("CBNMS adaptive provenance differs")
    if lock.get("adaptive_lineage") != _expected_configuration()[
        "prospective_status"
    ]["adaptive_lineage"]:
        raise CBNMSIntegrityError("CBNMS adaptive lineage coverage differs")
    return lock


def _load_lock() -> dict[str, Any]:
    if not LOCK_PATH.is_file():
        raise FileNotFoundError("CBNMS command requires the reviewed final lock")
    return verify_lock(_load_json(LOCK_PATH), verify_files=True)


def run_lock(split: str = "prospective_validation") -> dict[str, Any]:
    require_prospective_split(split)
    if LOCK_PATH.exists():
        raise FileExistsError("refusing to overwrite immutable CBNMS lock")
    value = proposed_lock(split)
    _write_new_json(LOCK_PATH, value)
    return value


def _require_offline() -> None:
    observed = {name: os.environ.get(name) for name in OFFLINE_ENVIRONMENT}
    if observed != OFFLINE_ENVIRONMENT:
        raise CBNMSIntegrityError("CBNMS requires exact local-only offline environment")


def run_preflight(split: str = "prospective_validation") -> dict[str, Any]:
    require_prospective_split(split)
    lock = _load_lock()
    if PREFLIGHT_PATH.exists():
        raise FileExistsError("refusing to overwrite immutable CBNMS preflight")
    _require_offline()
    _, forms = _source()
    backend = ResearchBackend.load(load_config(MODEL_CONFIG_PATH), with_lens=False)
    try:
        if (
            backend.config.model.id != MODEL["id"]
            or backend.config.model.revision != MODEL["revision"]
            or backend.device != MODEL["device"]
            or backend.dtype_name != MODEL["dtype"]
        ):
            raise CBNMSIntegrityError("resident backend differs from locked CBNMS model")
        core = build_tokenizer_preflight(backend, forms)
    finally:
        del backend
    value = _with_hash(
        {
            "schema_version": "sp_lense.cbnms_tokenizer_preflight_file.v1",
            "status": "complete",
            "split": split,
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "dataset_file_sha256": file_sha256(DATA_PATH),
            "offline_environment": OFFLINE_ENVIRONMENT,
            "core": core,
            "model_compute": {
                "model_forwards": 0,
                "model_backwards": 0,
                "generated_tokens": 0,
                "external_api_calls": 0,
                "external_model_judges": 0,
            },
            "sealed_data_accessed": False,
        },
        "preflight_file_sha256",
    )
    _write_new_json(PREFLIGHT_PATH, value)
    return value


def _load_preflight(lock: Mapping[str, Any]) -> dict[str, Any]:
    if not PREFLIGHT_PATH.is_file():
        raise FileNotFoundError("CBNMS capture requires completed tokenizer preflight")
    value = _load_json(PREFLIGHT_PATH)
    _verify_self_hash(value, "preflight_file_sha256")
    _, forms = _source()
    core = value.get("core")
    if not isinstance(core, Mapping):
        raise CBNMSIntegrityError("CBNMS preflight lacks its core record")
    _verify_self_hash(core, "preflight_sha256")
    rows = core.get("rows", [])
    if (
        value.get("schema_version") != "sp_lense.cbnms_tokenizer_preflight_file.v1"
        or value.get("status") != "complete"
        or value.get("split") != "prospective_validation"
        or value.get("lock_identity_sha256") != lock["lock_identity_sha256"]
        or value.get("dataset_file_sha256") != file_sha256(DATA_PATH)
        or value.get("offline_environment") != OFFLINE_ENVIRONMENT
        or core.get("schema_version") != "sp_lense.cbnms_prospective.v1.tokenizer_preflight"
        or core.get("split") != "prospective_validation"
        or core.get("model_forwards") != 0
        or core.get("model_backwards") != 0
        or core.get("generated_tokens") != 0
        or core.get("external_api_calls") != 0
        or core.get("external_model_judges") != 0
        or core.get("slot_rule") != [3, "anchor-8", "anchor-4", "anchor"]
        or core.get("layers") != list(range(23))
        or core.get("excluded_layer") != 23
        or core.get("semantic_anchor_rule")
        != (
            "source prompt and construction share an identical prefix ending in "
            "exactly one [FACTS COMPLETE] marker; token anchor is the last common "
            "token position, not a claimed UTF-8 byte boundary"
        )
        or len(rows) != 80
        or [row.get("form_id") for row in rows]
        != [form["form_id"] for form in forms]
        or value.get("sealed_data_accessed") is not False
    ):
        raise CBNMSIntegrityError("CBNMS tokenizer preflight contract differs")
    for index, (row, form) in enumerate(zip(rows, forms, strict=True)):
        _verify_self_hash(row, "row_sha256")
        anchor = row.get("anchor_index")
        if (
            row.get("tensor_index") != index
            or row.get("form_sha256") != form["form_sha256"]
            or row.get("prompt_sha256") != form["prompt_sha256"]
            or row.get("anchor_prefix_sha256")
            != text_sha256(str(form["anchor_prefix"]))
            or row.get("anchor_prefix_ends_facts_complete_marker") is not True
            or row.get("source_prompt_starts_with_anchor_prefix") is not True
            or row.get("source_construction_starts_with_anchor_prefix") is not True
            or type(anchor) is not int
            or row.get("slot_indices") != [3, anchor - 8, anchor - 4, anchor]
            or len(set(row["slot_indices"])) != 4
            or row["slot_indices"] != sorted(row["slot_indices"])
            or row.get("prompt_length", 0) <= anchor
            or len(row.get("slot_token_ids", [])) != 4
        ):
            raise CBNMSIntegrityError("one CBNMS preflight row contract differs")
    return value


def run_capture(split: str = "prospective_validation") -> dict[str, Any]:
    require_prospective_split(split)
    lock = _load_lock()
    preflight = _load_preflight(lock)
    if CAPTURE_PATH.exists() or CAPTURE_COMPLETE_PATH.exists():
        raise FileExistsError("refusing to overwrite immutable CBNMS capture")
    if CAPTURE_RESERVATION_PATH.exists():
        raise CBNMSIntegrityError("prior CBNMS reservation requires manual audit")
    _require_offline()
    reservation = _with_hash(
        {
            "schema_version": RESERVATION_SCHEMA,
            "status": "reserved_before_first_model_forward",
            "split": split,
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "preflight_file_sha256": file_sha256(PREFLIGHT_PATH),
            "offline_environment": OFFLINE_ENVIRONMENT,
            **COMPUTE_CEILING,
        },
        "reservation_sha256",
    )
    _write_new_json(CAPTURE_RESERVATION_PATH, reservation)
    _, forms = _source()
    backend = ResearchBackend.load(load_config(MODEL_CONFIG_PATH), with_lens=False)
    residuals = []
    gradients = []
    records = []
    try:
        recomputed_preflight = build_tokenizer_preflight(backend, forms)
        if recomputed_preflight != preflight["core"]:
            raise CBNMSIntegrityError(
                "resident tokenizer recomputation differs before first CBNMS forward"
            )
        for index, (form, token_row) in enumerate(
            zip(forms, preflight["core"]["rows"], strict=True)
        ):
            capture = capture_all_layers_four_slots(backend, form, token_row)
            residuals.append(capture.residuals)
            gradients.append(capture.gradients)
            records.append(
                _with_hash(
                    {
                        "tensor_index": index,
                        "form_id": form["form_id"],
                        "form_sha256": form["form_sha256"],
                        "preflight_row_sha256": token_row["row_sha256"],
                        "positive_minus_negative_log_odds": (
                            capture.positive_minus_negative_log_odds
                        ),
                        "residuals_float32_sha256": tensor_float32_sha256(
                            capture.residuals
                        ),
                        "gradients_float32_sha256": tensor_float32_sha256(
                            capture.gradients
                        ),
                        "full_logits_float32_sha256": tensor_float32_sha256(
                            capture.full_logits
                        ),
                        "capture_audit": dict(capture.audit),
                    },
                    "row_sha256",
                )
            )
            print(f"CBNMS capture {index + 1}/80 {form['form_id']}", flush=True)
        torch = backend.torch
        residual_tensor = torch.stack(residuals).float().contiguous()
        gradient_tensor = torch.stack(gradients).float().contiguous()
    finally:
        del backend
    if tuple(residual_tensor.shape) != (80, 23, 4, 1024) or gradient_tensor.shape != residual_tensor.shape:
        raise CBNMSIntegrityError("CBNMS aggregate capture tensors differ")
    metadata = _with_hash(
        {
            "schema_version": "sp_lense.cbnms_capture_checkpoint.v1",
            "status": "complete",
            "split": split,
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "preflight_file_sha256": file_sha256(PREFLIGHT_PATH),
            "reservation_file_sha256": file_sha256(CAPTURE_RESERVATION_PATH),
            "dataset_file_sha256": file_sha256(DATA_PATH),
            "record_count": 80,
            "records": records,
            "compute": COMPUTE_CEILING,
            "offline_environment": OFFLINE_ENVIRONMENT,
            "residuals_float32_sha256": tensor_float32_sha256(residual_tensor),
            "gradients_float32_sha256": tensor_float32_sha256(gradient_tensor),
            "prior_experiment_tensors_read": False,
            "sealed_data_accessed": False,
        },
        "checkpoint_sha256",
    )
    _write_new_checkpoint(
        torch,
        CAPTURE_PATH,
        metadata=metadata,
        tensors={"residuals": residual_tensor, "gradients": gradient_tensor},
    )
    complete = _with_hash(
        {
            "schema_version": COMPLETION_SCHEMA,
            "status": "complete",
            "split": split,
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "capture_file_sha256": file_sha256(CAPTURE_PATH),
            "capture_checkpoint_sha256": metadata["checkpoint_sha256"],
            "reservation_file_sha256": file_sha256(CAPTURE_RESERVATION_PATH),
        },
        "completion_sha256",
    )
    _write_new_json(CAPTURE_COMPLETE_PATH, complete)
    return complete


def _load_capture() -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    lock = _load_lock()
    preflight = _load_preflight(lock)
    if not CAPTURE_RESERVATION_PATH.is_file() or not CAPTURE_COMPLETE_PATH.is_file():
        raise CBNMSIntegrityError("CBNMS capture reservation/completion is missing")
    reservation = _load_json(CAPTURE_RESERVATION_PATH)
    complete = _load_json(CAPTURE_COMPLETE_PATH)
    _verify_self_hash(reservation, "reservation_sha256")
    _verify_self_hash(complete, "completion_sha256")
    metadata, tensors = _load_checkpoint(
        torch, CAPTURE_PATH, schema="sp_lense.cbnms_capture_checkpoint.v1"
    )
    if (
        reservation.get("schema_version") != RESERVATION_SCHEMA
        or reservation.get("status") != "reserved_before_first_model_forward"
        or reservation.get("split") != "prospective_validation"
        or reservation.get("lock_identity_sha256") != lock["lock_identity_sha256"]
        or reservation.get("preflight_file_sha256") != file_sha256(PREFLIGHT_PATH)
        or {key: reservation.get(key) for key in COMPUTE_CEILING} != COMPUTE_CEILING
        or reservation.get("offline_environment") != OFFLINE_ENVIRONMENT
        or complete.get("schema_version") != COMPLETION_SCHEMA
        or complete.get("status") != "complete"
        or complete.get("split") != "prospective_validation"
        or complete.get("lock_identity_sha256") != lock["lock_identity_sha256"]
        or complete.get("capture_file_sha256") != file_sha256(CAPTURE_PATH)
        or complete.get("capture_checkpoint_sha256") != metadata["checkpoint_sha256"]
        or complete.get("reservation_file_sha256")
        != file_sha256(CAPTURE_RESERVATION_PATH)
        or metadata.get("status") != "complete"
        or metadata.get("split") != "prospective_validation"
        or metadata.get("lock_identity_sha256") != lock["lock_identity_sha256"]
        or metadata.get("preflight_file_sha256") != file_sha256(PREFLIGHT_PATH)
        or metadata.get("reservation_file_sha256")
        != file_sha256(CAPTURE_RESERVATION_PATH)
        or metadata.get("dataset_file_sha256") != file_sha256(DATA_PATH)
        or metadata.get("offline_environment") != OFFLINE_ENVIRONMENT
        or metadata.get("compute") != COMPUTE_CEILING
        or metadata.get("record_count") != 80
        or metadata.get("prior_experiment_tensors_read") is not False
        or metadata.get("sealed_data_accessed") is not False
        or set(tensors) != {"residuals", "gradients"}
        or tuple(tensors["residuals"].shape) != (80, 23, 4, 1024)
        or tuple(tensors["gradients"].shape) != (80, 23, 4, 1024)
        or tensors["residuals"].dtype != torch.float32
        or tensors["gradients"].dtype != torch.float32
        or tensor_float32_sha256(tensors["residuals"])
        != metadata["residuals_float32_sha256"]
        or tensor_float32_sha256(tensors["gradients"])
        != metadata["gradients_float32_sha256"]
    ):
        raise CBNMSIntegrityError("CBNMS capture contract differs")
    _, forms = _source()
    records = metadata.get("records")
    if not isinstance(records, list) or len(records) != 80:
        raise CBNMSIntegrityError("CBNMS capture record coverage differs")
    for index, (record, form, token_row) in enumerate(
        zip(records, forms, preflight["core"]["rows"], strict=True)
    ):
        _verify_self_hash(record, "row_sha256")
        audit = record.get("capture_audit")
        if not isinstance(audit, Mapping):
            raise CBNMSIntegrityError("one CBNMS capture row lacks its audit")
        _verify_self_hash(audit, "audit_sha256")
        if (
            record.get("tensor_index") != index
            or record.get("form_id") != form["form_id"]
            or record.get("form_sha256") != form["form_sha256"]
            or record.get("preflight_row_sha256") != token_row["row_sha256"]
            or tensor_float32_sha256(tensors["residuals"][index])
            != record["residuals_float32_sha256"]
            or tensor_float32_sha256(tensors["gradients"][index])
            != record["gradients_float32_sha256"]
            or record.get("positive_minus_negative_log_odds")
            != audit.get("positive_minus_negative_log_odds")
            or record.get("residuals_float32_sha256")
            != audit.get("residuals_float32_sha256")
            or record.get("gradients_float32_sha256")
            != audit.get("gradients_float32_sha256")
            or record.get("full_logits_float32_sha256")
            != audit.get("full_logits_float32_sha256")
            or audit.get("schema_version")
            != "sp_lense.cbnms_prospective.v1.capture"
            or audit.get("capture_kind")
            != "fresh_state_zero_layers_0_through_22_four_slots_one_F_plus_one_B"
            or audit.get("form_id") != form["form_id"]
            or audit.get("layers") != list(range(23))
            or audit.get("excluded_layer") != 23
            or audit.get("excluded_layer_reason")
            != "hook_out_causal_position_excluded_a_priori"
            or audit.get("slot_indices") != token_row["slot_indices"]
            or audit.get("hook_call_counts")
            != {str(layer): 1 for layer in range(23)}
            or audit.get("model_forward_evaluations") != 1
            or audit.get("model_backward_evaluations") != 1
            or audit.get("generated_tokens") != 0
            or audit.get("external_api_calls") != 0
            or audit.get("external_model_judges") != 0
            or audit.get("finite_interventions") != 0
            or audit.get("maximum_abs_layer0_reconstruction_delta") != 0.0
            or audit.get("model_parameters_requires_grad_disabled_during_capture")
            is not True
            or audit.get("model_parameter_requires_grad_flags_restored_after_capture")
            is not True
            or audit.get("model_parameter_gradients_allocated") is not False
            or audit.get("later_layer_hooks_return_activation_unchanged") is not True
            or audit.get("prior_experiment_tensors_read") is not False
            or audit.get("sealed_data_accessed") is not False
        ):
            raise CBNMSIntegrityError("one CBNMS capture row differs")
    return metadata, tensors


def _training_path(index: int) -> Path:
    return TRAINING_ROOT / f"fold_{index}.json"


def _training_freeze_path(index: int) -> Path:
    return TRAINING_ROOT / f"fold_{index}_freeze.pt"


def _held_path(index: int) -> Path:
    return HELD_ROOT / f"fold_{index}.json"


def _write_training_fold(
    torch: Any,
    index: int,
    core: Mapping[str, Any],
    numeric: Mapping[str, Any] | None,
) -> dict[str, Any]:
    freeze_path = _training_freeze_path(index)
    freeze_record = None
    if numeric is not None:
        freeze_metadata = _with_hash(
            {
                "schema_version": "sp_lense.cbnms_training_freeze.v1",
                "fold_index": index,
                "training_record_sha256": core["record_sha256"],
                "tensor_identities": {
                    name: {
                        "shape": list(value.shape),
                        "dtype": "float64",
                        "raw_sha256": hashlib.sha256(
                            value.astype("<f8", copy=False).tobytes(order="C")
                        ).hexdigest(),
                    }
                    for name, value in numeric.items()
                },
            },
            "checkpoint_sha256",
        )
        tensors = {
            name: torch.from_numpy(value.copy()).double().contiguous()
            for name, value in numeric.items()
        }
        _write_new_checkpoint(torch, freeze_path, metadata=freeze_metadata, tensors=tensors)
        freeze_record = {
            "path": _relative(freeze_path),
            "file_sha256": file_sha256(freeze_path),
            "checkpoint_sha256": freeze_metadata["checkpoint_sha256"],
        }
    value = _with_hash(
        {
            "schema_version": "sp_lense.cbnms_training_fold_file.v1",
            "fold_index": index,
            "core": dict(core),
            "freeze": freeze_record,
            "passes": core["passes"],
        },
        "training_file_sha256",
    )
    _write_new_json(_training_path(index), value)
    return value


def _load_training_numeric(torch: Any, training_file: Mapping[str, Any]) -> dict[str, Any]:
    _verify_self_hash(training_file, "training_file_sha256")
    _verify_self_hash(training_file["core"], "record_sha256")
    freeze = training_file.get("freeze")
    if not isinstance(freeze, Mapping):
        raise CBNMSIntegrityError("passing training fold lacks numeric freeze")
    path = ROOT / str(freeze["path"])
    fold_index = int(training_file.get("fold_index", -1))
    if path.resolve() != _training_freeze_path(fold_index).resolve():
        raise CBNMSIntegrityError("CBNMS training freeze path differs")
    if file_sha256(path) != freeze["file_sha256"]:
        raise CBNMSIntegrityError("CBNMS training freeze bytes differ")
    metadata, tensors = _load_checkpoint(
        torch, path, schema="sp_lense.cbnms_training_freeze.v1"
    )
    if metadata["checkpoint_sha256"] != freeze["checkpoint_sha256"]:
        raise CBNMSIntegrityError("CBNMS training freeze identity differs")
    expected_names = {"scales", "nuisance_basis", "SP_bank", "target_only_bank"}
    if (
        metadata.get("fold_index") != fold_index
        or metadata.get("training_record_sha256")
        != training_file.get("core", {}).get("record_sha256")
        or set(tensors) != expected_names
        or set(metadata.get("tensor_identities", {})) != expected_names
    ):
        raise CBNMSIntegrityError("CBNMS training freeze metadata differs")
    result = {}
    for name, tensor in tensors.items():
        if tensor.dtype != torch.float64:
            raise CBNMSIntegrityError("one CBNMS training freeze tensor dtype differs")
        array = tensor.numpy()
        identity = {
            "shape": list(array.shape),
            "dtype": "float64",
            "raw_sha256": hashlib.sha256(
                array.astype("<f8", copy=False).tobytes(order="C")
            ).hexdigest(),
        }
        if identity != metadata["tensor_identities"][name]:
            raise CBNMSIntegrityError("one CBNMS training freeze tensor identity differs")
        result[name] = array
    if (
        result["scales"].shape != (23, 4)
        or result["nuisance_basis"].shape != (44, 94208)
        or result["SP_bank"].ndim != 2
        or result["SP_bank"].shape[1] != 94208
        or result["SP_bank"].shape[0] > 6
        or result["target_only_bank"].ndim != 2
        or result["target_only_bank"].shape[1] != 94208
        or result["target_only_bank"].shape[0] > 6
    ):
        raise CBNMSIntegrityError("CBNMS training freeze tensor shapes differ")
    core = training_file["core"]
    if (
        _float64_array_identity(result["scales"])
        != core["training_scales_identity"]
        or _float64_array_identity(result["nuisance_basis"])
        != core["training_nuisance_basis"]["svd_record"]["basis_identity"]
        or _float64_array_identity(result["SP_bank"])
        != core["training_SP_bank"]["bank_basis_identity"]
        or _float64_array_identity(result["target_only_bank"])
        != core["training_target_only_bank"]["bank_basis_identity"]
    ):
        raise CBNMSIntegrityError("CBNMS training freeze differs from core identities")
    return result


def _load_full_data_numeric(
    torch: Any, full_file: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Verify the exact full-data freeze bytes and their semantic core identities."""

    full_freeze = full_file.get("freeze")
    if full_file.get("passes") is not True:
        if full_freeze is not None or FULL_DATA_FREEZE_PATH.exists():
            raise CBNMSIntegrityError(
                "failed/skipped CBNMS full data must not have a freeze"
            )
        return None
    if not isinstance(full_freeze, Mapping):
        raise CBNMSIntegrityError("passing CBNMS full data lacks its numeric freeze")
    freeze_path = ROOT / str(full_freeze.get("path"))
    if not freeze_path.is_file():
        raise CBNMSIntegrityError("passing CBNMS full-data freeze is missing")
    if (
        freeze_path.resolve() != FULL_DATA_FREEZE_PATH.resolve()
        or file_sha256(freeze_path) != full_freeze.get("file_sha256")
    ):
        raise CBNMSIntegrityError("CBNMS full-data freeze bytes differ")
    freeze_metadata, freeze_tensors = _load_checkpoint(
        torch,
        freeze_path,
        schema="sp_lense.cbnms_full_data_freeze.v1",
    )
    expected_names = {"scales", "nuisance_basis", "SP_bank"}
    if (
        freeze_metadata.get("checkpoint_sha256")
        != full_freeze.get("checkpoint_sha256")
        or freeze_metadata.get("full_data_record_sha256")
        != full_file["core"]["record_sha256"]
        or freeze_metadata.get("tensor_names") != sorted(expected_names)
        or set(freeze_metadata.get("tensor_identities", {})) != expected_names
        or set(freeze_tensors) != expected_names
    ):
        raise CBNMSIntegrityError("CBNMS full-data freeze metadata differs")
    arrays = {}
    for name, tensor in freeze_tensors.items():
        if tensor.dtype != torch.float64:
            raise CBNMSIntegrityError("one full-data freeze tensor dtype differs")
        array = tensor.numpy()
        metadata_identity = {
            "shape": list(array.shape),
            "dtype": "float64",
            "raw_sha256": hashlib.sha256(
                array.astype("<f8", copy=False).tobytes(order="C")
            ).hexdigest(),
        }
        if metadata_identity != freeze_metadata["tensor_identities"][name]:
            raise CBNMSIntegrityError("one full-data freeze tensor identity differs")
        arrays[name] = array
    if (
        arrays["scales"].shape != (23, 4)
        or arrays["nuisance_basis"].shape != (64, 94208)
        or arrays["SP_bank"].ndim != 2
        or not 0 < arrays["SP_bank"].shape[0] <= 8
        or arrays["SP_bank"].shape[1] != 94208
    ):
        raise CBNMSIntegrityError("CBNMS full-data freeze tensor shapes differ")
    core = full_file["core"]
    if (
        _float64_array_identity(arrays["scales"])
        != core["training_scales_identity"]
        or _float64_array_identity(arrays["nuisance_basis"])
        != core["full_nuisance_basis"]["svd_record"]["basis_identity"]
        or _float64_array_identity(arrays["SP_bank"])
        != core["full_data_SP_bank"]["bank_basis_identity"]
    ):
        raise CBNMSIntegrityError("CBNMS full-data freeze differs from core identities")
    return arrays


def _skipped_held(fold: Mapping[str, Any], training_file: Mapping[str, Any]) -> dict[str, Any]:
    random_rows = [
        {
            "replicate": index,
            "passes_complete_fold_gate": False,
        }
        for index in range(32)
    ]
    return _with_hash(
        {
            "schema_version": "sp_lense.cbnms_held_fold_file.v1",
            "fold_index": fold["fold_index"],
            "fold": dict(fold),
            "status": "not_evaluated_because_training_fold_failed",
            "training_file_sha256": training_file["training_file_sha256"],
            "random_rank_matched_nullspace_controls": random_rows,
            "passes": False,
        },
        "held_file_sha256",
    )


def _verify_fold_source_binding(
    training_file: Mapping[str, Any],
    held_file: Mapping[str, Any],
    expected_fold: Mapping[str, Any],
    index: int,
) -> None:
    """Bind every stored fold, including skipped held rows, to fresh source logic."""

    held_core = held_file.get("core", held_file)
    if (
        training_file.get("fold_index") != index
        or training_file.get("core", {}).get("fold") != expected_fold
        or held_file.get("fold_index") != index
        or held_core.get("fold") != expected_fold
    ):
        raise CBNMSIntegrityError("one CBNMS fold differs from fresh source construction")


def _copy_training_only_numeric(
    residuals: Any,
    gradients: Any,
    margins: Sequence[float],
    fold: Mapping[str, Any],
) -> tuple[Any, Any, list[float]]:
    """Copy exactly the 56 training rows; excluded rows are never validated here."""

    indices = [int(value) for value in fold["training_all_indices"]]
    if len(indices) != 56 or len(set(indices)) != 56 or any(
        value < 0 or value >= 80 for value in indices
    ):
        raise CBNMSIntegrityError("one CBNMS training slice has invalid indices")
    return (
        residuals[indices].copy(),
        gradients[indices].copy(),
        [margins[value] for value in indices],
    )


def _full_data_stop_record(
    training_files: Sequence[Mapping[str, Any]],
    held_files: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Return an immutable no-go record when LOSO makes full-data rescue impossible."""

    if len(training_files) != 4 or len(held_files) != 4:
        raise CBNMSIntegrityError("CBNMS full-data stop check requires four LOSO folds")
    if all(value.get("passes") is True for value in training_files) and all(
        value.get("passes") is True for value in held_files
    ):
        return None
    return _with_hash(
        {
            "schema_version": "sp_lense.cbnms_prospective.v1.full_data_bank",
            "split": "prospective_validation",
            "status": "not_evaluated_because_one_or_more_LOSO_gates_failed",
            "run_only_after_all_four_fold_artifacts_are_immutable": True,
            "full_data_numeric_rows_used": False,
            "checks": {
                "all_training_and_held_LOSO_gates_pass_before_full_data": False
            },
            "passes": False,
        },
        "record_sha256",
    )


def run_analyze(split: str = "prospective_validation") -> dict[str, Any]:
    require_prospective_split(split)
    lock = _load_lock()
    if any(
        path.exists()
        for path in (
            TRAINING_COMPLETE_PATH,
            HELD_COMPLETE_PATH,
            FULL_DATA_PATH,
            FULL_DATA_FREEZE_PATH,
            RESULT_PATH,
        )
    ) or TRAINING_ROOT.exists() or HELD_ROOT.exists():
        raise FileExistsError("refusing to overwrite or resume partial CBNMS analysis")
    metadata, tensors = _load_capture()
    import torch

    _, forms = _source()
    folds = build_loso_folds(forms)
    residuals = tensors["residuals"].numpy()
    gradients = tensors["gradients"].numpy()
    margins = [float(record["positive_minus_negative_log_odds"]) for record in metadata["records"]]
    dataset_sha = file_sha256(DATA_PATH)
    training_files = []
    for index, fold in enumerate(folds):
        # The core receives copied training-only arrays. Held numeric rows are not
        # validated, standardized, scaled, or otherwise inspected before freeze.
        training_residuals, training_gradients, training_margins = (
            _copy_training_only_numeric(residuals, gradients, margins, fold)
        )
        core, numeric = analyze_training_fold(
            forms=forms,
            fold=fold,
            residuals=training_residuals,
            gradients=training_gradients,
            margins=training_margins,
            dataset_sha256=dataset_sha,
            random_replicates=32,
        )
        training_files.append(_write_training_fold(torch, index, core, numeric))
    training_complete = _with_hash(
        {
            "schema_version": TRAINING_COMPLETE_SCHEMA,
            "status": "complete",
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "capture_file_sha256": file_sha256(CAPTURE_PATH),
            "training_files": [
                {
                    "path": _relative(_training_path(index)),
                    "file_sha256": file_sha256(_training_path(index)),
                }
                for index in range(4)
            ],
            "all_training_folds_pass": all(value["passes"] for value in training_files),
            "held_arithmetic_preceded_this_artifact": False,
        },
        "training_complete_sha256",
    )
    _write_new_json(TRAINING_COMPLETE_PATH, training_complete)

    held_files = []
    for index, (fold, training_file) in enumerate(
        zip(folds, training_files, strict=True)
    ):
        if training_file["passes"]:
            numeric = _load_training_numeric(torch, training_file)
            core = evaluate_held_fold(
                forms=forms,
                fold=fold,
                training_record=training_file["core"],
                frozen_numeric=numeric,
                residuals=residuals,
                gradients=gradients,
                margins=margins,
                dataset_sha256=dataset_sha,
            )
            value = _with_hash(
                {
                    "schema_version": "sp_lense.cbnms_held_fold_file.v1",
                    "fold_index": index,
                    "status": "complete",
                    "training_file_sha256": training_file["training_file_sha256"],
                    "core": core,
                    "random_rank_matched_nullspace_controls": core[
                        "random_rank_matched_nullspace_controls"
                    ],
                    "passes": core["passes"],
                },
                "held_file_sha256",
            )
        else:
            value = _skipped_held(fold, training_file)
        _write_new_json(_held_path(index), value)
        held_files.append(value)
    held_complete = _with_hash(
        {
            "schema_version": HELD_COMPLETE_SCHEMA,
            "status": "complete",
            "training_complete_file_sha256": file_sha256(TRAINING_COMPLETE_PATH),
            "held_files": [
                {"path": _relative(_held_path(index)), "file_sha256": file_sha256(_held_path(index))}
                for index in range(4)
            ],
            "all_held_folds_pass": all(value["passes"] for value in held_files),
        },
        "held_complete_sha256",
    )
    _write_new_json(HELD_COMPLETE_PATH, held_complete)

    stopped_full_data = _full_data_stop_record(training_files, held_files)
    if stopped_full_data is None:
        full_core, full_numeric = analyze_full_data_bank(
            forms=forms,
            residuals=residuals,
            gradients=gradients,
            margins=margins,
        )
    else:
        full_core = stopped_full_data
        full_numeric = None
    full_freeze = None
    if full_numeric is not None:
        freeze_metadata = _with_hash(
            {
                "schema_version": "sp_lense.cbnms_full_data_freeze.v1",
                "full_data_record_sha256": full_core["record_sha256"],
                "tensor_names": sorted(full_numeric),
                "tensor_identities": {
                    name: {
                        "shape": list(value.shape),
                        "dtype": "float64",
                        "raw_sha256": hashlib.sha256(
                            value.astype("<f8", copy=False).tobytes(order="C")
                        ).hexdigest(),
                    }
                    for name, value in full_numeric.items()
                },
            },
            "checkpoint_sha256",
        )
        _write_new_checkpoint(
            torch,
            FULL_DATA_FREEZE_PATH,
            metadata=freeze_metadata,
            tensors={
                name: torch.from_numpy(value.copy()).double().contiguous()
                for name, value in full_numeric.items()
            },
        )
        full_freeze = {
            "path": _relative(FULL_DATA_FREEZE_PATH),
            "file_sha256": file_sha256(FULL_DATA_FREEZE_PATH),
            "checkpoint_sha256": freeze_metadata["checkpoint_sha256"],
        }
    full_file = _with_hash(
        {
            "schema_version": "sp_lense.cbnms_full_data_file.v1",
            "held_complete_file_sha256": file_sha256(HELD_COMPLETE_PATH),
            "core": full_core,
            "freeze": full_freeze,
            "passes": full_core["passes"],
        },
        "full_data_file_sha256",
    )
    _write_new_json(FULL_DATA_PATH, full_file)
    summary = summarize_geometry(
        training_folds=[value["core"] for value in training_files],
        held_folds=[value.get("core", value) for value in held_files],
        full_data=full_core,
    )
    result = _with_hash(
        {
            "schema_version": RESULT_SCHEMA,
            "status": summary["status"],
            "split": split,
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "capture_file_sha256": file_sha256(CAPTURE_PATH),
            "capture_checkpoint_sha256": metadata["checkpoint_sha256"],
            "training_complete_file_sha256": file_sha256(TRAINING_COMPLETE_PATH),
            "held_complete_file_sha256": file_sha256(HELD_COMPLETE_PATH),
            "full_data_file_sha256": file_sha256(FULL_DATA_PATH),
            "training_fold_files": training_complete["training_files"],
            "held_fold_files": held_complete["held_files"],
            "summary": summary,
            "model_compute": COMPUTE_CEILING,
            "analysis_model_compute": {
                "model_forwards": 0,
                "model_backwards": 0,
                "generated_tokens": 0,
                "external_api_calls": 0,
                "external_model_judges": 0,
                "finite_interventions": 0,
            },
            "prior_experiment_tensors_read": False,
            "sealed_data_accessed": False,
        },
        "result_sha256",
    )
    _write_new_json(RESULT_PATH, result)
    return result


def _verify_result() -> dict[str, Any]:
    lock = _load_lock()
    capture_metadata, capture_tensors = _load_capture()
    del capture_tensors
    result = _load_json(RESULT_PATH)
    _verify_self_hash(result, "result_sha256")
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("split") != "prospective_validation"
        or result.get("lock_identity_sha256") != lock["lock_identity_sha256"]
        or result.get("capture_file_sha256") != file_sha256(CAPTURE_PATH)
        or result.get("capture_checkpoint_sha256")
        != capture_metadata["checkpoint_sha256"]
        or result.get("training_complete_file_sha256") != file_sha256(TRAINING_COMPLETE_PATH)
        or result.get("held_complete_file_sha256") != file_sha256(HELD_COMPLETE_PATH)
        or result.get("full_data_file_sha256") != file_sha256(FULL_DATA_PATH)
        or result.get("prior_experiment_tensors_read") is not False
        or result.get("sealed_data_accessed") is not False
    ):
        raise CBNMSIntegrityError("CBNMS result provenance differs")
    for record in [*result["training_fold_files"], *result["held_fold_files"]]:
        if file_sha256(ROOT / record["path"]) != record["file_sha256"]:
            raise CBNMSIntegrityError("one CBNMS result artifact binding differs")
    _verify_self_hash(result["summary"], "record_sha256")
    training_complete = _load_json(TRAINING_COMPLETE_PATH)
    held_complete = _load_json(HELD_COMPLETE_PATH)
    full_file = _load_json(FULL_DATA_PATH)
    _verify_self_hash(training_complete, "training_complete_sha256")
    _verify_self_hash(held_complete, "held_complete_sha256")
    _verify_self_hash(full_file, "full_data_file_sha256")
    expected_training_records = [
        {
            "path": _relative(_training_path(index)),
            "file_sha256": file_sha256(_training_path(index)),
        }
        for index in range(4)
    ]
    expected_held_records = [
        {
            "path": _relative(_held_path(index)),
            "file_sha256": file_sha256(_held_path(index)),
        }
        for index in range(4)
    ]
    if (
        training_complete.get("schema_version") != TRAINING_COMPLETE_SCHEMA
        or training_complete.get("status") != "complete"
        or training_complete.get("lock_identity_sha256")
        != lock["lock_identity_sha256"]
        or training_complete.get("capture_file_sha256") != file_sha256(CAPTURE_PATH)
        or training_complete.get("held_arithmetic_preceded_this_artifact") is not False
        or training_complete.get("training_files") != expected_training_records
        or result.get("training_fold_files") != expected_training_records
        or held_complete.get("schema_version") != HELD_COMPLETE_SCHEMA
        or held_complete.get("status") != "complete"
        or held_complete.get("training_complete_file_sha256")
        != file_sha256(TRAINING_COMPLETE_PATH)
        or held_complete.get("held_files") != expected_held_records
        or result.get("held_fold_files") != expected_held_records
        or full_file.get("schema_version") != "sp_lense.cbnms_full_data_file.v1"
        or full_file.get("held_complete_file_sha256")
        != file_sha256(HELD_COMPLETE_PATH)
    ):
        raise CBNMSIntegrityError("CBNMS completion artifact contract differs")
    training_files = []
    held_files = []
    import torch

    _, forms = _source()
    expected_folds = build_loso_folds(forms)
    for index in range(4):
        training_file = _load_json(_training_path(index))
        held_file = _load_json(_held_path(index))
        _verify_self_hash(training_file, "training_file_sha256")
        _verify_self_hash(training_file["core"], "record_sha256")
        _verify_self_hash(held_file, "held_file_sha256")
        _verify_fold_source_binding(
            training_file, held_file, expected_folds[index], index
        )
        if (
            training_file.get("schema_version")
            != "sp_lense.cbnms_training_fold_file.v1"
            or training_file.get("fold_index") != index
            or training_file.get("passes") != training_file["core"].get("passes")
            or held_file.get("schema_version") != "sp_lense.cbnms_held_fold_file.v1"
            or held_file.get("fold_index") != index
            or held_file.get("training_file_sha256")
            != training_file["training_file_sha256"]
        ):
            raise CBNMSIntegrityError("one CBNMS fold artifact contract differs")
        if training_file["passes"]:
            _load_training_numeric(torch, training_file)
        elif training_file.get("freeze") is not None:
            raise CBNMSIntegrityError("a failed training fold has a numeric freeze")
        if "core" in held_file:
            _verify_self_hash(held_file["core"], "record_sha256")
            if held_file.get("passes") != held_file["core"].get("passes"):
                raise CBNMSIntegrityError("one held wrapper/core status differs")
        elif held_file.get("status") != "not_evaluated_because_training_fold_failed":
            raise CBNMSIntegrityError("one skipped held artifact status differs")
        training_files.append(training_file)
        held_files.append(held_file)
    _verify_self_hash(full_file["core"], "record_sha256")
    if full_file.get("passes") != full_file["core"].get("passes"):
        raise CBNMSIntegrityError("CBNMS full-data wrapper/core status differs")
    _load_full_data_numeric(torch, full_file)
    recomputed = summarize_geometry(
        training_folds=[value["core"] for value in training_files],
        held_folds=[value.get("core", value) for value in held_files],
        full_data=full_file["core"],
    )
    if recomputed != result["summary"]:
        raise CBNMSIntegrityError("CBNMS stored summary differs from exact artifacts")
    if (
        result.get("status") != recomputed["status"]
        or result.get("model_compute") != COMPUTE_CEILING
        or result.get("analysis_model_compute")
        != {
            "model_forwards": 0,
            "model_backwards": 0,
            "generated_tokens": 0,
            "external_api_calls": 0,
            "external_model_judges": 0,
            "finite_interventions": 0,
        }
    ):
        raise CBNMSIntegrityError("CBNMS result status or compute differs")
    return result


def run_report(split: str = "prospective_validation") -> Path:
    require_prospective_split(split)
    if REPORT_PATH.exists():
        raise FileExistsError("refusing to overwrite immutable CBNMS report")
    result = _verify_result()
    passed = result["summary"]["passes"]
    lines = [
        "# CBNMS prospective geometry report",
        "",
        f"Status: **{result['status']}**.",
        "",
        "This is a target-aware white-box transductive fixed-algorithm oracle, not a generalizing controller.",
        "No finite multi-layer intervention was run. All physical arithmetic is requested/state-zero-linearized.",
        "Dynamic, multi-layer, and nullspace steering components are prior art; this result is not a novelty or publication claim.",
        "",
        f"All strict geometry gates passed: **{passed}**.",
        f"Random banks passing the complete all-fold gate: **{result['summary']['random_complete_all_fold_count']}/32**.",
        "",
        "If false, the locked stop rule forbids layer pruning, alternate slots/caps, prompt rescue, or post-hoc retuning.",
        "If true, only drafting and auditing a separate finite prospective protocol is authorized.",
        "",
        f"Result SHA-256: `{result['result_sha256']}`",
    ]
    _write_new_text(REPORT_PATH, "\n".join(lines) + "\n")
    return REPORT_PATH


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("validate", "lock", "preflight", "capture", "analyze", "report")
    )
    parser.add_argument("--split", default="prospective_validation")
    args = parser.parse_args(argv)
    if args.command == "validate":
        value: Any = run_validate(args.split)
    elif args.command == "lock":
        value = run_lock(args.split)
    elif args.command == "preflight":
        value = run_preflight(args.split)
    elif args.command == "capture":
        value = run_capture(args.split)
    elif args.command == "analyze":
        value = run_analyze(args.split)
    else:
        value = str(run_report(args.split))
    print(json.dumps(_plain(value), indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
