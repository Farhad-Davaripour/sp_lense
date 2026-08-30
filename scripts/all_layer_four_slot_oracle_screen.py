#!/usr/bin/env python3
"""Local-only lock, capture, and staged analysis runner for ALFS.

The only model-compute command is ``capture`` and it is capped at exactly 80
forwards plus 80 backwards.  Held numeric rows in the monolithic capture are
excluded from every training-derived computation; training selections are
written immutably before separate held-derived arithmetic begins.  No command
accepts a sealed split or performs a finite intervention.
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
for value in (SRC, ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from sp_lense.all_layer_four_slot_oracle import (
    ANALYSIS_SCHEMA_VERSION,
    CAPTURE_SCHEMA_VERSION,
    DOUBLE_CERTIFICATE_TOLERANCE,
    FLOAT32_PHYSICAL_TOLERANCE,
    LAYER_COUNT,
    LEAKAGE_RATIO_MAXIMUM,
    QUALIFICATION_CAP,
    SCALAR_GATE_TOLERANCE,
    TARGET_MARGIN,
    ALFSIntegrityError,
    analyze_training_layer,
    build_outer_folds,
    capture_all_layer_four_slots,
    evaluate_held_oracles,
    frozen_nuisance_rowspace,
    require_opened_development_split,
    select_layer,
    training_only_slot_scales,
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

LOCK_SCHEMA = "sp_lense.all_layer_four_slot_oracle_screen_lock.v1"
LOCK_STATUS = "adaptive_opened_development_locked_before_alfs_model_compute"
RESULT_SCHEMA = "sp_lense.all_layer_four_slot_oracle_screen_result.v1"
RESERVATION_SCHEMA = "sp_lense.all_layer_four_slot_oracle_reservation.v1"
COMPLETION_SCHEMA = "sp_lense.all_layer_four_slot_oracle_completion.v1"
TRAINING_FREEZE_SCHEMA = "sp_lense.all_layer_four_slot_oracle_training_freeze.v1"

TEMPLATE_PATH = ROOT / "configs" / "all_layer_four_slot_oracle_screen_template.json"
LOCK_PATH = ROOT / "configs" / "all_layer_four_slot_oracle_screen_lock.json"
PROTOCOL_PATH = ROOT / "docs" / "ALL_LAYER_FOUR_SLOT_ORACLE_SCREEN_PROTOCOL.md"
CSMS_RUNNER_PATH = ROOT / "scripts" / "counterfactual_slot_matrix_steering.py"
CSMS_LOCK_PATH = ROOT / "configs" / "counterfactual_slot_matrix_steering_lock.json"
CSMS_CAPTURE_PATH = (
    ROOT
    / "results"
    / "counterfactual_slot_matrix_steering"
    / "qwen35_08b"
    / "capture.pt"
)
CSMS_RESERVATION_PATH = CSMS_CAPTURE_PATH.parent / "capture_reservation.json"
CSMS_COMPLETE_PATH = CSMS_CAPTURE_PATH.parent / "capture_complete.json"
CSMS_GEOMETRY_PATH = CSMS_CAPTURE_PATH.parent / "geometry.json"
DATA_PATH = ROOT / "data" / "ckes_v2_validation.json"

RESULT_ROOT = ROOT / "results" / "all_layer_four_slot_oracle_screen" / "qwen35_08b"
CAPTURE_PATH = RESULT_ROOT / "capture.pt"
CAPTURE_RESERVATION_PATH = RESULT_ROOT / "capture_reservation.json"
CAPTURE_COMPLETE_PATH = RESULT_ROOT / "capture_complete.json"
TRAINING_ROOT = RESULT_ROOT / "training_folds"
TRAINING_COMPLETE_PATH = RESULT_ROOT / "training_complete.json"
HELD_ROOT = RESULT_ROOT / "held_folds"
HELD_COMPLETE_PATH = RESULT_ROOT / "held_complete.json"
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
}

LOCKED_PATHS = (
    Path("src/sp_lense/all_layer_four_slot_oracle.py"),
    Path("scripts/all_layer_four_slot_oracle_screen.py"),
    Path("tests/test_all_layer_four_slot_oracle.py"),
    Path("tests/test_all_layer_four_slot_oracle_runner.py"),
    Path("configs/all_layer_four_slot_oracle_screen_template.json"),
    Path("docs/ALL_LAYER_FOUR_SLOT_ORACLE_SCREEN_PROTOCOL.md"),
    Path("src/sp_lense/counterfactual_slot_matrix_steering.py"),
    Path("scripts/counterfactual_slot_matrix_steering.py"),
    Path("src/sp_lense/comparison_runtime.py"),
    Path("src/sp_lense/factorial_causal_anchor.py"),
    Path("src/sp_lense/decision_margin_shield.py"),
    Path("src/sp_lense/decision_margin_shield_rowspace.py"),
    Path("src/sp_lense/counterfactual_tangent_shield.py"),
    Path("configs/counterfactual_slot_matrix_steering_lock.json"),
    Path("docs/COUNTERFACTUAL_SLOT_MATRIX_STEERING_PROTOCOL.md"),
    Path("data/ckes_v2_validation.json"),
    Path("results/counterfactual_slot_matrix_steering/qwen35_08b/capture.pt"),
    Path(
        "results/counterfactual_slot_matrix_steering/qwen35_08b/"
        "capture_reservation.json"
    ),
    Path(
        "results/counterfactual_slot_matrix_steering/qwen35_08b/"
        "capture_complete.json"
    ),
    Path("results/counterfactual_slot_matrix_steering/qwen35_08b/geometry.json"),
    Path("docs/DECISION_MARGIN_SHIELD_LAYER_SCREEN_PROTOCOL.md"),
    Path("docs/DECISION_MARGIN_SHIELD_LAYER_SCREEN_SOLVER_AMENDMENT.md"),
    Path(
        "results/decision_margin_shield_layer_screen_solver_amendment/"
        "qwen35_08b/layer_screen_result.json"
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
    Path("configs/qwen35_08b_aligned.json"),
    Path("pyproject.toml"),
    Path("requirements-research.txt"),
    Path("requirements-constrained-steering.txt"),
)

_CSMS: ModuleType | None = None


def _csms() -> ModuleType:
    global _CSMS
    if _CSMS is None:
        spec = importlib.util.spec_from_file_location("_alfs_csms_runner", CSMS_RUNNER_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load the immutable CSMS runner")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CSMS = module
    return _CSMS


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "item") and type(value).__module__.startswith("numpy"):
        return value.item()
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float64_array_identity(value: Any) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(value, dtype="<f8", order="C")
    return {
        "dtype": "float64",
        "shape": list(array.shape),
        "raw_little_endian_bytes_sha256": hashlib.sha256(
            array.tobytes(order="C")
        ).hexdigest(),
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ALFSIntegrityError(f"expected a JSON object: {path}")
    return value


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable ALFS artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(_plain(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    path.write_text(serialized, encoding="utf-8", newline="\n")


def _write_new_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable ALFS artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _with_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = _plain(value)
    result[field] = canonical_sha256(result)
    return result


def _verify_self_hash(value: Mapping[str, Any], field: str) -> None:
    observed = value.get(field)
    body = {key: item for key, item in value.items() if key != field}
    if not _is_sha256(observed) or observed != canonical_sha256(body):
        raise ALFSIntegrityError(f"{field} differs")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required ALFS lock input is missing: {path}")
    return {"path": _relative(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _expected_thresholds() -> dict[str, Any]:
    return {
        "target_margin": TARGET_MARGIN,
        "qualification_cap": QUALIFICATION_CAP,
        "held_absolute_nuisance_movement_maximum": TARGET_MARGIN,
        "held_leakage_ratio_maximum": LEAKAGE_RATIO_MAXIMUM,
        "double_certificate_tolerance": DOUBLE_CERTIFICATE_TOLERANCE,
        "float32_target_and_null_tolerance": FLOAT32_PHYSICAL_TOLERANCE,
        "held_scalar_norm_and_dose_tolerance": SCALAR_GATE_TOLERANCE,
        "choice_ambiguity_tolerance": FLOAT32_PHYSICAL_TOLERANCE,
        "all_four_folds_required": True,
        "fold_and_full_selected_layer_identity_required": True,
        "both_float32_signs_required": True,
        "zero_held_nuisance_choice_flips_required": True,
        "failure_action": "stop_all_layer_four_slot_search_without_rescue",
    }


def _expected_numerical_backend() -> dict[str, Any]:
    return {
        "paired_selecting_solver": "analytic_two_inequality_active_set",
        "paired_enumerated_active_sets": [[0], [1], [0, 1]],
        "paired_permutation_canonicalization": (
            "sha256_float64_projected_row_plus_required_slope"
        ),
        "global_nonselecting_solver": SOLVER_METHOD,
        "global_nonselecting_solver_max_iterations": SOLVER_MAX_ITERATIONS,
        "global_nonselecting_solver_function_tolerance": (
            SOLVER_FUNCTION_TOLERANCE
        ),
        "svd_rtol": DEFAULT_SVD_RTOL,
        "svd_atol": DEFAULT_SVD_ATOL,
        "strict_raw_feasibility_fraction": STRICT_RAW_FEASIBILITY_FRACTION,
        "strict_raw_roundoff_multiplier": STRICT_RAW_ROUNDOFF_MULTIPLIER,
    }


def _fold_contract(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "held_scenario_id": fold["held_scenario_id"],
            "held_unrelated_ids": fold["held_control_ids"],
        }
        for fold in build_outer_folds(records)
    ]


def _source_binding() -> dict[str, Any]:
    import torch

    metadata, tensors, state = _csms()._load_capture(torch)
    folds = _fold_contract(state["records"])
    template = _load_json(TEMPLATE_PATH)
    if folds != template["fold_construction"]["folds"]:
        raise ALFSIntegrityError("template fold literals differ from immutable source semantics")
    if tuple(tensors["residuals"].shape) != (80, 4, 1024) or tuple(
        tensors["gradients"].shape
    ) != (80, 4, 1024):
        raise ALFSIntegrityError("immutable CSMS layer-zero tensors differ")
    return {
        "dataset_file_sha256": file_sha256(DATA_PATH),
        "csms_lock_file_sha256": file_sha256(CSMS_LOCK_PATH),
        "csms_lock_identity_sha256": _load_json(CSMS_LOCK_PATH)["lock_identity_sha256"],
        "csms_capture_file_sha256": file_sha256(CSMS_CAPTURE_PATH),
        "csms_capture_checkpoint_sha256": metadata["checkpoint_sha256"],
        "csms_capture_reservation_file_sha256": file_sha256(CSMS_RESERVATION_PATH),
        "csms_capture_complete_file_sha256": file_sha256(CSMS_COMPLETE_PATH),
        "csms_geometry_file_sha256": file_sha256(CSMS_GEOMETRY_PATH),
        "csms_layer0_residuals_float32_sha256": tensor_float32_sha256(
            tensors["residuals"]
        ),
        "csms_layer0_gradients_float32_sha256": tensor_float32_sha256(
            tensors["gradients"]
        ),
        "csms_row_alignment_manifest_sha256": metadata["row_alignment_manifest"][
            "manifest_sha256"
        ],
        "source_record_ids_sha256": canonical_sha256(
            [record["form_id"] for record in state["records"]]
        ),
        "fold_contract": folds,
        "source_verification_model_forwards": 0,
        "source_verification_model_backwards": 0,
    }


def proposed_lock() -> dict[str, Any]:
    template = _load_json(TEMPLATE_PATH)
    csms_lock = _load_json(CSMS_LOCK_PATH)
    configuration = {
        **template,
        "status": LOCK_STATUS,
        "source_binding": _source_binding(),
        "pinned_runtime": csms_lock["configuration"]["pinned_runtime"],
        "chat_template_sha256": csms_lock["configuration"]["chat_template_sha256"],
        "numerical_backend": _expected_numerical_backend(),
        "local_only_environment_required_before_backend_load": OFFLINE_ENVIRONMENT,
        "lock_creation_compute": {
            **COMPUTE_CEILING,
            "model_forwards": 0,
            "model_backwards": 0,
        },
    }
    value = {
        "schema_version": LOCK_SCHEMA,
        "status": LOCK_STATUS,
        "file_hashes": {
            f"locked_{index:02d}": _file_record(ROOT / path)
            for index, path in enumerate(LOCKED_PATHS)
        },
        "configuration": configuration,
        "thresholds": _expected_thresholds(),
        "compute_ceiling": COMPUTE_CEILING,
        "sealed_access": {
            "permitted": False,
            "accepted_split": "opened_development",
            "sealed_paths_recorded": False,
            "sealed_bytes_must_never_be_read": True,
        },
    }
    value["lock_identity_sha256"] = canonical_sha256(value)
    return verify_lock(value, verify_files=True)


def verify_lock(value: Mapping[str, Any], *, verify_files: bool) -> dict[str, Any]:
    lock = _plain(value)
    identity = lock.pop("lock_identity_sha256", None)
    if not _is_sha256(identity) or identity != canonical_sha256(lock):
        raise ALFSIntegrityError("ALFS lock identity differs")
    lock["lock_identity_sha256"] = identity
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
        or lock["schema_version"] != LOCK_SCHEMA
        or lock["status"] != LOCK_STATUS
        or lock["thresholds"] != _expected_thresholds()
        or lock["compute_ceiling"] != COMPUTE_CEILING
        or lock["sealed_access"]
        != {
            "permitted": False,
            "accepted_split": "opened_development",
            "sealed_paths_recorded": False,
            "sealed_bytes_must_never_be_read": True,
        }
    ):
        raise ALFSIntegrityError("ALFS lock contract differs")
    file_records = lock["file_hashes"]
    expected_keys = [f"locked_{index:02d}" for index in range(len(LOCKED_PATHS))]
    expected_paths = [_relative(ROOT / path) for path in LOCKED_PATHS]
    if (
        not isinstance(file_records, Mapping)
        or list(file_records) != expected_keys
        or [record.get("path") for record in file_records.values()] != expected_paths
        or len(set(expected_paths)) != len(expected_paths)
    ):
        raise ALFSIntegrityError("ALFS locked file coverage differs")
    for record in file_records.values():
        if set(record) != {"path", "bytes", "sha256"} or not _is_sha256(
            record["sha256"]
        ):
            raise ALFSIntegrityError("one ALFS locked file record differs")
        if verify_files:
            path = ROOT / record["path"]
            if (
                not path.is_file()
                or path.stat().st_size != record["bytes"]
                or file_sha256(path) != record["sha256"]
            ):
                raise ALFSIntegrityError(f"locked ALFS bytes differ: {record['path']}")
    template = _load_json(TEMPLATE_PATH)
    expected_configuration = {
        **template,
        "status": LOCK_STATUS,
        "source_binding": _source_binding(),
        "pinned_runtime": _load_json(CSMS_LOCK_PATH)["configuration"]["pinned_runtime"],
        "chat_template_sha256": _load_json(CSMS_LOCK_PATH)["configuration"][
            "chat_template_sha256"
        ],
        "numerical_backend": _expected_numerical_backend(),
        "local_only_environment_required_before_backend_load": OFFLINE_ENVIRONMENT,
        "lock_creation_compute": {
            **COMPUTE_CEILING,
            "model_forwards": 0,
            "model_backwards": 0,
        },
    }
    if lock["configuration"] != expected_configuration:
        raise ALFSIntegrityError("ALFS locked configuration differs")
    return lock


def _load_lock() -> dict[str, Any]:
    if not LOCK_PATH.is_file():
        raise FileNotFoundError("ALFS compute and analysis require the reviewed final lock")
    return verify_lock(_load_json(LOCK_PATH), verify_files=True)


def run_lock(split: str = "opened_development") -> dict[str, Any]:
    require_opened_development_split(split)
    if LOCK_PATH.exists():
        raise FileExistsError("refusing to overwrite immutable ALFS lock")
    value = proposed_lock()
    _write_new_json(LOCK_PATH, value)
    return value


def run_preflight(split: str = "opened_development") -> dict[str, Any]:
    require_opened_development_split(split)
    lock = _load_lock() if LOCK_PATH.exists() else proposed_lock()
    return _with_hash(
        {
            "schema_version": "sp_lense.all_layer_four_slot_oracle_preflight.v1",
            "status": "locked_ready" if LOCK_PATH.exists() else "proposal_ready_not_locked",
            "split": split,
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "source_binding": lock["configuration"]["source_binding"],
            "next_expensive_phase": COMPUTE_CEILING,
            "preflight_model_compute": {**COMPUTE_CEILING, "model_forwards": 0, "model_backwards": 0},
        },
        "preflight_sha256",
    )


def _capture_alignment(
    *,
    source_records: Sequence[Mapping[str, Any]],
    csms_records: Sequence[Mapping[str, Any]],
    capture_records: Sequence[Mapping[str, Any]],
    residuals: Any,
    gradients: Any,
) -> dict[str, Any]:
    if not (len(source_records) == len(csms_records) == len(capture_records) == 80):
        raise ALFSIntegrityError("ALFS alignment requires exactly 80 rows")
    rows = []
    for index, (source, csms_row, capture) in enumerate(
        zip(source_records, csms_records, capture_records, strict=True)
    ):
        if (
            source["form_id"] != csms_row["form_id"]
            or source["form_id"] != capture["form_id"]
            or source["tensor_index"] != index
            or csms_row["tensor_index"] != index
            or capture["tensor_index"] != index
            or capture.get("prompt_token_ids_sha256")
            != source.get("prompt_token_ids_sha256")
            or capture.get("positive_minus_negative_log_odds")
            != source.get("positive_minus_negative_log_odds")
            or capture.get("full_logits_float32_sha256")
            != source.get("full_logits_float32_sha256")
            or capture.get("slot_indices") != csms_row.get("slot_indices")
            or tensor_float32_sha256(residuals[index])
            != capture["residuals_float32_sha256"]
            or tensor_float32_sha256(gradients[index])
            != capture["gradients_float32_sha256"]
            or tensor_float32_sha256(residuals[index, 0])
            != csms_row["residuals_float32_sha256"]
            or tensor_float32_sha256(gradients[index, 0])
            != csms_row["gradients_float32_sha256"]
        ):
            raise ALFSIntegrityError("one ALFS tensor row is misaligned")
        row = _with_hash(
            {
                "tensor_index": index,
                "form_id": source["form_id"],
                "source_row_sha256": source["row_sha256"],
                "csms_capture_row_sha256": csms_row["row_sha256"],
                "alfs_capture_row_sha256": capture["row_sha256"],
                "prompt_sha256": source["form"]["prompt_sha256"],
                "slot_indices": capture["slot_indices"],
                "residuals_float32_sha256": capture["residuals_float32_sha256"],
                "gradients_float32_sha256": capture["gradients_float32_sha256"],
            },
            "alignment_row_sha256",
        )
        rows.append(row)
    return _with_hash(
        {
            "schema_version": f"{CAPTURE_SCHEMA_VERSION}.row_alignment",
            "row_count": 80,
            "source_order_is_authoritative": True,
            "rows": rows,
            "rows_sha256": canonical_sha256(rows),
        },
        "manifest_sha256",
    )


def run_capture(split: str = "opened_development") -> dict[str, Any]:
    require_opened_development_split(split)
    lock = _load_lock()
    if CAPTURE_PATH.exists() or CAPTURE_COMPLETE_PATH.exists():
        raise FileExistsError("refusing to overwrite immutable ALFS capture")
    if CAPTURE_RESERVATION_PATH.exists():
        raise ALFSIntegrityError("a prior ALFS reservation exists; manual audit is required")
    observed_offline = {name: os.environ.get(name) for name in OFFLINE_ENVIRONMENT}
    if observed_offline != OFFLINE_ENVIRONMENT:
        raise ALFSIntegrityError("ALFS capture requires exact local-only offline environment")
    import torch

    csms_metadata, _csms_tensors, state = _csms()._load_capture(torch)
    reservation = _with_hash(
        {
            "schema_version": RESERVATION_SCHEMA,
            "status": "reserved_before_first_model_forward",
            "split": split,
            "lock_identity_sha256": lock["lock_identity_sha256"],
            **COMPUTE_CEILING,
            "offline_environment": observed_offline,
            "source_csms_capture_file_sha256": file_sha256(CSMS_CAPTURE_PATH),
        },
        "reservation_sha256",
    )
    _write_new_json(CAPTURE_RESERVATION_PATH, reservation)
    original = _csms()._base()._base()._finite()._load_original_runner()
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
        raise ALFSIntegrityError("resident backend differs from locked ALFS model")
    residuals = []
    gradients = []
    capture_records = []
    try:
        for index, (source, csms_row) in enumerate(
            zip(state["records"], csms_metadata["records"], strict=True)
        ):
            form = source["form"]
            capture = capture_all_layer_four_slots(
                backend,
                str(form["prompt"]),
                str(form["positive_label"]),
                str(form["negative_label"]),
                positive_semantic=str(form["positive_semantic"]),
                negative_semantic=str(form["negative_semantic"]),
                slot_indices=tuple(csms_row["slot_indices"]),
                expected_prompt_sha256=str(form["prompt_sha256"]),
                expected_choice_boundary_evidence_sha256=str(
                    csms_row["capture_audit"]["choice_boundary_evidence_sha256"]
                ),
                expected_prompt_token_ids_sha256=str(
                    csms_row["prompt_token_ids_sha256"]
                ),
                expected_full_logits_float32_sha256=str(
                    source["full_logits_float32_sha256"]
                ),
                expected_positive_minus_negative_log_odds=float(
                    source["positive_minus_negative_log_odds"]
                ),
                expected_layer0_residuals_float32_sha256=str(
                    csms_row["residuals_float32_sha256"]
                ),
                expected_layer0_gradients_float32_sha256=str(
                    csms_row["gradients_float32_sha256"]
                ),
            )
            residuals.append(capture.residuals)
            gradients.append(capture.gradients)
            row = _with_hash(
                {
                    "tensor_index": index,
                    "form_id": source["form_id"],
                    "prompt_sha256": form["prompt_sha256"],
                    "prompt_token_ids_sha256": csms_row["prompt_token_ids_sha256"],
                    "slot_indices": list(capture.slot_indices),
                    "positive_minus_negative_log_odds": (
                        capture.positive_minus_negative_log_odds
                    ),
                    "residuals_float32_sha256": tensor_float32_sha256(
                        capture.residuals
                    ),
                    "gradients_float32_sha256": tensor_float32_sha256(
                        capture.gradients
                    ),
                    "per_layer_residual_hashes": [
                        tensor_float32_sha256(capture.residuals[layer])
                        for layer in range(LAYER_COUNT)
                    ],
                    "per_layer_gradient_hashes": [
                        tensor_float32_sha256(capture.gradients[layer])
                        for layer in range(LAYER_COUNT)
                    ],
                    "full_logits_float32_sha256": tensor_float32_sha256(
                        capture.full_logits
                    ),
                    "capture_audit": _plain(capture.audit),
                },
                "row_sha256",
            )
            capture_records.append(row)
            print(f"ALFS capture {index + 1}/80 {source['form_id']}", flush=True)
        residual_tensor = torch.stack(residuals).float().contiguous()
        gradient_tensor = torch.stack(gradients).float().contiguous()
        if tuple(residual_tensor.shape) != (80, 24, 4, 1024) or tuple(
            gradient_tensor.shape
        ) != (80, 24, 4, 1024):
            raise ALFSIntegrityError("ALFS aggregate tensor shape differs")
        alignment = _capture_alignment(
            source_records=state["records"],
            csms_records=csms_metadata["records"],
            capture_records=capture_records,
            residuals=residual_tensor,
            gradients=gradient_tensor,
        )
        capture_metadata = {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "status": "complete",
            "split": split,
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "source_binding": lock["configuration"]["source_binding"],
            "record_count": 80,
            "records": capture_records,
            "row_alignment_manifest": alignment,
            "compute": COMPUTE_CEILING,
            "offline_environment": observed_offline,
        }
        _csms()._base()._save_checkpoint(
            torch,
            path=CAPTURE_PATH,
            metadata=capture_metadata,
            tensors={"residuals": residual_tensor, "gradients": gradient_tensor},
        )
    except Exception as error:
        raise ALFSIntegrityError("ALFS capture failed after immutable reservation") from error
    complete = _with_hash(
        {
            "schema_version": COMPLETION_SCHEMA,
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
    if not CAPTURE_RESERVATION_PATH.is_file() or not CAPTURE_COMPLETE_PATH.is_file():
        raise ALFSIntegrityError("ALFS capture lacks reservation or completion")
    reservation = _load_json(CAPTURE_RESERVATION_PATH)
    complete = _load_json(CAPTURE_COMPLETE_PATH)
    _verify_self_hash(reservation, "reservation_sha256")
    _verify_self_hash(complete, "completion_sha256")
    if (
        reservation.get("schema_version") != RESERVATION_SCHEMA
        or reservation.get("status") != "reserved_before_first_model_forward"
        or reservation.get("split") != "opened_development"
        or reservation.get("lock_identity_sha256") != lock["lock_identity_sha256"]
        or {key: reservation.get(key) for key in COMPUTE_CEILING} != COMPUTE_CEILING
        or reservation.get("offline_environment") != OFFLINE_ENVIRONMENT
        or reservation.get("source_csms_capture_file_sha256")
        != file_sha256(CSMS_CAPTURE_PATH)
        or complete.get("schema_version") != COMPLETION_SCHEMA
        or complete.get("status") != "complete"
        or complete.get("lock_identity_sha256") != lock["lock_identity_sha256"]
        or complete.get("capture_file_sha256") != file_sha256(CAPTURE_PATH)
        or complete.get("reservation_file_sha256")
        != file_sha256(CAPTURE_RESERVATION_PATH)
    ):
        raise ALFSIntegrityError("ALFS capture reservation/completion binding differs")
    metadata, tensors = _csms()._base()._load_checkpoint(
        torch, path=CAPTURE_PATH, schema=CAPTURE_SCHEMA_VERSION
    )
    csms_metadata, _, state = _csms()._load_capture(torch)
    if (
        metadata.get("schema_version") != CAPTURE_SCHEMA_VERSION
        or metadata.get("status") != "complete"
        or metadata.get("split") != "opened_development"
        or metadata.get("lock_identity_sha256") != lock["lock_identity_sha256"]
        or metadata.get("source_binding") != lock["configuration"]["source_binding"]
        or metadata.get("record_count") != 80
        or metadata.get("compute") != COMPUTE_CEILING
        or metadata.get("offline_environment") != OFFLINE_ENVIRONMENT
        or set(tensors) != {"residuals", "gradients"}
        or tuple(tensors["residuals"].shape) != (80, 24, 4, 1024)
        or tuple(tensors["gradients"].shape) != (80, 24, 4, 1024)
        or tensors["residuals"].dtype != torch.float32
        or tensors["gradients"].dtype != torch.float32
    ):
        raise ALFSIntegrityError("ALFS capture metadata/tensor contract differs")
    records = metadata.get("records")
    if not isinstance(records, list) or len(records) != 80:
        raise ALFSIntegrityError("ALFS capture record coverage differs")
    forwards = 0
    backwards = 0
    for index, row in enumerate(records):
        _verify_self_hash(row, "row_sha256")
        audit = row.get("capture_audit")
        if not isinstance(audit, Mapping):
            raise ALFSIntegrityError("ALFS capture row lacks audit")
        _verify_self_hash(audit, "audit_sha256")
        if (
            row.get("tensor_index") != index
            or len(row.get("per_layer_residual_hashes", ())) != 24
            or len(row.get("per_layer_gradient_hashes", ())) != 24
            or audit.get("layers") != list(range(24))
            or audit.get("hook_call_counts") != {str(layer): 1 for layer in range(24)}
            or audit.get("model_forward_evaluations") != 1
            or audit.get("model_backward_evaluations") != 1
            or audit.get("model_parameters_requires_grad_disabled_during_capture") is not True
            or audit.get("model_parameter_requires_grad_flags_restored_after_capture") is not True
            or audit.get("model_parameter_gradients_allocated") is not False
            or audit.get("maximum_abs_layer0_reconstruction_delta") != 0.0
            or audit.get("later_layer_hooks_return_activation_unchanged") is not True
            or any(
                audit.get(field) is not True
                for field in (
                    "source_layer0_residuals_reproduced",
                    "source_layer0_gradients_reproduced",
                    "source_full_logits_reproduced",
                    "source_margin_reproduced",
                    "source_tokenization_reproduced",
                )
            )
            or tensor_float32_sha256(tensors["residuals"][index])
            != row["residuals_float32_sha256"]
            or tensor_float32_sha256(tensors["gradients"][index])
            != row["gradients_float32_sha256"]
        ):
            raise ALFSIntegrityError("one ALFS capture row audit differs")
        for layer in range(24):
            if (
                tensor_float32_sha256(tensors["residuals"][index, layer])
                != row["per_layer_residual_hashes"][layer]
                or tensor_float32_sha256(tensors["gradients"][index, layer])
                != row["per_layer_gradient_hashes"][layer]
            ):
                raise ALFSIntegrityError("one ALFS per-layer tensor hash differs")
        forwards += 1
        backwards += 1
    if forwards != 80 or backwards != 80:
        raise ALFSIntegrityError("ALFS audited compute does not sum to 80F/80B")
    alignment = _capture_alignment(
        source_records=state["records"],
        csms_records=csms_metadata["records"],
        capture_records=records,
        residuals=tensors["residuals"],
        gradients=tensors["gradients"],
    )
    if alignment != metadata.get("row_alignment_manifest"):
        raise ALFSIntegrityError("ALFS row alignment manifest differs")
    return metadata, tensors, state


def _fold_training_path(index: int) -> Path:
    return TRAINING_ROOT / f"fold_{index}.json"


def _fold_training_freeze_path(index: int) -> Path:
    return TRAINING_ROOT / f"fold_{index}_freeze.pt"


def _fold_held_path(index: int) -> Path:
    return HELD_ROOT / f"fold_{index}.json"


def _compute_training_fold(
    *,
    records: Sequence[Mapping[str, Any]],
    residuals: Any,
    gradients: Any,
    fold: Mapping[str, Any],
) -> tuple[dict[str, Any], Any | None]:
    import numpy as np

    raw_residuals = np.asarray(residuals)
    raw_gradients = np.asarray(gradients)
    if raw_residuals.shape != raw_gradients.shape or raw_residuals.shape[:3] != (
        80,
        LAYER_COUNT,
        4,
    ):
        raise ALFSIntegrityError("ALFS training tensors have invalid shape")
    training_indices = tuple(int(value) for value in fold["training_all_indices"])
    masked_residuals = np.full(raw_residuals.shape, np.nan, dtype=np.float64)
    masked_gradients = np.full(raw_gradients.shape, np.nan, dtype=np.float64)
    masked_residuals[list(training_indices)] = raw_residuals[list(training_indices)]
    masked_gradients[list(training_indices)] = raw_gradients[list(training_indices)]
    scales = training_only_slot_scales(masked_residuals, training_indices)
    candidates = []
    for layer in range(LAYER_COUNT):
        record, _ = analyze_training_layer(
            records=records,
            residuals_at_layer=masked_residuals[:, layer],
            gradients_at_layer=masked_gradients[:, layer],
            slot_scales=scales[layer],
            training_target_indices=fold["training_target_indices"],
            training_nuisance_indices=fold["training_nuisance_indices"],
            training_all_indices=fold["training_all_indices"],
            layer=layer,
        )
        candidates.append(record)
    selection = select_layer(candidates)
    detail = None
    selected_basis = None
    if selection["passes"]:
        layer = int(selection["selected_layer"])
        detail = candidates[layer]
        nuisance_indices = tuple(
            int(value) for value in fold["training_nuisance_indices"]
        )
        nuisance_rows = (
            masked_gradients[list(nuisance_indices), layer]
            * scales[layer][None, :, None]
        ).reshape(len(nuisance_indices), -1)
        selected_basis, selected_basis_record = frozen_nuisance_rowspace(
            nuisance_rows
        )
        if selected_basis_record != detail["training_nuisance_rowspace"]:
            raise ALFSIntegrityError(
                "selected candidate differs from its frozen nuisance basis"
            )
    training = {
        "fold": fold,
        "training_only_slot_scales": scales.tolist(),
        "training_only_slot_scales_sha256": canonical_sha256(scales.tolist()),
        "held_numeric_rows_used_in_training_analysis": False,
        "layer_candidates": candidates,
        "selection": selection,
        "selected_layer_detail": detail,
        "passes": bool(selection["passes"]),
    }
    return training, selected_basis


def run_analyze_training(split: str = "opened_development") -> dict[str, Any]:
    require_opened_development_split(split)
    lock = _load_lock()
    targets = [
        *(_fold_training_path(index) for index in range(4)),
        *(_fold_training_freeze_path(index) for index in range(4)),
        TRAINING_COMPLETE_PATH,
    ]
    if any(path.exists() for path in targets):
        raise FileExistsError("refusing to overwrite or resume partial ALFS training artifacts")
    import torch

    capture, tensors, state = _load_capture(torch)
    folds = build_outer_folds(state["records"])
    records = []
    for fold in folds:
        training, frozen_basis = _compute_training_fold(
            records=state["records"],
            residuals=tensors["residuals"].numpy(),
            gradients=tensors["gradients"].numpy(),
            fold=fold,
        )
        artifact = _with_hash(
            {
                "schema_version": f"{ANALYSIS_SCHEMA_VERSION}.training_fold",
                "status": "selected" if training["passes"] else "no_go",
                "split": split,
                "lock_identity_sha256": lock["lock_identity_sha256"],
                "capture_file_sha256": file_sha256(CAPTURE_PATH),
                "capture_checkpoint_sha256": capture["checkpoint_sha256"],
                **training,
            },
            "training_fold_sha256",
        )
        path = _fold_training_path(int(fold["fold_index"]))
        _write_new_json(path, artifact)
        freeze_record = None
        if training["passes"]:
            if frozen_basis is None:
                raise ALFSIntegrityError("selected ALFS fold has no frozen basis")
            fold_index = int(fold["fold_index"])
            freeze_path = _fold_training_freeze_path(fold_index)
            scales = training["training_only_slot_scales"][
                int(training["selection"]["selected_layer"])
            ]
            basis_identity = _float64_array_identity(frozen_basis)
            if (
                basis_identity
                != training["selected_layer_detail"]["training_nuisance_rowspace"][
                    "basis_identity"
                ]
            ):
                raise ALFSIntegrityError("persisted ALFS basis identity differs")
            _csms()._base()._save_checkpoint(
                torch,
                path=freeze_path,
                metadata={
                    "schema_version": TRAINING_FREEZE_SCHEMA,
                    "status": "frozen_before_held_derived_computation",
                    "split": split,
                    "lock_identity_sha256": lock["lock_identity_sha256"],
                    "capture_file_sha256": file_sha256(CAPTURE_PATH),
                    "capture_checkpoint_sha256": capture["checkpoint_sha256"],
                    "training_fold_file_sha256": file_sha256(path),
                    "fold_index": fold_index,
                    "selected_layer": int(training["selection"]["selected_layer"]),
                    "frozen_training_nuisance_basis_identity": basis_identity,
                    "training_only_slot_scales_identity": _float64_array_identity(scales),
                },
                tensors={
                    "frozen_training_nuisance_basis": torch.from_numpy(
                        frozen_basis.copy()
                    ).to(dtype=torch.float64).contiguous(),
                    "training_only_slot_scales": torch.tensor(
                        scales, dtype=torch.float64
                    ).contiguous(),
                },
            )
            freeze_metadata, _ = _csms()._base()._load_checkpoint(
                torch,
                path=freeze_path,
                schema=TRAINING_FREEZE_SCHEMA,
            )
            freeze_record = {
                "path": _relative(freeze_path),
                "sha256": file_sha256(freeze_path),
                "checkpoint_sha256": freeze_metadata["checkpoint_sha256"],
            }
        records.append(
            {
                "path": _relative(path),
                "sha256": file_sha256(path),
                "frozen_basis_checkpoint": freeze_record,
            }
        )
    complete = _with_hash(
        {
            "schema_version": f"{ANALYSIS_SCHEMA_VERSION}.training_complete",
            "status": "complete",
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "capture_file_sha256": file_sha256(CAPTURE_PATH),
            "fold_artifacts": records,
            "all_four_written_before_any_held_derived_computation": True,
            "model_forwards": 0,
            "model_backwards": 0,
        },
        "training_complete_sha256",
    )
    _write_new_json(TRAINING_COMPLETE_PATH, complete)
    return complete


def _load_training_freeze(
    torch: Any,
    *,
    index: int,
    artifact: Mapping[str, Any],
) -> tuple[dict[str, Any], Any, Any]:
    import numpy as np

    path = _fold_training_freeze_path(index)
    metadata, tensors = _csms()._base()._load_checkpoint(
        torch,
        path=path,
        schema=TRAINING_FREEZE_SCHEMA,
    )
    if set(tensors) != {
        "frozen_training_nuisance_basis",
        "training_only_slot_scales",
    }:
        raise ALFSIntegrityError("ALFS training freeze tensor names differ")
    basis_tensor = tensors["frozen_training_nuisance_basis"]
    scales_tensor = tensors["training_only_slot_scales"]
    basis = basis_tensor.numpy()
    scales = scales_tensor.numpy()
    selected_layer = int(artifact["selection"]["selected_layer"])
    expected_basis_identity = artifact["selected_layer_detail"][
        "training_nuisance_rowspace"
    ]["basis_identity"]
    expected_scales = np.asarray(
        artifact["training_only_slot_scales"][selected_layer], dtype=np.float64
    )
    if (
        metadata.get("status") != "frozen_before_held_derived_computation"
        or metadata.get("split") != "opened_development"
        or metadata.get("lock_identity_sha256")
        != artifact["lock_identity_sha256"]
        or metadata.get("capture_file_sha256") != file_sha256(CAPTURE_PATH)
        or metadata.get("capture_checkpoint_sha256")
        != artifact["capture_checkpoint_sha256"]
        or metadata.get("training_fold_file_sha256")
        != file_sha256(_fold_training_path(index))
        or metadata.get("fold_index") != index
        or metadata.get("selected_layer") != selected_layer
        or basis_tensor.dtype != torch.float64
        or scales_tensor.dtype != torch.float64
        or basis.ndim != 2
        or basis.shape[1] != 4 * MODEL["d_model"]
        or scales.shape != (4,)
        or not np.isfinite(basis).all()
        or not np.isfinite(scales).all()
        or bool(np.any(scales <= 0.0))
        or _float64_array_identity(basis) != expected_basis_identity
        or metadata.get("frozen_training_nuisance_basis_identity")
        != expected_basis_identity
        or not np.array_equal(scales, expected_scales)
        or metadata.get("training_only_slot_scales_identity")
        != _float64_array_identity(scales)
    ):
        raise ALFSIntegrityError("ALFS persisted training freeze differs")
    return metadata, basis, scales


def _verify_selected_candidate_binding(artifact: Mapping[str, Any]) -> None:
    candidates = artifact.get("layer_candidates")
    if not isinstance(candidates, list):
        raise ALFSIntegrityError("ALFS training artifact lacks layer candidates")
    selection = artifact.get("selection")
    if selection != select_layer(candidates):
        raise ALFSIntegrityError("ALFS training selector differs from candidates")
    detail = artifact.get("selected_layer_detail")
    if selection["passes"]:
        selected_layer = int(selection["selected_layer"])
        if detail != candidates[selected_layer]:
            raise ALFSIntegrityError("selected ALFS detail differs from candidate")
    elif detail is not None:
        raise ALFSIntegrityError("no-go ALFS selector has selected detail")


def _load_training_artifacts(lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not TRAINING_COMPLETE_PATH.is_file():
        raise ALFSIntegrityError("ALFS held analysis requires completed training artifacts")
    complete = _load_json(TRAINING_COMPLETE_PATH)
    _verify_self_hash(complete, "training_complete_sha256")
    if (
        complete.get("schema_version") != f"{ANALYSIS_SCHEMA_VERSION}.training_complete"
        or complete.get("status") != "complete"
        or complete.get("lock_identity_sha256") != lock["lock_identity_sha256"]
        or complete.get("capture_file_sha256") != file_sha256(CAPTURE_PATH)
        or complete.get("all_four_written_before_any_held_derived_computation") is not True
        or complete.get("model_forwards") != 0
        or complete.get("model_backwards") != 0
    ):
        raise ALFSIntegrityError("ALFS training completion differs")
    file_records = complete.get("fold_artifacts")
    if not isinstance(file_records, list) or len(file_records) != 4:
        raise ALFSIntegrityError("ALFS requires exactly four training file records")
    import torch

    values = []
    for index, file_record in enumerate(file_records):
        path = _fold_training_path(index)
        if (
            not isinstance(file_record, Mapping)
            or file_record.get("path") != _relative(path)
            or file_record.get("sha256") != file_sha256(path)
        ):
            raise ALFSIntegrityError("one ALFS training artifact file binding differs")
        artifact = _load_json(path)
        _verify_self_hash(artifact, "training_fold_sha256")
        if (
            artifact.get("schema_version") != f"{ANALYSIS_SCHEMA_VERSION}.training_fold"
            or artifact.get("lock_identity_sha256") != lock["lock_identity_sha256"]
            or artifact.get("capture_file_sha256") != file_sha256(CAPTURE_PATH)
            or artifact.get("held_numeric_rows_used_in_training_analysis") is not False
            or artifact.get("fold", {}).get("fold_index") != index
            or artifact.get("passes") is not (
                artifact.get("selection", {}).get("passes") is True
            )
        ):
            raise ALFSIntegrityError("one ALFS training fold contract differs")
        _verify_selected_candidate_binding(artifact)
        if artifact.get("status") != (
            "selected" if artifact["passes"] else "no_go"
        ):
            raise ALFSIntegrityError("one ALFS training selector differs")
        if artifact["passes"]:
            freeze_path = _fold_training_freeze_path(index)
            freeze_metadata, _, _ = _load_training_freeze(
                torch,
                index=index,
                artifact=artifact,
            )
            if file_record.get("frozen_basis_checkpoint") != {
                "path": _relative(freeze_path),
                "sha256": file_sha256(freeze_path),
                "checkpoint_sha256": freeze_metadata["checkpoint_sha256"],
            }:
                raise ALFSIntegrityError("one ALFS frozen basis file binding differs")
        elif (
            file_record.get("frozen_basis_checkpoint") is not None
            or _fold_training_freeze_path(index).exists()
        ):
            raise ALFSIntegrityError("no-go ALFS fold unexpectedly has a frozen basis")
        values.append(artifact)
    if len(values) != 4:
        raise ALFSIntegrityError("ALFS requires exactly four training artifacts")
    return values


def run_analyze_held(split: str = "opened_development") -> dict[str, Any]:
    require_opened_development_split(split)
    lock = _load_lock()
    targets = [*(_fold_held_path(index) for index in range(4)), HELD_COMPLETE_PATH]
    if any(path.exists() for path in targets):
        raise FileExistsError("refusing to overwrite or resume partial ALFS held artifacts")
    training_artifacts = _load_training_artifacts(lock)
    import torch

    capture, tensors, state = _load_capture(torch)
    folds = build_outer_folds(state["records"])
    records = []
    for fold, training in zip(folds, training_artifacts, strict=True):
        if training["fold"] != fold:
            raise ALFSIntegrityError("frozen training fold semantics differ")
        selection = training["selection"]
        held = None
        if selection["passes"]:
            layer = int(selection["selected_layer"])
            scales = training["training_only_slot_scales"][layer]
            _, frozen_basis, frozen_scales = _load_training_freeze(
                torch,
                index=int(fold["fold_index"]),
                artifact=training,
            )
            if frozen_scales.tolist() != scales:
                raise ALFSIntegrityError("held scales differ from persisted training scales")
            held, _ = evaluate_held_oracles(
                records=state["records"],
                residuals_at_layer=tensors["residuals"][:, layer].numpy(),
                gradients_at_layer=tensors["gradients"][:, layer].numpy(),
                slot_scales=scales,
                training_nuisance_indices=fold["training_nuisance_indices"],
                held_target_indices=fold["held_target_indices"],
                held_nuisance_indices=fold["held_nuisance_indices"],
                layer=layer,
                frozen_training_basis=frozen_basis,
            )
            training_basis = training["selected_layer_detail"][
                "training_nuisance_rowspace"
            ]["basis_identity"]
            if (
                held["frozen_training_nuisance_rowspace"]["basis_identity"]
                != training_basis
                or held.get("persisted_frozen_training_basis_reused") is not True
            ):
                raise ALFSIntegrityError("held oracle did not reuse frozen training rowspace")
        artifact = _with_hash(
            {
                "schema_version": f"{ANALYSIS_SCHEMA_VERSION}.held_fold",
                "status": "pass" if held is not None and held["passes"] else "no_go",
                "split": split,
                "lock_identity_sha256": lock["lock_identity_sha256"],
                "capture_file_sha256": file_sha256(CAPTURE_PATH),
                "capture_checkpoint_sha256": capture["checkpoint_sha256"],
                "training_fold_file_sha256": file_sha256(
                    _fold_training_path(int(fold["fold_index"]))
                ),
                "training_selection_frozen_before_held_derived_computation": True,
                "fold": fold,
                "selection": selection,
                "held_oracle": held,
                "passes": bool(held is not None and held["passes"]),
            },
            "held_fold_sha256",
        )
        path = _fold_held_path(int(fold["fold_index"]))
        _write_new_json(path, artifact)
        records.append({"path": _relative(path), "sha256": file_sha256(path)})
    complete = _with_hash(
        {
            "schema_version": f"{ANALYSIS_SCHEMA_VERSION}.held_complete",
            "status": "complete",
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "capture_file_sha256": file_sha256(CAPTURE_PATH),
            "training_complete_file_sha256": file_sha256(TRAINING_COMPLETE_PATH),
            "fold_artifacts": records,
            "model_forwards": 0,
            "model_backwards": 0,
        },
        "held_complete_sha256",
    )
    _write_new_json(HELD_COMPLETE_PATH, complete)
    return complete


def _verify_held_training_binding(
    held: Mapping[str, Any],
    training: Mapping[str, Any],
    training_path: Path,
) -> None:
    if (
        held.get("training_fold_file_sha256") != file_sha256(training_path)
        or held.get("fold") != training.get("fold")
        or held.get("selection") != training.get("selection")
    ):
        raise ALFSIntegrityError("held artifact differs from its frozen training artifact")


def _load_held_artifacts(
    lock: Mapping[str, Any],
    training_artifacts: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not HELD_COMPLETE_PATH.is_file():
        raise ALFSIntegrityError("full selector requires completed held artifacts")
    complete = _load_json(HELD_COMPLETE_PATH)
    _verify_self_hash(complete, "held_complete_sha256")
    if (
        complete.get("schema_version") != f"{ANALYSIS_SCHEMA_VERSION}.held_complete"
        or complete.get("status") != "complete"
        or complete.get("lock_identity_sha256") != lock["lock_identity_sha256"]
        or complete.get("capture_file_sha256") != file_sha256(CAPTURE_PATH)
        or complete.get("training_complete_file_sha256")
        != file_sha256(TRAINING_COMPLETE_PATH)
    ):
        raise ALFSIntegrityError("ALFS held completion differs")
    frozen_training = (
        _load_training_artifacts(lock)
        if training_artifacts is None
        else list(training_artifacts)
    )
    if len(frozen_training) != 4:
        raise ALFSIntegrityError("held loading requires four frozen training artifacts")
    values = []
    for index, file_record in enumerate(complete.get("fold_artifacts", ())):
        path = _fold_held_path(index)
        if file_record != {"path": _relative(path), "sha256": file_sha256(path)}:
            raise ALFSIntegrityError("one ALFS held artifact file binding differs")
        artifact = _load_json(path)
        _verify_self_hash(artifact, "held_fold_sha256")
        if (
            artifact.get("schema_version") != f"{ANALYSIS_SCHEMA_VERSION}.held_fold"
            or artifact.get("lock_identity_sha256") != lock["lock_identity_sha256"]
            or artifact.get("capture_file_sha256") != file_sha256(CAPTURE_PATH)
            or artifact.get("fold", {}).get("fold_index") != index
            or artifact.get("training_selection_frozen_before_held_derived_computation")
            is not True
        ):
            raise ALFSIntegrityError("one ALFS held fold contract differs")
        _verify_held_training_binding(
            artifact,
            frozen_training[index],
            _fold_training_path(index),
        )
        selection_passes = artifact.get("selection", {}).get("passes") is True
        oracle = artifact.get("held_oracle")
        expected_pass = bool(
            selection_passes
            and isinstance(oracle, Mapping)
            and oracle.get("passes") is True
        )
        if (
            artifact.get("passes") is not expected_pass
            or artifact.get("status") != ("pass" if expected_pass else "no_go")
            or (selection_passes and not isinstance(oracle, Mapping))
            or (not selection_passes and oracle is not None)
        ):
            raise ALFSIntegrityError("one ALFS held result contract differs")
        if selection_passes:
            training_basis = frozen_training[index]["selected_layer_detail"][
                "training_nuisance_rowspace"
            ]["basis_identity"]
            if (
                oracle.get("layer")
                != artifact["selection"].get("selected_layer")
                or oracle.get("persisted_frozen_training_basis_reused") is not True
                or oracle.get("frozen_training_nuisance_rowspace", {}).get(
                    "basis_identity"
                )
                != training_basis
            ):
                raise ALFSIntegrityError("one ALFS held frozen basis contract differs")
        values.append(artifact)
    if len(values) != 4:
        raise ALFSIntegrityError("ALFS requires exactly four held artifacts")
    return values


def _global_qualification(
    training: Sequence[Mapping[str, Any]],
    held: Sequence[Mapping[str, Any]],
    full_selection: Mapping[str, Any],
) -> tuple[list[int], dict[str, bool], bool]:
    fold_layers = [
        int(artifact["selection"]["selected_layer"])
        for artifact in training
        if artifact.get("selection", {}).get("passes") is True
    ]
    same_layer = bool(
        len(fold_layers) == 4
        and full_selection.get("passes") is True
        and len({*fold_layers, int(full_selection["selected_layer"])}) == 1
    )
    checks = {
        "all_four_training_selectors_pass": len(training) == 4
        and all(value.get("passes") is True for value in training),
        "all_four_held_folds_pass": len(held) == 4
        and all(value.get("passes") is True for value in held),
        "full_data_selector_passes": full_selection.get("passes") is True,
        "all_fold_and_full_data_layers_identical": same_layer,
        "sealed_data_not_accessed": True,
    }
    return fold_layers, checks, bool(all(checks.values()))


def run_analyze_full(split: str = "opened_development") -> dict[str, Any]:
    require_opened_development_split(split)
    lock = _load_lock()
    if RESULT_PATH.exists():
        raise FileExistsError("refusing to overwrite immutable ALFS result")
    training = _load_training_artifacts(lock)
    held = _load_held_artifacts(lock, training)
    import torch

    capture, tensors, state = _load_capture(torch)
    all_indices = tuple(range(80))
    folds = build_outer_folds(state["records"])
    target_indices = tuple(
        index
        for fold in folds
        for index in fold["held_target_indices"]
    )
    target_indices = tuple(sorted(set(target_indices)))
    nuisance_indices = tuple(index for index in all_indices if index not in target_indices)
    if len(target_indices) != 16 or len(nuisance_indices) != 64:
        raise ALFSIntegrityError("full-data target/nuisance coverage differs")
    scales = training_only_slot_scales(tensors["residuals"].numpy(), all_indices)
    candidates = []
    for layer in range(LAYER_COUNT):
        candidate, _ = analyze_training_layer(
            records=state["records"],
            residuals_at_layer=tensors["residuals"][:, layer].numpy(),
            gradients_at_layer=tensors["gradients"][:, layer].numpy(),
            slot_scales=scales[layer],
            training_target_indices=target_indices,
            training_nuisance_indices=nuisance_indices,
            training_all_indices=all_indices,
            layer=layer,
        )
        candidates.append(candidate)
    selection = select_layer(candidates)
    detail = None
    if selection["passes"]:
        layer = int(selection["selected_layer"])
        detail = candidates[layer]
    fold_layers, checks, passes = _global_qualification(training, held, selection)
    result = _with_hash(
        {
            "schema_version": RESULT_SCHEMA,
            "status": "go_coordinate_only" if passes else "no_go",
            "split": split,
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "capture_file_sha256": file_sha256(CAPTURE_PATH),
            "capture_checkpoint_sha256": capture["checkpoint_sha256"],
            "training_complete_file_sha256": file_sha256(TRAINING_COMPLETE_PATH),
            "held_complete_file_sha256": file_sha256(HELD_COMPLETE_PATH),
            "training_fold_files": [
                {"path": _relative(_fold_training_path(index)), "sha256": file_sha256(_fold_training_path(index))}
                for index in range(4)
            ],
            "held_fold_files": [
                {"path": _relative(_fold_held_path(index)), "sha256": file_sha256(_fold_held_path(index))}
                for index in range(4)
            ],
            "full_data_scales": scales.tolist(),
            "full_data_layer_candidates": candidates,
            "full_data_selection": selection,
            "full_data_selected_layer_detail": detail,
            "fold_selected_layers": fold_layers,
            "checks": checks,
            "passes": passes,
            "finite_intervention_authorized": False,
            "next_authorized_action": (
                "write_separate_prospective_controller_protocol"
                if passes
                else "stop_all_layer_four_slot_search"
            ),
            "claim_boundary": (
                "transductive local controllability oracle only; not a controller, finite "
                "behavioral result, natural mechanism, safety result, or novelty claim"
            ),
            "sealed_data_accessed": False,
            "model_forwards_after_capture": 0,
            "model_backwards_after_capture": 0,
        },
        "result_sha256",
    )
    _write_new_json(RESULT_PATH, result)
    return result


def run_analyze(split: str = "opened_development") -> dict[str, Any]:
    require_opened_development_split(split)
    if not TRAINING_COMPLETE_PATH.exists():
        run_analyze_training(split)
    if not HELD_COMPLETE_PATH.exists():
        run_analyze_held(split)
    return run_analyze_full(split)


def _validate_result_contract(
    result: Mapping[str, Any],
    *,
    lock: Mapping[str, Any],
    capture_checkpoint_sha256: str,
    training: Sequence[Mapping[str, Any]],
    held: Sequence[Mapping[str, Any]],
) -> None:
    full_candidates = result.get("full_data_layer_candidates")
    if not isinstance(full_candidates, list):
        raise ALFSIntegrityError("ALFS result lacks full-data layer candidates")
    recomputed_selection = select_layer(full_candidates)
    selection = result.get("full_data_selection")
    if selection != recomputed_selection:
        raise ALFSIntegrityError("ALFS result full-data selection differs")
    detail = result.get("full_data_selected_layer_detail")
    if selection["passes"]:
        selected_layer = int(selection["selected_layer"])
        if detail != full_candidates[selected_layer]:
            raise ALFSIntegrityError("ALFS full-data detail differs from selected candidate")
    elif detail is not None:
        raise ALFSIntegrityError("no-go ALFS full selector has selected detail")
    fold_layers, checks, passes = _global_qualification(training, held, selection)
    expected_status = "go_coordinate_only" if passes else "no_go"
    expected_next = (
        "write_separate_prospective_controller_protocol"
        if passes
        else "stop_all_layer_four_slot_search"
    )
    expected_training_files = [
        {
            "path": _relative(_fold_training_path(index)),
            "sha256": file_sha256(_fold_training_path(index)),
        }
        for index in range(4)
    ]
    expected_held_files = [
        {
            "path": _relative(_fold_held_path(index)),
            "sha256": file_sha256(_fold_held_path(index)),
        }
        for index in range(4)
    ]
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("status") != expected_status
        or result.get("split") != "opened_development"
        or result.get("lock_identity_sha256") != lock["lock_identity_sha256"]
        or result.get("capture_file_sha256") != file_sha256(CAPTURE_PATH)
        or result.get("capture_checkpoint_sha256") != capture_checkpoint_sha256
        or result.get("training_complete_file_sha256")
        != file_sha256(TRAINING_COMPLETE_PATH)
        or result.get("held_complete_file_sha256")
        != file_sha256(HELD_COMPLETE_PATH)
        or result.get("training_fold_files") != expected_training_files
        or result.get("held_fold_files") != expected_held_files
        or result.get("fold_selected_layers") != fold_layers
        or result.get("checks") != checks
        or result.get("passes") is not passes
        or result.get("finite_intervention_authorized") is not False
        or result.get("next_authorized_action") != expected_next
        or result.get("sealed_data_accessed") is not False
        or result.get("model_forwards_after_capture") != 0
        or result.get("model_backwards_after_capture") != 0
    ):
        raise ALFSIntegrityError("ALFS result provenance or qualification differs")


def _load_validated_result() -> dict[str, Any]:
    lock = _load_lock()
    import torch

    capture, _, _ = _load_capture(torch)
    training = _load_training_artifacts(lock)
    held = _load_held_artifacts(lock, training)
    result = _load_json(RESULT_PATH)
    _verify_self_hash(result, "result_sha256")
    _validate_result_contract(
        result,
        lock=lock,
        capture_checkpoint_sha256=capture["checkpoint_sha256"],
        training=training,
        held=held,
    )
    return result


def run_report(split: str = "opened_development") -> str:
    require_opened_development_split(split)
    if REPORT_PATH.exists():
        raise FileExistsError("refusing to overwrite immutable ALFS report")
    result = _load_validated_result()
    selected = result["full_data_selection"].get("selected_layer")
    lines = [
        "# All-Layer Four-Slot Oracle Screen",
        "",
        f"Status: **{result['status']}**",
        "",
        f"Full-data selected layer: `{selected}`",
        "",
        "ALFS is a transductive local-controllability oracle. Held target gradients were",
        "used to construct held directions; this is not a deployable context controller.",
        "No finite intervention, generated output, sealed access, capability claim, safety",
        "claim, natural-mechanism claim, novelty claim, or publication claim is authorized.",
        "",
        "## Qualification checks",
        "",
        *[
            f"- {'PASS' if passed else 'FAIL'}: `{name}`"
            for name, passed in result["checks"].items()
        ],
        "",
        f"Result SHA-256: `{result['result_sha256']}`",
        "",
    ]
    text = "\n".join(lines)
    _write_new_text(REPORT_PATH, text)
    return text


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "lock",
            "preflight",
            "capture",
            "analyze-training",
            "analyze-held",
            "analyze-full",
            "analyze",
            "report",
        ),
    )
    parser.add_argument("--split", default="opened_development")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dispatch = {
        "lock": run_lock,
        "preflight": run_preflight,
        "capture": run_capture,
        "analyze-training": run_analyze_training,
        "analyze-held": run_analyze_held,
        "analyze-full": run_analyze_full,
        "analyze": run_analyze,
        "report": run_report,
    }
    value = dispatch[args.command](args.split)
    if isinstance(value, str):
        print(value)
    else:
        print(json.dumps(_plain(value), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
