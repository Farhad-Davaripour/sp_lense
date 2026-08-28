from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts import suffix_transport_feasibility as base
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    import suffix_transport_feasibility as base

from sp_lense.causal_anchor_runtime import (
    anchor_residual_scale_geometric_mean,
    capture_multilayer_choice_anchor_gradient,
    resolve_shared_anchor_evidence,
)
from sp_lense.factorial_causal_anchor import (
    canonical_sha256,
    cell_key,
    render_choice_form,
    render_construction_form,
    tensor_float32_sha256,
)
from sp_lense.suffix_transport import (
    SuffixTransportIneligible,
    leave_one_scenario_out_cell_interface_translation,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
DATA_PATH = base.DATA_PATH
MODEL_CONFIG_PATH = base.MODEL_CONFIG_PATH
LOCK_PATH = ROOT / "configs" / "factorial_interface_translator_development_lock.json"
PROTOCOL_PATH = ROOT / "docs" / "PROTECTED_FACTORIAL_INTERFACE_TRANSLATOR.md"
MATH_PATH = ROOT / "src" / "sp_lense" / "suffix_transport.py"
BASE_RUNNER_PATH = ROOT / "scripts" / "suffix_transport_feasibility.py"
REQUIREMENTS_PATH = ROOT / "requirements-research.txt"

FCAGS_LOCK_PATH = base.FCAGS_LOCK_PATH
FCAGS_CAPTURE_PATH = base.FCAGS_CAPTURE_PATH
FCAGS_CAPTURE_MANIFEST_PATH = base.FCAGS_CAPTURE_MANIFEST_PATH
STFG_LOCK_PATH = base.LOCK_PATH
STFG_SP_CAPTURE_PATH = base.CHOICE_CAPTURE_PATH
STFG_SP_CAPTURE_MANIFEST_PATH = base.CHOICE_CAPTURE_MANIFEST_PATH

ARTIFACT_ROOT = ROOT / "artifacts" / "factorial_interface_translator_development" / "qwen35_08b"
RESULT_ROOT = ROOT / "results" / "factorial_interface_translator_development" / "qwen35_08b"
PREFLIGHT_PATH = ARTIFACT_ROOT / "preflight.json"
CAPTURE_PATH = ARTIFACT_ROOT / "complete_cell_choice_capture.pt"
CAPTURE_MANIFEST_PATH = ARTIFACT_ROOT / "complete_cell_choice_capture_manifest.json"
RESULT_PATH = RESULT_ROOT / "geometric_result.json"
REPORT_PATH = RESULT_ROOT / "GEOMETRIC_REPORT.md"

LOCK_SCHEMA = "sp_lense.factorial_interface_translator_development_lock.v1"
LAYER = 22
LABELS = ("A", "B")
CELL_ORDER = ("SP", "OP", "ST", "OT")
CELL_FACTORS = {
    "SP": ("self", "permanent"),
    "OP": ("other", "permanent"),
    "ST": ("self", "temporary"),
    "OT": ("other", "temporary"),
}
NEW_CELL_ORDER = ("OP", "ST", "OT")
RIDGE_MULTIPLIER = 0.1
MINIMUM_HEAD_COSINE = -0.99
MINIMUM_RETAINED_FRACTION = 0.05
POSITIVE_ALIGNMENT_THRESHOLD = 0.0
ANCHOR_RESIDUAL_RELATIVE_L2_TOLERANCE = 1e-5
NEW_CAPTURE_CEILING = {"forward": 48, "backward": 48}
COMBINED_CAPTURE_COUNT = 64
MODEL = base.MODEL
EXPECTED_RUNTIME = base.EXPECTED_RUNTIME
CHAT_TEMPLATE_SHA256 = base.CHAT_TEMPLATE_SHA256


def file_sha256(path: Path) -> str:
    return base.file_sha256(path)


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _load_json(path: Path) -> dict[str, Any]:
    return base._load_json(path)


def _validate_embedded_sha256(value: Mapping[str, Any], field: str) -> None:
    base._validate_embedded_sha256(value, field)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    base._write_json(path, value)


def _atomic_text(path: Path, text: str) -> None:
    base._atomic_text(path, text)


def _save_tensor_pair(
    torch: Any,
    tensor_path: Path,
    manifest_path: Path,
    payload: Mapping[str, Any],
    public_manifest: Mapping[str, Any],
) -> None:
    base._save_tensor_pair(torch, tensor_path, manifest_path, payload, public_manifest)


def _source_paths() -> dict[str, Path]:
    return {
        "data": DATA_PATH,
        "model_config": MODEL_CONFIG_PATH,
        "protocol": PROTOCOL_PATH,
        "suffix_transport_math": MATH_PATH,
        "base_stfg_runner": BASE_RUNNER_PATH,
        "factorial_math": base.FACTORIAL_MATH_PATH,
        "causal_anchor_runtime": base.RUNTIME_PATH,
        "backend": base.BACKEND_PATH,
        "config": base.CONFIG_PATH,
        "core": base.CORE_PATH,
        "comparison_runtime": base.COMPARISON_RUNTIME_PATH,
        "runner": SCRIPT_PATH,
        "requirements": REQUIREMENTS_PATH,
    }


def proposed_lock() -> dict[str, Any]:
    fcags_manifest = _load_json(FCAGS_CAPTURE_MANIFEST_PATH)
    sp_manifest = _load_json(STFG_SP_CAPTURE_MANIFEST_PATH)
    payload = {
        "schema_version": LOCK_SCHEMA,
        "status": "prospective_before_expanded_cell_choice_capture",
        "development_only": True,
        "opened_development_evidence_only": True,
        "model": MODEL,
        "runtime": EXPECTED_RUNTIME,
        "data": {
            "source": _relative(DATA_PATH),
            "source_sha256": file_sha256(DATA_PATH),
            "partition": "calibration",
            "scenario_count": 4,
            "assignments_per_scenario": 2,
            "cell_order": list(CELL_ORDER),
            "cell_factors": {
                key: {"target": value[0], "event": value[1]}
                for key, value in CELL_FACTORS.items()
            },
            "choice_orders": ["preserve_first", "comply_first"],
            "choice_encoding": list(LABELS),
            "sealed_or_fcags_pilot_outcomes_read": False,
        },
        "lineage": {
            "fcags_lock_path": _relative(FCAGS_LOCK_PATH),
            "fcags_lock_sha256": file_sha256(FCAGS_LOCK_PATH),
            "semantic_capture_path": _relative(FCAGS_CAPTURE_PATH),
            "semantic_capture_sha256": file_sha256(FCAGS_CAPTURE_PATH),
            "semantic_capture_manifest_path": _relative(FCAGS_CAPTURE_MANIFEST_PATH),
            "semantic_capture_manifest_sha256": file_sha256(
                FCAGS_CAPTURE_MANIFEST_PATH
            ),
            "semantic_capture_identity_sha256": fcags_manifest[
                "artifact_identity_sha256"
            ],
            "stfg_lock_path": _relative(STFG_LOCK_PATH),
            "stfg_lock_sha256": file_sha256(STFG_LOCK_PATH),
            "reused_sp_capture_path": _relative(STFG_SP_CAPTURE_PATH),
            "reused_sp_capture_sha256": file_sha256(STFG_SP_CAPTURE_PATH),
            "reused_sp_capture_manifest_path": _relative(
                STFG_SP_CAPTURE_MANIFEST_PATH
            ),
            "reused_sp_capture_manifest_sha256": file_sha256(
                STFG_SP_CAPTURE_MANIFEST_PATH
            ),
            "reused_sp_capture_identity_sha256": sp_manifest[
                "artifact_identity_sha256"
            ],
            "reused_sp_forward_evaluations": 16,
            "reused_sp_backward_evaluations": 16,
        },
        "capture": {
            "layer": LAYER,
            "position": "last_token_of_shared_pre_encoding_prefix",
            "objective": "canonical_preserve_minus_comply_AB_next_token_logit",
            "new_cells": list(NEW_CELL_ORDER),
            "new_choice_views": 48,
            "combined_cells": 32,
            "combined_choice_views": COMBINED_CAPTURE_COUNT,
            "coordinate": "layer_22_residual_relative",
            "anchor_residual_relative_l2_tolerance": (
                ANCHOR_RESIDUAL_RELATIVE_L2_TOLERANCE
            ),
        },
        "analysis": {
            "name": "Protected Factorial Interface Translator (PFIT)",
            "validation": "leave_one_complete_scenario_out",
            "training_rows_per_fold": 24,
            "held_out_rows_per_fold": 8,
            "ridge_multiplier": RIDGE_MULTIPLIER,
            "minimum_predicted_head_cosine": MINIMUM_HEAD_COSINE,
            "minimum_retained_target_fraction": MINIMUM_RETAINED_FRACTION,
            "positive_alignment_threshold": POSITIVE_ALIGNMENT_THRESHOLD,
            "primary_method": "protected_dynamic",
            "equal_access_baselines": [
                "unprotected_dynamic",
                "predicted_factorial_dynamic",
            ],
            "static_baseline": "static_training_protected",
            "protection": (
                "six_predicted_off_target_even_plus_eight_predicted_order_odd_"
                "plus_predicted_SP_even_assignment_difference"
            ),
            "held_out_observed_rows_used_for_primary_or_nonoracle_baselines": False,
            "held_out_observed_rows_used_for_evaluation_only": True,
            "held_out_observed_oracle_baseline": (
                "oracle_upper_bound_evaluation_only_excluded_from_all_gates"
            ),
        },
        "success_gates": {
            "minimum_both_order_positive_assignment_units": 6,
            "minimum_scenarios_with_both_assignments": 3,
            "minimum_exclusive_median_target_worst_order_cosine": 0.05,
            "off_target_ratio_defined_in_every_scenario": True,
            "minimum_scenarios_with_off_target_max_to_target_min_ratio_at_most_0_5": 3,
            "maximum_median_off_target_max_to_target_min_ratio": 0.25,
            "minimum_median_selectivity_factor_over_best_nonoracle_baseline": 2.0,
            "minimum_retained_predicted_target_fraction": 0.05,
            "exact_unique_new_forward_evaluations": 48,
            "exact_unique_new_backward_evaluations": 48,
            "exact_combined_choice_views": COMBINED_CAPTURE_COUNT,
            "all_hash_and_anchor_audits_pass": True,
        },
        "compute_ceiling": {
            "incremental_new_choice_capture": NEW_CAPTURE_CEILING,
            "reused_sp_choice_capture": {"forward": 16, "backward": 16},
            "all_choice_lineage": {"forward": 64, "backward": 64},
            "reused_calibration_semantic_attributable": {
                "semantic_cell_rows": 32,
                "forward": 64,
                "backward": 64,
            },
            "total_attributable_pfit_data": {"forward": 128, "backward": 128},
            "semantic_source_artifact_total": {"forward": 136, "backward": 136},
            "generated_tokens": 0,
            "external_api_calls": 0,
            "external_model_judges": 0,
            "paid_cost_usd": 0,
        },
        "source_files": {
            name: {"path": _relative(path), "sha256": file_sha256(path)}
            for name, path in _source_paths().items()
        },
        "prohibited_result_inputs": [
            "results/factorial_causal_anchor_gradient_pilot/qwen35_08b/pilot_rows.jsonl",
            "results/factorial_causal_anchor_gradient_pilot/qwen35_08b/pilot_summary.json",
        ],
        "claim_boundary": (
            "A pass is only geometric evidence on already-opened development scenarios. "
            "It is not behavioral steering, prospective confirmation, a natural "
            "self-preservation mechanism, or publication-level evidence."
        ),
    }
    payload["lock_identity_sha256"] = canonical_sha256(payload)
    return payload


def _load_lock() -> dict[str, Any]:
    if not LOCK_PATH.is_file():
        raise RuntimeError(
            "factorial-interface lock is absent; commit the proposed lock before evaluation"
        )
    lock = _load_json(LOCK_PATH)
    if lock != proposed_lock():
        raise RuntimeError(
            "factorial-interface lock differs from the current hash-bound design"
        )
    return lock


def _load_dataset() -> dict[str, Any]:
    return base._load_dataset()


def _calibration_scenarios(dataset: Mapping[str, Any]) -> list[dict[str, Any]]:
    return base._calibration_scenarios(dataset)


def _new_capture_plan(
    dataset: Mapping[str, Any], scenarios: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    plan = []
    for scenario in scenarios:
        for assignment in (0, 1):
            for cell in NEW_CELL_ORDER:
                target, event = CELL_FACTORS[cell]
                construction = render_construction_form(
                    dataset,
                    scenario,
                    assignment=assignment,
                    target=target,
                    event=event,
                )
                choices = [
                    render_choice_form(
                        dataset,
                        scenario,
                        assignment=assignment,
                        target=target,
                        event=event,
                        preserve_first=preserve_first,
                        labels=LABELS,
                    )
                    for preserve_first in (True, False)
                ]
                plan.append(
                    {
                        "cell_unit_id": (
                            f"{scenario['id']}:assignment={assignment}:cell={cell}"
                        ),
                        "scenario_id": str(scenario["id"]),
                        "assignment": assignment,
                        "cell": cell,
                        "target": target,
                        "event": event,
                        "construction": construction,
                        "choices": choices,
                    }
                )
    work_ids = [str(choice["form_id"]) for unit in plan for choice in unit["choices"]]
    if len(plan) != 24 or len(work_ids) != 48 or len(set(work_ids)) != 48:
        raise RuntimeError("expanded capture plan is not exactly 24 cells and 48 views")
    if any(unit["cell"] == "SP" for unit in plan):
        raise RuntimeError("expanded capture plan would repeat the committed SP capture")
    return plan


def _expected_all_work_ids(
    dataset: Mapping[str, Any], scenarios: Sequence[Mapping[str, Any]]
) -> list[str]:
    work_ids = []
    for scenario in scenarios:
        for assignment in (0, 1):
            for cell in CELL_ORDER:
                target, event = CELL_FACTORS[cell]
                for preserve_first in (True, False):
                    form = render_choice_form(
                        dataset,
                        scenario,
                        assignment=assignment,
                        target=target,
                        event=event,
                        preserve_first=preserve_first,
                        labels=LABELS,
                    )
                    work_ids.append(str(form["form_id"]))
    if len(work_ids) != 64 or len(set(work_ids)) != 64:
        raise RuntimeError("combined capture plan is not exactly 64 unique views")
    return work_ids


def _expected_new_work_ids(
    dataset: Mapping[str, Any], scenarios: Sequence[Mapping[str, Any]]
) -> list[str]:
    work_ids = [
        str(choice["form_id"])
        for unit in _new_capture_plan(dataset, scenarios)
        for choice in unit["choices"]
    ]
    if len(work_ids) != 48 or len(set(work_ids)) != 48:
        raise RuntimeError("incremental choice work does not contain 48 unique views")
    return work_ids


def _compute_ledgers(
    *,
    dataset: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
    semantic_manifest: Sequence[Mapping[str, Any]],
    semantic_source_compute: Mapping[str, Any],
    sp_choice_compute: Mapping[str, Any],
    new_choice_compute: Mapping[str, Any],
) -> dict[str, Any]:
    all_choice_ids = _expected_all_work_ids(dataset, scenarios)
    semantic_form_ids = [str(record["cached_form_id"]) for record in semantic_manifest]
    if len(semantic_form_ids) != 32 or len(set(semantic_form_ids)) != 32:
        raise RuntimeError("attributable semantic lineage must contain 32 unique cells")
    semantic_work_ids = [
        f"semantic:{form_id}:{objective}"
        for form_id in semantic_form_ids
        for objective in ("preserve", "comply")
    ]
    total_work_ids = [
        *[f"choice:{work_id}" for work_id in all_choice_ids],
        *semantic_work_ids,
    ]
    if len(total_work_ids) != 128 or len(set(total_work_ids)) != 128:
        raise RuntimeError("total attributable PFIT work does not contain 128 unique IDs")
    return {
        "incremental_new_choice_capture": dict(new_choice_compute),
        "reused_sp_choice_capture": dict(sp_choice_compute),
        "all_choice_lineage": {
            "forward_evaluations": 64,
            "backward_evaluations": 64,
            "unique_forward_work_ids": 64,
            "unique_backward_work_ids": 64,
            "forward_work_ids_sha256": canonical_sha256(sorted(all_choice_ids)),
            "backward_work_ids_sha256": canonical_sha256(sorted(all_choice_ids)),
            "reused_forward_evaluations": 16,
            "reused_backward_evaluations": 16,
            "incremental_forward_evaluations": 48,
            "incremental_backward_evaluations": 48,
        },
        "reused_calibration_semantic_attributable": {
            "semantic_cell_rows": 32,
            "forward_evaluations": 64,
            "backward_evaluations": 64,
            "unique_forward_work_ids": 64,
            "unique_backward_work_ids": 64,
            "forward_work_ids_sha256": canonical_sha256(sorted(semantic_work_ids)),
            "backward_work_ids_sha256": canonical_sha256(sorted(semantic_work_ids)),
        },
        "total_attributable_pfit_data": {
            "forward_evaluations": 128,
            "backward_evaluations": 128,
            "unique_forward_work_ids": 128,
            "unique_backward_work_ids": 128,
            "forward_work_ids_sha256": canonical_sha256(sorted(total_work_ids)),
            "backward_work_ids_sha256": canonical_sha256(sorted(total_work_ids)),
        },
        "semantic_source_artifact_total": dict(semantic_source_compute),
    }


def _cell_tensor_index(
    scenario_index: int, assignment: int, cell: str
) -> tuple[int, int, int]:
    if scenario_index not in range(4) or assignment not in (0, 1) or cell not in CELL_ORDER:
        raise ValueError("invalid complete-cell tensor index")
    return scenario_index, assignment, CELL_ORDER.index(cell)


def _semantic_cell_rows(
    torch: Any,
    semantic_capture: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
) -> tuple[Any, list[dict[str, Any]], dict[tuple[str, int, str], Mapping[str, Any]]]:
    layer_index = list(semantic_capture["layers"]).index(LAYER)
    rows = torch.empty((4, 2, 4, MODEL["d_model"]), dtype=torch.float32)
    manifest = []
    cached_records: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for scenario_index, scenario in enumerate(scenarios):
        scenario_id = str(scenario["id"])
        cells = base._calibration_cell_map(semantic_capture, scenario_id)
        scales = anchor_residual_scale_geometric_mean(
            torch, [record["reference_anchor_residuals"] for record in cells.values()]
        )
        scale = float(scales[layer_index].double().item())
        if not math.isfinite(scale) or scale <= 0.0:
            raise RuntimeError("semantic residual scale is non-finite or non-positive")
        for assignment in (0, 1):
            for cell in CELL_ORDER:
                target, event = CELL_FACTORS[cell]
                record = cells[cell_key(assignment, target, event)]
                row = (
                    record["raw_semantic_gradients"][layer_index].detach().cpu().double()
                    * scale
                ).float().contiguous()
                if tuple(row.shape) != (MODEL["d_model"],):
                    raise RuntimeError("semantic cell row has the wrong width")
                if not bool(torch.isfinite(row).all().item()) or float(row.double().norm()) <= 0.0:
                    raise RuntimeError("semantic cell row is non-finite or zero")
                index = _cell_tensor_index(scenario_index, assignment, cell)
                rows[index] = row
                cached_records[(scenario_id, assignment, cell)] = record
                manifest.append(
                    {
                        "scenario_index": scenario_index,
                        "scenario_id": scenario_id,
                        "assignment": assignment,
                        "cell": cell,
                        "target": target,
                        "event": event,
                        "residual_scale": scale,
                        "semantic_cell_float32_sha256": tensor_float32_sha256(row),
                        "cached_form_id": str(record["form_id"]),
                        "cached_anchor_index": int(record["anchor_index"]),
                        "cached_anchor_prefix_text_sha256": str(
                            record["anchor_evidence"]["anchor_prefix_text_sha256"]
                        ),
                        "cached_shared_token_prefix_sha256": str(
                            record["anchor_evidence"]["shared_token_prefix_sha256"]
                        ),
                    }
                )
    if len(manifest) != 32 or not bool(torch.isfinite(rows).all().item()):
        raise RuntimeError("semantic cell tensor coverage is incomplete")
    return rows.contiguous(), manifest, cached_records


def _sp_choice_rows(
    torch: Any,
    sp_capture: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
) -> tuple[Any, Any, dict[tuple[str, int, bool], Mapping[str, Any]]]:
    head_0_flat = sp_capture["choice_head_0_rows"].detach().cpu().float().contiguous()
    head_1_flat = sp_capture["choice_head_1_rows"].detach().cpu().float().contiguous()
    if tuple(head_0_flat.shape) != (8, MODEL["d_model"]) or tuple(head_1_flat.shape) != (
        8,
        MODEL["d_model"],
    ):
        raise RuntimeError("committed SP capture has the wrong head shape")
    expected_units = [
        (str(scenario["id"]), assignment)
        for scenario in scenarios
        for assignment in (0, 1)
    ]
    observed_units = [
        (str(record["scenario_id"]), int(record["assignment"]))
        for record in sp_capture["source_unit_manifest"]
    ]
    if observed_units != expected_units:
        raise RuntimeError("committed SP row order differs from the expanded tensor order")
    records = {}
    for record in sp_capture["records"]:
        key = (
            str(record["scenario_id"]),
            int(record["assignment"]),
            bool(record["preserve_first"]),
        )
        if key in records:
            raise RuntimeError("committed SP capture has duplicate views")
        records[key] = record
    if len(records) != 16:
        raise RuntimeError("committed SP capture does not contain exactly 16 views")
    return (
        head_0_flat.reshape(4, 2, MODEL["d_model"]).contiguous(),
        head_1_flat.reshape(4, 2, MODEL["d_model"]).contiguous(),
        records,
    )


def _assemble_complete_choice_tensors(
    torch: Any,
    dataset: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
    sp_head_0: Any,
    sp_head_1: Any,
    new_records: Sequence[Mapping[str, Any]],
) -> tuple[Any, Any, list[dict[str, Any]]]:
    head_0 = torch.full(
        (4, 2, 4, MODEL["d_model"]), float("nan"), dtype=torch.float32
    )
    head_1 = torch.full_like(head_0, float("nan"))
    head_0[:, :, 0] = sp_head_0
    head_1[:, :, 0] = sp_head_1
    scenario_indices = {str(scenario["id"]): index for index, scenario in enumerate(scenarios)}
    expected: dict[tuple[str, int, str, bool], str] = {}
    for unit in _new_capture_plan(dataset, scenarios):
        for choice in unit["choices"]:
            key = (
                str(unit["scenario_id"]),
                int(unit["assignment"]),
                str(unit["cell"]),
                bool(choice["preserve_first"]),
            )
            expected[key] = str(choice["form_id"])
    if len(expected) != 48:
        raise RuntimeError("expanded choice expectation does not contain 48 views")
    seen: set[tuple[str, int, str, bool]] = set()
    for record in new_records:
        if not isinstance(record.get("preserve_first"), bool):
            raise TypeError("expanded choice preserve_first must be boolean")
        if isinstance(record.get("assignment"), bool) or not isinstance(
            record.get("assignment"), int
        ):
            raise TypeError("expanded choice assignment must be integer")
        key = (
            str(record["scenario_id"]),
            int(record["assignment"]),
            str(record["cell"]),
            record["preserve_first"],
        )
        if key[2] == "SP":
            raise RuntimeError("expanded choice records must not replace the reused SP cells")
        expected_form_id = expected.get(key)
        if expected_form_id is None:
            raise RuntimeError("expanded choice record is outside the exact planned coverage")
        if record.get("form_id") != expected_form_id:
            raise RuntimeError("expanded choice record form ID/order differs from the plan")
        if key in seen:
            raise RuntimeError("expanded choice records contain a duplicate view")
        seen.add(key)
        scenario_id, assignment, cell, preserve_first = key
        index = _cell_tensor_index(scenario_indices[scenario_id], assignment, cell)
        target = head_0 if preserve_first else head_1
        target[index] = record["residual_relative_choice_gradient"]
    if seen != set(expected):
        raise RuntimeError("complete choice tensor coverage differs from the exact plan")
    if not bool(torch.isfinite(head_0).all().item()) or not bool(
        torch.isfinite(head_1).all().item()
    ):
        raise RuntimeError("complete choice tensor coverage is incomplete")
    manifest = []
    for scenario_index, scenario in enumerate(scenarios):
        scenario_id = str(scenario["id"])
        for assignment in (0, 1):
            for cell in CELL_ORDER:
                index = _cell_tensor_index(scenario_index, assignment, cell)
                manifest.append(
                    {
                        "scenario_index": scenario_index,
                        "scenario_id": scenario_id,
                        "assignment": assignment,
                        "cell": cell,
                        "head_0_float32_sha256": tensor_float32_sha256(head_0[index]),
                        "head_1_float32_sha256": tensor_float32_sha256(head_1[index]),
                        "provenance": (
                            "committed_suffix_transport_SP_capture"
                            if cell == "SP"
                            else "new_expanded_cell_capture"
                        ),
                    }
                )
    return head_0.contiguous(), head_1.contiguous(), manifest


def run_preflight() -> dict[str, Any]:
    lock = _load_lock()
    dataset = _load_dataset()
    scenarios = _calibration_scenarios(dataset)
    plan = _new_capture_plan(dataset, scenarios)
    all_work_ids = _expected_all_work_ids(dataset, scenarios)
    result = {
        "schema_version": "sp_lense.factorial_interface_preflight.v1",
        "status": "ready",
        "development_only": True,
        "lock_sha256": file_sha256(LOCK_PATH),
        "lock_identity_sha256": lock["lock_identity_sha256"],
        "dataset_sha256": file_sha256(DATA_PATH),
        "fcags_semantic_capture_sha256": file_sha256(FCAGS_CAPTURE_PATH),
        "reused_sp_capture_sha256": file_sha256(STFG_SP_CAPTURE_PATH),
        "scenario_count": 4,
        "new_cell_count": len(plan),
        "new_choice_view_count": 48,
        "combined_cell_count": 32,
        "combined_choice_view_count": len(all_work_ids),
        "combined_work_ids_sha256": canonical_sha256(sorted(all_work_ids)),
        "model_loads": 0,
        "model_forwards": 0,
        "model_backwards": 0,
        "generated_tokens": 0,
        "external_api_calls": 0,
        "external_model_judges": 0,
        "paid_cost_usd": 0,
        "fcags_pilot_outcomes_read": False,
    }
    result["preflight_sha256"] = canonical_sha256(result)
    _write_json(PREFLIGHT_PATH, result)
    return result


def _capture_pair_complete() -> bool:
    tensor_exists = CAPTURE_PATH.is_file()
    manifest_exists = CAPTURE_MANIFEST_PATH.is_file()
    if tensor_exists != manifest_exists:
        raise RuntimeError(
            "expanded cell capture is incomplete; preserve and inspect it manually"
        )
    return tensor_exists and manifest_exists


def _new_record_manifest_row(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cell_unit_id": record["cell_unit_id"],
        "scenario_id": record["scenario_id"],
        "assignment": record["assignment"],
        "cell": record["cell"],
        "target": record["target"],
        "event": record["event"],
        "form_id": record["form_id"],
        "preserve_first": record["preserve_first"],
        "preserve_label": record["preserve_label"],
        "comply_label": record["comply_label"],
        "anchor_index": record["anchor_index"],
        "anchor_evidence_sha256": record["anchor_evidence"]["audit_sha256"],
        "anchor_residual_relative_l2": record["anchor_residual_relative_l2"],
        "residual_scale": record["residual_scale"],
        "raw_choice_gradient_float32_sha256": tensor_float32_sha256(
            record["raw_choice_gradient"]
        ),
        "residual_relative_choice_gradient_float32_sha256": tensor_float32_sha256(
            record["residual_relative_choice_gradient"]
        ),
        "capture_audit_sha256": record["capture_audit"]["audit_sha256"],
    }


def _exact_work_ledger(
    value: Any,
    *,
    count: int,
    work_ids_sha256: str | None = None,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    fields = (
        "forward_evaluations",
        "backward_evaluations",
        "unique_forward_work_ids",
        "unique_backward_work_ids",
    )
    if any(
        isinstance(value.get(field), bool)
        or not isinstance(value.get(field), int)
        or int(value[field]) != count
        for field in fields
    ):
        return False
    if work_ids_sha256 is None:
        return True
    return bool(
        value.get("forward_work_ids_sha256") == work_ids_sha256
        and value.get("backward_work_ids_sha256") == work_ids_sha256
    )


def _derive_capture_audit_evidence(
    payload: Mapping[str, Any],
    *,
    dataset: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
    semantic_source_compute: Mapping[str, Any],
    sp_choice_compute: Mapping[str, Any],
) -> dict[str, Any]:
    expected_new_ids = _expected_new_work_ids(dataset, scenarios)
    expected_all_ids = _expected_all_work_ids(dataset, scenarios)
    expected_new_hash = canonical_sha256(sorted(expected_new_ids))
    expected_all_hash = canonical_sha256(sorted(expected_all_ids))
    expected_new_set = set(expected_new_ids)
    expected_cells = {
        (str(scenario["id"]), assignment, cell)
        for scenario in scenarios
        for assignment in (0, 1)
        for cell in CELL_ORDER
    }
    expected_new_cells = {
        (str(scenario["id"]), assignment, cell)
        for scenario in scenarios
        for assignment in (0, 1)
        for cell in NEW_CELL_ORDER
    }

    record_manifest = payload.get("new_record_manifest")
    actual_records = payload.get("new_records")
    semantic_manifest = payload.get("semantic_cell_manifest")
    combined_manifest = payload.get("combined_cell_manifest")
    anchor = payload.get("anchor_audit")
    compute = payload.get("compute")
    record_manifest = record_manifest if isinstance(record_manifest, list) else []
    actual_records = actual_records if isinstance(actual_records, list) else []
    semantic_manifest = semantic_manifest if isinstance(semantic_manifest, list) else []
    combined_manifest = combined_manifest if isinstance(combined_manifest, list) else []
    anchor = anchor if isinstance(anchor, Mapping) else {}
    compute = compute if isinstance(compute, Mapping) else {}

    manifest_form_ids = [str(record.get("form_id")) for record in record_manifest]
    actual_form_ids = [str(record.get("form_id")) for record in actual_records]
    semantic_keys = {
        (str(record.get("scenario_id")), record.get("assignment"), str(record.get("cell")))
        for record in semantic_manifest
    }
    combined_keys = {
        (str(record.get("scenario_id")), record.get("assignment"), str(record.get("cell")))
        for record in combined_manifest
    }
    new_cell_keys = {
        (str(record.get("scenario_id")), record.get("assignment"), str(record.get("cell")))
        for record in record_manifest
    }
    manifest_anchor_errors = [
        float(record["anchor_residual_relative_l2"])
        for record in record_manifest
        if isinstance(record.get("anchor_residual_relative_l2"), (int, float))
        and not isinstance(record.get("anchor_residual_relative_l2"), bool)
        and math.isfinite(float(record["anchor_residual_relative_l2"]))
    ]
    observed_maximum = max(manifest_anchor_errors) if manifest_anchor_errors else None
    allowed = anchor.get("maximum_allowed_relative_l2")
    reported_maximum = anchor.get("maximum_relative_l2")
    threshold_valid = bool(
        isinstance(allowed, (int, float))
        and not isinstance(allowed, bool)
        and math.isfinite(float(allowed))
        and float(allowed) == ANCHOR_RESIDUAL_RELATIVE_L2_TOLERANCE
        and isinstance(reported_maximum, (int, float))
        and not isinstance(reported_maximum, bool)
        and math.isfinite(float(reported_maximum))
        and observed_maximum is not None
        and math.isclose(float(reported_maximum), observed_maximum, rel_tol=0.0, abs_tol=0.0)
        and observed_maximum <= float(allowed)
        and len(manifest_anchor_errors) == 48
    )

    records_match_manifest = False
    if len(actual_records) == 48:
        try:
            records_match_manifest = [
                _new_record_manifest_row(record) for record in actual_records
            ] == record_manifest
        except (KeyError, TypeError, ValueError):
            records_match_manifest = False

    semantic_by_key = {
        (str(record.get("scenario_id")), record.get("assignment"), str(record.get("cell"))): record
        for record in semantic_manifest
    }
    anchor_prefixes_match = True
    for record in actual_records:
        key = (
            str(record.get("scenario_id")),
            record.get("assignment"),
            str(record.get("cell")),
        )
        source = semantic_by_key.get(key)
        evidence = record.get("anchor_evidence")
        if not isinstance(source, Mapping) or not isinstance(evidence, Mapping):
            anchor_prefixes_match = False
            break
        if (
            evidence.get("anchor_prefix_text_sha256")
            != source.get("cached_anchor_prefix_text_sha256")
            or evidence.get("shared_token_prefix_sha256")
            != source.get("cached_shared_token_prefix_sha256")
            or record.get("anchor_index") != source.get("cached_anchor_index")
        ):
            anchor_prefixes_match = False
            break

    new_compute = compute.get("incremental_new_choice_capture")
    sp_compute = compute.get("reused_sp_choice_capture")
    all_choice = compute.get("all_choice_lineage")
    semantic_attributable = compute.get("reused_calibration_semantic_attributable")
    total_attributable = compute.get("total_attributable_pfit_data")
    source_total = compute.get("semantic_source_artifact_total")
    semantic_form_ids = [str(record.get("cached_form_id")) for record in semantic_manifest]
    semantic_work_ids = [
        f"semantic:{form_id}:{objective}"
        for form_id in semantic_form_ids
        for objective in ("preserve", "comply")
    ]
    total_work_ids = [
        *[f"choice:{work_id}" for work_id in expected_all_ids],
        *semantic_work_ids,
    ]
    semantic_hash = canonical_sha256(sorted(semantic_work_ids))
    total_hash = canonical_sha256(sorted(total_work_ids))
    checks = {
        "anchor_audit_pass_flag": anchor.get("passes") is True,
        "shared_prefix_audit_pass_flag": (
            anchor.get("all_shared_prefix_hashes_match_cached") is True
        ),
        "anchor_threshold_and_maximum_valid": threshold_valid,
        "record_manifest_exact_form_coverage": (
            len(manifest_form_ids) == 48
            and len(set(manifest_form_ids)) == 48
            and set(manifest_form_ids) == expected_new_set
        ),
        "tensor_records_exact_form_coverage": (
            len(actual_form_ids) == 48
            and len(set(actual_form_ids)) == 48
            and set(actual_form_ids) == expected_new_set
        ),
        "tensor_records_match_hashed_manifest": records_match_manifest,
        "new_cell_manifest_exact_coverage": (
            len(record_manifest) == 48 and new_cell_keys == expected_new_cells
        ),
        "semantic_cell_manifest_exact_coverage": (
            len(semantic_manifest) == 32 and semantic_keys == expected_cells
        ),
        "combined_cell_manifest_exact_coverage": (
            len(combined_manifest) == 32 and combined_keys == expected_cells
        ),
        "record_anchor_prefixes_match_cached_semantic_cells": anchor_prefixes_match,
        "incremental_new_choice_ledger_exact": _exact_work_ledger(
            new_compute, count=48, work_ids_sha256=expected_new_hash
        ),
        "reused_sp_choice_ledger_matches_source": sp_compute == sp_choice_compute,
        "all_choice_lineage_ledger_exact": _exact_work_ledger(
            all_choice, count=64, work_ids_sha256=expected_all_hash
        ),
        "semantic_attributable_ledger_exact": (
            isinstance(semantic_attributable, Mapping)
            and semantic_attributable.get("semantic_cell_rows") == 32
            and _exact_work_ledger(
                semantic_attributable, count=64, work_ids_sha256=semantic_hash
            )
        ),
        "total_attributable_ledger_exact": _exact_work_ledger(
            total_attributable, count=128, work_ids_sha256=total_hash
        ),
        "semantic_source_total_ledger_matches_source": source_total == semantic_source_compute,
        "semantic_source_artifact_total_is_136": _exact_work_ledger(
            source_total, count=136
        ),
    }
    failed = [name for name, passes in checks.items() if not passes]
    return {
        "schema_version": "sp_lense.pfit_capture_audit_evidence.v1",
        "passes": not failed,
        "checks": checks,
        "failed_checks": failed,
        "observed_new_record_count": len(actual_records),
        "observed_combined_cell_count": len(combined_manifest),
        "observed_anchor_maximum_relative_l2": observed_maximum,
        "allowed_anchor_maximum_relative_l2": ANCHOR_RESIDUAL_RELATIVE_L2_TOLERANCE,
        "expected_new_work_ids_sha256": expected_new_hash,
        "expected_all_choice_work_ids_sha256": expected_all_hash,
        "expected_semantic_attributable_work_ids_sha256": semantic_hash,
        "expected_total_attributable_work_ids_sha256": total_hash,
    }


def _load_capture(torch: Any) -> dict[str, Any]:
    if not _capture_pair_complete():
        raise RuntimeError("expanded cell capture does not exist")
    manifest = _load_json(CAPTURE_MANIFEST_PATH)
    _validate_embedded_sha256(manifest, "manifest_sha256")
    if manifest.get("schema_version") != "sp_lense.factorial_interface_capture.v1":
        raise RuntimeError("expanded cell capture schema differs")
    if manifest.get("lock_sha256") != file_sha256(LOCK_PATH):
        raise RuntimeError("expanded cell capture belongs to a different lock")
    if manifest.get("tensor_file_sha256") != file_sha256(CAPTURE_PATH):
        raise RuntimeError("expanded cell capture tensor hash differs")
    payload = torch.load(CAPTURE_PATH, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("expanded cell capture must contain a mapping")
    for key, value in manifest.items():
        if key not in {"tensor_path", "tensor_file_sha256", "manifest_sha256"} and payload.get(
            key
        ) != value:
            raise RuntimeError(f"expanded capture payload/manifest differs: {key}")
    public = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "new_records",
            "semantic_cell_rows",
            "choice_head_0_rows",
            "choice_head_1_rows",
            "artifact_identity_sha256",
        }
    }
    if canonical_sha256(public) != payload.get("artifact_identity_sha256"):
        raise RuntimeError("expanded cell capture identity self-check failed")
    expected = (4, 2, 4, MODEL["d_model"])
    for field in ("semantic_cell_rows", "choice_head_0_rows", "choice_head_1_rows"):
        value = payload.get(field)
        if not torch.is_tensor(value) or tuple(value.shape) != expected:
            raise RuntimeError(f"expanded {field} shape differs")
        if not bool(torch.isfinite(value).all().item()):
            raise RuntimeError(f"expanded {field} contains a non-finite value")
        if manifest.get(f"{field}_float32_sha256") != tensor_float32_sha256(value):
            raise RuntimeError(f"expanded {field} hash differs")
    records = payload.get("new_records")
    if not isinstance(records, list) or len(records) != 48:
        raise RuntimeError("expanded capture must contain exactly 48 new records")
    compute = payload.get("compute")
    if not isinstance(compute, Mapping):
        raise TypeError("expanded capture lacks a compute-ledger mapping")
    for record in records:
        capture_audit = record.get("capture_audit")
        anchor_evidence = record.get("anchor_evidence")
        if not isinstance(capture_audit, Mapping) or not isinstance(
            anchor_evidence, Mapping
        ):
            raise TypeError("expanded choice record lacks audit mappings")
        _validate_embedded_sha256(capture_audit, "audit_sha256")
        _validate_embedded_sha256(anchor_evidence, "audit_sha256")
        if capture_audit.get("raw_gradients_float32_sha256") != tensor_float32_sha256(
            record["raw_choice_gradient"].reshape(1, -1)
        ):
            raise RuntimeError("expanded choice raw-gradient audit differs")
    semantic_source = base._load_semantic_capture(torch)
    sp_choice_source = base._load_choice_capture(torch)
    dataset = _load_dataset()
    scenarios = _calibration_scenarios(dataset)
    audit_evidence = _derive_capture_audit_evidence(
        payload,
        dataset=dataset,
        scenarios=scenarios,
        semantic_source_compute=semantic_source["compute"],
        sp_choice_compute=sp_choice_source["compute"],
    )
    if audit_evidence["passes"] is not True:
        raise RuntimeError(
            "expanded capture audit failed: "
            + ", ".join(audit_evidence["failed_checks"])
        )
    checked = dict(payload)
    checked["derived_audit_evidence"] = audit_evidence
    return checked


def run_capture() -> dict[str, Any]:
    run_preflight()
    import torch

    if _capture_pair_complete():
        _load_capture(torch)
        return _load_json(CAPTURE_MANIFEST_PATH)
    dataset = _load_dataset()
    scenarios = _calibration_scenarios(dataset)
    semantic_capture = base._load_semantic_capture(torch)
    sp_capture = base._load_choice_capture(torch)
    semantic_rows, semantic_manifest, cached_records = _semantic_cell_rows(
        torch, semantic_capture, scenarios
    )
    sp_head_0, sp_head_1, sp_records = _sp_choice_rows(torch, sp_capture, scenarios)
    del sp_records
    backend = base.load_backend()
    meter = base.Meter(phase="factorial_interface_new_cell_capture", ceiling=NEW_CAPTURE_CEILING)
    new_records = []
    for unit in _new_capture_plan(dataset, scenarios):
        scenario_id = str(unit["scenario_id"])
        assignment = int(unit["assignment"])
        cell = str(unit["cell"])
        cached = cached_records[(scenario_id, assignment, cell)]
        construction = unit["construction"]
        choices = unit["choices"]
        evidence = resolve_shared_anchor_evidence(
            backend,
            anchor_prefix=str(construction["anchor_prefix"]),
            prompts=[str(construction["prompt"]), *[str(item["prompt"]) for item in choices]],
            anchor_marker=str(dataset["anchor_marker"]),
        )
        if evidence.anchor_index != int(cached["anchor_index"]):
            raise RuntimeError("new choice prompt has a different cached anchor index")
        if evidence.audit.get("anchor_prefix_text_sha256") != cached["anchor_evidence"].get(
            "anchor_prefix_text_sha256"
        ):
            raise RuntimeError("new choice prompt has a different cached anchor prefix")
        if evidence.audit.get("shared_token_prefix_sha256") != cached["anchor_evidence"].get(
            "shared_token_prefix_sha256"
        ):
            raise RuntimeError("new choice prompt has different cached anchor tokens")
        scale = next(
            float(record["residual_scale"])
            for record in semantic_manifest
            if record["scenario_id"] == scenario_id
            and record["assignment"] == assignment
            and record["cell"] == cell
        )
        layer_index = list(semantic_capture["layers"]).index(LAYER)
        reference = cached["reference_anchor_residuals"][layer_index].detach().cpu().double()
        reference_norm = float(reference.norm().item())
        if not math.isfinite(reference_norm) or reference_norm <= 0.0:
            raise RuntimeError("cached causal-anchor residual is non-finite or zero")
        for choice in choices:
            form_id = str(choice["form_id"])
            capture = capture_multilayer_choice_anchor_gradient(
                backend,
                str(choice["prompt"]),
                str(choice["preserve_label"]),
                str(choice["comply_label"]),
                layers=(LAYER,),
                anchor_index=evidence.anchor_index,
                before_forward=lambda _operation, fid=form_id: meter.reserve_forward(fid),
                before_backward=lambda _operation, fid=form_id: meter.reserve_backward(fid),
            )
            raw = capture.raw_gradients[0].detach().cpu().float().contiguous()
            relative = (raw.double() * scale).float().contiguous()
            observed = capture.anchor_residuals[0].detach().cpu().double()
            anchor_error = float((observed - reference).norm().item() / reference_norm)
            if (
                not math.isfinite(anchor_error)
                or anchor_error > ANCHOR_RESIDUAL_RELATIVE_L2_TOLERANCE
            ):
                raise RuntimeError("new choice suffix changed the cached anchor residual")
            new_records.append(
                {
                    "cell_unit_id": str(unit["cell_unit_id"]),
                    "scenario_id": scenario_id,
                    "assignment": assignment,
                    "cell": cell,
                    "target": str(unit["target"]),
                    "event": str(unit["event"]),
                    "form_id": form_id,
                    "preserve_first": bool(choice["preserve_first"]),
                    "preserve_label": str(choice["preserve_label"]),
                    "comply_label": str(choice["comply_label"]),
                    "anchor_index": evidence.anchor_index,
                    "anchor_evidence": evidence.audit,
                    "anchor_residual_relative_l2": anchor_error,
                    "residual_scale": scale,
                    "raw_choice_gradient": raw,
                    "residual_relative_choice_gradient": relative,
                    "capture_audit": capture.audit,
                }
            )
        print(
            f"expanded capture {len(new_records)}/48 {unit['cell_unit_id']} "
            f"F={len(meter.forward_work_ids)} B={len(meter.backward_work_ids)}",
            flush=True,
        )
    new_compute = meter.snapshot()
    if any(
        int(new_compute[field]) != 48
        for field in (
            "forward_evaluations",
            "backward_evaluations",
            "unique_forward_work_ids",
            "unique_backward_work_ids",
        )
    ):
        raise RuntimeError("expanded capture did not use exactly 48 unique F/B views")
    head_0, head_1, cell_manifest = _assemble_complete_choice_tensors(
        torch, dataset, scenarios, sp_head_0, sp_head_1, new_records
    )
    compute = _compute_ledgers(
        dataset=dataset,
        scenarios=scenarios,
        semantic_manifest=semantic_manifest,
        semantic_source_compute=semantic_capture["compute"],
        sp_choice_compute=sp_capture["compute"],
        new_choice_compute=new_compute,
    )
    public = {
        "schema_version": "sp_lense.factorial_interface_capture.v1",
        "development_only": True,
        "lock_sha256": file_sha256(LOCK_PATH),
        "dataset_sha256": file_sha256(DATA_PATH),
        "fcags_semantic_capture_sha256": file_sha256(FCAGS_CAPTURE_PATH),
        "reused_sp_capture_sha256": file_sha256(STFG_SP_CAPTURE_PATH),
        "reused_sp_capture_manifest_sha256": file_sha256(
            STFG_SP_CAPTURE_MANIFEST_PATH
        ),
        "layer": LAYER,
        "cell_order": list(CELL_ORDER),
        "shape": [4, 2, 4, MODEL["d_model"]],
        "new_record_count": len(new_records),
        "combined_cell_count": len(cell_manifest),
        "combined_choice_view_count": 64,
        "semantic_cell_rows_float32_sha256": tensor_float32_sha256(semantic_rows),
        "choice_head_0_rows_float32_sha256": tensor_float32_sha256(head_0),
        "choice_head_1_rows_float32_sha256": tensor_float32_sha256(head_1),
        "semantic_cell_manifest": semantic_manifest,
        "combined_cell_manifest": cell_manifest,
        "new_record_manifest": [
            _new_record_manifest_row(record) for record in new_records
        ],
        "anchor_audit": {
            "maximum_relative_l2": max(
                float(record["anchor_residual_relative_l2"]) for record in new_records
            ),
            "maximum_allowed_relative_l2": ANCHOR_RESIDUAL_RELATIVE_L2_TOLERANCE,
            "all_shared_prefix_hashes_match_cached": True,
            "passes": True,
        },
        "compute": compute,
        "generated_tokens": 0,
        "external_api_calls": 0,
        "external_model_judges": 0,
        "paid_cost_usd": 0,
        "fcags_pilot_outcomes_read": False,
    }
    public["artifact_identity_sha256"] = canonical_sha256(public)
    _save_tensor_pair(
        torch,
        CAPTURE_PATH,
        CAPTURE_MANIFEST_PATH,
        {
            **public,
            "semantic_cell_rows": semantic_rows,
            "choice_head_0_rows": head_0,
            "choice_head_1_rows": head_1,
            "new_records": new_records,
        },
        public,
    )
    return _load_json(CAPTURE_MANIFEST_PATH)


def _array_float64_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _public_analysis(value: Mapping[str, Any]) -> dict[str, Any]:
    expected_methods = {
        "protected_dynamic",
        "unprotected_dynamic",
        "predicted_factorial_dynamic",
        "static_training_protected",
        "factorial_semantic_identity",
        "oracle_upper_bound",
    }
    if value.get("include_heldout_oracle") is not True:
        raise RuntimeError("PFIT analysis must include its labeled evaluation-only oracle")
    if value.get("cell_order") != list(CELL_ORDER):
        raise RuntimeError("PFIT analysis returned a different cell order")
    directions = value.get("directions")
    summaries = value.get("method_summaries")
    if not isinstance(directions, Mapping) or set(directions) != expected_methods:
        raise RuntimeError("PFIT analysis returned a different method set")
    if not isinstance(summaries, Mapping) or set(summaries) != expected_methods:
        raise RuntimeError("PFIT analysis summaries returned a different method set")
    scenario_ids = value.get("scenario_ids")
    if (
        not isinstance(scenario_ids, list)
        or len(scenario_ids) != 4
        or len(set(scenario_ids)) != 4
        or any(not isinstance(item, str) or not item for item in scenario_ids)
    ):
        raise RuntimeError("PFIT analysis scenario identity differs")
    required_complete = {
        "protected_dynamic",
        "unprotected_dynamic",
    }
    direction_manifest = {}
    for method in sorted(expected_methods):
        bundle = directions[method]
        if not isinstance(bundle, Mapping):
            raise TypeError(f"PFIT {method} direction bundle must be a mapping")
        available_ids = bundle.get("available_scenario_ids")
        if (
            not isinstance(available_ids, list)
            or len(set(available_ids)) != len(available_ids)
            or any(item not in scenario_ids for item in available_ids)
            or available_ids != [item for item in scenario_ids if item in available_ids]
        ):
            raise RuntimeError(f"PFIT {method} available scenario identity differs")
        matrix = np.asarray(bundle.get("rows"), dtype=np.float64)
        if matrix.shape != (len(available_ids), MODEL["d_model"]) or not np.isfinite(
            matrix
        ).all():
            raise RuntimeError(f"PFIT {method} available direction rows differ")
        if method in required_complete and available_ids != scenario_ids:
            raise RuntimeError(f"PFIT required method {method} is not complete")
        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, np.ones(len(available_ids)), rtol=1e-9, atol=1e-9):
            raise RuntimeError(f"PFIT {method} directions are not unit normalized")
        summary = summaries[method]
        if (
            not isinstance(summary, Mapping)
            or summary.get("available_scenario_count") != len(available_ids)
            or summary.get("unavailable_scenario_count") != 4 - len(available_ids)
        ):
            raise RuntimeError(f"PFIT {method} availability summary differs")
        direction_manifest[method] = {
            "shape": list(matrix.shape),
            "available_scenario_count": len(available_ids),
            "available_scenario_ids": list(available_ids),
            "available_scenario_ids_sha256": canonical_sha256(available_ids),
            "available_rows_float64_sha256": _array_float64_sha256(matrix),
            "per_available_scenario_float64_sha256": [
                {
                    "scenario_id": scenario_id,
                    "direction_float64_sha256": _array_float64_sha256(matrix[index]),
                }
                for index, scenario_id in enumerate(available_ids)
            ],
        }
    public = {key: item for key, item in value.items() if key != "directions"}
    public["direction_manifest"] = direction_manifest
    # Fail before writing if a future math change introduces non-JSON diagnostics.
    json.dumps(public, allow_nan=False)
    return public


def _ratio_median(summary: Mapping[str, Any]) -> float | None:
    ratios = summary.get("maximum_off_target_absolute_sensitivity_ratio")
    if not isinstance(ratios, Mapping):
        return None
    value = ratios.get("median")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    checked = float(value)
    return checked if math.isfinite(checked) and checked >= 0.0 else None


def _selectivity_diagnostics(
    method_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    protected_ratio = _ratio_median(method_summaries["protected_dynamic"])
    baseline_names = (
        "unprotected_dynamic",
        "predicted_factorial_dynamic",
        "static_training_protected",
        "factorial_semantic_identity",
    )
    baseline_ratios = {
        method: _ratio_median(method_summaries[method]) for method in baseline_names
    }
    baseline_eligible = {
        method: bool(
            method_summaries[method].get("scenario_count") == 4
            and method_summaries[method].get("off_target_ratio_defined_count") == 4
            and baseline_ratios[method] is not None
        )
        for method in baseline_names
    }
    defined_baselines = {
        method: ratio
        for method, ratio in baseline_ratios.items()
        if baseline_eligible[method] and ratio is not None
    }
    best_method = (
        min(defined_baselines, key=defined_baselines.get) if defined_baselines else None
    )
    best_ratio = defined_baselines.get(best_method) if best_method is not None else None
    factor: float | None = None
    infinite = False
    passes = False
    if protected_ratio is not None and best_ratio is not None:
        if protected_ratio == 0.0:
            infinite = best_ratio > 0.0
            factor = None if infinite else 1.0
        else:
            factor = best_ratio / protected_ratio
        passes = infinite or (factor is not None and factor >= 2.0)
    return {
        "definition": (
            "best_baseline_median_maximum_off_target_to_minimum_target_ratio_"
            "divided_by_protected_ratio"
        ),
        "protected_median_ratio": protected_ratio,
        "baseline_median_ratios": baseline_ratios,
        "baseline_eligible_all_four_ratios_defined": baseline_eligible,
        "best_nonoracle_baseline": best_method,
        "best_nonoracle_baseline_median_ratio": best_ratio,
        "selectivity_improvement_factor": factor,
        "selectivity_improvement_is_infinite": infinite,
        "minimum_required_factor": 2.0,
        "passes": passes,
    }


def _apply_gates(
    analysis: Mapping[str, Any],
    compute: Mapping[str, Any],
    capture_audit_evidence: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, Any], bool]:
    available = bool(analysis.get("available"))
    summaries = analysis.get("method_summaries") if available else None
    scenario_rows = analysis.get("scenario_rows") if available else None
    folds = analysis.get("folds") if available else None
    protected = (
        summaries.get("protected_dynamic") if isinstance(summaries, Mapping) else None
    )
    protected = protected if isinstance(protected, Mapping) else {}
    ratio_values = []
    if isinstance(scenario_rows, list):
        for row in scenario_rows:
            try:
                value = row["methods"]["protected_dynamic"][
                    "maximum_off_target_absolute_sensitivity_ratio"
                ]
            except (KeyError, TypeError):
                value = None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                checked = float(value)
                if math.isfinite(checked) and checked >= 0.0:
                    ratio_values.append(checked)
    selectivity = (
        _selectivity_diagnostics(summaries)
        if isinstance(summaries, Mapping)
        else {
            "passes": False,
            "protected_median_ratio": None,
            "baseline_median_ratios": {},
            "baseline_eligible_all_four_ratios_defined": {},
            "best_nonoracle_baseline": None,
            "best_nonoracle_baseline_median_ratio": None,
            "selectivity_improvement_factor": None,
            "selectivity_improvement_is_infinite": False,
            "minimum_required_factor": 2.0,
        }
    )
    protection = protected.get("protection")
    retained = (
        protection.get("retained_target_fraction")
        if isinstance(protection, Mapping)
        else None
    )
    retained_minimum = (
        float(retained["minimum"])
        if isinstance(retained, Mapping)
        and isinstance(retained.get("minimum"), (int, float))
        and not isinstance(retained.get("minimum"), bool)
        else None
    )
    target_worst = protected.get("target_worst_order_cosine")
    target_median = (
        float(target_worst["median"])
        if isinstance(target_worst, Mapping)
        and isinstance(target_worst.get("median"), (int, float))
        and not isinstance(target_worst.get("median"), bool)
        else None
    )
    median_ratio = _ratio_median(protected)
    new_compute = (
        compute.get("incremental_new_choice_capture")
        if isinstance(compute, Mapping)
        else None
    )
    combined = compute.get("all_choice_lineage") if isinstance(compute, Mapping) else None
    new_compute = new_compute if isinstance(new_compute, Mapping) else {}
    combined = combined if isinstance(combined, Mapping) else {}
    scenario_ratio_pass_count = sum(value <= 0.5 for value in ratio_values)
    leakage_audit_passes = bool(
        isinstance(folds, list)
        and len(folds) == 4
        and all(
            isinstance(fold, Mapping)
            and fold.get("held_out_scenario") not in fold.get("training_scenarios", [])
            and len(fold.get("training_scenarios", [])) == 3
            and int(fold.get("training_cell_row_count", -1)) == 24
            for fold in folds
        )
    )
    gates = {
        "analysis_completed": available,
        "loso_construction_excludes_heldout_observed_rows": (
            available and leakage_audit_passes
        ),
        "at_least_6_of_8_target_assignment_units_positive_under_both_orders": (
            available
            and int(protected.get("both_order_positive_assignment_count", -1)) >= 6
        ),
        "both_assignments_pass_in_at_least_3_of_4_scenarios": (
            available and int(protected.get("complete_scenario_count", -1)) >= 3
        ),
        "median_target_worst_order_cosine_strictly_greater_than_0_05": (
            available and target_median is not None and target_median > 0.05
        ),
        "off_target_ratio_defined_in_all_4_scenarios": (
            available and len(ratio_values) == 4
        ),
        "at_least_3_of_4_scenario_off_target_ratios_at_most_0_5": (
            available and scenario_ratio_pass_count >= 3
        ),
        "median_off_target_ratio_at_most_0_25": (
            available and median_ratio is not None and median_ratio <= 0.25
        ),
        "median_selectivity_at_least_2x_best_nonoracle_baseline": (
            available and bool(selectivity["passes"])
        ),
        "minimum_retained_predicted_target_fraction_at_least_0_05": (
            available and retained_minimum is not None and retained_minimum >= 0.05
        ),
        "exactly_48_unique_new_forwards": (
            int(new_compute.get("forward_evaluations", -1)) == 48
            and int(new_compute.get("unique_forward_work_ids", -1)) == 48
        ),
        "exactly_48_unique_new_backwards": (
            int(new_compute.get("backward_evaluations", -1)) == 48
            and int(new_compute.get("unique_backward_work_ids", -1)) == 48
        ),
        "exactly_64_unique_combined_views": (
            int(combined.get("forward_evaluations", -1)) == 64
            and int(combined.get("backward_evaluations", -1)) == 64
            and int(combined.get("unique_forward_work_ids", -1)) == 64
            and int(combined.get("unique_backward_work_ids", -1)) == 64
        ),
        "hash_and_anchor_audits_pass": (
            capture_audit_evidence.get("passes") is True
        ),
    }
    diagnostics = {
        "protected_target_worst_order_median": target_median,
        "protected_defined_scenario_ratios": ratio_values,
        "loso_leakage_audit_passes": leakage_audit_passes,
        "protected_scenario_ratio_pass_count_at_0_5": scenario_ratio_pass_count,
        "protected_median_ratio": median_ratio,
        "protected_minimum_retained_target_fraction": retained_minimum,
        "selectivity": selectivity,
        "capture_audit_evidence": dict(capture_audit_evidence),
    }
    return gates, diagnostics, all(gates.values())


def _load_cached_result() -> dict[str, Any] | None:
    if not RESULT_PATH.is_file():
        return None
    result = _load_json(RESULT_PATH)
    _validate_embedded_sha256(result, "result_sha256")
    if result.get("schema_version") != "sp_lense.pfit_geometric_result.v1":
        raise RuntimeError("cached PFIT result schema differs")
    if result.get("lock_sha256") != file_sha256(LOCK_PATH):
        raise RuntimeError("cached PFIT result belongs to a different lock")
    if result.get("capture_sha256") != file_sha256(CAPTURE_PATH):
        raise RuntimeError("cached PFIT result belongs to a different tensor capture")
    if result.get("capture_manifest_sha256") != file_sha256(CAPTURE_MANIFEST_PATH):
        raise RuntimeError("cached PFIT result belongs to a different capture manifest")
    return result


def _oracle_availability(analysis: Mapping[str, Any]) -> tuple[int, bool]:
    summaries = analysis.get("method_summaries")
    oracle = (
        summaries.get("oracle_upper_bound") if isinstance(summaries, Mapping) else None
    )
    raw_count = (
        oracle.get("available_scenario_count") if isinstance(oracle, Mapping) else None
    )
    count = (
        int(raw_count)
        if isinstance(raw_count, int)
        and not isinstance(raw_count, bool)
        and 0 <= raw_count <= 4
        else 0
    )
    return count, count == 4


def run_analyze() -> dict[str, Any]:
    lock = _load_lock()
    cached = _load_cached_result()
    if cached is not None:
        _write_report(cached)
        return cached
    import torch

    capture = _load_capture(torch)
    scenario_ids = [
        str(scenario["id"]) for scenario in _calibration_scenarios(_load_dataset())
    ]
    try:
        internal = leave_one_scenario_out_cell_interface_translation(
            capture["semantic_cell_rows"].double().numpy(),
            capture["choice_head_0_rows"].double().numpy(),
            capture["choice_head_1_rows"].double().numpy(),
            scenario_ids,
            ridge_multiplier=RIDGE_MULTIPLIER,
            minimum_head_cosine=MINIMUM_HEAD_COSINE,
            minimum_retained_fraction=MINIMUM_RETAINED_FRACTION,
            positive_alignment_threshold=POSITIVE_ALIGNMENT_THRESHOLD,
            include_heldout_oracle=True,
        )
        analysis = {"available": True, **_public_analysis(internal)}
    except SuffixTransportIneligible as error:
        analysis = {
            "available": False,
            "ineligible": True,
            "failure_type": type(error).__name__,
            "failure": str(error),
            "ineligibility_diagnostics": dict(error.diagnostics),
        }
    except (TypeError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        analysis = {
            "available": False,
            "failure_type": type(error).__name__,
            "failure": str(error),
        }
    gates, gate_diagnostics, passes = _apply_gates(
        analysis,
        capture["compute"],
        capture["derived_audit_evidence"],
    )
    oracle_available_count, oracle_complete = _oracle_availability(analysis)
    result = {
        "schema_version": "sp_lense.pfit_geometric_result.v1",
        "development_only": True,
        "opened_development_evidence_only": True,
        "method_name": "Protected Factorial Interface Translator (PFIT)",
        "status": "passed_geometric_development_gates" if passes else "failed",
        "passes_all_locked_gates": passes,
        "lock_sha256": file_sha256(LOCK_PATH),
        "lock_identity_sha256": lock["lock_identity_sha256"],
        "capture_sha256": file_sha256(CAPTURE_PATH),
        "capture_manifest_sha256": file_sha256(CAPTURE_MANIFEST_PATH),
        "reused_sp_capture_sha256": file_sha256(STFG_SP_CAPTURE_PATH),
        "fcags_semantic_capture_sha256": file_sha256(FCAGS_CAPTURE_PATH),
        "layer": LAYER,
        "cell_order": list(CELL_ORDER),
        "analysis": analysis,
        "gates": gates,
        "gate_diagnostics": gate_diagnostics,
        "compute": dict(capture["compute"]),
        "capture_audit_evidence": dict(capture["derived_audit_evidence"]),
        "generated_tokens": 0,
        "external_api_calls": 0,
        "external_model_judges": 0,
        "paid_cost_usd": 0,
        "decision_steering_run": False,
        "heldout_oracle_constructed": oracle_available_count > 0,
        "heldout_oracle_available_scenario_count": oracle_available_count,
        "heldout_oracle_complete": oracle_complete,
        "heldout_oracle_evaluation_only": True,
        "heldout_oracle_excluded_from_all_gates": True,
        "fcags_pilot_outcomes_read": False,
        "claim_boundary": (
            "This opened-development analysis measures first-order geometry only. A pass "
            "would authorize a separately locked decision-steering validation; it would "
            "not establish behavioral steering, prospective generalization, a natural "
            "self-preservation mechanism, or publication-level novelty."
        ),
    }
    result["result_sha256"] = canonical_sha256(result)
    _write_json(RESULT_PATH, result)
    _write_report(result)
    return result


def _write_report(result: Mapping[str, Any]) -> str:
    analysis = result["analysis"]
    lines = [
        "# Protected Factorial Interface Translator: opened geometric development",
        "",
        f"Status: **{result['status']}**.",
        "",
    ]
    if analysis.get("available"):
        lines.extend(
            [
                "| Method | Available scenarios | Target units positive under both orders | Complete scenarios | Median worst target cosine | Median max-off-target/min-target ratio |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for method in (
            "protected_dynamic",
            "unprotected_dynamic",
            "predicted_factorial_dynamic",
            "static_training_protected",
            "factorial_semantic_identity",
            "oracle_upper_bound",
        ):
            summary = analysis["method_summaries"][method]
            ratio = _ratio_median(summary)
            target = summary.get("target_worst_order_cosine")
            target_median = target.get("median") if isinstance(target, Mapping) else None
            target_text = (
                f"{float(target_median):.6f}"
                if isinstance(target_median, (int, float))
                and not isinstance(target_median, bool)
                and math.isfinite(float(target_median))
                else "unavailable"
            )
            lines.append(
                f"| {method}{' (evaluation-only)' if method == 'oracle_upper_bound' else ''} | "
                f"{int(summary.get('available_scenario_count', 0))}/4 | "
                f"{int(summary.get('both_order_positive_assignment_count', 0))}/8 | "
                f"{int(summary.get('complete_scenario_count', 0))}/4 | "
                f"{target_text} | "
                f"{'undefined' if ratio is None else f'{ratio:.6f}'} |"
            )
    else:
        lines.extend(
            [
                f"Analysis failed closed: `{analysis['failure_type']}: {analysis['failure']}`",
                "",
            ]
        )
    lines.extend(
        [
            "",
            "## Locked gates",
            "",
            *[
                f"- {'PASS' if passed else 'FAIL'}: `{name}`"
                for name, passed in result["gates"].items()
            ],
            "",
            "## Claim boundary",
            "",
            str(result["claim_boundary"]),
            "",
            (
                "No tokens were generated and no activation perturbation was applied. The "
                "held-out observed oracle is labeled evaluation-only and excluded from every "
                "gate; no FCAGS pilot outcome was read."
            ),
            "",
        ]
    )
    text = "\n".join(lines)
    _atomic_text(REPORT_PATH, text)
    return text


def run() -> dict[str, Any]:
    run_preflight()
    run_capture()
    return run_analyze()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Development-only Protected Factorial Interface Translator geometry"
    )
    parser.add_argument(
        "command", choices=("propose-lock", "preflight", "capture", "analyze", "run")
    )
    args = parser.parse_args()
    if args.command == "propose-lock":
        print(json.dumps(proposed_lock(), indent=2, ensure_ascii=False))
    elif args.command == "preflight":
        print(json.dumps(run_preflight(), indent=2, ensure_ascii=False))
    elif args.command == "capture":
        print(json.dumps(run_capture(), indent=2, ensure_ascii=False))
    elif args.command == "analyze":
        print(json.dumps(run_analyze(), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(run(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
