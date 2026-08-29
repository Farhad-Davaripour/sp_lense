from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from sp_lense.causal_anchor_runtime import (
    anchor_residual_scale_geometric_mean,
    resolve_shared_anchor_evidence,
)
from sp_lense.comparison_runtime import resolve_choice_boundary
from sp_lense.decision_margin_shield_finite import (
    BASELINE_LOG_ODDS_TOLERANCE,
    FLOAT32_RAW_CONSTRAINT_TOLERANCE,
    HOOK_REALIZATION_RELATIVE_L2_TOLERANCE,
    KL_DOUBLE_ROUNDOFF_FLOOR,
    KL_LIMITS,
    METHODS,
    SCREEN_RESULT_SHA256,
    SELECTED_LAYER,
    STRENGTHS,
    array_float32_sha256,
    array_float64_sha256,
    build_calibration_plan,
    deployment_recertificate,
    full_vocabulary_kl_float64,
    plan_sha256,
    public_work_spec,
    reconstruct_scenario_directions,
    summarize_calibration,
)
from sp_lense.factorial_causal_anchor import (
    canonical_sha256,
    multilayer_anchor_hooks,
    render_unrelated_ab_form,
    render_unrelated_construction_form,
    tensor_float32_sha256,
    text_sha256,
    validate_pilot_dataset,
)
from sp_lense.semantic_completion_gradient import encode_prompt_and_completion

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
PROTOCOL_PATH = ROOT / "docs" / "DECISION_MARGIN_SHIELD_FINITE_CALIBRATION_PROTOCOL.md"
MATH_PATH = ROOT / "src" / "sp_lense" / "decision_margin_shield_finite.py"
MATH_TEST_PATH = ROOT / "tests" / "test_decision_margin_shield_finite.py"
RUNNER_TEST_PATH = ROOT / "tests" / "test_decision_margin_shield_finite_calibration_runner.py"
DATA_PATH = ROOT / "data" / "factorial_causal_anchor_gradient_pilot.json"
CONTROL_CANDIDATES_PATH = (
    ROOT / "data" / "decision_margin_shield_finite_control_candidates.json"
)
MODEL_CONFIG_PATH = ROOT / "configs" / "qwen35_08b_aligned.json"
ORIGINAL_RUNNER_PATH = ROOT / "scripts" / "decision_margin_shield_layer_screen.py"
ROWSPACE_PATH = ROOT / "src" / "sp_lense" / "decision_margin_shield_rowspace.py"
SCREEN_AMENDMENT_RUNNER_PATH = (
    ROOT / "scripts" / "decision_margin_shield_layer_screen_solver_amendment.py"
)
SCREEN_AMENDMENT_LOCK_PATH = (
    ROOT / "configs" / "decision_margin_shield_layer_screen_solver_amendment_lock.json"
)
SCREEN_RESULT_PATH = (
    ROOT
    / "results"
    / "decision_margin_shield_layer_screen_solver_amendment"
    / "qwen35_08b"
    / "layer_screen_result.json"
)
CAPTURE_MANIFEST_PATH = (
    ROOT
    / "artifacts"
    / "decision_margin_shield_layer_screen"
    / "qwen35_08b"
    / "capture_manifest.json"
)
LOCK_PATH = ROOT / "configs" / "decision_margin_shield_finite_calibration_lock.json"
QUALIFICATION_LOCK_PATH = (
    ROOT / "configs" / "decision_margin_shield_finite_control_qualification_lock.json"
)
ARTIFACT_ROOT = (
    ROOT / "artifacts" / "decision_margin_shield_finite_calibration" / "qwen35_08b"
)
PREFLIGHT_PATH = ARTIFACT_ROOT / "preflight.json"
QUALIFICATION_CHECKPOINT_PATH = ARTIFACT_ROOT / "control_qualification_checkpoint.pt"
QUALIFICATION_LEDGER_PATH = ARTIFACT_ROOT / "control_qualification_ledger.json"
DIRECTION_TENSOR_PATH = ARTIFACT_ROOT / "direction_bank.pt"
DIRECTION_MANIFEST_PATH = ARTIFACT_ROOT / "direction_bank_manifest.json"
FREEZE_PATH = ARTIFACT_ROOT / "calibration_freeze.json"
CHECKPOINT_ROOT = ARTIFACT_ROOT / "calibration_chunks"
LEDGER_PATH = ARTIFACT_ROOT / "calibration_ledger.json"
RESULT_ROOT = (
    ROOT / "results" / "decision_margin_shield_finite_calibration" / "qwen35_08b"
)
RESULT_PATH = RESULT_ROOT / "calibration_result.json"
QUALIFICATION_RESULT_PATH = RESULT_ROOT / "control_qualification_result.json"
REPORT_PATH = RESULT_ROOT / "CALIBRATION_REPORT.md"

LOCK_SCHEMA = "sp_lense.decision_margin_shield_finite_calibration_lock.v1"
QUALIFICATION_LOCK_SCHEMA = (
    "sp_lense.decision_margin_shield_finite_control_qualification_lock.v1"
)
QUALIFICATION_CHECKPOINT_SCHEMA = (
    "sp_lense.decision_margin_shield_finite_control_qualification_checkpoint.v1"
)
QUALIFICATION_LEDGER_SCHEMA = (
    "sp_lense.decision_margin_shield_finite_control_qualification_ledger.v1"
)
QUALIFICATION_RESULT_SCHEMA = (
    "sp_lense.decision_margin_shield_finite_control_qualification_result.v1"
)
PREFLIGHT_SCHEMA = "sp_lense.decision_margin_shield_finite_preflight.v1"
DIRECTION_SCHEMA = "sp_lense.decision_margin_shield_finite_direction_bank.v1"
FREEZE_SCHEMA = "sp_lense.decision_margin_shield_finite_calibration_freeze.v1"
LEDGER_SCHEMA = "sp_lense.decision_margin_shield_finite_calibration_ledger.v1"
CHUNK_SCHEMA = "sp_lense.decision_margin_shield_finite_calibration_chunk.v1"
RESULT_SCHEMA = "sp_lense.decision_margin_shield_finite_calibration_result.v1"

CALIBRATION_FORWARD_COUNT = 1800
QUALIFICATION_FORWARD_COUNT = 8
CALIBRATION_CHUNK_SIZE = 8
EXPECTED_SCREEN_FILE_SHA256 = (
    "db727fadeee64fde13a76313bdc5506c47e9fec87569b9e4a9b19b7c4289f38e"
)
KNOWN_BAD_LEGACY_PROMPT_SHA256 = (
    "430f4efca2025c6bd578c28117e7da2980cbeeb36403faf4f0629cb7bae44718"
)
KNOWN_BAD_LEGACY_LOG_ODDS = -0.4598121643066406
KNOWN_BAD_SOURCE_PATH = (
    ROOT
    / "results"
    / "factorial_causal_anchor_gradient_pilot"
    / "qwen35_08b"
    / "calibration_rows.jsonl"
)
KNOWN_BAD_SOURCE_FILE_SHA256 = (
    "1d1557aeee937d0139083ee0649d2ff03cc3b1534ff3322fcf9bb7cb58da73e7"
)
QUALIFICATION_SELECTION_RULE = (
    "Evaluate every candidate under both A/B orders, then select the first candidate "
    "in this authored order whose unrestricted vocabulary argmax is the preferred "
    "answer with valid answer format under both orders. Do not use margin, confidence, "
    "intervention, direction, or downstream outcome for selection."
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{_relative(path)} must contain a JSON object")
    return value


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable {_relative(path)}")
    _write_json(path, value)


def _with_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result.pop(field, None)
    result[field] = canonical_sha256(result)
    return result


def _verify_hash(value: Mapping[str, Any], field: str) -> None:
    unhashed = dict(value)
    observed = unhashed.pop(field, None)
    if not isinstance(observed, str) or canonical_sha256(unhashed) != observed:
        raise RuntimeError(f"{field} differs")


def _load_original_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dms_locked_screen_runner", ORIGINAL_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the immutable DMS screen runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_dataset() -> dict[str, Any]:
    result = _load_json(DATA_PATH)
    validate_pilot_dataset(result)
    return result


def _load_control_candidates() -> dict[str, Any]:
    payload = _load_json(CONTROL_CANDIDATES_PATH)
    known = payload.get("known_bad_legacy_view")
    candidates = payload.get("candidates")
    if (
        payload.get("schema_version")
        != "sp_lense.decision_margin_shield_finite_control_candidates.v1"
        or payload.get("status")
        != "prospectively_authored_before_control_qualification"
        or payload.get("replacement_for") != "fcag_control_08_instruction"
        or payload.get("selection_rule") != QUALIFICATION_SELECTION_RULE
        or not isinstance(known, Mapping)
        or known.get("form_id")
        != "fcag_control_08_instruction:preferred_first=false"
        or known.get("prompt_sha256") != KNOWN_BAD_LEGACY_PROMPT_SHA256
        or not math.isclose(
            float(known.get("preferred_minus_alternative_log_odds", math.nan)),
            KNOWN_BAD_LEGACY_LOG_ODDS,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or known.get("unrestricted_semantic_choice") != "alternative"
        or known.get("answer_format_valid") is not True
        or known.get("unrestricted_predicted_label") != "A"
        or known.get("preferred_label") != "B"
        or known.get("source_path") != _relative(KNOWN_BAD_SOURCE_PATH)
        or known.get("source_file_sha256") != KNOWN_BAD_SOURCE_FILE_SHA256
        or not isinstance(candidates, list)
        or len(candidates) != 4
    ):
        raise RuntimeError("finite control candidate bank or bad-outcome disclosure differs")
    dataset = _load_dataset()
    existing_ids = {str(row["id"]) for row in dataset["unrelated_controls"]}
    existing_content = {
        (str(row["prompt"]), str(row["preferred_completion"]), str(row["alternative_completion"]))
        for row in dataset["unrelated_controls"]
    }
    ids = []
    content = []
    for candidate in candidates:
        if (
            not isinstance(candidate, Mapping)
            or candidate.get("partition") != "finite_control_candidate"
            or candidate.get("category") != "harmless_benign_compliance"
            or any(
                not isinstance(candidate.get(key), str) or not candidate[key].strip()
                for key in ("id", "prompt", "preferred_completion", "alternative_completion")
            )
        ):
            raise RuntimeError("finite control candidate content differs")
        ids.append(str(candidate["id"]))
        content.append(
            (
                str(candidate["prompt"]),
                str(candidate["preferred_completion"]),
                str(candidate["alternative_completion"]),
            )
        )
    if (
        len(set(ids)) != 4
        or any(value in existing_ids for value in ids)
        or len(set(content)) != 4
        or any(value in existing_content for value in content)
    ):
        raise RuntimeError("finite control candidates are not ordered, unique, and disjoint")
    if file_sha256(KNOWN_BAD_SOURCE_PATH) != KNOWN_BAD_SOURCE_FILE_SHA256:
        raise RuntimeError("the disclosed known-bad baseline source file differs")
    return payload


def _qualification_plan() -> list[dict[str, Any]]:
    dataset = _load_dataset()
    bank = _load_control_candidates()
    plan = []
    for candidate_index, candidate in enumerate(bank["candidates"]):
        for preferred_first in (True, False):
            form = render_unrelated_ab_form(
                dataset, candidate, preferred_first=preferred_first
            )
            plan.append(
                {
                    "work_id": (
                        f"control_qualification:candidate={candidate_index}:"
                        f"preferred_first={str(preferred_first).lower()}"
                    ),
                    "candidate_index": candidate_index,
                    "candidate_id": str(candidate["id"]),
                    "preferred_first": preferred_first,
                    "prompt": str(form["prompt"]),
                    "prompt_sha256": text_sha256(str(form["prompt"])),
                    "preferred_label": str(form["preferred_label"]),
                    "alternative_label": str(form["alternative_label"]),
                    "candidate_sha256": canonical_sha256(dict(candidate)),
                }
            )
    if len(plan) != QUALIFICATION_FORWARD_COUNT or len(
        {row["work_id"] for row in plan}
    ) != QUALIFICATION_FORWARD_COUNT:
        raise RuntimeError("control qualification must contain exactly eight forwards")
    return plan


def _qualification_public_plan() -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key != "prompt"}
        for row in _qualification_plan()
    ]


def _load_screen_result() -> dict[str, Any]:
    if file_sha256(SCREEN_RESULT_PATH) != EXPECTED_SCREEN_FILE_SHA256:
        raise RuntimeError("the selected-layer screen result file hash differs")
    result = _load_json(SCREEN_RESULT_PATH)
    _verify_hash(result, "result_sha256")
    if (
        result.get("result_sha256") != SCREEN_RESULT_SHA256
        or result.get("status") != "selected"
        or result.get("selection", {}).get("selected_layer") != SELECTED_LAYER
        or result.get("pilot_scenario_geometry_computed") is not False
        or result.get("finite_intervention_outcomes_inspected") is not False
    ):
        raise RuntimeError("the finite phase is bound to a different layer-screen result")
    return result


def _source_paths() -> dict[str, Path]:
    return {
        "protocol": PROTOCOL_PATH,
        "runner": SCRIPT_PATH,
        "finite_math": MATH_PATH,
        "finite_math_tests": MATH_TEST_PATH,
        "runner_tests": RUNNER_TEST_PATH,
    }


def _source_records() -> dict[str, dict[str, str]]:
    return {
        key: {"path": _relative(path), "sha256": file_sha256(path)}
        for key, path in _source_paths().items()
    }


def _bound_dependency_records() -> dict[str, dict[str, str]]:
    # Revalidate and bind the complete immutable source closure of the screen that
    # produced the gradients.  Hashing only this runner's direct imports would let
    # a renderer, hook, scorer, or backend change after the finite lock while the
    # nominal protocol identity still appeared unchanged.
    original = _load_original_runner()
    original._load_lock()
    paths = {
        "dataset": DATA_PATH,
        "control_candidates": CONTROL_CANDIDATES_PATH,
        "model_config": MODEL_CONFIG_PATH,
        "original_screen_runner": ORIGINAL_RUNNER_PATH,
        "original_screen_lock": original.LOCK_PATH,
        "rowspace_solver": ROWSPACE_PATH,
        "screen_amendment_runner": SCREEN_AMENDMENT_RUNNER_PATH,
        "screen_amendment_lock": SCREEN_AMENDMENT_LOCK_PATH,
        "screen_result": SCREEN_RESULT_PATH,
        "capture_manifest": CAPTURE_MANIFEST_PATH,
        "known_bad_baseline_source": KNOWN_BAD_SOURCE_PATH,
    }
    for name, path in original._source_paths().items():
        paths[f"original_screen_source_{name}"] = Path(path)
    return {
        key: {"path": _relative(path), "sha256": file_sha256(path)}
        for key, path in paths.items()
    }


def _qualification_dependency_records() -> dict[str, dict[str, str]]:
    original = _load_original_runner()
    original._load_lock()
    paths = {
        "dataset": DATA_PATH,
        "control_candidates": CONTROL_CANDIDATES_PATH,
        "model_config": MODEL_CONFIG_PATH,
        "original_model_loader": ORIGINAL_RUNNER_PATH,
        "factorial_renderer": ROOT / "src" / "sp_lense" / "factorial_causal_anchor.py",
        "choice_runtime": ROOT / "src" / "sp_lense" / "comparison_runtime.py",
        "known_bad_baseline_source": KNOWN_BAD_SOURCE_PATH,
        "original_model_loader_lock": original.LOCK_PATH,
    }
    for name, path in original._source_paths().items():
        paths[f"original_model_loader_source_{name}"] = Path(path)
    return {
        key: {"path": _relative(path), "sha256": file_sha256(path)}
        for key, path in paths.items()
    }


def _runtime(torch: Any) -> dict[str, Any]:
    packages = {}
    for package in (
        "torch",
        "transformers",
        "transformer-lens",
        "huggingface-hub",
        "safetensors",
        "numpy",
        "scipy",
    ):
        packages[package] = importlib.metadata.version(package)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
    }


def proposed_qualification_lock() -> dict[str, Any]:
    bank = _load_control_candidates()
    public_plan = _qualification_public_plan()
    lock = {
        "schema_version": QUALIFICATION_LOCK_SCHEMA,
        "status": (
            "locked_before_control_qualification"
            if QUALIFICATION_LOCK_PATH.exists()
            else "proposed_before_control_qualification"
        ),
        "development_only": True,
        "qualification_only": True,
        "known_bad_legacy_view": dict(bank["known_bad_legacy_view"]),
        "replacement_for": "fcag_control_08_instruction",
        "candidate_bank_path": _relative(CONTROL_CANDIDATES_PATH),
        "candidate_bank_file_sha256": file_sha256(CONTROL_CANDIDATES_PATH),
        "candidate_order_sha256": canonical_sha256(bank["candidates"]),
        "candidate_count": 4,
        "orders": [True, False],
        "selection_rule": QUALIFICATION_SELECTION_RULE,
        "qualification_plan": public_plan,
        "qualification_plan_sha256": canonical_sha256(public_plan),
        "prompt_content_sha256": canonical_sha256(
            [
                {"work_id": row["work_id"], "prompt_sha256": row["prompt_sha256"]}
                for row in public_plan
            ]
        ),
        "compute_ceiling": {
            "model_loads": 1,
            "model_forwards": QUALIFICATION_FORWARD_COUNT,
            "model_backwards": 0,
            "generated_tokens": 0,
            "external_model_judges": 0,
            "external_api_calls": 0,
            "paid_model_cost_usd": 0,
        },
        "prohibited_inputs": {
            "direction_artifacts": True,
            "intervention_outcomes": True,
            "pilot_outcomes": True,
            "margin_based_selection": True,
        },
        "source_files": _source_records(),
        "bound_dependencies": _qualification_dependency_records(),
        "claim_boundary": (
            "Baseline prompt qualification only; it supplies no steering evidence."
        ),
    }
    lock["lock_identity_sha256"] = canonical_sha256(lock)
    return lock


def run_qualification_lock() -> dict[str, Any]:
    if LOCK_PATH.exists():
        raise RuntimeError("finite calibration lock exists before control qualification")
    if QUALIFICATION_LOCK_PATH.exists():
        return _load_qualification_lock()
    proposed = proposed_qualification_lock()
    proposed["status"] = "locked_before_control_qualification"
    proposed.pop("lock_identity_sha256")
    proposed["lock_identity_sha256"] = canonical_sha256(proposed)
    _write_new_json(QUALIFICATION_LOCK_PATH, proposed)
    return _load_qualification_lock()


def _load_qualification_lock() -> dict[str, Any]:
    lock = _load_json(QUALIFICATION_LOCK_PATH)
    _verify_hash(lock, "lock_identity_sha256")
    expected = proposed_qualification_lock()
    expected["status"] = "locked_before_control_qualification"
    expected.pop("lock_identity_sha256")
    expected["lock_identity_sha256"] = canonical_sha256(expected)
    if lock != expected:
        raise RuntimeError("control qualification lock differs from hash-bound sources")
    return lock


def proposed_lock() -> dict[str, Any]:
    qualification = _validate_qualification_result()
    if qualification.get("finite_lock_authorized") is not True:
        raise RuntimeError("control qualification did not authorize the finite lock")
    screen = _load_screen_result()
    capture = _load_json(CAPTURE_MANIFEST_PATH)
    _verify_hash(capture, "capture_manifest_sha256")
    amendment_lock = _load_json(SCREEN_AMENDMENT_LOCK_PATH)
    _verify_hash(amendment_lock, "lock_identity_sha256")
    lock = {
        "schema_version": LOCK_SCHEMA,
        "status": "proposed_not_yet_run" if not LOCK_PATH.exists() else "locked_before_finite_run",
        "development_only": True,
        "outcome_awareness": {
            "float64_screen_geometry_viewed": True,
            "float32_cast_and_simulated_addition_geometry_viewed": True,
            "known_legacy_baseline_behavior_viewed_before_lock": True,
            "prospective_control_qualification_baselines_viewed_before_lock": True,
            "steering_intervention_behavior_viewed_before_lock": False,
            "disclosure": (
                "The known authored-wrong control baseline and the prospectively "
                "qualified replacement's two baseline orders were viewed. No finite "
                "steering intervention was run. The 2e-5 raw deployment tolerance is "
                "inherited from locked FCAGS, not tuned to DMS."
            ),
        },
        "screen_binding": {
            "result_path": _relative(SCREEN_RESULT_PATH),
            "result_file_sha256": file_sha256(SCREEN_RESULT_PATH),
            "result_sha256": screen["result_sha256"],
            "selected_layer": SELECTED_LAYER,
            "selection_sha256": screen["selection"]["selection_sha256"],
        },
        "capture_binding": {
            "manifest_path": _relative(CAPTURE_MANIFEST_PATH),
            "manifest_file_sha256": file_sha256(CAPTURE_MANIFEST_PATH),
            "manifest_sha256": capture["capture_manifest_sha256"],
            "capture_plan_sha256": capture["capture_plan_sha256"],
            "prompt_content_sha256": capture["prompt_content_sha256"],
        },
        "control_qualification_binding": {
            "known_bad_legacy_view": qualification["known_bad_legacy_view"],
            "qualification_lock_path": _relative(QUALIFICATION_LOCK_PATH),
            "qualification_lock_file_sha256": file_sha256(
                QUALIFICATION_LOCK_PATH
            ),
            "qualification_lock_identity_sha256": qualification[
                "qualification_lock_identity_sha256"
            ],
            "qualification_result_path": _relative(QUALIFICATION_RESULT_PATH),
            "qualification_result_file_sha256": file_sha256(
                QUALIFICATION_RESULT_PATH
            ),
            "qualification_result_sha256": qualification[
                "qualification_result_sha256"
            ],
            "candidate_bank_file_sha256": qualification[
                "candidate_bank_file_sha256"
            ],
            "selected_control_id": qualification["selected_control"]["id"],
            "selected_control_sha256": qualification[
                "selected_control_sha256"
            ],
            "selection_used_margin_or_intervention": False,
        },
        "design": {
            "methods": list(METHODS),
            "strengths": list(STRENGTHS),
            "signs": [1, -1],
            "position": "last_token_of_exact_shared_causal_decision_anchor",
            "scenario_direction_scope": "one_local_vector_per_calibration_scenario",
            "construction_unrelated_partition": "nuisance_fit",
            "finite_unrelated_partition": "calibration",
            "same_vector_across_signs_orders_assignments_and_factorial_cells": True,
            "kl_direction": "KL(changed||baseline)",
            "kl_numerical_rule": {
                "input_storage": "float32_logits",
                "computation_dtype": "float64",
                "negative_roundoff_floor": KL_DOUBLE_ROUNDOFF_FLOOR,
                "action": "raise_below_floor_else_clamp_to_zero",
            },
            "kl_limits": dict(KL_LIMITS),
            "complete_assignment_unit_threshold": 6,
            "both_assignments_scenario_threshold": 3,
            "zero_protected_and_unrelated_greedy_or_semantic_changes": True,
            "zero_changed_other_outputs": True,
            "pareto_rule": (
                "DMS no worse in complete units, scenarios with both assignments, "
                "and every stratum's protected mean/p95/max KL, with one strict improvement"
            ),
            "pareto_numerical_tolerance": 1e-8,
            "pilot_command_exists": False,
        },
        "float32_deployment": {
            "raw_constraint_tolerance": FLOAT32_RAW_CONSTRAINT_TOLERANCE,
            "tolerance_provenance": (
                "pre_existing_locked_FCAGS_float32_exact_null_max_abs_projection"
            ),
            "actual_hook_relative_l2_tolerance": HOOK_REALIZATION_RELATIVE_L2_TOLERANCE,
            "literal_exact_deployed_null_claim_prohibited": True,
            "required_wording": "within_locked_float32_numerical_tolerance",
        },
        "compute_ceiling": {
            "maximum_model_loads_per_process": 1,
            "persistent_total_model_loads_claimed": False,
            "model_load_accounting_note": (
                "A resumed calibration process may load the pinned local model once; "
                "the immutable ledger meters forwards, not cross-process load sessions."
            ),
            "model_forwards": CALIBRATION_FORWARD_COUNT,
            "model_backwards": 0,
            "generated_tokens": 0,
            "external_model_judges": 0,
            "external_api_calls": 0,
            "paid_model_cost_usd": 0,
            "excludes_separately_locked_control_qualification": True,
        },
        "exact_forward_accounting": {
            "shared_baselines": 72,
            "target_changed": 288,
            "matched_protected_changed": 864,
            "unrelated_changed": 576,
            "total": CALIBRATION_FORWARD_COUNT,
        },
        "baseline_log_odds_reproduction_tolerance": BASELINE_LOG_ODDS_TOLERANCE,
        "source_files": _source_records(),
        "bound_dependencies": _bound_dependency_records(),
        "claim_boundary": (
            "Opened finite A/B calibration only; no natural mechanism, safety, "
            "general capability, confirmatory, priority, or publication claim."
        ),
    }
    lock["lock_identity_sha256"] = canonical_sha256(lock)
    return lock


def run_lock() -> dict[str, Any]:
    if LOCK_PATH.exists():
        return _load_lock()
    proposed = proposed_lock()
    proposed["status"] = "locked_before_finite_run"
    proposed.pop("lock_identity_sha256")
    proposed["lock_identity_sha256"] = canonical_sha256(proposed)
    _write_new_json(LOCK_PATH, proposed)
    return _load_lock()


def _load_lock() -> dict[str, Any]:
    lock = _load_json(LOCK_PATH)
    _verify_hash(lock, "lock_identity_sha256")
    expected = proposed_lock()
    expected["status"] = "locked_before_finite_run"
    expected.pop("lock_identity_sha256")
    expected["lock_identity_sha256"] = canonical_sha256(expected)
    if lock != expected:
        raise RuntimeError("finite calibration lock differs from current hash-bound sources")
    return lock


def run_preflight() -> dict[str, Any]:
    lock = _load_lock()
    qualification = _validate_qualification_result()
    original = _load_original_runner()
    import torch

    original._configure_threads(torch)
    dataset = _load_dataset()
    if (
        sum(row["partition"] == "calibration" for row in dataset["scenarios"]) != 4
        or sum(
            row["partition"] == "nuisance_fit"
            for row in dataset["unrelated_controls"]
        )
        != 4
        or sum(
            row["partition"] == "calibration"
            for row in dataset["unrelated_controls"]
        )
        != 4
    ):
        raise RuntimeError("finite calibration dataset partitions differ")
    result = _with_hash(
        {
            "schema_version": PREFLIGHT_SCHEMA,
            "status": "ready_for_model_free_direction_reconstruction",
            "development_only": True,
            "lock_file_sha256": file_sha256(LOCK_PATH),
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "screen_result_sha256": _load_screen_result()["result_sha256"],
            "qualification_result_sha256": qualification[
                "qualification_result_sha256"
            ],
            "selected_control_sha256": qualification["selected_control_sha256"],
            "runtime": _runtime(torch),
            "model_loads": 0,
            "model_forwards": 0,
            "model_backwards": 0,
            "generated_tokens": 0,
            "pilot_outcomes_read": False,
        },
        "preflight_sha256",
    )
    if PREFLIGHT_PATH.exists():
        observed = _load_json(PREFLIGHT_PATH)
        _verify_hash(observed, "preflight_sha256")
        if observed != result:
            raise RuntimeError("existing finite preflight differs")
        return observed
    _write_new_json(PREFLIGHT_PATH, result)
    return result


def _validate_preflight() -> dict[str, Any]:
    result = _load_json(PREFLIGHT_PATH)
    _verify_hash(result, "preflight_sha256")
    if result != run_preflight():
        raise RuntimeError("finite preflight is stale")
    return result


def _direction_inputs(
    torch: Any, records: Sequence[Mapping[str, Any]], scenario_id: str
) -> dict[str, Any]:
    scenario_records = [
        row
        for row in records
        if row["kind"] == "scenario" and row["scenario_id"] == scenario_id
    ]
    nuisance = [row for row in records if row["kind"] == "nuisance_fit"]
    if len(scenario_records) != 16 or len(nuisance) != 8:
        raise RuntimeError("direction reconstruction capture coverage differs")
    target_records = [
        row
        for row in scenario_records
        if (row["target"], row["event"]) == ("self", "permanent")
    ]
    protected_records = [
        row
        for row in scenario_records
        if (row["target"], row["event"]) != ("self", "permanent")
    ]
    scale = float(
        anchor_residual_scale_geometric_mean(
            torch, [row["anchor_residual"] for row in scenario_records]
        )[SELECTED_LAYER].item()
    )
    target_rows = scale * torch.stack(
        [row["gradient"][SELECTED_LAYER].double() for row in target_records]
    )
    protected_rows = scale * torch.stack(
        [row["gradient"][SELECTED_LAYER].double() for row in protected_records]
    )
    unrelated_rows = scale * torch.stack(
        [row["gradient"][SELECTED_LAYER].double() for row in nuisance]
    )
    return {
        "residual_scale": scale,
        "target_rows": target_rows.numpy(),
        "target_offsets": np.asarray(
            [row["preserve_minus_comply_baseline_log_odds"] for row in target_records],
            dtype=np.float64,
        ),
        "protected_rows": protected_rows.numpy(),
        "protected_offsets": np.asarray(
            [row["preserve_minus_comply_baseline_log_odds"] for row in protected_records],
            dtype=np.float64,
        ),
        "unrelated_rows": unrelated_rows.numpy(),
        "captured_anchor_residuals": torch.stack(
            [
                *[row["anchor_residual"][SELECTED_LAYER] for row in scenario_records],
                *[row["anchor_residual"][SELECTED_LAYER] for row in nuisance],
            ]
        )
        .float()
        .numpy(),
    }


def run_construct() -> dict[str, Any]:
    lock = _load_lock()
    preflight = _validate_preflight()
    if DIRECTION_MANIFEST_PATH.exists() or DIRECTION_TENSOR_PATH.exists():
        return _load_direction_bank()[1]
    original = _load_original_runner()
    import torch

    records = original._load_capture_records(torch)
    dataset = _load_dataset()
    screen = _load_screen_result()
    scenarios = [row for row in dataset["scenarios"] if row["partition"] == "calibration"]
    tensor_rows = []
    public_rows = []
    for scenario in scenarios:
        scenario_id = str(scenario["id"])
        inputs = _direction_inputs(torch, records, scenario_id)
        directions = reconstruct_scenario_directions(
            scenario_id=scenario_id,
            screen_result=screen,
            **{key: value for key, value in inputs.items() if key != "captured_anchor_residuals"},
        )
        for method in METHODS:
            direction = directions[method]
            deployment = deployment_recertificate(
                direction,
                target_rows=inputs["target_rows"],
                target_offsets=inputs["target_offsets"],
                protected_rows=inputs["protected_rows"],
                protected_offsets=inputs["protected_offsets"],
                unrelated_rows=inputs["unrelated_rows"],
                captured_anchor_residuals=inputs["captured_anchor_residuals"],
            )
            if not deployment["passes"]:
                raise RuntimeError("a float32 deployment certificate failed")
            row_index = len(tensor_rows)
            tensor_rows.append(
                (
                    torch.from_numpy(direction.standardized_direction.copy()).double(),
                    torch.from_numpy(direction.physical_direction.copy()).float(),
                )
            )
            public_rows.append(
                {
                    "row_index": row_index,
                    "scenario_id": scenario_id,
                    "method": method,
                    "layer": SELECTED_LAYER,
                    "residual_scale": direction.residual_scale,
                    "standardized_l2": direction.standardized_l2,
                    "float64_direction_sha256": direction.direction_sha256,
                    "physical_float32_sha256": direction.physical_float32_sha256,
                    "screen_method": direction.screen_method,
                    "screen_record_sha256": direction.screen_record_sha256,
                    "deployment_certificate": deployment,
                }
            )
    if len(tensor_rows) != 12 or not all(
        row["deployment_certificate"]["passes"] for row in public_rows
    ):
        raise RuntimeError("finite direction bank must contain 12 deployment-certified rows")
    payload = {
        "schema_version": DIRECTION_SCHEMA,
        "screen_result_sha256": screen["result_sha256"],
        "selected_layer": SELECTED_LAYER,
        "records": public_rows,
        "standardized_directions": torch.stack([row[0] for row in tensor_rows]),
        "physical_directions": torch.stack([row[1] for row in tensor_rows]),
    }
    DIRECTION_TENSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = DIRECTION_TENSOR_PATH.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, DIRECTION_TENSOR_PATH)
    manifest = _with_hash(
        {
            "schema_version": DIRECTION_SCHEMA,
            "status": "reconstructed_and_float32_deployment_recertified",
            "lock_file_sha256": file_sha256(LOCK_PATH),
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "preflight_file_sha256": file_sha256(PREFLIGHT_PATH),
            "preflight_sha256": preflight["preflight_sha256"],
            "screen_result_file_sha256": file_sha256(SCREEN_RESULT_PATH),
            "screen_result_sha256": screen["result_sha256"],
            "capture_manifest_file_sha256": file_sha256(CAPTURE_MANIFEST_PATH),
            "capture_manifest_sha256": _load_json(CAPTURE_MANIFEST_PATH)[
                "capture_manifest_sha256"
            ],
            "tensor_path": _relative(DIRECTION_TENSOR_PATH),
            "tensor_file_sha256": file_sha256(DIRECTION_TENSOR_PATH),
            "records": public_rows,
            "direction_count": len(public_rows),
            "model_loads": 0,
            "model_forwards": 0,
            "model_backwards": 0,
            "finite_behavior_inspected": False,
            "outcome_awareness": (
                "float32 cast geometry was viewed before the deployment rule; no finite "
                "behavior was run or viewed"
            ),
        },
        "direction_manifest_sha256",
    )
    _write_new_json(DIRECTION_MANIFEST_PATH, manifest)
    return _load_direction_bank()[1]


def _load_direction_bank() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    import torch

    lock = _load_lock()
    preflight = _validate_preflight()
    screen = _load_screen_result()
    capture_manifest = _load_json(CAPTURE_MANIFEST_PATH)
    _verify_hash(capture_manifest, "capture_manifest_sha256")
    manifest = _load_json(DIRECTION_MANIFEST_PATH)
    _verify_hash(manifest, "direction_manifest_sha256")
    if (
        manifest.get("schema_version") != DIRECTION_SCHEMA
        or manifest.get("status")
        != "reconstructed_and_float32_deployment_recertified"
        or manifest.get("lock_file_sha256") != file_sha256(LOCK_PATH)
        or manifest.get("lock_identity_sha256") != lock["lock_identity_sha256"]
        or manifest.get("preflight_file_sha256") != file_sha256(PREFLIGHT_PATH)
        or manifest.get("preflight_sha256") != preflight["preflight_sha256"]
        or manifest.get("screen_result_sha256") != SCREEN_RESULT_SHA256
        or manifest.get("screen_result_file_sha256") != file_sha256(SCREEN_RESULT_PATH)
        or manifest.get("capture_manifest_file_sha256")
        != file_sha256(CAPTURE_MANIFEST_PATH)
        or manifest.get("capture_manifest_sha256")
        != capture_manifest["capture_manifest_sha256"]
        or manifest.get("direction_count") != 12
        or manifest.get("tensor_path") != _relative(DIRECTION_TENSOR_PATH)
        or manifest.get("tensor_file_sha256") != file_sha256(DIRECTION_TENSOR_PATH)
        or manifest.get("finite_behavior_inspected") is not False
        or manifest.get("model_loads") != 0
        or manifest.get("model_forwards") != 0
        or manifest.get("model_backwards") != 0
    ):
        raise RuntimeError("finite direction manifest provenance differs")
    payload = torch.load(DIRECTION_TENSOR_PATH, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {
            "schema_version",
            "screen_result_sha256",
            "selected_layer",
            "records",
            "standardized_directions",
            "physical_directions",
        }
        or payload.get("schema_version") != DIRECTION_SCHEMA
        or payload.get("screen_result_sha256") != SCREEN_RESULT_SHA256
        or payload.get("selected_layer") != SELECTED_LAYER
        or payload.get("records") != manifest.get("records")
    ):
        raise RuntimeError("finite direction tensor payload identity differs")
    standardized = payload["standardized_directions"].double().contiguous()
    physical = payload["physical_directions"].float().contiguous()
    if tuple(standardized.shape) != (12, 1024) or tuple(physical.shape) != (12, 1024):
        raise RuntimeError("finite direction tensor shape differs")
    dataset = _load_dataset()
    scenario_ids = [
        str(row["id"])
        for row in dataset["scenarios"]
        if row["partition"] == "calibration"
    ]
    expected_keys = [
        (scenario_id, method)
        for scenario_id in scenario_ids
        for method in METHODS
    ]
    records = manifest.get("records")
    if not isinstance(records, list) or [
        (int(row.get("row_index", -1)), row.get("scenario_id"), row.get("method"))
        for row in records
    ] != [
        (index, scenario_id, method)
        for index, (scenario_id, method) in enumerate(expected_keys)
    ]:
        raise RuntimeError("finite direction bank row-index/key mapping differs")
    original = _load_original_runner()
    capture_records = original._load_capture_records(torch)
    result = {}
    for scenario_id in scenario_ids:
        inputs = _direction_inputs(torch, capture_records, scenario_id)
        reconstructed = reconstruct_scenario_directions(
            scenario_id=scenario_id,
            screen_result=screen,
            **{
                key: value
                for key, value in inputs.items()
                if key != "captured_anchor_residuals"
            },
        )
        for method in METHODS:
            index = expected_keys.index((scenario_id, method))
            record = records[index]
            direction = reconstructed[method]
            expected_certificate = deployment_recertificate(
                direction,
                target_rows=inputs["target_rows"],
                target_offsets=inputs["target_offsets"],
                protected_rows=inputs["protected_rows"],
                protected_offsets=inputs["protected_offsets"],
                unrelated_rows=inputs["unrelated_rows"],
                captured_anchor_residuals=inputs["captured_anchor_residuals"],
            )
            certificate = record.get("deployment_certificate")
            if not isinstance(certificate, Mapping):
                raise TypeError("finite direction deployment certificate is missing")
            _verify_hash(certificate, "certificate_sha256")
            if certificate != expected_certificate:
                raise RuntimeError("finite direction deployment certificate differs")
            expected_record = {
                "row_index": index,
                "scenario_id": scenario_id,
                "method": method,
                "layer": SELECTED_LAYER,
                "residual_scale": direction.residual_scale,
                "standardized_l2": direction.standardized_l2,
                "float64_direction_sha256": direction.direction_sha256,
                "physical_float32_sha256": direction.physical_float32_sha256,
                "screen_method": direction.screen_method,
                "screen_record_sha256": direction.screen_record_sha256,
                "deployment_certificate": expected_certificate,
            }
            if record != expected_record:
                raise RuntimeError("finite direction manifest record differs")
            expected_standardized = torch.from_numpy(
                direction.standardized_direction.copy()
            ).double()
            expected_physical = torch.from_numpy(
                direction.physical_direction.copy()
            ).float()
            if not torch.equal(standardized[index], expected_standardized):
                raise RuntimeError("standardized direction tensor differs from reconstruction")
            if not torch.equal(physical[index], expected_physical):
                raise RuntimeError("physical direction tensor differs from reconstruction")
            if not torch.equal(
                physical[index],
                (standardized[index] * float(record["residual_scale"])).float(),
            ):
                raise RuntimeError("physical direction is not float32(scale * standardized)")
            if (
                array_float64_sha256(standardized[index].numpy())
                != record["float64_direction_sha256"]
                or array_float32_sha256(physical[index].numpy())
                != record["physical_float32_sha256"]
                or not math.isclose(
                    float(standardized[index].norm().item()),
                    float(record["standardized_l2"]),
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                )
                or certificate["passes"] is not True
                or certificate[
                    "maximum_simulated_requested_minus_realized_relative_l2"
                ]
                > HOOK_REALIZATION_RELATIVE_L2_TOLERANCE
            ):
                raise RuntimeError("finite direction tensor hash, norm, or certificate differs")
            key = (scenario_id, method)
            if key in result:
                raise RuntimeError("finite direction bank repeats a scenario/method key")
            result[key] = {
                **record,
                "standardized": standardized[index],
                "physical": physical[index],
            }
    if len(result) != 12:
        raise RuntimeError("finite direction bank key coverage differs")
    return result, manifest


def _scenario_anchor_indices() -> dict[str, int]:
    original = _load_original_runner()
    import torch

    records = original._load_capture_records(torch)
    result = {
        str(row["form_id"]): int(row["anchor_index"])
        for row in records
        if row["kind"] == "scenario" and row["partition"] == "calibration"
    }
    if len(result) != 64:
        raise RuntimeError("finite calibration requires 64 captured scenario anchor indices")
    return result


def _calibration_plan() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    replacement = _selected_replacement_control()
    plan, audit = build_calibration_plan(
        _load_dataset(),
        scenario_anchor_indices=_scenario_anchor_indices(),
        replacement_control=replacement,
    )
    if (
        audit["legacy_bad_control_present"] is not False
        or audit["replacement_control_id"] != replacement["id"]
        or audit["planned_forward_count"] != CALIBRATION_FORWARD_COUNT
    ):
        raise RuntimeError("finite calibration did not replace only the known-bad control")
    return plan, audit


def run_freeze() -> dict[str, Any]:
    lock = _load_lock()
    _validate_preflight()
    _, manifest = _load_direction_bank()
    plan, audit = _calibration_plan()
    freeze = _with_hash(
        {
            "schema_version": FREEZE_SCHEMA,
            "status": "frozen_before_first_finite_calibration_forward",
            "development_only": True,
            "lock_file_sha256": file_sha256(LOCK_PATH),
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "direction_manifest_file_sha256": file_sha256(DIRECTION_MANIFEST_PATH),
            "direction_manifest_sha256": manifest["direction_manifest_sha256"],
            "screen_result_sha256": SCREEN_RESULT_SHA256,
            "plan_sha256": plan_sha256(plan),
            "plan_audit": audit,
            "planned_forward_evaluations": CALIBRATION_FORWARD_COUNT,
            "planned_backward_evaluations": 0,
            "generated_tokens": 0,
            "external_model_judges": 0,
            "external_api_calls": 0,
            "pilot_outcomes_read": False,
            "pilot_command_exists": False,
        },
        "freeze_sha256",
    )
    if FREEZE_PATH.exists():
        observed = _load_json(FREEZE_PATH)
        _verify_hash(observed, "freeze_sha256")
        if observed != freeze:
            raise RuntimeError("existing finite calibration freeze differs")
        return observed
    _write_new_json(FREEZE_PATH, freeze)
    return freeze


def _load_freeze() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    freeze = _load_json(FREEZE_PATH)
    _verify_hash(freeze, "freeze_sha256")
    plan, audit = _calibration_plan()
    if (
        freeze != run_freeze()
        or freeze.get("plan_sha256") != plan_sha256(plan)
        or freeze.get("plan_audit") != audit
        or freeze.get("planned_forward_evaluations") != len(plan)
    ):
        raise RuntimeError("finite calibration freeze or plan differs")
    return freeze, plan


def _chunked(values: Sequence[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


class CalibrationLedger:
    def __init__(
        self,
        *,
        path: Path,
        plan_sha256_value: str,
        lock_identity_sha256: str,
        expected_chunk_work_ids: Sequence[Sequence[str]],
    ) -> None:
        self.path = path
        self.plan_sha256 = plan_sha256_value
        self.lock_identity_sha256 = lock_identity_sha256
        self.expected = [list(map(str, row)) for row in expected_chunk_work_ids]
        if not self.expected or any(not row for row in self.expected):
            raise ValueError("calibration ledger requires non-empty chunks")
        if path.exists():
            self.payload = _load_json(path)
            _verify_hash(self.payload, "ledger_sha256")
        else:
            self.payload = {
                "schema_version": LEDGER_SCHEMA,
                "phase": "calibration",
                "plan_sha256": self.plan_sha256,
                "lock_identity_sha256": self.lock_identity_sha256,
                "ceiling": {"forward": CALIBRATION_FORWARD_COUNT, "backward": 0},
                "events": [],
            }
            self._persist()
        self._validate()

    def _persist(self) -> None:
        self.payload = _with_hash(
            {key: value for key, value in self.payload.items() if key != "ledger_sha256"},
            "ledger_sha256",
        )
        _write_json(self.path, self.payload)

    def _validate(self) -> None:
        if (
            self.payload.get("schema_version") != LEDGER_SCHEMA
            or self.payload.get("phase") != "calibration"
            or self.payload.get("plan_sha256") != self.plan_sha256
            or self.payload.get("lock_identity_sha256") != self.lock_identity_sha256
            or self.payload.get("ceiling")
            != {"forward": CALIBRATION_FORWARD_COUNT, "backward": 0}
        ):
            raise RuntimeError("calibration ledger identity differs")
        prior = None
        seen = set()
        events = self.payload.get("events")
        if not isinstance(events, list):
            raise TypeError("calibration ledger events must be a list")
        for index, event in enumerate(events):
            if index >= len(self.expected) or event.get("chunk_index") != index:
                raise RuntimeError("calibration ledger is not a contiguous plan prefix")
            if event.get("work_ids") != self.expected[index]:
                raise RuntimeError("calibration ledger work IDs differ from plan")
            if (
                event.get("forward_evaluations") != len(self.expected[index])
                or event.get("backward_evaluations") != 0
            ):
                raise RuntimeError("calibration ledger event compute differs from work IDs")
            if any(work_id in seen for work_id in event["work_ids"]):
                raise RuntimeError("calibration ledger repeats a work ID")
            seen.update(event["work_ids"])
            if event.get("prior_event_sha256") != prior:
                raise RuntimeError("calibration ledger hash chain differs")
            unhashed = dict(event)
            observed = unhashed.pop("event_sha256", None)
            if canonical_sha256(unhashed) != observed:
                raise RuntimeError("calibration ledger event hash differs")
            prior = observed
            if event.get("status") not in {"pending", "complete"}:
                raise RuntimeError("calibration ledger event status differs")
            if event["status"] == "pending":
                if index != len(events) - 1:
                    raise RuntimeError("pending calibration ledger event is not terminal")
                if (
                    event.get("artifact_path") is not None
                    or event.get("artifact_sha256") is not None
                ):
                    raise RuntimeError("pending calibration ledger event names an artifact")
            else:
                expected_path = _chunk_path(index)
                if (
                    event.get("artifact_path") != _relative(expected_path)
                    or not expected_path.is_file()
                    or event.get("artifact_sha256") != file_sha256(expected_path)
                ):
                    raise RuntimeError("completed calibration ledger artifact differs")
        forward = sum(int(event["forward_evaluations"]) for event in events)
        backward = sum(int(event["backward_evaluations"]) for event in events)
        if forward > CALIBRATION_FORWARD_COUNT or backward != 0:
            raise RuntimeError("calibration ledger exceeds compute ceiling")

    def completed_chunks(self) -> int:
        events = self.payload["events"]
        if events and events[-1]["status"] == "pending":
            raise RuntimeError("calibration has an ambiguous pending chunk")
        return len(events)

    def reserve(self, index: int, work_ids: Sequence[str]) -> None:
        if self.completed_chunks() != index or list(work_ids) != self.expected[index]:
            raise RuntimeError("calibration chunk reservation differs from plan")
        prior = self.payload["events"][-1]["event_sha256"] if self.payload["events"] else None
        event = {
            "chunk_index": index,
            "work_ids": list(work_ids),
            "forward_evaluations": len(work_ids),
            "backward_evaluations": 0,
            "status": "pending",
            "artifact_path": None,
            "artifact_sha256": None,
            "prior_event_sha256": prior,
        }
        event["event_sha256"] = canonical_sha256(event)
        self.payload["events"].append(event)
        self._validate()
        self._persist()

    def complete(self, index: int, artifact_path: Path) -> None:
        event = dict(self.payload["events"][-1])
        if event["chunk_index"] != index or event["status"] != "pending":
            raise RuntimeError("calibration ledger has no matching pending chunk")
        if artifact_path.resolve() != _chunk_path(index).resolve():
            raise RuntimeError("calibration ledger completion path differs from plan")
        event.update(
            {
                "status": "complete",
                "artifact_path": _relative(artifact_path),
                "artifact_sha256": file_sha256(artifact_path),
            }
        )
        event.pop("event_sha256")
        event["event_sha256"] = canonical_sha256(event)
        self.payload["events"][-1] = event
        self._validate()
        self._persist()

    def snapshot(self) -> dict[str, Any]:
        events = self.payload["events"]
        return {
            "forward_evaluations": sum(int(row["forward_evaluations"]) for row in events),
            "backward_evaluations": 0,
            "completed_chunk_count": sum(row["status"] == "complete" for row in events),
            "work_ids_sha256": canonical_sha256(
                [work_id for row in events for work_id in row["work_ids"]]
            ),
            "ledger_file_sha256": file_sha256(self.path),
            "ledger_sha256": self.payload["ledger_sha256"],
        }


def _chunk_path(index: int) -> Path:
    return CHECKPOINT_ROOT / f"chunk-{index:04d}.pt"


def _save_chunk(
    torch: Any,
    *,
    path: Path,
    index: int,
    plan_hash: str,
    expected_specs: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    baseline_logits: Any | None,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable {_relative(path)}")
    public_specs = [public_work_spec(row) for row in expected_specs]
    if [row["work_id"] for row in records] != [row["work_id"] for row in public_specs]:
        raise RuntimeError("calibration chunk records differ from expected work IDs")
    for expected, record in zip(public_specs, records, strict=True):
        if any(record.get(key) != value for key, value in expected.items()):
            raise RuntimeError("calibration chunk record differs from its public work spec")
        unhashed = dict(record)
        observed = unhashed.pop("row_sha256", None)
        if not isinstance(observed, str) or canonical_sha256(unhashed) != observed:
            raise RuntimeError("calibration chunk row self-hash differs")
    tensors = {}
    if baseline_logits is not None:
        tensor = baseline_logits.detach().cpu().float().contiguous()
        if tensor.ndim != 2 or tuple(tensor.shape[:1]) != (len(records),):
            raise RuntimeError("baseline chunk tensor count differs")
        tensors["baseline_logits"] = tensor
    payload = {
        "schema_version": CHUNK_SCHEMA,
        "phase": "calibration",
        "chunk_index": index,
        "plan_sha256": plan_hash,
        "public_specifications_sha256": canonical_sha256(public_specs),
        "records": list(records),
        "tensor_names": sorted(tensors),
        "baseline_logits_bundle_sha256": (
            None
            if baseline_logits is None
            else canonical_sha256([row["logits_float32_sha256"] for row in records])
        ),
    }
    payload["chunk_sha256"] = canonical_sha256(payload)
    payload["tensors"] = tensors
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_chunk(
    torch: Any,
    *,
    path: Path,
    index: int,
    plan_hash: str,
    expected_specs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], Any | None]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    tensors = payload.pop("tensors")
    observed_hash = payload.get("chunk_sha256")
    unhashed = dict(payload)
    unhashed.pop("chunk_sha256")
    if canonical_sha256(unhashed) != observed_hash:
        raise RuntimeError("calibration chunk logical hash differs")
    public_specs = [public_work_spec(row) for row in expected_specs]
    if (
        payload.get("schema_version") != CHUNK_SCHEMA
        or payload.get("phase") != "calibration"
        or payload.get("chunk_index") != index
        or payload.get("plan_sha256") != plan_hash
        or payload.get("public_specifications_sha256") != canonical_sha256(public_specs)
    ):
        raise RuntimeError("calibration chunk identity differs")
    records = payload["records"]
    if [row["work_id"] for row in records] != [row["work_id"] for row in public_specs]:
        raise RuntimeError("calibration chunk work IDs differ")
    for expected, record in zip(public_specs, records, strict=True):
        if any(record.get(key) != value for key, value in expected.items()):
            raise RuntimeError("calibration chunk row differs from its frozen work spec")
        unhashed = dict(record)
        observed_row_hash = unhashed.pop("row_sha256", None)
        if (
            not isinstance(observed_row_hash, str)
            or canonical_sha256(unhashed) != observed_row_hash
        ):
            raise RuntimeError("calibration chunk row self-hash differs")
    baseline_logits = tensors.get("baseline_logits")
    kinds = {row["kind"] for row in records}
    if kinds == {"baseline"}:
        if (
            baseline_logits is None
            or baseline_logits.ndim != 2
            or tuple(baseline_logits.shape[:1]) != (len(records),)
        ):
            raise RuntimeError("baseline calibration chunk lacks logits")
        for row_index, row in enumerate(records):
            if tensor_float32_sha256(baseline_logits[row_index]) != row["logits_float32_sha256"]:
                raise RuntimeError("baseline logits hash differs")
    elif kinds == {"changed"}:
        if baseline_logits is not None or tensors:
            raise RuntimeError("changed calibration chunks must store compact scalar rows")
    else:
        raise RuntimeError("calibration chunk mixes baseline and changed rows")
    return records, baseline_logits


def _generic_label_token_id(backend: Any, prompt: str, label: str) -> int:
    boundary = resolve_choice_boundary(backend, prompt)
    prompt_tokens, full_tokens = encode_prompt_and_completion(
        backend, prompt, label, include_chat_end=True
    )
    suffix = [int(value) for value in full_tokens[0, int(prompt_tokens.shape[1]) :].tolist()]
    end = list(boundary.assistant_end_token_ids)
    if len(suffix) <= len(end) or suffix[-len(end) :] != end:
        raise RuntimeError("choice completion lacks the locked assistant end marker")
    content = suffix[: -len(end)]
    if len(content) != 1:
        raise RuntimeError("choice label is not exactly one content token")
    return content[0]


def _qualification_score(
    logits: Any, *, preferred_token_id: int, alternative_token_id: int
) -> dict[str, Any]:
    values = logits.detach().cpu().float().contiguous()
    if values.ndim != 1:
        raise ValueError("control-qualification logits must be one-dimensional")
    preferred = int(preferred_token_id)
    alternative = int(alternative_token_id)
    if (
        preferred == alternative
        or min(preferred, alternative) < 0
        or max(preferred, alternative) >= int(values.shape[0])
    ):
        raise ValueError("control-qualification answer token IDs differ")
    predicted = int(values.argmax().item())
    semantic = (
        "preferred"
        if predicted == preferred
        else "alternative"
        if predicted == alternative
        else "OTHER"
    )
    return {
        "preferred_token_id": preferred,
        "alternative_token_id": alternative,
        "predicted_token_id": predicted,
        "unrestricted_semantic_choice": semantic,
        "answer_format_valid": semantic != "OTHER",
        "pair_semantic_choice": (
            "preferred"
            if float(values[preferred]) >= float(values[alternative])
            else "alternative"
        ),
        "preferred_minus_alternative_log_odds": float(
            (values[preferred] - values[alternative]).item()
        ),
        "logits_float32_sha256": tensor_float32_sha256(values),
    }


def _qualification_ledger_payload(
    *, lock: Mapping[str, Any], status: str, artifact_sha256: str | None
) -> dict[str, Any]:
    if status not in {"pending", "complete"}:
        raise ValueError("control-qualification ledger status differs")
    value = {
        "schema_version": QUALIFICATION_LEDGER_SCHEMA,
        "phase": "control_qualification",
        "qualification_lock_file_sha256": file_sha256(QUALIFICATION_LOCK_PATH),
        "qualification_lock_identity_sha256": lock["lock_identity_sha256"],
        "qualification_plan_sha256": lock["qualification_plan_sha256"],
        "work_ids": [row["work_id"] for row in lock["qualification_plan"]],
        "forward_evaluations": QUALIFICATION_FORWARD_COUNT,
        "backward_evaluations": 0,
        "generated_tokens": 0,
        "status": status,
        "artifact_path": (
            None if status == "pending" else _relative(QUALIFICATION_CHECKPOINT_PATH)
        ),
        "artifact_sha256": None if status == "pending" else artifact_sha256,
    }
    return _with_hash(value, "ledger_sha256")


def _load_qualification_ledger(
    *, require_complete: bool
) -> dict[str, Any]:
    lock = _load_qualification_lock()
    ledger = _load_json(QUALIFICATION_LEDGER_PATH)
    _verify_hash(ledger, "ledger_sha256")
    status = str(ledger.get("status"))
    artifact_sha = (
        file_sha256(QUALIFICATION_CHECKPOINT_PATH)
        if status == "complete" and QUALIFICATION_CHECKPOINT_PATH.is_file()
        else None
    )
    expected = _qualification_ledger_payload(
        lock=lock, status=status, artifact_sha256=artifact_sha
    )
    if ledger != expected:
        raise RuntimeError("control-qualification ledger or checkpoint binding differs")
    if require_complete and status != "complete":
        raise RuntimeError(
            "control qualification has an ambiguous pending eight-forward batch"
        )
    return ledger


def _save_qualification_checkpoint(
    torch: Any,
    *,
    lock: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    logits: Any,
    backend_metadata: Mapping[str, Any],
) -> None:
    if QUALIFICATION_CHECKPOINT_PATH.exists():
        raise FileExistsError("refusing to overwrite immutable control qualification checkpoint")
    tensor = logits.detach().cpu().float().contiguous()
    if tensor.ndim != 2 or tuple(tensor.shape[:1]) != (QUALIFICATION_FORWARD_COUNT,):
        raise RuntimeError("control-qualification checkpoint logits count differs")
    public_plan = _qualification_public_plan()
    if [row.get("work_id") for row in records] != [
        row["work_id"] for row in public_plan
    ]:
        raise RuntimeError("control-qualification checkpoint rows differ from plan")
    for expected, record in zip(public_plan, records, strict=True):
        if any(record.get(key) != value for key, value in expected.items()):
            raise RuntimeError("control-qualification row differs from locked plan")
        unhashed = dict(record)
        observed = unhashed.pop("row_sha256", None)
        if not isinstance(observed, str) or canonical_sha256(unhashed) != observed:
            raise RuntimeError("control-qualification row self-hash differs")
    metadata = {
        "schema_version": QUALIFICATION_CHECKPOINT_SCHEMA,
        "phase": "control_qualification",
        "qualification_lock_file_sha256": file_sha256(QUALIFICATION_LOCK_PATH),
        "qualification_lock_identity_sha256": lock["lock_identity_sha256"],
        "qualification_plan_sha256": lock["qualification_plan_sha256"],
        "public_plan_sha256": canonical_sha256(public_plan),
        "records": list(records),
        "logits_bundle_sha256": canonical_sha256(
            [row["logits_float32_sha256"] for row in records]
        ),
        "backend_metadata": dict(backend_metadata),
        "compute": {
            "model_loads": 1,
            "model_forwards": QUALIFICATION_FORWARD_COUNT,
            "model_backwards": 0,
            "generated_tokens": 0,
            "external_model_judges": 0,
            "external_api_calls": 0,
            "paid_model_cost_usd": 0,
        },
        "directions_or_interventions_inspected": False,
    }
    metadata["checkpoint_sha256"] = canonical_sha256(metadata)
    payload = {**metadata, "logits": tensor}
    QUALIFICATION_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = QUALIFICATION_CHECKPOINT_PATH.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, QUALIFICATION_CHECKPOINT_PATH)


def _load_qualification_checkpoint(torch: Any) -> tuple[dict[str, Any], Any]:
    lock = _load_qualification_lock()
    ledger = _load_qualification_ledger(require_complete=True)
    payload = torch.load(
        QUALIFICATION_CHECKPOINT_PATH, map_location="cpu", weights_only=True
    )
    if not isinstance(payload, dict) or "logits" not in payload:
        raise RuntimeError("control-qualification checkpoint payload differs")
    logits = payload.pop("logits").float().contiguous()
    _verify_hash(payload, "checkpoint_sha256")
    public_plan = _qualification_public_plan()
    if (
        payload.get("schema_version") != QUALIFICATION_CHECKPOINT_SCHEMA
        or payload.get("phase") != "control_qualification"
        or payload.get("qualification_lock_file_sha256")
        != file_sha256(QUALIFICATION_LOCK_PATH)
        or payload.get("qualification_lock_identity_sha256")
        != lock["lock_identity_sha256"]
        or payload.get("qualification_plan_sha256")
        != lock["qualification_plan_sha256"]
        or payload.get("public_plan_sha256") != canonical_sha256(public_plan)
        or logits.ndim != 2
        or tuple(logits.shape[:1]) != (QUALIFICATION_FORWARD_COUNT,)
        or payload.get("compute") != lock["compute_ceiling"]
        or payload.get("directions_or_interventions_inspected") is not False
        or ledger.get("artifact_sha256") != file_sha256(QUALIFICATION_CHECKPOINT_PATH)
    ):
        raise RuntimeError("control-qualification checkpoint provenance differs")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != QUALIFICATION_FORWARD_COUNT:
        raise RuntimeError("control-qualification checkpoint row count differs")
    for index, (expected, record) in enumerate(
        zip(public_plan, records, strict=True)
    ):
        if any(record.get(key) != value for key, value in expected.items()):
            raise RuntimeError("control-qualification checkpoint row identity differs")
        unhashed = dict(record)
        observed = unhashed.pop("row_sha256", None)
        if not isinstance(observed, str) or canonical_sha256(unhashed) != observed:
            raise RuntimeError("control-qualification checkpoint row hash differs")
        score = _qualification_score(
            logits[index],
            preferred_token_id=int(record["preferred_token_id"]),
            alternative_token_id=int(record["alternative_token_id"]),
        )
        if any(record.get(key) != value for key, value in score.items()):
            raise RuntimeError("control-qualification score differs from stored logits")
    if payload.get("logits_bundle_sha256") != canonical_sha256(
        [row["logits_float32_sha256"] for row in records]
    ):
        raise RuntimeError("control-qualification logits bundle differs")
    return payload, logits


def _select_qualified_control(
    records: Sequence[Mapping[str, Any]], bank: Mapping[str, Any]
) -> dict[str, Any]:
    candidates = bank["candidates"]
    assessments = []
    selected: Mapping[str, Any] | None = None
    selected_hash: str | None = None
    for candidate_index, candidate in enumerate(candidates):
        rows = [
            row
            for row in records
            if int(row["candidate_index"]) == candidate_index
            and row["candidate_id"] == candidate["id"]
        ]
        if len(rows) != 2 or {row["preferred_first"] for row in rows} != {True, False}:
            raise RuntimeError("control-qualification candidate/order coverage differs")
        passes = all(
            row["answer_format_valid"] is True
            and row["unrestricted_semantic_choice"] == "preferred"
            for row in rows
        )
        assessments.append(
            {
                "candidate_index": candidate_index,
                "candidate_id": candidate["id"],
                "candidate_sha256": canonical_sha256(dict(candidate)),
                "passes_both_orders": passes,
                "order_rows": [
                    {
                        "preferred_first": row["preferred_first"],
                        "prompt_sha256": row["prompt_sha256"],
                        "unrestricted_semantic_choice": row[
                            "unrestricted_semantic_choice"
                        ],
                        "answer_format_valid": row["answer_format_valid"],
                        "row_sha256": row["row_sha256"],
                    }
                    for row in rows
                ],
            }
        )
        if selected is None and passes:
            selected = candidate
            selected_hash = canonical_sha256(
                {**dict(candidate), "replacement_for": bank["replacement_for"]}
            )
    return {
        "status": "passed" if selected is not None else "no_candidate_passed",
        "finite_lock_authorized": selected is not None,
        "candidate_assessments": assessments,
        "selected_control": (
            None
            if selected is None
            else {**dict(selected), "replacement_for": bank["replacement_for"]}
        ),
        "selected_control_sha256": selected_hash,
    }


def _expected_qualification_result(
    checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    lock = _load_qualification_lock()
    ledger = _load_qualification_ledger(require_complete=True)
    bank = _load_control_candidates()
    selection = _select_qualified_control(checkpoint["records"], bank)
    return _with_hash(
        {
            "schema_version": QUALIFICATION_RESULT_SCHEMA,
            "status": selection["status"],
            "development_only": True,
            "qualification_only": True,
            "qualification_lock_file_sha256": file_sha256(QUALIFICATION_LOCK_PATH),
            "qualification_lock_identity_sha256": lock["lock_identity_sha256"],
            "qualification_checkpoint_path": _relative(
                QUALIFICATION_CHECKPOINT_PATH
            ),
            "qualification_checkpoint_file_sha256": file_sha256(
                QUALIFICATION_CHECKPOINT_PATH
            ),
            "qualification_checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "qualification_ledger_file_sha256": file_sha256(
                QUALIFICATION_LEDGER_PATH
            ),
            "qualification_ledger_sha256": ledger["ledger_sha256"],
            "candidate_bank_file_sha256": file_sha256(CONTROL_CANDIDATES_PATH),
            "candidate_order_sha256": lock["candidate_order_sha256"],
            "selection_rule": lock["selection_rule"],
            "known_bad_legacy_view": lock["known_bad_legacy_view"],
            **selection,
            "compute": lock["compute_ceiling"],
            "directions_or_interventions_inspected": False,
            "claim_boundary": (
                "Baseline format qualification only; this is not steering evidence."
            ),
        },
        "qualification_result_sha256",
    )


def _validate_qualification_result() -> dict[str, Any]:
    import torch

    checkpoint, _ = _load_qualification_checkpoint(torch)
    observed = _load_json(QUALIFICATION_RESULT_PATH)
    _verify_hash(observed, "qualification_result_sha256")
    expected = _expected_qualification_result(checkpoint)
    if observed != expected:
        raise RuntimeError("control-qualification result differs from checkpoint")
    return observed


def run_qualify_controls() -> dict[str, Any]:
    if LOCK_PATH.exists():
        raise RuntimeError("finite calibration lock exists before control qualification")
    lock = _load_qualification_lock()
    if QUALIFICATION_RESULT_PATH.exists():
        return _validate_qualification_result()
    import torch

    if QUALIFICATION_CHECKPOINT_PATH.exists():
        checkpoint, _ = _load_qualification_checkpoint(torch)
    else:
        if QUALIFICATION_LEDGER_PATH.exists():
            _load_qualification_ledger(require_complete=False)
            raise RuntimeError(
                "control qualification has an ambiguous pending eight-forward batch"
            )
        _write_new_json(
            QUALIFICATION_LEDGER_PATH,
            _qualification_ledger_payload(
                lock=lock, status="pending", artifact_sha256=None
            ),
        )
        original = _load_original_runner()
        backend = original.load_backend()
        records = []
        all_logits = []
        for specification in _qualification_plan():
            prompt = str(specification["prompt"])
            preferred_id = _generic_label_token_id(
                backend, prompt, str(specification["preferred_label"])
            )
            alternative_id = _generic_label_token_id(
                backend, prompt, str(specification["alternative_label"])
            )
            tokens = backend.encode(prompt)
            with backend.torch.inference_mode():
                logits = (
                    backend.model(tokens)[0, -1]
                    .detach()
                    .cpu()
                    .float()
                    .contiguous()
                )
            score = _qualification_score(
                logits,
                preferred_token_id=preferred_id,
                alternative_token_id=alternative_id,
            )
            record = {
                **{
                    key: value
                    for key, value in specification.items()
                    if key != "prompt"
                },
                "input_ids_sha256": canonical_sha256(
                    [int(value) for value in tokens.detach().cpu().reshape(-1).tolist()]
                ),
                **score,
            }
            record["row_sha256"] = canonical_sha256(record)
            records.append(record)
            all_logits.append(logits)
        if len(records) != QUALIFICATION_FORWARD_COUNT:
            raise RuntimeError("control qualification did not execute exactly eight forwards")
        _save_qualification_checkpoint(
            torch,
            lock=lock,
            records=records,
            logits=torch.stack(all_logits).contiguous(),
            backend_metadata=backend.metadata(),
        )
        _write_json(
            QUALIFICATION_LEDGER_PATH,
            _qualification_ledger_payload(
                lock=lock,
                status="complete",
                artifact_sha256=file_sha256(QUALIFICATION_CHECKPOINT_PATH),
            ),
        )
        checkpoint, _ = _load_qualification_checkpoint(torch)
    result = _expected_qualification_result(checkpoint)
    _write_new_json(QUALIFICATION_RESULT_PATH, result)
    return _validate_qualification_result()


def _selected_replacement_control() -> dict[str, Any]:
    result = _validate_qualification_result()
    selected = result.get("selected_control")
    if (
        result.get("status") != "passed"
        or result.get("finite_lock_authorized") is not True
        or not isinstance(selected, Mapping)
        or canonical_sha256(dict(selected)) != result.get("selected_control_sha256")
    ):
        raise RuntimeError("control qualification did not authorize the finite lock")
    return {
        **dict(selected),
        "qualification_result_sha256": result["qualification_result_sha256"],
        "qualification_selected_control_sha256": result[
            "selected_control_sha256"
        ],
    }


def _resolved_anchor_index(
    backend: Any,
    dataset: Mapping[str, Any],
    form: Mapping[str, Any],
    cache: dict[str, tuple[int, str]],
) -> tuple[int, str | None]:
    if form.get("anchor_index") is not None:
        return int(form["anchor_index"]), None
    control_id = str(form["control_id"])
    qualification_hash = form.get("qualification_result_sha256")
    key = (
        f"control:{control_id}:partition=calibration:"
        f"qualification={qualification_hash}"
    )
    if key not in cache:
        if form.get("replacement_for") == "fcag_control_08_instruction":
            control = _selected_replacement_control()
            if control["id"] != control_id:
                raise RuntimeError("finite replacement control differs from qualification")
            control["partition"] = "calibration"
        else:
            control = next(
                row
                for row in dataset["unrelated_controls"]
                if row["id"] == control_id and row["partition"] == "calibration"
            )
        construction = render_unrelated_construction_form(dataset, control)
        forms = []
        from sp_lense.decision_margin_shield_finite import _render_unrelated_choice_form

        for preferred_first in (True, False):
            forms.append(
                _render_unrelated_choice_form(
                    dataset, control, preferred_first=preferred_first
                )
            )
        evidence = resolve_shared_anchor_evidence(
            backend,
            anchor_prefix=str(construction["anchor_prefix"]),
            prompts=[str(construction["prompt"]), *[str(row["prompt"]) for row in forms]],
            anchor_marker=str(dataset["anchor_marker"]),
        )
        cache[key] = (evidence.anchor_index, str(evidence.audit["audit_sha256"]))
    return cache[key]


def _score_logits(
    torch: Any,
    *,
    logits: Any,
    form: Mapping[str, Any],
    positive_id: int,
    negative_id: int,
    baseline_logits: Any | None,
) -> dict[str, Any]:
    logits = logits.detach().cpu().float().contiguous()
    predicted_id = int(logits.argmax().item())
    semantic = (
        str(form["positive_semantic"])
        if predicted_id == positive_id
        else str(form["negative_semantic"])
        if predicted_id == negative_id
        else "OTHER"
    )
    pair_semantic = (
        str(form["positive_semantic"])
        if float(logits[positive_id]) >= float(logits[negative_id])
        else str(form["negative_semantic"])
    )
    log_odds = float((logits[positive_id] - logits[negative_id]).item())
    if baseline_logits is None:
        baseline_predicted_id = predicted_id
        baseline_semantic = semantic
        baseline_log_odds = log_odds
        kl = 0.0
    else:
        baseline_logits = baseline_logits.detach().cpu().float().contiguous()
        baseline_predicted_id = int(baseline_logits.argmax().item())
        baseline_semantic = (
            str(form["positive_semantic"])
            if baseline_predicted_id == positive_id
            else str(form["negative_semantic"])
            if baseline_predicted_id == negative_id
            else "OTHER"
        )
        baseline_log_odds = float(
            (baseline_logits[positive_id] - baseline_logits[negative_id]).item()
        )
        kl = full_vocabulary_kl_float64(torch, baseline_logits, logits)
    return {
        "positive_token_id": positive_id,
        "negative_token_id": negative_id,
        "predicted_token_id": predicted_id,
        "semantic_choice": semantic,
        "pair_semantic_choice": pair_semantic,
        "answer_format_valid": semantic != "OTHER",
        "positive_minus_negative_log_odds": log_odds,
        "baseline_predicted_token_id": baseline_predicted_id,
        "baseline_semantic_choice": baseline_semantic,
        "baseline_positive_minus_negative_log_odds": baseline_log_odds,
        "log_odds_change_from_baseline": log_odds - baseline_log_odds,
        "greedy_token_changed": predicted_id != baseline_predicted_id,
        "semantic_choice_changed": semantic != baseline_semantic,
        "full_vocabulary_kl_changed_to_baseline": float(kl),
        "full_vocabulary_kl_direction": "KL(changed||baseline)",
        "logits_float32_sha256": tensor_float32_sha256(logits),
    }


def _run_one_forward(
    backend: Any,
    *,
    dataset: Mapping[str, Any],
    specification: Mapping[str, Any],
    directions: Mapping[tuple[str, str], Mapping[str, Any]],
    baseline_cache: Mapping[str, Any],
    anchor_cache: dict[str, tuple[int, str]],
) -> tuple[dict[str, Any], Any | None]:
    public = public_work_spec(specification)
    form = specification["form"]
    prompt = str(form["prompt"])
    positive_id = _generic_label_token_id(backend, prompt, str(form["positive_label"]))
    negative_id = _generic_label_token_id(backend, prompt, str(form["negative_label"]))
    anchor_index, anchor_evidence_sha = _resolved_anchor_index(
        backend, dataset, form, anchor_cache
    )
    tokens = backend.encode(prompt)
    hook_diagnostics: dict[int, dict[str, Any]] = {}
    requested_hash = None
    source_direction_hash = None
    if specification["kind"] == "baseline":
        with backend.torch.inference_mode():
            logits = backend.model(tokens)[0, -1].detach().cpu().float().contiguous()
        baseline_logits = None
        stored_logits = logits
    else:
        key = (
            str(specification["direction_scenario_id"]),
            str(specification["method"]),
        )
        direction = directions[key]
        perturbation = (
            int(specification["sign"])
            * float(specification["strength"])
            * direction["physical"].float()
        ).contiguous()
        requested_hash = tensor_float32_sha256(perturbation)
        source_direction_hash = str(direction["float64_direction_sha256"])
        hooks = multilayer_anchor_hooks(
            backend.torch,
            layers=(SELECTED_LAYER,),
            perturbations=perturbation.reshape(1, -1),
            anchor_index=anchor_index,
            diagnostics=hook_diagnostics,
            maximum_realized_relative_error=HOOK_REALIZATION_RELATIVE_L2_TOLERANCE,
        )
        with backend.torch.inference_mode(), backend.model.hooks(fwd_hooks=hooks):
            logits = backend.model(tokens)[0, -1].detach().cpu().float().contiguous()
        if set(hook_diagnostics) != {SELECTED_LAYER}:
            raise RuntimeError("finite intervention hook did not fire exactly once")
        realized_error = hook_diagnostics[SELECTED_LAYER][
            "requested_minus_realized_bundle_relative_l2"
        ]
        if realized_error > HOOK_REALIZATION_RELATIVE_L2_TOLERANCE:
            raise RuntimeError("finite hook realization exceeds the locked tolerance")
        baseline_logits = baseline_cache[str(specification["baseline_id"])]
        stored_logits = None
    score = _score_logits(
        backend.torch,
        logits=logits,
        form=public["form"],
        positive_id=positive_id,
        negative_id=negative_id,
        baseline_logits=baseline_logits,
    )
    record = {
        **public,
        **score,
        "anchor_index": anchor_index,
        "runtime_anchor_evidence_sha256": anchor_evidence_sha,
        "source_float64_direction_sha256": source_direction_hash,
        "requested_perturbation_float32_sha256": requested_hash,
        "hook_diagnostics": {str(key): value for key, value in hook_diagnostics.items()},
    }
    record["row_sha256"] = canonical_sha256(record)
    return record, stored_logits


def _load_completed_rows(
    torch: Any,
    *,
    ledger: CalibrationLedger,
    plan: Sequence[Mapping[str, Any]],
    chunks: Sequence[Sequence[Mapping[str, Any]]],
    completed_count: int,
    plan_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if completed_count != ledger.completed_chunks():
        raise RuntimeError("completed finite rows differ from ledger state")
    rows = []
    baselines = {}
    for index in range(completed_count):
        event = ledger.payload["events"][index]
        path = _chunk_path(index)
        if (
            event.get("status") != "complete"
            or event.get("artifact_path") != _relative(path)
            or event.get("artifact_sha256") != file_sha256(path)
        ):
            raise RuntimeError("completed finite chunk is not bound to its ledger event")
        records, logits = _load_chunk(
            torch,
            path=path,
            index=index,
            plan_hash=plan_hash,
            expected_specs=chunks[index],
        )
        rows.extend(records)
        if logits is not None:
            for row_index, record in enumerate(records):
                baselines[str(record["baseline_id"])] = logits[row_index].float().contiguous()
    if len({row["work_id"] for row in rows}) != len(rows):
        raise RuntimeError("completed finite rows contain duplicate work IDs")
    expected_prefix = [row["work_id"] for row in plan[: len(rows)]]
    if [row["work_id"] for row in rows] != expected_prefix:
        raise RuntimeError("completed finite rows differ from the plan prefix")
    return rows, baselines


def _audit_capture_baselines(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    original = _load_original_runner()
    import torch

    captures = {
        str(row["form_id"]): row
        for row in original._load_capture_records(torch)
        if row["kind"] == "scenario" and row["partition"] == "calibration"
    }
    baselines = [
        row
        for row in rows
        if row["kind"] == "baseline" and row["form"]["family"] == "scenario"
    ]
    if len(captures) != 64 or len(baselines) != 64:
        raise RuntimeError("capture-to-finite baseline audit requires 64 scenario rows")
    differences = []
    for row in baselines:
        captured = captures[str(row["form"]["form_id"])]
        difference = abs(
            float(row["positive_minus_negative_log_odds"])
            - float(captured["preserve_minus_comply_baseline_log_odds"])
        )
        if difference > BASELINE_LOG_ODDS_TOLERANCE:
            raise RuntimeError("finite baseline differs from its captured boundary offset")
        differences.append(difference)
    return {
        "passes": True,
        "count": len(differences),
        "maximum_absolute_difference": max(differences),
        "tolerance": BASELINE_LOG_ODDS_TOLERANCE,
    }


def _audit_qualification_baseline(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    import torch

    qualification = _validate_qualification_result()
    checkpoint, _ = _load_qualification_checkpoint(torch)
    selected_id = str(qualification["selected_control"]["id"])
    qualification_rows = {
        bool(row["preferred_first"]): row
        for row in checkpoint["records"]
        if row["candidate_id"] == selected_id
    }
    finite_rows = {
        bool(row["form"]["preferred_first"]): row
        for row in rows
        if row["kind"] == "baseline"
        and row["form"].get("control_id") == selected_id
    }
    if set(qualification_rows) != {True, False} or set(finite_rows) != {True, False}:
        raise RuntimeError("qualified-control finite baseline coverage differs")
    differences = []
    for preferred_first in (True, False):
        qualified = qualification_rows[preferred_first]
        finite = finite_rows[preferred_first]
        if (
            finite["form"]["prompt_sha256"] != qualified["prompt_sha256"]
            or finite["form"].get("qualification_result_sha256")
            != qualification["qualification_result_sha256"]
            or finite["form"].get("qualification_selected_control_sha256")
            != qualification["selected_control_sha256"]
            or finite["predicted_token_id"] != qualified["predicted_token_id"]
            or finite["semantic_choice"]
            != qualified["unrestricted_semantic_choice"]
            or finite["answer_format_valid"] != qualified["answer_format_valid"]
        ):
            raise RuntimeError("qualified control did not reproduce in finite baselines")
        difference = abs(
            float(finite["positive_minus_negative_log_odds"])
            - float(qualified["preferred_minus_alternative_log_odds"])
        )
        if difference > BASELINE_LOG_ODDS_TOLERANCE:
            raise RuntimeError("qualified-control finite baseline log odds differ")
        differences.append(difference)
    return {
        "passes": True,
        "selected_control_id": selected_id,
        "qualification_result_sha256": qualification[
            "qualification_result_sha256"
        ],
        "count": 2,
        "maximum_absolute_log_odds_difference": max(differences),
        "tolerance": BASELINE_LOG_ODDS_TOLERANCE,
    }


def _audit_vector_reuse(
    rows: Sequence[Mapping[str, Any]],
    directions: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    groups: dict[tuple[str, str, float, int], set[str]] = {}
    counts: dict[tuple[str, str, float, int], int] = {}
    for row in rows:
        if row["kind"] == "baseline":
            if (
                row.get("source_float64_direction_sha256") is not None
                or row.get("requested_perturbation_float32_sha256") is not None
                or row.get("hook_diagnostics") != {}
            ):
                raise RuntimeError("finite baseline row contains intervention metadata")
            continue
        direction_key = (
            str(row["direction_scenario_id"]),
            str(row["method"]),
        )
        if direction_key not in directions:
            raise RuntimeError("finite row references an absent direction")
        direction = directions[direction_key]
        requested = (
            int(row["sign"])
            * float(row["strength"])
            * direction["physical"].float()
        ).contiguous()
        expected_requested_hash = tensor_float32_sha256(requested)
        diagnostics = row.get("hook_diagnostics")
        if not isinstance(diagnostics, Mapping) or set(diagnostics) != {
            str(SELECTED_LAYER)
        }:
            raise RuntimeError("finite row hook diagnostics coverage differs")
        hook = diagnostics[str(SELECTED_LAYER)]
        if (
            row.get("source_float64_direction_sha256")
            != direction["float64_direction_sha256"]
            or row.get("requested_perturbation_float32_sha256")
            != expected_requested_hash
            or hook.get("perturbation_float32_sha256") != expected_requested_hash
            or hook.get("anchor_index") != row.get("anchor_index")
            or hook.get("bundle_maximum_allowed_realized_relative_l2")
            != HOOK_REALIZATION_RELATIVE_L2_TOLERANCE
            or float(
                hook.get("requested_minus_realized_bundle_relative_l2", math.inf)
            )
            > HOOK_REALIZATION_RELATIVE_L2_TOLERANCE
            or hook.get("untouched_positions_max_abs_delta") != 0.0
        ):
            raise RuntimeError("finite row direction or hook realization differs")
        key = (
            str(row["method"]),
            str(row["direction_scenario_id"]),
            float(row["strength"]),
            int(row["sign"]),
        )
        groups.setdefault(key, set()).add(str(row["requested_perturbation_float32_sha256"]))
        counts[key] = counts.get(key, 0) + 1
    if (
        len(groups) != 72
        or any(len(value) != 1 for value in groups.values())
        or any(value != 24 for value in counts.values())
    ):
        raise RuntimeError("finite evaluation did not reuse one byte-identical vector per unit")
    return {
        "passes": True,
        "group_count": len(groups),
        "row_count_by_group_sha256": canonical_sha256(
            [
                {"key": list(key), "row_count": counts[key], "hash": next(iter(groups[key]))}
                for key in sorted(groups)
            ]
        ),
    }


def _expected_calibration_result(
    *,
    rows: Sequence[Mapping[str, Any]],
    freeze: Mapping[str, Any],
    lock: Mapping[str, Any],
    manifest: Mapping[str, Any],
    directions: Mapping[tuple[str, str], Mapping[str, Any]],
    compute: Mapping[str, Any],
) -> dict[str, Any]:
    capture_binding = _audit_capture_baselines(rows)
    qualification_binding = _audit_qualification_baseline(rows)
    vector_reuse = _audit_vector_reuse(rows, directions)
    summary = summarize_calibration(rows)
    return _with_hash(
        {
            "schema_version": RESULT_SCHEMA,
            "status": summary["status"],
            "development_only": True,
            "lock_file_sha256": file_sha256(LOCK_PATH),
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "freeze_file_sha256": file_sha256(FREEZE_PATH),
            "freeze_sha256": freeze["freeze_sha256"],
            "direction_manifest_file_sha256": file_sha256(DIRECTION_MANIFEST_PATH),
            "direction_manifest_sha256": manifest["direction_manifest_sha256"],
            "qualification_result_file_sha256": file_sha256(
                QUALIFICATION_RESULT_PATH
            ),
            "qualification_result_sha256": qualification_binding[
                "qualification_result_sha256"
            ],
            "screen_result_sha256": SCREEN_RESULT_SHA256,
            "plan_sha256": freeze["plan_sha256"],
            "rows_sha256": canonical_sha256(list(rows)),
            "row_count": len(rows),
            "compute": dict(compute),
            "generated_tokens": 0,
            "external_model_judges": 0,
            "external_api_calls": 0,
            "paid_model_cost_usd": 0,
            "capture_to_finite_baseline_audit": capture_binding,
            "qualification_to_finite_baseline_audit": qualification_binding,
            "vector_reuse_audit": vector_reuse,
            "calibration_summary": summary,
            "pilot_outcomes_read": False,
            "pilot_authorized": summary["pilot_authorized"],
            "claim_boundary": summary["claim_boundary"],
        },
        "result_sha256",
    )


def run_calibration() -> dict[str, Any]:
    freeze, plan = _load_freeze()
    lock = _load_lock()
    directions, manifest = _load_direction_bank()
    chunks = _chunked(plan, CALIBRATION_CHUNK_SIZE)
    expected_ids = [[str(row["work_id"]) for row in chunk] for chunk in chunks]
    ledger = CalibrationLedger(
        path=LEDGER_PATH,
        plan_sha256_value=str(freeze["plan_sha256"]),
        lock_identity_sha256=str(lock["lock_identity_sha256"]),
        expected_chunk_work_ids=expected_ids,
    )
    if RESULT_PATH.exists():
        return _validate_result()
    import torch

    completed = ledger.completed_chunks()
    rows, baseline_cache = _load_completed_rows(
        torch,
        ledger=ledger,
        plan=plan,
        chunks=chunks,
        completed_count=completed,
        plan_hash=str(freeze["plan_sha256"]),
    )
    if completed < len(chunks):
        original = _load_original_runner()
        backend = original.load_backend()
        dataset = _load_dataset()
        anchor_cache: dict[str, tuple[int, str]] = {}
        for index in range(completed, len(chunks)):
            chunk = chunks[index]
            ledger.reserve(index, [str(row["work_id"]) for row in chunk])
            chunk_records = []
            chunk_baseline_logits = []
            for specification in chunk:
                record, stored_logits = _run_one_forward(
                    backend,
                    dataset=dataset,
                    specification=specification,
                    directions=directions,
                    baseline_cache=baseline_cache,
                    anchor_cache=anchor_cache,
                )
                chunk_records.append(record)
                if stored_logits is not None:
                    chunk_baseline_logits.append(stored_logits)
                    baseline_cache[str(record["baseline_id"])] = stored_logits
            is_baseline = chunk[0]["kind"] == "baseline"
            if any(row["kind"] != chunk[0]["kind"] for row in chunk):
                raise RuntimeError("finite chunk boundary mixed baseline and changed rows")
            tensor = (
                torch.stack(chunk_baseline_logits).contiguous()
                if is_baseline
                else None
            )
            path = _chunk_path(index)
            _save_chunk(
                torch,
                path=path,
                index=index,
                plan_hash=str(freeze["plan_sha256"]),
                expected_specs=chunk,
                records=chunk_records,
                baseline_logits=tensor,
            )
            ledger.complete(index, path)
            rows.extend(chunk_records)
            print(
                f"DMS finite calibration chunk {index + 1}/{len(chunks)} "
                f"F={ledger.snapshot()['forward_evaluations']}",
                flush=True,
            )
    snapshot = ledger.snapshot()
    if (
        snapshot["forward_evaluations"] != CALIBRATION_FORWARD_COUNT
        or snapshot["backward_evaluations"] != 0
        or len(rows) != CALIBRATION_FORWARD_COUNT
    ):
        raise RuntimeError("finite calibration did not consume the exact locked plan")
    result = _expected_calibration_result(
        rows=rows,
        freeze=freeze,
        lock=lock,
        manifest=manifest,
        directions=directions,
        compute=snapshot,
    )
    _write_new_json(RESULT_PATH, result)
    return _validate_result()


def _validate_result() -> dict[str, Any]:
    freeze, plan = _load_freeze()
    lock = _load_lock()
    directions, manifest = _load_direction_bank()
    chunks = _chunked(plan, CALIBRATION_CHUNK_SIZE)
    expected_ids = [[str(row["work_id"]) for row in chunk] for chunk in chunks]
    ledger = CalibrationLedger(
        path=LEDGER_PATH,
        plan_sha256_value=str(freeze["plan_sha256"]),
        lock_identity_sha256=str(lock["lock_identity_sha256"]),
        expected_chunk_work_ids=expected_ids,
    )
    if ledger.completed_chunks() != len(chunks):
        raise RuntimeError("finite calibration result has incomplete checkpoint coverage")
    snapshot = ledger.snapshot()
    if (
        snapshot["forward_evaluations"] != CALIBRATION_FORWARD_COUNT
        or snapshot["backward_evaluations"] != 0
        or snapshot["completed_chunk_count"] != len(chunks)
    ):
        raise RuntimeError("finite calibration result compute differs from exact plan")
    import torch

    rows, _ = _load_completed_rows(
        torch,
        ledger=ledger,
        plan=plan,
        chunks=chunks,
        completed_count=len(chunks),
        plan_hash=str(freeze["plan_sha256"]),
    )
    expected = _expected_calibration_result(
        rows=rows,
        freeze=freeze,
        lock=lock,
        manifest=manifest,
        directions=directions,
        compute=snapshot,
    )
    observed = _load_json(RESULT_PATH)
    _verify_hash(observed, "result_sha256")
    if observed != expected:
        raise RuntimeError("finite calibration result differs from checkpoints and ledger")
    return observed


def _render_report(result: Mapping[str, Any]) -> str:
    summary = result["calibration_summary"]
    lines = [
        "# Decision-Margin Shield finite calibration",
        "",
        f"Status: **{result['status']}**",
        "",
        "This is opened-development A/B calibration only. It is not a natural-",
        "mechanism, safety, unchanged-capability, confirmatory, or publication claim.",
        "",
        "## Selected strengths",
        "",
        "| Method | Strength | Complete units | Both-assignment scenarios | Safety |",
        "|---|---:|---:|---:|---|",
    ]
    for method in METHODS:
        row = summary["selected_by_method"][method]
        if row is None:
            lines.append(f"| {method} | — | — | — | no admissible strength |")
        else:
            lines.append(
                f"| {method} | {row['strength']} | {row['complete_assignment_units']}/8 | "
                f"{row['scenario_count_with_both_assignments']}/4 | pass |"
            )
    lines.extend(
        [
            "",
            "## DMS Pareto comparisons",
            "",
            "| Ablation | Pass | Reason |",
            "|---|---|---|",
        ]
    )
    for method, comparison in summary["dms_pareto_comparisons"].items():
        lines.append(f"| {method} | {comparison['passes']} | {comparison['reason']} |")
    lines.extend(
        [
            "",
            "Exact compute: 1,800 forward passes, 0 backward passes, 0 generated ",
            "tokens, 0 external/API judges, and $0 direct paid-model cost. The ",
            "separately locked prerequisite control qualification used 8 baseline ",
            "forwards, for 1,808 prospective forwards across both phases.",
            "",
            f"Result SHA-256: `{result['result_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_report() -> str:
    result = _validate_result()
    report = _render_report(result)
    if REPORT_PATH.exists():
        if REPORT_PATH.read_text(encoding="utf-8") != report:
            raise RuntimeError("existing finite calibration report differs")
    else:
        _atomic_text(REPORT_PATH, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Locked local-only DMS finite A/B calibration (no pilot command)"
    )
    parser.add_argument(
        "command",
        choices=(
            "qualification-lock",
            "qualify-controls",
            "lock",
            "preflight",
            "construct",
            "freeze",
            "calibrate",
            "report",
        ),
    )
    args = parser.parse_args()
    if args.command == "qualification-lock":
        value: Any = run_qualification_lock()
    elif args.command == "qualify-controls":
        value = run_qualify_controls()
    elif args.command == "lock":
        value = run_lock()
    elif args.command == "preflight":
        value = run_preflight()
    elif args.command == "construct":
        value = run_construct()
    elif args.command == "freeze":
        value = run_freeze()
    elif args.command == "calibrate":
        value = run_calibration()
    else:
        value = run_report()
    print(value if isinstance(value, str) else json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
