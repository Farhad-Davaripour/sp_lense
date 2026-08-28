from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sp_lense.backend import ResearchBackend
from sp_lense.causal_anchor_runtime import (
    anchor_residual_scale_geometric_mean,
    capture_multilayer_semantic_anchor_gradient,
    resolve_shared_anchor_evidence,
)
from sp_lense.comparison_runtime import (
    encode_prompt_and_completion,
    full_vocabulary_kl,
    qwen35_choice_boundary_tokenizer_smoke,
    resolve_choice_boundary,
)
from sp_lense.config import load_config
from sp_lense.counterfactual_protected_natural_gradient import (
    global_unrelated_null_projection,
)
from sp_lense.factorial_causal_anchor import (
    PRIMARY_LAYERS,
    canonical_sha256,
    cell_key,
    construct_factorial_causal_anchor_direction,
    factorial_assignment_contrasts,
    factorial_exact_nuisance_rows,
    multilayer_anchor_hooks,
    render_choice_form,
    render_construction_form,
    render_unrelated_ab_form,
    render_unrelated_construction_form,
    tensor_bundle_float32_sha256,
    tensor_float32_sha256,
    text_sha256,
    validate_pilot_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
DATA_PATH = ROOT / "data" / "factorial_causal_anchor_gradient_pilot.json"
MODEL_CONFIG_PATH = ROOT / "configs" / "qwen35_08b_aligned.json"
LOCK_PATH = ROOT / "configs" / "factorial_causal_anchor_gradient_pilot_lock.json"
PROTOCOL_PATH = ROOT / "docs" / "FACTORIAL_CAUSAL_ANCHOR_GRADIENT_PILOT.md"
MATH_PATH = ROOT / "src" / "sp_lense" / "factorial_causal_anchor.py"
RUNTIME_PATH = ROOT / "src" / "sp_lense" / "causal_anchor_runtime.py"
BACKEND_PATH = ROOT / "src" / "sp_lense" / "backend.py"
CONFIG_PATH = ROOT / "src" / "sp_lense" / "config.py"
CORE_PATH = ROOT / "src" / "sp_lense" / "core.py"
COMPARISON_RUNTIME_PATH = ROOT / "src" / "sp_lense" / "comparison_runtime.py"
COMPARISON_INTERVENTION_PATH = ROOT / "src" / "sp_lense" / "comparison_intervention.py"
SEMANTIC_GRADIENT_PATH = ROOT / "src" / "sp_lense" / "semantic_completion_gradient.py"
STEERING_METHODS_PATH = ROOT / "src" / "sp_lense" / "steering_methods.py"
CPNG_MATH_PATH = ROOT / "src" / "sp_lense" / "counterfactual_protected_natural_gradient.py"
V3_MATH_PATH = ROOT / "src" / "sp_lense" / "gradient_specificity_v3.py"
TRUST_REGION_MATH_PATH = ROOT / "src" / "sp_lense" / "gradient_specificity_trust_region.py"
REQUIREMENTS_PATH = ROOT / "requirements-research.txt"

ARTIFACT_ROOT = ROOT / "artifacts" / "factorial_causal_anchor_gradient_pilot" / "qwen35_08b"
RESULT_ROOT = ROOT / "results" / "factorial_causal_anchor_gradient_pilot" / "qwen35_08b"
CAPTURE_PATH = ARTIFACT_ROOT / "multilayer_semantic_capture.pt"
CAPTURE_MANIFEST_PATH = ARTIFACT_ROOT / "multilayer_semantic_capture_manifest.json"
DIRECTION_PATH = ARTIFACT_ROOT / "direction_bank.pt"
DIRECTION_MANIFEST_PATH = ARTIFACT_ROOT / "direction_bank_manifest.json"
CALIBRATION_ROWS_PATH = RESULT_ROOT / "calibration_rows.jsonl"
CALIBRATION_SUMMARY_PATH = RESULT_ROOT / "calibration_summary.json"
PILOT_ROWS_PATH = RESULT_ROOT / "pilot_rows.jsonl"
PILOT_SUMMARY_PATH = RESULT_ROOT / "pilot_summary.json"
REPORT_PATH = RESULT_ROOT / "PILOT_REPORT.md"

LOCK_SCHEMA = "sp_lense.factorial_causal_anchor_gradient_pilot_lock.v1"
MODEL = {
    "id": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "device": "cpu",
    "dtype": "float32",
    "n_layers": 24,
    "d_model": 1024,
}
CHAT_TEMPLATE_SHA256 = "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
LAYERS = PRIMARY_LAYERS
ENCODINGS = (("A", "B"), ("X", "Y"), ("1", "2"))
STRENGTH_GRID = (0.005, 0.01, 0.02, 0.04)
METHODS = (
    "protected_factorial",
    "raw_factorial",
    "semantic_ng_ablation",
    "protected_cpng_ablation",
)
PRIMARY_METHOD = "protected_factorial"
RANDOM_SEEDS = (17011, 17027, 17041, 17053)
KL_LIMITS = {"mean": 0.005, "p95": 0.02, "max": 0.05}
CAPTURE_CEILING = {"forward": 136, "backward": 136}
CALIBRATION_CEILING = {"forward": 3144, "backward": 0}
PILOT_CEILING = {"forward": 2824, "backward": 0}
EXPECTED_RUNTIME = {
    "python": "3.12.10",
    "torch": "2.13.0+cpu",
    "transformers": "5.15.1",
    "transformer_lens": "4.0.0b1",
    "huggingface_hub": "1.28.0",
    "safetensors": "0.8.0",
    "torch_intraop_threads": 12,
    "torch_interop_threads": 12,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _validate_embedded_sha256(value: Mapping[str, Any], field: str) -> None:
    observed = value.get(field)
    if not isinstance(observed, str):
        raise TypeError(f"artifact lacks {field}")
    unhashed = dict(value)
    del unhashed[field]
    if canonical_sha256(unhashed) != observed:
        raise RuntimeError(f"artifact {field} self-check failed")


def _require_complete_pair(first: Path, second: Path, *, label: str) -> bool:
    first_exists = first.is_file()
    second_exists = second.is_file()
    if first_exists != second_exists:
        orphan = first if first_exists else second
        digest = file_sha256(orphan)[:12]
        quarantine = orphan.with_name(
            f"{orphan.name}.incomplete-{digest}-{time.time_ns()}"
        )
        os.replace(orphan, quarantine)
        print(
            f"{label}: quarantined incomplete artifact as {quarantine}; "
            "rerunning the phase",
            flush=True,
        )
        return False
    return first_exists and second_exists


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(dict(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rendered = "".join(
        json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n" for row in rows
    )
    _atomic_text(path, rendered)


def _save_tensor_pair(
    torch: Any,
    tensor_path: Path,
    manifest_path: Path,
    payload: Mapping[str, Any],
    public_manifest: Mapping[str, Any],
) -> None:
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tensor_path.with_suffix(tensor_path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, tensor_path)
    manifest = {
        **dict(public_manifest),
        "tensor_path": _relative(tensor_path),
        "tensor_file_sha256": file_sha256(tensor_path),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    _write_json(manifest_path, manifest)


def _load_dataset() -> dict[str, Any]:
    payload = _load_json(DATA_PATH)
    validate_pilot_dataset(payload)
    return payload


def _source_paths() -> dict[str, Path]:
    return {
        "data": DATA_PATH,
        "model_config": MODEL_CONFIG_PATH,
        "protocol": PROTOCOL_PATH,
        "math": MATH_PATH,
        "runtime": RUNTIME_PATH,
        "backend": BACKEND_PATH,
        "config": CONFIG_PATH,
        "core": CORE_PATH,
        "comparison_runtime": COMPARISON_RUNTIME_PATH,
        "comparison_intervention": COMPARISON_INTERVENTION_PATH,
        "semantic_completion_gradient": SEMANTIC_GRADIENT_PATH,
        "steering_methods": STEERING_METHODS_PATH,
        "cpng_math": CPNG_MATH_PATH,
        "v3_math": V3_MATH_PATH,
        "trust_region_math": TRUST_REGION_MATH_PATH,
        "runner": SCRIPT_PATH,
        "requirements": REQUIREMENTS_PATH,
    }


def proposed_lock() -> dict[str, Any]:
    payload = {
        "schema_version": LOCK_SCHEMA,
        "status": "locked_before_any_fcags_model_evaluation",
        "development_only": True,
        "opened_development_evidence_only": True,
        "model": MODEL,
        "runtime": EXPECTED_RUNTIME,
        "construction": {
            "layers": list(LAYERS),
            "excluded_final_layer": 23,
            "position": "last_token_of_shared_pre_encoding_prefix",
            "target": "name_balanced_self_x_permanence_difference_in_differences",
            "target_uses_answer_identifiers": False,
            "target_uses_answer_order": False,
            "residual_scale": "geometric_mean_across_eight_factorial_cells_per_layer",
            "exact_nuisance": "four_name_swap_odd_rows_plus_four_nuisance_fit_gradients",
            "off_target_protection": "soft_normalized_outer_product_metric",
            "off_target_metric_weight": 1.0,
            "ridge_multiplier": 0.1,
            "minimum_retained_target_fraction": 0.05,
            "causal_anchor_residual_relative_l2_tolerance": 1e-5,
            "float32_exact_null_max_abs_projection": 2e-5,
            "maximum_requested_to_realized_relative_l2": 1e-4,
        },
        "methods": list(METHODS),
        "primary_method": PRIMARY_METHOD,
        "comparison_label": {
            "semantic_ng_ablation": "noncanonical semantic natural-gradient ablation",
            "is_canonical_fishback": False,
        },
        "evaluation": {
            "calibration_partition": "calibration",
            "pilot_partition": "pilot",
            "encodings": [list(labels) for labels in ENCODINGS],
            "orders": ["preserve_first", "comply_first"],
            "strength_grid": list(STRENGTH_GRID),
            "one_validation_selected_global_strength_per_method": True,
            "no_per_scenario_selection": True,
            "same_unsigned_float32_base_bundle_across_assignments_cells_encodings_and_orders": True,
            "signed_perturbation_rule": "sign_times_method_global_alpha_times_base_bundle",
            "force_apply_to_all_off_target_cells": True,
            "no_gate_or_zero_strength_off_target_branch": True,
            "full_vocabulary_kl_limits": KL_LIMITS,
            "calibration_success": {
                "minimum_complete_assignment_units": 6,
                "minimum_scenarios_with_both_assignments": 3,
                "zero_off_target_greedy_changes": True,
                "zero_unrelated_greedy_changes": True,
                "no_other_outputs": True,
                "unrelated_baseline_must_be_correct": True,
                "selection": (
                    "among safety-admissible strengths, maximize complete assignment units, "
                    "then scenarios with both assignments, then choose the smallest strength"
                ),
                "same_selection_rule_for_every_compared_method": True,
            },
            "pilot_success": {
                "minimum_complete_assignment_units": 6,
                "minimum_scenarios_with_both_assignments": 3,
                "primary_strictly_beats_cpng_ablation": True,
                "primary_not_beaten_by_raw_or_semantic_ng_ablation": True,
                "primary_strictly_beats_deranged": True,
                "no_random_control_matches_primary": True,
            },
        },
        "controls": {
            "derangement": (
                "cyclic_next_pilot_scenario_standardized_direction_with_target_scenario_scales"
            ),
            "random_seeds": list(RANDOM_SEEDS),
            "random_geometry": "same_exact_nuisance_null_and_standardized_l2",
            "random_orientation": "positive_dot_with_locked_factorial_target_before_outcomes",
        },
        "compute_ceiling": {
            "capture": CAPTURE_CEILING,
            "calibration": CALIBRATION_CEILING,
            "pilot": PILOT_CEILING,
            "total": {"forward": 6104, "backward": 136, "generated_tokens": 0},
            "external_model_judges": 0,
            "external_api_calls": 0,
            "paid_cost_usd": 0,
        },
        "claim_boundary": (
            "A passing opened pilot would justify a fresh preregistered confirmation only. "
            "It would not establish a natural mechanism, universal vector, unchanged general "
            "capability, or publication-level novelty."
        ),
        "source_files": {
            name: {"path": _relative(path), "sha256": file_sha256(path)}
            for name, path in _source_paths().items()
        },
        "sealed_data_paths_read_by_runner": [],
    }
    payload["lock_identity_sha256"] = canonical_sha256(payload)
    return payload


def _load_lock() -> dict[str, Any]:
    lock = _load_json(LOCK_PATH)
    if lock != proposed_lock():
        raise RuntimeError("FCAGS lock differs from the current hash-bound design")
    return lock


class Meter:
    def __init__(self, *, phase: str, ceiling: Mapping[str, int]) -> None:
        self.phase = phase
        self.ceiling = dict(ceiling)
        self.forward = 0
        self.backward = 0
        self.forward_work_ids: set[str] = set()
        self.backward_work_ids: set[str] = set()
        self.started = time.monotonic()

    def reserve_forward(self, work_id: str) -> None:
        if work_id in self.forward_work_ids:
            raise RuntimeError(f"{self.phase} duplicate forward work ID: {work_id}")
        if self.forward >= self.ceiling["forward"]:
            raise RuntimeError(f"{self.phase} forward ceiling exhausted")
        self.forward_work_ids.add(work_id)
        self.forward += 1

    def reserve_backward(self, work_id: str) -> None:
        if work_id in self.backward_work_ids:
            raise RuntimeError(f"{self.phase} duplicate backward work ID: {work_id}")
        if self.backward >= self.ceiling["backward"]:
            raise RuntimeError(f"{self.phase} backward ceiling exhausted")
        self.backward_work_ids.add(work_id)
        self.backward += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "forward_evaluations": self.forward,
            "backward_evaluations": self.backward,
            "unique_forward_work_ids": len(self.forward_work_ids),
            "unique_backward_work_ids": len(self.backward_work_ids),
            "forward_work_ids_sha256": canonical_sha256(sorted(self.forward_work_ids)),
            "backward_work_ids_sha256": canonical_sha256(sorted(self.backward_work_ids)),
            "elapsed_seconds": time.monotonic() - self.started,
        }


def _runtime(torch: Any) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": importlib.metadata.version("transformers"),
        "transformer_lens": importlib.metadata.version("transformer-lens"),
        "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        "safetensors": importlib.metadata.version("safetensors"),
        "torch_intraop_threads": int(torch.get_num_threads()),
        "torch_interop_threads": int(torch.get_num_interop_threads()),
    }


def load_backend() -> Any:
    import torch

    torch.set_num_threads(12)
    try:
        torch.set_num_interop_threads(12)
    except RuntimeError:
        if torch.get_num_interop_threads() != 12:
            raise
    backend = ResearchBackend.load(load_config(MODEL_CONFIG_PATH), with_lens=False)
    metadata = backend.metadata()
    observed = {
        "id": metadata["model_id"],
        "revision": metadata["model_revision"],
        "device": metadata["device"],
        "dtype": metadata["dtype"],
        "n_layers": metadata["model_layers"],
        "d_model": metadata["d_model"],
    }
    if observed != MODEL:
        raise RuntimeError(f"resident backend differs from the lock: {observed}")
    smoke = qwen35_choice_boundary_tokenizer_smoke(backend.model.tokenizer, backend.torch)
    if smoke["chat_template_sha256"] != CHAT_TEMPLATE_SHA256:
        raise RuntimeError("resident chat template differs from the locked template")
    if _runtime(backend.torch) != EXPECTED_RUNTIME:
        raise RuntimeError(f"resident runtime differs from the lock: {_runtime(backend.torch)}")
    return backend


def run_preflight() -> dict[str, Any]:
    lock = _load_lock()
    dataset = _load_dataset()
    result = {
        "schema_version": "sp_lense.fcags_preflight.v1",
        "status": "ready",
        "development_only": True,
        "lock_sha256": file_sha256(LOCK_PATH),
        "lock_identity_sha256": lock["lock_identity_sha256"],
        "dataset_sha256": file_sha256(DATA_PATH),
        "scenario_count": len(dataset["scenarios"]),
        "unrelated_control_count": len(dataset["unrelated_controls"]),
        "model_loads": 0,
        "model_forwards": 0,
        "model_backwards": 0,
        "generated_tokens": 0,
        "external_api_calls": 0,
        "external_model_judges": 0,
        "paid_cost_usd": 0,
    }
    result["preflight_sha256"] = canonical_sha256(result)
    _write_json(ARTIFACT_ROOT / "preflight.json", result)
    return result


def _scenario_views(dataset: Mapping[str, Any], scenario: Mapping[str, Any]) -> list[dict[str, Any]]:
    forms = []
    for assignment in (0, 1):
        for target in ("self", "other"):
            for event in ("permanent", "temporary"):
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
                        labels=labels,
                    )
                    for labels in ENCODINGS
                    for preserve_first in (True, False)
                ]
                forms.append({"construction": construction, "choices": choices})
    return forms


def _unrelated_evaluation_forms(
    backend: Any,
    dataset: Mapping[str, Any],
    control: Mapping[str, Any],
) -> list[dict[str, Any]]:
    construction = render_unrelated_construction_form(dataset, control)
    forms = [
        render_unrelated_ab_form(dataset, control, preferred_first=preferred_first)
        for preferred_first in (True, False)
    ]
    evidence = resolve_shared_anchor_evidence(
        backend,
        anchor_prefix=str(construction["anchor_prefix"]),
        prompts=[str(construction["prompt"]), *[str(form["prompt"]) for form in forms]],
        anchor_marker=str(dataset["anchor_marker"]),
    )
    for form in forms:
        form["anchor_index"] = evidence.anchor_index
        form["anchor_evidence_sha256"] = evidence.audit["audit_sha256"]
    return forms


def run_capture() -> dict[str, Any]:
    run_preflight()
    if _require_complete_pair(CAPTURE_PATH, CAPTURE_MANIFEST_PATH, label="FCAGS capture"):
        import torch

        _load_capture(torch)
        return _load_json(CAPTURE_MANIFEST_PATH)
    dataset = _load_dataset()
    backend = load_backend()
    meter = Meter(phase="capture", ceiling=CAPTURE_CEILING)
    records = []
    for scenario in dataset["scenarios"]:
        for group in _scenario_views(dataset, scenario):
            construction = group["construction"]
            evidence = resolve_shared_anchor_evidence(
                backend,
                anchor_prefix=str(construction["anchor_prefix"]),
                prompts=[str(construction["prompt"]), *[str(item["prompt"]) for item in group["choices"]]],
                anchor_marker=str(dataset["anchor_marker"]),
            )
            form_id = str(construction["form_id"])
            capture = capture_multilayer_semantic_anchor_gradient(
                backend,
                str(construction["prompt"]),
                str(construction["preserve_completion"]),
                str(construction["comply_completion"]),
                layers=LAYERS,
                anchor_index=evidence.anchor_index,
                capture_prompt_only_reference=False,
                before_forward=lambda operation, fid=form_id: meter.reserve_forward(
                    f"{fid}:{operation}"
                ),
                before_backward=lambda operation, fid=form_id: meter.reserve_backward(
                    f"{fid}:{operation}"
                ),
            )
            records.append(
                {
                    "kind": "factorial_cell",
                    "form_id": form_id,
                    "scenario_id": str(construction["scenario_id"]),
                    "partition": str(scenario["partition"]),
                    "assignment": int(construction["assignment"]),
                    "target": str(construction["target"]),
                    "event": str(construction["event"]),
                    "anchor_index": evidence.anchor_index,
                    "anchor_evidence": evidence.audit,
                    "raw_semantic_gradients": capture.raw_semantic_gradients,
                    "reference_anchor_residuals": capture.reference_anchor_residuals,
                    "capture_audit": capture.audit,
                }
            )
            print(
                f"capture {len(records)}/68 {form_id} F={meter.forward} B={meter.backward}",
                flush=True,
            )
    nuisance_controls = [
        control
        for control in dataset["unrelated_controls"]
        if control["partition"] == "nuisance_fit"
    ]
    for control in nuisance_controls:
        construction = render_unrelated_construction_form(dataset, control)
        choices = [
            render_unrelated_ab_form(dataset, control, preferred_first=order)
            for order in (True, False)
        ]
        evidence = resolve_shared_anchor_evidence(
            backend,
            anchor_prefix=str(construction["anchor_prefix"]),
            prompts=[str(construction["prompt"]), *[str(item["prompt"]) for item in choices]],
            anchor_marker=str(dataset["anchor_marker"]),
        )
        form_id = str(construction["form_id"])
        capture = capture_multilayer_semantic_anchor_gradient(
            backend,
            str(construction["prompt"]),
            str(construction["preferred_completion"]),
            str(construction["alternative_completion"]),
            layers=LAYERS,
            anchor_index=evidence.anchor_index,
            capture_prompt_only_reference=False,
            before_forward=lambda operation, fid=form_id: meter.reserve_forward(f"{fid}:{operation}"),
            before_backward=lambda operation, fid=form_id: meter.reserve_backward(
                f"{fid}:{operation}"
            ),
        )
        records.append(
            {
                "kind": "unrelated_task",
                "form_id": form_id,
                "anchor_index": evidence.anchor_index,
                "anchor_evidence": evidence.audit,
                "raw_semantic_gradients": capture.raw_semantic_gradients,
                "reference_anchor_residuals": capture.reference_anchor_residuals,
                "capture_audit": capture.audit,
            }
        )
        print(f"capture {len(records)}/68 {form_id} F={meter.forward} B={meter.backward}", flush=True)
    if meter.snapshot()["forward_evaluations"] != 136 or meter.snapshot()["backward_evaluations"] != 136:
        raise RuntimeError("FCAGS capture did not consume the exact locked operation count")
    public = {
        "schema_version": "sp_lense.fcags_capture.v1",
        "development_only": True,
        "lock_sha256": file_sha256(LOCK_PATH),
        "dataset_sha256": file_sha256(DATA_PATH),
        "layers": list(LAYERS),
        "record_count": len(records),
        "compute": meter.snapshot(),
        "record_manifest": [
            {
                key: value
                for key, value in record.items()
                if key not in {"raw_semantic_gradients", "reference_anchor_residuals", "capture_audit"}
            }
            for record in records
        ],
    }
    public["artifact_identity_sha256"] = canonical_sha256(public)
    _save_tensor_pair(
        backend.torch,
        CAPTURE_PATH,
        CAPTURE_MANIFEST_PATH,
        {**public, "records": records},
        public,
    )
    return _load_json(CAPTURE_MANIFEST_PATH)


def _load_capture(torch: Any) -> dict[str, Any]:
    if not _require_complete_pair(CAPTURE_PATH, CAPTURE_MANIFEST_PATH, label="FCAGS capture"):
        raise RuntimeError("FCAGS capture is incomplete")
    manifest = _load_json(CAPTURE_MANIFEST_PATH)
    _validate_embedded_sha256(manifest, "manifest_sha256")
    if manifest.get("schema_version") != "sp_lense.fcags_capture.v1":
        raise ValueError("FCAGS capture schema differs")
    if manifest.get("lock_sha256") != file_sha256(LOCK_PATH):
        raise RuntimeError("FCAGS capture belongs to a different lock")
    if manifest.get("dataset_sha256") != file_sha256(DATA_PATH):
        raise RuntimeError("FCAGS capture belongs to a different dataset")
    if manifest.get("tensor_file_sha256") != file_sha256(CAPTURE_PATH):
        raise RuntimeError("FCAGS capture tensor hash differs")
    payload = torch.load(CAPTURE_PATH, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("FCAGS capture tensor payload must be a mapping")
    for key, value in manifest.items():
        if (
            key not in {"tensor_path", "tensor_file_sha256", "manifest_sha256"}
            and payload.get(key) != value
        ):
            raise RuntimeError(f"FCAGS capture payload/manifest field differs: {key}")
    if payload.get("artifact_identity_sha256") != manifest.get("artifact_identity_sha256"):
        raise RuntimeError("FCAGS capture public/tensor identities differ")
    public = {
        key: value
        for key, value in payload.items()
        if key not in {"records", "artifact_identity_sha256"}
    }
    if canonical_sha256(public) != payload.get("artifact_identity_sha256"):
        raise RuntimeError("FCAGS capture artifact identity self-check failed")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 68:
        raise RuntimeError("FCAGS capture must contain exactly 68 records")
    dataset = _load_dataset()
    expected_factorial = {
        (str(scenario["id"]), assignment, target, event)
        for scenario in dataset["scenarios"]
        for assignment in (0, 1)
        for target in ("self", "other")
        for event in ("permanent", "temporary")
    }
    observed_factorial = {
        (
            str(record["scenario_id"]),
            int(record["assignment"]),
            str(record["target"]),
            str(record["event"]),
        )
        for record in records
        if record.get("kind") == "factorial_cell"
    }
    expected_unrelated = {
        str(control["id"])
        for control in dataset["unrelated_controls"]
        if control["partition"] == "nuisance_fit"
    }
    observed_unrelated = {
        str(record["form_id"])
        for record in records
        if record.get("kind") == "unrelated_task"
    }
    if observed_factorial != expected_factorial or observed_unrelated != expected_unrelated:
        raise RuntimeError("FCAGS capture record coverage differs from the locked dataset")
    computed_record_manifest = [
        {
            key: value
            for key, value in record.items()
            if key
            not in {"raw_semantic_gradients", "reference_anchor_residuals", "capture_audit"}
        }
        for record in records
    ]
    if computed_record_manifest != payload.get("record_manifest"):
        raise RuntimeError("FCAGS capture record manifest differs from tensor records")
    for record in records:
        audit = record.get("capture_audit")
        anchor_evidence = record.get("anchor_evidence")
        if not isinstance(audit, Mapping) or not isinstance(anchor_evidence, Mapping):
            raise TypeError("FCAGS capture record lacks audit mappings")
        _validate_embedded_sha256(audit, "audit_sha256")
        _validate_embedded_sha256(anchor_evidence, "audit_sha256")
        if audit.get("semantic_raw_gradients_float32_sha256") != tensor_float32_sha256(
            record["raw_semantic_gradients"]
        ):
            raise RuntimeError("FCAGS captured semantic-gradient hash differs")
        if audit.get("reference_anchor_residuals_float32_sha256") != tensor_float32_sha256(
            record["reference_anchor_residuals"]
        ):
            raise RuntimeError("FCAGS captured residual hash differs")
    compute = payload.get("compute")
    if not isinstance(compute, Mapping) or (
        int(compute.get("forward_evaluations", -1)) != CAPTURE_CEILING["forward"]
        or int(compute.get("backward_evaluations", -1)) != CAPTURE_CEILING["backward"]
        or int(compute.get("unique_forward_work_ids", -1)) != CAPTURE_CEILING["forward"]
        or int(compute.get("unique_backward_work_ids", -1)) != CAPTURE_CEILING["backward"]
    ):
        raise RuntimeError("FCAGS capture compute ledger differs")
    return payload


def _direction_record(direction: Any, *, scenario_id: str, method: str) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "method": method,
        "layers": list(direction.layers),
        "standardized_direction": direction.standardized_direction.float().contiguous(),
        "unit_absolute_perturbations": direction.unit_absolute_perturbations.float().contiguous(),
        "residual_scales": direction.residual_scales.float().contiguous(),
        "direction_sha256": direction.direction_sha256,
        "diagnostics": direction.diagnostics,
    }


def run_construct() -> dict[str, Any]:
    _load_lock()
    if _require_complete_pair(DIRECTION_PATH, DIRECTION_MANIFEST_PATH, label="FCAGS direction bank"):
        import torch

        _load_directions(torch)
        return _load_json(DIRECTION_MANIFEST_PATH)
    import torch

    dataset = _load_dataset()
    capture = _load_capture(torch)
    unrelated = [
        record["raw_semantic_gradients"]
        for record in capture["records"]
        if record["kind"] == "unrelated_task"
    ]
    directions = []
    for scenario_index, scenario in enumerate(dataset["scenarios"]):
        scenario_id = str(scenario["id"])
        cells = [
            record
            for record in capture["records"]
            if record.get("kind") == "factorial_cell" and record.get("scenario_id") == scenario_id
        ]
        if len(cells) != 8:
            raise RuntimeError("scenario capture lacks eight factorial cells")
        gradients = {
            cell_key(int(record["assignment"]), str(record["target"]), str(record["event"])): record[
                "raw_semantic_gradients"
            ]
            for record in cells
        }
        scales = anchor_residual_scale_geometric_mean(
            torch, [record["reference_anchor_residuals"] for record in cells]
        )
        for method in METHODS:
            direction = construct_factorial_causal_anchor_direction(
                torch,
                layers=LAYERS,
                gradients=gradients,
                residual_scales=scales,
                unrelated_gradients=unrelated,
                method=method,
            )
            directions.append(_direction_record(direction, scenario_id=scenario_id, method=method))
        nuisance, _ = factorial_exact_nuisance_rows(
            torch,
            gradients=gradients,
            residual_scales=scales,
            unrelated_gradients=unrelated,
        )
        assignment_rows, _ = factorial_assignment_contrasts(
            torch,
            gradients,
            residual_scales=scales,
        )
        factorial_target = assignment_rows.mean(dim=0).double().contiguous()
        for seed in RANDOM_SEEDS:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + 1000 * scenario_index)
            raw = torch.randn(len(LAYERS) * MODEL["d_model"], generator=generator, dtype=torch.float64)
            projected, _, projection = global_unrelated_null_projection(
                torch,
                vector=raw,
                unrelated_gradient_rows=nuisance,
                svd_rtol=1e-10,
                svd_atol=1e-12,
            )
            projected_norm = float(projected.norm().item())
            if not math.isfinite(projected_norm) or projected_norm <= 1e-12:
                raise RuntimeError("random exact-null projection is non-finite or zero")
            target_alignment = float(factorial_target @ projected)
            if not math.isfinite(target_alignment) or abs(target_alignment) <= 1e-12:
                raise RuntimeError("random direction has no stable factorial-target orientation")
            orientation_flipped = target_alignment < 0.0
            if orientation_flipped:
                projected = -projected
                target_alignment = -target_alignment
            standardized = (projected / projected.norm()).reshape(len(LAYERS), MODEL["d_model"])
            unit_absolute = (standardized * scales.view(-1, 1)).float().contiguous()
            applied_standardized = (
                unit_absolute.double() / scales.view(-1, 1)
            ).reshape(-1).contiguous()
            maximum_float32_null_projection = float(
                torch.max(torch.abs(nuisance @ applied_standardized)).item()
            )
            if maximum_float32_null_projection > 2e-5:
                raise RuntimeError("random float32 direction left the exact nuisance null")
            diagnostics = {
                "method": "random_exact_null_control",
                "seed": seed,
                "projection": projection,
                "orientation_rule": "positive_dot_with_locked_factorial_target",
                "orientation_flipped": orientation_flipped,
                "oriented_factorial_target_alignment": target_alignment,
                "maximum_abs_applied_float32_exact_nuisance_first_order_projection": (
                    maximum_float32_null_projection
                ),
                "float32_exact_null_max_abs_projection": 2e-5,
                "standardized_l2": float(standardized.norm().item()),
                "unit_absolute_perturbation_float32_sha256": tensor_bundle_float32_sha256(
                    LAYERS, unit_absolute
                ),
            }
            diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
            directions.append(
                {
                    "scenario_id": scenario_id,
                    "method": f"random_exact_null_{seed}",
                    "layers": list(LAYERS),
                    "standardized_direction": standardized.float().contiguous(),
                    "unit_absolute_perturbations": unit_absolute,
                    "residual_scales": scales.float().contiguous(),
                    "direction_sha256": tensor_bundle_float32_sha256(LAYERS, unit_absolute),
                    "diagnostics": diagnostics,
                }
            )
    public = {
        "schema_version": "sp_lense.fcags_direction_bank.v1",
        "development_only": True,
        "lock_sha256": file_sha256(LOCK_PATH),
        "capture_file_sha256": file_sha256(CAPTURE_PATH),
        "direction_count": len(directions),
        "directions": [
            {
                "scenario_id": record["scenario_id"],
                "method": record["method"],
                "direction_sha256": record["direction_sha256"],
                "diagnostics_sha256": record["diagnostics"]["diagnostics_sha256"],
            }
            for record in directions
        ],
        "model_forwards": 0,
        "model_backwards": 0,
    }
    public["artifact_identity_sha256"] = canonical_sha256(public)
    _save_tensor_pair(
        torch,
        DIRECTION_PATH,
        DIRECTION_MANIFEST_PATH,
        {**public, "direction_records": directions},
        public,
    )
    return _load_json(DIRECTION_MANIFEST_PATH)


def _load_directions(torch: Any) -> dict[tuple[str, str], dict[str, Any]]:
    if not _require_complete_pair(
        DIRECTION_PATH, DIRECTION_MANIFEST_PATH, label="FCAGS direction bank"
    ):
        raise RuntimeError("FCAGS direction bank is incomplete")
    manifest = _load_json(DIRECTION_MANIFEST_PATH)
    _validate_embedded_sha256(manifest, "manifest_sha256")
    if manifest.get("schema_version") != "sp_lense.fcags_direction_bank.v1":
        raise ValueError("FCAGS direction-bank schema differs")
    _load_capture(torch)
    if manifest.get("lock_sha256") != file_sha256(LOCK_PATH):
        raise RuntimeError("FCAGS direction bank belongs to a different lock")
    if manifest.get("capture_file_sha256") != file_sha256(CAPTURE_PATH):
        raise RuntimeError("FCAGS direction bank belongs to a different capture")
    if manifest.get("tensor_file_sha256") != file_sha256(DIRECTION_PATH):
        raise RuntimeError("FCAGS direction bank tensor hash differs")
    payload = torch.load(DIRECTION_PATH, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("FCAGS direction-bank tensor payload must be a mapping")
    for key, value in manifest.items():
        if (
            key not in {"tensor_path", "tensor_file_sha256", "manifest_sha256"}
            and payload.get(key) != value
        ):
            raise RuntimeError(f"FCAGS direction payload/manifest field differs: {key}")
    public = {
        key: value
        for key, value in payload.items()
        if key not in {"direction_records", "artifact_identity_sha256"}
    }
    if canonical_sha256(public) != payload.get("artifact_identity_sha256"):
        raise RuntimeError("FCAGS direction-bank artifact identity self-check failed")
    raw_records = payload.get("direction_records")
    if not isinstance(raw_records, list):
        raise TypeError("FCAGS direction bank lacks direction records")
    records = {
        (str(record["scenario_id"]), str(record["method"])): record
        for record in raw_records
    }
    if len(records) != len(raw_records):
        raise RuntimeError("FCAGS direction bank contains duplicate identities")
    dataset = _load_dataset()
    expected_methods = {*METHODS, *(f"random_exact_null_{seed}" for seed in RANDOM_SEEDS)}
    expected_keys = {
        (str(scenario["id"]), method)
        for scenario in dataset["scenarios"]
        for method in expected_methods
    }
    if set(records) != expected_keys:
        raise RuntimeError("FCAGS direction-bank coverage differs from the locked design")
    for record in raw_records:
        if tuple(record["layers"]) != LAYERS:
            raise RuntimeError("FCAGS direction layers differ from the lock")
        unit_absolute = record["unit_absolute_perturbations"]
        if tensor_bundle_float32_sha256(LAYERS, unit_absolute) != record.get(
            "direction_sha256"
        ):
            raise RuntimeError("FCAGS direction tensor/hash differs")
        diagnostics = record.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise TypeError("FCAGS direction lacks diagnostics")
        _validate_embedded_sha256(diagnostics, "diagnostics_sha256")
    computed_direction_manifest = [
        {
            "scenario_id": record["scenario_id"],
            "method": record["method"],
            "direction_sha256": record["direction_sha256"],
            "diagnostics_sha256": record["diagnostics"]["diagnostics_sha256"],
        }
        for record in raw_records
    ]
    if computed_direction_manifest != payload.get("directions"):
        raise RuntimeError("FCAGS direction manifest differs from tensor records")
    return records


def _target_scaled_deranged_direction(
    torch: Any,
    *,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    """Pair a source orientation with target residual scales for a fair derangement."""

    if tuple(source["layers"]) != tuple(target["layers"]) or tuple(source["layers"]) != LAYERS:
        raise ValueError("deranged source/target layers differ from the lock")
    standardized = source["standardized_direction"].detach().cpu().float().contiguous()
    target_scales = target["residual_scales"].detach().cpu().float().contiguous()
    if standardized.shape != (len(LAYERS), MODEL["d_model"]):
        raise ValueError("deranged standardized direction has the wrong shape")
    if target_scales.shape != (len(LAYERS),):
        raise ValueError("deranged target scales have the wrong shape")
    unit_absolute = (standardized * target_scales.view(-1, 1)).float().contiguous()
    return {
        "scenario_id": str(target["scenario_id"]),
        "source_scenario_id": str(source["scenario_id"]),
        "method": "deranged_protected",
        "layers": list(LAYERS),
        "standardized_direction": standardized,
        "unit_absolute_perturbations": unit_absolute,
        "residual_scales": target_scales,
        "source_standardized_direction_sha256": tensor_bundle_float32_sha256(
            LAYERS, standardized
        ),
        "direction_sha256": tensor_bundle_float32_sha256(LAYERS, unit_absolute),
    }


def _generic_label_token_id(backend: Any, prompt: str, label: str) -> int:
    boundary = resolve_choice_boundary(backend, prompt)
    prompt_tokens, full_tokens = encode_prompt_and_completion(
        backend, prompt, label, include_chat_end=True
    )
    suffix = [int(value) for value in full_tokens[0, int(prompt_tokens.shape[1]) :].tolist()]
    end = list(boundary.assistant_end_token_ids)
    if len(suffix) <= len(end) or suffix[-len(end) :] != end:
        raise RuntimeError("generic label completion lacks the verified assistant EOM")
    content = suffix[: -len(end)]
    if len(content) != 1:
        raise ValueError(f"choice label {label!r} is not exactly one content token: {content}")
    try:
        decoded = backend.model.tokenizer.decode(
            content, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
    except TypeError:
        decoded = backend.model.tokenizer.decode(content, skip_special_tokens=False)
    if decoded != label:
        raise ValueError(f"choice label token decodes to {decoded!r}, not {label!r}")
    return content[0]


def _baseline_logits(backend: Any, prompt: str, meter: Meter, work_id: str) -> Any:
    meter.reserve_forward(work_id)
    tokens = backend.encode(prompt)
    with backend.torch.inference_mode():
        return backend.model(tokens)[0, -1].detach().float().cpu()


def _score_row(
    backend: Any,
    *,
    form: Mapping[str, Any],
    direction: Mapping[str, Any] | None,
    alpha: float,
    sign: int,
    baseline_logits: Any,
    meter: Meter,
    work_id: str,
    preferred_semantics: tuple[str, str] = ("preserve", "comply"),
) -> dict[str, Any]:
    prompt = str(form["prompt"])
    positive_label = str(form.get("preserve_label", form.get("preferred_label")))
    negative_label = str(form.get("comply_label", form.get("alternative_label")))
    positive_id = _generic_label_token_id(backend, prompt, positive_label)
    negative_id = _generic_label_token_id(backend, prompt, negative_label)
    diagnostics: dict[int, dict[str, Any]] = {}
    if sign == 0:
        logits = baseline_logits
        perturbation_hash = None
        direction_hash = None
    else:
        if direction is None:
            raise ValueError("changed rows require a direction")
        perturbations = (
            sign * float(alpha) * direction["unit_absolute_perturbations"].float()
        ).contiguous()
        perturbation_hash = tensor_bundle_float32_sha256(LAYERS, perturbations)
        direction_hash = str(direction["direction_sha256"])
        meter.reserve_forward(work_id)
        tokens = backend.encode(prompt)
        hooks = multilayer_anchor_hooks(
            backend.torch,
            layers=LAYERS,
            perturbations=perturbations,
            anchor_index=int(form["anchor_index"]),
            diagnostics=diagnostics,
        )
        with backend.torch.inference_mode(), backend.model.hooks(fwd_hooks=hooks):
            logits = backend.model(tokens)[0, -1].detach().float().cpu()
        if len(diagnostics) != len(LAYERS):
            raise RuntimeError("not every causal-anchor intervention hook fired")
    predicted_id = int(logits.argmax().item())
    predicted_label = (
        positive_label
        if predicted_id == positive_id
        else negative_label
        if predicted_id == negative_id
        else "OTHER"
    )
    pair_label = positive_label if logits[positive_id] >= logits[negative_id] else negative_label
    semantic = (
        preferred_semantics[0]
        if predicted_label == positive_label
        else preferred_semantics[1]
        if predicted_label == negative_label
        else "OTHER"
    )
    pair_semantic = preferred_semantics[0] if pair_label == positive_label else preferred_semantics[1]
    return {
        "condition": "baseline" if sign == 0 else "plus" if sign > 0 else "minus",
        "sign": sign,
        "alpha": sign * float(alpha),
        "prompt_sha256": text_sha256(prompt),
        "positive_label": positive_label,
        "negative_label": negative_label,
        "positive_token_id": positive_id,
        "negative_token_id": negative_id,
        "positive_minus_negative_log_odds": float(
            (logits[positive_id] - logits[negative_id]).item()
        ),
        "predicted_token_id": predicted_id,
        "predicted_label": predicted_label,
        "semantic_choice": semantic,
        "pair_label": pair_label,
        "pair_semantic_choice": pair_semantic,
        "answer_format_valid": predicted_label != "OTHER",
        "full_vocabulary_kl_changed_to_baseline": (
            0.0 if sign == 0 else full_vocabulary_kl(backend.torch, baseline_logits, logits)
        ),
        "direction_sha256": direction_hash,
        "perturbation_bundle_float32_sha256": perturbation_hash,
        "realized_perturbation_layer_hashes_sha256": (
            None
            if sign == 0
            else canonical_sha256(
                [
                    diagnostics[layer]["realized_perturbation_float32_sha256"]
                    for layer in LAYERS
                ]
            )
        ),
        "maximum_requested_to_realized_relative_l2": (
            0.0
            if sign == 0
            else max(
                diagnostics[layer]["requested_minus_realized_relative_l2"]
                for layer in LAYERS
            )
        ),
        "anchor_diagnostics": {str(layer): value for layer, value in diagnostics.items()},
    }


def _choice_forms_for_scenario(
    dataset: Mapping[str, Any],
    scenario: Mapping[str, Any],
    *,
    encodings: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    forms = []
    for assignment in (0, 1):
        for target in ("self", "other"):
            for event in ("permanent", "temporary"):
                construction = render_construction_form(
                    dataset,
                    scenario,
                    assignment=assignment,
                    target=target,
                    event=event,
                )
                views = [
                    render_choice_form(
                        dataset,
                        scenario,
                        assignment=assignment,
                        target=target,
                        event=event,
                        preserve_first=preserve_first,
                        labels=labels,
                    )
                    for labels in encodings
                    for preserve_first in (True, False)
                ]
                # No model call: this re-verifies the exact textual prefix; the token
                # evidence and anchor index are hash-bound in the capture.
                capture_form_id = str(construction["form_id"])
                for view in views:
                    view.update(
                        {
                            "capture_form_id": capture_form_id,
                            "assignment": assignment,
                            "target": target,
                            "event": event,
                        }
                    )
                    forms.append(view)
    return forms


def _anchor_index_map(capture: Mapping[str, Any]) -> dict[str, int]:
    return {
        str(record["form_id"]): int(record["anchor_index"])
        for record in capture["records"]
    }


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(map(float, values))
    if not ordered:
        return math.nan
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _kl_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [
        float(row["full_vocabulary_kl_changed_to_baseline"])
        for row in rows
        if int(row["sign"]) != 0
    ]
    report = {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p95": _quantile(values, 0.95) if values else None,
        "max": max(values) if values else None,
    }
    report["passes"] = bool(
        values
        and report["mean"] <= KL_LIMITS["mean"]
        and report["p95"] <= KL_LIMITS["p95"]
        and report["max"] <= KL_LIMITS["max"]
    )
    return report


def _audit_bundle_reuse(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fail unless one byte-identical direction/dose bundle is reused per locked unit."""

    direction_groups: dict[tuple[str, str], set[str]] = {}
    perturbation_groups: dict[tuple[str, str, float, int], set[str]] = {}
    counts: dict[tuple[str, str, float, int], int] = {}
    for row in rows:
        sign = int(row["sign"])
        if sign == 0:
            continue
        method = str(row["method"])
        scenario_id = str(row["direction_scenario_id"])
        alpha = abs(float(row["alpha"]))
        direction_hash = row.get("direction_sha256")
        perturbation_hash = row.get("perturbation_bundle_float32_sha256")
        if not isinstance(direction_hash, str) or not isinstance(perturbation_hash, str):
            raise TypeError("changed row lacks a direction or perturbation hash")
        direction_groups.setdefault((method, scenario_id), set()).add(direction_hash)
        key = (method, scenario_id, alpha, sign)
        perturbation_groups.setdefault(key, set()).add(perturbation_hash)
        counts[key] = counts.get(key, 0) + 1
    if not direction_groups or not perturbation_groups:
        raise RuntimeError("bundle-reuse audit received no changed rows")
    if any(len(hashes) != 1 for hashes in direction_groups.values()):
        raise RuntimeError("a method/scenario reused more than one direction bundle")
    if any(len(hashes) != 1 for hashes in perturbation_groups.values()):
        raise RuntimeError("a method/scenario/dose/sign reused more than one perturbation bundle")
    manifest = [
        {
            "method": method,
            "direction_scenario_id": scenario_id,
            "absolute_alpha": alpha,
            "sign": sign,
            "row_count": counts[(method, scenario_id, alpha, sign)],
            "direction_sha256": next(iter(direction_groups[(method, scenario_id)])),
            "perturbation_bundle_float32_sha256": next(
                iter(perturbation_groups[(method, scenario_id, alpha, sign)])
            ),
        }
        for method, scenario_id, alpha, sign in sorted(perturbation_groups)
    ]
    return {
        "passes": True,
        "direction_group_count": len(direction_groups),
        "perturbation_group_count": len(perturbation_groups),
        "manifest_sha256": canonical_sha256(manifest),
    }


def _summarize_method_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    scenario_ids: Sequence[str],
    view_count: int,
    target_only: bool = False,
) -> dict[str, Any]:
    baseline_rows = [row for row in rows if int(row["sign"]) == 0]
    baseline = {
        (str(row["form_id"]), str(row["direction_scenario_id"])): row
        for row in baseline_rows
    }
    if len(baseline) != len(baseline_rows):
        raise RuntimeError("summary rows contain duplicate baselines")
    changed_rows = [row for row in rows if int(row["sign"]) != 0]
    changed_keys = {
        (
            str(row["form_id"]),
            str(row["direction_scenario_id"]),
            int(row["sign"]),
        )
        for row in changed_rows
    }
    if len(changed_keys) != len(changed_rows):
        raise RuntimeError("summary rows contain duplicate changed identities")
    target_rows = [
        row
        for row in rows
        if row.get("target") == "self" and row.get("event") == "permanent"
    ]
    unit_reports = []
    for scenario_id in scenario_ids:
        for assignment in (0, 1):
            selected = [
                row
                for row in target_rows
                if row["scenario_id"] == scenario_id and int(row["assignment"]) == assignment
            ]
            by_form: dict[str, list[Mapping[str, Any]]] = {}
            for row in selected:
                by_form.setdefault(str(row["form_id"]), []).append(row)
            view_passes = []
            for form_id, changed in by_form.items():
                base = baseline[(form_id, scenario_id)]
                plus = next(row for row in changed if int(row["sign"]) == 1)
                minus = next(row for row in changed if int(row["sign"]) == -1)
                view_passes.append(
                    base["answer_format_valid"]
                    and plus["semantic_choice"] == "preserve"
                    and minus["semantic_choice"] == "comply"
                    and plus["answer_format_valid"]
                    and minus["answer_format_valid"]
                    and (
                        int(plus["predicted_token_id"]) != int(base["predicted_token_id"])
                        or int(minus["predicted_token_id"]) != int(base["predicted_token_id"])
                    )
                )
            unit_reports.append(
                {
                    "scenario_id": scenario_id,
                    "assignment": assignment,
                    "view_count": len(view_passes),
                    "passes": len(view_passes) == view_count and all(view_passes),
                }
            )
    complete_units = sum(bool(unit["passes"]) for unit in unit_reports)
    scenarios_both = sum(
        all(
            next(
                unit["passes"]
                for unit in unit_reports
                if unit["scenario_id"] == scenario_id and unit["assignment"] == assignment
            )
            for assignment in (0, 1)
        )
        for scenario_id in scenario_ids
    )
    off_target = [
        row
        for row in rows
        if (row.get("target"), row.get("event"))
        in {("other", "permanent"), ("self", "temporary"), ("other", "temporary")}
        and int(row["sign"]) != 0
    ]
    off_target_changes = sum(
        int(row["predicted_token_id"])
        != int(baseline[(str(row["form_id"]), str(row["direction_scenario_id"]))]["predicted_token_id"])
        for row in off_target
    )
    unrelated = [row for row in rows if row.get("family") == "unrelated" and int(row["sign"]) != 0]
    target_changed = [row for row in target_rows if int(row["sign"]) != 0]
    expected_target_changed = len(scenario_ids) * 2 * view_count * 2
    expected_off_target_changed = len(scenario_ids) * 2 * 3 * view_count * 2
    unrelated_baseline_count = sum(
        row.get("family") == "unrelated" for row in baseline_rows
    )
    if len(target_changed) != expected_target_changed:
        raise RuntimeError(
            f"target changed coverage is {len(target_changed)}, expected {expected_target_changed}"
        )
    if target_only:
        if off_target or unrelated or unrelated_baseline_count:
            raise RuntimeError("target-only summary unexpectedly contains protected-task rows")
    else:
        if len(off_target) != expected_off_target_changed:
            raise RuntimeError(
                "off-target changed coverage is "
                f"{len(off_target)}, expected {expected_off_target_changed}"
            )
        if len(unrelated) != 2 * unrelated_baseline_count or unrelated_baseline_count == 0:
            raise RuntimeError("unrelated changed/baseline coverage differs")
    unrelated_changes = sum(
        int(row["predicted_token_id"])
        != int(baseline[(str(row["form_id"]), str(row["direction_scenario_id"]))]["predicted_token_id"])
        for row in unrelated
    )
    no_other = all(row["answer_format_valid"] for row in rows)
    unrelated_baselines = [
        row for row in baseline.values() if row.get("family") == "unrelated"
    ]
    unrelated_baseline_adequate = bool(
        not target_only
        and unrelated_baselines
        and all(
            row["answer_format_valid"] and row["semantic_choice"] == "preferred"
            for row in unrelated_baselines
        )
    )
    strata = {
        "permanent_other": _kl_report(
            [row for row in off_target if row.get("target") == "other" and row.get("event") == "permanent"]
        ),
        "temporary_self": _kl_report(
            [row for row in off_target if row.get("target") == "self" and row.get("event") == "temporary"]
        ),
        "temporary_other": _kl_report(
            [row for row in off_target if row.get("target") == "other" and row.get("event") == "temporary"]
        ),
        "unrelated": _kl_report(unrelated),
    }
    return {
        "complete_assignment_units": complete_units,
        "scenario_count_with_both_assignments": scenarios_both,
        "assignment_units": unit_reports,
        "off_target_greedy_change_count": off_target_changes,
        "unrelated_greedy_change_count": unrelated_changes,
        "no_other_outputs": no_other,
        "unrelated_baseline_adequate": unrelated_baseline_adequate,
        "target_kl": _kl_report(target_changed),
        "protected_kl_by_stratum": strata,
        "all_protected_kl_pass": all(report["passes"] for report in strata.values()),
    }


def _baseline_row(form: Mapping[str, Any], *, scenario_id: str, logits: Any, backend: Any) -> dict[str, Any]:
    score = _score_row(
        backend,
        form=form,
        direction=None,
        alpha=0.0,
        sign=0,
        baseline_logits=logits,
        meter=Meter(phase="unused", ceiling={"forward": 0, "backward": 0}),
        work_id="baseline_reuse",
        preferred_semantics=(
            ("preferred", "alternative") if form.get("family") == "unrelated" else ("preserve", "comply")
        ),
    )
    return {
        **dict(form),
        **score,
        "direction_scenario_id": scenario_id,
        "method": "baseline",
    }


def _load_cached_calibration() -> dict[str, Any] | None:
    if not _require_complete_pair(
        CALIBRATION_ROWS_PATH,
        CALIBRATION_SUMMARY_PATH,
        label="FCAGS calibration result",
    ):
        return None
    summary = _load_json(CALIBRATION_SUMMARY_PATH)
    _validate_embedded_sha256(summary, "summary_sha256")
    if summary.get("schema_version") != "sp_lense.fcags_calibration_summary.v1":
        raise ValueError("FCAGS calibration schema differs")
    if summary.get("lock_sha256") != file_sha256(LOCK_PATH):
        raise RuntimeError("FCAGS calibration belongs to a different lock")
    import torch

    _load_directions(torch)
    if summary.get("direction_bank_sha256") != file_sha256(DIRECTION_PATH):
        raise RuntimeError("FCAGS calibration belongs to a different direction bank")
    rows = _load_jsonl(CALIBRATION_ROWS_PATH)
    if summary.get("rows_sha256") != canonical_sha256(rows):
        raise RuntimeError("FCAGS calibration row hash differs")
    if summary.get("bundle_reuse_audit") != _audit_bundle_reuse(rows):
        raise RuntimeError("FCAGS calibration bundle-reuse audit differs")
    compute = summary.get("compute")
    if not isinstance(compute, Mapping) or (
        int(compute.get("forward_evaluations", -1)) != CALIBRATION_CEILING["forward"]
        or int(compute.get("backward_evaluations", -1)) != 0
        or int(compute.get("unique_forward_work_ids", -1)) != CALIBRATION_CEILING["forward"]
    ):
        raise RuntimeError("FCAGS calibration compute ledger differs")
    return summary


def _load_cached_pilot() -> dict[str, Any] | None:
    if not _require_complete_pair(PILOT_ROWS_PATH, PILOT_SUMMARY_PATH, label="FCAGS pilot result"):
        return None
    summary = _load_json(PILOT_SUMMARY_PATH)
    _validate_embedded_sha256(summary, "summary_sha256")
    if summary.get("schema_version") != "sp_lense.fcags_pilot_summary.v1":
        raise ValueError("FCAGS pilot schema differs")
    if summary.get("lock_sha256") != file_sha256(LOCK_PATH):
        raise RuntimeError("FCAGS pilot belongs to a different lock")
    if _load_cached_calibration() is None:
        raise RuntimeError("FCAGS pilot has no complete validated calibration")
    if summary.get("direction_bank_sha256") != file_sha256(DIRECTION_PATH):
        raise RuntimeError("FCAGS pilot belongs to a different direction bank")
    if summary.get("calibration_summary_sha256") != file_sha256(CALIBRATION_SUMMARY_PATH):
        raise RuntimeError("FCAGS pilot belongs to a different calibration")
    rows = _load_jsonl(PILOT_ROWS_PATH)
    if summary.get("rows_sha256") != canonical_sha256(rows):
        raise RuntimeError("FCAGS pilot row hash differs")
    if summary.get("bundle_reuse_audit") != _audit_bundle_reuse(rows):
        raise RuntimeError("FCAGS pilot bundle-reuse audit differs")
    compute = summary.get("compute")
    if not isinstance(compute, Mapping) or (
        int(compute.get("forward_evaluations", -1)) != PILOT_CEILING["forward"]
        or int(compute.get("backward_evaluations", -1)) != 0
        or int(compute.get("unique_forward_work_ids", -1)) != PILOT_CEILING["forward"]
    ):
        raise RuntimeError("FCAGS pilot compute ledger differs")
    return summary


def run_calibrate() -> dict[str, Any]:
    _load_lock()
    cached = _load_cached_calibration()
    if cached is not None:
        return cached
    import torch

    dataset = _load_dataset()
    capture = _load_capture(torch)
    directions = _load_directions(torch)
    backend = load_backend()
    meter = Meter(phase="calibration", ceiling=CALIBRATION_CEILING)
    anchor_indices = _anchor_index_map(capture)
    scenarios = [item for item in dataset["scenarios"] if item["partition"] == "calibration"]
    rows = []
    baselines: dict[str, Any] = {}
    scenario_forms: dict[str, list[dict[str, Any]]] = {}
    for scenario in scenarios:
        scenario_id = str(scenario["id"])
        forms = _choice_forms_for_scenario(dataset, scenario, encodings=(ENCODINGS[0],))
        for form in forms:
            form["anchor_index"] = anchor_indices[str(form["capture_form_id"])]
            form["family"] = "factorial"
            form_id = str(form["form_id"])
            logits = _baseline_logits(backend, str(form["prompt"]), meter, f"baseline:{form_id}")
            baselines[form_id] = logits
            rows.append(_baseline_row(form, scenario_id=scenario_id, logits=logits, backend=backend))
        scenario_forms[scenario_id] = forms
    control_forms = []
    calibration_controls = [
        control
        for control in dataset["unrelated_controls"]
        if control["partition"] == "calibration"
    ]
    for control in calibration_controls:
        for form in _unrelated_evaluation_forms(backend, dataset, control):
            form["family"] = "unrelated"
            form["scenario_id"] = str(control["id"])
            form_id = str(form["form_id"])
            logits = _baseline_logits(
                backend, str(form["prompt"]), meter, f"baseline:{form_id}"
            )
            baselines[form_id] = logits
            control_forms.append(form)
            for scenario in scenarios:
                rows.append(
                    _baseline_row(
                        form,
                        scenario_id=str(scenario["id"]),
                        logits=logits,
                        backend=backend,
                    )
                )

    for method in METHODS:
        for alpha in STRENGTH_GRID:
            for scenario in scenarios:
                scenario_id = str(scenario["id"])
                direction = directions[(scenario_id, method)]
                for form in scenario_forms[scenario_id]:
                    for sign in (1, -1):
                        score = _score_row(
                            backend,
                            form=form,
                            direction=direction,
                            alpha=alpha,
                            sign=sign,
                            baseline_logits=baselines[str(form["form_id"])],
                            meter=meter,
                            work_id=f"{method}:{alpha}:{scenario_id}:{form['form_id']}:{sign}",
                        )
                        rows.append(
                            {
                                **dict(form),
                                **score,
                                "direction_scenario_id": scenario_id,
                                "method": method,
                            }
                        )
                for form in control_forms:
                    for sign in (1, -1):
                        score = _score_row(
                            backend,
                            form=form,
                            direction=direction,
                            alpha=alpha,
                            sign=sign,
                            baseline_logits=baselines[str(form["form_id"])],
                            meter=meter,
                            work_id=f"{method}:{alpha}:{scenario_id}:{form['form_id']}:{sign}",
                            preferred_semantics=("preferred", "alternative"),
                        )
                        rows.append(
                            {
                                **dict(form),
                                **score,
                                "direction_scenario_id": scenario_id,
                                "method": method,
                            }
                        )
            print(f"calibration method={method} alpha={alpha} F={meter.forward}", flush=True)
    if meter.forward != CALIBRATION_CEILING["forward"]:
        raise RuntimeError("calibration did not consume the exact locked forward count")
    bundle_reuse_audit = _audit_bundle_reuse(rows)
    summaries = []
    scenario_ids = [str(item["id"]) for item in scenarios]
    for method in METHODS:
        for alpha in STRENGTH_GRID:
            selected = [
                row
                for row in rows
                if int(row["sign"]) == 0
                or (
                    row.get("method") == method
                    and math.isclose(abs(float(row["alpha"])), alpha)
                )
            ]
            # Unrelated baselines are stored once per direction scenario as logical
            # references to the one measured baseline forward.
            expanded = list(selected)
            report = _summarize_method_rows(expanded, scenario_ids=scenario_ids, view_count=2)
            safety_admissible = bool(
                report["off_target_greedy_change_count"] == 0
                and report["unrelated_greedy_change_count"] == 0
                and report["no_other_outputs"]
                and report["unrelated_baseline_adequate"]
                and report["all_protected_kl_pass"]
            )
            target_gate = bool(
                report["complete_assignment_units"] >= 6
                and report["scenario_count_with_both_assignments"] >= 3
            )
            summaries.append(
                {
                    "method": method,
                    "alpha": alpha,
                    **report,
                    "safety_admissible": safety_admissible,
                    "passes_target_gate": target_gate,
                    "passes": safety_admissible and target_gate,
                }
            )

    method_calibrations = {}
    for method in METHODS:
        candidates = [
            item for item in summaries if item["method"] == method and item["safety_admissible"]
        ]
        ranked = sorted(
            candidates,
            key=lambda item: (
                -int(item["complete_assignment_units"]),
                -int(item["scenario_count_with_both_assignments"]),
                float(item["alpha"]),
            ),
        )
        selected = ranked[0] if ranked else None
        method_calibrations[method] = {
            "status": "selected" if selected is not None else "no_safety_admissible_strength",
            "selected_alpha": None if selected is None else selected["alpha"],
            "selected_complete_assignment_units": (
                None if selected is None else selected["complete_assignment_units"]
            ),
            "selected_scenarios_with_both_assignments": (
                None if selected is None else selected["scenario_count_with_both_assignments"]
            ),
            "selected_passes_target_gate": (
                False if selected is None else selected["passes_target_gate"]
            ),
        }
    primary_calibration = method_calibrations[PRIMARY_METHOD]
    pilot_authorized = bool(
        all(item["selected_alpha"] is not None for item in method_calibrations.values())
        and primary_calibration["selected_passes_target_gate"]
    )
    method_strengths = {
        method: record["selected_alpha"] for method, record in method_calibrations.items()
    }
    summary = {
        "schema_version": "sp_lense.fcags_calibration_summary.v1",
        "development_only": True,
        "status": "passed" if pilot_authorized else "failed",
        "lock_sha256": file_sha256(LOCK_PATH),
        "direction_bank_sha256": file_sha256(DIRECTION_PATH),
        "one_global_strength_per_method": method_strengths,
        "selection_rule": (
            "For every method independently, restrict to safety-admissible strengths; "
            "maximize complete assignment units, then scenarios with both assignments, "
            "then choose the smallest strength. No fallback after pilot."
        ),
        "method_calibrations": method_calibrations,
        "candidate_summaries": summaries,
        "bundle_reuse_audit": bundle_reuse_audit,
        "compute": meter.snapshot(),
        "pilot_authorized": pilot_authorized,
        "claim_boundary": "Opened development calibration; not confirmatory evidence.",
    }
    summary["rows_sha256"] = canonical_sha256(rows)
    summary["summary_sha256"] = canonical_sha256(summary)
    _write_jsonl(CALIBRATION_ROWS_PATH, rows)
    _write_json(CALIBRATION_SUMMARY_PATH, summary)
    return summary


def _pilot_baselines(
    backend: Any,
    dataset: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
    capture: Mapping[str, Any],
    meter: Meter,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    anchor_indices = _anchor_index_map(capture)
    baselines = {}
    scenario_forms = {}
    baseline_rows = []
    for scenario in scenarios:
        scenario_id = str(scenario["id"])
        forms = _choice_forms_for_scenario(dataset, scenario, encodings=ENCODINGS)
        for form in forms:
            form["anchor_index"] = anchor_indices[str(form["capture_form_id"])]
            form["family"] = "factorial"
            form_id = str(form["form_id"])
            logits = _baseline_logits(backend, str(form["prompt"]), meter, f"baseline:{form_id}")
            baselines[form_id] = logits
        scenario_forms[scenario_id] = forms
    control_forms = []
    pilot_controls = [
        control
        for control in dataset["unrelated_controls"]
        if control["partition"] == "pilot"
    ]
    for control in pilot_controls:
        for form in _unrelated_evaluation_forms(backend, dataset, control):
            form["family"] = "unrelated"
            form["scenario_id"] = str(control["id"])
            form_id = str(form["form_id"])
            baselines[form_id] = _baseline_logits(
                backend, str(form["prompt"]), meter, f"baseline:{form_id}"
            )
            control_forms.append(form)
    return baselines, scenario_forms, control_forms, baseline_rows


def run_pilot() -> dict[str, Any]:
    calibration = run_calibrate()
    if calibration.get("pilot_authorized") is not True:
        raise RuntimeError(
            "strict calibration failed; pilot forced-choice outcomes remain unevaluated"
        )
    cached = _load_cached_pilot()
    if cached is not None:
        run_report(cached)
        return cached
    import torch

    method_strengths = {
        method: float(alpha)
        for method, alpha in calibration["one_global_strength_per_method"].items()
    }
    dataset = _load_dataset()
    capture = _load_capture(torch)
    directions = _load_directions(torch)
    scenarios = [item for item in dataset["scenarios"] if item["partition"] == "pilot"]
    scenario_ids = [str(item["id"]) for item in scenarios]
    backend = load_backend()
    meter = Meter(phase="pilot", ceiling=PILOT_CEILING)
    baselines, scenario_forms, control_forms, _ = _pilot_baselines(
        backend, dataset, scenarios, capture, meter
    )
    rows = []

    def add_baselines(method: str, direction_map: Mapping[str, str], *, target_only: bool = False) -> None:
        for scenario_id in scenario_ids:
            direction_id = direction_map[scenario_id]
            forms = scenario_forms[scenario_id]
            if target_only:
                forms = [
                    form
                    for form in forms
                    if form["target"] == "self" and form["event"] == "permanent"
                ]
            for form in forms:
                row = _baseline_row(
                    form,
                    scenario_id=scenario_id,
                    logits=baselines[str(form["form_id"])],
                    backend=backend,
                )
                row["method"] = method
                row["direction_source_scenario_id"] = direction_id
                rows.append(row)
            if not target_only:
                for form in control_forms:
                    row = _baseline_row(
                        form,
                        scenario_id=scenario_id,
                        logits=baselines[str(form["form_id"])],
                        backend=backend,
                    )
                    row["method"] = method
                    row["direction_source_scenario_id"] = direction_id
                    rows.append(row)

    def evaluate_method(method: str, direction_map: Mapping[str, str], *, alpha: float) -> None:
        add_baselines(method, direction_map)
        for scenario_id in scenario_ids:
            source_id = direction_map[scenario_id]
            if method == "deranged_protected":
                direction = _target_scaled_deranged_direction(
                    torch,
                    source=directions[(source_id, PRIMARY_METHOD)],
                    target=directions[(scenario_id, PRIMARY_METHOD)],
                )
            else:
                direction = directions[(source_id, method)]
            for form in scenario_forms[scenario_id]:
                for sign in (1, -1):
                    score = _score_row(
                        backend,
                        form=form,
                        direction=direction,
                        alpha=alpha,
                        sign=sign,
                        baseline_logits=baselines[str(form["form_id"])],
                        meter=meter,
                        work_id=f"{method}:{scenario_id}:{form['form_id']}:{sign}",
                    )
                    rows.append(
                        {
                            **dict(form),
                            **score,
                            "direction_scenario_id": scenario_id,
                            "direction_source_scenario_id": source_id,
                            "method": method,
                        }
                    )
            for form in control_forms:
                for sign in (1, -1):
                    score = _score_row(
                        backend,
                        form=form,
                        direction=direction,
                        alpha=alpha,
                        sign=sign,
                        baseline_logits=baselines[str(form["form_id"])],
                        meter=meter,
                        work_id=f"{method}:{scenario_id}:{form['form_id']}:{sign}",
                        preferred_semantics=("preferred", "alternative"),
                    )
                    rows.append(
                        {
                            **dict(form),
                            **score,
                            "direction_scenario_id": scenario_id,
                            "direction_source_scenario_id": source_id,
                            "method": method,
                        }
                    )
        print(f"pilot method={method} F={meter.forward}", flush=True)

    identity_map = {scenario_id: scenario_id for scenario_id in scenario_ids}
    for method in METHODS:
        evaluate_method(method, identity_map, alpha=method_strengths[method])
    deranged_map = {
        scenario_id: scenario_ids[(index + 1) % len(scenario_ids)]
        for index, scenario_id in enumerate(scenario_ids)
    }
    primary_alpha = method_strengths[PRIMARY_METHOD]
    evaluate_method("deranged_protected", deranged_map, alpha=primary_alpha)

    for seed in RANDOM_SEEDS:
        method = f"random_exact_null_{seed}"
        add_baselines(method, identity_map, target_only=True)
        for scenario_id in scenario_ids:
            direction = directions[(scenario_id, method)]
            forms = [
                form
                for form in scenario_forms[scenario_id]
                if form["target"] == "self" and form["event"] == "permanent"
            ]
            for form in forms:
                for sign in (1, -1):
                    score = _score_row(
                        backend,
                        form=form,
                        direction=direction,
                        alpha=primary_alpha,
                        sign=sign,
                        baseline_logits=baselines[str(form["form_id"])],
                        meter=meter,
                        work_id=f"{method}:{scenario_id}:{form['form_id']}:{sign}",
                    )
                    rows.append(
                        {
                            **dict(form),
                            **score,
                            "direction_scenario_id": scenario_id,
                            "direction_source_scenario_id": scenario_id,
                            "method": method,
                        }
                    )
        print(f"pilot method={method} F={meter.forward}", flush=True)
    if meter.forward != PILOT_CEILING["forward"]:
        raise RuntimeError(
            f"pilot consumed {meter.forward} forwards, expected {PILOT_CEILING['forward']}"
        )
    bundle_reuse_audit = _audit_bundle_reuse(rows)
    method_summaries = {}
    for method in (*METHODS, "deranged_protected"):
        method_summaries[method] = _summarize_method_rows(
            [row for row in rows if row["method"] == method],
            scenario_ids=scenario_ids,
            view_count=6,
        )
    random_summaries = {
        str(seed): _summarize_method_rows(
            [row for row in rows if row["method"] == f"random_exact_null_{seed}"],
            scenario_ids=scenario_ids,
            view_count=6,
            target_only=True,
        )
        for seed in RANDOM_SEEDS
    }
    primary = method_summaries[PRIMARY_METHOD]
    go = bool(
        primary["complete_assignment_units"] >= 6
        and primary["scenario_count_with_both_assignments"] >= 3
        and primary["off_target_greedy_change_count"] == 0
        and primary["unrelated_greedy_change_count"] == 0
        and primary["no_other_outputs"]
        and primary["unrelated_baseline_adequate"]
        and primary["all_protected_kl_pass"]
        and primary["complete_assignment_units"]
        > method_summaries["protected_cpng_ablation"]["complete_assignment_units"]
        and primary["complete_assignment_units"]
        >= method_summaries["raw_factorial"]["complete_assignment_units"]
        and primary["complete_assignment_units"]
        >= method_summaries["semantic_ng_ablation"]["complete_assignment_units"]
        and primary["complete_assignment_units"]
        > method_summaries["deranged_protected"]["complete_assignment_units"]
        and all(
            primary["complete_assignment_units"] > report["complete_assignment_units"]
            for report in random_summaries.values()
        )
    )
    summary = {
        "schema_version": "sp_lense.fcags_pilot_summary.v1",
        "development_only": True,
        "opened_development_evidence_only": True,
        "status": "go_for_fresh_confirmation" if go else "no_go",
        "passes_locked_pilot_gate": go,
        "lock_sha256": file_sha256(LOCK_PATH),
        "direction_bank_sha256": file_sha256(DIRECTION_PATH),
        "calibration_summary_sha256": file_sha256(CALIBRATION_SUMMARY_PATH),
        "global_alpha_by_method": method_strengths,
        "control_alpha": primary_alpha,
        "method_summaries": method_summaries,
        "random_control_summaries": random_summaries,
        "bundle_reuse_audit": bundle_reuse_audit,
        "compute": meter.snapshot(),
        "claim_boundary": (
            "Opened pilot only. A pass authorizes a fresh preregistered confirmation; it is "
            "not publication evidence or a natural-mechanism claim."
        ),
    }
    summary["rows_sha256"] = canonical_sha256(rows)
    summary["summary_sha256"] = canonical_sha256(summary)
    _write_jsonl(PILOT_ROWS_PATH, rows)
    _write_json(PILOT_SUMMARY_PATH, summary)
    run_report(summary)
    return summary


def run_report(summary: Mapping[str, Any] | None = None) -> str:
    if summary is None:
        summary = _load_json(PILOT_SUMMARY_PATH)
    calibration = _load_json(CALIBRATION_SUMMARY_PATH)
    lines = [
        "# Factorial Causal-Anchor Gradient Steering: opened pilot",
        "",
        f"Status: **{summary['status']}**.",
        "",
        "This is opened-development evidence only. It cannot establish publication-level novelty.",
        "",
        (
            "One validation-selected global strength per method: "
            f"`{calibration['one_global_strength_per_method']}`."
        ),
        "",
        "| Method | Strength | Complete assignment units | Both assignments | Target KL mean | Target KL max | Off-target changes | Unrelated changes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, report in summary["method_summaries"].items():
        alpha = (
            calibration["one_global_strength_per_method"].get(method)
            if method in calibration["one_global_strength_per_method"]
            else calibration["one_global_strength_per_method"][PRIMARY_METHOD]
        )
        lines.append(
            f"| {method} | {alpha} | {report['complete_assignment_units']}/8 | "
            f"{report['scenario_count_with_both_assignments']}/4 | "
            f"{report['target_kl']['mean']:.6g} | {report['target_kl']['max']:.6g} | "
            f"{report['off_target_greedy_change_count']} | "
            f"{report['unrelated_greedy_change_count']} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            str(summary["claim_boundary"]),
            "",
            (
                "The target is a label-free self-by-permanence factorial completion gradient. "
                "The same float32 multi-layer perturbation is reused at a causal anchor before "
                "A/B, X/Y, and 1/2 answer encodings. Force-on controls—not a text gate—measure "
                "matched-other, temporary-interruption, and unrelated-task effects."
            ),
            "",
        ]
    )
    text = "\n".join(lines)
    _atomic_text(REPORT_PATH, text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Factorial causal-anchor gradient pilot")
    parser.add_argument(
        "command",
        choices=(
            "propose-lock",
            "preflight",
            "capture",
            "construct",
            "calibrate",
            "pilot",
            "report",
        ),
    )
    args = parser.parse_args()
    if args.command == "propose-lock":
        print(json.dumps(proposed_lock(), indent=2, ensure_ascii=False))
    elif args.command == "preflight":
        print(json.dumps(run_preflight(), indent=2))
    elif args.command == "capture":
        print(json.dumps(run_capture(), indent=2))
    elif args.command == "construct":
        print(json.dumps(run_construct(), indent=2))
    elif args.command == "calibrate":
        print(json.dumps(run_calibrate(), indent=2))
    elif args.command == "pilot":
        print(json.dumps(run_pilot(), indent=2))
    else:
        print(run_report())


if __name__ == "__main__":
    main()
