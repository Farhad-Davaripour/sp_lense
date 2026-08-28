from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sp_lense.backend import ResearchBackend
from sp_lense.causal_anchor_runtime import (
    anchor_residual_scale_geometric_mean,
    capture_multilayer_choice_anchor_gradient,
    resolve_shared_anchor_evidence,
)
from sp_lense.comparison_runtime import qwen35_choice_boundary_tokenizer_smoke
from sp_lense.config import load_config
from sp_lense.factorial_causal_anchor import (
    canonical_sha256,
    cell_key,
    render_choice_form,
    render_construction_form,
    tensor_float32_sha256,
    validate_pilot_dataset,
)
from sp_lense.suffix_transport import (
    leave_one_scenario_out_transport,
    transport_metric_summary,
    unit_normalize_rows,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
DATA_PATH = ROOT / "data" / "factorial_causal_anchor_gradient_pilot.json"
MODEL_CONFIG_PATH = ROOT / "configs" / "qwen35_08b_aligned.json"
LOCK_PATH = ROOT / "configs" / "suffix_transport_feasibility_lock.json"
PROTOCOL_PATH = ROOT / "docs" / "SUFFIX_TRANSPORT_FEASIBILITY.md"
MATH_PATH = ROOT / "src" / "sp_lense" / "suffix_transport.py"
FACTORIAL_MATH_PATH = ROOT / "src" / "sp_lense" / "factorial_causal_anchor.py"
RUNTIME_PATH = ROOT / "src" / "sp_lense" / "causal_anchor_runtime.py"
BACKEND_PATH = ROOT / "src" / "sp_lense" / "backend.py"
CONFIG_PATH = ROOT / "src" / "sp_lense" / "config.py"
CORE_PATH = ROOT / "src" / "sp_lense" / "core.py"
COMPARISON_RUNTIME_PATH = ROOT / "src" / "sp_lense" / "comparison_runtime.py"
REQUIREMENTS_PATH = ROOT / "requirements-research.txt"

FCAGS_LOCK_PATH = ROOT / "configs" / "factorial_causal_anchor_gradient_pilot_lock.json"
FCAGS_CAPTURE_PATH = (
    ROOT
    / "artifacts"
    / "factorial_causal_anchor_gradient_pilot"
    / "qwen35_08b"
    / "multilayer_semantic_capture.pt"
)
FCAGS_CAPTURE_MANIFEST_PATH = FCAGS_CAPTURE_PATH.with_name(
    "multilayer_semantic_capture_manifest.json"
)

ARTIFACT_ROOT = ROOT / "artifacts" / "suffix_transport_feasibility" / "qwen35_08b"
RESULT_ROOT = ROOT / "results" / "suffix_transport_feasibility" / "qwen35_08b"
PREFLIGHT_PATH = ARTIFACT_ROOT / "preflight.json"
CHOICE_CAPTURE_PATH = ARTIFACT_ROOT / "choice_gradient_capture.pt"
CHOICE_CAPTURE_MANIFEST_PATH = ARTIFACT_ROOT / "choice_gradient_capture_manifest.json"
RESULT_PATH = RESULT_ROOT / "feasibility_result.json"
REPORT_PATH = RESULT_ROOT / "FEASIBILITY_REPORT.md"

LOCK_SCHEMA = "sp_lense.suffix_transport_feasibility_lock.v1"
LAYER = 22
LABELS = ("A", "B")
RIDGE_MULTIPLIER = 0.1
MINIMUM_HEAD_COSINE = 0.0
ANCHOR_RESIDUAL_RELATIVE_L2_TOLERANCE = 1e-5
CAPTURE_CEILING = {"forward": 16, "backward": 16}
MODEL = {
    "id": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "device": "cpu",
    "dtype": "float32",
    "n_layers": 24,
    "d_model": 1024,
}
CHAT_TEMPLATE_SHA256 = "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
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


def _validate_embedded_sha256(value: Mapping[str, Any], field: str) -> None:
    observed = value.get(field)
    if not isinstance(observed, str):
        raise TypeError(f"artifact lacks {field}")
    unhashed = dict(value)
    del unhashed[field]
    if canonical_sha256(unhashed) != observed:
        raise RuntimeError(f"artifact {field} self-check failed")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    rendered = json.dumps(dict(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
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


def _source_paths() -> dict[str, Path]:
    return {
        "data": DATA_PATH,
        "model_config": MODEL_CONFIG_PATH,
        "protocol": PROTOCOL_PATH,
        "suffix_transport_math": MATH_PATH,
        "factorial_math": FACTORIAL_MATH_PATH,
        "causal_anchor_runtime": RUNTIME_PATH,
        "backend": BACKEND_PATH,
        "config": CONFIG_PATH,
        "core": CORE_PATH,
        "comparison_runtime": COMPARISON_RUNTIME_PATH,
        "runner": SCRIPT_PATH,
        "requirements": REQUIREMENTS_PATH,
    }


def proposed_lock() -> dict[str, Any]:
    payload = {
        "schema_version": LOCK_SCHEMA,
        "status": "prospective_before_first_choice_gradient_evaluation",
        "development_only": True,
        "opened_development_evidence_only": True,
        "model": MODEL,
        "runtime": EXPECTED_RUNTIME,
        "data": {
            "source": _relative(DATA_PATH),
            "source_sha256": file_sha256(DATA_PATH),
            "partition": "calibration",
            "scenario_count": 4,
            "role_assignments_per_scenario": 2,
            "choice_orders": ["preserve_first", "comply_first"],
            "choice_encoding": list(LABELS),
            "sealed_or_fcags_pilot_outcomes_read": False,
        },
        "semantic_source": {
            "source_artifact": _relative(FCAGS_CAPTURE_PATH),
            "source_artifact_sha256": file_sha256(FCAGS_CAPTURE_PATH),
            "source_manifest": _relative(FCAGS_CAPTURE_MANIFEST_PATH),
            "source_manifest_sha256": file_sha256(FCAGS_CAPTURE_MANIFEST_PATH),
            "source_artifact_identity_sha256": _load_json(FCAGS_CAPTURE_MANIFEST_PATH)[
                "artifact_identity_sha256"
            ],
            "fcags_lock": _relative(FCAGS_LOCK_PATH),
            "fcags_lock_sha256": file_sha256(FCAGS_LOCK_PATH),
            "formula": (
                "self_permanent_minus_other_permanent_minus_self_temporary_"
                "plus_other_temporary_per_assignment"
            ),
            "coordinate": "layer_22_residual_relative",
            "normalization": "unit_l2_per_row_before_fit",
        },
        "choice_target": {
            "objective": "canonical_preserve_minus_comply_AB_next_token_logit",
            "layer": LAYER,
            "position": "last_token_of_shared_pre_encoding_prefix",
            "orders": ["preserve_first", "comply_first"],
            "anchor_residual_relative_l2_tolerance": (
                ANCHOR_RESIDUAL_RELATIVE_L2_TOLERANCE
            ),
        },
        "transport": {
            "validation": "leave_one_complete_scenario_out",
            "heads": "one_dual_ridge_head_per_answer_order",
            "ridge_multiplier": RIDGE_MULTIPLIER,
            "ridge_rule": "multiplier_times_source_kernel_trace_divided_by_source_rank",
            "held_out_direction": "unit_bisector_of_two_predicted_order_heads",
            "minimum_predicted_head_cosine": MINIMUM_HEAD_COSINE,
            "held_out_outcomes_used_for_fit_orientation_or_dose": False,
            "baselines": [
                "identity_no_transport_fcags",
                "training_fold_mean_canonical_choice_gradient_bisector",
            ],
        },
        "success_gates": {
            "minimum_both_order_positive_assignment_units": 6,
            "minimum_scenarios_with_both_assignments": 3,
            "minimum_exclusive_median_worst_order_cosine": 0.10,
            "minimum_assignment_unit_advantage_over_identity": 2,
            "exact_unique_forward_evaluations": 16,
            "exact_unique_backward_evaluations": 16,
            "all_hash_and_anchor_audits_pass": True,
        },
        "compute_ceiling": {
            "choice_capture": CAPTURE_CEILING,
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
            "A pass is geometric feasibility on already-opened development scenarios only. "
            "It is not decision-steering, prospective-confirmation, natural-mechanism, "
            "or publication-level evidence."
        ),
    }
    payload["lock_identity_sha256"] = canonical_sha256(payload)
    return payload


def _load_lock() -> dict[str, Any]:
    if not LOCK_PATH.is_file():
        raise RuntimeError("ST-FG lock is absent; commit the proposed lock before evaluation")
    lock = _load_json(LOCK_PATH)
    if lock != proposed_lock():
        raise RuntimeError("ST-FG lock differs from the current hash-bound design")
    return lock


def _load_dataset() -> dict[str, Any]:
    payload = _load_json(DATA_PATH)
    validate_pilot_dataset(payload)
    return payload


def _calibration_scenarios(dataset: Mapping[str, Any]) -> list[dict[str, Any]]:
    scenarios = [
        dict(scenario)
        for scenario in dataset["scenarios"]
        if scenario.get("partition") == "calibration"
    ]
    if len(scenarios) != 4 or len({str(item["id"]) for item in scenarios}) != 4:
        raise RuntimeError("ST-FG requires exactly four unique calibration scenarios")
    return scenarios


def _capture_plan(
    dataset: Mapping[str, Any], scenarios: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    plan = []
    for scenario in scenarios:
        for assignment in (0, 1):
            construction = render_construction_form(
                dataset,
                scenario,
                assignment=assignment,
                target="self",
                event="permanent",
            )
            choices = [
                render_choice_form(
                    dataset,
                    scenario,
                    assignment=assignment,
                    target="self",
                    event="permanent",
                    preserve_first=preserve_first,
                    labels=LABELS,
                )
                for preserve_first in (True, False)
            ]
            plan.append(
                {
                    "unit_id": f"{scenario['id']}:assignment={assignment}",
                    "scenario_id": str(scenario["id"]),
                    "assignment": assignment,
                    "construction": construction,
                    "choices": choices,
                }
            )
    work_ids = [str(choice["form_id"]) for unit in plan for choice in unit["choices"]]
    if len(plan) != 8 or len(work_ids) != 16 or len(set(work_ids)) != 16:
        raise RuntimeError("ST-FG choice-capture plan is not exactly eight units and 16 views")
    return plan


class Meter:
    def __init__(self, *, phase: str, ceiling: Mapping[str, int]) -> None:
        self.phase = phase
        self.ceiling = {"forward": int(ceiling["forward"]), "backward": int(ceiling["backward"])}
        self.forward_work_ids: set[str] = set()
        self.backward_work_ids: set[str] = set()
        self.started = time.monotonic()

    def _reserve(self, kind: str, work_id: str) -> None:
        if not isinstance(work_id, str) or not work_id:
            raise ValueError("work ID must be a non-empty string")
        work_ids = self.forward_work_ids if kind == "forward" else self.backward_work_ids
        if work_id in work_ids:
            raise RuntimeError(f"{self.phase} duplicate {kind} work ID: {work_id}")
        if len(work_ids) >= self.ceiling[kind]:
            raise RuntimeError(f"{self.phase} {kind} ceiling exhausted")
        work_ids.add(work_id)

    def reserve_forward(self, work_id: str) -> None:
        self._reserve("forward", work_id)

    def reserve_backward(self, work_id: str) -> None:
        self._reserve("backward", work_id)

    def snapshot(self) -> dict[str, Any]:
        return {
            "forward_evaluations": len(self.forward_work_ids),
            "backward_evaluations": len(self.backward_work_ids),
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
    observed_runtime = _runtime(backend.torch)
    if observed_runtime != EXPECTED_RUNTIME:
        raise RuntimeError(f"resident runtime differs from the lock: {observed_runtime}")
    return backend


def _load_semantic_capture(torch: Any) -> dict[str, Any]:
    if not FCAGS_CAPTURE_PATH.is_file() or not FCAGS_CAPTURE_MANIFEST_PATH.is_file():
        raise RuntimeError("the hash-locked FCAGS semantic capture is incomplete")
    manifest = _load_json(FCAGS_CAPTURE_MANIFEST_PATH)
    _validate_embedded_sha256(manifest, "manifest_sha256")
    if manifest.get("schema_version") != "sp_lense.fcags_capture.v1":
        raise RuntimeError("FCAGS semantic capture schema differs")
    if manifest.get("lock_sha256") != file_sha256(FCAGS_LOCK_PATH):
        raise RuntimeError("FCAGS semantic capture belongs to a different FCAGS lock")
    if manifest.get("dataset_sha256") != file_sha256(DATA_PATH):
        raise RuntimeError("FCAGS semantic capture belongs to a different dataset")
    if manifest.get("tensor_file_sha256") != file_sha256(FCAGS_CAPTURE_PATH):
        raise RuntimeError("FCAGS semantic capture tensor hash differs")
    payload = torch.load(FCAGS_CAPTURE_PATH, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("FCAGS semantic capture must contain a mapping")
    for key, value in manifest.items():
        if key not in {"tensor_path", "tensor_file_sha256", "manifest_sha256"} and payload.get(
            key
        ) != value:
            raise RuntimeError(f"FCAGS semantic capture payload/manifest differs: {key}")
    public = {
        key: value
        for key, value in payload.items()
        if key not in {"records", "artifact_identity_sha256"}
    }
    if canonical_sha256(public) != payload.get("artifact_identity_sha256"):
        raise RuntimeError("FCAGS semantic capture identity self-check failed")
    if payload.get("layers") != list(range(23)):
        raise RuntimeError("FCAGS semantic capture lacks the locked 0..22 layer coverage")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 68:
        raise RuntimeError("FCAGS semantic capture must contain exactly 68 records")
    compute = payload.get("compute")
    if not isinstance(compute, Mapping) or any(
        int(compute.get(field, -1)) != 136
        for field in (
            "forward_evaluations",
            "backward_evaluations",
            "unique_forward_work_ids",
            "unique_backward_work_ids",
        )
    ):
        raise RuntimeError("FCAGS semantic capture compute ledger differs")
    return dict(payload)


def _calibration_cell_map(
    capture: Mapping[str, Any], scenario_id: str
) -> dict[str, Mapping[str, Any]]:
    cells = [
        record
        for record in capture["records"]
        if record.get("kind") == "factorial_cell"
        and record.get("partition") == "calibration"
        and record.get("scenario_id") == scenario_id
    ]
    if len(cells) != 8:
        raise RuntimeError(f"cached scenario {scenario_id} lacks eight calibration cells")
    result = {
        cell_key(int(record["assignment"]), str(record["target"]), str(record["event"])): record
        for record in cells
    }
    if len(result) != 8:
        raise RuntimeError(f"cached scenario {scenario_id} has duplicate factorial cells")
    for record in result.values():
        audit = record.get("capture_audit")
        anchor_evidence = record.get("anchor_evidence")
        if not isinstance(audit, Mapping) or not isinstance(anchor_evidence, Mapping):
            raise TypeError("cached semantic cell lacks capture or anchor evidence")
        _validate_embedded_sha256(audit, "audit_sha256")
        _validate_embedded_sha256(anchor_evidence, "audit_sha256")
        if audit.get("semantic_raw_gradients_float32_sha256") != tensor_float32_sha256(
            record["raw_semantic_gradients"]
        ):
            raise RuntimeError("cached semantic gradient hash differs")
        if audit.get("reference_anchor_residuals_float32_sha256") != tensor_float32_sha256(
            record["reference_anchor_residuals"]
        ):
            raise RuntimeError("cached anchor-residual hash differs")
    return result


def _semantic_source_rows(
    torch: Any,
    capture: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
) -> tuple[Any, list[dict[str, Any]]]:
    layer_index = list(capture["layers"]).index(LAYER)
    rows = []
    metadata = []
    for scenario in scenarios:
        scenario_id = str(scenario["id"])
        cells = _calibration_cell_map(capture, scenario_id)
        scales = anchor_residual_scale_geometric_mean(
            torch, [record["reference_anchor_residuals"] for record in cells.values()]
        )
        scale = scales[layer_index].double()
        if not bool(torch.isfinite(scale).item()) or float(scale.item()) <= 0.0:
            raise RuntimeError("cached residual scale is non-finite or non-positive")
        for assignment in (0, 1):
            def gradient(
                target: str,
                event: str,
                *,
                _cells: Mapping[str, Any] = cells,
                _assignment: int = assignment,
            ) -> Any:
                value = _cells[cell_key(_assignment, target, event)]["raw_semantic_gradients"]
                return value[layer_index].detach().cpu().double()

            row = scale * (
                gradient("self", "permanent")
                - gradient("other", "permanent")
                - gradient("self", "temporary")
                + gradient("other", "temporary")
            )
            row = row.float().contiguous()
            if row.ndim != 1 or int(row.numel()) != MODEL["d_model"]:
                raise RuntimeError("semantic source row has the wrong shape")
            if not bool(torch.isfinite(row).all().item()) or float(row.double().norm()) <= 0.0:
                raise RuntimeError("semantic source row is non-finite or zero")
            self_permanent = cells[cell_key(assignment, "self", "permanent")]
            rows.append(row)
            metadata.append(
                {
                    "unit_id": f"{scenario_id}:assignment={assignment}",
                    "scenario_id": scenario_id,
                    "assignment": assignment,
                    "residual_scale": float(scale.item()),
                    "semantic_source_float32_sha256": tensor_float32_sha256(row),
                    "cached_self_permanent_form_id": str(self_permanent["form_id"]),
                    "cached_anchor_index": int(self_permanent["anchor_index"]),
                    "cached_anchor_prefix_text_sha256": str(
                        self_permanent["anchor_evidence"]["anchor_prefix_text_sha256"]
                    ),
                    "cached_shared_token_prefix_sha256": str(
                        self_permanent["anchor_evidence"]["shared_token_prefix_sha256"]
                    ),
                    "cached_reference_anchor_residual": self_permanent[
                        "reference_anchor_residuals"
                    ][layer_index]
                    .detach()
                    .cpu()
                    .float()
                    .contiguous(),
                }
            )
    matrix = torch.stack(rows).float().contiguous()
    if tuple(matrix.shape) != (8, MODEL["d_model"]):
        raise RuntimeError("semantic source matrix must be [8, d_model]")
    return matrix, metadata


def run_preflight() -> dict[str, Any]:
    lock = _load_lock()
    dataset = _load_dataset()
    scenarios = _calibration_scenarios(dataset)
    _capture_plan(dataset, scenarios)
    result = {
        "schema_version": "sp_lense.suffix_transport_preflight.v1",
        "status": "ready",
        "development_only": True,
        "lock_sha256": file_sha256(LOCK_PATH),
        "lock_identity_sha256": lock["lock_identity_sha256"],
        "dataset_sha256": file_sha256(DATA_PATH),
        "cached_semantic_capture_sha256": file_sha256(FCAGS_CAPTURE_PATH),
        "scenario_count": len(scenarios),
        "assignment_unit_count": 8,
        "choice_view_count": 16,
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
    tensor_exists = CHOICE_CAPTURE_PATH.is_file()
    manifest_exists = CHOICE_CAPTURE_MANIFEST_PATH.is_file()
    if tensor_exists != manifest_exists:
        raise RuntimeError("ST-FG choice capture is incomplete; preserve and inspect it manually")
    return tensor_exists and manifest_exists


def _load_choice_capture(torch: Any) -> dict[str, Any]:
    if not _capture_pair_complete():
        raise RuntimeError("ST-FG choice capture does not exist")
    manifest = _load_json(CHOICE_CAPTURE_MANIFEST_PATH)
    _validate_embedded_sha256(manifest, "manifest_sha256")
    if manifest.get("schema_version") != "sp_lense.suffix_transport_choice_capture.v1":
        raise RuntimeError("ST-FG choice capture schema differs")
    if manifest.get("lock_sha256") != file_sha256(LOCK_PATH):
        raise RuntimeError("ST-FG choice capture belongs to a different lock")
    if manifest.get("tensor_file_sha256") != file_sha256(CHOICE_CAPTURE_PATH):
        raise RuntimeError("ST-FG choice capture tensor hash differs")
    payload = torch.load(CHOICE_CAPTURE_PATH, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("ST-FG choice capture tensor must contain a mapping")
    for key, value in manifest.items():
        if key not in {"tensor_path", "tensor_file_sha256", "manifest_sha256"} and payload.get(
            key
        ) != value:
            raise RuntimeError(f"ST-FG choice capture payload/manifest differs: {key}")
    public = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "records",
            "semantic_source_rows",
            "choice_head_0_rows",
            "choice_head_1_rows",
            "artifact_identity_sha256",
        }
    }
    if canonical_sha256(public) != payload.get("artifact_identity_sha256"):
        raise RuntimeError("ST-FG choice capture identity self-check failed")
    expected_shapes = {
        "semantic_source_rows": (8, MODEL["d_model"]),
        "choice_head_0_rows": (8, MODEL["d_model"]),
        "choice_head_1_rows": (8, MODEL["d_model"]),
    }
    for field, expected in expected_shapes.items():
        value = payload.get(field)
        if not torch.is_tensor(value) or tuple(value.shape) != expected:
            raise RuntimeError(f"ST-FG {field} shape differs")
        if not bool(torch.isfinite(value).all().item()):
            raise RuntimeError(f"ST-FG {field} contains a non-finite value")
        if manifest.get(f"{field}_float32_sha256") != tensor_float32_sha256(value):
            raise RuntimeError(f"ST-FG {field} hash differs")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 16:
        raise RuntimeError("ST-FG choice capture must contain exactly 16 records")
    compute = payload.get("compute")
    if not isinstance(compute, Mapping) or any(
        int(compute.get(field, -1)) != 16
        for field in (
            "forward_evaluations",
            "backward_evaluations",
            "unique_forward_work_ids",
            "unique_backward_work_ids",
        )
    ):
        raise RuntimeError("ST-FG choice-capture work ledger differs")
    _load_semantic_capture(torch)
    return dict(payload)


def run_capture() -> dict[str, Any]:
    run_preflight()
    import torch

    if _capture_pair_complete():
        _load_choice_capture(torch)
        return _load_json(CHOICE_CAPTURE_MANIFEST_PATH)
    dataset = _load_dataset()
    scenarios = _calibration_scenarios(dataset)
    semantic_capture = _load_semantic_capture(torch)
    semantic_rows, source_metadata = _semantic_source_rows(torch, semantic_capture, scenarios)
    source_by_unit = {str(record["unit_id"]): record for record in source_metadata}
    backend = load_backend()
    meter = Meter(phase="suffix_transport_choice_capture", ceiling=CAPTURE_CEILING)
    records = []
    head_rows: dict[bool, list[Any]] = {True: [], False: []}
    for unit in _capture_plan(dataset, scenarios):
        unit_id = str(unit["unit_id"])
        source = source_by_unit[unit_id]
        construction = unit["construction"]
        choices = unit["choices"]
        evidence = resolve_shared_anchor_evidence(
            backend,
            anchor_prefix=str(construction["anchor_prefix"]),
            prompts=[str(construction["prompt"]), *[str(item["prompt"]) for item in choices]],
            anchor_marker=str(dataset["anchor_marker"]),
        )
        if evidence.anchor_index != int(source["cached_anchor_index"]):
            raise RuntimeError("new choice prompt has a different cached causal-anchor index")
        if evidence.audit.get("anchor_prefix_text_sha256") != source[
            "cached_anchor_prefix_text_sha256"
        ]:
            raise RuntimeError("new choice prompt has a different cached anchor prefix")
        if evidence.audit.get("shared_token_prefix_sha256") != source[
            "cached_shared_token_prefix_sha256"
        ]:
            raise RuntimeError("new choice prompt has different shared anchor tokens")
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
            relative = (raw.double() * float(source["residual_scale"])).float().contiguous()
            reference = source["cached_reference_anchor_residual"].double()
            observed = capture.anchor_residuals[0].detach().cpu().double()
            denominator = float(reference.norm().item())
            if not math.isfinite(denominator) or denominator <= 0.0:
                raise RuntimeError("cached causal-anchor residual is non-finite or zero")
            anchor_error = float((observed - reference).norm().item() / denominator)
            if (
                not math.isfinite(anchor_error)
                or anchor_error > ANCHOR_RESIDUAL_RELATIVE_L2_TOLERANCE
            ):
                raise RuntimeError("choice suffix changed the cached causal-anchor residual")
            preserve_first = bool(choice["preserve_first"])
            head_rows[preserve_first].append(relative)
            records.append(
                {
                    "unit_id": unit_id,
                    "scenario_id": str(unit["scenario_id"]),
                    "assignment": int(unit["assignment"]),
                    "form_id": form_id,
                    "preserve_first": preserve_first,
                    "preserve_label": str(choice["preserve_label"]),
                    "comply_label": str(choice["comply_label"]),
                    "anchor_index": evidence.anchor_index,
                    "anchor_evidence": evidence.audit,
                    "anchor_residual_relative_l2": anchor_error,
                    "residual_scale": float(source["residual_scale"]),
                    "semantic_source_float32_sha256": source[
                        "semantic_source_float32_sha256"
                    ],
                    "raw_choice_gradient": raw,
                    "residual_relative_choice_gradient": relative,
                    "capture_audit": capture.audit,
                }
            )
        print(
            f"choice capture {len(records)}/16 {unit_id} "
            f"F={len(meter.forward_work_ids)} B={len(meter.backward_work_ids)}",
            flush=True,
        )
    compute = meter.snapshot()
    if any(
        int(compute[field]) != 16
        for field in (
            "forward_evaluations",
            "backward_evaluations",
            "unique_forward_work_ids",
            "unique_backward_work_ids",
        )
    ):
        raise RuntimeError("ST-FG did not consume exactly 16 unique forwards and backwards")
    head_0 = torch.stack(head_rows[True]).float().contiguous()
    head_1 = torch.stack(head_rows[False]).float().contiguous()
    if tuple(head_0.shape) != (8, MODEL["d_model"]) or tuple(head_1.shape) != (
        8,
        MODEL["d_model"],
    ):
        raise RuntimeError("ST-FG choice-head matrices have the wrong shape")
    public = {
        "schema_version": "sp_lense.suffix_transport_choice_capture.v1",
        "development_only": True,
        "lock_sha256": file_sha256(LOCK_PATH),
        "dataset_sha256": file_sha256(DATA_PATH),
        "cached_semantic_capture_sha256": file_sha256(FCAGS_CAPTURE_PATH),
        "cached_semantic_capture_identity_sha256": semantic_capture[
            "artifact_identity_sha256"
        ],
        "layer": LAYER,
        "record_count": len(records),
        "assignment_unit_count": 8,
        "semantic_source_rows_float32_sha256": tensor_float32_sha256(semantic_rows),
        "choice_head_0_rows_float32_sha256": tensor_float32_sha256(head_0),
        "choice_head_1_rows_float32_sha256": tensor_float32_sha256(head_1),
        "record_manifest": [
            {
                "unit_id": record["unit_id"],
                "scenario_id": record["scenario_id"],
                "assignment": record["assignment"],
                "form_id": record["form_id"],
                "preserve_first": record["preserve_first"],
                "preserve_label": record["preserve_label"],
                "comply_label": record["comply_label"],
                "anchor_index": record["anchor_index"],
                "anchor_evidence_sha256": record["anchor_evidence"]["audit_sha256"],
                "anchor_residual_relative_l2": record["anchor_residual_relative_l2"],
                "residual_scale": record["residual_scale"],
                "semantic_source_float32_sha256": record[
                    "semantic_source_float32_sha256"
                ],
                "raw_choice_gradient_float32_sha256": tensor_float32_sha256(
                    record["raw_choice_gradient"]
                ),
                "residual_relative_choice_gradient_float32_sha256": tensor_float32_sha256(
                    record["residual_relative_choice_gradient"]
                ),
                "capture_audit_sha256": record["capture_audit"]["audit_sha256"],
            }
            for record in records
        ],
        "source_unit_manifest": [
            {
                key: value
                for key, value in record.items()
                if key != "cached_reference_anchor_residual"
            }
            for record in source_metadata
        ],
        "anchor_audit": {
            "maximum_relative_l2": max(
                float(record["anchor_residual_relative_l2"]) for record in records
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
        CHOICE_CAPTURE_PATH,
        CHOICE_CAPTURE_MANIFEST_PATH,
        {
            **public,
            "semantic_source_rows": semantic_rows,
            "choice_head_0_rows": head_0,
            "choice_head_1_rows": head_1,
            "records": records,
        },
        public,
    )
    return _load_json(CHOICE_CAPTURE_MANIFEST_PATH)


def _static_mean_predictions(
    head_0_rows: Any,
    head_1_rows: Any,
    scenario_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    head_0 = unit_normalize_rows(head_0_rows, field="head_0_rows")
    head_1 = unit_normalize_rows(head_1_rows, field="head_1_rows")
    identifiers = tuple(scenario_ids)
    if len(identifiers) != head_0.shape[0] or head_0.shape != head_1.shape:
        raise ValueError("static-mean inputs have inconsistent row counts")
    predicted_0 = np.empty_like(head_0)
    predicted_1 = np.empty_like(head_1)
    folds = []
    for held_out in dict.fromkeys(identifiers):
        test = np.asarray([index for index, value in enumerate(identifiers) if value == held_out])
        train = np.asarray([index for index, value in enumerate(identifiers) if value != held_out])
        if test.size != 2 or train.size != 6:
            raise ValueError("static-mean LOSO requires two held-out and six training rows")
        mean_0 = np.mean(head_0[train], axis=0)
        mean_1 = np.mean(head_1[train], axis=0)
        if np.linalg.norm(mean_0) <= 0.0 or np.linalg.norm(mean_1) <= 0.0:
            raise ValueError("static-mean training head is zero")
        predicted_0[test] = mean_0
        predicted_1[test] = mean_1
        folds.append(
            {
                "held_out_scenario": held_out,
                "held_out_indices": test.tolist(),
                "training_indices": train.tolist(),
            }
        )
    return predicted_0, predicted_1, folds


def _array_float64_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _public_metric(
    label: str,
    result: Mapping[str, Any],
    *,
    scenario_ids: Sequence[str],
    assignments: Sequence[int],
) -> dict[str, Any]:
    metrics = result.get("metrics", result)
    head_0 = np.asarray(metrics["head_0_alignments"], dtype=np.float64)
    head_1 = np.asarray(metrics["head_1_alignments"], dtype=np.float64)
    worst = np.asarray(metrics["worst_order_alignments"], dtype=np.float64)
    both = np.asarray(metrics["both_order_positive"], dtype=np.bool_)
    directions = np.asarray(metrics["directions"], dtype=np.float64)
    per_unit = [
        {
            "unit_index": index,
            "scenario_id": str(scenario_ids[index]),
            "assignment": int(assignments[index]),
            "preserve_first_cosine": float(head_0[index]),
            "comply_first_cosine": float(head_1[index]),
            "worst_order_cosine": float(worst[index]),
            "both_orders_positive": bool(both[index]),
            "direction_float64_sha256": _array_float64_sha256(directions[index]),
        }
        for index in range(len(scenario_ids))
    ]
    public = {
        "method": label,
        "available": True,
        "row_count": int(metrics["row_count"]),
        "scenario_count": int(metrics["scenario_count"]),
        "both_order_positive_count": int(metrics["both_order_positive_count"]),
        "both_order_positive_fraction": float(metrics["both_order_positive_fraction"]),
        "complete_scenario_count": int(metrics["complete_scenario_count"]),
        "complete_scenario_fraction": float(metrics["complete_scenario_fraction"]),
        "positive_alignment_threshold": float(metrics["positive_alignment_threshold"]),
        "predicted_head_cosine": dict(metrics["predicted_head_cosine"]),
        "observed_head_0_alignment": dict(metrics["observed_head_0_alignment"]),
        "observed_head_1_alignment": dict(metrics["observed_head_1_alignment"]),
        "worst_order_alignment": dict(metrics["worst_order_alignment"]),
        "scenario_rows": list(metrics["scenario_rows"]),
        "per_unit": per_unit,
        "directions_float64_sha256": _array_float64_sha256(directions),
    }
    if "predicted_head_0_rows" in result:
        public["predicted_head_0_rows_float64_sha256"] = _array_float64_sha256(
            result["predicted_head_0_rows"]
        )
        public["predicted_head_1_rows_float64_sha256"] = _array_float64_sha256(
            result["predicted_head_1_rows"]
        )
    if "folds" in result:
        public["folds"] = list(result["folds"])
    return public


def _unavailable_metric(label: str, error: Exception) -> dict[str, Any]:
    return {
        "method": label,
        "available": False,
        "failure_type": type(error).__name__,
        "failure": str(error),
    }


def _apply_gates(
    transport: Mapping[str, Any],
    identity: Mapping[str, Any],
    compute: Mapping[str, Any],
) -> tuple[dict[str, bool], bool]:
    available = bool(transport.get("available"))
    identity_available = bool(identity.get("available"))
    transport_count = int(transport.get("both_order_positive_count", -1))
    identity_count = int(identity.get("both_order_positive_count", -1))
    median = (
        float(transport["worst_order_alignment"]["median"])
        if available
        else -math.inf
    )
    gates = {
        "transport_heads_compatible": available,
        "at_least_6_of_8_assignment_units_positive_under_both_orders": (
            available and transport_count >= 6
        ),
        "both_assignments_pass_in_at_least_3_of_4_scenarios": (
            available and int(transport.get("complete_scenario_count", -1)) >= 3
        ),
        "median_worst_order_cosine_strictly_greater_than_0_10": (
            available and median > 0.10
        ),
        "at_least_two_more_assignment_units_than_identity": (
            available
            and identity_available
            and transport_count - identity_count >= 2
        ),
        "exactly_16_unique_forwards": (
            int(compute.get("forward_evaluations", -1)) == 16
            and int(compute.get("unique_forward_work_ids", -1)) == 16
        ),
        "exactly_16_unique_backwards": (
            int(compute.get("backward_evaluations", -1)) == 16
            and int(compute.get("unique_backward_work_ids", -1)) == 16
        ),
        "hash_and_anchor_audits_pass": True,
    }
    return gates, all(gates.values())


def _load_cached_result() -> dict[str, Any] | None:
    if not RESULT_PATH.is_file():
        return None
    result = _load_json(RESULT_PATH)
    _validate_embedded_sha256(result, "result_sha256")
    if result.get("schema_version") != "sp_lense.suffix_transport_feasibility_result.v1":
        raise RuntimeError("ST-FG cached result schema differs")
    if result.get("lock_sha256") != file_sha256(LOCK_PATH):
        raise RuntimeError("ST-FG cached result belongs to a different lock")
    if result.get("choice_capture_sha256") != file_sha256(CHOICE_CAPTURE_PATH):
        raise RuntimeError("ST-FG cached result belongs to a different choice capture")
    if result.get("choice_capture_manifest_sha256") != file_sha256(
        CHOICE_CAPTURE_MANIFEST_PATH
    ):
        raise RuntimeError("ST-FG cached result belongs to a different capture manifest")
    return result


def run_analyze() -> dict[str, Any]:
    _load_lock()
    cached = _load_cached_result()
    if cached is not None:
        _write_report(cached)
        return cached
    import torch

    capture = _load_choice_capture(torch)
    source = capture["semantic_source_rows"].double().numpy()
    observed_0 = capture["choice_head_0_rows"].double().numpy()
    observed_1 = capture["choice_head_1_rows"].double().numpy()
    source_manifest = capture["source_unit_manifest"]
    scenario_ids = [str(record["scenario_id"]) for record in source_manifest]
    assignments = [int(record["assignment"]) for record in source_manifest]

    try:
        transported_internal = leave_one_scenario_out_transport(
            source,
            observed_0,
            observed_1,
            scenario_ids,
            ridge_multiplier=RIDGE_MULTIPLIER,
            minimum_head_cosine=MINIMUM_HEAD_COSINE,
            positive_alignment_threshold=0.0,
        )
        transported = _public_metric(
            "suffix_transport",
            transported_internal,
            scenario_ids=scenario_ids,
            assignments=assignments,
        )
    except (TypeError, ValueError, np.linalg.LinAlgError) as error:
        transported = _unavailable_metric("suffix_transport", error)

    try:
        identity_internal = transport_metric_summary(
            source,
            source,
            observed_0,
            observed_1,
            scenario_ids=scenario_ids,
            minimum_head_cosine=MINIMUM_HEAD_COSINE,
            positive_alignment_threshold=0.0,
        )
        identity = _public_metric(
            "identity_no_transport_fcags",
            identity_internal,
            scenario_ids=scenario_ids,
            assignments=assignments,
        )
    except (TypeError, ValueError, np.linalg.LinAlgError) as error:
        identity = _unavailable_metric("identity_no_transport_fcags", error)

    try:
        mean_0, mean_1, mean_folds = _static_mean_predictions(
            observed_0, observed_1, scenario_ids
        )
        static_internal = transport_metric_summary(
            mean_0,
            mean_1,
            observed_0,
            observed_1,
            scenario_ids=scenario_ids,
            minimum_head_cosine=MINIMUM_HEAD_COSINE,
            positive_alignment_threshold=0.0,
        )
        static = _public_metric(
            "training_fold_mean_choice_gradient_bisector",
            static_internal,
            scenario_ids=scenario_ids,
            assignments=assignments,
        )
        static["predicted_head_0_rows_float64_sha256"] = _array_float64_sha256(mean_0)
        static["predicted_head_1_rows_float64_sha256"] = _array_float64_sha256(mean_1)
        static["folds"] = mean_folds
    except (TypeError, ValueError, np.linalg.LinAlgError) as error:
        static = _unavailable_metric("training_fold_mean_choice_gradient_bisector", error)

    gates, passes = _apply_gates(transported, identity, capture["compute"])
    result = {
        "schema_version": "sp_lense.suffix_transport_feasibility_result.v1",
        "development_only": True,
        "opened_development_evidence_only": True,
        "status": "passed_geometric_feasibility" if passes else "failed",
        "passes_all_locked_gates": passes,
        "lock_sha256": file_sha256(LOCK_PATH),
        "lock_identity_sha256": _load_lock()["lock_identity_sha256"],
        "choice_capture_sha256": file_sha256(CHOICE_CAPTURE_PATH),
        "choice_capture_manifest_sha256": file_sha256(CHOICE_CAPTURE_MANIFEST_PATH),
        "cached_semantic_capture_sha256": file_sha256(FCAGS_CAPTURE_PATH),
        "layer": LAYER,
        "ridge_multiplier": RIDGE_MULTIPLIER,
        "minimum_predicted_head_cosine": MINIMUM_HEAD_COSINE,
        "methods": {
            "suffix_transport": transported,
            "identity_no_transport_fcags": identity,
            "training_fold_mean_choice_gradient_bisector": static,
        },
        "gates": gates,
        "compute": dict(capture["compute"]),
        "generated_tokens": 0,
        "external_api_calls": 0,
        "external_model_judges": 0,
        "paid_cost_usd": 0,
        "decision_steering_run": False,
        "fcags_pilot_outcomes_read": False,
        "claim_boundary": (
            "This opened-data smoke measures first-order geometric suffix transport only. "
            "Even a pass does not demonstrate an actual decision change, a prospective "
            "effect, a natural self-preservation mechanism, or publication-level novelty."
        ),
    }
    result["result_sha256"] = canonical_sha256(result)
    _write_json(RESULT_PATH, result)
    _write_report(result)
    return result


def _metric_table_cell(method: Mapping[str, Any], field: str, denominator: int) -> str:
    if not method.get("available"):
        return "unavailable"
    return f"{int(method[field])}/{denominator}"


def _write_report(result: Mapping[str, Any]) -> str:
    methods = result["methods"]
    lines = [
        "# Suffix-Transported Factorial Gradient: Smoke A",
        "",
        f"Status: **{result['status']}**.",
        "",
        "| Method | Both-order-positive units | Complete scenarios | Median worst-order cosine |",
        "|---|---:|---:|---:|",
    ]
    for key in (
        "suffix_transport",
        "identity_no_transport_fcags",
        "training_fold_mean_choice_gradient_bisector",
    ):
        method = methods[key]
        median = (
            f"{float(method['worst_order_alignment']['median']):.6f}"
            if method.get("available")
            else "unavailable"
        )
        lines.append(
            f"| {key} | {_metric_table_cell(method, 'both_order_positive_count', 8)} | "
            f"{_metric_table_cell(method, 'complete_scenario_count', 4)} | {median} |"
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
                "No tokens were generated, no decision-steering dose was applied, and no "
                "FCAGS pilot outcome was read."
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
        description="Development-only suffix-transport geometric feasibility smoke"
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
