from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sp_lense.counterfactual_protected_natural_gradient import (
    FISHER_RIDGE_MULTIPLIER_GRID,
    PREDICTED_COARSENED_NEXT_TOKEN_KL_BUDGET_GRID,
    RESIDUAL_RELATIVE_L2_CAP_GRID,
    CounterfactualConstructionIneligible,
    build_counterfactual_protected_natural_gradient,
    certify_applied_float32_perturbation,
    global_unrelated_null_projection,
    preregistered_candidate_grid,
    scale_to_predicted_coarsened_next_token_kl_budget,
)
from sp_lense.gradient_specificity_v2 import render_completion_form
from sp_lense.semantic_completion_gradient import capture_semantic_completion_gradient

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_ROOT = ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import gradient_specificity_trust_region_development as trust
import gradient_specificity_v3_development as base

LOCK_PATH = ROOT / "configs" / "counterfactual_protected_natural_gradient_development_lock.json"
PROTOCOL_PATH = ROOT / "docs" / "COUNTERFACTUAL_PROTECTED_NATURAL_GRADIENT_PROTOCOL.md"
MATH_PATH = ROOT / "src" / "sp_lense" / "counterfactual_protected_natural_gradient.py"
SEMANTIC_CAPTURE_PATH = ROOT / "src" / "sp_lense" / "semantic_completion_gradient.py"
RENDERER_PATH = ROOT / "src" / "sp_lense" / "gradient_specificity_v2.py"

ARTIFACT_ROOT = (
    ROOT / "artifacts" / "counterfactual_protected_natural_gradient_development" / "qwen35_08b"
)
RESULT_ROOT = (
    ROOT / "results" / "counterfactual_protected_natural_gradient_development" / "qwen35_08b"
)
PREFLIGHT_PATH = ARTIFACT_ROOT / "preflight.json"
CAPTURE_PATH = ARTIFACT_ROOT / "completion_capture.pt"
CAPTURE_MANIFEST_PATH = ARTIFACT_ROOT / "completion_capture_manifest.json"
CONSTRUCTION_PATH = ARTIFACT_ROOT / "construction_bank.pt"
CONSTRUCTION_MANIFEST_PATH = ARTIFACT_ROOT / "construction_bank_manifest.json"
ROWS_PATH = RESULT_ROOT / "calibration_rows.jsonl"
SUMMARY_PATH = RESULT_ROOT / "development_summary.json"
REPORT_PATH = RESULT_ROOT / "DEVELOPMENT_REPORT.md"
CAPTURE_LEDGER_PATH = ARTIFACT_ROOT / "capture_compute_ledger.json"
CALIBRATION_LEDGER_PATH = RESULT_ROOT / "calibration_compute_ledger.json"
STAGE_ONE_CHECKPOINT_ROOT = RESULT_ROOT / "stage_one_checkpoints"
STAGE_ONE_COMPLETE_PATH = RESULT_ROOT / "stage_one_complete.json"
STAGE_TWO_CHECKPOINT_ROOT = RESULT_ROOT / "stage_two_checkpoints"

LOCK_SCHEMA = "sp_lense.cpng_development_lock.v1"
PREFLIGHT_SCHEMA = "sp_lense.cpng_preflight.v1"
CAPTURE_SCHEMA = "sp_lense.cpng_completion_capture.v1"
CONSTRUCTION_SCHEMA = "sp_lense.cpng_construction_bank.v1"
ROW_SCHEMA = "sp_lense.cpng_calibration_row.v1"
SUMMARY_SCHEMA = "sp_lense.cpng_development_summary.v1"
LEDGER_SCHEMA = "sp_lense.cpng_compute_ledger.v1"
STAGE_ONE_CASE_SCHEMA = "sp_lense.cpng_stage_one_case.v1"
STAGE_ONE_COMPLETE_SCHEMA = "sp_lense.cpng_stage_one_complete.v1"
STAGE_TWO_CASE_SCHEMA = "sp_lense.cpng_stage_two_case.v1"

EXPECTED_PROTECTED_LIMITS = dict(trust.EXPECTED_PROTECTED_LIMITS)
EXPECTED_TARGET_MARGIN = 0.01
EXPECTED_MATCHED_OTHER_MARGIN = 0.0
EXPECTED_MAXIMUM_CALIBRATION_FORWARD_EVALUATIONS = 3648
EXPECTED_MAXIMUM_CAPTURE_FORWARD_EVALUATIONS = 48
EXPECTED_MAXIMUM_CAPTURE_BACKWARD_EVALUATIONS = 32
EXPECTED_MAXIMUM_TOTAL_FORWARD_EVALUATIONS = 3696
EXPECTED_MAXIMUM_TOTAL_BACKWARD_EVALUATIONS = 32
EXPECTED_EXTERNAL_MODEL_JUDGES = 0
EXPECTED_EXTERNAL_API_CALLS = 0
EXPECTED_VOCABULARY_SIZE = 248320
EXPECTED_RUNTIME = {
    "python": "3.12.10",
    "torch": "2.13.0+cpu",
    "transformers": "5.15.1",
    "torch_intraop_threads": 12,
    "torch_interop_threads": 12,
}
EXPECTED_MINIMUM_SEPARATION_HEURISTIC = {
    "unit_roundoff_rule": "u32 = torch.float32.eps / 2",
    "gamma_rule": "gamma_d = d * u32 / (1 - d * u32)",
    "strict_pass_rule": (
        "norm(projected_self_minus_other) / "
        "(norm(projected_self) + norm(projected_other)) > gamma_d"
    ),
    "is_vjp_or_backprop_error_bound": False,
    "interpretation": "locked conservative minimum-separation screen only",
}
TIE_STATISTIC_FORMULA = "self_half_span_minus_abs_matched_other_half_span"
CANDIDATE_LOCAL_NUMERICAL_FAILURE_MESSAGES = (
    "CPNG application diagnostics are non-finite",
    "CPNG changed logits are non-finite",
    "CPNG changed-to-baseline KL is invalid",
)
ALLOWLISTED_CONSTRUCTION_FAILURE_MESSAGES = (
    "projected counterfactual contrast is numerically zero",
    "projected counterfactual contrast failed the minimum-separation heuristic",
    "projected protected metric has no positive scale",
    "natural direction has no certifiable protected-metric energy",
)
EXPECTED_INTEGRITY_AND_RESTART = {
    "artifact_pair_policy": "both_absent_or_both_present_and_strictly_valid",
    "ledger": "atomic_hash_chained_pre_operation_reservations",
    "capture_interruption": "charged_and_failed_closed_no_replay",
    "stage_one_resume": "strict_contiguous_completed_case_checkpoints_only",
    "orphan_reservation": "charged_and_failed_closed_no_replay",
    "stage_one_complete_before_stage_two": True,
    "stage_two_interruption": "failed_closed_no_resume_shared_baselines_not_persisted",
    "completed_result_reuse": ("strict_recompute_rows_audits_selection_finalization_and_compute"),
    "malicious_rewrite_of_all_local_files": "outside_integrity_threat_model",
}
EXPECTED_CANDIDATE_LOCAL_FAILURE_TAXONOMY = {
    "construction": {
        "type": "CounterfactualConstructionIneligible",
        "allowlisted_messages": list(ALLOWLISTED_CONSTRUCTION_FAILURE_MESSAGES),
    },
    "evaluation": {
        "type": "CandidateLocalNumericalFailure",
        "allowlisted_messages": list(CANDIDATE_LOCAL_NUMERICAL_FAILURE_MESSAGES),
    },
    "unknown_integrity_runtime_oom_hook_or_token_errors": "abort",
}


class CandidateLocalNumericalFailure(RuntimeError):
    """An explicitly classified candidate-local numerical evaluation failure."""


def canonical_sha256(value: Any) -> str:
    return base.canonical_sha256(value)


def file_sha256(path: Path) -> str:
    return base.file_sha256(path)


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _replace_with_permission_retry(source: Path, destination: Path) -> None:
    """Retry only the demonstrated transient Windows replacement failure."""

    for delay_seconds in (0.01, 0.025, 0.05, None):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if delay_seconds is None:
                raise
            time.sleep(delay_seconds)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    _replace_with_permission_retry(temporary, path)


def _immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    rendered = json.dumps(dict(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"immutable JSON artifact differs: {_relative(path)}")
        return
    _atomic_text(path, rendered)


def _immutable_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rendered = "".join(
        json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n" for row in rows
    )
    if path.is_file():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"immutable JSONL artifact differs: {_relative(path)}")
        return
    _atomic_text(path, rendered)


def _hashed_payload(value: Mapping[str, Any], *, hash_field: str) -> dict[str, Any]:
    output = dict(value)
    output[hash_field] = canonical_sha256(output)
    return output


def _verify_internal_hash(value: Mapping[str, Any], *, hash_field: str) -> None:
    stored = value.get(hash_field)
    if not isinstance(stored, str) or stored != canonical_sha256(
        {key: item for key, item in value.items() if key != hash_field}
    ):
        raise RuntimeError(f"{hash_field} failed its internal hash")


class PersistentComputeLedger:
    """Persist a hash-chained reservation event before each model operation."""

    def __init__(
        self,
        *,
        path: Path,
        phase: str,
        study_identity_sha256: str,
        maximum_forwards: int,
        maximum_backwards: int,
        prior_phase_ledger_sha256: str | None = None,
    ) -> None:
        self.path = path
        self.phase = phase
        self.study_identity_sha256 = study_identity_sha256
        self.maximum_forwards = maximum_forwards
        self.maximum_backwards = maximum_backwards
        self.prior_phase_ledger_sha256 = prior_phase_ledger_sha256
        if path.exists():
            if not path.is_file():
                raise RuntimeError("compute ledger path is not a file")
            payload = _load_json(path)
            _verify_internal_hash(payload, hash_field="ledger_sha256")
            expected = {
                "schema_version": LEDGER_SCHEMA,
                "phase": phase,
                "study_identity_sha256": study_identity_sha256,
                "maximum_forward_evaluations": maximum_forwards,
                "maximum_backward_evaluations": maximum_backwards,
                "prior_phase_ledger_sha256": prior_phase_ledger_sha256,
            }
            if (
                set(payload)
                != {
                    *expected,
                    "forward_evaluations",
                    "backward_evaluations",
                    "events",
                    "ledger_sha256",
                }
                or any(payload.get(key) != value for key, value in expected.items())
                or any(
                    isinstance(payload.get(field), bool) or not isinstance(payload.get(field), int)
                    for field in (
                        "maximum_forward_evaluations",
                        "maximum_backward_evaluations",
                        "forward_evaluations",
                        "backward_evaluations",
                    )
                )
                or not isinstance(payload.get("events"), list)
            ):
                raise RuntimeError("compute ledger identity differs")
            self.forward_evaluations = payload["forward_evaluations"]
            self.backward_evaluations = payload["backward_evaluations"]
            self.events = list(payload["events"])
            prior_hash = None
            forward_count = 0
            backward_count = 0
            work_ids: set[str] = set()
            for sequence, event in enumerate(self.events):
                if (
                    not isinstance(event, Mapping)
                    or set(event)
                    != {
                        "sequence",
                        "phase",
                        "study_identity_sha256",
                        "work_id",
                        "operation",
                        "prior_event_sha256",
                        "cumulative_forward_evaluations",
                        "cumulative_backward_evaluations",
                        "event_sha256",
                    }
                    or event.get("sequence") != sequence
                    or event.get("phase") != phase
                    or event.get("study_identity_sha256") != study_identity_sha256
                    or event.get("prior_event_sha256") != prior_hash
                ):
                    raise RuntimeError("compute ledger event sequence is not contiguous")
                _verify_internal_hash(event, hash_field="event_sha256")
                operation = event.get("operation")
                work_id = event.get("work_id")
                if not isinstance(work_id, str) or not work_id or work_id in work_ids:
                    raise RuntimeError("compute ledger work IDs are invalid or duplicated")
                work_ids.add(work_id)
                if operation == "forward":
                    forward_count += 1
                elif operation == "backward":
                    backward_count += 1
                else:
                    raise RuntimeError("compute ledger event has an unknown operation")
                if (
                    event.get("cumulative_forward_evaluations") != forward_count
                    or event.get("cumulative_backward_evaluations") != backward_count
                ):
                    raise RuntimeError("compute ledger event counters are not monotonic")
                prior_hash = event["event_sha256"]
            if (
                forward_count != self.forward_evaluations
                or backward_count != self.backward_evaluations
            ):
                raise RuntimeError("compute ledger totals differ from its event chain")
        else:
            self.forward_evaluations = 0
            self.backward_evaluations = 0
            self.events = []
            self._persist()
        if not 0 <= self.forward_evaluations <= maximum_forwards:
            raise RuntimeError("compute ledger forward count exceeds its ceiling")
        if not 0 <= self.backward_evaluations <= maximum_backwards:
            raise RuntimeError("compute ledger backward count exceeds its ceiling")

    def _persist(self) -> None:
        payload = _hashed_payload(
            {
                "schema_version": LEDGER_SCHEMA,
                "phase": self.phase,
                "study_identity_sha256": self.study_identity_sha256,
                "maximum_forward_evaluations": self.maximum_forwards,
                "maximum_backward_evaluations": self.maximum_backwards,
                "prior_phase_ledger_sha256": self.prior_phase_ledger_sha256,
                "forward_evaluations": self.forward_evaluations,
                "backward_evaluations": self.backward_evaluations,
                "events": self.events,
            },
            hash_field="ledger_sha256",
        )
        _atomic_text(self.path, json.dumps(payload, indent=2, allow_nan=False) + "\n")

    def reserve(self, *, work_id: str, forward: int = 0, backward: int = 0) -> None:
        if not isinstance(work_id, str) or not work_id:
            raise ValueError("ledger work_id must be non-empty")
        if any(event.get("work_id") == work_id for event in self.events):
            raise RuntimeError("compute ledger refuses a duplicate work ID")
        if forward not in {0, 1} or backward not in {0, 1} or forward + backward != 1:
            raise ValueError("ledger reservations must contain exactly one model operation")
        if self.forward_evaluations + forward > self.maximum_forwards:
            raise trust.ComputeBudgetExhausted("cumulative forward ceiling exhausted")
        if self.backward_evaluations + backward > self.maximum_backwards:
            raise trust.ComputeBudgetExhausted("cumulative backward ceiling exhausted")
        self.forward_evaluations += forward
        self.backward_evaluations += backward
        event = _hashed_payload(
            {
                "sequence": len(self.events),
                "phase": self.phase,
                "study_identity_sha256": self.study_identity_sha256,
                "work_id": work_id,
                "operation": "forward" if forward else "backward",
                "prior_event_sha256": (self.events[-1]["event_sha256"] if self.events else None),
                "cumulative_forward_evaluations": self.forward_evaluations,
                "cumulative_backward_evaluations": self.backward_evaluations,
            },
            hash_field="event_sha256",
        )
        self.events.append(event)
        self._persist()

    def sync_budget(self, snapshot: Mapping[str, int], *, work_id: str) -> None:
        wanted_forward = int(snapshot["forward_evaluations"])
        wanted_backward = int(snapshot["backward_evaluations"])
        delta_forward = wanted_forward - self.forward_evaluations
        delta_backward = wanted_backward - self.backward_evaluations
        if (delta_forward, delta_backward) not in {(1, 0), (0, 1)}:
            raise RuntimeError("model budget and persistent ledger lost monotonic agreement")
        self.reserve(work_id=work_id, forward=delta_forward, backward=delta_backward)

    def snapshot(self) -> dict[str, int]:
        return {
            "forward_evaluations": self.forward_evaluations,
            "backward_evaluations": self.backward_evaluations,
            "event_count": len(self.events),
            "head_event_sha256": self.events[-1]["event_sha256"] if self.events else None,
        }


def _ledger_prefix_evidence(
    ledger: PersistentComputeLedger, event_count: int
) -> tuple[dict[str, Any], str]:
    if isinstance(event_count, bool) or not isinstance(event_count, int):
        raise TypeError("checkpoint ledger event count is invalid")
    if not 0 <= event_count <= len(ledger.events):
        raise RuntimeError("checkpoint ledger event count is outside the event chain")
    events = ledger.events[:event_count]
    forwards = sum(event["operation"] == "forward" for event in events)
    backwards = sum(event["operation"] == "backward" for event in events)
    snapshot = {
        "forward_evaluations": forwards,
        "backward_evaluations": backwards,
        "event_count": event_count,
        "head_event_sha256": events[-1]["event_sha256"] if events else None,
    }
    payload = _hashed_payload(
        {
            "schema_version": LEDGER_SCHEMA,
            "phase": ledger.phase,
            "study_identity_sha256": ledger.study_identity_sha256,
            "maximum_forward_evaluations": ledger.maximum_forwards,
            "maximum_backward_evaluations": ledger.maximum_backwards,
            "prior_phase_ledger_sha256": ledger.prior_phase_ledger_sha256,
            "forward_evaluations": forwards,
            "backward_evaluations": backwards,
            "events": events,
        },
        hash_field="ledger_sha256",
    )
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    return snapshot, _text_sha256(rendered)


def _validate_checkpoint_ledger_prefix(
    ledger: PersistentComputeLedger, checkpoint: Mapping[str, Any]
) -> None:
    snapshot = checkpoint.get("ledger")
    if not isinstance(snapshot, Mapping) or set(snapshot) != {
        "forward_evaluations",
        "backward_evaluations",
        "event_count",
        "head_event_sha256",
    }:
        raise RuntimeError("checkpoint ledger snapshot schema differs")
    expected_snapshot, expected_file_sha256 = _ledger_prefix_evidence(
        ledger, snapshot.get("event_count")
    )
    if (
        dict(snapshot) != expected_snapshot
        or checkpoint.get("ledger_file_sha256") != expected_file_sha256
    ):
        raise RuntimeError("checkpoint is not bound to the claimed compute-ledger prefix")


def _validate_capture_ledger_work_ids(
    ledger: PersistentComputeLedger, specifications: Sequence[Mapping[str, Any]]
) -> None:
    expected = []
    for specification in specifications:
        form_id = str(specification["form_id"])
        expected.extend(
            (
                (f"{form_id}:prompt_only_forward", "forward"),
                (f"{form_id}:preserve_forward", "forward"),
                (f"{form_id}:preserve_backward", "backward"),
                (f"{form_id}:comply_forward", "forward"),
                (f"{form_id}:comply_backward", "backward"),
            )
        )
    observed = [(str(event["work_id"]), str(event["operation"])) for event in ledger.events]
    if observed != expected:
        raise RuntimeError("capture compute-ledger work mapping differs")


def _validate_calibration_ledger_work_ids(
    ledger: PersistentComputeLedger,
    *,
    rows: Sequence[Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
    provisional: Mapping[str, Any] | None,
    candidate_map: Mapping[tuple[str, int, int], Mapping[str, Any]],
    frozen: Mapping[str, Any],
) -> None:
    assignments = _case_assignments(frozen)
    rows_by_attempt = {
        (str(row["case_id"]), int(row["assignment"]), int(row["grid_index"])): row for row in rows
    }
    expected_bases = []
    allowed_counts: dict[str, tuple[int, int]] = {}
    for case_id, assignment in assignments:
        seen = set()
        for grid_index in range(48):
            candidate = candidate_map[(case_id, assignment, grid_index)]
            if candidate["construction_status"] != "constructed":
                continue
            deduplication_key = str(candidate["deduplication_key"])
            if deduplication_key in seen:
                continue
            seen.add(deduplication_key)
            equivalent_rows = [
                rows_by_attempt[(case_id, assignment, other_grid_index)]
                for other_grid_index in range(48)
                if candidate_map[(case_id, assignment, other_grid_index)].get("deduplication_key")
                == deduplication_key
            ]
            reference = equivalent_rows[0]
            for equivalent in equivalent_rows[1:]:
                for field in (
                    "evaluation_status",
                    "evaluation_sha256",
                    "terminal_candidate",
                    "success",
                    "terminal_gate",
                    "matched_other_passed",
                    "matched_other_mean_kl",
                    "null_passed",
                    "self_minus_matched_other_effect",
                    "evaluation",
                ):
                    if equivalent.get(field) != reference.get(field):
                        raise RuntimeError("deduplicated CPNG rows have different evidence")
            base_id = f"stage_one:{case_id}:assignment={assignment}:{deduplication_key}"
            expected_bases.append(base_id)
            status = str(reference["evaluation_status"])
            if status == "evaluated":
                allowed_counts[base_id] = (8, 12)
            elif status == "failed_closed":
                allowed_counts[base_id] = (1, 12)
            else:
                raise RuntimeError("constructed CPNG candidate lacks an evaluation")
    if isinstance(provisional, Mapping):
        if len(audits) != len(assignments):
            raise RuntimeError("CPNG Stage-two ledger lacks all audits")
        for audit, (case_id, assignment) in zip(audits, assignments, strict=True):
            base_id = f"stage_two:{case_id}:assignment={assignment}"
            expected_bases.append(base_id)
            allowed_counts[base_id] = (64, 96) if audit["evaluated"] is True else (1, 96)
    elif audits:
        raise RuntimeError("CPNG ledger has audits without a provisional candidate")

    observed_bases = []
    counts: dict[str, int] = {}
    prior_base = None
    for sequence, event in enumerate(ledger.events):
        if event["operation"] != "forward":
            raise RuntimeError("calibration ledger contains a non-forward operation")
        work_id = str(event["work_id"])
        suffix = f":event={sequence}"
        if not work_id.endswith(suffix):
            raise RuntimeError("calibration ledger work ID is not bound to its event")
        base_id = work_id[: -len(suffix)]
        if base_id != prior_base:
            observed_bases.append(base_id)
            prior_base = base_id
        counts[base_id] = counts.get(base_id, 0) + 1
    if observed_bases != expected_bases or set(counts) != set(expected_bases):
        raise RuntimeError("calibration compute-ledger work mapping differs")
    for base_id, (minimum, maximum) in allowed_counts.items():
        if not minimum <= counts[base_id] <= maximum:
            raise RuntimeError("calibration compute-ledger work count differs")


def _save_tensor_artifact(
    torch: Any,
    *,
    tensor_path: Path,
    manifest_path: Path,
    payload: Mapping[str, Any],
    public_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if tensor_path.exists() or manifest_path.exists():
        if not tensor_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("immutable tensor artifact is only partially present")
        existing = _load_json(manifest_path)
        if existing.get("artifact_identity_sha256") != public_manifest.get(
            "artifact_identity_sha256"
        ) or existing.get("tensor_file_sha256") != file_sha256(tensor_path):
            raise RuntimeError("immutable tensor artifact identity differs")
        return existing
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tensor_path.with_suffix(tensor_path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    _replace_with_permission_retry(temporary, tensor_path)
    manifest = {
        **dict(public_manifest),
        "tensor_path": _relative(tensor_path),
        "tensor_file_sha256": file_sha256(tensor_path),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    _immutable_json(manifest_path, manifest)
    return manifest


def _expected_candidate_grid_lock() -> dict[str, Any]:
    expanded = list(preregistered_candidate_grid())
    return {
        "fisher_ridge_multipliers": list(FISHER_RIDGE_MULTIPLIER_GRID),
        "predicted_coarsened_next_token_kl_budgets": list(
            PREDICTED_COARSENED_NEXT_TOKEN_KL_BUDGET_GRID
        ),
        "residual_relative_l2_caps": list(RESIDUAL_RELATIVE_L2_CAP_GRID),
        "expansion_order": [
            "fisher_ridge_multiplier",
            "predicted_coarsened_next_token_kl_budget",
            "residual_relative_l2_cap",
        ],
        "candidate_count": len(expanded),
        "expanded_grid_sha256": canonical_sha256(expanded),
    }


def _load_lock() -> dict[str, Any]:
    if not LOCK_PATH.is_file():
        raise RuntimeError("CPNG requires its exact config lock before any execution")
    lock = _load_json(LOCK_PATH)
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise ValueError("CPNG lock has the wrong schema")
    if lock.get("status") != "locked_before_cpng_development_execution":
        raise ValueError("CPNG lock was not frozen before execution")
    if lock.get("development_only") is not True or lock.get("model") != base.EXPECTED_MODEL:
        raise ValueError("CPNG lock is not bound to the frozen development model")
    if lock.get("model_vocabulary_size") != EXPECTED_VOCABULARY_SIZE:
        raise ValueError("CPNG lock has the wrong model vocabulary size")
    if lock.get("runtime") != EXPECTED_RUNTIME:
        raise ValueError("CPNG lock is not bound to the frozen research runtime")
    if lock.get("candidate_grid") != _expected_candidate_grid_lock():
        raise ValueError("CPNG candidate grid differs from the preregistered math module")
    if lock.get("minimum_separation_heuristic") != EXPECTED_MINIMUM_SEPARATION_HEURISTIC:
        raise ValueError("CPNG minimum-separation heuristic differs from the lock")
    if lock.get("protected_limits") != EXPECTED_PROTECTED_LIMITS:
        raise ValueError("CPNG protected limits differ from the trust-region limits")
    if float(lock.get("target_margin_logit", math.nan)) != EXPECTED_TARGET_MARGIN:
        raise ValueError("CPNG target margin differs from the lock")
    if float(lock.get("matched_other_margin_logit", math.nan)) != EXPECTED_MATCHED_OTHER_MARGIN:
        raise ValueError("CPNG matched-other margin differs from the lock")
    compute = lock.get("compute_ceiling")
    expected_compute = {
        "calibration": {
            "maximum_forward_evaluations": (EXPECTED_MAXIMUM_CALIBRATION_FORWARD_EVALUATIONS),
            "maximum_backward_evaluations": 0,
        },
        "capture": {
            "maximum_forward_evaluations": EXPECTED_MAXIMUM_CAPTURE_FORWARD_EVALUATIONS,
            "maximum_backward_evaluations": EXPECTED_MAXIMUM_CAPTURE_BACKWARD_EVALUATIONS,
        },
        "total_experiment": {
            "maximum_forward_evaluations": EXPECTED_MAXIMUM_TOTAL_FORWARD_EVALUATIONS,
            "maximum_backward_evaluations": EXPECTED_MAXIMUM_TOTAL_BACKWARD_EVALUATIONS,
        },
        "external_model_judges": EXPECTED_EXTERNAL_MODEL_JUDGES,
        "external_api_calls": EXPECTED_EXTERNAL_API_CALLS,
    }
    if not isinstance(compute, Mapping) or compute != expected_compute:
        raise ValueError("CPNG compute ceiling differs from the exact development contract")
    if lock.get("completion_capture_dtype") != "torch.float32":
        raise ValueError("CPNG completion capture dtype is not locked to torch.float32")
    if lock.get("tie_statistic_formula") != TIE_STATISTIC_FORMULA:
        raise ValueError("CPNG tie statistic formula differs from the exact contract")
    if lock.get("integrity_and_restart") != EXPECTED_INTEGRITY_AND_RESTART:
        raise ValueError("CPNG integrity and restart contract differs from the exact lock")
    if lock.get("candidate_local_failure_taxonomy") != EXPECTED_CANDIDATE_LOCAL_FAILURE_TAXONOMY:
        raise ValueError("CPNG candidate-local failure taxonomy differs from the exact lock")
    selection = lock.get("selection")
    if not isinstance(selection, Mapping):
        raise TypeError("CPNG selection contract must be a mapping")
    if selection.get("incomplete_or_evaluation_failed_global_triple_ineligible") is not True:
        raise ValueError("CPNG selection does not fail closed on incomplete global triples")
    if selection.get("safety_and_efficacy_statuses_are_separate") is not True:
        raise ValueError("CPNG selection conflates safety with efficacy")
    sources = lock.get("source_files")
    if not isinstance(sources, Mapping) or not sources:
        raise TypeError("CPNG lock lacks source-file bindings")
    for name, binding in sources.items():
        if not isinstance(binding, Mapping):
            raise TypeError(f"source_files.{name} must be a mapping")
        path = ROOT / str(binding.get("path", ""))
        wanted = binding.get("sha256")
        if not path.is_file() or not isinstance(wanted, str) or file_sha256(path) != wanted:
            raise RuntimeError(f"CPNG source binding changed: {binding.get('path')}")
    stored = lock.get("lock_identity_sha256")
    recomputed = canonical_sha256(
        {key: value for key, value in lock.items() if key != "lock_identity_sha256"}
    )
    if stored != recomputed:
        raise RuntimeError("CPNG lock failed its internal identity hash")
    return lock


def _completion_specifications() -> list[dict[str, Any]]:
    manifest = base.load_development_manifest()
    source = base._load_json(base._source_path(manifest, "self_preservation_cases"))
    case_ids = list(map(str, manifest["stage_a_self_preservation_case_ids"]))
    cases = base._find_sp_cases(source, case_ids)
    output = []
    for case in cases:
        for assignment in (0, 1):
            for target in ("self", "other"):
                rendered = render_completion_form(case, assignment=assignment, target=target)
                form_id = f"completion:{case['id']}:assignment={assignment}:{target}"
                output.append(
                    {
                        "form_id": form_id,
                        "case_id": str(case["id"]),
                        "assignment": assignment,
                        "target": target,
                        "prompt": rendered["prompt"],
                        "preserve_completion": rendered["preserve_completion"],
                        "comply_completion": rendered["comply_completion"],
                        "prompt_sha256": _text_sha256(rendered["prompt"]),
                        "preserve_completion_sha256": _text_sha256(rendered["preserve_completion"]),
                        "comply_completion_sha256": _text_sha256(rendered["comply_completion"]),
                    }
                )
    if len(output) != 16 or len({item["form_id"] for item in output}) != 16:
        raise RuntimeError("CPNG requires exactly 16 unique authored-completion forms")
    return output


def _public_completion_manifest(
    specifications: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in specification.items()
            if key not in {"prompt", "preserve_completion", "comply_completion"}
        }
        for specification in specifications
    ]


def _study_identity(lock: Mapping[str, Any], frozen: Mapping[str, Any]) -> dict[str, Any]:
    identity = {
        "schema_version": "sp_lense.cpng_study_identity.v1",
        "development_only": True,
        "lock_sha256": file_sha256(LOCK_PATH),
        "lock_identity_sha256": str(lock["lock_identity_sha256"]),
        "model": lock["model"],
        "runtime": lock["runtime"],
        "candidate_grid_sha256": canonical_sha256(lock["candidate_grid"]),
        "global_unrelated_basis_sha256": base.v3.tensor_float64_sha256(
            frozen["global_nuisance_basis"]
        ),
        "global_unrelated_rank": int(frozen["global_nuisance_basis"].shape[0]),
        "frozen_source_hashes": dict(frozen["source_hashes"]),
        "completion_form_manifest_sha256": canonical_sha256(
            _public_completion_manifest(_completion_specifications())
        ),
        "target_contrast_objective": "mean_token_authored_completion_score_contrast",
        "target_contrast_uses_ab_labels": False,
        "target_contrast_uses_answer_order": False,
        "target_contrast_sign_fixed_by_authored_preserve_minus_comply_contrast": True,
        "target_contrast_sign_may_change_after_ab_evaluation": False,
        "protected_metric_uses_both_ab_order_renderings_symmetrically": True,
        "protected_metric_uses_ab_choice_prompt_tokens": True,
        "protected_metric_uses_ab_token_ids_and_coarsened_categories": True,
        "protected_metric_uses_answer_label_mapping": False,
        "protected_metric_uses_semantic_orientation": False,
        "protected_metric_uses_evaluation_outcomes": False,
        "protected_metric_kind": "prompt_group_balanced_topk_plus_aggregate_tail_next_token_pullback",
        "protected_metric_is_full_vocabulary_fisher": False,
        "actual_finite_protection_gate_uses_full_vocabulary_kl": True,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    return identity


def _actual_runtime(torch: Any) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": importlib.metadata.version("transformers"),
        "torch_intraop_threads": int(torch.get_num_threads()),
        "torch_interop_threads": int(torch.get_num_interop_threads()),
    }


def run_preflight(*, write: bool = True) -> dict[str, Any]:
    lock = _load_lock()
    import torch

    if _actual_runtime(torch) != lock["runtime"]:
        raise RuntimeError("CPNG active research runtime differs from the lock")
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(
        str(lock["model"]["id"]),
        revision=str(lock["model"]["revision"]),
        local_files_only=True,
        trust_remote_code=False,
    )
    if int(model_config.text_config.vocab_size) != EXPECTED_VOCABULARY_SIZE:
        raise RuntimeError("CPNG cached model vocabulary differs from the lock")
    frozen = trust._load_frozen_inputs(torch)
    specifications = _completion_specifications()
    case_assignments = sorted(
        {(str(item["case_id"]), int(item["assignment"])) for item in specifications}
    )
    output = {
        "schema_version": PREFLIGHT_SCHEMA,
        "development_only": True,
        "passes_preflight": True,
        "model_loads": 0,
        "model_forwards": 0,
        "external_model_judges": 0,
        "external_api_calls": 0,
        "case_assignment_count": len(case_assignments),
        "completion_form_count": len(specifications),
        "candidate_count_per_case_assignment": len(preregistered_candidate_grid()),
        "maximum_candidate_attempt_count": len(case_assignments)
        * len(preregistered_candidate_grid()),
        "capture_compute": {"forward_evaluations": 48, "backward_evaluations": 32},
        "calibration_compute_ceiling": {
            "stage_one_changed_forwards": 8 * 48 * 8,
            "stage_one_maximum_baseline_forwards": 32,
            "stage_two_changed_forwards": 8 * 64,
            "stage_two_maximum_baseline_forwards": 32,
            "maximum_calibration_forwards": (EXPECTED_MAXIMUM_CALIBRATION_FORWARD_EVALUATIONS),
            "selection_fallbacks_after_nuisance_audit": 0,
        },
        "capture_compute_ceiling": {
            "forward_evaluations": EXPECTED_MAXIMUM_CAPTURE_FORWARD_EVALUATIONS,
            "backward_evaluations": EXPECTED_MAXIMUM_CAPTURE_BACKWARD_EVALUATIONS,
        },
        "total_experiment_compute_ceiling": {
            "forward_evaluations": EXPECTED_MAXIMUM_TOTAL_FORWARD_EVALUATIONS,
            "backward_evaluations": EXPECTED_MAXIMUM_TOTAL_BACKWARD_EVALUATIONS,
            "strictly_below_4096_forwards": True,
        },
        "self_records_in_protected_metric": 0,
        "study_identity": _study_identity(lock, frozen),
    }
    output["preflight_sha256"] = canonical_sha256(output)
    if write:
        _immutable_json(PREFLIGHT_PATH, output)
    return output


def _capture_records(
    backend: Any,
    specifications: Sequence[Mapping[str, Any]],
    *,
    capture_fn: Callable[..., Any] = capture_semantic_completion_gradient,
    before_forward: Callable[[str], None] | None = None,
    before_backward: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records = []
    layer = int(base.EXPECTED_MODEL["layer_zero_based"])
    for specification in specifications:
        capture = capture_fn(
            backend,
            str(specification["prompt"]),
            str(specification["preserve_completion"]),
            str(specification["comply_completion"]),
            layer=layer,
            before_forward=(
                None
                if before_forward is None
                else lambda operation, form_id=str(specification["form_id"]): before_forward(
                    f"{form_id}:{operation}"
                )
            ),
            before_backward=(
                None
                if before_backward is None
                else lambda operation, form_id=str(specification["form_id"]): before_backward(
                    f"{form_id}:{operation}"
                )
            ),
        )
        if (
            capture.effective_gradient.dtype != backend.torch.float32
            or capture.prompt_residual.dtype != backend.torch.float32
        ):
            raise RuntimeError("completion capture is not in the locked float32 dtype")
        gradient = capture.effective_gradient.detach().cpu().contiguous()
        residual = capture.prompt_residual.detach().cpu().contiguous()
        if gradient.shape != residual.shape or gradient.ndim != 1:
            raise RuntimeError("completion capture returned incompatible residual coordinates")
        records.append(
            {
                "form_id": str(specification["form_id"]),
                "case_id": str(specification["case_id"]),
                "assignment": int(specification["assignment"]),
                "target": str(specification["target"]),
                "prompt_sha256": str(specification["prompt_sha256"]),
                "preserve_completion_sha256": str(specification["preserve_completion_sha256"]),
                "comply_completion_sha256": str(specification["comply_completion_sha256"]),
                "effective_gradient": gradient,
                "effective_gradient_sha256": base.v3.tensor_float32_sha256(gradient),
                "prompt_residual": residual,
                "prompt_residual_sha256": base.v3.tensor_float32_sha256(residual),
                "prompt_token_ids_sha256": str(capture.audit["prompt_token_ids_sha256"]),
                "preserve_content_token_ids_sha256": str(
                    capture.audit["preserve"]["content_token_ids_sha256"]
                ),
                "comply_content_token_ids_sha256": str(
                    capture.audit["comply"]["content_token_ids_sha256"]
                ),
                "audit": dict(capture.audit),
                "audit_sha256": canonical_sha256(capture.audit),
            }
        )
    return records, {
        "forward_evaluations": 3 * len(specifications),
        "backward_evaluations": 2 * len(specifications),
    }


def run_capture() -> dict[str, Any]:
    preflight = run_preflight()
    if CAPTURE_PATH.exists() != CAPTURE_MANIFEST_PATH.exists():
        raise RuntimeError("CPNG completion capture is only partially present")
    if CAPTURE_PATH.exists():
        import torch

        _load_capture(torch, preflight["study_identity"])
        return _load_json(CAPTURE_MANIFEST_PATH)
    ledger = PersistentComputeLedger(
        path=CAPTURE_LEDGER_PATH,
        phase="capture",
        study_identity_sha256=preflight["study_identity"]["identity_sha256"],
        maximum_forwards=EXPECTED_MAXIMUM_CAPTURE_FORWARD_EVALUATIONS,
        maximum_backwards=EXPECTED_MAXIMUM_CAPTURE_BACKWARD_EVALUATIONS,
    )
    if ledger.forward_evaluations or ledger.backward_evaluations:
        raise RuntimeError("interrupted capture is failed closed; recapture needs a new lock")
    backend = trust.load_backend()
    specifications = _completion_specifications()
    records, compute = _capture_records(
        backend,
        specifications,
        before_forward=lambda work_id: ledger.reserve(work_id=work_id, forward=1),
        before_backward=lambda work_id: ledger.reserve(work_id=work_id, backward=1),
    )
    if compute != {
        "forward_evaluations": ledger.forward_evaluations,
        "backward_evaluations": ledger.backward_evaluations,
    }:
        raise RuntimeError("capture compute report differs from persistent ledger")
    _validate_capture_ledger_work_ids(ledger, specifications)
    identity = {
        "schema_version": CAPTURE_SCHEMA,
        "development_only": True,
        "study_identity_sha256": preflight["study_identity"]["identity_sha256"],
        "form_manifest_sha256": canonical_sha256(_public_completion_manifest(specifications)),
        "form_count": len(specifications),
        "compute": compute,
        "compute_ledger": ledger.snapshot(),
        "compute_ledger_sha256": file_sha256(CAPTURE_LEDGER_PATH),
        "record_manifest": [
            {
                key: value
                for key, value in record.items()
                if key not in {"effective_gradient", "prompt_residual", "audit"}
            }
            for record in records
        ],
    }
    identity["artifact_identity_sha256"] = canonical_sha256(identity)
    payload = {**identity, "records": records}
    return _save_tensor_artifact(
        backend.torch,
        tensor_path=CAPTURE_PATH,
        manifest_path=CAPTURE_MANIFEST_PATH,
        payload=payload,
        public_manifest=identity,
    )


def _load_capture(torch: Any, study_identity: Mapping[str, Any]) -> dict[str, Any]:
    if CAPTURE_PATH.exists() != CAPTURE_MANIFEST_PATH.exists():
        raise RuntimeError("CPNG completion capture is only partially present")
    if not CAPTURE_PATH.is_file():
        raise RuntimeError("CPNG completion capture is incomplete")
    if not CAPTURE_LEDGER_PATH.is_file():
        raise RuntimeError("CPNG completed capture lacks its compute ledger")
    ledger = PersistentComputeLedger(
        path=CAPTURE_LEDGER_PATH,
        phase="capture",
        study_identity_sha256=str(study_identity["identity_sha256"]),
        maximum_forwards=EXPECTED_MAXIMUM_CAPTURE_FORWARD_EVALUATIONS,
        maximum_backwards=EXPECTED_MAXIMUM_CAPTURE_BACKWARD_EVALUATIONS,
    )
    if (
        ledger.forward_evaluations != EXPECTED_MAXIMUM_CAPTURE_FORWARD_EVALUATIONS
        or ledger.backward_evaluations != EXPECTED_MAXIMUM_CAPTURE_BACKWARD_EVALUATIONS
    ):
        raise RuntimeError("CPNG completed capture ledger has incomplete compute")
    manifest = _load_json(CAPTURE_MANIFEST_PATH)
    _verify_internal_hash(manifest, hash_field="manifest_sha256")
    tensor_hash = file_sha256(CAPTURE_PATH)
    if manifest.get("tensor_file_sha256") != tensor_hash:
        raise RuntimeError("CPNG completion capture file hash differs")
    payload = torch.load(CAPTURE_PATH, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("CPNG completion capture payload is not a mapping")
    payload_identity = {key: value for key, value in payload.items() if key not in {"records"}}
    stored_identity = payload_identity.get("artifact_identity_sha256")
    if stored_identity != canonical_sha256(
        {key: value for key, value in payload_identity.items() if key != "artifact_identity_sha256"}
    ):
        raise RuntimeError("CPNG completion capture payload identity hash differs")
    expected_manifest = {
        **payload_identity,
        "tensor_path": _relative(CAPTURE_PATH),
        "tensor_file_sha256": tensor_hash,
    }
    if {key: value for key, value in manifest.items() if key != "manifest_sha256"} != (
        expected_manifest
    ):
        raise RuntimeError("CPNG completion capture manifest content differs")
    if (
        payload.get("schema_version") != CAPTURE_SCHEMA
        or payload.get("development_only") is not True
        or payload.get("form_count") != 16
        or payload.get("compute") != {"forward_evaluations": 48, "backward_evaluations": 32}
        or payload.get("compute_ledger") != ledger.snapshot()
        or payload.get("compute_ledger_sha256") != file_sha256(CAPTURE_LEDGER_PATH)
        or payload.get("artifact_identity_sha256") != manifest.get("artifact_identity_sha256")
        or payload.get("study_identity_sha256") != study_identity.get("identity_sha256")
        or len(payload.get("records", [])) != 16
    ):
        raise RuntimeError("CPNG completion capture identity differs")
    specifications = _completion_specifications()
    _validate_capture_ledger_work_ids(ledger, specifications)
    expected_forms = {item["form_id"]: item for item in specifications}
    records = payload["records"]
    expected_form_order = [str(item["form_id"]) for item in specifications]
    observed_form_order = [str(record.get("form_id")) for record in records]
    if observed_form_order != expected_form_order or len(set(observed_form_order)) != 16:
        raise RuntimeError("CPNG completion capture form coverage differs")
    expected_record_keys = {
        "form_id",
        "case_id",
        "assignment",
        "target",
        "prompt_sha256",
        "preserve_completion_sha256",
        "comply_completion_sha256",
        "effective_gradient",
        "effective_gradient_sha256",
        "prompt_residual",
        "prompt_residual_sha256",
        "prompt_token_ids_sha256",
        "preserve_content_token_ids_sha256",
        "comply_content_token_ids_sha256",
        "audit",
        "audit_sha256",
    }
    for record in records:
        form_id = str(record["form_id"])
        specification = expected_forms[form_id]
        if set(record) != expected_record_keys:
            raise RuntimeError(f"CPNG completion capture {form_id} fields differ")
        for field in (
            "case_id",
            "assignment",
            "target",
            "prompt_sha256",
            "preserve_completion_sha256",
            "comply_completion_sha256",
        ):
            if record.get(field) != specification[field]:
                raise RuntimeError(f"CPNG completion capture {form_id} {field} differs")
        for tensor_field, hash_field in (
            ("effective_gradient", "effective_gradient_sha256"),
            ("prompt_residual", "prompt_residual_sha256"),
        ):
            tensor = record.get(tensor_field)
            if (
                not torch.is_tensor(tensor)
                or tensor.dtype != torch.float32
                or tensor.device.type != "cpu"
                or tensor.shape != (1024,)
                or not bool(torch.isfinite(tensor).all().item())
                or record.get(hash_field) != base.v3.tensor_float32_sha256(tensor)
            ):
                raise RuntimeError(f"CPNG completion capture {form_id} {tensor_field} differs")
        audit = record.get("audit")
        if not isinstance(audit, Mapping):
            raise TypeError(f"CPNG completion capture {form_id} audit is missing")
        if (
            record.get("audit_sha256") != canonical_sha256(audit)
            or audit.get("effective_gradient_sha256") != record["effective_gradient_sha256"]
            or audit.get("prompt_token_ids_sha256") != record["prompt_token_ids_sha256"]
            or audit.get("preserve", {}).get("content_token_ids_sha256")
            != record["preserve_content_token_ids_sha256"]
            or audit.get("comply", {}).get("content_token_ids_sha256")
            != record["comply_content_token_ids_sha256"]
        ):
            raise RuntimeError(f"CPNG completion capture {form_id} audit differs")
    expected_record_manifest = [
        {
            key: value
            for key, value in record.items()
            if key not in {"effective_gradient", "prompt_residual", "audit"}
        }
        for record in records
    ]
    if manifest.get("record_manifest") != expected_record_manifest:
        raise RuntimeError("CPNG completion capture record manifest differs")
    if manifest.get("form_manifest_sha256") != canonical_sha256(
        _public_completion_manifest(specifications)
    ):
        raise RuntimeError("CPNG completion form manifest differs")
    return payload


def _raw_unrelated_rows(torch: Any, frozen: Mapping[str, Any]) -> Any:
    records = sorted(frozen["nuisance_records"].values(), key=lambda record: str(record["form_id"]))
    rows = torch.cat(
        [
            torch.cat(
                (
                    record["semantic_gradient"].reshape(1, -1).double(),
                    record["greedy_competitor_gap_gradients"].double(),
                ),
                dim=0,
            )
            for record in records
        ],
        dim=0,
    ).contiguous()
    if rows.shape != (288, 1024):
        raise RuntimeError("CPNG unrelated gradient matrix differs from frozen v3")
    return rows


def _unrelated_basis_compatibility(
    torch: Any, *, unrelated_rows: Any, frozen_basis: Any
) -> dict[str, Any]:
    _projected, reconstructed_basis, _diagnostics = global_unrelated_null_projection(
        torch,
        vector=torch.zeros(int(unrelated_rows.shape[1]), dtype=torch.float64),
        unrelated_gradient_rows=unrelated_rows,
    )
    frozen = frozen_basis.detach().cpu().double().contiguous()
    if reconstructed_basis.shape != frozen.shape:
        raise RuntimeError("reconstructed unrelated basis rank differs from frozen basis")
    projector_difference = torch.linalg.matrix_norm(
        reconstructed_basis.transpose(0, 1) @ reconstructed_basis - frozen.transpose(0, 1) @ frozen
    )
    denominator = max(
        1.0,
        float(torch.linalg.matrix_norm(frozen.transpose(0, 1) @ frozen).item()),
    )
    relative_difference = float(projector_difference.item()) / denominator
    if relative_difference > 1e-8:
        raise RuntimeError("reconstructed unrelated projector differs from frozen projector")
    return {
        "reconstructed_rank": int(reconstructed_basis.shape[0]),
        "frozen_rank": int(frozen.shape[0]),
        "relative_projector_difference": relative_difference,
        "tolerance": 1e-8,
        "compatible": True,
    }


def _protected_metric_groups(
    frozen: Mapping[str, Any], *, case_id: str, assignment: int
) -> tuple[dict[str, list[Mapping[str, Any]]], list[dict[str, Any]]]:
    nuisance = sorted(frozen["nuisance_records"].values(), key=lambda row: str(row["form_id"]))
    other_forms = sorted(
        [
            form
            for form in frozen["sp_forms"]
            if str(form["case_id"]) == case_id
            and int(form["assignment"]) == assignment
            and str(form["target"]) == "other"
        ],
        key=lambda form: bool(form["preserve_first"]),
    )
    if len(nuisance) != 32 or len(other_forms) != 2:
        raise RuntimeError("protected metric requires 32 nuisance and two matched-other prompts")
    if {bool(form["preserve_first"]) for form in other_forms} != {True, False}:
        raise RuntimeError("matched-other metric prompts lack symmetric answer orders")
    other_records = [frozen["sp_records"][str(form["form_id"])] for form in other_forms]
    records = [*nuisance, *other_records]
    manifest = [
        {
            "form_id": str(record["form_id"]),
            "source": "nuisance_fit" if index < len(nuisance) else "matched_other",
            "answer_order_role": (
                None
                if index < len(nuisance)
                else ("first_rendering" if index == len(nuisance) else "second_rendering")
            ),
            "uses_answer_label_mapping": False,
            "uses_semantic_orientation": False,
            "uses_outcome": False,
            "uses_ab_choice_prompt_tokens": True,
            "uses_ab_token_ids_and_coarsened_categories": True,
        }
        for index, record in enumerate(records)
    ]
    if any(item["source"] == "self" for item in manifest):
        raise RuntimeError("self records leaked into protected metric")
    return {"unrelated": nuisance, "matched_other": other_records}, manifest


def _protected_metric_factors(
    torch: Any,
    *,
    frozen: Mapping[str, Any],
    case_id: str,
    assignment: int,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    """Give matched-other and unrelated prompt groups exactly one-half weight each."""

    groups, manifest = _protected_metric_groups(frozen, case_id=case_id, assignment=assignment)
    unrelated_factors, unrelated_diagnostics = base._fisher_factors(
        torch,
        groups["unrelated"],
        construction=frozen["manifest"]["construction"],
    )
    other_factors, other_diagnostics = base._fisher_factors(
        torch,
        groups["matched_other"],
        construction=frozen["manifest"]["construction"],
    )
    factors = (
        torch.cat(
            (math.sqrt(0.5) * other_factors, math.sqrt(0.5) * unrelated_factors),
            dim=0,
        )
        .double()
        .contiguous()
    )
    diagnostics = {
        "metric_kind": "prompt_group_balanced_topk_plus_aggregate_tail_next_token_pullback",
        "is_full_vocabulary_fisher": False,
        "group_weights": {"matched_other": 0.5, "unrelated": 0.5},
        "matched_other_prompt_count": len(groups["matched_other"]),
        "unrelated_prompt_count": len(groups["unrelated"]),
        "self_prompt_count": 0,
        "matched_other_diagnostics": other_diagnostics,
        "unrelated_diagnostics": unrelated_diagnostics,
        "factor_sha256": base.v3.tensor_float64_sha256(factors),
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return factors, diagnostics, manifest


def _case_assignments(frozen: Mapping[str, Any]) -> list[tuple[str, int]]:
    result = sorted(
        {(str(form["case_id"]), int(form["assignment"])) for form in frozen["sp_forms"]}
    )
    if len(result) != 8:
        raise RuntimeError("CPNG Stage A must contain eight case assignments")
    return result


def _is_allowlisted_candidate_failure(error: BaseException, *, phase: str) -> bool:
    expected = {
        "construction": CounterfactualConstructionIneligible,
        "evaluation": CandidateLocalNumericalFailure,
    }.get(phase)
    if expected is None or not isinstance(error, expected):
        return False
    if phase == "construction":
        return str(error) in ALLOWLISTED_CONSTRUCTION_FAILURE_MESSAGES
    return str(error) in CANDIDATE_LOCAL_NUMERICAL_FAILURE_MESSAGES


def _build_constructions(
    torch: Any,
    *,
    frozen: Mapping[str, Any],
    capture: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = {str(record["form_id"]): record for record in capture["records"]}
    unrelated_rows = _raw_unrelated_rows(torch, frozen)
    basis_compatibility = _unrelated_basis_compatibility(
        torch,
        unrelated_rows=unrelated_rows,
        frozen_basis=frozen["global_nuisance_basis"],
    )
    entries = []
    for case_id, assignment in _case_assignments(frozen):
        self_record = records[f"completion:{case_id}:assignment={assignment}:self"]
        other_record = records[f"completion:{case_id}:assignment={assignment}:other"]
        factors, factor_diagnostics, factor_manifest = _protected_metric_factors(
            torch, frozen=frozen, case_id=case_id, assignment=assignment
        )
        for ridge in FISHER_RIDGE_MULTIPLIER_GRID:
            common = {
                "case_id": case_id,
                "assignment": assignment,
                "fisher_ridge_multiplier": ridge,
                "protected_metric_factor_sha256": base.v3.tensor_float64_sha256(factors),
                "protected_metric_prompt_manifest": factor_manifest,
                "protected_metric_prompt_manifest_sha256": canonical_sha256(factor_manifest),
                "protected_metric_diagnostics": factor_diagnostics,
                "unrelated_basis_compatibility": basis_compatibility,
            }
            try:
                direction, diagnostics = build_counterfactual_protected_natural_gradient(
                    torch,
                    self_completion_gradient=self_record["effective_gradient"],
                    matched_other_completion_gradient=other_record["effective_gradient"],
                    unrelated_gradient_rows=unrelated_rows,
                    protected_metric_factors=factors,
                    fisher_ridge_multiplier=ridge,
                )
            except CounterfactualConstructionIneligible as error:
                if not _is_allowlisted_candidate_failure(error, phase="construction"):
                    raise RuntimeError(
                        "undeclared CPNG candidate-local construction failure"
                    ) from error
                entries.append(
                    {
                        **common,
                        "construction_status": "failed_closed",
                        "construction_failure_type": type(error).__name__,
                        "construction_failure_message": str(error),
                    }
                )
                continue
            entries.append(
                {
                    **common,
                    "construction_status": "constructed",
                    "direction": direction.double().contiguous(),
                    "direction_sha256": base.v3.tensor_float64_sha256(direction),
                    "construction_diagnostics": diagnostics,
                }
            )
            maximum_frozen_null_residual = float(
                torch.max(
                    torch.abs(frozen["global_nuisance_basis"].double() @ direction.double())
                ).item()
            )
            if maximum_frozen_null_residual > 1e-8 * (1.0 + float(direction.norm().item())):
                raise RuntimeError("constructed direction failed the frozen unrelated null")
            entries[-1]["maximum_abs_frozen_unrelated_basis_projection"] = (
                maximum_frozen_null_residual
            )
    if len(entries) != 24:
        raise RuntimeError("CPNG construction must contain 8 x 3 directions")
    return entries


def run_construct() -> dict[str, Any]:
    preflight = run_preflight()
    if CONSTRUCTION_PATH.exists() != CONSTRUCTION_MANIFEST_PATH.exists():
        raise RuntimeError("CPNG construction bank is only partially present")
    import torch

    frozen = trust._load_frozen_inputs(torch)
    capture = _load_capture(torch, preflight["study_identity"])
    if CONSTRUCTION_PATH.exists():
        _load_constructions(
            torch,
            preflight["study_identity"],
            frozen=frozen,
            capture=capture,
        )
        return _load_json(CONSTRUCTION_MANIFEST_PATH)
    entries = _build_constructions(torch, frozen=frozen, capture=capture)
    identity = {
        "schema_version": CONSTRUCTION_SCHEMA,
        "development_only": True,
        "study_identity_sha256": preflight["study_identity"]["identity_sha256"],
        "completion_capture_file_sha256": file_sha256(CAPTURE_PATH),
        "entry_count": len(entries),
        "entry_manifest": [_construction_entry_manifest(entry) for entry in entries],
        "model_forwards": 0,
        "model_backwards": 0,
    }
    identity["artifact_identity_sha256"] = canonical_sha256(identity)
    payload = {**identity, "entries": entries}
    return _save_tensor_artifact(
        torch,
        tensor_path=CONSTRUCTION_PATH,
        manifest_path=CONSTRUCTION_MANIFEST_PATH,
        payload=payload,
        public_manifest=identity,
    )


def _construction_entry_manifest(entry: Mapping[str, Any]) -> dict[str, Any]:
    output = {
        key: value
        for key, value in entry.items()
        if key not in {"direction", "construction_diagnostics", "protected_metric_diagnostics"}
    }
    if isinstance(entry.get("construction_diagnostics"), Mapping):
        output["construction_diagnostics_sha256"] = entry["construction_diagnostics"].get(
            "diagnostics_sha256"
        )
    if isinstance(entry.get("protected_metric_diagnostics"), Mapping):
        output["protected_metric_diagnostics_sha256"] = entry["protected_metric_diagnostics"].get(
            "diagnostics_sha256"
        )
    return output


def _load_constructions(
    torch: Any,
    study_identity: Mapping[str, Any],
    *,
    frozen: Mapping[str, Any],
    capture: Mapping[str, Any],
) -> dict[str, Any]:
    if CONSTRUCTION_PATH.exists() != CONSTRUCTION_MANIFEST_PATH.exists():
        raise RuntimeError("CPNG construction bank is only partially present")
    if not CONSTRUCTION_PATH.is_file():
        raise RuntimeError("CPNG construction bank is incomplete")
    manifest = _load_json(CONSTRUCTION_MANIFEST_PATH)
    _verify_internal_hash(manifest, hash_field="manifest_sha256")
    tensor_hash = file_sha256(CONSTRUCTION_PATH)
    if manifest.get("tensor_file_sha256") != tensor_hash:
        raise RuntimeError("CPNG construction file hash differs")
    payload = torch.load(CONSTRUCTION_PATH, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("CPNG construction payload is not a mapping")
    payload_identity = {key: value for key, value in payload.items() if key != "entries"}
    if payload_identity.get("artifact_identity_sha256") != canonical_sha256(
        {key: value for key, value in payload_identity.items() if key != "artifact_identity_sha256"}
    ):
        raise RuntimeError("CPNG construction payload identity hash differs")
    expected_manifest = {
        **payload_identity,
        "tensor_path": _relative(CONSTRUCTION_PATH),
        "tensor_file_sha256": tensor_hash,
    }
    if {key: value for key, value in manifest.items() if key != "manifest_sha256"} != (
        expected_manifest
    ):
        raise RuntimeError("CPNG construction manifest content differs")
    if (
        payload.get("artifact_identity_sha256") != manifest.get("artifact_identity_sha256")
        or payload.get("study_identity_sha256") != study_identity.get("identity_sha256")
        or len(payload.get("entries", [])) != 24
        or payload.get("completion_capture_file_sha256") != file_sha256(CAPTURE_PATH)
        or payload.get("completion_capture_file_sha256")
        != capture.get("tensor_file_sha256", file_sha256(CAPTURE_PATH))
    ):
        raise RuntimeError("CPNG construction bank identity differs")
    expected_key_order = [
        (case_id, assignment, float(ridge))
        for case_id, assignment in _case_assignments(frozen)
        for ridge in FISHER_RIDGE_MULTIPLIER_GRID
    ]
    entries = payload["entries"]
    observed_key_order = [
        (
            str(entry.get("case_id")),
            int(entry.get("assignment", -1)),
            float(entry.get("fisher_ridge_multiplier", math.nan)),
        )
        for entry in entries
    ]
    if observed_key_order != expected_key_order:
        raise RuntimeError("CPNG construction key coverage differs")
    factors_by_case = {}
    for case_id, assignment in _case_assignments(frozen):
        factors_by_case[(case_id, assignment)] = _protected_metric_factors(
            torch, frozen=frozen, case_id=case_id, assignment=assignment
        )
    for entry in entries:
        key = (str(entry["case_id"]), int(entry["assignment"]))
        factors, factor_diagnostics, factor_manifest = factors_by_case[key]
        factor_hash = base.v3.tensor_float64_sha256(factors)
        if (
            entry.get("protected_metric_factor_sha256") != factor_hash
            or entry.get("protected_metric_prompt_manifest") != factor_manifest
            or entry.get("protected_metric_prompt_manifest_sha256")
            != canonical_sha256(factor_manifest)
            or entry.get("protected_metric_diagnostics") != factor_diagnostics
        ):
            raise RuntimeError("CPNG construction recomputed factor identity differs")
        status = entry.get("construction_status")
        if status == "failed_closed":
            error = CounterfactualConstructionIneligible(
                str(entry.get("construction_failure_message"))
            )
            success_only = {
                "direction",
                "direction_sha256",
                "construction_diagnostics",
                "maximum_abs_frozen_unrelated_basis_projection",
            }
            if (
                entry.get("construction_failure_type") != "CounterfactualConstructionIneligible"
                or not _is_allowlisted_candidate_failure(error, phase="construction")
                or any(key in entry for key in success_only)
            ):
                raise RuntimeError("CPNG construction failure is not allowlisted")
            continue
        direction = entry.get("direction")
        diagnostics = entry.get("construction_diagnostics")
        if (
            status != "constructed"
            or not torch.is_tensor(direction)
            or direction.dtype != torch.float64
            or direction.device.type != "cpu"
            or direction.shape != (1024,)
            or not bool(torch.isfinite(direction).all().item())
            or entry.get("direction_sha256") != base.v3.tensor_float64_sha256(direction)
            or not isinstance(diagnostics, Mapping)
        ):
            raise RuntimeError("CPNG constructed direction content differs")
        _verify_internal_hash(diagnostics, hash_field="diagnostics_sha256")
        if diagnostics.get("direction_sha256") != entry["direction_sha256"]:
            raise RuntimeError("CPNG direction diagnostics hash chain differs")
        null_residual = float(
            torch.max(torch.abs(frozen["global_nuisance_basis"].double() @ direction)).item()
        )
        if (
            not math.isfinite(null_residual)
            or null_residual > 1e-8 * (1.0 + float(direction.norm().item()))
            or not math.isclose(
                null_residual,
                float(entry["maximum_abs_frozen_unrelated_basis_projection"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise RuntimeError("CPNG direction frozen-null certificate differs")
    expected_entry_manifest = [_construction_entry_manifest(entry) for entry in entries]
    if manifest.get("entry_manifest") != expected_entry_manifest:
        raise RuntimeError("CPNG construction entry manifest differs")
    reconstructed_entries = _build_constructions(torch, frozen=frozen, capture=capture)
    for stored_entry, reconstructed_entry in zip(entries, reconstructed_entries, strict=True):
        stored_direction = stored_entry.get("direction")
        reconstructed_direction = reconstructed_entry.get("direction")
        if (torch.is_tensor(stored_direction) or torch.is_tensor(reconstructed_direction)) and not (
            torch.is_tensor(stored_direction)
            and torch.is_tensor(reconstructed_direction)
            and torch.equal(stored_direction, reconstructed_direction)
        ):
            raise RuntimeError("CPNG reconstructed direction differs")
        stored_metadata = {key: value for key, value in stored_entry.items() if key != "direction"}
        reconstructed_metadata = {
            key: value for key, value in reconstructed_entry.items() if key != "direction"
        }
        if stored_metadata != reconstructed_metadata:
            raise RuntimeError("CPNG reconstructed construction metadata differs")
    return payload


def _candidate_perturbations(
    torch: Any,
    *,
    entries: Sequence[Mapping[str, Any]],
    factors_by_ridge: Mapping[float, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = []
    unique: dict[str, Any] = {}
    grid = preregistered_candidate_grid()
    by_ridge = {float(entry["fisher_ridge_multiplier"]): entry for entry in entries}
    for grid_index, candidate in enumerate(grid):
        ridge = float(candidate["fisher_ridge_multiplier"])
        entry = by_ridge[ridge]
        if entry.get("construction_status") != "constructed":
            candidates.append(
                {
                    "grid_index": grid_index,
                    **candidate,
                    "construction_status": "failed_closed",
                    "construction_failure_type": entry.get("construction_failure_type"),
                    "construction_failure_message": entry.get("construction_failure_message"),
                    "perturbation_sha256": None,
                    "direction_sha256": None,
                    "deduplication_key": None,
                    "scaling": None,
                }
            )
            continue
        factor_hash = str(entry["protected_metric_factor_sha256"])
        perturbation, scaling = scale_to_predicted_coarsened_next_token_kl_budget(
            torch,
            unit_coarsened_next_token_kl_direction=entry["direction"].double(),
            protected_metric_factors=factors_by_ridge[ridge],
            expected_protected_metric_factors_sha256=factor_hash,
            predicted_coarsened_next_token_kl_budget=float(
                candidate["predicted_coarsened_next_token_kl_budget"]
            ),
            residual_relative_l2_cap=float(candidate["residual_relative_l2_cap"]),
        )
        applied = perturbation.float().contiguous()
        applied_certificate = certify_applied_float32_perturbation(
            torch,
            requested_perturbation=perturbation,
            applied_float32_perturbation=applied,
            protected_metric_factors=factors_by_ridge[ridge],
            expected_protected_metric_factors_sha256=factor_hash,
            predicted_coarsened_next_token_kl_budget=float(
                candidate["predicted_coarsened_next_token_kl_budget"]
            ),
            residual_relative_l2_cap=float(candidate["residual_relative_l2_cap"]),
        )
        perturbation_sha256 = base.v3.tensor_float32_sha256(applied)
        deduplication_key = canonical_sha256(
            {
                "direction_sha256": str(entry["direction_sha256"]),
                "applied_float32_perturbation_sha256": perturbation_sha256,
            }
        )
        unique.setdefault(deduplication_key, applied)
        candidates.append(
            {
                "grid_index": grid_index,
                **candidate,
                "construction_status": "constructed",
                "perturbation_sha256": perturbation_sha256,
                "direction_sha256": str(entry["direction_sha256"]),
                "deduplication_key": deduplication_key,
                "scaling": scaling,
                "applied_float32_certificate": applied_certificate,
            }
        )
    if len(candidates) != 48:
        raise RuntimeError("CPNG calibration grid must contain exactly 48 candidates")
    return candidates, unique


def _json_primary(primary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "observations": [trust._observation_log(row) for row in primary["observations"]],
        "self_application": primary["self_application"],
        "matched_other": primary["matched_other"],
        "terminal_gate": primary["terminal_gate"],
    }


def _specificity_effect(primary: Mapping[str, Any]) -> float:
    rows = primary["observations"]
    self_rows = {
        (bool(row["preserve_first"]), int(row["sign"])): row
        for row in rows
        if row["family"] == "self"
    }
    other_rows = {
        (bool(row["preserve_first"]), int(row["sign"])): row
        for row in rows
        if row["family"] == "matched_other"
    }
    self_span = statistics.fmean(
        (
            float(self_rows[(order, 1)]["semantic_desired_gap"])
            + float(self_rows[(order, -1)]["semantic_desired_gap"])
        )
        / 2.0
        for order in (True, False)
    )
    other_span = statistics.fmean(
        (
            float(other_rows[(order, 1)]["semantic_desired_gap"])
            - float(other_rows[(order, -1)]["semantic_desired_gap"])
        )
        / 2.0
        for order in (True, False)
    )
    return self_span - abs(other_span)


def _candidate_logits_with_delta(
    backend: Any,
    *,
    form: Mapping[str, Any],
    delta: Any,
    sign: int,
    layer: int,
    budget: trust.EvaluationBudget,
) -> tuple[Any, dict[str, Any]]:
    """Apply one CPNG candidate while separating numerical failure from integrity failure."""

    if sign not in {-1, 1} or isinstance(sign, bool):
        raise ValueError("intervention sign must be +1 or -1")
    torch = backend.torch
    tokens, _boundary, _positive_id, _negative_id = trust._resolve_ids(backend, form)
    vector = delta.detach().to(device=tokens.device, dtype=torch.float32).contiguous()
    if vector.ndim != 1 or not bool(torch.isfinite(vector).all().item()):
        raise ValueError("delta must be a finite one-dimensional tensor")
    captured: dict[str, Any] = {"hook_calls": 0}

    def intervention_hook(activation: Any, hook: Any) -> Any:
        del hook
        captured["hook_calls"] += 1
        if captured["hook_calls"] != 1:
            raise RuntimeError("CPNG intervention hook fired more than once")
        if activation.ndim != 3 or int(activation.shape[0]) != 1:
            raise RuntimeError("CPNG residual activation must be [1, sequence, width]")
        if int(activation.shape[1]) != int(tokens.shape[-1]):
            raise RuntimeError("CPNG activation does not end at the prompt boundary")
        if int(activation.shape[-1]) != int(vector.numel()):
            raise RuntimeError("CPNG delta width differs from the residual width")
        prompt_index = int(tokens.shape[-1]) - 1
        original = activation.detach().float()
        residual_norm = original[0, prompt_index].norm().detach()
        if not bool(torch.isfinite(residual_norm).item()) or float(residual_norm.item()) <= 0.0:
            raise RuntimeError("CPNG prompt-final residual norm is not finite and positive")
        mask = torch.zeros_like(original)
        mask[:, prompt_index, :] = 1.0
        changed = original + sign * residual_norm * mask * vector.view(1, 1, -1)
        returned = changed.to(dtype=activation.dtype)
        returned_float = returned.detach().float()
        if not torch.equal(returned_float[:, :prompt_index], original[:, :prompt_index]):
            raise RuntimeError("CPNG hook changed a non-final prompt position")
        applied = returned_float[0, prompt_index] - original[0, prompt_index]
        expected_applied = sign * residual_norm * vector
        actual_norm = applied.norm()
        requested_relative_norm = vector.norm()
        realized_relative_norm = actual_norm / residual_norm
        maximum_abs_application_error = torch.max(torch.abs(applied - expected_applied))
        if not all(
            bool(torch.isfinite(value).item())
            for value in (
                actual_norm,
                requested_relative_norm,
                realized_relative_norm,
                maximum_abs_application_error,
            )
        ):
            raise CandidateLocalNumericalFailure("CPNG application diagnostics are non-finite")
        captured.update(
            {
                "selected_position_count": 1,
                "prompt_final_index": prompt_index,
                "residual_norm": float(residual_norm.item()),
                "actual_perturbation_norm": float(actual_norm.item()),
                "requested_relative_perturbation_norm": float(requested_relative_norm.item()),
                "realized_relative_perturbation_norm": float(realized_relative_norm.item()),
                "absolute_relative_perturbation_error": float(
                    torch.abs(realized_relative_norm - requested_relative_norm).item()
                ),
                "maximum_abs_application_coordinate_error": float(
                    maximum_abs_application_error.item()
                ),
                "maximum_abs_relative_application_coordinate_error": float(
                    (maximum_abs_application_error / residual_norm).item()
                ),
            }
        )
        return returned

    budget.require_capacity(forward=1)
    budget.record_forward()
    with (
        torch.inference_mode(),
        backend.model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_out", intervention_hook)]),
    ):
        model_output = backend.model(tokens)
    if captured["hook_calls"] != 1 or captured.get("selected_position_count") != 1:
        raise RuntimeError("CPNG intervention did not select exactly one position")
    if model_output.ndim != 3 or int(model_output.shape[0]) != 1:
        raise RuntimeError("CPNG model output is not [1, sequence, vocabulary]")
    if int(model_output.shape[1]) != int(tokens.shape[-1]):
        raise RuntimeError("CPNG model output does not end at the prompt boundary")
    logits = model_output[0, -1].detach().float().cpu().contiguous()
    if logits.ndim != 1:
        raise RuntimeError("CPNG changed logits are not a vocabulary vector")
    if not bool(torch.isfinite(logits).all().item()):
        raise CandidateLocalNumericalFailure("CPNG changed logits are non-finite")
    diagnostics = {
        **captured,
        "sign": sign,
        "requested_delta_norm": float(vector.norm().item()),
        "delta_float32_sha256": base.v3.tensor_float32_sha256(vector.cpu()),
    }
    return logits, diagnostics


def _candidate_changed_observation(
    backend: Any,
    *,
    specification: Mapping[str, Any],
    delta: Any,
    layer: int,
    baseline_cache: dict[str, dict[str, Any]],
    budget: trust.EvaluationBudget,
) -> dict[str, Any]:
    form = specification["form"]
    frozen_record = specification["frozen_record"]
    baseline = trust._baseline_observation(
        backend,
        form=form,
        frozen_record=frozen_record,
        cache=baseline_cache,
        budget=budget,
    )
    logits, intervention = _candidate_logits_with_delta(
        backend,
        form=form,
        delta=delta,
        sign=int(specification["sign"]),
        layer=layer,
        budget=budget,
    )
    positive_id = int(baseline["positive_id"])
    negative_id = int(baseline["negative_id"])
    sign = int(specification["sign"])
    desired_id = (
        positive_id
        if specification["family"] == "self" and sign == 1
        else (
            negative_id if specification["family"] == "self" else int(baseline["actual_token_id"])
        )
    )
    competitor_id, constraint_value = trust._strongest_competitor(backend.torch, logits, desired_id)
    actual_token_id = int(logits.argmax().item())
    actual_semantic = trust._semantic_choice(
        actual_token_id,
        positive_id=positive_id,
        negative_id=negative_id,
    )
    baseline_log_probs = backend.torch.log_softmax(baseline["logits"].float(), dim=-1)
    changed_log_probs = backend.torch.log_softmax(logits.float(), dim=-1)
    changed_to_baseline_kl = float(
        (changed_log_probs.exp() * (changed_log_probs - baseline_log_probs)).sum().item()
    )
    if not math.isfinite(changed_to_baseline_kl) or changed_to_baseline_kl < -1e-6:
        raise CandidateLocalNumericalFailure("CPNG changed-to-baseline KL is invalid")
    changed_to_baseline_kl = max(0.0, changed_to_baseline_kl)
    semantic_positive_gap = float((logits[positive_id] - logits[negative_id]).item())
    return {
        "constraint_id": str(specification["constraint_id"]),
        "family": str(specification["family"]),
        "form_id": str(form["form_id"]),
        "preserve_first": bool(specification["preserve_first"]),
        "sign": sign,
        "required_margin": float(specification["required_margin"]),
        "constraint_value": float(constraint_value.item()),
        "desired_token_id": desired_id,
        "strongest_competitor_token_id": competitor_id,
        "baseline_actual_token_id": int(baseline["actual_token_id"]),
        "baseline_semantic_choice": str(baseline["semantic"]),
        "choice_boundary_evidence_sha256": str(baseline["choice_boundary_evidence_sha256"]),
        "actual_token_id": actual_token_id,
        "actual_semantic_choice": actual_semantic,
        "positive_id": positive_id,
        "negative_id": negative_id,
        "semantic_desired_gap": (
            sign * semantic_positive_gap
            if specification["family"] == "self"
            else semantic_positive_gap
        ),
        "full_vocabulary_kl_changed_to_baseline": changed_to_baseline_kl,
        "new_other_output": baseline["semantic"] != "OTHER" and actual_semantic == "OTHER",
        "exact_token_changed": actual_token_id != int(baseline["actual_token_id"]),
        "semantic_decision_changed": actual_semantic != str(baseline["semantic"]),
        "intervention": intervention,
    }


def _candidate_primary_evaluation(
    backend: Any,
    *,
    specifications: Sequence[Mapping[str, Any]],
    delta: Any,
    layer: int,
    baseline_cache: dict[str, dict[str, Any]],
    limits: Mapping[str, Any],
    target_margin: float,
    budget: trust.EvaluationBudget,
) -> dict[str, Any]:
    observations = [
        _candidate_changed_observation(
            backend,
            specification=specification,
            delta=delta,
            layer=layer,
            baseline_cache=baseline_cache,
            budget=budget,
        )
        for specification in specifications
    ]
    if len(observations) != 8:
        raise RuntimeError("CPNG primary evaluation did not produce eight signed rows")
    self_rows = [row for row in observations if row["family"] == "self"]
    other_rows = [row for row in observations if row["family"] == "matched_other"]
    return {
        "observations": observations,
        "self_application": trust._application_report(self_rows, group="self"),
        "matched_other": trust._protection_report(other_rows, limits=limits, group="matched_other"),
        "terminal_gate": trust._terminal_from_self_observations(
            backend,
            observations=observations,
            target_margin=target_margin,
        ),
    }


def _candidate_nuisance_evaluation(
    backend: Any,
    *,
    frozen: Mapping[str, Any],
    delta: Any,
    layer: int,
    baseline_cache: dict[str, dict[str, Any]],
    limits: Mapping[str, Any],
    budget: trust.EvaluationBudget,
) -> dict[str, Any]:
    observations = [
        _candidate_changed_observation(
            backend,
            specification=specification,
            delta=delta,
            layer=layer,
            baseline_cache=baseline_cache,
            budget=budget,
        )
        for specification in trust._nuisance_specifications(frozen)
    ]
    return {
        "report": trust._protection_report(observations, limits=limits, group="nuisance_fit"),
        "observations": observations,
    }


def _evaluate_unique_perturbation(
    backend: Any,
    *,
    case_id: str,
    assignment: int,
    delta: Any,
    frozen: Mapping[str, Any],
    lock: Mapping[str, Any],
    baseline_cache: dict[str, dict[str, Any]],
    budget: trust.EvaluationBudget,
    primary_fn: Callable[..., Mapping[str, Any]] = _candidate_primary_evaluation,
) -> dict[str, Any]:
    optimizer = {
        "target_margin_logit": float(lock["target_margin_logit"]),
        "matched_other_margin_logit": float(lock["matched_other_margin_logit"]),
    }
    specifications = trust._constraint_specifications(
        case_id=case_id,
        assignment=assignment,
        frozen=frozen,
        optimizer=optimizer,
    )
    primary = primary_fn(
        backend,
        specifications=specifications,
        delta=delta,
        layer=int(lock["model"]["layer_zero_based"]),
        baseline_cache=baseline_cache,
        limits=lock["protected_limits"],
        target_margin=float(lock["target_margin_logit"]),
        budget=budget,
    )
    null_report = trust._null_certificate(
        backend.torch,
        delta=delta,
        global_basis=frozen["global_nuisance_basis"],
        absolute_cap=max(RESIDUAL_RELATIVE_L2_CAP_GRID),
    )
    terminal_candidate = (
        bool(primary["terminal_gate"]["passes_terminal_gate"])
        and bool(primary["self_application"]["passes"])
        and bool(primary["matched_other"]["passes"])
        and bool(null_report["passes"])
    )
    result = {
        "terminal_candidate": terminal_candidate,
        "stage_one_target_success": terminal_candidate,
        "primary": _json_primary(primary),
        "null_certificate": null_report,
        "self_minus_matched_other_effect": _specificity_effect(primary),
    }
    result["evaluation_sha256"] = canonical_sha256(result)
    return result


def _selection_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 8 * 48:
        raise RuntimeError("global selection requires every one of the 384 attempts")
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["grid_index"]), []).append(row)
    summaries = []
    for grid_index, candidate in enumerate(preregistered_candidate_grid()):
        attempts = grouped.get(grid_index, [])
        if len(attempts) != 8:
            raise RuntimeError("global candidate lacks all eight case assignments")
        evaluated = [row for row in attempts if row.get("evaluation_status") == "evaluated"]
        observed_violation = any(
            not bool(row["matched_other_passed"]) or not bool(row["null_passed"])
            for row in evaluated
        )
        success_count = sum(bool(row["success"]) for row in attempts)
        effects = [float(row["self_minus_matched_other_effect"]) for row in evaluated]
        protected_kls = [float(row["matched_other_mean_kl"]) for row in evaluated]
        summaries.append(
            {
                "grid_index": grid_index,
                **candidate,
                "attempt_count": len(attempts),
                "construction_failure_count": sum(
                    row["construction_status"] != "constructed" for row in attempts
                ),
                "evaluation_failure_count": sum(
                    row.get("evaluation_status") == "failed_closed" for row in attempts
                ),
                "completed_evaluation_count": len(evaluated),
                "success_count": success_count,
                "observed_protection_violation": observed_violation,
                "eligible": not observed_violation and len(evaluated) == 8,
                "median_self_minus_matched_other_effect": (
                    statistics.median(effects) if effects else -1e300
                ),
                "mean_matched_other_kl": (
                    statistics.fmean(protected_kls) if protected_kls else 1e300
                ),
                "failure_count": len(attempts) - success_count,
                "tie_statistic_formula": TIE_STATISTIC_FORMULA,
            }
        )
    eligible = [item for item in summaries if item["eligible"]]
    selected = None
    if eligible:
        selected = min(
            eligible,
            key=lambda item: (
                -int(item["success_count"]),
                -float(item["median_self_minus_matched_other_effect"]),
                float(item["mean_matched_other_kl"]),
                int(item["grid_index"]),
            ),
        )
    return {
        "candidate_summaries": summaries,
        "selected_candidate": selected,
        "no_safe_candidate": selected is None,
        "selection_rule": (
            "require all eight evaluations and reject observed Stage-one matched-other/null "
            "violations; maximize actual both-order "
            "successes; maximize median self-minus-matched-other effect; minimize mean "
            "matched-other KL; break exact ties by literal grid order; choose exactly one "
            "provisional triple before the method-wide unrelated audit; never fall back"
        ),
    }


def _finalize_provisional_selection(
    provisional: Mapping[str, Any] | None,
    unrelated_audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    method_wide_unrelated_passed = (
        isinstance(provisional, Mapping)
        and len(unrelated_audits) == 8
        and all(bool(audit.get("passes")) for audit in unrelated_audits)
    )
    provisional_is_effective = bool(
        isinstance(provisional, Mapping) and int(provisional["success_count"]) >= 1
    )
    selected = (
        dict(provisional) if method_wide_unrelated_passed and provisional_is_effective else None
    )
    safe_candidate = dict(provisional) if method_wide_unrelated_passed else None
    return {
        "safe_candidate": safe_candidate,
        "safe_candidate_exists": safe_candidate is not None,
        "no_safe_candidate": safe_candidate is None,
        "effective_candidate_selected": selected,
        "no_safe_effective_candidate": selected is None,
        "selected_candidate": selected,
        "method_wide_unrelated_passed": method_wide_unrelated_passed,
        "provisional_is_effective": provisional_is_effective,
        "safe_but_ineffective": method_wide_unrelated_passed and not provisional_is_effective,
    }


def _progress_line(
    *, phase: str, completed: int, total: int, unique: int, forwards: int, elapsed: float
) -> str:
    return (
        f"CPNG {phase} {completed}/{total}: unique_perturbations={unique} "
        f"forwards={forwards} elapsed_seconds={elapsed:.1f}"
    )


def _case_checkpoint_path(root: Path, case_id: str, assignment: int) -> Path:
    safe_case = "".join(character if character.isalnum() else "_" for character in case_id)
    return root / f"{safe_case}__assignment_{assignment}.json"


def _write_stage_one_case_checkpoint(
    *,
    study_identity_sha256: str,
    case_id: str,
    assignment: int,
    rows: Sequence[Mapping[str, Any]],
    unique_count: int,
    duplicate_count: int,
    ledger: PersistentComputeLedger,
) -> dict[str, Any]:
    payload = _hashed_payload(
        {
            "schema_version": STAGE_ONE_CASE_SCHEMA,
            "study_identity_sha256": study_identity_sha256,
            "case_id": case_id,
            "assignment": assignment,
            "row_count": len(rows),
            "rows": list(rows),
            "rows_sha256": canonical_sha256(list(rows)),
            "unique_perturbation_evaluation_count": unique_count,
            "deduplicated_candidate_count": duplicate_count,
            "ledger": ledger.snapshot(),
            "ledger_file_sha256": file_sha256(CALIBRATION_LEDGER_PATH),
        },
        hash_field="checkpoint_sha256",
    )
    path = _case_checkpoint_path(STAGE_ONE_CHECKPOINT_ROOT, case_id, assignment)
    _immutable_json(path, payload)
    return payload


def _load_stage_one_case_checkpoint(
    *, study_identity_sha256: str, case_id: str, assignment: int
) -> dict[str, Any] | None:
    path = _case_checkpoint_path(STAGE_ONE_CHECKPOINT_ROOT, case_id, assignment)
    if not path.exists():
        return None
    if not path.is_file():
        raise RuntimeError("Stage-one checkpoint path is not a file")
    payload = _load_json(path)
    _verify_internal_hash(payload, hash_field="checkpoint_sha256")
    rows = payload.get("rows")
    if (
        set(payload)
        != {
            "schema_version",
            "study_identity_sha256",
            "case_id",
            "assignment",
            "row_count",
            "rows",
            "rows_sha256",
            "unique_perturbation_evaluation_count",
            "deduplicated_candidate_count",
            "ledger",
            "ledger_file_sha256",
            "checkpoint_sha256",
        }
        or payload.get("schema_version") != STAGE_ONE_CASE_SCHEMA
        or payload.get("study_identity_sha256") != study_identity_sha256
        or payload.get("case_id") != case_id
        or payload.get("assignment") != assignment
        or not isinstance(rows, list)
        or len(rows) != 48
        or payload.get("row_count") != 48
        or payload.get("rows_sha256") != canonical_sha256(rows)
    ):
        raise RuntimeError("Stage-one case checkpoint identity differs")
    expected_ids = [(case_id, assignment, grid_index) for grid_index in range(48)]
    observed_ids = [
        (str(row.get("case_id")), int(row.get("assignment", -1)), int(row.get("grid_index", -1)))
        for row in rows
    ]
    if observed_ids != expected_ids:
        raise RuntimeError("Stage-one case checkpoint row order or coverage differs")
    unique_count = payload.get("unique_perturbation_evaluation_count")
    duplicate_count = payload.get("deduplicated_candidate_count")
    if (
        isinstance(unique_count, bool)
        or not isinstance(unique_count, int)
        or not 0 <= unique_count <= 48
        or isinstance(duplicate_count, bool)
        or not isinstance(duplicate_count, int)
        or not 0 <= duplicate_count <= 48
    ):
        raise RuntimeError("Stage-one case checkpoint candidate counts are invalid")
    for row in rows:
        _verify_internal_hash(row, hash_field="row_sha256")
        if row.get("study_identity_sha256") != study_identity_sha256:
            raise RuntimeError("Stage-one row study identity differs")
    return payload


def _write_stage_one_complete(
    *,
    study_identity_sha256: str,
    rows: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    unique_count: int,
    duplicate_count: int,
    ledger: PersistentComputeLedger,
) -> dict[str, Any]:
    payload = _hashed_payload(
        {
            "schema_version": STAGE_ONE_COMPLETE_SCHEMA,
            "study_identity_sha256": study_identity_sha256,
            "row_count": len(rows),
            "rows_sha256": canonical_sha256(list(rows)),
            "selection": dict(selection),
            "selection_sha256": canonical_sha256(selection),
            "unique_perturbation_evaluation_count": unique_count,
            "deduplicated_candidate_count": duplicate_count,
            "ledger": ledger.snapshot(),
            "ledger_file_sha256": file_sha256(CALIBRATION_LEDGER_PATH),
        },
        hash_field="checkpoint_sha256",
    )
    _immutable_json(STAGE_ONE_COMPLETE_PATH, payload)
    return payload


def _write_stage_two_case_checkpoint(
    *,
    study_identity_sha256: str,
    audit: Mapping[str, Any],
    ledger: PersistentComputeLedger,
) -> None:
    payload = _hashed_payload(
        {
            "schema_version": STAGE_TWO_CASE_SCHEMA,
            "study_identity_sha256": study_identity_sha256,
            "case_id": str(audit["case_id"]),
            "assignment": int(audit["assignment"]),
            "audit": dict(audit),
            "audit_sha256": str(audit["audit_sha256"]),
            "ledger": ledger.snapshot(),
            "ledger_file_sha256": file_sha256(CALIBRATION_LEDGER_PATH),
        },
        hash_field="checkpoint_sha256",
    )
    _immutable_json(
        _case_checkpoint_path(
            STAGE_TWO_CHECKPOINT_ROOT,
            str(audit["case_id"]),
            int(audit["assignment"]),
        ),
        payload,
    )


def _read_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n") or any(not line.strip() for line in text.splitlines()):
        raise RuntimeError("CPNG JSONL is blank-lined or truncated")
    rows = []
    for line in text.splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError("CPNG JSONL row is not an object")
        rows.append(value)
    return rows


def _candidate_maps_for_validation(
    torch: Any,
    *,
    frozen: Mapping[str, Any],
    constructions: Mapping[str, Any],
) -> tuple[dict[tuple[str, int, int], dict[str, Any]], dict[tuple[str, int, int], Any]]:
    entries_by_case: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for entry in constructions["entries"]:
        entries_by_case.setdefault((str(entry["case_id"]), int(entry["assignment"])), []).append(
            entry
        )
    candidate_map = {}
    delta_map = {}
    for case_id, assignment in _case_assignments(frozen):
        factors, _diagnostics, _manifest = _protected_metric_factors(
            torch, frozen=frozen, case_id=case_id, assignment=assignment
        )
        candidates, unique = _candidate_perturbations(
            torch,
            entries=entries_by_case[(case_id, assignment)],
            factors_by_ridge={float(ridge): factors for ridge in FISHER_RIDGE_MULTIPLIER_GRID},
        )
        for candidate in candidates:
            key = (case_id, assignment, int(candidate["grid_index"]))
            candidate_map[key] = candidate
            if candidate["construction_status"] == "constructed":
                delta_map[key] = unique[str(candidate["deduplication_key"])]
    return candidate_map, delta_map


ROW_ENVELOPE_FIELDS = frozenset(
    {
        "schema_version",
        "development_only",
        "study_identity_sha256",
        "case_id",
        "assignment",
        "evaluation_sha256",
        "evaluation_status",
        "terminal_candidate",
        "success",
        "terminal_gate",
        "matched_other_passed",
        "matched_other_mean_kl",
        "null_passed",
        "self_minus_matched_other_effect",
        "evaluation",
        "row_sha256",
    }
)


def _validate_observation_sequence(
    observations: Sequence[Mapping[str, Any]],
    *,
    specifications: Sequence[Mapping[str, Any]],
    perturbation: Any,
) -> None:
    expected_observation_fields = {
        "constraint_id",
        "family",
        "form_id",
        "preserve_first",
        "sign",
        "required_margin",
        "constraint_value",
        "desired_token_id",
        "strongest_competitor_token_id",
        "baseline_actual_token_id",
        "baseline_semantic_choice",
        "choice_boundary_evidence_sha256",
        "actual_token_id",
        "actual_semantic_choice",
        "positive_id",
        "negative_id",
        "semantic_desired_gap",
        "full_vocabulary_kl_changed_to_baseline",
        "new_other_output",
        "exact_token_changed",
        "semantic_decision_changed",
        "intervention",
    }
    expected_intervention_fields = {
        "hook_calls",
        "selected_position_count",
        "prompt_final_index",
        "residual_norm",
        "actual_perturbation_norm",
        "requested_relative_perturbation_norm",
        "realized_relative_perturbation_norm",
        "absolute_relative_perturbation_error",
        "maximum_abs_application_coordinate_error",
        "maximum_abs_relative_application_coordinate_error",
        "sign",
        "requested_delta_norm",
        "delta_float32_sha256",
    }
    if (
        not hasattr(perturbation, "dtype")
        or perturbation.dtype != perturbation.new_empty(()).float().dtype
        or perturbation.device.type != "cpu"
        or perturbation.ndim != 1
        or not bool(perturbation.isfinite().all().item())
    ):
        raise RuntimeError("CPNG observation perturbation is not finite CPU float32")
    perturbation_sha256 = base.v3.tensor_float32_sha256(perturbation)
    requested_norm = float(perturbation.norm().item())
    if len(observations) != len(specifications):
        raise RuntimeError("CPNG observation coverage differs")
    for observed, specification in zip(observations, specifications, strict=True):
        if not isinstance(observed, Mapping) or set(observed) != expected_observation_fields:
            raise RuntimeError("CPNG observation schema differs")
        form = specification["form"]
        frozen_record = specification["frozen_record"]
        expected_positive_id = (
            int(frozen_record["choice_a_token_id"])
            if str(form["positive_label"]) == "A"
            else int(frozen_record["choice_b_token_id"])
        )
        expected_negative_id = (
            int(frozen_record["choice_a_token_id"])
            if str(form["negative_label"]) == "A"
            else int(frozen_record["choice_b_token_id"])
        )
        identity = {
            "constraint_id": str(specification["constraint_id"]),
            "family": str(specification["family"]),
            "form_id": str(form["form_id"]),
            "preserve_first": bool(specification["preserve_first"]),
            "sign": int(specification["sign"]),
            "positive_id": expected_positive_id,
            "negative_id": expected_negative_id,
            "baseline_actual_token_id": int(frozen_record["baseline_greedy_token_id"]),
            "baseline_semantic_choice": str(frozen_record["baseline_actual_semantic_choice"]),
        }
        if any(observed.get(field) != value for field, value in identity.items()):
            raise RuntimeError("CPNG observation identity or order differs")
        if observed.get("choice_boundary_evidence_sha256") != str(
            frozen_record["choice_boundary_evidence_sha256"]
        ):
            raise RuntimeError("CPNG observation boundary identity differs")
        desired_id = (
            expected_positive_id
            if str(specification["family"]) == "self" and int(specification["sign"]) == 1
            else (
                expected_negative_id
                if str(specification["family"]) == "self"
                else int(frozen_record["baseline_greedy_token_id"])
            )
        )
        actual_token_id = observed.get("actual_token_id")
        competitor_id = observed.get("strongest_competitor_token_id")
        if (
            isinstance(actual_token_id, bool)
            or not isinstance(actual_token_id, int)
            or actual_token_id < 0
            or actual_token_id >= EXPECTED_VOCABULARY_SIZE
            or isinstance(competitor_id, bool)
            or not isinstance(competitor_id, int)
            or competitor_id < 0
            or competitor_id >= EXPECTED_VOCABULARY_SIZE
            or competitor_id == desired_id
        ):
            raise RuntimeError("CPNG observation token evidence is invalid")
        actual_semantic = trust._semantic_choice(
            actual_token_id,
            positive_id=expected_positive_id,
            negative_id=expected_negative_id,
        )
        baseline_token_id = int(frozen_record["baseline_greedy_token_id"])
        baseline_semantic = str(frozen_record["baseline_actual_semantic_choice"])
        numerical = (
            observed.get("required_margin"),
            observed.get("constraint_value"),
            observed.get("semantic_desired_gap"),
            observed.get("full_vocabulary_kl_changed_to_baseline"),
        )
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in numerical
            )
            or float(observed["full_vocabulary_kl_changed_to_baseline"]) < 0.0
        ):
            raise RuntimeError("CPNG observation numerical evidence is invalid")
        constraint_value = float(observed["constraint_value"])
        if (
            actual_token_id != desired_id
            and (competitor_id != actual_token_id or constraint_value > 1e-6)
        ) or (actual_token_id == desired_id and constraint_value < -1e-6):
            raise RuntimeError("CPNG observation argmax and margin evidence disagree")
        derived = {
            "required_margin": float(specification["required_margin"]),
            "desired_token_id": desired_id,
            "actual_semantic_choice": actual_semantic,
            "new_other_output": baseline_semantic != "OTHER" and actual_semantic == "OTHER",
            "exact_token_changed": actual_token_id != baseline_token_id,
            "semantic_decision_changed": actual_semantic != baseline_semantic,
        }
        if any(observed.get(field) != value for field, value in derived.items()):
            raise RuntimeError("CPNG observation derived semantics differ")
        intervention = observed.get("intervention")
        if (
            not isinstance(intervention, Mapping)
            or set(intervention) != expected_intervention_fields
            or intervention.get("sign") != int(specification["sign"])
            or intervention.get("delta_float32_sha256") != perturbation_sha256
        ):
            raise RuntimeError("CPNG observation is not bound to the candidate perturbation")
        intervention_numbers = {
            field: intervention.get(field)
            for field in expected_intervention_fields
            - {
                "hook_calls",
                "selected_position_count",
                "prompt_final_index",
                "sign",
                "delta_float32_sha256",
            }
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in intervention_numbers.values()
        ):
            raise RuntimeError("CPNG intervention numerical evidence is invalid")
        residual_norm = float(intervention["residual_norm"])
        actual_norm = float(intervention["actual_perturbation_norm"])
        realized_norm = float(intervention["realized_relative_perturbation_norm"])
        requested_relative_norm = float(intervention["requested_relative_perturbation_norm"])
        maximum_abs_error = float(intervention["maximum_abs_application_coordinate_error"])
        if (
            intervention.get("hook_calls") != 1
            or intervention.get("selected_position_count") != 1
            or intervention.get("prompt_final_index") != int(frozen_record["prompt_final_index"])
            or residual_norm <= 0.0
            or not math.isclose(
                residual_norm,
                float(frozen_record["residual_norm"]),
                rel_tol=2e-6,
                abs_tol=2e-7,
            )
            or not math.isclose(
                float(intervention["requested_delta_norm"]),
                requested_norm,
                rel_tol=2e-7,
                abs_tol=2e-8,
            )
            or not math.isclose(
                requested_relative_norm,
                requested_norm,
                rel_tol=2e-7,
                abs_tol=2e-8,
            )
            or not math.isclose(
                actual_norm,
                residual_norm * realized_norm,
                rel_tol=2e-6,
                abs_tol=2e-7,
            )
            or not math.isclose(
                float(intervention["absolute_relative_perturbation_error"]),
                abs(realized_norm - requested_relative_norm),
                rel_tol=2e-6,
                abs_tol=2e-7,
            )
            or not math.isclose(
                float(intervention["maximum_abs_relative_application_coordinate_error"]),
                maximum_abs_error / residual_norm,
                rel_tol=2e-6,
                abs_tol=2e-7,
            )
        ):
            raise RuntimeError("CPNG intervention arithmetic or frozen scale differs")


def _validate_candidate_row(
    torch: Any,
    *,
    row: Mapping[str, Any],
    candidate: Mapping[str, Any],
    delta: Any | None,
    frozen: Mapping[str, Any],
    study_identity_sha256: str,
) -> None:
    _verify_internal_hash(row, hash_field="row_sha256")
    expected_keys = ROW_ENVELOPE_FIELDS | set(candidate)
    if set(row) != expected_keys:
        raise RuntimeError("CPNG row schema differs from its reconstructed candidate")
    if any(row.get(field) != value for field, value in candidate.items()):
        raise RuntimeError("CPNG row candidate identity differs")
    if (
        row.get("schema_version") != ROW_SCHEMA
        or row.get("development_only") is not True
        or row.get("study_identity_sha256") != study_identity_sha256
    ):
        raise RuntimeError("CPNG row study identity differs")

    status = str(row.get("evaluation_status"))
    if candidate.get("construction_status") != "constructed":
        expected = {
            "evaluation_sha256": None,
            "evaluation_status": "not_evaluated_construction_failed",
            "terminal_candidate": False,
            "success": False,
            "terminal_gate": None,
            "matched_other_passed": None,
            "matched_other_mean_kl": None,
            "null_passed": None,
            "self_minus_matched_other_effect": None,
            "evaluation": None,
        }
        if any(row.get(field) != value for field, value in expected.items()):
            raise RuntimeError("CPNG construction-failed row evidence differs")
        return

    if (
        not torch.is_tensor(delta)
        or delta.dtype != torch.float32
        or delta.device.type != "cpu"
        or delta.shape != (1024,)
        or not bool(torch.isfinite(delta).all().item())
        or base.v3.tensor_float32_sha256(delta) != candidate.get("perturbation_sha256")
    ):
        raise RuntimeError("CPNG row candidate perturbation differs")

    evaluation = row.get("evaluation")
    if status == "failed_closed":
        if not isinstance(evaluation, Mapping) or set(evaluation) != {
            "evaluation_status",
            "failure_type",
            "failure_message",
            "evaluation_sha256",
        }:
            raise RuntimeError("CPNG failed-closed evaluation schema differs")
        if (
            evaluation.get("evaluation_status") != "failed_closed"
            or evaluation.get("failure_type") != "CandidateLocalNumericalFailure"
            or not isinstance(evaluation.get("failure_message"), str)
            or str(evaluation["failure_message"]) not in CANDIDATE_LOCAL_NUMERICAL_FAILURE_MESSAGES
            or evaluation.get("evaluation_sha256")
            != canonical_sha256(
                {key: value for key, value in evaluation.items() if key != "evaluation_sha256"}
            )
            or row.get("evaluation_sha256") != evaluation.get("evaluation_sha256")
        ):
            raise RuntimeError("CPNG failed-closed evaluation identity differs")
        expected = {
            "terminal_candidate": False,
            "success": False,
            "terminal_gate": None,
            "matched_other_passed": None,
            "matched_other_mean_kl": None,
            "null_passed": None,
            "self_minus_matched_other_effect": None,
        }
        if any(row.get(field) != value for field, value in expected.items()):
            raise RuntimeError("CPNG failed-closed row derived fields differ")
        return
    if status != "evaluated" or not isinstance(evaluation, Mapping):
        raise RuntimeError("CPNG constructed row has an invalid evaluation status")
    if set(evaluation) != {
        "terminal_candidate",
        "stage_one_target_success",
        "primary",
        "null_certificate",
        "self_minus_matched_other_effect",
        "evaluation_sha256",
    }:
        raise RuntimeError("CPNG evaluated evidence schema differs")
    if evaluation.get("evaluation_sha256") != canonical_sha256(
        {key: value for key, value in evaluation.items() if key != "evaluation_sha256"}
    ) or row.get("evaluation_sha256") != evaluation.get("evaluation_sha256"):
        raise RuntimeError("CPNG row evaluation hash differs")
    primary = evaluation.get("primary")
    if not isinstance(primary, Mapping) or set(primary) != {
        "observations",
        "self_application",
        "matched_other",
        "terminal_gate",
    }:
        raise RuntimeError("CPNG row primary evidence schema differs")
    observations = primary.get("observations")
    if not isinstance(observations, list):
        raise TypeError("CPNG row primary observations differ")
    specifications = trust._constraint_specifications(
        case_id=str(row["case_id"]),
        assignment=int(row["assignment"]),
        frozen=frozen,
        optimizer={
            "target_margin_logit": EXPECTED_TARGET_MARGIN,
            "matched_other_margin_logit": EXPECTED_MATCHED_OTHER_MARGIN,
        },
    )
    _validate_observation_sequence(
        observations,
        specifications=specifications,
        perturbation=delta,
    )
    self_rows = [item for item in observations if item.get("family") == "self"]
    other_rows = [item for item in observations if item.get("family") == "matched_other"]
    rebuilt_self = trust._application_report(self_rows, group="self")
    rebuilt_other = trust._protection_report(
        other_rows, limits=EXPECTED_PROTECTED_LIMITS, group="matched_other"
    )
    rebuilt_terminal = trust._terminal_from_self_observations(
        SimpleNamespace(torch=torch),
        observations=observations,
        target_margin=EXPECTED_TARGET_MARGIN,
    )
    rebuilt_null = trust._null_certificate(
        torch,
        delta=delta,
        global_basis=frozen["global_nuisance_basis"],
        absolute_cap=max(RESIDUAL_RELATIVE_L2_CAP_GRID),
    )
    rebuilt_effect = _specificity_effect({"observations": observations})
    terminal_candidate = all(
        (
            bool(rebuilt_terminal["passes_terminal_gate"]),
            bool(rebuilt_self["passes"]),
            bool(rebuilt_other["passes"]),
            bool(rebuilt_null["passes"]),
        )
    )
    if (
        primary.get("self_application") != rebuilt_self
        or primary.get("matched_other") != rebuilt_other
        or primary.get("terminal_gate") != rebuilt_terminal
        or evaluation.get("null_certificate") != rebuilt_null
        or not math.isclose(
            float(evaluation.get("self_minus_matched_other_effect")),
            rebuilt_effect,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or evaluation.get("terminal_candidate") != terminal_candidate
        or evaluation.get("stage_one_target_success") != terminal_candidate
        or row.get("terminal_candidate") != terminal_candidate
        or row.get("success") != terminal_candidate
        or row.get("terminal_gate") != rebuilt_terminal
        or row.get("matched_other_passed") != bool(rebuilt_other["passes"])
        or not math.isclose(
            float(row.get("matched_other_mean_kl")),
            float(rebuilt_other["full_vocabulary_kl_changed_to_baseline"]["mean"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or row.get("null_passed") != bool(rebuilt_null["passes"])
        or not math.isclose(
            float(row.get("self_minus_matched_other_effect")),
            rebuilt_effect,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise RuntimeError("CPNG row derived values differ")


def _validate_stage_two_evidence(
    torch: Any,
    *,
    audits: Sequence[Mapping[str, Any]],
    provisional: Mapping[str, Any] | None,
    candidate_map: Mapping[tuple[str, int, int], Mapping[str, Any]],
    delta_map: Mapping[tuple[str, int, int], Any],
    frozen: Mapping[str, Any],
    study_identity_sha256: str,
    ledger: PersistentComputeLedger,
    stage_one_event_count: int,
) -> None:
    assignments = _case_assignments(frozen)
    entries = []
    if STAGE_TWO_CHECKPOINT_ROOT.exists():
        if not STAGE_TWO_CHECKPOINT_ROOT.is_dir():
            raise RuntimeError("Stage-two checkpoint root is not a directory")
        entries = list(STAGE_TWO_CHECKPOINT_ROOT.iterdir())
    if provisional is None:
        if audits or entries:
            raise RuntimeError("CPNG has Stage-two evidence without a provisional candidate")
        if stage_one_event_count != len(ledger.events):
            raise RuntimeError("CPNG ledger contains work after Stage one without an audit")
        return
    if len(audits) != len(assignments):
        raise RuntimeError("CPNG Stage-two audit coverage differs")
    expected_paths = {
        _case_checkpoint_path(STAGE_TWO_CHECKPOINT_ROOT, case_id, assignment)
        for case_id, assignment in assignments
    }
    if set(entries) != expected_paths or any(not path.is_file() for path in entries):
        raise RuntimeError("CPNG Stage-two checkpoint file coverage differs")

    selected_grid_index = int(provisional["grid_index"])
    previous_event_count = stage_one_event_count
    specifications = trust._nuisance_specifications(frozen)
    for audit, (case_id, assignment) in zip(audits, assignments, strict=True):
        if not isinstance(audit, Mapping):
            raise TypeError("CPNG Stage-two audit is not an object")
        if audit.get("case_id") != case_id or audit.get("assignment") != assignment:
            raise RuntimeError("CPNG Stage-two audit identity or order differs")
        _verify_internal_hash(audit, hash_field="audit_sha256")
        key = (case_id, assignment, selected_grid_index)
        candidate = candidate_map.get(key)
        delta = delta_map.get(key)
        if (
            not isinstance(candidate, Mapping)
            or candidate.get("construction_status") != "constructed"
            or not torch.is_tensor(delta)
            or base.v3.tensor_float32_sha256(delta) != candidate.get("perturbation_sha256")
        ):
            raise RuntimeError("CPNG Stage-two candidate perturbation is unavailable")
        if audit.get("evaluated") is True:
            if set(audit) != {
                "case_id",
                "assignment",
                "evaluated",
                "passes",
                "report",
                "observations",
                "audit_sha256",
            }:
                raise RuntimeError("CPNG evaluated Stage-two audit schema differs")
            observations = audit.get("observations")
            if not isinstance(observations, list):
                raise RuntimeError("CPNG evaluated Stage-two observations are missing")
            _validate_observation_sequence(
                observations,
                specifications=specifications,
                perturbation=delta,
            )
            rebuilt = trust._protection_report(
                observations,
                limits=EXPECTED_PROTECTED_LIMITS,
                group="nuisance_fit",
            )
            if audit.get("report") != rebuilt or audit.get("passes") != bool(rebuilt["passes"]):
                raise RuntimeError("CPNG Stage-two audit derived report differs")
        else:
            if set(audit) != {
                "case_id",
                "assignment",
                "evaluated",
                "passes",
                "failure_type",
                "failure_message",
                "audit_sha256",
            }:
                raise RuntimeError("CPNG failed-closed Stage-two audit schema differs")
            if (
                audit.get("evaluated") is not False
                or audit.get("passes") is not False
                or audit.get("failure_type") != "CandidateLocalNumericalFailure"
                or not isinstance(audit.get("failure_message"), str)
                or str(audit["failure_message"]) not in CANDIDATE_LOCAL_NUMERICAL_FAILURE_MESSAGES
            ):
                raise RuntimeError("CPNG failed-closed Stage-two audit identity differs")

        checkpoint_path = _case_checkpoint_path(STAGE_TWO_CHECKPOINT_ROOT, case_id, assignment)
        checkpoint = _load_json(checkpoint_path)
        _verify_internal_hash(checkpoint, hash_field="checkpoint_sha256")
        if (
            set(checkpoint)
            != {
                "schema_version",
                "study_identity_sha256",
                "case_id",
                "assignment",
                "audit",
                "audit_sha256",
                "ledger",
                "ledger_file_sha256",
                "checkpoint_sha256",
            }
            or checkpoint.get("schema_version") != STAGE_TWO_CASE_SCHEMA
            or checkpoint.get("study_identity_sha256") != study_identity_sha256
            or checkpoint.get("case_id") != case_id
            or checkpoint.get("assignment") != assignment
            or checkpoint.get("audit") != audit
            or checkpoint.get("audit_sha256") != audit["audit_sha256"]
        ):
            raise RuntimeError("CPNG Stage-two checkpoint differs")
        _validate_checkpoint_ledger_prefix(ledger, checkpoint)
        event_count = int(checkpoint["ledger"]["event_count"])
        if event_count <= previous_event_count:
            raise RuntimeError("CPNG Stage-two checkpoint ledger order differs")
        previous_event_count = event_count
    if previous_event_count != len(ledger.events):
        raise RuntimeError("CPNG ledger contains work after the final Stage-two checkpoint")


def _load_completed_results(study_identity: Mapping[str, Any]) -> dict[str, Any]:
    if ROWS_PATH.exists() != SUMMARY_PATH.exists():
        raise RuntimeError("CPNG completed results are only partially present")
    if not ROWS_PATH.is_file():
        raise RuntimeError("CPNG completed results are absent")
    import torch

    summary = _load_json(SUMMARY_PATH)
    _verify_internal_hash(summary, hash_field="summary_sha256")
    if set(summary) != {
        "schema_version",
        "development_only",
        "status",
        "study_identity",
        "candidate_attempt_count",
        "unique_perturbation_evaluation_count",
        "deduplicated_candidate_count",
        "compute",
        "candidate_summaries",
        "provisional_selected_candidate",
        "safe_candidate",
        "safe_candidate_exists",
        "effective_candidate_selected",
        "no_safe_effective_candidate",
        "selected_candidate",
        "no_safe_candidate",
        "selection_rule",
        "unrelated_audit_rule",
        "method_wide_unrelated_passed",
        "provisional_is_effective",
        "safe_but_ineffective",
        "unrelated_audits",
        "calibration_compute_ledger",
        "calibration_compute_ledger_sha256",
        "stage_one_checkpoint_sha256",
        "claim_boundary",
        "rows_sha256",
        "summary_sha256",
    }:
        raise RuntimeError("CPNG completed summary schema differs")
    rows = _read_jsonl_strict(ROWS_PATH)
    if len(rows) != 384 or summary.get("candidate_attempt_count") != 384:
        raise RuntimeError("CPNG completed row count differs")
    if summary.get("rows_sha256") != canonical_sha256(rows):
        raise RuntimeError("CPNG completed rows hash differs")
    if (
        summary.get("schema_version") != SUMMARY_SCHEMA
        or summary.get("development_only") is not True
        or summary.get("status") != "complete"
        or summary.get("study_identity") != study_identity
    ):
        raise RuntimeError("CPNG completed summary identity differs")
    frozen = trust._load_frozen_inputs(torch)
    capture = _load_capture(torch, study_identity)
    constructions = _load_constructions(torch, study_identity, frozen=frozen, capture=capture)
    candidate_map, delta_map = _candidate_maps_for_validation(
        torch, frozen=frozen, constructions=constructions
    )
    if not CALIBRATION_LEDGER_PATH.is_file():
        raise RuntimeError("CPNG completed result lacks its calibration compute ledger")
    ledger = PersistentComputeLedger(
        path=CALIBRATION_LEDGER_PATH,
        phase="calibration",
        study_identity_sha256=str(study_identity["identity_sha256"]),
        maximum_forwards=EXPECTED_MAXIMUM_CALIBRATION_FORWARD_EVALUATIONS,
        maximum_backwards=0,
        prior_phase_ledger_sha256=file_sha256(CAPTURE_LEDGER_PATH),
    )
    expected_ids = set(candidate_map)
    observed_ids = {
        (str(row.get("case_id")), int(row.get("assignment", -1)), int(row.get("grid_index", -1)))
        for row in rows
    }
    if observed_ids != expected_ids or len(observed_ids) != len(rows):
        raise RuntimeError("CPNG completed row coverage differs")
    for row in rows:
        key = (str(row["case_id"]), int(row["assignment"]), int(row["grid_index"]))
        candidate = candidate_map[key]
        _validate_candidate_row(
            torch,
            row=row,
            candidate=candidate,
            delta=delta_map.get(key),
            frozen=frozen,
            study_identity_sha256=str(study_identity["identity_sha256"]),
        )
    selection = _selection_summary(rows)
    if (
        summary.get("candidate_summaries") != selection["candidate_summaries"]
        or summary.get("provisional_selected_candidate") != selection["selected_candidate"]
        or summary.get("selection_rule") != selection["selection_rule"]
    ):
        raise RuntimeError("CPNG completed selection differs")
    checkpoint_rows = []
    checkpoints = []
    checkpoint_unique_count = 0
    checkpoint_duplicate_count = 0
    prior_stage_one_event_count = -1
    assignments = _case_assignments(frozen)
    if not STAGE_ONE_CHECKPOINT_ROOT.is_dir():
        raise RuntimeError("CPNG completed result lacks its Stage-one checkpoint directory")
    expected_stage_one_paths = {
        _case_checkpoint_path(STAGE_ONE_CHECKPOINT_ROOT, case_id, assignment)
        for case_id, assignment in assignments
    }
    stage_one_entries = set(STAGE_ONE_CHECKPOINT_ROOT.iterdir())
    if stage_one_entries != expected_stage_one_paths or any(
        not path.is_file() for path in stage_one_entries
    ):
        raise RuntimeError("CPNG Stage-one checkpoint file coverage differs")
    for case_id, assignment in assignments:
        checkpoint = _load_stage_one_case_checkpoint(
            study_identity_sha256=str(study_identity["identity_sha256"]),
            case_id=case_id,
            assignment=assignment,
        )
        if checkpoint is None:
            raise RuntimeError("CPNG completed result lacks a Stage-one case checkpoint")
        candidates = [candidate_map[(case_id, assignment, grid_index)] for grid_index in range(48)]
        constructed = [
            candidate
            for candidate in candidates
            if candidate["construction_status"] == "constructed"
        ]
        reconstructed_unique_count = len(
            {str(candidate["deduplication_key"]) for candidate in constructed}
        )
        reconstructed_duplicate_count = len(constructed) - reconstructed_unique_count
        if (
            checkpoint.get("unique_perturbation_evaluation_count") != reconstructed_unique_count
            or checkpoint.get("deduplicated_candidate_count") != reconstructed_duplicate_count
        ):
            raise RuntimeError("CPNG Stage-one checkpoint candidate counts differ")
        _validate_checkpoint_ledger_prefix(ledger, checkpoint)
        checkpoint_event_count = int(checkpoint["ledger"]["event_count"])
        if checkpoint_event_count < prior_stage_one_event_count:
            raise RuntimeError("CPNG Stage-one checkpoint ledger order differs")
        prior_stage_one_event_count = checkpoint_event_count
        checkpoints.append(checkpoint)
        checkpoint_unique_count += reconstructed_unique_count
        checkpoint_duplicate_count += reconstructed_duplicate_count
        checkpoint_rows.extend(checkpoint["rows"])
    if checkpoint_rows != rows:
        raise RuntimeError("CPNG completed rows differ from Stage-one checkpoints")
    if not STAGE_ONE_COMPLETE_PATH.is_file():
        raise RuntimeError("CPNG completed result lacks its Stage-one freeze checkpoint")
    stage_one_complete = _load_json(STAGE_ONE_COMPLETE_PATH)
    _verify_internal_hash(stage_one_complete, hash_field="checkpoint_sha256")
    if (
        set(stage_one_complete)
        != {
            "schema_version",
            "study_identity_sha256",
            "row_count",
            "rows_sha256",
            "selection",
            "selection_sha256",
            "unique_perturbation_evaluation_count",
            "deduplicated_candidate_count",
            "ledger",
            "ledger_file_sha256",
            "checkpoint_sha256",
        }
        or stage_one_complete.get("schema_version") != STAGE_ONE_COMPLETE_SCHEMA
        or stage_one_complete.get("study_identity_sha256") != study_identity["identity_sha256"]
        or stage_one_complete.get("row_count") != 384
        or stage_one_complete.get("rows_sha256") != canonical_sha256(rows)
        or stage_one_complete.get("selection") != selection
        or stage_one_complete.get("selection_sha256") != canonical_sha256(selection)
        or stage_one_complete.get("unique_perturbation_evaluation_count") != checkpoint_unique_count
        or stage_one_complete.get("deduplicated_candidate_count") != checkpoint_duplicate_count
        or stage_one_complete.get("ledger") != checkpoints[-1]["ledger"]
        or summary.get("stage_one_checkpoint_sha256") != stage_one_complete.get("checkpoint_sha256")
    ):
        raise RuntimeError("CPNG Stage-one freeze checkpoint differs")
    _validate_checkpoint_ledger_prefix(ledger, stage_one_complete)
    audits = summary.get("unrelated_audits")
    if not isinstance(audits, list):
        raise TypeError("CPNG completed audits are missing")
    _validate_stage_two_evidence(
        torch,
        audits=audits,
        provisional=selection["selected_candidate"],
        candidate_map=candidate_map,
        delta_map=delta_map,
        frozen=frozen,
        study_identity_sha256=str(study_identity["identity_sha256"]),
        ledger=ledger,
        stage_one_event_count=int(stage_one_complete["ledger"]["event_count"]),
    )
    _validate_calibration_ledger_work_ids(
        ledger,
        rows=rows,
        audits=audits,
        provisional=selection["selected_candidate"],
        candidate_map=candidate_map,
        frozen=frozen,
    )
    finalization = _finalize_provisional_selection(selection["selected_candidate"], audits)
    for field in (
        "safe_candidate",
        "safe_candidate_exists",
        "no_safe_candidate",
        "effective_candidate_selected",
        "no_safe_effective_candidate",
        "selected_candidate",
        "method_wide_unrelated_passed",
        "provisional_is_effective",
        "safe_but_ineffective",
    ):
        if summary.get(field) != finalization[field]:
            raise RuntimeError(f"CPNG completed {field} differs")
    expected_compute = {
        "capture_forward_evaluations": EXPECTED_MAXIMUM_CAPTURE_FORWARD_EVALUATIONS,
        "capture_backward_evaluations": EXPECTED_MAXIMUM_CAPTURE_BACKWARD_EVALUATIONS,
        "calibration_forward_evaluations": ledger.forward_evaluations,
        "calibration_backward_evaluations": ledger.backward_evaluations,
        "total_experiment_forward_evaluations": (
            EXPECTED_MAXIMUM_CAPTURE_FORWARD_EVALUATIONS + ledger.forward_evaluations
        ),
        "total_experiment_backward_evaluations": (
            EXPECTED_MAXIMUM_CAPTURE_BACKWARD_EVALUATIONS + ledger.backward_evaluations
        ),
        "external_model_judges": EXPECTED_EXTERNAL_MODEL_JUDGES,
        "external_api_calls": EXPECTED_EXTERNAL_API_CALLS,
    }
    if (
        summary.get("unique_perturbation_evaluation_count") != checkpoint_unique_count
        or summary.get("deduplicated_candidate_count") != checkpoint_duplicate_count
        or summary.get("calibration_compute_ledger") != ledger.snapshot()
        or summary.get("calibration_compute_ledger_sha256") != file_sha256(CALIBRATION_LEDGER_PATH)
        or summary.get("compute") != expected_compute
    ):
        raise RuntimeError("CPNG completed compute accounting differs")
    return summary


def run_calibrate() -> dict[str, Any]:
    preflight = run_preflight()
    if ROWS_PATH.exists() != SUMMARY_PATH.exists():
        raise RuntimeError("CPNG calibration outputs are only partially present")
    if ROWS_PATH.exists():
        return _load_completed_results(preflight["study_identity"])
    lock = _load_lock()
    import torch

    frozen = trust._load_frozen_inputs(torch)
    capture = _load_capture(torch, preflight["study_identity"])
    constructions = _load_constructions(
        torch,
        preflight["study_identity"],
        frozen=frozen,
        capture=capture,
    )
    ledger = PersistentComputeLedger(
        path=CALIBRATION_LEDGER_PATH,
        phase="calibration",
        study_identity_sha256=preflight["study_identity"]["identity_sha256"],
        maximum_forwards=EXPECTED_MAXIMUM_CALIBRATION_FORWARD_EVALUATIONS,
        maximum_backwards=0,
        prior_phase_ledger_sha256=file_sha256(CAPTURE_LEDGER_PATH),
    )
    if STAGE_TWO_CHECKPOINT_ROOT.exists() and any(STAGE_TWO_CHECKPOINT_ROOT.iterdir()):
        raise RuntimeError(
            "interrupted Stage-two audit is failed closed to avoid replaying shared baselines"
        )
    entries_by_case: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for entry in constructions["entries"]:
        entries_by_case.setdefault((str(entry["case_id"]), int(entry["assignment"])), []).append(
            entry
        )
    budget = trust.EvaluationBudget(
        maximum_forward_evaluations=EXPECTED_MAXIMUM_CALIBRATION_FORWARD_EVALUATIONS,
        maximum_backward_evaluations=0,
        forward_evaluations=ledger.forward_evaluations,
        backward_evaluations=ledger.backward_evaluations,
    )
    current_work_id: dict[str, str | None] = {"value": None}

    def persist_budget(snapshot: Mapping[str, int]) -> None:
        work_id = current_work_id["value"]
        if work_id is None:
            raise RuntimeError("model operation has no active immutable work ID")
        ledger.sync_budget(
            snapshot,
            work_id=f"{work_id}:event={len(ledger.events)}",
        )

    budget.set_on_change(persist_budget)
    backend = None

    def require_backend() -> Any:
        nonlocal backend
        if backend is None:
            backend = trust.load_backend()
        return backend

    baseline_cache: dict[str, dict[str, Any]] = {}
    rows = []
    unique_evaluation_count = 0
    duplicate_candidate_count = 0
    perturbations_by_attempt: dict[tuple[str, int, int], Any] = {}
    calibration_started = time.monotonic()
    case_assignments = _case_assignments(frozen)
    if STAGE_ONE_CHECKPOINT_ROOT.exists():
        if not STAGE_ONE_CHECKPOINT_ROOT.is_dir():
            raise RuntimeError("Stage-one checkpoint root is not a directory")
        allowed_stage_one_paths = {
            _case_checkpoint_path(STAGE_ONE_CHECKPOINT_ROOT, case_id, assignment)
            for case_id, assignment in case_assignments
        }
        stage_one_entries = set(STAGE_ONE_CHECKPOINT_ROOT.iterdir())
        if not stage_one_entries <= allowed_stage_one_paths or any(
            not path.is_file() for path in stage_one_entries
        ):
            raise RuntimeError("Stage-one checkpoint file coverage differs")
    checkpoints = []
    missing_seen = False
    for case_id, assignment in case_assignments:
        checkpoint = _load_stage_one_case_checkpoint(
            study_identity_sha256=preflight["study_identity"]["identity_sha256"],
            case_id=case_id,
            assignment=assignment,
        )
        if checkpoint is None:
            missing_seen = True
        elif missing_seen:
            raise RuntimeError("Stage-one checkpoints are not a contiguous case prefix")
        checkpoints.append(checkpoint)
    completed_checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint is not None]
    expected_ledger = (
        completed_checkpoints[-1]["ledger"]
        if completed_checkpoints
        else {
            "forward_evaluations": 0,
            "backward_evaluations": 0,
            "event_count": 0,
            "head_event_sha256": None,
        }
    )
    if ledger.snapshot() != expected_ledger:
        raise RuntimeError("orphan calibration reservation cannot be replayed")
    previous_checkpoint_event_count = -1
    for completed_checkpoint in completed_checkpoints:
        _validate_checkpoint_ledger_prefix(ledger, completed_checkpoint)
        checkpoint_event_count = int(completed_checkpoint["ledger"]["event_count"])
        if checkpoint_event_count < previous_checkpoint_event_count:
            raise RuntimeError("Stage-one checkpoint ledger order differs")
        previous_checkpoint_event_count = checkpoint_event_count
    for case_index, (case_id, assignment) in enumerate(case_assignments, start=1):
        factors, _diagnostics, _manifest = _protected_metric_factors(
            torch, frozen=frozen, case_id=case_id, assignment=assignment
        )
        entries = entries_by_case[(case_id, assignment)]
        factors_by_ridge = {float(value): factors for value in FISHER_RIDGE_MULTIPLIER_GRID}
        candidates, unique = _candidate_perturbations(
            torch, entries=entries, factors_by_ridge=factors_by_ridge
        )
        for candidate in candidates:
            if candidate["construction_status"] == "constructed":
                perturbations_by_attempt[(case_id, assignment, int(candidate["grid_index"]))] = (
                    unique[str(candidate["deduplication_key"])]
                )
        checkpoint = checkpoints[case_index - 1]
        if checkpoint is not None:
            expected_duplicate_count = sum(
                candidate["construction_status"] == "constructed" for candidate in candidates
            ) - len(unique)
            if (
                checkpoint.get("unique_perturbation_evaluation_count") != len(unique)
                or checkpoint.get("deduplicated_candidate_count") != expected_duplicate_count
            ):
                raise RuntimeError("Stage-one checkpoint candidate counts differ")
            candidates_by_grid = {
                int(candidate["grid_index"]): candidate for candidate in candidates
            }
            for checkpoint_row in checkpoint["rows"]:
                candidate = candidates_by_grid[int(checkpoint_row["grid_index"])]
                delta = (
                    unique.get(str(candidate["deduplication_key"]))
                    if candidate["construction_status"] == "constructed"
                    else None
                )
                _validate_candidate_row(
                    torch,
                    row=checkpoint_row,
                    candidate=candidate,
                    delta=delta,
                    frozen=frozen,
                    study_identity_sha256=str(preflight["study_identity"]["identity_sha256"]),
                )
            rows.extend(checkpoint["rows"])
            unique_evaluation_count += int(checkpoint["unique_perturbation_evaluation_count"])
            duplicate_candidate_count += int(checkpoint["deduplicated_candidate_count"])
            continue
        evaluations = {}
        for deduplication_key, delta in unique.items():
            current_work_id["value"] = (
                f"stage_one:{case_id}:assignment={assignment}:{deduplication_key}"
            )
            try:
                evaluation = _evaluate_unique_perturbation(
                    require_backend(),
                    case_id=case_id,
                    assignment=assignment,
                    delta=delta,
                    frozen=frozen,
                    lock=lock,
                    baseline_cache=baseline_cache,
                    budget=budget,
                )
            except CandidateLocalNumericalFailure as error:
                if not _is_allowlisted_candidate_failure(error, phase="evaluation"):
                    raise RuntimeError(
                        "undeclared CPNG candidate-local evaluation failure"
                    ) from error
                failure_evidence = {
                    "evaluation_status": "failed_closed",
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                }
                evaluation = {
                    **failure_evidence,
                    "evaluation_sha256": canonical_sha256(failure_evidence),
                }
            finally:
                current_work_id["value"] = None
            evaluations[deduplication_key] = evaluation
            unique_evaluation_count += 1
        duplicate_candidate_count += sum(
            candidate["construction_status"] == "constructed" for candidate in candidates
        ) - len(unique)
        case_rows = []
        for candidate in candidates:
            if candidate["construction_status"] != "constructed":
                row = {
                    "schema_version": ROW_SCHEMA,
                    "development_only": True,
                    "study_identity_sha256": preflight["study_identity"]["identity_sha256"],
                    "case_id": case_id,
                    "assignment": assignment,
                    **candidate,
                    "evaluation_sha256": None,
                    "evaluation_status": "not_evaluated_construction_failed",
                    "terminal_candidate": False,
                    "success": False,
                    "terminal_gate": None,
                    "matched_other_passed": None,
                    "matched_other_mean_kl": None,
                    "null_passed": None,
                    "self_minus_matched_other_effect": None,
                    "evaluation": None,
                }
                row["row_sha256"] = canonical_sha256(row)
                rows.append(row)
                case_rows.append(row)
                continue
            deduplication_key = str(candidate["deduplication_key"])
            evaluation = evaluations[deduplication_key]
            if evaluation.get("evaluation_status") == "failed_closed":
                row = {
                    "schema_version": ROW_SCHEMA,
                    "development_only": True,
                    "study_identity_sha256": preflight["study_identity"]["identity_sha256"],
                    "case_id": case_id,
                    "assignment": assignment,
                    **candidate,
                    "evaluation_sha256": evaluation["evaluation_sha256"],
                    "evaluation_status": "failed_closed",
                    "terminal_candidate": False,
                    "success": False,
                    "terminal_gate": None,
                    "matched_other_passed": None,
                    "matched_other_mean_kl": None,
                    "null_passed": None,
                    "self_minus_matched_other_effect": None,
                    "evaluation": evaluation,
                }
                row["row_sha256"] = canonical_sha256(row)
                rows.append(row)
                case_rows.append(row)
                continue
            primary = evaluation["primary"]
            row = {
                "schema_version": ROW_SCHEMA,
                "development_only": True,
                "study_identity_sha256": preflight["study_identity"]["identity_sha256"],
                "case_id": case_id,
                "assignment": assignment,
                **candidate,
                "evaluation_status": "evaluated",
                "evaluation_sha256": evaluation["evaluation_sha256"],
                "terminal_candidate": bool(evaluation["terminal_candidate"]),
                "success": bool(evaluation["stage_one_target_success"]),
                "terminal_gate": primary["terminal_gate"],
                "matched_other_passed": bool(primary["matched_other"]["passes"]),
                "matched_other_mean_kl": float(
                    primary["matched_other"]["full_vocabulary_kl_changed_to_baseline"]["mean"]
                ),
                "null_passed": bool(evaluation["null_certificate"]["passes"]),
                "self_minus_matched_other_effect": float(
                    evaluation["self_minus_matched_other_effect"]
                ),
                "evaluation": evaluation,
            }
            row["row_sha256"] = canonical_sha256(row)
            rows.append(row)
            case_rows.append(row)
        candidates_by_grid = {int(candidate["grid_index"]): candidate for candidate in candidates}
        for case_row in case_rows:
            candidate = candidates_by_grid[int(case_row["grid_index"])]
            delta = (
                unique.get(str(candidate["deduplication_key"]))
                if candidate["construction_status"] == "constructed"
                else None
            )
            _validate_candidate_row(
                torch,
                row=case_row,
                candidate=candidate,
                delta=delta,
                frozen=frozen,
                study_identity_sha256=str(preflight["study_identity"]["identity_sha256"]),
            )
        _write_stage_one_case_checkpoint(
            study_identity_sha256=preflight["study_identity"]["identity_sha256"],
            case_id=case_id,
            assignment=assignment,
            rows=case_rows,
            unique_count=len(unique),
            duplicate_count=sum(
                candidate["construction_status"] == "constructed" for candidate in candidates
            )
            - len(unique),
            ledger=ledger,
        )
        print(
            _progress_line(
                phase="stage_one",
                completed=case_index,
                total=len(case_assignments),
                unique=unique_evaluation_count,
                forwards=budget.forward_evaluations,
                elapsed=time.monotonic() - calibration_started,
            ),
            flush=True,
        )
    selection = _selection_summary(rows)
    stage_one_complete = _write_stage_one_complete(
        study_identity_sha256=preflight["study_identity"]["identity_sha256"],
        rows=rows,
        selection=selection,
        unique_count=unique_evaluation_count,
        duplicate_count=duplicate_candidate_count,
        ledger=ledger,
    )
    _verify_internal_hash(stage_one_complete, hash_field="checkpoint_sha256")
    if stage_one_complete["selection"] != _selection_summary(rows):
        raise RuntimeError("Stage-one frozen selection differs after checkpoint")
    provisional = selection["selected_candidate"]
    unrelated_audits = []
    if isinstance(provisional, Mapping):
        selected_grid_index = int(provisional["grid_index"])
        for audit_index, (case_id, assignment) in enumerate(case_assignments, start=1):
            delta = perturbations_by_attempt.get((case_id, assignment, selected_grid_index))
            if delta is None:
                raise RuntimeError(
                    "provisional selection lacks its reconstructed Stage-two perturbation"
                )
            current_work_id["value"] = f"stage_two:{case_id}:assignment={assignment}"
            try:
                try:
                    nuisance = _candidate_nuisance_evaluation(
                        require_backend(),
                        frozen=frozen,
                        delta=delta,
                        layer=int(lock["model"]["layer_zero_based"]),
                        baseline_cache=baseline_cache,
                        limits=lock["protected_limits"],
                        budget=budget,
                    )
                except CandidateLocalNumericalFailure as error:
                    if not _is_allowlisted_candidate_failure(error, phase="evaluation"):
                        raise RuntimeError(
                            "undeclared CPNG Stage-two candidate-local failure"
                        ) from error
                    audit = {
                        "case_id": case_id,
                        "assignment": assignment,
                        "evaluated": False,
                        "passes": False,
                        "failure_type": type(error).__name__,
                        "failure_message": str(error),
                    }
                    audit["audit_sha256"] = canonical_sha256(audit)
                    unrelated_audits.append(audit)
                    _write_stage_two_case_checkpoint(
                        study_identity_sha256=preflight["study_identity"]["identity_sha256"],
                        audit=audit,
                        ledger=ledger,
                    )
                    print(
                        _progress_line(
                            phase="stage_two_audit",
                            completed=audit_index,
                            total=len(case_assignments),
                            unique=unique_evaluation_count,
                            forwards=budget.forward_evaluations,
                            elapsed=time.monotonic() - calibration_started,
                        ),
                        flush=True,
                    )
                    continue
            finally:
                current_work_id["value"] = None
            audit = {
                "case_id": case_id,
                "assignment": assignment,
                "evaluated": True,
                "passes": bool(nuisance["report"]["passes"]),
                "report": nuisance["report"],
                "observations": [trust._observation_log(row) for row in nuisance["observations"]],
            }
            audit["audit_sha256"] = canonical_sha256(audit)
            unrelated_audits.append(audit)
            _write_stage_two_case_checkpoint(
                study_identity_sha256=preflight["study_identity"]["identity_sha256"],
                audit=audit,
                ledger=ledger,
            )
            print(
                _progress_line(
                    phase="stage_two_audit",
                    completed=audit_index,
                    total=len(case_assignments),
                    unique=unique_evaluation_count,
                    forwards=budget.forward_evaluations,
                    elapsed=time.monotonic() - calibration_started,
                ),
                flush=True,
            )
    finalization = _finalize_provisional_selection(provisional, unrelated_audits)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "development_only": True,
        "status": "complete",
        "study_identity": preflight["study_identity"],
        "candidate_attempt_count": len(rows),
        "unique_perturbation_evaluation_count": unique_evaluation_count,
        "deduplicated_candidate_count": duplicate_candidate_count,
        "compute": {
            "capture_forward_evaluations": 48,
            "capture_backward_evaluations": 32,
            "calibration_forward_evaluations": budget.forward_evaluations,
            "calibration_backward_evaluations": budget.backward_evaluations,
            "total_experiment_forward_evaluations": (48 + budget.forward_evaluations),
            "total_experiment_backward_evaluations": 32,
            "external_model_judges": 0,
            "external_api_calls": 0,
        },
        "candidate_summaries": selection["candidate_summaries"],
        "provisional_selected_candidate": provisional,
        "safe_candidate": finalization["safe_candidate"],
        "safe_candidate_exists": finalization["safe_candidate_exists"],
        "effective_candidate_selected": finalization["effective_candidate_selected"],
        "no_safe_effective_candidate": finalization["no_safe_effective_candidate"],
        "selected_candidate": finalization["selected_candidate"],
        "no_safe_candidate": finalization["no_safe_candidate"],
        "selection_rule": selection["selection_rule"],
        "unrelated_audit_rule": (
            "Audit all eight directions at the one provisional global triple; any missing "
            "direction or unrelated-gate failure yields no safe selection, with no fallback."
        ),
        "method_wide_unrelated_passed": finalization["method_wide_unrelated_passed"],
        "provisional_is_effective": finalization["provisional_is_effective"],
        "safe_but_ineffective": finalization["safe_but_ineffective"],
        "unrelated_audits": unrelated_audits,
        "calibration_compute_ledger": ledger.snapshot(),
        "calibration_compute_ledger_sha256": file_sha256(CALIBRATION_LEDGER_PATH),
        "stage_one_checkpoint_sha256": stage_one_complete["checkpoint_sha256"],
        "claim_boundary": (
            "Development-only prompt-adaptive intervention evidence; not a natural, "
            "universal, persistent, or publication-confirmatory self-preservation mechanism."
        ),
    }
    summary["rows_sha256"] = canonical_sha256(rows)
    summary["summary_sha256"] = canonical_sha256(summary)
    _immutable_jsonl(ROWS_PATH, rows)
    _immutable_json(SUMMARY_PATH, summary)
    return _load_completed_results(preflight["study_identity"])


def _render_report(summary: Mapping[str, Any]) -> str:
    selected = summary.get("selected_candidate")
    selected_text = "No safe candidate was selectable."
    if summary.get("safe_but_ineffective") is True:
        selected_text = (
            "The provisional triple passed the method-wide unrelated audit but produced "
            "zero terminal successes, so it is safe-but-ineffective and not selected."
        )
    if isinstance(selected, Mapping):
        selected_text = (
            f"Selected grid index `{selected['grid_index']}`: ridge "
            f"`{selected['fisher_ridge_multiplier']}`, predicted coarsened next-token "
            f"KL budget `{selected['predicted_coarsened_next_token_kl_budget']}`, cap "
            f"`{selected['residual_relative_l2_cap']}`; both-order successes "
            f"`{selected['success_count']}/8`."
        )
    lines = [
        "# CPNG development report",
        "",
        "This is a development-only local result. No external judge or API was used.",
        "",
        selected_text,
        "",
        f"Candidate attempts: `{summary['candidate_attempt_count']}`.",
        f"Unique perturbations evaluated: `{summary['unique_perturbation_evaluation_count']}`.",
        f"Deduplicated candidate evaluations: `{summary['deduplicated_candidate_count']}`.",
        "",
        "## Claim boundary",
        "",
        str(summary["claim_boundary"]),
        "Only the mean-token authored-completion target contrast is label/order-free.",
        "The coarsened next-token protection metric symmetrically sees both matched-other",
        "A/B renderings and their A/B token IDs/categories, but receives no preserve/comply",
        "mapping, semantic orientation, or outcomes. Actual full-vocabulary KL is the finite gate.",
        "Exactly one provisional global triple was chosen before all eight unrelated audits;",
        "there is no fallback after an audit failure.",
        "",
        "## Complete grid",
        "",
        "| Grid | Ridge | Predicted coarsened next-token KL | Cap | Successes | Eligible | Median specificity | Mean other full-vocab KL |",
        "|---:|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for item in summary["candidate_summaries"]:
        lines.append(
            "| {grid_index} | {fisher_ridge_multiplier:.4g} | "
            "{predicted_coarsened_next_token_kl_budget:.4g} | {residual_relative_l2_cap:.4g} | "
            "{success_count}/8 | {eligible} | "
            "{median_self_minus_matched_other_effect:.6g} | {mean_matched_other_kl:.6g} |".format(
                **item
            )
        )
    return "\n".join(lines) + "\n"


def run_report() -> dict[str, Any]:
    preflight = run_preflight()
    summary = _load_completed_results(preflight["study_identity"])
    rendered = _render_report(summary)
    if REPORT_PATH.is_file() and REPORT_PATH.read_text(encoding="utf-8") != rendered:
        raise RuntimeError("immutable CPNG report differs from completed summary")
    if not REPORT_PATH.is_file():
        _atomic_text(REPORT_PATH, rendered)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Development-only local CPNG runner")
    parser.add_argument(
        "command", choices=("preflight", "capture", "construct", "calibrate", "report", "all")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    command = build_parser().parse_args(argv).command
    if command == "preflight":
        result = run_preflight()
    elif command == "capture":
        result = run_capture()
    elif command == "construct":
        result = run_construct()
    elif command == "calibrate":
        result = run_calibrate()
    elif command == "report":
        result = run_report()
    else:
        run_preflight()
        run_capture()
        run_construct()
        result = run_calibrate()
        run_report()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
